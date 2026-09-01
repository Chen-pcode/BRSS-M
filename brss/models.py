from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def groups(channels: int) -> int:
    return next((value for value in (8, 4, 2, 1) if channels % value == 0), 1)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, groups_: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, groups=groups_, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(in_channels, in_channels, 3, stride, in_channels),
            ConvNormAct(in_channels, out_channels, 1),
            ConvNormAct(out_channels, out_channels, 3, 1, out_channels),
            ConvNormAct(out_channels, out_channels, 1),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else ConvNormAct(in_channels, out_channels, 1, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class SelectiveStateSpace2D(nn.Module):
    """Reference 2-D selective SSM with input-dependent delta, B, and C.

    It is intentionally pure PyTorch for Kaggle portability. The recurrence is
    a real selective state-space update, not cumulative summation; apply it only
    at low resolutions where the reference implementation is practical.
    """

    def __init__(self, channels: int, use_selective_ssm: bool = True, use_local_path: bool = True, use_plain_scan: bool = False, use_boundary_router: bool = True):
        super().__init__()
        self.channels = channels
        self.use_selective_ssm = use_selective_ssm
        self.use_local_path = use_local_path
        self.use_plain_scan = use_plain_scan
        self.use_boundary_router = use_boundary_router
        self.in_proj = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.local = ConvNormAct(channels, channels, 3, groups_=channels)
        self.delta = nn.Conv2d(channels, channels, 1)
        self.b_proj = nn.Conv2d(channels, channels, 1)
        self.c_proj = nn.Conv2d(channels, channels, 1)
        self.direction = nn.Conv2d(channels, 4, 1)
        self.log_a = nn.Parameter(torch.zeros(channels))
        self.d = nn.Parameter(torch.ones(channels))
        self.out = ConvNormAct(channels, channels, 1)

    def _scan(self, u: torch.Tensor, delta: torch.Tensor, b: torch.Tensor, c: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            u, delta, b, c = (torch.flip(value, (-1,)) for value in (u, delta, b, c))
        state = torch.zeros_like(u[..., 0])
        # Each recurrence step has tensors shaped [batch, channels]. Keeping
        # A and D two-dimensional prevents batch/channel broadcast misalignment.
        a = -F.softplus(self.log_a).view(1, -1)
        outputs = []
        for index in range(u.shape[-1]):
            dt = delta[..., index]
            state = torch.exp(a * dt) * state + dt * b[..., index] * u[..., index]
            outputs.append(c[..., index] * state + self.d.view(1, -1) * u[..., index])
        result = torch.stack(outputs, dim=-1)
        return torch.flip(result, (-1,)) if reverse else result

    def forward(self, x: torch.Tensor, boundary_prior: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=1)
        local = self.local(u) if self.use_local_path else u
        if not self.use_selective_ssm:
            return x + self.out(local * gate.sigmoid())
        delta = F.softplus(self.delta(local)) + 1e-3
        b, c = self.b_proj(local), self.c_proj(local)
        if self.use_plain_scan:
            horizontal = (torch.cumsum(local, 3) + torch.flip(torch.cumsum(torch.flip(local, (3,)), 3), (3,))) / local.shape[3]
            vertical = (torch.cumsum(local, 2) + torch.flip(torch.cumsum(torch.flip(local, (2,)), 2), (2,))) / local.shape[2]
            return x + self.out((horizontal + vertical) * gate.sigmoid())
        row = local.flatten(2)
        row_delta, row_b, row_c = (value.flatten(2) for value in (delta, b, c))
        col = local.transpose(2, 3).flatten(2)
        col_delta, col_b, col_c = (value.transpose(2, 3).flatten(2) for value in (delta, b, c))
        col_shape = (local.shape[0], local.shape[1], local.shape[3], local.shape[2])
        outputs = [
            self._scan(row, row_delta, row_b, row_c, False).view_as(local),
            self._scan(row, row_delta, row_b, row_c, True).view_as(local),
            self._scan(col, col_delta, col_b, col_c, False).view(col_shape).transpose(2, 3),
            self._scan(col, col_delta, col_b, col_c, True).view(col_shape).transpose(2, 3),
        ]
        directional_weight = self.direction(local).softmax(dim=1)
        # Ambiguous contour regions retain more local evidence during global propagation.
        uncertainty = 4 * boundary_prior.sigmoid() * (1 - boundary_prior.sigmoid()) if self.use_boundary_router else 0.0
        mixed = sum(output * directional_weight[:, index : index + 1] for index, output in enumerate(outputs))
        mixed = mixed * (1 - 0.5 * uncertainty) + local * uncertainty
        return x + self.out(mixed * gate.sigmoid())


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_ssm: bool, use_local_path: bool, use_plain_scan: bool, use_ssm_boundary_router: bool):
        super().__init__()
        self.conv = DepthwiseResidual(in_channels, out_channels, stride)
        self.prior = nn.Conv2d(out_channels, 1, 1)
        self.ssm = SelectiveStateSpace2D(out_channels, use_ssm, use_local_path, use_plain_scan, use_ssm_boundary_router) if use_ssm else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        boundary_prior = self.prior(x)
        return (self.ssm(x, boundary_prior) if isinstance(self.ssm, SelectiveStateSpace2D) else x), boundary_prior


class BoundaryFusion(nn.Module):
    def __init__(self, high_channels: int, skip_channels: int, out_channels: int, use_boundary_router: bool, use_cross_scale: bool):
        super().__init__()
        self.use_boundary_router = use_boundary_router
        self.use_cross_scale = use_cross_scale
        self.high = ConvNormAct(high_channels, out_channels, 1)
        self.skip = ConvNormAct(skip_channels, out_channels, 1)
        self.boundary = nn.Conv2d(out_channels * 2, 1, 1)
        self.router = nn.Sequential(nn.Conv2d(1, out_channels, 1), nn.Sigmoid())
        self.mix = DepthwiseResidual(out_channels * 2, out_channels)

    def forward(self, high: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = F.interpolate(self.high(high), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        low = self.skip(skip)
        boundary = self.boundary(torch.cat((high, low), dim=1))
        if self.use_cross_scale:
            high = high + low.mean((2, 3), keepdim=True)
        if self.use_boundary_router:
            low = low * (1 + self.router(boundary))
        return self.mix(torch.cat((high, low), dim=1)), boundary


class BRSSMambaSeg(nn.Module):
    """Six-resolution CNN-SSM model with boundary-routed decoder fusion."""

    def __init__(self, base: int = 16, stages: int = 6, use_ssm: bool = True, use_ssm_boundary_router: bool = True, use_decoder_boundary_router: bool = True, use_local_path: bool = True, use_cross_scale: bool = True, use_plain_scan: bool = False):
        super().__init__()
        if stages not in {4, 5, 6}:
            raise ValueError("stages must be 4, 5 or 6")
        self.stages = stages
        widths = {4: [base, base, base * 2, base * 4], 5: [base, base, base * 2, base * 3, base * 4], 6: [base, base, base * 2, base * 3, base * 4, base * 6]}[stages]
        self.stem = ConvNormAct(3, widths[0])
        self.encoder = nn.ModuleList()
        for index in range(1, stages):
            # The portable reference recurrence is confined to the two lowest
            # resolutions (16/8 for the six-stage model). At 32x32 it creates
            # thousands of Python-level steps per batch and exceeds Kaggle's
            # session budget without using the GPU effectively.
            deep_start = stages - 2
            self.encoder.append(EncoderStage(widths[index - 1], widths[index], 2, use_ssm and index >= deep_start, use_local_path, use_plain_scan, use_ssm_boundary_router))
        self.decoder = nn.ModuleList()
        for index in range(stages - 1, 0, -1):
            self.decoder.append(BoundaryFusion(widths[index], widths[index - 1], widths[index - 1], use_decoder_boundary_router, use_cross_scale))
        self.output = nn.Conv2d(widths[0], 1, 1)
        self.auxiliary = nn.ModuleList([nn.Conv2d(widths[1], 1, 1), nn.Conv2d(widths[2], 1, 1)])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features, boundary_scales = [self.stem(x)], []
        for block in self.encoder:
            feature, boundary_prior = block(features[-1])
            features.append(feature)
            boundary_scales.append(boundary_prior)
        decoded, boundaries = features[-1], []
        decoder_features = []
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            decoded, boundary = block(decoded, skip)
            decoder_features.append(decoded)
            boundaries.append(boundary)
        logits = self.output(decoded)
        boundary = sum(F.interpolate(item, size=logits.shape[-2:], mode="bilinear", align_corners=False) for item in boundaries) / len(boundaries)
        aux = []
        for head, feature in zip(self.auxiliary, reversed(decoder_features[-3:-1])):
            aux.append(F.interpolate(head(feature), size=logits.shape[-2:], mode="bilinear", align_corners=False))
        return {"logits": logits, "boundary": boundary, "boundary_scales": boundary_scales, "aux": aux}


ABLATIONS = {
    "brss_mamba": {},
    "brss_4stage": {"stages": 4},
    "brss_4stage_matched": {"stages": 4, "base": 24},
    "brss_5stage": {"stages": 5},
    "brss_no_ssm": {"use_ssm": False},
    "brss_plain_scan": {"use_plain_scan": True},
    "brss_no_boundary_router": {"use_ssm_boundary_router": False, "use_decoder_boundary_router": False},
    "brss_ssm_router_only": {"use_decoder_boundary_router": False},
    "brss_decoder_router_only": {"use_ssm_boundary_router": False},
    "brss_no_local_path": {"use_local_path": False},
    "brss_no_cross_scale": {"use_cross_scale": False},
}


def get_model(name: str) -> BRSSMambaSeg:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown model variant: {name}. Choices: {', '.join(ABLATIONS)}")
    return BRSSMambaSeg(**ABLATIONS[name])
