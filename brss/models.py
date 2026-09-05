from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


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


class AxialMamba2D(nn.Module):
    """Two-dimensional adaptation of the official Mamba selective-scan block.

    Four independent official Mamba blocks scan row-major and column-major
    token orders in both directions. This retains the fused selective scan from
    mamba-ssm while removing the Python-level recurrence used by the prototype.
    """

    def __init__(self, channels: int, use_local_path: bool = True, use_axial_scan: bool = True):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaSeg requires mamba-ssm. Install a CUDA-compatible build with "
                "`pip install --no-build-isolation mamba-ssm causal-conv1d`."
            )
        self.use_local_path = use_local_path
        self.use_axial_scan = use_axial_scan
        self.local = ConvNormAct(channels, channels, 3, groups_=channels)
        self.norm = nn.LayerNorm(channels)
        self.row_forward = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        if use_axial_scan:
            self.row_backward = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
            self.column_forward = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
            self.column_backward = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.out = ConvNormAct(channels, channels, 1)

    @staticmethod
    def _tokens(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _feature(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

    def _scan(self, block: nn.Module, tokens: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            tokens = torch.flip(tokens, (1,))
        tokens = block(self.norm(tokens))
        return torch.flip(tokens, (1,)) if reverse else tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x) if self.use_local_path else x
        height, width = local.shape[-2:]
        row_tokens = self._tokens(local)
        if not self.use_axial_scan:
            return x + self.out(self._feature(self._scan(self.row_forward, row_tokens), height, width))
        column_tokens = self._tokens(local.transpose(2, 3))
        row = self._scan(self.row_forward, row_tokens)
        row = row + self._scan(self.row_backward, row_tokens, reverse=True)
        column = self._scan(self.column_forward, column_tokens)
        column = column + self._scan(self.column_backward, column_tokens, reverse=True)
        column = self._feature(column, width, height).transpose(2, 3)
        return x + self.out((self._feature(row, height, width) + column) * 0.25)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_mamba: bool, use_local_path: bool, use_axial_scan: bool):
        super().__init__()
        self.conv = DepthwiseResidual(in_channels, out_channels, stride)
        self.prior = nn.Conv2d(out_channels, 1, 1)
        self.mamba = AxialMamba2D(out_channels, use_local_path, use_axial_scan) if use_mamba else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        boundary_prior = self.prior(x)
        return self.mamba(x), boundary_prior


class BoundaryFusion(nn.Module):
    def __init__(self, high_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.high = ConvNormAct(high_channels, out_channels, 1)
        self.skip = ConvNormAct(skip_channels, out_channels, 1)
        self.boundary = nn.Conv2d(out_channels * 2, 1, 1)
        self.mix = DepthwiseResidual(out_channels * 2, out_channels)

    def forward(self, high: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = F.interpolate(self.high(high), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        low = self.skip(skip)
        fusion = torch.cat((high, low), dim=1)
        return self.mix(fusion), self.boundary(fusion)


class BRSSMambaSeg(nn.Module):
    """Six-resolution local-global skin lesion segmenter using official Mamba."""

    def __init__(self, base: int = 16, stages: int = 6, use_mamba: bool = True, use_local_path: bool = True, use_axial_scan: bool = True):
        super().__init__()
        if stages not in {4, 5, 6}:
            raise ValueError("stages must be 4, 5 or 6")
        self.stages = stages
        widths = {4: [base, base, base * 2, base * 4], 5: [base, base, base * 2, base * 3, base * 4], 6: [base, base, base * 2, base * 3, base * 4, base * 6]}[stages]
        self.stem = ConvNormAct(3, widths[0])
        self.encoder = nn.ModuleList()
        # Mamba is applied only at 16x16 and smaller features for 256x256 input.
        ssm_start_index = 4
        for index in range(1, stages):
            self.encoder.append(EncoderStage(widths[index - 1], widths[index], 2, use_mamba and index >= ssm_start_index, use_local_path, use_axial_scan))
        self.decoder = nn.ModuleList()
        for index in range(stages - 1, 0, -1):
            self.decoder.append(BoundaryFusion(widths[index], widths[index - 1], widths[index - 1]))
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
    "brss_raster_mamba": {"use_axial_scan": False},
    "brss_no_mamba": {"use_mamba": False},
    "brss_5stage": {"stages": 5},
    "brss_no_local_path": {"use_local_path": False},
}


def get_model(name: str) -> BRSSMambaSeg:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown model variant: {name}. Choices: {', '.join(ABLATIONS)}")
    return BRSSMambaSeg(**ABLATIONS[name])
