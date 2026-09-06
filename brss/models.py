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


class RasterMamba2D(nn.Module):
    """Low-resolution row-major adaptation of the official Mamba block."""

    def __init__(self, channels: int):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaSeg requires mamba-ssm. Install a CUDA-compatible build with "
                "`pip install --no-build-isolation mamba-ssm causal-conv1d`."
            )
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.out = ConvNormAct(channels, channels, 1)

    @staticmethod
    def _tokens(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _feature(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        tokens = self._tokens(x)
        output = self.mamba(self.norm(tokens))
        return x + self.out(self._feature(output, height, width))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_mamba: bool):
        super().__init__()
        self.conv = DepthwiseResidual(in_channels, out_channels, stride)
        self.prior = nn.Conv2d(out_channels, 1, 1)
        self.mamba = RasterMamba2D(out_channels) if use_mamba else nn.Identity()

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

    def __init__(self, base: int = 16, stages: int = 6, use_mamba: bool = True):
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
            self.encoder.append(EncoderStage(widths[index - 1], widths[index], 2, use_mamba and index >= ssm_start_index))
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
    "brss_raster_mamba": {},
    "brss_no_mamba": {"use_mamba": False},
    "brss_5stage": {"stages": 5},
}


def get_model(name: str) -> BRSSMambaSeg:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown model variant: {name}. Choices: {', '.join(ABLATIONS)}")
    return BRSSMambaSeg(**ABLATIONS[name])
