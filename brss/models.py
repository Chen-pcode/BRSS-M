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


class HighResolutionGroupedMamba(nn.Module):
    """Grouped shared Mamba for long 2-D sequences at 32x32 resolution."""

    def __init__(self, channels: int, compression: bool = True, dual_axis: bool = True, grouped: bool = True):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaSeg requires mamba-ssm. Install a CUDA-compatible build with "
                "`pip install --no-build-isolation mamba-ssm causal-conv1d`."
            )
        self.compression = compression
        self.dual_axis = dual_axis
        self.grouped = grouped
        compressed_channels = max(8, channels // 2) if compression else channels
        if grouped and compressed_channels % 2:
            compressed_channels += 1
        inner_channels = compressed_channels
        group_count = 2 if grouped else 1
        if inner_channels % group_count:
            raise ValueError("Mamba channels must be divisible by the group count")
        group_channels = inner_channels // group_count
        self.group_count = group_count
        self.reduce = nn.Conv2d(channels, inner_channels, 1) if compression else nn.Identity()
        self.expand = nn.Conv2d(inner_channels, channels, 1) if compression else nn.Identity()
        self.norm = nn.LayerNorm(group_channels)
        # One shared Mamba is applied to each channel group by folding groups
        # into the batch dimension. This keeps the parameter count constant.
        self.mamba = Mamba(d_model=group_channels, d_state=16, d_conv=4, expand=2)
        self.axis_fuse = nn.Conv2d(inner_channels * (2 if dual_axis else 1), inner_channels, 1) if dual_axis else nn.Identity()
        self.out = ConvNormAct(channels, channels, 1)

    @staticmethod
    def _tokens(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _feature(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

    def _group_scan(self, feature: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, channels = feature.shape[:2]
        group_channels = channels // self.group_count
        grouped = feature.reshape(batch * self.group_count, group_channels, height, width)
        tokens = self._tokens(grouped)
        output = self.mamba(self.norm(tokens))
        return self._feature(output, height, width).reshape(batch, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        reduced = self.reduce(x)
        row = self._group_scan(reduced, height, width)
        if self.dual_axis:
            column = self._group_scan(reduced.transpose(2, 3), width, height).transpose(2, 3)
            reduced = self.axis_fuse(torch.cat((row, column), dim=1))
        else:
            reduced = row
        return x + self.out(self.expand(reduced))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_mamba: bool, compression: bool, dual_axis: bool, grouped: bool):
        super().__init__()
        self.conv = DepthwiseResidual(in_channels, out_channels, stride)
        self.prior = nn.Conv2d(out_channels, 1, 1)
        self.mamba = HighResolutionGroupedMamba(out_channels, compression, dual_axis, grouped) if use_mamba else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        boundary_prior = self.prior(x)
        feature = self.mamba(x) if isinstance(self.mamba, HighResolutionGroupedMamba) else self.mamba(x)
        return feature, boundary_prior


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

    def __init__(
        self,
        base: int = 16,
        stages: int = 6,
        use_mamba: bool = True,
        compression: bool = True,
        dual_axis: bool = True,
        grouped: bool = True,
        mamba_indices: tuple[int, ...] = (3,),
    ):
        super().__init__()
        if stages not in {4, 5, 6}:
            raise ValueError("stages must be 4, 5 or 6")
        invalid_indices = set(mamba_indices) - set(range(1, stages))
        if invalid_indices:
            raise ValueError(f"Mamba encoder indices are outside this architecture: {sorted(invalid_indices)}")
        self.stages = stages
        widths = {4: [base, base, base * 2, base * 4], 5: [base, base, base * 2, base * 3, base * 4], 6: [base, base, base * 2, base * 3, base * 4, base * 6]}[stages]
        self.stem = ConvNormAct(3, widths[0])
        self.encoder = nn.ModuleList()
        # With a 256x256 input, encoder indices 3, 4 and 5 correspond to
        # 32x32, 16x16 and 8x8 features. The stage-location ablations keep
        # every other HGM component fixed and vary only this placement.
        for index in range(1, stages):
            self.encoder.append(EncoderStage(widths[index - 1], widths[index], 2, use_mamba and index in mamba_indices, compression, dual_axis, grouped))
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
    # Mamba placement ablations. All use the same compressed, two-group,
    # row/column HGM block; only the encoder resolution changes.
    "brss_hgm_mamba": {},
    "brss_s3_s4_mamba": {"mamba_indices": (3, 4)},
    "brss_s3_s4_s5_mamba": {"mamba_indices": (3, 4, 5)},
    "brss_s4_mamba": {"mamba_indices": (4,)},
    "brss_raster_mamba": {"dual_axis": False, "grouped": False, "compression": False},
    "brss_no_mamba": {"use_mamba": False},
    "brss_no_compression": {"compression": False},
    "brss_no_grouping": {"grouped": False},
    "brss_single_axis": {"dual_axis": False},
}


def get_model(name: str) -> BRSSMambaSeg:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown model variant: {name}. Choices: {', '.join(ABLATIONS)}")
    return BRSSMambaSeg(**ABLATIONS[name])
