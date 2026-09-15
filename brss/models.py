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


class BoundaryPreservingRasterMamba(nn.Module):
    """Boundary-conditioned residual update for one row-major Mamba scan.

    The official Mamba operator is kept unchanged. A predicted boundary
    probability controls how much of its residual update is accepted, reducing
    state-space feature propagation at uncertain lesion borders.
    """

    def __init__(
        self,
        channels: int,
        use_boundary_signal: bool = True,
        use_uncertainty: bool = True,
        fixed_gate: bool = False,
        direct_modulation: bool = False,
        preserve_local_residual: bool = True,
        scan_axis: str = "row",
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaSeg requires mamba-ssm. Install a CUDA-compatible build with "
                "`pip install --no-build-isolation mamba-ssm causal-conv1d`."
            )
        self.use_boundary_signal = use_boundary_signal
        self.use_uncertainty = use_uncertainty
        self.fixed_gate = fixed_gate
        self.direct_modulation = direct_modulation
        self.preserve_local_residual = preserve_local_residual
        if scan_axis not in {"row", "column"}:
            raise ValueError("scan_axis must be 'row' or 'column'")
        self.scan_axis = scan_axis
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        if not fixed_gate and not direct_modulation:
            gate_inputs = channels + (1 if use_boundary_signal else 0) + (1 if use_boundary_signal and use_uncertainty else 0)
            self.gate = nn.Sequential(nn.Linear(gate_inputs, 1), nn.Sigmoid())
        else:
            self.gate = None
        self.out = ConvNormAct(channels, channels, 1)

    def forward(self, x: torch.Tensor, boundary_logits: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        scan_x = x.transpose(2, 3) if self.scan_axis == "column" else x
        scan_boundary = boundary_logits.transpose(2, 3) if self.scan_axis == "column" else boundary_logits
        scan_height, scan_width = scan_x.shape[-2:]
        raw_tokens = scan_x.flatten(2).transpose(1, 2)
        normalized = self.norm(raw_tokens)
        scanned = self.mamba(normalized)

        probability = scan_boundary.sigmoid().flatten(2).transpose(1, 2)
        uncertainty = 4.0 * probability * (1.0 - probability)
        if self.direct_modulation:
            refined = scanned * (1.0 - 0.5 * probability)
        elif self.fixed_gate:
            learned_gate = torch.ones_like(probability)
        elif self.use_boundary_signal:
            gate_parts = [normalized, probability]
            if self.use_uncertainty:
                gate_parts.append(uncertainty)
            gate_input = torch.cat(gate_parts, dim=-1)
            learned_gate = self.gate(gate_input)
        else:
            learned_gate = self.gate(normalized)

        if not self.direct_modulation:
            # A fixed factor makes the mechanism explicitly boundary
            # preserving; the learned gate adapts the update to feature content.
            boundary_factor = 1.0 - 0.5 * uncertainty if self.use_uncertainty else 1.0
            update_gate = learned_gate * boundary_factor
            refined = normalized + update_gate * (scanned - normalized)
        feature = refined.transpose(1, 2).reshape(x.shape[0], x.shape[1], scan_height, scan_width)
        if self.scan_axis == "column":
            feature = feature.transpose(2, 3)
        output = self.out(feature)
        return x + output if self.preserve_local_residual else output


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_mamba: bool,
        compression: bool,
        dual_axis: bool,
        grouped: bool,
        boundary_gated: bool = False,
        boundary_signal: bool = True,
        boundary_uncertainty: bool = True,
        fixed_boundary_gate: bool = False,
        direct_boundary_modulation: bool = False,
        preserve_local_residual: bool = True,
        scan_axis: str = "row",
    ):
        super().__init__()
        self.conv = DepthwiseResidual(in_channels, out_channels, stride)
        self.prior = nn.Conv2d(out_channels, 1, 1)
        if use_mamba and boundary_gated:
            self.mamba = BoundaryPreservingRasterMamba(
                out_channels,
                use_boundary_signal=boundary_signal,
                use_uncertainty=boundary_uncertainty,
                fixed_gate=fixed_boundary_gate,
                direct_modulation=direct_boundary_modulation,
                preserve_local_residual=preserve_local_residual,
                scan_axis=scan_axis,
            )
        elif use_mamba:
            self.mamba = HighResolutionGroupedMamba(out_channels, compression, dual_axis, grouped)
        else:
            self.mamba = nn.Identity()
        self.boundary_gated = boundary_gated and use_mamba

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = self.conv(x)
        boundary_prior = self.prior(x)
        feature = self.mamba(x, boundary_prior) if self.boundary_gated else self.mamba(x)
        guidance = boundary_prior if self.boundary_gated else None
        return feature, boundary_prior, guidance


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


class MaskGuidedMambaBridge(nn.Module):
    """Refines a decoder fusion with a predicted or uniform coarse mask."""

    def __init__(self, channels: int, mask_guided: bool, use_mamba: bool, uniform_mask: bool = False):
        super().__init__()
        if use_mamba and Mamba is None:
            raise ImportError(
                "MambaSeg requires mamba-ssm. Install a CUDA-compatible build with "
                "`pip install --no-build-isolation mamba-ssm causal-conv1d`."
            )
        self.mask_guided = mask_guided
        self.use_mamba = use_mamba
        self.uniform_mask = uniform_mask
        inner_channels = max(8, channels // 2)
        self.reduce = nn.Conv2d(channels, inner_channels, 1)
        self.norm = nn.LayerNorm(inner_channels)
        # The same state-space model processes lesion and background streams.
        # Folding streams into the batch dimension prevents a parameter increase.
        self.mamba = Mamba(d_model=inner_channels, d_state=16, d_conv=4, expand=2) if use_mamba else None
        stream_count = 2 if mask_guided else 1
        self.fuse = nn.Conv2d(inner_channels * stream_count, inner_channels, 1)
        self.expand = nn.Conv2d(inner_channels, channels, 1)
        self.out = ConvNormAct(channels, channels, 1)

    def _scan(self, feature: torch.Tensor) -> torch.Tensor:
        if self.mamba is None:
            return feature
        height, width = feature.shape[-2:]
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = self.mamba(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(feature.shape[0], feature.shape[1], height, width)

    def forward(self, x: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        feature = self.reduce(x)
        if self.mask_guided:
            lesion_probability = torch.full_like(coarse_logits, 0.5) if self.uniform_mask else coarse_logits.sigmoid()
            streams = torch.stack(
                (feature * lesion_probability, feature * (1 - lesion_probability)), dim=1
            )
            batch, stream_count, channels, height, width = streams.shape
            streams = self._scan(streams.reshape(batch * stream_count, channels, height, width))
            feature = streams.reshape(batch, stream_count * channels, height, width)
        else:
            feature = self._scan(feature)
        return x + self.out(self.expand(self.fuse(feature)))


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
        decoder_bridge: bool = False,
        bridge_mask_guided: bool = False,
        bridge_use_mamba: bool = False,
        bridge_index: int = 1,
        bridge_uniform_mask: bool = False,
        boundary_gated: bool = False,
        boundary_signal: bool = True,
        boundary_uncertainty: bool = True,
        fixed_boundary_gate: bool = False,
        direct_boundary_modulation: bool = False,
        preserve_local_residual: bool = True,
        scan_axis: str = "row",
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
            self.encoder.append(
                EncoderStage(
                    widths[index - 1],
                    widths[index],
                    2,
                    use_mamba and index in mamba_indices,
                    compression,
                    dual_axis,
                    grouped,
                    boundary_gated=boundary_gated and index in mamba_indices,
                    boundary_signal=boundary_signal,
                    boundary_uncertainty=boundary_uncertainty,
                    fixed_boundary_gate=fixed_boundary_gate,
                    direct_boundary_modulation=direct_boundary_modulation,
                    preserve_local_residual=preserve_local_residual,
                    scan_axis=scan_axis,
                )
            )
        self.decoder = nn.ModuleList()
        for index in range(stages - 1, 0, -1):
            self.decoder.append(BoundaryFusion(widths[index], widths[index - 1], widths[index - 1]))
        self.decoder_bridge = decoder_bridge
        if decoder_bridge and stages != 6:
            raise ValueError("The mask-guided decoder bridge is defined for the six-stage architecture")
        if decoder_bridge and bridge_index not in {0, 1}:
            raise ValueError("The decoder bridge may be placed at 16x16 (0) or 32x32 (1)")
        self.bridge_index = bridge_index
        # At index 0, the 8x8 bottleneck creates a coarse map to condition the
        # first 16x16 fusion. At index 1, the first 16x16 decoder output
        # creates a map to condition the following 32x32 fusion.
        coarse_channels = widths[stages - 1 - bridge_index]
        bridge_channels = widths[stages - 2 - bridge_index]
        self.coarse_mask = nn.Conv2d(coarse_channels, 1, 1) if decoder_bridge else None
        self.bridge = (
            MaskGuidedMambaBridge(bridge_channels, bridge_mask_guided, bridge_use_mamba, bridge_uniform_mask)
            if decoder_bridge
            else None
        )
        self.output = nn.Conv2d(widths[0], 1, 1)
        self.auxiliary = nn.ModuleList([nn.Conv2d(widths[1], 1, 1), nn.Conv2d(widths[2], 1, 1)])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features, boundary_scales, boundary_guidance = [self.stem(x)], [], []
        for block in self.encoder:
            feature, boundary_prior, guidance = block(features[-1])
            features.append(feature)
            boundary_scales.append(boundary_prior)
            if guidance is not None:
                boundary_guidance.append(guidance)
        decoded, boundaries = features[-1], []
        decoder_features = []
        coarse_logits = self.coarse_mask(decoded) if self.bridge is not None and self.bridge_index == 0 else None
        for index, (block, skip) in enumerate(zip(self.decoder, reversed(features[:-1]))):
            decoded, boundary = block(decoded, skip)
            if index == self.bridge_index - 1 and self.coarse_mask is not None:
                coarse_logits = self.coarse_mask(decoded)
            if index == self.bridge_index and self.bridge is not None:
                if coarse_logits is None:
                    raise RuntimeError("Decoder bridge requires a coarse-mask prediction")
                coarse_logits = F.interpolate(coarse_logits, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
                decoded = self.bridge(decoded, coarse_logits)
            decoder_features.append(decoded)
            boundaries.append(boundary)
        logits = self.output(decoded)
        boundary = sum(F.interpolate(item, size=logits.shape[-2:], mode="bilinear", align_corners=False) for item in boundaries) / len(boundaries)
        aux = []
        for head, feature in zip(self.auxiliary, reversed(decoder_features[-3:-1])):
            aux.append(F.interpolate(head(feature), size=logits.shape[-2:], mode="bilinear", align_corners=False))
        return {
            "logits": logits,
            "boundary": boundary,
            "boundary_scales": boundary_scales,
            "boundary_guidance": boundary_guidance,
            "aux": aux,
        }


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
    # Decoder bridge study. Encoder Mamba is disabled in all bridge variants,
    # so the comparison isolates decoder placement and mask conditioning.
    "brss_cnn_final_boundary": {"use_mamba": False},
    "brss_raster_final_boundary": {"dual_axis": False, "grouped": False, "compression": False},
    "brss_decoder_mamba_bridge": {
        "use_mamba": False,
        "decoder_bridge": True,
        "bridge_mask_guided": False,
        "bridge_use_mamba": True,
    },
    "brss_mask_guided_fusion": {
        "use_mamba": False,
        "decoder_bridge": True,
        "bridge_mask_guided": True,
        "bridge_use_mamba": False,
    },
    "brss_mgmb_mamba_bridge": {
        "use_mamba": False,
        "decoder_bridge": True,
        "bridge_mask_guided": True,
        "bridge_use_mamba": True,
    },
    "brss_uniform_mask_mgmb": {
        "use_mamba": False,
        "decoder_bridge": True,
        "bridge_mask_guided": True,
        "bridge_use_mamba": True,
        "bridge_uniform_mask": True,
    },
    "brss_mgmb_16_bridge": {
        "use_mamba": False,
        "decoder_bridge": True,
        "bridge_mask_guided": True,
        "bridge_use_mamba": True,
        "bridge_index": 0,
    },
    # Boundary-Preserving Selective Raster Mamba (BPSR) study. The Mamba block
    # remains at 32x32; only the boundary-conditioned update rule changes.
    "brss_bpsr_mamba": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": True,
        "boundary_uncertainty": True,
    },
    "brss_bpsr_no_uncertainty": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": True,
        "boundary_uncertainty": False,
    },
    "brss_bpsr_fixed_gate": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": True,
        "boundary_uncertainty": True,
        "fixed_boundary_gate": True,
    },
    "brss_bpsr_no_boundary_signal": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": False,
        "boundary_uncertainty": False,
    },
    "brss_bpsr_direct_modulation": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": True,
        "boundary_uncertainty": False,
        "direct_boundary_modulation": True,
    },
    "brss_bpsr_no_local_residual": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": True,
        "boundary_uncertainty": True,
        "preserve_local_residual": False,
    },
    "brss_bpsr_column_scan": {
        "use_mamba": True,
        "boundary_gated": True,
        "boundary_signal": True,
        "boundary_uncertainty": True,
        "scan_axis": "column",
    },
}


def get_model(name: str) -> BRSSMambaSeg:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown model variant: {name}. Choices: {', '.join(ABLATIONS)}")
    return BRSSMambaSeg(**ABLATIONS[name])
