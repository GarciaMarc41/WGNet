"""Staged Wave-to-Gaussian Seismic Inversion (Rust + PyTorch)."""

from .renderer import render_gaussians, field_and_edge_loss
from .stage1_decoder import GeologicalAutoencoder, TriangleGaussianDecoder
from .stage2_wave_enc import (
    HybridVirtualSourceWaveEncoder,
    TraveltimeAttentionWaveEncoder,
    StagedWaveToGaussianModel,
)
from .stage3_physics import WavePhysicsFunction, WavePhysicsLoss
from .probes import WaveAttributionSuite

__all__ = [
    "render_gaussians",
    "field_and_edge_loss",
    "GeologicalAutoencoder",
    "TriangleGaussianDecoder",
    "HybridVirtualSourceWaveEncoder",
    "TraveltimeAttentionWaveEncoder",
    "StagedWaveToGaussianModel",
    "WavePhysicsFunction",
    "WavePhysicsLoss",
    "WaveAttributionSuite",
]
