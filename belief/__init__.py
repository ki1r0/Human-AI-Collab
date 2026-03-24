"""Belief module — belief state tracking and ghost visualization."""

from .manager import BeliefManager
try:
    from .ghost_visualizer import GhostVisualizer
except ImportError:
    GhostVisualizer = None  # type: ignore[assignment,misc]  # requires omni.usd
