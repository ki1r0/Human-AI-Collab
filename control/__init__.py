"""Control module — robot motion control and Franka arm policy."""

try:
    from .franka import FrankaControlPolicy
except ImportError:
    FrankaControlPolicy = None  # type: ignore[assignment,misc]  # requires omni / IsaacLab
