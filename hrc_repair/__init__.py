"""Minimal REPAIR-style M0 runner for the hub-cover placement POC.

The package is intentionally small and provider-agnostic.  It reuses the
existing Isaac/RoCo launcher as the physical skill executor, while keeping
planner observations, helper handoff, and the independent evaluator in a
separate contract boundary.
"""

__all__ = ["contracts", "config", "logger", "planner", "adapter", "observer", "evaluator"]
