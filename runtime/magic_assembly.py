"""Magic (kinematic/teleportation) assembly for tabletop parts.

Architecture
============
- Main-thread only for all USD Stage writes (combine / separate).
- The VLM background worker thread only enqueues AssemblyCommand objects.
- The main simulation update loop drains the queue via execute_pending().

Socket / Plug convention (Gearbox Assembly Manual)
===================================================
Socket : an Xform child of the parent prim named "socket_<desc>" (e.g.
         "socket_hub_output", "socket_gear_input", "socket_casing_mate").
Plug   : an Xform child of the child prim named "plug_main" or
         "plug_casing_mate" (for Casing_Top→Casing_Base mating).

Preferred runtime API:
    combine(partA, partB, plug_name, socket_name)

The 4-argument path is now the primary/strict path used by UI/model commands.
Legacy 2/3-argument combine() calls are still accepted for compatibility and
will auto-resolve plug/socket names.

Transform algebra  (USD row-vector convention)
==============================================
USD uses row-vector / post-multiply matrices:
    point_world = point_local * local_to_world_matrix
    child_world = child_local * parent_world

Given
  S_local = socket's local transform  (relative to parent prim)
  P_local = plug's   local transform  (relative to child  prim)

After reparenting child under parent, we want:
  plug_world_new  =  socket_world
  P_local * child_local_new * parent_world  =  S_local * parent_world

Therefore:
  P_local * child_local_new  =  S_local
  child_local_new  =  inv(P_local) * S_local
"""
from __future__ import annotations

import queue
import re
import time
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssemblyRecord:
    """Persistent state for a part that has been combined, needed by separate()."""

    child_path: str          # current USD path of the child (after reparenting)
    parent_path: str         # USD path of the parent it was combined with
    socket_name: Optional[str]  # socket used on parent (for auto socket occupancy)
    pre_combine_parent: str  # USD path of child's original parent (for restoring)
    pre_combine_world: Any   # Gf.Matrix4d — world transform captured before combine


@dataclass
class AssemblyCommand:
    """A thread-safe command from the VLM worker → main thread assembly queue."""

    action: str                                        # "combine" | "separate" | "focus"
    # --- combine fields ---
    child_name: str = ""
    parent_name: str = ""
    plug_name: Optional[str] = None
    socket_name: Optional[str] = None
    # --- separate fields ---
    part_name: str = ""
    # --- optional completion hook ---
    callback: Optional[Callable[[bool, str], None]] = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default socket lookup — maps (child_prim_name, parent_prim_name) → socket.
#
# When combine() is called WITHOUT an explicit socket_name, this dict is
# consulted.  It encodes the relationships from the I2R Gearbox Assembly
# Manual so that `combine("Input_Shaft", "Casing_Base")` automatically
# selects `socket_gear_input`.
# ---------------------------------------------------------------------------

DEFAULT_SOCKET_MAP: Dict[Tuple[str, str], str] = {
    # ── Sub-Assembly 1: Casing Top prep (steps 1-5) ──────────────────────
    ("Hub_Cover_Output", "Casing_Top"):  "socket_hub_output",
    ("Hub_Cover_Input",  "Casing_Top"):  "socket_hub_input",
    ("Hub_Cover_Small",  "Casing_Top"):  "socket_hub_small",
    ("Hub_Cover_Output_Top", "Casing_Top"):  "socket_hub_output",
    ("Hub_Cover_Input_Top",  "Casing_Top"):  "socket_hub_input",
    ("Hub_Cover_Small_Top",  "Casing_Top"):  "socket_hub_small",

    # ── Sub-Assembly 2: Casing Base prep (steps 6-10) ────────────────────
    ("Hub_Cover_Output", "Casing_Base"): "socket_hub_output",
    ("Hub_Cover_Small",  "Casing_Base"): "socket_hub_small",
    ("Hub_Cover_Output_Base", "Casing_Base"): "socket_hub_output",
    ("Hub_Cover_Small_Base",  "Casing_Base"): "socket_hub_small",
    ("Hub_Cover_Small_Base_01",  "Casing_Base"): "socket_hub_small_1",
    ("Hub_Cover_Small_Base_02",  "Casing_Base"): "socket_hub_small_2",

    # ── Pre-assembly: gear onto shaft ────────────────────────────────────
    ("Transfer_Gear",    "Transfer_Shaft"): "socket_gear",
    ("Output_Gear",      "Output_Shaft"):   "socket_gear",

    # ── Main assembly: shafts into base (step 11) ────────────────────────
    ("Input_Shaft",      "Casing_Base"): "socket_gear_input",
    ("Transfer_Shaft",   "Casing_Base"): "socket_gear_transfer",
    ("Output_Shaft",     "Casing_Base"): "socket_gear_output",
    ("Output_Shaft",     "Casing_Top"):  "socket_hub_output",

    # ── Bearing sub-assemblies (manual step 11 prep image) ───────────────
    # ── Casing mating (step 13) ──────────────────────────────────────────
    ("Casing_Top",       "Casing_Base"): "socket_casing_mate",

    # ── Casing bolts (step 15) ───────────────────────────────────────────
    ("M10_Casing_Bolt",  "Casing_Top"):  "socket_bolt_casing_1",

    # ── Accessories (steps 16-17) ────────────────────────────────────────
    ("Oil_Level_Indicator", "Casing_Base"): "socket_oil_1",
    ("Oil_Level_Indicator_02", "Casing_Base"): "socket_oil_1",
    ("Breather_Plug",       "Casing_Base"): "socket_breather",

    # ── Hub bolts (steps 4, 9) ───────────────────────────────────────────
    ("M6_Hub_Bolt",      "Casing_Top"):  "socket_bolt_hub_1",
    ("M6_Hub_Bolt",      "Casing_Base"): "socket_bolt_hub_1",
}

# Default plug lookup for legacy combine() paths that do not provide plug_name.
DEFAULT_PLUG_MAP: Dict[Tuple[str, str], str] = {
    ("Casing_Top", "Casing_Base"): "plug_casing_mate",
}

HUB_BOLT_SOCKET_INDICES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)


# Extra loose hub bolts requested by the user. These are cloned at runtime from
# the existing authored M6 hub bolt prims so the binary scene USD does not need
# to be hand-edited just to add more loose instances.
EXTRA_HUB_BOLT_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "M6_Hub_Bolt_08": (140.0, 0.0, 0.0),
    "M6_Hub_Bolt_09": (210.0, 0.0, 0.0),
    "M6_Hub_Bolt_10": (280.0, 0.0, 0.0),
    "M6_Hub_Bolt_11": (70.0, 60.0, 0.0),
    "M6_Hub_Bolt_13": (210.0, 60.0, 0.0),
    "M6_Hub_Bolt_14": (280.0, 60.0, 0.0),
}

# Extra Casing_Top hub-bolt sockets for the input and small hub covers.
# These are fitted to the actual bolt-hole circles on the Casing_Top top face,
# so the socket origin sits on the hole center rather than on the cover plate.
EXTRA_HUB_BOLT_TOP_SOCKETS: Dict[str, Tuple[float, float, float]] = {
    # Input-side lower cover (right)
    "socket_bolt_hub_8": (8.26, -26.47, 27.9),
    "socket_bolt_hub_9": (54.00, -72.69, 27.9),
    "socket_bolt_hub_10": (7.76, -72.52, 27.9),
    # Small-side lower cover (left)
    "socket_bolt_hub_11": (-7.96, -26.37, 27.9),
    "socket_bolt_hub_13": (-8.00, -72.69, 27.9),
    "socket_bolt_hub_14": (-53.82, -72.43, 27.9),
}

EXTRA_CASING_BOLT_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "M10_Casing_Bolt_02": (70.0, 0.0, 0.0),
    "M10_Casing_Bolt_03": (140.0, 0.0, 0.0),
}

EXTRA_ACCESSORY_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "Oil_Level_Indicator_02": (70.0, 0.0, 0.0),
}

TOP_CASING_BOLT_SOCKETS: Dict[str, Tuple[float, float, float]] = {
    "socket_bolt_casing_1": (60.0, 120.0, 20.0),
    "socket_bolt_casing_2": (-60.0, 120.0, 20.0),
    "socket_bolt_casing_3": (60.0, -125.0, 20.0),
    "socket_bolt_casing_4": (-60.0, -125.0, 20.0),
    "socket_bolt_casing_5": (95.0, 0.0, 20.0),
    "socket_bolt_casing_6": (-95.0, 0.0, 20.0),
}

BASE_HUB_COVER_SOCKET_ALIASES: Dict[str, str] = {
    "socket_hub_input": "socket_hub_small_1",
    "socket_hub_small": "socket_hub_small_2",
}

BASE_HUB_COVER_SOCKET_FALLBACKS: Dict[str, Tuple[float, float, float]] = {
    "socket_hub_input": (31.0, -49.69, -27.9),
    "socket_hub_small": (-31.0, -49.69, -27.9),
}

BASE_GEAR_SOCKET_FALLBACKS: Dict[str, Tuple[float, float, float]] = {
    "socket_gear_input": (31.0, -49.69, 0.0),
    "socket_gear_transfer": (-31.0, -49.69, 0.0),
    "socket_gear_output": (0.0, 39.75, 0.0),
}

BEARING_PART_NAMES: Tuple[str, ...] = ()
BEARING_SUBASSEMBLY_BATCHES: Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]] = {
    "combine_input_shaft": [
        ("Bearing_Input_Bottom", "Input_Shaft", "plug_main", "socket_gear"),
        ("Bearing_Input_Top", "Input_Shaft", "plug_main", "socket_gear"),
    ],
    "combine_transfer_shaft": [
        ("Bearing_Transfer", "Transfer_Shaft", "plug_main", "socket_gear"),
    ],
    "combine_output_shaft": [
        ("Bearing_Output_Top", "Output_Shaft", "plug_main", "socket_gear"),
        ("Bearing_Output_Bottom", "Output_Shaft", "plug_main", "socket_gear"),
    ],
}

CASE_ACCESSORY_BATCHES: Dict[str, List[Tuple[str, str, Optional[str], Optional[str]]]] = {
    "combine_accessories": [
        ("Oil_Level_Indicator", "Casing_Base", "plug_main", "socket_oil_1"),
        ("Oil_Level_Indicator_02", "Casing_Base", "plug_main", "socket_oil_2"),
        ("Breather_Plug", "Casing_Base", "plug_main", "socket_breather"),
    ],
}

CASING_BASE_INTERNAL_PART_HINTS = {
    # Only consider the internal transfer gear for the casing-top clearance
    # check. The output shaft/gear intentionally occupies the top hub cavity
    # and the shafts are expected to pass through top bores, so using their
    # whole BBoxes incorrectly lifts the lid.
    "Transfer_Gear",
}

# Hardcoded correction layer (user-authorized) applied in socket-local frame:
#   child_local = inv(plug_local) * fit_offset * socket_local
# Keys are (partA, partB, plug_name, socket_name).
# Translations are in the local units of authored sockets/plugs.
HARDCODED_FIT_OFFSETS: Dict[Tuple[str, str, str, str], Dict[str, Tuple[float, float, float]]] = {
    # ── Gear-on-shaft sub-assemblies ───────────────────────────────────
    # Gear disc normal is Z; rotate so hole axis aligns with shaft axis.
    # Translate positions the gear at the shaft's spline section.
    #
    ("Output_Gear", "Output_Shaft", "plug_main", "socket_gear"): {
        "translate": (0.0, 15.5, 0.0),
        "rotate_xyz": (-90.0, 0.0, 0.0),
    },
    ("Transfer_Gear", "Transfer_Shaft", "plug_main", "socket_gear"): {
        "translate": (22.1, 0.0, 0.0),
        "rotate_xyz": (0.0, 90.0, 0.0),
    },
    # ── Bearing-on-shaft sub-assemblies (hardcoded seat offsets) ──────────
    # All bearings attach through the existing shaft-center `socket_gear`.
    # Only the axial offset is part-specific; rotations align the bearing axis
    # to the shaft axis and preserve the taper orientation from the manual.
    ("Bearing_Input_Bottom", "Input_Shaft", "plug_main", "socket_gear"): {
        "translate": (0.0, -100.750000, 0.0),
        "rotate_xyz": (0.0, 0.0, 90.0),
    },
    ("Bearing_Input_Top", "Input_Shaft", "plug_main", "socket_gear"): {
        "translate": (0.0, 2.732481, 0.0),
        "rotate_xyz": (0.0, 0.0, -90.0),
    },
    ("Bearing_Transfer", "Transfer_Shaft", "plug_main", "socket_gear"): {
        "translate": (-50.563000, 0.0, 0.0),
        "rotate_xyz": (0.0, 0.0, 0.0),
    },
    ("Bearing_Output_Top", "Output_Shaft", "plug_main", "socket_gear"): {
        "translate": (0.0, 51.156502, 0.0),
        "rotate_xyz": (0.0, 0.0, -90.0),
    },
    ("Bearing_Output_Bottom", "Output_Shaft", "plug_main", "socket_gear"): {
        "translate": (0.0, -51.156502, 0.0),
        "rotate_xyz": (0.0, 0.0, 90.0),
    },
    # ── Shafts into Casing Base (step 11) ────────────────────────────────
    # Shaft axis must align with Z (vertical through bearing bores).
    # Plug and socket both at origin; shaft center goes to socket at Z=0.
    #
    # Input Shaft (axis Y): Rx(-90°) so long end (Y+) points upward in world.
    # Fine Z + tooth phase are clamped in HARDCODED_CHILD_LOCAL_* overrides.
    ("Input_Shaft", "Casing_Base", "plug_main", "socket_gear_input"): {
        "translate": (0.0, 0.0, -116.0),
        "rotate_xyz": (-90.0, 0.0, 0.0),
    },
    # Output Shaft (axis Y): Rx(-90°) so shoulder end faces downward.
    # Gear at Y=15.5 maps to Z=-15.5; offset positions output gear at
    # transfer-shaft pinion height for mesh contact.
    ("Output_Shaft", "Casing_Base", "plug_main", "socket_gear_output"): {
        "translate": (0.0, 0.0, 48.0),
        "rotate_xyz": (-90.0, 0.0, 0.0),
    },
    # Direct top-side snap path: same shaft, but flipped end-for-end relative
    # to the base-side insertion so the exposed shaft end matches the manual.
    ("Output_Shaft", "Casing_Top", "plug_main", "socket_hub_output"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    # Transfer Shaft (axis X): Ry(-90°) maps gear at X=22.1 to Z=-22.1.
    ("Transfer_Shaft", "Casing_Base", "plug_main", "socket_gear_transfer"): {
        "translate": (0.0, 0.0, -27.0),
        "rotate_xyz": (0.0, -90.0, 0.0),
    },
    # Casing_Base is authored with mirrored X/Z scale in the scene. The top
    # therefore needs a 180deg Y flip during mating so its exterior (hub-cover)
    # face ends up upward in world space, plus a local Z shift so it lands on
    # the upper parting face instead of underneath the base shell.
    ("Casing_Top", "Casing_Base", "plug_casing_mate", "socket_casing_mate"): {
        "translate": (0.0, 0.0, -55.8),
        "rotate_xyz": (0.0, 180.0, 0.0),
    },
    # ── Hub covers ──────────────────────────────────────────────────────
    # Rotation orients the disc normal to the surface normal.
    # Z-translate = ±half_thickness pushes the hub cover outward so its
    # body sits ON the surface and doesn't clip through the casing mesh.
    #
    # Hub_Cover_Output: keep the current seat depth, but flip 180deg about
    # local Y so the cover lands with the requested facing on both top/base.
    ("Hub_Cover_Output", "Casing_Top", "plug_main", "socket_hub_output"): {
        "translate": (0.0, 0.0, 7.03),
        "rotate_xyz": (-90.0, 180.0, 0.0),
    },
    ("Hub_Cover_Output", "Casing_Base", "plug_main", "socket_hub_output"): {
        # Seat the output cover so its flange just contacts the casing-base
        # surface. With the current authored socket this lands the child at
        # local Z ~= 35.3, which matches the manually fitted seat.
        "translate": (0.0, 0.0, 7.4),
        "rotate_xyz": (-90.0, 180.0, 0.0),
    },
    # Hub_Cover_Input: keep the current seat depth, but add the requested
    # 90deg Z-phase after the Y rotation.
    ("Hub_Cover_Input", "Casing_Top", "plug_main", "socket_hub_input"): {
        "translate": (0.0, 0.0, 6.3),
        "rotate_xyz": (0.0, 90.0, 90.0),
    },
    # Hub_Cover_Small: thin_axis=X, flipped so flange faces outward, Ry(+90°)
    ("Hub_Cover_Small", "Casing_Top", "plug_main", "socket_hub_small"): {
        "translate": (0.0, 0.0, -3.0),
        "rotate_xyz": (0.0, 90.0, 0.0),
    },
    ("Hub_Cover_Small", "Casing_Base", "plug_main", "socket_hub_small_1"): {
        # Small cover 01 uses the same axial seat depth as small cover 02, but
        # its lobe pattern is rotated 90deg about the socket normal.
        "translate": (0.0, 0.0, -3.1114057111257516),
        "rotate_xyz": (0.0, 90.0, 90.0),
    },
    ("Hub_Cover_Small", "Casing_Base", "plug_main", "socket_hub_small"): {
        "translate": (0.0, 0.0, -3.1114057111257516),
        "rotate_xyz": (0.0, 90.0, 0.0),
    },
    ("Hub_Cover_Small", "Casing_Base", "plug_main", "socket_hub_small_2"): {
        "translate": (0.0, 0.0, -3.1114057111257516),
        "rotate_xyz": (0.0, 90.0, 0.0),
    },
    # ── Bolts ───────────────────────────────────────────────────────────
    # Shaft axis is local Y; head at Y+ end.
    # Casing_Top (Z+ outward): Rx(+90°) maps Y+→Z+ so head faces outward.
    ("M6_Hub_Bolt", "Casing_Top", "plug_main", "socket_bolt_hub_*"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    ("M6_Hub_Bolt", "Casing_Base", "plug_main", "socket_bolt_hub_*"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    ("M10_Casing_Bolt", "Casing_Base", "plug_main", "socket_bolt_casing_*"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (-90.0, 0.0, 0.0),
    },
    ("M10_Casing_Bolt", "Casing_Top", "plug_main", "socket_bolt_casing_*"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    ("Oil_Level_Indicator", "Casing_Base", "plug_main", "socket_oil_1"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    ("Oil_Level_Indicator", "Casing_Base", "plug_main", "socket_oil_2"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    ("Breather_Plug", "Casing_Base", "plug_main", "socket_breather"): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (90.0, 0.0, 0.0),
    },
    # ── Nuts ────────────────────────────────────────────────────────────
    # Casing nuts are assembled as children of the already-installed casing
    # bolts.  Their authored axes match the bolt axis, so keep identity here
    # and apply the measured axial seat as a child-local translate override.
    ("M10_Casing_Nut", "M10_Casing_Bolt", "plug_main", ""): {
        "translate": (0.0, 0.0, 0.0),
        "rotate_xyz": (0.0, 0.0, 0.0),
    },
}

# Hard local-transform overrides after combine matrix solve.
# These are applied to child-local translation (under parent), while keeping
# solved rotation intact.  Key: (partA, partB, plug, socket)
HARDCODED_CHILD_LOCAL_Z: Dict[Tuple[str, str, str, str], float] = {
    # Requirement: Transfer_Shaft under Casing_Base must sit at Z=-27.
    ("Transfer_Shaft", "Casing_Base", "plug_main", "socket_gear_transfer"): -27.0,
    # Measured from the actual pinion band center on the one-piece input shaft.
    # The pinion tooth band is centered at local Y=-70, so Z=-74.9 places the
    # band center on the Transfer_Gear mid-plane at world Z=-4.9.
    ("Input_Shaft", "Casing_Base", "plug_main", "socket_gear_input"): -74.9,
    ("Output_Shaft", "Casing_Base", "plug_main", "socket_gear_output"): -18.32,
    # Hub-cover bolts should sit slightly proud of the flange surface.
    ("M6_Hub_Bolt", "Casing_Top", "plug_main", "socket_bolt_hub_*"): 29.2,
    ("M6_Hub_Bolt", "Casing_Base", "plug_main", "socket_bolt_hub_*"): 29.2,
    ("M10_Casing_Bolt", "Casing_Top", "plug_main", "socket_bolt_casing_*"): -31.6,
}

HARDCODED_CHILD_LOCAL_TRANSLATE: Dict[
    Tuple[str, str, str, str], Tuple[float, float, float]
] = {
    # User-tuned hub-cover seat positions. These are final child-local
    # translations after combine, not fit-offset deltas.
    ("Hub_Cover_Output", "Casing_Top", "plug_main", "socket_hub_output"): (
        9.304742279029631e-17,
        43.418589999999995,
        31.0,
    ),
    ("Hub_Cover_Output", "Casing_Base", "plug_main", "socket_hub_output"): (
        9.304742279029631e-17,
        43.418589999999995,
        31.0,
    ),
    ("Hub_Cover_Input", "Casing_Top", "plug_main", "socket_hub_input"): (
        35.90541548031911,
        -54.99766545623561,
        31.8,
    ),
    ("Hub_Cover_Small", "Casing_Base", "plug_main", "socket_hub_small_1"): (35.9, -55.0, 30.6),
    ("Hub_Cover_Small", "Casing_Base", "plug_main", "socket_hub_small_2"): (-36.3, -54.6, 30.6),
    ("Hub_Cover_Small", "Casing_Top", "plug_main", "socket_hub_small"): (-36.3, -54.6, 30.7),
    # Measured from the assembled casing pair. The outer base surface around
    # the bolt axis sits at bolt-local Y=-44.44816131591773. With the nut
    # thickness of 6.300000190734865, the flush center location is:
    #   -44.44816131591773 - 3.1500000953674325 = -47.59816141128516
    ("M10_Casing_Nut", "M10_Casing_Bolt", "plug_main", ""): (0.0, -47.59816141128516, 0.0),
}

# Exact authored XYZ rotations for cases where multiple Euler decompositions are
# equivalent mathematically, but the user wants a specific editable rotation
# triplet to appear in Isaac Sim's property panel after combine.
HARDCODED_CHILD_LOCAL_ROTATE_XYZ: Dict[
    Tuple[str, str, str, str], Tuple[float, float, float]
] = {
    ("Hub_Cover_Input", "Casing_Top", "plug_main", "socket_hub_input"): (0.0, 90.0, 90.0),
    ("Hub_Cover_Output", "Casing_Top", "plug_main", "socket_hub_output"): (-90.0, 180.0, 0.0),
    ("Hub_Cover_Output", "Casing_Base", "plug_main", "socket_hub_output"): (-90.0, 180.0, 0.0),
}

# Hard local phase override (degrees) around CHILD LOCAL Y axis.
# Used to avoid tooth tip-to-tip placement on initial snap.
HARDCODED_CHILD_LOCAL_ROTATE_Y: Dict[Tuple[str, str, str, str], float] = {
    # Phase values are measured against the fixed Transfer_Gear profile to place
    # each mating gear in a tooth-gap rather than tip-to-tip.
    ("Input_Shaft", "Casing_Base", "plug_main", "socket_gear_input"): 88.5,
    ("Output_Shaft", "Casing_Base", "plug_main", "socket_gear_output"): -52.5,
}


def _canonical_child_name(name: str) -> str:
    """Normalize child prim names for fit/socket lookup without touching parents."""
    out = re.sub(r"_\d+$", "", str(name or ""))
    if out.startswith("Hub_Cover_") and out.endswith(("_Top", "_Base")):
        out = out.rsplit("_", 1)[0]
    if out.startswith("M6_Hub_Bolt_") and out.lower().endswith(("_top", "_base")):
        out = re.sub(r"_(?:top|base)$", "", out, flags=re.IGNORECASE)
        out = re.sub(r"_\d+$", "", out)
    return out


class MagicAssemblyManager:
    """Kinematic (snap-to-socket) assembly manager.

    All USD Stage mutations MUST happen on the main thread.
    The VLM background thread only calls enqueue(); the main thread calls
    execute_pending() each update tick.

    Parameters
    ----------
    stage_fn:
        Callable returning the active Usd.Stage.  Defaults to the omni.usd
        context stage when running inside Isaac Sim.  Override in tests.
    use_omni_commands:
        When True (default), reparenting goes through omni.kit.commands so
        edits are undoable.  When False (or when omni is unavailable), falls
        back to the pure Sdf.BatchNamespaceEdit API.
    logger:
        Callable for diagnostic messages.
    """

    def __init__(
        self,
        stage_fn: Optional[Callable] = None,
        use_omni_commands: bool = True,
        logger: Callable[[str], None] = print,
    ) -> None:
        self._stage_fn = stage_fn if stage_fn is not None else self._default_stage
        self._use_omni = use_omni_commands
        self._log = logger
        # child_current_path -> AssemblyRecord
        self._records: Dict[str, AssemblyRecord] = {}
        self._queue: queue.Queue[AssemblyCommand] = queue.Queue()

    # ------------------------------------------------------------------
    # Thread-safe queue interface
    # ------------------------------------------------------------------

    def enqueue(self, cmd: AssemblyCommand) -> None:
        """Thread-safe: schedule an assembly command for main-thread execution."""
        self._queue.put_nowait(cmd)

    def execute_pending(self) -> int:
        """Main-thread: drain and execute all pending assembly commands.

        Returns the number of commands executed.
        """
        count = 0
        while not self._queue.empty():
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                break

            ok, msg = False, "not executed"
            try:
                if cmd.action == "combine":
                    ok = self.combine(
                        cmd.child_name,
                        cmd.parent_name,
                        cmd.plug_name,
                        cmd.socket_name,
                    )
                    msg = "ok" if ok else "failed"
                elif cmd.action == "separate":
                    ok = self.separate(cmd.part_name)
                    msg = "ok" if ok else "failed"
                elif cmd.action == "focus":
                    ok = self.focus(cmd.part_name)
                    msg = "ok" if ok else "failed"
                else:
                    msg = f"unknown action: {cmd.action!r}"
                    self._log(f"[MAGIC] {msg}")
            except Exception as exc:
                msg = str(exc)
                self._log(f"[MAGIC] execute_pending error: {exc}")
            finally:
                if cmd.callback is not None:
                    try:
                        cmd.callback(ok, msg)
                    except Exception:
                        pass

            count += 1
        return count

    # ------------------------------------------------------------------
    # Core operations  (MAIN-THREAD ONLY)
    # ------------------------------------------------------------------

    def combine(
        self,
        part_a: str,
        part_b: str,
        plug_name: Optional[str] = None,
        socket_name: Optional[str] = None,
    ) -> bool:
        """Snap partA to partB using explicit plug+socket frames.

        Preferred call signature:
            combine(partA, partB, plug_name, socket_name)

        Legacy 2/3-argument calls are accepted for compatibility and auto-resolve
        missing plug/socket names.

        Steps
        -----
        1. Find child and parent prims by name anywhere in the stage.
        2. Compute world transforms.
        3. Locate socket on parent (named Xform child or parent origin).
        4. Locate plug on child  (named Xform child or child  origin).
        5. Calculate the required child world transform so that
           plug_world == socket_world.
        6. Reparent child under parent via Sdf namespace edit (or Kit commands).
        7. Set the child's new local transform.

        Returns True on success, False on any error.
        """
        from pxr import UsdGeom, Gf, Sdf  # type: ignore

        child_name = str(part_a)
        parent_name = str(part_b)
        # Back-compat: older callers passed (child, parent, socket_name) positionally.
        if plug_name and not socket_name:
            ptxt = str(plug_name)
            if ptxt.lower().startswith("socket") or ptxt.lower().startswith("attach"):
                socket_name = ptxt
                plug_name = None
        plug_explicit = bool(str(plug_name or "").strip())
        socket_explicit = bool(str(socket_name or "").strip())
        self._log(
            f"[MAGIC] combine START: child={child_name!r}, parent={parent_name!r}, "
            f"plug={plug_name!r}, socket={socket_name!r}"
        )

        stage = self._stage_fn()
        if stage is None:
            self._log("[MAGIC] combine: stage not available")
            return False
        self._log(f"[MAGIC] combine: stage OK, root_layer={stage.GetRootLayer().identifier}")

        child_path = self._find_prim_path(stage, child_name)
        parent_path = self._find_prim_path(stage, parent_name)

        self._log(f"[MAGIC] combine: child_path={child_path}, parent_path={parent_path}")

        if child_path is None:
            self._log(f"[MAGIC] combine FAIL: child '{child_name}' not found in stage")
            # Dump top-level prims to help debug
            try:
                dp = stage.GetDefaultPrim()
                if dp and dp.IsValid():
                    children = [c.GetName() for c in dp.GetChildren()
                                if not c.GetName().startswith("socket_")
                                and not c.GetName().startswith("plug_")]
                    self._log(f"[MAGIC] combine: available prims under "
                              f"{dp.GetPath()}: {children[:20]}")
            except Exception:
                pass
            return False
        if parent_path is None:
            self._log(f"[MAGIC] combine FAIL: parent '{parent_name}' not found in stage")
            return False
        if str(parent_path).startswith(str(child_path) + "/"):
            self._log("[MAGIC] combine FAIL: parent is a descendant of child — abort")
            return False
        if str(child_path) == str(parent_path):
            self._log("[MAGIC] combine FAIL: child and parent are the same prim — abort")
            return False

        child_prim  = stage.GetPrimAtPath(child_path)
        parent_prim = stage.GetPrimAtPath(parent_path)
        tc = self._time_code(stage)

        # ---- Legacy auto-resolve when 4-param call is not used ----
        if socket_name is None:
            key = (child_prim.GetName(), parent_prim.GetName())
            socket_hint = DEFAULT_SOCKET_MAP.get(key)
            if socket_hint is None:
                # Instance fallback: M10_Casing_Bolt_01 -> M10_Casing_Bolt
                base_child_name = re.sub(r"_\d+$", "", child_prim.GetName())
                if base_child_name != child_prim.GetName():
                    socket_hint = DEFAULT_SOCKET_MAP.get((base_child_name, parent_prim.GetName()))
            if socket_hint is None:
                canonical_child_name = _canonical_child_name(child_prim.GetName())
                if canonical_child_name != child_prim.GetName():
                    socket_hint = DEFAULT_SOCKET_MAP.get((canonical_child_name, parent_prim.GetName()))

            socket_name = self._resolve_auto_socket_name(
                parent_prim=parent_prim,
                parent_path=str(parent_path),
                socket_hint=socket_hint,
            )
            if socket_name:
                self._log(
                    f"[MAGIC] auto-resolved socket: ({child_prim.GetName()}, {parent_prim.GetName()}) -> {socket_name}"
                )
            else:
                self._log(
                    f"[MAGIC] no default/free socket for ({child_prim.GetName()}, {parent_prim.GetName()}); "
                    f"will use parent origin"
                )
        if not plug_name:
            key = (child_prim.GetName(), parent_prim.GetName())
            plug_name = DEFAULT_PLUG_MAP.get(key, "plug_main")
            self._log(
                f"[MAGIC] auto-resolved plug: ({child_prim.GetName()}, {parent_prim.GetName()}) -> {plug_name}"
            )

        # ---- Capture pre-combine state ----
        child_world_before = self._world_xform(child_prim, tc)
        parent_world        = self._world_xform(parent_prim, tc)

        self._log(
            f"[MAGIC] combine: child_world_origin="
            f"({child_world_before.GetRow3(3)[0]:.1f}, "
            f"{child_world_before.GetRow3(3)[1]:.1f}, "
            f"{child_world_before.GetRow3(3)[2]:.1f})"
            f"  parent_world_origin="
            f"({parent_world.GetRow3(3)[0]:.1f}, "
            f"{parent_world.GetRow3(3)[1]:.1f}, "
            f"{parent_world.GetRow3(3)[2]:.1f})"
        )

        # ---- Locate socket (LOCAL, relative to parent prim) ----
        socket_local = self._find_socket_local(
            stage,
            parent_prim,
            parent_world,
            socket_name,
            tc,
            strict=socket_explicit,
        )
        if socket_local is None:
            self._log(
                f"[MAGIC] combine FAIL: socket '{socket_name}' not found on '{parent_prim.GetName()}'"
            )
            return False
        self._log(
            f"[MAGIC] combine: socket_local_origin="
            f"({socket_local.GetRow3(3)[0]:.1f}, "
            f"{socket_local.GetRow3(3)[1]:.1f}, "
            f"{socket_local.GetRow3(3)[2]:.1f})"
        )

        # ---- Locate plug (LOCAL, relative to child prim) ----
        plug_local = self._find_plug_local(
            stage,
            child_prim,
            child_world_before,
            tc,
            plug_name=plug_name,
            strict=plug_explicit,
        )
        if plug_local is None:
            self._log(
                f"[MAGIC] combine FAIL: plug '{plug_name}' not found on '{child_prim.GetName()}'"
            )
            return False
        self._log(
            f"[MAGIC] combine: plug_local_origin="
            f"({plug_local.GetRow3(3)[0]:.1f}, "
            f"{plug_local.GetRow3(3)[1]:.1f}, "
            f"{plug_local.GetRow3(3)[2]:.1f})"
        )

        # ---- Compute child local transform (LOCAL-SPACE alignment) ----
        # USD row-vector:
        # child_local_new = inv(plug_local) * fit_offset * socket_local
        fit_offset = self._resolve_fit_offset(
            child_prim.GetName(),
            parent_prim.GetName(),
            str(plug_name or ""),
            str(socket_name or ""),
        )
        child_local_new = plug_local.GetInverse() * fit_offset * socket_local
        child_local_new = self._apply_child_local_overrides(
            child_name=child_prim.GetName(),
            parent_name=parent_prim.GetName(),
            plug_name=str(plug_name or ""),
            socket_name=str(socket_name or ""),
            child_local_new=child_local_new,
        )
        child_local_new = self._apply_casing_top_clearance(
            stage=stage,
            child_prim=child_prim,
            parent_prim=parent_prim,
            child_local_new=child_local_new,
            parent_world=parent_world,
            tc=tc,
        )

        # ---- Save record (pre-combine, for separate()) ----
        pre_combine_parent = str(child_prim.GetParent().GetPath())
        record = AssemblyRecord(
            child_path=str(child_path),
            parent_path=str(parent_path),
            socket_name=socket_name,
            pre_combine_parent=pre_combine_parent,
            pre_combine_world=Gf.Matrix4d(child_world_before),
        )

        # ---- Reparent ----
        current_parent_path = child_prim.GetParent().GetPath()
        if current_parent_path == parent_path:
            self._log(
                f"[MAGIC] combine: child already under target parent {parent_path}; "
                "skipping reparent"
            )
            new_child_path = child_path
        else:
            self._log(
                f"[MAGIC] combine: reparenting {child_path} -> under {parent_path}"
            )
            new_child_path = self._reparent(stage, child_path, parent_path)
            if new_child_path is None:
                self._log(f"[MAGIC] combine FAIL: reparent failed ({child_path} -> {parent_path})")
                return False

        # ---- Set local transform ----
        new_child_prim = stage.GetPrimAtPath(new_child_path)
        if new_child_prim is None or not new_child_prim.IsValid():
            self._log(f"[MAGIC] combine FAIL: prim gone after reparent at {new_child_path}")
            return False
        self._set_xform_matrix(new_child_prim, child_local_new)
        self._apply_exact_authored_rotation_if_needed(
            new_child_prim,
            child_name=child_name,
            parent_name=parent_name,
            plug_name=str(plug_name or ""),
            socket_name=str(socket_name or ""),
        )

        # ---- Disable child subtree RigidBodyAPI to avoid nested rigid body conflict ----
        self._disable_rigid_body_recursive(new_child_prim)

        # ---- Verify final world position ----
        final_world = self._world_xform(new_child_prim, tc)
        self._log(
            f"[MAGIC] combine: final_world_origin="
            f"({final_world.GetRow3(3)[0]:.1f}, "
            f"{final_world.GetRow3(3)[1]:.1f}, "
            f"{final_world.GetRow3(3)[2]:.1f})"
        )

        # ---- Update records (fix stale paths for children moving with parent) ----
        old_child_str = str(child_path)
        new_child_str = str(new_child_path)
        record.child_path = new_child_str
        # Remove old record if the path changed
        self._records.pop(old_child_str, None)
        self._records[new_child_str] = record
        # Update any descendant records whose paths moved with this reparent
        self._rebase_descendant_records(old_child_str, new_child_str)

        self._log(
            f"[MAGIC] combine OK: '{child_name}' -> '{parent_name}'"
            f" | plug={plug_name or 'origin'}"
            f" | socket={socket_name or 'origin'}"
            f" | new_path={new_child_path}"
        )
        return True

    def _separate_resolved_path(self, stage, part_path) -> bool:
        """Detach a resolved prim path from its current assembly parent."""
        import zlib

        from pxr import UsdGeom, Gf, Sdf  # type: ignore

        record = self._records.get(str(part_path))

        part_prim = stage.GetPrimAtPath(part_path)
        if part_prim is None or not part_prim.IsValid():
            self._log(f"[MAGIC] separate FAIL: invalid prim at {part_path}")
            return False
        tc = self._time_code(stage)
        current_world = self._world_xform(part_prim, tc)

        target_parent_path = self._resolve_detach_target_parent_path(stage, part_prim, record)
        if target_parent_path is None:
            self._log(f"[MAGIC] separate: '{part_path}' has no assembly parent")
            return False

        # Do not restore pre-combine world poses. Disseminated parts should come
        # back as loose parts near the root origin, not jump to their old scene
        # positions. Keep the current orientation/scale, only replace translation
        # with a deterministic scatter offset so multiple detached parts remain
        # visible and separable.
        target_world = Gf.Matrix4d(current_world)
        scatter_key = str(part_path if record is None else record.child_path)
        scatter_idx = zlib.crc32(scatter_key.encode("utf-8")) % 25
        scatter_x = (scatter_idx % 5 - 2) * 120.0
        scatter_y = ((scatter_idx // 5) - 2) * 120.0
        scatter_z = 0.0
        target_world.SetRow3(3, Gf.Vec3d(scatter_x, scatter_y, scatter_z))

        # Get target parent world for computing new local transform
        if target_parent_path == Sdf.Path.absoluteRootPath:
            target_parent_world = Gf.Matrix4d(1.0)
        else:
            tpp = stage.GetPrimAtPath(target_parent_path)
            target_parent_world = (
                self._world_xform(tpp, tc)
                if (tpp and tpp.IsValid())
                else Gf.Matrix4d(1.0)
            )
        new_local = target_parent_world.GetInverse() * target_world

        # Reparent unless the part is already under the target parent.
        current_parent_path = part_prim.GetParent().GetPath()
        if current_parent_path == target_parent_path:
            new_path = part_path
        else:
            new_path = self._reparent(stage, part_path, target_parent_path)
            if new_path is None:
                self._log(f"[MAGIC] separate: reparent failed for {part_path}")
                return False

        new_prim = stage.GetPrimAtPath(new_path)
        if new_prim and new_prim.IsValid():
            self._set_xform_matrix(new_prim, new_local)
            # Re-enable RigidBodyAPI now that the part is standalone again
            self._enable_rigid_body(new_prim)

        # Clean up record (try both old and new paths in case of stale key)
        self._records.pop(str(part_path), None)
        self._records.pop(str(new_path), None)
        self._log(f"[MAGIC] separate OK: '{part_path}' restored to '{target_parent_path}'")
        return True

    def _resolve_detach_target_parent_path(self, stage, part_prim, record):
        """Return the parent path a detached part should live under."""
        from pxr import Sdf  # type: ignore

        if record is not None:
            return Sdf.Path(record.pre_combine_parent)

        parent_prim = part_prim.GetParent()
        if parent_prim is None or parent_prim.IsPseudoRoot():
            return None

        default_prim = stage.GetDefaultPrim()
        if (
            default_prim
            and default_prim.IsValid()
            and parent_prim.GetPath() == default_prim.GetPath()
        ):
            return parent_prim.GetPath()

        grandparent = parent_prim.GetParent()
        if grandparent is None or grandparent.IsPseudoRoot():
            return Sdf.Path.absoluteRootPath
        return grandparent.GetPath()

    def separate(self, part_name: str) -> bool:
        """Disseminate an assembled prim by detaching its tracked descendants first."""
        from pxr import Sdf  # type: ignore

        self._log(f"[MAGIC] separate START: part={part_name!r}")

        stage = self._stage_fn()
        if stage is None:
            self._log("[MAGIC] separate FAIL: stage not available")
            return False

        part_path = self._find_prim_path(stage, part_name)
        if part_path is None:
            self._log(f"[MAGIC] separate FAIL: part '{part_name}' not found")
            return False

        self._log(f"[MAGIC] separate: resolved path={part_path}")
        prefix = str(part_path).rstrip("/") + "/"
        separated_any = False
        while True:
            descendant_paths = [
                Sdf.Path(path)
                for path in self._records
                if str(path).startswith(prefix)
            ]
            if not descendant_paths:
                break
            descendant_paths.sort(key=lambda p: len(p.pathString.split("/")), reverse=True)
            progressed = False
            for child_path in descendant_paths:
                if self._separate_resolved_path(stage, child_path):
                    separated_any = True
                    progressed = True
            if not progressed:
                self._log(
                    f"[MAGIC] separate WARN: no progress while disseminating descendants of '{part_name}'"
                )
                break

        if str(part_path) in self._records:
            return self._separate_resolved_path(stage, part_path)

        part_prim = stage.GetPrimAtPath(part_path)
        if part_prim and part_prim.IsValid():
            parent_prim = part_prim.GetParent()
            default_prim = stage.GetDefaultPrim()
            if parent_prim is None or parent_prim.IsPseudoRoot():
                if separated_any:
                    self._log(
                        f"[MAGIC] separate OK: disseminated descendants of '{part_name}'"
                    )
                    return True
                self._log(f"[MAGIC] separate: '{part_path}' has no assembly parent")
                return False
            if (
                default_prim
                and default_prim.IsValid()
                and parent_prim.GetPath() == default_prim.GetPath()
            ):
                if separated_any:
                    self._log(
                        f"[MAGIC] separate OK: disseminated descendants of '{part_name}'"
                    )
                    return True
                self._log(f"[MAGIC] separate: '{part_name}' is already loose under default root")
                return False

        ok = self._separate_resolved_path(stage, part_path)
        return ok or separated_any

    def focus(self, part_name: str) -> bool:
        """Move a part, or its containing assembly, so the part lands at world (0, -0.85, current_z)."""
        from pxr import Gf  # type: ignore

        self._log(f"[MAGIC] focus START: part={part_name!r}")

        stage = self._stage_fn()
        if stage is None:
            self._log("[MAGIC] focus FAIL: stage not available")
            return False

        part_path = self._find_prim_path(stage, part_name)
        if part_path is None:
            self._log(f"[MAGIC] focus FAIL: part '{part_name}' not found")
            return False

        self._log(f"[MAGIC] focus: resolved path={part_path}")

        part_prim = stage.GetPrimAtPath(part_path)
        if part_prim is None or not part_prim.IsValid():
            self._log(f"[MAGIC] focus FAIL: invalid prim at {part_path}")
            return False

        focus_path = self._resolve_focus_target_path(stage, part_path)
        focus_prim = stage.GetPrimAtPath(focus_path)
        if focus_prim is None or not focus_prim.IsValid():
            self._log(f"[MAGIC] focus FAIL: focus target invalid at {focus_path}")
            return False

        tc = self._time_code(stage)
        part_world = self._world_xform(part_prim, tc)
        focus_world = self._world_xform(focus_prim, tc)
        part_world_row = part_world.GetRow3(3)
        focus_world_row = focus_world.GetRow3(3)

        delta_x = float(-float(part_world_row[0]))
        delta_y = float(-0.85 - float(part_world_row[1]))
        target_world = Gf.Matrix4d(focus_world)
        target_world.SetRow3(
            3,
            Gf.Vec3d(
                float(focus_world_row[0]) + delta_x,
                float(focus_world_row[1]) + delta_y,
                float(focus_world_row[2]),
            ),
        )

        focus_parent = focus_prim.GetParent()
        if focus_parent is None or focus_parent.IsPseudoRoot():
            target_parent_world = Gf.Matrix4d(1.0)
        else:
            target_parent_world = self._world_xform(focus_parent, tc)
        new_local = target_parent_world.GetInverse() * target_world
        self._set_xform_matrix(focus_prim, new_local)

        self._log(
            f"[MAGIC] focus OK: part='{part_name}' target='{focus_path}' "
            f"-> part_world (0.0, -0.85, {float(part_world_row[2]):.3f})"
        )
        return True

    def _resolve_focus_target_path(self, stage, part_path):
        """Return the top-level assembly node that should move when focusing a part."""
        focus_prim = stage.GetPrimAtPath(part_path)
        if focus_prim is None or not focus_prim.IsValid():
            return part_path

        default_prim = stage.GetDefaultPrim()
        while True:
            parent_prim = focus_prim.GetParent()
            if parent_prim is None or parent_prim.IsPseudoRoot():
                return focus_prim.GetPath()
            if default_prim and default_prim.IsValid() and parent_prim.GetPath() == default_prim.GetPath():
                return focus_prim.GetPath()
            focus_prim = parent_prim

    # ------------------------------------------------------------------
    # Casing orientation helper
    # ------------------------------------------------------------------

    def flip_casing_base(self) -> bool:
        """Rotate Casing_Base by 180° around the X-axis so inside faces upward (Z+ up).

        This corrects for a manually-flipped casing base whose Z+ currently
        points downward.  After the flip the cavity opening faces upward and
        positive Z offsets move parts UP into the cavity.
        """
        from pxr import UsdGeom, Gf  # type: ignore

        stage = self._stage_fn()
        if stage is None:
            self._log("[MAGIC] flip_casing_base: no stage")
            return False

        path = self._find_prim_path(stage, "Casing_Base")
        if path is None:
            self._log("[MAGIC] flip_casing_base: Casing_Base not found")
            return False

        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        tc = UsdGeom.XformCache()
        current = tc.GetLocalToWorldTransform(prim)

        # Apply 180° rotation about X to flip Z direction
        flip = Gf.Matrix4d()
        flip.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0))
        # Preserve current translation
        flip.SetTranslateOnly(current.ExtractTranslation())

        self._set_xform_matrix(prim, flip)
        self._log("[MAGIC] flip_casing_base: Casing_Base flipped — inside now faces upward (Z+)")
        return True

    # ------------------------------------------------------------------
    # Sub-assembly helpers
    # ------------------------------------------------------------------

    def assemble_subassemblies(self) -> List[bool]:
        """Build the three gear-shaft sub-assemblies (PDF steps pre-11).

        1. Output_Gear  → Output_Shaft   (socket_gear)
        2. Transfer_Gear → Transfer_Shaft (socket_gear)
        3. Input_Shaft is already one piece — no combine needed.

        Returns a list of bool results for each combine call.
        """
        results = []
        for child, parent in [
            ("Output_Gear", "Output_Shaft"),
            ("Transfer_Gear", "Transfer_Shaft"),
        ]:
            self._log(f"[MAGIC] assemble_subassemblies: {child} → {parent}")
            ok = self.combine(child, parent, "plug_main", "socket_gear")
            results.append(ok)
        return results

    # ------------------------------------------------------------------
    # Assembly state query
    # ------------------------------------------------------------------

    def get_assembly_belief(self) -> Dict[str, Any]:
        """Return a belief_update dict reflecting current assembly state.

        Suitable for merging into the AgentState belief_update dict that the
        VLM worker already produces.

        Example output::

            {
                "objects": {
                    "InputShaft": {
                        "belief_status": "attached",
                        "attached_to": "CasingBase",
                        "confidence": 1.0,
                    }
                }
            }
        """
        result: Dict[str, Any] = {"objects": {}}
        for child_path, rec in self._records.items():
            child_name  = child_path.rstrip("/").split("/")[-1]
            parent_name = rec.parent_path.rstrip("/").split("/")[-1]
            result["objects"][child_name] = {
                "belief_status": "attached",
                "attached_to": parent_name,
                "confidence": 1.0,
            }
        return result

    def list_assemblies(self) -> List[Tuple[str, str]]:
        """Return list of (child_name, parent_name) pairs for all active attachments."""
        out = []
        for child_path, rec in self._records.items():
            c = child_path.rstrip("/").split("/")[-1]
            p = rec.parent_path.rstrip("/").split("/")[-1]
            out.append((c, p))
        return out

    def ensure_extra_hub_bolt_assets(self) -> Dict[str, int]:
        """Runtime hub-bolt asset mutation disabled after revert."""
        return {"bolts": 0, "sockets": 0}

    def ensure_case_attachment_assets(self) -> Dict[str, int]:
        """Runtime case asset mutation disabled after revert."""
        return {
            "bolts": 0,
            "oils": 0,
            "top_sockets": 0,
            "base_alias_sockets": 0,
            "base_gear_sockets": 0,
        }

    def ensure_bearing_parts_prepared(self) -> Dict[str, int]:
        """Bearing preparation is disabled after revert to avoid mutating stage assets."""
        self._log("[MAGIC] ensure_bearing_parts_prepared: disabled after revert")
        return {
            "materialized": 0,
            "cleaned": 0,
            "recentered": 0,
            "plugs": 0,
            "shaft_sockets": 0,
            "removed_sockets": 0,
        }

    def ensure_shaft_gear_sockets(self) -> int:
        """Bearing socket authoring disabled after revert."""
        return 0

    def remove_bearing_shaft_sockets(self) -> int:
        """Bearing socket cleanup disabled after revert."""
        return 0

    def _materialize_bearing_payload(self, prim) -> bool:
        """Replace a payload-authored bearing root with local scene specs."""
        from pxr import Sdf  # type: ignore

        if not prim or not prim.IsValid() or not prim.HasAuthoredPayloads():
            return False

        stage = self._stage_fn()
        if stage is None:
            return False

        payload_meta = prim.GetMetadata("payload")
        payload_items = []
        if payload_meta is not None:
            for getter in ("GetExplicitItems", "GetAppliedItems"):
                fn = getattr(payload_meta, getter, None)
                if fn is None:
                    continue
                payload_items.extend(list(fn() or []))
        if not payload_items:
            return False

        asset_path = str(payload_items[0].assetPath or "").strip()
        if not asset_path:
            return False

        root_layer = stage.GetRootLayer()
        root_real_path = os.path.dirname(os.path.abspath(root_layer.realPath or root_layer.identifier or "."))
        source_path = asset_path if os.path.isabs(asset_path) else os.path.normpath(os.path.join(root_real_path, asset_path))
        source_layer = Sdf.Layer.FindOrOpen(source_path)
        if source_layer is None:
            self._log(f"[MAGIC] bearing materialize FAIL: could not open payload {source_path}")
            return False

        for child_name in ("Looks", "node_"):
            source_spec = Sdf.Path(f"/World/{child_name}")
            if not source_layer.GetPrimAtPath(source_spec):
                self._log(f"[MAGIC] bearing materialize FAIL: {source_path} missing {source_spec}")
                return False

        prim.GetPayloads().ClearPayloads()
        dest_layer = root_layer
        for child_name in ("Looks", "node_"):
            dst_path = prim.GetPath().AppendChild(child_name)
            existing = stage.GetPrimAtPath(dst_path)
            if existing and existing.IsValid():
                stage.RemovePrim(dst_path)
            Sdf.CopySpec(source_layer, Sdf.Path(f"/World/{child_name}"), dest_layer, dst_path)
        return True

    def _recenter_node_local_bbox(self, node_prim, tc) -> bool:
        """Shift `node_` so descendant mesh geometry is centered at the root origin."""
        from pxr import Gf, Usd, UsdGeom  # type: ignore

        if node_prim is None or not node_prim.IsValid():
            return False

        root_prim = node_prim.GetParent()
        if root_prim is None or not root_prim.IsValid():
            return False

        root_xf = UsdGeom.Xformable(root_prim)
        node_xf = UsdGeom.Xformable(node_prim)

        root_world = root_xf.ComputeLocalToWorldTransform(tc)
        root_world_inv = root_world.GetInverse()

        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]
        saw_points = False
        for prim in Usd.PrimRange(node_prim):
            if prim.GetTypeName() != "Mesh":
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get(tc) or mesh.GetPointsAttr().Get()
            if not points:
                continue
            mesh_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc)
            mesh_to_root = mesh_world * root_world_inv
            for point in points:
                p = mesh_to_root.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                for axis, value in enumerate((float(p[0]), float(p[1]), float(p[2]))):
                    mins[axis] = min(mins[axis], value)
                    maxs[axis] = max(maxs[axis], value)
                saw_points = True

        if not saw_points:
            return False

        centroid = Gf.Vec3d(
            0.5 * (mins[0] + maxs[0]),
            0.5 * (mins[1] + maxs[1]),
            0.5 * (mins[2] + maxs[2]),
        )
        if centroid.GetLength() < 1e-6:
            return False

        local_mat = node_xf.GetLocalTransformation(tc)
        tmp_stage = Usd.Stage.CreateInMemory()
        tmp_prim = UsdGeom.Xform.Define(tmp_stage, "/_decompose").GetPrim()
        tmp_xf = UsdGeom.Xformable(tmp_prim)
        tmp_xf.ClearXformOpOrder()
        tmp_xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_mat)
        api = UsdGeom.XformCommonAPI(tmp_prim)
        translate, rotate_xyz, scale, _pivot, _order = api.GetXformVectorsByAccumulation(Usd.TimeCode.Default())
        new_translate = Gf.Vec3d(
            float(translate[0]) - float(centroid[0]),
            float(translate[1]) - float(centroid[1]),
            float(translate[2]) - float(centroid[2]),
        )
        self._set_xform_trs(
            node_prim,
            translate=(float(new_translate[0]), float(new_translate[1]), float(new_translate[2])),
            rotate_xyz=(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])),
            scale=(float(scale[0]), float(scale[1]), float(scale[2])),
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_stage():
        """Return the active stage via omni.usd (Isaac Sim runtime only)."""
        try:
            import omni.usd  # type: ignore
            return omni.usd.get_context().get_stage()
        except Exception:
            return None

    def _find_prim_path(self, stage, name: str):
        """Search the stage for a prim whose name matches *name*.

        Accepts:
        - Absolute USD paths  (e.g. ``"/World/CasingBase"``)
        - Plain names         (e.g. ``"CasingBase"``)

        Disambiguation rules for repeated names (e.g. multiple M6_Hub_Bolt):
        1. Prefer the prim that is **not** already tracked in ``_records``
           (i.e. not already combined into an assembly).
        2. Among remaining candidates, prefer the **shallowest** path.

        Returns None when no match is found.
        """
        from pxr import Sdf  # type: ignore

        name = str(name).strip()
        if not name:
            return None

        # Exact absolute path lookup
        if name.startswith("/"):
            prim = stage.GetPrimAtPath(name)
            if prim and prim.IsValid():
                return Sdf.Path(name)

        # Search by last path component
        target = name.rstrip("/").split("/")[-1]
        matches = []
        for prim in stage.Traverse():
            if prim.GetName() == target:
                matches.append(prim.GetPath())

        if not matches:
            return None

        # Sort shallowest first (fewest '/' segments)
        matches.sort(key=lambda p: len(p.pathString.split("/")))

        if len(matches) == 1:
            return matches[0]

        # For duplicate names: prefer prims NOT already in an assembly record
        assembled_paths = set(self._records.keys())
        free = [m for m in matches if str(m) not in assembled_paths]
        if free:
            return free[0]

        # All instances assembled — return shallowest (caller may want to re-combine)
        return matches[0]

    def _find_socket_local(
        self,
        stage,
        parent_prim,
        parent_world,
        socket_name: Optional[str],
        tc,
        strict: bool = False,
    ):
        """Return the LOCAL transform of the named socket on parent_prim.

        The socket is a direct child Xform of parent_prim.  Its local transform
        is computed as:  socket_local = socket_world * inv(parent_world)
        (USD row-vector convention).

        Falls back to identity (parent's origin) if no socket found unless strict=True.
        """
        from pxr import UsdGeom, Gf  # type: ignore

        if socket_name:
            candidates = [socket_name]
        else:
            candidates = ["Socket", "socket", "Attach", "attach",
                          "SocketTop", "socket_top", "attach_point"]

        if socket_name:
            direct_child = parent_prim.GetChild(socket_name)
            if direct_child and direct_child.IsValid():
                xf = UsdGeom.Xformable(direct_child)
                if xf:
                    socket_local = xf.GetLocalTransformation(tc)
                    if isinstance(socket_local, tuple):
                        socket_local = socket_local[0]
                    return Gf.Matrix4d(socket_local)

            direct_path = parent_prim.GetPath().AppendChild(socket_name)
            direct_prim = stage.GetPrimAtPath(direct_path)
            if direct_prim and direct_prim.IsValid():
                xf = UsdGeom.Xformable(direct_prim)
                if xf:
                    socket_local = xf.GetLocalTransformation(tc)
                    if isinstance(socket_local, tuple):
                        socket_local = socket_local[0]
                    return Gf.Matrix4d(socket_local)

        for child in parent_prim.GetChildren():
            cname = child.GetName()
            for candidate in candidates:
                if cname == candidate or cname.startswith(candidate):
                    xf = UsdGeom.Xformable(child)
                    if xf:
                        socket_world = xf.ComputeLocalToWorldTransform(tc)
                        return socket_world * parent_world.GetInverse()

        if socket_name:
            self._log(
                f"[MAGIC] socket '{socket_name}' not found on '{parent_prim.GetName()}'; "
                f"using parent origin"
            )
            if strict:
                return None
        return Gf.Matrix4d(1.0)

    def _resolve_auto_socket_name(
        self,
        parent_prim,
        parent_path: str,
        socket_hint: Optional[str],
    ) -> Optional[str]:
        """Choose an auto socket, preferring unoccupied sockets on the same parent."""
        socket_children = []
        for child in parent_prim.GetChildren():
            name = child.GetName()
            if name.startswith("socket_"):
                socket_children.append(name)
        if not socket_children:
            return None

        def _natural_key(name: str):
            m = re.search(r"(\d+)$", name)
            if m:
                return (name[: m.start()], int(m.group(1)))
            return (name, -1)

        # Build candidate list from hint family (if any), else all sockets.
        candidates: List[str] = []
        if socket_hint:
            if socket_hint in socket_children:
                candidates.append(socket_hint)
            m = re.match(r"^(.*_)\d+$", socket_hint)
            if m:
                family_prefix = m.group(1)
                family = sorted(
                    [s for s in socket_children if s.startswith(family_prefix)],
                    key=_natural_key,
                )
                candidates.extend(family)
        if not candidates:
            candidates = sorted(socket_children, key=_natural_key)

        # De-duplicate while preserving order.
        seen = set()
        ordered_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered_candidates.append(c)

        # Prefer sockets not already occupied by current attachments under this parent.
        occupied = {
            str(rec.socket_name)
            for rec in self._records.values()
            if str(rec.parent_path) == str(parent_path) and rec.socket_name
        }
        for c in ordered_candidates:
            if c not in occupied:
                return c
        return ordered_candidates[0] if ordered_candidates else None

    def _find_plug_local(
        self,
        stage,
        child_prim,
        child_world,
        tc,
        plug_name: Optional[str] = None,
        strict: bool = False,
    ):
        """Return the LOCAL transform of the plug xform on child_prim.

        The plug is a direct child Xform of child_prim.  Its local transform
        is computed as:  plug_local = plug_world * inv(child_world)
        (USD row-vector convention).

        Returns identity (child origin is attachment point) when none found unless strict=True.
        """
        from pxr import UsdGeom, Gf  # type: ignore

        if plug_name:
            for grandchild in child_prim.GetChildren():
                if grandchild.GetName() != plug_name:
                    continue
                xf = UsdGeom.Xformable(grandchild)
                if xf:
                    plug_world = xf.ComputeLocalToWorldTransform(tc)
                    return plug_world * child_world.GetInverse()
            self._log(
                f"[MAGIC] plug '{plug_name}' not found on '{child_prim.GetName()}'"
            )
            if strict:
                return None

        for grandchild in child_prim.GetChildren():
            gname = grandchild.GetName()
            for candidate in ("plug_main", "plug_casing_mate",
                              "Plug", "plug", "PlugBottom", "plug_bottom",
                              "Attach", "attach", "attach_origin"):
                if gname == candidate or gname.startswith(candidate):
                    xf = UsdGeom.Xformable(grandchild)
                    if xf:
                        plug_world = xf.ComputeLocalToWorldTransform(tc)
                        return plug_world * child_world.GetInverse()

        if strict:
            return None
        return Gf.Matrix4d(1.0)

    def _resolve_fit_offset(
        self,
        child_name: str,
        parent_name: str,
        plug_name: str,
        socket_name: str,
    ):
        """Return hardcoded socket-local correction matrix for known tricky pairs."""
        from pxr import Gf  # type: ignore

        # Normalize instance suffixes for lookup (e.g., M10_Casing_Bolt_01 -> M10_Casing_Bolt).
        c_name = _canonical_child_name(child_name)
        p_name = re.sub(r"_\d+$", "", str(parent_name or ""))
        s_name = str(socket_name or "").strip()
        plug_str = str(plug_name or "").strip()
        key = (c_name, p_name, plug_str, s_name)
        cfg = HARDCODED_FIT_OFFSETS.get(key)
        if not cfg and plug_str:
            cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, "", s_name))
        if not cfg and s_name.startswith("socket_bolt_hub_"):
            cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, plug_str, "socket_bolt_hub_*"))
            if not cfg and plug_str:
                cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, "", "socket_bolt_hub_*"))
        if not cfg and s_name.startswith("socket_bolt_casing_"):
            cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, plug_str, "socket_bolt_casing_*"))
            if not cfg and plug_str:
                cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, "", "socket_bolt_casing_*"))
        if not cfg and s_name.startswith("socket_nut_casing_"):
            cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, plug_str, "socket_nut_casing_*"))
            if not cfg and plug_str:
                cfg = HARDCODED_FIT_OFFSETS.get((c_name, p_name, "", "socket_nut_casing_*"))
        if not cfg:
            self._log(
                f"[MAGIC] fit-offset MISS: key={key!r}  "
                f"available keys with child={c_name!r}: "
                f"{[k for k in HARDCODED_FIT_OFFSETS if k[0]==c_name]}"
            )
            return Gf.Matrix4d(1.0)

        t = cfg.get("translate", (0.0, 0.0, 0.0))
        r = cfg.get("rotate_xyz", (0.0, 0.0, 0.0))
        rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
        rot = (
            Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rx)
            * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), ry)
            * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rz)
        )
        out = Gf.Matrix4d(1.0)
        out.SetRotate(rot)
        out.SetTranslateOnly(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
        self._log(
            "[MAGIC] fit-offset applied: "
            f"({c_name}->{p_name}, plug={plug_name}, socket={socket_name}) "
            f"T={tuple(float(v) for v in t)} Rxyz={tuple(float(v) for v in r)}"
        )
        return out

    def _apply_child_local_overrides(
        self,
        *,
        child_name: str,
        parent_name: str,
        plug_name: str,
        socket_name: str,
        child_local_new,
    ):
        """Apply hard child-local translation/phase overrides for specific pairs."""
        from pxr import Gf  # type: ignore

        c_name = _canonical_child_name(child_name)
        p_name = re.sub(r"_\d+$", "", str(parent_name or ""))
        s_name = str(socket_name or "").strip()
        key = (c_name, p_name, str(plug_name or "").strip(), s_name)
        z_key = key
        rot_key = key
        if s_name.startswith("socket_bolt_hub_"):
            wildcard_key = (c_name, p_name, str(plug_name or "").strip(), "socket_bolt_hub_*")
            if wildcard_key in HARDCODED_CHILD_LOCAL_Z:
                z_key = wildcard_key
            if wildcard_key in HARDCODED_CHILD_LOCAL_ROTATE_Y:
                rot_key = wildcard_key
        if s_name.startswith("socket_bolt_casing_"):
            wildcard_key = (c_name, p_name, str(plug_name or "").strip(), "socket_bolt_casing_*")
            if wildcard_key in HARDCODED_CHILD_LOCAL_Z:
                z_key = wildcard_key
            if wildcard_key in HARDCODED_CHILD_LOCAL_ROTATE_Y:
                rot_key = wildcard_key
        out = Gf.Matrix4d(child_local_new)
        if key in HARDCODED_CHILD_LOCAL_TRANSLATE:
            target_tx, target_ty, target_tz = HARDCODED_CHILD_LOCAL_TRANSLATE[key]
            old_tx = float(out.GetRow3(3)[0])
            old_ty = float(out.GetRow3(3)[1])
            old_tz = float(out.GetRow3(3)[2])
            out.SetTranslateOnly(Gf.Vec3d(float(target_tx), float(target_ty), float(target_tz)))
            self._log(
                f"[MAGIC] child-local translate override: key={key} -> "
                f"T=({target_tx:.3f}, {target_ty:.3f}, {target_tz:.3f}) "
                f"(was ({old_tx:.3f}, {old_ty:.3f}, {old_tz:.3f}))"
            )
            return out
        if z_key in HARDCODED_CHILD_LOCAL_Z:
            target_z = float(HARDCODED_CHILD_LOCAL_Z[z_key])
            tx = float(out.GetRow3(3)[0])
            ty = float(out.GetRow3(3)[1])
            old_tz = float(out.GetRow3(3)[2])
            out.SetTranslateOnly(Gf.Vec3d(tx, ty, target_z))
            self._log(
                f"[MAGIC] child-local override: key={z_key} -> Tz={target_z:.3f} "
                f"(was {old_tz:.3f})"
            )

        if rot_key in HARDCODED_CHILD_LOCAL_ROTATE_Y:
            phase_deg = float(HARDCODED_CHILD_LOCAL_ROTATE_Y[rot_key])
            phase = Gf.Matrix4d(1.0)
            # Row-vector convention: left-multiply applies local-space phase
            # while preserving existing translation.
            phase.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), phase_deg))
            out = phase * out
            self._log(
                f"[MAGIC] child-local phase override: key={rot_key} -> Ry={phase_deg:.3f}deg"
            )

        return out

    def _apply_casing_top_clearance(
        self,
        *,
        stage,
        child_prim,
        parent_prim,
        child_local_new,
        parent_world,
        tc,
    ):
        """Lift Casing_Top if needed so it rests above the assembled internal stack."""
        from pxr import Gf, UsdGeom  # type: ignore

        if child_prim.GetName() != "Casing_Top" or parent_prim.GetName() != "Casing_Base":
            return child_local_new

        try:
            bbox_cache = UsdGeom.BBoxCache(
                tc,
                includedPurposes=[UsdGeom.Tokens.default_],
            )
            child_local_range = bbox_cache.ComputeLocalBound(child_prim).ComputeAlignedRange()
            child_min_local_z = float(child_local_range.GetMin()[2])
            child_world_after = Gf.Matrix4d(child_local_new) * parent_world
            current_child_world_min_z = float(child_world_after.GetRow3(3)[2]) + child_min_local_z

            parent_prefix = str(parent_prim.GetPath()).rstrip("/") + "/"
            max_internal_z = None
            for prim in stage.Traverse():
                prim_name = prim.GetName()
                prim_path = str(prim.GetPath())
                if not prim_path.startswith(parent_prefix):
                    continue
                if prim_name.startswith("socket_") or prim_name.startswith("plug_"):
                    continue
                if prim_path == str(parent_prim.GetPath()):
                    continue
                if prim_name not in CASING_BASE_INTERNAL_PART_HINTS:
                    continue
                rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                prim_max_z = float(rng.GetMax()[2])
                if max_internal_z is None or prim_max_z > max_internal_z:
                    max_internal_z = prim_max_z

            if max_internal_z is None:
                return child_local_new

            clearance_world = 0.001  # 1 mm in the scene's meter-scale units
            extra_lift = max(0.0, (max_internal_z + clearance_world) - current_child_world_min_z)
            if extra_lift <= 0.0:
                return child_local_new

            out = Gf.Matrix4d(child_local_new)
            row = out.GetRow3(3)
            out.SetTranslateOnly(Gf.Vec3d(float(row[0]), float(row[1]), float(row[2]) + extra_lift))
            self._log(
                "[MAGIC] casing-top clearance lift applied: "
                f"lift={extra_lift:.3f}mm max_internal_z={max_internal_z:.3f} "
                f"child_min_z_before={current_child_world_min_z:.3f}"
            )
            return out
        except Exception as exc:
            self._log(f"[MAGIC] casing-top clearance warning: {exc}")
            return child_local_new

    def _world_xform(self, prim, tc):
        """Return world transform of prim, or identity if not xformable."""
        from pxr import UsdGeom, Gf  # type: ignore
        if prim is None or not prim.IsValid():
            return Gf.Matrix4d(1.0)
        xf = UsdGeom.Xformable(prim)
        return xf.ComputeLocalToWorldTransform(tc)

    def _bake_clean_trs(self, prim, tc) -> bool:
        """Flatten an imported xform stack into clean translate/rotate/scale ops."""
        from pxr import Gf, Usd, UsdGeom  # type: ignore

        if prim is None or not prim.IsValid():
            return False
        xf = UsdGeom.Xformable(prim)

        ops = xf.GetOrderedXformOps()
        op_names = [op.GetOpName() for op in ops]
        if op_names == ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]:
            return False

        local_mat = xf.GetLocalTransformation(tc)

        tmp_stage = Usd.Stage.CreateInMemory()
        tmp_prim = UsdGeom.Xform.Define(tmp_stage, "/_decompose").GetPrim()
        tmp_xf = UsdGeom.Xformable(tmp_prim)
        tmp_xf.ClearXformOpOrder()
        tmp_xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_mat)
        api = UsdGeom.XformCommonAPI(tmp_prim)
        translate, rotate_xyz, scale, _pivot, _order = api.GetXformVectorsByAccumulation(Usd.TimeCode.Default())

        xf.ClearXformOpOrder()
        for attr in [a.GetName() for a in prim.GetAttributes() if a.GetName().startswith("xformOp:")]:
            prim.RemoveProperty(attr)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
        )
        xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2]))
        )
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
        )
        return True

    def _decompose_matrix_to_trs(self, matrix):
        """Decompose a local matrix into TRS values using USD's own accumulation logic."""
        from pxr import Gf, Usd, UsdGeom  # type: ignore

        tmp_stage = Usd.Stage.CreateInMemory()
        tmp_prim = UsdGeom.Xform.Define(tmp_stage, "/_decompose").GetPrim()
        tmp_xf = UsdGeom.Xformable(tmp_prim)
        tmp_xf.ClearXformOpOrder()
        tmp_xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(matrix))
        api = UsdGeom.XformCommonAPI(tmp_prim)
        translate, rotate_xyz, scale, _pivot, _order = api.GetXformVectorsByAccumulation(
            Usd.TimeCode.Default()
        )
        return (
            (float(translate[0]), float(translate[1]), float(translate[2])),
            (float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])),
            (float(scale[0]), float(scale[1]), float(scale[2])),
        )

    def _matrix_matches_trs(
        self,
        matrix,
        *,
        translate: Tuple[float, float, float],
        rotate_xyz: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        tol: float = 1e-4,
    ) -> bool:
        """Return True when clean TRS re-composes back to the original matrix."""
        from pxr import Gf, Usd, UsdGeom  # type: ignore

        tmp_stage = Usd.Stage.CreateInMemory()
        tmp_prim = UsdGeom.Xform.Define(tmp_stage, "/_recompose").GetPrim()
        tmp_xf = UsdGeom.Xformable(tmp_prim)
        tmp_xf.ClearXformOpOrder()
        api = UsdGeom.XformCommonAPI(tmp_prim)
        api.SetTranslate(Gf.Vec3d(*translate))
        api.SetRotate(
            Gf.Vec3f(*rotate_xyz),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        api.SetScale(Gf.Vec3f(*scale))
        recomposed = tmp_xf.GetLocalTransformation(Usd.TimeCode.Default())
        if isinstance(recomposed, tuple):
            recomposed = recomposed[0]

        lhs = Gf.Matrix4d(matrix)
        rhs = Gf.Matrix4d(recomposed)
        for row in range(4):
            for col in range(4):
                if abs(float(lhs[row][col]) - float(rhs[row][col])) > tol:
                    return False
        return True

    def _apply_exact_authored_rotation_if_needed(
        self,
        prim,
        *,
        child_name: str,
        parent_name: str,
        plug_name: str,
        socket_name: str,
    ) -> None:
        """Force a specific editable rotateXYZ triplet for selected combine results."""
        from pxr import Usd, UsdGeom  # type: ignore

        if prim is None or not prim.IsValid():
            return
        key = (
            _canonical_child_name(child_name),
            re.sub(r"_\d+$", "", str(parent_name or "")),
            str(plug_name or "").strip(),
            str(socket_name or "").strip(),
        )
        rotate_xyz = HARDCODED_CHILD_LOCAL_ROTATE_XYZ.get(key)
        if rotate_xyz is None:
            return

        xf = UsdGeom.Xformable(prim)
        local_mat = xf.GetLocalTransformation(Usd.TimeCode.Default())
        translate, _ignored_rotate, scale = self._decompose_matrix_to_trs(local_mat)
        self._set_xform_trs(
            prim,
            translate=translate,
            rotate_xyz=rotate_xyz,
            scale=scale,
        )
        self._log(
            f"[MAGIC] exact authored rotate override: key={key} -> "
            f"Rxyz=({rotate_xyz[0]:.3f}, {rotate_xyz[1]:.3f}, {rotate_xyz[2]:.3f})"
        )

    def _set_xform_matrix(self, prim, matrix) -> None:
        """Author the solved local pose, preferring clean TRS over a raw matrix op."""
        from pxr import UsdGeom, Gf  # type: ignore
        if prim is None or not prim.IsValid():
            return
        try:
            translate, rotate_xyz, scale = self._decompose_matrix_to_trs(matrix)
            if self._matrix_matches_trs(
                matrix,
                translate=translate,
                rotate_xyz=rotate_xyz,
                scale=scale,
            ):
                self._set_xform_trs(
                    prim,
                    translate=translate,
                    rotate_xyz=rotate_xyz,
                    scale=scale,
                )
                return
        except Exception as exc:
            self._log(
                f"[MAGIC] xform decompose warning for {prim.GetPath()}: {exc}; "
                "falling back to raw matrix op"
            )

        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        op = xf.AddXformOp(
            UsdGeom.XformOp.TypeTransform,
            UsdGeom.XformOp.PrecisionDouble,
        )
        op.Set(Gf.Matrix4d(matrix))

    def _set_xform_trs(
        self,
        prim,
        *,
        translate: Tuple[float, float, float],
        rotate_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Author simple translate/rotate ops so manipulators stay predictable."""
        from pxr import Gf, UsdGeom  # type: ignore

        if prim is None or not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
        api.SetRotate(
            Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        api.SetScale(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

    def _reparent(self, stage, child_path, new_parent_path):
        """Move child_path to be a child of new_parent_path.

        Tries omni.kit.commands first (supports undo/redo inside Kit).
        Falls back to Sdf.BatchNamespaceEdit for standalone / test mode.

        Returns the new Sdf.Path of the child, or None on failure.
        """
        from pxr import Sdf  # type: ignore

        child_path      = Sdf.Path(str(child_path))
        new_parent_path = Sdf.Path(str(new_parent_path))
        new_child_path  = new_parent_path.AppendChild(child_path.name)

        # ---- Try omni.kit.commands (Kit runtime) ----
        if self._use_omni:
            try:
                import omni.kit.commands  # type: ignore
                omni.kit.commands.execute(
                    "MovePrims",
                    paths_to_move={str(child_path): str(new_child_path)},
                )
                prim = stage.GetPrimAtPath(new_child_path)
                if prim and prim.IsValid():
                    return new_child_path
                # Fall through if the move silently failed
            except Exception as exc:
                self._log(f"[MAGIC] omni MovePrims failed: {exc}; using Sdf fallback")

        # ---- Pure Sdf namespace edit (test / standalone mode) ----
        try:
            layer = stage.GetRootLayer()
            edit  = Sdf.BatchNamespaceEdit()
            edit.Add(
                Sdf.NamespaceEdit.ReparentAndRename(
                    str(child_path),
                    str(new_parent_path),
                    child_path.name,
                    -1,  # append at end
                )
            )
            if layer.Apply(edit):
                return new_child_path
            self._log(
                f"[MAGIC] Sdf namespace edit returned False: "
                f"{child_path} → {new_child_path}"
            )
            return None
        except Exception as exc:
            self._log(f"[MAGIC] reparent error: {exc}")
            return None

    def _disable_rigid_body(self, prim) -> None:
        """Disable RigidBodyAPI on *prim* so it doesn't conflict with parent rigid body.

        PhysX does not support nested rigid bodies — a dynamic RB as a child of
        another dynamic RB produces undefined behaviour.  We set the
        ``physics:rigidBodyEnabled`` attribute to False rather than removing the
        API schema, so that separate() can simply flip it back to True.
        """
        try:
            from pxr import UsdPhysics  # type: ignore
            rb = UsdPhysics.RigidBodyAPI.Get(prim.GetStage(), prim.GetPath())
            enabled_attr = rb.GetRigidBodyEnabledAttr()
            if enabled_attr and enabled_attr.IsValid():
                enabled_attr.Set(False)
                self._log(f"[MAGIC] disabled RigidBodyAPI on {prim.GetPath()}")
            elif "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas():
                rb = UsdPhysics.RigidBodyAPI.Apply(prim)
                rb.GetRigidBodyEnabledAttr().Set(False)
                self._log(f"[MAGIC] disabled RigidBodyAPI on {prim.GetPath()}")
        except Exception as exc:
            self._log(f"[MAGIC] _disable_rigid_body warning: {exc}")

    def _disable_rigid_body_recursive(self, prim) -> None:
        """Disable rigid bodies on prim and all descendants after reparenting."""
        if prim is None or not prim.IsValid():
            return
        try:
            prefix = str(prim.GetPath()).rstrip("/") + "/"
            for subprim in prim.GetStage().Traverse():
                subpath = str(subprim.GetPath())
                if subpath == str(prim.GetPath()) or subpath.startswith(prefix):
                    self._disable_rigid_body(subprim)
        except Exception as exc:
            self._log(f"[MAGIC] _disable_rigid_body_recursive warning: {exc}")

    def _enable_rigid_body(self, prim) -> None:
        """Re-enable RigidBodyAPI on *prim* after separation from assembly."""
        try:
            from pxr import UsdPhysics  # type: ignore
            rb = UsdPhysics.RigidBodyAPI.Get(prim.GetStage(), prim.GetPath())
            enabled_attr = rb.GetRigidBodyEnabledAttr()
            if enabled_attr and enabled_attr.IsValid():
                enabled_attr.Set(True)
                self._log(f"[MAGIC] re-enabled RigidBodyAPI on {prim.GetPath()}")
        except Exception as exc:
            self._log(f"[MAGIC] _enable_rigid_body warning: {exc}")

    def _rebase_descendant_records(self, old_prefix: str, new_prefix: str) -> None:
        """After reparenting, update any assembly records whose paths moved.

        When Casing_Top (with children Hub_Cover_*) is reparented from /World
        to /World/Casing_Base, the children's paths change from
        ``/World/Casing_Top/Hub_Cover_*`` to
        ``/World/Casing_Base/Casing_Top/Hub_Cover_*``.

        This method fixes those stale record keys.
        """
        prefix_slash = old_prefix.rstrip("/") + "/"
        stale_keys = [k for k in self._records if k.startswith(prefix_slash)]
        for old_key in stale_keys:
            rec = self._records.pop(old_key)
            new_key = new_prefix + old_key[len(old_prefix):]
            rec.child_path = new_key
            self._records[new_key] = rec
            self._log(f"[MAGIC] rebased record: {old_key} -> {new_key}")

    @staticmethod
    def _time_code(stage):
        """Return a safe time code for transform queries."""
        from pxr import Usd  # type: ignore
        try:
            return Usd.TimeCode.Default()
        except Exception:
            return None
