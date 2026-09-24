"""Render a labelled, conceptual overview of the five constrained pairs.

The figure is intentionally a schematic: it shows the changed access/retention
topology and the A/B order, not the final CAD geometry or a physics result.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _draw_pair(ax, pair: str, title: str, action_a: str, action_b: str, hard: str, commutable: str, kind: str):
    import matplotlib.patches as patches

    ax.axis("off")
    ax.text(0.01, 0.90, f"{pair}  {title}", va="top", fontsize=12, fontweight="bold", color="#0f172a")
    ax.text(0.01, 0.82, f"A  {action_a}", va="top", fontsize=9, color="#334155")
    ax.text(0.01, 0.74, f"B  {action_b}", va="top", fontsize=9, color="#334155")
    for x, label, body, color in (
        (0.52, "HARD", hard, "#fee2e2"),
        (0.77, "COMMUTABLE", commutable, "#dcfce7"),
    ):
        ax.add_patch(patches.FancyBboxPatch((x, 0.04), 0.21, 0.60, boxstyle="round,pad=0.012", facecolor=color, edgecolor="#94a3b8", linewidth=0.8))
        ax.text(x + 0.105, 0.59, label, ha="center", fontsize=9, fontweight="bold", color="#1e293b")
        ax.text(x + 0.105, 0.07, body, ha="center", va="bottom", fontsize=7.5, color="#334155", wrap=True)
    # A compact icon occupies the upper portion of each card.  Icons are
    # deliberately abstract and are paired with the precise dimensions in the
    # accompanying report/configuration.
    for x, variant in ((0.52, "HARD"), (0.77, "COMMUTABLE")):
        cx, cy = x + 0.105, 0.37
        if kind == "hole":
            ax.add_patch(patches.Circle((cx, cy), 0.075, facecolor="white", edgecolor="#475569", linewidth=1.3))
            if variant == "COMMUTABLE":
                ax.add_patch(patches.Circle((cx + 0.07, cy), 0.045, facecolor="white", edgecolor="#16a34a", linewidth=1.2))
                ax.plot([cx, cx + 0.07], [cy, cy], color="#16a34a", linewidth=5, solid_capstyle="round")
        elif kind == "slot":
            ax.add_patch(patches.Circle((cx, cy), 0.075, facecolor="white", edgecolor="#475569", linewidth=1.3))
            ax.add_patch(patches.Circle((cx, cy), 0.035, facecolor="#94a3b8", edgecolor="none"))
            if variant == "COMMUTABLE":
                ax.add_patch(patches.Rectangle((cx, cy - 0.012), 0.10, 0.024, facecolor="white", edgecolor="#16a34a", linewidth=1.2))
        elif kind == "key":
            ax.add_patch(patches.Rectangle((cx - 0.07, cy - 0.03), 0.14, 0.06, facecolor="#94a3b8", edgecolor="#475569", linewidth=1))
            if variant == "COMMUTABLE":
                ax.add_patch(patches.Rectangle((cx + 0.02, cy - 0.055), 0.065, 0.11, facecolor="white", edgecolor="#16a34a", linewidth=1.2))
        elif kind == "pocket":
            ax.add_patch(patches.Rectangle((cx - 0.07, cy - 0.04), 0.14, 0.08, facecolor="#94a3b8", edgecolor="#475569", linewidth=1))
            if variant == "COMMUTABLE":
                ax.add_patch(patches.Circle((cx, cy), 0.038, facecolor="white", edgecolor="#16a34a", linewidth=1.2))
        elif kind == "dowel":
            ax.add_patch(patches.Rectangle((cx - 0.08, cy - 0.05), 0.16, 0.10, facecolor="#94a3b8", edgecolor="#475569", linewidth=1))
            if variant == "COMMUTABLE":
                ax.add_patch(patches.Rectangle((cx - 0.085, cy - 0.02), 0.17, 0.04, facecolor="white", edgecolor="#16a34a", linewidth=1.2))


def render(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), dpi=140, facecolor="#f8fafc")
    specs = [
        ("HCF-01", "Hub cover → M6 headed bolt", "seat small cover", "retain M6 bolt", "round hole: 11.70 mm head cannot pass 6.80 mm opening", "keyhole lobe: 13.60 mm head path before final lock", "hole"),
        ("WSG-01", "Washer → output gear", "seat 1.00 mm washer", "seat output gear", "closed shroud covers washer shoulder", "2.00 mm radial slot leaves side access", "slot"),
        ("KEY-01", "Output key → output gear", "insert 29.70 mm key", "mount output gear", "gear skirt closes the only keyway entry", "21.00 mm side window remains reachable", "key"),
        ("CAS-01", "Casing closure → M10 through-bolt", "close casing halves", "seat M10 bolt", "split lugs form a bore only after closure", "18.00 mm captive pocket retains bolt during closure", "pocket"),
        ("DOW-01", "Casing closure → locating dowel", "close casing halves", "seat locating dowel", "two half-bores do not retain a dowel alone", "15.20 mm open slot retains 13.80 mm dowel", "dowel"),
    ]
    for ax, spec in zip(axes, specs):
        _draw_pair(ax, *spec)
    fig.suptitle("五个 constrained pair：A→B 示范相同，只改变 B→A 的可达性", fontsize=18, fontweight="bold", color="#0f172a", y=0.99)
    # The first subplot sits directly below the suptitle in tight_layout; repeat
    # its text in figure coordinates so the first pair cannot be clipped.
    fig.text(0.02, 0.925, "HCF-01  Hub cover → M6 headed bolt", fontsize=12, fontweight="bold", color="#0f172a")
    fig.text(0.02, 0.905, "A  seat small cover", fontsize=9, color="#334155")
    fig.text(0.02, 0.888, "B  retain M6 bolt", fontsize=9, color="#334155")
    fig.text(0.5, 0.008, "概念示意图，不是最终 CAD/碰撞场景；精确尺寸与当前实现状态见 five_pair_mutations.zh-CN.md", ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), h_pad=1.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("pilot_12pair/outputs/geometry_demo/five_pair_mutation_schematic.png"))
    args = parser.parse_args()
    render(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
