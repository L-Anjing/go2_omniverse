"""
Automatic semantic labeling for the Isaac Sim Hospital USD scene.

Traverses every prim, infers a semantic class from its name/path, and writes
the label using the standard USD Semantics API (same format that
omni.replicator / Isaac Sim synthetic-data tools expect).

Usage (from the go2_omniverse project root):
    python label_hospital_semantics.py [--output <path>]

Output defaults to  assets/env/hospital_labeled.usd.
The original asset is NOT modified.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# ── AppLauncher must come first ───────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto-label hospital USD semantics")
parser.add_argument(
    "--output",
    type=str,
    default="assets/env/hospital_labeled.usd",
    help="Where to save the labeled USD (relative to this script or absolute).",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
# Force headless; we only need the USD runtime.
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Now safe to import USD / Omniverse modules ─────────────────────────────
import omni.usd                          # noqa: E402
from pxr import Usd, UsdGeom, Sdf       # noqa: E402

try:
    from pxr import Semantics            # available in recent USD / Isaac Sim
    HAS_SEMANTICS_API = True
except ImportError:
    HAS_SEMANTICS_API = False

# ── Semantic label rules ──────────────────────────────────────────────────────
# Each entry: (regex_pattern_on_prim_name_lower, label_string)
# Rules are tested in order; first match wins.
LABEL_RULES: list[tuple[re.Pattern, str]] = [
    # ── Structural ────────────────────────────────────────────────────────
    (re.compile(r"stair|step|ramp"),       "stairs"),
    (re.compile(r"elevator|lift"),         "elevator"),
    (re.compile(r"floor|ground|slab"),     "floor"),
    (re.compile(r"ceiling|ceil|roof"),     "ceiling"),
    (re.compile(r"wall"),                  "wall"),
    (re.compile(r"column|pillar"),         "column"),
    (re.compile(r"railing|handrail|rail"), "railing"),
    (re.compile(r"door"),                  "door"),
    (re.compile(r"window"),                "window"),
    (re.compile(r"corridor|hallway"),      "corridor"),
    # ── Medical furniture ─────────────────────────────────────────────────
    (re.compile(r"bed"),                   "bed"),
    (re.compile(r"gurney|stretcher|cart"), "gurney"),
    (re.compile(r"wheelchair"),            "wheelchair"),
    (re.compile(r"iv_stand|iv_pole|iv"),   "iv_stand"),
    (re.compile(r"monitor|screen|display"),"monitor"),
    # ── General furniture ─────────────────────────────────────────────────
    (re.compile(r"chair|seat|sofa"),       "chair"),
    (re.compile(r"table|desk|counter"),    "table"),
    (re.compile(r"cabinet|locker|drawer"), "cabinet"),
    (re.compile(r"shelf|rack|shelv"),      "shelf"),
    (re.compile(r"sink|basin"),            "sink"),
    (re.compile(r"toilet"),                "toilet"),
    # ── Props / misc ──────────────────────────────────────────────────────
    (re.compile(r"plant|tree|pot"),        "plant"),
    (re.compile(r"light|lamp|fixture"),    "light"),
    (re.compile(r"sign|board"),            "sign"),
    (re.compile(r"bottle|pill|medicine"),  "medicine"),
    (re.compile(r"trash|bin|waste"),       "trash"),
    (re.compile(r"pillow|blanket|curtain"),"textile"),
]


def infer_label(prim: Usd.Prim) -> str | None:
    """Return a semantic label for *prim*, or None to skip it."""
    name  = prim.GetName().lower()
    # Also check parent names for context (e.g. /hospital/Furniture/Chair_01)
    path  = str(prim.GetPath()).lower()

    for pattern, label in LABEL_RULES:
        if pattern.search(name) or pattern.search(path):
            return label
    return None


def apply_label(prim: Usd.Prim, label: str) -> None:
    """Write semantic label onto *prim* using the USD Semantics schema."""
    if HAS_SEMANTICS_API:
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(label)
    else:
        # Fallback: plain custom attributes understood by many tools
        prim.CreateAttribute("semantic:params:semanticType",
                             Sdf.ValueTypeNames.String).Set("class")
        prim.CreateAttribute("semantic:params:semanticData",
                             Sdf.ValueTypeNames.String).Set(label)


def main() -> None:
    _assets_root = os.environ.get(
        "ISAAC_ASSETS_ROOT",
        "/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5",
    )
    HOSPITAL_USD = os.path.join(_assets_root, "Isaac/Environments/Hospital/hospital.usd")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Opening stage: {HOSPITAL_USD}")
    stage = Usd.Stage.Open(HOSPITAL_USD)
    if not stage:
        print("[ERROR] Failed to open USD stage.")
        sys.exit(1)

    labeled   = 0
    skipped   = 0
    label_counts: dict[str, int] = {}

    for prim in stage.TraverseAll():
        # Only label Xform / Mesh prims (skip metadata, shaders, …)
        if not (prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh)):
            continue

        label = infer_label(prim)
        if label is None:
            skipped += 1
            continue

        apply_label(prim, label)
        labeled += 1
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"[INFO] Labeled  : {labeled}")
    print(f"[INFO] Skipped  : {skipped}")
    print(f"[INFO] Breakdown:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"         {lbl:20s}: {cnt}")

    print(f"[INFO] Saving labeled stage → {output_path}")
    stage.Export(str(output_path))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
