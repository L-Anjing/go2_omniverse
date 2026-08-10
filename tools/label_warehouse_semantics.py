"""
Automatic semantic labeling for the Isaac Sim Simple_Warehouse USD scene.

Traverses every prim, infers a semantic class from its name/path, and writes
the label using the standard USD Semantics API (same format that
omni.replicator / Isaac Sim synthetic-data tools expect).

Usage (from the go2_omniverse project root):
    python tools/label_warehouse_semantics.py [--output <path>]

Output defaults to  assets/env/full_warehouse_labeled.usd.
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

parser = argparse.ArgumentParser(description="Auto-label warehouse USD semantics")
parser.add_argument(
    "--output",
    type=str,
    default="assets/env/full_warehouse_labeled.usd",
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
#
# Labels are chosen to align with the X-MOBILITY warehouse semantic classes:
#   BACKGROUND(0), NAVIGABLE(1), FORKLIFT(2), PALLET(3), CONE(4), SIGN(5), FENCE(6)
# plus extended classes for richer semantic understanding.
LABEL_RULES: list[tuple[re.Pattern, str]] = [
    # ── Structural ────────────────────────────────────────────────────────
    (re.compile(r"floor|ground|slab|floor_decal|floor_?dec"), "floor"),
    (re.compile(r"ceiling|ceil"),                           "ceiling"),
    (re.compile(r"wall_a|wall_b|wall_wire|inner_corner"),   "wall"),
    (re.compile(r"pillar|column"),                          "pillar"),
    (re.compile(r"beam_a|beam_9m|bracket_beam|bracket_slot|bracketbeam|bracketslot"), "beam"),
    (re.compile(r"railing|handrail|rail"),                  "railing"),
    (re.compile(r"door"),                                   "door"),
    (re.compile(r"window"),                                 "window"),
    # ── Shelving / racking ────────────────────────────────────────────────
    (re.compile(r"rack_frame|rack_pile|rack_shelf|rack_shield|rackshield|rackframe|rackshelf|rackpile"), "shelf"),
    (re.compile(r"rack|shel[fv]|shelf_"),                   "shelf"),
    # ── Pallets / containers ──────────────────────────────────────────────
    (re.compile(r"palette|pallet"),                         "pallet"),
    (re.compile(r"crate_plastic|crateplastic|cardbox|card_box"), "crate"),
    (re.compile(r"barel_plastic|barrel|barelplastic"),      "barrel"),
    (re.compile(r"bottle_plastic|bottleplastic"),           "bottle"),
    (re.compile(r"bucket_plastic|bucketplastic"),           "bucket"),
    # ── Vehicles / carts ──────────────────────────────────────────────────
    (re.compile(r"forklift|fork_lift"),                     "forklift"),
    (re.compile(r"pushcart|push_cart|cart"),                "cart"),
    # ── Signs / labels ────────────────────────────────────────────────────
    (re.compile(r"aisle_sign|aislesign|sign_a|sign_b|sign_c|signcver|barcode"), "sign"),
    (re.compile(r"emergency_board|emergencyboard|fire_extinguisher|fireextinguisher|fuse_box|fusebox"), "safety"),
    (re.compile(r"wet_floor_sign|wetfloorsign"),            "sign"),
    # ── Hazard / safety ───────────────────────────────────────────────────
    (re.compile(r"traffic_cone|trafficcone|cone"),          "cone"),
    # ── Lighting / electrical ─────────────────────────────────────────────
    (re.compile(r"lamp_ceiling|lampceiling|lamp|light"),    "light"),
    (re.compile(r"fuse_box|fusebox|electrical"),            "electrical"),
    # ── Misc props ────────────────────────────────────────────────────────
    (re.compile(r"paper_note|papernote|paper_shortcut"),     "paper"),
    (re.compile(r"floor_decal|floordecal|keepclear|stripe"), "floor"),
    # ── Structural meshes (wall parts with SM_ prefix) ────────────────────
    (re.compile(r"sm_wall"),                                "wall"),
]


def infer_label(prim: Usd.Prim) -> str | None:
    """Return a semantic label for *prim*, or None to skip it."""
    name  = prim.GetName().lower()
    # Also check parent names for context (e.g. /warehouse/Props/SM_CardBoxA_01)
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
    WAREHOUSE_USD = os.path.join(
        _assets_root, "Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Opening stage: {WAREHOUSE_USD}")
    stage = Usd.Stage.Open(WAREHOUSE_USD)
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
