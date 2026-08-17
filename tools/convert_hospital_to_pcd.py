#!/usr/bin/env python3
"""Generate a dense world-frame PCD directly from a composed USD asset.

Features:
  - Traverses Instance Proxies.
  - Resolves UsdGeom.Cube (often used for floors, walls, and stairs).
  - Resolves UsdGeom.PointInstancer (used for scattered props like beds, chairs).
  - Pure NumPy surface sampling, no Open3D dependency required.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np


def import_usd():
    try:
        from pxr import Usd, UsdGeom, Gf  # type: ignore
        return Usd, UsdGeom, Gf
    except ModuleNotFoundError:
        pass

    candidates = []
    candidates.extend(glob.glob(
        os.path.join(sys.prefix, "lib/python*/site-packages/isaacsim/extscache/omni.usd.libs-*")
    ))
    isaac_path = os.environ.get("ISAAC_PATH")
    if isaac_path:
        candidates.append(isaac_path)
    candidates.extend([
        "/media/user/data1/carl/isaac-sim-4.5.0",
        "/media/user/data1/carl/isaac-sim4.5.0",
    ])

    for root in candidates:
        usd_libs = [root] if os.path.basename(root).startswith("omni.usd.libs-") else glob.glob(
            os.path.join(root, "extscache", "omni.usd.libs-*")
        )
        for usd_lib in usd_libs:
            if os.path.isdir(os.path.join(usd_lib, "pxr")) or os.path.isdir(usd_lib):
                bin_dir = os.path.join(usd_lib, "bin")
                ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if bin_dir not in ld_paths and os.environ.get("USD_LIBS_REEXEC") != "1":
                    env = os.environ.copy()
                    env["LD_LIBRARY_PATH"] = bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
                    env["PYTHONPATH"] = usd_lib + ":" + env.get("PYTHONPATH", "")
                    env["USD_LIBS_REEXEC"] = "1"
                    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
                sys.path.insert(0, usd_lib)
                from pxr import Usd, UsdGeom, Gf  # type: ignore
                return Usd, UsdGeom, Gf

    raise ModuleNotFoundError(
        "Cannot import pxr. Run with Isaac Sim's Python, or set ISAAC_PATH to the Isaac Sim root."
    )


Usd, UsdGeom, Gf = import_usd()


DEFAULT_ASSETS_ROOT = "/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5"
DEFAULT_USD = os.path.join(
    os.environ.get("ISAAC_ASSETS_ROOT", DEFAULT_ASSETS_ROOT),
    "Isaac/Environments/Hospital/hospital.usd",
)
DEFAULT_OUT_DIR = "/media/user/data1/carl/workspace/go2_omniverse/tmp"
DEFAULT_PCT_DIR = "/media/user/data1/carl/workspace/3d-navi/PCT_planner/rsc/pcd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a composed USD scene into a dense PCD")
    parser.add_argument("--usd", default=DEFAULT_USD, help="Input USD stage")
    parser.add_argument("--scene-id", default="", help="Output identifier; defaults to the USD filename")
    parser.add_argument("--out", default="", help="Optional diagnostic PCD path")
    parser.add_argument("--pct-out", default="", help="PCT PCD path; defaults to rsc/pcd/<scene-id>.pcd")
    parser.add_argument("--target-points", type=int, default=0, help="Explicit target samples; 0 uses density")
    parser.add_argument("--points-per-square-meter", type=float, default=2000.0, help="Sampling density (raised for thin-structure coverage: wheelchairs, bed frames)")
    parser.add_argument("--voxel-size", type=float, default=0.015, help="Voxel size (<=0 disables; reduced for thin-structure preservation)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--min-samples-per-triangle", type=int, default=2)
    parser.add_argument("--include-invisible", action="store_true", default=True, help="Sample invisible meshes (often collision maps)")
    parser.add_argument(
        "--exclude-regex",
        default=r"CollisionPlane|collisionPlane|SkySphere|SkyDome|Sky_Sphere",
        help="Skip presentation or collision-only meshes by regex",
    )
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--max-z", type=float, default=None)
    return parser.parse_args()


def is_visible(prim: Usd.Prim, time: Usd.TimeCode) -> bool:
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return True
    return imageable.ComputeVisibility(time) != UsdGeom.Tokens.invisible


def get_local_triangles(prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray | None:
    """Extract local triangle coordinates from Mesh or Cube Primitives."""
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(time)
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time)
        
        if points is None or counts is None or indices is None:
            return None
        if len(points) == 0 or len(counts) == 0 or len(indices) == 0:
            return None

        local = np.asarray(points, dtype=np.float64)
        faces = []
        cursor = 0
        for count in counts:
            count = int(count)
            if count >= 3:
                face = np.asarray(indices[cursor : cursor + count], dtype=np.int64)
                if np.all((face >= 0) & (face < len(local))):
                    for k in range(1, count - 1):
                        faces.append([int(face[0]), int(face[k]), int(face[k + 1])])
            cursor += count
            
        if not faces:
            return None
        return local[np.asarray(faces, dtype=np.int64)].astype(np.float64)

    elif prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        size_attr = cube.GetSizeAttr().Get(time)
        size = size_attr if size_attr is not None else 2.0
        half = size / 2.0
        
        local_pts = np.array([
            [-half, -half, -half], [half, -half, -half],
            [half, half, -half], [-half, half, -half],
            [-half, -half, half], [half, -half, half],
            [half, half, half], [-half, half, half]
        ], dtype=np.float64)
        
        faces = [
            [0,1,2],[0,2,3], [4,5,6],[4,6,7], # bottom, top
            [0,1,5],[0,5,4], [1,2,6],[1,6,5], # front, right
            [2,3,7],[2,7,6], [3,0,4],[3,4,7]  # back, left
        ]
        return local_pts[faces].astype(np.float64)
        
    return None


def transform_triangles(local_triangles: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 4x4 matrix to (N, 3, 3) triangles."""
    flat_verts = local_triangles.reshape(-1, 3)
    world_verts = flat_verts @ matrix[:3, :3] + matrix[3, :3]
    return world_verts.reshape(-1, 3, 3).astype(np.float32)


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float64)


def sample_triangles(triangles, areas, target_points, chunk_size, min_samples_per_triangle, rng):
    if target_points <= 0 or len(triangles) == 0: return np.empty((0, 3), dtype=np.float32)
    total_area = float(np.sum(areas))
    if total_area <= 0.0: return np.empty((0, 3), dtype=np.float32)

    guaranteed = min_samples_per_triangle * len(triangles)
    extra_target = max(0, int(target_points) - guaranteed)

    expected_extra = areas * (float(extra_target) / total_area)
    counts = np.floor(expected_extra).astype(np.int32)
    counts += (rng.random(len(counts)) < (expected_extra - counts)).astype(np.int32)
    counts += min_samples_per_triangle

    selected = np.repeat(np.arange(len(triangles), dtype=np.int32), counts)
    if len(selected) == 0: return np.empty((0, 3), dtype=np.float32)
    rng.shuffle(selected)

    out_chunks = []
    total = len(selected)
    offset = 0
    while offset < total:
        picks = selected[offset : offset + chunk_size]
        n = len(picks)
        tri = triangles[picks]
        r1, r2 = rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)
        sqrt_r1 = np.sqrt(r1)
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2
        pts = tri[:, 0] * a[:, None] + tri[:, 1] * b[:, None] + tri[:, 2] * c[:, None]
        out_chunks.append(pts.astype(np.float32, copy=False))
        offset += n
    return np.vstack(out_chunks)


def write_binary_pcd(path: str, points: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        points.tofile(f)


def main() -> None:
    args = parse_args()
    scene_id = args.scene_id.strip() or Path(args.usd).stem
    out_path = args.out.strip() or os.path.join(DEFAULT_OUT_DIR, f"{scene_id}_gt.pcd")
    pct_out_path = args.pct_out.strip() or os.path.join(DEFAULT_PCT_DIR, f"{scene_id}.pcd")
    stage = Usd.Stage.Open(args.usd)
    if stage is None: raise RuntimeError(f"Failed to open USD: {args.usd}")
    stage.Load()

    time = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time)
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    triangle_chunks = []
    obj_count = 0
    instancer_count = 0

    print("[INFO] Starting Deep USD Traversal (Meshes, Cubes, PointInstancers)...")
    iterator = Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
    
    for prim in iterator:
        path = str(prim.GetPath())
        if exclude_re and exclude_re.search(path): continue
        if not args.include_invisible and not is_visible(prim, time): continue

        # 1. Handle Standard Meshes and Primitive Cubes (Walls/Floors)
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
            local_tris = get_local_triangles(prim, time)
            if local_tris is not None:
                matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
                world_tris = transform_triangles(local_tris, matrix)
                triangle_chunks.append(world_tris)
                obj_count += 1
                
        # 2. Handle Point Instancers (Beds, Chairs, Props)
        elif prim.IsA(UsdGeom.PointInstancer):
            instancer = UsdGeom.PointInstancer(prim)
            proto_indices = instancer.GetProtoIndicesAttr().Get(time)
            
            if proto_indices is not None and len(proto_indices) > 0:
                # Get world matrices for every single instance directly from USD API
                # This perfectly handles position, quaternion orientation, and scale!
                matrices = instancer.ComputeInstanceTransformsAtTime(
                    time, baseTime=time, protoHierarchyTime=time
                )
                prototypes = instancer.GetPrototypesRel().GetTargets()
                
                # Cache local triangles for prototypes to avoid recalculating
                proto_tri_cache = {}
                for i, proto_path in enumerate(prototypes):
                    proto_prim = stage.GetPrimAtPath(proto_path)
                    if proto_prim:
                        # Some prototypes are roots containing multiple meshes, we need to traverse them
                        proto_tris_list = []
                        for sub_prim in Usd.PrimRange(proto_prim):
                            if sub_prim.IsA(UsdGeom.Mesh) or sub_prim.IsA(UsdGeom.Cube):
                                tris = get_local_triangles(sub_prim, time)
                                if tris is not None:
                                    # Get relative transform from proto root to sub_mesh
                                    rel_matrix = np.asarray(xform_cache.GetLocalToWorldTransform(sub_prim), dtype=np.float64)
                                    root_matrix_inv = np.linalg.inv(np.asarray(xform_cache.GetLocalToWorldTransform(proto_prim), dtype=np.float64))
                                    local_to_proto = rel_matrix @ root_matrix_inv
                                    proto_tris_list.append(transform_triangles(tris, local_to_proto))
                                    
                        if proto_tris_list:
                            proto_tri_cache[i] = np.vstack(proto_tris_list)

                # Apply matrices to instances
                for idx, proto_id in enumerate(proto_indices):
                    if proto_id in proto_tri_cache:
                        mat = np.array(matrices[idx], dtype=np.float64)
                        world_tris = transform_triangles(proto_tri_cache[proto_id], mat)
                        triangle_chunks.append(world_tris)
                        obj_count += 1
                instancer_count += 1

        if obj_count % 1000 == 0 and obj_count > 0:
            print(f"  ... processed {obj_count} objects")

    if not triangle_chunks:
        raise RuntimeError("No geometry found! Check paths and visibility.")

    triangles = np.vstack(triangle_chunks)
    areas = triangle_areas(triangles)
    valid = np.isfinite(areas) & (areas > 1e-10)
    triangles, areas = triangles[valid], areas[valid]

    surface_area = float(np.sum(areas))
    print(f"\n[INFO] Traversal Complete.")
    print(f"  Geometries Processed: {obj_count} (Instancers resolved: {instancer_count})")
    print(f"  Total Triangles:      {len(triangles)}")
    print(f"  Total Surface Area:   {surface_area:.2f} m^2")

    target_points = args.target_points if args.target_points > 0 else int(np.ceil(surface_area * args.points_per_square_meter))
    
    print(f"[INFO] Sampling {target_points} points based on surface area...")
    rng = np.random.default_rng(args.seed)
    points = sample_triangles(triangles, areas, target_points, args.chunk_size, args.min_samples_per_triangle, rng)

    if args.min_z is not None or args.max_z is not None:
        mask = np.ones(len(points), dtype=bool)
        if args.min_z is not None: mask &= points[:, 2] >= args.min_z
        if args.max_z is not None: mask &= points[:, 2] <= args.max_z
        points = points[mask]

    before_voxel = len(points)
    if args.voxel_size > 0:
        print(f"[INFO] Voxel downsampling (size: {args.voxel_size}m)...")
        keys = np.floor(points / args.voxel_size).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(keep)]

    write_binary_pcd(out_path, points)
    if os.path.abspath(out_path) != os.path.abspath(pct_out_path):
        write_binary_pcd(pct_out_path, points)

    print(f"\n[OK] Success! Final points: {len(points)} (Down from {before_voxel})")
    print(f"  Diagnostic PCD: {out_path}")
    print(f"  PCT PCD:        {pct_out_path}")

if __name__ == "__main__":
    main()
