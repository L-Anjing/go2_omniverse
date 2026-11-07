"""Static scene geometry for B2W simulation depth; no navigation inference."""
import numpy as np
from pxr import Usd, UsdGeom, Vt

from core import hospital_scene


SENSOR_MESH = "/World/SRU_DepthGeometry"


def triangulate(counts, indices):
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if np.all(counts == 3):
        return indices.reshape(-1, 3)
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset:offset + count]
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
        offset += count
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def build_sensor_mesh(stage, root_path):
    """RayCaster 2.1 needs one mesh; merge static visible Hospital meshes."""
    root = stage.GetPrimAtPath(root_path)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices, faces, floor_meshes = [], [], []
    vertex_offset = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        triangles = triangulate(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
        if not len(points) or not len(triangles):
            continue
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
        vertices.append((points @ matrix[:3, :3] + matrix[3, :3]).astype(np.float32))
        faces.append(triangles + vertex_offset)
        vertex_offset += len(points)
        if hospital_scene._spatial_semantic_class(prim, "hospital") == "floor":
            floor_meshes.append(prim)
    if not vertices:
        raise RuntimeError("Hospital has no visible meshes for the B2W depth sensor")
    merged = UsdGeom.Mesh.Define(stage, SENSOR_MESH)
    all_faces = np.concatenate(faces).astype(np.int32)
    merged.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(np.concatenate(vertices))))
    merged.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(all_faces), 3, dtype=np.int32)))
    merged.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(all_faces.reshape(-1)))
    merged.MakeInvisible()
    # No CollisionAPI: this is sensor geometry, not another physical collider.
    print(f"[sru] Depth geometry: {len(vertices)} meshes, {vertex_offset} vertices, {len(all_faces)} triangles", flush=True)
    hidden = hospital_scene.configure_default_ground(stage, floor_meshes, {})
    if not hidden:
        hospital_scene._label_ground_subtree(stage, "/World/ground", "floor")
