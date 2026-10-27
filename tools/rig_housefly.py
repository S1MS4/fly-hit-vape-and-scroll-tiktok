"""Auto-rig models/housefly.glb so the fly can move its legs, head and wings.

The model has no skeleton, but its body mesh is made of separate pieces:
thorax+abdomen, head parts, and one piece per leg segment. Pieces that touch
are joints; the chain of pieces from the body to a piece resting on the floor
is a leg. Writes models/housefly.rig.json with bone pivots and one bone index
per vertex of the body, hair and wing meshes.

Model units: glTF positions x 0.01 (the file's root scale), +Y up, fly faces +Z.
Run from the project root:  python tools/rig_housefly.py
"""
import json
import struct
from collections import deque
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
GLB = ROOT / "models" / "housefly.glb"
OUT = ROOT / "models" / "housefly.rig.json"
UNIT = 0.01
CONTACT = 0.012  # pieces closer than this are joined
FLOOR = 0.012    # pieces reaching below this are feet

BODY_MESH, HAIR_MESH, WING_MESH = "MultiMat_0_0", "MultiMat_0h_0", "MultiMat_0d_0"
COMPONENT_TYPES = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path):
    data = path.read_bytes()
    json_len, = struct.unpack_from("<I", data, 12)
    gltf = json.loads(data[20:20 + json_len])
    bin_len, = struct.unpack_from("<I", data, 20 + json_len)
    blob = data[28 + json_len: 28 + json_len + bin_len]

    def accessor(i):
        a = gltf["accessors"][i]
        view = gltf["bufferViews"][a["bufferView"]]
        dtype = np.dtype(COMPONENT_TYPES[a["componentType"]])
        width = COMPONENTS[a["type"]]
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        rows = np.arange(a["count"])[:, None] * stride + np.arange(dtype.itemsize * width)[None, :]
        raw = np.frombuffer(blob, np.uint8, stride * (a["count"] - 1) + dtype.itemsize * width, start)
        return raw[rows].copy().view(dtype).reshape(a["count"], width)

    def mesh(name_part):
        prim = next(m for m in gltf["meshes"] if name_part in m["name"])["primitives"][0]
        positions = accessor(prim["attributes"]["POSITION"]).astype(np.float64) * UNIT
        triangles = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
        return positions, triangles

    return mesh


def pieces(positions, triangles):
    """Connected-component label per vertex, welding duplicated seam vertices."""
    _, weld = np.unique(np.round(positions / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    tri = weld[triangles]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    n = weld.max() + 1
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels[weld]


def main():
    mesh = read_glb(GLB)
    body_p, body_t = mesh(BODY_MESH)
    label = pieces(body_p, body_t)
    count = label.max() + 1
    verts = [body_p[label == c] for c in range(count)]
    trees = [cKDTree(v) for v in verts]

    # Contact graph between pieces, with the contact point for each pair.
    contact = {}
    for a in range(count):
        for b in range(a + 1, count):
            d, j = trees[a].query(verts[b], k=1)
            if d.min() < CONTACT:
                k = int(np.argmin(d))
                point = (verts[b][k] + verts[a][j[k]]) / 2
                contact[(a, b)] = contact[(b, a)] = point
    neighbours = {c: [b for (a, b) in contact if a == c] for c in range(count)}

    body = int(np.argmax([len(v) for v in verts]))
    feet = [c for c in range(count) if c != body and verts[c][:, 1].min() < FLOOR]

    bones = [{"name": "body", "parent": -1, "pivot": verts[body].mean(0)}]
    bone_of_piece = {body: 0}

    def path_to_body(start):
        prev, queue = {start: None}, deque([start])
        while queue:
            c = queue.popleft()
            if c == body:
                break
            for n in neighbours[c]:
                if n not in prev:
                    prev[n] = c
                    queue.append(n)
        chain, c = [], body
        while c is not None:
            chain.append(c)
            c = prev[c]
        return chain  # body, ..., foot

    legs = {}
    by_side = {"L": [], "R": []}
    for foot in feet:
        side = "L" if verts[foot][:, 0].mean() > 0 else "R"  # the fly's left is +X
        by_side[side].append(foot)
    for side, side_feet in by_side.items():
        for name, foot in zip(["front", "mid", "hind"], sorted(side_feet, key=lambda c: -verts[c][:, 2].mean())):
            chain = path_to_body(foot)
            ids = []
            for parent_piece, piece in zip(chain, chain[1:]):
                bones.append({"name": f"{name}_{side}_{len(ids)}", "parent": bone_of_piece[parent_piece],
                              "pivot": contact[(parent_piece, piece)]})
                bone_of_piece[piece] = len(bones) - 1
                ids.append(len(bones) - 1)
            tip = verts[foot][np.argmax(np.linalg.norm(verts[foot] - bones[ids[-1]]["pivot"], axis=1))]
            legs[f"{name}_{side}"] = {"bones": ids, "tip": tip}

    # Head: the biggest remaining piece near the midline at the front, plus everything hanging off it.
    front = [c for c in range(count) if c not in bone_of_piece and abs(verts[c][:, 0].mean()) < 0.02
             and verts[c][:, 2].mean() > 0.2]
    head_core = max(front, key=lambda c: len(verts[c]))
    bones.append({"name": "head", "parent": 0, "pivot": contact[(body, head_core)]})
    head_bone = len(bones) - 1
    queue = deque([head_core])
    while queue:
        c = queue.popleft()
        if c in bone_of_piece:
            continue
        bone_of_piece[c] = head_bone
        queue.extend(n for n in neighbours[c] if n not in bone_of_piece and n != body)
    for c in range(count):
        bone_of_piece.setdefault(c, 0)  # small pieces on the thorax ride with the body

    head_verts = np.concatenate([verts[c] for c in range(count) if bone_of_piece[c] == head_bone])
    snout = head_verts[(np.abs(head_verts[:, 0]) < 0.03) & (head_verts[:, 2] > head_verts[:, 2].mean())]
    mouth = snout[np.argsort(snout[:, 1])[:15]].mean(0)

    # Wings: two pieces of the wing mesh, each hinged at its front inner corner.
    wing_p, wing_t = mesh(WING_MESH)
    wing_label = pieces(wing_p, wing_t)
    wing_bone = np.zeros(len(wing_p), np.int64)
    for c in range(wing_label.max() + 1):
        v = wing_p[wing_label == c]
        side = "L" if v[:, 0].mean() > 0 else "R"
        root = v[np.argsort(np.abs(v[:, 0]) - v[:, 2])[: max(3, len(v) // 20)]].mean(0)
        bones.append({"name": f"wing_{side}", "parent": 0, "pivot": root})
        wing_bone[wing_label == c] = len(bones) - 1

    body_bone = np.array([bone_of_piece[c] for c in label])

    # Hair strands follow the body vertex nearest to their root.
    hair_p, hair_t = mesh(HAIR_MESH)
    hair_label = pieces(hair_p, hair_t)
    body_tree = cKDTree(body_p)
    dist, nearest = body_tree.query(hair_p, k=1)
    hair_bone = np.zeros(len(hair_p), np.int64)
    for c in range(hair_label.max() + 1):
        members = np.flatnonzero(hair_label == c)
        root = members[np.argmin(dist[members])]
        hair_bone[members] = body_bone[nearest[root]] if dist[root] < 0.03 else 0

    rig = {
        "about": "Generated by tools/rig_housefly.py. Model units = glTF units x 0.01.",
        "bones": [{"name": b["name"], "parent": b["parent"], "pivot": np.round(b["pivot"], 5).tolist()} for b in bones],
        "legs": {k: {"bones": v["bones"], "tip": np.round(v["tip"], 5).tolist()} for k, v in legs.items()},
        "mouth": np.round(mouth, 5).tolist(),
        "vertexBones": {BODY_MESH: body_bone.tolist(), HAIR_MESH: hair_bone.tolist(), WING_MESH: wing_bone.tolist()},
    }
    OUT.write_text(json.dumps(rig, separators=(",", ":")))

    print(f"{count} body pieces, {len(bones)} bones -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    for b in bones:
        print(f"  {b['name']:9s} parent={b['parent']:2d} pivot={np.round(b['pivot'], 3)}")
    for k, v in legs.items():
        print(f"  leg {k}: bones {v['bones']} tip {np.round(v['tip'], 3)}")
    print("  mouth", np.round(mouth, 3))
    print("  vertices per bone (body):", np.bincount(body_bone, minlength=len(bones)).tolist())
    print("  hair strands on body/other:", int((hair_bone == 0).sum()), "/", int((hair_bone != 0).sum()))


if __name__ == "__main__":
    main()
