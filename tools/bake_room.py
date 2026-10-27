"""Bake the living room's lighting with Cycles.

Run from the project root (Blender 4.x):
  blender -b --factory-startup --python tools/bake_room.py -- data/room/cozy-living-room.decoded.glb models/room-baked

The input is the room decoded to a plain GLB (web/scripts/decode-glb.mjs), since
Blender can't read meshopt compression. Smooth, simple surfaces get a lightmap
(an extra UV set plus a PNG); dense clutter (plants, baskets, the merged furniture)
gets its lighting baked into vertex colours instead. Writes:
  <out>/room.glb         the room with lightmap UVs and baked vertex colours
  <out>/lightmaps/*.png  per-object irradiance, gamma-encoded and divided by a scale
  <out>/lightmaps.json   per object: lightmap file + UV set, or vertex colours; and the scale
Light positions are in glTF room metres (Y up); Blender itself is Z up.
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

SAMPLES = 256
TEXELS_PER_METRE = 200
SOFA_TEXELS_PER_METRE = 900  # the fly sits on the sofa, so it gets the sharpest lightmap
MIN_SIZE, MAX_SIZE = 256, 4096
MIN_UV_COVERAGE = 0.3  # a lightmap using less of its texture than this bakes to vertex colours
MAX_LIGHTMAP_VERTICES = 150_000  # denser meshes go straight to vertex colours
GAMMA = 2.2
# The painted view outside the window stays unlit; see-through lamp shades and glass stay as they are.
SKIP_NAMES = ("Plane.003",)
SKIP_MATERIALS = ("Leather_Fabric_01", "PaletteMaterial003")

LIGHTS = [
    dict(name="floor_lamp", type="POINT", pos=(-2.57, 1.30, -2.93), color=(1.0, 0.72, 0.45), power=90, radius=0.06),
    dict(name="pendant", type="POINT", pos=(-0.40, 2.38, -0.90), color=(1.0, 0.80, 0.55), power=120, radius=0.10),
    dict(name="candle", type="POINT", pos=(-0.82, 0.47, -0.20), color=(1.0, 0.60, 0.30), power=4, radius=0.01),
    dict(name="window", type="AREA", pos=(-0.80, 1.70, -3.80), color=(0.92, 0.95, 1.0), power=900,
         size=(3.2, 2.4), aim=(0.0, -0.15, 1.0)),
    dict(name="ceiling_fill", type="AREA", pos=(-0.30, 3.05, -0.30), color=(1.0, 0.93, 0.85), power=250,
         size=(3.0, 3.0), aim=(0.0, -1.0, 0.0)),
]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def log(*parts):
    print("[bake]", *parts, flush=True)


def to_blender(x, y, z):
    return Vector((x, -z, y))


def safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def supports(operator, prop):
    return prop in operator.get_rna_type().properties.keys()


def setup_cycles(scene):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            log("cycles device:", kind, [d.name for d in prefs.devices if d.use])
            break
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 2
    scene.cycles.transparent_max_bounces = 8
    scene.render.bake.margin = 8
    scene.render.bake.use_clear = True


def add_lights(scene):
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.55, 0.62, 0.75, 1.0)
    background.inputs["Strength"].default_value = 0.3
    for spec in LIGHTS:
        light = bpy.data.lights.new(spec["name"], spec["type"])
        light.color = spec["color"]
        light.energy = spec["power"]
        if spec["type"] == "POINT":
            light.shadow_soft_size = spec["radius"]
        else:
            light.shape = "RECTANGLE"
            light.size, light.size_y = spec["size"]
        obj = bpy.data.objects.new(spec["name"], light)
        scene.collection.objects.link(obj)
        obj.location = to_blender(*spec["pos"])
        if "aim" in spec:
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = to_blender(*spec["aim"]).normalized().to_track_quat("-Z", "Y")


def diffuse_only(obj):
    """Metal and see-through surfaces have no diffuse light to bake, so bake them as plain dielectrics."""
    for slot in obj.material_slots:
        material = slot.material
        if not material or not material.use_nodes:
            continue
        if material.users > 1:
            slot.material = material = material.copy()
        tree = material.node_tree
        for node in tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            for name in ("Metallic", "Transmission Weight", "Coat Weight"):
                socket = node.inputs.get(name)
                if socket is None:
                    continue
                for link in list(socket.links):
                    tree.links.remove(link)
                socket.default_value = 0.0


def world_area(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    matrix = np.array(obj.matrix_world)
    co = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    a, b, c = (co[tris.reshape(-1, 3)[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def lightmap_size(obj):
    density = SOFA_TEXELS_PER_METRE if "Sofa" in obj.name else TEXELS_PER_METRE
    wanted = math.sqrt(max(world_area(obj), 1e-4)) * density
    return int(min(MAX_SIZE, max(MIN_SIZE, 2 ** math.ceil(math.log2(wanted)))))


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def uv_coverage(mesh, layer):
    """Fraction of the UV square covered by the mesh's triangles."""
    mesh.calc_loop_triangles()
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uv)
    loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("loops", loops)
    t = uv.reshape(-1, 2)[loops.reshape(-1, 3)]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float(0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum())


def add_lightmap_uv(obj, size):
    """Unwrap a new 'lightmap' UV set. Returns its index and how much of the texture it uses."""
    mesh = obj.data
    original = mesh.uv_layers[0] if mesh.uv_layers else None
    layer = mesh.uv_layers.new(name="lightmap")
    if original:
        original.active_render = True  # materials keep using their own UVs
    mesh.uv_layers.active = layer
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0, correct_aspect=True)
    pack = {"rotate": True, "margin": 4 / size}
    if supports(bpy.ops.uv.pack_islands, "margin_method"):
        pack["margin_method"] = "FRACTION"
    bpy.ops.uv.pack_islands(**pack)
    bpy.ops.object.mode_set(mode="OBJECT")
    return mesh.uv_layers.find("lightmap"), uv_coverage(mesh, mesh.uv_layers["lightmap"])


def robust_scale(rgb):
    luminance = rgb @ LUMA
    lit = luminance[luminance > 0]
    return max(float(np.percentile(lit, 99.5)) if lit.size else 1.0, 1e-3)


def bake_lightmap(obj, size):
    image = bpy.data.images.new(f"bake_{safe(obj.name)}", size, size, alpha=False, float_buffer=True)
    added = []
    for slot in obj.material_slots:
        tree = slot.material.node_tree
        uv = tree.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "lightmap"
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tree.links.new(uv.outputs["UV"], tex.inputs["Vector"])
        for node in tree.nodes:
            node.select = False
        tex.select = True
        tree.nodes.active = tex
        added.append((tree, uv, tex))
    select_only(obj)
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, margin=8, use_clear=True)
    for tree, uv, tex in added:
        tree.nodes.remove(tex)
        tree.nodes.remove(uv)
    pixels = np.empty(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(size, size, 4)[..., :3]


def bake_vertex_colors(obj):
    """Bake lighting into a colour attribute, stored divided by the returned scale."""
    mesh = obj.data
    attribute = mesh.color_attributes.new(name="baked_light", type="FLOAT_COLOR", domain="POINT")
    mesh.color_attributes.active_color = attribute
    select_only(obj)
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, target="VERTEX_COLORS", use_clear=True)
    colors = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
    attribute.data.foreach_get("color", colors)
    rgb = colors.reshape(-1, 4)[:, :3]
    scale = robust_scale(rgb)
    rgba = np.column_stack([np.clip(rgb / scale, 0, 1), np.ones(len(rgb), dtype=np.float32)])
    attribute.data.foreach_set("color", rgba.ravel())
    return scale


def box_blur(a, r):
    """Separable box blur with edge clamping, on the first two axes."""
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
        hi = np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)
        a = (hi - lo) / (2 * r + 1)
    return a


def smooth(rgb, r=2):
    """Take the noise out of the bake without bleeding across UV islands."""
    mask = (rgb.max(axis=2) > 0).astype(np.float32)[..., None]
    blurred = box_blur(box_blur(rgb * mask, r), r)
    weight = box_blur(box_blur(mask, r), r)
    return np.where(mask > 0, blurred / np.maximum(weight, 1e-6), rgb)


def save_lightmap(rgb, path):
    """Store irradiance / scale, gamma-encoded, as an 8-bit PNG. Returns the scale."""
    scale = robust_scale(rgb.reshape(-1, 3))
    encoded = np.clip(rgb / scale, 0, 1) ** (1 / GAMMA)
    size = rgb.shape[0]
    image = bpy.data.images.new(path.stem, size, size, alpha=False, float_buffer=False)
    image.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate([encoded, np.ones((size, size, 1), dtype=np.float32)], axis=2)
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = str(path)  # must be absolute: Blender ignores relative paths when saving here
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    if not path.exists():
        raise RuntimeError(f"lightmap was not written: {path}")
    return scale


def main():
    src, out = (Path(p).resolve() for p in sys.argv[sys.argv.index("--") + 1:][:2])
    (out / "lightmaps").mkdir(parents=True, exist_ok=True)
    started = time.time()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(src))
    scene = bpy.context.scene
    setup_cycles(scene)
    add_lights(scene)

    manifest = {"about": "Generated by tools/bake_room.py", "gamma": GAMMA, "objects": {}, "unbaked": [],
                "lights": LIGHTS}
    meshes = [o for o in scene.objects if o.type == "MESH"]
    log(f"{len(meshes)} meshes")
    for obj in meshes:
        materials = [s.material.name for s in obj.material_slots if s.material]
        if obj.name.startswith(SKIP_NAMES) or any(m.startswith(SKIP_MATERIALS) for m in materials):
            manifest["unbaked"].append(obj.name)
            log(f"skip  {obj.name}")
            continue
        t0 = time.time()
        diffuse_only(obj)
        vertices = len(obj.data.vertices)
        use_lightmap = vertices <= MAX_LIGHTMAP_VERTICES
        if use_lightmap:
            size = lightmap_size(obj)
            uv_index, coverage = add_lightmap_uv(obj, size)
            if coverage < MIN_UV_COVERAGE:
                obj.data.uv_layers.remove(obj.data.uv_layers["lightmap"])
                use_lightmap = False
        t1 = time.time()
        if use_lightmap:
            rgb = smooth(bake_lightmap(obj, size))
            file = f"lightmaps/{safe(obj.name)}.png"
            scale = save_lightmap(rgb, out / file)
            manifest["objects"][obj.name] = {"file": file, "uv": uv_index, "scale": round(scale, 5), "size": size,
                                             "coverage": round(coverage, 3)}
            how = f"lightmap {size}px (uv {uv_index}, {coverage:.0%} used)"
        else:
            scale = bake_vertex_colors(obj)
            manifest["objects"][obj.name] = {"vertexColors": True, "scale": round(scale, 5)}
            how = "vertex colours"
        log(f"baked {obj.name}: {how}, {vertices} verts, unwrap {t1 - t0:.0f}s, bake {time.time() - t1:.0f}s, "
            f"scale {scale:.3f}")

    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj)
    export = dict(filepath=str(out / "room.glb"), export_format="GLB", export_texcoords=True, export_normals=True,
                  export_image_format="WEBP", export_yup=True, export_lights=False, export_cameras=False)
    exporter = bpy.ops.export_scene.gltf
    if supports(exporter, "export_vertex_color"):
        export["export_vertex_color"] = "ACTIVE"
    elif supports(exporter, "export_colors"):
        export["export_colors"] = True
    exporter(**export)
    (out / "lightmaps.json").write_text(json.dumps(manifest, indent=1))
    log(f"done in {time.time() - started:.0f}s -> {out}")


main()
