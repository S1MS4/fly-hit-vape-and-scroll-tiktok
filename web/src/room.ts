import * as THREE from 'three'

// Baked lighting for the living room, from tools/bake_room.py. A baked object
// shows its texture times its baked light (a lightmap, or vertex colours for
// dense clutter) and ignores real-time lights, so what you see is the Cycles
// bake. Objects that weren't baked (see-through lamp shades, the view outside
// the window) keep their own materials.

type BakedObject = {
  scale: number // baked light is stored divided by this
  file?: string // lightmap PNG
  uv?: number // which UV set the lightmap uses
  vertexColors?: boolean // light baked into COLOR_0 instead
}

export type LightmapManifest = {
  gamma: number
  objects: Record<string, BakedObject>
  unbaked: string[]
}

const BAKED = '/models/room-baked'
export const BAKED_ROOM_URL = `${BAKED}/room.glb`

// Cycles bakes in physical units, where the lit sofa reaches ~10; this brings it
// to about 1 before tone mapping.
const EXPOSURE = 0.13

// ---- golden hour ---------------------------------------------------------------------
// The bake lit the room through a window at room (-0.80, 1.70, -3.80); main.ts turns and
// scales the room around the sofa spot, which leaves that window out at world
// (9.4, 3.5, 4.5) — low, behind the fly and off to its left, so the late sun comes in
// over its shoulder. A baked object is a MeshBasicMaterial and cannot see a real-time
// light, so the sun is folded into its baked colour instead: whatever faces the window
// picks up amber, everything else falls back to the cooler skylight.
export const SUN_POSITION = new THREE.Vector3(11.5, 3.2, 6.2) // ~14 degrees above the horizon
export const SUN_COLOR = 0xffb066 // direct sun this late is about 3000 K
export const SKY_COLOR = 0x8fb2e8 // the sky away from the sun is far cooler

const SUN_DIR = SUN_POSITION.clone().normalize()
const SUN_TINT = new THREE.Color(1.5, 0.82, 0.14) // added where the sun lands
const SHADE_TINT = new THREE.Color(0.66, 0.63, 0.71) // and what is left of the light elsewhere
const OUTSIDE_TINT = new THREE.Color(1.45, 0.95, 0.52) // the view through the window, lit by the sunset
const SHADOW_TINT = new THREE.Color(0x1b1533) // shadows keep a little of the sky rather than going black

/** Relight a baked material with a low amber sun, since a real-time light cannot reach it. */
function relight(material: THREE.MeshBasicMaterial) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.sunDirection = { value: SUN_DIR }
    shader.uniforms.sunTint = { value: SUN_TINT }
    shader.uniforms.shadeTint = { value: SHADE_TINT }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vBakedNormal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n\tvBakedNormal = mat3(modelMatrix) * normal;')
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec3 vBakedNormal;\nuniform vec3 sunDirection;\nuniform vec3 sunTint;\nuniform vec3 shadeTint;',
      )
      .replace(
        '#include <opaque_fragment>',
        '\tfloat sun = max(dot(normalize(vBakedNormal), sunDirection), 0.0);\n' +
          '\toutgoingLight *= shadeTint + sunTint * pow(sun, 0.5);\n' +
          '#include <opaque_fragment>',
      )
  }
}

/** The painted view outside the window: unbaked, so give it the sunset directly. */
function sunlitOutside(mesh: THREE.Mesh) {
  const old = mesh.material as THREE.MeshStandardMaterial
  mesh.material = new THREE.MeshBasicMaterial({
    map: old.map,
    color: old.color.clone().multiply(OUTSIDE_TINT),
    side: old.side,
  })
  old.dispose()
}

/** The bake's manifest, or null if the room hasn't been baked yet. */
export async function fetchLightmaps(): Promise<LightmapManifest | null> {
  const res = await fetch(`${BAKED}/lightmaps.json`)
  return res.ok ? res.json() : null
}

function bakedMaterial(old: THREE.MeshStandardMaterial, params: THREE.MeshBasicMaterialParameters) {
  const material = new THREE.MeshBasicMaterial({
    map: old.map,
    color: old.color,
    transparent: old.transparent,
    opacity: old.opacity,
    alphaTest: old.alphaTest,
    side: old.side,
    ...params,
  })
  old.dispose()
  relight(material)
  return material
}

export async function applyLightmaps(room: THREE.Object3D, manifest: LightmapManifest) {
  const entries = new Map(
    Object.entries(manifest.objects).map(([name, entry]) => [THREE.PropertyBinding.sanitizeNodeName(name), entry]),
  )
  const loader = new THREE.TextureLoader()
  const jobs: Promise<void>[] = []
  room.traverse((object) => {
    const mesh = object as THREE.Mesh
    if (!mesh.isMesh) return
    // Single-material objects are named after their node; multi-material ones sit under it.
    const entry = entries.get(mesh.name) ?? entries.get(mesh.parent?.name ?? '')
    if (!entry) {
      if (/^Plane\.?_?003/.test(mesh.name)) sunlitOutside(mesh) // sanitizeNodeName drops the dot
      return
    }
    const old = mesh.material as THREE.MeshStandardMaterial
    if (entry.vertexColors) {
      mesh.material = bakedMaterial(old, { color: old.color.clone().multiplyScalar(entry.scale * EXPOSURE), vertexColors: true })
      return
    }
    if (!entry.file) return
    jobs.push(
      loader.loadAsync(`${BAKED}/${entry.file}`).then((lightMap) => {
        lightMap.flipY = false // glTF UVs
        lightMap.colorSpace = THREE.SRGBColorSpace // stored gamma-encoded
        lightMap.channel = entry.uv ?? 1
        lightMap.generateMipmaps = false // mipmaps would bleed dark seams between UV islands
        lightMap.minFilter = THREE.LinearFilter
        mesh.material = bakedMaterial(old, { lightMap, lightMapIntensity: entry.scale * EXPOSURE * Math.PI }) // three.js divides lightmaps by pi
      }),
    )
  })
  await Promise.all(jobs)
}

/** An invisible copy of `mesh` that only shows the real-time shadows cast onto it. */
export function addShadowCatcher(mesh: THREE.Mesh, opacity = 0.52) {
  const catcher = new THREE.Mesh(
    mesh.geometry,
    new THREE.ShadowMaterial({ color: SHADOW_TINT, opacity, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
  )
  catcher.receiveShadow = true
  mesh.add(catcher)
  return catcher
}

/** Height of `surface` straight below world (x, z), or null where there is none. */
export function heightProbe(surface: THREE.Object3D) {
  const ray = new THREE.Raycaster()
  const down = new THREE.Vector3(0, -1, 0)
  const origin = new THREE.Vector3()
  return (x: number, z: number): number | null => {
    ray.set(origin.set(x, 1000, z), down)
    const hit = ray.intersectObject(surface, false)[0]
    return hit ? hit.point.y : null
  }
}

/** Move `object` up or down so its lowest point rests on the highest ground under its footprint. */
export function restOn(object: THREE.Object3D, ground: (x: number, z: number) => number | null, gap = 0.004) {
  object.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(object)
  let top = -Infinity
  for (let i = 0; i <= 4; i++) {
    for (let j = 0; j <= 4; j++) {
      const h = ground(THREE.MathUtils.lerp(box.min.x, box.max.x, i / 4), THREE.MathUtils.lerp(box.min.z, box.max.z, j / 4))
      if (h !== null) top = Math.max(top, h)
    }
  }
  if (top > -Infinity) object.position.y += top + gap - box.min.y
  object.updateMatrixWorld(true)
}
