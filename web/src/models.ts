import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { BAKED_ROOM_URL, fetchLightmaps, type LightmapManifest } from './room'

// The models in /models (served by the brain server), plus the fly's rig from
// tools/rig_housefly.py. Models come back in their own units with shadows on,
// except the room, where only the bare shell casts: enough for the window to stripe
// the sofa, without the rest of the room shading the fly.

export type FlyRig = {
  bones: { name: string; parent: number; pivot: [number, number, number] }[]
  legs: Record<string, { bones: number[]; tip: [number, number, number] }>
  mouth: [number, number, number]
  vertexBones: Record<string, number[]>
}

export type Models = {
  fly: THREE.Group
  phone: THREE.Group
  vape: THREE.Group
  room: THREE.Group
  lightmaps: LightmapManifest | null
  rig: FlyRig
}

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder) // the room is meshopt-compressed

/**
 * Everything loads with shadows on. `casts` narrows that, and the room passes a predicate
 * rather than a flag: only the 83-triangle shell of walls, ceiling and window reveals casts,
 * so the piers between the window openings fall between the sun and the sofa and rake their
 * shadows across it. Letting the whole room cast would put its own ceiling in the way.
 */
async function load(url: string, casts: boolean | ((mesh: THREE.Mesh) => boolean) = true) {
  const gltf = await loader.loadAsync(url)
  gltf.scene.traverse((o) => {
    const mesh = o as THREE.Mesh
    if (mesh.isMesh) {
      mesh.castShadow = typeof casts === 'function' ? casts(mesh) : casts
      mesh.receiveShadow = true
    }
  })
  return gltf.scene
}

async function loadRig(url: string): Promise<FlyRig> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status} (run python tools/rig_housefly.py)`)
  return res.json()
}

export async function loadModels(): Promise<Models> {
  const lightmaps = await fetchLightmaps() // the baked room if tools/bake_room.py has run
  const [fly, phone, vape, room, rig] = await Promise.all([
    load('/models/housefly.glb'),
    load('/models/iphone-13-pro-max.glb'),
    load('/models/elf-bar-1500-watermelon.glb'),
    load(lightmaps ? BAKED_ROOM_URL : '/models/cozy-living-room.glb', (mesh) => mesh.name === 'Cube_Material_0'),
    loadRig('/models/housefly.rig.json'),
  ])
  return { fly, phone, vape, room, lightmaps, rig }
}

/** First mesh whose (sanitised) name contains `part`. */
export function findMesh(root: THREE.Object3D, part: string): THREE.Mesh | null {
  const hits: THREE.Mesh[] = []
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && o.name.includes(part)) hits.push(o as THREE.Mesh)
  })
  return hits[0] ?? null
}
