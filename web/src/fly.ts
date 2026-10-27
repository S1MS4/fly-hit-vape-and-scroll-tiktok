import * as THREE from 'three'
import { findMesh, type FlyRig } from './models'
import { SUN_POSITION } from './room'
import type { Smoke } from './smoke'
import type { Vape } from './vape'

// The housefly, sitting back on its abdomen. Its skeleton comes from
// tools/rig_housefly.py: the static mesh becomes skinned meshes with one bone
// per body piece. The front legs work as arms: they pull the vape up to the
// mouth for a hit and swipe the phone. The other four legs stay planted with
// two-bone IK. The fly faces +Z, up is +Y, its left side is +X.

const SIT_ANGLE = 0.95 // how far the body tips back (radians)
const SIT_PIVOT = new THREE.Vector3(0, 0.08, -0.28) // abdomen, near the back
const HEAD_NOD = 0.75 // head tilts down to look at the phone
const MESHES = ['MultiMat_0_0', 'MultiMat_0h_0', 'MultiMat_0d_0'] // body, hair, wings
// Sunlight that has come through the wing membrane: the sun's amber, pushed pale because
// what you see has crossed two thin sheets. Linear, because the fly's shader tone-maps it
// — the opposite of smoke.ts, whose shader writes straight to the frame buffer.
// Saturated harder and pulled down than the arithmetic suggests, because ACES desaturates
// as it rolls off: fed the obvious amber the wing tone-maps to chalky cream, not to sunlight.
const WING_GLOW = new THREE.Color(1.25, 0.68, 0.28)
const SUN_DIR = SUN_POSITION.clone().normalize()
const VAPE_REST = new THREE.Vector3(-0.12, 0, 0.18) // where the vape stands, relative to the fly
// A contact point sits this far off the surface it touches, so the leg's own thickness
// rests against the vape or the glass instead of sinking into it. Measured off the rig
// rather than guessed: a front tarsus is a rod of radius 0.013 all the way to its tip,
// so 0.013 is exactly the gap that lays its surface on the pen. The mid and hind tarsi
// are blunter (0.0165 and 0.0185) but meet the cushion at a slant, so they need less.
const CONTACT_CLEARANCE = 0.013
const FOOT_CLEARANCE = 0.007 // feet on the cushion
const FETCH_S = 4.7 // the whole trip out to a spare vape and back
// Fallbacks only. The brain says how long it is actually coughing or gone, and these
// match brain/world.py's cough_s and whitey_s defaults.
const COUGH_S = 3.0
const WHITEY_S = 3.5

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
const smooth = (x: number) => clamp01(x) * clamp01(x) * (3 - 2 * clamp01(x))
/** 0 before a, easing to 1 at b. */
const ramp = (t: number, a: number, b: number) => smooth((t - a) / (b - a))
const bump = (t: number, a: number, b: number) => (t <= a || t >= b ? 0 : Math.sin(((t - a) / (b - a)) * Math.PI))

type Leg = {
  femur: THREE.Bone
  tibia: THREE.Bone
  tarsus: THREE.Bone
  tip: THREE.Vector3 // foot tip, in the tarsus bone's local units
  l1: number
  l2: number
  l3: number // tarsus length, in world units like l1 and l2
  side: number // +1 = the fly's left
}

type Pose = { joint: THREE.Vector3; foot: THREE.Vector3 }

type Action = {
  kind: 'hit' | 'swipe' | 'like' | 'follow' | 'repost' | 'fetch' | 'cough' | 'whitey'
  t: number
  side: number
  fired: Set<string>
  target?: Vape // the spare it is flying out to fetch
  run?: number // how long this one lasts, when the brain rather than the rig decides
  swatAt?: THREE.Vector3 // where the dead vape stood when the backhand started, in root space
  spot?: THREE.Vector3 // where the spare stood when the trip began, in world space
}

export type FlyProps = { vape: Vape; screen: THREE.Mesh; smoke: Smoke }

/** Turn a static mesh into a skinned mesh whose vertices each follow one bone. */
function skin(mesh: THREE.Mesh, boneOfVertex: number[]) {
  const geometry = mesh.geometry
  const count = geometry.attributes.position.count
  if (boneOfVertex.length !== count) throw new Error(`rig does not match ${mesh.name}; re-run tools/rig_housefly.py`)
  const index = new Uint16Array(count * 4)
  const weight = new Float32Array(count * 4)
  for (let i = 0; i < count; i++) {
    index[i * 4] = boneOfVertex[i]
    weight[i * 4] = 1
  }
  geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(index, 4))
  geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weight, 4))
  const skinned = new THREE.SkinnedMesh(geometry, mesh.material)
  skinned.name = mesh.name
  skinned.castShadow = skinned.receiveShadow = true
  skinned.frustumCulled = false
  skinned.position.copy(mesh.position)
  skinned.quaternion.copy(mesh.quaternion)
  skinned.scale.copy(mesh.scale)
  mesh.parent!.add(skinned)
  mesh.removeFromParent()
  return skinned
}

/**
 * A housefly's wing is membrane stretched between thick veins, and with the sun this low it
 * is lit from the far side: the membrane burns amber and the venation reads as dark lines
 * across it. The model ships with KHR transmission set, but three's transmission refracts
 * the backdrop and never looks at a light, so all it can show is the fly's own dark abdomen
 * behind the wing — and it costs a full multisampled re-render of the whole room every
 * frame. Drop it and add the glow by hand, from the angle between the wing and the sun.
 * (The material already has depthWrite off, so the far wing shows through the near one.)
 */
function backlitWings(mesh: THREE.SkinnedMesh) {
  const material = mesh.material as THREE.MeshPhysicalMaterial
  const membrane = material.transmissionMap // white on the membrane, dark along the veins
  material.transmission = 0 // this alone is what takes the extra pass out of the frame
  material.transmissionMap = null
  material.metalness = 0 // a wing is not metal; 0.5 was crushing everything that lit it
  material.roughness = 0.55
  material.onBeforeCompile = (shader) => {
    shader.uniforms.sunDirection = { value: SUN_DIR }
    shader.uniforms.wingGlow = { value: WING_GLOW }
    shader.uniforms.membrane = { value: membrane }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWingNormal;')
      // objectNormal, not normal: <skinnormal_vertex> has already run, so this one is skinned.
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n\tvWingNormal = mat3(modelMatrix) * objectNormal;')
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec3 vWingNormal;\nuniform vec3 sunDirection;\nuniform vec3 wingGlow;\nuniform sampler2D membrane;',
      )
      .replace(
        '#include <opaque_fragment>',
        // abs(), because a sheet this thin passes light whichever face the sun is on.
        '\tfloat through = pow(abs(dot(normalize(vWingNormal), sunDirection)), 0.6);\n' +
          '\tfloat pane = texture2D(membrane, vMapUv).r;\n' +
          '\toutgoingLight += wingGlow * pane * through;\n' +
          // The membrane is barely there in the alpha, so lit it has to hold itself up.
          '\tdiffuseColor.a = clamp(diffuseColor.a + pane * through * 0.42, 0.0, 1.0);\n' +
          '#include <opaque_fragment>',
      )
  }
  material.needsUpdate = true
}

/** Rotation whose +Y is `y`, with +X as close to `xHint` as possible. */
function basis(y: THREE.Vector3, xHint: THREE.Vector3) {
  const yn = y.clone().normalize()
  const x = xHint.clone().addScaledVector(yn, -xHint.dot(yn)).normalize()
  const z = new THREE.Vector3().crossVectors(x, yn)
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, yn, z))
}

export class Fly {
  readonly root = new THREE.Group()
  private sit = new THREE.Group()
  private sitY = 0
  private bones: THREE.Bone[]
  private legs: Record<string, Leg> = {}
  private planted: Record<string, Pose> = {} // mid and hind feet, in root space
  private head: THREE.Bone
  private wings: THREE.Bone[]
  private mouth = new THREE.Object3D()
  private vape: Vape
  private screen: THREE.Mesh
  private smoke: Smoke

  private queue: Action[] = []
  private action: Action | null = null
  private swipeSide = -1
  private escapeT = -1
  private vapeRestY = 0 // cushion height under the vape, in root space
  private turn = 0
  private arousal = 0
  private home = new THREE.Vector3() // its seat, to fly back to
  private released = false // the held vape has been thrown away and moves on its own now
  private lift = 0 // eased vape height: 0 standing on the cushion, 1 at the mouth
  private hold = 0 // eased arm reach from rest to the vape
  private ground: ((x: number, z: number) => number | null) | null = null // cushion height probe

  onSwipeStroke: () => void = () => {}
  onTap: (kind: 'like' | 'follow' | 'repost') => void = () => {} // a finger landing on the screen
  /** A live spare to go and get, or null when there are none left in the room. */
  onNeedVape: () => Vape | null = () => null
  /** A dead vape the fly has thrown away. It tumbles on its own from here. */
  onFlick: (vape: Vape) => void = () => {}

  constructor(model: THREE.Object3D, rig: FlyRig, props: FlyProps) {
    this.vape = props.vape
    this.screen = props.screen
    this.smoke = props.smoke
    this.screen.geometry.computeBoundingBox()

    this.root.add(this.sit)
    this.sit.add(model)
    model.position.copy(SIT_PIVOT).negate()
    this.sit.rotation.x = -SIT_ANGLE
    findMesh(model, 'Cylinder001')?.removeFromParent() // display pedestal

    // Bones in mesh-local units (model units / mesh scale).
    const meshes = MESHES.map((part) => {
      const mesh = findMesh(model, part)
      if (!mesh) throw new Error(`housefly.glb has no mesh ${part}`)
      return mesh
    })
    model.updateMatrixWorld(true)
    const k = 1 / meshes[0].getWorldScale(new THREE.Vector3()).x
    const pivots = rig.bones.map((b) => new THREE.Vector3(...b.pivot))
    this.bones = rig.bones.map((b) => Object.assign(new THREE.Bone(), { name: b.name }))
    rig.bones.forEach((b, i) => {
      const offset = b.parent < 0 ? pivots[i].clone() : pivots[i].clone().sub(pivots[b.parent])
      this.bones[i].position.copy(offset.multiplyScalar(k))
      if (b.parent >= 0) this.bones[b.parent].add(this.bones[i])
    })
    const skinned = meshes.map((mesh, i) => skin(mesh, rig.vertexBones[MESHES[i]]))
    backlitWings(skinned[2])
    skinned[2].castShadow = false // a wing you can see through should not drop a solid one
    skinned[0].add(this.bones[0])
    model.updateMatrixWorld(true)
    const skeleton = new THREE.Skeleton(this.bones)
    for (const mesh of skinned) mesh.bind(skeleton)

    const boneIndex = (name: string) => rig.bones.findIndex((b) => b.name === name)
    for (const [key, spec] of Object.entries(rig.legs)) {
      const [f, t, s] = spec.bones.slice(-3)
      const tip = new THREE.Vector3(...spec.tip).sub(pivots[s])
      this.legs[key] = {
        femur: this.bones[f],
        tibia: this.bones[t],
        tarsus: this.bones[s],
        l3: tip.length(), // read before the bone-local rescale, so it matches l1 and l2
        tip: tip.multiplyScalar(k),
        l1: pivots[t].distanceTo(pivots[f]),
        l2: pivots[s].distanceTo(pivots[t]),
        side: key.endsWith('_L') ? 1 : -1,
      }
    }
    const h = boneIndex('head')
    this.head = this.bones[h]
    this.head.add(this.mouth)
    this.mouth.position.set(...rig.mouth).sub(pivots[h]).multiplyScalar(k)
    this.wings = [this.bones[boneIndex('wing_L')], this.bones[boneIndex('wing_R')]]

    // Rest the abdomen on the table: lift the tipped-back body until its lowest point touches.
    const tilt = new THREE.Quaternion().setFromEuler(this.sit.rotation)
    const positions = skinned[0].geometry.attributes.position
    const bodyBone = rig.vertexBones[MESHES[0]]
    const v = new THREE.Vector3()
    let low = Infinity
    for (let i = 0; i < positions.count; i++) {
      if (bodyBone[i] !== 0) continue
      low = Math.min(low, v.fromBufferAttribute(positions, i).divideScalar(k).sub(SIT_PIVOT).applyQuaternion(tilt).y)
    }
    this.sitY = 0.004 - low
    this.sit.position.set(SIT_PIVOT.x, this.sitY, SIT_PIVOT.z)
    this.root.updateMatrixWorld(true)

    // Planted feet for the middle and hind legs, beside the body.
    for (const [key, leg] of Object.entries(this.legs)) {
      if (key.startsWith('front')) continue
      const hip = leg.femur.getWorldPosition(new THREE.Vector3())
      const mid = key.startsWith('mid')
      // Sitting: back legs stretched forward and out, middle legs to the side.
      const foot = new THREE.Vector3(hip.x + leg.side * (mid ? 0.16 : 0.2), FOOT_CLEARANCE, hip.z + (mid ? 0.06 : 0.18))
      // Exactly one tarsus up the leg's own slope, so solve() lands the tip on `foot`.
      // It used to be a flat 0.1, well short of the 0.157-0.170 tarsus, which drove the
      // foot a further 0.05 down and buried all four of them in the cushion.
      const joint = foot.clone().addScaledVector(new THREE.Vector3(leg.side * 0.4, 1, 0).normalize(), leg.l3)
      this.planted[key] = { joint, foot }
    }

    this.vape.pen.position.copy(VAPE_REST).setY(this.vape.length / 2 + 0.002)
  }

  /** Put the planted feet and the vape on the real cushion, given ground height at world (x, z). */
  settle(ground: (x: number, z: number) => number | null) {
    this.ground = ground // kept so a flicked vape can follow the cushion as it skids
    this.root.updateMatrixWorld(true)
    const localGround = (point: THREE.Vector3) => {
      const world = this.root.localToWorld(point.clone())
      const h = ground(world.x, world.z)
      return h === null ? null : this.root.worldToLocal(world.setY(h)).y
    }
    for (const feet of Object.values(this.planted)) {
      const y = localGround(feet.foot)
      if (y === null) continue
      const lift = y + FOOT_CLEARANCE - feet.foot.y
      feet.foot.y += lift
      feet.joint.y += lift
    }
    this.vapeRestY = localGround(VAPE_REST) ?? 0
    this.home.copy(this.root.position)
  }

  /** walk is ignored while sitting; turn nudges the head; arousal (0..1) makes it twitchy. */
  setDrive(_walk: number, turn: number, arousal: number) {
    this.turn = THREE.MathUtils.clamp(turn, -1, 1)
    this.arousal = clamp01(arousal)
  }

  puff() {
    const busy = this.action?.kind === 'hit' || this.queue.some((a) => a.kind === 'hit')
    if (!busy) this.queue.push({ kind: 'hit', t: 0, side: 0, fired: new Set() })
  }

  swipe() {
    if (this.queue.length >= 3) return
    this.swipeSide = -this.swipeSide
    this.queue.push({ kind: 'swipe', t: 0, side: this.swipeSide, fired: new Set() })
  }

  /** Double-tap the screen with the arm that didn't swipe last. */
  like() {
    if (this.queue.length >= 3) return
    this.queue.push({ kind: 'like', t: 0, side: -this.swipeSide, fired: new Set() })
  }

  /** Tap the creator's avatar in the app's right rail. */
  follow() {
    if (this.queue.length >= 3) return
    this.queue.push({ kind: 'follow', t: 0, side: -1, fired: new Set() }) // the rail is on the fly's right
  }

  /** Tap the share button to repost. */
  repost() {
    if (this.queue.length >= 3) return
    this.queue.push({ kind: 'repost', t: 0, side: -1, fired: new Set() })
  }

  escape() {
    if (this.escapeT < 0) this.escapeT = 0
  }

  /** The vape the fly is holding. */
  setVape(vape: Vape) {
    this.vape = vape
    this.released = false
  }

  /**
   * Give up on an action without losing what the brain already counted on it doing.
   * A swipe and a tap only reach the feed from onSwipeStroke/onTap partway through
   * the animation, but the brain has by then already set liked_video and started its
   * cooldown. Dropping one silently would leave the phone on a video the brain thinks
   * it has scrolled past, and the hidden TikTok window drifting out of step for good.
   */
  private abandon(a: Action | null) {
    if (!a) return
    if (a.kind === 'swipe' && !a.fired.has('stroke')) this.onSwipeStroke()
    if ((a.kind === 'like' || a.kind === 'follow' || a.kind === 'repost') && !a.fired.has('tap')) this.onTap(a.kind)
    // A draw that never got exhaled: the lungful has to go somewhere, and coughing it
    // straight back out is exactly what the fly is about to do anyway.
    if (a.kind === 'hit' && a.fired.has('draw') && !a.fired.has('exhale')) {
      const dir = new THREE.Vector3(0, 0.45, 1).applyQuaternion(this.root.quaternion)
      const mouth = this.mouth.getWorldPosition(new THREE.Vector3())
      this.smoke.stream(mouth, dir, { duration: 1.6, rate: 160, speed: 0.34, size: 0.045 })
    }
  }

  /**
   * Throw away every draw the fly has lined up, and keep everything else. A pull it can
   * no longer take is not worth saving, but a swipe or a like is a decision the brain
   * has already made and reported, so those wait their turn.
   *
   * A queued action has to be dropped exactly as firmly as a running one. The brain
   * emits 'puff' and the cough that pull caused in the same tick, and main.ts dispatches
   * puff first, so at the moment the cough arrives the hit is still sitting in the queue
   * and has not been promoted. Left there it would survive the fit and play afterwards,
   * and the fly would convulse for three seconds and then calmly raise the vape it was
   * choking on. It also made the brain behave differently from the Give puff button,
   * where the onset lands a tick later and the hit is already running.
   */
  private dropQueuedHits() {
    const waiting = this.queue.filter((q) => q.kind !== 'hit')
    for (const q of this.queue) if (q.kind === 'hit') this.abandon(q)
    this.queue = waiting
  }

  /**
   * The battery is dead. Throw it away and go and get a live one. The fly gives up on
   * drawing, because there is nothing to draw, but anything it was told to do to the
   * phone still happens once it is back in its seat.
   */
  private fetchVape() {
    if (this.action?.kind === 'fetch' || this.queue.some((a) => a.kind === 'fetch')) return
    const spare = this.onNeedVape()
    if (!spare) return // nothing left in the room, so it keeps drawing on a dead one
    this.dropQueuedHits()
    const trip: Action = { kind: 'fetch', t: 0, side: 0, fired: new Set(), target: spare, spot: spare.pen.position.clone() }
    this.queue = [trip, ...this.queue]
  }

  /** A bad pull: it doubles over, wings rattling and head shaking. */
  cough(seconds = COUGH_S) {
    this.overcome('cough', seconds)
  }

  /** The head rush after a big hit on a low tolerance: its legs fold and it just watches. */
  whitey(seconds = WHITEY_S) {
    this.overcome('whitey', seconds)
  }

  /**
   * Both take over from whatever the fly was doing, except a fetch: it is in mid-air on
   * a trip, and cutting that off would leave the dead vape thrown and no new one picked
   * up. Anything already queued survives and plays once the fly has recovered.
   */
  private overcome(kind: 'cough' | 'whitey', seconds: number) {
    if (this.action?.kind === 'fetch' || this.action?.kind === kind) return
    this.abandon(this.action) // whatever it was told to do to the phone still counts
    this.dropQueuedHits() // including the puff that caused this, which is still queued
    this.action = { kind, t: 0, side: 0, fired: new Set(), run: Math.max(0.5, seconds) }
  }

  update(dt: number, t: number) {
    if (!this.action && this.queue.length) this.action = this.queue.shift()!
    const a = this.action
    if (a) a.t += dt

    // Body: breathing, a startled hop, head looking at the phone.
    let hop = 0
    let buzz = 0
    if (this.escapeT >= 0) {
      this.escapeT += dt
      hop = bump(this.escapeT, 0, 0.5) * 0.05
      buzz = bump(this.escapeT, 0, 0.9)
      if (this.escapeT > 0.9) this.escapeT = -1
    }
    // Fetching: a shake of irritation, then up off the sofa, over to the spare and back.
    let airborne = 0
    let carry = 0
    let shake = 0
    if (a?.kind === 'fetch') {
      airborne = ramp(a.t, 0.8, 1.3) * (1 - ramp(a.t, 4.1, 4.6))
      carry = ramp(a.t, 2.45, 2.8)
      shake = bump(a.t, 0, 0.45)
      buzz = Math.max(buzz, airborne)
      if (a.spot) {
        const trip = ramp(a.t, 1.0, 2.45) * (1 - ramp(a.t, 2.85, 4.2))
        // Aimed at where the spare was standing when the trip began, never at where it
        // is now. Once the carry starts lifting the spare up under the fly, a hover point
        // read off the live vape chases its own tail: the vape follows the fly, the fly
        // follows the vape, and the pair climb out of the room together — measured at
        // 2.14 up, about two and a half body lengths above the cushion.
        const reach = new THREE.Vector3(VAPE_REST.x, 0, VAPE_REST.z).applyQuaternion(this.root.quaternion)
        const hover = a.spot.clone().sub(reach) // so the vape ends up where the fly holds one
        hover.y = a.spot.y + 0.24
        this.root.position.lerpVectors(this.home, hover, trip)
        this.root.position.y += airborne * (0.05 + 0.012 * Math.sin(t * 26))
      }
    }

    // A bad pull doubles it over in sharp jerks. A head rush folds it down slowly and
    // lets it back up at the end.
    const convulse = a?.kind === 'cough' ? bump(a.t, 0, a.run ?? COUGH_S) : 0
    let slump = 0
    if (a?.kind === 'whitey') {
      const span = a.run ?? WHITEY_S
      slump = ramp(a.t, 0, 0.5) * (1 - ramp(a.t, Math.max(0.55, span - 0.9), span))
    }
    if (convulse > 0) buzz = Math.max(buzz, convulse * (0.55 + 0.45 * Math.sin(t * 72)))

    const jitter = this.arousal * 0.01 * Math.sin(t * 23)
    // In the air it levels out; on the sofa it stays tipped back on its abdomen.
    this.sit.rotation.x =
      -SIT_ANGLE * (1 - 0.72 * airborne) +
      0.012 * Math.sin(t * 1.8) +
      jitter +
      convulse * 0.24 * Math.sin(t * 34) + // the cough itself
      slump * 0.5 // tipping forward onto the phone
    this.sit.position.y = this.sitY + hop - slump * 0.05 // it sinks as its legs give out

    // Eased toward the target rather than read straight off the clock. A cough or a
    // head rush takes the hit away mid-draw, and read straight the vape would jump from
    // the fly's mouth back down to the cushion in a single frame.
    let wantUp = 0 // 0 = vape standing on the table, 1 = at the mouth
    let wantGrab = 0 // how far the arms have moved from rest to the vape
    if (a?.kind === 'hit') {
      wantGrab = ramp(a.t, 0, 0.5) * (1 - ramp(a.t, 2.8, 3.3))
      wantUp = ramp(a.t, 0.5, 1.1) * (1 - ramp(a.t, 2.2, 2.8))
    }
    const ease = Math.min(1, dt * 9)
    this.lift += (wantUp - this.lift) * ease
    this.hold += (wantGrab - this.hold) * ease
    const vapeUp = this.lift
    const grab = this.hold
    this.head.rotation.set(
      HEAD_NOD * (1 - 0.6 * airborne) + 0.12 * vapeUp + convulse * 0.35 + slump * 0.45,
      0.15 * this.turn + 0.04 * Math.sin(t * 0.7) + 0.5 * shake * Math.sin(t * 42) + convulse * 0.55 * Math.sin(t * 47),
      slump * 0.3 * Math.sin(t * 1.3), // the head lolls while it comes round
    )
    this.wings[0].rotation.set(0, -0.6 * buzz, 0.9 * buzz * Math.sin(t * 90))
    this.wings[1].rotation.set(0, 0.6 * buzz, -0.9 * buzz * Math.sin(t * 90))
    this.root.updateMatrixWorld(true)

    // The vape: standing on the table, or held with its mouthpiece at the mouth.
    const toWorld = (p: THREE.Vector3) => this.root.localToWorld(p.clone())
    const rest = toWorld(VAPE_REST.clone().setY(this.vapeRestY + this.vape.length / 2 + 0.002))
    const mouth = this.mouth.getWorldPosition(new THREE.Vector3())
    const out = new THREE.Vector3(-0.7, -0.6, 0.2).normalize().applyQuaternion(this.root.quaternion) // mouthpiece -> body
    const held = mouth.clone().addScaledVector(out, this.vape.length / 2 + 0.012)
    const heldTurn = basis(out.clone().negate(), new THREE.Vector3(1, 0, 0).applyQuaternion(this.root.quaternion))
    if (!this.released) {
      this.vape.pen.position.lerpVectors(rest, held, vapeUp)
      this.vape.pen.quaternion.copy(this.root.quaternion).slerp(heldTurn, vapeUp)
      this.vape.pen.updateMatrixWorld(true)
    }
    // A fetched vape hangs under the fly on the way home, then settles into its place
    // as the fly lands, so there is no jump when it takes over as the held one.
    if (a?.kind === 'fetch' && a.target && carry > 0) {
      const spare = a.target
      const settled = new THREE.Vector3(VAPE_REST.x, this.vapeRestY + spare.length / 2 + 0.002, VAPE_REST.z)
      const slung = new THREE.Vector3(VAPE_REST.x, -0.1, VAPE_REST.z + 0.02)
      spare.pen.position.lerp(toWorld(settled.lerp(slung, airborne)), Math.min(1, dt * 20))
      spare.pen.quaternion.slerp(this.root.quaternion, Math.min(1, dt * 12))
      spare.pen.updateMatrixWorld(true)
    }

    // Arms: rest in front of the chest, hold the vape, or swipe the screen.
    for (const side of [1, -1]) {
      const leg = this.legs[side > 0 ? 'front_L' : 'front_R']
      const hip = leg.femur.getWorldPosition(new THREE.Vector3())
      const pose = this.restArm(hip, side)
      // The eased reach only decays toward zero, so stop once it is past noticing.
      if (grab > 0.001) {
        const grip = { joint: this.vape.grip(side, 0.55, CONTACT_CLEARANCE), foot: this.vape.grip(side, 0.85, CONTACT_CLEARANCE) }
        this.blend(pose, grip, grab)
      }
      if (a?.kind === 'fetch') {
        // The backhand that sends the dead one skidding away, then the grip on the new one.
        if (side === -1) {
          const swat = bump(a.t, 0.3, 0.85)
          // Latched where the vape stood, in the fly's own frame. The flick happens
          // halfway through this swing, and an arm still tracking the live vape would
          // snap across the cushion after it; latched, the backhand follows through
          // and relaxes as the bump eases out.
          if (swat > 0 && !a.swatAt) a.swatAt = this.root.worldToLocal(this.vape.grip(side, 0.7, CONTACT_CLEARANCE))
          if (a.swatAt && swat > 0) {
            const at = toWorld(a.swatAt)
            this.blend(pose, { joint: at.clone(), foot: at }, swat)
          }
        }
        if (a.target && carry > 0) {
          const hold = { joint: a.target.grip(side, 0.5, CONTACT_CLEARANCE), foot: a.target.grip(side, 0.8, CONTACT_CLEARANCE) }
          this.blend(pose, hold, carry)
        }
      }
      if (a?.kind === 'swipe' && a.side === side) {
        const reach = ramp(a.t, 0, 0.35) * (1 - ramp(a.t, 0.75, 1.2))
        const touch = this.screenPoint(0.35 * side, -0.4).lerp(this.screenPoint(0.35 * side, 0), ramp(a.t, 0.35, 0.7))
        this.blend(pose, this.contact(leg, touch, this.reachAngle()), reach)
        if (a.t > 0.5 && !a.fired.has('stroke')) {
          a.fired.add('stroke')
          this.onSwipeStroke()
        }
      }
      if ((a?.kind === 'like' || a?.kind === 'follow' || a?.kind === 'repost') && a.side === side) {
        const reach = ramp(a.t, 0, 0.3) * (1 - ramp(a.t, 0.95, 1.3))
        // like: double-tap the video; follow: tap the avatar; repost: tap share (both in the right rail)
        const tap = a.kind === 'like' ? bump(a.t, 0.3, 0.5) + bump(a.t, 0.55, 0.75) : bump(a.t, 0.3, 0.7)
        const [u, v] = a.kind === 'like' ? [0.25 * side, 0.05] : a.kind === 'follow' ? [-0.84, 0.05] : [-0.84, -0.6]
        const touch = this.screenPoint(u, v)
        const hover = touch.addScaledVector(this.screenNormal(), 0.05 * (1 - tap))
        this.blend(pose, this.contact(leg, hover, this.reachAngle()), reach)
        if (a.t > 0.4 && !a.fired.has('tap')) {
          a.fired.add('tap')
          this.onTap(a.kind)
        }
      }
      const pole = new THREE.Vector3(side, -0.3, -0.5).applyQuaternion(this.root.quaternion)
      this.solve(leg, pose.joint, pose.foot, pole)
    }

    for (const [key, feet] of Object.entries(this.planted)) {
      const leg = this.legs[key]
      const pole = new THREE.Vector3(leg.side * 0.6, 1, 0).applyQuaternion(this.root.quaternion)
      const joint = toWorld(feet.joint)
      const foot = toWorld(feet.foot)
      // Off the ground the legs trail under the body; in a whitey they fold under it.
      const fold = Math.max(airborne, slump * 0.55)
      if (fold > 0) {
        const hip = leg.femur.getWorldPosition(new THREE.Vector3())
        const tuck = new THREE.Vector3(leg.side * 0.45, -0.62, -0.55).normalize().applyQuaternion(this.root.quaternion)
        const tuckJoint = hip.clone().addScaledVector(tuck, leg.l1 * 0.85)
        joint.lerp(tuckJoint, fold)
        foot.lerp(tuckJoint.clone().addScaledVector(tuck, leg.l2 * 0.85), fold)
      }
      this.solve(leg, joint, foot, pole)
    }

    // Hit: the band glows while it draws, then a cloud rolls out of the mouth.
    if (a?.kind === 'hit') {
      if (a.t > 1.1 && !a.fired.has('draw')) {
        a.fired.add('draw')
        if (this.vape.hit()) {
          this.smoke.stream(this.vape.mouthpiece(), new THREE.Vector3(0, 1, 0), { duration: 1.0, rate: 16, speed: 0.04, size: 0.012 })
        } else {
          // Dead battery: it only blinks, so there is no cloud and nothing to blow out.
          a.fired.add('exhale')
          this.fetchVape()
        }
      }
      if (a.t > 2.45 && !a.fired.has('exhale')) {
        a.fired.add('exhale')
        const dir = new THREE.Vector3(0, 0.45, 1).applyQuaternion(this.root.quaternion)
        this.smoke.stream(mouth, dir, { duration: 1.6, rate: 160, speed: 0.34, size: 0.045 })
      }
    }
    if (a?.kind === 'fetch') {
      if (a.t > 0.55 && !a.fired.has('flick')) {
        a.fired.add('flick')
        // Thrown off at a different angle and speed every time. The rest of the flick
        // is deterministic, so without this every dead vape retraces the last one's arc
        // and they pile up as a single vape-shaped heap of coincident meshes.
        const away = new THREE.Vector3(-0.85, 0, -0.5)
          .applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.randFloatSpread(1.2))
          .applyQuaternion(this.root.quaternion)
        const dead = this.vape
        this.released = true // it belongs to the room now, not to the fly
        const from = toWorld(VAPE_REST.clone().setY(this.vapeRestY)).y
        dead.flick(away, THREE.MathUtils.randFloat(2.8, 3.6), from, this.ground)
        this.onFlick(dead)
      }
      if (a.t > FETCH_S && a.target) this.setVape(a.target)
    }
    const runs = { hit: 3.4, swipe: 1.2, like: 1.3, follow: 1.3, repost: 1.3, fetch: FETCH_S, cough: COUGH_S, whitey: WHITEY_S }
    if (a && a.t > (a.run ?? runs[a.kind])) this.action = null
  }

  private restArm(hip: THREE.Vector3, side: number): Pose {
    const q = this.root.quaternion
    const joint = hip.clone().add(new THREE.Vector3(side * 0.07, -0.19, 0.12).applyQuaternion(q))
    const foot = joint.clone().add(new THREE.Vector3(side * 0.01, -0.07, 0.08).applyQuaternion(q))
    return { joint, foot }
  }

  private blend(pose: Pose, to: Pose, k: number) {
    pose.joint.lerp(to.joint, k)
    pose.foot.lerp(to.foot, k)
  }

  /**
   * A pose that puts the foot tip on `point`, with the tarsus coming in along `away`.
   *
   * solve() treats a pose's `foot` as a direction to aim the tarsus, not a place to put
   * it: the tip ends up one tarsus from the knee whatever `foot` says. So the knee has
   * to be set one tarsus back along the approach, or the tip overshoots the surface and
   * the leg lands inside it. That overshoot was the whole limb-clipping complaint.
   */
  private contact(leg: Leg, point: THREE.Vector3, away: THREE.Vector3): Pose {
    return { joint: point.clone().addScaledVector(away, leg.l3), foot: point.clone() }
  }

  /** How a front leg comes at the glass: out of the screen and a bit below, so the
   *  tarsus slants up to the point it taps rather than stabbing straight in. */
  private reachAngle() {
    return this.screenNormal().multiplyScalar(0.93).add(new THREE.Vector3(0, -0.37, 0)).normalize()
  }

  /** A point on the phone screen: u, v in -1..1 across and up, just in front of the glass. */
  private screenPoint(u: number, v: number) {
    const box = this.screen.geometry.boundingBox!
    const local = new THREE.Vector3(
      THREE.MathUtils.lerp(box.min.x, box.max.x, (u + 1) / 2),
      THREE.MathUtils.lerp(box.min.y, box.max.y, (v + 1) / 2),
      box.min.z,
    )
    return this.screen.localToWorld(local).addScaledVector(this.screenNormal(), CONTACT_CLEARANCE)
  }

  /** Unit vector out of the screen, toward the fly. */
  private screenNormal() {
    const origin = this.screen.localToWorld(new THREE.Vector3())
    return this.screen.localToWorld(new THREE.Vector3(0, 0, -1)).sub(origin).normalize()
  }

  /** Two-bone IK for femur + tibia to reach `joint`, then point the tarsus at `foot`. */
  private solve(leg: Leg, joint: THREE.Vector3, foot: THREE.Vector3, pole: THREE.Vector3) {
    const hip = leg.femur.getWorldPosition(new THREE.Vector3())
    const toJoint = joint.clone().sub(hip)
    const d = THREE.MathUtils.clamp(toJoint.length(), Math.abs(leg.l1 - leg.l2) + 1e-4, leg.l1 + leg.l2 - 1e-4)
    const dir = toJoint.normalize()
    const cos = (leg.l1 ** 2 + d ** 2 - leg.l2 ** 2) / (2 * leg.l1 * d)
    const sin = Math.sqrt(Math.max(0, 1 - cos * cos))
    const bend = pole.clone().addScaledVector(dir, -pole.dot(dir))
    if (bend.lengthSq() < 1e-8) bend.set(0, 1, 0)
    bend.normalize()
    const knee = hip.clone().addScaledVector(dir, leg.l1 * cos).addScaledVector(bend, leg.l1 * sin)
    const end = hip.clone().addScaledVector(dir, d)
    this.aim(leg.femur, leg.tibia.position, knee.clone().sub(hip))
    this.aim(leg.tibia, leg.tarsus.position, end.clone().sub(knee))
    this.aim(leg.tarsus, leg.tip, foot.clone().sub(end))
  }

  /** Rotate `bone` so its child at `childLocal` lies along `worldDir`. */
  private aim(bone: THREE.Bone, childLocal: THREE.Vector3, worldDir: THREE.Vector3) {
    const parent = bone.parent!.getWorldQuaternion(new THREE.Quaternion()).invert()
    const want = worldDir.clone().normalize().applyQuaternion(parent)
    bone.quaternion.setFromUnitVectors(childLocal.clone().normalize(), want)
    bone.updateMatrixWorld(true)
  }
}
