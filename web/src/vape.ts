import * as THREE from 'three'
import { findMesh } from './models'

// The Elf Bar. `pen` is what the fly picks up: its origin is the middle of the
// pen and +Y points at the mouthpiece. Place it with pen.position/quaternion.
//
// A vape carries a finite number of puffs. Once they are gone the battery is dead:
// drawing on it only blinks the band, the way a spent disposable does, and the fly
// has to go and fetch another one. A dead vape gets flicked away and tumbles across
// the cushion — a flick from a sitting fly does not carry it as far as the floor —
// where it stays as a small monument to how much the fly has smoked.

// 9.81 m/s^2 at ROOM_SCALE = 3 in main.ts, where one scene unit is a third of a metre.
const GRAVITY = 29.4
const UP = new THREE.Vector3(0, 1, 0)

export type VapeOptions = {
  /** Hue rotation applied to the body, 0..1, so each flavour is its own colour. */
  hue?: number
  /** Puffs before the battery dies. */
  puffs?: number
}

/** A vape in mid-air after being flicked away. */
type Toss = {
  velocity: THREE.Vector3
  spin: THREE.Vector3
  groundY: number
  groundAt: ((x: number, z: number) => number | null) | null // the surface it is skidding over
  lie: THREE.Vector3 // horizontal direction it ends up lying along
  bounces: number
}

export class Vape {
  readonly pen = new THREE.Group()
  readonly length: number
  readonly radius: number
  private band: THREE.MeshStandardMaterial | null = null
  private glow = 0
  private blinkT = -1
  private puffsLeft: number
  private toss: Toss | null = null

  /** model: the Elf Bar GLB, upright along +Y with the mouthpiece on top. */
  constructor(model: THREE.Object3D, length = 0.34, options: VapeOptions = {}) {
    const box = new THREE.Box3().setFromObject(model)
    const scale = length / (box.max.y - box.min.y)
    model.scale.multiplyScalar(scale)
    model.position.set(0, -box.min.y * scale - length / 2, 0)
    this.length = length
    this.radius = ((box.max.x - box.min.x) * scale) / 2
    this.puffsLeft = options.puffs ?? Infinity
    this.pen.add(model)

    // The red band lights up while the fly draws on it.
    const band = findMesh(model, 'Material001')
    if (band) {
      this.band = (band.material as THREE.MeshStandardMaterial).clone()
      this.band.emissive = new THREE.Color(0xff2a3a)
      this.band.emissiveIntensity = 0
      band.material = this.band
    }
    // A different flavour for each spare, so a fresh one doesn't read as the same
    // vape respawning. The band keeps its own colour.
    if (options.hue !== undefined) this.tint(model, options.hue)
  }

  /**
   * Turn the printed wrap's hue. Only the wrap carries a texture: the shell and the
   * end caps are untextured black plastic, and the band is a thin untextured ring.
   *
   * The wrap has to be recoloured in the shader rather than through `material.color`.
   * The artwork is a photograph of a red watermelon Elf Bar, and multiplying a red
   * texture by a colour can only ever darken it toward a duller red — a blue-razz
   * tint came out as washed-out pink. Rotating the sampled colour about the grey axis
   * moves the artwork itself onto another hue and keeps its printing legible.
   */
  private tint(model: THREE.Object3D, hue: number) {
    model.traverse((object) => {
      const mesh = object as THREE.Mesh
      if (!mesh.isMesh) return
      const material = mesh.material as THREE.MeshStandardMaterial
      if (!material.map) return
      const tinted = material.clone()
      tinted.onBeforeCompile = (shader) => {
        shader.uniforms.flavourHue = { value: hue }
        shader.fragmentShader = shader.fragmentShader
          .replace(
            '#include <common>',
            `#include <common>
uniform float flavourHue;
vec3 turnHue(vec3 c, float h) {
  const vec3 grey = vec3(0.57735027);
  float a = h * 6.2831853;
  return c * cos(a) + cross(grey, c) * sin(a) + grey * dot(grey, c) * (1.0 - cos(a));
}`,
          )
          .replace(
            '#include <color_fragment>',
            `#include <color_fragment>
	diffuseColor.rgb = clamp(turnHue(diffuseColor.rgb, flavourHue), 0.0, 1.0);`,
          )
      }
      // Without this every flavour shares the first one's compiled program, and all
      // the spares come out the same colour again.
      tinted.customProgramCacheKey = () => `flavour${hue}`
      mesh.material = tinted
    })
  }

  /** No charge left: it only blinks now. */
  get dead() {
    return this.puffsLeft <= 0
  }

  /** Puffs before the battery dies. */
  get remaining() {
    return this.puffsLeft
  }

  /** True once it has been flicked away and is no longer the fly's to hold. */
  get loose() {
    return this.toss !== null
  }

  /** Mouthpiece tip in world space. */
  mouthpiece(target = new THREE.Vector3()) {
    return this.pen.localToWorld(target.set(0, this.length / 2, 0))
  }

  /**
   * A point to hold: side ±1 around the pen, along 0 (base) … 1 (mouthpiece).
   * `clearance` pushes it out from the surface so the leg's own thickness rests
   * against the pen instead of sinking into it.
   */
  grip(side: number, along: number, clearance = 0, target = new THREE.Vector3()) {
    return this.pen.localToWorld(target.set(side * (this.radius + clearance), (along - 0.5) * this.length, 0))
  }

  /** The fly is drawing on it. False means the battery is dead and nothing came out. */
  hit() {
    if (this.dead) {
      this.blinkT = 0
      return false
    }
    this.puffsLeft--
    this.glow = 1
    return true
  }

  /** Blink the band the way a spent disposable does when you draw on it. */
  blink() {
    this.blinkT = 0
  }

  /**
   * Flick it away. It tumbles, bounces and comes to rest lying on the cushion.
   * `groundAt` is the height of the surface it is skidding over, sampled as it goes,
   * so a vape thrown across a dip in the cushion settles into it instead of hovering.
   */
  flick(
    direction: THREE.Vector3,
    speed = 3.2,
    groundY = 0,
    groundAt: ((x: number, z: number) => number | null) | null = null,
  ) {
    const lie = new THREE.Vector3(direction.x, 0, direction.z)
    if (lie.lengthSq() < 1e-6) lie.set(1, 0, 0)
    lie.normalize()
    this.toss = {
      // Thrown flat and fast rather than lobbed: at 29.4 units/s^2 a high toss is over
      // in a tenth of a second and reads as the vape simply vanishing sideways.
      velocity: lie.clone().multiplyScalar(speed).setY(Math.abs(speed) * 0.5),
      spin: new THREE.Vector3(THREE.MathUtils.randFloatSpread(24), THREE.MathUtils.randFloatSpread(10), THREE.MathUtils.randFloatSpread(24)),
      groundY,
      groundAt,
      lie,
      bounces: 0,
    }
  }

  update(dt: number) {
    this.glow = Math.max(0, this.glow - dt / 1.6)
    // A dead battery answers with a few quick flashes and nothing else.
    if (this.blinkT >= 0) {
      this.blinkT += dt
      const flashes = 4
      const period = 0.22
      if (this.blinkT > flashes * period) this.blinkT = -1
      else this.glow = Math.max(this.glow, this.blinkT % period < period / 2 ? 0.85 : 0)
    }
    if (this.band) this.band.emissiveIntensity = 3 * this.glow

    const toss = this.toss
    if (!toss) return
    toss.velocity.y -= GRAVITY * dt
    this.pen.position.addScaledVector(toss.velocity, dt)
    this.pen.quaternion.multiply(
      new THREE.Quaternion().setFromEuler(new THREE.Euler(toss.spin.x * dt, toss.spin.y * dt, toss.spin.z * dt)),
    )
    const under = toss.groundAt?.(this.pen.position.x, this.pen.position.z)
    if (under !== null && under !== undefined) toss.groundY = under
    const restY = toss.groundY + this.radius
    if (this.pen.position.y > restY || toss.velocity.y > 0) return
    // Landed: bounce, losing most of the energy, until it gives up and lies still.
    this.pen.position.y = restY
    toss.bounces++
    toss.velocity.multiplyScalar(0.34)
    toss.velocity.y = Math.abs(toss.velocity.y)
    toss.spin.multiplyScalar(0.4)
    if (toss.bounces < 3 && toss.velocity.lengthSq() > 0.05) return
    this.pen.quaternion.setFromUnitVectors(UP, toss.lie)
    this.toss = null
  }
}
