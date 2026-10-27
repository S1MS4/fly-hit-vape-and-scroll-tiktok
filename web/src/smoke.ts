import * as THREE from 'three'
import { SUN_COLOR, SUN_POSITION } from './room'

// Vape smoke: soft, noisy, rotating sprites that billow out, swirl, rise and
// thin out. Shaded from pale grey to white so it still reads in a white room,
// and burning amber along its edges when the camera looks through it at the sun.

const CAPACITY = 1500

const SUN_DIR = SUN_POSITION.clone().normalize()
// Unconverted, unlike everywhere else. This shader includes neither <tonemapping_fragment>
// nor <colorspace_fragment>, so it writes straight to the frame buffer; the default sRGB
// reading would hand back a far redder linear value than the frame it has to sit in.
const SUN_TINT = new THREE.Color().setHex(SUN_COLOR, THREE.LinearSRGBColorSpace)

function valueNoise(seed: number) {
  let s = seed >>> 0 || 1
  const rand = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296
  const p = Array.from({ length: 256 }, (_, i) => i)
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[p[i], p[j]] = [p[j], p[i]]
  }
  const perm = Uint8Array.from({ length: 512 }, (_, i) => p[i & 255])
  const lattice = Float32Array.from({ length: 256 }, rand)
  const fade = (t: number) => t * t * (3 - 2 * t)
  const at = (i: number, j: number) => lattice[perm[(perm[i & 255] + j) & 255]]
  return (x: number, y: number) => {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const u = fade(x - xi)
    const v = fade(y - yi)
    const a = at(xi, yi) + u * (at(xi + 1, yi) - at(xi, yi))
    const b = at(xi, yi + 1) + u * (at(xi + 1, yi + 1) - at(xi, yi + 1))
    return a + v * (b - a)
  }
}

/** 2×2 atlas of cloudy puffs: red = brightness, alpha = density. */
function puffAtlas() {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size * 2
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(size * 2, size * 2)
  for (let cell = 0; cell < 4; cell++) {
    const noise = valueNoise(cell * 7919 + 17)
    const ox = (cell % 2) * size
    const oy = Math.floor(cell / 2) * size
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const falloff = Math.max(0, 1 - Math.hypot((x / size) * 2 - 1, (y / size) * 2 - 1)) ** 1.5
        let n = 0
        let amp = 0.5
        let freq = 2.5
        for (let o = 0; o < 5; o++) {
          n += amp * noise((x / size) * freq, (y / size) * freq)
          amp *= 0.5
          freq *= 2
        }
        const i = ((oy + y) * size * 2 + ox + x) * 4
        img.data[i] = img.data[i + 1] = img.data[i + 2] = 255 * Math.min(1, 0.45 + 0.7 * n)
        img.data[i + 3] = 255 * Math.min(1, Math.max(0, falloff * (1.6 * n - 0.1)))
      }
    }
  }
  ctx.putImageData(img, 0, 0)
  return new THREE.CanvasTexture(canvas)
}

// The logdepthbuf chunks keep smoke depth-tested correctly with a logarithmic depth buffer.
const vertexShader = /* glsl */ `
  #include <common>
  #include <logdepthbuf_pars_vertex>
  attribute float aLife;
  attribute float aSize;
  attribute float aRot;
  attribute float aCell;
  uniform float uScale;
  uniform vec3 uSunDirection;
  varying float vLife;
  varying float vRot;
  varying float vCell;
  varying float vBacklit;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    #include <logdepthbuf_vertex>
    float grow = 1.0 + 4.0 * sqrt(clamp(aLife, 0.0, 1.0));
    gl_PointSize = aLife >= 1.0 ? 0.0 : uScale * aSize * grow / -mv.z;
    vLife = aLife;
    vRot = aRot;
    vCell = aCell;
    // A sprite is screen-aligned and has no normal, so the only handle on it is how close
    // the view ray runs to the sun: 1 looking straight through this puff at it, 0 with the
    // sun behind the camera. That is forward scattering, which is what smoke actually does.
    vec3 world = (modelMatrix * vec4(position, 1.0)).xyz;
    vBacklit = smoothstep(-0.15, 0.85, dot(normalize(world - cameraPosition), uSunDirection));
  }
`

const fragmentShader = /* glsl */ `
  #include <logdepthbuf_pars_fragment>
  uniform sampler2D uMap;
  uniform vec3 uSunTint;
  varying float vLife;
  varying float vRot;
  varying float vCell;
  varying float vBacklit;
  void main() {
    #include <logdepthbuf_fragment>
    if (vLife >= 1.0) discard;
    vec2 p = gl_PointCoord - 0.5;
    float c = cos(vRot);
    float s = sin(vRot);
    p = mat2(c, s, -s, c) * p * 1.41 + 0.5; // rotate; shrink the puff so corners never clip
    if (any(lessThan(p, vec2(0.0))) || any(greaterThan(p, vec2(1.0)))) discard;
    vec2 cell = vec2(mod(vCell, 2.0), floor(vCell / 2.0));
    vec4 puff = texture2D(uMap, (p + cell) * 0.5);
    float a = puff.a * smoothstep(0.0, 0.06, vLife) * (1.0 - smoothstep(0.3, 1.0, vLife)) * 0.75;
    vec3 body = mix(vec3(0.66, 0.68, 0.72), vec3(1.0), puff.r);
    // Sunlight that has come through the cloud. It keeps a pale core, because that is where
    // the smoke is thick enough to scatter the amber back out white, and burns golden along
    // the thin edges, where the sun gets through almost unscattered. Tinting the body rather
    // than replacing it matters: multiplying straight by the sun's colour leaves no white at
    // all and the exhale reads as a puff of flame.
    float edge = 1.0 - smoothstep(0.05, 0.45, puff.a);
    vec3 lit = body * mix(vec3(1.0), uSunTint, 0.38 + 0.45 * edge) * (1.0 + 0.3 * edge);
    gl_FragColor = vec4(mix(body, min(lit, vec3(1.0)), vBacklit), a);
  }
`

type Stream = {
  origin: THREE.Vector3
  dir: THREE.Vector3
  speed: number
  rate: number
  size: number
  duration: number
  left: number
  carry: number
}

export type StreamOptions = { speed?: number; rate?: number; duration?: number; size?: number }

export class Smoke {
  readonly points: THREE.Points
  private geo = new THREE.BufferGeometry()
  private material: THREE.ShaderMaterial
  private pos = new Float32Array(CAPACITY * 3)
  private vel = new Float32Array(CAPACITY * 3)
  private life = new Float32Array(CAPACITY).fill(1)
  private span = new Float32Array(CAPACITY).fill(1)
  private size = new Float32Array(CAPACITY)
  private rot = new Float32Array(CAPACITY)
  private spin = new Float32Array(CAPACITY)
  private cell = new Float32Array(CAPACITY)
  private seed = new Float32Array(CAPACITY)
  private next = 0
  private streams: Stream[] = []

  constructor() {
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3))
    this.geo.setAttribute('aLife', new THREE.BufferAttribute(this.life, 1))
    this.geo.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1))
    this.geo.setAttribute('aRot', new THREE.BufferAttribute(this.rot, 1))
    this.geo.setAttribute('aCell', new THREE.BufferAttribute(this.cell, 1))
    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uMap: { value: puffAtlas() },
        uScale: { value: 800 },
        uSunDirection: { value: SUN_DIR },
        uSunTint: { value: SUN_TINT },
      },
      transparent: true,
      depthWrite: false,
    })
    this.points = new THREE.Points(this.geo, this.material)
    this.points.frustumCulled = false
    this.points.renderOrder = 10
  }

  /** Pixels per world unit at distance 1: drawing-buffer height / (2 tan(fov / 2)). */
  setScale(pixelsPerUnit: number) {
    this.material.uniforms.uScale.value = pixelsPerUnit
  }

  /** Emit smoke continuously from `origin` along `dir`, e.g. an exhale. */
  stream(origin: THREE.Vector3, dir: THREE.Vector3, opts: StreamOptions = {}) {
    const duration = opts.duration ?? 1.2
    this.streams.push({
      origin: origin.clone(),
      dir: dir.clone().normalize(),
      speed: opts.speed ?? 0.35,
      rate: opts.rate ?? 120,
      size: opts.size ?? 0.035,
      duration,
      left: duration,
      carry: 0,
    })
  }

  private spawn(s: Stream) {
    const i = this.next
    this.next = (this.next + 1) % CAPACITY
    const strength = Math.min(1, s.left / (0.35 * s.duration)) // the stream weakens as it ends
    const speed = s.speed * (0.6 + 0.6 * Math.random()) * strength
    const jitter = () => (Math.random() - 0.5) * 0.1
    this.pos.set([s.origin.x, s.origin.y, s.origin.z], i * 3)
    this.vel.set([s.dir.x * speed + jitter(), s.dir.y * speed + jitter(), s.dir.z * speed + jitter()], i * 3)
    this.life[i] = 0
    this.span[i] = 2.8 + 2.2 * Math.random()
    this.size[i] = s.size * (0.7 + 0.6 * Math.random())
    this.rot[i] = Math.random() * Math.PI * 2
    this.spin[i] = (Math.random() - 0.5) * 0.8
    this.cell[i] = Math.floor(Math.random() * 4)
    this.seed[i] = Math.random() * 100
  }

  update(dt: number, t: number) {
    for (const s of this.streams) {
      s.carry += s.rate * dt
      for (; s.carry >= 1; s.carry--) this.spawn(s)
      s.left -= dt
    }
    this.streams = this.streams.filter((s) => s.left > 0)

    const drag = Math.exp(-dt * 1.6)
    for (let i = 0; i < CAPACITY; i++) {
      if (this.life[i] >= 1) continue
      const o = i * 3
      const sd = this.seed[i]
      // Drag, buoyancy and a cheap swirling flow field.
      this.vel[o] = this.vel[o] * drag + dt * 0.12 * Math.sin(this.pos[o + 1] * 7 + t * 1.3 + sd)
      this.vel[o + 1] = this.vel[o + 1] * drag + dt * 0.09
      this.vel[o + 2] = this.vel[o + 2] * drag + dt * 0.12 * Math.cos(this.pos[o] * 6 - t * 1.1 + sd)
      this.pos[o] += this.vel[o] * dt
      this.pos[o + 1] += this.vel[o + 1] * dt
      this.pos[o + 2] += this.vel[o + 2] * dt
      this.rot[i] += this.spin[i] * dt
      this.life[i] = Math.min(1, this.life[i] + dt / this.span[i])
    }
    this.geo.attributes.position.needsUpdate = true
    this.geo.attributes.aLife.needsUpdate = true
    this.geo.attributes.aRot.needsUpdate = true
    this.geo.attributes.aSize.needsUpdate = true
    this.geo.attributes.aCell.needsUpdate = true
  }
}
