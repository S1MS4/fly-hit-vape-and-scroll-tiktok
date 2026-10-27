import * as THREE from 'three'

// Every neuron with a measured soma as a glowing point; spikes light it up.

const CATEGORY_COLORS: Record<string, number> = {
  other: 0x8a8f99,
  optic: 0x4cc9f0,
  central: 0xb5a8ff,
  sensory: 0x80ed99,
  descending: 0xffd166,
  motor: 0xff9f1c,
  mushroom_body: 0xf72585,
  dopamine: 0xff4d6d,
  vnc: 0x90a4ae,
}

const vertexShader = /* glsl */ `
  attribute vec3 aColor;
  attribute float aHeat;
  uniform float uSize;
  varying vec3 vColor;
  varying float vHeat;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * (0.6 + 2.4 * aHeat) / -mv.z;
    vColor = aColor;
    vHeat = aHeat;
  }
`

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vHeat;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = dot(c, c);
    if (d > 0.25) discard;
    float a = smoothstep(0.25, 0.0, d);
  #ifdef GLOW
    // Glowing points on a dark panel.
    gl_FragColor = vec4(vColor * (0.18 + 1.6 * vHeat), a * (0.35 + 0.65 * vHeat));
  #else
    // Dark tinted points on a light background; spiking neurons flare orange.
    vec3 col = mix(vColor * 0.6, vec3(1.0, 0.36, 0.08), vHeat * 0.75);
    gl_FragColor = vec4(col, a * (0.22 + 0.78 * vHeat));
  #endif
  }
`

export class BrainView {
  readonly points: THREE.Points
  private heat: Float32Array
  private heatAttr: THREE.BufferAttribute
  private pointOfNeuron: Int32Array

  /** somata: float32 records [x, y, z, neuronIndex, category]. */
  /** glow: additive glowing points for a dark background. */
  constructor(somata: Float32Array, categories: string[], neuronCount: number, verticalSign = -1, glow = false) {
    const count = somata.length / 5
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    this.heat = new Float32Array(count)
    this.pointOfNeuron = new Int32Array(neuronCount).fill(-1)
    const color = new THREE.Color()

    for (let i = 0; i < count; i++) {
      const o = i * 5
      // Dataset z runs brain -> ventral nerve cord; show that axis vertically.
      positions[i * 3] = somata[o]
      positions[i * 3 + 1] = verticalSign * somata[o + 2]
      positions[i * 3 + 2] = somata[o + 1]
      this.pointOfNeuron[somata[o + 3]] = i
      color.set(CATEGORY_COLORS[categories[somata[o + 4]]] ?? 0xffffff)
      colors.set([color.r, color.g, color.b], i * 3)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('aColor', new THREE.BufferAttribute(colors, 3))
    this.heatAttr = new THREE.BufferAttribute(this.heat, 1)
    this.heatAttr.setUsage(THREE.DynamicDrawUsage)
    geo.setAttribute('aHeat', this.heatAttr)

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: { uSize: { value: 5.0 } },
      transparent: true,
      depthWrite: false,
      blending: glow ? THREE.AdditiveBlending : THREE.NormalBlending,
      defines: glow ? { GLOW: '' } : {},
    })
    this.points = new THREE.Points(geo, material)
    this.points.frustumCulled = false
  }

  flash(neurons: Uint32Array) {
    for (let i = 0; i < neurons.length; i++) {
      const p = this.pointOfNeuron[neurons[i]]
      if (p >= 0) this.heat[p] = 1
    }
  }

  update(dt: number) {
    const k = Math.exp(-dt / 0.25)
    for (let i = 0; i < this.heat.length; i++) this.heat[i] *= k
    this.heatAttr.needsUpdate = true
  }
}

export const categoryColors = CATEGORY_COLORS
