// Link to the Python brain server (brain/server.py) through the Vite proxy.

export type BrainState = {
  t: number
  bio_ms: number
  rates: Record<string, number>
  z: Record<string, number>
  turn: number
  walk: number
  nicotine: number
  tolerance: number
  withdrawal: number
  odor: number
  puffs: number
  ach_gain: number
  /** Seconds left of a coughing fit, 0 when it is not coughing (brain/world.py). */
  cough_s_left: number
  /** Seconds left of a head rush, 0 when it is not slumped. */
  whitey_s_left: number
  actions: string[]
}

export type Meta = {
  n: number
  edges: number
  categories: string[]
  front_leg_dn_types: string[]
  eliquid_orn_types: string[]
  group_sizes: Record<string, number>
  tick_ms: number
}

export async function fetchMeta(): Promise<Meta> {
  const res = await fetch('/api/meta')
  if (!res.ok) throw new Error(`/api/meta ${res.status}`)
  return res.json()
}

/** Records of [x, y, z, neuronIndex, category] as float32. */
export async function fetchSomata(): Promise<Float32Array> {
  const res = await fetch('/api/somata')
  if (!res.ok) throw new Error(`/api/somata ${res.status}`)
  return new Float32Array(await res.arrayBuffer())
}

export class BrainLink {
  private ws: WebSocket | null = null
  onState: (s: BrainState) => void = () => {}
  onSpikes: (neurons: Uint32Array) => void = () => {}
  onStatus: (text: string, live: boolean) => void = () => {}

  connect() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${scheme}://${location.host}/ws`)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    this.onStatus('connecting…', false)
    ws.onopen = () => this.onStatus('brain live', true)
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') this.onState(JSON.parse(ev.data))
      else this.onSpikes(new Uint32Array(ev.data))
    }
    ws.onclose = (ev) => {
      this.ws = null
      if (ev.code === 4001) {
        this.onStatus('another tab took over the brain', false)
        return
      }
      this.onStatus('brain offline, retrying…', false)
      setTimeout(() => this.connect(), 1500)
    }
  }

  private get open() {
    return this.ws?.readyState === WebSocket.OPEN
  }

  command(cmd: 'puff' | 'reset' | 'pause' | 'resume') {
    if (this.open) this.ws!.send(JSON.stringify({ cmd }))
  }

  /** What the fly sees: a small luminance image of the phone screen. */
  sendFrame(width: number, height: number, lum: Uint8Array) {
    if (!this.open || this.ws!.bufferedAmount > 1 << 16) return
    const buf = new Uint8Array(4 + lum.length)
    const view = new DataView(buf.buffer)
    view.setUint16(0, width, true)
    view.setUint16(2, height, true)
    buf.set(lum, 4)
    this.ws!.send(buf)
  }
}
