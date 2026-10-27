import type { BrainState, Meta } from './net'

// Readout bars: [group key, label, rate (Hz) that fills the bar].
const BARS: [string, string, number][] = [
  ['photoreceptors', 'photoreceptors', 80],
  ['eliquid_orns', 'e-liquid ORNs', 80],
  ['kc', 'Kenyon cells', 40],
  ['mbon', 'MBONs', 80],
  ['pam', 'PAM reward', 80],
  ['ppl1', 'PPL1 punish', 80],
  ['proboscis', 'MN9 proboscis', 80],
  ['front_leg', 'front-leg DNs', 80],
  ['steer_left', 'DNa01/02 L', 80],
  ['steer_right', 'DNa01/02 R', 80],
  ['escape', 'giant fiber', 40],
]

const $ = (id: string) => document.getElementById(id)!

export class Hud {
  private bars = new Map<string, { fill: HTMLElement; val: HTMLElement; max: number }>()

  constructor() {
    const root = $('bars')
    for (const [key, label, max] of BARS) {
      const row = document.createElement('div')
      row.className = 'bar'
      row.innerHTML = `<label>${label}</label><div class="track"><div class="fill"></div></div><span class="val">0</span>`
      root.appendChild(row)
      this.bars.set(key, { fill: row.querySelector('.fill')!, val: row.querySelector('.val')!, max })
    }
  }

  setMeta(meta: Meta) {
    $('dataset').textContent =
      `MaleCNS v1.0 · ${meta.n.toLocaleString()} neurons · ${(meta.edges / 1e6).toFixed(1)}M connections`
  }

  update(s: BrainState, firing: number, counts: Record<'swipes' | 'likes' | 'follows' | 'reposts', number>) {
    $('bio').textContent = `${(s.bio_ms / 1000).toFixed(2)} s`
    $('spk').textContent = firing.toLocaleString()
    $('puffs').textContent = String(s.puffs)
    for (const [id, n] of Object.entries(counts)) $(id).textContent = String(n)
    $('nic').textContent = s.nicotine.toFixed(2)
    $('tol').textContent = s.tolerance.toFixed(2)
    $('wd').textContent = s.withdrawal.toFixed(2)
    $('gain').textContent = s.ach_gain.toFixed(2)
    for (const [key, bar] of this.bars) {
      const rate = s.rates[key] ?? 0
      bar.fill.style.width = `${Math.min(100, (rate / bar.max) * 100)}%`
      bar.fill.classList.toggle('hot', (s.z[key] ?? 0) > 2.5)
      bar.val.textContent = rate.toFixed(rate < 10 ? 1 : 0)
    }
  }

  log(action: string, t: number) {
    const icons: Record<string, string> = {
      puff: '💨 puff', swipe: '👆 swipe', like: '❤️ like', follow: '➕ follow', repost: '🔁 repost', escape: '⚡ escape',
      cough: '🤮 coughing fit', whitey: '😵 whitey', fetch: '🔋 fresh vape', dead: '🪫 dead battery',
      'out of vapes': '🪫 out of vapes',
    }
    const line = document.createElement('div')
    line.className = action.replace(/\s+/g, '-') // a multi-word action would split into several classes
    line.textContent = `${t.toFixed(1)}s  ${icons[action] ?? action}`
    const log = $('log')
    log.prepend(line)
    while (log.children.length > 14) log.lastChild!.remove()
  }

  clearLog() {
    $('log').replaceChildren()
  }

  status(text: string, live: boolean) {
    const el = $('status')
    el.textContent = text
    el.classList.toggle('live', live)
  }
}
