import * as THREE from 'three'

// The phone screen: TikTok's iPhone app, drawn into a canvas that is shown on the
// 3D phone and downsampled into what the fly's eyes see. The videos are the live
// tiktok.com feed's, downloaded a few ahead by the desktop app (electron/tiktok.cjs)
// so a swipe lands on the next one at once, or built-in clips; the app itself (status
// bar, tabs, action rail, caption, tab bar) is drawn here, filled with the real
// video's author, caption, sound and counts when live.

// iPhone 13 Pro Max screen in points, drawn at 1.5x for a sharp texture.
const W = 428
const H = 926
const SCALE = 1.5
const TAB_H = 83 // tab bar, including the home indicator
const VIDEO_H = H - TAB_H
export const EYE_W = 36
export const EYE_H = 64

export type VideoMeta = {
  author: string
  desc: string
  music: string
  likes: string
  comments: string
  saves: string
  shares: string
  liked: boolean
  avatarUrl: string
  following: boolean
  repostedBy: string // "name reposted" label from TikTok
}

/** The desktop app's bridge to a live tiktok.com feed (electron/preload.cjs); absent in a plain browser. */
export type TikTokStatus = { loggedIn: boolean; video: boolean }
/** What the page reports about the video TikTok is playing. */
type LiveMeta = VideoMeta & { id: string; live?: boolean }
/** A feed item downloaded ahead of time: its video (or a photo post's pictures) and avatar. */
type ItemPayload = { id: string; meta: VideoMeta; duration: number; video: Uint8Array | null; avatar: Uint8Array | null; images: Uint8Array[] }
/** The video TikTok is playing and the ids around it, in feed order. */
type FeedOrder = { activeId: string; order: string[] }
type TikTokBridge = {
  onStatus(callback: (status: TikTokStatus) => void): void
  onMeta(callback: (meta: LiveMeta) => void): void
  onItem(callback: (item: ItemPayload) => void): void
  onFeed(callback: (feed: FeedOrder) => void): void
  showing(id: string): void
  swipe(): void
  like(): void
  follow(): void
  repost(): void
  login(): void
}
declare global {
  interface Window {
    tiktok?: TikTokBridge
  }
}
export const tiktok = window.tiktok

type Clip = {
  meta: VideoMeta
  draw: (ctx: CanvasRenderingContext2D, t: number) => void // fills W x VIDEO_H
  video?: HTMLVideoElement
}

const easeInOut = (k: number) => (k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2)

const fakeMeta = (author: string, desc: string, likes: string, comments: string, saves: string, shares: string): VideoMeta => ({
  author, desc, likes, comments, saves, shares, music: `original sound - ${author}`, liked: false, avatarUrl: '', following: false, repostedBy: '',
})

function builtInClips(): Clip[] {
  const VH = VIDEO_H
  return [
    {
      meta: fakeMeta('discoball', 'POV: the club at 3am 🪩 #nightlife #fyp', '1.2M', '8,412', '96.1K', '31.2K'),
      draw: (ctx, t) => {
        ctx.fillStyle = '#05010a'; ctx.fillRect(0, 0, W, VH)
        for (let i = 0; i < 14; i++) {
          const a = t * 1.7 + (i / 14) * Math.PI * 2
          ctx.fillStyle = `hsla(${(i * 26 + t * 120) % 360}, 95%, 60%, ${0.5 + 0.5 * Math.sin(t * 9 + i)})`
          ctx.beginPath(); ctx.moveTo(W / 2, VH / 2)
          ctx.arc(W / 2, VH / 2, VH, a, a + 0.18); ctx.fill()
        }
      },
    },
    {
      meta: fakeMeta('loomzilla', 'wait for it… 😳 #satisfying', '384K', '2,101', '12.8K', '5,930'),
      draw: (ctx, t) => {
        ctx.fillStyle = '#e9e4d8'; ctx.fillRect(0, 0, W, VH)
        const k = (t % 3) / 3 // an expanding dark disc: a looming object
        ctx.fillStyle = '#111'
        ctx.beginPath(); ctx.arc(W / 2, VH * 0.45, 4 + k ** 3 * 520, 0, Math.PI * 2); ctx.fill()
      },
    },
    {
      meta: fakeMeta('calmwaters', 'ocean asmr for sleep 🌊 #asmr #sleep #relax', '2.4M', '19.5K', '410K', '88K'),
      draw: (ctx, t) => {
        const g = ctx.createLinearGradient(0, 0, 0, VH)
        g.addColorStop(0, '#7fb2d6'); g.addColorStop(1, '#1d4e74')
        ctx.fillStyle = g; ctx.fillRect(0, 0, W, VH)
        ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 3
        for (let y = 260; y < VH; y += 70) {
          ctx.beginPath()
          for (let x = 0; x <= W; x += 12) ctx.lineTo(x, y + 8 * Math.sin(x / 40 + t * 0.8 + y))
          ctx.stroke()
        }
      },
    },
    {
      meta: fakeMeta('bugcam', 'tiny guy doing tiny things 🪰 #insects #macro', '97.3K', '1,204', '4,310', '2,877'),
      draw: (ctx, t) => {
        ctx.fillStyle = '#c9d6a3'; ctx.fillRect(0, 0, W, VH)
        const x = W / 2 + 140 * Math.sin(t * 1.3) * Math.cos(t * 0.4)
        const y = VH / 2 + 260 * Math.sin(t * 0.9)
        ctx.fillStyle = '#1b1b12'
        ctx.beginPath(); ctx.ellipse(x, y, 18, 30, t, 0, Math.PI * 2); ctx.fill()
      },
    },
    {
      meta: fakeMeta('glitchcore', 'my brain on 0.5x speed #glitch #edit', '651K', '6,033', '52K', '19.4K'),
      draw: (ctx, t) => {
        const seed = Math.floor(t * 12)
        for (let i = 0; i < 70; i++) {
          const r = Math.sin(seed * 12.9898 + i * 78.233) * 43758.5453
          const f = r - Math.floor(r)
          ctx.fillStyle = `hsl(${f * 360}, 80%, ${20 + f * 60}%)`
          ctx.fillRect(((f * 7) % 1) * W, ((f * 13) % 1) * VH, 40 + f * 180, 20 + f * 110)
        }
      },
    },
  ]
}

function drawCover(ctx: CanvasRenderingContext2D, source: HTMLVideoElement | ImageBitmap | HTMLCanvasElement, w: number, h: number) {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h)
  const sw = source instanceof HTMLVideoElement ? source.videoWidth : source.width
  const sh = source instanceof HTMLVideoElement ? source.videoHeight : source.height
  if (!sw || !sh) return
  const s = Math.max(w / sw, h / sh)
  ctx.drawImage(source, (w - sw * s) / 2, (h - sh * s) / 2, sw * s, sh * s)
}

/** Like the app: fill the screen with near-portrait video, letterbox anything wider. */
function drawLive(ctx: CanvasRenderingContext2D, frame: HTMLVideoElement | ImageBitmap | HTMLCanvasElement) {
  const fw = frame instanceof HTMLVideoElement ? frame.videoWidth : frame.width
  const fh = frame instanceof HTMLVideoElement ? frame.videoHeight : frame.height
  if (!fw || !fh) return
  const aspect = fw / fh
  if (Math.abs(aspect - W / VIDEO_H) / (W / VIDEO_H) < 0.2) {
    drawCover(ctx, frame, W, VIDEO_H)
    return
  }
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, VIDEO_H)
  const h = W / aspect
  ctx.drawImage(frame, 0, (VIDEO_H - h) / 2, W, h)
}

// ---- icons, in points ---------------------------------------------------------------

function heartPath(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number) {
  const k = size / 100
  ctx.beginPath()
  ctx.moveTo(cx, cy + 32 * k)
  ctx.bezierCurveTo(cx - 62 * k, cy - 8 * k, cx - 36 * k, cy - 62 * k, cx, cy - 28 * k)
  ctx.bezierCurveTo(cx + 36 * k, cy - 62 * k, cx + 62 * k, cy - 8 * k, cx, cy + 32 * k)
  ctx.closePath()
}

function commentIcon(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  ctx.beginPath()
  ctx.ellipse(cx, cy - 1, 17, 14.5, 0, 0, Math.PI * 2)
  ctx.moveTo(cx - 9, cy + 10)
  ctx.lineTo(cx - 13, cy + 19)
  ctx.lineTo(cx + 1, cy + 12)
  ctx.fill()
  ctx.save()
  ctx.shadowColor = 'transparent'
  ctx.fillStyle = '#161823'
  for (const dx of [-7, 0, 7]) {
    ctx.beginPath(); ctx.arc(cx + dx, cy - 1, 2.3, 0, Math.PI * 2); ctx.fill()
  }
  ctx.restore()
}

function bookmarkIcon(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  ctx.beginPath()
  ctx.moveTo(cx - 12, cy - 11)
  ctx.quadraticCurveTo(cx - 12, cy - 16, cx - 7, cy - 16)
  ctx.lineTo(cx + 7, cy - 16)
  ctx.quadraticCurveTo(cx + 12, cy - 16, cx + 12, cy - 11)
  ctx.lineTo(cx + 12, cy + 15)
  ctx.lineTo(cx, cy + 6)
  ctx.lineTo(cx - 12, cy + 15)
  ctx.closePath()
  ctx.fill()
}

function shareIcon(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  ctx.beginPath()
  ctx.moveTo(cx + 16, cy - 2)
  ctx.lineTo(cx + 2, cy - 15)
  ctx.lineTo(cx + 2, cy - 8)
  ctx.bezierCurveTo(cx - 11, cy - 7, cx - 16, cy + 4, cx - 16, cy + 15)
  ctx.bezierCurveTo(cx - 10, cy + 7, cx - 4, cy + 4, cx + 2, cy + 4)
  ctx.lineTo(cx + 2, cy + 11)
  ctx.closePath()
  ctx.fill()
}

function statusBar(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#fff'
  ctx.font = '600 17px -apple-system, "SF Pro Text", system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('9:41', 54, 24)
  for (let i = 0; i < 4; i++) ctx.fillRect(W - 100 + i * 5.5, 29 - (i + 1) * 2.6, 3.5, (i + 1) * 2.6)
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2.2
  ctx.lineCap = 'round'
  for (const r of [5, 9.5]) {
    ctx.beginPath(); ctx.arc(W - 66, 30, r, -Math.PI * 0.75, -Math.PI * 0.25); ctx.stroke()
  }
  ctx.beginPath(); ctx.arc(W - 66, 30, 1.8, 0, Math.PI * 2); ctx.fill()
  ctx.globalAlpha = 0.4
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.roundRect(W - 49, 18, 26, 13, 4); ctx.stroke()
  ctx.fillRect(W - 21.5, 22, 1.8, 5)
  ctx.globalAlpha = 1
  ctx.beginPath(); ctx.roundRect(W - 47, 20, 22, 9, 2.5); ctx.fill()
}

function topTabs(ctx: CanvasRenderingContext2D) {
  const cy = 72
  ctx.textBaseline = 'middle'
  // LIVE
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2
  ctx.beginPath(); ctx.roundRect(14, cy - 11, 30, 22, 6); ctx.stroke()
  ctx.fillStyle = '#fff'
  ctx.font = '800 8px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('LIVE', 29, cy + 0.5)
  // Explore · Following · For You
  const tabs: [string, boolean][] = [['Explore', false], ['Following', false], ['For You', true]]
  ctx.font = '600 17px system-ui, sans-serif'
  const gap = 20
  const widths = tabs.map(([label]) => ctx.measureText(label).width)
  let x = W / 2 - (widths.reduce((a, b) => a + b, 0) + gap * (tabs.length - 1)) / 2
  tabs.forEach(([label, active], i) => {
    ctx.font = `${active ? 700 : 600} 17px system-ui, sans-serif`
    ctx.fillStyle = active ? '#fff' : 'rgba(255,255,255,0.62)'
    ctx.textAlign = 'left'
    ctx.fillText(label, x, cy)
    if (active) {
      ctx.fillStyle = '#fff'
      ctx.beginPath(); ctx.roundRect(x + widths[i] / 2 - 14, cy + 15, 28, 3, 1.5); ctx.fill()
    }
    x += widths[i] + gap
  })
  // search
  ctx.lineWidth = 2.6
  ctx.beginPath(); ctx.arc(W - 27, cy - 2, 8.5, 0, Math.PI * 2); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(W - 20.5, cy + 4.5); ctx.lineTo(W - 14, cy + 11); ctx.stroke()
}

function tabBar(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#000'
  ctx.fillRect(0, VIDEO_H, W, TAB_H)
  const y = VIDEO_H + 21
  const xs = [0, 1, 2, 3, 4].map((i) => (W * (2 * i + 1)) / 10)
  ctx.strokeStyle = '#fff'
  ctx.fillStyle = '#fff'
  ctx.lineWidth = 2
  ctx.lineJoin = 'round'
  // Home (selected, filled)
  ctx.beginPath()
  ctx.moveTo(xs[0] - 11, y - 1); ctx.lineTo(xs[0], y - 11); ctx.lineTo(xs[0] + 11, y - 1)
  ctx.lineTo(xs[0] + 11, y + 11); ctx.lineTo(xs[0] - 11, y + 11); ctx.closePath(); ctx.fill()
  ctx.fillStyle = '#000'; ctx.fillRect(xs[0] - 3, y + 3, 6, 8); ctx.fillStyle = '#fff'
  // Friends
  ctx.globalAlpha = 0.9
  ctx.beginPath(); ctx.arc(xs[1] - 4, y - 5, 5, 0, Math.PI * 2); ctx.stroke()
  ctx.beginPath(); ctx.arc(xs[1] - 4, y + 12, 9, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke()
  ctx.beginPath(); ctx.arc(xs[1] + 7, y - 3, 4, -Math.PI * 0.6, Math.PI * 0.6); ctx.stroke()
  // Inbox
  ctx.beginPath(); ctx.roundRect(xs[3] - 11, y - 9, 22, 19, 5); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(xs[3] - 6, y - 1); ctx.lineTo(xs[3], y + 3); ctx.lineTo(xs[3] + 6, y - 1); ctx.stroke()
  // Profile
  ctx.beginPath(); ctx.arc(xs[4], y - 4, 5.5, 0, Math.PI * 2); ctx.stroke()
  ctx.beginPath(); ctx.arc(xs[4], y + 14, 10, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke()
  ctx.globalAlpha = 1
  // Create (+)
  const px = xs[2]
  ctx.fillStyle = '#25f4ee'; ctx.beginPath(); ctx.roundRect(px - 22, y - 13, 40, 27, 8); ctx.fill()
  ctx.fillStyle = '#fe2c55'; ctx.beginPath(); ctx.roundRect(px - 18, y - 13, 40, 27, 8); ctx.fill()
  ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.roundRect(px - 18, y - 13, 36, 27, 7); ctx.fill()
  ctx.fillStyle = '#000'; ctx.fillRect(px - 7, y - 1.5, 14, 3); ctx.fillRect(px - 1.5, y - 7.5, 3, 14)
  // labels
  ctx.font = '600 10px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const labels: [number, string][] = [[0, 'Home'], [1, 'Friends'], [3, 'Inbox'], [4, 'Profile']]
  for (const [i, label] of labels) {
    ctx.fillStyle = i === 0 ? '#fff' : 'rgba(255,255,255,0.75)'
    ctx.fillText(label, xs[i], y + 25)
  }
  // home indicator
  ctx.fillStyle = '#fff'
  ctx.beginPath(); ctx.roundRect(W / 2 - 67, H - 10, 134, 5, 2.5); ctx.fill()
}

/** Words of `text` laid out in up to `maxLines` lines; hashtags and mentions are bold. */
function layoutCaption(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number) {
  const regular = '400 15px system-ui, sans-serif'
  const bold = '700 15px system-ui, sans-serif'
  const lines: { word: string; bold: boolean; x: number }[][] = [[]]
  let x = 0
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const isTag = /^[#@]/.test(word)
    ctx.font = isTag ? bold : regular
    const w = ctx.measureText(word + ' ').width
    if (x + w > maxWidth && lines[lines.length - 1].length) {
      if (lines.length === maxLines) {
        const last = lines[lines.length - 1]
        last[last.length - 1].word += '…'
        break
      }
      lines.push([])
      x = 0
    }
    lines[lines.length - 1].push({ word, bold: isTag, x })
    x += w
  }
  return { lines, regular, bold }
}

/** The app's fixed parts: top shading, status bar, top tabs and tab bar. */
function drawFixedChrome(ctx: CanvasRenderingContext2D) {
  ctx.save()
  const top = ctx.createLinearGradient(0, 0, 0, 130)
  top.addColorStop(0, 'rgba(0,0,0,0.45)'); top.addColorStop(1, 'rgba(0,0,0,0)')
  ctx.fillStyle = top; ctx.fillRect(0, 0, W, 130)
  statusBar(ctx)
  topTabs(ctx)
  tabBar(ctx)
  ctx.restore()
}

const EMPTY_META = fakeMeta('', '', '', '', '', '')
/** The app shows "original sound - author" when a video has no named sound. */
const withMusic = (meta: VideoMeta): VideoMeta => (meta.music || !meta.author ? meta : { ...meta, music: `original sound - ${meta.author}` })
const decodeImage = (bytes: Uint8Array) => createImageBitmap(new Blob([bytes.slice()]))

const PHOTO_S = 3.5 // how long each picture of a photo post stays up
type Frame = HTMLVideoElement | ImageBitmap

/** A downloaded feed item, decoded and ready to play on the phone. */
class Item {
  readonly id: string
  readonly meta: VideoMeta
  video: HTMLVideoElement | null = null
  avatar: ImageBitmap | null = null
  images: ImageBitmap[] = []
  private ready = false
  private playing = false
  private url = ''
  private disposed = false
  /** The video arrived but the browser could not decode it, so this item can only be black. */
  failed = false

  constructor(payload: ItemPayload, muted: boolean) {
    this.id = payload.id
    this.meta = payload.meta
    // Decoding outlives a swipe, so an item can be thrown away while its pictures are
    // still coming back. Closing them here is the only chance: dispose() has already run
    // and would otherwise leave decoded bitmaps to the garbage collector.
    if (payload.avatar) {
      decodeImage(payload.avatar)
        .then((bitmap) => (this.disposed ? bitmap.close() : (this.avatar = bitmap)))
        .catch(() => {})
    }
    Promise.all(payload.images.map((bytes) => decodeImage(bytes).catch(() => null)))
      .then((bitmaps) => {
        const decoded = bitmaps.filter((b): b is ImageBitmap => b !== null)
        if (this.disposed) for (const image of decoded) image.close()
        else this.images = decoded
      })
    if (!payload.video) return
    const video = document.createElement('video')
    video.muted = muted
    video.loop = true
    video.playsInline = true
    video.preload = 'auto'
    video.addEventListener('loadeddata', () => (this.ready = true))
    // Without this an undecodable file is a black page that never recovers and never
    // explains itself. TikTok also serves H.265, which this window cannot always decode.
    video.addEventListener('error', () => {
      this.failed = true
      console.warn(`TikTok video ${this.id} would not decode: ${video.error?.message || 'unknown error'}`)
    })
    this.url = URL.createObjectURL(new Blob([payload.video.slice()], { type: 'video/mp4' }))
    video.src = this.url
    video.load()
    this.video = video
  }

  /** The current video frame, or the picture showing `t` seconds into a photo post. */
  frame(t: number): Frame | null {
    if (this.video && this.ready && this.video.videoWidth) return this.video
    if (this.images.length) return this.images[Math.floor(t / PHOTO_S) % this.images.length]
    return null
  }

  get progress() {
    const v = this.video
    return v && v.duration ? (v.currentTime / v.duration) % 1 : 0
  }

  setPlaying(on: boolean) {
    if (!this.video || on === this.playing) return
    this.playing = on
    // Retry a refused play, but never a broken one. setPlaying runs for every item on every
    // frame, so a video the browser will not decode would be asked again sixty times a
    // second for as long as the fly sits on it.
    if (on) this.video.play().catch(() => { if (!this.video?.error) this.playing = false })
    else this.video.pause()
  }

  dispose() {
    this.disposed = true
    if (this.video) {
      this.video.pause()
      this.video.removeAttribute('src')
      this.video.load()
    }
    if (this.url) URL.revokeObjectURL(this.url)
    this.avatar?.close()
    for (const image of this.images) image.close()
  }
}

/** What the fly has done to a video, as the app shows it. */
type PageState = { liked: boolean; following: boolean; reposted: boolean }
const untouched = (meta: VideoMeta): PageState => ({ liked: meta.liked, following: meta.following, reposted: false })

/** What fills a feed page: the picture, the author's avatar and how far along the video is. */
type PageContent = { draw: (ctx: CanvasRenderingContext2D) => void; avatar: ImageBitmap | null; progress: number }

/** A swipe in progress: the current video slides up as the next comes in. */
type Slide = {
  k: number
  fromId: string
  fromMeta: VideoMeta
  fromTime: number
  fromState: PageState
  toId: string
}

const SLIDE_S = 0.32

export class Feed {
  readonly canvas = document.createElement('canvas')
  readonly texture: THREE.CanvasTexture
  private ctx: CanvasRenderingContext2D
  private eyeCtx: CanvasRenderingContext2D
  private clips: Clip[] = builtInClips()
  private index = 0
  private clipTime = 0
  private swipeK = -1
  // The live TikTok feed: videos the desktop app downloaded ahead of time, in feed order.
  private items = new Map<string, Item>()
  private order: string[] = [] // ids around the playing video
  private activeId = '' // what TikTok says it is playing
  private currentId = '' // what the phone shows
  private liveMeta: LiveMeta | null = null // the page's own details of the playing video
  private slide: Slide | null = null
  private shown: string[] = [] // what the phone has already put on screen
  private muted = true
  private likedThis = false
  private followedThis = false
  private repostedThis = false
  private toastT = -1
  private toastText = ''
  private heartT = -1
  private disc = 0
  useLive = true
  swipes = 0
  likes = 0
  follows = 0
  reposts = 0

  constructor() {
    this.canvas.width = W * SCALE
    this.canvas.height = H * SCALE
    this.ctx = this.canvas.getContext('2d')!
    this.ctx.scale(SCALE, SCALE)
    const eye = document.createElement('canvas')
    eye.width = EYE_W
    eye.height = EYE_H
    this.eyeCtx = eye.getContext('2d', { willReadFrequently: true })!
    this.texture = new THREE.CanvasTexture(this.canvas)
    this.texture.colorSpace = THREE.SRGBColorSpace
    this.texture.anisotropy = 8

    if (!tiktok) return
    tiktok.onMeta((meta) => {
      this.liveMeta = meta
    })
    tiktok.onItem((payload) => {
      const existing = this.items.get(payload.id)
      // Replacing what is on screen would restart it from zero mid-watch, so keep the copy
      // that is playing. The exception is a copy that finally has a video where the one
      // showing has none, since refusing that would leave a black page rejecting its own fix.
      const better = Boolean(payload.video) && !existing?.video
      if (existing && !better && (payload.id === this.currentId || payload.id === this.slide?.fromId)) return
      existing?.dispose()
      this.items.set(payload.id, new Item(payload, this.muted))
    })
    tiktok.onFeed(({ activeId, order }) => {
      this.activeId = activeId
      this.order = order
      // Drop what the feed has left behind, except what is on screen.
      for (const [id, item] of this.items) {
        if (order.includes(id) || id === this.currentId || id === this.slide?.fromId) continue
        item.dispose()
        this.items.delete(id)
      }
      this.followTikTok()
    })
  }

  /** The live feed shows in the desktop app; a plain browser gets the built-in clips. */
  get showingLive() {
    return this.useLive && tiktok !== undefined
  }

  get current() {
    return this.clips[this.index]
  }

  addVideos(files: FileList) {
    this.useLive = false // your own videos replace the live feed
    const added: Clip[] = []
    for (const file of Array.from(files)) {
      const video = document.createElement('video')
      video.src = URL.createObjectURL(file)
      video.muted = true
      video.loop = true
      video.playsInline = true
      const meta = fakeMeta('you', file.name.replace(/\.[^.]+$/, ''), '0', '0', '0', '0')
      added.push({ meta, video, draw: (ctx) => drawCover(ctx, video, W, VIDEO_H) })
    }
    this.current.video?.pause()
    this.clips.splice(this.index, 0, ...added)
    this.clipTime = 0
    this.current.video?.play()
  }

  swipe() {
    if (!this.showingLive) {
      this.likedThis = this.followedThis = this.repostedThis = false
      if (this.swipeK < 0) this.swipeK = 0
      return
    }
    if (this.slide) return
    tiktok!.swipe() // TikTok scrolls to the next video itself
    this.swipes++
    const fromId = this.currentId
    const next = this.nextId(fromId)
    // Nothing known to move to: the fly has got ahead of TikTok's own list. TikTok is
    // scrolling regardless, so stay on this video and slide when the next one arrives,
    // rather than sliding onto a blank page.
    if (!next) return
    // Otherwise the next video is already downloaded, so the phone doesn't wait for TikTok.
    const fromMeta = this.metaFor(fromId)
    const fromState = { liked: this.likedThis || fromMeta.liked, following: this.followedThis || fromMeta.following, reposted: this.repostedThis }
    this.likedThis = this.followedThis = this.repostedThis = false
    this.startSlide(fromId, next, fromState)
  }

  /** The video after `id` in the feed, as far as the app knows. */
  private nextId(id: string) {
    let pos = this.order.indexOf(id)
    if (pos < 0) pos = this.order.indexOf(this.activeId)
    const next = this.order[pos + 1] ?? ''
    // With neither id in the window the index above is -1, and this would otherwise hand
    // back order[0], which is the slot kept behind the phone: the video just left. Empty
    // means the swipe holds the current video until the feed says what really comes next.
    return next && this.shown.includes(next) ? '' : next
  }

  private startSlide(fromId: string, toId: string, fromState: PageState) {
    this.slide = { k: 0, fromId, fromMeta: this.metaFor(fromId), fromTime: this.clipTime, fromState, toId }
    this.currentId = toId
    this.clipTime = 0
    if (!toId) return
    // The app downloads around whatever the phone is showing, so it has to be told.
    this.shown.push(toId)
    if (this.shown.length > 200) this.shown.splice(0, 100)
    tiktok?.showing(toId)
  }

  /** Catch up when TikTok is further down the feed than the phone: it moved by itself, or the guess at the next video was wrong. */
  private followTikTok() {
    if (this.slide || !this.showingLive || !this.activeId || this.activeId === this.currentId) return
    const active = this.order.indexOf(this.activeId)
    const current = this.order.indexOf(this.currentId)
    // Only ever forward. TikTok reports the video it is playing, which lags the phone
    // while it scrolls, and it reports nothing at all for a photo post, which has no
    // video to read; neither is a reason to go back to what the fly has left.
    if (this.shown.includes(this.activeId)) return // already watched and left
    if (active < 0 || (current >= 0 && active <= current)) return
    this.startSlide(this.currentId, this.activeId, this.pageState(this.metaFor(this.currentId)))
    this.likedThis = this.followedThis = this.repostedThis = false
  }

  /** A video's details: what the page says while it plays, over what the feed API listed. */
  private metaFor(id: string): VideoMeta {
    const item = this.items.get(id)
    const live = this.liveMeta && this.liveMeta.id === id ? withMusic(this.liveMeta) : null
    if (!item) return live ?? EMPTY_META
    if (!live) return item.meta
    const meta = { ...item.meta, liked: live.liked, following: live.following, repostedBy: live.repostedBy }
    for (const key of ['desc', 'music'] as const) if (live[key]) meta[key] = live[key]
    for (const key of ['likes', 'comments', 'saves', 'shares'] as const) if (/\d/.test(live[key])) meta[key] = live[key]
    return meta
  }

  /** Sound from the phone's videos; off by default. */
  setMuted(muted: boolean) {
    this.muted = muted
    for (const item of this.items.values()) if (item.video) item.video.muted = muted
  }

  /** For checking the feed from outside: what is downloaded and what is playing. */
  debug() {
    const item = this.items.get(this.currentId) ?? null
    const video = item?.video ?? null
    return {
      currentId: this.currentId,
      activeId: this.activeId,
      order: this.order,
      items: [...this.items.values()].map((i) => ({ id: i.id, author: i.meta.author, video: !!i.video, ready: !!i.frame(0), failed: i.failed, avatar: !!i.avatar, images: i.images.length })),
      playing: video ? { t: video.currentTime, paused: video.paused, w: video.videoWidth, h: video.videoHeight, muted: video.muted } : null,
      sliding: !!this.slide,
      swipes: this.swipes,
    }
  }

  snapshot(quality = 0.6) {
    return this.canvas.toDataURL('image/jpeg', quality)
  }

  /** Like what's on screen, at most once per video. */
  like() {
    if (this.likedThis) return
    this.likedThis = true
    this.likes++
    this.heartT = 0
    if (this.showingLive) tiktok!.like()
  }

  /** Follow the creator on screen. */
  follow() {
    if (this.followedThis || (this.showingLive && this.metaFor(this.currentId).following)) return
    this.followedThis = true
    this.follows++
    if (this.showingLive) tiktok!.follow()
  }

  /** Repost what's on screen, at most once per video. */
  repost() {
    if (this.repostedThis) return
    this.repostedThis = true
    this.reposts++
    this.toastT = 0
    this.toastText = 'Reposted'
    if (this.showingLive) tiktok!.repost()
  }

  private pageState(meta: VideoMeta): PageState {
    return { liked: this.likedThis || meta.liked, following: this.followedThis || meta.following, reposted: this.repostedThis }
  }

  update(dt: number) {
    this.clipTime += dt
    this.disc += dt * 0.9
    const ctx = this.ctx
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, W, H)
    if (this.showingLive) this.drawLiveFeed(dt)
    else this.drawClips(dt)
    this.drawHeart(dt)
    this.drawToast(dt)
    drawFixedChrome(ctx)
    this.texture.needsUpdate = true
  }

  /** Downsample the screen to EYE_W x EYE_H luminance for the photoreceptors. */
  sampleLuminance(): Uint8Array {
    this.eyeCtx.drawImage(this.canvas, 0, 0, EYE_W, EYE_H)
    const px = this.eyeCtx.getImageData(0, 0, EYE_W, EYE_H).data
    const lum = new Uint8Array(EYE_W * EYE_H)
    for (let i = 0; i < lum.length; i++) {
      lum[i] = (0.2126 * px[i * 4] + 0.7152 * px[i * 4 + 1] + 0.0722 * px[i * 4 + 2]) | 0
    }
    return lum
  }

  /** An item's page: its video, a photo, or black while it downloads. */
  private pageOf(item: Item | null, time: number): PageContent {
    const frame = item?.frame(time) ?? null
    return {
      draw: (ctx) => {
        if (frame) drawLive(ctx, frame)
        else { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, VIDEO_H) }
      },
      avatar: item?.avatar ?? null,
      progress: item?.video ? item.progress : (time % 15) / 15,
    }
  }

  private drawLiveFeed(dt: number) {
    this.followTikTok()
    for (const [id, item] of this.items) item.setPlaying(id === this.currentId || id === this.slide?.fromId)
    const s = this.slide
    if (s) {
      s.k = Math.min(1, s.k + dt / SLIDE_S)
      const k = easeInOut(s.k)
      const toMeta = this.metaFor(s.toId)
      this.drawPage(this.pageOf(this.items.get(s.fromId) ?? null, s.fromTime), s.fromMeta, -k * VIDEO_H, s.fromTime, s.fromState)
      this.drawPage(this.pageOf(this.items.get(s.toId) ?? null, this.clipTime), toMeta, (1 - k) * VIDEO_H, this.clipTime, untouched(toMeta))
      if (s.k >= 1) this.slide = null
      return
    }
    const meta = this.metaFor(this.currentId)
    this.drawPage(this.pageOf(this.items.get(this.currentId) ?? null, this.clipTime), meta, 0, this.clipTime, this.pageState(meta))
  }

  private clipPage(clip: Clip, time: number): PageContent {
    return { draw: (ctx) => clip.draw(ctx, time), avatar: null, progress: (time % 15) / 15 }
  }

  private drawClips(dt: number) {
    if (this.swipeK >= 0) {
      this.swipeK = Math.min(1, this.swipeK + dt / SLIDE_S)
      const k = easeInOut(this.swipeK)
      const current = this.current
      const next = this.clips[(this.index + 1) % this.clips.length]
      const time = this.clipTime
      this.drawPage(this.clipPage(current, time), current.meta, -k * VIDEO_H, time, untouched(current.meta))
      this.drawPage(this.clipPage(next, 0), next.meta, (1 - k) * VIDEO_H, 0, untouched(next.meta))
      if (this.swipeK >= 1) {
        current.video?.pause()
        this.index = (this.index + 1) % this.clips.length
        this.clipTime = 0
        this.swipeK = -1
        this.swipes++
        this.current.video?.play()
      }
      return
    }
    const clip = this.current
    const time = this.clipTime
    this.drawPage(this.clipPage(clip, time), clip.meta, 0, time, this.pageState(clip.meta))
  }

  /** One feed item (video, shading, action rail, caption, progress), shifted down by offsetY. */
  private drawPage(content: PageContent, meta: VideoMeta, offsetY: number, time: number, state: PageState) {
    if (Math.abs(offsetY) >= VIDEO_H) return
    const ctx = this.ctx
    ctx.save()
    ctx.beginPath(); ctx.rect(0, 0, W, VIDEO_H); ctx.clip()
    ctx.translate(0, offsetY)
    content.draw(ctx)
    // Shade the bottom of the video so white text reads on any footage.
    const bottom = ctx.createLinearGradient(0, VIDEO_H - 260, 0, VIDEO_H)
    bottom.addColorStop(0, 'rgba(0,0,0,0)'); bottom.addColorStop(1, 'rgba(0,0,0,0.5)')
    ctx.fillStyle = bottom; ctx.fillRect(0, VIDEO_H - 260, W, 260)
    this.drawRail(meta, state, content.avatar)
    this.drawCaption(meta, time, state)
    // playback progress
    ctx.fillStyle = 'rgba(255,255,255,0.22)'
    ctx.fillRect(0, VIDEO_H - 2, W, 2)
    ctx.fillStyle = 'rgba(255,255,255,0.85)'
    ctx.fillRect(0, VIDEO_H - 2, W * content.progress, 2)
    ctx.restore()
  }

  private drawRail(meta: VideoMeta, state: PageState, avatar: ImageBitmap | null) {
    const ctx = this.ctx
    const cx = W - 34
    ctx.save()
    ctx.shadowColor = 'rgba(0,0,0,0.35)'
    ctx.shadowBlur = 4
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.font = '600 13px system-ui, sans-serif'

    // avatar, with the follow badge until the fly follows
    const ay = VIDEO_H - 404
    this.drawAvatar(avatar, meta.author, cx, ay, 24)
    ctx.shadowColor = 'transparent'
    if (!state.following) {
      ctx.fillStyle = '#fe2c55'
      ctx.beginPath(); ctx.arc(cx, ay + 24, 10, 0, Math.PI * 2); ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.fillRect(cx - 5, ay + 23, 10, 2); ctx.fillRect(cx - 1, ay + 19, 2, 10)
    }
    ctx.shadowColor = 'rgba(0,0,0,0.35)'

    const rows: [number, (y: number) => void, string][] = [
      [VIDEO_H - 318, (y) => { heartPath(ctx, cx, y, 38); ctx.fillStyle = state.liked ? '#fe2c55' : '#fff'; ctx.fill() }, meta.likes],
      [VIDEO_H - 244, (y) => { ctx.fillStyle = '#fff'; commentIcon(ctx, cx, y) }, meta.comments],
      [VIDEO_H - 172, (y) => { ctx.fillStyle = '#fff'; bookmarkIcon(ctx, cx, y) }, meta.saves],
      [VIDEO_H - 100, (y) => { ctx.fillStyle = '#fff'; shareIcon(ctx, cx, y) }, meta.shares],
    ]
    for (const [y, icon, text] of rows) {
      icon(y)
      ctx.fillStyle = '#fff'
      ctx.fillText(text, cx, y + 30)
    }

    // spinning sound disc
    const dy = VIDEO_H - 36
    ctx.save()
    ctx.translate(cx, dy)
    ctx.rotate(this.disc)
    const ring = ctx.createRadialGradient(0, 0, 8, 0, 0, 23)
    ring.addColorStop(0, '#333'); ring.addColorStop(0.6, '#161616'); ring.addColorStop(1, '#3a3a3a')
    ctx.fillStyle = ring
    ctx.beginPath(); ctx.arc(0, 0, 23, 0, Math.PI * 2); ctx.fill()
    ctx.restore()
    ctx.shadowColor = 'transparent'
    this.drawAvatar(avatar, meta.author, cx, dy, 12, this.disc)
    ctx.restore()
  }

  private drawAvatar(image: ImageBitmap | null, author: string, cx: number, cy: number, r: number, rotation = 0) {
    const ctx = this.ctx
    ctx.save()
    ctx.fillStyle = '#fff'
    ctx.beginPath(); ctx.arc(cx, cy, r + 1.5, 0, Math.PI * 2); ctx.fill()
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip()
    ctx.translate(cx, cy)
    ctx.rotate(rotation)
    if (image) {
      ctx.drawImage(image, -r, -r, r * 2, r * 2)
    } else {
      const hue = [...author].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 200)
      ctx.fillStyle = `hsl(${hue}, 45%, 45%)`
      ctx.fillRect(-r, -r, r * 2, r * 2)
      ctx.fillStyle = '#fff'
      ctx.font = `700 ${Math.round(r * 1.1)}px system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText((author[0] ?? '').toUpperCase(), 0, 1)
    }
    ctx.restore()
  }

  private drawCaption(meta: VideoMeta, time: number, state: PageState) {
    const ctx = this.ctx
    const left = 14
    const maxWidth = W - 100
    ctx.save()
    ctx.shadowColor = 'rgba(0,0,0,0.4)'
    ctx.shadowBlur = 3
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'

    // sound: ♫ + scrolling title
    const musicY = VIDEO_H - 20
    ctx.font = '400 14px system-ui, sans-serif'
    if (meta.music) {
      ctx.fillText('♫', left, musicY)
      const text = `${meta.music}     `
      const width = ctx.measureText(text).width
      const offset = (time * 32) % width
      ctx.save()
      ctx.beginPath(); ctx.rect(left + 20, musicY - 18, 190, 26); ctx.clip()
      ctx.fillText(text + text, left + 20 - offset, musicY)
      ctx.restore()
    }

    // description, up to two lines
    const { lines, regular, bold } = layoutCaption(ctx, meta.desc, maxWidth, 2)
    let y = musicY - 30 - (lines.length - 1) * 21
    const descTop = y
    for (const line of lines) {
      for (const { word, bold: isBold, x } of line) {
        ctx.font = isBold ? bold : regular
        ctx.fillText(word, left + x, y)
      }
      y += 21
    }

    // author, and who reposted it
    const authorY = descTop - (meta.desc ? 28 : 6)
    if (meta.author) {
      ctx.font = '700 17px system-ui, sans-serif'
      ctx.fillText(meta.author, left, authorY)
    }
    const repost = state.reposted ? 'You reposted' : meta.repostedBy ? `${meta.repostedBy} reposted` : ''
    if (repost) {
      ctx.font = '600 13px system-ui, sans-serif'
      const width = ctx.measureText(repost).width + 34
      ctx.save()
      ctx.shadowColor = 'transparent'
      ctx.fillStyle = 'rgba(255,255,255,0.2)'
      ctx.beginPath(); ctx.roundRect(left, authorY - 46, width, 26, 13); ctx.fill()
      ctx.restore()
      ctx.fillText('↻', left + 10, authorY - 28)
      ctx.fillText(repost, left + 26, authorY - 28)
    }
    ctx.restore()
  }

  /** The app's small confirmation toast, e.g. after a repost. */
  private drawToast(dt: number) {
    if (this.toastT < 0) return
    this.toastT += dt
    if (this.toastT > 1.6) {
      this.toastT = -1
      return
    }
    const ctx = this.ctx
    ctx.save()
    ctx.globalAlpha = Math.min(1, this.toastT / 0.15, (1.6 - this.toastT) / 0.3)
    ctx.font = '600 15px system-ui, sans-serif'
    const width = ctx.measureText(this.toastText).width + 44
    const y = VIDEO_H * 0.62
    ctx.fillStyle = 'rgba(37,37,37,0.92)'
    ctx.beginPath(); ctx.roundRect(W / 2 - width / 2, y, width, 42, 8); ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(this.toastText, W / 2, y + 21)
    ctx.restore()
  }

  /** TikTok's pop-up heart after a double-tap like. */
  private drawHeart(dt: number) {
    if (this.heartT < 0) return
    this.heartT += dt
    const k = this.heartT / 0.9
    if (k >= 1) {
      this.heartT = -1
      return
    }
    const ctx = this.ctx
    const size = 110 + 60 * Math.sin((Math.min(1, k * 3) * Math.PI) / 2)
    ctx.save()
    ctx.globalAlpha = 1 - k * k
    ctx.translate(W / 2, VIDEO_H * 0.45 - 50 * k)
    ctx.rotate(-0.25)
    heartPath(ctx, 0, 0, size)
    const fill = ctx.createLinearGradient(0, -size / 2, 0, size / 2)
    fill.addColorStop(0, '#ff5a7a'); fill.addColorStop(1, '#fe2c55')
    ctx.fillStyle = fill
    ctx.fill()
    ctx.restore()
  }
}
