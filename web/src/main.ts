import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import './app.css'
import { BrainPanel } from './brainPanel'
import { BrainView } from './brainView'
import { Fly } from './fly'
import { Hud } from './hud'
import { findMesh, loadModels } from './models'
import { BrainLink, fetchMeta, fetchSomata } from './net'
import { EYE_H, EYE_W, Feed, tiktok } from './phone'
import { addShadowCatcher, applyLightmaps, heightProbe, restOn, SKY_COLOR, SUN_COLOR, SUN_POSITION } from './room'
import { Smoke } from './smoke'
import { Vape } from './vape'

// ---- the fly on the living-room sofa ------------------------------------------------
// Scene units are fly-sized (the fly's body is 0.85 long); the room is in metres.
// ROOM_SCALE is units per metre: 3 makes the fly ~28 cm, so the sofa and room read at
// normal detail. 120 would be a real 7 mm fly, but the sofa then looks like rock.
const ROOM_SCALE = 3
const SOFA_SPOT = new THREE.Vector3(-2.3, 0.527, -0.675) // centre of a seat cushion (not the seam), room metres
// Disposables: puffs in each one, the flavour hues, and where the spares stand on the
// cushion. A spot with no cushion under it is skipped, so these can be generous.
const VAPE_PUFFS = 6
const FLAVOURS = [0.02, 0.3, 0.55, 0.83] // watermelon, apple, blue razz, grape
const SPARE_SPOTS: [number, number][] = [
  [0.95, -0.3],
  [-1.05, 0.3],
  [1.35, 0.55],
]

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', logarithmicDepthBuffer: true })
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
renderer.setSize(innerWidth, innerHeight)
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFShadowMap
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 0.95 // a stop down: the sun is doing the lifting, and the shade should stay deep
document.getElementById('scene')!.appendChild(renderer.domElement)

/** Late afternoon outside: warm haze along the horizon, deeper blue overhead. */
function duskSky() {
  const canvas = document.createElement('canvas')
  canvas.width = 2
  canvas.height = 256
  const ctx = canvas.getContext('2d')!
  const sky = ctx.createLinearGradient(0, 0, 0, 256)
  sky.addColorStop(0.0, '#2b3a63') // zenith
  sky.addColorStop(0.45, '#7c6a83')
  sky.addColorStop(0.62, '#e0914f') // haze along the horizon
  sky.addColorStop(0.78, '#ffbb6d')
  sky.addColorStop(1.0, '#4a3524') // ground
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, 2, 256)
  const texture = new THREE.CanvasTexture(canvas)
  texture.mapping = THREE.EquirectangularReflectionMapping
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

const scene = new THREE.Scene()
scene.background = duskSky()
const pmrem = new THREE.PMREMGenerator(renderer)
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture // until the room is captured

const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.05, 2000)
// Three-quarter view from the fly's right: the fly, the vape, the screen and the sofa.
camera.position.set(-2.3, 1.5, 0.05)
const controls = new OrbitControls(camera, renderer.domElement)
controls.target.set(0, 0.35, 0.3)
controls.enableDamping = true
controls.minDistance = 0.3
controls.maxDistance = 80 // free to roam the whole room
controls.zoomToCursor = true
controls.listenToKeyEvents(window) // arrow keys pan; right-drag pans too

// Golden hour. The sun is low and off to the fly's left, where the window is (room.ts
// works out where the bake put it), so it comes in over the fly's shoulder: an amber key
// that rims it against the sofa and rakes a long shadow back toward the camera.
const KEY_TARGET = new THREE.Vector3(0, 0.2, 0.15)
const key = new THREE.DirectionalLight(SUN_COLOR, 6.5)
key.target.position.copy(KEY_TARGET)
// SUN_POSITION sits a fraction inside the window reveal, which leaves the piers between the
// openings behind the shadow camera, where they can never cast. Slide the light back along
// its own direction until it is outside the glass: only the distance changes, so the
// direction — and every rim the room was baked and relit to match — is exactly as it was.
key.position
  .copy(SUN_POSITION)
  .sub(KEY_TARGET)
  .setLength(SUN_POSITION.distanceTo(KEY_TARGET) + 4)
  .add(KEY_TARGET)
key.castShadow = true
key.shadow.mapSize.set(2048, 2048)
key.shadow.radius = 3
key.shadow.bias = -0.0004
key.shadow.normalBias = 0.02 // a sun this low grazes the cushion, which would otherwise self-shadow
Object.assign(key.shadow.camera, { left: -4.6, right: 4.6, top: 4.6, bottom: -4.6, near: 1.5, far: 24 })
// LightShadow.updateMatrices never calls this and the shadow map code only does it when it
// allocates, so a frustum assigned in place has to be committed by hand.
key.shadow.camera.updateProjectionMatrix()
scene.add(key, key.target)

// Skylight from the other side, so the shadow side of the fly stays blue rather than black.
const skyFill = new THREE.DirectionalLight(SKY_COLOR, 0.7)
skyFill.position.set(-6, 5, -4)
scene.add(skyFill)

// Cool from the sky above, warm from the sunlit cushion below.
const bounce = new THREE.HemisphereLight(SKY_COLOR, 0xffa864, 0.5)
scene.add(bounce)

const smoke = new Smoke()
scene.add(smoke.points)
const fitSmoke = () =>
  smoke.setScale(renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)))
fitSmoke()

// ---- brain link, HUD and the brain panel -------------------------------------------
const hud = new Hud()
const link = new BrainLink()
const feed = new Feed()
;(window as unknown as { __feed: Feed }).__feed = feed // for looking at the feed from the devtools or a debugger
const brainPanel = new BrainPanel(document.getElementById('brain-canvas')!)
const brainFiring = document.getElementById('brain-firing')!
let firing = 0
let brainT = 0 // brain seconds, for the log
let puffBlocked = false // the brain is coughing or slumped, so it would refuse a puff
let fly: Fly | null = null
let vape: Vape | null = null
// Every vape in the room: the one being held, the spares the fly can go and fetch, and
// the dead ones lying where they were thrown. All of them get updated, because a flicked
// one is still tumbling for a second after it leaves the fly's hand.
const vapes: Vape[] = []
const spares: Vape[] = []
// For poking at the scene from the devtools or a test driver: the fly, every vape in the
// room and the camera. `fly` is a getter because the models load asynchronously.
;(window as unknown as Record<string, unknown>).__debug = {
  get fly() {
    return fly
  },
  vapes,
  spares,
  camera,
  controls,
  feed,
  get ground() {
    return groundProbe
  },
}
let phone: THREE.Group | null = null
let brainView: BrainView | null = null
// The cushion-height probe, once the sofa has loaded. Exposed on __debug so the spare
// spots can be checked against the real cushion from the devtools.
let groundProbe: ((x: number, z: number) => number | null) | null = null

link.onStatus = (text, live) => hud.status(text, live)
link.onSpikes = (neurons) => {
  firing = neurons.length
  brainView?.flash(neurons)
  brainFiring.textContent = `${firing.toLocaleString()} firing`
}
link.onState = (s) => {
  brainT = s.t
  puffBlocked = s.cough_s_left > 0 || s.whitey_s_left > 0
  hud.update(s, firing, { swipes: feed.swipes, likes: feed.likes, follows: feed.follows, reposts: feed.reposts })
  fly?.setDrive(s.walk * 0.3, s.turn * 0.3, Math.min(1, s.nicotine))
  for (const action of s.actions) {
    hud.log(action, s.t)
    if (action === 'puff') fly?.puff()
    if (action === 'swipe') fly?.swipe()
    if (action === 'like') fly?.like()
    if (action === 'follow') fly?.follow()
    if (action === 'repost') fly?.repost()
    if (action === 'escape') fly?.escape()
    // The brain says how long each of these lasts, so the pose holds for exactly that.
    if (action === 'cough') fly?.cough(s.cough_s_left)
    if (action === 'whitey') fly?.whitey(s.whitey_s_left)
  }
}

document.getElementById('btn-puff')!.onclick = () => {
  // A convulsing or slumped fly can't get the vape to its mouth, and brain/world.py
  // refuses the dose, so don't play a puff the brain is going to throw away.
  if (puffBlocked) return
  link.command('puff')
  fly?.puff()
}
document.getElementById('btn-like')!.onclick = () => fly?.like()
document.getElementById('btn-follow')!.onclick = () => fly?.follow()
document.getElementById('btn-repost')!.onclick = () => fly?.repost()

// Live TikTok (desktop app only): log the fly's account in once; sound starts muted.
const tiktokBtn = document.getElementById('btn-tiktok')!
const soundBtn = document.getElementById('btn-sound')!
if (tiktok) {
  const bridge = tiktok
  tiktokBtn.hidden = soundBtn.hidden = false
  tiktokBtn.onclick = () => bridge.login()
  let muted = true
  soundBtn.onclick = () => {
    muted = !muted
    feed.setMuted(muted)
    soundBtn.textContent = muted ? 'Sound off' : 'Sound on'
  }
  tiktok.onStatus((s) => {
    tiktokBtn.textContent = s.loggedIn ? 'TikTok: logged in' : 'Log in to TikTok'
  })
}

const pauseBtn = document.getElementById('btn-pause')!
let paused = false
pauseBtn.onclick = () => {
  paused = !paused
  link.command(paused ? 'pause' : 'resume')
  pauseBtn.textContent = paused ? 'Resume' : 'Pause'
}
document.getElementById('btn-reset')!.onclick = () => {
  link.command('reset')
  hud.clearLog()
}
const videoInput = document.getElementById('videos') as HTMLInputElement
videoInput.onchange = () => {
  if (videoInput.files?.length) feed.addVideos(videoInput.files)
}

// Which GPU is drawing this, e.g. "NVIDIA GeForce RTX 4060 Laptop GPU".
const gl = renderer.getContext()
const info = gl.getExtension('WEBGL_debug_renderer_info')
const rendererName = info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : 'unknown'
const gpuName = (/ANGLE \([^,]+, ([^(,]+)/.exec(rendererName)?.[1] ?? rendererName).trim()
const gpuEl = document.getElementById('gpu')!
gpuEl.textContent = gpuName
gpuEl.title = rendererName

/** Light the fly and its props with the room itself, as seen from the sofa. */
function captureRoomLight() {
  const target = new THREE.WebGLCubeRenderTarget(256, { type: THREE.HalfFloatType })
  const cubeCamera = new THREE.CubeCamera(0.3, 2000, target)
  cubeCamera.position.set(0, 0.7, 0.1)
  const props = [fly?.root, phone, smoke.points, ...vapes.map((pen) => pen.pen)]
  for (const o of props) if (o) o.visible = false
  cubeCamera.update(renderer, scene)
  for (const o of props) if (o) o.visible = true
  scene.environment = pmrem.fromCubemap(target.texture).texture
  target.dispose()
}

// ---- models ---------------------------------------------------------------------------
loadModels()
  .then(async (models) => {
    // Put the sofa spot at the origin, turned so the sofa faces +Z like the fly.
    const room = new THREE.Group()
    room.scale.setScalar(ROOM_SCALE)
    room.rotation.y = -Math.PI / 2
    models.room.position.copy(SOFA_SPOT).negate()
    room.add(models.room)
    scene.add(room)
    if (models.lightmaps) await applyLightmaps(models.room, models.lightmaps)
    else console.warn('Room lighting is not baked yet: run tools/bake_room.py')
    const sofa = findMesh(models.room, 'Leather_Sofa')
    if (sofa) addShadowCatcher(sofa)
    room.updateMatrixWorld(true)
    // The phone, the vape and the fly's feet rest on the real cushion surface.
    const ground: (x: number, z: number) => number | null = sofa ? heightProbe(sofa) : () => 0
    groundProbe = ground

    const screen = findMesh(models.phone, 'Wallpaper')
    if (!screen) throw new Error('the phone model has no screen mesh')
    screen.material = new THREE.MeshBasicMaterial({ map: feed.texture, toneMapped: false, side: THREE.DoubleSide })
    phone = new THREE.Group()
    phone.add(models.phone)
    phone.position.set(0, 0.5, 0.44) // screen faces -Z, toward the fly
    phone.rotation.x = 0.08
    scene.add(phone)
    restOn(phone, ground)

    // One in its hand and a few spares around the cushion, each a different flavour.
    // Every one gets its own clone, including the first: the Vape constructor rescales and
    // repositions whatever it is handed, so the loaded model has to stay untouched or each
    // clone would inherit the last one's transform. Clones share geometry and textures, so
    // a spare costs almost nothing.
    const addVape = (index: number) => {
      const made = new Vape(models.vape.clone(), 0.46, {
        hue: FLAVOURS[index % FLAVOURS.length],
        puffs: VAPE_PUFFS,
      })
      scene.add(made.pen)
      vapes.push(made)
      return made
    }
    vape = addVape(0)
    for (const [x, z] of SPARE_SPOTS) {
      const y = ground(x, z)
      if (y === null) continue // no cushion there, so nothing could stand on it
      const spare = addVape(vapes.length)
      spare.pen.position.set(x, y + spare.length / 2 + 0.002, z)
      spares.push(spare)
    }

    fly = new Fly(models.fly, models.rig, { vape, screen, smoke })
    fly.onSwipeStroke = () => feed.swipe()
    fly.onTap = (kind) => (kind === 'like' ? feed.like() : kind === 'follow' ? feed.follow() : feed.repost())
    // A dead battery sends it off to the nearest spare it hasn't used yet.
    fly.onNeedVape = () => {
      if (!spares.length) {
        // Distinct from the 'dead' logged when one is thrown away, so the log tells
        // "it went and got another" apart from "there are none left".
        hud.log('out of vapes', brainT) // it keeps sucking on a flat one
        return null
      }
      const here = fly?.root.position ?? new THREE.Vector3()
      spares.sort((a, b) => a.pen.position.distanceToSquared(here) - b.pen.position.distanceToSquared(here))
      hud.log('fetch', brainT)
      return spares.shift()!
    }
    fly.onFlick = () => hud.log('dead', brainT)
    fly.root.position.z = 0.12 // close enough to reach the vape and the screen
    fly.root.position.y = ground(0, 0.12 - 0.28) ?? 0 // abdomen on the cushion
    scene.add(fly.root)
    fly.settle(ground)

    scene.updateMatrixWorld(true)
    captureRoomLight()
  })
  .catch((err) => hud.status(`could not load models: ${err}`, false))

Promise.all([fetchMeta(), fetchSomata()])
  .then(([meta, somata]) => {
    hud.setMeta(meta)
    brainView = new BrainView(somata, meta.categories, meta.n, -1, true)
    brainPanel.setBrain(brainView)
  })
  .catch(() => hud.status('start the brain: python brain/server.py', false))
link.connect()

// ---- main loop -------------------------------------------------------------------------
const timer = new THREE.Timer()
timer.connect(document)
let sinceFrame = 0

renderer.setAnimationLoop((time) => {
  timer.update(time)
  const dt = Math.min(timer.getDelta(), 0.05)
  const t = timer.getElapsed()

  feed.update(dt)
  sinceFrame += dt
  if (sinceFrame >= 0.05) {
    sinceFrame = 0
    link.sendFrame(EYE_W, EYE_H, feed.sampleLuminance())
  }

  fly?.update(dt, t)
  for (const pen of vapes) pen.update(dt) // a flicked one is still tumbling
  smoke.update(dt, t)

  controls.update()
  renderer.render(scene, camera)
  brainPanel.update(dt)
})

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
  fitSmoke()
})
