// Desktop shell: starts the brain server, then shows the scene in its own
// window, rendered on the high-performance GPU (the RTX 4060 on hybrid laptops).
const { app, BrowserWindow, dialog } = require('electron')
const { spawn } = require('node:child_process')
const http = require('node:http')
const path = require('node:path')
const { startTikTok } = require('./tiktok.cjs')

app.commandLine.appendSwitch('force_high_performance_gpu')
app.commandLine.appendSwitch('ignore-gpu-blocklist')
if (process.env.FLY_DEBUG_PORT) app.commandLine.appendSwitch('remote-debugging-port', process.env.FLY_DEBUG_PORT)

const ROOT = path.resolve(__dirname, '..', '..')
const SERVER = 'http://127.0.0.1:8765/'
let brain = null

function serverUp() {
  return new Promise((resolve) => {
    const req = http.get(`${SERVER}api/meta`, (res) => {
      res.resume()
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.setTimeout(1000, () => {
      req.destroy()
      resolve(false)
    })
  })
}

async function startBrain() {
  if (await serverUp()) return // already running, e.g. started by hand
  const python = process.env.PYTHON || 'python'
  brain = spawn(python, [path.join(ROOT, 'brain', 'server.py')], { cwd: ROOT, stdio: 'inherit', windowsHide: true })
  brain.on('exit', (code) => {
    brain = null
    if (code) console.error(`brain server exited with code ${code}`)
  })
  for (let i = 0; i < 120; i++) {
    if (await serverUp()) return
    if (!brain) throw new Error('The brain server failed to start. Run "python brain/server.py" to see why.')
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error('The brain server did not come up within 60 seconds.')
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: '#ffffff',
    title: 'Fly Brain: Vape & Scroll',
    autoHideMenuBar: true,
    // Keep animating behind other windows: the fly's arm movements are what scroll the feed.
    // The phone plays downloaded TikTok videos, with sound if asked, without anyone clicking.
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), backgroundThrottling: false, autoplayPolicy: 'no-user-gesture-required' },
  })
  // Say why the app goes away: a closed window, or a crashed renderer or GPU process.
  const stamp = () => new Date().toLocaleTimeString()
  win.on('closed', () => console.log(`${stamp()} window closed`))
  win.webContents.on('render-process-gone', (_event, details) =>
    console.error(`${stamp()} renderer process gone: ${details.reason} (exit code ${details.exitCode})`))
  app.on('child-process-gone', (_event, details) => {
    if (details.reason !== 'clean-exit') console.error(`${stamp()} ${details.type} process gone: ${details.reason}`)
  })
  try {
    await startBrain()
  } catch (err) {
    dialog.showErrorBox('Brain server', String(err.message || err))
    app.quit()
    return
  }
  await win.loadURL(process.env.FLY_DEV_URL || SERVER)
  if (!process.env.FLY_NO_TIKTOK) startTikTok(win) // FLY_NO_TIKTOK=1 keeps the built-in clips

  const gpu = await app.getGPUInfo('complete')
  const active = (gpu.gpuDevice || []).find((d) => d.active)
  console.log('GPU in use:', gpu.auxAttributes?.glRenderer || active?.deviceString || 'unknown')
}

app.whenReady().then(createWindow)
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => {
  if (brain) brain.kill()
})
