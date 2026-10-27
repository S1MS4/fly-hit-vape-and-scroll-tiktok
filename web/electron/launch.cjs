// Starts the desktop app with a clean environment. Tools that themselves run
// inside Electron (VS Code extensions, for one) set ELECTRON_RUN_AS_NODE, which
// makes Electron act like plain Node and never open a window.
const { spawn } = require('node:child_process')
const path = require('node:path')
const electron = require('electron') // from Node, this is the path to the Electron binary

const env = { ...process.env }
delete env.ELECTRON_RUN_AS_NODE

const child = spawn(electron, [path.join(__dirname, '..')], { stdio: 'inherit', env })
child.on('exit', (code) => process.exit(code ?? 0))
