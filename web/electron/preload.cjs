// Gives the scene page a small bridge to the live TikTok feed (electron/tiktok.cjs).
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('tiktok', {
  onStatus: (callback) => ipcRenderer.on('tiktok:status', (_event, status) => callback(status)),
  onMeta: (callback) => ipcRenderer.on('tiktok:meta', (_event, meta) => callback(meta)),
  onItem: (callback) => ipcRenderer.on('tiktok:item', (_event, item) => callback(item)),
  onFeed: (callback) => ipcRenderer.on('tiktok:feed', (_event, feed) => callback(feed)),
  showing: (id) => ipcRenderer.send('tiktok:showing', id),
  swipe: () => ipcRenderer.send('tiktok:swipe'),
  like: () => ipcRenderer.send('tiktok:like'),
  follow: () => ipcRenderer.send('tiktok:follow'),
  repost: () => ipcRenderer.send('tiktok:repost'),
  login: () => ipcRenderer.send('tiktok:login'),
})
