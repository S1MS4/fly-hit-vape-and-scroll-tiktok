// Runs inside the hidden tiktok.com window. Reports which video is playing (its id
// from TikTok's player wrapper) with its live details (author, caption, sound, counts,
// like/follow state), and hands over the first feed items embedded in the page. The
// video itself no longer comes from here: electron/tiktok.cjs downloads each item's
// mp4 ahead of time from TikTok's feed API and the phone plays the copy.
const { ipcRenderer } = require('electron')

function visibleArea(video) {
  const r = video.getBoundingClientRect()
  const w = Math.min(r.right, innerWidth) - Math.max(r.left, 0)
  const h = Math.min(r.bottom, innerHeight) - Math.max(r.top, 0)
  return w > 0 && h > 0 ? w * h : 0
}

/** The video that takes up the most of the viewport. */
function activeVideo() {
  let best = null
  let area = 0
  for (const video of document.querySelectorAll('video')) {
    const a = visibleArea(video)
    if (a > area) {
      area = a
      best = video
    }
  }
  return area > 40000 ? best : null
}

/** TikTok wraps each player in <div id="xgwrapper-0-<item id>">. */
function itemId(video) {
  const wrapper = video.closest('[id^="xgwrapper-"]')
  const id = wrapper ? wrapper.id.split('-').pop() : ''
  return /^\d{6,}$/.test(id) ? id : ''
}

/** "name reposted" on a feed item, as TikTok labels videos reposted by accounts you follow. */
function repostedBy(box) {
  for (const el of box.querySelectorAll('span, p, div')) {
    if (el.children.length) continue
    const match = /^(.+?)\s+reposted$/i.exec((el.textContent || '').trim())
    if (match) return match[1]
  }
  return ''
}

/** Author, caption, sound and counts from the feed item around `video`. */
function details(video) {
  let box = video
  while (box && box !== document.body && !box.querySelector('[data-e2e="like-icon"]')) box = box.parentElement
  if (!box || box === document.body) return null
  const find = (selector) => box.querySelector(selector)
  const text = (selector) => (find(selector)?.innerText || '').trim()
  const author = find('a[href^="/@"]')
  const music = find('a[href^="/music/"]')
  const avatar = find('[data-e2e="video-author-avatar"] img')?.src || ''
  return {
    id: itemId(video),
    author: author ? decodeURIComponent(author.getAttribute('href').slice(2).split(/[/?]/)[0]) : '',
    avatarUrl: avatar.startsWith('http') ? avatar : '', // a data: pixel while the picture lazy-loads
    desc: text('[data-e2e="video-desc"]'),
    music: music ? music.innerText.trim() : '',
    likes: text('[data-e2e="like-count"]'),
    comments: text('[data-e2e="comment-count"]'),
    saves: text('[data-e2e="favorite-count"]'),
    shares: text('[data-e2e="share-count"]'),
    liked: find('[data-e2e="like-icon"]')?.getAttribute('aria-pressed') === 'true',
    following: !find('[data-e2e="feed-follow"]'), // TikTok drops the + button once you follow
    repostedBy: repostedBy(box),
    live: video.duration === Infinity,
  }
}

// ---- the first items, embedded in the page --------------------------------------------
// The For You page arrives with its first couple of items in a JSON script; the rest
// come from the feed API, which tiktok.cjs reads off the network.
function sendEmbeddedItems() {
  const script = document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__')
  if (!script) return
  try {
    const scope = JSON.parse(script.textContent).__DEFAULT_SCOPE__ || {}
    const items = scope['webapp.updated-items']
    if (Array.isArray(items) && items.length) ipcRenderer.send('tiktok:embedded-items', items)
  } catch {
    // no embedded items: the feed API fills the order in
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', sendEmbeddedItems)
else sendEmbeddedItems()

// ---- details of the playing video -----------------------------------------------------
let lastMeta = ''
setInterval(() => {
  const video = activeVideo()
  const meta = video && details(video)
  if (!meta) return
  const key = JSON.stringify(meta)
  if (key === lastMeta) return
  lastMeta = key
  ipcRenderer.send('tiktok:video-meta', meta)
}, 250)
