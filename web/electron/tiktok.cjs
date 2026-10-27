// A live tiktok.com feed for the fly. The feed runs in a hidden offscreen window, so
// TikTok sees real watching and swiping. What the phone plays is a copy: this reads
// TikTok's feed API off the window's network traffic, downloads the next few videos
// before the fly gets to them, and sends each one to the phone with its author,
// caption, sound, counts and avatar. The fly's swipes and likes become TikTok's
// keyboard shortcuts (↓ and L); follows and reposts click TikTok's own buttons inside
// the page. The account lives in a persistent session: log in once and it stays.
const path = require('node:path')
const { BrowserWindow, ipcMain, session } = require('electron')

const PARTITION = 'persist:fly-tiktok'
const FEED_URL = 'https://www.tiktok.com/foryou'
const LOGIN_URL = 'https://www.tiktok.com/login'
const ITEM_LIST = /\/api\/recommend\/item_list\//
// How many videos are downloaded before the fly reaches them, and how many are kept
// after it moves on (the one it slides away from).
const AHEAD = Math.max(1, Number(process.env.FLY_TIKTOK_PRELOAD) || 4)
const BEHIND = 1
const DOWNLOADS_AT_ONCE = 3
// The order, the item records and the failure counts are otherwise only emptied on a page
// load, which a healthy multi-hour session never does. Keep a window around where the fly
// is instead: far wider than the downloads reach, so nothing in use is ever dropped, and
// wide enough that the order's membership scans stay short.
const KEEP_BEHIND = 50
const KEEP_AHEAD = 150
// Without a timeout a hung fetch keeps its id in `loading` for the rest of the session and
// that download slot never comes back. Videos get the longer one; avatars and photo posts'
// pictures are small, and are awaited alongside the video, so a slow one must not hold the
// whole item up.
// Long enough for a big video on a slow link, short enough to be worth having. The fly
// crosses the whole preload window in about ten seconds, so a download that takes longer
// than this is evicted before it is ever watched: waiting longer buys nothing and holds a
// slot the videos just ahead of the fly need.
const VIDEO_TIMEOUT = 20000
// Strikes before a video is written off. A refusal is the CDN saying no, so two is plenty.
// A timeout only says "slow", which is worth more goes before giving up on it.
const HARD_TRIES = 2
const SLOW_TRIES = 4
// The whole verified swipe, however slow the page is to answer. Without this the budget
// counted only the sleeps and not the page queries, so a page that answered slowly could
// hold a swipe open for the better part of a minute, dropping every swipe behind it.
const SWIPE_BUDGET_MS = 6000
const IMAGE_TIMEOUT = 10000
// How long the tap waits before climbing back on after a detach, doubling up to a cap.
const TAP_RETRY_MIN = 500
const TAP_RETRY_MAX = 30000

// Finds the first visible match of `selector` in the feed item that fills most of the
// page (or in the whole page, for the share panel), optionally clicks it, and returns its
// text. Clicks run inside the page: mouse events sent to the hidden window don't reach
// TikTok's buttons.
const locate = (selector, inActiveItem, click = false) => `(() => {
  let root = document
  if (${inActiveItem}) {
    let best = null, area = 0
    for (const v of document.querySelectorAll('video')) {
      const r = v.getBoundingClientRect()
      const w = Math.min(r.right, innerWidth) - Math.max(r.left, 0)
      const h = Math.min(r.bottom, innerHeight) - Math.max(r.top, 0)
      if (w > 0 && h > 0 && w * h > area) { area = w * h; best = v }
    }
    root = best
    while (root && root !== document.body && !root.querySelector('[data-e2e="like-icon"]')) root = root.parentElement
    if (!root || root === document.body) return null
  }
  const el = root.querySelector(${JSON.stringify(selector)})
  if (!el) return null
  const r = el.getBoundingClientRect()
  if (r.width === 0 || r.height === 0) return null
  const text = el.innerText || ''
  if (${click}) (el.closest('button') || el).click()
  return { text }
})()`

// The id of the feed item filling most of the hidden window, photo posts included (they
// have no <video>, so the player wrapper the rest of the code keys on isn't there).
const ACTIVE_ITEM = `(() => {
  const area = (r) => Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0)) *
                      Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0))
  let best = null, most = 40000
  for (const item of document.querySelectorAll('[data-e2e="recommend-list-item-container"], article')) {
    const a = area(item.getBoundingClientRect())
    if (a > most) { most = a; best = item }
  }
  if (!best) return null
  const player = best.querySelector('[id^="xgwrapper-"]')
  if (player) return player.id.split('-').pop()
  const link = best.querySelector('a[href*="/video/"], a[href*="/photo/"]')
  const match = link && link.getAttribute('href').match(/\\/(?:video|photo)\\/(\\d+)/)
  return match ? match[1] : null
})()`

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/** Counts the way TikTok shows them: 1234, 12.3K, 1.2M. */
function formatCount(n) {
  n = Number(n) || 0
  if (n < 10000) return String(n)
  const [value, unit] = n < 1e6 ? [n / 1e3, 'K'] : [n / 1e6, 'M']
  return value.toFixed(1).replace(/\.0$/, '') + unit
}

/** The phone's view of a feed item, from TikTok's API record. */
function slimItem(item) {
  const video = item.video || {}
  const author = item.author || {}
  const music = item.music || {}
  const stats = item.statsV2 || item.stats || {}
  // The sharpest copy the phone needs, H.264 first: TikTok also serves H.265, which the
  // app window can't always decode.
  const copies = (video.bitrateInfo || [])
    .filter((b) => b.PlayAddr && b.PlayAddr.UrlList && b.PlayAddr.UrlList.length)
    .sort((a, b) => (b.Bitrate || 0) - (a.Bitrate || 0))
  const h264 = copies.filter((b) => b.CodecType === 'h264')
  const pool = h264.length ? h264 : copies
  const copy = pool.find((b) => (b.PlayAddr.Height || 0) <= 1280) || pool[0]
  const title = music.title || ''
  const sound = title && music.authorName && !title.endsWith(music.authorName) ? `${title} - ${music.authorName}` : title
  return {
    id: item.id,
    meta: {
      author: author.uniqueId || '',
      desc: item.desc || '',
      music: sound,
      likes: formatCount(stats.diggCount),
      comments: formatCount(stats.commentCount),
      saves: formatCount(stats.collectCount),
      shares: formatCount(stats.shareCount),
      liked: Boolean(item.digged),
      avatarUrl: author.avatarThumb || '',
      following: false, // the page says once the video is playing
      repostedBy: '',
    },
    videoUrl: copy ? copy.PlayAddr.UrlList[0] : video.playAddr || '',
    imageUrls: ((item.imagePost && item.imagePost.images) || [])
      .map((image) => image.imageURL && image.imageURL.urlList && image.imageURL.urlList[0])
      .filter(Boolean),
    duration: video.duration || 0,
  }
}

async function startTikTok(appWindow) {
  const ses = session.fromPartition(PARTITION)
  // Present as the Chromium it is, without the "Electron" and app-name tokens, and
  // ask for English so the feed isn't pulled toward the local language.
  ses.setUserAgent(ses.getUserAgent().replace(/\s(Electron|web)\/\S+/g, ''), 'en-US,en')
  // TikTok picks For You mostly by the country it sees. FLY_TIKTOK_PROXY (for example
  // socks5://127.0.0.1:1080 from a VPN client) sends only the fly's TikTok traffic through it.
  const proxy = process.env.FLY_TIKTOK_PROXY
  if (proxy) {
    await ses.setProxy({ proxyRules: proxy })
    console.log(`TikTok traffic goes through ${proxy}`)
  }

  const feed = new BrowserWindow({
    show: false,
    width: 1280,
    height: 900,
    webPreferences: {
      offscreen: true, // keeps the page "visible", so videos keep playing
      partition: PARTITION,
      backgroundThrottling: false,
      preload: path.join(__dirname, 'feed-preload.cjs'),
    },
  })
  feed.webContents.setAudioMuted(true) // the phone plays the sound
  feed.webContents.setFrameRate(30) // nobody looks at this window, but TikTok needs to keep scrolling and playing
  feed.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  const toApp = (channel, payload) => {
    if (!appWindow.isDestroyed()) appWindow.webContents.send(channel, payload)
  }
  const fromFeed = (event) => !feed.isDestroyed() && event.sender === feed.webContents
  // The channels the scene window drives. Checking the sender both ways keeps any other
  // renderer -- a login pop-up on tiktok.com, say -- from steering the fly's account.
  const fromApp = (event) => !appWindow.isDestroyed() && event.sender === appWindow.webContents

  // ---- what's coming up in the feed ----------------------------------------------------
  // TikTok plays the items embedded in the page first, then each batch it fetches from
  // /api/recommend/item_list/, in order. The batches are read off the window's network
  // traffic through the debugger, which the page can't see; the ids the page reports as
  // videos play confirm the order.
  const order = [] // item ids, in feed order
  const items = new Map() // id -> slim item
  const loaded = new Map() // id -> what the phone was sent (null after a failed download)
  const loading = new Set()
  const failures = new Map()
  const retryAt = new Map() // id -> when a failed download may be tried again
  let activeId = '' // what TikTok is playing
  let phoneId = '' // what the phone is showing, which is usually a step ahead
  let generation = 0 // bumps on every page load, so downloads for the old page are dropped

  // Downloads follow the phone, not TikTok. The phone slides to the next video the
  // moment the fly swipes, while the hidden window is still scrolling, so anchoring on
  // TikTok would leave the fly outrunning the downloads onto a video that isn't here yet.
  const anchor = () => {
    const phone = order.indexOf(phoneId)
    return phone >= 0 ? phone : order.indexOf(activeId)
  }
  const span = () => {
    const pos = anchor()
    return pos < 0 ? [0, AHEAD + 1] : [Math.max(0, pos - BEHIND), pos + AHEAD + 1]
  }

  /**
   * Forgets the far past and far future of the feed. Nothing here is reachable: the
   * downloads only ever cover a handful either side of the fly, so the window kept is
   * orders of magnitude wider than anything in use.
   */
  const prune = () => {
    const pos = Math.max(0, anchor()) // nothing playing yet: the front is what plays next
    const from = Math.max(0, pos - KEEP_BEHIND)
    const to = pos + KEEP_AHEAD
    if (from === 0 && to >= order.length) return
    // The tail first, so the front's removal can't shift the index it's taken at.
    const dropped = order.splice(to).concat(order.splice(0, from))
    for (const id of dropped) {
      items.delete(id)
      failures.delete(id)
      retryAt.delete(id)
    }
  }

  /** Downloads the playing video and the next few, drops the ones left behind, and tells the phone the order. */
  const preload = () => {
    prune()
    const [from, to] = span()
    const wanted = order.slice(from, to)
    for (const id of loaded.keys()) if (!wanted.includes(id)) loaded.delete(id)
    const now = Date.now()
    const ahead = order.slice(Math.max(from, anchor()), to)
    // Two passes. Anything never tried gets a slot before anything that has already failed,
    // or one broken item sitting nearest the fly holds a slot on every pass while the videos
    // behind it are never fetched at all.
    for (const retry of [false, true]) {
      for (const id of ahead) {
        if (loading.size >= DOWNLOADS_AT_ONCE) break
        if (loaded.has(id) || loading.has(id) || !items.has(id)) continue
        if ((retryAt.get(id) || 0) > now || failures.has(id) !== retry) continue
        download(id)
      }
    }
    toApp('tiktok:feed', { activeId, order: wanted })
  }

  const addItems = (list, first = false) => {
    const fresh = []
    for (const raw of list) {
      if (!raw || !raw.id) continue
      if (!items.has(raw.id)) items.set(raw.id, slimItem(raw))
      if (!order.includes(raw.id) && !fresh.includes(raw.id)) fresh.push(raw.id)
    }
    if (first) order.unshift(...fresh)
    else order.push(...fresh)
    if (fresh.length) preload()
  }

  // The video CDN answers 403 without the session's cookies and a tiktok.com referer.
  // The signal covers the body as well as the headers, so a CDN that accepts the
  // connection and then stalls mid-download gives up too instead of wedging the slot.
  const fetchBytes = async (url, timeout = VIDEO_TIMEOUT) => {
    const res = await ses.fetch(url, {
      credentials: 'include',
      headers: { Referer: 'https://www.tiktok.com/' },
      signal: AbortSignal.timeout(timeout),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return new Uint8Array(await res.arrayBuffer())
  }

  const download = async (id) => {
    const item = items.get(id)
    const gen = generation
    loading.add(id)
    const started = Date.now()
    try {
      const [video, avatar, images] = await Promise.all([
        item.videoUrl ? fetchBytes(item.videoUrl) : null,
        item.meta.avatarUrl ? fetchBytes(item.meta.avatarUrl, IMAGE_TIMEOUT).catch(() => null) : null,
        Promise.all(item.imageUrls.slice(0, 12).map((url) => fetchBytes(url, IMAGE_TIMEOUT).catch(() => null))),
      ])
      if (gen !== generation) return
      const payload = { id, meta: item.meta, duration: item.duration, video, avatar, images: images.filter(Boolean) }
      loaded.set(id, payload)
      toApp('tiktok:item', payload)
      failures.delete(id) // it came good, so earlier strikes shouldn't count against it later
      retryAt.delete(id)
      const size = video ? `${(video.byteLength / 1e6).toFixed(1)} MB` : `${payload.images.length} pictures`
      console.log(`TikTok video ${id} by @${item.meta.author}: ${size} in ${Date.now() - started} ms`)
    } catch (err) {
      if (gen !== generation) return
      // A stall before the reply times out; one partway through the body comes back from
      // the stream as an abort. Both mean slow rather than broken.
      const slow = err.name === 'TimeoutError' || err.name === 'AbortError'
      const strikes = (failures.get(id) || 0) + 1
      failures.set(id, strikes)
      // Back off before trying this one again. Without it a stalled download retries the
      // instant its slot frees, so the same broken item takes the slot over and over while
      // everything behind it waits.
      retryAt.set(id, Date.now() + Math.min(30000, 2000 * 2 ** (strikes - 1)))
      console.warn(`TikTok video ${id} not downloaded (${err.message})`)
      if (strikes >= (slow ? SLOW_TRIES : HARD_TRIES)) {
        // Give up: the phone shows the caption and counts over a black screen.
        loaded.set(id, null)
        toApp('tiktok:item', { id, meta: item.meta, duration: item.duration, video: null, avatar: null, images: [] })
      }
    } finally {
      // Guarded like the paths above. A download from the old page settling after a
      // navigation must not remove an id the new page has already started downloading
      // again: the navigation emptied `loading` for it, so the delete would be retiring
      // somebody else's entry and the next pass would start a second copy of the same
      // video. Two copies arriving at the phone rebuild the item, restarting playback
      // from zero if the fly is watching it.
      if (gen === generation) {
        loading.delete(id)
        preload()
      }
    }
  }

  const tap = feed.webContents.debugger
  const pendingBodies = new Map() // CDP request id -> the generation that queued it
  tap.on('message', async (_event, method, params) => {
    if (method === 'Network.responseReceived') {
      if (ITEM_LIST.test(params.response.url) && params.response.status === 200) pendingBodies.set(params.requestId, generation)
    } else if (method === 'Network.loadingFinished' && pendingBodies.has(params.requestId)) {
      const gen = pendingBodies.get(params.requestId)
      pendingBodies.delete(params.requestId)
      try {
        const { body, base64Encoded } = await tap.sendCommand('Network.getResponseBody', { requestId: params.requestId })
        // The page can navigate while the body is being read. Without this the old page's
        // batch lands in the order the navigation just emptied, and the phone preloads and
        // plays videos TikTok is nowhere near -- which the catch-up can't undo, because
        // those ids look like they are ahead.
        if (gen !== generation) return
        const json = JSON.parse(base64Encoded ? Buffer.from(body, 'base64').toString() : body)
        addItems(json.itemList || [])
      } catch (err) {
        console.warn('TikTok feed batch unreadable:', err.message)
      }
    } else if (method === 'Network.loadingFailed') {
      // A cancelled or failed feed request never reaches loadingFinished, so without this
      // its id stays in the map for the rest of the session.
      pendingBodies.delete(params.requestId)
    }
  })
  // The tap is the only source of feed data, and losing it is quiet: the page still reports
  // which video plays, so ids keep reaching the order and the phone keeps its captions and
  // counts over a black screen, for ever. Anything can take it away -- DevTools opening on
  // this window, or a client reaching the remote debugging port -- so climb back on, with a
  // backoff so a target that keeps refusing can't spin.
  let tapRetry = 0
  let tapTimer = null
  const scheduleTap = () => {
    if (tapTimer || feed.isDestroyed()) return
    const wait = Math.min(TAP_RETRY_MAX, TAP_RETRY_MIN * 2 ** tapRetry)
    tapRetry++
    tapTimer = setTimeout(() => {
      tapTimer = null
      attachTap()
    }, wait)
  }
  tap.on('detach', (_event, reason) => {
    console.error(`TikTok feed tap detached (${reason}): no new videos reach the phone until it is back`)
    scheduleTap()
  })
  const attachTap = () => {
    if (feed.isDestroyed()) return
    try {
      if (!tap.isAttached()) tap.attach('1.3')
    } catch (err) {
      console.error(`TikTok feed tap could not attach (${err.message}): the phone gets no videos until it does`)
      scheduleTap()
      return
    }
    // Answers once the page's renderer exists, so don't wait for it here: until then this
    // rejects harmlessly, which is why it used to be swallowed whole. But that made a
    // lasting failure look identical, with the same silent black-page ending and nothing in
    // the log. Retrying covers both, and only a tap that actually works clears the backoff.
    tap.sendCommand('Network.enable', { maxResourceBufferSize: 32 << 20, maxTotalBufferSize: 128 << 20 })
      .then(() => {
        tapRetry = 0
      })
      .catch((err) => {
        if (!tap.isAttached()) return // detached meanwhile: the detach handler owns the retry
        console.warn(`TikTok feed tap: Network.enable failed (${err.message}); retrying`)
        scheduleTap()
      })
  }
  attachTap()
  feed.loadURL(FEED_URL)

  feed.webContents.on('did-navigate', (_event, url) => {
    console.log('TikTok page loaded:', url)
    generation++
    order.length = 0
    items.clear()
    loaded.clear()
    loading.clear()
    failures.clear()
    retryAt.clear()
    pendingBodies.clear()
    activeId = ''
    phoneId = ''
    toApp('tiktok:feed', { activeId, order: [] })
  })
  feed.webContents.on('render-process-gone', (_event, details) => {
    console.error('TikTok page crashed:', details.reason)
    if (feed.isDestroyed()) return
    attachTap()
    feed.loadURL(FEED_URL)
  })
  // The app page reloaded (say, to pick up a new build): send it what's downloaded.
  appWindow.webContents.on('did-finish-load', () => {
    for (const payload of loaded.values()) if (payload) toApp('tiktok:item', payload)
    preload()
  })

  let lastMetaAt = 0
  let loggedIn = false
  const reportStatus = setInterval(async () => {
    try {
      const cookies = await ses.cookies.get({ url: 'https://www.tiktok.com', name: 'sessionid' })
      if (cookies.length > 0 && !loggedIn) ses.cookies.flushStore() // write a new login to disk now
      loggedIn = cookies.length > 0
      toApp('tiktok:status', { loggedIn, video: Date.now() - lastMetaAt < 5000 })
      // Last resort. Both the detach handler and the failed-attach path queue a retry, so
      // the tap being off with nothing pending means one of them was missed, and the feed
      // has gone quiet for good: the phone would show black pages under live captions
      // until the app is restarted. Cheap to check, and it can only ever revive a dead tap.
      if (!tap.isAttached() && !tapTimer && !feed.isDestroyed()) {
        console.error('TikTok feed tap is off with no retry pending; putting it back')
        attachTap()
      }
    } catch (err) {
      // An async interval has nobody to catch for it, and the cookie read rejects while the
      // session is being torn down: that was an unhandled rejection on the way out.
      console.warn('TikTok status unavailable:', err.message)
    }
  }, 2000)

  const press = (keyCode) => {
    if (feed.isDestroyed()) return
    feed.webContents.focus()
    feed.webContents.sendInputEvent({ type: 'keyDown', keyCode })
    if (keyCode.length === 1) feed.webContents.sendInputEvent({ type: 'char', keyCode })
    feed.webContents.sendInputEvent({ type: 'keyUp', keyCode })
  }
  // executeJavaScript can hang for good if TikTok swaps its page frame mid-call, so give up
  // after 5 s. Three outcomes, kept apart: the element's text when it is there, null when
  // the page answered and nothing matched, and undefined when the query itself failed or
  // timed out. Folding the last two together reported a follow that never ran as "already
  // following".
  const find = (selector, inActiveItem, click = false) => {
    if (feed.isDestroyed()) return Promise.resolve(undefined)
    const call = feed.webContents.executeJavaScript(locate(selector, inActiveItem, click), true).catch(() => undefined)
    return Promise.race([call, sleep(5000).then(() => undefined)])
  }

  // The phone slides to the next video the moment the fly swipes, so this only has to
  // get TikTok itself onto the same video: press ↓, check the page moved, press again if
  // it didn't. Without the check a dropped keypress leaves the fly's watching on the
  // wrong video. One swipe runs at a time; a second arriving meanwhile waits its turn
  // instead of being lost, which would leave the phone a step ahead of TikTok for good.
  // Guarded like press() and find(): reaching webContents on a destroyed window throws
  // where nobody catches, and a swipe can still be mid-poll when the app window closes.
  const probe = () => {
    if (feed.isDestroyed()) return Promise.resolve(null)
    return Promise.race([
      feed.webContents.executeJavaScript(ACTIVE_ITEM, true).catch(() => null),
      sleep(2000).then(() => null),
    ])
  }

  const swipeOnce = async () => {
    const before = await probe()
    // The order already knows which id comes next, so the check can tell "moved one" from
    // "moved three". Asking only whether anything changed passes a double scroll as
    // happily as a single one and logs it a success either way.
    const pos = before ? order.indexOf(before) : -1
    const expected = pos >= 0 ? order[pos + 1] || '' : ''
    // Bounded by the clock, not by how many times round the loop: every pass also waits on
    // a page query, which has its own fallback, so counting only the sleeps understated the
    // worst case by a factor of five. A swipe held open this long drops the ones behind it.
    const deadline = Date.now() + SWIPE_BUDGET_MS
    for (let attempt = 0; attempt < 3 && Date.now() < deadline; attempt++) {
      press('Down')
      // Poll instead of waiting a fixed stretch and pressing again. A second press sent
      // while TikTok is still scrolling is queued rather than ignored, and the feed jumps
      // two or three items for the fly's one swipe.
      // A fixed number of polls, not a wall-clock budget. A budget that counts the page
      // query gives the page the fewest chances exactly when it is slowest, and so re-presses
      // while the first key is still queued on a congested renderer. Both then land and the
      // feed jumps two videos for one swipe, which is in the account's watch history for
      // good. An unconfirmed single press heals itself; a double press cannot be undone.
      for (let poll = 0; poll < 6; poll++) {
        await sleep(250)
        const now = await probe()
        if (!before || !now) return // can't read the page: assume the press landed
        if (expected ? now === expected : now !== before) return
        if (Date.now() >= deadline) {
          console.warn('TikTok swipe: gave up waiting for the page to answer')
          return
        }
        if (now !== before) {
          // An overshoot, or something the feed API never listed (an ad, a live). Either
          // way the page has moved, so pressing again would only take it further.
          console.warn(`TikTok swipe: expected ${expected}, landed on ${now}`)
          return
        }
      }
    }
    console.warn('TikTok did not move on after three swipes')
  }

  // A swipe arriving while one is in flight cannot be thrown away. The renderer has already
  // counted it and started its slide, and its catch-up only ever moves forward, so a
  // swallowed swipe leaves the phone a step ahead of TikTok for the rest of the session --
  // and like, follow and repost act on the hidden window, landing on a video the fly was
  // never shown. Remember one and run it on the way out.
  let swiping = false
  let pendingSwipe = false
  const swipe = async () => {
    if (feed.isDestroyed()) return
    if (swiping) {
      if (pendingSwipe) console.warn('TikTok swipe dropped: one in flight and one already waiting')
      else console.log('TikTok swipe deferred: one already in flight')
      pendingSwipe = true
      return
    }
    swiping = true
    try {
      do {
        pendingSwipe = false // cleared before the run, so a swipe during it is remembered
        await swipeOnce()
      } while (pendingSwipe && !feed.isDestroyed())
    } finally {
      swiping = false
      pendingSwipe = false
    }
  }

  const follow = async () => {
    const button = await find('[data-e2e="feed-follow"]', true, true)
    if (button === undefined) return 'could not tell: the page did not answer'
    if (!button) return 'already following'
    await sleep(1500)
    const still = await find('[data-e2e="feed-follow"]', true)
    if (still === undefined) return 'clicked, but the page did not answer'
    return still ? 'follow button still there' : 'followed'
  }

  const repost = async () => {
    const share = await find('[data-e2e="share-icon"]', true, true)
    if (share === undefined) return 'could not open share: the page did not answer'
    if (!share) return 'no share button'
    let option = null
    for (let i = 0; i < 20 && !option; i++) {
      await sleep(100)
      option = await find('[data-e2e="share-repost"] [tabindex]', false)
    }
    let result = 'no repost option'
    if (option && /remove/i.test(option.text)) {
      result = 'already reposted'
    } else if (option) {
      await find('[data-e2e="share-repost"] [tabindex]', false, true)
      result = 'reposted'
      await sleep(800)
    }
    await find('button[aria-label="close"]', false, true)
    return result
  }

  let login = null
  const handlers = {
    'tiktok:embedded-items': (event, list) => {
      if (fromFeed(event)) addItems(list, true)
    },
    'tiktok:showing': (event, id) => {
      if (!fromApp(event)) return
      if (typeof id !== 'string' || !id || id === phoneId) return
      phoneId = id
      preload()
    },
    'tiktok:video-meta': (event, meta) => {
      if (!fromFeed(event)) return
      lastMetaAt = Date.now()
      if (meta.id && meta.id !== activeId) {
        // Something the feed API didn't list (an ad, a live) goes in after the playing one.
        // Splicing at indexOf(activeId) + 1 unguarded puts it at the very front whenever
        // that id isn't in the order, which is right only by accident on the first report
        // after a page load -- and stops being right as soon as the order is pruned.
        if (!order.includes(meta.id)) {
          const at = activeId ? order.indexOf(activeId) : -1
          if (at >= 0) order.splice(at + 1, 0, meta.id)
          else if (!activeId) order.unshift(meta.id) // nothing playing yet: this is the front
          else console.warn(`TikTok is playing ${meta.id}, which the feed order cannot place`)
        }
        activeId = meta.id
        preload()
      }
      toApp('tiktok:meta', meta)
    },
    'tiktok:swipe': (event) => {
      if (fromApp(event)) swipe()
    },
    // Logged out, these only open a login pop-up that then swallows the swipe keys.
    'tiktok:like': (event) => {
      if (fromApp(event) && loggedIn) press('l')
    },
    'tiktok:follow': async (event) => {
      if (fromApp(event) && loggedIn) console.log('TikTok follow:', await follow())
    },
    'tiktok:repost': async (event) => {
      if (fromApp(event) && loggedIn) console.log('TikTok repost:', await repost())
    },
    'tiktok:login': (event) => {
      if (!fromApp(event)) return
      if (login) return login.focus()
      login = new BrowserWindow({
        width: 480,
        height: 820,
        title: "Log the fly into TikTok (close this window when you're done)",
        autoHideMenuBar: true,
        webPreferences: { partition: PARTITION },
      })
      login.loadURL(LOGIN_URL)
      login.on('closed', () => {
        login = null
        ses.cookies.flushStore()
        if (!feed.isDestroyed()) feed.loadURL(FEED_URL) // pick up the new session
      })
    },
  }
  for (const [channel, handler] of Object.entries(handlers)) ipcMain.on(channel, handler)

  // The hidden window would otherwise keep the app alive after its window closes.
  appWindow.on('closed', () => {
    clearInterval(reportStatus)
    clearTimeout(tapTimer) // no reattach after the app is gone
    for (const [channel, handler] of Object.entries(handlers)) ipcMain.removeListener(channel, handler)
    ses.cookies.flushStore()
    if (login) login.destroy()
    if (!feed.isDestroyed()) feed.destroy()
  })
}

module.exports = { startTikTok }
