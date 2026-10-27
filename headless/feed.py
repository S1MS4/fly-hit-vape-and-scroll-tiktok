"""A real TikTok For You feed in Microsoft Edge, driven for the headless fly.

Playwright's bundled Chromium can't play TikTok's H.264, so this drives the
installed Edge (channel="msedge") on a persistent profile. Frames come from
drawing the active <video> into a 36x64 canvas inside the page (its src is a
same-origin blob, so the canvas isn't tainted). Photo posts, or a tainted canvas,
fall back to a JPEG screenshot of the media area.

Nothing here logs in, signs up or solves captchas: log the fly in yourself with
`python headless/run.py --login`, or provide exported session cookies.
"""
import base64
import io
import json
import os
import time
import urllib.parse

import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

FEED_URL = "https://www.tiktok.com/foryou"
LOGIN_URL = "https://www.tiktok.com/login"
LIKE_API = "/api/commit/item/digg/"
EYE_W, EYE_H = 36, 64          # what the desktop app sends the brain: 36 wide, 64 tall
VIEWPORT = {"width": 1280, "height": 900}


class FeedError(RuntimeError):
    pass


# The active post: the most visible <video> (or, without one, the most visible feed
# item), scoped to its nearest ancestor holding a like button. Cached on window.__fly.
ACTIVE_JS = r"""() => {
  const S = window.__fly || (window.__fly = {})
  const area = r => Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0)) *
                    Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0))
  const ITEM = '[data-e2e="recommend-list-item-container"], article'
  let video = null, best = 0
  for (const v of document.querySelectorAll('video')) {
    const a = area(v.getBoundingClientRect())
    if (a > best) { best = a; video = v }
  }
  if (best < 40000) video = null
  let scope = null
  if (video) {
    for (let el = video.parentElement; el && el !== document.body; el = el.parentElement) {
      if (el.querySelector('[data-e2e="like-icon"]') || el.matches(ITEM)) { scope = el; break }
    }
  }
  if (!scope) {
    let bestItem = 0
    for (const it of document.querySelectorAll(ITEM)) {
      const a = area(it.getBoundingClientRect())
      if (a > bestItem) { bestItem = a; scope = it }
    }
    if (scope && !video) video = scope.querySelector('video')
  }
  S.video = video; S.scope = scope
  if (!scope) return null

  const q = s => scope.querySelector(s)
  const txt = e => e ? (e.innerText || e.textContent || '').replace(/\s+/g, ' ').trim() : null
  const num = e => { const s = txt(e); return s && /\d/.test(s) ? s : null }   // "Like" until the count loads
  const href = e => e ? (e.getAttribute('href') || '') : ''
  // Which feed slot is showing. Metadata fills in over a few seconds, so it can't be the key.
  const itemEl = scope.closest(ITEM) || scope
  if (!itemEl.dataset.flyPost) itemEl.dataset.flyPost = String(S.seq = (S.seq || 0) + 1)
  const post = itemEl.id || ('post-' + itemEl.dataset.flyPost)
  const player = q('[id^="xgwrapper-"]')
  let id = player ? ((player.id.match(/(\d{6,})$/) || [])[1] || null) : null
  const link = q('a[href*="/video/"], a[href*="/photo/"]')
  if (!id && link) id = (href(link).match(/\/(?:video|photo)\/(\d+)/) || [])[1] || null

  const avatar = q('[data-e2e="video-author-avatar"]')
  let authorA = avatar ? avatar.closest('a[href^="/@"]') || avatar.querySelector('a[href^="/@"]') : null
  if (!authorA) authorA = [...scope.querySelectorAll('a[href^="/@"]')].find(a => !a.closest('[data-e2e="video-desc"]')) || null
  const author = authorA ? decodeURIComponent(href(authorA).slice(2).split(/[/?#]/)[0]) : null
  const nick = txt(q('[data-e2e="video-author-nickname"]')) || null

  const descEl = q('[data-e2e="video-desc"]')
  const tags = [...scope.querySelectorAll('a[href^="/tag/"]')]
    .map(a => decodeURIComponent(href(a).slice(5).split(/[?#]/)[0]).toLowerCase())
  const mentions = descEl ? [...descEl.querySelectorAll('a[href^="/@"]')]
    .map(a => decodeURIComponent(href(a).slice(2).split(/[/?#]/)[0])) : []

  const musicA = q('a[href^="/music/"]')
  let sound = null, soundId = null
  if (musicA) {
    sound = txt(musicA) || (musicA.getAttribute('aria-label') || '').replace(/^Watch more videos with music\s*/i, '') || null
    const slug = decodeURIComponent(href(musicA).slice(7).split(/[?#]/)[0])
    const m = slug.match(/^(.*)-(\d{6,})$/)
    if (m) { soundId = m[2]; if (!sound) sound = m[1].replace(/-/g, ' ') }
    else if (!sound) sound = slug.replace(/-/g, ' ')
  }

  const likeIcon = q('[data-e2e="like-icon"]')
  const pressedEl = likeIcon && (likeIcon.hasAttribute('aria-pressed') ? likeIcon
                    : likeIcon.closest('[aria-pressed]') || likeIcon.querySelector('[aria-pressed]'))
  const text = scope.innerText || ''
  const kind = video ? (video.duration === Infinity ? 'live' : 'video')
             : (/\bLIVE\b/.test(text) && q('a[href*="/live"]') ? 'live' : (q('img') ? 'photo' : 'other'))
  const media = video || q('[data-e2e="feed-video"]') || q('section') || scope
  const r = media.getBoundingClientRect()
  const x = Math.max(0, r.left), y = Math.max(0, r.top)
  return {
    key: post, video_id: id, author, author_name: nick,
    url: author && id ? `https://www.tiktok.com/@${author}/${kind === 'photo' ? 'photo' : 'video'}/${id}` : null,
    desc: txt(descEl), hashtags: [...new Set(tags)], mentions,
    sound, sound_id: soundId, kind,
    sponsored: /(^|\n)\s*(Sponsored|Promoted|Ad)\s*(\n|$)/.test(text),
    duration_s: video && isFinite(video.duration) ? video.duration : null,
    ready: video ? video.readyState : null,
    liked: pressedEl ? pressedEl.getAttribute('aria-pressed') === 'true' : null,
    like_count: num(q('[data-e2e="like-count"]')), comment_count: num(q('[data-e2e="comment-count"]')),
    save_count: num(q('[data-e2e="favorite-count"]')), share_count: num(q('[data-e2e="share-count"]')),
    rect: {x, y, width: Math.min(r.right, innerWidth) - x, height: Math.min(r.bottom, innerHeight) - y},
    login_button: !!document.querySelector('[data-e2e="top-login-button"]'),
  }
}"""

# One brain tick's view of the phone: set the playback rate the servo asked for,
# then draw the current frame, letterboxed like a phone would show it, as luminance.
GRAB_JS = r"""({w, h, rate}) => {
  const S = window.__fly
  const v = S && S.video
  if (!v) return {novideo: true}
  if (!v.isConnected) return {gone: true}
  if (rate <= 0) { if (!v.paused) v.pause() }
  else {
    const r = Math.min(16, Math.max(0.0625, rate))
    if (Math.abs(v.playbackRate - r) > 0.01) v.playbackRate = r
    if (v.paused) v.play().catch(() => {})
  }
  const out = {t: v.currentTime, dur: isFinite(v.duration) ? v.duration : null, paused: v.paused}
  if (v.readyState < 2 || !v.videoWidth) { out.wait = true; return out }
  if (!S.cv) {
    S.cv = document.createElement('canvas'); S.cv.width = w; S.cv.height = h
    S.cx = S.cv.getContext('2d', {willReadFrequently: true})
  }
  const g = S.cx
  g.fillStyle = '#000'; g.fillRect(0, 0, w, h)
  const s = Math.min(w / v.videoWidth, h / v.videoHeight), dw = v.videoWidth * s, dh = v.videoHeight * s
  try {
    g.drawImage(v, (w - dw) / 2, (h - dh) / 2, dw, dh)
    const px = g.getImageData(0, 0, w, h).data
    let str = ''
    for (let i = 0; i < w * h; i++) str += String.fromCharCode((0.2126 * px[4*i] + 0.7152 * px[4*i+1] + 0.0722 * px[4*i+2]) | 0)
    out.b64 = btoa(str)
  } catch (e) { out.tainted = String(e).slice(0, 80) }
  return out
}"""

LIKE_STATE_JS = r"""() => {
  const scope = window.__fly && window.__fly.scope
  const icon = scope && scope.querySelector('[data-e2e="like-icon"]')
  if (!icon || !icon.isConnected) return null
  const p = icon.hasAttribute('aria-pressed') ? icon : icon.closest('[aria-pressed]') || icon.querySelector('[aria-pressed]')
  const c = scope.querySelector('[data-e2e="like-count"]')
  const r = icon.getBoundingClientRect()
  return {pressed: p ? p.getAttribute('aria-pressed') === 'true' : null, count: c ? c.innerText.trim() : null,
          x: r.left + r.width / 2, y: r.top + r.height / 2, visible: r.width > 0}
}"""

COOKIE_BANNER_JS = r"""() => {
  const b = document.querySelector('tiktok-cookie-banner')
  if (!b || !b.shadowRoot) return 'none'
  for (const x of b.shadowRoot.querySelectorAll('button'))
    if (/decline|reject/i.test(x.innerText)) { x.click(); return 'declined' }
  return 'pending'
}"""

# The centre of one close (X) button on a visible blocking pop-up, or null. Only close
# icons: never "Log in", "Sign up", "Continue" or the interest choices. The point is
# hit-tested so a click can't land on the video underneath.
MODAL_CLOSER_JS = r"""() => {
  const vis = e => { const r = e.getBoundingClientRect(); const cs = getComputedStyle(e)
    return r.width > 120 && r.height > 120 && r.bottom > 0 && r.top < innerHeight && cs.visibility !== 'hidden' && cs.display !== 'none' }
  // Blocking modals only: a bare role="dialog" also matches the player's captions/speed menu.
  const boxes = [...document.querySelectorAll('[aria-modal="true"], [data-e2e="login-modal"], [class*="DivModalContainer"], [class*="tux-modal__container"]')]
    .filter(vis).filter(b => !b.closest('.xgplayer, [class*="xgplayer"], [class*="Player"]'))
  for (const box of boxes) {
    const br = box.getBoundingClientRect()
    let target = box.querySelector('[data-e2e="modal-close-inner-button"], [aria-label="Close"], [aria-label="close"]')
    if (target && target.getBoundingClientRect().width === 0) target = null
    if (!target) {
      for (const el of box.querySelectorAll('svg, button, [role="button"]')) {
        const r = el.getBoundingClientRect()
        if (r.width === 0 || r.width > 48 || r.height > 48) continue
        if (/log ?in|sign ?up|continue/i.test(el.innerText || '')) continue
        if (r.left + r.width / 2 > br.right - 80 && r.top + r.height / 2 < br.top + 80) { target = el; break }
      }
    }
    if (!target) continue
    const r = target.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2
    const hit = document.elementFromPoint(x, y)
    const small = hit && hit.getBoundingClientRect().width <= 60
    if (!hit || !(hit === target || target.contains(hit) || (small && hit.contains(target)))) continue
    return {x, y, text: (box.innerText || '').slice(0, 60)}
  }
  return null
}"""


def proxy_from_env():
    """FLY_TIKTOK_PROXY=http://host:port or socks5://host:port (optionally user:pass@)."""
    raw = os.environ.get("FLY_TIKTOK_PROXY", "").strip()
    if not raw:
        return None
    u = urllib.parse.urlsplit(raw if "://" in raw else "http://" + raw)
    proxy = {"server": f"{u.scheme}://{u.hostname}" + (f":{u.port}" if u.port else "")}
    if u.username:
        proxy["username"] = urllib.parse.unquote(u.username)
    if u.password:
        proxy["password"] = urllib.parse.unquote(u.password)
    return proxy


def lum_from_b64(b64):
    return np.frombuffer(base64.b64decode(b64), np.uint8).reshape(EYE_H, EYE_W).astype(np.float32) / 255.0


def lum_from_image(img):
    """Letterbox an RGB image into EYE_W x EYE_H and return Rec. 709 luminance in [0, 1]."""
    s = min(EYE_W / img.width, EYE_H / img.height)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (EYE_W, EYE_H))
    canvas.paste(img.convert("RGB").resize((w, h), Image.BILINEAR), ((EYE_W - w) // 2, (EYE_H - h) // 2))
    rgb = np.asarray(canvas, np.float32)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


class TikTokFeed:
    def __init__(self, profile_dir, window="headless", cookies_path=None, proxy=None):
        self.profile_dir = profile_dir
        self.window = window
        self.cookies_path = cookies_path
        self.proxy = proxy
        self._pw = self.ctx = self.page = None
        self.meta = None
        self.cookie_note = ""
        self._banner_checks = 0

    # --- lifecycle -------------------------------------------------------------------

    def launch(self, headless=None):
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        args = ["--mute-audio"]
        if headless is None:
            headless = self.window == "headless"
        if self.window == "offscreen":
            args.append("--window-position=-32000,-32000")
        self._pw = sync_playwright().start()
        try:
            self.ctx = self._pw.chromium.launch_persistent_context(
                str(self.profile_dir), channel="msedge", headless=headless, viewport=VIEWPORT,
                locale="en-US", extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                proxy=self.proxy, args=args)
        except Exception:
            self._pw.stop()
            raise
        self.page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
        return self.page

    def open(self, timeout=45.0):
        self.launch()
        self.cookie_note = self._import_cookies()
        self.page.goto(FEED_URL, wait_until="domcontentloaded", timeout=60000)
        return self.wait_active(timeout)

    def close(self):
        for fn in (lambda: self.ctx and self.ctx.close(), lambda: self._pw and self._pw.stop()):
            try:
                fn()
            except Exception:
                pass
        self.ctx = self._pw = self.page = None

    def logged_in(self):
        try:
            return any(c["name"] == "sessionid" and c.get("value")
                       for c in self.ctx.cookies("https://www.tiktok.com"))
        except PlaywrightError:
            return False

    def _import_cookies(self):
        """Add exported session cookies, but only if the profile has no session of its own."""
        if self.logged_in():
            return "profile already has a TikTok session"
        if not self.cookies_path or not self.cookies_path.exists():
            return "no session cookies found; running logged out"
        allowed = {"name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite"}
        cookies = []
        for c in json.loads(self.cookies_path.read_text(encoding="utf-8")):
            c = {k: v for k, v in c.items() if k in allowed}
            if c.get("sameSite") not in ("Strict", "Lax", "None"):
                c.pop("sameSite", None)
            if c.get("sameSite") == "None" and not c.get("secure"):
                c["sameSite"] = "Lax"
            cookies.append(c)
        added = 0
        for c in cookies:  # one at a time so a single odd cookie can't sink the rest
            try:
                self.ctx.add_cookies([c])
                added += 1
            except PlaywrightError:
                pass
        return f"imported {added}/{len(cookies)} session cookies from {self.cookies_path.name}"

    # --- the feed ----------------------------------------------------------------------

    def active(self):
        self.meta = self.page.evaluate(ACTIVE_JS)
        return self.meta

    def wait_active(self, timeout=20.0, not_key=None, settle=2.5):
        """Wait for a post (different from not_key) to be active and, if a video, decodable.
        Holds out up to `settle` s for an id or author, so a loading placeholder isn't taken."""
        t0 = last_dismiss = time.monotonic()
        bare_since = None
        while True:
            meta = self.active()
            if meta and meta["key"] != not_key and (meta["kind"] != "video" or (meta["ready"] or 0) >= 2):
                if meta["video_id"] or meta["author"]:
                    return meta
                bare_since = bare_since or time.monotonic()
                if time.monotonic() - bare_since > settle:
                    return meta
            now = time.monotonic()
            if now - last_dismiss > 1.0:
                self.dismiss()
                last_dismiss = now
            if now - t0 > timeout:
                return None
            time.sleep(0.15)

    def dismiss(self):
        """Decline the cookie banner and close blocking pop-ups by their X. Returns what it closed."""
        closed = []
        try:
            # The banner can be injected seconds after load, so keep looking for a while.
            if self._banner_checks < 30:
                self._banner_checks += 1
                if self.page.evaluate(COOKIE_BANNER_JS) == "declined":
                    closed.append("cookie banner")
                    self._banner_checks = 30
            for _ in range(3):
                target = self.page.evaluate(MODAL_CLOSER_JS)
                if not target:
                    break
                self.page.mouse.click(target["x"], target["y"])
                closed.append("pop-up: " + target["text"].replace("\n", " ")[:40])
                time.sleep(0.4)
        except PlaywrightError:
            pass
        return closed

    def recover(self):
        """After a page error: return to the For You feed if we left it, and find the active post."""
        try:
            url = self.page.url
            if "/foryou" not in url and url.rstrip("/") != "https://www.tiktok.com":
                self.page.goto(FEED_URL, wait_until="domcontentloaded", timeout=60000)
        except PlaywrightError:
            pass
        return self.wait_active(20.0)

    def grab(self, rate):
        """Return (lum or None, info). Sets the video's playback rate first (0 pauses)."""
        info = self.page.evaluate(GRAB_JS, {"w": EYE_W, "h": EYE_H, "rate": float(rate)})
        if info.get("b64"):
            return lum_from_b64(info["b64"]), info
        if info.get("novideo") or info.get("tainted"):
            return self.screenshot_lum(), info
        return None, info

    def screenshot_lum(self):
        r = (self.meta or {}).get("rect")
        if not r or r["width"] < 10 or r["height"] < 10:
            return None
        jpg = self.page.screenshot(type="jpeg", quality=50, clip=r)
        return lum_from_image(Image.open(io.BytesIO(jpg)))

    def _blur(self):
        self.page.evaluate("() => { const a = document.activeElement; if (a && a !== document.body) a.blur() }")

    def swipe(self, tries=3, timeout=3.0):
        """Next video via ArrowDown, verified by the active post changing. Returns (meta, presses)."""
        old = self.meta["key"] if self.meta else None
        for attempt in range(1, tries + 1):
            self._blur()
            nxt = self.page.query_selector('[data-e2e="feed-navigation-next"]') if attempt == tries else None
            if nxt and nxt.is_visible():
                nxt.click()
            else:
                self.page.keyboard.press("ArrowDown")
            meta = self.wait_active(timeout, not_key=old)
            if meta:
                return meta, attempt
            self.dismiss()
        # Last resort: reload the feed.
        self.page.goto(FEED_URL, wait_until="domcontentloaded", timeout=60000)
        meta = self.wait_active(30.0, not_key=old)
        if not meta:
            raise FeedError("the feed stopped advancing (a pop-up, a captcha or a block?)")
        return meta, tries + 1

    def like(self):
        """Like the active post with TikTok's L shortcut. Never unlikes.

        The heart turns red at once whatever happens next, so it proves nothing. The
        like is confirmed only by TikTok's reply to its like API (JSON with status_code
        0 and is_digg 1). Seen with this automated Edge, headless or in a window: HTTP
        200 with an empty body, and the heart goes grey again ~3 s later."""
        before = self.page.evaluate(LIKE_STATE_JS)
        if not before:
            return {"pressed": False, "confirmed": False, "note": "no like button"}
        if before["pressed"]:
            return {"pressed": False, "confirmed": True, "note": "already liked"}
        out = {"pressed": True, "method": "key", "count_before": before["count"]}
        self._blur()
        api = self._like_request(lambda: self.page.keyboard.press("l"))
        if api is None:
            state = self.page.evaluate(LIKE_STATE_JS)
            if state and state["pressed"] is False and before.get("visible"):
                out["method"] = "click"  # the shortcut sent nothing: click the heart once
                api = self._like_request(lambda: self.page.mouse.click(before["x"], before["y"]))
        out["api"] = api
        out["confirmed"] = bool(api and api.get("status_code") == 0 and api.get("is_digg") == 1)
        if api is None:
            out["note"] = "no like request was sent"
        elif not out["confirmed"]:
            out["note"] = "TikTok didn't accept it" + (" (empty reply)" if api.get("body_len") == 0 else "")
        return out

    def _like_request(self, action, timeout_ms=5000):
        """Run action and return the like API's answer (status fields only), or None if none came."""
        try:
            with self.page.expect_response(lambda r: LIKE_API in r.url, timeout=timeout_ms) as info:
                action()
            resp = info.value
        except PlaywrightTimeoutError:
            return None
        api = {"http": resp.status}
        try:
            body = resp.text()
            api["body_len"] = len(body)
            data = json.loads(body) if body else None
            if isinstance(data, dict):
                api.update({k: data[k] for k in ("status_code", "status_msg", "is_digg") if k in data})
        except Exception:
            pass
        return api
