"""The connectome fly scrolls a real TikTok For You feed, headless.

Run from the project root:
  python headless/run.py --login              log the fly's account in, in a visible Edge window
  python headless/run.py                      watch until Ctrl+C, at natural speed
  python headless/run.py --videos 50 --speed 5
  python headless/run.py --dry                no browser: brain throughput on synthetic frames
  python headless/report.py                   summarize the latest run

Brain and video time run in lockstep: every 50 ms brain tick sees one frame, and the
video's playback rate is steered so video time advances `speed` x brain time. A
faster --speed compresses what the fly watches; it does not make TikTok requests any
faster than --min-swipe-gap and --min-like-gap allow in real time.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from paths import BUILD, PROFILE_DIR, ROOT, RUNS_DIR
from lif import Brain
from world import World
from fastbrain import FastBrain

TICK_S = 0.05
EYE_W, EYE_H = 36, 64
SESSION_COOKIES = ROOT / "data" / "tiktok-session" / "cookies.json"


def load_brain(eager=False):
    g = np.load(BUILD / "graph.npz")
    meta = json.loads((BUILD / "groups.json").read_text())
    r = np.load(BUILD / "retina.npz")
    cls = Brain if eager else FastBrain
    brain = cls(g["indptr"], g["indices"], g["counts"], g["is_ach"])

    def make_world():
        return World(meta["groups"], r["idx"], r["eye"], r["u"], r["v"])

    return brain, make_world


class Stepper:
    """One 50 ms tick, wired like brain/server.py's tick(). With FastBrain the GPU
    works on the tick between start() and finish(), so the caller can grab the next
    frame in between."""

    def __init__(self, brain, world):
        self.brain, self.world = brain, world
        self.steps = int(round(TICK_S * 1000 / brain.p.dt))
        self.fast = isinstance(brain, FastBrain)
        self.world_s = 0.0

    def start(self, lum):
        w = self.world
        w.set_frame(lum)
        idx, rates = w.sensory_inputs()
        self.brain.set_inputs(idx, rates)
        self.brain.ach_gain = w.ach_gain()
        if self.fast:
            self.brain.launch(self.steps)
        else:
            self._counts = self.brain.run(self.steps)

    def finish(self):
        if self.fast:
            self.brain.wait(self.steps)
            counts = self.brain.counts_numpy()
        else:
            counts = self._counts.cpu().numpy()
        t0 = time.perf_counter()
        state = self.world.update(counts, TICK_S)
        self.world_s += time.perf_counter() - t0
        return state


def settle(brain, make_world, lum, seconds):
    """Run the brain on the first frame with a throwaway World, then return a fresh World.

    A cold network answers its first input with a burst (PAM near 50 Hz for one tick
    before settling at 1-3 Hz). A World that watches from the first tick seeds its
    running baselines with that burst; its 30 s PAM average then sits far above PAM
    for minutes and no video can build enjoyment. seconds=0 keeps the cold start that
    brain/server.py uses."""
    if seconds > 0:
        st = Stepper(brain, make_world())
        for _ in range(int(round(seconds / TICK_S))):
            st.start(lum)
            st.finish()
    return make_world()


def new_video_for_world(world):
    """Every video starts fresh, as World does after the fly's own swipe."""
    world.enjoyment = 0.0
    for flag in ("liked_video", "followed_video", "reposted_video"):
        if hasattr(world, flag):
            setattr(world, flag, False)


# --- dry benchmark -------------------------------------------------------------------

_YY, _XX = np.mgrid[0:EYE_H, 0:EYE_W]


def synthetic_frame(t, seed):
    """A busy vertical video: a new shot every 1.5 s with a few moving blobs."""
    shot = int(t / 1.5)
    r = np.random.default_rng(seed * 7919 + shot)
    img = np.full((EYE_H, EYE_W), r.uniform(0.05, 0.6), np.float32)
    local = t - shot * 1.5
    for _ in range(3):
        cx = (r.uniform(0, EYE_W) + r.uniform(-15, 15) * local) % EYE_W
        cy = (r.uniform(0, EYE_H) + r.uniform(-15, 15) * local) % EYE_H
        img[(_XX - cx) ** 2 + (_YY - cy) ** 2 < r.uniform(3, 10) ** 2] = r.uniform(0, 1)
    return img


def dry(args):
    brain, make_world = load_brain(args.eager)
    world = settle(brain, make_world, synthetic_frame(0.0, 0), args.preroll)
    if args.like_threshold is not None:
        world.like_threshold = args.like_threshold
    st = Stepper(brain, world)
    sync = torch.cuda.synchronize if brain.dev.type == "cuda" else (lambda: None)
    name = torch.cuda.get_device_name(brain.dev) if brain.dev.type == "cuda" else "cpu"
    print(f"{type(brain).__name__} on {name}, speed {args.speed}, {args.dry_seconds:g} s of brain time")
    warm = 40  # graph captures and CUDA warm-up, not timed
    n = int(round(args.dry_seconds / TICK_S))
    seed, video_bio = 0, 0.0
    acts = {"swipe": 0, "like": 0, "puff": 0, "escape": 0, "cough": 0, "whitey": 0, "timeout": 0}
    for i in range(warm + n):
        if i == warm:
            sync()
            wall0, st.world_s = time.perf_counter(), 0.0
            acts = dict.fromkeys(acts, 0)
            if isinstance(brain, FastBrain):
                brain.stats.update(ticks=0, reruns=0, eager_ticks=0)
                edges = {}
        st.start(synthetic_frame(video_bio * args.speed, seed))
        state = st.finish()
        video_bio += TICK_S
        if i >= warm and isinstance(brain, FastBrain):
            edges[brain._E] = edges.get(brain._E, 0) + 1
        for a in state["actions"]:
            acts[a] = acts.get(a, 0) + (i >= warm)
        if "swipe" in state["actions"] or video_bio >= args.max_watch:
            if "swipe" not in state["actions"]:
                acts["timeout"] += i >= warm
                new_video_for_world(world)
            seed, video_bio = seed + 1, 0.0
    sync()
    wall = time.perf_counter() - wall0
    bio = n * TICK_S
    print(f"brain time {bio:.1f} s in {wall:.1f} s wall  ->  {bio / wall:.2f} bio s per wall s"
          f"  ({wall / n * 1000:.1f} ms per 50 ms tick, of which world.update {st.world_s / n * 1000:.2f} ms)")
    print(f"at --speed {args.speed:g}: {bio * args.speed / wall:.2f} video s per wall s")
    print("actions:", acts)
    if isinstance(brain, FastBrain):
        print("FastBrain:", {k: brain.stats[k] for k in ("reruns", "eager_ticks", "captures")},
              "edge capacity per tick:", dict(sorted(edges.items())))
        print(f"GPU memory peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


# --- login ---------------------------------------------------------------------------

def login(args):
    from feed import LOGIN_URL, TikTokFeed, proxy_from_env
    from playwright.sync_api import Error as PlaywrightError

    feed = TikTokFeed(Path(args.profile), window="headed", proxy=proxy_from_env())
    page = feed.launch(headless=False)
    page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
    print(f"Edge is open on the fly's profile ({args.profile}).\n"
          "Log in to TikTok there yourself, answer any first-run questions, then close the window.")
    announced = False
    try:
        while feed.ctx.pages:
            if not announced and feed.logged_in():
                print("Logged in; the session is saved in the profile. Close the window when you're done.")
                announced = True
            feed.ctx.pages[0].wait_for_timeout(1500)
    except PlaywrightError:
        pass
    finally:
        feed.close()
    print("Window closed." + ("" if announced else " No TikTok session was detected."))


# --- live ----------------------------------------------------------------------------

class VideoClock:
    """Steers the video's playback rate so video time stays at speed x brain time."""

    def __init__(self, speed):
        self.speed, self.bio, self.vt, self.last_t = speed, 0.0, 0.0, None

    def rate(self, tick_wall):
        ahead = self.vt - self.speed * self.bio
        if ahead > 0.25 * max(1.0, self.speed):
            return 0.0                                   # hold the frame until the brain catches up
        feedforward = self.speed * TICK_S / max(tick_wall, 1e-3)
        return min(16.0, max(0.0625, feedforward - ahead))  # and close any gap over about a second

    def advance(self, info):
        self.bio += TICK_S
        t = info.get("t") if info else None
        if t is None:
            return
        if self.last_t is not None:
            d = t - self.last_t
            if d < -0.25 and info.get("dur"):
                d += info["dur"]                           # the video looped
            if d > 0:
                self.vt += d
        self.last_t = t


META_KEYS = ("video_id", "url", "author", "author_name", "desc", "hashtags", "mentions", "sound", "sound_id",
             "kind", "sponsored", "duration_s", "like_count", "comment_count", "save_count", "share_count")


class VideoRecord:
    def __init__(self, index, meta, world):
        self.i, self.meta, self.key = index, meta, meta["key"]
        self.started = datetime.now().isoformat(timespec="seconds")
        self.t0 = time.monotonic()
        self.ticks = 0
        self.pam = self.photo = self.frame_mean = self.frame_change = self.enjoy_peak = 0.0
        self.prev = None
        self.puffs = self.escapes = self.missing_frames = 0
        # Nicotine side effects (brain/world.py): how many fits and head rushes started on
        # this video, and the brain seconds it spent in each. Without these a long run
        # leaves no trace of either, so there is no way to tell afterwards how often they
        # fired or whether their tuning held.
        self.coughs = self.whiteys = 0
        self.coughing_bio_s = self.whitey_bio_s = 0.0
        self.fly_liked, self.like_bio_s, self.like = False, None, None
        self.fly_followed = self.fly_reposted = False   # logged only; never pressed on TikTok
        self.swipe_bio_s, self.exit, self.swipe_presses = None, None, None
        self.state = None
        self.world = world
        self.logged = False

    @property
    def bio_s(self):
        return self.ticks * TICK_S

    def tick(self, state, lum):
        self.ticks += 1
        self.state = state
        self.pam += state["rates"]["pam"]
        self.photo += state["rates"]["photoreceptors"]
        self.enjoy_peak = max(self.enjoy_peak, self.world.enjoyment)
        if state["cough_s_left"] > 0:
            self.coughing_bio_s += TICK_S
        if state["whitey_s_left"] > 0:
            self.whitey_bio_s += TICK_S
        self.frame_mean += float(lum.mean())
        if self.prev is not None:
            self.frame_change += float(np.abs(lum - self.prev).mean())
        self.prev = lum

    def merge(self, m):
        """TikTok fills a post's caption, tags and counts in over a few seconds: keep the latest."""
        if not m or m.get("key") != self.key:
            return
        for k in META_KEYS + ("liked",):
            if m.get(k) not in (None, "", []):
                self.meta[k] = m[k]

    def to_json(self, clock, now_meta, speed):
        self.merge(now_meta)
        m = self.meta
        n, s = max(self.ticks, 1), self.state or {}
        return {
            "type": "video", "i": self.i, "started": self.started,
            **{k: m.get(k) for k in META_KEYS},
            "bio_s": round(self.bio_s, 2), "real_s": round(time.monotonic() - self.t0, 2),
            "video_s": round(clock.vt, 2), "speed": speed,
            "fly_liked": self.fly_liked, "like_at_bio_s": self.like_bio_s, "like": self.like,
            "liked_on_tiktok": m.get("liked"),
            "fly_followed": self.fly_followed, "fly_reposted": self.fly_reposted,
            "enjoyment_peak": round(self.enjoy_peak, 3), "like_threshold": getattr(self.world, "like_threshold", None),
            "pam_hz_mean": round(self.pam / n, 3), "photoreceptor_hz_mean": round(self.photo / n, 2),
            "frame_mean": round(self.frame_mean / n, 3), "frame_change": round(self.frame_change / max(n - 1, 1), 4),
            "missing_frames": self.missing_frames,
            "nicotine": s.get("nicotine"), "tolerance": s.get("tolerance"), "withdrawal": s.get("withdrawal"),
            "puffs_video": self.puffs, "puffs_total": s.get("puffs"), "escapes": self.escapes,
            "coughs": self.coughs, "whiteys": self.whiteys,
            "coughing_bio_s": round(self.coughing_bio_s, 2), "whitey_bio_s": round(self.whitey_bio_s, 2),
            "exit": self.exit, "fly_swipe_at_bio_s": self.swipe_bio_s,
        }


def video_line(rec):
    like = ""
    if rec["fly_liked"]:
        lr = rec["like"] or {}
        like = "LIKE" + ("d on TikTok" if lr.get("confirmed") else f" ({lr.get('note') or 'unconfirmed'})")
    tags = " ".join("#" + t for t in (rec["hashtags"] or [])[:4])
    return (f"#{rec['i']:<4} @{(rec['author'] or '?')[:20]:<20} bio {rec['bio_s']:5.1f}s  video {rec['video_s']:5.1f}s"
            f"  real {rec['real_s']:5.1f}s  {rec['exit']:<9} {like:<14} {tags}  | {(rec['sound'] or '')[:40]}")


def first_frame(feed, timeout=6.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        lum, info = feed.grab(1.0)
        if lum is not None:
            return lum
        time.sleep(0.1)
    return np.zeros((EYE_H, EYE_W), np.float32)


def live(args):
    from playwright.sync_api import Error as PlaywrightError
    from feed import FeedError, TikTokFeed, proxy_from_env

    proxy = proxy_from_env()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print("Loading the brain ...")
    brain, make_world = load_brain(args.eager)
    feed = TikTokFeed(Path(args.profile), window=args.window,
                      cookies_path=None if args.no_cookies else Path(args.cookies), proxy=proxy)
    print(f"Opening TikTok in Edge ({args.window}{', proxy ' + proxy['server'] if proxy else ''}) ...")
    try:
        meta = feed.open()
    except PlaywrightError as e:
        feed.close()
        raise SystemExit(f"Couldn't start Microsoft Edge through Playwright: {e}")
    print(feed.cookie_note)
    if meta is None:
        feed.close()
        raise SystemExit("No post appeared on the For You page within 45 s. If headless Edge is being "
                         "blocked, try --window offscreen (a normal Edge window placed off-screen).")
    logged_in = feed.logged_in()
    print("Logged in." if logged_in else "NOT logged in: this is TikTok's logged-out feed, and likes are skipped.")

    lum = first_frame(feed)
    world = settle(brain, make_world, lum, args.preroll)
    if args.like_threshold is not None:
        world.like_threshold = args.like_threshold

    header = {
        "type": "run", "started": datetime.now().isoformat(timespec="seconds"), "speed": args.speed,
        "preroll_s": args.preroll,
        "max_watch_s": args.max_watch, "min_swipe_gap_s": args.min_swipe_gap, "min_like_gap_s": args.min_like_gap,
        "like_threshold": getattr(world, "like_threshold", None), "likes_enabled": not args.no_like,
        # The side-effect tuning this run ran under, so its cough and whitey counts can be
        # read against the constants that produced them.
        "side_effects": {k: getattr(world, k, None) for k in
                         ("cough_load_threshold", "cough_chance", "cough_s", "cough_dose_lost",
                          "whitey_ratio", "whitey_tolerance_floor", "whitey_s")},
        "max_likes": args.max_likes, "logged_in": logged_in, "window": args.window, "proxy": bool(proxy),
        "brain": type(brain).__name__, "videos_requested": args.videos,
    }
    log = out.open("a", encoding="utf-8")
    log.write(json.dumps(header, ensure_ascii=False) + "\n")
    log.flush()
    print(f"Logging to {out}\n")

    st = Stepper(brain, world)
    rec = VideoRecord(1, meta, world)
    clock = VideoClock(args.speed)
    last_swipe = last_like = -1e9
    likes_pressed, likes_rejected, done, errors = 0, 0, 0, 0
    pending_swipe = pending_like = None
    tick_wall, t_prev = TICK_S, time.perf_counter()
    last_check = last_dismiss = time.monotonic()
    wall0 = time.monotonic()

    def finish(reason, now_meta=None):
        nonlocal done
        rec.exit = rec.exit or reason
        j = rec.to_json(clock, now_meta, args.speed)
        log.write(json.dumps(j, ensure_ascii=False) + "\n")
        log.flush()
        rec.logged = True
        done += 1
        if not args.quiet:
            print(video_line(j), flush=True)

    try:
        while True:
            st.start(lum)                                      # the GPU runs this tick ...
            try:
                nxt, info = feed.grab(clock.rate(tick_wall))   # ... while the page draws the next frame
            except PlaywrightError:
                nxt, info = None, {"error": True}
            state = st.finish()
            now = time.perf_counter()
            tick_wall += 0.1 * ((now - t_prev) - tick_wall)
            t_prev = now
            rec.tick(state, lum)
            clock.advance(info)
            if nxt is not None:
                lum = nxt
            else:
                rec.missing_frames += 1

            for a in state["actions"]:
                if a == "puff":
                    rec.puffs += 1
                elif a == "escape":
                    rec.escapes += 1
                elif a == "cough":
                    rec.coughs += 1
                elif a == "whitey":
                    rec.whiteys += 1
                elif a == "like" and not rec.fly_liked:
                    rec.fly_liked, rec.like_bio_s, pending_like = True, round(rec.bio_s, 2), True
                elif a == "follow":
                    rec.fly_followed = True
                elif a == "repost":
                    rec.fly_reposted = True
                elif a == "swipe" and not pending_swipe:
                    pending_swipe, rec.swipe_bio_s = "fly_swipe", round(rec.bio_s, 2)
            if not pending_swipe and rec.bio_s >= args.max_watch:
                pending_swipe = "timeout"

            mono = time.monotonic()
            if info.get("gone") or info.get("error") or mono - last_check > 1.0:
                last_check = mono
                try:
                    m = feed.active() if not info.get("error") else feed.recover()
                    errors = 0
                except PlaywrightError:
                    errors += 1
                    if errors > 5:
                        raise FeedError("the browser stopped responding")
                    m = None
                if m and m["key"] != rec.key:                  # the feed moved without us
                    finish("feed_moved")
                    if args.videos and done >= args.videos:
                        break
                    new_video_for_world(world)
                    rec, clock = VideoRecord(done + 1, m, world), VideoClock(args.speed)
                    pending_swipe = pending_like = None
                    lum = first_frame(feed)
                    t_prev = time.perf_counter()
                    continue
                rec.merge(m)
            if mono - last_dismiss > 2.0:
                last_dismiss = mono
                for what in feed.dismiss():
                    print(f"  (closed {what})")

            if pending_like:
                skip = ("likes disabled" if args.no_like else "logged out" if not logged_in
                        else "like limit reached" if args.max_likes and likes_pressed >= args.max_likes
                        else "TikTok rejected earlier likes" if likes_rejected >= 2 else None)
                if skip:
                    rec.like, pending_like = {"pressed": False, "confirmed": False, "note": skip}, None
                elif mono - last_like >= args.min_like_gap:
                    rec.like = feed.like()
                    last_like, pending_like = time.monotonic(), None
                    if rec.like.get("pressed"):
                        likes_pressed += 1
                        likes_rejected = 0 if rec.like.get("confirmed") else likes_rejected + 1
                        if likes_rejected == 2:
                            print("  TikTok didn't accept the last 2 likes; not pressing like again this run.")
                    t_prev = time.perf_counter()

            if pending_swipe and mono - last_swipe >= args.min_swipe_gap:
                if pending_like:
                    rec.like = {"pressed": False, "confirmed": False, "note": "swiped before the like gap"}
                try:
                    end_meta = feed.active()
                except PlaywrightError:
                    end_meta = None
                finish(pending_swipe, end_meta)
                if args.videos and done >= args.videos:
                    break
                last_swipe = time.monotonic()
                meta, presses = feed.swipe()
                new_video_for_world(world)
                rec, clock = VideoRecord(done + 1, meta, world), VideoClock(args.speed)
                rec.swipe_presses = presses
                pending_swipe = pending_like = None
                lum = first_frame(feed)
                last_check, t_prev = time.monotonic(), time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopping ...")
        if rec.ticks and not rec.logged:
            finish("interrupted")
    except FeedError as e:
        print(f"\nStopped: {e}")
        if rec.ticks and not rec.logged:
            finish("error")
    finally:
        wall = time.monotonic() - wall0
        feed.close()
        log.close()
    print(f"\n{done} videos in {wall / 60:.1f} min ({done / max(wall, 1e-9) * 60:.1f} per minute). "
          f"Log: {out}\nSummary: python headless/report.py {out}")


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Headless fly scrolls real TikTok.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--login", action="store_true", help="open a visible Edge window on the fly's profile to log in by hand")
    p.add_argument("--dry", action="store_true", help="no browser: measure brain throughput on synthetic frames")
    p.add_argument("--dry-seconds", type=float, default=20.0, help="brain seconds to simulate in --dry")
    p.add_argument("--videos", type=int, default=0, help="stop after this many videos (0: until Ctrl+C)")
    p.add_argument("--speed", type=float, default=1.0, help="video seconds per brain second")
    p.add_argument("--max-watch", type=float, default=30.0, help="brain seconds on one video before a forced swipe")
    p.add_argument("--min-swipe-gap", type=float, default=2.0, help="minimum real seconds between swipes")
    p.add_argument("--min-like-gap", type=float, default=5.0, help="minimum real seconds between likes")
    p.add_argument("--no-like", action="store_true", help="never press like (the fly's likes are still logged)")
    p.add_argument("--max-likes", type=int, default=0, help="press like at most this many times (0: no limit)")
    p.add_argument("--like-threshold", type=float, default=None, help="override World.like_threshold for this run")
    p.add_argument("--preroll", type=float, default=2.0,
                   help="brain seconds on the first frame before World starts watching, so the network's "
                        "start-up burst doesn't seed World's baselines (0: cold start, as brain/server.py)")
    p.add_argument("--window", choices=["headless", "offscreen", "headed"], default="headless",
                   help="headless Edge, a normal window placed off-screen, or a visible window")
    p.add_argument("--profile", default=str(PROFILE_DIR), help="Edge profile directory for the fly's account")
    p.add_argument("--cookies", default=str(SESSION_COOKIES),
                   help="exported session cookies, imported only when the profile has no session")
    p.add_argument("--no-cookies", action="store_true", help="don't import session cookies")
    p.add_argument("--eager", action="store_true", help="use brain/lif.py's Brain.run instead of FastBrain")
    p.add_argument("--out", default=None, help="JSONL log path (default data/tiktok-runs/<timestamp>.jsonl)")
    p.add_argument("--quiet", action="store_true", help="don't print a line per video")
    args = p.parse_args()
    if args.login:
        login(args)
    elif args.dry:
        dry(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
