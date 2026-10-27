"""Summarize headless TikTok runs as markdown: where the For You feed took the fly.

Run from the project root:
  python headless/report.py                          the latest run
  python headless/report.py data/tiktok-runs/A.jsonl data/tiktok-runs/B.jsonl
  python headless/report.py --all                    every run, in time order
  python headless/report.py --all --out report.md
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from paths import RUNS_DIR

# Tags that ride on almost everything and say nothing about content.
GENERIC = re.compile(r"^(fyp.*|foryou.*|for_you.*|fy|fypage|4u|foru|viral.*|trend.*|xyz.*|parati.*|"
                     r"tiktok.*|capcut.*|explore.*|blowup|goviral|recommendations?|global.*)$")


def load(paths):
    runs, videos = [], []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by a crash
            if row.get("type") == "run":
                runs.append({**row, "file": path.name})
            elif row.get("type") == "video":
                videos.append({**row, "file": path.name})
    return runs, videos


def tags_of(v, generic):
    return [t for t in (v.get("hashtags") or []) if generic or not GENERIC.match(t)]


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def short(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def cell(s):
    return str(s).replace("|", "\\|")


def counter(videos, field, generic=False):
    c = Counter()
    for v in videos:
        if field == "hashtags":
            c.update(set(tags_of(v, generic)))
        elif v.get(field):
            c[v[field]] += 1
    return c


def top_table(title, c, n_videos, k, label):
    lines = [f"### {title}", ""]
    if not c:
        return lines + ["(none recorded)", ""]
    rows = [(i + 1, cell(name if label != "@" else "@" + name), n, pct(n, n_videos)) for i, (name, n) in enumerate(c.most_common(k))]
    return lines + table(["#", title.split()[-1].rstrip("s").lower(), "videos", "share"], rows) + [""]


def quarter_stats(vs):
    n = len(vs)
    return {
        "videos": n,
        "like": pct(sum(v.get("fly_liked", False) for v in vs), n),
        "watch": f"{statistics.mean(v['bio_s'] for v in vs):.1f}" if vs else "-",
        "sponsored": pct(sum(bool(v.get("sponsored")) for v in vs), n),
    }


def drift(videos, generic):
    n = len(videos)
    q = max(1, n // 4)
    first, last = videos[:q], videos[-q:]
    lines = [f"## Drift: first {q} vs last {q} videos", ""]
    if n < 8:
        lines += [f"(only {n} videos; drift is noisy below ~40)", ""]
    fs, ls = quarter_stats(first), quarter_stats(last)
    lines += table(["", "first", "last"], [
        ("like rate", fs["like"], ls["like"]),
        ("mean brain s watched", fs["watch"], ls["watch"]),
        ("sponsored", fs["sponsored"], ls["sponsored"]),
    ]) + [""]
    for field, name in (("hashtags", "hashtags"), ("author", "authors"), ("sound", "sounds")):
        cf, cl = counter(first, field, generic), counter(last, field, generic)
        a, b = set(cf), set(cl)
        jac = len(a & b) / len(a | b) if a | b else 0.0
        tf, tl = cf.most_common(8), cl.most_common(8)
        rows = []
        for i in range(max(len(tf), len(tl))):
            left = f"{cell(tf[i][0])} ({tf[i][1]})" if i < len(tf) else ""
            right = f"{cell(tl[i][0])} ({tl[i][1]})" if i < len(tl) else ""
            rows.append((i + 1, left, right))
        lines += [f"**{name.capitalize()}** (overlap of the two sets: {jac:.0%})", ""]
        lines += (table(["#", "first", "last"], rows) if rows else ["(none recorded)"]) + [""]
    rising = [(t, cl_n, counter(first, "hashtags", generic).get(t, 0))
              for t, cl_n in counter(last, "hashtags", generic).most_common(30)]
    rising = [r for r in rising if r[1] >= 2 and r[1] > 2 * r[2]][:10]
    if rising:
        lines += ["Rising hashtags (at least 2 in the last quarter, more than twice the first): "
                  + ", ".join(f"#{cell(t)} {a}←{b}" for t, a, b in rising), ""]
    return lines


def rolling(videos, window, generic):
    lines = [f"## Rolling windows of {window} videos", ""]
    rows = []
    for s in range(0, len(videos), window):
        vs = videos[s: s + window]
        tags = ", ".join(f"#{cell(t)}" for t, _ in counter(vs, "hashtags", generic).most_common(4))
        rows.append((f"{s + 1}–{s + len(vs)}", pct(sum(v.get("fly_liked", False) for v in vs), len(vs)),
                     f"{statistics.mean(v['bio_s'] for v in vs):.1f}", tags or "-"))
    return lines + table(["videos", "like rate", "mean brain s", "top hashtags"], rows) + [""]


def watch_by_tag(videos, generic, min_n=3):
    per = {}
    for v in videos:
        for t in set(tags_of(v, generic)):
            per.setdefault(t, []).append(v)
    rows = [(t, len(vs), statistics.mean(v["bio_s"] for v in vs), sum(v.get("fly_liked", False) for v in vs) / len(vs))
            for t, vs in per.items() if len(vs) >= min_n]
    if not rows:
        return []
    overall = statistics.mean(v["bio_s"] for v in videos)
    rows.sort(key=lambda r: -r[2])
    lines = [f"## Watch time by hashtag (tags on at least {min_n} videos; overall mean {overall:.1f} brain s)", ""]
    return lines + table(["hashtag", "videos", "mean brain s", "like rate"],
                         [(f"#{cell(t)}", n, f"{w:.1f}", f"{lr:.0%}") for t, n, w, lr in rows[:15]]) + [""]


def liked(videos, generic):
    lv = [v for v in videos if v.get("fly_liked")]
    lines = [f"## What the fly liked ({len(lv)})", ""]
    if not lv:
        return lines + ["Nothing yet.", ""]
    rows = []
    for v in lv:
        lr = v.get("like") or {}
        status = "on TikTok" if lr.get("confirmed") else (lr.get("note") or "unconfirmed")
        link = f"[link]({v['url']})" if v.get("url") else ""
        rows.append((v["i"], cell("@" + (v.get("author") or "?")), cell(short(v.get("desc"), 70)),
                     " ".join(f"#{cell(t)}" for t in tags_of(v, True)[:5]), f"{v.get('enjoyment_peak', 0):.2f}", status, link))
    lines += table(["#", "author", "caption", "hashtags", "enjoyment", "like", "url"], rows) + [""]
    all_c, liked_c = counter(videos, "hashtags", generic), counter(lv, "hashtags", generic)
    lift = [(t, n, n / len(lv) / (all_c[t] / len(videos))) for t, n in liked_c.items() if n >= 2]
    lift.sort(key=lambda r: -r[2])
    if lift:
        lines += ["Hashtags over-represented in liked videos: "
                  + ", ".join(f"#{cell(t)} ({n} liked, {x:.1f}x)" for t, n, x in lift[:10]), ""]
    return lines


def report(paths, window, generic):
    runs, videos = load(paths)
    lines = ["# Where TikTok took the fly", ""]
    if not videos:
        return "\n".join(lines + ["No videos logged in " + ", ".join(p.name for p in paths)])
    n = len(videos)
    liked_n = sum(v.get("fly_liked", False) for v in videos)
    confirmed = sum(bool((v.get("like") or {}).get("confirmed")) for v in videos)
    exits = Counter(v.get("exit") for v in videos)
    kinds = Counter(v.get("kind") for v in videos)
    bio = sum(v["bio_s"] for v in videos)
    real = sum(v.get("real_s", 0) for v in videos)
    thresholds = sorted({r.get("like_threshold") for r in runs if r.get("like_threshold") is not None})
    speeds = sorted({r.get("speed") for r in runs if r.get("speed") is not None})
    lines += table(["", ""], [
        ("runs", ", ".join(f"{r['file']} ({r.get('started', '?')})" for r in runs) or ", ".join(p.name for p in paths)),
        ("logged in", ", ".join(sorted({str(r.get("logged_in")) for r in runs})) or "?"),
        ("speed", ", ".join(f"{s:g}x" for s in speeds) or "?"),
        ("like threshold", ", ".join(f"{t:g}" for t in thresholds) or "?"),
        ("videos", n),
        ("brain time", f"{bio / 60:.1f} min ({statistics.mean(v['bio_s'] for v in videos):.1f} s per video, median {statistics.median(v['bio_s'] for v in videos):.1f})"),
        ("real time", f"{real / 60:.1f} min ({n / max(real, 1e-9) * 60:.1f} videos per minute, swipe waits excluded)"),
        ("fly likes", f"{liked_n} ({pct(liked_n, n)}), {confirmed} confirmed on TikTok"),
        ("would follow / repost (logged, not pressed)",
         f"{sum(bool(v.get('fly_followed')) for v in videos)} / {sum(bool(v.get('fly_reposted')) for v in videos)}"),
        ("moved on by", ", ".join(f"{k} {c}" for k, c in exits.most_common())),
        ("posts", ", ".join(f"{k} {c}" for k, c in kinds.most_common())),
        ("sponsored", f"{sum(bool(v.get('sponsored')) for v in videos)} ({pct(sum(bool(v.get('sponsored')) for v in videos), n)})"),
        ("puffs", max((v.get("puffs_total") or 0) for v in videos)),
        ("mean PAM rate", f"{statistics.mean(v.get('pam_hz_mean') or 0 for v in videos):.2f} Hz"),
    ]) + [""]
    note = "" if generic else " (generic tags like #fyp and #viral left out; --generic keeps them)"
    lines += [f"## Overall{note}", ""]
    lines += top_table("Top hashtags", counter(videos, "hashtags", generic), n, 15, "#")
    lines += top_table("Top authors", counter(videos, "author"), n, 10, "@")
    lines += top_table("Top sounds", counter(videos, "sound"), n, 10, "")
    lines += drift(videos, generic)
    lines += rolling(videos, window or max(5, n // 8), generic)
    lines += watch_by_tag(videos, generic)
    lines += liked(videos, generic)
    return "\n".join(lines)


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="JSONL run logs (default: the latest in data/tiktok-runs)")
    p.add_argument("--all", action="store_true", help="every run in data/tiktok-runs")
    p.add_argument("--window", type=int, default=0, help="rolling window size in videos (default: about an eighth of the run)")
    p.add_argument("--generic", action="store_true", help="keep generic hashtags like #fyp")
    p.add_argument("--out", help="also write the markdown to this file")
    args = p.parse_args()
    if args.runs:
        paths = [Path(r) for r in args.runs]
    else:
        found = sorted(RUNS_DIR.glob("*.jsonl"))
        if not found:
            raise SystemExit(f"No runs in {RUNS_DIR}")
        paths = found if args.all else found[-1:]
    md = report(paths, args.window, args.generic)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
