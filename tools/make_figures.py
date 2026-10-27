"""Regenerate the README's data figures in docs/img/fig/, each as a light/dark PNG pair.

Run from the project root:  python tools/make_figures.py [--skip-run] [--cache F] [--only N ...]

  fig-somata   every neuron with a soma, the readout groups on top
  fig-raster   a real run of brain/lif.py through World (GPU): moving bar, one forced puff
  fig-degree   partner and synapse-count distributions and sign shares of graph.npz
  fig-groups   direct signed weight between the readout groups

--skip-run leaves fig-raster alone. --cache keeps the run in an .npz, so restyling the
raster doesn't need the GPU again.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "data" / "build"
OUT = ROOT / "docs" / "img" / "fig"
sys.path.insert(0, str(ROOT / "brain"))

TICK_S = 0.05
RUN_S, PUFF_AT_S = 8.0, 3.0
EYE_W, EYE_H = 36, 64      # the phone frame the app sends, portrait (web/src/phone.ts)
WIDTH_IN, DPI = 8.0, 200   # 1600 px wide
SEED = 0

# One colour per readout group in every figure. Steering is steer_left + steer_right.
GROUPS = {
    "photoreceptors": ("Photoreceptors", "#2f81f7"),
    "eliquid_orns": ("E-liquid ORNs", "#f0883e"),
    "kc": ("Kenyon cells", "#a371f7"),
    "mbon": ("MBONs", "#8957e5"),
    "pam": ("PAM dopamine", "#db61a2"),
    "ppl1": ("PPL1", "#f85149"),
    "steer": ("DNa01/DNa02", "#56d364"),
    "proboscis": ("MN9 proboscis", "#d29922"),
    "front_leg": ("Front-leg DNs", "#3fb950"),
    "escape": ("Giant fibre", "#39c5cf"),
}
REST = ("Rest of network", "#8b949e")
THEMES = {  # ink, muted ink, page colour (for rings around markers)
    "light": ("#1f2328", "#59636e", "#ffffff"),
    "dark": ("#e6edf3", "#9198a1", "#0d1117"),
}


def load_groups():
    g = json.loads((BUILD / "groups.json").read_text())["groups"]
    g["steer"] = g["steer_left"] + g["steer_right"]
    return {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}


# ---------------------------------------------------------------- output


def save(name, draw, **kw):
    """Draw the figure once per theme and write <name>-<theme>.png, transparent."""
    for theme, (ink, muted, page) in THEMES.items():
        rc = {"text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": muted,
              "xtick.color": muted, "ytick.color": muted, "xtick.labelcolor": ink,
              "ytick.labelcolor": ink, "font.family": "sans-serif",
              "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5,
              "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
              "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
              "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
              "legend.frameon": False}
        with plt.rc_context(rc):
            fig = draw(theme=theme, ink=ink, muted=muted, page=page, **kw)
            buf = io.BytesIO()
            fig.savefig(buf, dpi=DPI, transparent=True)
            plt.close(fig)
        img = Image.open(buf)
        path = OUT / f"{name}-{theme}.png"
        img.save(path, optimize=True)
        if path.stat().st_size > 400 * 1024:
            # Too big for the README: 256 colours is plenty for these, gradients aside.
            img = img.quantize(256, method=Image.Quantize.FASTOCTREE)
            img.save(path, optimize=True)
        print(f"wrote {path.relative_to(ROOT)}  {img.size[0]}x{img.size[1]}"
              f"  {path.stat().st_size / 1024:.0f} KB")


def panel(ax, letter, x=-0.06, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="right")


# ---------------------------------------------------------------- the brain run


def phone_frame(t, rng):
    """bench.py's moving bright bar on a dark screen, drawn on the app's portrait frame."""
    img = np.full((EYE_H, EYE_W), 0.1, np.float32)
    x = int((t * 40) % EYE_W)
    img[:, max(0, x - 4):x + 4] = 0.9
    return img + 0.05 * rng.random(img.shape).astype(np.float32)


def raster_rows(groups, retina, n, rng, per_group=40):
    """Neuron ids for the raster: up to per_group from each population, then a random
    sample of everything outside the readout groups. Photoreceptors are drawn from those
    placed on the eye and sorted by the screen column they look at, so the bar shows."""
    rows = {}
    for k in GROUPS:
        if k == "photoreceptors":
            x = np.where(retina["eye"] == 0, 0.6 * retina["u"], 0.4 + 0.6 * retina["u"])
            pick = rng.choice(len(x), per_group, replace=False)
            rows[k] = retina["idx"][pick[np.argsort(x[pick])]]
        else:
            ids = groups[k]
            rows[k] = np.sort(rng.choice(ids, min(per_group, len(ids)), replace=False))
    rest = np.setdiff1d(np.arange(n), np.concatenate(list(groups.values())))
    rows["rest"] = np.sort(rng.choice(rest, per_group, replace=False))
    return rows, rest


def run_brain():
    """Cold start, RUN_S of brain time on the moving bar, one forced puff at PUFF_AT_S.

    Wired like brain/server.py's tick(), but the 50 steps of a tick are run one at a time
    so the raster rows keep 1 ms spike times.
    """
    import torch
    from lif import Brain
    from world import World

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    g = np.load(BUILD / "graph.npz")
    meta = json.loads((BUILD / "groups.json").read_text())
    retina = np.load(BUILD / "retina.npz")
    brain = Brain(g["indptr"], g["indices"], g["counts"], g["is_ach"])
    world = World(meta["groups"], retina["idx"], retina["eye"], retina["u"], retina["v"],
                  rng=np.random.default_rng(SEED))
    groups = load_groups()
    rows, rest = raster_rows(groups, retina, brain.n, rng)
    keys = list(GROUPS) + ["rest"]
    pops = {**{k: groups[k] for k in GROUPS}, "rest": rest}
    row_ids = torch.as_tensor(np.concatenate([rows[k] for k in keys]), device=brain.dev)
    steps = int(round(TICK_S * 1000 / brain.p.dt))
    n_ticks = int(round(RUN_S / TICK_S))
    spikes = torch.zeros((n_ticks * steps, len(row_ids)), dtype=torch.bool, device=brain.dev)
    rates = np.zeros((n_ticks, len(keys)), np.float32)
    chem = np.zeros((n_ticks, 5), np.float32)   # t, nicotine, tolerance, ach gain, smell
    total = np.zeros(n_ticks, np.int64)
    puffs, actions, forced = [], [], False

    t0 = time.perf_counter()
    for k in range(n_ticks):
        if not forced and world.t >= PUFF_AT_S - 1e-9:
            forced = world.puff()
            if forced:
                puffs.append(world.t)
        world.set_frame(phone_frame(world.t, rng))
        idx, r = world.sensory_inputs()
        brain.set_inputs(idx, r)
        brain.ach_gain = world.ach_gain()
        counts = torch.zeros(brain.n, dtype=torch.int16, device=brain.dev)
        for s in range(steps):
            c = brain.run(1)
            counts += c
            spikes[k * steps + s] = c[row_ids] > 0
        c = counts.cpu().numpy()
        st = world.update(c, TICK_S)
        rates[k] = [c[pops[q]].sum() / (len(pops[q]) * TICK_S) for q in keys]
        chem[k] = st["t"], st["nicotine"], st["tolerance"], st["ach_gain"], st["odor"]
        total[k] = c.sum()
        actions += [(st["t"], a) for a in st["actions"]]
        if "puff" in st["actions"]:
            puffs.append(st["t"])
    wall = time.perf_counter() - t0
    print(f"ran {RUN_S:.0f} s of brain time in {wall:.0f} s wall; actions {actions}")
    t_spk, row = np.nonzero(spikes.cpu().numpy())
    return dict(t_spk=t_spk.astype(np.int32), row=row.astype(np.int32),
                n_rows=np.array([len(rows[q]) for q in keys]),
                sizes=np.array([len(pops[q]) for q in keys]), rates=rates, chem=chem,
                total=total, puffs=np.array(puffs), wall=np.array(wall),
                actions=np.array([f"{t:.2f} {a}" for t, a in actions]))


def nice_top(x):
    """x rounded up to 1, 2 or 5 times a power of ten."""
    e = 10 ** np.floor(np.log10(x))
    return next(m * e for m in (1, 2, 5, 10) if m * e >= x)


def draw_raster(run, ink, muted, **_):
    keys = list(GROUPS) + ["rest"]
    style = {**GROUPS, "rest": REST}
    t_end = run["chem"][:, 0]
    edges = np.concatenate([[0.0], t_end])
    fig = plt.figure(figsize=(WIDTH_IN, 9.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.9, 3.3, 1.3], hspace=0.13,
                          left=0.215, right=0.885, top=0.965, bottom=0.055)
    ax = fig.add_subplot(gs[0])
    rate_ax = gs[1].subgridspec(len(keys), 1, hspace=0.35).subplots(sharex=True)
    chem_ax = gs[2].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.35).subplots(sharex=True)
    axes = [ax, *rate_ax, *chem_ax]

    # A: raster. Small groups get a band of at least 8 rows' height so they stay visible.
    y, first, pos, off, length, col = 0.0, 0, [], [], [], []
    spk_s = (run["t_spk"] + 1) / 1000.0
    for q, nr, size in zip(keys, run["n_rows"], run["sizes"]):
        band = max(nr, 8)
        step = band / nr
        for j in range(nr):
            pos.append(spk_s[run["row"] == first + j])
            off.append(y - (j + 0.5) * step)
            length.append(1.0 if step == 1 else min(step, 2.5) * 0.9)
            col.append(style[q][1])
        ax.plot([-0.012, -0.012], [y, y - band], color=style[q][1], lw=3,
                transform=ax.get_yaxis_transform(), clip_on=False, solid_capstyle="butt")
        ax.text(-0.025, y - band / 2, f"{style[q][0]}  {nr}/{size:,}", ha="right",
                va="center", transform=ax.get_yaxis_transform(), fontsize=7.5)
        y -= band + 4
        first += nr
    ax.eventplot(pos, lineoffsets=off, linelengths=length, colors=col, linewidths=0.55)
    ax.set_ylim(y + 2, 2)
    ax.text(-0.025, 3, "rows / neurons in group", ha="right", va="bottom", fontsize=6.5,
            color=muted, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    panel(ax, "A", x=-0.255)

    # B: population rate per 50 ms tick. Each strip is scaled to what comes after the
    # first 0.5 s; the start-up burst is clipped at the top and its peak written in.
    late = t_end > 0.5
    for a, q, i in zip(rate_ax, keys, range(len(keys))):
        r = run["rates"][:, i]
        top = nice_top(max(5.0, 1.1 * r[late].max()))
        a.stairs(np.minimum(r, top), edges, color=style[q][1], fill=True, alpha=0.3, lw=0)
        a.stairs(np.minimum(r, top), edges, color=style[q][1], lw=0.9)
        if r.max() > top:
            k = int(r.argmax())
            a.text(t_end[k] + 0.06, top * 0.98, f"{r.max():.0f} Hz", fontsize=6.5,
                   va="top", color=muted)
        a.set_ylim(0, top)
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.text(-0.025, 0.4, style[q][0], transform=a.transAxes, ha="right", va="center",
               fontsize=7.5)
        a.text(1.01, 0.4, f"0–{top:g} Hz", transform=a.transAxes, ha="left", va="center",
               fontsize=6.5, color=muted)
        a.tick_params(axis="x", length=0, labelbottom=False)
    panel(rate_ax[0], "B", x=-0.255, y=1.15)

    # C: nicotine, tolerance and smell share a scale; the ACh gain gets its own strip.
    c = np.vstack([[0, 0, 0, 1, 0], run["chem"]])
    a = chem_ax[0]
    for j, lab, ls in ((1, "nicotine N", "-"), (2, "tolerance T", "--"), (4, "smell ω", ":")):
        a.plot(c[:, 0], c[:, j], color=ink if j != 4 else muted, ls=ls, lw=1.3)
        a.text(1.01, c[-1, j], lab, transform=a.get_yaxis_transform(), fontsize=7,
               va="center", color=ink if j != 4 else muted)
    a.set_ylim(0, 1.3)
    a.set_yticks([0, 0.5, 1])
    a.set_ylabel("a.u.", fontsize=7.5)
    b = chem_ax[1]
    b.plot(c[:, 0], c[:, 3], color=ink, lw=1.3)
    b.text(1.01, c[-1, 3], "ACh gain γ", transform=b.get_yaxis_transform(), fontsize=7,
           va="center")
    b.set_ylim(0.98, 1.25)
    b.set_yticks([1, 1.2])
    a.tick_params(axis="x", labelbottom=False)
    b.set_xlabel("brain time since a cold start (s)")
    panel(a, "C", x=-0.255, y=1.0)

    for i, t in enumerate(run["puffs"]):
        for a in axes:
            a.axvline(t, color=ink, lw=0.8, alpha=0.6, ls="-" if i == 0 else "--", zorder=0)
        ax.text(t, 3, "forced puff" if i == 0 else "the fly's own puff", ha="center",
                va="bottom", fontsize=7.5)
    for a in axes:
        a.set_xlim(0, t_end[-1])
    b.set_xticks(np.arange(0, t_end[-1] + 0.01, 1))
    ax.tick_params(axis="x", labelbottom=False)
    return fig


def sign_classes(g, groups):
    """Sender class of every stored connection: 0 ACh, 1 photoreceptor relay (+1),
    2 unclear transmitter (+1/2), 3 inhibitory (GABA, glutamate, histamine)."""
    n = len(g["indptr"]) - 1
    pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(g["indptr"]))
    inhib = np.zeros(n, bool)
    inhib[pre[g["counts"] < 0]] = True
    photo = np.zeros(n, bool)
    photo[groups["photoreceptors"]] = True
    cls = np.select([g["is_ach"], inhib, photo], [0, 3, 1], 2)
    return pre, cls[pre]


def log_density(x, per_decade=10):
    """Share of x per unit value, over log-spaced integer bins (x >= 1)."""
    hi = np.log10(x.max() + 1)
    e = np.unique(np.floor(np.logspace(0, hi, int(hi * per_decade) + 2)).astype(int))
    h, _ = np.histogram(x, e)
    keep = h > 0
    return np.sqrt(e[:-1] * (e[1:] - 1))[keep], (h / (len(x) * np.diff(e)))[keep]


E_COL, I_COL = "#e5534b", "#4184e4"   # sign, not a group: excitatory / inhibitory


def draw_degree(g, cls, ink, muted, **_):
    n = len(g["indptr"]) - 1
    out_deg = np.diff(g["indptr"])
    in_deg = np.bincount(g["indices"], minlength=n)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.9),
                                  gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.6,
                                               "left": 0.075, "right": 0.97, "top": 0.9,
                                               "bottom": 0.17})
    # A: partners per neuron.
    for d, lab, color, ls in ((out_deg, "out (sends to)", ink, "-"),
                              (in_deg, "in (receives from)", muted, "--")):
        x, y = log_density(d[d > 0])
        a.plot(x, y, color=color, lw=1.3, ls=ls, label=f"{lab}\nmedian {np.median(d):.0f}")
    a.set(xscale="log", yscale="log", xlabel="partners per neuron, k",
          ylabel="share of neurons with k partners")
    a.legend(loc="lower left", bbox_to_anchor=(0, 0.15), fontsize=6.8, handlelength=1.8,
             labelspacing=0.8)
    a.text(0.03, 0.03, f"not shown: {(out_deg == 0).sum():,} send to nobody,\n"
           f"{(in_deg == 0).sum():,} receive from nobody", transform=a.transAxes,
           fontsize=6.5, color=muted)
    panel(a, "A", x=-0.2)

    # B: synapses per connection, for senders whose sign is +-1 (count stored exactly).
    s = np.abs(g["counts"]).astype(np.int64)
    for sel, lab, color in ((cls <= 1, "excitatory", E_COL), (cls == 3, "inhibitory", I_COL)):
        x, y = log_density(s[sel])
        b.plot(x, y, color=color, lw=1.2, marker="o", ms=2.2,
               label=f"{lab}, median {np.median(s[sel]):.0f}")
    b.set(xscale="log", yscale="log", xlabel="synapses per connection, s",
          ylabel="share of connections with s synapses")
    b.legend(loc="lower left", fontsize=6.8, handlelength=1.5)
    b.set_ylim(top=3)
    single = (s[cls != 2] == 1).mean()
    b.annotate(f"{100 * single:.0f}% are a\nsingle synapse", (1, single), (2.5, 0.25),
               fontsize=6.5, color=muted, arrowprops={"arrowstyle": "-", "color": muted,
                                                      "lw": 0.6})
    panel(b, "B", x=-0.2)

    # C: shares by sender sign, of connections and of |stored weight|.
    zero = s == 0
    rows = [("ACh  (+1)", (cls == 0), E_COL),
            ("photoreceptors  (+1)", (cls == 1), E_COL),
            ("unclear transmitter  (+½)", (cls == 2) & ~zero, "#e5534b80"),
            ("unclear, rounds to 0", zero, muted),
            ("GABA, Glu, His  (−1)", (cls == 3), I_COL)]
    total_w = s.sum()
    for i, (lab, sel, color) in enumerate(rows):
        conn, syn = 100 * sel.mean(), 100 * s[sel].sum() / total_w
        c.barh(i - 0.19, conn, 0.36, color=color)
        c.barh(i + 0.19, syn, 0.36, color=color, alpha=0.45)
        c.text(conn + 1, i - 0.19, f"{conn:.2g}%" if conn < 10 else f"{conn:.1f}%",
               va="center", fontsize=6.5)
        c.text(syn + 1, i + 0.19, f"{syn:.2g}%" if syn < 10 else f"{syn:.1f}%",
               va="center", fontsize=6.5, color=muted)
    c.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    c.invert_yaxis()
    c.set_xlim(0, 75)
    c.set_xlabel("share (%)")
    c.tick_params(axis="y", length=0)
    c.spines["left"].set_visible(False)
    c.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=muted),
                      plt.Rectangle((0, 0), 1, 1, color=muted, alpha=0.45)],
             labels=["of connections", "of synapses (|c|)"], loc="center right",
             fontsize=6.8, handlelength=1, borderaxespad=0)
    panel(c, "C", x=-0.02, y=1.02)
    return fig


def fmt(v):
    a = abs(v)
    s = f"{a / 1e6:.2g}M" if a >= 1e6 else f"{a / 1e3:.3g}k" if a >= 1e4 else f"{a:,}"
    return ("−" if v < 0 else "") + s


def draw_groups(M, ink, muted, theme, **_):
    keys = list(GROUPS)
    mid = "#eaeef2" if theme == "light" else "#262c36"
    cmap = LinearSegmentedColormap.from_list("ei", [I_COL, mid, E_COL])
    vmax = np.abs(M).max()
    norm = SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.7),
                           gridspec_kw={"left": 0.2, "right": 0.88, "top": 0.8, "bottom": 0.03})
    im = ax.imshow(np.ma.masked_equal(M, 0), cmap=cmap, norm=norm)
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if v == 0:
            ax.text(j, i, "·", ha="center", va="center", color=muted, fontsize=8)
            continue
        r, g_, b = cmap(norm(v))[:3]
        dark = 0.2126 * r + 0.7152 * g_ + 0.0722 * b < 0.45
        ax.text(j, i, fmt(int(v)), ha="center", va="center", fontsize=6.8,
                color="#ffffff" if dark else "#1f2328")
    d = keys.index("pam")
    ax.text(len(keys) / 2 - 0.5, d + 0.5, "PAM and PPL1 are dopaminergic (σ = 0): "
            "no fast synapses in the model", ha="center", va="center", fontsize=7, color=muted)
    names = [GROUPS[k][0] for k in keys]
    ax.set_yticks(range(len(keys)), names)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(keys)), names, rotation=40, ha="left", rotation_mode="anchor")
    ax.tick_params(length=0, pad=17)
    for i, k in enumerate(keys):
        ax.scatter(-0.8, i, s=36, marker="s", color=GROUPS[k][1], clip_on=False)
        ax.scatter(i, -0.8, s=36, marker="s", color=GROUPS[k][1], clip_on=False)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("from (presynaptic)")
    ax.set_xlabel("to (postsynaptic)")
    ax.xaxis.set_label_position("top")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03,
                      ticks=[-1e4, -100, -1, 0, 1, 100, 1e4, 1e6])
    cb.ax.set_yticklabels(["−10k", "−100", "−1", "0", "1", "100", "10k", "1M"], fontsize=7)
    cb.outline.set_visible(False)
    cb.set_label("Σ signed synapse count c  (W = 0.275 mV · c)", fontsize=7.5)
    return fig


def draw_somata(ink, muted, page, theme, **_):
    s = np.fromfile(BUILD / "somata.bin", np.float32).reshape(-1, 5)
    groups = load_groups()
    where = np.full(json.loads((BUILD / "groups.json").read_text())["n"], -1)
    where[s[:, 3].astype(int)] = np.arange(len(s))
    grey, alpha = ("#57606a", 0.16) if theme == "light" else ("#8b949e", 0.2)
    fig = plt.figure(figsize=(WIDTH_IN, 4.3))
    a = fig.add_axes([0.0, 0.01, 0.34, 0.97])
    b = fig.add_axes([0.37, 0.27, 0.62, 0.71])
    brain = s[:, 2] < -0.3      # the brain and the ventral nerve cord are apart along z
    order = sorted(GROUPS, key=lambda k: -len(groups[k]))  # small groups drawn last
    for ax, (i, j), keep, scale in ((a, (0, 2), slice(None), 0.6), (b, (0, 1), brain, 1.0)):
        p = s[keep]
        ax.scatter(p[:, i], p[:, j], s=0.3 * scale, c=grey, alpha=alpha, lw=0, rasterized=True)
        for k in order:
            rows = where[groups[k]]
            rows = rows[rows >= 0]
            rows = rows[s[rows, 2] < -0.3] if ax is b else rows
            big = len(rows) > 1000
            size = 1.2 if big else 5 if len(rows) > 50 else 24
            ax.scatter(s[rows, i], s[rows, j], s=size * scale, c=GROUPS[k][1],
                       lw=0 if big else 0.5, edgecolors=page, alpha=0.8 if big else 1)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    a.text(0.5, 0.985, "brain", transform=a.transAxes, ha="center", va="top", fontsize=7.5,
           color=muted)
    a.text(0.5, 0.0, "ventral nerve cord", transform=a.transAxes, ha="center", fontsize=7.5,
           color=muted)
    b.text(0.5, -0.01, "brain, seen from the front", transform=b.transAxes, ha="center",
           va="top", fontsize=7.5, color=muted)
    panel(a, "A", x=0.08, y=0.95)
    panel(b, "B", x=0.0, y=0.96)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=GROUPS[k][1]) for k in GROUPS]
    shown = {k: int((where[groups[k]] >= 0).sum()) for k in GROUPS}
    labels = [f"{GROUPS[k][0]}  {shown[k]:,}/{len(groups[k]):,}" for k in GROUPS]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.38, 0.0), ncol=3,
               fontsize=7, handletextpad=0.3, columnspacing=1.2, labelspacing=0.45)
    fig.text(0.385, 0.2, f"{len(s):,} of {len(where):,} neurons have a soma in the volume; "
             "numbers are somata shown / neurons in group", fontsize=6.5, color=muted)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-run", action="store_true", help="don't run the brain; no fig-raster")
    p.add_argument("--cache", type=Path, help="reuse the run saved in this .npz, or save it there")
    p.add_argument("--only", nargs="*", help="just these figures, e.g. --only raster degree")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = lambda f: not a.only or f in a.only
    if want("somata"):
        save("fig-somata", draw_somata)
    if want("degree") or want("groups"):
        g = dict(np.load(BUILD / "graph.npz"))
        groups = load_groups()
        pre, cls = sign_classes(g, groups)
        if want("degree"):
            save("fig-degree", draw_degree, g=g, cls=cls)
        if want("groups"):
            lab = np.full(len(g["indptr"]) - 1, -1)
            for i, k in enumerate(GROUPS):
                lab[groups[k]] = i
            lp, lq = lab[pre], lab[g["indices"]]
            sel = (lp >= 0) & (lq >= 0)
            M = np.zeros((len(GROUPS),) * 2, np.int64)
            np.add.at(M, (lp[sel], lq[sel]), g["counts"][sel])
            save("fig-groups", draw_groups, M=M)
    if want("raster") and not a.skip_run:
        if a.cache and a.cache.exists():
            run = dict(np.load(a.cache))
        else:
            run = run_brain()
            if a.cache:
                np.savez_compressed(a.cache, **run)
        save("fig-raster", draw_raster, run=run)


if __name__ == "__main__":
    main()
