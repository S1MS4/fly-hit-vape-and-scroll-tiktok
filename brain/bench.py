"""Headless check of speed and activity: video only, then a puff, then withdrawal.

Run from the project root:  python brain/bench.py [seconds_per_phase]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lif import Brain
from world import World

BUILD = Path(__file__).resolve().parent.parent / "data" / "build"
TICK_S = 0.05


def main():
    phase_s = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    g = np.load(BUILD / "graph.npz")
    meta = json.loads((BUILD / "groups.json").read_text())
    r = np.load(BUILD / "retina.npz")
    brain = Brain(g["indptr"], g["indices"], g["counts"], g["is_ach"])
    world = World(meta["groups"], r["idx"], r["eye"], r["u"], r["v"])
    steps = int(round(TICK_S * 1000 / brain.p.dt))
    rng = np.random.default_rng(0)
    sync = torch.cuda.synchronize if brain.dev.type == "cuda" else (lambda: None)

    # A moving bright bar on a dark screen, like a busy video.
    def frame(t):
        img = np.full((36, 64), 0.1, np.float32)
        x = int((t * 40) % 64)
        img[:, max(0, x - 4):x + 4] = 0.9
        return img + 0.05 * rng.random(img.shape).astype(np.float32)

    for phase in ["video", "puff", "later"]:
        # puff() refuses while the fly is coughing or in a whitey. At the default phase
        # length the fly has never puffed by itself yet, but a longer one gives it time to
        # during the video phase, and it can still be down when this phase starts. A
        # refused forced puff would leave the before/after comparison measuring nothing,
        # with nothing in the output to say so, so keep asking until it lands and report.
        want_puff, refused, puffed_at = phase == "puff", 0, None
        wall, spikes, actions, st = 0.0, 0, [], None
        for _ in range(int(phase_s / TICK_S)):
            if want_puff:
                if world.puff():
                    want_puff, puffed_at = False, world.t
                else:
                    refused += 1
            world.set_frame(frame(world.t))
            idx, rates = world.sensory_inputs()
            brain.set_inputs(idx, rates)
            brain.ach_gain = world.ach_gain()
            sync(); t0 = time.perf_counter()
            counts = brain.run(steps)
            sync(); wall += time.perf_counter() - t0
            c = counts.cpu().numpy()
            spikes += int(c.sum())
            st = world.update(c, TICK_S)
            actions += st["actions"]
        print(f"[{phase}] {phase_s:.1f}s bio in {wall:.2f}s wall | {spikes / phase_s:,.0f} spikes/s"
              f" | active last tick {int((c > 0).sum())}")
        print("   rates", {k: v for k, v in st["rates"].items()})
        print("   nicotine", st["nicotine"], "ach_gain", st["ach_gain"], "actions", actions)
        # Onsets only reach `actions` after World's warm-up, so read the side effects off
        # the state itself: these are the seconds left at the end of the phase.
        print("   cough_s_left", st["cough_s_left"], "whitey_s_left", st["whitey_s_left"])
        if phase == "puff" and refused:
            landed = f"{refused * TICK_S:.2f}s late" if puffed_at is not None else \
                     "never landed, so this phase is NOT a puff phase"
            print(f"   !! forced puff refused {refused}x (fly coughing or in a whitey): {landed}")


if __name__ == "__main__":
    main()
