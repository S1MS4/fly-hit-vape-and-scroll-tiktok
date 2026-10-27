"""The fly's world: phone pixels, vape smoke, nicotine, and behaviour readouts.

Sensory side turns the environment into Poisson rates for identified neurons.
Motor side turns population firing of identified neurons into actions. Actions
fire when a population rises well above its own running baseline (z-score), so
behaviour comes from the network's dynamics rather than a script.
"""
import math
from dataclasses import dataclass, field

import numpy as np

READOUTS = ["pam", "ppl1", "mbon", "kc", "steer_left", "steer_right", "forward",
            "backward", "escape", "proboscis", "front_leg", "eliquid_orns", "photoreceptors"]


@dataclass
class Baseline:
    """Exponential running mean/variance of a population rate."""
    tau_s: float = 8.0
    mean: float = 0.0
    var: float = 1.0
    ready: bool = False

    def update(self, x, dt):
        if not self.ready:
            self.mean, self.ready = x, True
        a = 1 - math.exp(-dt / self.tau_s)
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)

    def z(self, x):
        return (x - self.mean) / math.sqrt(self.var + 1.0)


@dataclass
class Chemistry:
    odor: float = 0.0          # e-liquid smell around the antennae, 0..1
    nicotine: float = 0.0      # brain nicotine level, arbitrary units
    gut: float = 0.0           # absorbed dose not yet in the brain
    tolerance: float = 0.0     # slow adaptation to nicotine
    puffs: int = 0

    def step(self, dt):
        absorbed = self.gut * (1 - math.exp(-dt / 1.5))
        self.gut -= absorbed
        self.nicotine += absorbed
        self.nicotine *= math.exp(-dt / 25.0)
        self.odor *= math.exp(-dt / 3.0)
        self.tolerance += (self.nicotine - self.tolerance) * (1 - math.exp(-dt / 60.0))

    @property
    def withdrawal(self):
        return max(0.0, self.tolerance - self.nicotine)


@dataclass
class World:
    groups: dict
    retina_idx: np.ndarray
    retina_eye: np.ndarray
    retina_u: np.ndarray
    retina_v: np.ndarray
    chem: Chemistry = field(default_factory=Chemistry)
    baselines: dict = field(default_factory=dict)
    cooldowns: dict = field(default_factory=dict)
    frame: np.ndarray | None = None
    prev_frame: np.ndarray | None = None
    t: float = 0.0
    warmup_s: float = 4.0
    puff_dose: float = 1.0
    enjoyment: float = 0.0     # reward accumulated during the current video
    liked_video: bool = False
    pam_slow: Baseline = field(default_factory=lambda: Baseline(tau_s=30.0))
    # Enjoyment (Hz x s of PAM firing above its slow average) needed to like a video.
    # The 8.7-14.4 peaks this was set against come from headless runs on real TikTok
    # videos (after warm-up) taken before the side effects below existed. A whitey now
    # banks nothing while it lasts, so a video interrupted by one reaches a lower peak
    # than that calibration ever saw. The intent is unchanged - like the better-watched
    # videos, follow rarely, repost very rarely - but the range wants re-measuring on
    # the real network before anyone trusts the exact number again.
    like_threshold: float = 12.0
    # Videos enjoyed far more than usual get a follow, and the very best a repost. Both
    # are real actions on the account, so they also need a like and a long gap.
    follow_threshold: float = 14.0
    repost_threshold: float = 16.0
    follow_gap_s: float = 120.0
    repost_gap_s: float = 300.0
    followed_video: bool = False
    reposted_video: bool = False
    # Nicotine side effects. Both are hand-tuned models of what a hit does to a body, not
    # connectome wiring: no neuron decides them, puff() does, from the chemistry alone.
    # Coughing fit: a harsh pull, meaning one taken while a previous dose is still in the
    # gut or on top of an already loaded brain. cough_load_threshold is the gut + nicotine
    # already on board at which a pull starts to be harsh, and cough_chance the ceiling on
    # how often a harsh one actually turns into a fit (a bad pull is a risk, not a rule).
    cough_load_threshold: float = 3.5
    cough_chance: float = 0.25
    cough_s: float = 3.0
    # What a fit costs. It is a pull the fly couldn't hold, not a pull it never took: this
    # much of the dose comes straight back out before the gut absorbs any of it, and the
    # rest stays down. Coughing the whole hit back would make a harsh pull free, and fits
    # are the brake on a chain of them; keeping none of it would make them cost nothing.
    cough_dose_lost: float = 0.6
    # Whitey (head rush): the hit about to land (the gut, so an un-absorbed remainder of the
    # last pull counts towards it) is whitey_ratio times what the fly is used to. The floor
    # is how big a hit a fly with no tolerance at all can take without greying out, and it
    # keeps the ratio finite at tolerance 0. Common early, gone once tolerance catches up.
    whitey_ratio: float = 1.6
    whitey_tolerance_floor: float = 0.5
    whitey_s: float = 3.5
    cough_t: float = 0.0        # seconds left of the current fit, 0 when not coughing
    whitey_t: float = 0.0       # seconds left of the current head rush, 0 when not gone
    pending_actions: list = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in self.groups.items()}
        self.baselines = {k: Baseline() for k in READOUTS}
        self.cooldowns = {"puff": 0.0, "swipe": 0.0, "escape": 0.0, "follow": 0.0, "repost": 0.0}

    def set_frame(self, lum):
        """lum: HxW float array in [0, 1] of what is on the phone screen."""
        self.prev_frame = self.frame if self.frame is not None else lum
        self.frame = lum

    def puff(self):
        """Take a hit; returns False if the fly couldn't.

        What a pull does depends on what was already on board, so both side effects are
        rolled here, around the moment the dose lands. They are hand-tuned models of a
        body, not connectome wiring, and they are rolled for a puff forced from the
        browser button too. update() emits their onsets: in the same tick when the
        proboscis gate there took the hit, on the next tick when the button called this
        between two of them, and not at all until the warm-up is over.
        """
        if self.cough_t > 0 or self.whitey_t > 0:
            return False       # a convulsing or slumped fly can't get the vape to its mouth
        stacked = self.chem.gut + self.chem.nicotine
        self.chem.odor = 1.0
        self.chem.gut += self.puff_dose
        self.chem.puffs += 1
        self.cooldowns["puff"] = 4.0
        # Coughing fit: a harsh pull, taken while the last dose is still being absorbed or
        # on top of an already loaded brain. Past the threshold it is a chance that grows
        # with how loaded the fly is, not a certainty. One pull gives at most one state and
        # the cough wins it.
        harsh = stacked / self.cough_load_threshold - 1.0
        if harsh > 0 and self.rng.random() < self.cough_chance * min(1.0, harsh):
            # The vape still fired and the smoke is still in the air, so the puff counts
            # and the fly still smells of it. What it loses is the part it couldn't hold:
            # most of the dose leaves the gut before any of it is absorbed. It convulses
            # too hard to work the phone until it stops, so it can't swipe either.
            self.chem.gut = max(0.0, self.chem.gut - self.cough_dose_lost * self.puff_dose)
            self.cough_t = self.cough_s
            self.cooldowns["puff"] = max(self.cooldowns["puff"], self.cough_s)
            self.cooldowns["swipe"] = max(self.cooldowns["swipe"], self.cough_s)
            self.pending_actions.append("cough")
        elif self.chem.gut > self.whitey_ratio * (self.chem.tolerance + self.whitey_tolerance_floor):
            # Whitey: the hit now on its way in is big next to what the fly is used to. It
            # slumps and watches, so it can neither pull again nor swipe until it passes.
            self.whitey_t = self.whitey_s
            self.cooldowns["puff"] = max(self.cooldowns["puff"], self.whitey_s)
            self.cooldowns["swipe"] = max(self.cooldowns["swipe"], self.whitey_s)
            self.pending_actions.append("whitey")
        return True

    def sensory_inputs(self):
        """Return (neuron indices, Poisson rates in Hz) for this tick."""
        idx, rates = [], []

        if self.frame is not None:
            h, w = self.frame.shape
            # Both eyes face the phone with a central overlap.
            x = np.where(self.retina_eye == 0, 0.6 * self.retina_u, 0.4 + 0.6 * self.retina_u)
            px = np.clip((x * (w - 1)).astype(int), 0, w - 1)
            py = np.clip((self.retina_v * (h - 1)).astype(int), 0, h - 1)
            lum = self.frame[py, px]
            change = np.abs(lum - self.prev_frame[py, px])
            idx.append(self.retina_idx)
            rates.append(np.minimum(5 + 50 * lum + 250 * change, 300))

        orns = self.groups["eliquid_orns"]
        idx.append(orns)
        rates.append(np.full(len(orns), 4 + 160 * self.chem.odor))

        # Nicotine gives the reward neurons only a mild lift that tolerance blunts, as in
        # real users: a small rise in dopamine firing, not a flood. Withdrawal drives
        # the punishment-signalling PPL1 cluster.
        pam, ppl1 = self.groups["pam"], self.groups["ppl1"]
        effective = max(0.0, self.chem.nicotine - 0.5 * self.chem.tolerance)
        idx += [pam, ppl1]
        rates += [np.full(len(pam), 6 * effective / (effective + 1.0)),
                  np.full(len(ppl1), 120 * self.chem.withdrawal)]

        return np.concatenate(idx), np.concatenate(rates).astype(np.float32)

    def ach_gain(self):
        # Nicotinic receptor agonism, with tolerance blunting the effect.
        effective = max(0.0, self.chem.nicotine - 0.5 * self.chem.tolerance)
        return 1.0 + 0.35 * effective / (effective + 1.0)

    @staticmethod
    def _tick_down(left, dt):
        """Take one tick off a countdown, snapping to exactly 0 within half a tick.

        Subtracting dt over and over leaves a float residue of about 1e-15, which is
        enough to keep a timer internally alive for one tick after it reads as 0.00 in
        the state dict and after every `== 0` test here should have opened. That gap is
        real: the front end would be told the fly is free while puff() still refuses.
        Snapping closes it, so what update() reports and what puff() enforces agree.
        """
        left -= dt
        return 0.0 if left < 0.5 * dt else left

    def update(self, counts, dt):
        """counts: per-neuron spike counts (numpy) for a window of dt seconds."""
        self.t += dt
        self.chem.step(dt)
        for k in self.cooldowns:
            self.cooldowns[k] = self._tick_down(self.cooldowns[k], dt)
        self.cough_t = self._tick_down(self.cough_t, dt)
        self.whitey_t = self._tick_down(self.whitey_t, dt)

        rates, z = {}, {}
        for k in READOUTS:
            g = self.groups[k]
            r = float(counts[g].sum()) / (max(len(g), 1) * dt) if len(g) else 0.0
            b = self.baselines[k]
            z[k] = b.z(r) if b.ready else 0.0
            b.update(r, dt)
            rates[k] = r
        if self.t > self.warmup_s:
            # Skip the brain's start-up burst, which would otherwise set the "usual" reward
            # level so high that nothing is enjoyed for minutes.
            self.pam_slow.update(rates["pam"], dt)

        actions = []
        if self.t > self.warmup_s:
            if z["proboscis"] > 2.5 and self.cooldowns["puff"] == 0:
                if self.puff():          # it can refuse: a convulsing or slumped fly can't reach the vape
                    actions.append("puff")
            # What that pull may have just started decides the rest of this tick. A whitey
            # is a blackout, not a pause: the fly is slumped against the phone and cannot
            # move on from the video it is stuck on, so nothing it is shown while it lasts
            # is worth anything and it banks no reward at all. A fit leaves it watching,
            # so reward still builds, but it is convulsing too hard to work the phone.
            # Everything below this line reaches the owner's real account and a follow or
            # a repost cannot be taken back, so neither state gets to press anything.
            gone = self.whitey_t > 0
            can_touch_phone = not gone and self.cough_t == 0
            if not gone:
                # Enjoyment builds while the PAM dopamine (reward) cluster fires above its own
                # slow average during a video (Hz x seconds). Enough of it likes the video, once,
                # and the fly keeps watching for a moment.
                self.enjoyment += max(0.0, rates["pam"] - self.pam_slow.mean) * dt
            if can_touch_phone and self.enjoyment > self.like_threshold and not self.liked_video:
                self.liked_video = True
                self.cooldowns["swipe"] = max(self.cooldowns["swipe"], 1.5)
                actions.append("like")
            if (can_touch_phone and self.enjoyment > self.follow_threshold and self.liked_video
                    and not self.followed_video and self.cooldowns["follow"] == 0):
                self.followed_video = True
                self.cooldowns["follow"] = self.follow_gap_s
                self.cooldowns["swipe"] = max(self.cooldowns["swipe"], 1.5)
                actions.append("follow")
            if (can_touch_phone and self.enjoyment > self.repost_threshold and self.liked_video
                    and not self.reposted_video and self.cooldowns["repost"] == 0):
                self.reposted_video = True
                self.cooldowns["repost"] = self.repost_gap_s
                self.cooldowns["swipe"] = max(self.cooldowns["swipe"], 2.0)
                actions.append("repost")
            # Both states also hold the swipe cooldown down for as long as they run, so the
            # test on can_touch_phone is belt and braces; it keeps the rule in one place
            # rather than resting on a cooldown someone could shorten later.
            if can_touch_phone and z["front_leg"] > 2.5 and self.cooldowns["swipe"] == 0:
                self.cooldowns["swipe"] = 2.5
                self.enjoyment, self.liked_video = 0.0, False
                self.followed_video = self.reposted_video = False
                actions.append("swipe")
            # Escape is deliberately left alone: the giant fibre is the most hard-wired
            # reflex the animal has, a fly too gone to swipe would still jump from a
            # looming shadow, and unlike everything above it touches nothing on the account.
            if rates["escape"] > 0 and self.cooldowns["escape"] == 0:
                self.cooldowns["escape"] = 3.0
                actions.append("escape")
        if self.pending_actions:
            # Onsets rolled in puff(), which the browser's Give puff button also calls
            # between ticks, so they are emitted whatever caused the hit. The cooldowns
            # they set are what actually blocks puffing and swiping above.
            #
            # They pass the same warm-up gate as every other action. A puff forced at time
            # zero is not reported as a puff, so reporting the state it caused would log a
            # fit with no pull in front of it. The state itself still runs and still
            # blocks, and update() still reports its seconds left; only the onset that the
            # animation keys off is dropped, for the first warmup_s of a session.
            if self.t > self.warmup_s:
                actions += self.pending_actions
            self.pending_actions.clear()

        return {
            "t": round(self.t, 3),
            "rates": {k: round(v, 2) for k, v in rates.items()},
            "z": {k: round(v, 2) for k, v in z.items()},
            "turn": round(z["steer_right"] - z["steer_left"], 2),
            "walk": round(z["forward"] - z["backward"], 2),
            "nicotine": round(self.chem.nicotine, 3),
            "tolerance": round(self.chem.tolerance, 3),
            "withdrawal": round(self.chem.withdrawal, 3),
            "odor": round(self.chem.odor, 3),
            "puffs": self.chem.puffs,
            "ach_gain": round(self.ach_gain(), 3),
            # Seconds left of each side effect, 0.0 when it is not running: the animation
            # holds its pose while these count down.
            "cough_s_left": round(self.cough_t, 2),
            "whitey_s_left": round(self.whitey_t, 2),
            "actions": actions,
        }
