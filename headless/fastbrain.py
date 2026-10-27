"""The same brain as brain/lif.py, replayed as CUDA graphs for headless runs.

`Brain.run` spends ~85% of its time on the CPU: ~70 kernel launches and several
device syncs per 1 ms step (nonzero, boolean-mask indexing, the edge count),
while the GPU itself does ~0.25 ms of work per step. `FastBrain` keeps the model,
its constants, its state tensors and its update order, and only changes how a
step executes:

- Spikes travel through the synaptic delay as boolean masks instead of index lists.
- Refractory and reset updates use masked in-place ops instead of boolean indexing.
- Event-driven delivery reads the spiking neurons' CSR rows through fixed-capacity
  buffers (at most K spiking neurons and E edges per step), so a whole 50-step tick
  has static shapes and is captured once and replayed as a single CUDA graph.
- The tick records the largest spike and edge count it saw. One sync per tick
  checks them; if a step overflowed its capacity, the tick is rolled back and
  re-run with a larger capacity (or eagerly, exactly like Brain.run), so no
  spike is ever dropped. The re-run restores the random-number state too, so it
  replays the same Poisson input and yields the tick that would have happened;
  redrawing instead would quietly discard the busiest ticks.

Constants are never copied: they come from the parent's LIFParams.
"""
import numpy as np
import torch

from paths import BRAIN_DIR  # noqa: F401  (puts brain/ on sys.path)
from lif import Brain


MIN_EDGES = 1 << 16
MAX_EDGES = 1 << 24


def _bucket_for(edges, headroom=1.25, step=2 ** 0.5):
    """Smallest edge capacity on a geometric ladder (multiples of 4096) with some headroom."""
    e, k = MIN_EDGES, 0
    while e < edges * headroom + 1024 and e < MAX_EDGES:
        k += 1
        e = min(MAX_EDGES, int(MIN_EDGES * step ** k) // 4096 * 4096)
    return e


class FastBrain(Brain):
    # Edge-capacity policy. A tick costs about linearly in capacity; too little capacity
    # costs a re-run. These only trade speed; the simulation is the same either way.
    headroom = 1.05          # capacity >= headroom x the busiest step of the last tick
    bucket_step = 2 ** 0.25  # ratio between capacities (each one is a captured graph)
    shrink_after = 10        # quieter ticks before the capacity comes down

    def __init__(self, indptr, indices, counts, is_ach, params=None, device=None, graphs=True):
        self._ach_gain = 1.0
        super().__init__(indptr, indices, counts, is_ach, params=params, device=device)
        dev, n = self.dev, self.n
        self.graphs = bool(graphs) and dev.type == "cuda"
        self.nnz = int(self.indices.numel())
        # One zero-weight sentinel edge at position nnz absorbs unused edge slots. Rows
        # below nnz are untouched, so the parent's delivery reads the same values.
        self.weights = torch.cat([self.weights, torch.zeros(1, device=dev)])
        self.indices = torch.cat([self.indices, torch.zeros(1, dtype=torch.int32, device=dev)])
        # Per-row start and out-degree (int32), plus a sink row n with no edges.
        self.starts_ext = torch.cat([self.indptr[:-1], self.indptr[-1:]]).to(torch.int32)
        self.deg_ext = torch.cat([self.indptr[1:] - self.indptr[:-1],
                                  torch.zeros(1, dtype=torch.int64, device=dev)]).to(torch.int32)
        # Spikes in flight: pipe[0] is delivered next step, pipe[-1] holds this step's spikes.
        self.pipe = [torch.zeros(n, dtype=torch.bool, device=dev) for _ in range(self.delay_steps)]
        self.counts = torch.zeros(n, dtype=torch.int16, device=dev)
        self.max_cnt = torch.zeros((), dtype=torch.int64, device=dev)
        self.max_tot = torch.zeros((), dtype=torch.int64, device=dev)
        self.gain_t = torch.ones((), device=dev)
        self._gain_set = 1.0
        self.in_prob = torch.zeros(0, device=dev)
        self._snap = None
        self._graphs = {}
        self._pool = None
        self._arange = {}
        self._ones = {}
        self.edges_hint = MIN_EDGES   # edge capacity for the next tick
        self.stats = {"ticks": 0, "reruns": 0, "eager_ticks": 0, "captures": 0}
        self._flags_cpu = torch.zeros(2, dtype=torch.int64, pin_memory=dev.type == "cuda")
        self._counts_cpu = torch.zeros(n, dtype=torch.int16, pin_memory=dev.type == "cuda")

    # --- API compatible with Brain -------------------------------------------------

    @property
    def ach_gain(self):
        return self._ach_gain

    @ach_gain.setter
    def ach_gain(self, value):
        self._ach_gain = float(value)

    def set_inputs(self, idx, rates_hz):
        idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        rates = torch.as_tensor(np.asarray(rates_hz, dtype=np.float32))
        if self.input_idx.numel() != idx.numel():
            # New input size: fresh static buffers, so captured graphs are stale.
            self.input_idx = torch.zeros(idx.numel(), dtype=torch.int64, device=self.dev)
            self.input_rate = torch.zeros(idx.numel(), device=self.dev)
            self.in_prob = torch.zeros(idx.numel(), device=self.dev)
            self._graphs.clear()
        self.input_idx.copy_(idx, non_blocking=True)
        self.input_rate.copy_(rates, non_blocking=True)

    def reset(self):
        self.v.fill_(self.p.v_rest)
        self.g.zero_()
        self.ref.zero_()
        self.alive.fill_(True)
        self.resource.fill_(1.0)
        for m in self.pipe:
            m.zero_()
        self.t_ms = 0.0

    @torch.no_grad()
    def run(self, n_steps):
        """Advance n_steps; return per-neuron spike counts (a fresh CUDA tensor)."""
        self.launch(n_steps)
        self.wait(n_steps)
        return self.counts.clone()

    # --- Split API: launch the tick, do other work (grab a frame), then collect ------

    @torch.no_grad()
    def launch(self, n_steps):
        """Queue a tick on the GPU and return immediately."""
        if self._gain_set != self._ach_gain:
            self.gain_t.fill_(self._ach_gain)
            self._gain_set = self._ach_gain
        self._E = self.edges_hint
        self._K = max(1024, self._E // 16)
        if self.graphs:
            self._rng = torch.cuda.get_rng_state(self.dev)
            self._replay(n_steps, self._K, self._E)
        else:
            self._tick_eager(n_steps)

    @torch.no_grad()
    def wait(self, n_steps):
        """Block until the launched tick is done, re-running it if it overflowed."""
        self.stats["ticks"] += 1
        if self.graphs:
            self._flags_cpu[0].copy_(self.max_cnt, non_blocking=True)
            self._flags_cpu[1].copy_(self.max_tot, non_blocking=True)
            torch.cuda.current_stream(self.dev).synchronize()
            cnt, tot = int(self._flags_cpu[0]), int(self._flags_cpu[1])
            while cnt >= self._K or tot > self._E:
                self.stats["reruns"] += 1
                self._restore()
                torch.cuda.set_rng_state(self._rng, self.dev)   # same input spikes as the first try
                if self._E >= MAX_EDGES:
                    self._tick_eager(n_steps)
                    torch.cuda.current_stream(self.dev).synchronize()
                    break
                self._E = max(self._E * 4 if cnt >= self._K else 0,
                              _bucket_for(tot, self.headroom, self.bucket_step))
                self._E = min(self._E, MAX_EDGES)
                self._K = max(1024, self._E // 16)
                self._replay(n_steps, self._K, self._E)
                self._flags_cpu[0].copy_(self.max_cnt, non_blocking=True)
                self._flags_cpu[1].copy_(self.max_tot, non_blocking=True)
                torch.cuda.current_stream(self.dev).synchronize()
                cnt, tot = int(self._flags_cpu[0]), int(self._flags_cpu[1])
            else:
                # Size the next tick from this one: grow at once, shrink after quieter ticks.
                want = _bucket_for(tot, self.headroom, self.bucket_step)
                if want >= self.edges_hint:
                    self.edges_hint = want
                    self._quiet, self._quiet_max = 0, 0
                else:
                    self._quiet = getattr(self, "_quiet", 0) + 1
                    self._quiet_max = max(getattr(self, "_quiet_max", 0), want)
                    if self._quiet >= self.shrink_after:
                        self.edges_hint, self._quiet, self._quiet_max = self._quiet_max, 0, 0
        self.t_ms += n_steps * self.p.dt

    def counts_numpy(self):
        """Counts of the last tick as numpy (a view of a pinned buffer reused each tick)."""
        self._counts_cpu.copy_(self.counts)
        return self._counts_cpu.numpy()

    # --- Implementation -------------------------------------------------------------

    def _state(self):
        return [self.v, self.g, self.ref, self.resource, *self.pipe]

    def _snapshot(self):
        if self._snap is None:
            self._snap = [torch.empty_like(t) for t in self._state()]
        for d, s in zip(self._snap, self._state()):
            d.copy_(s)

    def _restore(self):
        for snap, cur in zip(self._snap, self._state()):
            cur.copy_(snap)

    def _deliver_static(self, K, E):
        """Deliver pipe[0] through K row slots and E edge slots, with no device sync.

        Exact while fewer than K neurons spiked (so the last row slot is always an
        empty sink) and their rows hold at most E edges; wait() checks both.
        """
        buf0 = self.pipe[0]
        pre = torch.nonzero_static(buf0, size=K, fill_value=self.n).squeeze(1)
        lens = self.deg_ext[pre]
        cum = torch.cumsum(lens, 0, dtype=torch.int32)
        begin = cum - lens                              # first edge slot of each row
        off = self.starts_ext[pre] - begin              # CSR position = off[row] + slot
        # Row of each edge slot = number of rows after the first that begin at or before it.
        # Slots past the last real edge fall in the sink row and land on the sentinel.
        marks = torch.zeros(E + 1, dtype=torch.int32, device=self.dev)
        marks.index_add_(0, begin[1:].clamp(max=E), self._ones[K])
        row = torch.cumsum(marks[:E], 0, dtype=torch.int32)
        pos = torch.index_select(off, 0, row)
        pos += self._arange[E]
        pos.clamp_(max=self.nnz)
        pre_c = pre.clamp(max=self.n - 1)
        eff = self.resource[pre_c] * torch.where(self.is_ach[pre_c], self.gain_t, 1.0)
        w = torch.index_select(self.weights, 0, pos)
        w *= torch.index_select(eff, 0, row)
        self.g.index_add_(0, torch.index_select(self.indices, 0, pos), w)
        torch.maximum(self.max_cnt, buf0.sum(), out=self.max_cnt)
        torch.maximum(self.max_tot, cum[-1].long(), out=self.max_tot)

    def _deliver_exact(self):
        Brain._deliver(self, self.pipe[0].nonzero().squeeze(1))  # the parent's own delivery

    def _step_tail(self):
        """Everything after delivery, as in Brain.run, with masks instead of indices."""
        p = self.p
        if self.input_idx.numel():
            hits = torch.rand_like(self.in_prob) < self.in_prob
            self.g.index_add_(0, self.input_idx, hits.float() * p.w_input)

        self.v += self.decay_m * (p.v_rest - self.v + self.g)
        self.g *= self.decay_syn
        self.resource.mul_(self.recover).add_(1.0 - self.recover)

        refractory = self.ref > 0
        self.v.masked_fill_(refractory, p.v_reset)
        self.ref.sub_(refractory.to(torch.int16))

        spiked = (self.v >= p.v_thresh) & self.alive
        self.v.masked_fill_(spiked, p.v_reset)
        self.ref.masked_fill_(spiked, self.ref_steps)
        self.counts.add_(spiked.to(torch.int16))
        self.resource.copy_(torch.where(spiked, self.resource * (1.0 - p.stp_u), self.resource))
        for i in range(len(self.pipe) - 1):
            self.pipe[i].copy_(self.pipe[i + 1])
        self.pipe[-1].copy_(spiked)

    def _tick_body(self, n_steps, K, E):
        self._snapshot()
        torch.mul(self.input_rate, self.p.dt / 1000.0, out=self.in_prob)
        self.counts.zero_()
        self.max_cnt.zero_()
        self.max_tot.zero_()
        for _ in range(n_steps):
            self._deliver_static(K, E)
            self._step_tail()

    def _tick_eager(self, n_steps):
        self.stats["eager_ticks"] += 1
        self._snapshot()
        torch.mul(self.input_rate, self.p.dt / 1000.0, out=self.in_prob)
        self.counts.zero_()
        for _ in range(n_steps):
            self._deliver_exact()
            self._step_tail()

    def _replay(self, n_steps, K, E):
        key = (n_steps, K, E)
        graph = self._graphs.get(key)
        if graph is None:
            rng = torch.cuda.get_rng_state(self.dev)
            graph = self._capture(n_steps, K, E)      # warm-up and capture draw random numbers
            torch.cuda.set_rng_state(rng, self.dev)
        graph.replay()

    def _buffers(self, K, E):
        if E not in self._arange:
            self._arange[E] = torch.arange(E, dtype=torch.int32, device=self.dev)
        if K not in self._ones:
            self._ones[K] = torch.ones(K - 1, dtype=torch.int32, device=self.dev)

    def _capture(self, n_steps, K, E):
        self.stats["captures"] += 1
        self._buffers(K, E)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        saved = [t.clone() for t in self._state()]
        side = torch.cuda.Stream(self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._tick_body(2, K, E)
        torch.cuda.current_stream(self.dev).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            self._tick_body(n_steps, K, E)
        torch.cuda.current_stream(self.dev).synchronize()
        # Warm-up and capture really ran; put the brain back where it was.
        for sv, cur in zip(saved, self._state()):
            cur.copy_(sv)
        self._graphs[(n_steps, K, E)] = graph
        return graph
