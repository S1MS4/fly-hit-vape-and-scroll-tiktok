"""Whole-CNS leaky integrate-and-fire network on the GPU.

Neuron and synapse constants follow the connectome LIF model described in
Shiu et al. 2024 (Nature, "A Drosophila computational brain model reveals
sensorimotor processing"). Synaptic propagation is event-driven: each step only
the rows of neurons that actually spiked are read from the CSR matrix, so cost
scales with activity rather than with the 25M edges.

Added on top of that model: short-term synaptic depression. Without it, driving
thousands of sensory neurons at once pushes the whole network into a
self-sustaining seizure (Kenyon cells, MBONs and dopamine neurons pinned near
their maximum rate). Depleting a neuron's transmitter as it fires keeps
recurrent loops bounded, as real synapses do.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LIFParams:
    v_rest: float = -52.0    # mV
    v_reset: float = -52.0   # mV
    v_thresh: float = -45.0  # mV
    tau_m: float = 20.0      # ms
    tau_syn: float = 5.0     # ms
    t_ref: float = 2.2       # ms
    w_syn: float = 0.275     # mV per synapse
    delay: float = 1.8       # ms
    dt: float = 1.0          # ms (paper uses 0.1; 1.0 gives similar rates at 2x the speed of 0.5)
    w_input: float = 40.0    # mV kick per external Poisson spike; relays ~1:1 (own calibration)
    stp_u: float = 0.1       # fraction of a neuron's transmitter used per spike
    tau_rec: float = 300.0   # ms, transmitter recovery time


class Brain:
    """Spiking network over a presynaptic-row CSR connectome.

    indptr/indices/counts: CSR where row i holds the postsynaptic targets of i and
    the signed synapse count of each connection (negative = inhibitory).
    is_ach: bool per neuron, True for cholinergic neurons (nicotine targets).
    """

    def __init__(self, indptr, indices, counts, is_ach, params=None, device=None):
        self.p = params or LIFParams()
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n = len(indptr) - 1
        dev, p = self.dev, self.p

        self.indptr = torch.as_tensor(np.asarray(indptr, dtype=np.int64), device=dev)
        self.indices = torch.as_tensor(np.asarray(indices, dtype=np.int32), device=dev)
        self.weights = torch.as_tensor(np.asarray(counts, dtype=np.float32) * p.w_syn, device=dev)
        self.is_ach = torch.as_tensor(np.asarray(is_ach, dtype=bool), device=dev)

        self.v = torch.full((self.n,), p.v_rest, device=dev)
        self.g = torch.zeros(self.n, device=dev)
        self.ref = torch.zeros(self.n, dtype=torch.int16, device=dev)
        self.alive = torch.ones(self.n, dtype=torch.bool, device=dev)
        self.resource = torch.ones(self.n, device=dev)  # available transmitter, 0..1

        self.decay_m = p.dt / p.tau_m
        self.decay_syn = float(np.exp(-p.dt / p.tau_syn))
        self.recover = float(np.exp(-p.dt / p.tau_rec))
        self.ref_steps = int(round(p.t_ref / p.dt))
        self.delay_steps = max(1, int(round(p.delay / p.dt)))
        self.pipeline = deque([self._empty() for _ in range(self.delay_steps)])

        # Multiplier on cholinergic synapses; nicotine raises it above 1.
        self.ach_gain = 1.0

        # External Poisson drive, set per tick by the world: neuron ids + rates (Hz).
        self.input_idx = torch.zeros(0, dtype=torch.int64, device=dev)
        self.input_rate = torch.zeros(0, device=dev)

        self.t_ms = 0.0

    def _empty(self):
        return torch.zeros(0, dtype=torch.int64, device=self.dev)

    def set_inputs(self, idx, rates_hz):
        self.input_idx = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.dev)
        self.input_rate = torch.as_tensor(np.asarray(rates_hz, dtype=np.float32), device=self.dev)

    def reset(self):
        self.v.fill_(self.p.v_rest)
        self.g.zero_()
        self.ref.zero_()
        self.alive.fill_(True)
        self.resource.fill_(1.0)
        self.pipeline = deque([self._empty() for _ in range(self.delay_steps)])
        self.t_ms = 0.0

    def _deliver(self, pre):
        if pre.numel() == 0:
            return
        starts = self.indptr[pre]
        lens = self.indptr[pre + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return
        # Flat positions of every outgoing edge of every spiking neuron.
        row_offsets = torch.cumsum(lens, 0) - lens
        flat = torch.arange(total, device=self.dev) + torch.repeat_interleave(
            starts - row_offsets, lens, output_size=total)
        efficacy = self.resource[pre]
        if self.ach_gain != 1.0:
            efficacy = efficacy * torch.where(self.is_ach[pre], self.ach_gain, 1.0)
        w = self.weights[flat] * torch.repeat_interleave(efficacy, lens, output_size=total)
        self.g.index_add_(0, self.indices[flat].long(), w)

    @torch.no_grad()
    def run(self, n_steps):
        """Advance n_steps; return per-neuron spike counts over the window."""
        p = self.p
        counts = torch.zeros(self.n, dtype=torch.int16, device=self.dev)
        in_prob = self.input_rate * (p.dt / 1000.0)
        for _ in range(n_steps):
            self._deliver(self.pipeline.popleft())

            if self.input_idx.numel():
                hits = torch.rand_like(in_prob) < in_prob
                self.g.index_add_(0, self.input_idx, hits.float() * p.w_input)

            self.v += self.decay_m * (p.v_rest - self.v + self.g)
            self.g *= self.decay_syn
            self.resource.mul_(self.recover).add_(1.0 - self.recover)

            refractory = self.ref > 0
            self.v[refractory] = p.v_reset
            self.ref[refractory] -= 1

            spiked = (self.v >= p.v_thresh) & self.alive
            idx = spiked.nonzero().squeeze(1)
            self.v[idx] = p.v_reset
            self.ref[idx] = self.ref_steps
            counts[idx] += 1
            self.resource[idx] *= 1.0 - p.stp_u
            self.pipeline.append(idx)
            self.t_ms += p.dt
        return counts
