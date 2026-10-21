"""Python mirror of the cost model in tuner/plugin.c.

Every function here mirrors a C function of the same purpose; keep them in
lock-step (tests/test_parity.py compares against the compiled C driver).

Latency is expressed as a linear combination of parameters
(`latency_coeffs`) so the fitter can solve the latency part as a linear
least-squares problem; `cell_terms` evaluates the same expression.
"""

import math

import numpy as np

from .constants import (
    ALGO_NVLS, ALGO_NVLS_TREE, ALGO_PAT, ALGO_RING, ALGO_TREE, HW_NET, HW_NVLINK,
    NUM_ALGOS, NUM_PROTOS, PROTO_LL, PROTO_LL128, PROTO_SIMPLE,
)
from .params import get_param

# Mirrors COST_MODEL_DISABLED_COST in plugin.c.
DISABLED_COST = 1.0e9


def log2i(n):
    r = 0
    while n > 1:
        n >>= 1
        r += 1
    return r


def _scale(n_nodes):
    return 0 if n_nodes <= 1 else (1 if n_nodes <= 2 else 2)


def per_channel_bw(P, algo, proto, scale, n_nodes, n_channels, nvls_pipeline_factor):
    bw = P["bwIntra"] if n_nodes <= 2 else P["bwInter"]
    cap = P["ringBwCapInter"]
    if algo == ALGO_RING and n_nodes >= 2 and cap > 0.0 and bw > cap:
        bw = cap
    if algo in (ALGO_NVLS, ALGO_NVLS_TREE):
        if proto != PROTO_SIMPLE or n_channels < 2:
            return -1.0
        intra = P["bwIntra"] * P["nvlsEfficiency"] * (n_channels - 1) / n_channels
        intra *= nvls_pipeline_factor
        inter = P["bwInter"] * (2.0 if (n_nodes <= 2 and algo == ALGO_NVLS_TREE) else 1.0)
        nvls_cap = P["perChMaxNVLSTreeBw"][scale] if algo == ALGO_NVLS_TREE else 1e30
        bw = min(intra, inter, nvls_cap)
    return bw


def apply_protocol_cap(bus_bw, algo, proto, P, scale, n_nodes, n_channels):
    if algo == ALGO_RING and proto == PROTO_LL:
        bus_bw = min(P["llMaxBw"][scale], bus_bw * 0.5)
    elif algo == ALGO_RING and proto == PROTO_LL128:
        bus_bw = min(n_channels * P["perChMaxRingLL128Bw"][scale], bus_bw * 0.78)
    elif algo == ALGO_TREE:
        bus_bw = min(n_channels * P["perChMaxTreeBw"][scale], bus_bw * 0.92)
        if proto == PROTO_LL:
            bus_bw = min(P["llMaxBw"][scale], bus_bw / 3.8)
        elif proto == PROTO_LL128:
            factor = (7.0 / 9.0) if n_nodes == 1 else (120.0 / 128.0)
            bus_bw = min(n_channels * P["perChMaxTreeLL128Bw"][scale], bus_bw * factor)
    return bus_bw


def bus_to_algo_ratio(algo, coll, n_ranks):
    """busBw -> algoBw ratio (busBwToAlgoBw); None when unsupported."""
    if n_ranks <= 1:
        return 1.0
    if coll == "allreduce":
        if algo in (ALGO_RING, ALGO_NVLS, ALGO_NVLS_TREE):
            return n_ranks / (2.0 * (n_ranks - 1))
        if algo == ALGO_TREE:
            return 0.5
        return None
    if coll in ("allgather", "reducescatter"):
        if algo in (ALGO_RING, ALGO_NVLS):
            return n_ranks / (n_ranks - 1.0)
        if algo == ALGO_PAT:
            return 1.0  # NCCL uses PAT busBw as algoBw directly.
        return None
    return None


def reg_bw_scale(P, algo, proto, reg_buff):
    if not reg_buff:
        return 1.0
    s = float(P["regBwScale"][algo][proto])
    return s if s > 0.0 else 1.0


def latency_coeffs(P, algo, proto, coll, n_ranks, n_nodes):
    """Latency as {param_key: coefficient} plus a constant, or None if the
    (coll, algo) combination is not modelled.  Mirrors algoLatency() and the
    PAT branch of computeAllGatherOrRSTime()."""
    ppn = n_ranks // n_nodes
    g = P["graphLatencyInter"]
    g_terms = g * (2.0 if proto == PROTO_SIMPLE else 1.0)
    base = f"baseLatencies[{algo}][{proto}]"
    intra = f"hwLatencies[{HW_NVLINK}][{algo}][{proto}]"

    if algo == ALGO_PAT:
        if coll not in ("allgather", "reducescatter"):
            return None
        l2 = log2i(n_nodes)
        return ({base: 1.0, f"hwLatencies[{HW_NET}][{ALGO_PAT}][{proto}]": float(l2),
                 "patLatPerRank": float(n_ranks)}, l2 * 2.0 * g)

    algo_net = ALGO_TREE if ppn == 1 else algo
    inter = f"hwLatencies[{HW_NET}][{algo_net}][{proto}]"
    if coll == "allreduce":
        nsteps = 2 * (n_ranks - 1)
        n_inter = 0 if n_nodes == 1 else 2 * (n_nodes - 1)
    elif coll in ("allgather", "reducescatter"):
        nsteps = n_ranks - 1
        n_inter = 0 if n_nodes == 1 else n_nodes - 1
    else:
        return None

    coeffs = {}

    def add(key, c):
        if c:
            coeffs[key] = coeffs.get(key, 0.0) + float(c)

    const = 0.0
    if algo == ALGO_RING:
        add(base, 1.0)
        add(intra, nsteps - n_inter)
        add(inter, n_inter)
        const += n_inter * g_terms
    elif algo == ALGO_TREE and coll == "allreduce":
        l2 = log2i(n_nodes)
        add(base, 1.0)
        add(intra, 2.0 * (ppn - 1))
        add(inter, 2.0 * l2)
        const += 2.0 * l2 * g_terms
    elif algo == ALGO_NVLS:
        add(intra, 1.0)
        if n_nodes > 1:
            add(inter, 1.0)
            const += g_terms
    elif algo == ALGO_NVLS_TREE and coll == "allreduce":
        l2 = log2i(n_nodes)
        add(base, 1.0)
        add(intra, 1.0)
        add(inter, 2.0 * l2)
        const += 2.0 * l2 * g_terms
    else:
        return None
    return coeffs, const


def _eval_coeffs(P, coeffs, const):
    return const + sum(c * get_param(P, k) for k, c in coeffs.items())


def cell_terms(P, coll, algo, proto, n_ranks, n_nodes, reg_buff=0):
    """(latency_us, algoBw_GBps, busBw_GBps) for one cell, or None when the
    model does not price it.  Time is latency*latCount + nBytes/(1000*algoBw)."""
    if n_ranks <= 1:
        return (0.0, math.inf, math.inf)
    n_ch = int(P["nChannels"])
    scale = _scale(n_nodes)

    if coll == "allreduce":
        if algo == ALGO_NVLS and n_nodes > 1:
            return None
        bw = per_channel_bw(P, algo, proto, scale, n_nodes, n_ch, 2.0)
        if bw < 0.0:
            return None
        cap = P["ringBwCapIntra"]
        if algo == ALGO_RING and proto == PROTO_SIMPLE and n_nodes == 1 and cap > 0.0 and bw > cap:
            bw = cap
        bus = n_ch * bw
        bus = apply_protocol_cap(bus, algo, proto, P, scale, n_nodes, n_ch)
    elif coll in ("allgather", "reducescatter"):
        if algo == ALGO_PAT:
            if P["patBwEff"] <= 0.0 or proto != PROTO_SIMPLE or n_nodes != n_ranks:
                return None
            per_ch = P["bwIntra"] if n_nodes <= 2 else P["bwInter"]
            bus = n_ch * per_ch * P["patBwEff"]
        else:
            if algo == ALGO_NVLS and n_nodes > 1:
                return None
            if algo in (ALGO_TREE, ALGO_NVLS_TREE):
                return None
            bw = per_channel_bw(P, algo, proto, scale, n_nodes, n_ch, 1.0)
            if bw < 0.0:
                return None
            bus = n_ch * bw
            bus = apply_protocol_cap(bus, algo, proto, P, scale, n_nodes, n_ch)
    else:
        return None

    bus *= reg_bw_scale(P, algo, proto, reg_buff)
    ratio = bus_to_algo_ratio(algo, coll, n_ranks)
    if ratio is None:
        return None
    algo_bw = bus * ratio
    if algo_bw <= 0.0:
        return None
    lc = latency_coeffs(P, algo, proto, coll, n_ranks, n_nodes)
    if lc is None:
        return None
    lat = _eval_coeffs(P, *lc)
    if lat < 0.0:
        return None
    return (lat, algo_bw, bus)


def lat_count(algo, num_pipe_ops=1):
    c = num_pipe_ops if algo == ALGO_RING else (num_pipe_ops + 3) // 4
    return max(1, c)


def cell_time(P, coll, algo, proto, n_ranks, n_nodes, n_bytes, reg_buff=0, num_pipe_ops=1):
    """Predicted time (us) as float32, like the C code; None if unpriced."""
    t = cell_terms(P, coll, algo, proto, n_ranks, n_nodes, reg_buff)
    if t is None:
        return None
    lat, algo_bw, _ = t
    us = lat * lat_count(algo, num_pipe_ops) + n_bytes / (1000.0 * algo_bw)
    return float(np.float32(us))


def cell_times(P, coll, algo, proto, n_ranks, n_nodes, sizes, reg_buff=0, num_pipe_ops=1):
    """Vectorised cell_time over an array of sizes (float64); None if unpriced."""
    t = cell_terms(P, coll, algo, proto, n_ranks, n_nodes, reg_buff)
    if t is None:
        return None
    lat, algo_bw, _ = t
    return lat * lat_count(algo, num_pipe_ops) + np.asarray(sizes, dtype=float) / (1000.0 * algo_bw)


def choose(P, coll, cells, n_ranks, n_nodes, n_bytes, reg_buff=0):
    """NCCL's pick among `cells` (list of (algo, proto)) given this model:
    priced cells compete; unpriced cells get DISABLED_COST when the profile
    sets disableUnmodeledCells, otherwise they are treated as unknown (None)."""
    best, best_t = None, math.inf
    for algo, proto in cells:
        t = cell_time(P, coll, algo, proto, n_ranks, n_nodes, n_bytes, reg_buff)
        if t is None:
            if not P["disableUnmodeledCells"]:
                continue
            t = DISABLED_COST
        if t < best_t:
            best, best_t = (algo, proto), t
    return best, best_t


ALL_CELLS = [(a, p) for a in range(NUM_ALGOS) for p in range(NUM_PROTOS)]
