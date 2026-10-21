"""Fit the rail cost-model profile to forced-cell sweep data.

Stages (see ../README.md for the reasoning):
  1. per-cell affine fit  T(s) = L + k*s  (robust; non-affine points tagged
     as anomalies and excluded from the model fit);
  2. closed-form start: latency parameters by bounded linear least squares
     over all (topology, cell) L values (hop counts from the model), bandwidth
     parameters by inverting each cell's cap chain;
  3. joint refinement of all free parameters on log(time) (soft_l1) with a
     weak pull toward the stage-2 values;
  4. selection-quality polish: minimise a smooth surrogate of the regret
     against the oracle (busbw lost by the model's pick), keeping the
     log-time loss within `loss_slack` of stage 3;
  5. cross-scale validation: leave one node count out, fit, evaluate;
  6. final fit on all data + acceptance gate at the nightly scale.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize

from . import costmodel as cm
from . import db as dbm
from . import oracle as orc
from .constants import (
    ALGO_NVLS_TREE, ALGO_PAT, ALGO_RING, ALGO_TREE, PROTO_LL, PROTO_LL128, PROTO_SIMPLE,
    cell_name, parse_cell,
)
from .params import copy_params, get_param, key_label, set_param


@dataclass
class Group:
    run_id: str
    coll: str
    nRanks: int
    nNodes: int
    R: int
    regbuff: int
    nvls: int
    algo: int
    proto: int
    sizes: np.ndarray
    times: np.ndarray
    busbw: np.ndarray
    mask: np.ndarray = None          # True = usable (not anomaly)
    L_obs: float = None
    k_obs: float = None

    @property
    def cell(self):
        return cell_name(self.algo, self.proto)

    @property
    def scale(self):
        return cm._scale(self.nNodes)


@dataclass
class FitConfig:
    profile: str = "rail_nop2p"
    anomaly_floor: float = 0.25
    reg_weight: float = 0.01
    loss_slack: float = 0.05
    polish: bool = True
    polish_maxiter: int = 4000
    temperature: float = 0.02
    gate_nodes: list = None
    gate_nvls: tuple = (0,)
    gate_weight: float = 3.0
    min_tol: float = orc.MIN_TOL
    fit_regbuff: str = "auto"         # auto | on | off
    verbose: bool = False


@dataclass
class FitState:
    params: dict
    free: list
    theta0: np.ndarray = None
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_groups(con, profile):
    groups = []
    for run in dbm.runs(con, "status='ok' AND kind='forced' AND profile=?", (profile,)):
        if run["cell"] in (None, "default") or run["nNodes"] is None:
            continue
        pts = dbm.last_cycle(con, run["run_id"])
        if len(pts) < 3:
            continue
        a, p = parse_cell(run["cell"])
        groups.append(Group(
            run_id=run["run_id"], coll=run["coll"], nRanks=run["nRanks"], nNodes=run["nNodes"],
            R=run["R"], regbuff=run["regbuff"], nvls=run["nvls"], algo=a, proto=p,
            sizes=np.array([x["size"] for x in pts], dtype=float),
            times=np.array([x["time_oop"] for x in pts], dtype=float),
            busbw=np.array([x["busbw_oop"] for x in pts], dtype=float)))
    return groups


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def fit_affine(g, floor):
    s, t = g.sizes, g.times
    x0 = [max(float(t.min()) * 0.5, 1e-3), max(float(t[-1] - t.min()) / float(s[-1]), 1e-12)]

    def res(x):
        return (x[0] + x[1] * s) / t - 1.0

    r = least_squares(res, x0, bounds=([0.0, 0.0], [np.inf, np.inf]), loss="soft_l1",
                      f_scale=0.05, x_scale=[max(x0[0], 1.0), max(x0[1], 1e-9)])
    resid = res(r.x)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    thr = max(floor, 5 * 1.4826 * mad)
    g.mask = np.abs(resid) <= thr
    g.L_obs, g.k_obs = float(r.x[0]), float(r.x[1])
    return resid


def stage1(groups, cfg):
    anomalies = []
    for g in groups:
        resid = fit_affine(g, cfg.anomaly_floor)
        for i in np.where(~g.mask)[0]:
            anomalies.append({"run_id": g.run_id, "coll": g.coll, "nNodes": g.nNodes,
                              "nRanks": g.nRanks, "R": g.R, "cell": g.cell,
                              "size": int(g.sizes[i]), "rel_resid": float(resid[i])})
    return anomalies


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
def latency_keys(params, groups):
    keys = []
    for g in groups:
        lc = cm.latency_coeffs(params, g.algo, g.proto, g.coll, g.nRanks, g.nNodes)
        if lc:
            for k in lc[0]:
                if k not in keys:
                    keys.append(k)
    return keys


def stage2_latency(params, groups, keys, prior_weight=0.05):
    rows, rhs, w = [], [], []
    for g in groups:
        lc = cm.latency_coeffs(params, g.algo, g.proto, g.coll, g.nRanks, g.nNodes)
        if not lc or g.L_obs is None:
            continue
        coeffs, const = lc
        rows.append([coeffs.get(k, 0.0) for k in keys])
        rhs.append(g.L_obs - const)
        w.append(1.0 / max(g.L_obs, 1.0))
    if not rows:
        return
    A = np.array(rows) * np.array(w)[:, None]
    b = np.array(rhs) * np.array(w)
    theta0 = np.array([get_param(params, k) for k in keys])
    scale = np.maximum(np.abs(theta0), 1.0)
    A = np.vstack([A, prior_weight * np.diag(1.0 / scale)])
    b = np.concatenate([b, prior_weight * theta0 / scale])
    sol = lsq_linear(A, b, bounds=(0.0, 1e5))
    for k, v in zip(keys, sol.x):
        set_param(params, k, float(v))


def _bus_obs(g):
    ratio = cm.bus_to_algo_ratio(g.algo, g.coll, g.nRanks)
    if not g.k_obs or ratio is None:
        return None
    algo_bw = 1.0 / (1000.0 * g.k_obs)
    return algo_bw / ratio


def stage2_bandwidth(params, groups):
    """Invert each cell's cap chain (rail regime) to start the bandwidth
    parameters; medians across topologies."""
    est = defaultdict(list)
    n_ch = int(params["nChannels"])
    for g in groups:
        if g.regbuff:
            continue
        bus = _bus_obs(g)
        if not bus:
            continue
        sc = g.scale
        a, p = g.algo, g.proto
        if a == ALGO_RING and p == PROTO_SIMPLE:
            est["bwInter" if g.nNodes > 2 else "bwIntra"].append(bus / n_ch)
        elif a == ALGO_RING and p == PROTO_LL:
            est[f"llMaxBw[{sc}]"].append(bus)
        elif a == ALGO_RING and p == PROTO_LL128:
            est[f"perChMaxRingLL128Bw[{sc}]"].append(bus / n_ch)
        elif a == ALGO_TREE and p == PROTO_SIMPLE and g.coll == "allreduce":
            est[f"perChMaxTreeBw[{sc}]"].append(bus / n_ch)
        elif a == ALGO_TREE and p == PROTO_LL128 and g.coll == "allreduce":
            est[f"perChMaxTreeLL128Bw[{sc}]"].append(bus / n_ch)
        elif a == ALGO_NVLS_TREE:
            est[f"perChMaxNVLSTreeBw[{sc}]"].append(bus / n_ch)
        elif a == ALGO_PAT:
            per = params["bwInter"] if g.nNodes > 2 else params["bwIntra"]
            if per > 0:
                est["patBwEff"].append(bus / (n_ch * per))
    for k, vals in est.items():
        if k != "patBwEff":
            set_param(params, k, float(np.median(vals)))
    if est.get("patBwEff"):
        per = params["bwInter"]
        vals = []
        for g in groups:
            if g.algo == ALGO_PAT and not g.regbuff and _bus_obs(g):
                vals.append(_bus_obs(g) / (n_ch * per))
        set_param(params, "patBwEff", float(np.median(vals)))
    return sorted(est)


def regbuff_keys(groups, params, tol, mode):
    """regBwScale keys to fit: cells whose stage-1 bandwidth differs between
    R=0 and R>=1 by more than the noise band (mode auto), all cells with
    regBuff data (on), or none (off)."""
    if mode == "off":
        return {}
    pairs = defaultdict(lambda: {0: [], 1: []})
    for g in groups:
        bus = _bus_obs(g)
        if bus:
            pairs[(g.coll, g.nRanks, g.nNodes, g.algo, g.proto)][g.regbuff].append(bus)
    ratios = defaultdict(list)
    for (coll, _nr, _nn, a, p), d in pairs.items():
        if d[0] and d[1]:
            ratios[(a, p)].append(np.median(d[1]) / np.median(d[0]))
    out = {}
    for (a, p), rs in ratios.items():
        r = float(np.median(rs))
        if mode == "on" or abs(r - 1.0) > tol:
            out[f"regBwScale[{a}][{p}]"] = r
    return out


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------
def with_theta(base, keys, theta):
    P = copy_params(base)
    for k, v in zip(keys, theta):
        set_param(P, k, float(v))
    return P


def log_residuals(P, groups):
    out = []
    for g in groups:
        m = g.mask if g.mask is not None else np.ones_like(g.sizes, dtype=bool)
        pred = cm.cell_times(P, g.coll, g.algo, g.proto, g.nRanks, g.nNodes, g.sizes[m], g.regbuff)
        if pred is None:
            out.append(np.full(int(m.sum()), 10.0))
        else:
            out.append(np.log(np.maximum(pred, 1e-9)) - np.log(g.times[m]))
    return np.concatenate(out) if out else np.zeros(0)


def log_loss(P, groups):
    r = log_residuals(P, groups)
    return float(np.mean(r * r)) if len(r) else 0.0


def bounds_for(key):
    name = key.split("[")[0]
    if name in ("baseLatencies", "hwLatencies", "patLatPerRank", "graphLatencyInter"):
        return 0.0, 1e5
    if name == "nvlsEfficiency":
        return 1e-3, 1.0
    if name == "patBwEff":
        return 1e-3, 2.0
    if name == "regBwScale":
        return 0.05, 5.0
    return 1e-4, 1e4


# ---------------------------------------------------------------------------
# Regret (selection quality) on the oracle
# ---------------------------------------------------------------------------
class RegretProblem:
    """Precomputed oracle matrices per key: sizes x available cells, with the
    regret of each cell at each size."""

    def __init__(self, oracle, profile, weights=None):
        self.keys = []
        weights = weights or {}
        for key, ents in orc.index(oracle).items():
            if key[0] != profile:
                continue
            sizes = sorted(ents)
            cells = sorted({c for s in sizes for c in ents[s]["cells"]})
            reg = np.ones((len(sizes), len(cells)))
            for i, s in enumerate(sizes):
                for j, c in enumerate(cells):
                    reg[i, j] = orc.regret(ents[s], c)
            self.keys.append({
                "key": key, "coll": key[1], "nNodes": key[2], "nRanks": key[3], "R": key[4],
                "regbuff": 1 if key[4] >= 1 else 0, "nvls": key[5],
                "sizes": np.array(sizes, dtype=float), "cells": cells,
                "cell_idx": [parse_cell(c) for c in cells], "regret": reg,
                "entries": [ents[s] for s in sizes],
                "w": weights.get(key[2], 1.0),
            })

    def times(self, P, k):
        T = np.full((len(k["sizes"]), len(k["cells"])), np.inf)
        for j, (a, p) in enumerate(k["cell_idx"]):
            t = cm.cell_times(P, k["coll"], a, p, k["nRanks"], k["nNodes"], k["sizes"], k["regbuff"])
            if t is None:
                t = np.full(len(k["sizes"]), cm.DISABLED_COST if P["disableUnmodeledCells"] else np.inf)
            T[:, j] = t
        return T

    def hard(self, P, select=None):
        """Per-entry picks and regrets for keys passing `select(key_dict)`."""
        rows = []
        for k in self.keys:
            if select and not select(k):
                continue
            T = self.times(P, k)
            pick = np.argmin(T, axis=1)
            for i, j in enumerate(pick):
                e = k["entries"][i]
                rows.append({
                    "coll": k["coll"], "nNodes": k["nNodes"], "nRanks": k["nRanks"], "R": k["R"],
                    "nvls": k["nvls"], "size": int(k["sizes"][i]), "model": k["cells"][j],
                    "best": e["best"], "equivalent": e["equivalent"], "default": e["default_cell"],
                    "regret": float(k["regret"][i, j]), "tol": e["tol"],
                    "default_gap": e["default_gap"], "w": k["w"],
                })
        return rows

    def soft(self, P, tau):
        tot, wsum = 0.0, 0.0
        for k in self.keys:
            T = self.times(P, k)
            lt = np.log(np.maximum(T, 1e-9))
            lt -= lt.min(axis=1, keepdims=True)
            wts = np.exp(-lt / tau)
            wts /= wts.sum(axis=1, keepdims=True)
            tot += k["w"] * float(np.sum(wts * k["regret"]))
            wsum += k["w"] * len(k["sizes"])
        return tot / max(wsum, 1e-9)


def summarize_picks(rows):
    if not rows:
        return {"n": 0}
    reg = np.array([r["regret"] for r in rows])
    return {
        "n": len(rows),
        "top1": float(np.mean(reg == 0.0)),
        "mean_regret": float(np.mean(reg)),
        "max_regret": float(np.max(reg)),
        "violations": int(np.sum(reg > 0.0)),
    }


# ---------------------------------------------------------------------------
# Full fit
# ---------------------------------------------------------------------------
def run_fit(seed, groups, oracle, cfg, log=print):
    """Stages 1-4 on `groups`; returns (params, info)."""
    info = {}
    P = copy_params(seed)
    anomalies = stage1(groups, cfg)
    info["anomalies"] = anomalies
    lat_keys = latency_keys(P, groups)
    bw_keys = stage2_bandwidth(P, groups)
    stage2_latency(P, groups, lat_keys)
    if "patBwEff" in bw_keys and "patLatPerRank" not in lat_keys:
        lat_keys.append("patLatPerRank")
    tol = max([v["tol"] for v in oracle.get("noise", {}).values()] + [cfg.min_tol])
    reg = regbuff_keys(groups, P, tol, cfg.fit_regbuff)
    for k, v in reg.items():
        set_param(P, k, v)
    free = lat_keys + [k for k in bw_keys if k not in lat_keys] + list(reg)
    if "nvlsEfficiency" not in free and any(g.algo == ALGO_NVLS_TREE for g in groups):
        free.append("nvlsEfficiency")
    info["free"] = free
    info["loss_stage2"] = log_loss(P, groups)

    # Stage 3: joint log-time fit.
    theta0 = np.array([get_param(P, k) for k in free])
    lo = np.array([bounds_for(k)[0] for k in free])
    hi = np.array([bounds_for(k)[1] for k in free])
    theta0 = np.clip(theta0, lo + 1e-12, hi - 1e-12)
    sc = np.maximum(np.abs(theta0), 1e-3)
    npts = sum(int(g.mask.sum()) for g in groups)
    rw = math.sqrt(cfg.reg_weight)

    def res3(th):
        Pth = with_theta(P, free, th)
        return np.concatenate([log_residuals(Pth, groups), rw * (th - theta0) / sc])

    r3 = least_squares(res3, theta0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1,
                       x_scale=sc, max_nfev=200 * max(len(free), 1))
    P3 = with_theta(P, free, r3.x)
    loss3 = log_loss(P3, groups)
    info["loss_stage3"] = loss3
    info["points"] = npts
    log(f"  stage 3: {len(free)} free params, {npts} points, log-loss {info['loss_stage2']:.4f} -> {loss3:.4f}")

    # Stage 4: regret polish.
    weights = {n: (cfg.gate_weight if cfg.gate_nodes and n in cfg.gate_nodes else 1.0)
               for n in {g.nNodes for g in groups}}
    prob = RegretProblem(oracle, cfg.profile, weights)
    nodes_fit = {g.nNodes for g in groups}
    prob.keys = [k for k in prob.keys if k["nNodes"] in nodes_fit]
    hard3 = summarize_picks(prob.hard(P3))
    info["regret_stage3"] = hard3
    Pf = P3
    if cfg.polish and prob.keys and hard3.get("violations", 0) > 0:
        base = r3.x.copy()

        def obj(u):
            th = np.clip(base * np.exp(np.clip(u, -2.0, 2.0)), lo, hi)
            Pth = with_theta(P, free, th)
            pen = max(0.0, log_loss(Pth, groups) / max(loss3, 1e-12) - (1.0 + cfg.loss_slack))
            return prob.soft(Pth, cfg.temperature) + 10.0 * pen

        r4 = minimize(obj, np.zeros(len(free)), method="Nelder-Mead",
                      options={"maxiter": cfg.polish_maxiter, "xatol": 1e-3, "fatol": 1e-6,
                               "adaptive": True})
        th4 = np.clip(base * np.exp(np.clip(r4.x, -2.0, 2.0)), lo, hi)
        P4 = with_theta(P, free, th4)
        hard4 = summarize_picks(prob.hard(P4))
        loss4 = log_loss(P4, groups)
        info["regret_stage4"] = hard4
        info["loss_stage4"] = loss4
        better = (hard4["mean_regret"] < hard3["mean_regret"]
                  and loss4 <= loss3 * (1.0 + cfg.loss_slack) + 1e-12)
        log(f"  stage 4: mean regret {hard3['mean_regret']:.4f} -> {hard4['mean_regret']:.4f}, "
            f"violations {hard3['violations']} -> {hard4['violations']}, log-loss {loss4:.4f}"
            + ("" if better else " (rejected)"))
        if better:
            Pf = P4
    info["final_regret"] = summarize_picks(prob.hard(Pf))
    return Pf, info


def anomaly_set(anomalies):
    return {(a["coll"], a["nNodes"], a["nRanks"], a["R"], a["cell"], a["size"]) for a in anomalies}


def gate(P, oracle, cfg, anomalies):
    """Acceptance at the nightly scale: every (key, size) pick within the noise
    band of the oracle, except violations explained by anomaly-tagged points
    (non-affine notches the model cannot represent)."""
    prob = RegretProblem(oracle, cfg.profile)
    nodes = set(cfg.gate_nodes or [])
    rows = prob.hard(P, lambda k: k["nNodes"] in nodes and k["nvls"] in cfg.gate_nvls)
    an = anomaly_set(anomalies)
    viol, exempt = [], []
    for r in rows:
        if r["regret"] <= 0.0:
            continue
        base = (r["coll"], r["nNodes"], r["nRanks"], r["R"])
        if base + (r["best"], r["size"]) in an or base + (r["model"], r["size"]) in an:
            exempt.append(r)
        else:
            viol.append(r)
    return {"nodes": sorted(nodes), "nvls": list(cfg.gate_nvls), "checked": len(rows),
            "violations": viol, "exempt": exempt, "passed": bool(rows) and not viol}


def default_baseline(oracle, cfg, nodes=None):
    """How NCCL's own choice (default run) does against the oracle."""
    rows = []
    for e in oracle["entries"]:
        if e["profile"] != cfg.profile or e["default_gap"] is None:
            continue
        if nodes and e["nNodes"] not in nodes:
            continue
        g = e["default_gap"]
        rows.append({"regret": 0.0 if g <= e["tol"] else g})
    return summarize_picks(rows)


def fit_all(con, seed, oracle, cfg, log=print):
    groups = load_groups(con, cfg.profile)
    if not groups:
        raise RuntimeError(f"no forced-cell runs for profile {cfg.profile} in the store")
    nodes = sorted({g.nNodes for g in groups})
    if not cfg.gate_nodes:
        cfg.gate_nodes = [max(nodes)]
    log(f"data: {len(groups)} forced-cell curves at nNodes={nodes}")

    seed_prob = RegretProblem(oracle, cfg.profile)
    result = {
        "profile": cfg.profile, "nodes": nodes, "gate_nodes": cfg.gate_nodes,
        "seed_regret": summarize_picks(seed_prob.hard(seed)),
        "nccl_default": default_baseline(oracle, cfg),
        "nccl_default_gate": default_baseline(oracle, cfg, set(cfg.gate_nodes)),
        "cv": [],
    }

    # Stage 5: cross-scale validation.
    if len(nodes) >= 2:
        for hold in nodes:
            train = [g for g in groups if g.nNodes != hold]
            log(f"CV: train nNodes={[n for n in nodes if n != hold]} -> predict {hold}")
            Pcv, _ = run_fit(seed, train, oracle, cfg, log=lambda *_: None)
            prob = RegretProblem(oracle, cfg.profile)
            rows = prob.hard(Pcv, lambda k, h=hold: k["nNodes"] == h)
            s = summarize_picks(rows)
            s["held_out"] = hold
            result["cv"].append(s)
            log(f"  held-out {hold}: top1 {s.get('top1', 0):.3f}, mean regret {s.get('mean_regret', 0):.4f}, "
                f"violations {s.get('violations', 0)}/{s.get('n', 0)}")

    # Stage 6: final fit on everything + gate.
    log("final fit on all scales")
    Pf, info = run_fit(seed, groups, oracle, cfg, log=log)
    g = gate(Pf, oracle, cfg, info["anomalies"])
    log(f"gate at nNodes={g['nodes']} nvls={g['nvls']}: {len(g['violations'])} violations, "
        f"{len(g['exempt'])} anomaly-exempt, {g['checked']} checked -> {'PASS' if g['passed'] else 'FAIL'}")
    prob = RegretProblem(oracle, cfg.profile)
    result.update({
        "info": {k: v for k, v in info.items() if k != "anomalies"},
        "anomalies": info["anomalies"], "gate": g, "validated": g["passed"],
        "picks": prob.hard(Pf),
        "free": [{"key": k, "label": key_label(k), "seed": get_param(seed, k), "fit": get_param(Pf, k)}
                 for k in info["free"]],
    })
    return Pf, result
