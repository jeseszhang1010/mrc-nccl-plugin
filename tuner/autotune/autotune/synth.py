"""Synthetic sweep data for tests and pipeline dry runs.

Measurements are generated from a known "truth" parameter set with the same
cost model the plugin uses, plus multiplicative noise and an optional notch
(non-affine dip) in one cell, so that the full pipeline (oracle -> fit ->
gate) can be exercised without cluster time.  The default runs pick cells
with a deliberately different "NCCL" parameter set, mimicking NCCL's
sub-optimal choices.
"""

import numpy as np

from . import costmodel as cm
from . import db as dbm
from . import topology
from .constants import cell_name, parse_cell
from .params import copy_params, set_param
from .sweep import candidate_cells

SIZES = [4096 << i for i in range(21)]  # 4K .. 4G


def perturb(params, seed=0, spread=0.35):
    """A plausible 'truth' that differs from `params` (the seed)."""
    rng = np.random.default_rng(seed)
    P = copy_params(params)
    for a in range(7):
        for p in range(3):
            P["baseLatencies"][a][p] *= float(np.exp(rng.normal(0, spread)))
            for h in range(3):
                P["hwLatencies"][h][a][p] *= float(np.exp(rng.normal(0, spread)))
    for k in ("bwInter", "patBwEff", "nvlsEfficiency"):
        set_param(P, k, P[k] * float(np.exp(rng.normal(0, spread / 2))))
    for k in ("llMaxBw", "perChMaxRingLL128Bw", "perChMaxTreeBw", "perChMaxTreeLL128Bw",
              "perChMaxNVLSTreeBw"):
        P[k] = P[k] * np.exp(rng.normal(0, spread, size=3))
    # Make the inter-node caps actually bind somewhere (as on real fabrics).
    P["perChMaxTreeBw"][2] = min(P["perChMaxTreeBw"][2], 0.85 * P["bwInter"])
    P["perChMaxRingLL128Bw"][2] = min(P["perChMaxRingLL128Bw"][2], 0.6 * P["bwInter"])
    P["perChMaxTreeLL128Bw"][2] = min(P["perChMaxTreeLL128Bw"][2], 0.5 * P["bwInter"])
    P["llMaxBw"][2] = min(P["llMaxBw"][2], 0.25 * P["nChannels"] * P["bwInter"])
    return P


def _busbw_factor(coll, n):
    return 2.0 * (n - 1) / n if coll == "allreduce" else (n - 1) / n


def _cycle(coll, n_ranks, sizes, times):
    rows = []
    f = _busbw_factor(coll, n_ranks)
    for s, t in zip(sizes, times):
        algbw = s / (t * 1e3)
        rows.append({"size": int(s), "count": int(s // 4), "time_oop": float(t), "algbw_oop": algbw,
                     "busbw_oop": algbw * f, "wrong_oop": "0", "time_ip": float(t),
                     "algbw_ip": algbw, "busbw_ip": algbw * f, "wrong_ip": "0"})
    return rows


def generate(con, truth, nccl_params, scales=(8, 32, 128), colls=("allreduce", "allgather", "reducescatter"),
             splits=("0x1", "0x3"), Rs=(0, 1), ppr=4, noise=0.01, seed=1, notch=None,
             reg_scale=None, profile="rail_nop2p"):
    """Write probe/default/forced runs into the store.  `notch` =
    (nNodes, coll, split, R, cell, size, factor) slows one point down.
    `reg_scale` = {cell: factor} applies to regBuff=1 bandwidth in the truth."""
    rng = np.random.default_rng(seed)
    T = copy_params(truth)
    if reg_scale:
        for cell, fac in reg_scale.items():
            a, p = parse_cell(cell)
            T["regBwScale"][a][p] = fac
    sizes = np.array(SIZES, dtype=float)
    n = 0
    for nodes in scales:
        for coll in colls:
            for split in splits:
                env = {"NCCL_TESTS_SPLIT_MASK": split, "NCCL_MNNVL_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}
                topo = topology.comm_topology(nodes, ppr, env)
                nr, nn = topo["nRanks"], topo["nNodes"]
                for R in Rs:
                    rb = topology.regbuff_for_R(R)
                    for nvls in ((0, 1) if coll == "allreduce" and topo["ppn"] >= 2 else (0,)):
                        cells = candidate_cells(coll, topo["ppn"], nvls)
                        if nvls:
                            cells = [c for c in cells if c.startswith("nvls")]
                        all_cells = candidate_cells(coll, topo["ppn"], nvls)
                        meas = {}
                        for c in all_cells:
                            a, p = parse_cell(c)
                            t = cm.cell_times(T, coll, a, p, nr, nn, sizes, rb)
                            if t is None:
                                continue
                            t = t * np.exp(rng.normal(0, noise, size=len(sizes)))
                            if notch and notch[:5] == (nodes, coll, split, R, c):
                                t[SIZES.index(notch[5])] *= notch[6]
                            meas[c] = t
                        base = {"source": "synthetic", "coll": coll, "phys_nodes": nodes, "ppr": ppr,
                                "split": split, "R": R, "regbuff": rb, "p2p_disable": 1, "nvls": nvls,
                                "mnnvl": 0, "profile": profile, "nNodes": nn, "nRanks": nr,
                                "ppn": topo["ppn"], "tuner": "none", "cycles_expected": 3,
                                "cycles_found": 3, "status": "ok", "note": "", "rc": 0,
                                "env": {**env, "NCCL_NVLS_ENABLE": "2" if nvls else "0"}, "cmd": "",
                                "probe_cells": [], "nccl_table": [], "log_lines": [],
                                "nvls_available": 1 if nvls else None, "forced_algo": None,
                                "forced_proto": None}
                        tag = f"{nodes}n-{coll}-m{split}-R{R}-nvls{nvls}"
                        for c in cells:
                            if c not in meas:
                                continue
                            rows = _cycle(coll, nr, sizes, meas[c])
                            dbm.store_run(con, {**base, "run_id": f"{tag}-forced-{c.replace('/', '-')}",
                                                "kind": "forced", "case_name": tag, "cell": c,
                                                "cycles": [rows, rows, rows]})
                            n += 1
                        # NCCL's default: argmin of a different parameter set over valid cells.
                        avail = [parse_cell(c) for c in all_cells if c in meas]
                        dflt, probe = [], []
                        for i, s in enumerate(SIZES):
                            pick, _ = cm.choose(nccl_params, coll, avail, nr, nn, s, rb)
                            cname = cell_name(*pick)
                            dflt.append(meas[cname][i] * float(np.exp(rng.normal(0, noise))))
                            probe.append({"coll": coll, "size": s, "cell": cname, "ch_lo": 0, "ch_hi": 31})
                        rows = _cycle(coll, nr, sizes, dflt)
                        dbm.store_run(con, {**base, "run_id": f"{tag}-default", "kind": "default",
                                            "case_name": tag, "cell": "default", "cycles": [rows, rows, rows]})
                        dbm.store_run(con, {**base, "run_id": f"{tag}-probe", "kind": "probe",
                                            "case_name": tag, "cell": "default", "cycles": [],
                                            "probe_cells": probe, "status": "ok"})
                        n += 2
    return n
