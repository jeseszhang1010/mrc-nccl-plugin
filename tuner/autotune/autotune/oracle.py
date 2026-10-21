"""Oracle: the best measured (algo, proto) per (configuration, size).

A configuration ("key") is (profile, coll, nNodes, nRanks, R, nvls).  The
oracle is built from forced-cell runs (last cycle, out-of-place busbw).  Any
cell within the noise band `tol` of the best is an equivalent winner.

Noise is estimated from duplicates that the sweep already contains:
  * the default run vs the forced run of the cell NCCL chose by default (from
    the probe), per size;
  * allreduce R=0 vs R=1 runs of the same forced cell.
sigma = 1.4826 * median(|d|) / sqrt(2) on relative differences d, and
tol = max(min_tol, 2 * sigma), per nNodes.
"""

import json
import math
from collections import defaultdict

import numpy as np

from . import db as dbm

MIN_TOL = 0.03
SWEEP_KINDS = ("forced", "default", "probe")


def key_of(run):
    return (run["profile"], run["coll"], run["nNodes"], run["nRanks"], run["R"], run["nvls"])


def key_dict(key):
    return dict(zip(("profile", "coll", "nNodes", "nRanks", "R", "nvls"), key))


def key_str(key):
    p, c, nn, nr, r, nv = key
    return f"{p}/{c}/N{nn}/R{nr}/reg{r}/nvls{nv}"


def _curve(con, run_id):
    return {p["size"]: (p["busbw_oop"], p["time_oop"]) for p in dbm.last_cycle(con, run_id)}


def collect(con, profile=None, kinds=SWEEP_KINDS):
    """Group usable sweep runs by key: {key: {"forced": {cell: curve},
    "default": curve, "probe": {size: cell}}}; curve = {size: (busbw, time)}."""
    where = "status='ok' AND kind IN (%s)" % ",".join("?" * len(kinds))
    args = list(kinds)
    if profile:
        where += " AND profile=?"
        args.append(profile)
    groups = defaultdict(lambda: {"forced": {}, "default": None, "probe": {}, "runs": []})
    for run in dbm.runs(con, where, tuple(args)):
        if run["nNodes"] is None:
            continue
        g = groups[key_of(run)]
        g["runs"].append(run["run_id"])
        if run["kind"] == "forced" and run["cell"] not in (None, "default"):
            g["forced"][run["cell"]] = _curve(con, run["run_id"])
        elif run["kind"] == "default":
            g["default"] = _curve(con, run["run_id"])
        elif run["kind"] == "probe":
            cells = dbm.probe_cells(con, run["run_id"])
            g["probe"].update(cells)
    # NVLS-on keys inherit the cells measured with NVLS off for the same
    # communicator (only nvls_tree is re-measured with NVLS on).
    for key, g in list(groups.items()):
        if key[5] == 1:
            off = groups.get(key[:5] + (0,))
            if off:
                for cell, curve in off["forced"].items():
                    g["forced"].setdefault(cell, curve)
    return dict(groups)


def _rel_diffs(a, b):
    out = []
    for s in set(a) & set(b):
        x, y = a[s][0], b[s][0]
        if x > 0 and y > 0:
            out.append(abs(x - y) / max(x, y))
    return out


def estimate_noise(groups, min_tol=MIN_TOL):
    by_nodes = defaultdict(list)
    for key, g in groups.items():
        if g["default"] and g["probe"]:
            for s, (bw, _t) in g["default"].items():
                cell = g["probe"].get(s)
                cur = g["forced"].get(cell) if cell else None
                if cur and s in cur and bw > 0 and cur[s][0] > 0:
                    by_nodes[key[2]].append(abs(bw - cur[s][0]) / max(bw, cur[s][0]))
        if key[1] == "allreduce" and key[4] == 0:
            other = groups.get(key[:4] + (1, key[5]))
            if other:
                for cell, curve in g["forced"].items():
                    if cell in other["forced"]:
                        by_nodes[key[2]].extend(_rel_diffs(curve, other["forced"][cell]))
    noise = {}
    for n, d in by_nodes.items():
        sigma = 1.4826 * float(np.median(d)) / math.sqrt(2) if d else 0.0
        noise[n] = {"sigma": sigma, "tol": max(min_tol, 2 * sigma), "pairs": len(d)}
    return noise


def build(con, profile=None, min_tol=MIN_TOL):
    groups = collect(con, profile)
    noise = estimate_noise(groups, min_tol)
    entries = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        if not g["forced"]:
            continue
        tol = noise.get(key[2], {"tol": min_tol})["tol"]
        sizes = sorted({s for c in g["forced"].values() for s in c})
        for s in sizes:
            cells = {c: cur[s][0] for c, cur in g["forced"].items() if s in cur}
            times = {c: cur[s][1] for c, cur in g["forced"].items() if s in cur}
            if not cells:
                continue
            best = max(cells, key=cells.get)
            bbw = cells[best]
            equiv = sorted(c for c, v in cells.items() if v >= bbw * (1 - tol))
            dcell = g["probe"].get(s)
            dbw = g["default"][s][0] if g["default"] and s in g["default"] else None
            entries.append({
                **key_dict(key), "size": s, "cells": cells, "times": times,
                "best": best, "best_busbw": bbw, "equivalent": equiv, "tol": tol,
                "default_cell": dcell, "default_busbw": dbw,
                "default_gap": (1 - dbw / bbw) if dbw and bbw > 0 else None,
            })
    return {"profile": profile, "min_tol": min_tol,
            "noise": {str(k): v for k, v in noise.items()}, "entries": entries}


def regret(entry, cell):
    """Busbw lost by picking `cell` instead of the oracle (0 for equivalent
    winners; 1.0 for a cell that was not measured/available)."""
    if cell in entry["equivalent"]:
        return 0.0
    v = entry["cells"].get(cell)
    if v is None:
        return 1.0
    return max(0.0, (entry["best_busbw"] - v) / entry["best_busbw"])


def index(oracle):
    idx = defaultdict(dict)
    for e in oracle["entries"]:
        idx[(e["profile"], e["coll"], e["nNodes"], e["nRanks"], e["R"], e["nvls"])][e["size"]] = e
    return dict(idx)


def save(oracle, path):
    with open(path, "w") as f:
        json.dump(oracle, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)


def write_csv(oracle, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "coll", "nNodes", "nRanks", "R", "nvls", "size", "best",
                    "best_busbw", "equivalent", "tol", "default_cell", "default_busbw", "default_gap"])
        for e in oracle["entries"]:
            w.writerow([e["profile"], e["coll"], e["nNodes"], e["nRanks"], e["R"], e["nvls"],
                        e["size"], e["best"], f"{e['best_busbw']:.3f}", " ".join(e["equivalent"]),
                        f"{e['tol']:.3f}", e["default_cell"] or "",
                        "" if e["default_busbw"] is None else f"{e['default_busbw']:.3f}",
                        "" if e["default_gap"] is None else f"{e['default_gap']:.4f}"])
