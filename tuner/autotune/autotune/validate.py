"""Validation: offline (compiled C model vs oracle) and online A/B."""

import csv
from collections import defaultdict

import numpy as np

from . import codegen
from . import db as dbm
from . import dips as dipmod
from . import oracle as orc


def offline(oracle, driver, profile, gate_nodes, gate_nvls=(0,)):
    """For every oracle (key, size), let the compiled cost model pick among the
    cells that were measured available (NCCL marks the rest IGNORE) and score
    the pick against the oracle."""
    idx = orc.index(oracle)
    by_rb = defaultdict(set)
    for key in idx:
        if key[0] == profile:
            by_rb[1 if key[4] >= 1 else 0].add((key[3], key[2]))
    cells = {}
    for rb, topos in by_rb.items():
        c, _ = codegen.run_driver(driver, profile, sorted(topos), regbuff=rb)
        cells.update(c)
    rows = []
    for key, ents in idx.items():
        if key[0] != profile:
            continue
        _, coll, nn, nr, R, nvls = key
        rb = 1 if R >= 1 else 0
        for s, e in sorted(ents.items()):
            table = cells.get((coll, nr, nn, rb, s), {})
            avail = {c: table[c] for c in e["cells"] if c in table}
            pick = min(avail, key=avail.get) if avail else None
            reg = orc.regret(e, pick) if pick else 1.0
            rows.append({"coll": coll, "nNodes": nn, "nRanks": nr, "R": R, "nvls": nvls, "size": s,
                         "default": e["default_cell"], "default_gap": e["default_gap"],
                         "oracle": e["best"], "equivalent": " ".join(e["equivalent"]),
                         "model": pick, "regret": reg, "tol": e["tol"],
                         "gate": nn in gate_nodes and nvls in gate_nvls})
    g = [r for r in rows if r["gate"]]
    viol = [r for r in g if r["regret"] > 0]
    return {"rows": rows, "checked": len(g), "violations": viol, "passed": bool(g) and not viol}


def write_rows_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _median_curve(con, run_ids):
    acc = defaultdict(list)
    for rid in run_ids:
        for p in dbm.last_cycle(con, rid):
            acc[p["size"]].append(p["busbw_oop"])
    return {s: float(np.median(v)) for s, v in acc.items()}


def compare_ab(con, oracle=None, max_regression=0.03, notch=dipmod.NOTCH):
    """Gates per configuration: (1) B within the oracle noise band at every
    size, (2) B no more than `max_regression` below A at any size, (3) every A
    dip resolved in B, (4) the cells NCCL used under the tuner (B probe TUNING
    lines) are oracle-equivalent."""
    idx = orc.index(oracle) if oracle else {}
    groups = defaultdict(lambda: {"ab_a": [], "ab_b": [], "ab_probe": []})
    for run in dbm.runs(con, "status='ok' AND kind IN ('ab_a','ab_b','ab_probe')"):
        groups[orc.key_of(run)][run["kind"]].append(run["run_id"])
    report = []
    for key, g in sorted(groups.items(), key=lambda kv: orc.key_str(kv[0])):
        A, B = _median_curve(con, g["ab_a"]), _median_curve(con, g["ab_b"])
        probe = {}
        for rid in g["ab_probe"]:
            probe.update(dbm.probe_cells(con, rid))
        ents = idx.get(key, {})
        rows = []
        sizes = sorted(set(A) | set(B))
        for i, s in enumerate(sizes):
            a, b = A.get(s), B.get(s)
            e = ents.get(s)
            row = {"size": s, "A": a, "B": b, "B_vs_A": (b / a - 1) if a and b else None,
                   "oracle": e["best_busbw"] if e else None,
                   "B_gap": (1 - b / e["best_busbw"]) if e and b else None,
                   "tol": e["tol"] if e else None, "B_cell": probe.get(s),
                   "B_cell_ok": (probe.get(s) in e["equivalent"]) if e and probe.get(s) else None}
            prev, nxt = (sizes[i - 1] if i else None), (sizes[i + 1] if i + 1 < len(sizes) else None)

            def is_dip(curve):
                if s < dipmod.MIN_NOTCH_SIZE or prev is None or nxt is None:
                    return False
                if curve.get(s) is None or curve.get(prev) is None or curve.get(nxt) is None:
                    return False
                return curve[s] < (1 - notch) * min(curve[prev], curve[nxt])

            row["A_dip"], row["B_dip"] = is_dip(A), is_dip(B)
            rows.append(row)
        fails = {
            "oracle": [r["size"] for r in rows if r["B_gap"] is not None and r["B_gap"] > r["tol"]],
            "regression": [r["size"] for r in rows if r["B_vs_A"] is not None and r["B_vs_A"] < -max_regression],
            "dips": [r["size"] for r in rows if r["A_dip"] and r["B_dip"]],
            "cells": [r["size"] for r in rows if r["B_cell_ok"] is False],
        }
        report.append({"key": orc.key_dict(key), "key_str": orc.key_str(key), "rows": rows, "fails": fails,
                       "passed": not any(fails.values()) and bool(A) and bool(B),
                       "mean_gain": float(np.mean([r["B_vs_A"] for r in rows if r["B_vs_A"] is not None]))
                       if rows else None})
    return report


def ab_markdown(report):
    lines = ["| configuration | A/B | oracle fails | regressions | unresolved dips | wrong cells | mean B/A | result |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report:
        f = r["fails"]
        gain = "" if r["mean_gain"] is None else "%+.1f%%" % (100 * r["mean_gain"])
        cols = [r["key_str"], "%d sizes" % len(r["rows"]), f["oracle"] or "-", f["regression"] or "-",
                f["dips"] or "-", f["cells"] or "-", gain, "PASS" if r["passed"] else "FAIL"]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    return "\n".join(lines)
