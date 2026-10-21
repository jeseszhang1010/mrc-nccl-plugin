"""Dip report for nightly curves.

Only the collectives the tuner cost model selects for (AllReduce, AllGather,
ReduceScatter) are reported; alltoall and sendrecv are out of scope.

Signals per (run, size), out-of-place busbw of the selected (last) cycle:
  * notch    : busbw(s) < (1 - notch) * min(busbw(s-1), busbw(s+1)) for
               s >= 1 MiB, i.e. lower than both neighbours (a plateau before
               a jump is not a notch; the envelope signal covers it);
  * envelope : busbw(s) below the oracle best for the same configuration by
               more than the oracle noise band (needs a sweep in the store).
"""

import csv
import os

from . import db as dbm
from . import oracle as orc
from .constants import COLLS

NOTCH = 0.10
MIN_NOTCH_SIZE = 1 << 20


def find_dips(con, oracle=None, kinds=("nightly",), notch=NOTCH, colls=tuple(COLLS)):
    idx = orc.index(oracle) if oracle else {}
    rows = []
    where = "status='ok' AND kind IN (%s)" % ",".join("?" * len(kinds))
    for run in dbm.runs(con, where, tuple(kinds)):
        if run["coll"] not in colls:
            continue
        pts = dbm.last_cycle(con, run["run_id"])
        if not pts:
            continue
        ent = idx.get(orc.key_of(run), {}) if run["nNodes"] is not None else {}
        for i, p in enumerate(pts):
            bw = p["busbw_oop"]
            neigh = None
            if 0 < i < len(pts) - 1:
                neigh = min(pts[i - 1]["busbw_oop"], pts[i + 1]["busbw_oop"])
            drop = (1 - bw / neigh) if neigh and p["size"] >= MIN_NOTCH_SIZE else None
            e = ent.get(p["size"])
            gap = (1 - bw / e["best_busbw"]) if e and e["best_busbw"] > 0 else None
            is_notch = drop is not None and drop > notch
            is_env = gap is not None and gap > e["tol"]
            if not (is_notch or is_env):
                continue
            rows.append({
                "run_id": run["run_id"], "case": run["case_name"], "coll": run["coll"],
                "profile": run["profile"], "nNodes": run["nNodes"], "nRanks": run["nRanks"],
                "R": run["R"], "nvls": run["nvls"], "size": p["size"], "busbw": bw,
                "neighbor_min": neigh, "notch_drop": drop,
                "oracle_best": e["best"] if e else None,
                "oracle_busbw": e["best_busbw"] if e else None, "oracle_gap": gap,
                "lost_gbps": max((neigh - bw) if is_notch else 0.0,
                                 (e["best_busbw"] - bw) if is_env else 0.0),
            })
    rows.sort(key=lambda r: -r["lost_gbps"])
    return rows


def write_csv(rows, path):
    cols = ["run_id", "case", "coll", "profile", "nNodes", "nRanks", "R", "nvls", "size", "busbw",
            "neighbor_min", "notch_drop", "oracle_best", "oracle_busbw", "oracle_gap", "lost_gbps"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_runs(con, rows, out_dir, oracle=None):
    """One PNG per run with dips: busbw curve, dip markers and the oracle best."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    idx = orc.index(oracle) if oracle else {}
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for rid in sorted({r["run_id"] for r in rows}):
        run = dbm.runs(con, "run_id=?", (rid,))[0]
        pts = dbm.last_cycle(con, rid)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogx([p["size"] for p in pts], [p["busbw_oop"] for p in pts], "o-", label="nightly", base=2)
        ent = idx.get(orc.key_of(run), {}) if run["nNodes"] is not None else {}
        if ent:
            ss = sorted(ent)
            ax.plot(ss, [ent[s]["best_busbw"] for s in ss], "--", label="oracle best")
        d = [r for r in rows if r["run_id"] == rid]
        ax.plot([r["size"] for r in d], [r["busbw"] for r in d], "rx", ms=10, label="dip")
        ax.set_xlabel("bytes")
        ax.set_ylabel("busbw (GB/s)")
        ax.set_title(f"{run['case_name']} ({run['coll']}, nNodes={run['nNodes']}, nRanks={run['nRanks']}, R={run['R']})",
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(out_dir, f"{rid}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def markdown(rows, top=30):
    lines = ["| run | coll | nNodes | nRanks | R | size | busbw | neighbours | notch | oracle | gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows[:top]:
        lines.append("| {case} | {coll} | {nNodes} | {nRanks} | {R} | {size} | {busbw:.2f} | {nb} | {nd} | {ob} | {og} |".format(
            nb="" if r["neighbor_min"] is None else f"{r['neighbor_min']:.2f}",
            nd="" if r["notch_drop"] is None else f"{100 * r['notch_drop']:.0f}%",
            ob="" if r["oracle_best"] is None else f"{r['oracle_best']} {r['oracle_busbw']:.2f}",
            og="" if r["oracle_gap"] is None else f"{100 * r['oracle_gap']:.0f}%", **r))
    return "\n".join(lines)
