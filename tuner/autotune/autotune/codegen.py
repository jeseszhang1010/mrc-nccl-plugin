"""Write fitted parameters into plugin.c, rebuild, and check C/Python parity."""

import datetime
import json
import os
import subprocess

from . import costmodel as cm
from . import params as pm
from .constants import ALGO_NAMES, PROTO_NAMES


def tuner_dir_of(plugin_path):
    return os.path.dirname(os.path.abspath(plugin_path))


def write(plugin_path, params, profile, validated, provenance):
    with open(plugin_path) as f:
        text = f.read()
    old = pm.parse_params(text, profile)
    block = pm.render_block(params, profile, validated, provenance)
    new_text = pm.replace_block(text, profile, block)
    back = pm.parse_params(new_text, profile)
    a, b = pm.params_to_values(back), pm.params_to_values(params)
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if abs(float(x) - float(y)) > 1e-5 * max(1.0, abs(float(y)))]
    if bad:
        raise RuntimeError(f"rendered block does not round-trip (value indices {bad[:5]})")
    with open(plugin_path, "w") as f:
        f.write(new_text)
    changes = []
    va, vb = pm.params_to_values(old), pm.params_to_values(back)
    names = _flat_names()
    for n, x, y in zip(names, va, vb):
        if abs(float(x) - float(y)) > 1e-9 * max(1.0, abs(float(x))):
            changes.append((n, float(x), float(y)))
    return changes


def _flat_names():
    names = []
    for name, shape, _ in pm.FIELDS:
        if shape == ():
            names.append(name)
        elif len(shape) == 1:
            names += [f"{name}[{i}]" for i in range(shape[0])]
        elif len(shape) == 2:
            names += [f"{name}[{i}][{j}]" for i in range(shape[0]) for j in range(shape[1])]
        else:
            names += [f"{name}[{h}][{i}][{j}]" for h in range(shape[0])
                      for i in range(shape[1]) for j in range(shape[2])]
    return [pm.key_label(n) for n in names]


def build(plugin_path):
    """Rebuild the tuner library and the host test drivers."""
    tdir = tuner_dir_of(plugin_path)
    root = os.path.dirname(tdir)
    if os.path.exists(os.path.join(root, "Makefile")) and os.path.basename(tdir) == "tuner":
        cmds = [["make", "-C", root, "tuner"], ["make", "-C", root, "test-tuner"]]
    else:
        cmds = [["make", "-C", tdir, "all"]]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(c)} failed:\n{r.stdout}\n{r.stderr}")
    return os.path.join(tdir, "test", "cost_model_main")


def run_driver(driver, profile, topos, colls=("allreduce", "allgather", "reducescatter"),
               regbuff=0, max_bytes=None):
    """Run cost_model_main --all-cells; returns {(coll,nRanks,nNodes,regBuff,nBytes): {cell: time}}
    and the winners."""
    args = [driver, "--profile", profile, "--all-cells", "--regbuff", str(regbuff)]
    for nr, nn in topos:
        args += ["--topo", f"{nr}:{nn}"]
    for c in colls:
        args += ["--coll", c]
    if max_bytes:
        args += ["--max-bytes", str(max_bytes)]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    cells, winners = {}, {}
    for line in out.splitlines():
        kind, coll, nr, nn, rb, nb, a, p, t = line.split(",")
        key = (coll, int(nr), int(nn), int(rb), int(nb))
        if kind == "cell":
            cells.setdefault(key, {})[f"{a}/{p}"] = float(t)
        elif kind == "winner":
            winners[key] = f"{a}/{p}"
    return cells, winners


def parity(driver, plugin_path, profile, topos, colls=("allreduce", "allgather", "reducescatter"), rtol=1e-4):
    """Compare every priced cell of the compiled driver with costmodel.py."""
    params = pm.load_params(plugin_path, profile)
    worst, n, bad = 0.0, 0, []
    for rb in (0, 1):
        cells, _ = run_driver(driver, profile, topos, colls, rb)
        for (coll, nr, nn, r, nb), d in cells.items():
            for cell, t in d.items():
                a, p = cell.split("/")
                py = cm.cell_time(params, coll, ALGO_NAMES.index(a), PROTO_NAMES.index(p), nr, nn, nb, r)
                if py is None:
                    py = cm.DISABLED_COST
                e = abs(py - t) / max(abs(t), 1e-9)
                worst = max(worst, e)
                n += 1
                if e > rtol:
                    bad.append((coll, nr, nn, r, nb, cell, t, py))
    return {"cells": n, "worst_rel_err": worst, "mismatches": bad[:20], "ok": not bad and n > 0}


def load_fit(path):
    with open(path) as f:
        res = json.load(f)
    params = pm.params_from_values(res["values"])
    return res, params


def provenance(res, fit_path):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    g = res.get("gate", {})
    fr = res.get("info", {}).get("final_regret", {})
    return [
        f"Source: {os.path.basename(fit_path)} ({now}); data nNodes={res.get('nodes')}",
        f"Fit: final top1={fr.get('top1', 0):.3f} mean_regret={fr.get('mean_regret', 0):.4f} "
        f"violations={fr.get('violations', 0)}/{fr.get('n', 0)}",
        f"Gate nNodes={g.get('nodes')} nvls={g.get('nvls')}: "
        f"{'PASS' if g.get('passed') else 'FAIL'} ({len(g.get('violations', []))} violations, "
        f"{len(g.get('exempt', []))} anomaly-exempt)",
    ]
