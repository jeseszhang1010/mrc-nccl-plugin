"""Generate the measurement sweep (probe / default / forced-cell runs) from a
nightly mpirun command used as a template, plus the A/B validation runs.

Every generated run reuses the template verbatim (launcher options, numa
binding wrapper, every -x setting) except for: -np, the hostfile, the
benchmark binary and its arguments, and the managed env vars below.
"""

import json
import os
import shlex
import stat

from . import topology
from .constants import COLL_BINARY, NCCL_ENV_ALGO, NCCL_ENV_PROTO, parse_cell
from .ingest import parse_mpirun

MANAGED_ENV = [
    "NCCL_TUNER_PLUGIN", "NCCL_TUNER_FORCE_COST_MODEL", "NCCL_TUNER_COST_MODEL_PROFILE",
    "NCCL_TUNER_CONFIG_FILE", "NCCL_ALGO", "NCCL_PROTO", "NCCL_NVLS_ENABLE",
    "NCCL_TESTS_SPLIT_MASK", "NCCL_TESTS_SPLIT", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    "NCCL_DEBUG_FILE",
]

AR_CELLS = ["ring/ll", "ring/ll128", "ring/simple", "tree/ll", "tree/ll128", "tree/simple"]
RING_CELLS = ["ring/ll", "ring/ll128", "ring/simple"]


def candidate_cells(coll, ppn, nvls_on):
    """Cells NCCL 2.30 can run for this communicator (CollNet excluded: no
    SHARP on MRC).  nvls is impossible for nNodes>1 without CollNet; PAT is
    AG/RS only with one rank per node; nvls_tree is AR only, needs >= 2 local
    ranks and NVLS enabled."""
    if coll == "allreduce":
        cells = list(AR_CELLS)
        if nvls_on and ppn >= 2:
            cells.append("nvls_tree/simple")
        return cells
    if coll in ("allgather", "reducescatter"):
        cells = list(RING_CELLS)
        if ppn == 1:
            cells.append("pat/simple")
        return cells
    raise ValueError(f"unsupported collective {coll}")


class Template:
    """A parsed nightly mpirun command."""

    def __init__(self, cmd):
        self.cmd = cmd
        self.p = parse_mpirun(cmd)
        if self.p["binary_index"] is None:
            raise ValueError("template command has no nccl-tests *_perf binary")
        if not self.p["np"]:
            raise ValueError("template command has no -np")
        if not self.p["ppr"]:
            raise ValueError("template command has no --map-by ppr:K:node")
        self.ppr = self.p["ppr"]
        self.env = dict(self.p["env"])
        self.bench = dict(self.p["bench"])
        self.hostfile = self.p["hostfile"]

    def build(self, *, nodes, hostfile, coll, env, bench_args):
        toks = list(self.p["tokens"])
        bi = self.p["binary_index"]
        launcher, binary = toks[:bi], toks[bi]
        out = []
        i = 0
        while i < len(launcher):
            t = launcher[i]
            if t in ("-np", "-n", "--np") and i + 1 < len(launcher):
                out += [t, str(nodes * self.ppr)]
                out += [x for k, v in env.items() if v is not None for x in ("-x", f"{k}={v}")]
                i += 2
                continue
            if t in ("--hostfile", "-hostfile", "--machinefile", "-machinefile") and i + 1 < len(launcher):
                out += [t, hostfile]
                i += 2
                continue
            if t == "-x" and i + 1 < len(launcher) and launcher[i + 1].split("=", 1)[0] in MANAGED_ENV:
                i += 2
                continue
            out.append(t)
            i += 1
        if self.hostfile is None:
            out[1:1] = ["--hostfile", hostfile]
        bdir = binary.rsplit("/", 1)[0] + "/" if "/" in binary else ""
        out.append(bdir + COLL_BINARY[coll])
        out += bench_args
        return shlex.join(out)


def bench_args(template, R, kind, cycles=3, smoke=False):
    b = template.bench
    if kind == "probe":
        args = {"-b": "4K", "-e": "4G", "-f": "2", "-g": b.get("ngpus", "1"), "-c": "0",
                "-R": str(R), "-w": "1", "-n": "1", "-N": "1"}
    else:
        args = {"-b": "4K", "-e": "4G", "-f": "2", "-g": b.get("ngpus", "1"),
                "-c": b.get("check", "1"), "-R": str(R), "-w": b.get("warmup_iters", "50"),
                "-n": b.get("iters", "50"), "-N": str(cycles)}
    if smoke:
        args.update({"-e": "1M", "-w": "1", "-n": "5", "-N": "1"})
    return [x for kv in args.items() for x in kv]


def _run_env(base_nvls, split, kind, cell=None, nvls=None, tuner="none", extra=None):
    env = {k: None for k in MANAGED_ENV}
    env["NCCL_TUNER_PLUGIN"] = tuner
    env["NCCL_TESTS_SPLIT_MASK"] = split
    env["NCCL_NVLS_ENABLE"] = str(nvls if nvls is not None else base_nvls)
    if kind == "probe":
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING"
    else:
        env["NCCL_DEBUG"] = "WARN"
    if cell:
        algo, proto = cell.split("/")
        env["NCCL_ALGO"] = NCCL_ENV_ALGO[algo]
        env["NCCL_PROTO"] = NCCL_ENV_PROTO[proto]
    if extra:
        env.update(extra)
    return {k: v for k, v in env.items() if v is not None}


def _rid(*parts):
    return "-".join(str(p).replace("/", "-") for p in parts if p not in (None, ""))


def plan_sweep(template, scales, colls, splits, Rs, nvls_variant=True, cycles=3, smoke_scales=()):
    """List of run dicts (not yet written)."""
    base_nvls = template.env.get("NCCL_NVLS_ENABLE", "2")
    runs = []
    phases = [(f"smoke-{n}n", n, True) for n in smoke_scales] + [(f"{n}n", n, False) for n in scales]
    for phase, nodes, smoke in phases:
        for coll in colls:
            for split in splits:
                topo = topology.comm_topology(nodes, template.ppr, {**template.env, "NCCL_TESTS_SPLIT_MASK": split,
                                                                    "NCCL_MNNVL_ENABLE": "0"})
                for R in Rs:
                    variants = [(int(base_nvls != "0"), base_nvls)]
                    if (nvls_variant and coll == "allreduce" and topo["ppn"] >= 2 and base_nvls == "0"):
                        variants.append((1, "2"))
                    for vi, (nvls_on, nvls_env) in enumerate(variants):
                        tag = f"nvls{nvls_env}"
                        cells = candidate_cells(coll, topo["ppn"], nvls_on)
                        if vi > 0:
                            cells = [c for c in cells if c.startswith("nvls")]
                        kinds = [("probe", None), ("default", None)] + [("forced", c) for c in cells]
                        for kind, cell in kinds:
                            rid = _rid(phase, coll, f"m{split}", f"R{R}", tag, kind, cell)
                            env = _run_env(base_nvls, split, kind, cell, nvls_env)
                            runs.append({
                                "run_id": rid, "phase": phase, "smoke": smoke,
                                "kind": "smoke" if smoke else kind, "role": kind,
                                "coll": coll, "phys_nodes": nodes, "ppr": template.ppr,
                                "np": nodes * template.ppr, "split": split, "R": R,
                                "nvls_env": nvls_env, "cell": cell or "default",
                                "nNodes": topo["nNodes"], "nRanks": topo["nRanks"], "ppn": topo["ppn"],
                                "env": env,
                                "bench": bench_args(template, R, kind, cycles, smoke),
                            })
    return runs


def _write_exec(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_runs(out_dir, template, runs, hostfile_pattern=None, hostfile_source=None, title="sweep"):
    """Write scripts/, logs/, hostfiles/, run_all.sh and manifest.json."""
    for d in ("scripts", "logs", "hostfiles"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    source = hostfile_source or (None if hostfile_pattern else template.hostfile)
    node_counts = sorted({r["phys_nodes"] for r in runs})

    def hostfile_for(n):
        if hostfile_pattern:
            return hostfile_pattern.format(nodes=n)
        return f"hostfiles/hostfile-{n}"

    if not hostfile_pattern:
        lines = ["#!/bin/bash", "# Create per-scale hostfiles: the first N unique hosts of the source.",
                 "set -euo pipefail", 'cd "$(dirname "$0")"',
                 f"SRC={shlex.quote(source or '')}",
                 '[ -n "$SRC" ] && [ -f "$SRC" ] || { echo "hostfile source $SRC not found" >&2; exit 1; }']
        for n in node_counts:
            lines.append(f"awk 'NF && !seen[$1]++' \"$SRC\" | head -n {n} > hostfile-{n}")
            lines.append(f'[ "$(wc -l < hostfile-{n})" -eq {n} ] || '
                         f'{{ echo "need {n} hosts in $SRC" >&2; exit 1; }}')
        _write_exec(os.path.join(out_dir, "hostfiles", "prepare_hostfiles.sh"), "\n".join(lines) + "\n")

    manifest = {"title": title, "template": template.cmd, "runs": []}
    for r in runs:
        cmd = template.build(nodes=r["phys_nodes"], hostfile=hostfile_for(r["phys_nodes"]),
                             coll=r["coll"], env=r["env"], bench_args=r["bench"])
        script = f"scripts/{r['run_id']}.sh"
        log = f"logs/{r['run_id']}.log"
        meta = {k: r[k] for k in ("run_id", "kind", "role", "phase", "coll", "phys_nodes", "ppr",
                                  "np", "split", "R", "cell")}
        meta["cmd"] = cmd
        body = "\n".join([
            "#!/bin/bash",
            f"# autotune {title} run {r['run_id']}",
            "set -o pipefail",
            'cd "$(dirname "$0")/.."',
            "{",
            f"  printf '%s\\n' {shlex.quote('AUTOTUNE_RUN ' + json.dumps(meta, sort_keys=True))}",
            "  date -u +'AUTOTUNE_START %Y-%m-%dT%H:%M:%SZ'",
            f"  {cmd}",
            "  rc=$?",
            '  echo "AUTOTUNE_EXIT $rc"',
            f"}} 2>&1 | tee {shlex.quote(log)}",
            "",
        ])
        _write_exec(os.path.join(out_dir, script), body)
        manifest["runs"].append({**meta, "script": script, "log": log, "env": r["env"],
                                 "nNodes": r.get("nNodes"), "nRanks": r.get("nRanks"),
                                 "ppn": r.get("ppn"), "nvls_env": r.get("nvls_env")})

    phases = []
    for r in manifest["runs"]:
        if r["phase"] not in phases:
            phases.append(r["phase"])
    runner = [
        "#!/bin/bash",
        f"# Run the autotune {title} in order; completed runs (AUTOTUNE_EXIT 0) are skipped.",
        "# Usage: ./run_all.sh [phase ...]   phases: " + " ".join(phases),
        'cd "$(dirname "$0")"',
        f"if [ $# -gt 0 ]; then PHASES=(\"$@\"); else PHASES=({' '.join(phases)}); fi",
    ]
    if not hostfile_pattern:
        runner.append("hostfiles/prepare_hostfiles.sh || exit 1")
    runner += ["fail=0", "for phase in \"${PHASES[@]}\"; do"]
    runner.append("  case \"$phase\" in")
    for ph in phases:
        ids = [r["run_id"] for r in manifest["runs"] if r["phase"] == ph]
        runner.append(f"    {ph}) RUNS=({' '.join(ids)}) ;;")
    runner += [
        '    *) echo "unknown phase $phase" >&2; exit 2 ;;',
        "  esac",
        '  for id in "${RUNS[@]}"; do',
        '    if grep -q "^AUTOTUNE_EXIT 0$" "logs/$id.log" 2>/dev/null; then continue; fi',
        '    echo "=== $phase: $id"',
        '    "scripts/$id.sh" > /dev/null || { echo "FAILED: $id" >&2; fail=$((fail+1)); }',
        "  done",
        "done",
        'echo "done; failed runs: $fail"',
        "",
    ]
    _write_exec(os.path.join(out_dir, "run_all.sh"), "\n".join(runner))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def plan_ab(template, scales, colls, splits, Rs, tuner_lib, profile="rail_nop2p", cycles=3, repeats=1):
    """A/B validation runs: A = NCCL_TUNER_PLUGIN=none (today's nightly), B =
    MRC tuner with FORCE_COST_MODEL=1, plus one B probe per key to record the
    applied cells.  A and B alternate (ABAB...) so drift hits both."""
    base_nvls = template.env.get("NCCL_NVLS_ENABLE", "2")
    tuner_env = {"NCCL_TUNER_FORCE_COST_MODEL": "1", "NCCL_TUNER_COST_MODEL_PROFILE": profile}
    runs = []
    for nodes in scales:
        for coll in colls:
            for split in splits:
                topo = topology.comm_topology(nodes, template.ppr, {**template.env, "NCCL_TESTS_SPLIT_MASK": split,
                                                                    "NCCL_MNNVL_ENABLE": "0"})
                for R in Rs:
                    seq = [("ab_probe", "B", "probe")]
                    for k in range(repeats):
                        seq += [("ab_a", "A", "default"), ("ab_b", "B", "default")] if k % 2 == 0 else \
                               [("ab_b", "B", "default"), ("ab_a", "A", "default")]
                    for k, (kind, arm, role) in enumerate(seq):
                        tuner = "none" if arm == "A" else tuner_lib
                        extra = None if arm == "A" else tuner_env
                        env = _run_env(base_nvls, split, role, None, None, tuner, extra)
                        if kind == "ab_probe":
                            env["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING"
                        rid = _rid(f"{nodes}n", coll, f"m{split}", f"R{R}", kind, k)
                        runs.append({
                            "run_id": rid, "phase": f"ab-{nodes}n", "smoke": False, "kind": kind,
                            "role": role, "coll": coll, "phys_nodes": nodes, "ppr": template.ppr,
                            "np": nodes * template.ppr, "split": split, "R": R,
                            "nvls_env": base_nvls, "cell": "default",
                            "nNodes": topo["nNodes"], "nRanks": topo["nRanks"], "ppn": topo["ppn"],
                            "env": env, "bench": bench_args(template, R, role, cycles),
                        })
    return runs


def summarize(runs):
    out = {}
    for r in runs:
        key = (r["phase"], r["coll"], r["split"])
        out[key] = out.get(key, 0) + 1
    return out


__all__ = ["Template", "candidate_cells", "plan_sweep", "plan_ab", "write_runs", "parse_cell"]
