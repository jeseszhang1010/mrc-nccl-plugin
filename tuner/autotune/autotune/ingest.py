"""Parsers for nccl-tests output and the nightly report.

Inputs understood:
  * the nightly report text ("RUN n: <case>" blocks with header fields, the
    captured mpirun line, the printed cycle and the captured env JSON);
  * raw nccl-tests stdout, optionally with NCCL INFO lines, as written by the
    scripts that `autotune sweep` generates (self-describing via a leading
    `AUTOTUNE_RUN {json}` line and a trailing `AUTOTUNE_EXIT <rc>` line).

Cycle rule: nccl-tests prints the -N cycles back to back without a
separator, so a new cycle starts whenever the size column does not increase.
Only the LAST cycle is used for selection and fitting (earlier cycles can be
skewed by dead BT0-BT1 links); it must be complete and all N cycles present,
otherwise the run is marked `incomplete`.
"""

import json
import re
import shlex

from . import topology
from .constants import (
    ALGO_NAMES, BINARY_COLL, NCCL_FUNC_NAMES, NCCL_LOG_ALGO, NCCL_LOG_PROTO, PROTO_NAMES,
    REPORT_COLL,
)

_SIZE_UNITS = {"": 1, "B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}

# nccl-tests result row (optionally prefixed by "NNNN:" in the report).
_ROW_RE = re.compile(
    r"^\s*(?:\d+:\s+)?(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$")

_TUNING_RE = re.compile(
    r"NCCL INFO (AllReduce|AllGather|ReduceScatter|Broadcast|Reduce): (\d+) Bytes -> "
    r"Algo (\S+) proto (\S+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")
_NVLS_RE = re.compile(r"NVLS multicast support is (not )?available")
_ERR_RE = re.compile(r"(NCCL WARN|Test NCCL failure|NCCL error|Test failure|ncclInvalidUsage|"
                     r"no algorithm|ncclInternalError|ncclSystemError)", re.I)


def parse_size(s):
    m = re.match(r"^\s*(\d+)\s*([KMGT]?)B?\s*$", str(s), re.I)
    if not m:
        raise ValueError(f"bad size '{s}'")
    return int(m.group(1)) * _SIZE_UNITS[m.group(2).upper()]


# ---------------------------------------------------------------------------
# mpirun command line
# ---------------------------------------------------------------------------
_BENCH_FLAGS = {
    "-b": "minbytes", "--minbytes": "minbytes", "-e": "maxbytes", "--maxbytes": "maxbytes",
    "-f": "stepfactor", "--stepfactor": "stepfactor", "-g": "ngpus", "--ngpus": "ngpus",
    "-c": "check", "--check": "check", "-R": "local_register", "--local_register": "local_register",
    "-w": "warmup_iters", "--warmup_iters": "warmup_iters", "-n": "iters", "--iters": "iters",
    "-N": "run_cycles", "--run_cycles": "run_cycles",
}


def parse_mpirun(cmd):
    """Split an mpirun command line into launcher options, env and benchmark."""
    toks = shlex.split(cmd)
    out = {"tokens": toks, "np": None, "ppr": None, "env": {}, "hostfile": None,
           "binary": None, "binary_index": None, "coll": None, "bench": {}}
    i = 0
    while i < len(toks):
        t = toks[i]
        if (t.endswith("_perf") and "/" in t) or t in BINARY_COLL:
            out["binary"] = t
            out["binary_index"] = i
            base = t.rsplit("/", 1)[-1]
            out["coll"] = BINARY_COLL.get(base)
            break
        if t in ("-np", "-n", "--np") and i + 1 < len(toks):
            out["np"] = int(toks[i + 1])
            i += 2
            continue
        if t in ("--map-by", "-map-by") and i + 1 < len(toks):
            m = re.match(r"ppr:(\d+):node", toks[i + 1])
            if m:
                out["ppr"] = int(m.group(1))
            i += 2
            continue
        if t in ("--hostfile", "-hostfile", "--machinefile", "-machinefile") and i + 1 < len(toks):
            out["hostfile"] = toks[i + 1]
            i += 2
            continue
        if t == "-x" and i + 1 < len(toks):
            kv = toks[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                out["env"][k] = v
            i += 2
            continue
        i += 1
    if out["binary_index"] is not None:
        args = toks[out["binary_index"] + 1:]
        j = 0
        while j < len(args):
            key = _BENCH_FLAGS.get(args[j])
            if key and j + 1 < len(args):
                out["bench"][key] = args[j + 1]
                j += 2
            else:
                j += 1
    return out


def expected_sizes(bench):
    """Message sizes nccl-tests will sweep for -b/-e/-f."""
    lo = parse_size(bench.get("minbytes", "32M"))
    hi = parse_size(bench.get("maxbytes", "32M"))
    f = int(bench.get("stepfactor", "1"))
    sizes, s = [], lo
    while s <= hi:
        sizes.append(s)
        if f <= 1:
            break
        s *= f
    return sizes


# ---------------------------------------------------------------------------
# nccl-tests rows and cycles
# ---------------------------------------------------------------------------
def parse_rows(lines):
    rows = []
    for line in lines:
        m = _ROW_RE.match(line)
        if not m:
            continue
        rows.append({
            "size": int(m.group(1)), "count": int(m.group(2)),
            "time_oop": float(m.group(6)), "algbw_oop": float(m.group(7)),
            "busbw_oop": float(m.group(8)), "wrong_oop": m.group(9),
            "time_ip": float(m.group(10)), "algbw_ip": float(m.group(11)),
            "busbw_ip": float(m.group(12)), "wrong_ip": m.group(13),
        })
    return rows


def split_cycles(rows):
    cycles, cur, prev = [], [], None
    for r in rows:
        if prev is not None and r["size"] <= prev:
            cycles.append(cur)
            cur = []
        cur.append(r)
        prev = r["size"]
    if cur:
        cycles.append(cur)
    return cycles


def check_cycles(cycles, cycles_expected, n_sizes_expected):
    """(status, note) for the cycle list given the expected -N and sweep length."""
    if not cycles:
        return "missing", "no result rows"
    if cycles_expected and len(cycles) < cycles_expected:
        return "incomplete", f"{len(cycles)}/{cycles_expected} cycles"
    last = cycles[-1]
    want = n_sizes_expected or len(cycles[0])
    if len(last) < want:
        return "incomplete", f"last cycle has {len(last)}/{want} sizes"
    bad = [r["size"] for r in last if r["wrong_oop"] not in ("0", "N/A") or r["wrong_ip"] not in ("0", "N/A")]
    if bad:
        return "invalid", f"correctness errors at sizes {bad[:5]}"
    return "ok", ""


# ---------------------------------------------------------------------------
# NCCL INFO evidence
# ---------------------------------------------------------------------------
def _canon_cell(algo, proto):
    a = NCCL_LOG_ALGO.get(algo.lower())
    p = NCCL_LOG_PROTO.get(proto.lower())
    if a is None or p is None:
        return None
    return f"{a}/{p}"


def parse_tuning_lines(text, comm_nranks):
    """Rank-0 `<Coll>: <bytes> Bytes -> Algo X proto Y channel{Lo..Hi}` lines.

    NCCL prints count*typesize, which for AllGather/ReduceScatter is the
    per-rank size; it is scaled by nRanks so it matches the nccl-tests size
    column (and the nBytes NCCL hands to the tuner).  Returns a list of
    {coll, size, cell, ch_lo, ch_hi}, deduplicated per (coll, size, cell).
    """
    seen, out = set(), []
    for m in _TUNING_RE.finditer(text):
        coll = NCCL_FUNC_NAMES[m.group(1)]
        nbytes = int(m.group(2))
        if coll in ("allgather", "reducescatter") and comm_nranks:
            nbytes *= comm_nranks
        cell = _canon_cell(m.group(3), m.group(4))
        if cell is None:
            continue
        key = (coll, nbytes, cell)
        if key in seen:
            continue
        seen.add(key)
        out.append({"coll": coll, "size": nbytes, "cell": cell,
                    "ch_lo": int(m.group(5)), "ch_hi": int(m.group(6))})
    return out


def parse_nccl_table(text):
    """NCCL's init-time TUNING table ("Algorithm | ..." blocks): latency (us)
    and bandwidth (GB/s) per (coll, algo, proto) as NCCL models them."""
    out = []
    algos = None
    func_map = {"Broadcast": "broadcast", "Reduce": "reduce", "AllGather": "allgather",
                "ReduceScatter": "reducescatter", "AllReduce": "allreduce"}
    for line in text.splitlines():
        body = line.split("NCCL INFO", 1)[-1]
        if "Algorithm   |" in body or "Algorithm |" in body:
            names = [c.strip() for c in body.split("|")[1:] if c.strip()]
            algos = [n.lower() for n in names]
            continue
        if algos is None:
            continue
        m = re.match(r"^\s*(\w+)\s*\|(.*)$", body)
        if not m or m.group(1) not in func_map:
            continue
        pairs = re.findall(r"([\d.]+)\s*/\s*([\d.]+)", m.group(2))
        for k, (lat, bw) in enumerate(pairs):
            ai, pi = divmod(k, 3)
            if ai >= len(algos):
                break
            cell = _canon_cell(algos[ai], PROTO_NAMES[pi])
            if cell:
                out.append({"coll": func_map[m.group(1)], "cell": cell,
                            "lat": float(lat), "bw": float(bw)})
    return out


def parse_nvls_available(text):
    m = None
    for m in _NVLS_RE.finditer(text):
        pass
    if m is None:
        return None
    return 0 if m.group(1) else 1


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------
def forced_cell(env):
    """(algo, proto, cell) forced through NCCL_ALGO/NCCL_PROTO.  `cell` is
    'default' when nothing is forced and None when only one of them is forced
    or a list is given (not a single measurable cell)."""
    a = env.get("NCCL_ALGO")
    p = env.get("NCCL_PROTO")
    if not a and not p:
        return None, None, "default"
    ca = NCCL_LOG_ALGO.get(a.lower()) if a and "," not in a and ":" not in a else None
    cp = NCCL_LOG_PROTO.get(p.lower()) if p and "," not in p and ":" not in p else None
    cell = f"{ca}/{cp}" if ca and cp else None
    return a, p, cell


def build_record(run_id, source, kind, case_name, coll, phys_nodes, ppr, env, bench, cmd,
                 cycles, status, note, text="", rc=None):
    env = dict(env or {})
    R = int(bench.get("local_register", 0) or 0)
    topo = None
    if phys_nodes and ppr:
        topo = topology.comm_topology(phys_nodes, ppr, env)
    fa, fp, cell = forced_cell(env)
    rec = {
        "run_id": run_id, "source": source, "kind": kind, "case_name": case_name,
        "coll": coll, "phys_nodes": phys_nodes, "ppr": ppr,
        "split": env.get("NCCL_TESTS_SPLIT_MASK") or env.get("NCCL_TESTS_SPLIT"),
        "R": R, "regbuff": topology.regbuff_for_R(R),
        "p2p_disable": 1 if topology.p2p_disabled(env) else 0,
        "nvls": 1 if topology.nvls_enabled(env) else 0,
        "mnnvl": 1 if topology.mnnvl_enabled(env) else 0,
        "profile": topology.profile_for_env(env),
        "nNodes": topo["nNodes"] if topo else None,
        "nRanks": topo["nRanks"] if topo else None,
        "ppn": topo["ppn"] if topo else None,
        "cell": cell, "forced_algo": fa, "forced_proto": fp,
        "tuner": env.get("NCCL_TUNER_PLUGIN", ""),
        "cycles_expected": int(bench.get("run_cycles", 1) or 1),
        "cycles_found": len(cycles), "cycles": cycles,
        "status": status, "note": note, "rc": rc,
        "env": env, "cmd": cmd,
        "probe_cells": [], "nccl_table": [], "log_lines": [], "nvls_available": None,
    }
    if topo is None and status == "ok":
        rec["status"], rec["note"] = "incomplete", "topology unknown (no command/split recorded)"
    if text:
        rec["probe_cells"] = parse_tuning_lines(text, rec["nRanks"])
        rec["nccl_table"] = parse_nccl_table(text)
        rec["nvls_available"] = parse_nvls_available(text)
        for line in text.splitlines():
            if "TUNER/MRC" in line:
                rec["log_lines"].append(("tuner", line.strip()))
            elif _ERR_RE.search(line):
                rec["log_lines"].append(("error", line.strip()))
    _verify_forced(rec)
    return rec


def _verify_forced(rec):
    """A forced run whose own TUNING lines show another cell did not measure
    the cell it claims; mark it invalid."""
    if rec["status"] != "ok" or rec["cell"] in (None, "default") or not rec["probe_cells"]:
        return
    seen = {p["cell"] for p in rec["probe_cells"] if p["coll"] == rec["coll"]}
    if seen and seen != {rec["cell"]}:
        rec["status"] = "invalid"
        rec["note"] = f"forced {rec['cell']} but NCCL used {sorted(seen)}"


def parse_raw_log(text, source, meta=None):
    """Parse one raw nccl-tests stdout log written by an autotune script."""
    meta = dict(meta or {})
    rc = None
    for line in text.splitlines():
        if line.startswith("AUTOTUNE_RUN "):
            meta = {**json.loads(line[len("AUTOTUNE_RUN "):]), **meta}
        elif line.startswith("AUTOTUNE_EXIT "):
            try:
                rc = int(line.split()[1])
            except (IndexError, ValueError):
                rc = None
    cmd = meta.get("cmd")
    if not cmd:
        m = re.search(r"^\s*(?:\+\s*)?(mpirun .*)$", text, re.M)
        cmd = m.group(1) if m else None
    if not cmd:
        raise ValueError(f"{source}: no AUTOTUNE_RUN header or mpirun line")
    p = parse_mpirun(cmd)
    ppr = meta.get("ppr") or p["ppr"]
    np_ = meta.get("np") or p["np"]
    phys = meta.get("phys_nodes") or (np_ // ppr if np_ and ppr else None)
    rows = parse_rows(text.splitlines())
    cycles = split_cycles(rows)
    status, note = check_cycles(cycles, int(p["bench"].get("run_cycles", 1)),
                                len(expected_sizes(p["bench"])) if p["bench"] else 0)
    if status == "missing" and (rc not in (None, 0) or _ERR_RE.search(text)):
        status, note = "invalid", f"no results (rc={rc})"
    if status == "ok" and rc not in (None, 0):
        note = f"rc={rc}"
    run_id = meta.get("run_id") or source
    rec = build_record(run_id, source, meta.get("kind", "raw"), meta.get("case", run_id),
                       meta.get("coll") or p["coll"], phys, ppr, p["env"], p["bench"], cmd,
                       cycles, status, note, text=text, rc=rc)
    # Probe runs log NCCL_DEBUG=INFO to stdout, which interleaves with (and
    # breaks) the result rows; their purpose is the TUNING cells, not timings.
    if (meta.get("role", rec["kind"]) == "probe" and rec["status"] in ("missing", "incomplete") and rc == 0
            and rec["probe_cells"] and rec["nNodes"] is not None):
        rec["status"], rec["note"] = "ok", f"probe: {len(rec['probe_cells'])} TUNING cells ({note})"
    return rec


# ---------------------------------------------------------------------------
# Nightly report text ("RUN n: <case>" blocks)
# ---------------------------------------------------------------------------
_RUN_HDR_RE = re.compile(r"^=+\nRUN (\d+): (\S+)\n=+\n", re.M)


def parse_nightly_report(text, source="nightly"):
    """Parse the nightly report; each RUN block becomes a record of kind
    'nightly'.  The report already holds a single selected cycle (it states
    which one); that cycle is stored as the last cycle."""
    heads = list(_RUN_HDR_RE.finditer(text))
    out = []
    for k, h in enumerate(heads):
        body = text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        num, case = int(h.group(1)), h.group(2)

        def field(name):
            m = re.search(r"^" + re.escape(name) + r":\s*(.*)$", body, re.M)
            return m.group(1).strip() if m else None

        cmd_m = re.search(r"^mpirun .*$", body, re.M)
        cmd = cmd_m.group(0) if cmd_m else None
        env = {}
        env_m = re.search(r"CAPTURED ENVIRONMENT\n-+\n(\{.*?\})\s*(?:\n=|\Z)", body, re.S)
        if env_m:
            try:
                env = json.loads(env_m.group(1))
            except json.JSONDecodeError:
                env = {}
        p = parse_mpirun(cmd) if cmd else {"env": {}, "bench": {}, "ppr": None, "np": None, "coll": None}
        env = {**env, **p["env"]}
        bench = dict(p["bench"])
        if "local_register" not in bench and field("Registration (-R)") not in (None, "Not recorded"):
            bench["local_register"] = field("Registration (-R)")
        split = field("Split mask")
        if split and split != "Not recorded" and "NCCL_TESTS_SPLIT_MASK" not in env:
            env["NCCL_TESTS_SPLIT_MASK"] = split
        coll = p["coll"] or REPORT_COLL.get((field("Collective") or "").lower())
        bench_name = field("Benchmark")
        if bench_name:
            coll = BINARY_COLL.get(bench_name + "_perf", coll)
        phys = int(field("Nodes")) if (field("Nodes") or "").isdigit() else None
        if not cmd and split in (None, "Not recorded"):
            phys = None  # communicator split unknown -> topology unknown
        ranks = int(field("Ranks")) if (field("Ranks") or "").isdigit() else None
        ppr = p["ppr"] or (ranks // phys if phys and ranks else None)
        rows = parse_rows(body.splitlines())
        sel = re.search(r"cycle (\d+)", field("Selected data") or "")
        cycles = [rows] if rows else []
        status, note = ("ok", "") if rows else ("missing", "no rows")
        exp = int(bench.get("run_cycles", 1) or 1)
        if sel and int(sel.group(1)) < exp:
            status, note = "incomplete", f"report selected cycle {sel.group(1)} of {exp}"
        rec = build_record(f"nightly-{num:02d}-{case}", source, "nightly", case, coll, phys, ppr,
                           env, bench, cmd, cycles, status, note)
        rec["cycles_found"] = int(sel.group(1)) if sel else len(cycles)
        out.append(rec)
    return out


def is_nightly_report(text):
    return bool(_RUN_HDR_RE.search(text))


def ingest_paths(con, paths, kind=None):
    """Ingest files into the store; returns the stored records."""
    from . import db as dbm
    recs = []
    for path in paths:
        with open(path, errors="replace") as f:
            text = f.read()
        if is_nightly_report(text):
            new = parse_nightly_report(text, source=path)
        else:
            meta = {"kind": kind} if kind else None
            new = [parse_raw_log(text, path, meta)]
        for r in new:
            dbm.store_run(con, r)
        recs.extend(new)
    return recs


def ingest_manifest(con, manifest_path):
    """Ingest every run listed in an autotune sweep/A-B manifest whose log exists."""
    import os
    from . import db as dbm
    with open(manifest_path) as f:
        man = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest_path))
    recs, missing = [], []
    for run in man["runs"]:
        log = os.path.join(base, run["log"])
        if not os.path.exists(log):
            missing.append(run["run_id"])
            continue
        with open(log, errors="replace") as f:
            text = f.read()
        meta = {k: run[k] for k in ("run_id", "kind", "coll", "phys_nodes", "ppr", "cmd") if k in run}
        rec = parse_raw_log(text, log, meta)
        dbm.store_run(con, rec)
        recs.append(rec)
    return recs, missing


__all__ = [
    "ALGO_NAMES", "parse_size", "parse_mpirun", "expected_sizes", "parse_rows", "split_cycles",
    "check_cycles", "parse_tuning_lines", "parse_nccl_table", "parse_raw_log",
    "parse_nightly_report", "ingest_paths", "ingest_manifest",
]
