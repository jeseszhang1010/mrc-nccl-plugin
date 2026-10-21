import json
import os

import pytest

from autotune import db, ingest, topology

from .conftest import FIXTURES


def _row(size, t, busbw, wrong="0"):
    algbw = size / (t * 1e3)
    return (f"  {size:>12}  {size // 4:>12}     float     sum      -1  {t:8.2f}  {algbw:6.2f}  {busbw:6.2f}"
            f"  {wrong:>6}  {t:8.2f}  {algbw:6.2f}  {busbw:6.2f}  {wrong:>6}")


SIZES = [4096, 8192, 16384, 32768]


def _raw_log(cycles=3, cmd_extra="", sizes=SIZES, tuning="", rc=0, n_last=None):
    meta = {"run_id": "t1", "kind": "forced", "coll": "allgather", "phys_nodes": 8, "ppr": 4,
            "cmd": ("mpirun -np 32 --map-by ppr:4:node --hostfile h -x NCCL_TESTS_SPLIT_MASK=0x1 "
                    "-x NCCL_MNNVL_ENABLE=0 -x NCCL_P2P_DISABLE=1 -x NCCL_ALGO=ring -x NCCL_PROTO=Simple "
                    f"{cmd_extra}/opt/nccl-tests/build/all_gather_perf -b 4K -e 32K -f 2 -g 1 -R 0 "
                    f"-w 5 -n 5 -N {cycles}")}
    lines = ["AUTOTUNE_RUN " + json.dumps(meta), "# nccl-tests header", tuning]
    for c in range(cycles):
        ss = sizes if (c < cycles - 1 or n_last is None) else sizes[:n_last]
        for i, s in enumerate(ss):
            lines.append(_row(s, 10.0 + i + (50.0 if c == 0 else 0.0), 1.0 + i))
    lines.append(f"AUTOTUNE_EXIT {rc}")
    return "\n".join(lines)


def test_parse_size():
    assert ingest.parse_size("4K") == 4096
    assert ingest.parse_size("4G") == 4 << 30
    assert ingest.parse_size("123") == 123


def test_parse_mpirun():
    p = ingest.parse_mpirun("mpirun -np 512 --map-by ppr:4:node --hostfile /h -x A=1 -x LD_LIBRARY_PATH "
                            "/bin/bash wrap.sh --x 1 /opt/b/all_reduce_perf -b 4K -e 4G -f 2 -R 1 -N 3")
    assert p["np"] == 512 and p["ppr"] == 4 and p["hostfile"] == "/h"
    assert p["env"] == {"A": "1"}
    assert p["coll"] == "allreduce"
    assert p["bench"]["local_register"] == "1" and p["bench"]["run_cycles"] == "3"
    assert ingest.expected_sizes(p["bench"])[-1] == 4 << 30


def test_last_cycle_selected_and_cycles_split():
    rec = ingest.parse_raw_log(_raw_log(), "x.log")
    assert rec["status"] == "ok", rec["note"]
    assert rec["cycles_found"] == 3
    # First cycle is slower (dead links); the stored last cycle is not.
    assert rec["cycles"][0][0]["time_oop"] == pytest.approx(60.0)
    assert rec["cycles"][-1][0]["time_oop"] == pytest.approx(10.0)
    assert rec["cell"] == "ring/simple"
    assert (rec["nNodes"], rec["nRanks"], rec["ppn"]) == (8, 16, 2)


def test_incomplete_runs():
    rec = ingest.parse_raw_log(_raw_log(cycles=3, n_last=2), "x.log")
    assert rec["status"] == "incomplete"
    rec = ingest.parse_raw_log(_raw_log(cycles=3).replace("-N 3", "-N 4"), "x.log")
    assert rec["status"] == "incomplete" and "3/4" in rec["note"]


def test_failed_run_is_invalid():
    text = "\n".join(_raw_log().splitlines()[:3]) + "\nhost:1:1 [0] NCCL WARN no algorithm\nAUTOTUNE_EXIT 1\n"
    rec = ingest.parse_raw_log(text, "x.log")
    assert rec["status"] == "invalid"


def test_tuning_lines_scale_allgather_bytes_and_verify_forced_cell():
    tuning = ("h:1:1 [0] NCCL INFO AllGather: 256 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..7}\n"
              "h:1:1 [0] NCCL INFO AllGather: 512 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..7}")
    rec = ingest.parse_raw_log(_raw_log(tuning=tuning), "x.log")
    assert [(p["size"], p["cell"]) for p in rec["probe_cells"]] == [(4096, "ring/simple"), (8192, "ring/simple")]
    assert rec["status"] == "ok"
    bad = tuning.replace("proto SIMPLE", "proto LL")
    rec = ingest.parse_raw_log(_raw_log(tuning=bad), "x.log")
    assert rec["status"] == "invalid" and "ring/ll" in rec["note"]


def test_probe_with_interleaved_info_lines_is_usable():
    info = "host:1:1 [0] NCCL INFO AllGather: {b} Bytes -> Algo RING proto LL channel{{Lo..Hi}}={{0..0}}"
    text = _raw_log(cycles=1).replace('"kind": "forced"', '"kind": "probe"').replace(
        "-x NCCL_ALGO=ring -x NCCL_PROTO=Simple ", "")
    lines = []
    for ln in text.splitlines():
        if ln.startswith("  ") and ln.split()[0].isdigit():
            ln = ln[:60] + info.format(b=int(ln.split()[0]) // 32)
        lines.append(ln)
    rec = ingest.parse_raw_log("\n".join(lines), "probe.log")
    assert rec["status"] == "ok" and rec["probe_cells"]
    rec = ingest.parse_raw_log("\n".join(lines).replace("AUTOTUNE_EXIT 0", "AUTOTUNE_EXIT 1"), "probe.log")
    assert rec["status"] != "ok"


def test_nccl_table_parse():
    text = """h [0] NCCL INFO   Algorithm   |            Tree            |            Ring            |  CollNetDirect |
h [0] NCCL INFO   Protocol    |             LL |          LL128 |         Simple |             LL |          LL128 |         Simple |
h [0] NCCL INFO  Max NThreads |            512 |            640 |            512 |            512 |            640 |            512 |
h [0] NCCL INFO     AllReduce |    14.4/  9.1 |    20.0/ 20.0 |    56.0/ 60.0 |     5.6/ 12.0 |     8.4/ 40.0 |    30.0/ 90.0 |"""
    rows = ingest.parse_nccl_table(text)
    d = {(r["coll"], r["cell"]): (r["lat"], r["bw"]) for r in rows}
    assert d[("allreduce", "tree/ll")] == (14.4, 9.1)
    assert d[("allreduce", "ring/simple")] == (30.0, 90.0)


def test_nightly_report_excerpt(tmp_path):
    text = open(os.path.join(FIXTURES, "nightly_excerpt.txt")).read()
    recs = ingest.parse_nightly_report(text)
    assert len(recs) == 3
    r1, r2, r3 = recs
    assert (r1["coll"], r1["nNodes"], r1["nRanks"], r1["R"], r1["profile"], r1["cell"]) == \
        ("allreduce", 256, 256, 0, "rail_nop2p", "default")
    assert r1["status"] == "ok" and len(r1["cycles"][0]) == 21
    assert r1["cycles"][0][16]["busbw_oop"] == pytest.approx(51.29)  # the 256MB dip
    assert r2["cell"] == "tree/simple" and r2["R"] == 1
    assert r3["status"] == "incomplete"  # no command, split mask not recorded
    con = db.connect(str(tmp_path / "a.db"))
    for r in recs:
        db.store_run(con, r)
    assert len(db.last_cycle(con, r1["run_id"])) == 21
    assert db.last_cycle(con, r3["run_id"]) == []


def test_topology():
    t = topology.comm_topology(128, 4, {"NCCL_TESTS_SPLIT_MASK": "0x1", "NCCL_MNNVL_ENABLE": "0"})
    assert (t["nRanks"], t["nNodes"], t["ppn"], t["groups"]) == (256, 128, 2, 2)
    t = topology.comm_topology(128, 4, {"NCCL_TESTS_SPLIT_MASK": "0x3", "NCCL_MNNVL_ENABLE": "0"})
    assert (t["nRanks"], t["nNodes"], t["ppn"], t["groups"]) == (128, 128, 1, 4)
    t = topology.comm_topology(128, 4, {})
    assert (t["nRanks"], t["nNodes"], t["profile"]) == (512, 8, "mnnvl")
    t = topology.comm_topology(8, 4, {"NCCL_TESTS_SPLIT": "MOD 2", "NCCL_MNNVL_ENABLE": "0",
                                      "NCCL_P2P_DISABLE": "1"})
    assert (t["nRanks"], t["ppn"], t["profile"]) == (16, 2, "rail_nop2p")
    # P2P and SHM both disabled: every rank is its own NCCL node.
    t = topology.comm_topology(128, 4, {"NCCL_TESTS_SPLIT_MASK": "0x1", "NCCL_MNNVL_ENABLE": "0",
                                        "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"})
    assert (t["nRanks"], t["nNodes"], t["ppn"], t["physNodes"]) == (256, 256, 1, 128)
