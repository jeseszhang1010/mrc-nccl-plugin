import os
import shlex
import subprocess
from collections import Counter

from autotune import ingest, sweep

from .conftest import FIXTURES


def _template():
    text = open(os.path.join(FIXTURES, "nightly_excerpt.txt")).read()
    rec = ingest.parse_nightly_report(text)[0]
    return sweep.Template(rec["cmd"])


def test_candidate_cells():
    assert sweep.candidate_cells("allreduce", 2, 0) == sweep.AR_CELLS
    assert "nvls_tree/simple" in sweep.candidate_cells("allreduce", 2, 1)
    assert "nvls_tree/simple" not in sweep.candidate_cells("allreduce", 1, 1)
    assert "pat/simple" in sweep.candidate_cells("allgather", 1, 0)
    assert "pat/simple" not in sweep.candidate_cells("reducescatter", 2, 0)
    assert all(not c.startswith("pat") for c in sweep.candidate_cells("allreduce", 1, 1))


def test_plan_counts_match_plan():
    runs = sweep.plan_sweep(_template(), [128], ["allreduce", "allgather", "reducescatter"],
                            ["0x1", "0x3"], [0, 1])
    c = Counter((r["coll"], r["split"], r["R"]) for r in runs)
    # Nightly template has P2P and SHM disabled: NCCL sees one rank per node,
    # so PAT is a candidate for both splits and there is no nvls_tree variant.
    assert c[("allreduce", "0x1", 0)] == 8
    assert c[("allreduce", "0x3", 1)] == 8
    assert c[("allgather", "0x1", 0)] == 6
    assert c[("reducescatter", "0x3", 1)] == 6
    assert len(runs) == 80
    assert len({r["run_id"] for r in runs}) == len(runs)


def test_command_rewrite_keeps_template_and_overrides_managed():
    tpl = _template()
    runs = sweep.plan_sweep(tpl, [32], ["allgather"], ["0x3"], [1])
    r = next(x for x in runs if x["cell"] == "pat/simple")
    cmd = tpl.build(nodes=32, hostfile="hostfiles/hostfile-32", coll="allgather", env=r["env"],
                    bench_args=r["bench"])
    p = ingest.parse_mpirun(cmd)
    assert p["np"] == 128 and p["hostfile"] == "hostfiles/hostfile-32"
    assert p["coll"] == "allgather" and p["binary"].endswith("/all_gather_perf")
    env = p["env"]
    assert env["NCCL_ALGO"] == "pat" and env["NCCL_PROTO"] == "Simple"
    assert env["NCCL_TUNER_PLUGIN"] == "none" and env["NCCL_TESTS_SPLIT_MASK"] == "0x3"
    assert env["NCCL_DEBUG"] == "WARN" and "NCCL_DEBUG_SUBSYS" not in env
    # Unmanaged template settings are preserved verbatim.
    assert env["NCCL_IB_QPS_PER_CONNECTION"] == "2" and env["NCCL_MNNVL_ENABLE"] == "0"
    assert "numa-bind.sh" in cmd and "--cpu_bind_range" in cmd
    assert p["bench"]["run_cycles"] == "3" and p["bench"]["local_register"] == "1"
    toks = shlex.split(cmd)
    assert toks.count("-x") == len([t for t in toks if t == "-x"])  # well-formed pairs


def test_probe_and_smoke_args():
    tpl = _template()
    runs = sweep.plan_sweep(tpl, [], ["allreduce"], ["0x1"], [0], smoke_scales=[8])
    probe = next(r for r in runs if r["role"] == "probe")
    assert probe["env"]["NCCL_DEBUG_SUBSYS"] == "INIT,TUNING"
    b = dict(zip(probe["bench"][::2], probe["bench"][1::2]))
    assert b["-N"] == "1" and b["-e"] == "1M" and probe["kind"] == "smoke"
    forced = next(r for r in runs if r["role"] == "forced")
    assert dict(zip(forced["bench"][::2], forced["bench"][1::2]))["-e"] == "1M"


def test_write_runs_scripts_are_valid_bash(tmp_path):
    tpl = _template()
    runs = sweep.plan_sweep(tpl, [8], ["allreduce"], ["0x1"], [0])
    man = sweep.write_runs(str(tmp_path), tpl, runs, hostfile_source="/share/hosts")
    assert len(man["runs"]) == 8
    for f in ["run_all.sh", "hostfiles/prepare_hostfiles.sh", man["runs"][0]["script"]]:
        subprocess.run(["bash", "-n", str(tmp_path / f)], check=True)
    text = (tmp_path / man["runs"][0]["script"]).read_text()
    assert "AUTOTUNE_RUN" in text and "AUTOTUNE_EXIT" in text


def test_ab_plan():
    tpl = _template()
    runs = sweep.plan_ab(tpl, [128], ["allreduce"], ["0x1"], [0], "/x/libnccl-tuner-mrc.so", repeats=2)
    kinds = [r["kind"] for r in runs]
    assert kinds == ["ab_probe", "ab_a", "ab_b", "ab_b", "ab_a"]
    b = next(r for r in runs if r["kind"] == "ab_b")
    assert b["env"]["NCCL_TUNER_PLUGIN"] == "/x/libnccl-tuner-mrc.so"
    assert b["env"]["NCCL_TUNER_FORCE_COST_MODEL"] == "1"
    a = next(r for r in runs if r["kind"] == "ab_a")
    assert a["env"]["NCCL_TUNER_PLUGIN"] == "none" and "NCCL_TUNER_FORCE_COST_MODEL" not in a["env"]


def test_cli_rejects_untuned_collectives(tmp_path):
    import pytest
    from autotune import cli
    from .conftest import FIXTURES
    with pytest.raises(SystemExit) as e:
        cli.main(["sweep", "--template", os.path.join(FIXTURES, "nightly_excerpt.txt"), "--template-run", "1",
                  "--out", str(tmp_path / "s"), "--colls", "allreduce,alltoall"])
    assert "alltoall" in str(e.value)
