import pytest

from autotune import codegen, db, dips, fit, ingest, oracle, params as pm, synth, validate

from .conftest import _build_driver


def _cfg(**kw):
    return fit.FitConfig(polish_maxiter=800, **kw)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory, request):
    from .conftest import PLUGIN
    seed = pm.load_params(PLUGIN, "rail_nop2p")
    truth = synth.perturb(seed, seed=3)
    con = db.connect(str(tmp_path_factory.mktemp("syn") / "s.db"))
    synth.generate(con, truth, seed)
    return con, seed, truth


def test_oracle_and_noise(synthetic):
    con, _, _ = synthetic
    o = oracle.build(con, "rail_nop2p")
    assert len(o["entries"]) == 3 * 14 * 21  # 3 scales x 14 configurations x 21 sizes
    for n in o["noise"].values():
        assert n["pairs"] > 100 and 0.0 < n["sigma"] < 0.03 and n["tol"] == pytest.approx(0.03)
    e = o["entries"][0]
    assert e["best"] in e["equivalent"] and oracle.regret(e, e["best"]) == 0.0
    assert oracle.regret(e, "nope/cell") == 1.0


def test_fit_recovers_selection_and_passes_gate(synthetic):
    con, seed, _ = synthetic
    o = oracle.build(con, "rail_nop2p")
    Pf, res = fit.fit_all(con, seed, o, _cfg(), log=lambda *_: None)
    assert res["seed_regret"]["violations"] > 20          # NCCL-like seed is wrong often
    assert res["info"]["final_regret"]["violations"] <= 2  # fit fixes (almost) all
    assert res["validated"] and res["gate"]["passed"]
    held = {c["held_out"]: c for c in res["cv"]}
    assert held[128]["violations"] == 0                    # train 8+32 -> predict 128


def test_anomaly_is_tagged_and_exempted(tmp_path, plugin_path):
    seed = pm.load_params(plugin_path, "rail_nop2p")
    truth = synth.perturb(seed, seed=3)
    con = db.connect(str(tmp_path / "n.db"))
    # A single-size slowdown (notch) of the large-message winner at the nightly
    # scale: the oracle winner flips there, which an affine model cannot follow.
    synth.generate(con, truth, seed, scales=(8, 32, 128),
                   notch=(128, "allreduce", "0x3", 0, "ring/simple", 1 << 31, 1.6))
    o = oracle.build(con, "rail_nop2p")
    e = oracle.index(o)[("rail_nop2p", "allreduce", 128, 128, 0, 0)][1 << 31]
    assert e["best"] != "ring/simple"
    _, res = fit.fit_all(con, seed, o, _cfg(), log=lambda *_: None)
    tagged = [a for a in res["anomalies"] if a["size"] == 1 << 31 and a["cell"] == "ring/simple"
              and a["nNodes"] == 128 and a["nRanks"] == 128 and a["R"] == 0]
    assert tagged
    hit = [x for x in res["gate"]["exempt"] if x["size"] == 1 << 31 and x["nRanks"] == 128
           and x["R"] == 0 and x["coll"] == "allreduce"]
    assert hit and hit[0]["model"] == "ring/simple"
    assert res["validated"]


def test_regbuff_scale_is_fitted_when_needed(tmp_path, plugin_path):
    seed = pm.load_params(plugin_path, "rail_nop2p")
    truth = synth.perturb(seed, seed=5)
    con = db.connect(str(tmp_path / "r.db"))
    synth.generate(con, truth, seed, scales=(8, 32, 128), colls=("allgather",),
                   reg_scale={"ring/simple": 0.75})
    o = oracle.build(con, "rail_nop2p")
    Pf, res = fit.fit_all(con, seed, o, _cfg(), log=lambda *_: None)
    keys = {p["key"] for p in res["free"]}
    assert "regBwScale[1][2]" in keys
    assert Pf["regBwScale"][1][2] == pytest.approx(0.75, rel=0.05)


def test_codegen_build_parity_and_offline_validate(synthetic, tuner_copy):
    con, seed, _ = synthetic
    o = oracle.build(con, "rail_nop2p")
    Pf, res = fit.fit_all(con, seed, o, _cfg(), log=lambda *_: None)
    res["values"] = pm.params_to_values(Pf)
    plugin = str(tuner_copy / "plugin.c")
    changes = codegen.write(plugin, Pf, "rail_nop2p", res["validated"], codegen.provenance(res, "fit_result.json"))
    assert changes
    text = open(plugin).read()
    assert pm.parse_validated(text, "rail_nop2p") is True
    assert "Gate nNodes=[128]" in text
    drv = _build_driver(str(tuner_copy), str(tuner_copy / "test" / "cost_model_main"))
    r = codegen.parity(drv, plugin, "rail_nop2p", [(256, 128), (128, 128), (64, 32)])
    assert r["ok"], r["mismatches"]
    off = validate.offline(o, drv, "rail_nop2p", [128])
    assert off["passed"], off["violations"][:3]


def test_dips_notch_rule(tmp_path):
    con = db.connect(str(tmp_path / "d.db"))
    sizes = [1 << i for i in range(20, 26)]
    busbw = [10, 20, 30, 25, 60, 70]   # 25 < min(30, 60) by >10% -> notch at 8M
    plateau = [10, 10.5, 10.4, 10.6, 60, 70]  # plateau then jump: no notch
    for rid, bws in (("notch", busbw), ("plateau", plateau)):
        rows = [{"size": s, "count": s // 4, "time_oop": 1.0, "algbw_oop": b, "busbw_oop": b,
                 "wrong_oop": "0", "time_ip": 1.0, "algbw_ip": b, "busbw_ip": b, "wrong_ip": "0"}
                for s, b in zip(sizes, bws)]
        rec = ingest.build_record(rid, "t", "nightly", rid, "allreduce", 8, 4,
                                  {"NCCL_MNNVL_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}, {}, "",
                                  [rows], "ok", "")
        db.store_run(con, rec)
    rows = dips.find_dips(con)
    assert [(r["run_id"], r["size"]) for r in rows] == [("notch", 1 << 23)]
    # Collectives the tuner does not select for (alltoall, sendrecv) are not reported.
    rows_a2a = [{**r, "busbw_oop": r["busbw_oop"]} for r in db.last_cycle(con, "notch")]
    rec = ingest.build_record("a2a", "t", "nightly", "a2a", "alltoall", 8, 4,
                              {"NCCL_MNNVL_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}, {}, "",
                              [[{**r, "wrong_oop": "0", "wrong_ip": "0"} for r in rows_a2a]], "ok", "")
    db.store_run(con, rec)
    assert [r["run_id"] for r in dips.find_dips(con)] == ["notch"]
