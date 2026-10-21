"""autotune command line.  Run `python -m autotune -h` from tuner/autotune."""

import argparse
import json
import os
import sys

from . import codegen, db, dips, fit, ingest, oracle, sweep, synth, validate
from . import params as pm
from .constants import COLLS


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def _strs(s):
    return [x for x in s.split(",") if x]


def _colls(s):
    colls = _strs(s)
    bad = [c for c in colls if c not in COLLS]
    if bad:
        sys.exit(f"--colls: {','.join(bad)} not tuned by the cost model (supported: {','.join(COLLS)})")
    return colls


def _default_plugin():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "plugin.c"))


def _template(args):
    if not args.template and not args.template_cmd:
        sys.exit("give --template (nightly report or file with an mpirun line) or --template-cmd")
    if args.template_cmd:
        return sweep.Template(args.template_cmd)
    with open(args.template) as f:
        text = f.read()
    if not ingest.is_nightly_report(text):
        line = next((ln for ln in text.splitlines() if ln.strip().startswith("mpirun ")), None)
        if not line:
            sys.exit(f"{args.template}: no mpirun line found")
        return sweep.Template(line.strip())
    recs = [r for r in ingest.parse_nightly_report(text) if r["cmd"]]
    if args.template_run:
        recs = [r for r in recs if r["run_id"].startswith(f"nightly-{args.template_run:02d}-")]
    else:
        recs = [r for r in recs if r["profile"] == args.profile and r["coll"] in COLLS
                and r["cell"] == "default"]
    if not recs:
        sys.exit("no usable template run in the nightly report (use --template-run N)")
    print(f"template: {recs[0]['run_id']}")
    return sweep.Template(recs[0]["cmd"])


def cmd_ingest(args):
    con = db.connect(args.db)
    recs = ingest.ingest_paths(con, args.files, kind=args.kind) if args.files else []
    missing = []
    for m in args.manifest or []:
        r, miss = ingest.ingest_manifest(con, m)
        recs += r
        missing += miss
    by = {}
    for r in recs:
        by.setdefault((r["kind"], r["status"]), 0)
        by[(r["kind"], r["status"])] += 1
    for (k, s), n in sorted(by.items()):
        print(f"  {k:<10} {s:<11} {n}")
    for r in recs:
        if r["status"] != "ok":
            print(f"  {r['status']}: {r['run_id']} {r['note']}")
    if missing:
        print(f"  {len(missing)} manifest runs have no log yet")


def cmd_sweep(args):
    _colls(args.colls)
    tpl = _template(args)
    runs = sweep.plan_sweep(tpl, _ints(args.scales), _colls(args.colls), _strs(args.splits), _ints(args.R),
                            nvls_variant=not args.no_nvls_variant, cycles=args.cycles,
                            smoke_scales=_ints(args.smoke_scales))
    man = sweep.write_runs(args.out, tpl, runs, args.hostfile_pattern, args.hostfile_source, "sweep")
    phases = {}
    for r in man["runs"]:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    print(f"wrote {len(man['runs'])} runs to {args.out}: " + ", ".join(f"{p}={n}" for p, n in phases.items()))
    print(f"run: {os.path.join(args.out, 'run_all.sh')} [phase ...]")


def cmd_oracle(args):
    con = db.connect(args.db)
    o = oracle.build(con, args.profile, args.min_tol)
    oracle.save(o, args.out)
    if args.csv:
        oracle.write_csv(o, args.csv)
    print(f"oracle: {len(o['entries'])} (configuration, size) entries; noise {o['noise']}")


def _load_or_build_oracle(con, args):
    if args.oracle and os.path.exists(args.oracle):
        return oracle.load(args.oracle)
    return oracle.build(con, args.profile)


def cmd_dips(args):
    con = db.connect(args.db)
    o = oracle.load(args.oracle) if args.oracle and os.path.exists(args.oracle) else None
    rows = dips.find_dips(con, o, kinds=tuple(_strs(args.kinds)))
    os.makedirs(args.out, exist_ok=True)
    dips.write_csv(rows, os.path.join(args.out, "dips.csv"))
    pngs = dips.plot_runs(con, rows, os.path.join(args.out, "plots"), o) if not args.no_plots else []
    with open(os.path.join(args.out, "dips.md"), "w") as f:
        f.write("# Dip report\n\n" + dips.markdown(rows, top=args.top) + "\n")
    print(f"{len(rows)} dips; wrote {args.out}/dips.csv, dips.md, {len(pngs)} plots")


def _fit_report(res, path):
    L = [f"# Cost-model fit: profile `{res['profile']}`", ""]
    L.append(f"Data nNodes: {res['nodes']}; gate nNodes: {res['gate_nodes']}; "
             f"**validated: {res['validated']}**")
    L += ["", "## Selection quality (oracle regret)", "",
          "| model | entries | top-1 | mean regret | max regret | violations |", "|---|---|---|---|---|---|"]

    def row(name, s):
        if not s or not s.get("n"):
            return f"| {name} | 0 | | | | |"
        return (f"| {name} | {s['n']} | {s['top1']:.3f} | {s['mean_regret']:.4f} | "
                f"{s['max_regret']:.3f} | {s['violations']} |")
    L.append(row("NCCL default (all scales)", res["nccl_default"]))
    L.append(row("NCCL default (gate scale)", res["nccl_default_gate"]))
    L.append(row("seed parameters", res["seed_regret"]))
    L.append(row("fitted (all scales)", res["info"].get("final_regret")))
    L += ["", "## Cross-scale validation", "", "| held-out nNodes | entries | top-1 | mean regret | violations |",
          "|---|---|---|---|---|"]
    for c in res["cv"]:
        L.append(f"| {c['held_out']} | {c.get('n', 0)} | {c.get('top1', 0):.3f} | "
                 f"{c.get('mean_regret', 0):.4f} | {c.get('violations', 0)} |")
    g = res["gate"]
    L += ["", f"## Gate (nNodes={g['nodes']}, nvls={g['nvls']}): {'PASS' if g['passed'] else 'FAIL'}", "",
          f"{g['checked']} entries checked, {len(g['violations'])} violations, "
          f"{len(g['exempt'])} anomaly-exempt.", ""]
    if g["violations"] or g["exempt"]:
        L += ["| status | coll | nRanks | R | nvls | size | model | oracle | regret |", "|---|---|---|---|---|---|---|---|---|"]
        for tag, rows in (("violation", g["violations"]), ("exempt", g["exempt"])):
            for r in rows:
                L.append(f"| {tag} | {r['coll']} | {r['nRanks']} | {r['R']} | {r['nvls']} | {r['size']} | "
                         f"{r['model']} | {r['best']} | {100 * r['regret']:.1f}% |")
    info = res["info"]
    L += ["", "## Fit", "", f"- points: {info.get('points')}, free parameters: {len(res['free'])}",
          f"- log-loss: stage 2 {info.get('loss_stage2', 0):.4f}, stage 3 {info.get('loss_stage3', 0):.4f}"
          + (f", stage 4 {info['loss_stage4']:.4f}" if "loss_stage4" in info else ""),
          f"- anomaly-tagged points (excluded from the fit): {len(res['anomalies'])}", "",
          "| parameter | seed | fitted |", "|---|---|---|"]
    for p in res["free"]:
        L.append(f"| {p['label']} | {p['seed']:.6g} | {p['fit']:.6g} |")
    if res["anomalies"]:
        L += ["", "### Anomalies", "", "| coll | nNodes | nRanks | R | cell | size | rel. residual |",
              "|---|---|---|---|---|---|---|"]
        for a in res["anomalies"][:50]:
            L.append(f"| {a['coll']} | {a['nNodes']} | {a['nRanks']} | {a['R']} | {a['cell']} | {a['size']} | "
                     f"{100 * a['rel_resid']:+.0f}% |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def cmd_fit(args):
    con = db.connect(args.db)
    o = _load_or_build_oracle(con, args)
    seed = pm.load_params(args.plugin, args.profile)
    cfg = fit.FitConfig(profile=args.profile, polish=not args.no_polish, polish_maxiter=args.polish_maxiter,
                        gate_nodes=_ints(args.gate_nodes) or None, gate_nvls=tuple(_ints(args.gate_nvls)),
                        fit_regbuff=args.regbuff, min_tol=args.min_tol)
    Pf, res = fit.fit_all(con, seed, o, cfg)
    os.makedirs(args.out, exist_ok=True)
    res["values"] = pm.params_to_values(Pf)
    with open(os.path.join(args.out, "fit_result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    validate.write_rows_csv(res["picks"], os.path.join(args.out, "picks.csv"))
    _fit_report(res, os.path.join(args.out, "fit_report.md"))
    print(f"wrote {args.out}/fit_result.json, fit_report.md, picks.csv; validated={res['validated']}")


def cmd_codegen(args):
    res, params = codegen.load_fit(args.fit)
    validated = res["validated"]
    if args.force_validated:
        validated = True
    if args.not_validated:
        validated = False
    changes = codegen.write(args.plugin, params, res["profile"], validated, codegen.provenance(res, args.fit))
    print(f"updated {args.plugin}: {len(changes)} values changed, validated={validated}")
    for n, a, b in changes[:40]:
        print(f"  {n}: {a:.6g} -> {b:.6g}")
    if args.build or args.parity:
        driver = codegen.build(args.plugin)
        print(f"built tuner and {driver}")
        if args.parity:
            topos = sorted({(p["nRanks"], p["nNodes"]) for p in res.get("picks", [])}) or [(256, 128), (128, 128)]
            r = codegen.parity(driver, args.plugin, res["profile"], topos)
            print(f"parity: {r['cells']} cells, worst rel err {r['worst_rel_err']:.2e} -> {'OK' if r['ok'] else 'FAIL'}")
            if not r["ok"]:
                sys.exit(1)


def cmd_validate_offline(args):
    con = db.connect(args.db)
    o = _load_or_build_oracle(con, args)
    driver = args.driver or os.path.join(os.path.dirname(os.path.abspath(args.plugin)), "test", "cost_model_main")
    gate_nodes = _ints(args.gate_nodes) or [max(e["nNodes"] for e in o["entries"])]
    r = validate.offline(o, driver, args.profile, gate_nodes, tuple(_ints(args.gate_nvls)))
    os.makedirs(args.out, exist_ok=True)
    validate.write_rows_csv(r["rows"], os.path.join(args.out, "offline.csv"))
    lines = [f"# Offline validation ({args.profile}, compiled model)", "",
             f"Gate nNodes={gate_nodes}: {r['checked']} entries, {len(r['violations'])} violations -> "
             f"**{'PASS' if r['passed'] else 'FAIL'}**", "",
             "| coll | nNodes | nRanks | R | nvls | size | NCCL default | oracle (equivalent) | model | regret |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        if x["regret"] > 0 or (x["default_gap"] or 0) > x["tol"]:
            lines.append(f"| {x['coll']} | {x['nNodes']} | {x['nRanks']} | {x['R']} | {x['nvls']} | {x['size']} | "
                         f"{x['default']} | {x['equivalent']} | {x['model']} | {100 * x['regret']:.1f}% |")
    with open(os.path.join(args.out, "offline.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"offline: {r['checked']} gate entries, {len(r['violations'])} violations -> "
          f"{'PASS' if r['passed'] else 'FAIL'}; wrote {args.out}/offline.md, offline.csv")


def cmd_validate_ab_scripts(args):
    _colls(args.colls)
    tpl = _template(args)
    runs = sweep.plan_ab(tpl, _ints(args.scales), _colls(args.colls), _strs(args.splits), _ints(args.R),
                         args.tuner_lib, args.profile, args.cycles, args.repeats)
    man = sweep.write_runs(args.out, tpl, runs, args.hostfile_pattern, args.hostfile_source, "ab")
    print(f"wrote {len(man['runs'])} A/B runs to {args.out}; run {os.path.join(args.out, 'run_all.sh')}")


def cmd_validate_compare(args):
    con = db.connect(args.db)
    o = oracle.load(args.oracle) if args.oracle and os.path.exists(args.oracle) else None
    rep = validate.compare_ab(con, o, args.max_regression)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ab_report.md"), "w") as f:
        f.write("# A/B: NCCL_TUNER_PLUGIN=none (A) vs MRC tuner cost model (B)\n\n" + validate.ab_markdown(rep) + "\n")
    with open(os.path.join(args.out, "ab_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    ok = all(r["passed"] for r in rep) and bool(rep)
    print(f"A/B: {sum(r['passed'] for r in rep)}/{len(rep)} configurations pass -> {'PASS' if ok else 'FAIL'}")


def cmd_report(args):
    con = db.connect(args.db)
    recs = ingest.ingest_paths(con, [args.nightly])
    o = oracle.load(args.oracle) if args.oracle and os.path.exists(args.oracle) else None
    rows = dips.find_dips(con, o)
    os.makedirs(args.out, exist_ok=True)
    dips.write_csv(rows, os.path.join(args.out, "dips.csv"))
    pngs = dips.plot_runs(con, rows, os.path.join(args.out, "plots"), o) if not args.no_plots else []
    tuned = [r for r in recs if r["coll"] in COLLS]
    skipped = sorted({r["coll"] for r in recs if r["coll"] not in COLLS})
    lines = [f"# Nightly report: {os.path.basename(args.nightly)}", "",
             f"{len(recs)} runs ingested; {len(tuned)} AllReduce/AllGather/ReduceScatter runs analysed"
             + (f" ({', '.join(skipped)} not tuned, skipped)" if skipped else "") + "; "
             f"{len(rows)} dips" + ("" if o else " (notch only; no oracle given)"), "",
             "| run | coll | profile | nNodes | nRanks | R | tuner | cell | status |", "|---|---|---|---|---|---|---|---|---|"]
    for r in tuned:
        lines.append(f"| {r['case_name']} | {r['coll']} | {r['profile']} | {r['nNodes']} | {r['nRanks']} | {r['R']} | "
                     f"{r['tuner'] or '-'} | {r['cell']} | {r['status']} |")
    lines += ["", "## Dips", "", dips.markdown(rows, top=args.top)]
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}/report.md, dips.csv, {len(pngs)} plots")


def cmd_simulate(args):
    con = db.connect(args.db)
    seed = pm.load_params(args.plugin, args.profile)
    truth = synth.perturb(seed, args.seed)
    notch = (max(_ints(args.scales)), "allreduce", "0x1", 0, "ring/simple", 1 << 28, 1.6) if args.notch else None
    n = synth.generate(con, truth, seed, scales=_ints(args.scales), noise=args.noise, seed=args.seed, notch=notch)
    print(f"wrote {n} synthetic runs to {args.db} (truth = seed perturbed with seed {args.seed})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autotune", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, plugin=False, oracle_arg=False):
        p.add_argument("--db", default="autotune.db", help="SQLite store (default: autotune.db)")
        p.add_argument("--profile", default="rail_nop2p")
        if plugin:
            p.add_argument("--plugin", default=_default_plugin(), help="path to tuner/plugin.c")
        if oracle_arg:
            p.add_argument("--oracle", default="oracle.json", help="oracle JSON (built from --db if missing)")

    def tmpl(p):
        p.add_argument("--template", help="nightly report (txt) or file containing an mpirun line")
        p.add_argument("--template-run", type=int, help="RUN number in the nightly report to use as template")
        p.add_argument("--template-cmd", help="mpirun command line to use as template")
        p.add_argument("--hostfile-pattern", help="per-scale hostfile path, e.g. /share/hosts-{nodes}")
        p.add_argument("--hostfile-source", help="hostfile to take the first N unique hosts from")
        p.add_argument("--colls", default=",".join(COLLS))
        p.add_argument("--splits", default="0x1,0x3")
        p.add_argument("--R", default="0,1")
        p.add_argument("--cycles", type=int, default=3)

    p = sub.add_parser("ingest", help="ingest nightly report / raw logs / sweep manifests")
    common(p)
    p.add_argument("files", nargs="*")
    p.add_argument("--manifest", action="append", help="sweep or A/B manifest.json (repeatable)")
    p.add_argument("--kind", help="kind for raw logs without an AUTOTUNE_RUN header")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("sweep", help="generate probe/default/forced-cell run scripts")
    common(p)
    tmpl(p)
    p.add_argument("--out", required=True)
    p.add_argument("--scales", default="8,32,128")
    p.add_argument("--smoke-scales", default="8")
    p.add_argument("--no-nvls-variant", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("oracle", help="best measured cell per (configuration, size)")
    common(p)
    p.add_argument("--out", default="oracle.json")
    p.add_argument("--csv")
    p.add_argument("--min-tol", type=float, default=oracle.MIN_TOL)
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("dips", help="dip report on stored runs")
    common(p, oracle_arg=True)
    p.add_argument("--out", default="dips")
    p.add_argument("--kinds", default="nightly")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_dips)

    p = sub.add_parser("fit", help="fit the cost-model profile (stages 1-6)")
    common(p, plugin=True, oracle_arg=True)
    p.add_argument("--out", default="fit")
    p.add_argument("--gate-nodes", default="", help="nNodes for the acceptance gate (default: largest)")
    p.add_argument("--gate-nvls", default="0")
    p.add_argument("--regbuff", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--min-tol", type=float, default=oracle.MIN_TOL)
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--polish-maxiter", type=int, default=4000)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("codegen", help="write fitted params into plugin.c (+ build, parity)")
    p.add_argument("--fit", required=True, help="fit_result.json")
    p.add_argument("--plugin", default=_default_plugin())
    p.add_argument("--build", action="store_true")
    p.add_argument("--parity", action="store_true", help="build and compare C vs Python model")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--force-validated", action="store_true")
    g.add_argument("--not-validated", action="store_true")
    p.set_defaults(func=cmd_codegen)

    p = sub.add_parser("validate", help="offline check / A-B scripts / A-B compare")
    vs = p.add_subparsers(dest="vcmd", required=True)
    q = vs.add_parser("offline", help="compiled model picks vs oracle")
    common(q, plugin=True, oracle_arg=True)
    q.add_argument("--driver", help="cost_model_main path (default: next to plugin.c)")
    q.add_argument("--gate-nodes", default="")
    q.add_argument("--gate-nvls", default="0")
    q.add_argument("--out", default="validate")
    q.set_defaults(func=cmd_validate_offline)
    q = vs.add_parser("ab-scripts", help="generate A/B run scripts")
    common(q)
    tmpl(q)
    q.add_argument("--out", required=True)
    q.add_argument("--scales", default="128,32")
    q.add_argument("--tuner-lib", required=True, help="path of libnccl-tuner-mrc.so on the cluster")
    q.add_argument("--repeats", type=int, default=2)
    q.set_defaults(func=cmd_validate_ab_scripts)
    q = vs.add_parser("compare", help="A/B gates from ingested A/B logs")
    common(q, oracle_arg=True)
    q.add_argument("--max-regression", type=float, default=0.03)
    q.add_argument("--out", default="validate")
    q.set_defaults(func=cmd_validate_compare)

    p = sub.add_parser("report", help="ingest a nightly report and write the dip report")
    common(p, oracle_arg=True)
    p.add_argument("nightly")
    p.add_argument("--out", default="report")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("simulate", help="write a synthetic sweep (pipeline dry run)")
    common(p, plugin=True)
    p.add_argument("--scales", default="8,32,128")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--notch", action="store_true", help="inject a non-affine notch at the largest scale")
    p.set_defaults(func=cmd_simulate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
