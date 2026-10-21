"""SQLite store for runs, last-cycle points and NCCL TUNING evidence."""

import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  source TEXT, kind TEXT, case_name TEXT, coll TEXT,
  phys_nodes INTEGER, ppr INTEGER, split TEXT, R INTEGER, regbuff INTEGER,
  p2p_disable INTEGER, nvls INTEGER, mnnvl INTEGER, profile TEXT,
  nNodes INTEGER, nRanks INTEGER, ppn INTEGER,
  cell TEXT, forced_algo TEXT, forced_proto TEXT, tuner TEXT,
  cycles_expected INTEGER, cycles_found INTEGER,
  status TEXT, note TEXT, rc INTEGER, nvls_available INTEGER,
  env_json TEXT, cmd TEXT
);
CREATE TABLE IF NOT EXISTS points (
  run_id TEXT, cycle INTEGER, last INTEGER, size INTEGER,
  time_oop REAL, algbw_oop REAL, busbw_oop REAL,
  time_ip REAL, algbw_ip REAL, busbw_ip REAL
);
CREATE INDEX IF NOT EXISTS points_run ON points(run_id, last);
CREATE TABLE IF NOT EXISTS probe_cells (
  run_id TEXT, coll TEXT, size INTEGER, cell TEXT, ch_lo INTEGER, ch_hi INTEGER
);
CREATE TABLE IF NOT EXISTS nccl_table (
  run_id TEXT, coll TEXT, cell TEXT, lat REAL, bw REAL
);
CREATE TABLE IF NOT EXISTS log_lines (
  run_id TEXT, tag TEXT, line TEXT
);
"""

RUN_COLS = [
    "run_id", "source", "kind", "case_name", "coll", "phys_nodes", "ppr", "split", "R",
    "regbuff", "p2p_disable", "nvls", "mnnvl", "profile", "nNodes", "nRanks", "ppn",
    "cell", "forced_algo", "forced_proto", "tuner", "cycles_expected", "cycles_found",
    "status", "note", "rc", "nvls_available", "env_json", "cmd",
]


def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def store_run(con, rec):
    """Insert or replace one parsed run (see ingest.RunRecord.to_db)."""
    rid = rec["run_id"]
    for table in ("points", "probe_cells", "nccl_table", "log_lines"):
        con.execute(f"DELETE FROM {table} WHERE run_id=?", (rid,))
    row = dict(rec)
    row["env_json"] = json.dumps(rec.get("env", {}), sort_keys=True)
    con.execute(
        f"INSERT OR REPLACE INTO runs ({','.join(RUN_COLS)}) VALUES ({','.join('?' * len(RUN_COLS))})",
        [row.get(c) for c in RUN_COLS],
    )
    ncyc = len(rec.get("cycles", []))
    for ci, cyc in enumerate(rec.get("cycles", [])):
        last = 1 if (ci == ncyc - 1 and rec.get("status") == "ok") else 0
        con.executemany(
            "INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(rid, ci + 1, last, r["size"], r["time_oop"], r["algbw_oop"], r["busbw_oop"],
              r["time_ip"], r["algbw_ip"], r["busbw_ip"]) for r in cyc],
        )
    con.executemany("INSERT INTO probe_cells VALUES (?,?,?,?,?,?)",
                    [(rid, p["coll"], p["size"], p["cell"], p["ch_lo"], p["ch_hi"])
                     for p in rec.get("probe_cells", [])])
    con.executemany("INSERT INTO nccl_table VALUES (?,?,?,?,?)",
                    [(rid, t["coll"], t["cell"], t["lat"], t["bw"]) for t in rec.get("nccl_table", [])])
    con.executemany("INSERT INTO log_lines VALUES (?,?,?)",
                    [(rid, tag, line) for tag, line in rec.get("log_lines", [])])
    con.commit()


def runs(con, where="1=1", args=()):
    return [dict(r) for r in con.execute(f"SELECT * FROM runs WHERE {where} ORDER BY run_id", args)]


def last_cycle(con, run_id):
    """Rows of the selected (last, complete) cycle, ordered by size."""
    return [dict(r) for r in con.execute(
        "SELECT * FROM points WHERE run_id=? AND last=1 ORDER BY size", (run_id,))]


def probe_cells(con, run_id):
    return {r["size"]: r["cell"] for r in con.execute(
        "SELECT size, cell FROM probe_cells WHERE run_id=? ORDER BY size", (run_id,))}
