"""The Python mirror must match the compiled C cost model cell for cell."""

import pytest

from autotune import codegen, costmodel as cm, params as pm
from autotune.constants import ALGO_NAMES, PROTO_NAMES

MNNVL_TOPOS = [(8, 1), (64, 1), (128, 2), (512, 8), (4096, 64)]
RAIL_TOPOS = [(16, 8), (8, 8), (64, 32), (32, 32), (256, 128), (128, 128), (254, 127)]


@pytest.mark.parametrize("profile,topos", [("mnnvl", MNNVL_TOPOS), ("rail_nop2p", RAIL_TOPOS)])
def test_python_matches_c(driver, plugin_path, profile, topos):
    r = codegen.parity(driver, plugin_path, profile, topos, rtol=1e-6)
    assert r["ok"], r["mismatches"]
    assert r["cells"] > 1000


def test_rail_profile_prices_pat_only_with_one_rank_per_node(seed_params):
    pat, simple = ALGO_NAMES.index("pat"), PROTO_NAMES.index("simple")
    assert cm.cell_time(seed_params, "allgather", pat, simple, 128, 128, 1 << 20) is not None
    assert cm.cell_time(seed_params, "allgather", pat, simple, 256, 128, 1 << 20) is None
    assert cm.cell_time(seed_params, "allreduce", pat, simple, 128, 128, 1 << 20) is None


def test_regbuff_scale_applies(seed_params):
    p = pm.copy_params(seed_params)
    ring, simple = ALGO_NAMES.index("ring"), PROTO_NAMES.index("simple")
    p["regBwScale"][ring][simple] = 0.5
    t0 = cm.cell_terms(p, "allreduce", ring, simple, 256, 128, 0)
    t1 = cm.cell_terms(p, "allreduce", ring, simple, 256, 128, 1)
    assert t1[1] == pytest.approx(0.5 * t0[1])


def test_unmodelled_cells_disabled_in_rail(driver):
    cells, _ = codegen.run_driver(driver, "rail_nop2p", [(256, 128)], ["allgather"], max_bytes=4096)
    # The driver offers NVLS for AG; the rail profile cannot price it on >1 node.
    assert cells[("allgather", 256, 128, 0, 4096)]["nvls/simple"] == pytest.approx(cm.DISABLED_COST)
    cells, _ = codegen.run_driver(driver, "mnnvl", [(256, 4)], ["allgather"], max_bytes=4096)
    assert "nvls/simple" not in cells[("allgather", 256, 4, 0, 4096)]
