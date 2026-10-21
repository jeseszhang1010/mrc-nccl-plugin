import numpy as np
import pytest

from autotune import params as pm


def test_schema_size_matches_struct():
    # 2 ints + 4 doubles + 7x3 + 3x7x3 + 5x3 caps + 4 doubles + 7x3 + 1 int
    assert pm.num_values() == 2 + 4 + 21 + 63 + 15 + 2 + 2 + 21 + 1


def test_parse_gb200f_profile(plugin_path):
    p = pm.load_params(plugin_path, "mnnvl")
    assert p["gpusPerNode"] == 64
    assert p["nChannels"] == 32
    assert p["bwInter"] == pytest.approx(28.19)
    assert p["ringBwCapIntra"] == pytest.approx(17.5)
    # Fields appended after the original struct default to 0 for mnnvl.
    assert p["patBwEff"] == 0.0
    assert p["disableUnmodeledCells"] == 0
    assert np.all(p["regBwScale"] == 0.0)


def test_rail_profile_present_and_not_validated(plugin_path):
    text = open(plugin_path).read()
    p = pm.parse_params(text, "rail_nop2p")
    assert p["gpusPerNode"] == 4
    assert p["disableUnmodeledCells"] == 1
    assert p["patBwEff"] > 0
    assert pm.parse_validated(text, "rail_nop2p") is False


def test_render_replace_roundtrip(plugin_path):
    text = open(plugin_path).read()
    p = pm.parse_params(text, "rail_nop2p")
    p["bwInter"] = 3.25
    p["hwLatencies"][2][0][2] = 17.5
    block = pm.render_block(p, "rail_nop2p", True, ["test"])
    new = pm.replace_block(text, "rail_nop2p", block)
    back = pm.parse_params(new, "rail_nop2p")
    assert pm.params_to_values(back) == pytest.approx(pm.params_to_values(p))
    assert pm.parse_validated(new, "rail_nop2p") is True
    # The mnnvl profile is untouched.
    assert pm.params_to_values(pm.parse_params(new, "mnnvl")) == \
        pm.params_to_values(pm.parse_params(text, "mnnvl"))


def test_param_keys():
    p = pm.params_from_values([])
    pm.set_param(p, "hwLatencies[2][1][0]", 4.5)
    assert pm.get_param(p, "hwLatencies[2][1][0]") == 4.5
    assert pm.key_label("hwLatencies[2][1][0]") == "hwLatencies[NET][ring][ll]"
    assert pm.key_label("baseLatencies[6][2]") == "baseLatencies[pat][simple]"
