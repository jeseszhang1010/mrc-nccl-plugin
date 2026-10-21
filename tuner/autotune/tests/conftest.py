import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
TUNER_DIR = os.path.dirname(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)

from autotune import params as pm  # noqa: E402

PLUGIN = os.path.join(TUNER_DIR, "plugin.c")
FIXTURES = os.path.join(HERE, "fixtures")


@pytest.fixture(scope="session")
def plugin_path():
    return PLUGIN


@pytest.fixture(scope="session")
def seed_params():
    return pm.load_params(PLUGIN, "rail_nop2p")


def _build_driver(tuner_dir, out):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler")
    subprocess.run([cc, "-I", os.path.join(tuner_dir, "nccl"), "-O2", "-o", out,
                    os.path.join(tuner_dir, "test", "cost_model_main.c")], check=True)
    return out


@pytest.fixture(scope="session")
def driver(tmp_path_factory):
    """cost_model_main compiled from the checked-in plugin.c."""
    return _build_driver(TUNER_DIR, str(tmp_path_factory.mktemp("drv") / "cost_model_main"))


@pytest.fixture
def tuner_copy(tmp_path):
    """A writable copy of the tuner sources (plugin.c, headers, test drivers)."""
    dst = tmp_path / "tuner"
    dst.mkdir()
    for name in ("plugin.c", "Makefile", "nccl_tuner.conf"):
        shutil.copy(os.path.join(TUNER_DIR, name), dst / name)
    shutil.copytree(os.path.join(TUNER_DIR, "nccl"), dst / "nccl")
    (dst / "test").mkdir()
    for name in ("cost_model_main.c", "csv_tuning_main.c"):
        shutil.copy(os.path.join(TUNER_DIR, "test", name), dst / "test" / name)
    return dst
