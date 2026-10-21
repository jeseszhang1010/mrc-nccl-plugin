"""Map a benchmark launch (physical nodes, ranks per node, NCCL_TESTS_SPLIT*)
to the communicator NCCL actually builds, i.e. what the tuner sees."""

import math
import re

DEFAULT_NVL_DOMAIN_NODES = 16


def parse_split(env):
    """Return a function rank -> color from NCCL_TESTS_SPLIT / _SPLIT_MASK, or
    None when the run is not split (see nccl-tests README)."""
    mask = env.get("NCCL_TESTS_SPLIT_MASK")
    if mask not in (None, ""):
        m = int(mask, 0)
        return lambda r: r & m
    spec = env.get("NCCL_TESTS_SPLIT")
    if spec in (None, ""):
        return None
    m = re.match(r"^\s*(AND|OR|MOD|DIV|&|\||%|/)\s*(\S+)\s*$", spec, re.I)
    if not m:
        raise ValueError(f"unsupported NCCL_TESTS_SPLIT '{spec}'")
    op, v = m.group(1).upper(), int(m.group(2), 0)
    ops = {
        "AND": lambda r: r & v, "&": lambda r: r & v,
        "OR": lambda r: r | v, "|": lambda r: r | v,
        "MOD": lambda r: r % v, "%": lambda r: r % v,
        "DIV": lambda r: r // v, "/": lambda r: r // v,
    }
    return ops[op]


def comm_topology(phys_nodes, ranks_per_node, env=None, nvl_domain_nodes=DEFAULT_NVL_DOMAIN_NODES):
    """Topology of the communicator containing global rank 0.

    Ranks are assumed to be placed by node (`--map-by ppr:K:node`), so global
    rank r lives on node r // K.  Returns a dict with nRanks, nNodes (NCCL's
    view), ppn, physNodes (of the communicator), groups (number of
    communicators), mnnvl and profile.
    """
    env = env or {}
    color_of = parse_split(env)
    n = phys_nodes * ranks_per_node
    colors = [color_of(r) if color_of else 0 for r in range(n)]
    c0 = colors[0]
    members = [r for r in range(n) if colors[r] == c0]
    nodes = {r // ranks_per_node for r in members}
    n_ranks = len(members)
    comm_phys = len(nodes)
    mnnvl = mnnvl_enabled(env)
    if mnnvl:
        n_nodes = max(1, math.ceil(comm_phys / nvl_domain_nodes))
    elif p2p_disabled(env) and shm_disabled(env):
        # No intra-node transport: NCCL reports localRanks 1, nNodes == nRanks.
        n_nodes = n_ranks
    else:
        n_nodes = comm_phys
    return {
        "nRanks": n_ranks,
        "nNodes": n_nodes,
        "ppn": n_ranks // n_nodes,
        "physNodes": comm_phys,
        "groups": len(set(colors)),
        "mnnvl": mnnvl,
        "profile": profile_for_env(env),
    }


def mnnvl_enabled(env):
    # NCCL_MNNVL_ENABLE defaults to auto (on for GB200 NVL racks).
    return env.get("NCCL_MNNVL_ENABLE", "2") != "0"


def p2p_disabled(env):
    return env.get("NCCL_P2P_DISABLE", "0") == "1"


def shm_disabled(env):
    return env.get("NCCL_SHM_DISABLE", "0") == "1"


def nvls_enabled(env):
    # NCCL_NVLS_ENABLE: 0 = off, 1 = force, 2 = auto (default).
    return env.get("NCCL_NVLS_ENABLE", "2") != "0"


def profile_for_env(env):
    if mnnvl_enabled(env):
        return "mnnvl"
    return "rail_nop2p" if p2p_disabled(env) else "rail_p2p"


def regbuff_for_R(r):
    """nccl-tests -R (0 none, 1 local, 2 symmetric) -> tuner regBuff flag."""
    return 1 if int(r or 0) >= 1 else 0
