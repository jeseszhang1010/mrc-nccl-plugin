"""Index maps and spellings shared with nccl/tuner.h and NCCL itself."""

# Algorithm indices (nccl/tuner.h).
ALGO_TREE = 0
ALGO_RING = 1
ALGO_COLLNET_DIRECT = 2
ALGO_COLLNET_CHAIN = 3
ALGO_NVLS = 4
ALGO_NVLS_TREE = 5
ALGO_PAT = 6
NUM_ALGOS = 7

# Protocol indices (nccl/tuner.h).
PROTO_LL = 0
PROTO_LL128 = 1
PROTO_SIMPLE = 2
NUM_PROTOS = 3

HW_NVLINK = 0
HW_PCI = 1
HW_NET = 2
NUM_HW = 3

NUM_SCALES = 3

# Canonical (plugin CSV / tuner log) names.
ALGO_NAMES = ["tree", "ring", "collnet_direct", "collnet_chain", "nvls", "nvls_tree", "pat"]
PROTO_NAMES = ["ll", "ll128", "simple"]

# Spellings accepted by NCCL_ALGO / NCCL_PROTO.
NCCL_ENV_ALGO = {
    "tree": "tree", "ring": "ring", "nvls": "nvls", "nvls_tree": "nvlstree",
    "pat": "pat", "collnet_direct": "collnetdirect", "collnet_chain": "collnetchain",
}
NCCL_ENV_PROTO = {"ll": "LL", "ll128": "LL128", "simple": "Simple"}

# Spellings NCCL prints in INFO/TUNING lines (ncclAlgoStr / ncclProtoStr),
# lower-cased by the parser before lookup.
NCCL_LOG_ALGO = {
    "tree": "tree", "ring": "ring", "collnetdirect": "collnet_direct",
    "collnetchain": "collnet_chain", "nvls": "nvls", "nvlstree": "nvls_tree",
    "pat": "pat",
}
NCCL_LOG_PROTO = {"ll": "ll", "ll128": "ll128", "simple": "simple"}

COLLS = ["allreduce", "allgather", "reducescatter"]

# nccl-tests binaries per collective.
COLL_BINARY = {
    "allreduce": "all_reduce_perf",
    "allgather": "all_gather_perf",
    "reducescatter": "reduce_scatter_perf",
    "alltoall": "alltoall_perf",
    "sendrecv": "sendrecv_perf",
    "broadcast": "broadcast_perf",
    "reduce": "reduce_perf",
}
BINARY_COLL = {v: k for k, v in COLL_BINARY.items()}

# NCCL function names as printed in TUNING lines.
NCCL_FUNC_NAMES = {
    "AllReduce": "allreduce", "AllGather": "allgather",
    "ReduceScatter": "reducescatter", "Broadcast": "broadcast", "Reduce": "reduce",
}

# Collective names used in the nightly report header ("Collective: All-reduce").
REPORT_COLL = {
    "all-reduce": "allreduce", "all-gather": "allgather",
    "reduce-scatter": "reducescatter", "all-to-all": "alltoall",
    "sendrecv": "sendrecv",
}


def cell_name(algo, proto):
    a = ALGO_NAMES[algo] if isinstance(algo, int) else algo
    p = PROTO_NAMES[proto] if isinstance(proto, int) else proto
    return f"{a}/{p}"


def parse_cell(cell):
    a, p = cell.split("/")
    return ALGO_NAMES.index(a), PROTO_NAMES.index(p)
