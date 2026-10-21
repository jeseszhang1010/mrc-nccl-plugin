// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Standalone host-side driver for the MRC tuner cost model.
 * Builds without CUDA / NCCL by including plugin.c directly (so static
 * helpers and TunerContext are visible) and bypassing the v5 plugin symbol.
 * 
 * For each (topology, collective, message size) the driver populates
 * collCostTable via the cost model and prints the winning (algo, proto, time).
 *
 * With no arguments it sweeps the built-in GB200F (mnnvl) topologies.  Options:
 *   --profile mnnvl|rail_nop2p   parameter set to use (default: mnnvl)
 *   --topo NRANKS:NNODES         topology to evaluate (repeatable; replaces
 *                                the built-in list)
 *   --coll allreduce|allgather|reducescatter   (repeatable; default: all)
 *   --regbuff 0|1                regBuff passed to the model (default: 0)
 *   --min-bytes N / --max-bytes N  limit the message-size sweep
 *   --all-cells                  machine-readable output: one CSV line per
 *                                priced cell plus one per winner:
 *     cell,<coll>,<nRanks>,<nNodes>,<regBuff>,<nBytes>,<algo>,<proto>,<time_us>
 *     winner,<coll>,<nRanks>,<nNodes>,<regBuff>,<nBytes>,<algo>,<proto>,<time_us>
 */

#define NCCL_TUNER_TEST_MAIN 1
#include "../plugin.c"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void formatBytes(size_t b, char* out, size_t n) {
  if (b >= (1ULL << 30))      snprintf(out, n, "%.1fGB", b / (double)(1ULL << 30));
  else if (b >= (1ULL << 20)) snprintf(out, n, "%.1fMB", b / (double)(1ULL << 20));
  else if (b >= (1ULL << 10)) snprintf(out, n, "%.1fKB", b / (double)(1ULL << 10));
  else                        snprintf(out, n, "%zuB",   b);
}

// Allocate a numAlgo x numProto cost table.  Cells start at a large sentinel
// (no cap), except cells that are invalid for the collective which are set
// to NCCL_ALGO_PROTO_IGNORE so the model leaves them alone (mirrors what
// NCCL core does internally).
//
// NCCL core declares this as a contiguous float[NCCL_NUM_ALGORITHMS]
// [NCCL_NUM_PROTOCOLS] (see nccl/src/enqueue.cc) and passes it as float**.
// Allocate it the same way (a single contiguous block) so the plugin's 2D
// indexing matches production.
static float** allocCostTable(ncclFunc_t coll, size_t nRanks, size_t nNodes) {
  float* flat = (float*)malloc(NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS * sizeof(float));
  float (*t)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])flat;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      // Default: open cell.  Ignore cells the cost model doesn't support
      // anyway so the post-call scan can't accidentally pick an unfilled
      // sentinel as the winner.
      int valid = 0;
      if (coll == ncclFuncAllReduce) {
        if (a == NCCL_ALGO_RING || a == NCCL_ALGO_TREE) valid = 1;
        if (a == NCCL_ALGO_NVLS && p == NCCL_PROTO_SIMPLE) valid = 1;
        if (a == NCCL_ALGO_NVLS_TREE && p == NCCL_PROTO_SIMPLE) valid = 1;
      } else if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
        if (a == NCCL_ALGO_RING) valid = 1;
        if (a == NCCL_ALGO_NVLS && p == NCCL_PROTO_SIMPLE) valid = 1;
        // NCCL offers PAT (SIMPLE only) when every node has a single rank.
        if (a == NCCL_ALGO_PAT && p == NCCL_PROTO_SIMPLE && nRanks == nNodes) valid = 1;
      }
      t[a][p] = valid ? 1e30f : (float)NCCL_ALGO_PROTO_IGNORE;
    }
  }
  return (float**)flat;
}

static void freeCostTable(float** t) {
  free(t);
}

static int findWinner(float** tbl, int* outAlgo, int* outProto, float* outTime) {
  float (*t)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])tbl;
  float best = 1e30f;
  int ba = -1, bp = -1;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      float v = t[a][p];
      if (v == (float)NCCL_ALGO_PROTO_IGNORE) continue;
      if (v >= 1e29f) continue;  // untouched
      if (v < best) { best = v; ba = a; bp = p; }
    }
  }
  if (ba < 0) return 0;
  *outAlgo = ba; *outProto = bp; *outTime = best;
  return 1;
}

static int applyModel(ncclFunc_t coll, TunerContext* ctx, size_t nBytes, int regBuff, float** t) {
  switch (coll) {
    case ncclFuncAllReduce:
      return applyCostModelAllReduce(ctx, nBytes, 1, regBuff, t, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS);
    case ncclFuncAllGather:
      return applyCostModelAllGather(ctx, nBytes, 1, regBuff, t, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS);
    case ncclFuncReduceScatter:
      return applyCostModelReduceScatter(ctx, nBytes, 1, regBuff, t, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS);
    default:
      return 0;
  }
}

static void runOne(ncclFunc_t coll, TunerContext* ctx, size_t nBytes) {
  float** t = allocCostTable(coll, ctx->nRanks, ctx->nNodes);
  int n = applyModel(coll, ctx, nBytes, 0, t);
  int a = -1, p = -1; float ti = 0.0f;
  char sz[16];
  formatBytes(nBytes, sz, sizeof(sz));
  if (n > 0 && findWinner(t, &a, &p, &ti)) {
    printf("  nNodes=%-5zu nRanks=%-6zu nBytes=%-7s -> algo=%-9s proto=%-7s time_us=%.2f\n",
           ctx->nNodes, ctx->nRanks, sz,
           algorithmToString(a), protocolToString(p), ti);
  } else {
    printf("  nNodes=%-5zu nRanks=%-6zu nBytes=%-7s -> (no viable cell)\n",
           ctx->nNodes, ctx->nRanks, sz);
  }
  freeCostTable(t);
}

// Machine-readable variant: every priced cell, then the winner.
static void runOneAllCells(ncclFunc_t coll, TunerContext* ctx, size_t nBytes, int regBuff) {
  float** tbl = allocCostTable(coll, ctx->nRanks, ctx->nNodes);
  int n = applyModel(coll, ctx, nBytes, regBuff, tbl);
  float (*t)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])tbl;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      float v = t[a][p];
      if (v == (float)NCCL_ALGO_PROTO_IGNORE || v >= 1e29f) continue;
      printf("cell,%s,%zu,%zu,%d,%zu,%s,%s,%.9g\n", collTypeToString(coll), ctx->nRanks,
             ctx->nNodes, regBuff, nBytes, algorithmToString(a), protocolToString(p), v);
    }
  }
  int a = -1, p = -1; float ti = 0.0f;
  if (n > 0 && findWinner(tbl, &a, &p, &ti)) {
    printf("winner,%s,%zu,%zu,%d,%zu,%s,%s,%.9g\n", collTypeToString(coll), ctx->nRanks,
           ctx->nNodes, regBuff, nBytes, algorithmToString(a), protocolToString(p), ti);
  }
  freeCostTable(tbl);
}

typedef struct { size_t nRanks; size_t nNodes; } Topology;

static void initContext(TunerContext* ctx, const Topology* topo, const char* profile) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->nRanks = topo->nRanks;
  ctx->nNodes = topo->nNodes;
  ctx->logFunction = NULL;
  ctx->costModelProfile = profile;
  ctx->costModel = *costModelProfileParams(profile);
  ctx->useCostModel = 1;
  ctx->numConfigs = 0;
  ctx->configs = NULL;
}

static void runCollective(ncclFunc_t coll,
                          const Topology* topos, size_t nTopos,
                          const size_t* sizes, size_t nSizes,
                          const char* profile, int regBuff, int allCells) {
  if (!allCells) {
    printf("\n=================================================\n");
    printf("  Collective: %s\n", collTypeToString(coll));
    printf("=================================================\n");
  }
  for (size_t ti = 0; ti < nTopos; ti++) {
    TunerContext ctx;
    initContext(&ctx, &topos[ti], profile);

    if (!allCells) {
      printf("\n--- topology: nRanks=%zu nNodes=%zu (ppn=%zu) ---\n",
             ctx.nRanks, ctx.nNodes, ctx.nNodes ? ctx.nRanks / ctx.nNodes : 0);
    }
    for (size_t si = 0; si < nSizes; si++) {
      if (allCells) runOneAllCells(coll, &ctx, sizes[si], regBuff);
      else runOne(coll, &ctx, sizes[si]);
    }
  }
}

static int parseColl(const char* s, ncclFunc_t* out) {
  if (strcmp(s, "allreduce") == 0) { *out = ncclFuncAllReduce; return 1; }
  if (strcmp(s, "allgather") == 0) { *out = ncclFuncAllGather; return 1; }
  if (strcmp(s, "reducescatter") == 0) { *out = ncclFuncReduceScatter; return 1; }
  return 0;
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--profile mnnvl|rail_nop2p] [--topo NRANKS:NNODES]... "
          "[--coll NAME]... [--regbuff 0|1] [--min-bytes N] [--max-bytes N] [--all-cells]\n",
          argv0);
}

#define MAX_CLI_ITEMS 64

int main(int argc, char** argv) {
  // Topology sweep in NCCL's view (one NVLink domain == one NCCL node,
  // ppn up to 64 in a full GB200F NVL domain).  nNodes=1 sweeps the
  // intra-domain fill levels; nNodes>=2 exercises the inter-domain NET path.
  const Topology defaultTopos[] = {
    {     8,    1 },    // 2 physical nodes inside one NVL domain
    {    16,    1 },    // 4 physical
    {    32,    1 },    // 8 physical
    {    64,    1 },    // 16 physical (one full NVL domain)
    {   128,    2 },    // 32 physical = 2 NVL domains
    {   256,    4 },    // 64 physical
    {   512,    8 },    // 128 physical
    {  1024,   16 },    // 256 physical
    {  2048,   32 },    // 512 physical
    {  4096,   64 },    // 1024 physical
  };
  const size_t allSizes[] = {
    1ULL << 12,    // 4KB
    1ULL << 13,    // 8KB
    1ULL << 14,    // 16KB
    1ULL << 15,    // 32KB
    1ULL << 16,    // 64KB
    1ULL << 17,    // 128KB
    1ULL << 18,    // 256KB
    1ULL << 19,    // 512KB
    1ULL << 20,    // 1MB
    1ULL << 21,    // 2MB
    1ULL << 22,    // 4MB
    1ULL << 23,    // 8MB
    1ULL << 24,    // 16MB
    1ULL << 25,    // 32MB
    1ULL << 26,    // 64MB
    1ULL << 27,    // 128MB
    1ULL << 28,    // 256MB
    1ULL << 29,    // 512MB
    1ULL << 30,    // 1GB
    1ULL << 31,    // 2GB
    1ULL << 32,    // 4GB
    1ULL << 33,    // 8GB
    1ULL << 34,    // 16GB
  };
  const ncclFunc_t defaultColls[] = {
    ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter,
  };

  const char* profile = "mnnvl";
  int regBuff = 0, allCells = 0;
  unsigned long long minBytes = 0, maxBytes = ~0ULL;
  Topology topos[MAX_CLI_ITEMS];
  size_t nTopos = 0;
  ncclFunc_t colls[MAX_CLI_ITEMS];
  size_t nColls = 0;

  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];
    const char* val = (i + 1 < argc) ? argv[i + 1] : NULL;
    if (strcmp(arg, "--all-cells") == 0) { allCells = 1; continue; }
    if (!val) { usage(argv[0]); return 2; }
    if (strcmp(arg, "--profile") == 0) {
      if (!costModelProfileParams(val)) { fprintf(stderr, "unknown profile %s\n", val); return 2; }
      profile = val;
    } else if (strcmp(arg, "--topo") == 0) {
      unsigned long r = 0, n = 0;
      if (sscanf(val, "%lu:%lu", &r, &n) != 2 || r == 0 || n == 0 || nTopos >= MAX_CLI_ITEMS) {
        fprintf(stderr, "bad --topo %s (want NRANKS:NNODES)\n", val); return 2;
      }
      topos[nTopos].nRanks = r; topos[nTopos].nNodes = n; nTopos++;
    } else if (strcmp(arg, "--coll") == 0) {
      if (nColls >= MAX_CLI_ITEMS || !parseColl(val, &colls[nColls])) {
        fprintf(stderr, "bad --coll %s\n", val); return 2;
      }
      nColls++;
    } else if (strcmp(arg, "--regbuff") == 0) {
      regBuff = atoi(val) ? 1 : 0;
    } else if (strcmp(arg, "--min-bytes") == 0) {
      minBytes = strtoull(val, NULL, 0);
    } else if (strcmp(arg, "--max-bytes") == 0) {
      maxBytes = strtoull(val, NULL, 0);
    } else {
      usage(argv[0]); return 2;
    }
    i++;
  }

  if (nTopos == 0) {
    nTopos = sizeof(defaultTopos) / sizeof(defaultTopos[0]);
    memcpy(topos, defaultTopos, sizeof(defaultTopos));
  }
  if (nColls == 0) {
    nColls = sizeof(defaultColls) / sizeof(defaultColls[0]);
    memcpy(colls, defaultColls, sizeof(defaultColls));
  }
  size_t sizes[sizeof(allSizes) / sizeof(allSizes[0])];
  size_t nSizes = 0;
  for (size_t i = 0; i < sizeof(allSizes) / sizeof(allSizes[0]); i++) {
    if (allSizes[i] >= minBytes && allSizes[i] <= maxBytes) sizes[nSizes++] = allSizes[i];
  }

  if (!allCells) {
    if (strcmp(profile, "mnnvl") == 0) {
      printf("MRC tuner cost-model sweep (GB200F defaults)\n");
      printf("--------------------------------------------\n");
    } else {
      printf("MRC tuner cost-model sweep (profile %s, regBuff=%d)\n", profile, regBuff);
      printf("--------------------------------------------\n");
    }
  }
  for (size_t ci = 0; ci < nColls; ci++) {
    runCollective(colls[ci], topos, nTopos, sizes, nSizes, profile, regBuff, allCells);
  }
  return 0;
}
