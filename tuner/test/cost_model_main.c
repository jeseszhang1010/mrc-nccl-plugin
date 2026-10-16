// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Standalone host-side driver for the MRC tuner cost model.
 * Builds without CUDA / NCCL by including plugin.c directly (so static
 * helpers and TunerContext are visible) and bypassing the v5 plugin symbol.
 * 
 * For each (topology, collective, message size) the driver populates
 * collCostTable via the cost model and prints the winning (algo, proto, time).
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
static float** allocCostTable(ncclFunc_t coll) {
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

static void runOne(ncclFunc_t coll, TunerContext* ctx, size_t nBytes) {
  float** t = allocCostTable(coll);
  int n = 0;
  switch (coll) {
    case ncclFuncAllReduce:
      n = applyCostModelAllReduce(ctx, nBytes, 1, t, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS);
      break;
    case ncclFuncAllGather:
      n = applyCostModelAllGather(ctx, nBytes, 1, t, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS);
      break;
    case ncclFuncReduceScatter:
      n = applyCostModelReduceScatter(ctx, nBytes, 1, t, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS);
      break;
    default: break;
  }
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

typedef struct { size_t nRanks; size_t nNodes; } Topology;

static void runCollective(ncclFunc_t coll,
                          const Topology* topos, size_t nTopos,
                          const size_t* sizes, size_t nSizes) {
  printf("\n=================================================\n");
  printf("  Collective: %s\n", collTypeToString(coll));
  printf("=================================================\n");
  for (size_t ti = 0; ti < nTopos; ti++) {
    TunerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.nRanks = topos[ti].nRanks;
    ctx.nNodes = topos[ti].nNodes;
    ctx.logFunction = NULL;
    ctx.costModel = gb200fParamsDefault;
    ctx.useCostModel = 1;
    ctx.numConfigs = 0;
    ctx.configs = NULL;

    printf("\n--- topology: nRanks=%zu nNodes=%zu (ppn=%zu) ---\n",
           ctx.nRanks, ctx.nNodes, ctx.nNodes ? ctx.nRanks / ctx.nNodes : 0);
    for (size_t si = 0; si < nSizes; si++) {
      runOne(coll, &ctx, sizes[si]);
    }
  }
}

int main(void) {
  printf("MRC tuner cost-model sweep (GB200F defaults)\n");
  printf("--------------------------------------------\n");

  // Topology sweep in NCCL's view (one NVLink domain == one NCCL node,
  // ppn up to 64 in a full GB200F NVL domain).  nNodes=1 sweeps the
  // intra-domain fill levels; nNodes>=2 exercises the inter-domain NET path.
  const Topology topos[] = {
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
  const size_t sizes[] = {
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
  const ncclFunc_t colls[] = {
    ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter,
  };

  const size_t nTopos = sizeof(topos) / sizeof(topos[0]);
  const size_t nSizes = sizeof(sizes) / sizeof(sizes[0]);
  const size_t nColls = sizeof(colls) / sizeof(colls[0]);
  for (size_t ci = 0; ci < nColls; ci++) {
    runCollective(colls[ci], topos, nTopos, sizes, nSizes);
  }
  return 0;
}
