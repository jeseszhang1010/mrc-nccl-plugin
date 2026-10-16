// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Standalone host-side driver for the MRC tuner CSV path.
 * Exercises the CSV-match logic in pluginGetCollInfo, so each row prints
 * the CSV pinned (algo, proto) — or "(no match)" when no CSV rule applies.
 *
 * Builds without CUDA / NCCL by including plugin.c directly.
 * To run with a custom CSV, set CSV_CONFIG_PATH or pass the path as the first argument.
 * 
 * Example usage:
 *   $ make test
 *   $ ./csv_tuning_main nccl_tuner.conf
 */

#define NCCL_TUNER_TEST_MAIN 1
#include "../plugin.c"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CSV_CONFIG_PATH
#define CSV_CONFIG_PATH "nccl_tuner.conf"
#endif

static void formatBytes(size_t b, char* out, size_t n) {
  if (b >= (1ULL << 30))      snprintf(out, n, "%.1fGB", b / (double)(1ULL << 30));
  else if (b >= (1ULL << 20)) snprintf(out, n, "%.1fMB", b / (double)(1ULL << 20));
  else if (b >= (1ULL << 10)) snprintf(out, n, "%.1fKB", b / (double)(1ULL << 10));
  else                        snprintf(out, n, "%zuB",   b);
}

// Allocate an NCCL_NUM_ALGORITHMS x NCCL_NUM_PROTOCOLS table.  Cells start at
// a large sentinel; the CSV path pins the matched cell to 0.0.  No cells are
// marked IGNORE here because the CSV may legitimately target any (algo, proto).
//
// NCCL core declares this as a contiguous float[NCCL_NUM_ALGORITHMS]
// [NCCL_NUM_PROTOCOLS] (see nccl/src/enqueue.cc) and passes it as float**.
// Allocate it the same way (a single contiguous block) so the plugin's 2D
// indexing matches production.
static float** allocCostTable(void) {
  float* flat = (float*)malloc(NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS * sizeof(float));
  float (*t)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])flat;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) t[a][p] = 1e30f;
  return (float**)flat;
}

static void freeCostTable(float** t) {
  free(t);
}

// A CSV match pins one cell to 0.0; find that cell.
static int findPinned(float** tbl, int* outAlgo, int* outProto) {
  float (*t)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])tbl;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      if (t[a][p] == 0.0f) { *outAlgo = a; *outProto = p; return 1; }
    }
  }
  return 0;
}

static void runOne(ncclFunc_t coll, TunerContext* ctx, size_t nBytes) {
  float** t = allocCostTable();
  int nChannels = 0;
  ncclResult_t r = pluginGetCollInfo(ctx, coll, nBytes,
                                     /*numPipeOps=*/1, t,
                                     NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                                     /*regBuff=*/0, &nChannels);
  int a = -1, p = -1;
  char sz[16];
  formatBytes(nBytes, sz, sizeof(sz));
  if (r == ncclSuccess && findPinned(t, &a, &p)) {
    printf("  nNodes=%-5zu nRanks=%-6zu nBytes=%-7s -> algo=%-9s proto=%-7s channels=%d\n",
           ctx->nNodes, ctx->nRanks, sz,
           algorithmToString(a), protocolToString(p), nChannels);
  } else {
    printf("  nNodes=%-5zu nRanks=%-6zu nBytes=%-7s -> (no CSV match)\n",
           ctx->nNodes, ctx->nRanks, sz);
  }
  freeCostTable(t);
}

typedef struct { size_t nRanks; size_t nNodes; } Topology;

static void runCollective(ncclFunc_t coll, TunerContext* base,
                          const Topology* topos, size_t nTopos,
                          const size_t* sizes, size_t nSizes) {
  printf("\n=================================================\n");
  printf("  Collective: %s\n", collTypeToString(coll));
  printf("=================================================\n");
  for (size_t ti = 0; ti < nTopos; ti++) {
    base->nRanks = topos[ti].nRanks;
    base->nNodes = topos[ti].nNodes;
    printf("\n--- topology: nRanks=%zu nNodes=%zu (ppn=%zu) ---\n",
           base->nRanks, base->nNodes,
           base->nNodes ? base->nRanks / base->nNodes : 0);
    for (size_t si = 0; si < nSizes; si++) {
      runOne(coll, base, sizes[si]);
    }
  }
}

int main(int argc, char** argv) {
  const char* cfg = (argc > 1) ? argv[1] : CSV_CONFIG_PATH;
  printf("MRC tuner CSV sweep\n");
  printf("config file: %s\n", cfg);
  printf("--------------------------------------------------\n");

  TunerContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.logFunction = NULL;

  if (loadConfig(&ctx, cfg) != ncclSuccess) {
    fprintf(stderr, "Failed to load config file: %s\n", cfg);
    return 1;
  }
  printf("loaded %d CSV rules\n", ctx.numConfigs);

  // Topologies in NCCL's view (NVL domain == 1 NCCL node, ppn up to 64 on
  // GB200F).  Mirrors the (nNodes, nRanks) grid used by nccl_tuner.conf.
  const Topology topos[] = {
    {     8,    1 },
    {    16,    1 },
    {    32,    1 },
    {    64,    1 },
    {   128,    2 },
    {   256,    4 },
    {   512,    8 },
    {  1024,   16 },
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
    runCollective(colls[ci], &ctx, topos, nTopos, sizes, nSizes);
  }

  if (ctx.configs) free(ctx.configs);
  return 0;
}
