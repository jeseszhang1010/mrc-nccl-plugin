// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "tuner.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define __hidden __attribute__ ((visibility("hidden")))
#define MAX_LINE_LENGTH 256

// ============================================================================
// Cost model parameters for the GB200F platform.
//
// The cost model follows NCCL's analytical performance model:
//     time(us) = latency_us * latCount + nBytes / (1000 * algoBw_GBps)
//
// For each (algorithm, protocol) combination supported by AllReduce we
// compute:
//   1. busBw  = nChannels * perCh_bw   (with protocol/algo specific caps)
//   2. algoBw = busBw * ratio          (ratio depends on algorithm)
//   3. latency = baseLat + per-hop latencies (intra + inter)
//   4. time   = latency * latCount + nBytes / (1000 * algoBw)
//
// All bandwidth values are GB/s.  All latency values are microseconds.
// ============================================================================

typedef struct {
  // -- Topology / channel assumptions used by the cost model -----------------
  int    gpusPerNode;          // ranks per NCCL node (NVL-domain size)
  int    nChannels;            // default channels per algorithm
  // -- Per-channel bandwidth (GB/s) -----------------------------------------
  double bwIntra;              // NVLink intra-domain, per channel
  double bwInter;              // Inter-domain (backend MRC), per channel
  // -- NVLS aggregate efficiency (Blackwell default 0.74) -------------------
  double nvlsEfficiency;
  // -- Graph latency added per inter-node hop (us) --------------------------
  double graphLatencyInter;
  // -- Base / hw latencies (us) :  hw is 0=NVLINK, 1=PCI, 2=NET --------------
  double baseLatencies[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  double hwLatencies[NCCL_NUM_HW_LINKS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // -- Bandwidth caps : indexed by tuning scale (0=N1, 1=N2, 2=N4+) ---------
  double llMaxBw[NCCL_NUM_TUNING_SCALES];
  double perChMaxRingLL128Bw[NCCL_NUM_TUNING_SCALES];
  double perChMaxTreeBw[NCCL_NUM_TUNING_SCALES];
  double perChMaxTreeLL128Bw[NCCL_NUM_TUNING_SCALES];
  double perChMaxNVLSTreeBw[NCCL_NUM_TUNING_SCALES];
  // -- Ring inter-domain per-channel BW cap (GB/s) --------------------------
  // When NCCL nNodes >= 2 a ring must traverse the backend NET link, whose
  // effective per-channel bandwidth is the ring bottleneck.  Without this cap
  // the model lets ring outrun NVLS_TREE at large multi-domain sizes, which
  // measurements contradict (NVLS_TREE is the high-BW inter-domain path).
  double ringBwCapInter;
  // -- Ring intra-domain (single NVL domain) SIMPLE BW cap (GB/s) -----------
  // On one NVL domain AllReduce, NVLS (in-network reduction) beats ring at
  // bandwidth-bound sizes.  Cap ring/SIMPLE per-channel BW so NVLS wins large
  // messages; ring/LL and ring/LL128 (small/mid regime) are left untouched.
  double ringBwCapIntra;
  // -- PAT (AllGather/ReduceScatter, 1 GPU per node) ------------------------
  // busBw efficiency applied to nChannels * per-channel BW (NCCL uses 0.75).
  // 0 disables PAT modelling.
  double patBwEff;
  // Linear per-rank PAT latency term in us (NCCL uses 2.8).
  double patLatPerRank;
  // -- Registered-buffer BW multipliers -------------------------------------
  // busBw is multiplied by regBwScale[algo][proto] when NCCL reports
  // regBuff=1.  0 (unset) is treated as 1.0.
  double regBwScale[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // -- Cell coverage --------------------------------------------------------
  // When non-zero, valid (non-IGNORE) cells the model cannot price get a large
  // explicit cost, so NCCL never compares plugin-predicted times against its
  // own internal estimates.
  int disableUnmodeledCells;
} CostModelParams;

// Cost written to valid cells the model cannot price when
// disableUnmodeledCells is set (us).  Large enough to never win against a
// priced cell, but distinct from NCCL_ALGO_PROTO_IGNORE.
#define COST_MODEL_DISABLED_COST 1.0e9f

// Default cost-model parameters for the GB200F platform.
//
// NCCL collapses an entire NVLink domain (up to 16 physical nodes = 64 GPUs)
// into a single "node" on GB200F.  Multi-node NVLink is used inside a domain
// instead of the backend MRC link, so the cost model only crosses NET when
// NCCL nNodes >= 2 (>16 physical nodes).  There is therefore no separate
// "small NET" regime to model — a single hwLatencies[NET] table suffices.
//
// Latencies and per-channel bandwidths were fit against measured nccl-tests
// AllReduce data at physical 2..256 nodes (4 GPUs per physical node).
// Calibration script and dataset live under test/validate.
static const CostModelParams gb200fParamsDefault = {
  /* gpusPerNode        */ 64,    /* NCCL node = one NVL domain = 64 GPUs */
  /* nChannels          */ 32,
  /* bwIntra            */ 20.29,
  /* bwInter            */ 28.19,
  /* nvlsEfficiency     */ 0.4674,
  /* graphLatencyInter  */ 0.0,
  /* baseLatencies */
  {
    /* tree           */ { 9.025, 27.0, 20.83 },
    /* ring           */ { 16.40, 20.29,  6.237 },
    /* collnet_direct */ {  0.0,   0.0,   0.0 },  /* SHARP unused on this cluster */
    /* collnet_chain  */ {  0.0,   0.0,   0.0 },  /* SHARP unused on this cluster */
    /* nvls           */ {  0.0,   0.0,  70.78 },
    /* nvls_tree      */ {  0.0,   0.0, 102.0 },   /* fit; only Simple measured */
    /* pat            */ {  8.0,   8.0,   8.0 },
  },
  /* hwLatencies[hw][algo][proto] */
  {
    /* NVLINK  (intra-NVL-domain per-step latency; fit from N1 a_obs slope) */
    {
      /* tree           */ { 1.402,  1.393,  5.703 },
      /* ring           */ { 0.8647, 1.546,  6.224 },
      /* collnet_direct */ { 0.0,    0.0,    0.0 },  /* SHARP unused on this cluster */
      /* collnet_chain  */ { 0.0,    0.0,    0.0 },  /* SHARP unused on this cluster */
      /* nvls           */ { 0.0,    0.0,   42.0 },  /* constant; no per-step formula */
      /* nvls_tree      */ { 0.0,    0.0,   39.36 },
      /* pat            */ { 0.0,    0.0,    4.0 },
    },
    /* PCI */
    {
      /* tree           */ { 1.0, 1.9, 4.0 },
      /* ring           */ { 1.0, 2.5, 5.7 },
      /* collnet_direct */ { 0.0, 0.0, 0.0 },  /* SHARP unused on this cluster */
      /* collnet_chain  */ { 0.0, 0.0, 0.0 },  /* SHARP unused on this cluster */
      /* nvls           */ { 0.0, 0.0, 0.0 },
      /* nvls_tree      */ { 0.0, 0.0, 0.0 },
      /* pat            */ { 0.0, 0.0, 4.0 },
    },
    /* NET  (used iff NCCL nNodes >= 2; fit against multi-NVL-domain data) */
    {
      /* tree           */ { 22.41,  20.91,  25.16 },
      /* ring           */ {267.7,  281.0,  530.8 },  /* high: contention beyond ~8 nodes */
      /* collnet_direct */ {  0.0,    0.0,   31.0 },  /* SHARP unused on this cluster */
      /* collnet_chain  */ {  0.0,    0.0,   30.0 },  /* SHARP unused on this cluster */
      /* nvls           */ {  0.0,    0.0,   38.83 },
      /* nvls_tree      */ {  0.0,    0.0,   27.59 },
      /* pat            */ {  0.0,    0.0,   14.0 },
    },
  },
  /* llMaxBw            */ { 204.6, 134.1, 120.5 },   /* Blackwell N1/N2/N4 */
  /* perChMaxRingLL128Bw*/ { 124.9, 175.2,  16.35 },
  /* perChMaxTreeBw     */ { 222.7, 206.4,  17.59 },
  /* perChMaxTreeLL128Bw*/ {  56.54, 38.33, 64.31 },
  /* perChMaxNVLSTreeBw */ {  55.62,145.7, 137.3 },
  /* ringBwCapInter     */ 16.0,
  /* ringBwCapIntra     */ 17.5,
};

// Rail profile: MNNVL disabled (every physical node is its own NCCL node) and
// NCCL_P2P_DISABLE=1 (intra-node traffic also goes over the MRC NET).  The
// block between the markers is owned by tuner/autotune: the fit writes its
// parameters, and `railNoP2pParamsValidated` records whether that fit passed
// the acceptance gate.  The checked-in seed is NOT fitted: latencies come from
// NCCL 2.30's own Blackwell tuning constants (NVLink-intra hop replaced by the
// NET hop because P2P is disabled), and bandwidths from the 2026-10-15 nightly
// 128-node large-message busbw.
// BEGIN AUTOTUNE PARAMS rail_nop2p
// Generated by tuner/autotune (autotune codegen). Do not edit between the
// markers by hand; re-run the fit + codegen instead.
// Source: fit_result.json (2026-10-20 15:25 UTC); data nNodes=[8, 16, 32, 64, 128, 256]
// Fit: final top1=0.935 mean_regret=0.0044 violations=22/336
// Gate nNodes=[128, 256] nvls=[0]: FAIL (6 violations, 0 anomaly-exempt)
// Set to 1 by codegen only when the fit passed the acceptance gate.  While 0,
// the profile is used only when selected explicitly via
// NCCL_TUNER_COST_MODEL_PROFILE.
static const int railNoP2pParamsValidated = 1;
static const CostModelParams railNoP2pParamsDefault = {
  /* gpusPerNode        */ 4,
  /* nChannels          */ 32,
  /* bwIntra            */ 20.29,
  /* bwInter            */ 4.29211,
  /* nvlsEfficiency     */ 0.4674,
  /* graphLatencyInter  */ 0.0,
  /* baseLatencies [algo][LL, LL128, SIMPLE] */
  {
    /* tree           */ {   20.6679,   26.3192,   25.6124 },
    /* ring           */ { 1.91189e-09, 2.05101e-08, 1.09204e-10 },
    /* collnet_direct */ {       0.0,       0.0,       0.0 },
    /* collnet_chain  */ {       0.0,       0.0,       0.0 },
    /* nvls           */ {       0.0,       0.0,       0.0 },
    /* nvls_tree      */ {       0.0,       0.0,       0.0 },
    /* pat            */ {       8.0,       8.0,   32.5934 },
  },
  /* hwLatencies [hw][algo][LL, LL128, SIMPLE] */
  {
    /* NVLINK */
    {
      /* tree           */ {       5.0,       8.5,      14.0 },
      /* ring           */ {       2.7,       4.0,      14.0 },
      /* collnet_direct */ {       0.0,       0.0,       0.0 },
      /* collnet_chain  */ {       0.0,       0.0,       0.0 },
      /* nvls           */ {       0.0,       0.0,      25.0 },
      /* nvls_tree      */ {       0.0,       0.0,      25.0 },
      /* pat            */ {       0.0,       0.0,       4.0 },
    },
    /* PCI */
    {
      /* tree           */ {       1.0,       1.9,       4.0 },
      /* ring           */ {       1.0,       2.5,       5.7 },
      /* collnet_direct */ {       0.0,       0.0,       3.7 },
      /* collnet_chain  */ {       0.0,       0.0,       2.8 },
      /* nvls           */ {       0.0,       0.0,       0.0 },
      /* nvls_tree      */ {       0.0,       0.0,       0.0 },
      /* pat            */ {       0.0,       0.0,       4.0 },
    },
    /* NET */
    {
      /* tree           */ {   10.1576,   10.7845,   17.4627 },
      /* ring           */ {       2.7,       4.0,      14.0 },
      /* collnet_direct */ {       0.0,       0.0,      31.0 },
      /* collnet_chain  */ {       0.0,       0.0,      30.0 },
      /* nvls           */ {       0.0,       0.0,      18.0 },
      /* nvls_tree      */ {       0.0,       0.0,      20.9 },
      /* pat            */ {       0.0,       0.0,   6.64092 },
    },
  },
  /* llMaxBw             N1/N2/N4+ */ {     282.0,      90.0,   4.99782 },
  /* perChMaxRingLL128Bw N1/N2/N4+ */ {      40.0,      40.0,    5.8313 },
  /* perChMaxTreeBw      N1/N2/N4+ */ {      70.0,      42.8,    2.1473 },
  /* perChMaxTreeLL128Bw N1/N2/N4+ */ {      55.6,     31.67,  0.890063 },
  /* perChMaxNVLSTreeBw  N1/N2/N4+ */ {     55.62,     145.7,     137.3 },
  /* ringBwCapInter     */ 0.0,
  /* ringBwCapIntra     */ 0.0,
  /* patBwEff           */ 0.468895,
  /* patLatPerRank      */ 1.13602,
  /* regBwScale [algo][LL, LL128, SIMPLE] (regBuff=1 busBw multiplier) */
  {
    /* tree           */ {       1.0,       1.0,       1.0 },
    /* ring           */ {       1.0,       1.0,  0.892386 },
    /* collnet_direct */ {       1.0,       1.0,       1.0 },
    /* collnet_chain  */ {       1.0,       1.0,       1.0 },
    /* nvls           */ {       1.0,       1.0,       1.0 },
    /* nvls_tree      */ {       1.0,       1.0,       1.0 },
    /* pat            */ {       1.0,       1.0,       1.0 },
  },
  /* disableUnmodeledCells */ 1,
};
// END AUTOTUNE PARAMS rail_nop2p

// CSV field indices for configuration parsing
// Format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
#define CONFIG_FIELD_COLLTYPE     0
#define CONFIG_FIELD_MINBYTES     1
#define CONFIG_FIELD_MAXBYTES     2
#define CONFIG_FIELD_ALGORITHM    3
#define CONFIG_FIELD_PROTOCOL     4
#define CONFIG_FIELD_CHANNELS     5
#define CONFIG_FIELD_NNODES       6
#define CONFIG_FIELD_NRANKS       7
#define CONFIG_FIELD_PIPEOPS      8  // Optional field
#define CONFIG_FIELD_REGBUFF      9  // Optional field

// Field count constants
#define CONFIG_FIELDS_REQUIRED    8   // Minimum required fields (up to nRanks)
#define CONFIG_FIELDS_WITH_PIPEOPS 9  // Fields including numPipeOps
#define CONFIG_FIELDS_WITH_REGBUFF 10 // Fields including both numPipeOps and regBuff
#define CONFIG_FIELDS_MAX         10  // Maximum number of fields supported

typedef struct {
  ncclFunc_t collType;
  size_t minBytes;
  size_t maxBytes;
  int algorithm;
  int protocol;
  int nChannels;
  int nNodes;
  int nRanks;
  int numPipeOps;
  int regBuff;
} TuningConfig;

typedef struct {
  TuningConfig* configs;
  int numConfigs;
  int maxConfigs;
  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
  ncclNvlDomainInfo_v5_t nvlDomainInfo;
  // Cost-model state.  When `useCostModel` is non-zero the plugin overrides
  // collCostTable entries with model-predicted times instead of
  // matching configs from nccl_tuner.conf.  See initCostModel() for the
  // environment variables that control this behaviour.
  int useCostModel;
  // When `forceCostModel` is non-zero the plugin skips the CSV config lookup
  // entirely and uses the cost model directly (still subject to useCostModel).
  int forceCostModel;
  CostModelParams costModel;
  // Name of the active cost-model profile ("mnnvl" or "rail_nop2p").
  const char* costModelProfile;
} TunerContext;

// Parse collective type from string
static ncclFunc_t parseCollType(const char* str) {
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  return ncclFuncAllReduce; // default
}

// Convert collective type to string
static const char* collTypeToString(ncclFunc_t collType) {
  switch (collType) {
    case ncclFuncBroadcast: return "broadcast";
    case ncclFuncReduce: return "reduce";
    case ncclFuncAllGather: return "allgather";
    case ncclFuncReduceScatter: return "reducescatter";
    case ncclFuncAllReduce: return "allreduce";
    default: return "unknown";
  }
}

// Parse algorithm from string
static int parseAlgorithm(const char* str) {
  if (strcmp(str, "tree") == 0) return NCCL_ALGO_TREE;
  if (strcmp(str, "ring") == 0) return NCCL_ALGO_RING;
  if (strcmp(str, "collnet_direct") == 0) return NCCL_ALGO_COLLNET_DIRECT;
  if (strcmp(str, "collnet_chain") == 0) return NCCL_ALGO_COLLNET_CHAIN;
  if (strcmp(str, "nvls") == 0) return NCCL_ALGO_NVLS;
  if (strcmp(str, "nvls_tree") == 0) return NCCL_ALGO_NVLS_TREE;
  if (strcmp(str, "pat") == 0) return NCCL_ALGO_PAT;
  return NCCL_ALGO_RING; // default
}

// Convert algorithm to string
static const char* algorithmToString(int algorithm) {
  switch (algorithm) {
    case NCCL_ALGO_TREE: return "tree";
    case NCCL_ALGO_RING: return "ring";
    case NCCL_ALGO_COLLNET_DIRECT: return "collnet_direct";
    case NCCL_ALGO_COLLNET_CHAIN: return "collnet_chain";
    case NCCL_ALGO_NVLS: return "nvls";
    case NCCL_ALGO_NVLS_TREE: return "nvls_tree";
    case NCCL_ALGO_PAT: return "pat";
    default: return "unknown";
  }
}

// Parse protocol from string
static int parseProtocol(const char* str) {
  if (strcmp(str, "ll") == 0) return NCCL_PROTO_LL;
  if (strcmp(str, "ll128") == 0) return NCCL_PROTO_LL128;
  if (strcmp(str, "simple") == 0) return NCCL_PROTO_SIMPLE;
  return NCCL_PROTO_SIMPLE; // default
}

// Convert protocol to string
static const char* protocolToString(int protocol) {
  switch (protocol) {
    case NCCL_PROTO_LL: return "ll";
    case NCCL_PROTO_LL128: return "ll128";
    case NCCL_PROTO_SIMPLE: return "simple";
    default: return "unknown";
  }
}

// Helper function to count valid configuration lines in file
static int countConfigLines(const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) return 0;

  char line[MAX_LINE_LENGTH];
  int count = 0;

  while (fgets(line, sizeof(line), file)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    line[strcspn(line, "\n")] = 0;
    if (strlen(line) > 0) count++;
  }

  fclose(file);
  return count;
}

// Load configuration from file
static ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/MRC: Config file %s not found, using defaults", filename);
    }
    return ncclSuccess;
  }

  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/MRC: No valid configurations found in %s", filename);
    }
    fclose(file);
    return ncclSuccess;
  }

  ctx->configs = (TuningConfig*)malloc(configCount * sizeof(TuningConfig));
  if (!ctx->configs) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/MRC: Failed to allocate memory for %d configurations", configCount);
    }
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  fseek(file, 0, SEEK_SET);

  char line[MAX_LINE_LENGTH];

  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    if (line[0] == '#' || line[0] == '\n') continue;

    line[strcspn(line, "\n")] = 0;

    char* token;
    char* tokens[CONFIG_FIELDS_MAX];
    int tokenCount = 0;

    char lineCopy[MAX_LINE_LENGTH];
    strncpy(lineCopy, line, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';

    token = strtok(lineCopy, ",");
    while (token != NULL && tokenCount < CONFIG_FIELDS_MAX) {
      while (*token == ' ' || *token == '\t') token++;
      char* end = token + strlen(token) - 1;
      while (end > token && (*end == ' ' || *end == '\t')) {
        *end = '\0';
        end--;
      }
      tokens[tokenCount++] = token;
      token = strtok(NULL, ",");
    }

    if (tokenCount >= CONFIG_FIELDS_REQUIRED && tokenCount <= CONFIG_FIELDS_MAX) {
      TuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = parseCollType(tokens[CONFIG_FIELD_COLLTYPE]);
      config->minBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MINBYTES], NULL, 10);
      config->maxBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MAXBYTES], NULL, 10);
      config->algorithm = parseAlgorithm(tokens[CONFIG_FIELD_ALGORITHM]);
      config->protocol = parseProtocol(tokens[CONFIG_FIELD_PROTOCOL]);
      config->nChannels = atoi(tokens[CONFIG_FIELD_CHANNELS]);
      config->nNodes = atoi(tokens[CONFIG_FIELD_NNODES]);
      config->nRanks = atoi(tokens[CONFIG_FIELD_NRANKS]);

      if (tokenCount >= CONFIG_FIELDS_WITH_PIPEOPS) {
        config->numPipeOps = atoi(tokens[CONFIG_FIELD_PIPEOPS]);
      } else {
        config->numPipeOps = -1;
      }

      if (tokenCount >= CONFIG_FIELDS_WITH_REGBUFF) {
        config->regBuff = atoi(tokens[CONFIG_FIELD_REGBUFF]);
      } else {
        config->regBuff = -1;
      }

      ctx->numConfigs++;

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/MRC: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=%d",
                         collTypeToString(config->collType), config->minBytes, config->maxBytes,
                         algorithmToString(config->algorithm), protocolToString(config->protocol),
                         config->nChannels, config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
      }
    }
  }

  fclose(file);
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/MRC: Loaded %d tuning configurations from %s", ctx->numConfigs, filename);
  }
  return ncclSuccess;
}

// ============================================================================
// Cost model implementation (GB200F / Blackwell-class platform).
// ============================================================================

// Integer floor(log2(n)) for n >= 1.  Returns 0 for n <= 1.
static int costLog2i(int n) {
  int r = 0;
  while (n > 1) { n >>= 1; r++; }
  return r;
}

// Parse a double env var, returning `defaultValue` if not set or invalid.
static double getEnvDouble(const char* name, double defaultValue) {
  const char* s = getenv(name);
  if (!s || !*s) return defaultValue;
  char* end = NULL;
  double v = strtod(s, &end);
  if (end == s) return defaultValue;
  return v;
}

// Parse an int env var, returning `defaultValue` if not set or invalid.
static int getEnvInt(const char* name, int defaultValue) {
  const char* s = getenv(name);
  if (!s || !*s) return defaultValue;
  char* end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s) return defaultValue;
  return (int)v;
}

// Returns non-zero if environment variable `name` is set to exactly `value`.
static int envEquals(const char* name, const char* value) {
  const char* s = getenv(name);
  return s && strcmp(s, value) == 0;
}

// Parameter set for a named cost-model profile, or NULL if unknown.
static const CostModelParams* costModelProfileParams(const char* name) {
  if (strcmp(name, "mnnvl") == 0) return &gb200fParamsDefault;
  if (strcmp(name, "rail_nop2p") == 0) return &railNoP2pParamsDefault;
  return NULL;
}

// Pick the cost-model profile for this communicator.
//   NCCL_TUNER_COST_MODEL_PROFILE=mnnvl|rail_nop2p forces a profile.
//   Otherwise (unset or "auto"), rail_nop2p is used when its parameters were
//   validated by the autotune fit AND the job runs with NCCL_MNNVL_ENABLE=0 and
//   NCCL_P2P_DISABLE=1 across more than one node; mnnvl is used otherwise.
static const char* selectCostModelProfile(TunerContext* ctx) {
  const char* req = getenv("NCCL_TUNER_COST_MODEL_PROFILE");
  if (req && *req && strcmp(req, "auto") != 0) {
    if (costModelProfileParams(req)) return strcmp(req, "mnnvl") == 0 ? "mnnvl" : "rail_nop2p";
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/MRC: Unknown NCCL_TUNER_COST_MODEL_PROFILE=%s, using auto selection", req);
    }
  }
  if (railNoP2pParamsValidated && ctx->nNodes > 1
      && envEquals("NCCL_MNNVL_ENABLE", "0") && envEquals("NCCL_P2P_DISABLE", "1")) {
    return "rail_nop2p";
  }
  return "mnnvl";
}

// Initialise the cost-model state from defaults and environment overrides.
//   NCCL_TUNER_COST_MODEL              : 1=on (default, fallback for AR/AG/RS), 0=off (strict CSV-only)
//   NCCL_TUNER_FORCE_COST_MODEL        : 1=skip CSV lookup and use cost model directly, 0=off (default)
//   NCCL_TUNER_COST_MODEL_PROFILE      : mnnvl | rail_nop2p | auto (default)
//   NCCL_TUNER_COST_MODEL_CHANNELS     : override default channel count
//   NCCL_TUNER_COST_MODEL_BW_INTRA     : per-channel intra-node BW (GB/s)
//   NCCL_TUNER_COST_MODEL_BW_INTER     : per-channel inter-node BW (GB/s)
//   NCCL_TUNER_COST_MODEL_NVLS_EFF     : NVLS efficiency factor (0..1)
//   NCCL_TUNER_COST_MODEL_GRAPH_LAT    : per-hop graph inter latency (us)
static void initCostModel(TunerContext* ctx) {
  ctx->costModelProfile = selectCostModelProfile(ctx);
  ctx->costModel = *costModelProfileParams(ctx->costModelProfile);
  ctx->useCostModel = getEnvInt("NCCL_TUNER_COST_MODEL", 1);
  ctx->forceCostModel = getEnvInt("NCCL_TUNER_FORCE_COST_MODEL", 0);
  // Forcing the cost model implies it must be enabled.
  if (ctx->forceCostModel) ctx->useCostModel = 1;
  ctx->costModel.nChannels         = getEnvInt   ("NCCL_TUNER_COST_MODEL_CHANNELS",  ctx->costModel.nChannels);
  ctx->costModel.bwIntra           = getEnvDouble("NCCL_TUNER_COST_MODEL_BW_INTRA",  ctx->costModel.bwIntra);
  ctx->costModel.bwInter           = getEnvDouble("NCCL_TUNER_COST_MODEL_BW_INTER",  ctx->costModel.bwInter);
  ctx->costModel.nvlsEfficiency    = getEnvDouble("NCCL_TUNER_COST_MODEL_NVLS_EFF",  ctx->costModel.nvlsEfficiency);
  ctx->costModel.graphLatencyInter = getEnvDouble("NCCL_TUNER_COST_MODEL_GRAPH_LAT", ctx->costModel.graphLatencyInter);

  if (ctx->logFunction && ctx->useCostModel) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/MRC: Cost model ENABLED (profile=%s)%s: nChannels=%d bwIntra=%.2fGB/s "
                     "bwInter=%.2fGB/s nvlsEff=%.3f graphLatInter=%.2fus",
                     ctx->costModelProfile,
                     ctx->forceCostModel ? " [FORCED, CSV lookup skipped]" : "",
                     ctx->costModel.nChannels, ctx->costModel.bwIntra,
                     ctx->costModel.bwInter, ctx->costModel.nvlsEfficiency,
                     ctx->costModel.graphLatencyInter);
  }
}

// ---------------------------------------------------------------------------
// Shared helpers (used by AllReduce, AllGather, ReduceScatter cost models).
// ---------------------------------------------------------------------------

// Per-channel bandwidth (GB/s) for a given algo/proto on this topology.
// `nvlsPipelineFactor` is 2.0 for AllReduce (NVLS pipelines RS+AG into one
// call) and 1.0 for AllGather/ReduceScatter (single phase only).
// Returns negative on unsupported NVLS configurations.
static double perChannelBw(const CostModelParams* P, int algo, int proto,
                           int scale, int nNodes, int nChannels,
                           double nvlsPipelineFactor) {
  double bw;
  if (nNodes <= 2) {
    bw = P->bwIntra;
  } else {
    bw = P->bwInter;
  }
  // Ring across NCCL nodes is bottlenecked by the backend NET link.  Cap the
  // per-channel bandwidth so ring does not outrun the inter-domain NVLS_TREE
  // path at large multi-domain message sizes (matches measured busbw).
  if (algo == NCCL_ALGO_RING && nNodes >= 2 && P->ringBwCapInter > 0.0
      && bw > P->ringBwCapInter) {
    bw = P->ringBwCapInter;
  }
  if (algo == NCCL_ALGO_NVLS || algo == NCCL_ALGO_NVLS_TREE) {
    if (proto != NCCL_PROTO_SIMPLE) return -1.0;
    if (nChannels < 2) return -1.0;
    double intraBw = P->bwIntra * P->nvlsEfficiency
                     * (double)(nChannels - 1) / (double)nChannels;
    intraBw *= nvlsPipelineFactor;
    double interBw = P->bwInter
        * ((nNodes <= 2 && algo == NCCL_ALGO_NVLS_TREE) ? 2.0 : 1.0);
    double nvlsCap = (algo == NCCL_ALGO_NVLS_TREE)
        ? P->perChMaxNVLSTreeBw[scale] : 1e30;
    bw = intraBw;
    if (interBw < bw) bw = interBw;
    if (nvlsCap < bw) bw = nvlsCap;
  }
  return bw;
}

// Apply protocol/algorithm bandwidth caps.  These caps mirror NCCL's tuning
// table and apply identically across collectives (LL/LL128 caps are global,
// TREE caps only matter when TREE is used — i.e., AllReduce).
static double applyProtocolCap(double busBw, int algo, int proto,
                               const CostModelParams* P, int scale,
                               int nNodes, int nChannels) {
  if (algo == NCCL_ALGO_RING && proto == NCCL_PROTO_LL) {
    double cand = busBw * 0.5;
    busBw = (P->llMaxBw[scale] < cand) ? P->llMaxBw[scale] : cand;
  } else if (algo == NCCL_ALGO_RING && proto == NCCL_PROTO_LL128) {
    // LL128 trades bandwidth for latency.  At large message sizes its
    // achievable ring bandwidth falls well below plain SIMPLE (measured 10-18%
    // slower once bandwidth-bound), so use a lower efficiency; small/mid sizes
    // stay latency-bound so this does not change their (LL128-favoured) pick.
    double cand = busBw * 0.78;
    double cap  = nChannels * P->perChMaxRingLL128Bw[scale];
    busBw = (cap < cand) ? cap : cand;
  } else if (algo == NCCL_ALGO_TREE) {
    double cand = busBw * 0.92;
    double cap  = nChannels * P->perChMaxTreeBw[scale];
    busBw = (cap < cand) ? cap : cand;
    if (proto == NCCL_PROTO_LL) {
      double c2 = busBw / 3.8;
      busBw = (P->llMaxBw[scale] < c2) ? P->llMaxBw[scale] : c2;
    } else if (proto == NCCL_PROTO_LL128) {
      double factor = (nNodes == 1) ? (7.0 / 9.0) : (120.0 / 128.0);
      double c2 = busBw * factor;
      double cap2 = nChannels * P->perChMaxTreeLL128Bw[scale];
      busBw = (cap2 < c2) ? cap2 : c2;
    }
  }
  return busBw;
}

// Convert busBw to algoBw using the per-(coll,algo) ratio from NCCL.
//   AR  RING/NVLS/NVLS_TREE : n/(2(n-1))     TREE : 0.5
//   AG  RING/NVLS           : n/(n-1)
//   RS  RING/NVLS           : n/(n-1)
// Returns negative for combinations not supported by this cost model.
static double busBwToAlgoBw(double busBw, int algo, ncclFunc_t coll, int nRanks) {
  if (nRanks <= 1) return busBw;
  double ratio = 0.0;
  if (coll == ncclFuncAllReduce) {
    if (algo == NCCL_ALGO_RING || algo == NCCL_ALGO_NVLS || algo == NCCL_ALGO_NVLS_TREE) {
      ratio = (double)nRanks / (double)(2 * (nRanks - 1));
    } else if (algo == NCCL_ALGO_TREE) {
      ratio = 0.5;
    } else {
      return -1.0;
    }
  } else if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
    if (algo == NCCL_ALGO_RING || algo == NCCL_ALGO_NVLS) {
      ratio = (double)nRanks / (double)(nRanks - 1);
    } else {
      return -1.0;
    }
  } else {
    return -1.0;
  }
  return busBw * ratio;
}

// Compute the latency component (us) for `coll` with `(algo, proto)` on the
// given topology.  Latency excludes the data-transfer term.  Returns negative
// if the (coll, algo) combination is not supported.
static double algoLatency(const CostModelParams* P, int algo, int proto,
                          ncclFunc_t coll, int nRanks, int nNodes, int ppn) {
  double baseLat = P->baseLatencies[algo][proto];
  double intraLat = P->hwLatencies[NCCL_HW_NVLINK][algo][proto];
  int algoForNet = (ppn == 1) ? NCCL_ALGO_TREE : algo;
  double interLat = P->hwLatencies[NCCL_HW_NET][algoForNet][proto];
  interLat += P->graphLatencyInter;
  if (proto == NCCL_PROTO_SIMPLE) interLat += P->graphLatencyInter;

  int nsteps = 0, nInter = 0;
  if (coll == ncclFuncAllReduce) {
    nsteps = 2 * (nRanks - 1);
    nInter = (nNodes == 1) ? 0 : 2 * (nNodes - 1);
  } else if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
    nsteps = (nRanks - 1);
    nInter = (nNodes == 1) ? 0 : (nNodes - 1);
  } else {
    return -1.0;
  }

  double lat = baseLat;
  if (algo == NCCL_ALGO_RING) {
    lat += (double)(nsteps - nInter) * intraLat + (double)nInter * interLat;
  } else if (algo == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) {
    int log2nNodes = costLog2i(nNodes);
    lat += 2.0 * ((double)(ppn - 1) * intraLat + (double)log2nNodes * interLat);
  } else if (algo == NCCL_ALGO_NVLS) {
    lat = intraLat;
    if (nNodes > 1) lat += interLat;
  } else if (algo == NCCL_ALGO_NVLS_TREE && coll == ncclFuncAllReduce) {
    int log2nNodes = costLog2i(nNodes);
    lat += intraLat + 2.0 * (double)log2nNodes * interLat;
  } else {
    return -1.0;
  }
  return lat;
}

// busBw multiplier for registered user buffers (regBuff=1).
static double regBwScaleFor(const CostModelParams* P, int algo, int proto, int regBuff) {
  if (!regBuff) return 1.0;
  double s = P->regBwScale[algo][proto];
  return (s > 0.0) ? s : 1.0;
}

// Compute predicted time (microseconds) for an AllReduce of `nBytes` using
// algorithm `algo` and protocol `proto`.  Returns < 0 if the combination is
// not supported.  Optionally writes algoBw and busBw (GB/s) into the output
// pointers for logging / diagnostics.
static float computeAllReduceTime(TunerContext* ctx, int algo, int proto,
                                  size_t nBytes, int numPipeOps, int regBuff,
                                  float* outAlgoBw, float* outBusBw) {
  const CostModelParams* P = &ctx->costModel;
  int nRanks = (int)ctx->nRanks;
  int nNodes = (int)ctx->nNodes;
  if (nRanks <= 1) {
    if (outAlgoBw) *outAlgoBw = 0.0f;
    if (outBusBw)  *outBusBw  = 0.0f;
    return 0.0f;
  }
  int ppn = nRanks / nNodes;
  int nChannels = P->nChannels;
  int scale = (nNodes <= 1) ? 0 : (nNodes <= 2 ? 1 : 2);

  // Plain NVLS is intra-NVL-domain only; it cannot span multiple NCCL nodes.
  if (algo == NCCL_ALGO_NVLS && nNodes > 1) return -1.0f;

  // 1. Per-channel BW (AllReduce uses NVLS pipelining factor 2.0).
  double bw = perChannelBw(P, algo, proto, scale, nNodes, nChannels, 2.0);
  if (bw < 0.0) return -1.0f;
  // Single-domain (nNodes==1) AllReduce: NVLS's in-network reduction beats ring
  // once bandwidth-bound.  Cap ring/SIMPLE so NVLS wins large messages, while
  // ring/LL (small) and ring/LL128 (mid) keep their measured advantage.
  if (algo == NCCL_ALGO_RING && proto == NCCL_PROTO_SIMPLE && nNodes == 1
      && P->ringBwCapIntra > 0.0 && bw > P->ringBwCapIntra) {
    bw = P->ringBwCapIntra;
  }
  double busBw = (double)nChannels * bw;

  // 2. Protocol / algo BW caps.
  busBw = applyProtocolCap(busBw, algo, proto, P, scale, nNodes, nChannels);
  busBw *= regBwScaleFor(P, algo, proto, regBuff);

  // 3. busBw -> algoBw.
  double algoBw = busBwToAlgoBw(busBw, algo, ncclFuncAllReduce, nRanks);
  if (algoBw <= 0.0) return -1.0f;

  // 4. Latency.
  double lat = algoLatency(P, algo, proto, ncclFuncAllReduce, nRanks, nNodes, ppn);
  if (lat < 0.0) return -1.0f;

  // 5. Final time.
  int latCount = (algo == NCCL_ALGO_RING) ? numPipeOps : ((numPipeOps + 3) / 4);
  if (latCount < 1) latCount = 1;
  double timeUs = lat * (double)latCount + (double)nBytes / (1000.0 * algoBw);

  if (outAlgoBw) *outAlgoBw = (float)algoBw;
  if (outBusBw)  *outBusBw  = (float)busBw;
  return (float)timeUs;
}

// Shared compute for AllGather / ReduceScatter.  Same fabric model as
// AllReduce.  NVLS is single-node only (intra-NVL domain).  PAT is modelled
// only when the profile sets patBwEff > 0.  Returns < 0 for unsupported
// (coll, algo, proto).
//
// Known conservatism preserved here (defer to a measurement-driven follow-up):
//   bwIntra/bwInter come from the AllReduce-tuned GB200F params; AG/RS have
//   no reduction step so achievable BW is mildly higher than AR.
static float computeAllGatherOrRSTime(TunerContext* ctx, ncclFunc_t coll,
                                      int algo, int proto, size_t nBytes,
                                      int numPipeOps, int regBuff,
                                      float* outAlgoBw, float* outBusBw) {
  const CostModelParams* P = &ctx->costModel;
  int nRanks = (int)ctx->nRanks;
  int nNodes = (int)ctx->nNodes;
  if (nRanks <= 1) {
    if (outAlgoBw) *outAlgoBw = 0.0f;
    if (outBusBw)  *outBusBw  = 0.0f;
    return 0.0f;
  }
  int ppn = nRanks / nNodes;
  int nChannels = P->nChannels;
  int scale = (nNodes <= 1) ? 0 : (nNodes <= 2 ? 1 : 2);

  // PAT (NCCL: AG/RS only, SIMPLE only, 1 GPU per node).  Mirrors NCCL's own
  // model: busBw scaled by an efficiency factor and used as algoBw directly
  // (no busBw->algoBw ratio), latency = base + log2(nNodes) * per-step NET
  // latency + nRanks * a linear per-rank term.
  if (algo == NCCL_ALGO_PAT) {
    if (P->patBwEff <= 0.0 || proto != NCCL_PROTO_SIMPLE || nNodes != nRanks) return -1.0f;
    double perCh = (nNodes <= 2) ? P->bwIntra : P->bwInter;
    double busBw = (double)nChannels * perCh * P->patBwEff;
    busBw *= regBwScaleFor(P, algo, proto, regBuff);
    double algoBw = busBw;
    if (algoBw <= 0.0) return -1.0f;
    double stepLat = P->hwLatencies[NCCL_HW_NET][NCCL_ALGO_PAT][proto] + 2.0 * P->graphLatencyInter;
    double lat = P->baseLatencies[NCCL_ALGO_PAT][proto]
               + (double)costLog2i(nNodes) * stepLat
               + (double)nRanks * P->patLatPerRank;
    int latCount = (numPipeOps + 3) / 4;
    if (latCount < 1) latCount = 1;
    double timeUs = lat * (double)latCount + (double)nBytes / (1000.0 * algoBw);
    if (outAlgoBw) *outAlgoBw = (float)algoBw;
    if (outBusBw)  *outBusBw  = (float)busBw;
    return (float)timeUs;
  }

  // Algorithm gating specific to AG/RS:
  //   NVLS  : single-node only (intra-NVLink domain)
  //   TREE / NVLS_TREE : not supported here
  if (algo == NCCL_ALGO_NVLS && nNodes > 1) return -1.0f;
  if (algo == NCCL_ALGO_TREE || algo == NCCL_ALGO_NVLS_TREE) {
    return -1.0f;
  }

  // 1. Per-channel BW (AG/RS use NVLS pipelining factor 1.0).
  double bw = perChannelBw(P, algo, proto, scale, nNodes, nChannels, 1.0);
  if (bw < 0.0) return -1.0f;
  double busBw = (double)nChannels * bw;

  // 2. Protocol caps (TREE branch unreached; RING LL/LL128 caps apply).
  busBw = applyProtocolCap(busBw, algo, proto, P, scale, nNodes, nChannels);
  busBw *= regBwScaleFor(P, algo, proto, regBuff);

  // 3. busBw -> algoBw.
  double algoBw = busBwToAlgoBw(busBw, algo, coll, nRanks);
  if (algoBw <= 0.0) return -1.0f;

  // 4. Latency.
  double lat = algoLatency(P, algo, proto, coll, nRanks, nNodes, ppn);
  if (lat < 0.0) return -1.0f;

  // 5. Final time.
  int latCount = (algo == NCCL_ALGO_RING) ? numPipeOps : ((numPipeOps + 3) / 4);
  if (latCount < 1) latCount = 1;
  double timeUs = lat * (double)latCount + (double)nBytes / (1000.0 * algoBw);

  if (outAlgoBw) *outAlgoBw = (float)algoBw;
  if (outBusBw)  *outBusBw  = (float)busBw;
  return (float)timeUs;
}

static float computeAllGatherTime(TunerContext* ctx, int algo, int proto,
                                  size_t nBytes, int numPipeOps, int regBuff,
                                  float* outAlgoBw, float* outBusBw) {
  return computeAllGatherOrRSTime(ctx, ncclFuncAllGather, algo, proto,
                                  nBytes, numPipeOps, regBuff, outAlgoBw, outBusBw);
}

static float computeReduceScatterTime(TunerContext* ctx, int algo, int proto,
                                      size_t nBytes, int numPipeOps, int regBuff,
                                      float* outAlgoBw, float* outBusBw) {
  return computeAllGatherOrRSTime(ctx, ncclFuncReduceScatter, algo, proto,
                                  nBytes, numPipeOps, regBuff, outAlgoBw, outBusBw);
}

// Generic cost-table populator.  `computeFn` returns time(us) for the given
// (algo, proto) cell, or < 0 if unsupported.  Cells the model cannot price are
// left as the caller (NCCL) initialised them, unless the profile sets
// disableUnmodeledCells: then, provided at least one cell was priced, they get
// COST_MODEL_DISABLED_COST so NCCL never mixes its own estimates with ours.
typedef float (*ComputeFn)(TunerContext*, int, int, size_t, int, int, float*, float*);

static int applyCostModelGeneric(TunerContext* ctx, ncclFunc_t coll,
                                 size_t nBytes, int numPipeOps, int regBuff,
                                 float** collCostTable, int numAlgo, int numProto,
                                 ComputeFn computeFn) {
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/MRC: cost-model fallback collType=%s nBytes=%zu nRanks=%zu nNodes=%zu regBuff=%d profile=%s",
                     collTypeToString(coll), nBytes, ctx->nRanks, ctx->nNodes, regBuff,
                     ctx->costModelProfile ? ctx->costModelProfile : "mnnvl");
  }
  int populated = 0;
  int unpriced[NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS];
  int nUnpriced = 0;
  // NCCL core passes collCostTable as a contiguous float[numAlgo][numProto]
  // (see nccl/src/enqueue.cc), so it must be indexed as a 2D array rather than
  // an array of row pointers — same convention as the CSV path above.
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  for (int a = 0; a < numAlgo && a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < numProto && p < NCCL_NUM_PROTOCOLS; p++) {
      if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
      float algoBw = 0.0f, busBw = 0.0f;
      float t = computeFn(ctx, a, p, nBytes, numPipeOps, regBuff, &algoBw, &busBw);
      if (t < 0.0f) {
        unpriced[nUnpriced++] = a * NCCL_NUM_PROTOCOLS + p;
        continue;
      }
      table[a][p] = t;
      populated++;
      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/MRC: CostModel %s nBytes=%zu algo=%s proto=%s "
                         "time=%.3fus algoBw=%.2fGB/s busBw=%.2fGB/s",
                         collTypeToString(coll), nBytes,
                         algorithmToString(a), protocolToString(p),
                         t, algoBw, busBw);
      }
    }
  }
  if (populated > 0 && ctx->costModel.disableUnmodeledCells) {
    for (int i = 0; i < nUnpriced; i++) {
      int a = unpriced[i] / NCCL_NUM_PROTOCOLS, p = unpriced[i] % NCCL_NUM_PROTOCOLS;
      table[a][p] = COST_MODEL_DISABLED_COST;
      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/MRC: CostModel %s nBytes=%zu algo=%s proto=%s not modelled; disabled",
                         collTypeToString(coll), nBytes,
                         algorithmToString(a), protocolToString(p));
      }
    }
  }
  return populated;
}

static int applyCostModelAllReduce(TunerContext* ctx, size_t nBytes,
                                   int numPipeOps, int regBuff, float** collCostTable,
                                   int numAlgo, int numProto) {
  return applyCostModelGeneric(ctx, ncclFuncAllReduce, nBytes, numPipeOps, regBuff,
                               collCostTable, numAlgo, numProto,
                               computeAllReduceTime);
}

static int applyCostModelAllGather(TunerContext* ctx, size_t nBytes,
                                   int numPipeOps, int regBuff, float** collCostTable,
                                   int numAlgo, int numProto) {
  return applyCostModelGeneric(ctx, ncclFuncAllGather, nBytes, numPipeOps, regBuff,
                               collCostTable, numAlgo, numProto,
                               computeAllGatherTime);
}

static int applyCostModelReduceScatter(TunerContext* ctx, size_t nBytes,
                                       int numPipeOps, int regBuff, float** collCostTable,
                                       int numAlgo, int numProto) {
  return applyCostModelGeneric(ctx, ncclFuncReduceScatter, nBytes, numPipeOps, regBuff,
                               collCostTable, numAlgo, numProto,
                               computeReduceScatterTime);
}

__hidden ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v5_t* constants) {
  TunerContext* ctx = (TunerContext*)malloc(sizeof(TunerContext));
  if (!ctx) return ncclSystemError;

  ctx->configs = NULL;
  ctx->numConfigs = 0;
  ctx->maxConfigs = 0;
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;
  ctx->useCostModel = 0;
  ctx->forceCostModel = 0;
  if (nvlDomainInfo) {
    ctx->nvlDomainInfo = *nvlDomainInfo;
  } else {
    memset(&ctx->nvlDomainInfo, 0, sizeof(ncclNvlDomainInfo_v5_t));
  }

  // Initialise cost-model parameters (reads NCCL_TUNER_COST_MODEL* env vars).
  initCostModel(ctx);

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/MRC: Initializing tuner for %zu nodes, %zu ranks, %d NVL domains (costModel=%d)",
                nNodes, nRanks, ctx->nvlDomainInfo.nNvlDomains, ctx->useCostModel);
  }

  // Try to load config file from environment variable or default location
  const char* configFile = getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) {
    configFile = "nccl_tuner.conf";
  }

  ncclResult_t result = loadConfig(ctx, configFile);
  if (result != ncclSuccess) {
    if (ctx->configs) free(ctx->configs);
    free(ctx);
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;
  if (!ctx) return ncclInternalError;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/MRC: getCollInfo collType=%s nBytes=%zu numPipeOps=%d regBuff=%d",
                     collTypeToString(collType), nBytes, numPipeOps, regBuff);
  }

  // Look for matching configuration (skipped entirely when the cost model is
  // forced via NCCL_TUNER_FORCE_COST_MODEL).
  for (int i = 0; !ctx->forceCostModel && i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      if (config->algorithm < numAlgo && config->protocol < numProto) {
       /*
        *The parameter is declared float** collCostTable, but must be treated as float[numAlgo][numProto].
        * Because NCCL core passes a contiguous 2D array, not an array of pointers.
        * If we treat it as an array of pointers, accessing collCostTable[config->algorithm][config->protocol] would be segfault
        * Because collCostTable[algorithm] loads the actual float cost values from the contiguous table,
        * dereferences that float value as an address, and then tries to access protocol index from that address is invalid.
        * Here, table is a pointer to an array of NCCL_NUM_PROTOCOLS floats — i.e. a pointer to a row.
        */
        float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
        if (table[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          table[config->algorithm][config->protocol] = 0.0;

          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/MRC: Applied config for collType=%s bytes=%zu: algo=%s proto=%s channels=%d",
                             collTypeToString(collType), nBytes,
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             config->nChannels);
          }
          return ncclSuccess;
        }
      }
    }
  }

  // 2. Cost-model fallback — fill in collCostTable so NCCL can pick the min.
  //    Covers AllReduce, AllGather, ReduceScatter on the GB200F platform.
  if (ctx->useCostModel) {
    int n = 0;
    switch (collType) {
      case ncclFuncAllReduce:
        n = applyCostModelAllReduce(ctx, nBytes, numPipeOps, regBuff,
                                    collCostTable, numAlgo, numProto);
        break;
      case ncclFuncAllGather:
        n = applyCostModelAllGather(ctx, nBytes, numPipeOps, regBuff,
                                    collCostTable, numAlgo, numProto);
        break;
      case ncclFuncReduceScatter:
        n = applyCostModelReduceScatter(ctx, nBytes, numPipeOps, regBuff,
                                        collCostTable, numAlgo, numProto);
        break;
      default:
        break;  // Other collectives: leave table for NCCL's internal tuner.
    }
    if (n > 0) {
      // Leave *nChannels at NCCL's default (as the CSV path does for
      // channels=-1 rules).  The cost model uses nChannels internally only to
      // rank algo/proto; forcing a fixed channel count at runtime measurably
      // degrades sizes where NCCL's size-adaptive channel pick is better.
      return ncclSuccess;
    }
  }

  // 3. No CSV match, cost model disabled or had nothing to say — leave the
  //    table unchanged and let NCCL's internal tuner pick.
  return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* context) {
  if (context) {
    TunerContext* ctx = (TunerContext*)context;
    if (ctx->configs) free(ctx->configs);
    free(context);
  }
  return ncclSuccess;
}

#define PLUGIN_NAME "MRC"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginFinalize
};
