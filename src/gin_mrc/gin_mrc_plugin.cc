/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NCCL GIN plugin for MRC.
 *
 * Host side of the NCCL_NET_DEVICE_GIN_MRC backend: discovers MRC-capable
 * NICs, builds the RCX queue pairs, registers symmetric memory, and publishes
 * the GPU-resident context that NCCL's in-tree device code
 * (nccl_device/gin/mrc/gin_mrc.h) drives from CUDA kernels.
 *
 * Built on the open-source DOCA GPUNetIO library (DOCA_HOME), running in SDK
 * mode for the RCX QP type. MRC QPs speak RDMA Write and Write-with-Immediate
 * only: signals arrive as inbound Write-with-Immediate, so every QP has a
 * receive queue; there are no companion/atomic QPs, and Get is a
 * Receiver-Driven Write served by the data owner's device code.
 *
 * Who drains the receive queues follows the device code NCCL was built with,
 * which NCCL reports as the backend version (see gin_mrc_dev.h):
 *   - CPU RQ: a progress thread of this plugin drains them into per-context
 *     DOCA GPUNetIO signal maps.
 *   - GPU RQ: each QP is exported to the DOCA SDK (doca_gpu_verbs_export_sdk_qp,
 *     needs a DOCA_HOME with it) and NCCL's device code drains them.
 ************************************************************************/

#include <assert.h>
#include <endian.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "transport/net_ib/common.h"

#include "gin/gin_v14.h"
#include "mrc_qp.h"
#include "nccl_device/gin/mrc/gin_mrc_dev.h"
#ifdef MRC_HAVE_GPU_RQ
#include "mrc_rq_init.h"
#endif

// DOCA SDK libraries the open-source GPUNetIO library forwards to for RCX QPs (SDK mode), unless
// DOCA_SDK_LIB_PATH is already set. Defaults to the build-time DOCA SDK location.
#ifndef MRC_DOCA_SDK_LIB_PATH
#define MRC_DOCA_SDK_LIB_PATH "/opt/mellanox/doca/lib/aarch64-linux-gnu"
#endif

#define DOCACHECK(call) \
  do { \
    doca_error_t err = call; \
    if (err != DOCA_SUCCESS) { \
      WARN("GIN/MRC : DOCA failure %d", err); \
      return ncclSystemError; \
    } \
  } while (0)

#define DOCACHECKGOTO(call, RES, label) \
  do { \
    doca_error_t err = call; \
    if (err != DOCA_SUCCESS) { \
      WARN("GIN/MRC : DOCA failure %d", err); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

#define MAX_PCI_ADDRESS_LEN 32U

NCCL_PARAM(GinMrcQpDepth, "GIN_MRC_QP_DEPTH", 128);
// Receive WQEs per QP. Each inbound signal or RDW message consumes one until its drain returns it;
// senders wait in RNR retries beyond. -1: 1024 in CPU RQ mode, to absorb bursts while the progress
// thread catches up, and the SQ depth in GPU RQ mode.
NCCL_PARAM(GinMrcRqDepth, "GIN_MRC_RQ_DEPTH", -1);
// RNR NAK timer the responder advertises (IB encoding; 1 = 0.01 ms, 12 = 0.64 ms). -1: 1 in CPU RQ
// mode, 12 in GPU RQ mode.
NCCL_PARAM(GinMrcMinRnrTimer, "GIN_MRC_MIN_RNR_TIMER", -1);
// GPU RQ mode: receive WQE (re)post mode, 0 = lightweight DBR-only update, 1 = regular post-recv
// (see gin_mrc_dev.h).
NCCL_PARAM(GinMrcRqPostMode, "GIN_MRC_RQ_POST_MODE", NCCL_GIN_MRC_RQ_POST_MODE_DBR);
NCCL_PARAM(GinMrcNicHandler, "GIN_MRC_NIC_HANDLER", 0);
NCCL_PARAM(GinMrcGidIndex, "IB_GID_INDEX", 3);
NCCL_PARAM(GinMrcTc, "GIN_MRC_TC", -1);
NCCL_PARAM(GinMrcSl, "GIN_MRC_SL", -1);
NCCL_PARAM(GinMrcTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(GinMrcRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(GinMrcPkeyIndex, "IB_PKEY_INDEX", 0);
// In-flight RDW get requests per (context, peer); a power of two.
NCCL_PARAM(GinMrcGetWindow, "GIN_MRC_GET_WINDOW", 4);
// GPU RQ mode: generation tag for RDW get requests, unique per created context set (6 bits on the wire).
static std::atomic<uint32_t> mrcGinRdwGen{0};
// Register GIN windows over a NIC's Data Direct PCIe link when this GPU sits behind it. Independent
// of NCCL_IB_DATA_DIRECT, which also moves the NET plugin and NCCL's host RMA to Data Direct.
NCCL_PARAM(GinMrcDataDirect, "GIN_MRC_DATA_DIRECT", 1);

/* ── CUDA memory helpers ──────────────────────────────────────────── */

// GPU allocation via the VMM API. Needed (over cudaMalloc) so the buffer can be exported as a
// DMA-BUF for MR registration, which is the supported GPUDirect path on this fabric.
static ncclResult_t mrcCuMemAlloc(void** ptr, CUmemGenericAllocationHandle* handleOut, size_t size) {
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  CUdeviceptr dptr = 0;
  size_t granularity = 0;
  int cudaDev = 0;

  CUDACHECK(cudaGetDevice(&cudaDev));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cudaDev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  // Lets GDRCopy pin the allocation, which the signal maps need for their CPU mappings.
  prop.allocFlags.gpuDirectRDMACapable = 1;

  if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
    return ncclSystemError;
  size = ROUNDUP(size, granularity);

  if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) return ncclSystemError;
  if (cuMemAddressReserve(&dptr, size, granularity, 0, 0) != CUDA_SUCCESS) goto fail_release;
  if (cuMemMap(dptr, size, 0, handle, 0) != CUDA_SUCCESS) goto fail_free_addr;

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (cuMemSetAccess(dptr, size, &accessDesc, 1) != CUDA_SUCCESS) goto fail_unmap;

  CUDACHECK(cudaMemset((void*)dptr, 0, size));
  *ptr = (void*)dptr;
  *handleOut = handle;
  return ncclSuccess;

fail_unmap:
  cuMemUnmap(dptr, size);
fail_free_addr:
  cuMemAddressFree(dptr, size);
fail_release:
  cuMemRelease(handle);
  WARN("GIN/MRC : GPU memory allocation of %zu bytes failed", size);
  return ncclSystemError;
}

static ncclResult_t mrcCuMemFree(void* ptr) {
  if (ptr == nullptr) return ncclSuccess;
  CUdeviceptr dptr = (CUdeviceptr)ptr;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  if (cuMemRetainAllocationHandle(&handle, ptr) != CUDA_SUCCESS) return ncclSuccess;
  cuMemRelease(handle);
  if (cuMemGetAddressRange(NULL, &size, dptr) != CUDA_SUCCESS) return ncclSystemError;
  cuMemUnmap(dptr, size);
  cuMemRelease(handle);
  cuMemAddressFree(dptr, size);
  return ncclSuccess;
}

// Paired host staging + GPU buffer. The host copy is filled during setup and pushed once.
template <typename T>
struct MrcHostGpuBuf {
  T* host = nullptr;
  T* gpu = nullptr;
  CUmemGenericAllocationHandle handle = 0;
  size_t count = 0;

  ncclResult_t allocate(size_t n) {
    count = n;
    host = (T*)calloc(n, sizeof(T));
    if (host == nullptr) return ncclSystemError;
    NCCLCHECK(mrcCuMemAlloc((void**)&gpu, &handle, n * sizeof(T)));
    return ncclSuccess;
  }
  ncclResult_t push() {
    CUDACHECK(cudaMemcpy(gpu, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return ncclSuccess;
  }
  ncclResult_t free_() {
    if (host) {
      free(host);
      host = nullptr;
    }
    if (gpu) {
      NCCLCHECK(mrcCuMemFree(gpu));
      gpu = nullptr;
    }
    return ncclSuccess;
  }
};

/* ── Memory registration ──────────────────────────────────────────── */

static ncclResult_t mrcRegMrDmaBuf(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access) {
  int dmabufFd = -1;
  size_t pageSize = sysconf(_SC_PAGESIZE);
  size_t alignedSize = ROUNDUP(length, pageSize);

  if (cuMemGetHandleForAddressRange((void*)&dmabufFd, (CUdeviceptr)addr, alignedSize,
                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0) != CUDA_SUCCESS)
    return ncclSystemError;

  ncclResult_t status = wrap_ibv_reg_dmabuf_mr(mr, pd, 0, alignedSize, 0, dmabufFd, access);
  close(dmabufFd);
  return status;
}

// Data Direct registration: the NIC reaches the buffer over its direct PCIe link to the GPU
// instead of through the CPU (C2C). Fails quietly so callers can fall back.
static ncclResult_t mrcRegMrDataDirect(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access) {
  int dmabufFd = -1;
  size_t alignedSize = ROUNDUP(length, (size_t)sysconf(_SC_PAGESIZE));
  if (cuMemGetHandleForAddressRange((void*)&dmabufFd, (CUdeviceptr)addr, alignedSize,
                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                    CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE) != CUDA_SUCCESS)
    return ncclSystemError;
  *mr = wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0, alignedSize, 0, dmabufFd, access,
                                         MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  close(dmabufFd);
  return *mr ? ncclSuccess : ncclSystemError;
}

static ncclResult_t mrcRegMr(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access,
                             bool forceStrictOrdering = false, bool dataDirect = false) {
  if (!forceStrictOrdering && ncclIbRelaxedOrderingEnabled) access |= IBV_ACCESS_RELAXED_ORDERING;
  if (dataDirect && mrcRegMrDataDirect(mr, pd, addr, length, access) == ncclSuccess) return ncclSuccess;
  if (mrcRegMrDmaBuf(mr, pd, addr, length, access) == ncclSuccess) return ncclSuccess;
  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, access));
  return ncclSuccess;
}

// GPU-resident table of uint64 slots (signals or counters) shared by all contexts of a
// connection, registered once and its rkey shared with every peer.
struct MrcGlobalTable {
  uint64_t* gpu = nullptr;
  CUmemGenericAllocationHandle handle = 0;
  struct ibv_mr* mr = nullptr;
  unsigned int count = 0;
  unsigned int nextFree = 0;
  MrcHostGpuBuf<uint32_t> rkeys; // big-endian, one per peer

  ncclResult_t allocate(unsigned int n, int nranks) {
    count = n;
    nextFree = 0;
    if (n == 0) return ncclSuccess;
    NCCLCHECK(mrcCuMemAlloc((void**)&gpu, &handle, (size_t)n * sizeof(uint64_t)));
    NCCLCHECK(rkeys.allocate(nranks));
    return ncclSuccess;
  }
  ncclResult_t registerMr(struct ibv_pd* pd) {
    if (count == 0) return ncclSuccess;
    NCCLCHECK(mrcRegMr(&mr, pd, gpu, (size_t)count * sizeof(uint64_t),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
                       /*forceStrictOrdering=*/true));
    return ncclSuccess;
  }
  ncclResult_t take(unsigned int n, unsigned int* outStart) {
    if (nextFree + n > count) {
      WARN("GIN/MRC : global table exhausted (%u requested, %u of %u used)", n, nextFree, count);
      return ncclInvalidUsage;
    }
    *outStart = nextFree;
    nextFree += n;
    return ncclSuccess;
  }
  ncclResult_t free_() {
    if (mr) {
      NCCLCHECK(wrap_ibv_dereg_mr(mr));
      mr = nullptr;
    }
    if (gpu) {
      NCCLCHECK(mrcCuMemFree(gpu));
      gpu = nullptr;
    }
    NCCLCHECK(rkeys.free_());
    return ncclSuccess;
  }
};

/* ── Collective comm: ring of sockets for setup-time exchanges ────── */

struct mrcGinHandle {
  union ncclSocketAddress connectAddr;
  uint64_t magic;
};
static_assert(sizeof(struct mrcGinHandle) <= NCCL_GIN_HANDLE_MAXSIZE, "mrcGinHandle too large");

struct mrcGinListenComm {
  int dev;
  struct ncclSocket sock;
};

struct mrcGinCollComm {
  int dev;
  int rank;
  int nranks;
  struct ncclSocket sendSock; // to rank+1
  struct ncclSocket recvSock; // from rank-1
  struct ibv_context* context;
  struct ibv_pd* pd;
  // This rank's registered windows, mirrored to the GPU so the RDW get server can validate a
  // request's source and translate its rkey into an lkey. Allocated on first use.
  MrcHostGpuBuf<struct ncclGinMrcMrTable> mrTable;
  // Data Direct (NCCL_GIN_MRC_DATA_DIRECT, on by default for capable NICs): windows are registered
  // over the NIC's direct PCIe link. This 8-byte buffer, registered the same way, probes that the
  // NIC accepts Data Direct registrations; windows only use Data Direct when it succeeded. The
  // QPs are then created for Data Direct, so the signal maps deliver through forced-PCIe
  // mappings that cannot overtake the data.
  struct {
    void* addr;
    CUmemGenericAllocationHandle handle;
    struct ibv_mr* mr;
  } dd;
};

// ncclSocketAccept() memcpy's the listen socket, so the accepted socket inherits its async flag
// and can return before the handshake finishes. Drive both ends to Ready explicitly.
static ncclResult_t mrcGinSocketWaitReady(struct ncclSocket* sock) {
  int ready = 0;
  while (!ready) NCCLCHECK(ncclSocketReady(sock, &ready));
  return ncclSuccess;
}

// Ring allgather: rotate each rank's chunk around the ring nranks-1 times. Setup-path only,
// so simplicity beats bandwidth.
static ncclResult_t mrcGinAllGather(struct mrcGinCollComm* cComm, void* srcBuf, void* recvBuf, size_t len) {
  const int nranks = cComm->nranks;
  char* buf = (char*)recvBuf;
  int speer = cComm->rank;

  memcpy(buf + (size_t)speer * len, srcBuf, len);
  for (int i = 0; i < nranks - 1; i++) {
    int rpeer = (speer - 1 + nranks) % nranks;
    NCCLCHECK(ncclSocketSend(&cComm->sendSock, buf + (size_t)speer * len, (int)len));
    NCCLCHECK(ncclSocketRecv(&cComm->recvSock, buf + (size_t)rpeer * len, (int)len));
    speer = rpeer;
  }
  return ncclSuccess;
}

// All-to-all built on the allgather: every rank contributes nranks items and keeps the column
// addressed to it.
static ncclResult_t mrcGinAllToAll(struct mrcGinCollComm* cComm, void* srcBuf, void* recvBuf, size_t len) {
  const int nranks = cComm->nranks;
  ncclResult_t status = ncclSuccess;
  char* tmp = (char*)malloc((size_t)nranks * nranks * len);
  if (tmp == nullptr) return ncclSystemError;

  NCCLCHECKGOTO(mrcGinAllGather(cComm, srcBuf, tmp, (size_t)nranks * len), status, out);
  for (int i = 0; i < nranks; i++) {
    memcpy((char*)recvBuf + (size_t)i * len, tmp + ((size_t)i * nranks + cComm->rank) * len, len);
  }

out:
  free(tmp);
  return status;
}

/* ── Per-context state ────────────────────────────────────────────── */

struct mrcGinExchInfo {
  int lid;
  uint32_t qpn;
  struct doca_verbs_gid vgid;
  int rank;
  char hostname[HOST_NAME_MAX];
  char ibDevName[MAXNAMESIZE];
  int cudaId;
};

struct mrcGinMemHandle {
  int type;
  struct ibv_mr* mr;
  MrcHostGpuBuf<struct ncclGinMrcMemHandle> devHandle;
  MrcHostGpuBuf<uint32_t> rkeys;
};

struct mrcGinContext {
  struct mrcGinCollComm* collComm;
  int cudaId;
  int rank;
  int nContexts;

  doca_gpu_t* gdev;
  doca_dev_t* netDev;
  doca_verbs_ah_attr_t* ah;
  struct ibv_device_attr devAttr;
  struct ibv_port_attr portAttr;
  union ibv_gid rgid;
  uint8_t portNum;
  int gidIndex;
  int pkeyIndex;
  char hostname[HOST_NAME_MAX];
  char ibDevName[MAXNAMESIZE];

  uint32_t qpSqSize;
  uint32_t qpRqSize;
  uint8_t minRnrTimer;
  struct doca_gpu_verbs_qp_hl** qps; // [nContexts * (nranks + 1)]
  // GPU RQ mode (the device code drains the receive queues): the DOCA SDK GPU handles of the QPs,
  // [nContexts][nranks] per-peer QPs followed by [nContexts] loopback responders.
  bool gpuRq;
  MrcHostGpuBuf<void*> sdkQps;

  MrcGlobalTable* counters;
  MrcGlobalTable* signals;

  struct ncclGinMrcGPUContext* ctxStaging; // host mirror of the device array
  MrcHostGpuBuf<struct ncclGinMrcGPUContext> ctxBuf;

  struct ncclGinMrcPeerState* peers; // [nContexts * nranks]
  uint32_t* counterSlots;            // [nContexts * nranks * counterCap]
  uint32_t counterCap;

  // Signal delivery. Each context's signal table slice is registered in its own signal map (ids
  // are per context, see gin_mrc_dev.h), and one progress thread drains the receive queues of
  // every context into its map.
  unsigned int signalTableSize; // NCCL_GIN_MRC_SIGNAL_TABLE_SIZE(nSignals, nranks)
  int nSignals;
  struct doca_gpu_verbs_signal_map** maps;     // [nContexts]
  bool* mapRegistered;                         // [nContexts]
  uint64_t* signalBase;                        // [nContexts * nSignals] reset baselines (GPU)
  std::vector<struct doca_gpu_verbs_qp*>* rqQps; // [nContexts] receive queues drained into maps[i]
  bool dataDirect;
  pthread_t progressThread;
  bool progressStarted;
  std::atomic<bool> progressStop;
  std::atomic<int> progressError; // first fatal doca_error_t of the receive path (0 = healthy)

  struct {
    void* addr;
    CUmemGenericAllocationHandle handle;
    struct ibv_mr* mr;
  } sink;

  // RDW get: request rings, descriptor staging and response landing for every context
  // (layout in gin_mrc_dev.h), one zero-based MR whose rkey is shared with every peer.
  struct {
    void* addr;
    CUmemGenericAllocationHandle handle;
    struct ibv_mr* mr;
    size_t ctxStride;
    uint32_t window;
  } rdw;
  MrcHostGpuBuf<uint32_t> rdwRkeys; // [nranks], big-endian

  ncclNetDeviceHandle_t* devHandle;
};

/* ── RDW get: registered-window table ─────────────────────────────── */

// The GPU may be serving gets while windows are (de)registered, so the table is never compacted
// or re-uploaded wholesale: a live entry is never moved, and each update is published by a final
// aligned 4-byte copy (an entry's rkey, or the count) once everything it exposes is in place.
// A removed entry is tombstoned with rkey 0 / size 0, which no lookup can match usefully.

static ncclResult_t mrcGinMrTableEnsure(struct mrcGinCollComm* cComm) {
  if (cComm->mrTable.gpu) return ncclSuccess;
  NCCLCHECK(cComm->mrTable.allocate(1));
  return cComm->mrTable.push();
}

static ncclResult_t mrcGinMrTableCopy(struct mrcGinCollComm* cComm, const void* hostField, size_t bytes) {
  const size_t offset = (const char*)hostField - (const char*)cComm->mrTable.host;
  CUDACHECK(cudaMemcpy((char*)cComm->mrTable.gpu + offset, hostField, bytes, cudaMemcpyHostToDevice));
  return ncclSuccess;
}

static ncclResult_t mrcGinMrTableAdd(struct mrcGinCollComm* cComm, struct ibv_mr* mr, size_t size) {
  NCCLCHECK(mrcGinMrTableEnsure(cComm));
  struct ncclGinMrcMrTable* table = cComm->mrTable.host;
  uint32_t slot = 0;
  while (slot < table->count && table->entries[slot].rkey != 0) slot++;
  if (slot == NCCL_GIN_MRC_MR_TABLE_CAPACITY) {
    WARN("GIN/MRC : registered-window table is full (%d entries); gets from this window will be rejected",
         NCCL_GIN_MRC_MR_TABLE_CAPACITY);
    return ncclSuccess;
  }
  struct ncclGinMrcMrEntry* entry = &table->entries[slot];
  entry->lkey = htobe32(mr->lkey);
  entry->size = size;
  if (slot < table->count) {
    // Reusing a tombstone: lkey/size first, then the rkey that makes the entry matchable.
    NCCLCHECK(mrcGinMrTableCopy(cComm, &entry->lkey, sizeof(entry->lkey)));
    NCCLCHECK(mrcGinMrTableCopy(cComm, &entry->size, sizeof(entry->size)));
    entry->rkey = htobe32(mr->rkey);
    NCCLCHECK(mrcGinMrTableCopy(cComm, &entry->rkey, sizeof(entry->rkey)));
  } else {
    // Appending: the whole entry first, then the count that makes it visible.
    entry->rkey = htobe32(mr->rkey);
    NCCLCHECK(mrcGinMrTableCopy(cComm, entry, sizeof(*entry)));
    table->count++;
    NCCLCHECK(mrcGinMrTableCopy(cComm, &table->count, sizeof(table->count)));
  }
  INFO(NCCL_NET, "GIN/MRC : window rkey 0x%x (%zu bytes) registered for RDW get serving", mr->rkey, size);
  return ncclSuccess;
}

static ncclResult_t mrcGinMrTableRemove(struct mrcGinCollComm* cComm, struct ibv_mr* mr) {
  struct ncclGinMrcMrTable* table = cComm->mrTable.host;
  if (table == nullptr) return ncclSuccess;
  const uint32_t rkey = htobe32(mr->rkey);
  for (uint32_t i = 0; i < table->count; i++) {
    struct ncclGinMrcMrEntry* entry = &table->entries[i];
    if (entry->rkey != rkey) continue;
    entry->rkey = 0;
    NCCLCHECK(mrcGinMrTableCopy(cComm, &entry->rkey, sizeof(entry->rkey)));
    entry->size = 0;
    NCCLCHECK(mrcGinMrTableCopy(cComm, &entry->size, sizeof(entry->size)));
    return ncclSuccess;
  }
  return ncclSuccess;
}

/* ── QP setup ─────────────────────────────────────────────────────── */

static void mrcGinFillExchInfo(struct mrcGinExchInfo* info, struct mrcGinContext* ctx,
                               struct doca_gpu_verbs_qp_hl* qp) {
  memset(info, 0, sizeof(*info));
  info->lid = ctx->portAttr.lid;
  doca_verbs_qp_get_qpn(qp->qp, &info->qpn);
  memcpy(info->vgid.raw, ctx->rgid.raw, sizeof(union ibv_gid));
  info->rank = ctx->rank;
  info->cudaId = ctx->cudaId;
  snprintf(info->hostname, HOST_NAME_MAX, "%s", ctx->hostname);
  snprintf(info->ibDevName, MAXNAMESIZE, "%s", ctx->ibDevName);
}

static ncclResult_t mrcGinCreateAh(struct mrcGinContext* ctx, int sl, int tc, int gidIndex) {
  ncclResult_t status = ncclSuccess;

  DOCACHECK(doca_verbs_ah_attr_create(ctx->netDev, &ctx->ah));

  if (ctx->portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_sl(ctx->ah, sl), status, fail);
  } else {
    // RoCE: use the IPv4 L3 addr type, matching the DOCA GPUNetIO verbs samples. Selecting IPv6
    // here produces a header the NIC silently drops, so remote writes never land.
    DOCACHECKGOTO(doca_verbs_ah_attr_set_traffic_class(ctx->ah, tc), status, fail);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IPv4), status, fail);
  }
  DOCACHECKGOTO(doca_verbs_ah_attr_set_sgid_index(ctx->ah, gidIndex), status, fail);
  DOCACHECKGOTO(doca_verbs_ah_attr_set_hop_limit(ctx->ah, 255), status, fail);
  return ncclSuccess;

fail:
  DOCACHECK(doca_verbs_ah_attr_destroy(ctx->ah));
  ctx->ah = nullptr;
  return status;
}

static ncclResult_t mrcGinConnectQp(struct mrcGinContext* ctx, struct doca_gpu_verbs_qp_hl* qp,
                                    struct mrcGinExchInfo* peer) {
  ncclResult_t status = ncclSuccess;
  doca_verbs_qp_attr_t* attr = nullptr;

  DOCACHECK(doca_verbs_ah_attr_set_gid(ctx->ah, peer->vgid));
  DOCACHECK(doca_verbs_ah_attr_set_dlid(ctx->ah, peer->lid));
  DOCACHECK(doca_verbs_qp_attr_create(&attr));

  DOCACHECKGOTO(doca_verbs_qp_attr_set_path_mtu(attr, DOCA_VERBS_MTU_SIZE_4K_BYTES), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(attr, 0), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(attr, 0), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(attr, ctx->portNum), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(attr, ncclParamGinMrcTimeout()), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(attr, ncclParamGinMrcRetryCnt()), status, fail);
  // Receive credits return asynchronously from the receive drain: retry RNR indefinitely while it
  // catches up instead of failing the QP.
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(attr, 7), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(attr, ctx->minRnrTimer), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_write(attr, 1), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_read(attr, 1), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_atomic_mode(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ah_attr(attr, ctx->ah), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_dest_qp_num(attr, peer->qpn), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_pkey_index(attr, ctx->pkeyIndex), status, fail);

  DOCACHECKGOTO(doca_verbs_qp_modify(qp->qp, attr,
                                     DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                       DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                       DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM),
                status, fail);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, ctx->devAttr.max_qp_rd_atom), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_modify(qp->qp, attr,
                                     DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                       DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                       DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                       DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC | DOCA_VERBS_QP_ATTR_ATOMIC_MODE),
                status, fail);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_max_rd_atomic(attr, ctx->devAttr.max_qp_rd_atom), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_modify(qp->qp, attr,
                                     DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                       DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                       DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC),
                status, fail);

  DOCACHECK(doca_verbs_qp_attr_destroy(attr));
  return ncclSuccess;

fail:
  WARN("GIN/MRC : failed to connect QP: local=%s/gpu%d/%s remote_rank=%d remote=%s/gpu%d/%s", ctx->hostname,
       ctx->cudaId, ctx->ibDevName, peer->rank, peer->hostname, peer->cudaId, peer->ibDevName);
  DOCACHECK(doca_verbs_qp_attr_destroy(attr));
  return status;
}

/* ── Plugin entry points ──────────────────────────────────────────── */

static ncclResult_t mrcGinInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  ncclMrcSetLogger(logFunction);
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));

  // GIN needs GPUDirect RDMA; fail here rather than at connect time so NCCL can fall back to
  // another backend cleanly. Either registration path will do, and DMA-BUF is the common one
  // since GDAKI-style QPs cannot use nv_peer_mem.
  if (ncclIbGdrSupport() != ncclSuccess && ncclIbDmaBufSupport(0) != ncclSuccess) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/MRC : neither peermem nor DMA-BUF is available, disabling MRC GIN");
    return ncclInternalError;
  }

  // RCX QPs are a DOCA SDK feature: the open-source GPUNetIO library provides them by forwarding to
  // the SDK found through DOCA_SDK_LIB_PATH, which must be set before its first call.
  if (getenv("DOCA_SDK_LIB_PATH") == nullptr) setenv("DOCA_SDK_LIB_PATH", MRC_DOCA_SDK_LIB_PATH, 0);
  INFO(NCCL_INIT | NCCL_NET, "GIN/MRC : open-source DOCA GPUNetIO in SDK mode, DOCA_SDK_LIB_PATH=%s",
       getenv("DOCA_SDK_LIB_PATH"));

  if (ctx) *ctx = nullptr;
  (void)commId;
  return ncclSuccess;
}

static ncclResult_t mrcGinDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

static ncclResult_t mrcGinGetGinProperties(ncclGinProperties_v14_t* ginProps) {
  // A signal is the immediate on the data write itself and rides the same QP, so ordered
  // delivery gives it the strong-signal guarantee.
  ginProps->supportsStrongSignals = true;
  // Reported as supported so callers that merely inherit ginVaSignalsRequired=true from
  // NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER are not rejected. A VA signal is not actually
  // representable -- the 32-bit immediate carries a signal index, not an address -- so the
  // device path asserts if one is ever issued (NCCL builds without -DNDEBUG, so it fires).
  // Notably NCCL's GIN symmetric kernels do use VA signals: set
  // NCCL_SYM_GIN_KERNELS_ENABLE=0 when a workload would engage them.
  ginProps->supportsVASignals = true;
  return ncclSuccess;
}

extern int64_t ncclParamIbMergeVfs();

// With Data Direct, net_ib places a NIC at its data-direct PCIe function, which is local to one
// GPU only. GIN needs every GPU of a NIC pair to use both NICs (one connection per plane), so the
// GIN plugin reports the NIC's regular PCI path, normalized like net_ib does for other devices;
// mrcGinConnect still uses Data Direct when the GPU sits behind the data-direct function.
static const char* mrcGinTopoPciPath(int dev) {
  static std::mutex mutex;
  static char paths[MAX_IB_DEVS][PATH_MAX];
  std::lock_guard<std::mutex> lock(mutex);
  char* p = paths[dev];
  if (p[0] == '\0') {
    snprintf(p, PATH_MAX, "%s", ncclIbDevs[dev].fullPciPath);
    size_t len = strlen(p);
    if (len >= 4) {
      p[len - 1] = '0';
      if (ncclParamIbMergeVfs()) p[len - 3] = p[len - 4] = '0';
    }
  }
  return p;
}

// Whether to use Data Direct for windows on dev: the NIC has a data-direct PCIe function and the
// current GPU sits behind it (they share a PCI domain). Only that GPU reaches the NIC directly;
// the other GPU of the pair keeps the C2C path on this NIC.
static bool mrcGinUseDataDirect(int dev) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  if (!ncclParamGinMrcDataDirect() || ibDev->ibProvider != IB_PROVIDER_MLX5) return false;
  char ddPath[PATH_MAX] = "/sys";
  if (ibDev->capsProvider.mlx5.dataDirect) {
    snprintf(ddPath, sizeof(ddPath), "%s", ibDev->pciPath);
  } else if (wrap_mlx5dv_get_data_direct_sysfs_path(ibDev->context, ddPath + 4, sizeof(ddPath) - 4) != ncclSuccess) {
    return false;
  }
  int cudaDev, domain;
  if (cudaGetDevice(&cudaDev) != cudaSuccess ||
      cudaDeviceGetAttribute(&domain, cudaDevAttrPciDomainId, cudaDev) != cudaSuccess)
    return false;
  char prefix[64];
  snprintf(prefix, sizeof(prefix), "/sys/devices/pci%04x:", domain);
  if (strncmp(ddPath, prefix, strlen(prefix)) != 0) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/MRC : GPU is not behind the Data Direct function of %s; using the C2C path",
         ibDev->devName);
    return false;
  }
  return true;
}

static ncclResult_t mrcGinGetProperties(int dev, ncclNetProperties_v12_t* props) {
  NCCLCHECK(ncclIbGetProperties(dev, props));
  if (ncclIbDevs[dev].capsProvider.mlx5.dataDirect && ncclIbDevs[dev].fullPciPath[0] != '\0')
    props->pciPath = (char*)mrcGinTopoPciPath(dev);

  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  // Keep railId/planeId from NCCL_IB_HCA (<dev>:<port>:<rail>:<plane>): NCCL requires the NET and GIN
  // plugins to agree for a device, and NCCL >= 2.32 orders each GPU's GIN connections by rail and
  // plane, so on multi-plane fabrics they keep connection i of every rank on the same plane.

  props->netDeviceType = (ncclNetDeviceType)NCCL_NET_DEVICE_GIN_MRC;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

static ncclResult_t mrcGinListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  struct mrcGinListenComm* comm = (struct mrcGinListenComm*)calloc(1, sizeof(*comm));
  struct mrcGinHandle* handle = (struct mrcGinHandle*)opaqueHandle;
  ncclResult_t ret = ncclSuccess;
  if (comm == nullptr) return ncclSystemError;

  static_assert(sizeof(struct mrcGinHandle) <= NCCL_NET_HANDLE_MAXSIZE, "mrcGinHandle exceeds handle size");
  memset(handle, 0, sizeof(*handle));
  handle->magic = NCCL_SOCKET_MAGIC;
  comm->dev = dev;

  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);

  *listenComm = comm;
  (void)ctx;
  return ncclSuccess;

fail:
  free(comm);
  return ret;
}

static ncclResult_t mrcGinConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                  void** collComm) {
  struct mrcGinListenComm* lComm = (struct mrcGinListenComm*)listenComm;
  struct mrcGinCollComm* cComm = nullptr;
  ncclResult_t ret = ncclSuccess;
  int next = (rank + 1) % nranks;
  struct mrcGinHandle* nextHandle = (struct mrcGinHandle*)handles[next];
  struct ncclIbDev* ibDev = ncclIbDevs + lComm->dev;

  *collComm = nullptr;
  cComm = (struct mrcGinCollComm*)calloc(1, sizeof(*cComm));
  if (cComm == nullptr) return ncclSystemError;
  cComm->dev = lComm->dev;
  cComm->rank = rank;
  cComm->nranks = nranks;

  // Ring: connect forward, accept from behind. Both sides of every link are established by
  // some rank's connect(), so this completes without a separate barrier.
  NCCLCHECKGOTO(ncclSocketInit(&cComm->sendSock, &nextHandle->connectAddr, nextHandle->magic, ncclSocketTypeNetIb,
                               NULL, 0, 0),
                ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&cComm->sendSock), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&cComm->recvSock, NULL, NCCL_SOCKET_MAGIC, ncclSocketTypeUnknown, NULL, 0, 0), ret,
                fail);
  NCCLCHECKGOTO(ncclSocketAccept(&cComm->recvSock, &lComm->sock), ret, fail);
  NCCLCHECKGOTO(mrcGinSocketWaitReady(&cComm->sendSock), ret, fail);
  NCCLCHECKGOTO(mrcGinSocketWaitReady(&cComm->recvSock), ret, fail);

  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (ibDev->pdRefs++ == 0) {
      ret = wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context);
      if (ret != ncclSuccess) ibDev->pdRefs--;
    }
  }
  if (ret != ncclSuccess) goto fail;

  cComm->context = ibDev->context;
  cComm->pd = ibDev->pd;

  if (mrcGinUseDataDirect(cComm->dev)) {
    if (mrcCuMemAlloc(&cComm->dd.addr, &cComm->dd.handle, sizeof(uint64_t)) == ncclSuccess &&
        mrcRegMrDataDirect(&cComm->dd.mr, cComm->pd, cComm->dd.addr, sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE) == ncclSuccess) {
      INFO(NCCL_INIT | NCCL_NET, "GIN/MRC : Data Direct enabled on %s", ibDev->devName);
    } else {
      INFO(NCCL_INIT | NCCL_NET, "GIN/MRC : Data Direct registration failed on %s; using the C2C path",
           ibDev->devName);
      cComm->dd.mr = nullptr;
    }
  }

  *collComm = cComm;
  (void)ctx;
  return ncclSuccess;

fail:
  ncclSocketClose(&cComm->sendSock, false);
  ncclSocketClose(&cComm->recvSock, false);
  free(cComm);
  return ret;
}

static ncclResult_t mrcGinCloseColl(void* collComm) {
  struct mrcGinCollComm* cComm = (struct mrcGinCollComm*)collComm;
  if (cComm == nullptr) return ncclSuccess;

  struct ncclIbDev* ibDev = ncclIbDevs + cComm->dev;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (--ibDev->pdRefs == 0) {
      wrap_ibv_dealloc_pd(ibDev->pd);
      ibDev->pd = nullptr;
    }
  }

  NCCLCHECK(ncclSocketClose(&cComm->sendSock, false));
  NCCLCHECK(ncclSocketClose(&cComm->recvSock, false));
  NCCLCHECK(cComm->mrTable.free_());
  if (cComm->dd.mr) NCCLCHECK(wrap_ibv_dereg_mr(cComm->dd.mr));
  if (cComm->dd.addr) NCCLCHECK(mrcCuMemFree(cComm->dd.addr));
  free(cComm);
  return ncclSuccess;
}

static ncclResult_t mrcGinCloseListen(void* listenComm) {
  struct mrcGinListenComm* lComm = (struct mrcGinListenComm*)listenComm;
  if (lComm == nullptr) return ncclSuccess;
  NCCLCHECK(ncclSocketClose(&lComm->sock, false));
  free(lComm);
  return ncclSuccess;
}

static ncclResult_t mrcGinDestroyContext(void* ginCtx);

/* ── Signal delivery: CPU-managed receive queues ──────────────────── */

// Drains every context's receive queues into that context's signal map, then returns the consumed
// receive credits with one fenced doorbell-record commit per sweep (doca_gpu_verbs_rq_progress
// contract). An RQ error stops that RQ for good; it is latched for mrcGinQueryLastError.
static void* mrcGinProgressMain(void* arg) {
  struct mrcGinContext* ctx = (struct mrcGinContext*)arg;
  const int ncontexts = ctx->nContexts;
  while (!ctx->progressStop.load(std::memory_order_relaxed)) {
    bool any = false;
    for (int c = 0; c < ncontexts; c++) {
      for (struct doca_gpu_verbs_qp* qp : ctx->rqQps[c]) {
        bool progressed = false;
        doca_error_t err = doca_gpu_verbs_rq_progress(qp, ctx->maps[c], &progressed);
        if (err != DOCA_SUCCESS && err != DOCA_ERROR_SKIPPED) {
          int healthy = 0;
          if (ctx->progressError.compare_exchange_strong(healthy, (int)err))
            WARN("GIN/MRC : rank %d context %d: receive queue failed with DOCA error %d; its signals are lost",
                 ctx->rank, c, err);
        }
        any |= progressed;
      }
    }
    if (any) {
      doca_gpu_verbs_rq_commit_fence(ctx->dataDirect);
      for (int c = 0; c < ncontexts; c++)
        for (struct doca_gpu_verbs_qp* qp : ctx->rqQps[c]) doca_gpu_verbs_rq_commit_dbr(qp);
      doca_gpu_verbs_rq_commit_fence(ctx->dataDirect);
    } else {
      sched_yield();
    }
  }
  return nullptr;
}

// Creates context c's signal map over its slice of the signal table: the NCCL signals and the RDW
// counters. The sink slot between them stays out of the map (see gin_mrc_dev.h).
static ncclResult_t mrcGinCreateSignalMap(struct mrcGinContext* ctx, int c, uint64_t* slots, int nranks) {
  const uint32_t rdwBase = (uint32_t)ctx->nSignals + 1;
  DOCACHECK(doca_gpu_verbs_signal_map_create(&ctx->maps[c]));
  if (ctx->nSignals > 0) DOCACHECK(doca_gpu_verbs_signal_register_range(ctx->maps[c], 0, ctx->nSignals - 1, slots));
  DOCACHECK(doca_gpu_verbs_signal_register_range(ctx->maps[c], rdwBase, rdwBase + 3 * nranks - 1, slots + rdwBase));
  ctx->mapRegistered[c] = true;
  return ncclSuccess;
}

static ncclResult_t mrcGinCreateContext(void* collComm, ncclGinConfig_v14_t* config, void** outGinCtx,
                                        ncclNetDeviceHandle_v11_t** outDevHandle) {
  struct mrcGinCollComm* cComm = (struct mrcGinCollComm*)collComm;
  ncclResult_t status = ncclSuccess;

  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int ncontexts = config->nContexts;
  const int nCounters = config->nCounters;
  const int nSignals = config->nSignals;
  // The device code NCCL was built with decides who drains the receive queues (gin_mrc_dev.h).
  const bool gpuRq = config->backendVersion == NCCL_GIN_MRC_VERSION_GPU_RQ;
  // CPU RQ: signals and RDW counters, each context's slice on its own 64 KiB GPU page, the unit a
  // signal map pins, so the contexts' maps never pin the same page. GPU RQ: signals and the sink.
  const unsigned int signalTableSize =
    gpuRq ? (unsigned)nSignals + 1 : NCCL_GIN_MRC_SIGNAL_TABLE_SIZE(nSignals, nranks);
  const unsigned int kSlotsPerPinPage = 65536 / sizeof(uint64_t);
  const unsigned int signalTableStride = gpuRq ? signalTableSize : ROUNDUP(signalTableSize, kSlotsPerPinPage);
  const int64_t rqPostMode = ncclParamGinMrcRqPostMode();
  uint32_t rdwGen = 0;
  const int nqpsForComm = ncontexts * nranks;
  const int nqps = nqpsForComm + ncontexts; // + one self-loop responder QP per context

  struct mrcGinContext* ctx = nullptr;
  ncclNetDeviceHandle_t* devHandle = nullptr;
  struct mrcGinExchInfo* localInfo = nullptr;
  struct mrcGinExchInfo* remoteInfo = nullptr;
  struct doca_gpu_verbs_qp** gverbsQps = nullptr;
  struct ncclGinMrcPeerState* peersHost = nullptr;
  struct mrcQpAttr qpAttr;
  char pciBusId[MAX_PCI_ADDRESS_LEN];
  int gidIndex = (int)ncclParamGinMrcGidIndex();
  const int tc = config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF ? config->trafficClass :
                 ncclParamGinMrcTc() != -1                            ? (int)ncclParamGinMrcTc() :
                                                                        0;
  const int sl = ncclParamGinMrcSl() != -1 ? (int)ncclParamGinMrcSl() : 0;
  const int64_t getWindow = ncclParamGinMrcGetWindow();
  if (getWindow < 1 || getWindow > 256 || (getWindow & (getWindow - 1)) != 0) {
    WARN("GIN/MRC : invalid NCCL_GIN_MRC_GET_WINDOW=%ld (must be a power of two in [1, 256])", (long)getWindow);
    return ncclInvalidUsage;
  }
  if (config->backendVersion != NCCL_GIN_MRC_VERSION_CPU_RQ && !gpuRq) {
    WARN("GIN/MRC : NCCL's MRC device code has layout version %d; this plugin supports %d (CPU RQ) and %d (GPU RQ)",
         config->backendVersion, NCCL_GIN_MRC_VERSION_CPU_RQ, NCCL_GIN_MRC_VERSION_GPU_RQ);
    return ncclInvalidUsage;
  }
#ifndef MRC_HAVE_GPU_RQ
  if (gpuRq) {
    WARN("GIN/MRC : NCCL was built with GIN_MRC_RQ=gpu, but this plugin was built without GPU RQ support: rebuild "
         "it with a DOCA_HOME that provides doca_gpu_verbs_export_sdk_qp (GPUNetIO feature/export_sdk_qp)");
    return ncclInvalidUsage;
  }
#endif
  if (gpuRq && rqPostMode != NCCL_GIN_MRC_RQ_POST_MODE_DBR && rqPostMode != NCCL_GIN_MRC_RQ_POST_MODE_WQE) {
    WARN("GIN/MRC : invalid NCCL_GIN_MRC_RQ_POST_MODE=%ld (0 = DBR-only update, 1 = regular post-recv)",
         (long)rqPostMode);
    return ncclInvalidUsage;
  }
  // CPU RQ: signal ids travel in the immediate, so a context's whole table must fit the id field.
  if (!gpuRq && signalTableSize > DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS) {
    WARN("GIN/MRC : %d signals + %d RDW counters exceed the %u signal ids of a Write-with-Immediate; rebuild "
         "GPUNetIO, NCCL and this plugin with -DDOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_SHIFT=24 for 8M ids",
         nSignals, 3 * nranks + 1, (unsigned)DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS);
    return ncclInvalidUsage;
  }
  // A rank stride > 1 would make the device-side peer index (compacted by NCCL) disagree with
  // the stride-sparse QP array that the progress loops walk.
  if (config->rankStride != 1) {
    WARN("GIN/MRC : rank stride %d is not supported (set NCCL_P2P_DISABLE=1)", config->rankStride);
    return ncclInvalidUsage;
  }

  ctx = (struct mrcGinContext*)calloc(1, sizeof(*ctx));
  if (ctx == nullptr) return ncclSystemError;
  new (&ctx->progressStop) std::atomic<bool>(false);
  new (&ctx->progressError) std::atomic<int>(0);
  ctx->collComm = cComm;
  ctx->rank = rank;
  ctx->nContexts = ncontexts;
  ctx->nSignals = nSignals;
  ctx->signalTableSize = signalTableSize;
  ctx->dataDirect = cComm->dd.mr != nullptr;
  ctx->gpuRq = gpuRq;
  ctx->counters = new MrcGlobalTable();
  ctx->signals = new MrcGlobalTable();
  ctx->rqQps = new std::vector<struct doca_gpu_verbs_qp*>[ncontexts];

  devHandle = (ncclNetDeviceHandle_t*)calloc(1, sizeof(*devHandle));
  EQCHECKGOTO(devHandle, nullptr, status, out);
  ctx->devHandle = devHandle;

  ctx->qps = (struct doca_gpu_verbs_qp_hl**)calloc(nqps, sizeof(*ctx->qps));
  EQCHECKGOTO(ctx->qps, nullptr, status, out);
  ctx->ctxStaging = (struct ncclGinMrcGPUContext*)calloc(ncontexts, sizeof(*ctx->ctxStaging));
  EQCHECKGOTO(ctx->ctxStaging, nullptr, status, out);
  ctx->maps = (struct doca_gpu_verbs_signal_map**)calloc(ncontexts, sizeof(*ctx->maps));
  EQCHECKGOTO(ctx->maps, nullptr, status, out);
  ctx->mapRegistered = (bool*)calloc(ncontexts, sizeof(*ctx->mapRegistered));
  EQCHECKGOTO(ctx->mapRegistered, nullptr, status, out);
  localInfo = (struct mrcGinExchInfo*)calloc(nranks, sizeof(*localInfo));
  EQCHECKGOTO(localInfo, nullptr, status, out);
  remoteInfo = (struct mrcGinExchInfo*)calloc((size_t)ncontexts * nranks, sizeof(*remoteInfo));
  EQCHECKGOTO(remoteInfo, nullptr, status, out);
  gverbsQps = (struct doca_gpu_verbs_qp**)calloc(nranks, sizeof(*gverbsQps));
  EQCHECKGOTO(gverbsQps, nullptr, status, out);

  gethostname(ctx->hostname, HOST_NAME_MAX);
  snprintf(ctx->ibDevName, MAXNAMESIZE, "%s", ncclIbDevs[cComm->dev].devName);

  CUDACHECKGOTO(cudaGetDevice(&ctx->cudaId), status, out);
  CUDACHECKGOTO(cudaDeviceGetPCIBusId(pciBusId, MAX_PCI_ADDRESS_LEN, ctx->cudaId), status, out);
  DOCACHECKGOTO(doca_gpu_create(pciBusId, &ctx->gdev), status, out);
  DOCACHECKGOTO(doca_verbs_dev_open(cComm->pd, &ctx->netDev), status, out);

  NCCLCHECKGOTO(wrap_ibv_query_device(cComm->context, &ctx->devAttr), status, out);
  ctx->portNum = ncclIbDevs[cComm->dev].portNum;
  NCCLCHECKGOTO(wrap_ibv_query_port(cComm->context, ctx->portNum, &ctx->portAttr), status, out);
  NCCLCHECKGOTO(wrap_ibv_query_gid(cComm->context, ctx->portNum, gidIndex, &ctx->rgid), status, out);
  ctx->gidIndex = gidIndex;
  ctx->pkeyIndex = (int)ncclParamGinMrcPkeyIndex();

  // Only the sink slots of the signal table are written by the NIC (bare-signal payloads); the
  // rest is delivered by the signal maps, which need no MR.
  NCCLCHECKGOTO(ctx->counters->allocate((unsigned)nCounters * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(ctx->signals->allocate(signalTableStride * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(ctx->counters->registerMr(cComm->pd), status, out);
  NCCLCHECKGOTO(ctx->signals->registerMr(cComm->pd), status, out);
  {
    uint32_t rkey = htobe32(ctx->signals->mr->rkey);
    NCCLCHECKGOTO(mrcGinAllGather(cComm, &rkey, ctx->signals->rkeys.host, sizeof(uint32_t)), status, out);
    NCCLCHECKGOTO(ctx->signals->rkeys.push(), status, out);
  }
  if (!gpuRq && nSignals > 0) {
    CUDACHECKGOTO(cudaMalloc(&ctx->signalBase, (size_t)ncontexts * nSignals * sizeof(uint64_t)), status, out);
    CUDACHECKGOTO(cudaMemset(ctx->signalBase, 0, (size_t)ncontexts * nSignals * sizeof(uint64_t)), status, out);
  }

  NCCLCHECKGOTO(mrcGinCreateAh(ctx, sl, tc, gidIndex), status, out);

  ctx->qpSqSize = config->queueDepth > 0 ? config->queueDepth : (uint32_t)ncclParamGinMrcQpDepth();
  ctx->qpRqSize = ncclParamGinMrcRqDepth() > 0 ? (uint32_t)ncclParamGinMrcRqDepth() : gpuRq ? ctx->qpSqSize : 1024;
  if (ctx->qpRqSize < ctx->qpSqSize) ctx->qpRqSize = ctx->qpSqSize;
  ctx->minRnrTimer = ncclParamGinMrcMinRnrTimer() >= 0 ? (uint8_t)ncclParamGinMrcMinRnrTimer() : gpuRq ? 12 : 1;
  if ((uint32_t)getWindow > ctx->qpRqSize) {
    WARN("GIN/MRC : NCCL_GIN_MRC_GET_WINDOW=%ld exceeds the receive queue depth %u", (long)getWindow, ctx->qpRqSize);
    status = ncclInvalidUsage;
    goto out;
  }

  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.gpu_dev = ctx->gdev;
  qpAttr.net_dev = ctx->netDev;
  qpAttr.qp_type = MRC_QP_TYPE_RCX;
  qpAttr.sq_nwqe = ctx->qpSqSize;
  qpAttr.rq_nwqe = ctx->qpRqSize;
  // Data Direct ordering: in CPU RQ mode the signal maps deliver through forced-PCIe mappings; in
  // GPU RQ mode the device drain flushes before publishing (ncclGinMrcGPUContext::dd_flush).
  qpAttr.data_direct = ctx->dataDirect && !gpuRq;
  qpAttr.rq_gpu = gpuRq;
  // The GPU must ring the doorbell directly; the CPU-proxy path cannot drive RCX. Force a GPU-SM
  // handler rather than AUTO, which could silently fall back to the proxy.
  qpAttr.nic_handler = (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinMrcNicHandler();
  if (qpAttr.nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
      qpAttr.nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF)
    qpAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;

  for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
    DOCACHECKGOTO(mrcQpCreate(&qpAttr, &ctx->qps[qpIdx]), status, out);
  }
  // GPU RQ: hand each QP to the DOCA SDK, whose device code (NCCL's) posts and drains its receive
  // queue. The queues stay owned by the open-source library, which releases the export on destroy.
  if (gpuRq) {
    NCCLCHECKGOTO(ctx->sdkQps.allocate(nqps), status, out);
    for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
      DOCACHECKGOTO(doca_gpu_verbs_export_sdk_qp(ctx->qps[qpIdx]->qp_gverbs, &ctx->sdkQps.host[qpIdx]), status, out);
    }
    NCCLCHECKGOTO(ctx->sdkQps.push(), status, out);
  }

  // Exchange QP numbers, one round per context: rank r's entry of the all-to-all carries the QP
  // that r will use to talk to the receiving rank.
  for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
    for (int peer = 0; peer < nranks; peer++) {
      mrcGinFillExchInfo(&localInfo[peer], ctx, ctx->qps[peer + ctxIdx * nranks]);
    }
    NCCLCHECKGOTO(mrcGinAllToAll(cComm, localInfo, &remoteInfo[(size_t)ctxIdx * nranks], sizeof(*localInfo)), status,
                  out);
  }

  {
    int nConnected = 0;
    for (int peer = 0; peer < nranks; peer++) {
      if (peer == rank) continue;
      for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
        status = mrcGinConnectQp(ctx, ctx->qps[peer + ctxIdx * nranks], &remoteInfo[(size_t)ctxIdx * nranks + peer]);
        if (status != ncclSuccess) {
          // The firmware reports a generic bad-parameter error both when the peer NIC is on another
          // network plane and when this NIC hit its cap of connected MRC QPs (~3K, shared by every
          // process on it).
          WARN("GIN/MRC : QP connect failed after %d of %d QPs (%d contexts x %d peers) on %s. Check that "
               "NCCL_IB_HCA rail/plane IDs pair NICs on the same plane; at 0 QPs that is the likely cause, "
               "otherwise the NIC may have run out of MRC QPs (request fewer GIN contexts).",
               nConnected, nqpsForComm - ncontexts, ncontexts, nranks - 1, ctx->ibDevName);
          goto out;
        }
        nConnected++;
      }
    }
  }

  // Self traffic loops through a dedicated responder QP: an MRC QP cannot target itself, and the
  // responder's receive queue takes the Write-with-Immediate signals this rank sends itself.
  for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
    struct mrcGinExchInfo self;
    mrcGinFillExchInfo(&self, ctx, ctx->qps[nqpsForComm + ctxIdx]);
    NCCLCHECKGOTO(mrcGinConnectQp(ctx, ctx->qps[rank + ctxIdx * nranks], &self), status, out);
    mrcGinFillExchInfo(&self, ctx, ctx->qps[rank + ctxIdx * nranks]);
    NCCLCHECKGOTO(mrcGinConnectQp(ctx, ctx->qps[nqpsForComm + ctxIdx], &self), status, out);
  }

  NCCLCHECKGOTO(mrcCuMemAlloc(&ctx->sink.addr, &ctx->sink.handle, sizeof(uint64_t)), status, out);
  NCCLCHECKGOTO(mrcRegMr(&ctx->sink.mr, cComm->pd, ctx->sink.addr, sizeof(uint64_t),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
                status, out);

  // RDW get buffer: every rank uses the same layout, so a context's region sits at the same
  // offset of every peer's zero-based RDW MR. In CPU RQ mode with Data Direct it is registered like
  // the windows, so a request descriptor cannot be overtaken by the request counter the signal map
  // delivers; in GPU RQ mode it stays on C2C, so the drain can serve a request without a flush.
  ctx->rdw.window = (uint32_t)getWindow;
  ctx->rdw.ctxStride =
    2 * (size_t)nranks * ctx->rdw.window * NCCL_GIN_MRC_RDW_DESC_BYTES + NCCL_GIN_MRC_RDW_DESC_BYTES;
  NCCLCHECKGOTO(mrcCuMemAlloc(&ctx->rdw.addr, &ctx->rdw.handle, (size_t)ncontexts * ctx->rdw.ctxStride), status, out);
  NCCLCHECKGOTO(mrcRegMr(&ctx->rdw.mr, cComm->pd, ctx->rdw.addr, (size_t)ncontexts * ctx->rdw.ctxStride,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE, /*forceStrictOrdering=*/true,
                         ctx->dataDirect && !gpuRq),
                status, out);
  {
    uint32_t rdwRkey = htobe32(ctx->rdw.mr->rkey);
    NCCLCHECKGOTO(ctx->rdwRkeys.allocate(nranks), status, out);
    NCCLCHECKGOTO(mrcGinAllGather(cComm, &rdwRkey, ctx->rdwRkeys.host, sizeof(uint32_t)), status, out);
    NCCLCHECKGOTO(ctx->rdwRkeys.push(), status, out);
  }
  NCCLCHECKGOTO(mrcGinMrTableEnsure(cComm), status, out);
  rdwGen = mrcGinRdwGen.fetch_add(1) & NCCL_GIN_MRC_RDW_GEN_MASK;

  // Per-peer device state. Counter-drain slot tables are a power of two at least as large as the
  // SQ depth, so each outstanding WQE index maps to a unique slot.
  CUDACHECKGOTO(cudaMalloc(&ctx->peers, (size_t)ncontexts * nranks * sizeof(struct ncclGinMrcPeerState)), status, out);
  if (nCounters > 0) {
    ctx->counterCap = 1;
    while (ctx->counterCap < ctx->qpSqSize) ctx->counterCap <<= 1;
    size_t slotBytes = (size_t)ncontexts * nranks * ctx->counterCap * sizeof(uint32_t);
    CUDACHECKGOTO(cudaMalloc(&ctx->counterSlots, slotBytes), status, out);
    CUDACHECKGOTO(cudaMemset(ctx->counterSlots, 0, slotBytes), status, out);
  }
  peersHost = (struct ncclGinMrcPeerState*)calloc((size_t)ncontexts * nranks, sizeof(*peersHost));
  EQCHECKGOTO(peersHost, nullptr, status, out);
  for (size_t i = 0; i < (size_t)ncontexts * nranks; i++) {
    if (ctx->counterSlots == nullptr) continue;
    peersHost[i].counter_slots = ctx->counterSlots + i * ctx->counterCap;
    peersHost[i].sq_wqe_mask = ctx->counterCap - 1;
  }
  CUDACHECKGOTO(cudaMemcpy(ctx->peers, peersHost, (size_t)ncontexts * nranks * sizeof(*peersHost),
                           cudaMemcpyHostToDevice),
                status, out);

  NCCLCHECKGOTO(ctx->ctxBuf.allocate(ncontexts), status, out);

  for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
    struct ncclGinMrcGPUContext* gpuCtx = &ctx->ctxStaging[ctxIdx];
    unsigned int start;

    if (gpuRq) {
      gpuCtx->sdk_qps = ctx->sdkQps.gpu + (size_t)ctxIdx * nranks;
      gpuCtx->self_rq_qp = ctx->sdkQps.host[nqpsForComm + ctxIdx];
      gpuCtx->dd_flush = cComm->dd.mr != nullptr;
      gpuCtx->dd_flush_lkey = cComm->dd.mr ? htobe32(cComm->dd.mr->lkey) : 0;
      gpuCtx->rdw_gen = rdwGen;
    } else {
      for (int peer = 0; peer < nranks; peer++) gverbsQps[peer] = ctx->qps[ctxIdx * nranks + peer]->qp_gverbs;
      DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(ctx->gdev, gverbsQps, nranks,
                                                        (struct doca_gpu_dev_verbs_qp**)&gpuCtx->qps),
                    status, out);
    }

    if (nCounters) {
      NCCLCHECKGOTO(ctx->counters->take(nCounters, &start), status, out);
      gpuCtx->counters = ctx->counters->gpu + start;
    }
    NCCLCHECKGOTO(ctx->signals->take(signalTableStride, &start), status, out);
    gpuCtx->signals = ctx->signals->gpu + start;
    gpuCtx->signal_base = ctx->signalBase ? ctx->signalBase + (size_t)ctxIdx * nSignals : nullptr;
    gpuCtx->signal_rkeys = ctx->signals->rkeys.gpu;
    // Absolute element index within the signal MR, since it is used as a remote offset.
    gpuCtx->signals_sink_offset = start + nSignals;
    if (!gpuRq) {
      gpuCtx->rdw_signal_base = nSignals + 1;
      NCCLCHECKGOTO(mrcGinCreateSignalMap(ctx, ctxIdx, gpuCtx->signals, nranks), status, out);

      // Peers deliver into this context through their QP to us; our own traffic through the responder.
      for (int peer = 0; peer < nranks; peer++)
        if (peer != rank) ctx->rqQps[ctxIdx].push_back(ctx->qps[ctxIdx * nranks + peer]->qp_gverbs);
      ctx->rqQps[ctxIdx].push_back(ctx->qps[nqpsForComm + ctxIdx]->qp_gverbs);
    }

    gpuCtx->peers = ctx->peers + (size_t)ctxIdx * nranks;
    gpuCtx->sink_lkey = htobe32(ctx->sink.mr->lkey);
    gpuCtx->mr_table = cComm->mrTable.gpu;
    gpuCtx->rdw_region = (uint8_t*)ctx->rdw.addr + (size_t)ctxIdx * ctx->rdw.ctxStride;
    gpuCtx->rdw_rkeys = ctx->rdwRkeys.gpu;
    gpuCtx->rdw_ctx_offset = (uint64_t)ctxIdx * ctx->rdw.ctxStride;
    gpuCtx->rdw_lkey = htobe32(ctx->rdw.mr->lkey);
    gpuCtx->rdw_window = ctx->rdw.window;

    ctx->ctxBuf.host[ctxIdx] = *gpuCtx;
  }
  NCCLCHECKGOTO(ctx->ctxBuf.push(), status, out);

#ifdef MRC_HAVE_GPU_RQ
  // GPU RQ: the receive rings are posted by the device code, before any peer can send.
  if (gpuRq) {
    for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
      uint32_t* slots = ctx->counterSlots ? ctx->counterSlots + (size_t)ctxIdx * nranks * ctx->counterCap : nullptr;
      cudaError_t cerr = mrcGinInitRqsLaunch(ctx->sdkQps.gpu + (size_t)ctxIdx * nranks,
                                             ctx->sdkQps.host[nqpsForComm + ctxIdx], rank,
                                             ctx->peers + (size_t)ctxIdx * nranks, nranks, slots, ctx->counterCap,
                                             (uint32_t)rqPostMode, 0);
      if (cerr != cudaSuccess) {
        WARN("GIN/MRC : receive-queue init kernel launch failed: %s", cudaGetErrorString(cerr));
        status = ncclUnhandledCudaError;
        goto out;
      }
    }
    CUDACHECKGOTO(cudaDeviceSynchronize(), status, out);
  }
#endif

  // Data Direct QPs need signal maps with forced-PCIe mappings, which GDRCopy may not support;
  // rq_progress refuses them up front, before touching the queue.
  if (ctx->dataDirect && !gpuRq) {
    bool progressed = false;
    doca_error_t err = doca_gpu_verbs_rq_progress(ctx->rqQps[0][0], ctx->maps[0], &progressed);
    if (err != DOCA_SUCCESS) {
      WARN("GIN/MRC : signal delivery over Data Direct is unavailable (DOCA error %d, GDRCopy without forced-PCIe "
           "mappings?); set NCCL_GIN_MRC_DATA_DIRECT=0",
           err);
      status = ncclSystemError;
      goto out;
    }
  }

  if (!gpuRq) {
    if (pthread_create(&ctx->progressThread, nullptr, mrcGinProgressMain, ctx) != 0) {
      WARN("GIN/MRC : failed to start the receive-queue progress thread");
      status = ncclSystemError;
      goto out;
    }
    ctx->progressStarted = true;
  }

  devHandle->netDeviceType = (ncclNetDeviceType)NCCL_NET_DEVICE_GIN_MRC;
  devHandle->netDeviceVersion = config->backendVersion;
  devHandle->handle = (void*)ctx->ctxBuf.gpu;
  devHandle->size = 0;
  devHandle->needsProxyProgress = 0;

  *outDevHandle = devHandle;
  *outGinCtx = ctx;

  INFO(NCCL_INIT | NCCL_NET,
       "GIN/MRC : rank %d created %d contexts over %d ranks on %s (%s RQ, RQ depth %u, RDW get window %u, Data Direct "
       "%s, %u signal ids per context)",
       rank, ncontexts, nranks, ctx->ibDevName, gpuRq ? "GPU" : "CPU", ctx->qpRqSize, ctx->rdw.window,
       ctx->dataDirect ? "on" : "off", signalTableSize);

out:
  free(localInfo);
  free(remoteInfo);
  free(gverbsQps);
  free(peersHost);
  if (status != ncclSuccess) mrcGinDestroyContext(ctx);
  return status;
}

static ncclResult_t mrcGinDestroyContext(void* ginCtx) {
  struct mrcGinContext* ctx = (struct mrcGinContext*)ginCtx;
  if (ctx == nullptr) return ncclSuccess;

  const int nranks = ctx->collComm->nranks;
  const int ncontexts = ctx->nContexts;
  const int nqpsForComm = ncontexts * nranks;
  const int nqps = nqpsForComm + ncontexts;

  // Delivery reads the maps and queues without locks, so it must stop before either goes away.
  if (ctx->progressStarted) {
    ctx->progressStop.store(true);
    pthread_join(ctx->progressThread, nullptr);
  }

  if (ctx->ctxStaging && ctx->qps) {
    struct doca_gpu_verbs_qp** gverbsQps =
      (struct doca_gpu_verbs_qp**)calloc(nranks, sizeof(struct doca_gpu_verbs_qp*));
    for (int ctxIdx = 0; ctxIdx < ncontexts && gverbsQps; ctxIdx++) {
      struct ncclGinMrcGPUContext* gpuCtx = &ctx->ctxStaging[ctxIdx];
      if (gpuCtx->qps == nullptr) continue; // CPU RQ only
      for (int peer = 0; peer < nranks; peer++) gverbsQps[peer] = ctx->qps[ctxIdx * nranks + peer]->qp_gverbs;
      doca_gpu_verbs_unexport_multi_qps_dev(ctx->gdev, gverbsQps, nranks, gpuCtx->qps);
      gpuCtx->qps = nullptr;
    }
    free(gverbsQps);
  }

  if (ctx->qps) {
    for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
      if (ctx->qps[qpIdx]) mrcQpDestroy(ctx->qps[qpIdx]);
    }
    free(ctx->qps);
  }

  if (ctx->maps) {
    const uint32_t rdwBase = (uint32_t)ctx->nSignals + 1;
    for (int c = 0; c < ncontexts; c++) {
      if (ctx->maps[c] == nullptr) continue;
      if (ctx->mapRegistered[c]) {
        if (ctx->nSignals > 0) doca_gpu_verbs_signal_unregister_range(ctx->maps[c], 0, ctx->nSignals - 1);
        doca_gpu_verbs_signal_unregister_range(ctx->maps[c], rdwBase, rdwBase + 3 * nranks - 1);
      }
      doca_gpu_verbs_signal_map_destroy(ctx->maps[c]);
    }
    free(ctx->maps);
  }
  free(ctx->mapRegistered);
  delete[] ctx->rqQps;
  ctx->sdkQps.free_();

  free(ctx->ctxStaging);
  ctx->ctxBuf.free_();

  if (ctx->counters) {
    ctx->counters->free_();
    delete ctx->counters;
  }
  // The signal table outlives its maps: they pin its pages until destroyed.
  if (ctx->signals) {
    ctx->signals->free_();
    delete ctx->signals;
  }
  if (ctx->signalBase) cudaFree(ctx->signalBase);

  if (ctx->peers) cudaFree(ctx->peers);
  if (ctx->counterSlots) cudaFree(ctx->counterSlots);

  if (ctx->sink.mr) wrap_ibv_dereg_mr(ctx->sink.mr);
  if (ctx->sink.addr) mrcCuMemFree(ctx->sink.addr);

  if (ctx->rdw.mr) wrap_ibv_dereg_mr(ctx->rdw.mr);
  if (ctx->rdw.addr) mrcCuMemFree(ctx->rdw.addr);
  ctx->rdwRkeys.free_();

  if (ctx->ah) doca_verbs_ah_attr_destroy(ctx->ah);
  if (ctx->netDev) doca_verbs_dev_close(ctx->netDev);
  if (ctx->gdev) doca_gpu_destroy(ctx->gdev);
  free(ctx->devHandle);

  free(ctx);
  return ncclSuccess;
}

static ncclResult_t mrcGinRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle,
                                   void** ginHandle) {
  struct mrcGinCollComm* cComm = (struct mrcGinCollComm*)collComm;
  struct mrcGinMemHandle* mh = new mrcGinMemHandle();
  ncclResult_t status = ncclSuccess;
  uint32_t rkey;

  NCCLCHECKGOTO(mrcRegMr(&mh->mr, cComm->pd, data, size,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
                         (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) != 0, /*dataDirect=*/cComm->dd.mr != nullptr),
                status, fail);

  rkey = htobe32(mh->mr->rkey);
  NCCLCHECKGOTO(mh->rkeys.allocate(cComm->nranks), status, fail);
  NCCLCHECKGOTO(mrcGinAllGather(cComm, &rkey, mh->rkeys.host, sizeof(uint32_t)), status, fail);
  NCCLCHECKGOTO(mh->rkeys.push(), status, fail);

  NCCLCHECKGOTO(mh->devHandle.allocate(1), status, fail);
  mh->devHandle.host->rkeys = mh->rkeys.gpu;
  mh->devHandle.host->lkey = htobe32(mh->mr->lkey);
  NCCLCHECKGOTO(mh->devHandle.push(), status, fail);
  NCCLCHECKGOTO(mrcGinMrTableAdd(cComm, mh->mr, size), status, fail);

  mh->type = type;
  *mhandle = mh;
  *ginHandle = mh->devHandle.gpu;
  return ncclSuccess;

fail:
  if (mh->mr) wrap_ibv_dereg_mr(mh->mr);
  mh->rkeys.free_();
  mh->devHandle.free_();
  delete mh;
  return status;
}

static ncclResult_t mrcGinRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                         uint64_t mrFlags, void** mhandle, void** ginHandle) {
  (void)offset;
  (void)fd;
  return mrcGinRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t mrcGinDeregMrSym(void* collComm, void* mhandle) {
  struct mrcGinMemHandle* mh = (struct mrcGinMemHandle*)mhandle;
  if (mh == nullptr) return ncclSuccess;

  NCCLCHECK(mrcGinMrTableRemove((struct mrcGinCollComm*)collComm, mh->mr));
  NCCLCHECK(wrap_ibv_dereg_mr(mh->mr));
  NCCLCHECK(mh->rkeys.free_());
  NCCLCHECK(mh->devHandle.free_());
  delete mh;
  return ncclSuccess;
}

// All doorbells are rung by the GPU, and the receive queues are drained by the plugin's own
// progress thread, so there is nothing for NCCL's proxy to advance.
static ncclResult_t mrcGinProgress(void* ginCtx) {
  (void)ginCtx;
  return ncclSuccess;
}

// Receive-queue errors are latched by the progress thread; send-side errors surface as CQ errors
// on the device side.
static ncclResult_t mrcGinQueryLastError(void* ginCtx, bool* hasError) {
  struct mrcGinContext* ctx = (struct mrcGinContext*)ginCtx;
  *hasError = ctx != nullptr && ctx->progressError.load(std::memory_order_relaxed) != 0;
  return ncclSuccess;
}

static ncclResult_t mrcGinFinalize(void* ctx) {
  (void)ctx;
  return ncclIbFinalizeDevices();
}

ncclGin_v14_t NCCL_GIN_PLUGIN_SYMBOL = {
  "MRC",
  mrcGinInit,
  mrcGinDevices,
  mrcGinGetGinProperties,
  mrcGinGetProperties,
  mrcGinListen,
  mrcGinConnect,
  mrcGinCreateContext,
  mrcGinRegMrSym,
  mrcGinRegMrSymDmaBuf,
  mrcGinDeregMrSym,
  mrcGinDestroyContext,
  mrcGinCloseColl,
  mrcGinCloseListen,
  mrcGinProgress,
  mrcGinQueryLastError,
  mrcGinFinalize,
};
