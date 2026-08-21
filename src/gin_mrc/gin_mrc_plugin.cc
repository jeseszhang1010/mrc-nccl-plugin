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
 * MRC QPs speak RDMA Write and Write-with-Immediate only, which shapes the
 * setup here: every QP needs a receive ring (signals arrive as inbound
 * Write-with-Immediate), there are no companion/atomic QPs, and Get is
 * unsupported.
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

#include <mutex>

#include "transport/net_ib/common.h"

#include "gin/gin_v14.h"
#include "gin_mrc.h"
#include "mrc_doca_ext.h"

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
NCCL_PARAM(GinMrcNicHandler, "GIN_MRC_NIC_HANDLER", 0);
NCCL_PARAM(GinMrcGidIndex, "IB_GID_INDEX", 3);
NCCL_PARAM(GinMrcTc, "GIN_MRC_TC", -1);
NCCL_PARAM(GinMrcSl, "GIN_MRC_SL", -1);
NCCL_PARAM(GinMrcTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(GinMrcRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(GinMrcPkeyIndex, "IB_PKEY_INDEX", 0);

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

static ncclResult_t mrcRegMr(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access,
                             bool forceStrictOrdering = false) {
  if (!forceStrictOrdering && ncclIbRelaxedOrderingEnabled) access |= IBV_ACCESS_RELAXED_ORDERING;
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

  struct doca_gpu* gdev;
  struct doca_verbs_context* verbsContext;
  struct doca_verbs_ah_attr* ah;
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
  struct doca_gpu_verbs_qp_hl** qps; // [nContexts * (nranks + 1)]

  MrcGlobalTable* counters;
  MrcGlobalTable* signals;

  struct ncclGinMrcGPUContext* ctxStaging; // host mirror of the device array
  MrcHostGpuBuf<struct ncclGinMrcGPUContext> ctxBuf;

  struct ncclGinMrcRqState* rq;   // [nContexts * nranks]
  uint32_t* counterSlots;         // [nContexts * nranks * counterCap]
  uint32_t counterCap;

  struct {
    void* addr;
    CUmemGenericAllocationHandle handle;
    struct ibv_mr* mr;
  } sink;

  ncclNetDeviceHandle_t* devHandle;
};

/* ── QP setup ─────────────────────────────────────────────────────── */

static void mrcGinFillExchInfo(struct mrcGinExchInfo* info, struct mrcGinContext* ctx,
                               struct doca_gpu_verbs_qp_hl* qp) {
  memset(info, 0, sizeof(*info));
  info->lid = ctx->portAttr.lid;
  info->qpn = doca_verbs_qp_get_qpn(qp->qp);
  memcpy(info->vgid.raw, ctx->rgid.raw, sizeof(union ibv_gid));
  info->rank = ctx->rank;
  info->cudaId = ctx->cudaId;
  snprintf(info->hostname, HOST_NAME_MAX, "%s", ctx->hostname);
  snprintf(info->ibDevName, MAXNAMESIZE, "%s", ctx->ibDevName);
}

static ncclResult_t mrcGinCreateAh(struct mrcGinContext* ctx, struct ibv_context* ibContext, int sl, int tc,
                                   int gidIndex) {
  ncclResult_t status = ncclSuccess;

  DOCACHECK(doca_verbs_bridge_verbs_context_import(ibContext, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
                                                   &ctx->verbsContext));
  DOCACHECK(doca_verbs_ah_attr_create(ctx->verbsContext, &ctx->ah));

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
  struct doca_verbs_qp_attr* attr = nullptr;

  DOCACHECK(doca_verbs_ah_attr_set_gid(ctx->ah, peer->vgid));
  DOCACHECK(doca_verbs_ah_attr_set_dlid(ctx->ah, peer->lid));
  DOCACHECK(doca_verbs_qp_attr_create(&attr));

  DOCACHECKGOTO(doca_verbs_qp_attr_set_path_mtu(attr, DOCA_MTU_SIZE_4K_BYTES), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(attr, 0), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(attr, 0), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(attr, ctx->portNum), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(attr, ncclParamGinMrcTimeout()), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(attr, ncclParamGinMrcRetryCnt()), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(attr, 7), status, fail);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(attr, 12), status, fail);
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

static ncclResult_t mrcGinGetProperties(int dev, ncclNetProperties_v12_t* props) {
  NCCLCHECK(ncclIbGetProperties(dev, props));

  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->railId = -1;
  props->planeId = -1;

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

static ncclResult_t mrcGinCreateContext(void* collComm, ncclGinConfig_v14_t* config, void** outGinCtx,
                                        ncclNetDeviceHandle_v11_t** outDevHandle) {
  struct mrcGinCollComm* cComm = (struct mrcGinCollComm*)collComm;
  ncclResult_t status = ncclSuccess;

  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int ncontexts = config->nContexts;
  const int nCounters = config->nCounters;
  const int nSignals = config->nSignals;
  // One extra slot per context backs the bare-signal Write-with-Immediate payload: MRC rejects
  // zero-length writes, and the payload must not land on a signal slot.
  const int numSignals = nSignals ? nSignals + 1 : 0;
  const int nqpsForComm = ncontexts * nranks;
  const int nqps = nqpsForComm + ncontexts; // + one self-loop responder QP per context

  struct mrcGinContext* ctx = nullptr;
  ncclNetDeviceHandle_t* devHandle = nullptr;
  struct mrcGinExchInfo* localInfo = nullptr;
  struct mrcGinExchInfo* remoteInfo = nullptr;
  struct doca_gpu_verbs_qp** gverbsQps = nullptr;
  struct doca_gpu_verbs_qp_init_attr_hl qpInitAttr;
  char pciBusId[MAX_PCI_ADDRESS_LEN];
  int gidIndex = (int)ncclParamGinMrcGidIndex();
  const int tc = config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF ? config->trafficClass :
                 ncclParamGinMrcTc() != -1                            ? (int)ncclParamGinMrcTc() :
                                                                        0;
  const int sl = ncclParamGinMrcSl() != -1 ? (int)ncclParamGinMrcSl() : 0;

  if (nCounters > 0 && nSignals == 0) {
    WARN("GIN/MRC : counters require a signal table to be present");
    return ncclInvalidUsage;
  }
  // A rank stride > 1 would make the device-side peer index (compacted by NCCL) disagree with
  // the stride-sparse QP array that the RQ init and drain loops walk.
  if (config->rankStride != 1) {
    WARN("GIN/MRC : rank stride %d is not supported (set NCCL_P2P_DISABLE=1)", config->rankStride);
    return ncclInvalidUsage;
  }

  ctx = (struct mrcGinContext*)calloc(1, sizeof(*ctx));
  if (ctx == nullptr) return ncclSystemError;
  ctx->collComm = cComm;
  ctx->rank = rank;
  ctx->nContexts = ncontexts;
  ctx->counters = new MrcGlobalTable();
  ctx->signals = new MrcGlobalTable();

  devHandle = (ncclNetDeviceHandle_t*)calloc(1, sizeof(*devHandle));
  EQCHECKGOTO(devHandle, nullptr, status, out);
  ctx->devHandle = devHandle;

  ctx->qps = (struct doca_gpu_verbs_qp_hl**)calloc(nqps, sizeof(*ctx->qps));
  EQCHECKGOTO(ctx->qps, nullptr, status, out);
  ctx->ctxStaging = (struct ncclGinMrcGPUContext*)calloc(ncontexts, sizeof(*ctx->ctxStaging));
  EQCHECKGOTO(ctx->ctxStaging, nullptr, status, out);
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

  NCCLCHECKGOTO(wrap_ibv_query_device(cComm->context, &ctx->devAttr), status, out);
  ctx->portNum = ncclIbDevs[cComm->dev].portNum;
  NCCLCHECKGOTO(wrap_ibv_query_port(cComm->context, ctx->portNum, &ctx->portAttr), status, out);
  NCCLCHECKGOTO(wrap_ibv_query_gid(cComm->context, ctx->portNum, gidIndex, &ctx->rgid), status, out);
  ctx->gidIndex = gidIndex;
  ctx->pkeyIndex = (int)ncclParamGinMrcPkeyIndex();

  NCCLCHECKGOTO(ctx->counters->allocate((unsigned)nCounters * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(ctx->signals->allocate((unsigned)numSignals * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(ctx->counters->registerMr(cComm->pd), status, out);
  NCCLCHECKGOTO(ctx->signals->registerMr(cComm->pd), status, out);
  if (ctx->signals->count) {
    uint32_t rkey = htobe32(ctx->signals->mr->rkey);
    NCCLCHECKGOTO(mrcGinAllGather(cComm, &rkey, ctx->signals->rkeys.host, sizeof(uint32_t)), status, out);
    NCCLCHECKGOTO(ctx->signals->rkeys.push(), status, out);
  }

  NCCLCHECKGOTO(mrcGinCreateAh(ctx, cComm->context, sl, tc, gidIndex), status, out);

  // MRC has no atomics: signals ride Write-with-Immediate, which needs a receive queue.
  ctx->qpSqSize = config->queueDepth > 0 ? config->queueDepth : (uint32_t)ncclParamGinMrcQpDepth();
  ctx->qpRqSize = ctx->qpSqSize;

  memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.gpu_dev = ctx->gdev;
  qpInitAttr.ibpd = cComm->pd;
  qpInitAttr.sq_nwqe = ctx->qpSqSize;
  qpInitAttr.rq_nwqe = ctx->qpRqSize;
  qpInitAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  qpInitAttr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  qpInitAttr.rcx_type = 0;
  // The GPU must ring the doorbell directly; the CPU-proxy path cannot drive RCX. Force a GPU-SM
  // handler rather than AUTO, which could silently fall back to the proxy.
  qpInitAttr.nic_handler = (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinMrcNicHandler();
  if (qpInitAttr.nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
      qpInitAttr.nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF)
    qpInitAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;

  for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
    DOCACHECKGOTO(doca_gpu_verbs_create_rcx_qp_hl(&qpInitAttr, &ctx->qps[qpIdx]), status, out);
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

  for (int peer = 0; peer < nranks; peer++) {
    if (peer == rank) continue;
    for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
      NCCLCHECKGOTO(mrcGinConnectQp(ctx, ctx->qps[peer + ctxIdx * nranks], &remoteInfo[(size_t)ctxIdx * nranks + peer]),
                    status, out);
    }
  }

  // Self traffic loops through a dedicated responder QP: an MRC QP cannot target itself, and the
  // responder needs its own receive ring for the write-with-immediate signals.
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

  CUDACHECKGOTO(cudaMalloc(&ctx->rq, (size_t)ncontexts * nranks * sizeof(struct ncclGinMrcRqState)), status, out);
  CUDACHECKGOTO(cudaMemset(ctx->rq, 0, (size_t)ncontexts * nranks * sizeof(struct ncclGinMrcRqState)), status, out);
  if (nCounters > 0) {
    // Power-of-two capacity at least as large as the SQ depth, so each outstanding WQE index maps
    // to a unique drain slot.
    ctx->counterCap = 1;
    while (ctx->counterCap < ctx->qpSqSize) ctx->counterCap <<= 1;
    size_t slotBytes = (size_t)ncontexts * nranks * ctx->counterCap * sizeof(uint32_t);
    CUDACHECKGOTO(cudaMalloc(&ctx->counterSlots, slotBytes), status, out);
    CUDACHECKGOTO(cudaMemset(ctx->counterSlots, 0, slotBytes), status, out);
  }

  NCCLCHECKGOTO(ctx->ctxBuf.allocate(ncontexts), status, out);

  for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
    struct ncclGinMrcGPUContext* gpuCtx = &ctx->ctxStaging[ctxIdx];
    unsigned int start;

    for (int peer = 0; peer < nranks; peer++) gverbsQps[peer] = ctx->qps[ctxIdx * nranks + peer]->qp_gverbs;
    DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(ctx->gdev, gverbsQps, nranks,
                                                      (struct doca_gpu_dev_verbs_qp**)&gpuCtx->qps),
                  status, out);

    if (nCounters) {
      NCCLCHECKGOTO(ctx->counters->take(nCounters, &start), status, out);
      gpuCtx->counters = ctx->counters->gpu + start;
    }
    if (numSignals) {
      NCCLCHECKGOTO(ctx->signals->take(numSignals, &start), status, out);
      gpuCtx->signals = ctx->signals->gpu + start;
      gpuCtx->signal_rkeys = ctx->signals->rkeys.gpu;
      // Absolute element index within the signal MR, since it is used as a remote offset.
      gpuCtx->signals_sink_offset = start + nSignals;
    }
    gpuCtx->rq = ctx->rq + (size_t)ctxIdx * nranks;
    gpuCtx->sink_lkey = htobe32(ctx->sink.mr->lkey);

    ctx->ctxBuf.host[ctxIdx] = *gpuCtx;
  }
  NCCLCHECKGOTO(ctx->ctxBuf.push(), status, out);

  for (int ctxIdx = 0; ctxIdx < ncontexts; ctxIdx++) {
    struct ncclGinMrcGPUContext* gpuCtx = &ctx->ctxStaging[ctxIdx];
    uint32_t* slots =
      ctx->counterSlots ? ctx->counterSlots + (size_t)ctxIdx * nranks * ctx->counterCap : nullptr;
    cudaError_t cerr = ncclGinMrcInitRqsLaunch((struct doca_gpu_dev_verbs_qp*)gpuCtx->qps, gpuCtx->rq, nranks, slots,
                                               ctx->counterCap, 0);
    if (cerr != cudaSuccess) {
      WARN("GIN/MRC : receive-queue init kernel launch failed: %s", cudaGetErrorString(cerr));
      status = ncclUnhandledCudaError;
      goto out;
    }
  }
  CUDACHECKGOTO(cudaDeviceSynchronize(), status, out);

  devHandle->netDeviceType = (ncclNetDeviceType)NCCL_NET_DEVICE_GIN_MRC;
  devHandle->netDeviceVersion = NCCL_GIN_MRC_VERSION;
  devHandle->handle = (void*)ctx->ctxBuf.gpu;
  devHandle->size = 0;
  devHandle->needsProxyProgress = 0;

  *outDevHandle = devHandle;
  *outGinCtx = ctx;

  INFO(NCCL_INIT | NCCL_NET, "GIN/MRC : rank %d created %d contexts over %d ranks on %s", rank, ncontexts, nranks,
       ctx->ibDevName);

out:
  free(localInfo);
  free(remoteInfo);
  free(gverbsQps);
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

  if (ctx->ctxStaging && ctx->qps) {
    struct doca_gpu_verbs_qp** gverbsQps =
      (struct doca_gpu_verbs_qp**)calloc(nranks, sizeof(struct doca_gpu_verbs_qp*));
    for (int ctxIdx = 0; ctxIdx < ncontexts && gverbsQps; ctxIdx++) {
      struct ncclGinMrcGPUContext* gpuCtx = &ctx->ctxStaging[ctxIdx];
      if (gpuCtx->qps == nullptr) continue;
      for (int peer = 0; peer < nranks; peer++) gverbsQps[peer] = ctx->qps[ctxIdx * nranks + peer]->qp_gverbs;
      doca_gpu_verbs_unexport_multi_qps_dev(ctx->gdev, gverbsQps, nranks, (struct doca_gpu_dev_verbs_qp*)gpuCtx->qps);
      gpuCtx->qps = nullptr;
    }
    free(gverbsQps);
  }

  if (ctx->qps) {
    for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
      if (ctx->qps[qpIdx]) doca_gpu_verbs_destroy_qp_hl(ctx->qps[qpIdx]);
    }
    free(ctx->qps);
  }
  free(ctx->ctxStaging);
  ctx->ctxBuf.free_();

  if (ctx->counters) {
    ctx->counters->free_();
    delete ctx->counters;
  }
  if (ctx->signals) {
    ctx->signals->free_();
    delete ctx->signals;
  }

  if (ctx->rq) cudaFree(ctx->rq);
  if (ctx->counterSlots) cudaFree(ctx->counterSlots);

  if (ctx->sink.mr) wrap_ibv_dereg_mr(ctx->sink.mr);
  if (ctx->sink.addr) mrcCuMemFree(ctx->sink.addr);

  if (ctx->ah) doca_verbs_ah_attr_destroy(ctx->ah);
  if (ctx->verbsContext) doca_verbs_context_destroy(ctx->verbsContext);
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
                         (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) != 0),
                status, fail);

  rkey = htobe32(mh->mr->rkey);
  NCCLCHECKGOTO(mh->rkeys.allocate(cComm->nranks), status, fail);
  NCCLCHECKGOTO(mrcGinAllGather(cComm, &rkey, mh->rkeys.host, sizeof(uint32_t)), status, fail);
  NCCLCHECKGOTO(mh->rkeys.push(), status, fail);

  NCCLCHECKGOTO(mh->devHandle.allocate(1), status, fail);
  mh->devHandle.host->rkeys = mh->rkeys.gpu;
  mh->devHandle.host->lkey = htobe32(mh->mr->lkey);
  NCCLCHECKGOTO(mh->devHandle.push(), status, fail);

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
  (void)collComm;
  if (mh == nullptr) return ncclSuccess;

  NCCLCHECK(wrap_ibv_dereg_mr(mh->mr));
  NCCLCHECK(mh->rkeys.free_());
  NCCLCHECK(mh->devHandle.free_());
  delete mh;
  return ncclSuccess;
}

// All doorbells are rung by the GPU, so there is nothing for a CPU proxy to advance.
static ncclResult_t mrcGinProgress(void* ginCtx) {
  (void)ginCtx;
  return ncclSuccess;
}

// The DOCA public API exposes no per-QP error query, so errors surface as CQ errors on the
// device side instead.
static ncclResult_t mrcGinQueryLastError(void* ginCtx, bool* hasError) {
  (void)ginCtx;
  *hasError = false;
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
