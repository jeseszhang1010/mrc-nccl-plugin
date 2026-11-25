/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * CPU RQ mode of the MRC GIN plugin (see mrc_cpu_rq.h). Built against the
 * open-source GPUNetIO headers installed by NCCL, never the DOCA SDK ones.
 ************************************************************************/

#include "mrc_cpu_rq.h"

#include <dlfcn.h>
#include <endian.h>
#include <errno.h>
#include <linux/types.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <common/doca_gpunetio_verbs_dev.h>

#define MRC_GPU_PAGE_SIZE (1UL << 16)
// Bounds the work of one poll, as doca_gpu_verbs_rq_progress() does.
#define MRC_RQ_MAX_CQES_PER_POLL 64

uint32_t mrcCpuRqMaxSignalIds(void) { return DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK + UINT32_C(1); }

/* ── Device QPs for the open-source GPUNetIO device code ──────────── */

int mrcCpuRqExportQps(const struct mrcQpDesc* descs, int n, void** gpuQps) {
  struct doca_gpu_dev_verbs_qp* qps = (struct doca_gpu_dev_verbs_qp*)calloc(n, sizeof(*qps));
  cudaError_t err;
  if (qps == nullptr) return cudaErrorMemoryAllocation;
  for (int i = 0; i < n; i++) {
    const struct mrcQpDesc* d = &descs[i];
    struct doca_gpu_dev_verbs_qp* qp = &qps[i];
    qp->sq_num = d->qpn;
    qp->sq_num_shift8 = d->qpn << 8;
    qp->sq_num_shift8_be = htobe32(qp->sq_num_shift8);
    qp->sq_num_shift8_be_1ds = htobe32(qp->sq_num_shift8 | 1);
    qp->sq_num_shift8_be_2ds = htobe32(qp->sq_num_shift8 | 2);
    qp->sq_num_shift8_be_3ds = htobe32(qp->sq_num_shift8 | 3);
    qp->sq_num_shift8_be_4ds = htobe32(qp->sq_num_shift8 | 4);
    qp->sq_num_shift8_be_5ds = htobe32(qp->sq_num_shift8 | 5);
    qp->sq_wqe_num = (uint16_t)d->sq_wqe_num;
    qp->sq_wqe_mask = (uint16_t)(d->sq_wqe_num - 1);
    qp->sq_wqe_daddr = (uint8_t*)d->sq_wqe;
    qp->sq_dbrec = (__be32*)d->sq_dbrec;
    qp->sq_db = d->sq_db;
    qp->cq_sq.cqe_daddr = (uint8_t*)d->sq_cqe;
    qp->cq_sq.cq_num = d->sq_cqn;
    qp->cq_sq.cqe_num = d->sq_cqe_num;
    qp->cq_sq.dbrec = (__be32*)d->sq_cq_dbrec;
    qp->cq_sq.cqe_mask = d->sq_cqe_num - 1;
    qp->cq_sq.cqe_size = d->sq_cqe_size;
    qp->cq_sq.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
    qp->cq_sq.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
    qp->nic_handler = d->sq_bf ? DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF : DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
    qp->mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
  }
  err = cudaMalloc(gpuQps, (size_t)n * sizeof(*qps));
  if (err == cudaSuccess) {
    err = cudaMemcpy(*gpuQps, qps, (size_t)n * sizeof(*qps), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(*gpuQps);
      *gpuQps = nullptr;
    }
  }
  free(qps);
  return err;
}

void mrcCpuRqFreeQps(void* gpuQps) {
  if (gpuQps) cudaFree(gpuQps);
}

/* ── GDRCopy mappings ─────────────────────────────────────────────── */

// The subset of gdrapi.h used here, resolved at run time so that GDRCopy is only needed in CPU RQ mode.
typedef struct gdr* gdr_t;
typedef struct {
  unsigned long h;
} gdr_mh_t;
enum { MRC_GDR_PIN_FLAG_FORCE_PCIE = 1, MRC_GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE = 2 };

static struct {
  pthread_mutex_t lock;
  bool loaded;
  gdr_t g;
  int (*pin_buffer)(gdr_t, unsigned long, size_t, uint64_t, uint32_t, gdr_mh_t*);
  int (*pin_buffer_v2)(gdr_t, unsigned long, size_t, uint32_t, gdr_mh_t*);
  int (*unpin_buffer)(gdr_t, gdr_mh_t);
  int (*map)(gdr_t, gdr_mh_t, void**, size_t);
  int (*unmap)(gdr_t, gdr_mh_t, void*, size_t);
  int (*get_attribute)(gdr_t, int, int*);
} gdrcopy = {PTHREAD_MUTEX_INITIALIZER, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

// Opens GDRCopy once per process and keeps it open, like the open-source GPUNetIO library.
static gdr_t gdrcopyOpen() {
  pthread_mutex_lock(&gdrcopy.lock);
  if (!gdrcopy.loaded) {
    gdrcopy.loaded = true;
    void* h = dlopen("libgdrapi.so.2", RTLD_NOW);
    if (h == nullptr) h = dlopen("libgdrapi.so", RTLD_NOW);
    if (h != nullptr) {
      gdr_t (*open)() = (gdr_t(*)())dlsym(h, "gdr_open");
      gdrcopy.pin_buffer = (decltype(gdrcopy.pin_buffer))dlsym(h, "gdr_pin_buffer");
      gdrcopy.pin_buffer_v2 = (decltype(gdrcopy.pin_buffer_v2))dlsym(h, "gdr_pin_buffer_v2");
      gdrcopy.unpin_buffer = (decltype(gdrcopy.unpin_buffer))dlsym(h, "gdr_unpin_buffer");
      gdrcopy.map = (decltype(gdrcopy.map))dlsym(h, "gdr_map");
      gdrcopy.unmap = (decltype(gdrcopy.unmap))dlsym(h, "gdr_unmap");
      gdrcopy.get_attribute = (decltype(gdrcopy.get_attribute))dlsym(h, "gdr_get_attribute");
      if (open && gdrcopy.pin_buffer && gdrcopy.unpin_buffer && gdrcopy.map && gdrcopy.unmap) gdrcopy.g = open();
    }
  }
  gdr_t g = gdrcopy.g;
  pthread_mutex_unlock(&gdrcopy.lock);
  return g;
}

struct mrcCpuRqMap {
  uintptr_t gpu;
  size_t size;
  gdr_mh_t mh;
  void* cpu;
};

int mrcCpuRqMapCreate(void* gpu, size_t size, bool forcePcie, struct mrcCpuRqMap** out) {
  gdr_t g = gdrcopyOpen();
  if (g == nullptr) return -ENOSYS;
  if (((uintptr_t)gpu & (MRC_GPU_PAGE_SIZE - 1)) != 0) return -EINVAL;
  if (forcePcie) {
    int supported = 0;
    if (gdrcopy.get_attribute == nullptr || gdrcopy.pin_buffer_v2 == nullptr ||
        gdrcopy.get_attribute(g, MRC_GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE, &supported) != 0 || !supported)
      return -ENOTSUP;
  }

  struct mrcCpuRqMap* m = (struct mrcCpuRqMap*)calloc(1, sizeof(*m));
  if (m == nullptr) return -ENOMEM;
  m->gpu = (uintptr_t)gpu;
  m->size = (size + MRC_GPU_PAGE_SIZE - 1) & ~(MRC_GPU_PAGE_SIZE - 1);
  int rc = forcePcie ? gdrcopy.pin_buffer_v2(g, m->gpu, m->size, MRC_GDR_PIN_FLAG_FORCE_PCIE, &m->mh)
                     : gdrcopy.pin_buffer(g, m->gpu, m->size, 0, 0, &m->mh);
  if (rc != 0) {
    free(m);
    return -EIO;
  }
  if (gdrcopy.map(g, m->mh, &m->cpu, m->size) != 0) {
    gdrcopy.unpin_buffer(g, m->mh);
    free(m);
    return -EIO;
  }
  *out = m;
  return 0;
}

volatile uint64_t* mrcCpuRqMapCpu(struct mrcCpuRqMap* m, void* gpu) {
  return (volatile uint64_t*)((uint8_t*)m->cpu + ((uintptr_t)gpu - m->gpu));
}

void mrcCpuRqMapDestroy(struct mrcCpuRqMap* m) {
  if (m == nullptr) return;
  gdrcopy.unmap(gdrcopy.g, m->mh, m->cpu, m->size);
  gdrcopy.unpin_buffer(gdrcopy.g, m->mh);
  free(m);
}

bool mrcCpuRqNeedsMcst(int cudaDev) {
  int ordering = 0;
  if (cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, cudaDev) != cudaSuccess) {
    cudaGetLastError();
    return true;
  }
  return ordering == 0;
}

/* ── Receive queues ───────────────────────────────────────────────── */

static inline void storeFence() {
#if defined(__aarch64__)
  asm volatile("dmb st" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("sfence" ::: "memory");
#else
  __sync_synchronize();
#endif
}

static inline void fullFence() {
#if defined(__aarch64__)
  asm volatile("dmb sy" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("mfence" ::: "memory");
#else
  __sync_synchronize();
#endif
}

void mrcCpuRqInit(struct mrcCpuRq* rq, const struct mrcQpDesc* d) {
  memset(rq, 0, sizeof(*rq));
  rq->cqes = (const volatile uint8_t*)d->rq_cqe;
  rq->cqeNum = d->rq_cqe_num;
  rq->dbrec = (volatile uint32_t*)d->rq_dbrec + DOCA_GPUNETIO_IB_MLX5_RCV_DBR;
  rq->dbrPi = rq->dbrPiShadow = d->rq_wqe_num;
}

static void applySignal(struct mrcCpuRqSignals* s, uint32_t imm) {
  const uint32_t op = (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_OP_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_OP_MASK;
  const uint32_t id = (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK;
  const uint64_t val = (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_MASK;
  // Undeliverable immediates are counted and consumed, so they cannot stall the queue.
  if ((op != DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD && op != DOCA_GPUNETIO_VERBS_SIGNAL_OP_SET) || id >= s->nIds) {
    if (s->dropped++ == 0) s->droppedImm = imm;
    return;
  }
  volatile uint64_t* slot = &s->slots[id];
  if (s->mcst) (void)*(volatile uint8_t*)slot;
  const uint64_t v = op == DOCA_GPUNETIO_VERBS_SIGNAL_OP_SET ? val : s->shadow[id] + val;
  s->shadow[id] = v;
  *slot = v;
}

int mrcCpuRqPoll(struct mrcCpuRq* rq, struct mrcCpuRqSignals* sigs, bool* progressed, struct mrcCpuRqError* err) {
  const uint32_t maxBatch = rq->cqeNum < MRC_RQ_MAX_CQES_PER_POLL ? rq->cqeNum : MRC_RQ_MAX_CQES_PER_POLL;
  uint32_t consumed = 0;
  int status = 0;

  *progressed = false;
  if (rq->fatal) return rq->fatal;

  while (consumed < maxBatch) {
    const uint64_t ci = rq->cqeCi + consumed;
    const struct doca_gpunetio_ib_mlx5_cqe64* cqe =
      (const struct doca_gpunetio_ib_mlx5_cqe64*)(rq->cqes + (ci & (rq->cqeNum - 1)) * sizeof(*cqe));
    // The NIC writes this word last, so acquiring it makes the rest of the CQE safe to read.
    const uint32_t ctrl = be32toh(__atomic_load_n(&cqe->wqe_counter_sig_op_own_raw, __ATOMIC_ACQUIRE));
    const uint8_t opOwn = ctrl & 0xff;
    const uint8_t opcode = opOwn >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
    const bool owner = (opOwn & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) != 0;
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_INVALID || owner != ((ci & rq->cqeNum) != 0)) break;

    if (opcode != DOCA_GPUNETIO_IB_MLX5_CQE_RESP_WR_IMM) {
      memset(err, 0, sizeof(*err));
      err->opcode = opcode;
      if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_RESP_ERR) {
        const struct doca_gpunetio_ib_mlx5_err_cqe_ex* e = (const struct doca_gpunetio_ib_mlx5_err_cqe_ex*)cqe;
        err->syndrome = e->syndrome;
        err->vendorSyndrome = e->vendor_err_synd;
        err->hwSyndrome = e->hw_err_synd;
        err->wqeCounter = be16toh(e->wqe_counter);
        status = -EIO;
      } else {
        status = -EPROTO;
      }
      break;
    }
    applySignal(sigs, le32toh(cqe->imm_inval_pkey));
    consumed++;
  }

  if (consumed != 0) {
    rq->cqeCi += consumed;
    rq->dbrPi += consumed;
    *progressed = true;
  }
  if (status != 0) rq->fatal = status;
  return status;
}

void mrcCpuRqCommitFence(bool strongFence) {
  if (strongFence)
    fullFence();
  else
    storeFence();
}

void mrcCpuRqCommitDbr(struct mrcCpuRq* rq) {
  if (rq->dbrPi == rq->dbrPiShadow) return;
  *rq->dbrec = htobe32((uint32_t)(rq->dbrPi & UINT16_MAX));
  rq->dbrPiShadow = rq->dbrPi;
}
