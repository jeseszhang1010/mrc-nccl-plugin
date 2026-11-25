/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RCX queue-pair builder for the MRC GIN plugin on the DOCA SDK (see mrc_qp.h).
 * Same resource layout as the DOCA GPUNetIO verbs samples (verbs_high_level.c):
 * external UAR, work queues, doorbell record and CQs in caller-owned memory.
 ************************************************************************/

#include "mrc_qp.h"

#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include <doca_dev.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_mp.h>

#define MRC_QP_ERR(fmt, ...) fprintf(stderr, "[GIN/MRC QP] " fmt "\n", ##__VA_ARGS__)

static constexpr uint32_t kDbrSize = 8;
static constexpr uint32_t kRcvWqeSize = sizeof(struct mlx5_wqe_data_seg);
static constexpr uint32_t kSndWqeSize = sizeof(struct doca_gpu_dev_verbs_wqe);
static constexpr uint32_t kMaxSegs = 1;
static constexpr uint32_t kUmemAccess = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC;

struct mrcQp {
  struct doca_gpu* gpu;
  enum doca_gpu_dev_verbs_nic_handler nic;
  struct doca_uar* uar;

  struct doca_verbs_cq* cq_sq;
  void* cq_sq_mem; // GPU
  struct doca_umem* cq_sq_umem;

  struct doca_verbs_cq* cq_rq;
  void* cq_rq_mem; // GPU
  struct doca_umem* cq_rq_umem;

  struct doca_verbs_qp* qp;
  void* wq_mem; // GPU: receive ring, then send ring
  struct doca_umem* wq_umem;
  void* dbr_mem; // GPU
  struct doca_umem* dbr_umem;
  uint32_t rq_wqe_num;

  struct doca_gpu_verbs_qp* gverbs;
  struct doca_gpu_dev_verbs_qp* dev; // GPU
};

static size_t pageSize() {
  long sz = sysconf(_SC_PAGESIZE);
  return sz > 0 ? (size_t)sz : 4096;
}

static uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

static uint32_t nextPow2(uint32_t v) {
  uint32_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

static uint32_t cqUmemSize(uint32_t ncqes) {
  return alignUp(ncqes * (uint32_t)sizeof(struct mlx5_cqe64) + kDbrSize, (uint32_t)pageSize());
}

static void initCqes(struct mlx5_cqe64* cqes, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    cqes[i].op_own = (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
}

static doca_error_t createUar(struct mrcQp* q, struct doca_dev* dev) {
  if (q->nic != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
    if (doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &q->uar) == DOCA_SUCCESS)
      return DOCA_SUCCESS;
    if (doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &q->uar) == DOCA_SUCCESS) return DOCA_SUCCESS;
  }
  return doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_BLUEFLAME, &q->uar);
}

// CQ over caller GPU memory, polled by device code.
static doca_error_t createCq(struct mrcQp* q, const struct mrcQpAttr* a, uint32_t ncqes, void** mem,
                             struct doca_umem** umem, struct doca_verbs_cq** cq) {
  const uint32_t size = cqUmemSize(ncqes);
  struct doca_verbs_cq_attr* attr = nullptr;
  doca_error_t status;

  status = doca_gpu_mem_alloc(q->gpu, size, pageSize(), DOCA_GPU_MEM_TYPE_GPU, mem, nullptr);
  if (status != DOCA_SUCCESS) return status;
  struct mlx5_cqe64* ring = (struct mlx5_cqe64*)calloc(1, size);
  if (ring == nullptr) return DOCA_ERROR_NO_MEMORY;
  initCqes(ring, ncqes);
  if (cudaMemcpy(*mem, ring, size, cudaMemcpyHostToDevice) != cudaSuccess) status = DOCA_ERROR_DRIVER;
  free(ring);
  if (status != DOCA_SUCCESS) return status;
  status = doca_umem_gpu_create(q->gpu, a->dev, *mem, size, kUmemAccess, umem);
  if (status != DOCA_SUCCESS) return status;

  if ((status = doca_verbs_cq_attr_create(&attr)) != DOCA_SUCCESS) return status;
  if ((status = doca_verbs_cq_attr_set_external_datapath_en(attr, 1)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_external_umem(attr, *umem, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_cq_size(attr, ncqes)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_cq_overrun(attr, 1)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_external_uar(attr, q->uar)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_create(a->verbs_ctx, attr, cq)) != DOCA_SUCCESS) {
    doca_verbs_cq_attr_destroy(attr);
    return status;
  }
  return doca_verbs_cq_attr_destroy(attr);
}

// Work queues and doorbell record in GPU memory.
static doca_error_t createQp(struct mrcQp* q, const struct mrcQpAttr* a, uint32_t sqN, uint32_t rqN) {
  const uint32_t wqSize = alignUp(alignUp(rqN * kRcvWqeSize, kSndWqeSize) + sqN * kSndWqeSize, (uint32_t)pageSize());
  const uint32_t dbrSize = alignUp(kDbrSize, (uint32_t)pageSize());
  struct doca_verbs_qp_init_attr* attr = nullptr;
  doca_error_t status;

  status = doca_gpu_mem_alloc(q->gpu, wqSize, pageSize(), DOCA_GPU_MEM_TYPE_GPU, &q->wq_mem, nullptr);
  if (status != DOCA_SUCCESS) return status;
  if (cudaMemset(q->wq_mem, 0, wqSize) != cudaSuccess) return DOCA_ERROR_DRIVER;
  status = doca_umem_gpu_create(q->gpu, a->dev, q->wq_mem, wqSize, kUmemAccess, &q->wq_umem);
  if (status != DOCA_SUCCESS) return status;

  status = doca_gpu_mem_alloc(q->gpu, dbrSize, pageSize(), DOCA_GPU_MEM_TYPE_GPU, &q->dbr_mem, nullptr);
  if (status != DOCA_SUCCESS) return status;
  if (cudaMemset(q->dbr_mem, 0, dbrSize) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    return DOCA_ERROR_DRIVER;
  status = doca_umem_gpu_create(q->gpu, a->dev, q->dbr_mem, dbrSize, kUmemAccess, &q->dbr_umem);
  if (status != DOCA_SUCCESS) return status;

  if ((status = doca_verbs_qp_init_attr_create(&attr)) != DOCA_SUCCESS) return status;
  if ((status = doca_verbs_qp_init_attr_set_external_datapath_en(attr, 1)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_external_uar(attr, q->uar)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_external_umem(attr, q->wq_umem, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_external_dbr_umem(attr, q->dbr_umem, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_pd(attr, a->verbs_pd)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_sq_wr(attr, sqN)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_rq_wr(attr, rqN)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_qp_type(attr, a->rcx ? DOCA_VERBS_QP_TYPE_RCX : DOCA_VERBS_QP_TYPE_RC)) !=
        DOCA_SUCCESS ||
      (a->rcx && (status = doca_verbs_qp_init_attr_set_rcx_type(attr, 0)) != DOCA_SUCCESS) ||
      (status = doca_verbs_qp_init_attr_set_send_cq(attr, q->cq_sq)) != DOCA_SUCCESS ||
      (rqN > 0 && (status = doca_verbs_qp_init_attr_set_receive_cq(attr, q->cq_rq)) != DOCA_SUCCESS) ||
      (status = doca_verbs_qp_init_attr_set_send_max_sges(attr, kMaxSegs)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_receive_max_sges(attr, kMaxSegs)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_create(a->verbs_ctx, attr, &q->qp)) != DOCA_SUCCESS) {
    doca_verbs_qp_init_attr_destroy(attr);
    return status;
  }
  return doca_verbs_qp_init_attr_destroy(attr);
}

int mrcQpCreate(const struct mrcQpAttr* a, struct mrcQp** out) {
  if (a == nullptr || out == nullptr || a->gpu_dev == nullptr || a->dev == nullptr || a->verbs_ctx == nullptr ||
      a->verbs_pd == nullptr || a->sq_nwqe == 0)
    return DOCA_ERROR_INVALID_VALUE;
  if (a->nic_handler != MRC_QP_NIC_HANDLER_GPU_SM_DB && a->nic_handler != MRC_QP_NIC_HANDLER_GPU_SM_BF) {
    MRC_QP_ERR("nic_handler %d must be GPU_SM_DB or GPU_SM_BF", (int)a->nic_handler);
    return DOCA_ERROR_INVALID_VALUE;
  }

  const uint32_t sqN = nextPow2(a->sq_nwqe);
  // At least one send WQE basic block of receive WQEs, so the send ring stays aligned.
  const uint32_t minRq = kSndWqeSize / kRcvWqeSize;
  const uint32_t rqN = a->rq_nwqe ? nextPow2(a->rq_nwqe < minRq ? minRq : a->rq_nwqe) : 0;
  const char* step = "UAR";
  doca_error_t status;

  struct mrcQp* q = (struct mrcQp*)calloc(1, sizeof(*q));
  if (q == nullptr) return DOCA_ERROR_NO_MEMORY;
  q->gpu = a->gpu_dev;
  q->nic = (enum doca_gpu_dev_verbs_nic_handler)a->nic_handler;
  q->rq_wqe_num = rqN;

  if ((status = createUar(q, a->dev)) != DOCA_SUCCESS) goto fail;
  step = "send CQ";
  if ((status = createCq(q, a, sqN, &q->cq_sq_mem, &q->cq_sq_umem, &q->cq_sq)) != DOCA_SUCCESS) goto fail;
  step = "receive CQ";
  if (rqN > 0 && (status = createCq(q, a, rqN, &q->cq_rq_mem, &q->cq_rq_umem, &q->cq_rq)) != DOCA_SUCCESS)
    goto fail;
  step = "QP";
  if ((status = createQp(q, a, sqN, rqN)) != DOCA_SUCCESS) goto fail;
  step = "QP export";
  // The DOCA SDK handler gets the receive CQ too: device code posts and drains the receive queue.
  if ((status = doca_gpu_verbs_export_qp(q->gpu, a->dev, q->qp, q->nic, q->wq_mem, q->cq_sq, q->cq_rq,
                                         &q->gverbs)) != DOCA_SUCCESS)
    goto fail;
  if ((status = doca_gpu_verbs_get_qp_dev(q->gverbs, &q->dev)) != DOCA_SUCCESS) goto fail;

  *out = q;
  return DOCA_SUCCESS;

fail:
  MRC_QP_ERR("failed to create the %s (%s QP, sq %u, rq %u): %s", step, a->rcx ? "RCX" : "RC", sqN, rqN,
             doca_error_get_descr(status));
  mrcQpDestroy(q);
  return status;
}

void mrcQpDestroy(struct mrcQp* q) {
  if (q == nullptr) return;
  if (q->gverbs) doca_gpu_verbs_unexport_qp(q->gpu, q->gverbs);
  if (q->qp) doca_verbs_qp_destroy(q->qp);
  if (q->wq_umem) doca_umem_destroy(q->wq_umem);
  if (q->wq_mem) doca_gpu_mem_free(q->gpu, q->wq_mem);
  if (q->dbr_umem) doca_umem_destroy(q->dbr_umem);
  if (q->dbr_mem) doca_gpu_mem_free(q->gpu, q->dbr_mem);
  if (q->cq_rq) doca_verbs_cq_destroy(q->cq_rq);
  if (q->cq_rq_umem) doca_umem_destroy(q->cq_rq_umem);
  if (q->cq_rq_mem) doca_gpu_mem_free(q->gpu, q->cq_rq_mem);
  if (q->cq_sq) doca_verbs_cq_destroy(q->cq_sq);
  if (q->cq_sq_umem) doca_umem_destroy(q->cq_sq_umem);
  if (q->cq_sq_mem) doca_gpu_mem_free(q->gpu, q->cq_sq_mem);
  if (q->uar) doca_uar_destroy(q->uar);
  free(q);
}

struct doca_verbs_qp* mrcQpVerbs(struct mrcQp* q) { return q->qp; }

uint32_t mrcQpNum(struct mrcQp* q) { return doca_verbs_qp_get_qpn(q->qp); }

void* mrcQpDevice(struct mrcQp* q) { return q->dev; }
