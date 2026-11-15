/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RCX queue-pair builder for the MRC GIN plugin (see mrc_qp.h). Mirrors the
 * open-source doca_gpu_verbs_create_qp_hl() resource layout, so the QP is
 * torn down by doca_gpu_verbs_destroy_qp_hl(), but sets the QP type, and can
 * keep the receive CQ in GPU memory.
 ************************************************************************/

#include "mrc_qp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#define MRC_QP_ERR(fmt, ...) fprintf(stderr, "[GIN/MRC QP] " fmt "\n", ##__VA_ARGS__)

// Same as the open-source high-level layer (doca_gpunetio_high_level.cpp).
static constexpr uint32_t kDbrSize = 8;
static constexpr uint32_t kWqeBbSize = 64; // send WQE and WQE basic block
static constexpr uint32_t kRcvWqeSize = 16; // one receive data segment
static constexpr uint32_t kMaxSegs = 1;
static constexpr int kIbvAccessLocalWrite = 1; // IBV_ACCESS_LOCAL_WRITE

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

// The high-level QP plus what doca_gpu_verbs_destroy_qp_hl() does not know about. hl comes first, so
// that the library's free() of the high-level QP releases the whole allocation.
struct mrcQpHl {
  struct doca_gpu_verbs_qp_hl hl;
  void* rq_cq_gpu_ptr; // GPU-memory receive CQ (rq_gpu), whose CQ object hl owns
};

static uint32_t cqUmemSize(uint32_t ncqes, uint32_t dbrSize) {
  return alignUp(ncqes * (uint32_t)sizeof(struct doca_gpunetio_ib_mlx5_cqe64) + dbrSize, (uint32_t)pageSize());
}

static void initCqes(struct doca_gpunetio_ib_mlx5_cqe64* cqes, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    cqes[i].op_own = (DOCA_GPUNETIO_IB_MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
                     DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK;
}

// Register GPU memory with the NIC, preferring DMA-BUF over nvidia-peermem.
static doca_error_t gpuUmem(doca_gpu_t* gpu, doca_dev_t* dev, void* ptr, size_t size, doca_verbs_umem_t** umem) {
  int fd = (int)DOCA_VERBS_DMABUF_INVALID_FD;
  if (doca_gpu_get_dmabuf_fd(gpu, ptr, size, &fd) != DOCA_SUCCESS) fd = (int)DOCA_VERBS_DMABUF_INVALID_FD;
  doca_error_t status = doca_verbs_umem_create(dev, gpu, ptr, size, kIbvAccessLocalWrite, fd, 0, umem);
  if (status != DOCA_SUCCESS && fd != (int)DOCA_VERBS_DMABUF_INVALID_FD)
    status = doca_verbs_umem_create(dev, gpu, ptr, size, kIbvAccessLocalWrite, DOCA_VERBS_DMABUF_INVALID_FD, 0, umem);
  if (fd != (int)DOCA_VERBS_DMABUF_INVALID_FD) close(fd);
  return status;
}

// CQ with CQEs and doorbell record in GPU memory, polled by device code.
static doca_error_t createGpuCq(doca_gpu_t* gpu, doca_dev_t* dev, uint32_t ncqes, void** gpuPtr,
                                doca_verbs_umem_t** umem, doca_verbs_cq_t** cq) {
  const uint32_t size = cqUmemSize(ncqes, kDbrSize);
  doca_verbs_cq_attr_t* attr = nullptr;
  struct doca_gpunetio_ib_mlx5_cqe64* ring = nullptr;
  doca_error_t status = doca_gpu_mem_alloc(gpu, size, pageSize(), DOCA_GPU_MEM_TYPE_GPU, gpuPtr, nullptr);
  if (status != DOCA_SUCCESS) return status;

  ring = (struct doca_gpunetio_ib_mlx5_cqe64*)calloc(1, size);
  if (ring == nullptr) return DOCA_ERROR_NO_MEMORY;
  initCqes(ring, ncqes);
  if (cudaMemcpy(*gpuPtr, ring, size, cudaMemcpyDefault) != cudaSuccess) status = DOCA_ERROR_DRIVER;
  free(ring);
  if (status != DOCA_SUCCESS) return status;

  status = gpuUmem(gpu, dev, *gpuPtr, size, umem);
  if (status != DOCA_SUCCESS) return status;

  status = doca_verbs_cq_attr_create(&attr);
  if (status != DOCA_SUCCESS) return status;
  if ((status = doca_verbs_cq_attr_set_external_umem(attr, *umem, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_cq_size(attr, ncqes)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_cq_overrun(attr, DOCA_VERBS_CQ_ENABLE_OVERRUN)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_create(dev, attr, cq)) != DOCA_SUCCESS) {
    doca_verbs_cq_attr_destroy(attr);
    return status;
  }
  return doca_verbs_cq_attr_destroy(attr);
}

// Receive CQ: CQEs and doorbell record in host memory, polled by doca_gpu_verbs_rq_progress().
static doca_error_t createRqCq(struct doca_gpu_verbs_qp_hl* qp, doca_dev_t* dev, uint32_t ncqes) {
  const uint32_t size = cqUmemSize(ncqes, kDbrSize);
  doca_verbs_cq_attr_t* attr = nullptr;
  void* buf = nullptr;
  uint32_t n = 0;
  uint8_t esz = 0;
  doca_error_t status;

  if (posix_memalign(&qp->cq_rq_umem_cpu_ptr, pageSize(), size) != 0) {
    qp->cq_rq_umem_cpu_ptr = nullptr;
    return DOCA_ERROR_NO_MEMORY;
  }
  memset(qp->cq_rq_umem_cpu_ptr, 0, size);
  initCqes((struct doca_gpunetio_ib_mlx5_cqe64*)qp->cq_rq_umem_cpu_ptr, ncqes);

  status = doca_verbs_umem_create(dev, qp->gpu_dev, qp->cq_rq_umem_cpu_ptr, size, kIbvAccessLocalWrite,
                                  DOCA_VERBS_DMABUF_INVALID_FD, 0, &qp->cq_rq_umem);
  if (status != DOCA_SUCCESS) return status;

  status = doca_verbs_cq_attr_create(&attr);
  if (status != DOCA_SUCCESS) return status;
  if ((status = doca_verbs_cq_attr_set_external_umem(attr, qp->cq_rq_umem, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_cq_size(attr, ncqes)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_attr_set_cq_overrun(attr, DOCA_VERBS_CQ_ENABLE_OVERRUN)) != DOCA_SUCCESS ||
      (status = doca_verbs_cq_create(dev, attr, &qp->cq_rq)) != DOCA_SUCCESS) {
    doca_verbs_cq_attr_destroy(attr);
    return status;
  }
  doca_verbs_cq_attr_destroy(attr);

  // rq_progress walks the ring through this address, so it must be the buffer just registered.
  status = doca_verbs_cq_get_wq(qp->cq_rq, &buf, &n, &esz);
  if (status != DOCA_SUCCESS) return status;
  if (buf != qp->cq_rq_umem_cpu_ptr || n != ncqes || esz != sizeof(struct doca_gpunetio_ib_mlx5_cqe64)) {
    MRC_QP_ERR("receive CQ has unexpected ring properties (buf %p/%p, %u/%u CQEs, %u B)", buf,
               qp->cq_rq_umem_cpu_ptr, n, ncqes, esz);
    return DOCA_ERROR_INVALID_VALUE;
  }
  return DOCA_SUCCESS;
}

static doca_error_t createUar(struct doca_gpu_verbs_qp_hl* qp, doca_dev_t* dev,
                              enum doca_gpu_dev_verbs_nic_handler nicHandler) {
  if (nicHandler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
    if (doca_verbs_uar_create(dev, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &qp->external_uar) ==
        DOCA_SUCCESS)
      return DOCA_SUCCESS;
    return doca_verbs_uar_create(dev, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE, &qp->external_uar);
  }
  return doca_verbs_uar_create(dev, DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME, &qp->external_uar);
}

// Work queues in GPU memory. With a CPU-managed receive queue the doorbell record is GDRCopy-mapped
// GPU memory: the NIC and the GPU use its GPU address and the CPU returns receive credits through
// the mapping.
static doca_error_t createQp(struct doca_gpu_verbs_qp_hl* qp, const struct mrcQpAttr* a, uint32_t sqN, uint32_t rqN,
                             void** dbrCpuPtr) {
  const uint32_t wqSize = alignUp(sqN * kWqeBbSize + alignUp(rqN * kRcvWqeSize, kWqeBbSize), (uint32_t)pageSize());
  const uint32_t dbrSize = alignUp(kDbrSize, (uint32_t)pageSize());
  doca_verbs_qp_init_attr_t* attr = nullptr;
  doca_error_t status;

  *dbrCpuPtr = nullptr;
  status = doca_gpu_mem_alloc(qp->gpu_dev, wqSize, pageSize(), DOCA_GPU_MEM_TYPE_GPU, &qp->qp_umem_gpu_ptr, nullptr);
  if (status != DOCA_SUCCESS) return status;
  status = gpuUmem(qp->gpu_dev, a->net_dev, qp->qp_umem_gpu_ptr, wqSize, &qp->qp_umem);
  if (status != DOCA_SUCCESS) return status;

  const bool cpuRq = rqN > 0 && !a->rq_gpu;
  status = doca_gpu_mem_alloc(qp->gpu_dev, dbrSize, pageSize(), cpuRq ? DOCA_GPU_MEM_TYPE_GPU_CPU : DOCA_GPU_MEM_TYPE_GPU,
                              &qp->qp_umem_dbr_gpu_ptr, cpuRq ? dbrCpuPtr : nullptr);
  if (status != DOCA_SUCCESS) return status;
  status = gpuUmem(qp->gpu_dev, a->net_dev, qp->qp_umem_dbr_gpu_ptr, dbrSize, &qp->qp_umem_dbr);
  if (status != DOCA_SUCCESS) return status;

  status = doca_verbs_qp_init_attr_create(&attr);
  if (status != DOCA_SUCCESS) return status;
  if ((status = doca_verbs_qp_init_attr_set_external_uar(attr, qp->external_uar)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_external_umem(attr, qp->qp_umem, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_external_umem_dbr(attr, qp->qp_umem_dbr, 0)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_pd(attr, a->net_dev)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_sq_wr(attr, sqN)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_rq_wr(attr, rqN)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_qp_type(attr, a->qp_type)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_send_cq(attr, qp->cq_sq)) != DOCA_SUCCESS ||
      (rqN > 0 && (status = doca_verbs_qp_init_attr_set_receive_cq(attr, qp->cq_rq)) != DOCA_SUCCESS) ||
      (status = doca_verbs_qp_init_attr_set_send_max_sges(attr, kMaxSegs)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_init_attr_set_receive_max_sges(attr, kMaxSegs)) != DOCA_SUCCESS ||
      (status = doca_verbs_qp_create(a->net_dev, attr, &qp->qp)) != DOCA_SUCCESS) {
    doca_verbs_qp_init_attr_destroy(attr);
    return status;
  }
  return doca_verbs_qp_init_attr_destroy(attr);
}

doca_error_t mrcQpCreate(const struct mrcQpAttr* a, struct doca_gpu_verbs_qp_hl** out) {
  if (a == nullptr || out == nullptr || a->gpu_dev == nullptr || a->net_dev == nullptr || a->sq_nwqe == 0)
    return DOCA_ERROR_INVALID_VALUE;
  if (a->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
      a->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
    MRC_QP_ERR("nic_handler %d must be GPU_SM_DB or GPU_SM_BF", (int)a->nic_handler);
    return DOCA_ERROR_INVALID_VALUE;
  }

  const uint32_t sqN = nextPow2(a->sq_nwqe);
  const uint32_t rqN = a->rq_nwqe ? nextPow2(a->rq_nwqe < kWqeBbSize / kRcvWqeSize ? kWqeBbSize / kRcvWqeSize
                                                                                     : a->rq_nwqe)
                                  : 0;
  void* dbrCpuPtr = nullptr;
  const char* step = "send CQ";
  doca_error_t status;

  struct mrcQpHl* w = (struct mrcQpHl*)calloc(1, sizeof(*w));
  if (w == nullptr) return DOCA_ERROR_NO_MEMORY;
  struct doca_gpu_verbs_qp_hl* qp = &w->hl;
  qp->gpu_dev = a->gpu_dev;
  qp->nic_handler = a->nic_handler;
  qp->send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;

  if ((status = createGpuCq(qp->gpu_dev, a->net_dev, sqN, &qp->cq_sq_umem_gpu_ptr, &qp->cq_sq_umem, &qp->cq_sq)) !=
      DOCA_SUCCESS)
    goto fail;
  step = "receive CQ";
  if (rqN > 0) {
    status = a->rq_gpu ? createGpuCq(qp->gpu_dev, a->net_dev, rqN, &w->rq_cq_gpu_ptr, &qp->cq_rq_umem, &qp->cq_rq)
                       : createRqCq(qp, a->net_dev, rqN);
    if (status != DOCA_SUCCESS) goto fail;
  }
  step = "UAR";
  if ((status = createUar(qp, a->net_dev, a->nic_handler)) != DOCA_SUCCESS) goto fail;
  step = "QP";
  if ((status = createQp(qp, a, sqN, rqN, &dbrCpuPtr)) != DOCA_SUCCESS) goto fail;
  step = "QP export";
  // A GPU-managed receive queue skips the library's CPU receive setup, which writes the receive CQ
  // from the CPU, and only attaches its CQ for doca_gpu_verbs_export_sdk_qp().
  if ((status = doca_gpu_verbs_export_qp(qp->gpu_dev, qp->qp, qp->nic_handler, qp->qp_umem_gpu_ptr, qp->cq_sq,
                                         a->rq_gpu ? nullptr : qp->cq_rq, dbrCpuPtr, qp->send_dbr_mode_ext,
                                         DOCA_GPUNETIO_VERBS_CQ_64B, a->data_direct, &qp->qp_gverbs)) != DOCA_SUCCESS)
    goto fail;
  if (a->rq_gpu) qp->qp_gverbs->cq_rq = qp->cq_rq;

  *out = qp;
  return DOCA_SUCCESS;

fail:
  MRC_QP_ERR("failed to create the %s (qp type 0x%x, sq %u, rq %u): DOCA error %d", step, a->qp_type, sqN, rqN,
             (int)status);
  mrcQpDestroy(qp);
  return status;
}

doca_error_t mrcQpDestroy(struct doca_gpu_verbs_qp_hl* qp) {
  if (qp == nullptr) return DOCA_SUCCESS;
  struct mrcQpHl* w = (struct mrcQpHl*)qp;
  doca_gpu_t* gpu = qp->gpu_dev;
  void* rqCq = w->rq_cq_gpu_ptr;
  // Releases the DOCA SDK export first, then the queues; the GPU receive CQ memory goes last.
  doca_error_t status = doca_gpu_verbs_destroy_qp_hl(qp);
  if (status != DOCA_SUCCESS) return status;
  if (rqCq != nullptr) doca_gpu_mem_free(gpu, rqCq);
  return DOCA_SUCCESS;
}
