/*************************************************************************
 * High-level DOCA QP helpers for the MRC GIN plugin.
 *
 * Implements the helpers declared in mrc_doca_ext.h on top of the DOCA 3.2
 * public API (doca_verbs / doca_gpunetio / bridge), which is the only surface
 * the prebuilt DOCA package ships.
 *
 * The create path is ported from the DOCA sample verbs_high_level.c (which
 * uses only the public API and supports RCX), adapted to an init-attr layout
 * that takes an ibv_pd; the DOCA doca_dev / doca_verbs_context / doca_verbs_pd
 * are derived from it via the DOCA bridge.
 ************************************************************************/

#include "mrc_doca_ext.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <doca_types.h>
#include <doca_dev.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_rdma_bridge.h>
#include <doca_gpunetio_verbs_def.h>

#define EXT_ERR(fmt, ...) fprintf(stderr, "[GDAKI-EXT] " fmt "\n", ##__VA_ARGS__)

#define VERBS_TEST_DBR_SIZE (8)
#define VERBS_TEST_MAX_SEND_SEGS (1)
#define VERBS_TEST_MAX_RECEIVE_SEGS (1)

#define EXT_ROUND_UP(sz, align) ((((sz) + (align) - 1) / (align)) * (align))

static size_t ext_page_size(void)
{
	long sz = sysconf(_SC_PAGESIZE);
	return (sz > 0) ? (size_t)sz : 4096;
}

static uint32_t ext_align_up_u32(uint32_t value, uint32_t alignment)
{
	if (alignment == 0)
		return value;
	return ((value + alignment - 1) / alignment) * alignment;
}

static uint32_t ext_next_pow2(uint32_t x)
{
	if (x < 2)
		return 1;
	return 1u << (32 - __builtin_clz(x - 1));
}

static uint32_t calc_cq_external_umem_size(uint32_t queue_size)
{
	uint32_t cqe_buf_size = 0;

	if (queue_size != 0)
		cqe_buf_size = (uint32_t)(queue_size * sizeof(struct mlx5_cqe64));

	return ext_align_up_u32(cqe_buf_size + VERBS_TEST_DBR_SIZE, (uint32_t)ext_page_size());
}

static uint32_t calc_qp_external_umem_size(uint32_t rq_nwqes, uint32_t sq_nwqes)
{
	uint32_t rq_ring_size = 0;
	uint32_t sq_ring_size = 0;

	if (rq_nwqes != 0)
		rq_ring_size = (uint32_t)(rq_nwqes * sizeof(struct mlx5_wqe_data_seg));
	if (sq_nwqes != 0)
		sq_ring_size = (uint32_t)(sq_nwqes * sizeof(struct doca_gpu_dev_verbs_wqe));

	return ext_align_up_u32(rq_ring_size + sq_ring_size, (uint32_t)ext_page_size());
}

static void mlx5_init_cqes(struct mlx5_cqe64 *cqes, uint32_t nb_cqes)
{
	for (uint32_t i = 0; i < nb_cqes; i++)
		cqes[i].op_own = (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
				 MLX5_CQE_OWNER_MASK;
}

/* Derive the DOCA doca_dev / doca_verbs_context / doca_verbs_pd from an ibv_pd. */
static doca_error_t import_from_pd(struct ibv_pd *ibpd,
				   struct doca_dev **dev,
				   struct doca_verbs_context **verbs_context,
				   struct doca_verbs_pd **verbs_pd)
{
	doca_error_t status;

	status = doca_rdma_bridge_open_dev_from_pd(ibpd, dev);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to open doca_dev from ibv_pd: %s", doca_error_get_descr(status));
		return status;
	}

	status = doca_verbs_bridge_verbs_context_import(ibpd->context,
							DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
							verbs_context);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to import doca_verbs_context: %s", doca_error_get_descr(status));
		doca_dev_close(*dev);
		*dev = NULL;
		return status;
	}

	status = doca_verbs_bridge_verbs_pd_import(ibpd, verbs_pd);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to import doca_verbs_pd: %s", doca_error_get_descr(status));
		doca_verbs_context_destroy(*verbs_context);
		*verbs_context = NULL;
		doca_dev_close(*dev);
		*dev = NULL;
		return status;
	}

	return DOCA_SUCCESS;
}

static doca_error_t create_uar(struct doca_dev *dev,
			       enum doca_gpu_dev_verbs_nic_handler nic_handler,
			       struct doca_uar **external_uar)
{
	doca_error_t status;

	if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
		status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, external_uar);
		if (status == DOCA_SUCCESS)
			return DOCA_SUCCESS;

		status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, external_uar);
		if (status == DOCA_SUCCESS)
			return DOCA_SUCCESS;
	}

	status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_BLUEFLAME, external_uar);
	if (status != DOCA_SUCCESS)
		EXT_ERR("Failed to create doca_uar: %s", doca_error_get_descr(status));

	return status;
}

static doca_error_t create_cq(struct doca_gpu *gpu_dev,
			      struct doca_dev *dev,
			      struct doca_verbs_context *verbs_ctx,
			      uint32_t ncqes,
			      void **gpu_umem_dev_ptr,
			      struct doca_umem **gpu_umem,
			      struct doca_uar *external_uar,
			      struct doca_verbs_cq **verbs_cq)
{
	doca_error_t status, tmp_status;
	struct doca_verbs_cq_attr *verbs_cq_attr = NULL;
	struct doca_verbs_cq *new_cq = NULL;
	struct mlx5_cqe64 *cq_ring_haddr = NULL;
	uint32_t external_umem_size = calc_cq_external_umem_size(ncqes);

	*gpu_umem = NULL;
	*gpu_umem_dev_ptr = NULL;

	status = doca_verbs_cq_attr_create(&verbs_cq_attr);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to create doca verbs cq attributes");
		return status;
	}

	status = doca_verbs_cq_attr_set_external_datapath_en(verbs_cq_attr, 1);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, ext_page_size(),
				    DOCA_GPU_MEM_TYPE_GPU, gpu_umem_dev_ptr, NULL);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to alloc gpu memory for external umem cq");
		goto destroy_resources;
	}

	cq_ring_haddr = (struct mlx5_cqe64 *)calloc(external_umem_size, sizeof(uint8_t));
	if (cq_ring_haddr == NULL) {
		status = DOCA_ERROR_NO_MEMORY;
		goto destroy_resources;
	}

	mlx5_init_cqes(cq_ring_haddr, ncqes);

	if (cudaMemcpy(*gpu_umem_dev_ptr, cq_ring_haddr, external_umem_size, cudaMemcpyDefault) !=
	    cudaSuccess) {
		EXT_ERR("Failed to cudaMemcpy gpu cq ring buffer");
		status = DOCA_ERROR_DRIVER;
		goto destroy_resources;
	}

	free(cq_ring_haddr);
	cq_ring_haddr = NULL;

	status = doca_umem_gpu_create(gpu_dev, dev, *gpu_umem_dev_ptr, external_umem_size,
				      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
					      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
				      gpu_umem);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_cq_attr_set_external_umem(verbs_cq_attr, *gpu_umem, 0);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_cq_attr_set_cq_size(verbs_cq_attr, ncqes);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_cq_attr_set_cq_overrun(verbs_cq_attr, 1);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	if (external_uar != NULL) {
		status = doca_verbs_cq_attr_set_external_uar(verbs_cq_attr, external_uar);
		if (status != DOCA_SUCCESS)
			goto destroy_resources;
	}

	status = doca_verbs_cq_create(verbs_ctx, verbs_cq_attr, &new_cq);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to create doca verbs cq");
		goto destroy_resources;
	}

	doca_verbs_cq_attr_destroy(verbs_cq_attr);
	*verbs_cq = new_cq;
	return DOCA_SUCCESS;

destroy_resources:
	if (new_cq != NULL)
		doca_verbs_cq_destroy(new_cq);
	if (verbs_cq_attr != NULL)
		doca_verbs_cq_attr_destroy(verbs_cq_attr);
	if (*gpu_umem != NULL) {
		doca_umem_destroy(*gpu_umem);
		*gpu_umem = NULL;
	}
	if (cq_ring_haddr != NULL)
		free(cq_ring_haddr);
	if (*gpu_umem_dev_ptr != NULL) {
		tmp_status = doca_gpu_mem_free(gpu_dev, *gpu_umem_dev_ptr);
		(void)tmp_status;
		*gpu_umem_dev_ptr = NULL;
	}
	return status;
}

static doca_error_t set_qp_transport_type(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
					  bool use_rcx,
					  uint8_t rcx_type)
{
	if (!use_rcx)
		return doca_verbs_qp_init_attr_set_qp_type(verbs_qp_init_attr, DOCA_VERBS_QP_TYPE_RC);

#ifdef GPUNETIO_VERBS_RCX_SUPPORTED
	doca_error_t status =
		doca_verbs_qp_init_attr_set_qp_type(verbs_qp_init_attr, DOCA_VERBS_QP_TYPE_RCX);
	if (status != DOCA_SUCCESS)
		return status;

	return doca_verbs_qp_init_attr_set_rcx_type(verbs_qp_init_attr, rcx_type);
#else
	(void)rcx_type;
	EXT_ERR("RCX QP creation requires the DOCA Verbs multi-path API");
	return DOCA_ERROR_NOT_SUPPORTED;
#endif
}

static doca_error_t create_qp(struct doca_gpu *gpu_dev,
			      struct doca_dev *dev,
			      struct doca_verbs_context *verbs_ctx,
			      struct doca_verbs_pd *verbs_pd,
			      struct doca_verbs_cq *cq_sq,
			      uint32_t sq_nwqe,
			      struct doca_verbs_cq *cq_rq,
			      uint32_t rq_nwqe,
			      void **gpu_umem_dev_ptr,
			      struct doca_umem **gpu_umem,
			      void **gpu_umem_dbr_dev_ptr,
			      struct doca_umem **gpu_umem_dbr,
			      struct doca_uar *external_uar,
			      enum doca_gpu_dev_verbs_nic_handler nic_handler,
			      bool use_rcx,
			      uint8_t rcx_type,
			      bool set_core_direct,
			      struct doca_verbs_qp **verbs_qp)
{
	doca_error_t status;
	struct doca_verbs_qp_init_attr *verbs_qp_init_attr = NULL;
	struct doca_verbs_qp *new_qp = NULL;
	uint32_t external_umem_size;
	size_t dbr_umem_align_sz = EXT_ROUND_UP(VERBS_TEST_DBR_SIZE, ext_page_size());

	*gpu_umem = NULL;
	*gpu_umem_dev_ptr = NULL;
	*gpu_umem_dbr = NULL;
	*gpu_umem_dbr_dev_ptr = NULL;

	status = doca_verbs_qp_init_attr_create(&verbs_qp_init_attr);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to create doca verbs qp attributes");
		return status;
	}

	status = doca_verbs_qp_init_attr_set_external_datapath_en(verbs_qp_init_attr, 1);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_external_uar(verbs_qp_init_attr, external_uar);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	external_umem_size = calc_qp_external_umem_size(rq_nwqe, sq_nwqe);

	status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, ext_page_size(),
				    DOCA_GPU_MEM_TYPE_GPU, gpu_umem_dev_ptr, NULL);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to alloc gpu memory for external umem qp");
		goto destroy_resources;
	}

	status = doca_umem_gpu_create(gpu_dev, dev, *gpu_umem_dev_ptr, external_umem_size,
				      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
					      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
				      gpu_umem);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_external_umem(verbs_qp_init_attr, *gpu_umem, 0);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
		*gpu_umem_dbr_dev_ptr = calloc(dbr_umem_align_sz, sizeof(uint8_t));
		if (*gpu_umem_dbr_dev_ptr == NULL) {
			status = DOCA_ERROR_NO_MEMORY;
			goto destroy_resources;
		}

		status = doca_umem_create(dev, *gpu_umem_dbr_dev_ptr, dbr_umem_align_sz,
					  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
						  DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
					  gpu_umem_dbr);
		if (status != DOCA_SUCCESS)
			goto destroy_resources;
	} else {
		status = doca_gpu_mem_alloc(gpu_dev, dbr_umem_align_sz, ext_page_size(),
					    DOCA_GPU_MEM_TYPE_GPU, gpu_umem_dbr_dev_ptr, NULL);
		if (status != DOCA_SUCCESS)
			goto destroy_resources;

		status = doca_umem_gpu_create(gpu_dev, dev, *gpu_umem_dbr_dev_ptr, dbr_umem_align_sz,
					      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
						      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
					      gpu_umem_dbr);
		if (status != DOCA_SUCCESS)
			goto destroy_resources;
	}

	status = doca_verbs_qp_init_attr_set_external_dbr_umem(verbs_qp_init_attr, *gpu_umem_dbr, 0);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_pd(verbs_qp_init_attr, verbs_pd);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_sq_wr(verbs_qp_init_attr, sq_nwqe);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_rq_wr(verbs_qp_init_attr, rq_nwqe);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = set_qp_transport_type(verbs_qp_init_attr, use_rcx, rcx_type);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_send_cq(verbs_qp_init_attr, cq_sq);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_send_max_sges(verbs_qp_init_attr, VERBS_TEST_MAX_SEND_SEGS);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	status = doca_verbs_qp_init_attr_set_receive_max_sges(verbs_qp_init_attr, VERBS_TEST_MAX_RECEIVE_SEGS);
	if (status != DOCA_SUCCESS)
		goto destroy_resources;

	if (cq_rq != NULL) {
		status = doca_verbs_qp_init_attr_set_receive_cq(verbs_qp_init_attr, cq_rq);
		if (status != DOCA_SUCCESS)
			goto destroy_resources;
	}

	if (set_core_direct)
		doca_verbs_qp_init_attr_set_core_direct_master(verbs_qp_init_attr, 1);

	status = doca_verbs_qp_create(verbs_ctx, verbs_qp_init_attr, &new_qp);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to create doca verbs QP");
		goto destroy_resources;
	}

	doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
	*verbs_qp = new_qp;
	return DOCA_SUCCESS;

destroy_resources:
	if (new_qp != NULL)
		doca_verbs_qp_destroy(new_qp);
	if (verbs_qp_init_attr != NULL)
		doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
	if (*gpu_umem_dbr != NULL) {
		doca_umem_destroy(*gpu_umem_dbr);
		*gpu_umem_dbr = NULL;
	}
	if (*gpu_umem != NULL) {
		doca_umem_destroy(*gpu_umem);
		*gpu_umem = NULL;
	}
	if (*gpu_umem_dev_ptr != NULL) {
		doca_gpu_mem_free(gpu_dev, *gpu_umem_dev_ptr);
		*gpu_umem_dev_ptr = NULL;
	}
	return status;
}

/* Release everything held by a single qp_hl. Does not free the struct itself. */
static void destroy_qp_hl_internal(struct doca_gpu_verbs_qp_hl *qp)
{
	if (qp->qp_gverbs != NULL)
		doca_gpu_verbs_unexport_qp(qp->gpu_dev, qp->qp_gverbs);
	if (qp->qp != NULL)
		doca_verbs_qp_destroy(qp->qp);
	if (qp->qp_umem != NULL)
		doca_umem_destroy(qp->qp_umem);
	if (qp->qp_umem_gpu_ptr != NULL)
		doca_gpu_mem_free(qp->gpu_dev, qp->qp_umem_gpu_ptr);
	if (qp->qp_umem_dbr != NULL)
		doca_umem_destroy(qp->qp_umem_dbr);
	if (qp->cq_rq != NULL)
		doca_verbs_cq_destroy(qp->cq_rq);
	if (qp->cq_rq_umem != NULL)
		doca_umem_destroy(qp->cq_rq_umem);
	if (qp->cq_rq_umem_gpu_ptr != NULL)
		doca_gpu_mem_free(qp->gpu_dev, qp->cq_rq_umem_gpu_ptr);
	if (qp->cq_sq != NULL)
		doca_verbs_cq_destroy(qp->cq_sq);
	if (qp->cq_sq_umem != NULL)
		doca_umem_destroy(qp->cq_sq_umem);
	if (qp->cq_sq_umem_gpu_ptr != NULL)
		doca_gpu_mem_free(qp->gpu_dev, qp->cq_sq_umem_gpu_ptr);
	if (qp->external_uar != NULL)
		doca_uar_destroy(qp->external_uar);
	if (qp->verbs_pd != NULL)
		doca_verbs_pd_destroy(qp->verbs_pd);
	if (qp->verbs_context != NULL)
		doca_verbs_context_destroy(qp->verbs_context);
	if (qp->dev != NULL)
		doca_dev_close(qp->dev);
}

/*
 * Fill an already-allocated qp_hl. If shared_uar != NULL it is used instead of
 * creating a new one (caller retains ownership; the shared UAR must be cleared
 * from this qp_hl before teardown).
 */
static doca_error_t build_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *ia,
				bool use_rcx,
				uint8_t rcx_type,
				uint32_t rq_nwqe,
				bool set_core_direct,
				struct doca_uar *shared_uar,
				struct doca_gpu_verbs_qp_hl *qp_)
{
	doca_error_t status;
	uint32_t sq_nwqe = ext_next_pow2(ia->sq_nwqe);

	qp_->gpu_dev = ia->gpu_dev;
	qp_->nic_handler = ia->nic_handler;

	status = import_from_pd(ia->ibpd, &qp_->dev, &qp_->verbs_context, &qp_->verbs_pd);
	if (status != DOCA_SUCCESS)
		return status;

	if (shared_uar != NULL)
		qp_->external_uar = shared_uar;
	else {
		status = create_uar(qp_->dev, qp_->nic_handler, &qp_->external_uar);
		if (status != DOCA_SUCCESS)
			goto error;
	}

	if (sq_nwqe > 0) {
		status = create_cq(qp_->gpu_dev, qp_->dev, qp_->verbs_context, sq_nwqe,
				   &qp_->cq_sq_umem_gpu_ptr, &qp_->cq_sq_umem, qp_->external_uar,
				   &qp_->cq_sq);
		if (status != DOCA_SUCCESS)
			goto error;
	}

	if (rq_nwqe > 0) {
		rq_nwqe = ext_next_pow2(rq_nwqe);
		status = create_cq(qp_->gpu_dev, qp_->dev, qp_->verbs_context, rq_nwqe,
				   &qp_->cq_rq_umem_gpu_ptr, &qp_->cq_rq_umem, qp_->external_uar,
				   &qp_->cq_rq);
		if (status != DOCA_SUCCESS)
			goto error;
	}

	status = create_qp(qp_->gpu_dev, qp_->dev, qp_->verbs_context, qp_->verbs_pd, qp_->cq_sq,
			   sq_nwqe, qp_->cq_rq, rq_nwqe, &qp_->qp_umem_gpu_ptr, &qp_->qp_umem,
			   &qp_->qp_umem_dbr_gpu_ptr, &qp_->qp_umem_dbr, qp_->external_uar,
			   qp_->nic_handler, use_rcx, rcx_type, set_core_direct, &qp_->qp);
	if (status != DOCA_SUCCESS)
		goto error;

	status = doca_gpu_verbs_export_qp(qp_->gpu_dev, qp_->dev, qp_->qp, qp_->nic_handler,
					 qp_->qp_umem_gpu_ptr, qp_->cq_sq, qp_->cq_rq,
					 &qp_->qp_gverbs);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to export GPU verbs QP");
		goto error;
	}

	return DOCA_SUCCESS;

error:
	if (shared_uar != NULL)
		qp_->external_uar = NULL;
	destroy_qp_hl_internal(qp_);
	memset(qp_, 0, sizeof(*qp_));
	return status;
}

static doca_error_t create_qp_hl_common(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					bool use_rcx,
					uint8_t rcx_type,
					uint32_t rq_nwqe,
					struct doca_gpu_verbs_qp_hl **qp)
{
	doca_error_t status;

	if (qp_init_attr == NULL || qp == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	if (qp_init_attr->gpu_dev == NULL || qp_init_attr->ibpd == NULL ||
	    qp_init_attr->sq_nwqe == 0) {
		EXT_ERR("Invalid input: gpu_dev %p ibpd %p sq_nwqe %u",
			(void *)qp_init_attr->gpu_dev, (void *)qp_init_attr->ibpd,
			qp_init_attr->sq_nwqe);
		return DOCA_ERROR_INVALID_VALUE;
	}

	struct doca_gpu_verbs_qp_hl *qp_ =
		(struct doca_gpu_verbs_qp_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_hl));
	if (qp_ == NULL)
		return DOCA_ERROR_NO_MEMORY;

	status = build_qp_hl(qp_init_attr, use_rcx, rcx_type, rq_nwqe, false, NULL, qp_);
	if (status != DOCA_SUCCESS) {
		free(qp_);
		return status;
	}

	*qp = qp_;
	return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_create_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					 struct doca_gpu_verbs_qp_hl **qp)
{
	return create_qp_hl_common(qp_init_attr, false, 0, 0, qp);
}

doca_error_t doca_gpu_verbs_create_rcx_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					     struct doca_gpu_verbs_qp_hl **qp)
{
#ifdef GPUNETIO_VERBS_RCX_SUPPORTED
	uint32_t rq_nwqe;

	if (qp_init_attr == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	/* RCX requires a receive queue to consume the Write-with-Immediate CQEs. */
	rq_nwqe = (qp_init_attr->rq_nwqe != 0) ? qp_init_attr->rq_nwqe : qp_init_attr->sq_nwqe;

	return create_qp_hl_common(qp_init_attr, true, qp_init_attr->rcx_type, rq_nwqe, qp);
#else
	(void)qp_init_attr;
	(void)qp;
	EXT_ERR("RCX QP creation requires the DOCA Verbs multi-path API");
	return DOCA_ERROR_NOT_SUPPORTED;
#endif
}

doca_error_t doca_gpu_verbs_create_qp_group_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					       struct doca_gpu_verbs_qp_group_hl **qpg)
{
	doca_error_t status;

	if (qp_init_attr == NULL || qpg == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	if (qp_init_attr->gpu_dev == NULL || qp_init_attr->ibpd == NULL ||
	    qp_init_attr->sq_nwqe == 0) {
		EXT_ERR("Invalid input: gpu_dev %p ibpd %p sq_nwqe %u",
			(void *)qp_init_attr->gpu_dev, (void *)qp_init_attr->ibpd,
			qp_init_attr->sq_nwqe);
		return DOCA_ERROR_INVALID_VALUE;
	}

	struct doca_gpu_verbs_qp_group_hl *qpg_ = (struct doca_gpu_verbs_qp_group_hl *)calloc(
		1, sizeof(struct doca_gpu_verbs_qp_group_hl));
	if (qpg_ == NULL)
		return DOCA_ERROR_NO_MEMORY;

	/* Main QP: not the core-direct master. */
	status = build_qp_hl(qp_init_attr, false, 0, 0, false, NULL, &qpg_->qp_main);
	if (status != DOCA_SUCCESS)
		goto error;

	/* Companion QP: core-direct master, sharing the main QP's UAR. */
	status = build_qp_hl(qp_init_attr, false, 0, 0, true, qpg_->qp_main.external_uar,
			     &qpg_->qp_companion);
	if (status != DOCA_SUCCESS)
		goto error;

	*qpg = qpg_;
	return DOCA_SUCCESS;

error:
	/* Companion never owns the shared UAR. */
	qpg_->qp_companion.external_uar = NULL;
	destroy_qp_hl_internal(&qpg_->qp_companion);
	destroy_qp_hl_internal(&qpg_->qp_main);
	free(qpg_);
	return status;
}

doca_error_t doca_gpu_verbs_destroy_qp_hl(struct doca_gpu_verbs_qp_hl *qp)
{
	if (qp == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	destroy_qp_hl_internal(qp);
	free(qp);
	return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_destroy_qp_group_hl(struct doca_gpu_verbs_qp_group_hl *qpg)
{
	if (qpg == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	/* Companion shares the main QP's UAR; clear it so it is freed only once. */
	qpg->qp_companion.external_uar = NULL;
	destroy_qp_hl_internal(&qpg->qp_companion);
	destroy_qp_hl_internal(&qpg->qp_main);
	free(qpg);
	return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_export_multi_qps_dev(struct doca_gpu *gpu_dev,
						 struct doca_gpu_verbs_qp **qps,
						 uint32_t num_qps,
						 struct doca_gpu_dev_verbs_qp **qps_dev)
{
	doca_error_t status = DOCA_SUCCESS;
	struct doca_gpu_dev_verbs_qp *dev_array = NULL;
	size_t elem_sz = sizeof(struct doca_gpu_dev_verbs_qp);

	if (gpu_dev == NULL || qps == NULL || qps_dev == NULL || num_qps == 0)
		return DOCA_ERROR_INVALID_VALUE;

	status = doca_gpu_mem_alloc(gpu_dev, num_qps * elem_sz, ext_page_size(),
				    DOCA_GPU_MEM_TYPE_GPU, (void **)&dev_array, NULL);
	if (status != DOCA_SUCCESS) {
		EXT_ERR("Failed to alloc gpu memory for multi-qp device array");
		return status;
	}

	for (uint32_t i = 0; i < num_qps; i++) {
		struct doca_gpu_dev_verbs_qp *qp_gpu = NULL;

		status = doca_gpu_verbs_get_qp_dev(qps[i], &qp_gpu);
		if (status != DOCA_SUCCESS) {
			EXT_ERR("Failed to get device QP handle for index %u", i);
			goto error;
		}

		if (cudaMemcpy((void *)(dev_array + i), (void *)qp_gpu, elem_sz,
			       cudaMemcpyDeviceToDevice) != cudaSuccess) {
			EXT_ERR("Failed to pack device QP %u into multi-qp array", i);
			status = DOCA_ERROR_DRIVER;
			goto error;
		}
	}

	*qps_dev = dev_array;
	return DOCA_SUCCESS;

error:
	doca_gpu_mem_free(gpu_dev, dev_array);
	return status;
}

doca_error_t doca_gpu_verbs_unexport_multi_qps_dev(struct doca_gpu *gpu_dev,
						   struct doca_gpu_verbs_qp **qps,
						   uint32_t num_qps,
						   struct doca_gpu_dev_verbs_qp *qps_dev)
{
	(void)qps;
	(void)num_qps;

	if (gpu_dev == NULL || qps_dev == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	return doca_gpu_mem_free(gpu_dev, qps_dev);
}
