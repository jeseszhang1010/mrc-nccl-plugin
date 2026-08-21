/*************************************************************************
 * NCCL-owned external-DOCA compatibility layer for the GDAKI backend.
 *
 * When NCCL is built against the prebuilt external DOCA 3.2 package
 * (DOCA_ROOT set, NCCL_GIN_GDAKI_RCX defined) the vendored high-level
 * host layer (doca_gpunetio_host.h / doca_gpunetio_high_level.h) is not
 * available: the external package only ships the low-level public API
 * (doca_gpunetio.h, doca_verbs.h, doca_verbs_mp.h and the device .cuh).
 *
 * This header re-declares the vendored-compatible high-level types and
 * helpers that gin_host_gdaki.cc depends on, implemented on top of the
 * external public API (see gin_gdaki_doca_ext.cc). It is included in
 * place of "doca_gpunetio_host.h" when NCCL_GIN_GDAKI_RCX is defined.
 ************************************************************************/

#ifndef MRC_DOCA_EXT_H
#define MRC_DOCA_EXT_H

#include <stdint.h>

#include <cuda_runtime_api.h>

#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#ifdef GPUNETIO_VERBS_RCX_SUPPORTED
#include <doca_verbs_mp.h>
#endif

// NOTE: we deliberately do NOT include <doca_gpunetio_verbs_def.h> or
// <doca_rdma_bridge.h> here. The former pulls in the real
// <infiniband/mlx5dv.h> and the latter the real <infiniband/verbs.h>, both of
// which conflict with NCCL's fallback ib/mlx5 definitions (mlx5/mlx5dvcore.h,
// ibvwrap.h) that are compiled into gin_host_gdaki.cc when NCCL_BUILD_MLX5DV /
// NCCL_BUILD_RDMA_CORE are not set. Host code only ever handles
// doca_gpu_dev_verbs_qp as an opaque pointer and does not call the rdma bridge,
// so a forward declaration is sufficient. The glue translation unit
// (gin_gdaki_doca_ext.cc) includes those headers directly where the concrete
// mlx5/verbs types are actually needed.
struct doca_gpu_dev_verbs_qp;
struct ibv_pd;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The following two enumerations are part of the vendored high-level API
 * (doca_gpunetio_high_level.h / common/doca_gpunetio_verbs_def.h) and are
 * not shipped by the external package. They are re-declared here with the
 * same names and values so that gin_host_gdaki.cc compiles unchanged. In
 * external mode the QP is created through the public doca_verbs create path,
 * which does not expose the DBR-mode knob, so send_dbr_mode_ext is accepted
 * for source compatibility but otherwise ignored by the glue.
 */
enum doca_gpu_verbs_mem_reg_type {
	DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT = 0,
	DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_DMABUF = 1,
	DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM = 2,
	DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_MAX,
};

enum doca_gpu_verbs_send_dbr_mode_ext {
	DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR = 0,
	DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW = 1,
	DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED = 2,
};

/*
 * High-level QP init attributes. Field names/semantics mirror the vendored
 * struct doca_gpu_verbs_qp_init_attr_hl so gin_host_gdaki.cc fills it the
 * same way. Two extra fields (rq_nwqe, rcx_type) support the RCX path, which
 * requires a receive queue; they default to 0 for the RC path.
 */
struct doca_gpu_verbs_qp_init_attr_hl {
	struct doca_gpu *gpu_dev;
	struct ibv_pd *ibpd;
	uint32_t sq_nwqe;
	uint32_t rq_nwqe;
	enum doca_gpu_dev_verbs_nic_handler nic_handler;
	enum doca_gpu_verbs_mem_reg_type mreg_type;
	enum doca_gpu_verbs_send_dbr_mode_ext send_dbr_mode_ext;
	uint8_t rcx_type;
	bool cq_collapsed;
};

/*
 * High-level QP. Holds every resource created for a single GDAKI QP so it can
 * be torn down by doca_gpu_verbs_destroy_qp_hl(). gin_host_gdaki.cc only reads
 * ->qp (for doca_verbs_qp_get_qpn / doca_verbs_qp_modify) and ->qp_gverbs (the
 * opaque GPUNetIO handle passed to the device-export step).
 */
struct doca_gpu_verbs_qp_hl {
	struct doca_gpu *gpu_dev;

	/* imported from the ibv_pd via the DOCA bridge */
	struct doca_dev *dev;
	struct doca_verbs_context *verbs_context;
	struct doca_verbs_pd *verbs_pd;

	/* send-completion queue */
	struct doca_verbs_cq *cq_sq;
	void *cq_sq_umem_gpu_ptr;
	struct doca_umem *cq_sq_umem;

	/* receive-completion queue (RCX only) */
	struct doca_verbs_cq *cq_rq;
	void *cq_rq_umem_gpu_ptr;
	struct doca_umem *cq_rq_umem;

	/* queue-pair */
	struct doca_verbs_qp *qp;
	void *qp_umem_gpu_ptr;
	struct doca_umem *qp_umem;
	void *qp_umem_dbr_gpu_ptr;
	struct doca_umem *qp_umem_dbr;
	struct doca_uar *external_uar;

	enum doca_gpu_dev_verbs_nic_handler nic_handler;

	/* GPUNetIO device handle (opaque) */
	struct doca_gpu_verbs_qp *qp_gverbs;
};

struct doca_gpu_verbs_qp_group_hl {
	struct doca_gpu_verbs_qp_hl qp_main;
	struct doca_gpu_verbs_qp_hl qp_companion;
};

/* Create a reliable-connection (RC) high-level QP. */
doca_error_t doca_gpu_verbs_create_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					 struct doca_gpu_verbs_qp_hl **qp);

/* Create an RCX high-level QP (RDMA Write / Write-with-Immediate only). */
doca_error_t doca_gpu_verbs_create_rcx_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					     struct doca_gpu_verbs_qp_hl **qp);

/* Create an RC QP group (main + companion self-loop for core-direct atomics). */
doca_error_t doca_gpu_verbs_create_qp_group_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
					       struct doca_gpu_verbs_qp_group_hl **qpg);

doca_error_t doca_gpu_verbs_destroy_qp_hl(struct doca_gpu_verbs_qp_hl *qp);
doca_error_t doca_gpu_verbs_destroy_qp_group_hl(struct doca_gpu_verbs_qp_group_hl *qpg);

/*
 * Build the contiguous device-side array of doca_gpu_dev_verbs_qp indexed by
 * peer that the MRC device context (ncclGinMrcGPUContext::qps) expects.
 * Replaces the vendored doca_gpu_verbs_export_multi_qps_dev, which is not part
 * of the DOCA public API, by fetching each per-QP device handle via
 * doca_gpu_verbs_get_qp_dev() and packing them into one GPU allocation.
 */
doca_error_t doca_gpu_verbs_export_multi_qps_dev(struct doca_gpu *gpu_dev,
						 struct doca_gpu_verbs_qp **qps,
						 uint32_t num_qps,
						 struct doca_gpu_dev_verbs_qp **qps_dev);

/* Release the contiguous device-side QP array produced by
 * doca_gpu_verbs_export_multi_qps_dev(). */
doca_error_t doca_gpu_verbs_unexport_multi_qps_dev(struct doca_gpu *gpu_dev,
						   struct doca_gpu_verbs_qp **qps,
						   uint32_t num_qps,
						   struct doca_gpu_dev_verbs_qp *qps_dev);

#ifdef __cplusplus
}
#endif

#endif /* MRC_DOCA_EXT_H */
