/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RCX queue-pair builder for the MRC GIN plugin, on top of the DOCA SDK
 * (doca_verbs, doca_gpunetio).
 *
 * The send queue and its CQ live in GPU memory and are driven by device code.
 * Every QP is exported to the DOCA SDK GPU handler (doca_gpu_verbs_export_qp).
 * The receive queue is GPU managed: the receive CQ is in GPU memory and is
 * exported with the QP, so DOCA SDK device code posts and drains the receive
 * queue through the SDK GPU handler (mrcQpDevice).
 *
 * The header only forward-declares the DOCA types.
 ************************************************************************/

#ifndef MRC_QP_H_
#define MRC_QP_H_

#include <stdbool.h>
#include <stdint.h>

struct doca_gpu;
struct doca_dev;
struct doca_verbs_context;
struct doca_verbs_pd;
struct doca_verbs_qp;

#ifdef __cplusplus
extern "C" {
#endif

// DOCA SDK NIC handlers the device code may ring the doorbell with (enum doca_gpu_dev_verbs_nic_handler).
enum mrcQpNicHandler { MRC_QP_NIC_HANDLER_GPU_SM_DB = 2, MRC_QP_NIC_HANDLER_GPU_SM_BF = 3 };

struct mrcQpAttr {
  struct doca_gpu* gpu_dev;
  // The NIC, opened once per context from the ibv_pd through the DOCA bridges.
  struct doca_dev* dev;
  struct doca_verbs_context* verbs_ctx;
  struct doca_verbs_pd* verbs_pd;
  bool rcx;         // RCX (MRC) QP, or RC
  uint32_t sq_nwqe; // rounded up to a power of two
  uint32_t rq_nwqe; // rounded up to a power of two; 0 = no receive queue
  enum mrcQpNicHandler nic_handler;
};

struct mrcQp;

// Functions returning int return a doca_error_t (DOCA_SUCCESS = 0).
int mrcQpCreate(const struct mrcQpAttr* attr, struct mrcQp** out);
void mrcQpDestroy(struct mrcQp* qp);

struct doca_verbs_qp* mrcQpVerbs(struct mrcQp* qp);
uint32_t mrcQpNum(struct mrcQp* qp);
// GPU pointer to the DOCA SDK struct doca_gpu_dev_verbs_qp of the QP.
void* mrcQpDevice(struct mrcQp* qp);

#ifdef __cplusplus
}
#endif

#endif // MRC_QP_H_
