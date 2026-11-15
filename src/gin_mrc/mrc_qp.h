/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RCX queue-pair builder for the MRC GIN plugin, on top of the open-source
 * DOCA GPUNetIO host library (DOCA_HOME).
 *
 * The open-source high-level QP API always creates RC QPs. MRC needs RCX QPs,
 * whose creation is a DOCA SDK restricted feature: the open-source library
 * forwards every verbs call to the DOCA SDK when DOCA_SDK_LIB_PATH points at
 * it (SDK mode), including the raw QP type set here. The send queue and its
 * CQ live in GPU memory and are driven by device code. The receive queue is
 * either CPU managed, with a host-memory receive CQ and a GDRCopy mapping of
 * the doorbell record, so that doca_gpu_verbs_rq_progress() can deliver
 * inbound Write-with-Immediate signals into a signal map; or GPU managed, with
 * a GPU-memory receive CQ that DOCA SDK device code drains after the QP is
 * exported with doca_gpu_verbs_export_sdk_qp().
 *
 * Only depends on the open-source DOCA host headers, so it can be linked into
 * standalone tests as well as the plugin.
 ************************************************************************/

#ifndef MRC_QP_H_
#define MRC_QP_H_

#include <stdint.h>

// Not doca_gpunetio_host.h: its mlx5 PRM headers collide with NCCL's bundled mlx5 definitions.
#include <host/doca_gpunetio.h>
#include <host/doca_gpunetio_high_level.h>
#include <host/doca_gpunetio_signal.h>
#include <host/doca_verbs.h>

// DOCA SDK RCX QP type (doca_verbs_mp.h), passed through to the SDK in SDK mode.
#define MRC_QP_TYPE_RCX 0x10000u

#ifdef __cplusplus
extern "C" {
#endif

struct mrcQpAttr {
  doca_gpu_t* gpu_dev;
  doca_dev_t* net_dev;
  uint32_t qp_type; // MRC_QP_TYPE_RCX, or DOCA_VERBS_QP_TYPE_RC
  uint32_t sq_nwqe; // rounded up to a power of two
  uint32_t rq_nwqe; // rounded up to a power of two; 0 = no receive queue
  enum doca_gpu_dev_verbs_nic_handler nic_handler;
  // The QP targets windows registered for Data Direct: the signal map must then deliver through
  // its forced-PCIe mappings so that signals are not observed ahead of the data.
  bool data_direct;
  // GPU-managed receive queue: the receive CQ is in GPU memory and the library's CPU receive path
  // (doca_gpu_verbs_rq_progress) is not set up; the receive CQ is still attached to the QP handle
  // (doca_gpu_verbs_qp::cq_rq), where doca_gpu_verbs_export_sdk_qp() finds it. The device code
  // posts the receive WQEs.
  bool rq_gpu;
};

// Creates and exports one QP: a regular high-level QP of the open-source library, released with
// mrcQpDestroy().
doca_error_t mrcQpCreate(const struct mrcQpAttr* attr, struct doca_gpu_verbs_qp_hl** out);

// Destroys a QP of mrcQpCreate(), including its DOCA SDK export if any.
doca_error_t mrcQpDestroy(struct doca_gpu_verbs_qp_hl* qp);

#ifdef __cplusplus
}
#endif

#endif // MRC_QP_H_
