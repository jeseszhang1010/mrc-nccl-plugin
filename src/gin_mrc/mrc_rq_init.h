/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * GPU RQ mode of the MRC GIN plugin: the one-time receive-ring fill, done on the
 * GPU because only the device code knows how to post receive WQEs on a QP
 * exported to the DOCA SDK (doca_gpu_verbs_export_sdk_qp). The host code is
 * plain C++ and cannot launch kernels.
 ************************************************************************/

#ifndef MRC_RQ_INIT_H_
#define MRC_RQ_INIT_H_

#include <cuda_runtime.h>
#include <stdint.h>

#include "nccl_device/gin/mrc/gin_mrc_dev.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pre-posts every peer QP's receive ring, from NCCL's own GPU RQ device code (rqInit) so that the
// fill and the device drain always agree on the ring state. qps[peer] are the DOCA SDK GPU handles
// of a context's QPs; the local rank's ring is selfRqQp's, the loopback responder that receives
// this rank's own traffic. Must run before any peer issues an operation. postMode is
// NCCL_GIN_MRC_RQ_POST_MODE_*.
cudaError_t mrcGinInitRqsLaunch(void* const* qps, void* selfRqQp, int rank, struct ncclGinMrcPeerState* peers,
                                int numQps, uint32_t* counterSlots, uint32_t counterCapacity, uint32_t postMode,
                                cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // MRC_RQ_INIT_H_
