/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Interfaces shared between the MRC GIN plugin's host code and its CUDA
 * translation unit.
 ************************************************************************/

#ifndef GIN_MRC_H_
#define GIN_MRC_H_

#include <cuda_runtime.h>
#include <stdint.h>

// Layout of the GPU-resident context the device backend reads. Comes from the
// NCCL headers so the two definitions can never drift.
#include "nccl_device/gin/mrc/gin_mrc_dev.h"

// Bump whenever ncclGinMrcGPUContext / ncclGinMrcRqState change shape, together with
// mrcBackendMinVersions[] in NCCL's src/gin/gin_host.cc.
#define NCCL_GIN_MRC_VERSION 1

struct doca_gpu_dev_verbs_qp;

#ifdef __cplusplus
extern "C" {
#endif

// Pre-post every peer QP's receive ring. MRC signals arrive as Write-with-Immediate and
// consume a receive WQE, so this must run before any peer issues an operation.
cudaError_t ncclGinMrcInitRqsLaunch(struct doca_gpu_dev_verbs_qp* qps, struct ncclGinMrcRqState* rq, int numQps,
                                    uint32_t* counterSlots, uint32_t counterCapacity, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // end include guard
