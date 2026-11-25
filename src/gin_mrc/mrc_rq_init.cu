/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Receive-ring fill (see mrc_rq_init.h). Compiled with NCCL's device code and
 * the DOCA SDK device headers.
 ************************************************************************/

#include <cuda_runtime.h>

#include "nccl_device/gin/mrc/gin_mrc.h"

#include "mrc_rq_init.h"

// One thread per peer performs the single-threaded RQ fill for its slot. Without a self loop
// (selfRqQp NULL) nothing arrives for the local rank, whose ring is left unfilled.
__global__ static void mrcGinInitRqKernel(void* const* qps, void* selfRqQp, int rank, struct ncclGinMrcPeerState* peers,
                                          int numQps, uint32_t* counterSlots, uint32_t counterCapacity,
                                          uint32_t postMode) {
  int peer = blockIdx.x * blockDim.x + threadIdx.x;
  if (peer < numQps && (peer != rank || selfRqQp != nullptr)) {
    uint32_t* slots = counterSlots ? counterSlots + (size_t)peer * counterCapacity : nullptr;
    void* qp = peer == rank ? selfRqQp : qps[peer];
    nccl::gin::mrc::rqInit((struct doca_gpu_dev_verbs_qp*)qp, &peers[peer], slots, counterCapacity, postMode);
  }
}

extern "C" cudaError_t mrcGinInitRqsLaunch(void* const* qps, void* selfRqQp, int rank, struct ncclGinMrcPeerState* peers,
                                           int numQps, uint32_t* counterSlots, uint32_t counterCapacity,
                                           uint32_t postMode, cudaStream_t stream) {
  if (numQps <= 0) return cudaSuccess;
  const int threads = 64;
  const int blocks = (numQps + threads - 1) / threads;
  mrcGinInitRqKernel<<<blocks, threads, 0, stream>>>(qps, selfRqQp, rank, peers, numQps, counterSlots, counterCapacity,
                                                      postMode);
  return cudaGetLastError();
}
