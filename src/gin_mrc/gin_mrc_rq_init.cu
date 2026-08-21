/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Receive-queue init kernel for the MRC GIN plugin. The host code is plain C++
 * and cannot launch kernels, so the one-time receive-ring fill lives here.
 *
 * The fill primitive itself is NCCL's, pulled from the installed device header
 * so host-side setup and device-side draining always agree on the ring state.
 ************************************************************************/

#include <cuda_runtime.h>

#include "nccl_device/gin/mrc/gin_mrc.h"

#include "gin_mrc.h"

// One thread per peer QP performs the single-threaded RQ fill for its slot.
__global__ static void ncclGinMrcInitRqKernel(struct doca_gpu_dev_verbs_qp* qps, struct ncclGinMrcRqState* rq,
                                              int numQps, uint32_t* counterSlots, uint32_t counterCapacity) {
  int peer = blockIdx.x * blockDim.x + threadIdx.x;
  if (peer < numQps) {
    uint32_t* slots = counterSlots ? counterSlots + (size_t)peer * counterCapacity : nullptr;
    nccl::gin::mrc::rqInit(&qps[peer], &rq[peer], slots, counterCapacity);
  }
}

extern "C" cudaError_t ncclGinMrcInitRqsLaunch(struct doca_gpu_dev_verbs_qp* qps, struct ncclGinMrcRqState* rq,
                                               int numQps, uint32_t* counterSlots, uint32_t counterCapacity,
                                               cudaStream_t stream) {
  if (numQps <= 0) return cudaSuccess;
  const int threads = 64;
  const int blocks = (numQps + threads - 1) / threads;
  ncclGinMrcInitRqKernel<<<blocks, threads, 0, stream>>>(qps, rq, numQps, counterSlots, counterCapacity);
  return cudaGetLastError();
}
