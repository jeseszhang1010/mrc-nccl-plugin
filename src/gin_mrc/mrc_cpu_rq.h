/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * CPU RQ mode of the MRC GIN plugin. NCCL's device code is built on the
 * open-source DOCA GPUNetIO that NCCL embeds: it sends with
 * doca_gpu_dev_verbs_put_immediate, whose immediate carries a signal op, id and
 * value, and never touches the receive queues. This module, built against the
 * open-source GPUNetIO headers NCCL installs (and nothing from the DOCA SDK):
 *   - lays out the open-source struct doca_gpu_dev_verbs_qp that device code
 *     drives, from the send queue of the DOCA SDK QP (mrcQpDesc);
 *   - drains the host-memory receive CQs on the CPU, applies each immediate
 *     to its GPU signal slot through a GDRCopy mapping, and returns the
 *     receive credits through the doorbell record.
 * Same wire format and semantics as the open-source signal maps and
 * doca_gpu_verbs_rq_progress().
 ************************************************************************/

#ifndef MRC_CPU_RQ_H_
#define MRC_CPU_RQ_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mrc_qp_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Signal ids a Write-with-Immediate can address (DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS).
uint32_t mrcCpuRqMaxSignalIds(void);

// GPU array of n open-source struct doca_gpu_dev_verbs_qp, one per description. Returns a cudaError_t.
int mrcCpuRqExportQps(const struct mrcQpDesc* descs, int n, void** gpuQps);
void mrcCpuRqFreeQps(void* gpuQps);

// CPU mapping of GPU memory through GDRCopy. forcePcie maps it over the GPU's PCIe BAR, so CPU
// writes are ordered with Data Direct NIC writes. gpu must be 64 KiB aligned. Returns 0, -ENOSYS
// without GDRCopy, -ENOTSUP without forced-PCIe support, or -EIO.
struct mrcCpuRqMap;
int mrcCpuRqMapCreate(void* gpu, size_t size, bool forcePcie, struct mrcCpuRqMap** map);
volatile uint64_t* mrcCpuRqMapCpu(struct mrcCpuRqMap* map, void* gpu);
void mrcCpuRqMapDestroy(struct mrcCpuRqMap* map);

// Whether the GPU needs a read of a slot before the CPU writes it (no GPUDirect RDMA write ordering).
bool mrcCpuRqNeedsMcst(int cudaDev);

// Delivery target of a set of receive queues: signal ids [0, nIds) at slots[id].
struct mrcCpuRqSignals {
  volatile uint64_t* slots; // CPU mapping of the GPU slots
  uint64_t* shadow;         // [nIds] last value written to each slot
  uint32_t nIds;
  bool mcst;
  uint64_t dropped;     // undeliverable immediates, consumed
  uint32_t droppedImm;  // the first of them
};

// One CPU-managed receive queue.
struct mrcCpuRq {
  const volatile uint8_t* cqes;
  uint32_t cqeNum;
  uint64_t cqeCi;
  volatile uint32_t* dbrec;
  uint64_t dbrPi;
  uint64_t dbrPiShadow;
  int fatal;
};

struct mrcCpuRqError {
  uint8_t opcode;
  uint8_t syndrome;
  uint8_t vendorSyndrome;
  uint8_t hwSyndrome;
  uint16_t wqeCounter;
};

// Starts tracking a receive queue whose ring the QP builder posted.
void mrcCpuRqInit(struct mrcCpuRq* rq, const struct mrcQpDesc* desc);

// Applies up to 64 inbound Write-with-Immediate completions to sigs. Returns 0, or a negative errno
// once the queue failed (filling *err the first time); a failed queue is never polled again. The
// new signal values and the consumed credits are published by the next commit: mrcCpuRqCommitFence,
// mrcCpuRqCommitDbr on every queue of the sweep, then mrcCpuRqCommitFence again. strongFence is
// needed with Data Direct.
int mrcCpuRqPoll(struct mrcCpuRq* rq, struct mrcCpuRqSignals* sigs, bool* progressed, struct mrcCpuRqError* err);
void mrcCpuRqCommitFence(bool strongFence);
void mrcCpuRqCommitDbr(struct mrcCpuRq* rq);

#ifdef __cplusplus
}
#endif

#endif // MRC_CPU_RQ_H_
