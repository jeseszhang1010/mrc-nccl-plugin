/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Library-neutral description of an MRC GIN QP: the queue addresses that the
 * DOCA SDK side (mrc_qp.cc) hands to the code built against the open-source
 * GPUNetIO headers (mrc_cpu_rq.cc). The two cannot share a translation unit:
 * their headers use the same include guards for different layouts.
 ************************************************************************/

#ifndef MRC_QP_DESC_H_
#define MRC_QP_DESC_H_

#include <stdbool.h>
#include <stdint.h>

struct mrcQpDesc {
  // Send queue, as the DOCA SDK GPU handler of the QP drives it. GPU addresses.
  uint32_t qpn;
  void* sq_wqe;
  uint32_t sq_wqe_num;
  uint32_t* sq_dbrec;
  uint64_t* sq_db; // doorbell register, mapped for the GPU
  bool sq_bf;      // ring it with BlueFlame writes
  void* sq_cqe;
  uint32_t sq_cqn;
  uint32_t sq_cqe_num;
  uint8_t sq_cqe_size;
  uint32_t* sq_cq_dbrec;

  // CPU-managed receive queue (NULL addresses otherwise). The CQ is in host memory, and the
  // doorbell record is reached through its CPU mapping.
  void* rq_cqe;
  uint32_t rq_cqe_num;
  uint8_t rq_cqe_size;
  uint32_t* rq_dbrec;
  uint32_t rq_wqe_num; // credits posted at creation
};

#endif // MRC_QP_DESC_H_
