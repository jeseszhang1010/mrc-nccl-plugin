/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DOCA SDK side of the RCX loopback test: the kernels use the open-source
 * GPUNetIO device code, whose headers cannot share a translation unit with
 * the DOCA SDK ones.
 ************************************************************************/

#ifndef RCX_LOOPBACK_HOST_H_
#define RCX_LOOPBACK_HOST_H_

#include <stdint.h>

#include <infiniband/verbs.h>

#include "../mrc_qp.h"

struct lbHost;

// Opens the GPU and the NIC behind pd for the DOCA SDK, filling the device fields of *qa.
struct lbHost* lbHostOpen(int gpu, struct ibv_pd* pd, int gidIndex, const union ibv_gid* gid, struct mrcQpAttr* qa);
void lbHostConnect(struct lbHost* h, struct mrcQp* qp, uint32_t remoteQpn, int rdAtom);
void lbHostClose(struct lbHost* h);

#endif // RCX_LOOPBACK_HOST_H_
