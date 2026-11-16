/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DOCA SDK side of the RCX loopback test (see rcx_loopback_host.h).
 ************************************************************************/

#include "rcx_loopback_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <doca_dev.h>
#include <doca_gpunetio.h>
#include <doca_rdma_bridge.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#define DOCA_OK(call)                                                                                   \
  do {                                                                                                  \
    doca_error_t s_ = (doca_error_t)(call);                                                             \
    if (s_ != DOCA_SUCCESS) {                                                                           \
      fprintf(stderr, "FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, #call, doca_error_get_descr(s_));     \
      exit(1);                                                                                          \
    }                                                                                                   \
  } while (0)

struct lbHost {
  struct doca_gpu* gpu;
  struct doca_dev* dev;
  struct doca_verbs_context* ctx;
  struct doca_verbs_pd* pd;
  struct doca_verbs_ah_attr* ah;
};

struct lbHost* lbHostOpen(int gpu, struct ibv_pd* pd, int gidIndex, const union ibv_gid* gid, struct mrcQpAttr* qa) {
  struct lbHost* h = (struct lbHost*)calloc(1, sizeof(*h));
  char busId[32];
  if (h == nullptr || cudaDeviceGetPCIBusId(busId, sizeof(busId), gpu) != cudaSuccess) {
    fprintf(stderr, "FAIL: GPU %d\n", gpu);
    exit(1);
  }
  DOCA_OK(doca_gpu_create(busId, &h->gpu));
  DOCA_OK(doca_rdma_bridge_open_dev_from_pd(pd, &h->dev));
  DOCA_OK(doca_verbs_bridge_verbs_context_import(pd->context, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &h->ctx));
  DOCA_OK(doca_verbs_bridge_verbs_pd_import(pd, &h->pd));

  struct doca_verbs_gid vgid;
  memcpy(vgid.raw, gid->raw, sizeof(gid->raw));
  DOCA_OK(doca_verbs_ah_attr_create(h->ctx, &h->ah));
  DOCA_OK(doca_verbs_ah_attr_set_addr_type(h->ah, DOCA_VERBS_ADDR_TYPE_IPv4));
  DOCA_OK(doca_verbs_ah_attr_set_gid(h->ah, vgid));
  DOCA_OK(doca_verbs_ah_attr_set_sgid_index(h->ah, gidIndex));
  DOCA_OK(doca_verbs_ah_attr_set_hop_limit(h->ah, 255));

  qa->gpu_dev = h->gpu;
  qa->dev = h->dev;
  qa->verbs_ctx = h->ctx;
  qa->verbs_pd = h->pd;
  return h;
}

void lbHostConnect(struct lbHost* h, struct mrcQp* qp, uint32_t remoteQpn, int rdAtom) {
  struct doca_verbs_qp_attr* attr = nullptr;
  DOCA_OK(doca_verbs_qp_attr_create(&attr));
  DOCA_OK(doca_verbs_qp_attr_set_path_mtu(attr, DOCA_MTU_SIZE_4K_BYTES));
  DOCA_OK(doca_verbs_qp_attr_set_rq_psn(attr, 0));
  DOCA_OK(doca_verbs_qp_attr_set_sq_psn(attr, 0));
  DOCA_OK(doca_verbs_qp_attr_set_port_num(attr, 1));
  DOCA_OK(doca_verbs_qp_attr_set_ack_timeout(attr, 20));
  DOCA_OK(doca_verbs_qp_attr_set_retry_cnt(attr, 7));
  DOCA_OK(doca_verbs_qp_attr_set_rnr_retry(attr, 7));
  DOCA_OK(doca_verbs_qp_attr_set_min_rnr_timer(attr, 1));
  DOCA_OK(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
  DOCA_OK(doca_verbs_qp_attr_set_allow_remote_write(attr, 1));
  DOCA_OK(doca_verbs_qp_attr_set_allow_remote_read(attr, 1));
  DOCA_OK(doca_verbs_qp_attr_set_atomic_mode(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
  DOCA_OK(doca_verbs_qp_attr_set_ah_attr(attr, h->ah));
  DOCA_OK(doca_verbs_qp_attr_set_dest_qp_num(attr, remoteQpn));
  DOCA_OK(doca_verbs_qp_attr_set_pkey_index(attr, 0));
  DOCA_OK(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, rdAtom));
  DOCA_OK(doca_verbs_qp_attr_set_max_rd_atomic(attr, rdAtom));
  DOCA_OK(doca_verbs_qp_modify(mrcQpVerbs(qp), attr,
                               DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                 DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                 DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM));
  DOCA_OK(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
  DOCA_OK(doca_verbs_qp_modify(mrcQpVerbs(qp), attr,
                               DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                 DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                 DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                 DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC | DOCA_VERBS_QP_ATTR_ATOMIC_MODE));
  DOCA_OK(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS));
  DOCA_OK(doca_verbs_qp_modify(mrcQpVerbs(qp), attr,
                               DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                 DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                 DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));
  DOCA_OK(doca_verbs_qp_attr_destroy(attr));
}

void lbHostClose(struct lbHost* h) {
  doca_verbs_ah_attr_destroy(h->ah);
  doca_verbs_pd_destroy(h->pd);
  doca_verbs_context_destroy(h->ctx);
  doca_dev_close(h->dev);
  doca_gpu_destroy(h->gpu);
  free(h);
}
