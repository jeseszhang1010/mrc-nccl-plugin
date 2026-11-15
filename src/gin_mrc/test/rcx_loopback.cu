/*************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Single-node loopback check of the MRC GIN data path on the open-source DOCA
 * GPUNetIO stack: two QPs of one NIC (RCX by default) connected to each other,
 * Write-with-Immediate posted from a CUDA kernel with
 * doca_gpu_dev_verbs_put_immediate(), and the receive queues drained on the
 * CPU by doca_gpu_verbs_rq_progress() into a signal map, as the GIN plugin
 * does for its loopback responder.
 *
 * Checks that every signal is delivered and that each payload is visible
 * once its signal is, and reports the ping-pong latency (one way = half a
 * round trip, each half crossing a CPU delivery) and the streaming rate.
 *
 * RCX QPs are a DOCA SDK feature: run with DOCA_SDK_LIB_PATH set to the DOCA
 * SDK library directory.
 *
 * Usage: rcx_loopback [-d mlx5_0] [-g 0] [-x 3] [-t rcx|rc] [-n iters] [-s bytes] [-z]
 ************************************************************************/

#include <endian.h>
#include <getopt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <atomic>

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <doca_gpunetio_device.h>

#include "../mrc_qp.h"

#define CHECK(cond, ...)                                                                                              \
  do {                                                                                                                \
    if (!(cond)) {                                                                                                    \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                            \
      fprintf(stderr, __VA_ARGS__);                                                                                   \
      fprintf(stderr, "\n");                                                                                          \
      exit(1);                                                                                                        \
    }                                                                                                                 \
  } while (0)
#define DOCA_OK(call) CHECK((call) == DOCA_SUCCESS, "%s", #call)
#define CUDA_OK(call) CHECK((call) == cudaSuccess, "%s", #call)

enum : uint32_t { kSigAtB = 0, kSigAtA = 1, kNumSigs = 2 };

struct Progress {
  std::atomic<bool> run{true};
  struct doca_gpu_verbs_qp* qps[2];
  struct doca_gpu_verbs_signal_map* map;
  bool strongFence;
  doca_error_t error = DOCA_SUCCESS;
  // Delivery statistics, reported per phase.
  uint64_t sweeps = 0, busySweeps = 0, progressNs = 0, commitNs = 0;
};

static uint64_t hostNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void* progressMain(void* arg) {
  Progress* p = (Progress*)arg;
  while (p->run.load(std::memory_order_relaxed)) {
    bool any = false;
    const uint64_t t0 = hostNs();
    for (auto* qp : p->qps) {
      bool progressed = false;
      doca_error_t s = doca_gpu_verbs_rq_progress(qp, p->map, &progressed);
      if (s != DOCA_SUCCESS && p->error == DOCA_SUCCESS) p->error = s;
      any |= progressed;
    }
    p->sweeps++;
    if (any) {
      const uint64_t t1 = hostNs();
      doca_gpu_verbs_rq_commit_fence(p->strongFence);
      for (auto* qp : p->qps) doca_gpu_verbs_rq_commit_dbr(qp);
      doca_gpu_verbs_rq_commit_fence(p->strongFence);
      p->busySweeps++;
      p->progressNs += t1 - t0;
      p->commitNs += hostNs() - t1;
    }
  }
  return nullptr;
}

__device__ static uint64_t loadSys(uint64_t* p) {
  return cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*p).load(cuda::memory_order_acquire);
}

__device__ static uint64_t nowNs() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Ping-pong: A writes B's buffer and adds 1 to B's signal; once seen, B answers likewise.
__global__ void pingPong(doca_gpu_dev_verbs_qp* qpA, doca_gpu_dev_verbs_qp* qpB, uint8_t* bufA, uint8_t* bufB,
                         uint32_t keyA, uint32_t keyB, size_t bytes, uint64_t* sigAtB, uint64_t* sigAtA, int iters,
                         int warmup, uint64_t* out) {
  constexpr auto kMode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
  constexpr auto kNic = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  constexpr auto kCq = DOCA_GPUNETIO_VERBS_CQ_64B;
  doca_gpu_dev_verbs_addr a = {(uint64_t)bufA, keyA}, b = {(uint64_t)bufB, keyB};
  uint64_t start = 0;
  doca_gpu_dev_verbs_ticket_t t;
  for (int i = 0; i < warmup + iters; i++) {
    if (i == warmup) start = nowNs();
    *(volatile uint8_t*)bufA = (uint8_t)i;
    __threadfence_system();
    doca_gpu_dev_verbs_put_immediate<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, kMode, kNic, kCq>(qpA, b, a, bytes, kSigAtB, 1,
                                                                                          &t);
    while (loadSys(sigAtB) < (uint64_t)i + 1) {
    }
    if (*(volatile uint8_t*)bufB != (uint8_t)i) {
      printf("ping %d: payload %u not visible with its signal\n", i, *(volatile uint8_t*)bufB);
      out[1] = 1;
    }
    doca_gpu_dev_verbs_put_immediate<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, kMode, kNic, kCq>(qpB, a, b, bytes, kSigAtA, 1,
                                                                                          &t);
    while (loadSys(sigAtA) < (uint64_t)i + 1) {
    }
  }
  out[0] = nowNs() - start;
}

// Reference ping-pong without CPU delivery: plain 8-byte RDMA Writes of a sequence number, polled
// in GPU memory by the receiving side. The difference to pingPong is the cost of the RQ/CPU path.
__global__ void pingPongPolled(doca_gpu_dev_verbs_qp* qpA, doca_gpu_dev_verbs_qp* qpB, uint64_t* bufA, uint64_t* bufB,
                               uint32_t keyA, uint32_t keyB, int iters, int warmup, uint64_t* out) {
  constexpr auto kMode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
  constexpr auto kNic = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  constexpr auto kCq = DOCA_GPUNETIO_VERBS_CQ_64B;
  doca_gpu_dev_verbs_addr a = {(uint64_t)bufA, keyA}, b = {(uint64_t)bufB, keyB};
  uint64_t start = 0;
  doca_gpu_dev_verbs_ticket_t t;
  for (int i = 0; i < warmup + iters; i++) {
    if (i == warmup) start = nowNs();
    const uint64_t seq = 0x100000000ull + i;
    *(volatile uint64_t*)bufA = seq;
    __threadfence_system();
    doca_gpu_dev_verbs_put<kMode, kNic, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD, kCq>(qpA, b, a, sizeof(uint64_t), &t);
    while (loadSys(bufB) != seq) {
    }
    // The answer is a different word and value, so A cannot mistake its own store for it.
    *(volatile uint64_t*)(bufB + 1) = ~seq;
    __threadfence_system();
    doca_gpu_dev_verbs_addr a1 = {(uint64_t)(bufA + 1), keyA}, b1 = {(uint64_t)(bufB + 1), keyB};
    doca_gpu_dev_verbs_put<kMode, kNic, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD, kCq>(qpB, a1, b1, sizeof(uint64_t), &t);
    while (loadSys(bufA + 1) != ~seq) {
    }
  }
  out[0] = nowNs() - start;
}

// Streaming: the threads of the block post Write-with-Immediate ops on the shared QP; thread 0
// waits until all their signals are delivered.
__global__ void stream(doca_gpu_dev_verbs_qp* qpA, uint8_t* bufA, uint8_t* bufB, uint32_t keyA, uint32_t keyB,
                       size_t bytes, uint64_t* sigAtB, uint64_t base, int msgs, uint64_t* out) {
  constexpr auto kMode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
  constexpr auto kNic = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  constexpr auto kCq = DOCA_GPUNETIO_VERBS_CQ_64B;
  __syncthreads();
  uint64_t start = nowNs();
  for (int i = threadIdx.x; i < msgs; i += blockDim.x) {
    doca_gpu_dev_verbs_addr a = {(uint64_t)bufA + (size_t)threadIdx.x * bytes, keyA};
    doca_gpu_dev_verbs_addr b = {(uint64_t)bufB + (size_t)threadIdx.x * bytes, keyB};
    doca_gpu_dev_verbs_ticket_t t;
    doca_gpu_dev_verbs_put_immediate<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, kMode, kNic, kCq>(qpA, b, a, bytes, kSigAtB, 1,
                                                                                          &t);
  }
  if (threadIdx.x == 0) {
    while (loadSys(sigAtB) < base + msgs) {
    }
    out[0] = nowNs() - start;
  }
}

// Returns the big-endian key the device API expects (lkey == rkey for these MRs).
static uint32_t regMr(struct ibv_pd* pd, void* ptr, size_t size, bool dataDirect) {
  int fd = -1;
  struct ibv_mr* mr = nullptr;
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  CHECK(cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)ptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                      dataDirect ? CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE : 0) == CUDA_SUCCESS,
        "DMA-BUF export");
  if (dataDirect)
    mr = mlx5dv_reg_dmabuf_mr(pd, 0, size, (uintptr_t)ptr, fd, access, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  else
    mr = ibv_reg_dmabuf_mr(pd, 0, size, (uintptr_t)ptr, fd, access);
  close(fd);
  CHECK(mr != nullptr, "MR registration failed");
  CHECK(mr->lkey == mr->rkey, "lkey != rkey");
  return htobe32(mr->lkey);
}

static void connectQp(struct doca_gpu_verbs_qp_hl* qp, doca_verbs_ah_attr_t* ah, uint32_t remoteQpn, int rdAtom) {
  doca_verbs_qp_attr_t* attr = nullptr;
  DOCA_OK(doca_verbs_qp_attr_create(&attr));
  DOCA_OK(doca_verbs_qp_attr_set_path_mtu(attr, DOCA_VERBS_MTU_SIZE_4K_BYTES));
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
  DOCA_OK(doca_verbs_qp_attr_set_ah_attr(attr, ah));
  DOCA_OK(doca_verbs_qp_attr_set_dest_qp_num(attr, remoteQpn));
  DOCA_OK(doca_verbs_qp_attr_set_pkey_index(attr, 0));
  DOCA_OK(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, rdAtom));
  DOCA_OK(doca_verbs_qp_attr_set_max_rd_atomic(attr, rdAtom));
  DOCA_OK(doca_verbs_qp_modify(qp->qp, attr,
                               DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                 DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                 DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM));
  DOCA_OK(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
  DOCA_OK(doca_verbs_qp_modify(qp->qp, attr,
                               DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                 DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                 DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                 DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC | DOCA_VERBS_QP_ATTR_ATOMIC_MODE));
  DOCA_OK(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS));
  DOCA_OK(doca_verbs_qp_modify(qp->qp, attr,
                               DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                 DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                 DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));
  DOCA_OK(doca_verbs_qp_attr_destroy(attr));
}

// Waits for the kernel; on a hang or RQ error, reports both QPs' last error and fails.
static void printStats(Progress* p, uint64_t signals) {
  printf("  delivery: %lu signals in %lu busy sweeps (%.1f/sweep) of %lu; per busy sweep %.0f ns poll+apply, %.0f ns "
         "commit\n",
         signals, p->busySweeps, p->busySweeps ? (double)signals / p->busySweeps : 0.0, p->sweeps,
         p->busySweeps ? (double)p->progressNs / p->busySweeps : 0.0,
         p->busySweeps ? (double)p->commitNs / p->busySweeps : 0.0);
  p->sweeps = p->busySweeps = p->progressNs = p->commitNs = 0;
}

static void waitKernel(Progress* prog, struct doca_gpu_verbs_qp* qpA, struct doca_gpu_verbs_qp* qpB, const char* what) {
  for (int ms = 0; ms < 20000; ms++) {
    cudaError_t e = cudaStreamQuery(0);
    if (e == cudaSuccess && prog->error == DOCA_SUCCESS) return;
    if ((e != cudaErrorNotReady && e != cudaSuccess) || prog->error != DOCA_SUCCESS) break;
    usleep(1000);
  }
  struct doca_gpu_verbs_qp* qps[2] = {qpA, qpB};
  for (int i = 0; i < 2; i++) {
    struct doca_gpu_verbs_qp_error_info info = {};
    doca_gpu_verbs_query_last_error(qps[i], &info);
    fprintf(stderr, "QP %c: has_error %d syndrome 0x%x vendor 0x%x hw 0x%x type 0x%x wqe %d\n", 'A' + i,
            info.has_error, info.syndrome, info.vendor_err_synd, info.hw_err_synd, info.hw_synd_type, info.wqe_counter);
  }
  CHECK(false, "%s did not complete (rq_progress status %d)", what, prog->error);
}

int main(int argc, char** argv) {
  const char* devName = "mlx5_0";
  int gpu = 0, gidIndex = 3, iters = 10000, opt;
  size_t bytes = 8;
  bool rcx = true, dataDirect = false;
  while ((opt = getopt(argc, argv, "d:g:x:t:n:s:z")) != -1) {
    switch (opt) {
    case 'd': devName = optarg; break;
    case 'g': gpu = atoi(optarg); break;
    case 'x': gidIndex = atoi(optarg); break;
    case 't': rcx = strcmp(optarg, "rc") != 0; break;
    case 'n': iters = atoi(optarg); break;
    case 's': bytes = strtoull(optarg, nullptr, 0); break;
    case 'z': dataDirect = true; break;
    default:
      fprintf(stderr, "usage: %s [-d dev] [-g gpu] [-x gid] [-t rcx|rc] [-n iters] [-s bytes] [-z]\n", argv[0]);
      return 2;
    }
  }
  const int threads = 128;
  const size_t bufBytes = 1 << 26;
  CHECK(bytes > 0, "MRC rejects zero-length Write-with-Immediate");
  CHECK(bytes * threads <= bufBytes, "message too large");

  CUDA_OK(cudaSetDevice(gpu));
  CUDA_OK(cudaFree(0));
  char busId[32];
  CUDA_OK(cudaDeviceGetPCIBusId(busId, sizeof(busId), gpu));
  doca_gpu_t* gdev = nullptr;
  DOCA_OK(doca_gpu_create(busId, &gdev));

  int ndev = 0;
  struct ibv_device** list = ibv_get_device_list(&ndev);
  struct ibv_context* ctx = nullptr;
  for (int i = 0; i < ndev && !ctx; i++)
    if (strcmp(ibv_get_device_name(list[i]), devName) == 0) ctx = ibv_open_device(list[i]);
  ibv_free_device_list(list);
  CHECK(ctx != nullptr, "device %s not found", devName);
  struct ibv_pd* pd = ibv_alloc_pd(ctx);
  struct ibv_device_attr devAttr;
  union ibv_gid gid;
  CHECK(pd && ibv_query_device(ctx, &devAttr) == 0 && ibv_query_gid(ctx, 1, gidIndex, &gid) == 0, "ibv query");
  doca_dev_t* netDev = nullptr;
  DOCA_OK(doca_verbs_dev_open(pd, &netDev));

  struct mrcQpAttr qa = {};
  qa.gpu_dev = gdev;
  qa.net_dev = netDev;
  qa.qp_type = rcx ? MRC_QP_TYPE_RCX : DOCA_VERBS_QP_TYPE_RC;
  qa.sq_nwqe = 256;
  qa.rq_nwqe = 1024;
  qa.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  qa.data_direct = dataDirect;
  struct doca_gpu_verbs_qp_hl *qpA = nullptr, *qpB = nullptr;
  DOCA_OK(mrcQpCreate(&qa, &qpA));
  DOCA_OK(mrcQpCreate(&qa, &qpB));
  uint32_t qpnA = 0, qpnB = 0;
  DOCA_OK(doca_verbs_qp_get_qpn(qpA->qp, &qpnA));
  DOCA_OK(doca_verbs_qp_get_qpn(qpB->qp, &qpnB));

  doca_verbs_ah_attr_t* ah = nullptr;
  struct doca_verbs_gid vgid;
  memcpy(vgid.raw, gid.raw, sizeof(gid.raw));
  DOCA_OK(doca_verbs_ah_attr_create(netDev, &ah));
  DOCA_OK(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IPv4));
  DOCA_OK(doca_verbs_ah_attr_set_gid(ah, vgid));
  DOCA_OK(doca_verbs_ah_attr_set_sgid_index(ah, gidIndex));
  DOCA_OK(doca_verbs_ah_attr_set_hop_limit(ah, 255));
  connectQp(qpA, ah, qpnB, devAttr.max_qp_rd_atom);
  connectQp(qpB, ah, qpnA, devAttr.max_qp_rd_atom);

  doca_gpu_dev_verbs_qp *dA = nullptr, *dB = nullptr;
  DOCA_OK(doca_gpu_verbs_get_qp_dev(qpA->qp_gverbs, &dA));
  DOCA_OK(doca_gpu_verbs_get_qp_dev(qpB->qp_gverbs, &dB));

  uint8_t *bufA, *bufB;
  CUDA_OK(cudaMalloc(&bufA, bufBytes));
  CUDA_OK(cudaMalloc(&bufB, bufBytes));
  CUDA_OK(cudaMemset(bufA, 0, bufBytes));
  CUDA_OK(cudaMemset(bufB, 0xff, bufBytes));
  const uint32_t keyA = regMr(pd, bufA, bufBytes, dataDirect), keyB = regMr(pd, bufB, bufBytes, dataDirect);

  // Signal slots are application GPU memory, written only by the CPU delivery path.
  uint64_t* sigs;
  CUDA_OK(cudaMalloc(&sigs, kNumSigs * sizeof(uint64_t)));
  struct doca_gpu_verbs_signal_map* map = nullptr;
  DOCA_OK(doca_gpu_verbs_signal_map_create(&map));
  DOCA_OK(doca_gpu_verbs_signal_register_range(map, 0, kNumSigs - 1, sigs));

  Progress prog;
  prog.qps[0] = qpA->qp_gverbs;
  prog.qps[1] = qpB->qp_gverbs;
  prog.map = map;
  prog.strongFence = dataDirect;
  pthread_t thr;
  CHECK(pthread_create(&thr, nullptr, progressMain, &prog) == 0, "pthread_create");

  uint64_t* out;
  CUDA_OK(cudaMallocManaged(&out, 2 * sizeof(uint64_t)));
  out[0] = out[1] = 0;
  const int warmup = iters / 10;
  pingPong<<<1, 1>>>(dA, dB, bufA, bufB, keyA, keyB, bytes, sigs + kSigAtB, sigs + kSigAtA, iters, warmup, out);
  waitKernel(&prog, qpA->qp_gverbs, qpB->qp_gverbs, "ping-pong");
  CHECK(out[1] == 0, "a payload was not visible when its signal was");
  printf("%s%s ping-pong %zu B: %.2f us one-way over %d round trips\n", rcx ? "RCX" : "RC", dataDirect ? "+DD" : "",
         bytes, out[0] / 1e3 / iters / 2, iters);

  printStats(&prog, 2ull * (iters + warmup));

  CUDA_OK(cudaMemset(bufA, 0, bufBytes));
  CUDA_OK(cudaMemset(bufB, 0, bufBytes));
  pingPongPolled<<<1, 1>>>(dA, dB, (uint64_t*)bufA, (uint64_t*)bufB, keyA, keyB, iters, warmup, out);
  waitKernel(&prog, qpA->qp_gverbs, qpB->qp_gverbs, "polled ping-pong");
  printf("%s%s polled ping-pong 8 B (plain writes, no CPU): %.2f us one-way\n", rcx ? "RCX" : "RC",
         dataDirect ? "+DD" : "", out[0] / 1e3 / iters / 2);

  uint64_t hostSigs[kNumSigs];
  CUDA_OK(cudaMemcpy(hostSigs, sigs, sizeof(hostSigs), cudaMemcpyDeviceToHost));
  CHECK(hostSigs[kSigAtB] == (uint64_t)(iters + warmup) && hostSigs[kSigAtA] == (uint64_t)(iters + warmup),
        "signal totals %lu/%lu, expected %d", hostSigs[kSigAtB], hostSigs[kSigAtA], iters + warmup);

  const int msgs = iters * 10;
  stream<<<1, threads>>>(dA, bufA, bufB, keyA, keyB, bytes, sigs + kSigAtB, hostSigs[kSigAtB], msgs, out);
  waitKernel(&prog, qpA->qp_gverbs, qpB->qp_gverbs, "stream");
  printf("%s%s stream %zu B x %d threads: %.2f Mmsg/s, %.2f GB/s\n", rcx ? "RCX" : "RC", dataDirect ? "+DD" : "",
         bytes, threads, msgs / (out[0] / 1e3), msgs * (double)bytes / out[0]);
  printStats(&prog, msgs);

  prog.run = false;
  pthread_join(thr, nullptr);
  CHECK(prog.error == DOCA_SUCCESS, "rq_progress error %d", prog.error);
  CUDA_OK(cudaMemcpy(hostSigs, sigs, sizeof(hostSigs), cudaMemcpyDeviceToHost));
  CHECK(hostSigs[kSigAtB] == (uint64_t)(iters + warmup + msgs), "stream signal total %lu, expected %d",
        hostSigs[kSigAtB], iters + warmup + msgs);

  DOCA_OK(doca_gpu_verbs_signal_unregister_range(map, 0, kNumSigs - 1));
  DOCA_OK(doca_gpu_verbs_signal_map_destroy(map));
  doca_verbs_ah_attr_destroy(ah);
  mrcQpDestroy(qpA);
  mrcQpDestroy(qpB);
  printf("PASSED\n");
  return 0;
}
