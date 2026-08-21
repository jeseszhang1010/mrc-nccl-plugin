/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NCCL net properties v12: identical to v11 plus railId/planeId. Declared here
 * because the GIN v14 plugin ABI passes this struct and the net plugin in this
 * repo only carries v11.
 ************************************************************************/

#ifndef NET_V12_H_
#define NET_V12_H_

#include "net_v11.h"

typedef ncclNetVDeviceProps_v11_t ncclNetVDeviceProps_v12_t;

typedef struct {
  char* name;
  char* pciPath;
  uint64_t guid;
  int ptrSupport;
  int regIsGlobal;
  int forceFlush;
  int speed;
  int port;
  float latency;
  int maxComms;
  int maxRecvs;
  ncclNetDeviceType netDeviceType;
  int netDeviceVersion;
  ncclNetVDeviceProps_v12_t vProps;
  size_t maxP2pBytes;
  size_t maxCollBytes;
  int maxMultiRequestSize;
  int16_t railId;
  int16_t planeId;
} ncclNetProperties_v12_t;

#endif // end include guard
