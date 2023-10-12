// Copyright Jiazheng Liu, EthStorage
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define MAX_THREAD_NUM 1024

__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void gate_constraint_sat_kernel(const uint lg_domain_size, fr_t* a, fr_t* b, fr_t* c)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
#endif
    uint domain_size = 1 << lg_domain_size;
    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    if (tid < domain_size) {
        c[tid] = a[tid] + b[tid];
    }
}

#undef MAX_THREAD_NUM