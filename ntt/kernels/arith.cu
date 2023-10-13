// Copyright Jiazheng Liu, EthStorage
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define MAX_THREAD_NUM 1024

// Hardcode the SBOX in constant time
#define POW_SBOX(n) ((n) * (n) * (n) * (n) * (n))

// This MACRO is used to generate the argument list of following function
// And it would be used in following files as well
#define POINTER_LIST(X) \
    X(w_l)       \
    X(w_r)       \
    X(w_o)       \
    X(w_4)       \
    X(q_l)       \
    X(q_r)       \
    X(q_o)       \
    X(q_4)       \
    X(q_hl)      \
    X(q_hr)      \
    X(q_h4)      \
    X(q_c)       \
    X(q_arith)   \
    X(q_m)

// Compose a argument list of following function
#define MAKE_ARGUMENT(var) , const fr_t* var 

// Compose a parameter list of following function
#define MAKE_PARAMETER(var) , var

__device__ fr_t compute_quotient_i(size_t i POINTER_LIST(MAKE_ARGUMENT))
{
    return ((w_l[i] * w_r[i] * q_m[i])
            + (w_l[i] * q_l[i])
            + (w_r[i] * q_r[i])
            + (w_o[i] * q_o[i])
            + (w_4[i] * q_4[i])
            + (POW_SBOX(w_l[i]) * q_hl[i])
            + (POW_SBOX(w_r[i]) * q_hr[i])
            + (POW_SBOX(w_4[i]) * q_h4[i])
            + q_c[i])
            * q_arith[i];
}

__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void gate_constraint_sat_kernel(const uint lg_domain_size, fr_t* out
                                POINTER_LIST(MAKE_ARGUMENT))
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
#endif
    uint domain_size = 1 << lg_domain_size;
    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    // out of range, just return
    if (tid > domain_size) {
        return;
    }

    out[tid] = compute_quotient_i(tid POINTER_LIST(MAKE_PARAMETER));
}

#undef MAX_THREAD_NUM