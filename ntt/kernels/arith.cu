// Copyright Jiazheng Liu, EthStorage
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Some constant macro only be used in this file
#define MAX_THREAD_NUM 1024
#define NEXT 8 // 8 is the row of gate constraint, because we extend the domain by 8
#define ONE fr_t::one()
#define TWO (fr_t::one() + fr_t::one())
#define THREE (fr_t::one() + fr_t::one() + fr_t::one())
#define FOUR (fr_t::one() + fr_t::one() + fr_t::one() + fr_t::one())
#define NINE (THREE * THREE)
#define EIGHTEEN (NINE * TWO)
#define EIGHTY_ONE (NINE * NINE)
#define EIGHTY_THREE (EIGHTY_ONE + TWO)

// Hardcode the SBOX in constant time
#define POW_SBOX(n) ((n) * (n) * (n) * (n) * (n))
#define SQAURE(n) ((n) * (n))

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
    X(q_m)       \
    X(r_s)       \
    X(l_s)

#define AUX_LIST(X) \
    X(challenges)

#define RANGE_CHALLENGE challenges[0]
#define LOGIC_CHALLENGE challenges[1]

// Compose a argument list of following function
#define MAKE_PTR_ARGUMENT(var) , const fr_t* var

// Compose a parameter list of following function
#define MAKE_PARAMETER(var) , var

// Total argument
#define TOTAL_ARGUMENT \
    POINTER_LIST(MAKE_PTR_ARGUMENT) AUX_LIST(MAKE_PTR_ARGUMENT)

#define TOTAL_PARAMETER \
    POINTER_LIST(MAKE_PARAMETER) AUX_LIST(MAKE_PARAMETER)

__device__ fr_t compute_quotient_i(size_t i TOTAL_ARGUMENT)
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

__device__ fr_t delta(fr_t f)
{
    fr_t f_1 = f - ONE;
    fr_t f_2 = f - TWO;
    fr_t f_3 = f - THREE;
    return f * f_1 * f_2 * f_3; 
}

__device__ fr_t range_quoteint_term(size_t i TOTAL_ARGUMENT)
{
   fr_t kappa = RANGE_CHALLENGE * RANGE_CHALLENGE;
   fr_t kappa_sq = kappa * kappa;
   fr_t kappa_cu = kappa_sq * kappa;
   fr_t b1 = delta(w_o[i] - FOUR * w_4[i]);
   fr_t b2 = delta(w_r[i] - FOUR * w_o[i]) * kappa;
   fr_t b3 = delta(w_l[i] - FOUR * w_r[i]) * kappa_sq;
   // NOTICE: w_4 is next one, should add next line
   fr_t b4 = delta(w_4[i + NEXT] - FOUR * w_l[i]) * kappa_cu;

   return r_s[i] * (b1 + b2 + b3 + b4) * RANGE_CHALLENGE;
}

__device__ fr_t delta_xor_and(fr_t a, fr_t b, fr_t w, fr_t c, fr_t q_c)
{
    fr_t F = w
        * (w * (FOUR * w - EIGHTEEN * (a + b) + EIGHTY_ONE)
            + EIGHTEEN * (SQAURE(a) + SQAURE(b))
            - EIGHTY_ONE * (a + b)
            + EIGHTY_THREE);
    fr_t E = THREE * (a + b + c) - (TWO * F);
    fr_t B = q_c * ((NINE * c) - THREE * (a + b));
    return E + B;
}

__device__ fr_t logic_quotient_term(size_t i TOTAL_ARGUMENT)
{
    fr_t kappa = RANGE_CHALLENGE * RANGE_CHALLENGE;
    fr_t kappa_sq = kappa * kappa;
    fr_t kappa_cu = kappa_sq * kappa;
    fr_t kappa_qu = kappa_cu * kappa;

    fr_t a = w_l[i + NEXT] - FOUR * w_l[i];
    fr_t c_0 = delta(a);

    fr_t b = w_r[i + NEXT] - FOUR * w_r[i];
    fr_t c_1 = delta(b) * kappa;

    fr_t d = w_4[i + NEXT] - FOUR * w_4[i];
    fr_t c_2 = delta(d) * kappa_sq;

    fr_t w = w_o[i];
    fr_t c_3 = (w - a * b) * kappa_cu;

    fr_t c_4 = delta_xor_and(a, b, w, d, q_c[i]) * kappa_qu;

    return l_s[i] * (c_0 + c_1 + c_2 + c_3 + c_4) * LOGIC_CHALLENGE;
}

__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void gate_constraint_sat_kernel(const uint lg_domain_size, fr_t* out
                                TOTAL_ARGUMENT)
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

    out[tid] = compute_quotient_i(tid TOTAL_PARAMETER) + 
                range_quoteint_term(tid TOTAL_PARAMETER) +
                logic_quotient_term(tid TOTAL_PARAMETER);
}

#undef MAX_THREAD_NUM
#undef NEXT
#undef ONE
#undef TWO
#undef THREE
#undef FOUR
#undef NINE
#undef EIGHTEEN
#undef EIGHTY_ONE
#undef EIGHTY_THREE
#undef POW_SBOX