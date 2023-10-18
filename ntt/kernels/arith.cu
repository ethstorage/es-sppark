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
#define K1 (THREE + FOUR)  // 7
#define K2 (THREE * FOUR + ONE)  // 13
#define K3 (TWO * K1 + THREE) // 17

// Hardcode the SBOX in constant time
#define POW_SBOX(n) ((n) * (n) * (n) * (n) * (n))
#define SQAURE(n) ((n) * (n))

// This MACRO is used to generate the argument list of following function
// And it would be used in following files as well
// All of them are equal length, except w_l, w_r, w_4, they are extended by 8
#define POINTER_LIST(X) \
    X(w_l)              \
    X(w_r)              \
    X(w_o)              \
    X(w_4)              \
    X(q_l)              \
    X(q_r)              \
    X(q_o)              \
    X(q_4)              \
    X(q_hl)             \
    X(q_hr)             \
    X(q_h4)             \
    X(q_c)              \
    X(q_arith)          \
    X(q_m)              \
    X(r_s)              \
    X(l_s)              \
    X(fbsm_s)           \
    X(vgca_s)           \
    X(pi)               \
    X(z)                \
    X(perm_linear)      \
    X(sigma_l)          \
    X(sigma_r)          \
    X(sigma_o)          \
    X(sigma_4)          \
    X(l1_alpha_sq)  


// Auxilary list
// Challenges has 4 elements, curve_params has 2 elements
// Permutation parameters has 3 elements
#define AUX_LIST(X) \
    X(challenges)   \
    X(curve_params) \
    X(perm_params)

#define RANGE_CHALLENGE challenges[0]
#define LOGIC_CHALLENGE challenges[1]
#define FIXED_BASE_CHALLENGE challenges[2]
#define VAR_BASE_CHALLENGE challenges[3]

#define P_COEFF_A curve_params[0]
#define P_COEFF_D curve_params[1]

#define ALPHA perm_params[0]
#define BETA perm_params[1]
#define GAMMA perm_params[2]

// Compose a argument list of following function
#define MAKE_PTR_ARGUMENT(var) , const fr_t* var

// Compose a parameter list of following function
#define MAKE_PARAMETER(var) , var

// Total argument
#define TOTAL_ARGUMENT \
    POINTER_LIST(MAKE_PTR_ARGUMENT) AUX_LIST(MAKE_PTR_ARGUMENT)

#define TOTAL_PARAMETER \
    POINTER_LIST(MAKE_PARAMETER) AUX_LIST(MAKE_PARAMETER)

/*-------------------------GATE SAT---------------------------------------*/
__device__ __forceinline__ fr_t compute_quotient_i(size_t i TOTAL_ARGUMENT)
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

__device__ __forceinline__ fr_t delta(fr_t f)
{
    fr_t f_1 = f - ONE;
    fr_t f_2 = f - TWO;
    fr_t f_3 = f - THREE;
    return f * f_1 * f_2 * f_3; 
}

__device__ __forceinline__ fr_t range_quoteint_term(size_t i TOTAL_ARGUMENT)
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

__device__ __forceinline__ fr_t delta_xor_and(fr_t a, fr_t b, fr_t w, fr_t c, fr_t q_c)
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

__device__ __forceinline__ fr_t logic_quotient_term(size_t i TOTAL_ARGUMENT)
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

// Extracts the bit value from the accumulated bit.
__device__ __forceinline__ fr_t extract_bit(fr_t curr_acc, fr_t next_acc)
{
    return next_acc - curr_acc - curr_acc;
}

/// Ensures that the bit is either `+1`, `-1`, or `0`
__device__ __forceinline__ fr_t check_bit_consistency(fr_t bit)
{
    return bit * (bit - ONE) * (bit + ONE);
}

__device__ __forceinline__ fr_t fixed_base_quoteint_term(size_t i TOTAL_ARGUMENT)
{
    fr_t kappa = SQAURE(FIXED_BASE_CHALLENGE);
    fr_t kappa_sq = SQAURE(kappa);
    fr_t kappa_cu = kappa_sq * kappa;

    fr_t x_beta_eval = q_l[i];
    fr_t y_beta_eval = q_r[i];

    fr_t acc_x = w_l[i];
    fr_t acc_x_next = w_l[i + NEXT];
    fr_t acc_y = w_r[i];
    fr_t acc_y_next = w_r[i + NEXT];

    fr_t xy_alpha = w_o[i];

    fr_t accumulated_bit = w_4[i];
    fr_t accumulated_bit_next = w_4[i + NEXT];
    fr_t bit = extract_bit(accumulated_bit, accumulated_bit_next);

    // Check bit consistency
    fr_t bit_consistency = check_bit_consistency(bit);

    fr_t y_alpha = SQAURE(bit) * (y_beta_eval - ONE) + ONE;
    fr_t x_alpha = x_beta_eval * bit;

    // xy_alpha consistency check
    fr_t xy_consistency = ((bit * q_c[i]) - xy_alpha) * kappa;

    // x accumulator consistency check
    fr_t x_3 = acc_x_next;
    fr_t lhs = x_3 + (x_3 * xy_alpha * acc_x * acc_y * P_COEFF_D);
    fr_t rhs = (x_alpha * acc_y) + (y_alpha * acc_x);
    fr_t x_acc_consistency = (lhs - rhs) * kappa_sq;

    // y accumulator consistency check
    fr_t y_3 = acc_y_next;
    lhs = y_3 - (y_3 * xy_alpha * acc_x * acc_y * P_COEFF_D);
    rhs = y_alpha * acc_y - P_COEFF_A * x_alpha * acc_x;
    fr_t y_acc_consistency = (lhs - rhs) * kappa_cu;

    fr_t checks = bit_consistency
        + x_acc_consistency
        + y_acc_consistency
        + xy_consistency;

    return fbsm_s[i] * checks * FIXED_BASE_CHALLENGE;
}

__device__ __forceinline__ fr_t curve_addition_quotient_term(size_t i TOTAL_ARGUMENT)
{
    fr_t x_1 = w_l[i];
    fr_t x_3 = w_l[i + NEXT];
    fr_t y_1 = w_r[i];
    fr_t y_3 = w_r[i + NEXT];
    fr_t x_2 = w_o[i];
    fr_t y_2 = w_4[i];
    fr_t x1_y2 = w_4[i + NEXT];

    fr_t kappa = SQAURE(VAR_BASE_CHALLENGE);

    // Check that `x1 * y2` is correct
    fr_t xy_consistency = x_1 * y_2 - x1_y2;

    fr_t y1_x2 = y_1 * x_2;
    fr_t y1_y2 = y_1 * y_2;
    fr_t x1_x2 = x_1 * x_2;

    // Check that `x_3` is correct
    fr_t x3_lhs = x1_y2 + y1_x2;
    fr_t x3_rhs = x_3 + (x_3 * P_COEFF_D * x1_y2 * y1_x2);
    fr_t x3_consistency = (x3_lhs - x3_rhs) * kappa;

    // Check that `y_3` is correct
    fr_t y3_lhs = y1_y2 - P_COEFF_A * x1_x2;
    fr_t y3_rhs = y_3 - y_3 * P_COEFF_D * x1_y2 * y1_x2;
    fr_t y3_consistency = (y3_lhs - y3_rhs) * SQAURE(kappa);

    return vgca_s[i] * (xy_consistency + x3_consistency + y3_consistency) * VAR_BASE_CHALLENGE;
}

__device__ __forceinline__ fr_t gate_sat_term(size_t i TOTAL_ARGUMENT)
{
    return compute_quotient_i(i TOTAL_PARAMETER) + 
                range_quoteint_term(i TOTAL_PARAMETER) +
                logic_quotient_term(i TOTAL_PARAMETER) +
                fixed_base_quoteint_term(i TOTAL_PARAMETER) +
                curve_addition_quotient_term(i TOTAL_PARAMETER) +
                pi[i];
}

/*--------------------------------------PERMUTATION--------------------------------------------*/
__device__ __forceinline__ fr_t compute_quotient_identity_range_check_i(size_t i TOTAL_ARGUMENT)
{
    fr_t x = perm_linear[i];
    return (w_l[i] + BETA * x + GAMMA)
        * (w_r[i] + (BETA * K1 * x) + GAMMA)
        * (w_o[i] + (BETA * K2 * x) + GAMMA)
        * (w_4[i] + (BETA * K3 * x) + GAMMA)
        * z[i]
        * ALPHA;
}

__device__ __forceinline__ fr_t compute_quotient_copy_range_check_i(size_t i TOTAL_ARGUMENT)
{
    fr_t left_sigma_eval = sigma_l[i];
    fr_t right_sigma_eval = sigma_r[i];
    fr_t out_sigma_eval = sigma_o[i];
    fr_t fourth_sigma_eval = sigma_4[i];
    fr_t product = (w_l[i] + (BETA * left_sigma_eval) + GAMMA)
        * (w_r[i] + (BETA * right_sigma_eval) + GAMMA)
        * (w_o[i] + (BETA * out_sigma_eval) + GAMMA)
        * (w_4[i] + (BETA * fourth_sigma_eval) + GAMMA)
        * z[i + NEXT]
        * ALPHA;
    return -product;
}

__device__ __forceinline__ fr_t compute_quotient_term_check_one_i(size_t i TOTAL_ARGUMENT)
{
    return (z[i] - ONE) * l1_alpha_sq[i];
}

__device__ __forceinline__ fr_t permutation_term(size i TOTAL_ARGUMENT)
{
    return compute_quotient_identity_range_check_i(i TOTAL_PARAMETER) +
                compute_quotient_copy_range_check_i(i TOTAL_PARAMETER) +
                compute_quotient_term_check_one_i(i TOTAL_PARAMETER);
}

/*----------------------------------FINAL KERNEL FUNCTION---------------------------------------*/
__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void quotient_poly_kernel(const uint lg_domain_size, fr_t* out
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

    out[tid] = gate_sat_term(tid TOTAL_PARAMETER)
               + permutation_term(tid TOTAL_PARAMETER);
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
#undef K1
#undef K2
#undef K3
#undef POW_SBOX
#undef SQAURE
#undef RANGE_CHALLENGE
#undef LOGIC_CHALLENGE
#undef FIXED_BASE_CHALLENGE
#undef VAR_BASE_CHALLENGE
#undef P_COEFF_A
#undef P_COEFF_D
#undef ALPHA
#undef BETA
#undef GAMMA
