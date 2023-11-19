// Copyright Jiazheng Liu, EthStorage
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Some constant macro only be used in this file
#define MAX_THREAD_NUM 1024
// 8 is the row of gate constraint, because we extend the domain by 8
// NOTE: The caller must make sure the buffer is at length of domain_size + 8
//       Otherwise it would cause out of bound error
#define NEXT(index, domain_size) (index + 8)
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
    X(q_lookup)         \
    X(table)            \
    X(f)                \
    X(h1)               \
    X(h2)               \
    X(z2)               \
    X(l1)               \
    X(l1_alpha_sq)      \
    X(v_h_coset) 


// Auxilary list
// Challenges has 5 elements, curve_params has 2 elements
// Permutation parameters has 6 elements
#define AUX_LIST(X) \
    X(challenges)   \
    X(curve_params) \
    X(perm_params)

#define RANGE_CHALLENGE challenges[0]
#define LOGIC_CHALLENGE challenges[1]
#define FIXED_BASE_CHALLENGE challenges[2]
#define VAR_BASE_CHALLENGE challenges[3]
#define LOOKUP_CHALLENGE challenges[4]

#define P_COEFF_A curve_params[0]
#define P_COEFF_D curve_params[1]

#define ALPHA perm_params[0]
#define BETA perm_params[1]
#define GAMMA perm_params[2]
#define DELTA perm_params[3]
#define EPSILON perm_params[4]
#define ZETA perm_params[5]

// Compose a argument list of following function
#define MAKE_PTR_ARGUMENT(var) , const fr_t* var

// Compose a parameter list of following function
#define MAKE_PARAMETER(var) , var

// Total argument
#define TOTAL_ARGUMENT \
    POINTER_LIST(MAKE_PTR_ARGUMENT) AUX_LIST(MAKE_PTR_ARGUMENT)

#define TOTAL_PARAMETER \
    POINTER_LIST(MAKE_PARAMETER) AUX_LIST(MAKE_PARAMETER)

#define PRODUCT_POINTER_LIST(X) \
    X(root) \
    X(gate_sigma0) \
    X(gate_sigma1) \
    X(gate_sigma2) \
    X(gate_sigma3) \
    X(gate_wire0) \
    X(gate_wire1) \
    X(gate_wire2) \
    X(gate_wire3) \

#define PRODUCT_AUX_LIST(X) \
    X(ks) \
    X(beta) \
    X(gamma) \

#define PRODUCT_ARGUMENT \
    PRODUCT_POINTER_LIST(MAKE_PTR_ARGUMENT) PRODUCT_AUX_LIST(MAKE_PTR_ARGUMENT)

#define PRODUCT_PARAMETER \
    PRODUCT_POINTER_LIST(MAKE_PARAMETER) PRODUCT_AUX_LIST(MAKE_PARAMETER)

#define LOOKUP_PRODUCT_POINTER_LIST(X) \
    X(f) \
    X(t) \
    X(t_next) \
    X(h_1) \
    X(h_1_next) \
    X(h_2) \

#define LOOKUP_PRODUCT_AUX_LIST(X) \
    X(delta) \
    X(epsilon) \

#define LOOKUP_PRODUCT_ARGUMENT \
    LOOKUP_PRODUCT_POINTER_LIST(MAKE_PTR_ARGUMENT) LOOKUP_PRODUCT_AUX_LIST(MAKE_PTR_ARGUMENT)

#define LOOKUP_PRODUCT_PARAMETER \
    LOOKUP_PRODUCT_POINTER_LIST(MAKE_PARAMETER) LOOKUP_PRODUCT_AUX_LIST(MAKE_PARAMETER)

#define LINEAR_POLY_POINTER_LIST(X) \
    X(q_m) \
    X(q_l) \
    X(q_r) \
    X(q_o) \
    X(q_4) \
    X(q_hl) \
    X(q_hr) \
    X(q_h4) \
    X(q_c) \
    X(z_poly) \
    X(fourth_sigma) \
    X(t_1_poly) \
    X(t_2_poly) \
    X(t_3_poly) \
    X(t_4_poly) \
    X(t_5_poly) \
    X(t_6_poly) \
    X(t_7_poly) \
    X(t_8_poly) \

#define LINEAR_POLY_AUX_LIST(X) \
    X(wit_vals) \
    X(perm_vals) \

#define LINEAR_POLY_ARGUMENT \
    LINEAR_POLY_POINTER_LIST(MAKE_PTR_ARGUMENT) LINEAR_POLY_AUX_LIST(MAKE_PTR_ARGUMENT), const uint64_t* power 

#define LINEAR_POLY_PARAMETER \
    LINEAR_POLY_POINTER_LIST(MAKE_PARAMETER) LINEAR_POLY_AUX_LIST(MAKE_PARAMETER), power

/*-------------------------GATE SAT---------------------------------------*/
__device__ __forceinline__ fr_t compute_quotient_i(size_t i, size_t domain_size TOTAL_ARGUMENT)
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

__device__ __forceinline__ fr_t range_quoteint_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
   fr_t kappa = RANGE_CHALLENGE * RANGE_CHALLENGE;
   fr_t kappa_sq = kappa * kappa;
   fr_t kappa_cu = kappa_sq * kappa;
   fr_t b1 = delta(w_o[i] - FOUR * w_4[i]);
   fr_t b2 = delta(w_r[i] - FOUR * w_o[i]) * kappa;
   fr_t b3 = delta(w_l[i] - FOUR * w_r[i]) * kappa_sq;
   // NOTICE: w_4 is next one, should add next line
   fr_t b4 = delta(w_4[NEXT(i, domain_size)] - FOUR * w_l[i]) * kappa_cu;

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

__device__ __forceinline__ fr_t logic_quotient_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    fr_t kappa = RANGE_CHALLENGE * RANGE_CHALLENGE;
    fr_t kappa_sq = kappa * kappa;
    fr_t kappa_cu = kappa_sq * kappa;
    fr_t kappa_qu = kappa_cu * kappa;

    fr_t a = w_l[NEXT(i, domain_size)] - FOUR * w_l[i];
    fr_t c_0 = delta(a);

    fr_t b = w_r[NEXT(i, domain_size)] - FOUR * w_r[i];
    fr_t c_1 = delta(b) * kappa;

    fr_t d = w_4[NEXT(i, domain_size)] - FOUR * w_4[i];
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

__device__ __forceinline__ fr_t fixed_base_quoteint_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    fr_t kappa = SQAURE(FIXED_BASE_CHALLENGE);
    fr_t kappa_sq = SQAURE(kappa);
    fr_t kappa_cu = kappa_sq * kappa;

    fr_t x_beta_eval = q_l[i];
    fr_t y_beta_eval = q_r[i];

    fr_t acc_x = w_l[i];
    fr_t acc_x_next = w_l[NEXT(i, domain_size)];
    fr_t acc_y = w_r[i];
    fr_t acc_y_next = w_r[NEXT(i, domain_size)];

    fr_t xy_alpha = w_o[i];

    fr_t accumulated_bit = w_4[i];
    fr_t accumulated_bit_next = w_4[NEXT(i, domain_size)];
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

__device__ __forceinline__ fr_t curve_addition_quotient_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    fr_t x_1 = w_l[i];
    fr_t x_3 = w_l[NEXT(i, domain_size)];
    fr_t y_1 = w_r[i];
    fr_t y_3 = w_r[NEXT(i, domain_size)];
    fr_t x_2 = w_o[i];
    fr_t y_2 = w_4[i];
    fr_t x1_y2 = w_4[NEXT(i, domain_size)];

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

__device__ __forceinline__ fr_t gate_sat_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    return compute_quotient_i(i, domain_size TOTAL_PARAMETER) + 
                range_quoteint_term(i, domain_size TOTAL_PARAMETER) +
                logic_quotient_term(i, domain_size TOTAL_PARAMETER) +
                fixed_base_quoteint_term(i, domain_size TOTAL_PARAMETER) +
                curve_addition_quotient_term(i, domain_size TOTAL_PARAMETER) +
                pi[i];
}

/*--------------------------------------PERMUTATION--------------------------------------------*/
__device__ __forceinline__ fr_t compute_quotient_identity_range_check_i(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    fr_t x = perm_linear[i];
    return (w_l[i] + BETA * x + GAMMA)
        * (w_r[i] + (BETA * K1 * x) + GAMMA)
        * (w_o[i] + (BETA * K2 * x) + GAMMA)
        * (w_4[i] + (BETA * K3 * x) + GAMMA)
        * z[i]
        * ALPHA;
}

__device__ __forceinline__ fr_t compute_quotient_copy_range_check_i(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    fr_t left_sigma_eval = sigma_l[i];
    fr_t right_sigma_eval = sigma_r[i];
    fr_t out_sigma_eval = sigma_o[i];
    fr_t fourth_sigma_eval = sigma_4[i];
    fr_t product = (w_l[i] + (BETA * left_sigma_eval) + GAMMA)
        * (w_r[i] + (BETA * right_sigma_eval) + GAMMA)
        * (w_o[i] + (BETA * out_sigma_eval) + GAMMA)
        * (w_4[i] + (BETA * fourth_sigma_eval) + GAMMA)
        * z[NEXT(i, domain_size)]
        * ALPHA;
    return product; 
}

__device__ __forceinline__ fr_t compute_quotient_term_check_one_i(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    return (z[i] - ONE) * l1_alpha_sq[i];
}

__device__ __forceinline__ fr_t permutation_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
    return compute_quotient_identity_range_check_i(i, domain_size TOTAL_PARAMETER)
           - compute_quotient_copy_range_check_i(i, domain_size TOTAL_PARAMETER)
           + compute_quotient_term_check_one_i(i, domain_size TOTAL_PARAMETER);
}

/*---------------------------------------LOOKUP-------------------------------------------------*/
// Linear combination of a series of values
__device__ __forceinline__ fr_t lc(const fr_t* coeffs, fr_t x, size_t n)
{
    // Horner's method
    fr_t acc = coeffs[n];
    for (size_t i = 1; i < n; ++i) {
        acc = acc * x + coeffs[n - i];
    }
    return acc;
}

__device__ __forceinline__ fr_t lookup_term(size_t i, size_t domain_size TOTAL_ARGUMENT)
{
   
    fr_t lookup_sep_sq = SQAURE(LOOKUP_CHALLENGE);
    fr_t lookup_sep_cu = lookup_sep_sq * LOOKUP_CHALLENGE;
    fr_t one_plus_delta = DELTA + ONE;
    fr_t epsilon_one_plus_delta = EPSILON * one_plus_delta;

    // q_lookup(X) * (a(X) + zeta * b(X) + (zeta^2 * c(X)) + (zeta^3 * d(X)
    // - f(X))) * α_1
    fr_t wit[4] = {w_l[i], w_r[i], w_o[i], w_4[i]};
    fr_t compressed_tuple = lc(wit, ZETA, 4);
    fr_t a = q_lookup[i] * (compressed_tuple - f[i]) * LOOKUP_CHALLENGE;

    // z2(X) * (1+δ) * (ε+f(X)) * (ε*(1+δ) + t(X) + δt(Xω)) * lookup_sep^2
    fr_t b_0 = EPSILON + f[i];
    fr_t b_1 = epsilon_one_plus_delta + table[i] + DELTA * table[NEXT(i, domain_size)];
    fr_t b = z2[i] * one_plus_delta * b_0 * b_1 * lookup_sep_sq;

    // − z2(Xω) * (ε*(1+δ) + h1(X) + δ*h2(X)) * (ε*(1+δ) + h2(X) + δ*h1(Xω))
    // * lookup_sep^2
    fr_t c_0 = epsilon_one_plus_delta + h1[i] + DELTA * h2[i];
    fr_t c_1 = epsilon_one_plus_delta + h2[i] + DELTA * h1[NEXT(i, domain_size)];
    fr_t c = z2[NEXT(i, domain_size)] * c_0 * c_1 * lookup_sep_sq;

    fr_t d = (z2[i] - ONE) * l1[i] * lookup_sep_cu;

    return a + b - c + d;
}

/*--------------------------------------PRODUCT ARGUMENT--------------------------------------------*/
__device__ __forceinline__ fr_t product_argment(size_t i, size_t domain_size PRODUCT_ARGUMENT) {
    fr_t _beta = beta[0];
    fr_t _gamma = gamma[0];
    fr_t numerator_product = (gate_wire0[i] + _beta * ks[0] * root[i] + _gamma)
        * (gate_wire1[i] + _beta * ks[1] * root[i] + _gamma) 
        * (gate_wire2[i] + _beta * ks[2] * root[i] + _gamma)
        * (gate_wire3[i] + _beta * ks[3] * root[i] + _gamma); 
    fr_t denominator_product = (gate_wire0[i] + _beta * gate_sigma0[i] + _gamma)
        * (gate_wire1[i] + _beta * gate_sigma1[i] + _gamma)
        * (gate_wire2[i] + _beta * gate_sigma2[i] + _gamma)
        * (gate_wire3[i] + _beta * gate_sigma3[i] + _gamma);
    return numerator_product / denominator_product;
}

/*--------------------------------------LOOKUP PRODUCT ARGUMENT--------------------------------------------*/
__device__ __forceinline__ fr_t lookup_product_argment(size_t i, size_t domain_size LOOKUP_PRODUCT_ARGUMENT) {
    fr_t _epsilon = epsilon[0];
    fr_t _delta = delta[0];
    fr_t one_plus_delta = _delta + ONE;
    fr_t epsilon_one_plus_delta = _epsilon * one_plus_delta;
    fr_t part_1 = one_plus_delta * (_epsilon + f[i]) 
        * (epsilon_one_plus_delta + t[i] + _delta * t_next[i]);
    fr_t part_2 = ((epsilon_one_plus_delta + h_1[i] + _delta * h_2[i])
        * (epsilon_one_plus_delta + h_2[i] + _delta * h_1_next[i])).reciprocal();
    return part_1 * part_2;
}

/*--------------------------------------LINEAR POLY: linear_poly_arithmetic--------------------------------------------*/
__device__ __forceinline__ fr_t linear_poly_arithmetic(size_t i, size_t domain_size LINEAR_POLY_ARGUMENT) {
    fr_t a_eval = wit_vals[0];
    fr_t b_eval = wit_vals[1];
    fr_t c_eval = wit_vals[2];
    fr_t d_eval = wit_vals[3];
    fr_t q_arith_eval = wit_vals[4];
    fr_t result = (
        q_m[i] * a_eval * b_eval
        + q_l[i] * a_eval 
        + q_r[i] * b_eval
        + q_o[i] * c_eval
        + q_4[i] * d_eval
        + q_hl[i] * POW_SBOX(a_eval)
        + q_hr[i] * POW_SBOX(b_eval)
        + q_h4[i] * POW_SBOX(d_eval)
        + q_c[i]
    ) * q_arith_eval;
    return result;
}

/*--------------------------------------LINEAR POLY: compute_lineariser_identity_range_check---------------------------------*/
__device__ __forceinline__ fr_t compute_lineariser_identity_range_check(size_t i, size_t domain_size LINEAR_POLY_ARGUMENT) {
    fr_t a_eval = wit_vals[0];
    fr_t b_eval = wit_vals[1];
    fr_t c_eval = wit_vals[2];
    fr_t d_eval = wit_vals[3];
    fr_t z_challenge = perm_vals[1];
    fr_t alpha = perm_vals[2];
    fr_t beta = perm_vals[3];
    fr_t gamma = perm_vals[4];

    fr_t beta_z = beta * z_challenge;
    // a_eval + beta * z_challenge + gamma
    fr_t a_0 = a_eval + beta_z;
    a_0 += gamma;

    // b_eval + beta * K1 * z_challenge + gamma
    fr_t beta_z_k1 = K1 * beta_z;
    fr_t a_1 = b_eval + beta_z_k1;
    a_1 += gamma;

    // c_eval + beta * K2 * z_challenge + gamma
    fr_t beta_z_k2 = K2 * beta_z;
    fr_t a_2 = c_eval + beta_z_k2;
    a_2 += gamma;

    // d_eval + beta * K3 * z_challenge + gamma
    fr_t beta_z_k3 = K3 * beta_z;
    fr_t a_3 = d_eval + beta_z_k3;
    a_3 += gamma;

    fr_t a = a_0 * a_1;
    a *= a_2;
    a *= a_3;
    a *= alpha; 

    return z_poly[i] * a;
}

/*--------------------------------------LINEAR POLY: compute_lineariser_copy_range_check---------------------------------*/
__device__ __forceinline__ fr_t compute_lineariser_copy_range_check(size_t i, size_t domain_size LINEAR_POLY_ARGUMENT) {
    fr_t a_eval = wit_vals[0];
    fr_t b_eval = wit_vals[1];
    fr_t c_eval = wit_vals[2];
    fr_t alpha = perm_vals[2];
    fr_t beta = perm_vals[3];
    fr_t gamma = perm_vals[4];
    fr_t sigma_1_eval = perm_vals[5];
    fr_t sigma_2_eval = perm_vals[6];
    fr_t sigma_3_eval = perm_vals[7];
    fr_t z_eval = perm_vals[8];

    // a_eval + beta * sigma_1 + gamma
    fr_t beta_sigma_1 = beta * sigma_1_eval;
    fr_t a_0 = a_eval + beta_sigma_1;
    a_0 += gamma;

    // b_eval + beta * sigma_2 + gamma
    fr_t beta_sigma_2 = beta * sigma_2_eval;
    fr_t a_1 = b_eval + beta_sigma_2;
    a_1 += gamma;

    // c_eval + beta * sigma_3 + gamma
    fr_t beta_sigma_3 = beta * sigma_3_eval;
    fr_t a_2 = c_eval + beta_sigma_3;
    a_2 += gamma;

    fr_t beta_z_eval = beta * z_eval;

    fr_t a = a_0 * a_1 * a_2;
    a *= beta_z_eval;
    a *= alpha; // (a_eval + beta * sigma_1 + gamma)(b_eval + beta * sigma_2 +
                // gamma)(c_eval + beta * sigma_3 + gamma) * beta * z_eval * alpha
    fr_t result = fourth_sigma[i] * a;
    // to negate a Fr
    return result.cneg(true);
}

/*--------------------------------------LINEAR POLY: compute_lineariser_check_is_one---------------------------------*/
__device__ __forceinline__ fr_t compute_lineariser_check_is_one(size_t i, size_t domain_size LINEAR_POLY_ARGUMENT) {
    fr_t a_eval = wit_vals[0];
    fr_t z_challenge = perm_vals[1];
    fr_t alpha = perm_vals[2];
    fr_t beta = perm_vals[3];
    fr_t gamma = perm_vals[4];
    fr_t sigma_1_eval = perm_vals[5];
    fr_t alpha_sq = SQAURE(alpha);

    // a_eval + beta * sigma_1 + gamma
    fr_t beta_sigma_1 = beta * sigma_1_eval;
    fr_t a_0 = a_eval + beta_sigma_1;
    a_0 += gamma;

    fr_t m = perm_vals[0]; 
    fr_t h = ONE;
    fr_t v_0_inv = m;
    const uint64_t p = power[0];
    // There is no explicit arithmetic precedence, so a brack must used for ^
    fr_t l_1_z = ((z_challenge^p) - h) /v_0_inv / (z_challenge - h);
    return z_poly[i] * (l_1_z * alpha_sq);
}

/*--------------------------------------LINEAR POLY: compute_quotient_tem---------------------------------*/
__device__ __forceinline__ fr_t compute_quotient_tem(size_t i, size_t domain_size LINEAR_POLY_ARGUMENT) {
    fr_t z_challenge_to_n = perm_vals[9];
    fr_t vanishing_poly_eval = perm_vals[10];
    fr_t z_2 = SQAURE(z_challenge_to_n);
    fr_t z_3 = z_2 * z_challenge_to_n;
    fr_t z_4 = z_3 * z_challenge_to_n;
    fr_t z_5 = z_4 * z_challenge_to_n;
    fr_t z_6 = z_5 * z_challenge_to_n;
    fr_t z_7 = z_6 * z_challenge_to_n;
    fr_t result = (t_1_poly[i] 
        + t_2_poly[i]*z_challenge_to_n 
        + t_3_poly[i]*z_2 
        + t_4_poly[i]*z_3
        + t_5_poly[i]*z_4
        + t_6_poly[i]*z_5
        + t_7_poly[i]*z_6
        + t_8_poly[i]*z_7
        ) * vanishing_poly_eval;
    // to negate a Fr
    return result.cneg(true);
}


/*----------------------------------FINAL KERNEL FUNCTION---------------------------------------*/
__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void quotient_poly_kernel(const size_t domain_size, fr_t* out
                                TOTAL_ARGUMENT)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(domain_size <= (1 << MAX_LG_DOMAIN_SIZE));
#endif
    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    // out of range, just return
    if (tid > domain_size) {
        return;
    }

    fr_t numerator =   gate_sat_term(tid, domain_size TOTAL_PARAMETER)
                     + permutation_term(tid, domain_size TOTAL_PARAMETER)
                     + lookup_term(tid, domain_size TOTAL_PARAMETER);
    out[tid] = numerator / v_h_coset[tid];
    
}

__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void product_argment_kernel(const uint lg_domain_size, fr_t* out
                                PRODUCT_ARGUMENT)
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

    out[tid] =  product_argment(tid, domain_size PRODUCT_PARAMETER);
}

__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void lookup_product_argment_kernel(const uint lg_domain_size, fr_t* out
                                LOOKUP_PRODUCT_ARGUMENT)
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

    out[tid] =  lookup_product_argment(tid, domain_size LOOKUP_PRODUCT_PARAMETER);
}

__launch_bounds__(MAX_THREAD_NUM, 1) __global__
void linear_poly_kernel(const uint lg_domain_size, fr_t* out
                                LINEAR_POLY_ARGUMENT)
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
    out[tid] = linear_poly_arithmetic(tid, domain_size LINEAR_POLY_PARAMETER)
        + compute_lineariser_identity_range_check(tid, domain_size LINEAR_POLY_PARAMETER)
        + compute_lineariser_copy_range_check(tid, domain_size LINEAR_POLY_PARAMETER)
        + compute_lineariser_check_is_one(tid, domain_size LINEAR_POLY_PARAMETER)
        + compute_quotient_tem(tid, domain_size LINEAR_POLY_PARAMETER);
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
#undef LOOKUP_CHALLENGE
#undef P_COEFF_A
#undef P_COEFF_D
#undef ALPHA
#undef BETA
#undef GAMMA
#undef DELTA
#undef EPSILON
#undef ZETA