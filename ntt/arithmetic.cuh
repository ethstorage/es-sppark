// Copyright Jiazheng Liu, EthStorage
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_NTT_ARITHMETIC_CUH__
#define __SPPARK_NTT_ARITHMETIC_CUH__

#include <cassert>
#include <iostream>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include "parameters.cuh"
#include "kernels.cu"

#ifndef __CUDA_ARCH__

#define MAX_THREAD_SIZE 1024

#define EVAL_LIST(X)  \
    X(w_o)            \
    X(q_l)            \
    X(q_r)            \
    X(q_o)            \
    X(q_4)            \
    X(q_hl)           \
    X(q_hr)           \
    X(q_h4)           \
    X(q_c)            \
    X(q_arith)        \
    X(q_m)            \
    X(r_s)            \
    X(l_s)            \
    X(fbsm_s)         \
    X(vgca_s)         \
    X(pi)             \
    X(perm_linear)    \
    X(sigma_l)        \
    X(sigma_r)        \
    X(sigma_o)        \
    X(sigma_4)        \
    X(q_lookup)       \
    X(f)              \
    X(h2)             \
    X(l1)             \
    X(l1_alpha_sq)    \
    X(v_h_coset)

#define EXTENDED_LIST(X) \
    X(w_l)               \
    X(w_r)               \
    X(w_4)               \
    X(z)                 \
    X(z2)                \
    X(table)             \
    X(h1)

class ARITHMETIC {

public:
    static RustError quotient_poly_gpu(const gpu_t& gpu, size_t domain_size, fr_t* out
                                       TOTAL_ARGUMENT)
    {
        if (domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            // For normal length buffer
#define MAKE_DEV_PTR(var) dev_ptr_t<fr_t> d_##var{domain_size, gpu}; 
#define MAKE_HOST2DEVICE(var) gpu.HtoD(&d_##var[0], var, domain_size);

            EVAL_LIST(MAKE_DEV_PTR);
            EVAL_LIST(MAKE_HOST2DEVICE);

            // For extended length buffer
#define MAKE_DEV_PTR_EXTEND(var) dev_ptr_t<fr_t> d_##var{domain_size+8, gpu}; 
#define MAKE_HOST2DEVICE_EXTEND(var) gpu.HtoD(&d_##var[0], var, domain_size+8);

            EXTENDED_LIST(MAKE_DEV_PTR_EXTEND);
            EXTENDED_LIST(MAKE_HOST2DEVICE_EXTEND);

            // We only have 5 challenges
            // Because it is associated with Macro, if you want to change it name
            // change the macro as well
            dev_ptr_t<fr_t> d_challenges{5, gpu};
            gpu.HtoD(&d_challenges[0], challenges, 5);

            // We have two curve parameters
            dev_ptr_t<fr_t> d_curve_params{2, gpu};
            gpu.HtoD(&d_curve_params[0], curve_params, 2);

            // 6 permutation parameters
            dev_ptr_t<fr_t> d_perm_params{6, gpu};
            gpu.HtoD(&d_perm_params[0], perm_params, 6);

            dev_ptr_t<fr_t> d_out{domain_size, gpu};

            // First check if it could be stored inside one block
            size_t thread_size = domain_size <= MAX_THREAD_SIZE ? domain_size : MAX_THREAD_SIZE;
            size_t block_size = (domain_size + thread_size - 1) / thread_size;

#define MAKE_KERNEL_PARAMETER(var) , d_##var
            quotient_poly_kernel<<<block_size, thread_size, 0, gpu>>>(domain_size, d_out POINTER_LIST(MAKE_KERNEL_PARAMETER)
                                                                            AUX_LIST(MAKE_KERNEL_PARAMETER));

            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                auto name = cudaGetErrorString(err);
                std::cerr << "Error: " << name << std::endl;
                throw cuda_error{err};
            }
            // Only needs to sync var `out`
            gpu.DtoH(out, &d_out[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

public: 
    static RustError product_argument_gpu(const gpu_t& gpu, uint32_t lg_domain_size, fr_t* out PRODUCT_ARGUMENT) {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;

#define MAKE_DEV_PTR(var) dev_ptr_t<fr_t> d_##var{domain_size, gpu};
#define MAKE_HOST2DEVICE(var) gpu.HtoD(&d_##var[0], var, domain_size);
#define MAKE_KERNEL_PARAMETER(var) , d_##var

            PRODUCT_POINTER_LIST(MAKE_DEV_PTR);
            PRODUCT_POINTER_LIST(MAKE_HOST2DEVICE);

            // We only have 5 challenges
            // Because it is associated with Macro, if you want to change it name
            // change the macro as well
            dev_ptr_t<fr_t> d_ks{4, gpu};
            gpu.HtoD(&d_ks[0], ks, 4);

            // We have two curve parameters
            dev_ptr_t<fr_t> d_beta{1, gpu};
            gpu.HtoD(&d_beta[0], beta, 1);

            // 6 permutation parameters
            dev_ptr_t<fr_t> d_gamma{1, gpu};
            gpu.HtoD(&d_gamma[0], gamma, 1);

            dev_ptr_t<fr_t> d_out{domain_size, gpu};

            // First check if it could be stored inside one block
            size_t thread_size = domain_size <= MAX_THREAD_SIZE ? domain_size : MAX_THREAD_SIZE;
            size_t block_size = (domain_size + thread_size - 1) / thread_size;

            product_argment_kernel<<<block_size, thread_size, 0, gpu>>>(
                lg_domain_size, d_out PRODUCT_POINTER_LIST(MAKE_KERNEL_PARAMETER) PRODUCT_AUX_LIST(MAKE_KERNEL_PARAMETER));

            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                auto name = cudaGetErrorString(err);
                std::cerr << "Error: " << name << std::endl;
                throw cuda_error{err};
            }
            // Only needs to sync var `out`
            gpu.DtoH(out, &d_out[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

public: 
    static RustError lookup_product_argument_gpu(const gpu_t& gpu, uint32_t lg_domain_size, fr_t* out LOOKUP_PRODUCT_ARGUMENT) {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;

#define MAKE_DEV_PTR(var) dev_ptr_t<fr_t> d_##var{domain_size, gpu};
#define MAKE_HOST2DEVICE(var) gpu.HtoD(&d_##var[0], var, domain_size);
#define MAKE_KERNEL_PARAMETER(var) , d_##var

            LOOKUP_PRODUCT_POINTER_LIST(MAKE_DEV_PTR);
            LOOKUP_PRODUCT_POINTER_LIST(MAKE_HOST2DEVICE);

            // We only have 5 challenges
            // Because it is associated with Macro, if you want to change it name
            // change the macro as well
            dev_ptr_t<fr_t> d_delta{1, gpu};
            gpu.HtoD(&d_delta[0], delta, 1);

            // We have two curve parameters
            dev_ptr_t<fr_t> d_epsilon{1, gpu};
            gpu.HtoD(&d_epsilon[0], epsilon, 1);

            dev_ptr_t<fr_t> d_out{domain_size, gpu};

            // First check if it could be stored inside one block
            size_t thread_size = domain_size <= MAX_THREAD_SIZE ? domain_size : MAX_THREAD_SIZE;
            size_t block_size = (domain_size + thread_size - 1) / thread_size;

            lookup_product_argment_kernel<<<block_size, thread_size, 0, gpu>>>(
                lg_domain_size, d_out LOOKUP_PRODUCT_POINTER_LIST(MAKE_KERNEL_PARAMETER) LOOKUP_PRODUCT_AUX_LIST(MAKE_KERNEL_PARAMETER));

            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                auto name = cudaGetErrorString(err);
                std::cerr << "Error: " << name << std::endl;
                throw cuda_error{err};
            }
            // Only needs to sync var `out`
            gpu.DtoH(out, &d_out[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

public: 
    static RustError linear_poly_gpu(const gpu_t& gpu, uint32_t lg_domain_size, fr_t* out LINEAR_POLY_ARGUMENT) {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;

#define MAKE_DEV_PTR(var) dev_ptr_t<fr_t> d_##var{domain_size, gpu};
#define MAKE_HOST2DEVICE(var) gpu.HtoD(&d_##var[0], var, domain_size);

#define MAKE_KERNEL_PARAMETER(var) , d_##var

            LINEAR_POLY_POINTER_LIST(MAKE_DEV_PTR);
            LINEAR_POLY_POINTER_LIST(MAKE_HOST2DEVICE);

            // 5 wit_vals
            // dev_ptr_t<fr_t> d_challenges{4, gpu};
            // gpu.HtoD(&d_challenges[0], challenges, 4);
            // dev_ptr_t<fr_t> d_custom_evals{9, gpu};
            // gpu.HtoD(&d_custom_evals[0], custom_evals, 9);
            dev_ptr_t<fr_t> d_wit_vals{5, gpu};
            gpu.HtoD(&d_wit_vals[0], wit_vals, 5);

            dev_ptr_t<fr_t> d_out{domain_size, gpu};

            // First check if it could be stored inside one block
            size_t thread_size = domain_size <= MAX_THREAD_SIZE ? domain_size : MAX_THREAD_SIZE;
            size_t block_size = (domain_size + thread_size - 1) / thread_size;

            linear_poly_kernel<<<block_size, thread_size, 0, gpu>>>(
                lg_domain_size, d_out LINEAR_POLY_POINTER_LIST(MAKE_KERNEL_PARAMETER) LINEAR_POLY_AUX_LIST(MAKE_KERNEL_PARAMETER));

            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                auto name = cudaGetErrorString(err);
                std::cerr << "Error: " << name << std::endl;
                throw cuda_error{err};
            }
            // Only needs to sync var `out`
            gpu.DtoH(out, &d_out[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

};

#undef MAX_THREAD_SIZE
#undef MAKE_DEV_PTR
#undef MAKE_HOST2DEVICE
#undef MAKE_KERNEL_PARAMETER
#undef MAKE_DEV_PTR_EXTEND
#undef MAKE_HOST2DEVICE_EXTEND
#undef EVAL_LIST
#undef EXTENDED_LIST
#endif
#endif