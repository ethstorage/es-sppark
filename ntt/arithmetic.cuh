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

class ARITHMETIC {

public:
    static RustError quotient_poly_gpu(const gpu_t& gpu, uint32_t lg_domain_size, fr_t* out
                                     TOTAL_ARGUMENT)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;

#define MAKE_DEV_PTR(var) dev_ptr_t<fr_t> d_##var{domain_size, gpu};
#define MAKE_HOST2DEVICE(var) gpu.HtoD(&d_##var[0], var, domain_size);
#define MAKE_KERNEL_PARAMETER(var) , d_##var

            POINTER_LIST(MAKE_DEV_PTR);
            POINTER_LIST(MAKE_HOST2DEVICE);

            // We only have 4 challenges
            // Because it is associated with Macro, if you want to change it name
            // change the macro as well
            dev_ptr_t<fr_t> d_challenges{4, gpu};
            gpu.HtoD(&d_challenges[0], challenges, 4);

            // We have two curve parameters
            dev_ptr_t<fr_t> d_curve_params{2, gpu};
            gpu.HtoD(&d_curve_params[0], curve_params, 2);

            // Three permutation parameters
            dev_ptr_t<fr_t> d_perm_params{3, gpu};
            gpu.HtoD(&d_perm_params[0], perm_params, 3);

            dev_ptr_t<fr_t> d_out{domain_size, gpu};

            // First check if it could be stored inside one block
            size_t thread_size = domain_size <= MAX_THREAD_SIZE ? domain_size : MAX_THREAD_SIZE;
            size_t block_size = (domain_size + thread_size - 1) / thread_size;

            quotient_poly_kernel<<<block_size, thread_size, 0, gpu>>>(lg_domain_size, d_out POINTER_LIST(MAKE_KERNEL_PARAMETER)
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
};

#undef MAX_THREAD_SIZE
#undef MAKE_DEV_PTR
#undef MAKE_HOST2DEVICE
#undef MAKE_KERNEL_PARAMETER
#endif
#endif