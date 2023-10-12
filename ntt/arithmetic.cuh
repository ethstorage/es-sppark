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

#define GATE_CONSTRAINT_THREAD_SIZE 1024

class ARITHMETIC {

public:
    static RustError gate_constraint(const gpu_t& gpu, uint32_t lg_domain_size,
                                      fr_t* a, fr_t* b, fr_t* c)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_a{domain_size, gpu};
            dev_ptr_t<fr_t> d_b{domain_size, gpu};
            dev_ptr_t<fr_t> d_c{domain_size, gpu};
            gpu.HtoD(&d_a[0], a, domain_size);
            gpu.HtoD(&d_b[0], b, domain_size);
            gpu.HtoD(&d_c[0], c, domain_size);

            // First check if it could be stored inside one block
            size_t thread_size = domain_size <= GATE_CONSTRAINT_THREAD_SIZE ? domain_size : GATE_CONSTRAINT_THREAD_SIZE;
            size_t block_size = (domain_size + thread_size - 1) / thread_size;

            gate_constraint_sat_kernel<<<block_size, thread_size, 0, gpu>>>(lg_domain_size, d_a, d_b, d_c);

            gpu.DtoH(c, &d_c[0], domain_size);
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

#undef GATE_CONSTRAINT_THREAD_SIZE
#endif
#endif