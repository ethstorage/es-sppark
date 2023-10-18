// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_PALLAS)
# include <ff/pasta.hpp>
#elif defined(FEATURE_VESTA)
# include <ff/pasta.hpp>
#else
# error "no FEATURE"
#endif

#include <ntt/ntt.cuh>
#include <ntt/arithmetic.cuh>

#ifndef __CUDA_ARCH__

extern "C"
RustError compute_ntt(size_t device_id, fr_t* inout, uint32_t lg_domain_size,
                      NTT::InputOutputOrder ntt_order,
                      NTT::Direction ntt_direction,
                      NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    return NTT::Base(gpu, inout, lg_domain_size,
                     ntt_order, ntt_direction, ntt_type);
}

extern "C"
RustError compute_quotient_term(size_t device_id, uint32_t lg_domain_size,
                                  fr_t* out TOTAL_ARGUMENT)
{
    auto& gpu = select_gpu(device_id);

    return ARITHMETIC::quotient_poly_gpu(gpu, lg_domain_size, out TOTAL_PARAMETER);
}
#endif
