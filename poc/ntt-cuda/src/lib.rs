// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

sppark::cuda_error!();

#[repr(C)]
pub enum NTTInputOutputOrder {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[repr(C)]
enum NTTDirection {
    Forward = 0,
    Inverse = 1,
}

#[repr(C)]
enum NTTType {
    Standard = 0,
    Coset = 1,
}

extern "C" {
    fn compute_ntt(
        device_id: usize,
        inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> cuda::Error;
}

extern "C" {
    fn compute_gate_constraint(
          device_id: usize,
          lg_domain_size: u32,
          a: *mut core::ffi::c_void,
          b: *mut core::ffi::c_void,
          c: *mut core::ffi::c_void,
    ) -> cuda::Error;
}

/// Compute an in-place NTT on the input data.
#[allow(non_snake_case)]
pub fn NTT<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            len.trailing_zeros(),
            order,
            NTTDirection::Forward,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

/// Compute an in-place iNTT on the input data.
#[allow(non_snake_case)]
pub fn iNTT<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            len.trailing_zeros(),
            order,
            NTTDirection::Inverse,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

#[allow(non_snake_case)]
pub fn coset_NTT<T>(
    device_id: usize,
    inout: &mut [T],
    order: NTTInputOutputOrder,
) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            len.trailing_zeros(),
            order,
            NTTDirection::Forward,
            NTTType::Coset,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

#[allow(non_snake_case)]
pub fn coset_iNTT<T>(
    device_id: usize,
    inout: &mut [T],
    order: NTTInputOutputOrder,
) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            len.trailing_zeros(),
            order,
            NTTDirection::Inverse,
            NTTType::Coset,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn gate_constraint_sat<T>(device_id: usize, a: &mut [T], b : &mut [T], c : &mut [T]) {
    // First check all the lengths are the same
    let aux = vec![a.len(), b.len(), c.len()];
    let all_same_length = aux.iter().all(|v| *v == aux[0]);
    if !all_same_length {
        panic!("All vectors must have the same length");
    }
    let len = aux[0];
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_gate_constraint(
            device_id,
            len.trailing_zeros(),
            a.as_mut_ptr() as *mut core::ffi::c_void,
            b.as_mut_ptr() as *mut core::ffi::c_void,
            c.as_mut_ptr() as *mut core::ffi::c_void,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}