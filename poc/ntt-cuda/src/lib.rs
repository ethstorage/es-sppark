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
    fn compute_quotient_term(
        device_id: usize,
        lg_domain_size: u32,
        out: *mut core::ffi::c_void,
        w_l: *const core::ffi::c_void,
        w_r: *const core::ffi::c_void,
        w_o: *const core::ffi::c_void,
        w_4: *const core::ffi::c_void,
        q_l: *const core::ffi::c_void,
        q_r: *const core::ffi::c_void,
        q_o: *const core::ffi::c_void,
        q_4: *const core::ffi::c_void,
        q_hl: *const core::ffi::c_void,
        q_hr: *const core::ffi::c_void,
        q_h4: *const core::ffi::c_void,
        q_c: *const core::ffi::c_void,
        q_arith: *const core::ffi::c_void,
        q_m: *const core::ffi::c_void,
        r_s: *const core::ffi::c_void,
        l_s: *const core::ffi::c_void,
        fbsm_s: *const core::ffi::c_void,
        vgca_s: *const core::ffi::c_void,
        pi: *const core::ffi::c_void,
        z: *const core::ffi::c_void,
        perm_linear: *const core::ffi::c_void,
        sigma_l: *const core::ffi::c_void,
        sigma_r: *const core::ffi::c_void,
        sigma_o: *const core::ffi::c_void,
        sigma_4: *const core::ffi::c_void,
        q_lookup: *const core::ffi::c_void,
        table: *const core::ffi::c_void,
        f: *const core::ffi::c_void,
        h1: *const core::ffi::c_void,
        h2: *const core::ffi::c_void,
        z2: *const core::ffi::c_void,
        l1: *const core::ffi::c_void,
        l1_alpha_sq: *const core::ffi::c_void,
        challenges: *const core::ffi::c_void,
        curve_params: *const core::ffi::c_void,
        perm_params: *const core::ffi::c_void,
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

pub fn quotient_term_gpu<T>(
    device_id: usize,
    out: &mut [T],
    w_l: &[T],
    w_r: &[T],
    w_o: &[T],
    w_4: &[T],
    q_l: &[T],
    q_r: &[T],
    q_o: &[T],
    q_4: &[T],
    q_hl: &[T],
    q_hr: &[T],
    q_h4: &[T],
    q_c: &[T],
    q_arith: &[T],
    q_m: &[T],
    r_s: &[T],
    l_s: &[T],
    fbms_s: &[T],
    vgca_s: &[T],
    pi: &[T],
    z: &[T],
    perm_linear: &[T],
    sigma_l: &[T],
    sigma_r: &[T],
    sigma_o: &[T],
    sigma_4: &[T],
    q_lookup: &[T],
    table: &[T],
    f: &[T],
    h1: &[T],
    h2: &[T],
    z2: &[T],
    l1: &[T],
    l1_alpha_sq: &[T],
    challenges: &[T],
    curve_parameters: &[T],
    perm_parameters: &[T],
) {
    // First check whether majority of the vectors have the same length
    // except for w_l w_r and W_4, they are longer than the rest
    let aux = vec![
        out.len(),
        w_o.len(),
        q_l.len(),
        q_r.len(),
        q_o.len(),
        q_4.len(),
        q_hl.len(),
        q_hr.len(),
        q_h4.len(),
        q_c.len(),
        q_arith.len(),
        q_m.len(),
        r_s.len(),
        l_s.len(),
        fbms_s.len(),
        vgca_s.len(),
        pi.len(),
        perm_linear.len(),
        sigma_l.len(),
        sigma_r.len(),
        sigma_o.len(),
        sigma_4.len(),
        q_lookup.len(),
        f.len(),
        h2.len(),
        l1.len(),
        l1_alpha_sq.len(),
    ];
    let all_same_length = aux.iter().all(|v| *v == aux[0]);
    if !all_same_length {
        panic!("8n series must have the same length ");
    }
    let len = aux[0];

    // Second check w series, they should be 8 elements longer than the q series
    let aux2 = vec![
        w_l.len(),
        w_r.len(),
        w_4.len(),
        z.len(),
        z2.len(),
        h1.len(),
        table.len(),
    ];
    let all_same_length = aux2.iter().all(|v| *v == (len + 8));
    if !all_same_length {
        panic!("8n+8 series must have the same length ");
    }

    // challenges only have 5 elements
    assert!(challenges.len() == 5);

    // curve_parameters only have 2 elements
    assert!(curve_parameters.len() == 2);

    // permutation parameters have 6 elements
    assert!(perm_parameters.len() == 6);

    // Third check the length of the input vectors, should be power of 2
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    // Call GPU kernel
    let err = unsafe {
        compute_quotient_term(
            device_id,
            len.trailing_zeros(),
            out.as_mut_ptr() as *mut core::ffi::c_void,
            w_l.as_ptr() as *const core::ffi::c_void,
            w_r.as_ptr() as *const core::ffi::c_void,
            w_o.as_ptr() as *const core::ffi::c_void,
            w_4.as_ptr() as *const core::ffi::c_void,
            q_l.as_ptr() as *const core::ffi::c_void,
            q_r.as_ptr() as *const core::ffi::c_void,
            q_o.as_ptr() as *const core::ffi::c_void,
            q_4.as_ptr() as *const core::ffi::c_void,
            q_hl.as_ptr() as *const core::ffi::c_void,
            q_hr.as_ptr() as *const core::ffi::c_void,
            q_h4.as_ptr() as *const core::ffi::c_void,
            q_c.as_ptr() as *const core::ffi::c_void,
            q_arith.as_ptr() as *const core::ffi::c_void,
            q_m.as_ptr() as *const core::ffi::c_void,
            r_s.as_ptr() as *const core::ffi::c_void,
            l_s.as_ptr() as *const core::ffi::c_void,
            fbms_s.as_ptr() as *const core::ffi::c_void,
            vgca_s.as_ptr() as *const core::ffi::c_void,
            pi.as_ptr() as *const core::ffi::c_void,
            z.as_ptr() as *const core::ffi::c_void,
            perm_linear.as_ptr() as *const core::ffi::c_void,
            sigma_l.as_ptr() as *const core::ffi::c_void,
            sigma_r.as_ptr() as *const core::ffi::c_void,
            sigma_o.as_ptr() as *const core::ffi::c_void,
            sigma_4.as_ptr() as *const core::ffi::c_void,
            q_lookup.as_ptr() as *const core::ffi::c_void,
            table.as_ptr() as *const core::ffi::c_void,
            f.as_ptr() as *const core::ffi::c_void,
            h1.as_ptr() as *const core::ffi::c_void,
            h2.as_ptr() as *const core::ffi::c_void,
            z2.as_ptr() as *const core::ffi::c_void,
            l1.as_ptr() as *const core::ffi::c_void,
            l1_alpha_sq.as_ptr() as *const core::ffi::c_void,
            challenges.as_ptr() as *const core::ffi::c_void,
            curve_parameters.as_ptr() as *const core::ffi::c_void,
            perm_parameters.as_ptr() as *const core::ffi::c_void,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}
