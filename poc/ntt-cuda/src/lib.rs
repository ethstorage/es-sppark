// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

sppark::cuda_error!();

#[macro_use]
extern crate lazy_static;

lazy_static! {
    /// The default GPU device ID
    static ref DEFAULT_GPU: usize = 0;

    static ref DEFAULT_GPU_MAX: (u64, u64) = {
        let id = *DEFAULT_GPU;
        let mut max_memory: u64 = 0;
        let mut max_threading: u64 = 0;
        unsafe { cuda_get_info(id as i32, &mut max_memory, &mut max_threading) };
        (max_memory, max_threading)
    };

    static ref DEFAULT_GPU_MAX_MEMORY: u64 = DEFAULT_GPU_MAX.0;
    static ref DEFAULT_GPU_MAX_THREADING: u64 = DEFAULT_GPU_MAX.1;

    // Reserve 512MB memory for each GPU
    static ref MEMORY_RESERVE: u64 = 512 * 1024 * 1024;
}

#[derive(Clone, Copy)]
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
        domain_size: usize,
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
        v_h_coset: *const core::ffi::c_void,
        challenges: *const core::ffi::c_void,
        curve_params: *const core::ffi::c_void,
        perm_params: *const core::ffi::c_void,
    ) -> cuda::Error;
}

extern "C" {
    fn compute_product_argument(
        device_id: usize,
        lg_domain_size: u32,
        out: *mut core::ffi::c_void,
        root: *const core::ffi::c_void,
        gate_sigma0: *const core::ffi::c_void,
        gate_sigma1: *const core::ffi::c_void,
        gate_sigma2: *const core::ffi::c_void,
        gate_sigma3: *const core::ffi::c_void,
        gate_wire0: *const core::ffi::c_void,
        gate_wire1: *const core::ffi::c_void,
        gate_wire2: *const core::ffi::c_void,
        gate_wire3: *const core::ffi::c_void,
        ks: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        gamma: *const core::ffi::c_void,
    ) -> cuda::Error;

    fn compute_lookup_product_argument(
        device_id: usize,
        lg_domain_size: u32,
        out: *mut core::ffi::c_void,
        f: *const core::ffi::c_void,
        t: *const core::ffi::c_void,
        t_next: *const core::ffi::c_void,
        h_1: *const core::ffi::c_void,
        h_1_next: *const core::ffi::c_void,
        h_2: *const core::ffi::c_void,
        delta_arr: *const core::ffi::c_void,
        epsilon_arr: *const core::ffi::c_void,
    ) -> cuda::Error;

    fn compute_linear_poly(
        device_id: usize,
        lg_domain_size: u32,
        out: *mut core::ffi::c_void,
        q_m: *const core::ffi::c_void,
        q_l: *const core::ffi::c_void,
        q_r: *const core::ffi::c_void,
        q_o: *const core::ffi::c_void,
        q_4: *const core::ffi::c_void,
        q_hl: *const core::ffi::c_void,
        q_hr: *const core::ffi::c_void,
        q_h4: *const core::ffi::c_void,
        q_c: *const core::ffi::c_void,
        z_poly: *const core::ffi::c_void,
        fourth_sigma: *const core::ffi::c_void,
        t_1_poly: *const core::ffi::c_void,
        t_2_poly: *const core::ffi::c_void,
        t_3_poly: *const core::ffi::c_void,
        t_4_poly: *const core::ffi::c_void,
        t_5_poly: *const core::ffi::c_void,
        t_6_poly: *const core::ffi::c_void,
        t_7_poly: *const core::ffi::c_void,
        t_8_poly: *const core::ffi::c_void,
        wit_vals: *const core::ffi::c_void,
        perm_vals: *const core::ffi::c_void,
        power: *const u64,
    ) -> cuda::Error;
}

extern "C" {
    fn cuda_get_info(id: i32, max_memory: *mut u64, max_threading: *mut u64);
}

// Helper function to floor a number to the nearest power of 2
#[allow(unused)]
fn floor_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }

    let mut n = n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if std::mem::size_of::<usize>() == 8 {
        n |= n >> 32;
    }
    n = n - (n >> 1);

    n
}

macro_rules! check_len {
    ($len:expr, $device_id:expr, $T:expr) => {
        if ($len & ($len - 1)) != 0 {
            panic!("inout.len() is not power of 2");
        }

        // Available memory
        let gpu_memory = {
            let gpu_mem = if $device_id != *DEFAULT_GPU {
                get_cuda_info($device_id as i32).0
            } else {
                *DEFAULT_GPU_MAX_MEMORY
            };
            gpu_mem - *MEMORY_RESERVE
        };

        // Thread number
        let thread_num = if $device_id != *DEFAULT_GPU {
            get_cuda_info($device_id as i32).1
        } else {
            *DEFAULT_GPU_MAX_THREADING
        };

        // Single size of T
        let size_of_t = std::mem::size_of::<T>();

        // FFT must be done within one round
        // DFT X_k = sum_{m=0}^{N-1} x_m * exp(-2*pi*i*m*k/N)
        // One cannot do it in piecewise, X_k would be wrong then
        assert!($len * size_of_t <= gpu_memory as usize);
        assert!($len <= thread_num as usize);
    };
}

/// Compute an in-place NTT on the input data.
#[allow(non_snake_case)]
pub fn NTT<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    check_len!(len, device_id, T);

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
    check_len!(len, device_id, T);

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
    check_len!(len, device_id, T);

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
    check_len!(len, device_id, T);

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

pub fn parallelized_range<T>(device_id: usize, len: usize) -> usize {
    // Single size of T
    let size_of_t = std::mem::size_of::<T>();

    // Total memory required
    let total_memory = size_of_t * len;

    // Available memory
    let gpu_memory = {
        let gpu_mem = if device_id != *DEFAULT_GPU {
            get_cuda_info(device_id as i32).0
        } else {
            *DEFAULT_GPU_MAX_MEMORY
        };
        gpu_mem - *MEMORY_RESERVE
    };

    gpu_memory as usize / total_memory
}

pub fn compute_mem_segment(
    device_id: usize,
    len: usize,
    mem_requied: usize,
) -> (usize, usize) {
    // Available memory
    let gpu_memory = {
        let gpu_mem = if device_id != *DEFAULT_GPU {
            get_cuda_info(device_id as i32).0
        } else {
            *DEFAULT_GPU_MAX_MEMORY
        };
        gpu_mem - *MEMORY_RESERVE
    };
    // How many round if memory is not enough
    let round = (mem_requied as u64 + gpu_memory - 1) / gpu_memory;
    let round = round as usize;
    // Proper size (of buffer length)
    let domain_size = (len + round - 1) / round as usize;

    // Thread number
    let thread_num = if device_id != *DEFAULT_GPU {
        get_cuda_info(device_id as i32).1
    } else {
        *DEFAULT_GPU_MAX_THREADING
    };

    // Normally GPU threading is around 2^63 or so, simply calculation will
    // leads to u64 overflow So we only need to check if domain size is
    // smaller than thread number
    assert!(domain_size <= thread_num as usize);
    // println!("buffer size {}, total memory needed {} bytes, gpu memory
    // available {} bytes, domain_size: {}, rounds: {}", len, mem_requied,
    // gpu_memory, domain_size, len / domain_size);
    (round, domain_size)
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
    v_h_coset: &[T],
    challenges: &[T],
    curve_parameters: &[T],
    perm_parameters: &[T],
) {
    // First check whether majority of the vectors have the same length
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
        v_h_coset.len(),
    ];
    let all_same_length = aux.iter().all(|v| *v == aux[0]);
    if !all_same_length {
        panic!(" input series must have the same length ");
    }
    let len = aux[0];

    let aux2 = vec![
        w_l.len(),
        w_r.len(),
        w_4.len(),
        z.len(),
        z2.len(),
        h1.len(),
        table.len(),
    ];
    let all_same_length = aux2.iter().all(|v| *v == len + 8);
    if !all_same_length {
        panic!(" extended series must have the same length, len + 8 ");
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

    // Single size of T
    let size_of_t = std::mem::size_of::<T>();

    // Total memory required
    let total_memory =
        size_of_t * (aux.len() * len + aux2.len() * (len + 8) + 5 + 2 + 6);

    let (round, domain_size) =
        compute_mem_segment(device_id, len, total_memory);

    // Burden same GPU for now
    // Submit all buffer to same GPU in sequence
    for i in 0..round {
        // Compute the start and end index
        let start = i * domain_size;
        let end = std::cmp::min((i + 1) * domain_size, len);
        let extend_end = end + 8;

        // println!(
        //     "round {}, start {}, end {}, extend_end {}",
        //     i, start, end, extend_end
        // );

        // Simple memory sanitizer
        {
            let _ = w_l[start];
            let _ = w_r[end - 1];
            let _ = table[end];
            let _ = w_4[extend_end - 1];
        }

        // Call GPU kernel
        let err = unsafe {
            compute_quotient_term(
                device_id,
                end - start,
                out[start..end].as_mut_ptr() as *mut core::ffi::c_void,
                w_l[start..extend_end].as_ptr() as *const core::ffi::c_void,
                w_r[start..extend_end].as_ptr() as *const core::ffi::c_void,
                w_o[start..end].as_ptr() as *const core::ffi::c_void,
                w_4[start..extend_end].as_ptr() as *const core::ffi::c_void,
                q_l[start..end].as_ptr() as *const core::ffi::c_void,
                q_r[start..end].as_ptr() as *const core::ffi::c_void,
                q_o[start..end].as_ptr() as *const core::ffi::c_void,
                q_4[start..end].as_ptr() as *const core::ffi::c_void,
                q_hl[start..end].as_ptr() as *const core::ffi::c_void,
                q_hr[start..end].as_ptr() as *const core::ffi::c_void,
                q_h4[start..end].as_ptr() as *const core::ffi::c_void,
                q_c[start..end].as_ptr() as *const core::ffi::c_void,
                q_arith[start..end].as_ptr() as *const core::ffi::c_void,
                q_m[start..end].as_ptr() as *const core::ffi::c_void,
                r_s[start..end].as_ptr() as *const core::ffi::c_void,
                l_s[start..end].as_ptr() as *const core::ffi::c_void,
                fbms_s[start..end].as_ptr() as *const core::ffi::c_void,
                vgca_s[start..end].as_ptr() as *const core::ffi::c_void,
                pi[start..end].as_ptr() as *const core::ffi::c_void,
                z[start..extend_end].as_ptr() as *const core::ffi::c_void,
                perm_linear[start..end].as_ptr() as *const core::ffi::c_void,
                sigma_l[start..end].as_ptr() as *const core::ffi::c_void,
                sigma_r[start..end].as_ptr() as *const core::ffi::c_void,
                sigma_o[start..end].as_ptr() as *const core::ffi::c_void,
                sigma_4[start..end].as_ptr() as *const core::ffi::c_void,
                q_lookup[start..end].as_ptr() as *const core::ffi::c_void,
                table[start..extend_end].as_ptr() as *const core::ffi::c_void,
                f[start..end].as_ptr() as *const core::ffi::c_void,
                h1[start..extend_end].as_ptr() as *const core::ffi::c_void,
                h2[start..end].as_ptr() as *const core::ffi::c_void,
                z2[start..extend_end].as_ptr() as *const core::ffi::c_void,
                l1[start..end].as_ptr() as *const core::ffi::c_void,
                l1_alpha_sq[start..end].as_ptr() as *const core::ffi::c_void,
                v_h_coset[start..end].as_ptr() as *const core::ffi::c_void,
                challenges.as_ptr() as *const core::ffi::c_void,
                curve_parameters.as_ptr() as *const core::ffi::c_void,
                perm_parameters.as_ptr() as *const core::ffi::c_void,
            )
        };

        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }
}

pub fn product_argument_gpu<T>(
    device_id: usize,
    out: &mut [T],
    root: &[T],
    gate_sigma0: &[T],
    gate_sigma1: &[T],
    gate_sigma2: &[T],
    gate_sigma3: &[T],
    gate_wire0: &[T],
    gate_wire1: &[T],
    gate_wire2: &[T],
    gate_wire3: &[T],
    ks: &[T],
    beta: T,
    gamma: T,
) {
    let aux = vec![
        out.len(),
        root.len(),
        gate_sigma0.len(),
        gate_sigma1.len(),
        gate_sigma2.len(),
        gate_sigma3.len(),
        gate_wire0.len(),
        gate_wire1.len(),
        gate_wire2.len(),
        gate_wire3.len(),
    ];

    let all_same_length = aux.iter().all(|v| *v == aux[0]);
    if !all_same_length {
        panic!(" input series must have the same length ");
    }

    assert!(ks.len() == 4);

    let len = aux[0];

    let beta_arr = &[beta];
    let gamma_arr = &[gamma];

    // Single size of T
    let size_of_t = std::mem::size_of::<T>();
    // Total memory required
    let total_memory = size_of_t * (aux.len() * len + ks.len() + 2);

    let (round, domain_size) =
        compute_mem_segment(device_id, len, total_memory);

    // Burden same GPU for now
    // Submit all buffer to same GPU in sequence
    for i in 0..round {
        // Compute the start and end index
        let start = i * domain_size;
        let end = std::cmp::min((i + 1) * domain_size, len);
        let err = unsafe {
            compute_product_argument(
                device_id,
                len.trailing_zeros(),
                out[start..end].as_mut_ptr() as *mut core::ffi::c_void,
                root[start..end].as_ptr() as *const core::ffi::c_void,
                gate_sigma0[start..end].as_ptr() as *const core::ffi::c_void,
                gate_sigma1[start..end].as_ptr() as *const core::ffi::c_void,
                gate_sigma2[start..end].as_ptr() as *const core::ffi::c_void,
                gate_sigma3[start..end].as_ptr() as *const core::ffi::c_void,
                gate_wire0[start..end].as_ptr() as *const core::ffi::c_void,
                gate_wire1[start..end].as_ptr() as *const core::ffi::c_void,
                gate_wire2[start..end].as_ptr() as *const core::ffi::c_void,
                gate_wire3[start..end].as_ptr() as *const core::ffi::c_void,
                ks.as_ptr() as *const core::ffi::c_void,
                beta_arr.as_ptr() as *const core::ffi::c_void,
                gamma_arr.as_ptr() as *const core::ffi::c_void,
            )
        };

        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }
}

pub fn lookup_product_argument_gpu<T>(
    device_id: usize,
    out: &mut [T],
    f: &[T],
    t: &[T],
    t_next: &[T],
    h_1: &[T],
    h_1_next: &[T],
    h_2: &[T],
    delta: T,
    epsilon: T,
) {
    let aux = vec![
        f.len(),
        t.len(),
        t_next.len(),
        h_1.len(),
        h_1_next.len(),
        h_2.len(),
    ];

    let all_same_length = aux.iter().all(|v| *v == aux[0]);
    if !all_same_length {
        panic!(" input series must have the same length ");
    }

    let len = aux[0];

    let delta_arr = &[delta];
    let epsilon_arr = &[epsilon];

    // Single size of T
    let size_of_t = std::mem::size_of::<T>();
    // Total memory required
    let total_memory = size_of_t * (aux.len() * len + 2);

    let (round, domain_size) =
        compute_mem_segment(device_id, len, total_memory);

    // Burden same GPU for now
    // Submit all buffer to same GPU in sequence
    for i in 0..round {
        // Compute the start and end index
        let start = i * domain_size;
        let end = std::cmp::min((i + 1) * domain_size, len);
        let err = unsafe {
            compute_lookup_product_argument(
                device_id,
                len.trailing_zeros(),
                out[start..end].as_mut_ptr() as *mut core::ffi::c_void,
                f[start..end].as_ptr() as *const core::ffi::c_void,
                t[start..end].as_ptr() as *const core::ffi::c_void,
                t_next[start..end].as_ptr() as *const core::ffi::c_void,
                h_1[start..end].as_ptr() as *const core::ffi::c_void,
                h_1_next[start..end].as_ptr() as *const core::ffi::c_void,
                h_2[start..end].as_ptr() as *const core::ffi::c_void,
                delta_arr.as_ptr() as *const core::ffi::c_void,
                epsilon_arr.as_ptr() as *const core::ffi::c_void,
            )
        };

        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }
}

pub fn get_cuda_info(device_id: i32) -> (u64, u64) {
    let mut max_memory: u64 = 0;
    let mut max_threading: u64 = 0;
    unsafe { cuda_get_info(device_id, &mut max_memory, &mut max_threading) };
    (max_memory, max_threading)
}

#[allow(unused_macros)]
macro_rules! param {
    ($var: ident, $len: ident, $start: ident, $end: ident) => {
        if $var.len() == $len {
            $var[$start..$end].as_ptr() as *const core::ffi::c_void
        } else {
            $var.as_ptr() as *const core::ffi::c_void
        }
    };
}

// for plonk linearisation_poly optimization
pub fn linear_poly_gpu<T>(
    device_id: usize,
    out: &mut [T],
    q_m: &[T],
    q_l: &[T],
    q_r: &[T],
    q_o: &[T],
    q_4: &[T],
    q_hl: &[T],
    q_hr: &[T],
    q_h4: &[T],
    q_c: &[T],
    z_poly: &[T],
    fourth_sigma: &[T],
    t_1_poly: &[T],
    t_2_poly: &[T],
    t_3_poly: &[T],
    t_4_poly: &[T],
    t_5_poly: &[T],
    t_6_poly: &[T],
    t_7_poly: &[T],
    t_8_poly: &[T],
    wit_vals: &[T],
    perm_vals: &[T],
    power: &[u64],
) {
    let aux = vec![
        out.len(),
        q_l.len(),
        q_r.len(),
        q_o.len(),
        q_4.len(),
        q_hl.len(),
        q_hr.len(),
        q_h4.len(),
        q_c.len(),
        z_poly.len(),
        fourth_sigma.len(),
        t_1_poly.len(),
        t_2_poly.len(),
        t_3_poly.len(),
        t_4_poly.len(),
        t_5_poly.len(),
    ];

    let all_same_length = aux.iter().all(|v| *v == aux[0]);
    if !all_same_length {
        panic!(" input series must have the same length ");
    }

    assert!(wit_vals.len() == 5, "wit_vals must have 5 vals");
    assert!(perm_vals.len() == 11);
    assert!(power.len() == 1);

    let len = aux[0];

    // Single size of T
    let size_of_t = std::mem::size_of::<T>();
    // Total memory required
    let total_memory = size_of_t
        * (aux.len() * len
            + q_m.len()  // q_m's len is zero in this case
            + t_6_poly.len() // len is -6 less than aux[0] in this case
            + t_7_poly.len() // len is zero in this case
            + t_8_poly.len() // len is zero in this case
            + wit_vals.len()
            + perm_vals.len()
            + power.len());

    let (round, domain_size) =
        compute_mem_segment(device_id, len, total_memory);

    // Burden same GPU for now
    // Submit all buffer to same GPU in sequence
    for i in 0..round {
        // Compute the start and end index
        let start = i * domain_size;
        let end = std::cmp::min((i + 1) * domain_size, len);
        let t_poly_6_end = std::cmp::min((i + 1) * domain_size, t_6_poly.len());

        let err = unsafe {
            compute_linear_poly(
                device_id,
                len.trailing_zeros(),
                out[start..end].as_mut_ptr() as *mut core::ffi::c_void,
                param!(q_m, len, start, end),
                q_l[start..end].as_ptr() as *const core::ffi::c_void,
                q_r[start..end].as_ptr() as *const core::ffi::c_void,
                q_o[start..end].as_ptr() as *const core::ffi::c_void,
                q_4[start..end].as_ptr() as *const core::ffi::c_void,
                q_hl[start..end].as_ptr() as *const core::ffi::c_void,
                q_hr[start..end].as_ptr() as *const core::ffi::c_void,
                q_h4[start..end].as_ptr() as *const core::ffi::c_void,
                q_c[start..end].as_ptr() as *const core::ffi::c_void,
                z_poly[start..end].as_ptr() as *const core::ffi::c_void,
                fourth_sigma[start..end].as_ptr() as *const core::ffi::c_void,
                t_1_poly[start..end].as_ptr() as *const core::ffi::c_void,
                t_2_poly[start..end].as_ptr() as *const core::ffi::c_void,
                t_3_poly[start..end].as_ptr() as *const core::ffi::c_void,
                t_4_poly[start..end].as_ptr() as *const core::ffi::c_void,
                t_5_poly[start..end].as_ptr() as *const core::ffi::c_void,
                t_6_poly[start..t_poly_6_end].as_ptr()
                    as *const core::ffi::c_void,
                param!(t_7_poly, len, start, end),
                param!(t_8_poly, len, start, end),
                wit_vals.as_ptr() as *const core::ffi::c_void,
                perm_vals.as_ptr() as *const core::ffi::c_void,
                power.as_ptr() as *const u64,
            )
        };

        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }
}
