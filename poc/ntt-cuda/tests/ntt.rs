// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use ntt_cuda::NTTInputOutputOrder;

const DEFAULT_GPU: usize = 0;

#[cfg(any(
    feature = "bls12_377",
    feature = "bls12_381",
    feature = "pallas",
    feature = "vesta"
))]
#[test]
fn test_against_arkworks() {
    #[cfg(feature = "bls12_377")]
    use ark_bls12_377::Fr;
    #[cfg(feature = "bls12_381")]
    use ark_bls12_381::Fr;
    use ark_ff::{PrimeField, UniformRand};
    #[cfg(feature = "pallas")]
    use ark_pallas::Fr;
    use ark_poly::{
        domain::DomainCoeff, EvaluationDomain, GeneralEvaluationDomain,
    };
    use ark_std::test_rng;
    #[cfg(feature = "vesta")]
    use ark_vesta::Fr;

    fn test_ntt<
        F: PrimeField,
        T: DomainCoeff<F> + UniformRand + core::fmt::Debug + Eq,
        R: ark_std::rand::Rng,
        D: EvaluationDomain<F>,
    >(
        rng: &mut R,
    ) {
        for lg_domain_size in 1..20 + 4 * !cfg!(debug_assertions) as i32 {
            let domain_size = 1usize << lg_domain_size;

            let domain = D::new(domain_size).unwrap();

            let mut v = vec![];
            for _ in 0..domain_size {
                v.push(T::rand(rng));
            }

            v.resize(domain.size(), T::zero());
            let mut vtest = v.clone();

            domain.fft_in_place(&mut v);
            ntt_cuda::NTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NN);
            assert!(vtest == v);

            domain.ifft_in_place(&mut v);
            ntt_cuda::iNTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NN);
            assert!(vtest == v);

            ntt_cuda::NTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NR);
            ntt_cuda::iNTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::RN);
            assert!(vtest == v);

            domain.coset_fft_in_place(&mut v);
            ntt_cuda::coset_NTT(
                DEFAULT_GPU,
                &mut vtest,
                NTTInputOutputOrder::NN,
            );
            assert!(vtest == v);

            domain.coset_ifft_in_place(&mut v);
            ntt_cuda::coset_iNTT(
                DEFAULT_GPU,
                &mut vtest,
                NTTInputOutputOrder::NN,
            );
            assert!(vtest == v);

            ntt_cuda::coset_NTT(
                DEFAULT_GPU,
                &mut vtest,
                NTTInputOutputOrder::NR,
            );
            ntt_cuda::coset_iNTT(
                DEFAULT_GPU,
                &mut vtest,
                NTTInputOutputOrder::RN,
            );
            assert!(vtest == v);
        }
    }

    /* Mute this unused test */
    /*
    fn test_arith<
        F: PrimeField,
        T: DomainCoeff<F> + UniformRand + core::fmt::Debug + Eq,
        R: ark_std::rand::Rng,
        D: EvaluationDomain<F>,
    >(
        rng: &mut R,
    ) {
        for lg_domain_size in 1..20 + 4 * !cfg!(debug_assertions) as i32 {
            let domain_size = 1usize << lg_domain_size;

            let _domain = D::new(domain_size).unwrap();

            let mut a = vec![];
            for _ in 0..domain_size {
                a.push(T::rand(rng));
            }
            let mut b = vec![];
            for _ in 0..domain_size {
                b.push(T::rand(rng));
            }
            let mut c = vec![];
            for i in 0..domain_size {
                c.push(a[i] + b[i]);
            }

            let mut atest = a.clone();
            let mut btest = b.clone();
            // Then we fill ctest with zeros
            let mut ctest = vec![T::zero(); domain_size];

            ntt_cuda::gate_constraint_sat(
                DEFAULT_GPU,
                &mut atest,
                &mut btest,
                &mut ctest,
            );
            assert!(ctest == c);
        }
    }
    */

    let rng = &mut test_rng();

    test_ntt::<Fr, Fr, _, GeneralEvaluationDomain<Fr>>(rng);
    //test_arith::<Fr, Fr, _, GeneralEvaluationDomain<Fr>>(rng);
}
