use std::io::{self, Write};
use std::sync::{atomic::{self, AtomicU64}, Arc};
use std::time::Instant;
use rayon::prelude::*;
mod utils;
use utils::*;

// idea:
// comb_filled - filled for sums without k-2 and k-1 elements
// filled_l1 - filled for sums with k-2 elements
// filled_l2 - filled for sums with k-1 elements
fn init_sum_fill_diff_change(n: usize, comb: &[usize], comb_filled: &mut [u64],
            filled_l1: &mut [u64], filled_l1l2_sums: &mut [Vec<usize>], filled_l2: &mut [u64]) {
    let k = comb.len();
    comb_filled.fill(0);
    filled_l1.fill(0);
    filled_l2.fill(0);
    filled_l1l2_sums.iter_mut().for_each(|sums| sums.clear());
    let filled_clen = comb_filled.len();
    let mut numr_iter = CombineWithRepIter::new(k, k);
    let fix_sh =  if (n & 63) != 0 {
        (u64::BITS - ((n as u32) & 63)) as usize
    } else {
        0
    };
    loop {
        let numc = numr_iter.get();
        //let sum = numc.iter().map(|x| comb[*x]).sum::<usize>() % n;
        let sum = numc.iter().map(|x| comb[*x]).fold(0, |a, x| {
            let a = a + x;
            if a >= n {
                a - n
            } else {
                a
            }
        });
        
        let fixsum = fix_sh + sum;
        let mut l1count = 0;
        let mut l2count = 0;
        for c in numc {
            if *c == k - 2 {
                l1count += 1;
            } else if *c == k - 1 {
                l2count += 1;
            }
        }
        if l1count == 0 && l2count == 0 {
            comb_filled[fixsum >> 6] |= 1u64 << (fixsum & 63);
        } else {
            if l1count != 0 {
                if l2count == 0 {
                    filled_l1[filled_clen*(l1count-1) + (fixsum >> 6)] |= 1u64 << (fixsum & 63);
                } else {
                    filled_l1l2_sums[k*(l1count-1) + (l2count-1)].push(sum);
                }
            } else if l2count != 0 {
                filled_l2[filled_clen*(l2count-1) + (fixsum >> 6)] |= 1u64 << (fixsum & 63);
            }
        }
        if !numr_iter.next() {
            break;
        }
    }
}

fn shift_filled_lx(len: usize, k: usize, filled_l1: &mut [u64], fix_sh: usize) {
    for i in 0..k {
        let filled = &mut filled_l1[len*i..len*(i+1)];
        let shift = i+1;
        let mut vprev = filled[len-1];
        for j in 0..len {
            let vcur = filled[j];
            filled[j] = (vcur << shift) | (vprev >> (64-shift));
            vprev = vcur;
        }
        if fix_sh != 0 {
            let mask = (1u64 << shift) - 1;
            // fix first bits
            let vold = filled[0] & mask;
            filled[0] = (filled[0] & !mask) | (vold << fix_sh);
            if (64 - fix_sh) < shift {
                filled[1] |= vold >> (64-fix_sh);
            }
        }
    }
}

fn apply_filled_lx(len: usize, k: usize, filled_l1: &[u64], comb_filled: &[u64],
                    out_filled: &mut [u64]) {
    out_filled.copy_from_slice(comb_filled);
    for i in 0..k {
        let filled = &filled_l1[len*i..len*(i+1)];
        for j in 0..len {
            out_filled[j] |= filled[j];
        }
    }
}

#[inline]
fn check_all_filled(filled: &[u64], fix_sh: usize) -> bool {
    if fix_sh != 0 {
        filled[0] == (!((1u64 << fix_sh) - 1)) &&
            filled[1..].iter().all(|x| *x == u64::MAX)
    } else {
        filled.iter().all(|x| *x == u64::MAX)
    }
}

fn process_comb_l1l2(n: usize, k: usize, start: usize, comb_filled: &[u64],
        filled_l1: &[u64], filled_l1l2_sums: &[Vec<usize>], filled_l2: &[u64],
        mut found_call: impl FnMut(usize, usize)) {
    let filled_clen = comb_filled.len();
    let fix_sh =  if (n & 63) != 0 {
        (u64::BITS - ((n as u32) & 63)) as usize
    } else {
        0
    };
    let mut l1_filled = vec![0; comb_filled.len()];
    let mut l2_filled = vec![0; comb_filled.len()];
    let mut l1_filled_l1 = Vec::from(filled_l1);
    let mut l1_filled_l1l2_sums = Vec::from(filled_l1l2_sums);
    let mut l1_filled_l2_templ = Vec::from(filled_l2);
    let mut l2_filled_l2 = l1_filled_l2_templ.clone();
    for i in start..n-1 {
        apply_filled_lx(filled_clen, k, &l1_filled_l1, &comb_filled, &mut l1_filled);
        
        l2_filled_l2.copy_from_slice(&l1_filled_l2_templ);
        for j0 in 0..k {
            for j1 in 0..(k-j0) {
                for sum in &l1_filled_l1l2_sums[j0*k + j1] {
                    let fixsum = sum + fix_sh;
                    l2_filled_l2[filled_clen*j1 + (fixsum >> 6)] |= 1u64 << (fixsum & 63);
                }
            }
        }
        
        // apply to comb_filled
        for j in i+1..n {
            apply_filled_lx(filled_clen, k, &l2_filled_l2, &l1_filled, &mut l2_filled);
            if check_all_filled(&l2_filled, fix_sh) {
                found_call(i, j);
            }
            shift_filled_lx(filled_clen, k, &mut l2_filled_l2, fix_sh);
        }
        // shift l1
        shift_filled_lx(filled_clen, k, &mut l1_filled_l1, fix_sh);
        for j0 in 0..k {
            for j1 in 0..(k-j0) {
                l1_filled_l1l2_sums[j0*k + j1].iter_mut().for_each(|sum| {
                    *sum = modulo_add(*sum, j0+j1+2, n);
                });
            }
        }
        shift_filled_lx(filled_clen, k, &mut l1_filled_l2_templ, fix_sh);
    }
}

#[inline]
fn modulo_add(a: usize, b: usize, n: usize) -> usize {
    let c = a + b;
    if c < n {
        c
    } else {
        c - n
    }
}

fn fill_sums_3(n: usize, comb: &[usize], filled: &mut [bool], shift: usize) {
    let c0 = comb[0];
    let c00 = modulo_add(shift, modulo_add(c0, c0, n), n);
    filled[modulo_add(c00, c0, n)] = true;
    if let Some(c1) = comb.get(1).copied() {
        filled[modulo_add(c00, c1, n)] = true;
        let c01 = modulo_add(shift, modulo_add(c0, c1, n), n);
        filled[modulo_add(c01, c1, n)] = true;
        let c11 = modulo_add(shift, modulo_add(c1, c1, n), n);
        filled[modulo_add(c11, c1, n)] = true;
        
        if let Some(c2) = comb.get(2).copied() {
            filled[modulo_add(c00, c2, n)] = true;
            filled[modulo_add(c01, c2, n)] = true;
            filled[modulo_add(c11, c2, n)] = true;
            
            let c02 = modulo_add(shift, modulo_add(c0, c2, n), n);
            filled[modulo_add(c02, c2, n)] = true;
            let c12 = modulo_add(shift, modulo_add(c1, c2, n), n);
            filled[modulo_add(c12, c2, n)] = true;
            let c22 = modulo_add(shift, modulo_add(c2, c2, n), n);
            filled[modulo_add(c22, c2, n)] = true;
            
            if let Some(c3) = comb.get(3).copied() {
                filled[modulo_add(c00, c3, n)] = true;
                filled[modulo_add(c01, c3, n)] = true;
                filled[modulo_add(c11, c3, n)] = true;
                filled[modulo_add(c02, c3, n)] = true;
                filled[modulo_add(c12, c3, n)] = true;
                filled[modulo_add(c22, c3, n)] = true;
                
                let c03 = modulo_add(shift, modulo_add(c0, c3, n), n);
                filled[modulo_add(c03, c3, n)] = true;
                let c13 = modulo_add(shift, modulo_add(c1, c3, n), n);
                filled[modulo_add(c13, c3, n)] = true;
                let c23 = modulo_add(shift, modulo_add(c2, c3, n), n);
                filled[modulo_add(c23, c3, n)] = true;
                let c33 = modulo_add(shift, modulo_add(c3, c3, n), n);
                filled[modulo_add(c33, c3, n)] = true;
                
                if let Some(c4) = comb.get(4).copied() {
                    filled[modulo_add(c00, c4, n)] = true;
                    filled[modulo_add(c01, c4, n)] = true;
                    filled[modulo_add(c11, c4, n)] = true;
                    filled[modulo_add(c02, c4, n)] = true;
                    filled[modulo_add(c12, c4, n)] = true;
                    filled[modulo_add(c22, c4, n)] = true;
                    filled[modulo_add(c03, c4, n)] = true;
                    filled[modulo_add(c13, c4, n)] = true;
                    filled[modulo_add(c23, c4, n)] = true;
                    filled[modulo_add(c33, c4, n)] = true;
                    
                    let c04 = modulo_add(shift, modulo_add(c0, c4, n), n);
                    filled[modulo_add(c04, c4, n)] = true;
                    let c14 = modulo_add(shift, modulo_add(c1, c4, n), n);
                    filled[modulo_add(c14, c4, n)] = true;
                    let c24 = modulo_add(shift, modulo_add(c2, c4, n), n);
                    filled[modulo_add(c24, c4, n)] = true;
                    let c34 = modulo_add(shift, modulo_add(c3, c4, n), n);
                    filled[modulo_add(c34, c4, n)] = true;
                    let c44 = modulo_add(shift, modulo_add(c4, c4, n), n);
                    filled[modulo_add(c44, c4, n)] = true;
                    
                    if let Some(c5) = comb.get(5).copied() {
                        filled[modulo_add(c00, c5, n)] = true;
                        filled[modulo_add(c01, c5, n)] = true;
                        filled[modulo_add(c11, c5, n)] = true;
                        filled[modulo_add(c02, c5, n)] = true;
                        filled[modulo_add(c12, c5, n)] = true;
                        filled[modulo_add(c22, c5, n)] = true;
                        filled[modulo_add(c03, c5, n)] = true;
                        filled[modulo_add(c13, c5, n)] = true;
                        filled[modulo_add(c23, c5, n)] = true;
                        filled[modulo_add(c33, c5, n)] = true;
                        filled[modulo_add(c04, c5, n)] = true;
                        filled[modulo_add(c14, c5, n)] = true;
                        filled[modulo_add(c24, c5, n)] = true;
                        filled[modulo_add(c34, c5, n)] = true;
                        filled[modulo_add(c44, c5, n)] = true;
                        
                        let c05 = modulo_add(shift, modulo_add(c0, c5, n), n);
                        filled[modulo_add(c05, c5, n)] = true;
                        let c15 = modulo_add(shift, modulo_add(c1, c5, n), n);
                        filled[modulo_add(c15, c5, n)] = true;
                        let c25 = modulo_add(shift, modulo_add(c2, c5, n), n);
                        filled[modulo_add(c25, c5, n)] = true;
                        let c35 = modulo_add(shift, modulo_add(c3, c5, n), n);
                        filled[modulo_add(c35, c5, n)] = true;
                        let c45 = modulo_add(shift, modulo_add(c4, c5, n), n);
                        filled[modulo_add(c45, c5, n)] = true;
                        let c55 = modulo_add(shift, modulo_add(c5, c5, n), n);
                        filled[modulo_add(c55, c5, n)] = true;
                        
                        if let Some(c6) = comb.get(6).copied() {
                            filled[modulo_add(c00, c6, n)] = true;
                            filled[modulo_add(c01, c6, n)] = true;
                            filled[modulo_add(c11, c6, n)] = true;
                            filled[modulo_add(c02, c6, n)] = true;
                            filled[modulo_add(c12, c6, n)] = true;
                            filled[modulo_add(c22, c6, n)] = true;
                            filled[modulo_add(c03, c6, n)] = true;
                            filled[modulo_add(c13, c6, n)] = true;
                            filled[modulo_add(c23, c6, n)] = true;
                            filled[modulo_add(c33, c6, n)] = true;
                            filled[modulo_add(c04, c6, n)] = true;
                            filled[modulo_add(c14, c6, n)] = true;
                            filled[modulo_add(c24, c6, n)] = true;
                            filled[modulo_add(c34, c6, n)] = true;
                            filled[modulo_add(c44, c6, n)] = true;
                            filled[modulo_add(c05, c6, n)] = true;
                            filled[modulo_add(c15, c6, n)] = true;
                            filled[modulo_add(c25, c6, n)] = true;
                            filled[modulo_add(c35, c6, n)] = true;
                            filled[modulo_add(c45, c6, n)] = true;
                            filled[modulo_add(c55, c6, n)] = true;
                            
                            let c06 = modulo_add(shift, modulo_add(c0, c6, n), n);
                            filled[modulo_add(c06, c6, n)] = true;
                            let c16 = modulo_add(shift, modulo_add(c1, c6, n), n);
                            filled[modulo_add(c16, c6, n)] = true;
                            let c26 = modulo_add(shift, modulo_add(c2, c6, n), n);
                            filled[modulo_add(c26, c6, n)] = true;
                            let c36 = modulo_add(shift, modulo_add(c3, c6, n), n);
                            filled[modulo_add(c36, c6, n)] = true;
                            let c46 = modulo_add(shift, modulo_add(c4, c6, n), n);
                            filled[modulo_add(c46, c6, n)] = true;
                            let c56 = modulo_add(shift, modulo_add(c5, c6, n), n);
                            filled[modulo_add(c56, c6, n)] = true;
                            let c66 = modulo_add(shift, modulo_add(c6, c6, n), n);
                            filled[modulo_add(c66, c6, n)] = true;
                            
                            if let Some(c7) = comb.get(7).copied() {
                                filled[modulo_add(c00, c7, n)] = true;
                                filled[modulo_add(c01, c7, n)] = true;
                                filled[modulo_add(c11, c7, n)] = true;
                                filled[modulo_add(c02, c7, n)] = true;
                                filled[modulo_add(c12, c7, n)] = true;
                                filled[modulo_add(c22, c7, n)] = true;
                                filled[modulo_add(c03, c7, n)] = true;
                                filled[modulo_add(c13, c7, n)] = true;
                                filled[modulo_add(c23, c7, n)] = true;
                                filled[modulo_add(c33, c7, n)] = true;
                                filled[modulo_add(c04, c7, n)] = true;
                                filled[modulo_add(c14, c7, n)] = true;
                                filled[modulo_add(c24, c7, n)] = true;
                                filled[modulo_add(c34, c7, n)] = true;
                                filled[modulo_add(c44, c7, n)] = true;
                                filled[modulo_add(c05, c7, n)] = true;
                                filled[modulo_add(c15, c7, n)] = true;
                                filled[modulo_add(c25, c7, n)] = true;
                                filled[modulo_add(c35, c7, n)] = true;
                                filled[modulo_add(c45, c7, n)] = true;
                                filled[modulo_add(c55, c7, n)] = true;
                                filled[modulo_add(c06, c7, n)] = true;
                                filled[modulo_add(c16, c7, n)] = true;
                                filled[modulo_add(c26, c7, n)] = true;
                                filled[modulo_add(c36, c7, n)] = true;
                                filled[modulo_add(c46, c7, n)] = true;
                                filled[modulo_add(c56, c7, n)] = true;
                                filled[modulo_add(c66, c7, n)] = true;
                                
                                let c07 = modulo_add(shift, modulo_add(c0, c7, n), n);
                                filled[modulo_add(c07, c7, n)] = true;
                                let c17 = modulo_add(shift, modulo_add(c1, c7, n), n);
                                filled[modulo_add(c17, c7, n)] = true;
                                let c27 = modulo_add(shift, modulo_add(c2, c7, n), n);
                                filled[modulo_add(c27, c7, n)] = true;
                                let c37 = modulo_add(shift, modulo_add(c3, c7, n), n);
                                filled[modulo_add(c37, c7, n)] = true;
                                let c47 = modulo_add(shift, modulo_add(c4, c7, n), n);
                                filled[modulo_add(c47, c7, n)] = true;
                                let c57 = modulo_add(shift, modulo_add(c5, c7, n), n);
                                filled[modulo_add(c57, c7, n)] = true;
                                let c67 = modulo_add(shift, modulo_add(c6, c7, n), n);
                                filled[modulo_add(c67, c7, n)] = true;
                                let c77 = modulo_add(shift, modulo_add(c7, c7, n), n);
                                filled[modulo_add(c77, c7, n)] = true;
                                
                                if let Some(c8) = comb.get(8).copied() {
                                    filled[modulo_add(c00, c8, n)] = true;
                                    filled[modulo_add(c01, c8, n)] = true;
                                    filled[modulo_add(c11, c8, n)] = true;
                                    filled[modulo_add(c02, c8, n)] = true;
                                    filled[modulo_add(c12, c8, n)] = true;
                                    filled[modulo_add(c22, c8, n)] = true;
                                    filled[modulo_add(c03, c8, n)] = true;
                                    filled[modulo_add(c13, c8, n)] = true;
                                    filled[modulo_add(c23, c8, n)] = true;
                                    filled[modulo_add(c33, c8, n)] = true;
                                    filled[modulo_add(c04, c8, n)] = true;
                                    filled[modulo_add(c14, c8, n)] = true;
                                    filled[modulo_add(c24, c8, n)] = true;
                                    filled[modulo_add(c34, c8, n)] = true;
                                    filled[modulo_add(c44, c8, n)] = true;
                                    filled[modulo_add(c05, c8, n)] = true;
                                    filled[modulo_add(c15, c8, n)] = true;
                                    filled[modulo_add(c25, c8, n)] = true;
                                    filled[modulo_add(c35, c8, n)] = true;
                                    filled[modulo_add(c45, c8, n)] = true;
                                    filled[modulo_add(c55, c8, n)] = true;
                                    filled[modulo_add(c06, c8, n)] = true;
                                    filled[modulo_add(c16, c8, n)] = true;
                                    filled[modulo_add(c26, c8, n)] = true;
                                    filled[modulo_add(c36, c8, n)] = true;
                                    filled[modulo_add(c46, c8, n)] = true;
                                    filled[modulo_add(c56, c8, n)] = true;
                                    filled[modulo_add(c66, c8, n)] = true;
                                    filled[modulo_add(c07, c8, n)] = true;
                                    filled[modulo_add(c17, c8, n)] = true;
                                    filled[modulo_add(c27, c8, n)] = true;
                                    filled[modulo_add(c37, c8, n)] = true;
                                    filled[modulo_add(c47, c8, n)] = true;
                                    filled[modulo_add(c57, c8, n)] = true;
                                    filled[modulo_add(c67, c8, n)] = true;
                                    filled[modulo_add(c77, c8, n)] = true;
                                    
                                    let c08 = modulo_add(shift, modulo_add(c0, c8, n), n);
                                    filled[modulo_add(c08, c8, n)] = true;
                                    let c18 = modulo_add(shift, modulo_add(c1, c8, n), n);
                                    filled[modulo_add(c18, c8, n)] = true;
                                    let c28 = modulo_add(shift, modulo_add(c2, c8, n), n);
                                    filled[modulo_add(c28, c8, n)] = true;
                                    let c38 = modulo_add(shift, modulo_add(c3, c8, n), n);
                                    filled[modulo_add(c38, c8, n)] = true;
                                    let c48 = modulo_add(shift, modulo_add(c4, c8, n), n);
                                    filled[modulo_add(c48, c8, n)] = true;
                                    let c58 = modulo_add(shift, modulo_add(c5, c8, n), n);
                                    filled[modulo_add(c58, c8, n)] = true;
                                    let c68 = modulo_add(shift, modulo_add(c6, c8, n), n);
                                    filled[modulo_add(c68, c8, n)] = true;
                                    let c78 = modulo_add(shift, modulo_add(c7, c8, n), n);
                                    filled[modulo_add(c78, c8, n)] = true;
                                    let c88 = modulo_add(shift, modulo_add(c8, c8, n), n);
                                    filled[modulo_add(c88, c8, n)] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn fill_sums(n: usize, comb: &[usize], filled: &mut [bool]) {
    match comb.len() {
        1 => {
            filled[comb[0]] = true;
        }
        2 => {
            let (c0, c1) = (comb[0], comb[1]);
            filled[modulo_add(c0, c0, n)] = true;
            filled[modulo_add(c0, c1, n)] = true;
            filled[modulo_add(c1, c1, n)] = true;
        }
        3 => {
            fill_sums_3(n, comb, filled, 0);
        }
        4 => {
            fill_sums_3(n, comb, filled, comb[0]);
            fill_sums_3(n, &comb[1..], filled, comb[1]);
            fill_sums_3(n, &comb[2..], filled, comb[2]);
            fill_sums_3(n, &comb[3..], filled, comb[3]);
        }
        5 => {
            let (c0, c1, c2, c3, c4) = (comb[0], comb[1], comb[2], comb[3], comb[4]);
            let c00 = modulo_add(c0, c0, n);
            fill_sums_3(n, comb, filled, c00);
            let c01 = modulo_add(c0, c1, n);
            fill_sums_3(n, &comb[1..], filled, c01);
            let c02 = modulo_add(c0, c2, n);
            fill_sums_3(n, &comb[2..], filled, c02);
            let c03 = modulo_add(c0, c3, n);
            fill_sums_3(n, &comb[3..], filled, c03);
            let c04 = modulo_add(c0, c4, n);
            fill_sums_3(n, &comb[4..], filled, c04);
            
            let c11 = modulo_add(c1, c1, n);
            fill_sums_3(n, &comb[1..], filled, c11);
            let c12 = modulo_add(c1, c2, n);
            fill_sums_3(n, &comb[2..], filled, c12);
            let c13 = modulo_add(c1, c3, n);
            fill_sums_3(n, &comb[3..], filled, c13);
            let c14 = modulo_add(c1, c4, n);
            fill_sums_3(n, &comb[4..], filled, c14);
            
            let c22 = modulo_add(c2, c2, n);
            fill_sums_3(n, &comb[2..], filled, c22);
            let c23 = modulo_add(c2, c3, n);
            fill_sums_3(n, &comb[3..], filled, c23);
            let c24 = modulo_add(c2, c4, n);
            fill_sums_3(n, &comb[4..], filled, c24);
            
            let c33 = modulo_add(c3, c3, n);
            fill_sums_3(n, &comb[3..], filled, c33);
            let c34 = modulo_add(c3, c4, n);
            fill_sums_3(n, &comb[4..], filled, c34);
            
            let c44 = modulo_add(c4, c4, n);
            fill_sums_3(n, &comb[4..], filled, c44);
        }
        6|7|8 => {
            let k = comb.len();
            let mut numr_iter = CombineWithRepIter::new(k - 3, k);
            loop {
                let numc = numr_iter.get();
                let sum = numc.iter().map(|x| comb[*x]).fold(0, |a, x| {
                    let a = a + x;
                    if a >= n {
                        a - n
                    } else {
                        a
                    }
                });
                fill_sums_3(n, &comb[*numc.last().unwrap()..], filled, sum);
                if !numr_iter.next() {
                    break;
                }
            }
        }
        _ => { panic!("unsupported!"); }
    }
}

fn calc_min_sumn_to_fill_par_all_test(n: usize) {
    // find k_start
    let ks = (1..64).find(|&x| {
        let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
        //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
        max_n >= n
    }).unwrap().try_into().unwrap();
    
    let nsq = u64::try_from((n*n)/10).unwrap();
    let max_result = std::cmp::max(nsq, 100);
    
    for k in ks..64 {
        let found_count = Arc::new(AtomicU64::new(0));
        // TESTING!!!
        let filled_clen = (n + 63) >> 6;
        let mut comb_filled = vec![0u64; filled_clen];
        let mut filled_l1 = vec![0u64; filled_clen*k];
        let mut filled_l1l2_sums = vec![vec![]; k*k];
        let mut filled_l2 = vec![0u64; filled_clen*k];
        let mut expected_found: Vec<(usize, usize)> = vec![];
        let mut comb_start_pos = 0;
        // TESTING!!!
        //if k < 5
        {
            let mut comb_iter = CombineIter::new(k, n);
            let mut count = 0;
            loop {
                let comb = comb_iter.get();
                if (count & ((1 << 20) - 1)) == 0 {
                    writeln!(io::stderr().lock(), "Progress: {} {} {:?}", n, k, comb).unwrap();
                }
                if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                    break;
                }
                
                let mut filled = vec![false; n];
                fill_sums(n, comb, &mut filled);
                //assert_eq!(filled, filled2);
                
                // TESTING!!!
                if k > 3 {
                    if comb[k-3]+1==comb[k-2] && comb[k-3]+2==comb[k-1] {
                        init_sum_fill_diff_change(n, comb, &mut comb_filled,
                                &mut filled_l1, &mut filled_l1l2_sums, &mut filled_l2);
                        expected_found.clear();
                        comb_start_pos = comb[k-2];
                    }
                }
                // TESTING!!!
                
                if filled.into_iter().all(|x| x) {
                    if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
                        writeln!(io::stdout().lock(), "Result {}: {} {:?}", n, k, comb).unwrap();
                    }
                    if k > 3 {
                        expected_found.push((comb[k-2], comb[k-1]));
                    }
                }
                
                // TESTING!!!
                if k > 3 {
                    if n-2==comb[k-2] && n-1==comb[k-1] {
                        let mut result_found = vec![];
                        //println!("TestEnd {:?}", comb);
                        process_comb_l1l2(n, k, comb_start_pos, &comb_filled,
                            &filled_l1, &filled_l1l2_sums, &filled_l2,
                            |i,j| {
                                result_found.push((i, j));
                            });
                        assert_eq!(expected_found, result_found);
                    }
                }
                // TESTING!!!
                
                if !comb_iter.next() {
                    break;
                }
                count += 1;
            }
        }
//         else {
//             // let mut count = 0;
//             // loop {
//             // }
//             const PAR_LEVEL: usize = 4;
//             writeln!(io::stderr().lock(), "Tasks: {} {}", n, k).unwrap();
//             
//             let time = Instant::now();
//             
//             CombineIterStd::new(PAR_LEVEL, n)
//                 .take_while(|parent_comb| parent_comb[0] == 0 && parent_comb[1] == 1)
//                 .par_bridge()
//                 .for_each(|parent_comb| {
//                     if time.elapsed().as_millis() % 1000 < 50 {
//                         writeln!(io::stderr().lock(), "Task: {} {} {:?}", n, k,
//                                  parent_comb).unwrap();
//                     }
//                     
//                     let next_p = parent_comb[PAR_LEVEL - 1] + 1;
//                     
//                     if k - PAR_LEVEL > n - next_p {
//                         return;
//                     }
//                     let mut comb_iter = CombineIter::new(k - PAR_LEVEL, n - next_p);
//                     let mut count = 1;
//                     loop {
//                         let comb = parent_comb.iter().copied().chain(
//                             comb_iter.get().iter().map(|x| *x + next_p))
//                             .collect::<Vec<_>>();
//                         if (count & ((1 << 18) - 1)) == 0 {
//                             writeln!(io::stderr().lock(),
//                                      "ParProgress: {} {} {:?}", n, k, comb).unwrap();
//                         }
//                         
//                         let mut filled = vec![false; n];
//                         fill_sums(n, &comb, &mut filled);
//                         //assert_eq!(filled, filled2);
//                         
//                         if filled.into_iter().all(|x| x) {
//                             if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
//                                 writeln!(io::stdout().lock(),
//                                         "Result {}: {} {:?}", n, k, comb).unwrap();
//                             }
//                         }
//                         
//                         if !comb_iter.next() {
//                             break;
//                         }
//                         count += 1;
//                     }
//                 });
//         };
        if found_count.load(atomic::Ordering::SeqCst) != 0 {
            writeln!(io::stdout().lock(), "Total results {}: {} {}", n, k,
                        found_count.load(atomic::Ordering::SeqCst)).unwrap();
            break;
        }
    }
}

fn calc_min_sumn_to_fill_par_all(n: usize) {
    // find k_start
    let ks = (1..64).find(|&x| {
        let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
        //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
        max_n >= n
    }).unwrap().try_into().unwrap();
    
    let nsq = u64::try_from((n*n)/10).unwrap();
    let max_result = std::cmp::max(nsq, 100);
    
    for k in ks..64 {
        let found_count = Arc::new(AtomicU64::new(0));
        //if k < 5
        {
            let mut comb_iter = CombineIter::new(k, n);
            let mut count = 0;
            loop {
                let comb = comb_iter.get();
                if (count & ((1 << 20) - 1)) == 0 {
                    writeln!(io::stderr().lock(), "Progress: {} {} {:?}", n, k, comb).unwrap();
                }
                if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                    break;
                }
                
                let mut filled = vec![false; n];
                fill_sums(n, comb, &mut filled);
                
                if filled.into_iter().all(|x| x) {
                    if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
                        writeln!(io::stdout().lock(), "Result {}: {} {:?}", n, k, comb).unwrap();
                    }
                }
                
                if !comb_iter.next() {
                    break;
                }
                count += 1;
            }
        }
//         else {
//             // let mut count = 0;
//             // loop {
//             // }
//             const PAR_LEVEL: usize = 4;
//             writeln!(io::stderr().lock(), "Tasks: {} {}", n, k).unwrap();
//             
//             let time = Instant::now();
//             
//             CombineIterStd::new(PAR_LEVEL, n)
//                 .take_while(|parent_comb| parent_comb[0] == 0 && parent_comb[1] == 1)
//                 .par_bridge()
//                 .for_each(|parent_comb| {
//                     if time.elapsed().as_millis() % 1000 < 50 {
//                         writeln!(io::stderr().lock(), "Task: {} {} {:?}", n, k,
//                                  parent_comb).unwrap();
//                     }
//                     
//                     let next_p = parent_comb[PAR_LEVEL - 1] + 1;
//                     
//                     if k - PAR_LEVEL > n - next_p {
//                         return;
//                     }
//                     let mut comb_iter = CombineIter::new(k - PAR_LEVEL, n - next_p);
//                     let mut count = 1;
//                     loop {
//                         let comb = parent_comb.iter().copied().chain(
//                             comb_iter.get().iter().map(|x| *x + next_p))
//                             .collect::<Vec<_>>();
//                         if (count & ((1 << 18) - 1)) == 0 {
//                             writeln!(io::stderr().lock(),
//                                      "ParProgress: {} {} {:?}", n, k, comb).unwrap();
//                         }
//                         
//                         let mut filled = vec![false; n];
//                         fill_sums(n, &comb, &mut filled);
//                         //assert_eq!(filled, filled2);
//                         
//                         if filled.into_iter().all(|x| x) {
//                             if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
//                                 writeln!(io::stdout().lock(),
//                                         "Result {}: {} {:?}", n, k, comb).unwrap();
//                             }
//                         }
//                         
//                         if !comb_iter.next() {
//                             break;
//                         }
//                         count += 1;
//                     }
//                 });
//         };
        if found_count.load(atomic::Ordering::SeqCst) != 0 {
            writeln!(io::stdout().lock(), "Total results {}: {} {}", n, k,
                        found_count.load(atomic::Ordering::SeqCst)).unwrap();
            break;
        }
    }
}

fn calc_min_sumn_to_fill_par_all_2(n: usize) {
    // find k_start
    let ks = (1..64).find(|&x| {
        let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
        //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
        max_n >= n
    }).unwrap().try_into().unwrap();
    
    let nsq = u64::try_from((n*n)/10).unwrap();
    let max_result = std::cmp::max(nsq, 100);
    
    for k in ks..64 {
        let found_count = Arc::new(AtomicU64::new(0));
        if k < 4 {
            let mut comb_iter = CombineIter::new(k, n);
            let mut count = 0;
            loop {
                let comb = comb_iter.get();
                if (count & ((1 << 20) - 1)) == 0 {
                    writeln!(io::stderr().lock(), "Progress: {} {} {:?}", n, k, comb).unwrap();
                }
                if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                    break;
                }
                
                let mut filled = vec![false; n];
                fill_sums(n, comb, &mut filled);
                
                if filled.into_iter().all(|x| x) {
                    if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
                        writeln!(io::stdout().lock(), "Result {}: {} {:?}", n, k, comb).unwrap();
                    }
                }
                
                if !comb_iter.next() {
                    break;
                }
                count += 1;
            }
        }
        else {
            let mut comb_iter = CombineIter::new(k - 2, n - 2);
            let mut count = 0;
            let mut final_comb = vec![0; k];
            
            let filled_clen = (n + 63) >> 6;
            let mut comb_filled = vec![0u64; filled_clen];
            let mut filled_l1 = vec![0u64; filled_clen*k];
            let mut filled_l1l2_sums = vec![vec![]; k*k];
            let mut filled_l2 = vec![0u64; filled_clen*k];
            
            loop {
                let comb = comb_iter.get();
                if (count & ((1 << 15) - 1)) == 0 {
                    writeln!(io::stderr().lock(), "Progress: {} {} {:?}", n, k, comb).unwrap();
                }
                if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                    break;
                }
                
                final_comb[0..k-2].copy_from_slice(comb);
                final_comb[k-2] = final_comb[k-3] + 1;
                final_comb[k-1] = final_comb[k-3] + 2;
                
                init_sum_fill_diff_change(n, &final_comb, &mut comb_filled,
                                &mut filled_l1, &mut filled_l1l2_sums, &mut filled_l2);
                process_comb_l1l2(n, k, final_comb[k-2], &comb_filled,
                &filled_l1, &filled_l1l2_sums, &filled_l2,
                    |i,j| {
                        final_comb[k-2] = i;
                        final_comb[k-1] = j;
                        if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
                            writeln!(io::stdout().lock(), "Result {}: {} {:?}", n, k, final_comb)
                                .unwrap();
                        }
                    });
                
                if !comb_iter.next() {
                    break;
                }
                count += 1;
            }
        }
        
//         else {
//             // let mut count = 0;
//             // loop {
//             // }
//             const PAR_LEVEL: usize = 4;
//             writeln!(io::stderr().lock(), "Tasks: {} {}", n, k).unwrap();
//             
//             let time = Instant::now();
//             
//             CombineIterStd::new(PAR_LEVEL, n)
//                 .take_while(|parent_comb| parent_comb[0] == 0 && parent_comb[1] == 1)
//                 .par_bridge()
//                 .for_each(|parent_comb| {
//                     if time.elapsed().as_millis() % 1000 < 50 {
//                         writeln!(io::stderr().lock(), "Task: {} {} {:?}", n, k,
//                                  parent_comb).unwrap();
//                     }
//                     
//                     let next_p = parent_comb[PAR_LEVEL - 1] + 1;
//                     
//                     if k - PAR_LEVEL > n - next_p {
//                         return;
//                     }
//                     let mut comb_iter = CombineIter::new(k - PAR_LEVEL, n - next_p);
//                     let mut count = 1;
//                     loop {
//                         let comb = parent_comb.iter().copied().chain(
//                             comb_iter.get().iter().map(|x| *x + next_p))
//                             .collect::<Vec<_>>();
//                         if (count & ((1 << 18) - 1)) == 0 {
//                             writeln!(io::stderr().lock(),
//                                      "ParProgress: {} {} {:?}", n, k, comb).unwrap();
//                         }
//                         
//                         let mut filled = vec![false; n];
//                         fill_sums(n, &comb, &mut filled);
//                         //assert_eq!(filled, filled2);
//                         
//                         if filled.into_iter().all(|x| x) {
//                             if found_count.fetch_add(1, atomic::Ordering::SeqCst) < max_result {
//                                 writeln!(io::stdout().lock(),
//                                         "Result {}: {} {:?}", n, k, comb).unwrap();
//                             }
//                         }
//                         
//                         if !comb_iter.next() {
//                             break;
//                         }
//                         count += 1;
//                     }
//                 });
//         };
        if found_count.load(atomic::Ordering::SeqCst) != 0 {
            writeln!(io::stdout().lock(), "Total results {}: {} {}", n, k,
                        found_count.load(atomic::Ordering::SeqCst)).unwrap();
            break;
        }
    }
}

fn main() {
    for i in 1..100 {
        calc_min_sumn_to_fill_par_all_2(i);
    }
}
