use std::io::{self, Write};
use std::sync::{atomic::{self, AtomicU32}, Arc};
use std::time::Instant;
use rayon::prelude::*;
mod utils;
use utils::*;

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

fn calc_min_sumn_to_fill(n: usize, ks: usize) -> Option<Vec<usize>> {
    for k in ks..64 {
        let mut comb_iter = CombineIter::new(k, n);
        let mut count = 0;
        loop {
            let comb = comb_iter.get();
            if (count & ((1 << 20) - 1)) == 0 {
                writeln!(io::stdout().lock(), "Progress: {} {} {:?}", n, k, comb).unwrap();
            }
            if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                break;
            }
            // let mut filled = vec![false; n];
            // let mut numr_iter = CombineWithRepIter::new(k, k);
            // loop {
            //     let numc = numr_iter.get();
            //     //let sum = numc.iter().map(|x| comb[*x]).sum::<usize>() % n;
            //     let sum = numc.iter().map(|x| comb[*x]).fold(0, |a, x| {
            //         let a = a + x;
            //         if a >= n {
            //             a - n
            //         } else {
            //             a
            //         }
            //     });
            //     filled[sum] = true;
            //     if !numr_iter.next() {
            //         break;
            //     }
            // }
            
            let mut filled = vec![false; n];
            fill_sums(n, comb, &mut filled);
            //assert_eq!(filled, filled2);
            
            if filled.into_iter().all(|x| x) {
                return Some(Vec::from(comb));
            }
            
            if !comb_iter.next() {
                break;
            }
            count += 1;
        }
    }
    None
}

fn calc_min_sumn_to_fill_par_all(n: usize) {
    // find k_start
    let ks = (1..64).find(|&x| {
        let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
        //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
        max_n >= n
    }).unwrap().try_into().unwrap();
    
    let nsq = u32::try_from((n*n)/10).unwrap();
    let max_result = std::cmp::max(nsq, 100);
    
    for k in ks..64 {
        let found_count = Arc::new(AtomicU32::new(0));
        if k < 5 {
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
                // let mut filled = vec![false; n];
                // let mut numr_iter = CombineWithRepIter::new(k, k);
                // loop {
                //     let numc = numr_iter.get();
                //     //let sum = numc.iter().map(|x| comb[*x]).sum::<usize>() % n;
                //     let sum = numc.iter().map(|x| comb[*x]).fold(0, |a, x| {
                //         let a = a + x;
                //         if a >= n {
                //             a - n
                //         } else {
                //             a
                //         }
                //     });
                //     filled[sum] = true;
                //     if !numr_iter.next() {
                //         break;
                //     }
                // }
                
                let mut filled = vec![false; n];
                fill_sums(n, comb, &mut filled);
                //assert_eq!(filled, filled2);
                
                if filled.into_iter().all(|x| x) {
                    writeln!(io::stdout().lock(), "Result {}: {} {:?}", n, k, comb).unwrap();
                    if found_count.fetch_add(1, atomic::Ordering::SeqCst) >= max_result {
                        return;
                    }
                }
                
                if !comb_iter.next() {
                    break;
                }
                count += 1;
            }
        } else {
            // let mut count = 0;
            // loop {
            // }
            const PAR_LEVEL: usize = 4;
            writeln!(io::stderr().lock(), "Tasks: {} {}", n, k).unwrap();
            
            let time = Instant::now();
            
            CombineIterStd::new(PAR_LEVEL, n)
                .take_while(|parent_comb|
                    parent_comb[0] == 0 && parent_comb[1] == 1 &&
                    found_count.load(atomic::Ordering::SeqCst) <= max_result)
                .par_bridge()
                .for_each(|parent_comb| {
                    if time.elapsed().as_millis() % 1000 < 50 {
                        writeln!(io::stderr().lock(), "Task: {} {} {:?}", n, k,
                                 parent_comb).unwrap();
                    }
                    
                    let next_p = parent_comb[PAR_LEVEL - 1] + 1;
                    
                    if k - PAR_LEVEL > n - next_p {
                        return;
                    }
                    let mut comb_iter = CombineIter::new(k - PAR_LEVEL, n - next_p);
                    let mut count = 1;
                    loop {
                        let comb = parent_comb.iter().copied().chain(
                            comb_iter.get().iter().map(|x| *x + next_p))
                            .collect::<Vec<_>>();
                        if (count & ((1 << 18) - 1)) == 0 {
                            writeln!(io::stderr().lock(),
                                     "ParProgress: {} {} {:?}", n, k, comb).unwrap();
                            if found_count.load(atomic::Ordering::SeqCst) > max_result {
                                return;
                            }
                        }
                        // let mut filled = vec![false; n];
                        // let mut numr_iter = CombineWithRepIter::new(k, k);
                        // loop {
                        //     let numc = numr_iter.get();
                        //     //let sum = numc.iter().map(|x| comb[*x]).sum::<usize>() % n;
                        //     let sum = numc.iter().map(|x| comb[*x]).fold(0, |a, x| {
                        //         let a = a + x;
                        //         if a >= n {
                        //             a - n
                        //         } else {
                        //             a
                        //         }
                        //     });
                        //     filled[sum] = true;
                        //     if !numr_iter.next() {
                        //         break;
                        //     }
                        // }
                        
                        let mut filled = vec![false; n];
                        fill_sums(n, &comb, &mut filled);
                        //assert_eq!(filled, filled2);
                        
                        if filled.into_iter().all(|x| x) {
                            if found_count.load(atomic::Ordering::SeqCst) > max_result {
                                return;
                            }
                            writeln!(io::stdout().lock(),
                                     "Result {}: {} {:?}", n, k, comb).unwrap();
                            if found_count.fetch_add(1, atomic::Ordering::SeqCst) >= max_result {
                                return;
                            }
                        }
                        
                        if !comb_iter.next() {
                            break;
                        }
                        count += 1;
                    }
                });
        };
        if found_count.load(atomic::Ordering::SeqCst) != 0 {
            break;
        }
    }
}

fn main() {
    // let mut comb_iter = CombineWithRepIter::new(4, 7);
    // loop {
    //     println!("comb: {:?}", comb_iter.get());
    //     if !comb_iter.next() {
    //         break;
    //     }
    // }
    for i in 1..3000 {
        calc_min_sumn_to_fill_par_all(i);
    }
    
//     (601..=1000).into_par_iter().for_each(|i| {
//         // find k_start
//         let ks = (1..64).find(|&x| {
//             let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
//             //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
//             max_n >= i
//         }).unwrap().try_into().unwrap();
//         //writeln!(io::stdout().lock(), "KS {}: {}", i, ks);
//         
//         if let Some(comb) = calc_min_sumn_to_fill(i, ks) {
//             writeln!(io::stdout().lock(), "Result {}: {} {:?}", i, comb.len(), comb).unwrap();
//             //k = std::cmp::max(1, comb.len()-1);
//         }
//     });
    
//     (251..=462).into_par_iter().for_each(|i| {
//         // find k_start
//         let ks = (1..64).find(|&x| {
//             let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
//             //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
//             max_n >= i
//         }).unwrap().try_into().unwrap();
//         //writeln!(io::stdout().lock(), "KS {}: {}", i, ks);
//         
//         if let Some(comb) = calc_min_sumn_to_fill_all(i, ks) {
//             writeln!(io::stdout().lock(), "Result {}: {} {:?}", i, comb.len(), comb);
//             //k = std::cmp::max(1, comb.len()-1);
//         }
//     });
}
