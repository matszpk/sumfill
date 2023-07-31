mod utils;
use utils::*;

fn calc_min_sumn_to_fill(n: usize, ks: usize) -> Option<Vec<usize>> {
    for k in ks..64 {
        let mut comb_iter = CombineIter::new(k, n);
        loop {
            let comb = comb_iter.get();
            if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                break;
            }
            let mut filled = vec![0usize; n];
            let mut numr_iter = CombineWithRepIter::new(k, k);
            loop {
                let numc = numr_iter.get();
                let sum = numc.iter().map(|x| comb[*x]).sum::<usize>() % n;
                filled[sum] += 1;
                if !numr_iter.next() {
                    break;
                }
            }
            
            if filled.into_iter().all(|x| x!=0) {
                return Some(Vec::from(comb));
            }
            
            if !comb_iter.next() {
                break;
            }
        }
    }
    None
}

fn main() {
    // let mut comb_iter = CombineWithRepIter::new(4, 7);
    // loop {
    //     println!("comb: {:?}", comb_iter.get());
    //     if !comb_iter.next() {
    //         break;
    //     }
    // }
    let mut k = 1;
    for i in 1..100 {
        if let Some(comb) = calc_min_sumn_to_fill(i, k) {
            println!("{}: {} {:?}", i, comb.len(), comb);
            k = std::cmp::max(1, comb.len()-1);
        }
    }
}
