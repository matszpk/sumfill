use std::env;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_uint, cl_ulong, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use rayon::prelude::*;
use std::sync::{atomic::{self, AtomicU64}, Arc};
use std::ptr;
mod utils;
use utils::*;

const L2_LEN_STEP_SIZE: usize = 32;

// OPENCL VERSION DOESN'T WORK!!!! - bad results!!!

// CPU routines

fn init_sum_fill_diff_change(n: usize, comb: &[usize], comb_filled: &mut [u32],
            filled_l1: &mut [u32], filled_l1l2_sums: &mut [Vec<usize>], filled_l2: &mut [u32]) {
    let k = comb.len();
    comb_filled.fill(0);
    filled_l1.fill(0);
    filled_l2.fill(0);
    filled_l1l2_sums.iter_mut().for_each(|sums| sums.clear());
    let filled_clen = comb_filled.len();
    let mut numr_iter = CombineWithRepIter::new(k, k);
    let fix_sh =  if (n & 31) != 0 {
        (u32::BITS - ((n as u32) & 31)) as usize
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
            comb_filled[fixsum >> 5] |= 1u32 << (fixsum & 31);
        } else {
            if l1count != 0 {
                if l2count == 0 {
                    filled_l1[filled_clen*(l1count-1) + (fixsum >> 5)] |= 1u32 << (fixsum & 31);
                } else {
                    filled_l1l2_sums[k*(l1count-1) + (l2count-1)].push(sum);
                }
            } else if l2count != 0 {
                filled_l2[filled_clen*(l2count-1) + (fixsum >> 5)] |= 1u32 << (fixsum & 31);
            }
        }
        if !numr_iter.next() {
            break;
        }
    }
}

fn shift_filled_lx(len: usize, k: usize, filled_l1: &mut [u32], fix_sh: usize) {
    for i in 0..k {
        let filled = &mut filled_l1[len*i..len*(i+1)];
        let shift = i+1;
        let mut vprev = filled[len-1];
        for j in 0..len {
            let vcur = filled[j];
            filled[j] = (vcur << shift) | (vprev >> (32-shift));
            vprev = vcur;
        }
        if fix_sh != 0 {
            let mask = (1u32 << shift) - 1;
            // fix first bits
            let vold = filled[0] & mask;
            filled[0] = (filled[0] & !mask) | (vold << fix_sh);
            if (32 - fix_sh) < shift {
                filled[1] |= vold >> (32-fix_sh);
            }
        }
    }
}

fn apply_filled_lx(len: usize, k: usize, filled_l1: &[u32], comb_filled: &[u32],
                    out_filled: &mut [u32]) {
    out_filled.copy_from_slice(comb_filled);
    for i in 0..k {
        let filled = &filled_l1[len*i..len*(i+1)];
        for j in 0..len {
            out_filled[j] |= filled[j];
        }
    }
}

#[inline]
fn check_all_filled(filled: &[u32], fix_sh: usize) -> bool {
    if fix_sh != 0 {
        filled[0] == (!((1u32 << fix_sh) - 1)) &&
            filled[1..].iter().all(|x| *x == u32::MAX)
    } else {
        filled.iter().all(|x| *x == u32::MAX)
    }
}

fn process_comb_l1l2(n: usize, k: usize, start: usize, comb_filled: &[u32],
        filled_l1: &[u32], filled_l1l2_sums: &[Vec<usize>], filled_l2: &[u32],
        mut found_call: impl FnMut(usize, usize)) {
    let filled_clen = comb_filled.len();
    let fix_sh =  if (n & 31) != 0 {
        (u32::BITS - ((n as u32) & 31)) as usize
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
                    l2_filled_l2[filled_clen*j1 + (fixsum >> 5)] |= 1u32 << (fixsum & 31);
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CombL2Task {
    l1_task_id: usize,
    l1: usize,
    l2_filled_l2: Vec<u32>,
    l1_filled: Vec<u32>,
}

impl CombL2Task {
    fn process_comb_l2(&self, n: usize, k: usize, mut found_call: impl FnMut(usize, usize)) {
        let filled_clen = self.l1_filled.len();
        let mut l2_filled_l2 = self.l2_filled_l2.clone();
        let l1_filled = self.l1_filled.clone();
        let mut l2_filled = vec![0; filled_clen];
        let fix_sh =  if (n & 31) != 0 {
            (u32::BITS - ((n as u32) & 31)) as usize
        } else {
            0
        };
        for j in self.l1+1..n {
            apply_filled_lx(filled_clen, k, &l2_filled_l2, &l1_filled, &mut l2_filled);
            if check_all_filled(&l2_filled, fix_sh) {
                found_call(self.l1, j);
            }
            shift_filled_lx(filled_clen, k, &mut l2_filled_l2, fix_sh);
        }
    }
}

#[derive(Clone)]
struct CombTask {
    n: usize,
    comb: Vec<usize>,
    comb_filled: Vec<u32>,
    filled_l1: Vec<u32>,
    filled_l1l2_sums: Vec<Vec<usize>>,
    filled_l2: Vec<u32>,
    l1: usize,
}

impl CombTask {
    fn new(n: usize, comb: &[usize]) -> Self {
        let filled_clen = (n + 31) >> 5;
        let k = comb.len();
        let mut comb_filled = vec![0u32; filled_clen];
        let mut filled_l1 = vec![0u32; filled_clen*k];
        let mut filled_l1l2_sums = vec![vec![]; k*k];
        let mut filled_l2 = vec![0u32; filled_clen*k];
        init_sum_fill_diff_change(n, comb, &mut comb_filled, &mut filled_l1,
                                  &mut filled_l1l2_sums, &mut filled_l2);
        Self {
            n,
            comb: Vec::from(comb),
            comb_filled,
            filled_l1,
            filled_l1l2_sums,
            filled_l2,
            l1: comb[comb.len()-2],
        }
    }
    
    fn process_comb_l1(&mut self, task_id: usize, min_iter: usize,
                       l2_tasks: &mut Vec<CombL2Task>) {
        let n = self.n;
        let k = self.comb.len();
        let start = self.l1;
        
        if min_iter > n-(start+1) {
            return;
        }
        
        let filled_clen = self.comb_filled.len();
        let fix_sh =  if (n & 31) != 0 {
            (u32::BITS - ((n as u32) & 31)) as usize
        } else {
            0
        };
        let mut l1_filled = vec![0; self.comb_filled.len()];
        let l1_filled_l1 = &mut self.filled_l1;
        let l1_filled_l1l2_sums = &mut self.filled_l1l2_sums;
        let l1_filled_l2_templ = &mut self.filled_l2;
        let mut l2_filled_l2 = l1_filled_l2_templ.clone();
        
        for i in start..n-1 {
            if min_iter > n-(i+1) {
                self.l1 = i;
                return;
            }    
            
            apply_filled_lx(filled_clen, k, l1_filled_l1, &self.comb_filled, &mut l1_filled);
            l2_filled_l2.copy_from_slice(l1_filled_l2_templ);
            for j0 in 0..k {
                for j1 in 0..(k-j0) {
                    for sum in &l1_filled_l1l2_sums[j0*k + j1] {
                        let fixsum = sum + fix_sh;
                        l2_filled_l2[filled_clen*j1 + (fixsum >> 5)] |= 1u32 << (fixsum & 31);
                    }
                }
            }
            // put new L2 task
            l2_tasks.push(CombL2Task{ l1_task_id: task_id, l2_filled_l2: l2_filled_l2.clone(),
                                l1_filled: l1_filled.clone(), l1: i });
            // shift l1
            shift_filled_lx(filled_clen, k, l1_filled_l1, fix_sh);
            for j0 in 0..k {
                for j1 in 0..(k-j0) {
                    l1_filled_l1l2_sums[j0*k + j1].iter_mut().for_each(|sum| {
                        *sum = modulo_add(*sum, j0+j1+2, n);
                    });
                }
            }
            shift_filled_lx(filled_clen, k, l1_filled_l2_templ, fix_sh);
        }
        self.l1 = n-1;
    }
}

// CPU routines - end

const PROGRAM_SOURCE: &str = r#"
// CONST_K - k value
// CONST_N - n value
// FIX_SH - (32-n)%32 (if n%32!=0 else 0)
// FCLEN - n/32
// GROUP_LEN - wavefront length
// MAX_RESULTS - max results number

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#if CONST_K < 5 || CONST_K > 9
#error "Unsupported CONST_K"
#endif

// #if CONST_N < 161
// #error "Unsupported CONST_N"
// #endif

#if CONST_K == 5
#define L1L2_TOTAL_SUMS (35)
constant uint l1l2_sum_pos[25] = {
    0, 10, 16, 19, 20, 20, 26, 29, 30, 30, 30, 33, 34, 34, 34, 34, 35, 35, 35, 35,
    35, 35, 35, 35, 35
};
#endif

#if CONST_K == 6
#define L1L2_TOTAL_SUMS (126)
constant uint l1l2_sum_pos[36] = {
    0, 35, 55, 65, 69, 70, 70, 90, 100, 104, 105, 105, 105, 115, 119,
    120, 120, 120, 120, 124, 125, 125, 125, 125, 125, 126, 126, 126,
    126, 126, 126, 126, 126, 126, 126, 126
};
#endif

#if CONST_K == 7
#define L1L2_TOTAL_SUMS (462)
constant uint l1l2_sum_pos[49] = {
    0, 126, 196, 231, 246, 251, 252, 252, 322, 357, 372, 377, 378, 378, 378, 413, 428, 433,
    434, 434, 434, 434, 449, 454, 455, 455, 455, 455, 455, 460, 461, 461, 461, 461, 461, 461,
    462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462
};
#endif

#if CONST_K == 8
#define L1L2_TOTAL_SUMS (1716)
constant uint l1l2_sum_pos[64] = {
    0, 462, 714, 840, 896, 917, 923, 924, 924, 1176, 1302, 1358, 1379, 1385, 1386, 1386, 1386,
    1512, 1568, 1589, 1595, 1596, 1596, 1596, 1596, 1652, 1673, 1679, 1680, 1680, 1680, 1680,
    1680, 1701, 1707, 1708, 1708, 1708, 1708, 1708, 1708, 1714, 1715, 1715, 1715, 1715, 1715,
    1715, 1715, 1716, 1716, 1716, 1716, 1716, 1716, 1716, 1716, 1716, 1716, 1716, 1716, 1716,
    1716, 1716
};
#endif

#if CONST_K == 9
#define L1L2_TOTAL_SUMS (6435)
constant uint l1l2_sum_pos[81] = {
    0, 1716, 2640, 3102, 3312, 3396, 3424, 3431, 3432, 3432, 4356, 4818, 5028, 5112, 5140,
    5147, 5148, 5148, 5148, 5610, 5820, 5904, 5932, 5939, 5940, 5940, 5940, 5940, 6150, 6234,
    6262, 6269, 6270, 6270, 6270, 6270, 6270, 6354, 6382, 6389, 6390, 6390, 6390, 6390, 6390,
    6390, 6418, 6425, 6426, 6426, 6426, 6426, 6426, 6426, 6426, 6433, 6434, 6434, 6434, 6434,
    6434, 6434, 6434, 6434, 6435, 6435, 6435, 6435, 6435, 6435, 6435, 6435, 6435, 6435, 6435,
    6435, 6435, 6435, 6435, 6435, 6435
};
#endif

typedef struct _CombTask {
    uint comb[CONST_K-2];
    uint comb_filled[FCLEN];
    uint filled_l1[FCLEN*CONST_K];
    uint filled_l1l2_sums[L1L2_TOTAL_SUMS];
    uint filled_l2[FCLEN*CONST_K];
    uint comb_k_l1;
} CombTask;

kernel void init_sum_fill_diff_change(uint task_num, global const uint* combs,
                    global CombTask* comb_tasks) {
    const uint gid = get_global_id(0);
    if (gid >= task_num)
        return;
    const global uint* comb = combs + CONST_K*gid;
    global CombTask* comb_task = comb_tasks + gid;
    //
    uint i;
    for (i = 0; i < CONST_K-2; i++)
        comb_task->comb[i] = comb[i];
    for (i = 0; i < FCLEN; i++)
        comb_task->comb_filled[i] = 0;
    for (i = 0; i < FCLEN*CONST_K; i++) {
        comb_task->filled_l1[i] = 0;
        comb_task->filled_l2[i] = 0;
    }
    for (i = 0; i < L1L2_TOTAL_SUMS; i++)
        comb_task->filled_l1l2_sums[i] = 0;
    comb_task->comb_k_l1 = comb[CONST_K-3] + 1;
    // initialize iterator
    const uint lid = get_local_id(0);
    local uint numcomb_group[GROUP_LEN*CONST_K];
    local uint* numcomb = numcomb_group + CONST_K*lid;
    for (i = 0; i < CONST_K; i++)
        numcomb[i] = 0;
    
    local uint l1l2idx_idx_group[GROUP_LEN*CONST_K*CONST_K];
    local uint* l1l2idx_idx = l1l2idx_idx_group + CONST_K*CONST_K*lid;
    for (i = 0; i < CONST_K*CONST_K; i++)
        l1l2idx_idx[i] = 0;
    // main loop
    while (true) {
        // fill up comb task
        uint sum = 0;
        for (i = 0; i < CONST_K; i++) {
            sum += comb[numcomb[i]];
            if (sum >= CONST_N)
                sum -= CONST_N;
        }
        
        uint l1count = 0;
        uint l2count = 0;
        for (i = 0; i < CONST_K; i++) {
            const uint c = numcomb[i];
            if (c == CONST_K - 2) {
                l1count++;
            } else if (c == CONST_K - 1) {
                l2count++;
            }
        }
        const uint fixsum = sum + FIX_SH;
        
        if ((l1count == 0) && (l2count == 0)) {
            comb_task->comb_filled[fixsum >> 5] |= 1 << (fixsum & 31);
        } else {
            if (l1count != 0) {
                if (l2count == 0)
                    comb_task->filled_l1[FCLEN*(l1count-1) + (fixsum>>5)] |= 1<<(fixsum & 31);
                else {
                    const uint vec_id = CONST_K*(l1count-1) + (l2count-1);
                    comb_task->filled_l1l2_sums[
                        l1l2_sum_pos[vec_id] + l1l2idx_idx[vec_id]] = sum;
                    l1l2idx_idx[vec_id] += 1;
                }
            } else if (l2count != 0) {
                comb_task->filled_l2[FCLEN*(l2count-1) + (fixsum>>5)] |= 1<<(fixsum & 31);
            }
        }
        
        // next iteration of numc (combinations with replacements) (k,k)
        {
            const uint len = CONST_K;
            uint ki;
            for (ki = 0; ki < len; ki++) {
                const uint i = len - ki - 1;
                numcomb[i] += 1;
                if (numcomb[i] < CONST_K) {
                    uint j;
                    for (j = 1; j <= ki; j++)
                        numcomb[i + j] = numcomb[i];
                    break;
                }
            }
            if (ki == len)
                break;
        }
    }
}

typedef struct _CombL2Task {
    uint l1_task_id;
    uint l2_filled_l2[CONST_K*FCLEN];
    uint l1_filled[FCLEN];
    uint l1;
} CombL2Task;

inline void shift_filled_lx_global(global uint* filled_l1) {
    uint i;
    for (i = 0; i < CONST_K; i++) {
        global uint* filled = filled_l1 + FCLEN*i;
        uint shift = i+1;
        uint vprev = filled[FCLEN-1];
        uint j;
        for (j = 0; j < FCLEN; j++) {
            uint vcur = filled[j];
            filled[j] = (vcur << shift) | (vprev >> (32-shift));
            vprev = vcur;
        }
        if (FIX_SH != 0) {
            uint mask = (1 << shift) - 1;
            // fix first bits
            uint vold = filled[0] & mask;
            filled[0] = (filled[0] & ~mask) | (vold << FIX_SH);
            if ((32 - FIX_SH) < shift) {
                filled[1] |= vold >> (32-FIX_SH);
            }
        }
    }
}

inline void apply_filled_lx_global(global uint* filled_l1, private uint* comb_filled,
                    private uint* out_filled) {
    uint i;
    for (i = 0; i < FCLEN; i++)
        out_filled[i] = comb_filled[i];
    for (i = 0; i < CONST_K; i++) {
        uint j = 0;
        for (j = 0; j < FCLEN; j++) {
            out_filled[j] |= filled_l1[FCLEN*i + j];
        }
    }
}

inline void shift_filled_lx_private(private uint* filled_l1) {
    uint i;
    for (i = 0; i < CONST_K; i++) {
        private uint* filled = filled_l1 + FCLEN*i;
        uint shift = i+1;
        uint vprev = filled[FCLEN-1];
        uint j;
        for (j = 0; j < FCLEN; j++) {
            uint vcur = filled[j];
            filled[j] = (vcur << shift) | (vprev >> (32-shift));
            vprev = vcur;
        }
        if (FIX_SH != 0) {
            uint mask = (1 << shift) - 1;
            // fix first bits
            uint vold = filled[0] & mask;
            filled[0] = (filled[0] & ~mask) | (vold << FIX_SH);
            if ((32 - FIX_SH) < shift) {
                filled[1] |= vold >> (32-FIX_SH);
            }
        }
    }
}

inline void apply_filled_lx_private(private uint* filled_l1, private uint* comb_filled,
                    private uint* out_filled) {
    uint i;
    for (i = 0; i < FCLEN; i++)
        out_filled[i] = comb_filled[i];
    for (i = 0; i < CONST_K; i++) {
        uint j = 0;
        for (j = 0; j < FCLEN; j++) {
            out_filled[j] |= filled_l1[FCLEN*i + j];
        }
    }
}

kernel void process_comb_l1(uint task_num, uint min_iter, global CombTask* comb_tasks,
                    global CombL2Task* comb_l2_tasks, global uint* comb_l2_task_num) {
    const uint gid = get_global_id(0);
    if (gid >= task_num)
        return;
    global CombTask* comb_task = comb_tasks + gid;
    
    uint start = comb_task->comb_k_l1;
    if (min_iter > CONST_N - (start+1))
        return;
    
    global uint* l1_filled_l1 = comb_task->filled_l1;
    global uint* l1_filled_l1l2_sums = comb_task->filled_l1l2_sums;
    global uint* l1_filled_l2_templ = comb_task->filled_l2;
    private uint comb_filled[FCLEN];
    private uint l1_filled[FCLEN];
    uint i;
    for (i = 0; i < FCLEN; i++)
        comb_filled[i] = comb_task->comb_filled[i];
    
    for (i = start; i < CONST_N-1; i++) {
        uint j0, j1;
        if (min_iter > CONST_N-(i+1)) {
            comb_task->comb_k_l1 = i;
            return;
        }
        
        apply_filled_lx_global(l1_filled_l1, comb_filled, l1_filled);
        
        {   // put new L2 task
            const uint l2pos = atomic_inc(comb_l2_task_num);
            global CombL2Task* l2_task = comb_l2_tasks + l2pos;
            global uint* l2_filled_l2 = l2_task->l2_filled_l2;
            for (j0 = 0; j0 < CONST_K*FCLEN; j0++)
                l2_filled_l2[j0] = l1_filled_l2_templ[j0];
            for (j0 = 0; j0 < CONST_K; j0++)
                for (j1 = 0; j1 < CONST_K-j0; j1++) {
                    uint vidx = l1l2_sum_pos[j0*CONST_K+j1];
                    const uint vend = l1l2_sum_pos[j0*CONST_K+j1+1];
                    for (; vidx < vend; vidx++) {
                        uint fixsum = l1_filled_l1l2_sums[vidx] + FIX_SH;
                        l2_filled_l2[FCLEN*j1 + (fixsum>>5)] |= 1<<(fixsum&31);
                    }
                }
            for (j0 = 0; j0 < CONST_K*FCLEN; j0++)
                l2_task->l2_filled_l2[j0] = l2_filled_l2[j0];
            for (j0 = 0; j0 < FCLEN; j0++)
                l2_task->l1_filled[j0] = l1_filled[j0];
            l2_task->l1 = i;
            l2_task->l1_task_id = gid;
        }
        
        // shift
        shift_filled_lx_global(l1_filled_l1);
        for (j0 = 0; j0 < CONST_K; j0++)
            for (j1 = 0; j1 < CONST_K-j0; j1++) {
                uint vidx = l1l2_sum_pos[j0*CONST_K+j1];
                const uint vend = l1l2_sum_pos[j0*CONST_K+j1+1];
                for (; vidx < vend; vidx++) {
                    uint sum = l1_filled_l1l2_sums[vidx] + j0+j1 + 2;
                    if (sum >= CONST_N)
                        sum -= CONST_N;
                    l1_filled_l1l2_sums[vidx] = sum;
                }
            }
        shift_filled_lx_global(l1_filled_l2_templ);
    }
    comb_task->comb_k_l1 = CONST_N-1;
}

kernel void process_comb_l2(uint task_num, global CombTask* comb_tasks,
            const global CombL2Task* comb_l2_tasks,
            global uint* results, global ulong* result_count) {
    const uint gid = get_global_id(0);
    if (gid >= task_num)
        return;
    const global CombL2Task* l2_task = comb_l2_tasks + gid;
    //global uint* l2_filled_l2 = l2_task->l2_filled_l2;
    private uint l2_filled_l2[FCLEN*CONST_K];
    private uint l1_filled[FCLEN];
    private uint l2_filled[FCLEN];
    uint j;
    for (j = 0; j < FCLEN; j++)
        l1_filled[j] = l2_task->l1_filled[j];
    for (j = 0; j < FCLEN*CONST_K; j++)
        l2_filled_l2[j] = l2_task->l2_filled_l2[j];
    
    uint l1 = l2_task->l1;
    for (j = l1 + 1; j < CONST_N; j++) {
        apply_filled_lx_private(l2_filled_l2, l1_filled, l2_filled);
        uint val = UINT_MAX;
        uint i = 0;
        if (FIX_SH != 0) {
            i = 1;
            val &= (l2_filled[0] | ((1 << FIX_SH) - 1));
        }
        for (; i < FCLEN; i++)
            val &= l2_filled[i];
        if (val == UINT_MAX) {
            ulong old = atom_inc(result_count);
            if (old < MAX_RESULTS) {
                uint i2;
                global const uint* comb = comb_tasks[l2_task->l1_task_id].comb;
                for (i2 = 0; i2 < CONST_K-2; i2++)
                    results[old*CONST_K + i2] = comb[i2];
                results[old*CONST_K + CONST_K-2] = l1;
                results[old*CONST_K + CONST_K-1] = j;
            }
        }
        shift_filled_lx_private(l2_filled_l2);
    }
}
"#;

/// OpenCL stuff

pub struct CLNWork {
    n: usize,
    k: usize,
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
    init_sum_fill_diff_change_kernel: Kernel,
    process_comb_l1_kernel: Kernel,
    process_comb_l2_kernel: Kernel,
    max_results: usize,
    group_num: usize,
    group_len: usize,
    task_num: usize,
    comb_task_len: usize,
    comb_l2_task_len: usize,
    combs: Buffer<cl_uint>,
    comb_tasks: Buffer<cl_uint>,
    comb_l2_tasks: Buffer<cl_uint>,
    comb_l2_task_num: Buffer<cl_uint>,
    results: Buffer<cl_uint>,
    result_count: Buffer<cl_ulong>,
}

impl CLNWork {
    fn new(device_index: usize, n: usize, k: usize) -> Result<Self> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
            .get(device_index).expect("No device in platform");
        let device = Device::new(device_id);
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, 0)?;
        
        let max_results = 10000;
        
        let fix_sh =  if (n & 31) != 0 {
            (u32::BITS - ((n as u32) & 31)) as usize
        } else {
            0
        };
        
        let group_len = usize::try_from(device.max_work_group_size()? >> 2).unwrap();
        let group_num = usize::try_from(8 * device.max_compute_units()?).unwrap();
        println!("CLNWork: GroupLen: {}, GroupNum: {}", group_len, group_num);
        
        let fclen = (n + 31) >> 5;
        let prog_opts = format!(
                concat!("-DCONST_N=({}) -DCONST_K=({}) -DFIX_SH=({}) -DFCLEN=({}) ",
                        "-DGROUP_LEN=({}) -DMAX_RESULTS=({})"),
                n, k, fix_sh, fclen, group_len, max_results);
        let program = match Program::create_and_build_from_source(&context, PROGRAM_SOURCE,
                &prog_opts) {
            Ok(program) => program,
            Err(err) => {
                panic!("Can't compile program: {}", err);
            }
        };
        let init_kernel = Kernel::create(&program, "init_sum_fill_diff_change")?;
        let process_comb_l1_kernel = Kernel::create(&program, "process_comb_l1")?;
        let process_comb_l2_kernel = Kernel::create(&program, "process_comb_l2")?;
        
        let l1l2_total_sums = match k {
            5 => 32,
            6 => 126,
            7 => 462,
            8 => 1716,
            9 => 6435,
            _ => { panic!("Unsupported k"); }
        };
        let comb_task_len = (k-2) + fclen + k*fclen*2 + l1l2_total_sums + 1;
        let comb_l2_task_len = 1 + k*fclen + fclen + 1;
        let task_num = group_len * group_num;
        
        let combs = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                            k * task_num, ptr::null_mut())?
        };
        let comb_tasks = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                    comb_task_len * task_num, ptr::null_mut())?
        };
        let comb_l2_tasks = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                    comb_l2_task_len * task_num * L2_LEN_STEP_SIZE, ptr::null_mut())?
        };
        let comb_l2_task_num = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut())?
        };
        let results = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                            k * max_results, ptr::null_mut())?
        };
        let result_count = unsafe {
            Buffer::<cl_ulong>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut())?
        };
        
        Ok(Self {
           n,
           k,
           device,
           context,
           queue,
           program,
           init_sum_fill_diff_change_kernel: init_kernel,
           process_comb_l1_kernel,
           process_comb_l2_kernel,
           group_num,
           group_len,
           comb_task_len,
           comb_l2_task_len,
           max_results,
           task_num,
           combs,
           comb_tasks,
           comb_l2_tasks,
           comb_l2_task_num,
           results,
           result_count
        })
    }
    
    fn test_init_kernel(&mut self) {
        let filled_clen = (self.n + 31) >> 5;
        let mut count = 0;
        let mut exp_cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
        let mut cl_combs: Vec<cl_uint> = vec![0; self.k*self.task_num];
        let mut cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
        let mut comb_task_data: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = vec![
                        (vec![], vec![], vec![]); self.task_num];
        
        let mut comb_iter = CombineIter::new(self.k - 2, self.n - 2);
        let mut final_comb = vec![0; self.k];
        let mut comb_filled = vec![0u32; filled_clen];
        let mut filled_l1 = vec![0u32; filled_clen*self.k];
        let mut filled_l1l2_sums = vec![vec![]; self.k*self.k];
        let mut filled_l2 = vec![0u32; filled_clen*self.k];
        
        // compare results
        loop {
            let comb = comb_iter.get();
            
            let has_next_1 = comb[0] == 0 && comb.get(1).copied().unwrap_or(1) == 1;
            
            if has_next_1 {
                final_comb[0..self.k-2].copy_from_slice(comb);
                final_comb[self.k-2] = final_comb[self.k-3] + 1;
                final_comb[self.k-1] = final_comb[self.k-3] + 2;
                
                init_sum_fill_diff_change(self.n, &final_comb, &mut comb_filled,
                                &mut filled_l1, &mut filled_l1l2_sums, &mut filled_l2);
                
                {
                    // put comb to cl_combs
                    cl_combs[count*self.k..(count+1)*self.k].iter_mut().enumerate()
                            .for_each(|(i,x)| *x = final_comb[i] as cl_uint);
                    // put to expected cl comb_task
                    let mut exp_comb_task = &mut exp_cl_comb_tasks[
                            self.comb_task_len*count..self.comb_task_len*(count+1)];
                    // copy comb
                    comb.iter().take(self.k-2).enumerate().for_each(|(i, x)|
                        exp_comb_task[i] = *x as cl_uint);
                    // copy comb_filled
                    comb_filled.iter().enumerate().for_each(|(i, x)|
                        exp_comb_task[(self.k-2) + i] = *x as cl_uint);
                    // copy filled_l1
                    filled_l1.iter().enumerate().for_each(|(i, x)|
                        exp_comb_task[(self.k-2) + filled_clen + i] = *x as cl_uint);
                    // copy filled_l1l2_sum
                    let mut idx = (self.k-2) + filled_clen + filled_clen*self.k;
                    for j in 0..self.k*self.k {
                        filled_l1l2_sums[j].iter().enumerate().for_each(|(i,x)|
                            exp_comb_task[idx + i] = *x as cl_uint);
                        idx += filled_l1l2_sums[j].len();
                    }
                    // copy filled_l2
                    filled_l2.iter().enumerate().for_each(|(i, x)|
                        exp_comb_task[idx + i] = *x as cl_uint);
                    exp_comb_task[exp_comb_task.len() - 1] = final_comb[self.k-2] as cl_uint;
                }
                count += 1;
            }
            
            let has_next = has_next_1 && comb_iter.next();
            
            if !has_next || count == self.task_num {
                unsafe {
                    self.queue.enqueue_write_buffer(&mut self.combs, CL_BLOCKING,
                                0, &cl_combs, &[]).unwrap();
                    let cl_task_num = count as cl_uint;
                    println!("NDrange: {} {}", count, self.task_num);
                    // call init_kernel
                    ExecuteKernel::new(&self.init_sum_fill_diff_change_kernel)
                            .set_arg(&cl_task_num)
                            .set_arg(&self.combs)
                            .set_arg(&self.comb_tasks)
                            .set_local_work_size(self.group_len)
                            .set_global_work_size((((count + self.group_len - 1)
                                    / self.group_len)) * self.group_len)
                            .enqueue_nd_range(&self.queue).unwrap();
                    self.queue.finish().unwrap();
                    self.queue.enqueue_read_buffer(&mut self.comb_tasks, CL_BLOCKING,
                                0, &mut cl_comb_tasks, &[]).unwrap();
                    // compare results
                    for i in 0..count {
                        let exp_comb_task = &mut exp_cl_comb_tasks[
                                self.comb_task_len*i..self.comb_task_len*(i+1)];
                        let res_comb_task = &mut cl_comb_tasks[
                                self.comb_task_len*i..self.comb_task_len*(i+1)];
                        assert_eq!(exp_comb_task, res_comb_task, "comb_task {}", i);
                        // if exp_comb_task != res_comb_task {
                        //     println!("Not Equal {}: {:?}!={:?}", i, 
                        //              exp_comb_task, res_comb_task);
                        // }
                    }
                }
                count = 0;
                println!("CCX: {:?}", final_comb);
            }
            
            if !has_next {
                break;
            }
        }
        //
        
        // call kernel
        // end of OpenCL stuff
    }
    
    fn test_calc(&mut self) {
        let filled_clen = (self.n + 31) >> 5;
        let mut count = 0;
        
        let mut comb_iter = CombineIter::new(self.k - 2, self.n - 2);
        let mut final_comb = vec![0; self.k];
        
        let task_num = 2048;
        let mut l1_tasks: Vec<CombTask> = vec![];
        
        let result_count = Arc::new(AtomicU64::new(0u64));
        
        println!("TestCalc");
        // compare results
        loop {
            let comb = comb_iter.get();
            
            let has_next_1 = comb[0] == 0 && comb.get(1).copied().unwrap_or(1) == 1;
            
            if has_next_1 {
                final_comb[0..self.k-2].copy_from_slice(comb);
                final_comb[self.k-2] = final_comb[self.k-3] + 1;
                final_comb[self.k-1] = final_comb[self.k-3] + 2;
            }
            
            // init_sum_fill_diff_change(self.n, &final_comb, &mut comb_filled,
            //                 &mut filled_l1, &mut filled_l1l2_sums, &mut filled_l2);
            if has_next_1 {
                l1_tasks.push(CombTask::new(self.n, &final_comb));
                count += 1;
            }
            
            let has_next = has_next_1 && comb_iter.next();
            
            if !has_next || count == task_num {
                let max_step_num = (self.n + L2_LEN_STEP_SIZE - 1) / L2_LEN_STEP_SIZE;
                for i in (0..max_step_num).rev() {
                    println!("Cyyy: {}", i*L2_LEN_STEP_SIZE);
                    let mut l2_tasks: Vec<CombL2Task> = vec![];
                    for (id, l1_task) in l1_tasks.iter_mut().enumerate() {
                        l1_task.process_comb_l1(id, i*L2_LEN_STEP_SIZE, &mut l2_tasks);
                    }
                    // process l2 tasks
                    // for l2_task in l2_tasks {
                    //     l2_task.process_comb_l2(self.n, self.k, |i, j| {
                    //         result_count += 1;
                    //     });
                    // }
                    l2_tasks.into_par_iter().for_each(|l2_task| {
                        l2_task.process_comb_l2(self.n, self.k, |i, j| {
                            let rc = result_count.fetch_add(1, atomic::Ordering::SeqCst);
                            if rc < u64::try_from(self.max_results).unwrap() {
                                let mut comb = Vec::from(
                                    &l1_tasks[l2_task.l1_task_id].comb[0..self.k-2]);
                                comb.push(i);
                                comb.push(j);
                                println!("Result {}: {} {:?}", self.n, self.k, comb);
                            }
                        });
                    });
                }
                l1_tasks.clear();
                count = 0;
                println!("CCX: {:?}", final_comb);
            }
            
            if !has_next {
                break;
            }
        }
        println!("Total results: {}", result_count.load(atomic::Ordering::SeqCst));
    }
    
    fn test_calc_cl(&mut self) {
        let filled_clen = (self.n + 31) >> 5;
        let mut count = 0;
        
        let mut comb_iter = CombineIter::new(self.k - 2, self.n - 2);
        let mut final_comb = vec![0; self.k];
        
        let task_num = self.task_num;
        let mut l1_tasks: Vec<CombTask> = vec![];
        let mut cl_combs = vec![];
        let mut cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
        let mut exp_cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
        
        let result_count = Arc::new(AtomicU64::new(0u64));
        
        println!("TestCalc CL");
        {
            let result_count_cl = [0];
            unsafe {
                self.queue.enqueue_write_buffer(&mut self.result_count, CL_BLOCKING,
                                0, &result_count_cl[..], &[]).unwrap();
            }
        }
        
        let mut cl_result_count_old = 0;
        
        // compare results
        loop {
            let comb = comb_iter.get();
            
            let has_next_1 = comb[0] == 0 && comb.get(1).copied().unwrap_or(1) == 1;
            
            if has_next_1 {
                final_comb[0..self.k-2].copy_from_slice(comb);
                final_comb[self.k-2] = final_comb[self.k-3] + 1;
                final_comb[self.k-1] = final_comb[self.k-3] + 2;
            }
            
            // init_sum_fill_diff_change(self.n, &final_comb, &mut comb_filled,
            //                 &mut filled_l1, &mut filled_l1l2_sums, &mut filled_l2);
            if has_next_1 {
                l1_tasks.push(CombTask::new(self.n, &final_comb));
                cl_combs.extend(final_comb.iter().map(|x| u32::try_from(*x).unwrap()));
                
                {
                    let comb_task = &l1_tasks[count];
                    // put to expected cl comb_task
                    let mut exp_comb_task = &mut exp_cl_comb_tasks[
                            self.comb_task_len*count..self.comb_task_len*(count+1)];
                    // copy comb
                    comb_task.comb.iter().take(self.k-2).enumerate().for_each(|(i, x)|
                        exp_comb_task[i] = *x as cl_uint);
                    // copy comb_filled
                    comb_task.comb_filled.iter().enumerate().for_each(|(i, x)|
                        exp_comb_task[(self.k-2) + i] = *x as cl_uint);
                    // copy filled_l1
                    comb_task.filled_l1.iter().enumerate().for_each(|(i, x)|
                        exp_comb_task[(self.k-2) + filled_clen + i] = *x as cl_uint);
                    // copy filled_l1l2_sum
                    let mut idx = (self.k-2) + filled_clen + filled_clen*self.k;
                    for j in 0..self.k*self.k {
                        comb_task.filled_l1l2_sums[j].iter().enumerate().for_each(|(i,x)|
                            exp_comb_task[idx + i] = *x as cl_uint);
                        idx += comb_task.filled_l1l2_sums[j].len();
                    }
                    // copy filled_l2
                    comb_task.filled_l2.iter().enumerate().for_each(|(i, x)|
                        exp_comb_task[idx + i] = *x as cl_uint);
                    exp_comb_task[exp_comb_task.len() - 1] = final_comb[self.k-2] as cl_uint;
                }
                count += 1;
            }
            
            let has_next = has_next_1 && comb_iter.next();
            
            if !has_next || count == task_num {
                unsafe {
                    // call init kernel
                    self.queue.enqueue_write_buffer(&mut self.combs, CL_BLOCKING,
                                0, &cl_combs, &[]).unwrap();
                    let cl_task_num = count as cl_uint;
                    println!("Count xxx: {} {} {}", cl_task_num,
                             (((count + self.group_len - 1)
                                    / self.group_len)) * self.group_len,
                             cl_combs.len()
                             );
                    // call init_kernel
                    ExecuteKernel::new(&self.init_sum_fill_diff_change_kernel)
                            .set_arg(&cl_task_num)
                            .set_arg(&self.combs)
                            .set_arg(&self.comb_tasks)
                            .set_local_work_size(self.group_len)
                            .set_global_work_size((((count + self.group_len - 1)
                                    / self.group_len)) * self.group_len)
                            .enqueue_nd_range(&self.queue).unwrap();
                    self.queue.finish().unwrap();
                    
                    // TESTING!
                    self.queue.enqueue_read_buffer(&mut self.comb_tasks, CL_BLOCKING,
                                0, &mut cl_comb_tasks, &[]).unwrap();
                    // compare results
                    for i in 0..count {
                        let exp_comb_task = &mut exp_cl_comb_tasks[
                                self.comb_task_len*i..self.comb_task_len*(i+1)];
                        let res_comb_task = &mut cl_comb_tasks[
                                self.comb_task_len*i..self.comb_task_len*(i+1)];
                        assert_eq!(exp_comb_task, res_comb_task, "comb_task {}", i);
                    }
                    // TESTING!
                }
                
                let max_step_num = (self.n + L2_LEN_STEP_SIZE - 1) / L2_LEN_STEP_SIZE;
                for i in (0..max_step_num).rev() {
                    println!("Cyyy: {}", i*L2_LEN_STEP_SIZE);
                    let cl_min_iter = (i*L2_LEN_STEP_SIZE) as cl_uint;
                    let cl_task_num = count as cl_uint;
                    // call process_comb_l1 kernel
                    unsafe {
                        let comb_l2_task_num_cl = [0];
                        self.queue.enqueue_write_buffer(
                                &mut self.comb_l2_task_num, CL_BLOCKING,
                                0, &comb_l2_task_num_cl[..], &[]).unwrap();
                        ExecuteKernel::new(&self.process_comb_l1_kernel)
                                .set_arg(&cl_task_num)
                                .set_arg(&cl_min_iter)
                                .set_arg(&self.comb_tasks)
                                .set_arg(&self.comb_l2_tasks)
                                .set_arg(&self.comb_l2_task_num)
                                .set_local_work_size(self.group_len)
                                .set_global_work_size((((count + self.group_len - 1)
                                        / self.group_len)) * self.group_len)
                                .enqueue_nd_range(&self.queue).unwrap();
                    }
                    self.queue.finish().unwrap();
                    // TESTING!
                    let mut l2_tasks: Vec<CombL2Task> = vec![];
                    for (id, l1_task) in l1_tasks.iter_mut().enumerate() {
                        l1_task.process_comb_l1(id, i*L2_LEN_STEP_SIZE, &mut l2_tasks);
                    }
                    let res_l2_task_num = unsafe {
                        let mut comb_l2_task_num_cl = [0];
                        self.queue.enqueue_read_buffer(
                                &mut self.comb_l2_task_num, CL_BLOCKING,
                                0, &mut comb_l2_task_num_cl[..], &[]).unwrap();
                        comb_l2_task_num_cl[0] as usize
                    };
                    {
                        assert_eq!(l2_tasks.len(), res_l2_task_num);
                        if res_l2_task_num != 0 {
                            let mut cl_l2_tasks =
                                vec![0u32; res_l2_task_num*self.comb_l2_task_len];
                            unsafe {
                                self.queue.enqueue_read_buffer(
                                        &mut self.comb_l2_tasks, CL_BLOCKING,
                                        0, &mut cl_l2_tasks[..], &[]).unwrap();
                            }
                            let mut exp_l2_tasks = l2_tasks.clone();
                            exp_l2_tasks.sort();
                            let mut res_l2_tasks = cl_l2_tasks
                                .chunks(self.comb_l2_task_len).map(|ch|
                                    CombL2Task {
                                        l1_task_id: *ch.first().unwrap() as usize,
                                        l2_filled_l2: Vec::from(&ch[1..1 + filled_clen*self.k]),
                                        l1_filled: Vec::from(
                                            &ch[ch.len() - filled_clen - 1..ch.len()-1]),
                                        l1: *ch.last().unwrap() as usize,
                                    }
                                ).collect::<Vec<_>>();
                            res_l2_tasks.sort();
                            for i in 0..res_l2_task_num {
                                assert_eq!(exp_l2_tasks[i], res_l2_tasks[i], "l2_task {}", i);
                            }
                        }
                    }
                    // TESTING!
                    
                    if res_l2_task_num != 0 {
                        // call process_comb_l2 kernel
                        unsafe {
                            let cl_l2_task_num = res_l2_task_num as cl_uint;
                            ExecuteKernel::new(&self.process_comb_l2_kernel)
                                    .set_arg(&cl_l2_task_num)
                                    .set_arg(&self.comb_tasks)
                                    .set_arg(&self.comb_l2_tasks)
                                    .set_arg(&self.results)
                                    .set_arg(&self.result_count)
                                    .set_local_work_size(self.group_len)
                                    .set_global_work_size((((res_l2_task_num + self.group_len - 1)
                                            / self.group_len)) * self.group_len)
                                    .enqueue_nd_range(&self.queue).unwrap();
                        }
                        self.queue.finish().unwrap();
                    }
                    
                    // TESTING!
                    l2_tasks.into_par_iter().for_each(|l2_task| {
                        l2_task.process_comb_l2(self.n, self.k, |i, j| {
                            result_count.fetch_add(1, atomic::Ordering::SeqCst);
                        });
                    });
                    let mut result_count_cl = [0];
                    unsafe {
                        self.queue.enqueue_read_buffer(&mut self.result_count, CL_BLOCKING,
                                        0, &mut result_count_cl[..], &[]).unwrap();
                    }
                    assert_eq!(result_count.load(atomic::Ordering::SeqCst),
                            result_count_cl[0]);
                    let result_count_cl_val = result_count_cl[0];
                    if result_count_cl_val > cl_result_count_old &&
                            cl_result_count_old < u64::try_from(self.max_results).unwrap() {
                        let result_pos = usize::try_from(cl_result_count_old).unwrap();
                        let result_count_cl_val = std::cmp::min(result_count_cl_val,
                                    self.max_results as u64);
                        let result_len = usize::try_from(
                                    result_count_cl_val - cl_result_count_old).unwrap();
                        let mut cl_results = vec![0; result_len*self.k];
                        println!("New results: {} {}", result_pos, result_len);
                        unsafe {
                            self.queue.enqueue_read_buffer(&mut self.results, CL_BLOCKING,
                                        4*result_pos*self.k, &mut cl_results[..], &[]).unwrap();
                        }
                        for ch in cl_results.chunks(self.k) {
                            println!("Result {}: {} {:?}", self.n, self.k, ch);
                        }
                    }
                    cl_result_count_old = result_count_cl_val;
                    // TESTING!
                }
                
                l1_tasks.clear();
                count = 0;
                cl_combs.clear();
                println!("CCX: {:?}", final_comb);
            }
            
            if !has_next {
                break;
            }
        }
        println!("Total results: {}", result_count.load(atomic::Ordering::SeqCst));
    }
    
    fn calc_cl(&mut self) -> bool {
        let filled_clen = (self.n + 31) >> 5;
        let mut count = 0;
        
        let mut comb_iter = CombineIter::new(self.k - 2, self.n - 2);
        let mut final_comb = vec![0; self.k];
        
        let task_num = self.task_num;
        let mut cl_combs = vec![];
        let mut cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
                
        println!("Calc CL: {} {}", self.n, self.k);
        {
            let result_count_cl = [0];
            unsafe {
                self.queue.enqueue_write_buffer(&mut self.result_count, CL_BLOCKING,
                                0, &result_count_cl[..], &[]).unwrap();
            }
        }
        
        let mut cl_result_count_old = 0;
        
        // compare results
        loop {
            let comb = comb_iter.get();
            
            let has_next_1 = comb[0] == 0 && comb.get(1).copied().unwrap_or(1) == 1;
            
            if has_next_1 {
                final_comb[0..self.k-2].copy_from_slice(comb);
                final_comb[self.k-2] = final_comb[self.k-3] + 1;
                final_comb[self.k-1] = final_comb[self.k-3] + 2;
            }
            
            if has_next_1 {
                cl_combs.extend(final_comb.iter().map(|x| u32::try_from(*x).unwrap()));
                count += 1;
            }
            
            let has_next = has_next_1 && comb_iter.next();
            
            if !has_next || count == task_num {
                unsafe {
                    // call init kernel
                    self.queue.enqueue_write_buffer(&mut self.combs, CL_BLOCKING,
                                0, &cl_combs, &[]).unwrap();
                    let cl_task_num = count as cl_uint;
                    // println!("Count xxx: {} {} {}", cl_task_num,
                    //          (((count + self.group_len - 1)
                    //                 / self.group_len)) * self.group_len,
                    //          cl_combs.len()
                    //          );
                    // call init_kernel
                    ExecuteKernel::new(&self.init_sum_fill_diff_change_kernel)
                            .set_arg(&cl_task_num)
                            .set_arg(&self.combs)
                            .set_arg(&self.comb_tasks)
                            .set_local_work_size(self.group_len)
                            .set_global_work_size((((count + self.group_len - 1)
                                    / self.group_len)) * self.group_len)
                            .enqueue_nd_range(&self.queue).unwrap();
                    self.queue.finish().unwrap();
                }
                
                let max_step_num = (self.n + L2_LEN_STEP_SIZE - 1) / L2_LEN_STEP_SIZE;
                for i in (0..max_step_num).rev() {
                    //println!("Cyyy: {}", i*L2_LEN_STEP_SIZE);
                    let cl_min_iter = (i*L2_LEN_STEP_SIZE) as cl_uint;
                    let cl_task_num = count as cl_uint;
                    // call process_comb_l1 kernel
                    unsafe {
                        let comb_l2_task_num_cl = [0];
                        self.queue.enqueue_write_buffer(
                                &mut self.comb_l2_task_num, CL_BLOCKING,
                                0, &comb_l2_task_num_cl[..], &[]).unwrap();
                        ExecuteKernel::new(&self.process_comb_l1_kernel)
                                .set_arg(&cl_task_num)
                                .set_arg(&cl_min_iter)
                                .set_arg(&self.comb_tasks)
                                .set_arg(&self.comb_l2_tasks)
                                .set_arg(&self.comb_l2_task_num)
                                .set_local_work_size(self.group_len)
                                .set_global_work_size((((count + self.group_len - 1)
                                        / self.group_len)) * self.group_len)
                                .enqueue_nd_range(&self.queue).unwrap();
                    }
                    self.queue.finish().unwrap();
                    let res_l2_task_num = unsafe {
                        let mut comb_l2_task_num_cl = [0];
                        self.queue.enqueue_read_buffer(
                                &mut self.comb_l2_task_num, CL_BLOCKING,
                                0, &mut comb_l2_task_num_cl[..], &[]).unwrap();
                        comb_l2_task_num_cl[0] as usize
                    };
                    
                    if res_l2_task_num != 0 {
                        // call process_comb_l2 kernel
                        unsafe {
                            let cl_l2_task_num = res_l2_task_num as cl_uint;
                            ExecuteKernel::new(&self.process_comb_l2_kernel)
                                    .set_arg(&cl_l2_task_num)
                                    .set_arg(&self.comb_tasks)
                                    .set_arg(&self.comb_l2_tasks)
                                    .set_arg(&self.results)
                                    .set_arg(&self.result_count)
                                    .set_local_work_size(self.group_len)
                                    .set_global_work_size((((res_l2_task_num + self.group_len - 1)
                                            / self.group_len)) * self.group_len)
                                    .enqueue_nd_range(&self.queue).unwrap();
                        }
                        self.queue.finish().unwrap();
                    }
                    
                    let mut result_count_cl = [0];
                    unsafe {
                        self.queue.enqueue_read_buffer(&mut self.result_count, CL_BLOCKING,
                                        0, &mut result_count_cl[..], &[]).unwrap();
                    }
                    let result_count_cl_val = result_count_cl[0];
                    if (result_count_cl_val > cl_result_count_old &&
                            cl_result_count_old < u64::try_from(self.max_results).unwrap()) {
                        let result_pos = usize::try_from(cl_result_count_old).unwrap();
                        let result_count_cl_val = std::cmp::min(result_count_cl_val,
                                    self.max_results as u64);
                        let result_len = usize::try_from(
                                    result_count_cl_val - cl_result_count_old).unwrap();
                        let mut cl_results = vec![0; result_len*self.k];
                        unsafe {
                            self.queue.enqueue_read_buffer(&mut self.results, CL_BLOCKING,
                                        4*result_pos*self.k, &mut cl_results[..], &[]).unwrap();
                        }
                        for ch in cl_results.chunks(self.k) {
                            println!("Result {}: {} {:?}", self.n, self.k, ch);
                        }
                    }
                    cl_result_count_old = result_count_cl_val;
                }
                
                count = 0;
                cl_combs.clear();
                eprintln!("Tasks: {} {}: {:?}", self.n, self.k, final_comb);
            }
            
            if !has_next {
                break;
            }
        }
        
        if cl_result_count_old != 0 {
            println!("Total results {} {}: {}", self.n, self.k, cl_result_count_old);
            true
        } else {
            false
        }
    }
}

// OpenCL stuff - end

fn get_sum_numbers(k: usize, filled_l1l2_sums: &mut [usize]) {
    let mut numr_iter = CombineWithRepIter::new(k, k);
    loop {
        let numc = numr_iter.get();
        
        // let fixsum = fix_sh + sum;
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
            //comb_filled[fixsum >> 6] |= 1u64 << (fixsum & 63);
        } else {
            if l1count != 0 {
                if l2count == 0 {
                    //filled_l1[filled_clen*(l1count-1) + (fixsum >> 6)] |= 1u64 << (fixsum & 63);
                } else {
                    filled_l1l2_sums[k*(l1count-1) + (l2count-1)] += 1;
                }
            } else if l2count != 0 {
                //filled_l2[filled_clen*(l2count-1) + (fixsum >> 6)] |= 1u64 << (fixsum & 63);
            }
        }
        if !numr_iter.next() {
            break;
        }
    }
}

fn gen_l1l2_tables() {
    for k in 5..10 {
        let mut l1l2_sum_numbers = vec![0; k*k];
        let mut l1l2_sumsum_pos = vec![0; k*k];
        get_sum_numbers(k, &mut l1l2_sum_numbers);
        for i in 1..k*k {
            l1l2_sumsum_pos[i] += l1l2_sumsum_pos[i-1] + l1l2_sum_numbers[i-1];
        }
        let sumsum = l1l2_sum_numbers.iter().sum::<usize>();
        println!("sum numbers: {}: {:?} - {}\nsumsumpos: {:?}", k, l1l2_sum_numbers,
                sumsum, l1l2_sumsum_pos);
        let mut l1l2_ij_table = vec![];
        {
            let mut p = 0;
            for i in 0..sumsum {
                while p+1 < l1l2_sumsum_pos.len() && i == l1l2_sumsum_pos[p] {
                    p += 1;
                }
                l1l2_ij_table.push(((p-1)/k, (p-1)%k));
            }
        }
        // println!("l1l2_ij_table {:?}", l1l2_ij_table.iter().map(|(i,j)|
        //         format!("{{{},{}}}", i, j)).collect::<Vec<_>>()
        // );
        println!("constant uchar l1l2_ij_table[][2] = {{");
        for ch in l1l2_ij_table.chunks(10) {
            for (i,j) in ch {
                print!("  {{{},{}}},", i, j);
            }
            println!("");
        }
        println!("\n}};");
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let n_start: usize = args.next().expect("Required n_start argument")
        .parse().expect("Required n_start argument");
    let n_end: usize = args.next().expect("Required n_end argument")
        .parse().expect("Required n_end argument");
    for i in n_start..n_end {
        // find k_start
        let ks = (1..64).find(|&x| {
            let max_n = usize::try_from(combinations(x as u64, x+x-1 as u64)).unwrap();
            //writeln!(io::stdout().lock(), "KSmax {}: {}", i, max_n);
            max_n >= i
        }).unwrap().try_into().unwrap();
        for k in ks..64 {
            let mut clnwork = CLNWork::new(0, i, k).unwrap();
            if clnwork.calc_cl() {
                break;
            }
        }
    }
    // {
    //     let mut clnwork = CLNWork::new(0, 544, 7).unwrap();
    //     //clnwork.test_init_kernel();
    //     clnwork.test_calc();
    //     //clnwork.test_calc_cl();
    //     clnwork.calc_cl();
    // }
    // gen_l1l2_tables();
}
