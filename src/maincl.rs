use std::env;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_uint, cl_ulong, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;
mod utils;
use utils::*;

// CPU routines

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

// for 32-bit only testing

fn init_sum_fill_diff_change_32(n: usize, comb: &[usize], comb_filled: &mut [u32],
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

// CPU routines - end

const PROGRAM_SOURCE: &str = r#"
// CONST_K - k value
// CONST_N - n value
// FIX_SH - (32-n)%32 (if n%32!=0 else 0)
// FCLEN - n/32
// WFLEN - wavefront length

#define GROUP_LEN (WFLEN)

#if CONST_K < 5 || CONST_K > 9
#error "Unsupported CONST_K"
#endif

#if CONST_K == 5
#define L1L2_TOTAL_SUMS (35)
constant uint l1l2_sum_pos[25] = {
    0, 10, 16, 19, 20, 20, 26, 29, 30, 30, 30, 33, 34, 34, 34, 34, 35,
    35, 35, 35, 35, 35, 35, 35, 35
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
    uint comb_filled[FCLEN];
    uint filled_l1[FCLEN*CONST_K];
    uint filled_l1l2_sums[L1L2_TOTAL_SUMS];
    uint filled_l2[FCLEN*CONST_K];
    uint to_process;
} CombTask;

kernel void init_sum_fill_diff_change(uint task_num, global const uint* combs,
                    global const uint* free_list, global CombTask* comb_tasks) {
    const uint gid = get_global_id(0);
    if (gid >= task_num)
        return;
    const uint cbidx = free_list[gid];
    const global uint* comb = combs + CONST_K*gid;
    global CombTask* comb_task = &comb_tasks[gid];
    //
    uint i;
    for (i = 0; i < FCLEN; i++)
        comb_tasks->comb_filled[i] = 0;
    for (i = 0; i < FCLEN*CONST_K; i++) {
        comb_task->filled_l1[i] = 0;
        comb_task->filled_l2[i] = 0;
    }
    for (i = 0; i < L1L2_TOTAL_SUMS; i++)
        comb_tasks->filled_l1l2_sums[i] = 0;
    comb_tasks->to_process = 1;
    // initialize iterator
    uint numcomb[CONST_K];
    for (i = 0; i < CONST_K; i++)
        numcomb[i] = 0;
    
    local uint l1l2idx_idx_group[GROUP_LEN*CONST_K*CONST_K];
    local uint* l1l2idx_idx = l1l2idx_idx_group + CONST_K*CONST_K*(get_local_id(0));
    for (i = 0; i < CONST_K*CONST_K; i++)
        l1l2idx_idx[i] = 0;
    
    // main loop
    while (true) {
    //uint ii;
    //for (ii = 0; ii < 100; ii++) {
        // fill up comb task
        uint sum = 0;
        for (i = 0; i < CONST_K; i++)
            sum += comb[numcomb[i]];
        
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
            comb_tasks->comb_filled[fixsum >> 5] |= 1 << (fixsum & 31);
        } else {
            if (l1count != 0) {
                if (l2count == 0)
                    comb_tasks->filled_l1[FCLEN*(l1count-1) + (fixsum>>5)] |= 1<<(fixsum & 31);
                else {
                    const uint vec_id = CONST_K*(l1count-1) + (l2count-1);
                    comb_tasks->filled_l1l2_sums[
                        l1l2_sum_pos[vec_id] + l1l2idx_idx[vec_id]]  = sum;
                    l1l2idx_idx[vec_id] += 1;
                }
            } else if (l2count != 0) {
                comb_tasks->filled_l2[FCLEN*(l2count-1) + (fixsum>>5)] |= 1<<(fixsum & 31);
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

kernel void process_comb_l1l2(uint task_num, global uint* free_list,
            global uint* free_list_num,
            global CombTask* comb_tasks,
            global uint* results,
            global uint* result_count) {
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
    process_comb_l1l2_kernel: Kernel,
    group_num: usize,
    task_num: usize,
    comb_task_len: usize,
    combs: Buffer<cl_uint>,
    free_list: Buffer<cl_uint>,
    free_list_num: Buffer<cl_uint>,
    comb_tasks: Buffer<cl_uint>,
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
        
        let fix_sh =  if (n & 31) != 0 {
            (u32::BITS - ((n as u32) & 31)) as usize
        } else {
            0
        };
        
        let fclen = (n + 31) >> 5;
        let prog_opts = format!(
                "-DCONST_N=({}) -DCONST_K=({}) -DFIX_SH=({}) -DFCLEN=({}) -DWFLEN=({})",
                n, k, fix_sh, fclen, 64);
        let program = match Program::create_and_build_from_source(&context, PROGRAM_SOURCE,
                &prog_opts) {
            Ok(program) => program,
            Err(err) => {
                panic!("Can't compile program: {}", err);
            }
        };
        let init_kernel = Kernel::create(&program, "init_sum_fill_diff_change")?;
        let process_kernel = Kernel::create(&program, "process_comb_l1l2")?;
        
        let group_num = 1024;
        let l1l2_total_sums = match k {
            5 => 32,
            6 => 126,
            7 => 462,
            8 => 1716,
            9 => 6435,
            _ => { panic!("Unsupported k"); }
        };
        let comb_task_len = fclen + k*fclen*2 + l1l2_total_sums + 1;
        let task_num = (group_num + fclen-1) / fclen;
        
        let combs = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                            k * task_num, ptr::null_mut())?
        };
        let comb_tasks = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                    comb_task_len * task_num, ptr::null_mut())?
        };
        let free_list = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                            task_num, ptr::null_mut())?
        };
        let results = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE,
                            3 * 10000, ptr::null_mut())?
        };
        let free_list_num = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut())?
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
           process_comb_l1l2_kernel: process_kernel,
           group_num,
           comb_task_len,
           task_num,
           combs,
           comb_tasks,
           free_list,
           free_list_num,
           results,
           result_count
        })
    }
    
    fn test_init_kernel(&mut self) -> Result<()> {
        let filled_clen = (self.n + 31) >> 5;
        let mut count = 0;
        let mut exp_cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
        let mut cl_combs: Vec<cl_uint> = vec![0; self.k*self.task_num];
        let mut cl_comb_tasks: Vec<cl_uint> = vec![0; self.comb_task_len*self.task_num];
        let mut comb_task_data: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = vec![
                        (vec![], vec![], vec![]); self.task_num];
        
        {
            let free_list = (0..(self.task_num as cl_uint)).collect::<Vec<_>>();
            unsafe {
                self.queue.enqueue_write_buffer(&mut self.free_list, CL_BLOCKING,
                        0, &free_list, &[])?;
            }
            self.queue.finish()?;
        }
        
        let mut comb_iter = CombineIter::new(self.k - 2, self.n - 2);
        let mut final_comb = vec![0; self.k];
        let mut comb_filled = vec![0u32; filled_clen];
        let mut filled_l1 = vec![0u32; filled_clen*self.k];
        let mut filled_l1l2_sums = vec![vec![]; self.k*self.k];
        let mut filled_l2 = vec![0u32; filled_clen*self.k];
        
        // compare results
        loop {
            let comb = comb_iter.get();
            
            if comb[0] != 0 || comb.get(1).copied().unwrap_or(1) != 1 {
                break;
            }
            
            final_comb[0..self.k-2].copy_from_slice(comb);
            final_comb[self.k-2] = final_comb[self.k-3] + 1;
            final_comb[self.k-1] = final_comb[self.k-3] + 2;
            
            init_sum_fill_diff_change_32(self.n, &final_comb, &mut comb_filled,
                            &mut filled_l1, &mut filled_l1l2_sums, &mut filled_l2);
            
            {
                // put comb to cl_combs
                cl_combs[count*self.k..(count+1)*self.k].iter_mut().enumerate()
                        .for_each(|(i,x)| *x = final_comb[i] as cl_uint);
                // put to expected cl comb_task
                let mut exp_comb_task = &mut exp_cl_comb_tasks[
                        self.comb_task_len*count..self.comb_task_len*(count+1)];
                // copy comb_filled
                comb_filled.iter().enumerate().for_each(|(i, x)|
                    exp_comb_task[i] = *x as cl_uint);
                // copy filled_l1
                filled_l1.iter().enumerate().for_each(|(i, x)|
                    exp_comb_task[filled_clen + i] = *x as cl_uint);
                // copy filled_l1l2_sum
                let mut idx = filled_clen + filled_clen*self.k;
                for j in 0..self.k*self.k {
                    filled_l1l2_sums[j].iter().enumerate().for_each(|(i,x)|
                        exp_comb_task[idx + i] = *x as cl_uint);
                    idx += filled_l1l2_sums[j].len();
                }
                // copy filled_l2
                filled_l2.iter().enumerate().for_each(|(i, x)|
                    exp_comb_task[idx + i] = *x as cl_uint);
                *exp_comb_task.last_mut().unwrap() = 1;
            }
            
            let has_next = comb_iter.next();
            
            count += 1;
            if !has_next || count == self.task_num {
                unsafe {
                    self.queue.enqueue_write_buffer(&mut self.combs, CL_BLOCKING,
                                0, &cl_combs, &[])?;
                    let cl_task_num = count as cl_uint;
                    println!("NDrange: {} {}", count, self.task_num);
                    ExecuteKernel::new(&self.init_sum_fill_diff_change_kernel)
                            .set_arg(&cl_task_num)
                            .set_arg(&self.combs)
                            .set_arg(&self.free_list)
                            .set_arg(&self.comb_tasks)
                            .set_local_work_size(64)
                            .set_global_work_size((count + 63) >> 6)
                            .enqueue_nd_range(&self.queue)?;
                    // //
                    //             uint task_num, global const uint* combs,
                    // global const uint* free_list, global CombTask* comb_tasks
                    // get results
                    self.queue.finish()?;
                    self.queue.enqueue_read_buffer(&mut self.comb_tasks, CL_BLOCKING,
                                0, &mut cl_comb_tasks, &[])?;
                    self.queue.finish()?;
                    // compare results
                    for i in 0..count {
                        let exp_comb_task = &mut exp_cl_comb_tasks[
                                self.comb_task_len*i..self.comb_task_len*(i+1)];
                        let res_comb_task = &mut cl_comb_tasks[
                                self.comb_task_len*i..self.comb_task_len*(i+1)];
                        assert_eq!(exp_comb_task, res_comb_task, "comb_task {}", i);
                    }
                }
                self.queue.finish()?;
                // call init_kernel
                count = 0;
            }
            
            if !has_next {
                break;
            }
        }
        //
        
        // call kernel
        // end of OpenCL stuff
        Ok(())
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

fn main() {
    // let mut args = env::args().skip(1);
    // let n_start: usize = args.next().expect("Required n_start argument")
    //     .parse().expect("Required n_start argument");
    // let n_end: usize = args.next().expect("Required n_end argument")
    //     .parse().expect("Required n_end argument");
    // for i in n_start..n_end {
    //     //calc_min_sumn_to_fill_par_all_opencl(i);
    // }
    {
        let mut clnwork = CLNWork::new(0, 300, 7).unwrap();
        clnwork.test_init_kernel().unwrap();
    }
    // for k in 5..10 {
    //     let mut l1l2_sum_numbers = vec![0; k*k];
    //     let mut l1l2_sumsum_pos = vec![0; k*k];
    //     get_sum_numbers(k, &mut l1l2_sum_numbers);
    //     for i in 1..k*k {
    //         l1l2_sumsum_pos[i] += l1l2_sumsum_pos[i-1] + l1l2_sum_numbers[i-1];
    //     }
    //     println!("sum numbers: {}: {:?} - {}\nsumsumpos: {:?}", k, l1l2_sum_numbers,
    //             l1l2_sum_numbers.iter().sum::<usize>(),
    //             l1l2_sumsum_pos);
    // }
}
