//! Auto-vectorization hints and compiler optimizations
//!
//! This module provides compiler hints, attributes, and patterns that help
//! the Rust compiler generate better vectorized code automatically.

use crate::{Result, VeritasError};
use std::hint;

/// Macro for loop unrolling hint
#[macro_export]
macro_rules! unroll_loop {
    (for $i:ident in 0..$n:expr => $body:expr) => {
        match $n {
            2 => {
                let $i = 0; $body;
                let $i = 1; $body;
            }
            4 => {
                let $i = 0; $body;
                let $i = 1; $body;
                let $i = 2; $body;
                let $i = 3; $body;
            }
            8 => {
                let $i = 0; $body;
                let $i = 1; $body;
                let $i = 2; $body;
                let $i = 3; $body;
                let $i = 4; $body;
                let $i = 5; $body;
                let $i = 6; $body;
                let $i = 7; $body;
            }
            _ => {
                for $i in 0..$n {
                    $body
                }
            }
        }
    };
}

/// Macro for prefetch hints
#[macro_export]
macro_rules! prefetch_read {
    ($ptr:expr) => {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch($ptr as *const i8, 0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::__prefetch;
            __prefetch($ptr as *const i8);
        }
    };
}

/// Vectorization-friendly loop patterns
pub struct VectorizedOps;

impl VectorizedOps {
    /// Dot product with auto-vectorization hints
    #[inline(always)]
    #[target_feature(enable = "sse2")]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub unsafe fn dot_product_autovec(a: &[f32], b: &[f32]) -> f32 {
        // Ensure alignment and length requirements
        debug_assert_eq!(a.len(), b.len());
        debug_assert!(a.len() % 8 == 0);
        
        let len = a.len();
        let mut sum = 0.0f32;
        
        // Use restrict pointers to help compiler
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        
        // Main vectorized loop
        let mut i = 0;
        while i < len {
            // Prefetch next cache lines
            if i + 16 < len {
                prefetch_read!(a_ptr.add(i + 16));
                prefetch_read!(b_ptr.add(i + 16));
            }
            
            // Process 8 elements at a time
            let mut local_sum = 0.0f32;
            
            // Help compiler unroll and vectorize
            #[allow(clippy::needless_range_loop)]
            for j in 0..8 {
                local_sum += *a_ptr.add(i + j) * *b_ptr.add(i + j);
            }
            
            sum += local_sum;
            i += 8;
        }
        
        sum
    }
    
    /// Element-wise addition with auto-vectorization
    #[inline(always)]
    pub fn add_autovec(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let len = a.len();
        
        // Process aligned chunks
        let chunks = len / 16;
        let remainder = len % 16;
        
        // Main vectorized loop
        for i in 0..chunks {
            let offset = i * 16;
            
            // Use slice patterns to help compiler vectorize
            let a_chunk = &a[offset..offset + 16];
            let b_chunk = &b[offset..offset + 16];
            let result_chunk = &mut result[offset..offset + 16];
            
            // Explicit vectorization pattern
            for j in 0..16 {
                result_chunk[j] = a_chunk[j] + b_chunk[j];
            }
        }
        
        // Handle remainder
        let remainder_offset = chunks * 16;
        for i in 0..remainder {
            result[remainder_offset + i] = a[remainder_offset + i] + b[remainder_offset + i];
        }
        
        Ok(())
    }
    
    /// Matrix multiplication with cache-friendly access pattern
    #[inline(always)]
    pub fn matmul_autovec(
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<()> {
        if a.len() != rows_a * cols_a || b.len() != cols_a * cols_b {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Initialize result
        result.fill(0.0);
        
        // Cache-friendly loop ordering (ikj instead of ijk)
        const TILE_SIZE: usize = 64;
        
        // Tiled matrix multiplication
        for i_tile in (0..rows_a).step_by(TILE_SIZE) {
            for k_tile in (0..cols_a).step_by(TILE_SIZE) {
                for j_tile in (0..cols_b).step_by(TILE_SIZE) {
                    // Process tile
                    let i_end = (i_tile + TILE_SIZE).min(rows_a);
                    let k_end = (k_tile + TILE_SIZE).min(cols_a);
                    let j_end = (j_tile + TILE_SIZE).min(cols_b);
                    
                    for i in i_tile..i_end {
                        for k in k_tile..k_end {
                            let a_val = a[i * cols_a + k];
                            
                            // Vectorizable inner loop
                            let row_offset = i * cols_b;
                            let b_row_offset = k * cols_b;
                            
                            // Process 8 elements at a time
                            let mut j = j_tile;
                            while j + 8 <= j_end {
                                for jj in 0..8 {
                                    result[row_offset + j + jj] += a_val * b[b_row_offset + j + jj];
                                }
                                j += 8;
                            }
                            
                            // Handle remainder
                            while j < j_end {
                                result[row_offset + j] += a_val * b[b_row_offset + j];
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Reduction operation with vectorization hints
    #[inline(always)]
    pub fn reduce_sum_autovec(data: &[f32]) -> f32 {
        // Use tree reduction pattern
        let len = data.len();
        
        if len == 0 {
            return 0.0;
        }
        
        // Process 8 elements at a time
        let chunks = len / 8;
        let remainder = len % 8;
        
        // Accumulate in multiple variables to avoid dependencies
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        
        for i in 0..chunks {
            let offset = i * 8;
            sum0 += data[offset];
            sum1 += data[offset + 1];
            sum2 += data[offset + 2];
            sum3 += data[offset + 3];
            sum0 += data[offset + 4];
            sum1 += data[offset + 5];
            sum2 += data[offset + 6];
            sum3 += data[offset + 7];
        }
        
        // Combine partial sums
        let mut total = sum0 + sum1 + sum2 + sum3;
        
        // Handle remainder
        let remainder_offset = chunks * 8;
        for i in 0..remainder {
            total += data[remainder_offset + i];
        }
        
        total
    }
    
    /// Convolution with vectorization hints
    #[inline(always)]
    pub fn convolve_1d_autovec(
        signal: &[f32],
        kernel: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let result_len = result.len();
        
        if result_len != signal_len + kernel_len - 1 {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Clear result
        result.fill(0.0);
        
        // Reverse kernel for convolution
        let mut reversed_kernel = vec![0.0f32; kernel_len];
        for i in 0..kernel_len {
            reversed_kernel[i] = kernel[kernel_len - 1 - i];
        }
        
        // Main convolution loop
        for i in 0..signal_len {
            let signal_val = signal[i];
            
            // Vectorizable inner loop
            let start = i;
            let end = (i + kernel_len).min(result_len);
            
            // Process 4 elements at a time
            let mut j = start;
            while j + 4 <= end {
                for k in 0..4 {
                    result[j + k] += signal_val * reversed_kernel[j + k - i];
                }
                j += 4;
            }
            
            // Handle remainder
            while j < end {
                result[j] += signal_val * reversed_kernel[j - i];
                j += 1;
            }
        }
        
        Ok(())
    }
}

/// Compiler optimization attributes
pub mod attributes {
    /// Mark function as hot (frequently called)
    #[macro_export]
    macro_rules! hot {
        ($item:item) => {
            #[inline(always)]
            #[cfg_attr(feature = "unstable", optimize(speed))]
            $item
        };
    }
    
    /// Mark function as cold (rarely called)
    #[macro_export]
    macro_rules! cold {
        ($item:item) => {
            #[inline(never)]
            #[cold]
            $item
        };
    }
    
    /// Force no vectorization (for debugging)
    #[macro_export]
    macro_rules! no_vectorize {
        ($item:item) => {
            #[cfg_attr(feature = "unstable", optimize(size))]
            $item
        };
    }
}

/// Memory alignment helpers
pub mod alignment {
    use std::alloc::{alloc, dealloc, Layout};
    use std::mem;
    use std::ptr;
    use std::slice;
    
    /// Aligned vector for SIMD operations
    #[repr(align(64))]
    pub struct AlignedVec<T> {
        ptr: *mut T,
        len: usize,
        capacity: usize,
    }
    
    impl<T> AlignedVec<T> {
        /// Create new aligned vector with capacity
        pub fn with_capacity(capacity: usize) -> Self {
            let layout = Layout::from_size_align(
                capacity * mem::size_of::<T>(),
                64, // 64-byte alignment for cache lines
            ).unwrap();
            
            let ptr = unsafe { alloc(layout) as *mut T };
            
            Self {
                ptr,
                len: 0,
                capacity,
            }
        }
        
        /// Push element to vector
        pub fn push(&mut self, value: T) {
            if self.len >= self.capacity {
                panic!("AlignedVec capacity exceeded");
            }
            
            unsafe {
                ptr::write(self.ptr.add(self.len), value);
            }
            self.len += 1;
        }
        
        /// Get slice reference
        pub fn as_slice(&self) -> &[T] {
            unsafe { slice::from_raw_parts(self.ptr, self.len) }
        }
        
        /// Get mutable slice reference
        pub fn as_mut_slice(&mut self) -> &mut [T] {
            unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
        }
        
        /// Check if pointer is aligned
        pub fn is_aligned(&self) -> bool {
            self.ptr as usize % 64 == 0
        }
    }
    
    impl<T> Drop for AlignedVec<T> {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    // Drop all elements
                    for i in 0..self.len {
                        ptr::drop_in_place(self.ptr.add(i));
                    }
                    
                    // Deallocate memory
                    let layout = Layout::from_size_align(
                        self.capacity * mem::size_of::<T>(),
                        64,
                    ).unwrap();
                    dealloc(self.ptr as *mut u8, layout);
                }
            }
        }
    }
}

/// Loop optimization patterns
pub mod loops {
    /// Strip mining for better cache usage
    pub fn strip_mine<F>(n: usize, strip_size: usize, mut f: F)
    where
        F: FnMut(usize, usize),
    {
        let full_strips = n / strip_size;
        let remainder = n % strip_size;
        
        // Process full strips
        for i in 0..full_strips {
            f(i * strip_size, strip_size);
        }
        
        // Process remainder
        if remainder > 0 {
            f(full_strips * strip_size, remainder);
        }
    }
    
    /// Software pipelining helper
    pub struct PipelinedLoop<T> {
        buffer: Vec<T>,
        stages: usize,
    }
    
    impl<T: Default + Clone> PipelinedLoop<T> {
        pub fn new(stages: usize) -> Self {
            Self {
                buffer: vec![T::default(); stages],
                stages,
            }
        }
        
        pub fn execute<F1, F2, F3>(
            &mut self,
            n: usize,
            mut stage1: F1,
            mut stage2: F2,
            mut stage3: F3,
        ) where
            F1: FnMut(usize) -> T,
            F2: FnMut(&T) -> T,
            F3: FnMut(&T),
        {
            // Prologue
            for i in 0..self.stages.min(n) {
                self.buffer[i % self.stages] = stage1(i);
            }
            
            // Steady state
            for i in self.stages..n {
                let idx = i % self.stages;
                let prev_idx = (i - 1) % self.stages;
                
                // Execute stages in parallel
                let temp = stage2(&self.buffer[prev_idx]);
                stage3(&temp);
                self.buffer[idx] = stage1(i);
            }
            
            // Epilogue
            for i in n..n + self.stages - 1 {
                let prev_idx = (i - 1) % self.stages;
                let temp = stage2(&self.buffer[prev_idx]);
                stage3(&temp);
            }
        }
    }
}

/// Branch prediction hints
pub mod branches {
    /// Likely branch hint
    #[inline(always)]
    pub fn likely(cond: bool) -> bool {
        if std::intrinsics::likely(cond) {
            true
        } else {
            false
        }
    }
    
    /// Unlikely branch hint
    #[inline(always)]
    pub fn unlikely(cond: bool) -> bool {
        if std::intrinsics::unlikely(cond) {
            true
        } else {
            false
        }
    }
    
    /// Branchless select
    #[inline(always)]
    pub fn select<T: Copy>(cond: bool, a: T, b: T) -> T {
        // Compiler will optimize this to cmov on x86
        if cond { a } else { b }
    }
    
    /// Branchless min
    #[inline(always)]
    pub fn branchless_min(a: f32, b: f32) -> f32 {
        let mask = (a < b) as i32 as f32;
        a * mask + b * (1.0 - mask)
    }
    
    /// Branchless max
    #[inline(always)]
    pub fn branchless_max(a: f32, b: f32) -> f32 {
        let mask = (a > b) as i32 as f32;
        a * mask + b * (1.0 - mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::alignment::AlignedVec;
    
    #[test]
    fn test_aligned_vec() {
        let mut vec: AlignedVec<f32> = AlignedVec::with_capacity(64);
        assert!(vec.is_aligned());
        
        for i in 0..64 {
            vec.push(i as f32);
        }
        
        let slice = vec.as_slice();
        assert_eq!(slice.len(), 64);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[63], 63.0);
    }
    
    #[test]
    fn test_vectorized_add() {
        let a = vec![1.0f32; 64];
        let b = vec![2.0f32; 64];
        let mut result = vec![0.0f32; 64];
        
        VectorizedOps::add_autovec(&a, &b, &mut result).unwrap();
        
        for val in &result {
            assert_eq!(*val, 3.0);
        }
    }
    
    #[test]
    fn test_reduce_sum() {
        let data = vec![1.0f32; 100];
        let sum = VectorizedOps::reduce_sum_autovec(&data);
        assert_eq!(sum, 100.0);
    }
    
    #[test]
    fn test_strip_mining() {
        let mut sum = 0;
        loops::strip_mine(100, 16, |start, count| {
            sum += count;
        });
        assert_eq!(sum, 100);
    }
    
    #[test]
    fn test_branchless_operations() {
        use branches::*;
        
        assert_eq!(branchless_min(3.0, 5.0), 3.0);
        assert_eq!(branchless_max(3.0, 5.0), 5.0);
        assert_eq!(select(true, 10, 20), 10);
        assert_eq!(select(false, 10, 20), 20);
    }
}