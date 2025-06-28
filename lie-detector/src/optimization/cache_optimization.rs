//! Advanced CPU cache optimization techniques
//!
//! This module provides advanced cache optimization strategies including:
//! - Loop tiling and blocking
//! - Data structure padding and alignment
//! - Cache-aware algorithms
//! - Prefetching strategies

use crate::{Result, VeritasError};
use std::mem;
use std::arch::x86_64::*;

/// Cache line size (typically 64 bytes on modern CPUs)
pub const CACHE_LINE_SIZE: usize = 64;

/// L1 cache size hint (typical: 32KB)
pub const L1_CACHE_SIZE: usize = 32 * 1024;

/// L2 cache size hint (typical: 256KB)
pub const L2_CACHE_SIZE: usize = 256 * 1024;

/// L3 cache size hint (typical: 8MB)
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;

/// Cache-aligned data structure
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct CacheAligned<T> {
    pub data: T,
    _padding: [u8; 0], // Zero-sized for alignment
}

impl<T> CacheAligned<T> {
    /// Create new cache-aligned wrapper
    pub fn new(data: T) -> Self {
        Self {
            data,
            _padding: [],
        }
    }
    
    /// Get reference to inner data
    pub fn get(&self) -> &T {
        &self.data
    }
    
    /// Get mutable reference to inner data
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

/// Cache-optimized matrix with padding to avoid false sharing
#[repr(C)]
pub struct CacheOptimizedMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    padded_cols: usize, // Actual column count with padding
}

impl<T: Default + Clone> CacheOptimizedMatrix<T> {
    /// Create new cache-optimized matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        // Pad columns to cache line boundary
        let elements_per_cache_line = CACHE_LINE_SIZE / mem::size_of::<T>().max(1);
        let padded_cols = ((cols + elements_per_cache_line - 1) / elements_per_cache_line) 
            * elements_per_cache_line;
        
        let data = vec![T::default(); rows * padded_cols];
        
        Self {
            data,
            rows,
            cols,
            padded_cols,
        }
    }
    
    /// Get element at (row, col)
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> &T {
        debug_assert!(row < self.rows && col < self.cols);
        &self.data[row * self.padded_cols + col]
    }
    
    /// Get mutable element at (row, col)
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        debug_assert!(row < self.rows && col < self.cols);
        &mut self.data[row * self.padded_cols + col]
    }
    
    /// Get row as slice (without padding)
    pub fn row(&self, row: usize) -> &[T] {
        debug_assert!(row < self.rows);
        let start = row * self.padded_cols;
        &self.data[start..start + self.cols]
    }
}

/// Loop tiling for matrix operations
pub struct LoopTiling;

impl LoopTiling {
    /// Tiled matrix multiplication for better cache usage
    pub fn tiled_matmul<T: Default + Clone + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>(
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize, // rows of A
        n: usize, // cols of A, rows of B
        p: usize, // cols of B
        tile_size: usize,
    ) -> Result<()> {
        if a.len() != m * n || b.len() != n * p || c.len() != m * p {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Initialize result matrix
        c.fill(T::default());
        
        // Triple-nested tiled loops
        for i0 in (0..m).step_by(tile_size) {
            for j0 in (0..p).step_by(tile_size) {
                for k0 in (0..n).step_by(tile_size) {
                    // Compute tile boundaries
                    let i_max = (i0 + tile_size).min(m);
                    let j_max = (j0 + tile_size).min(p);
                    let k_max = (k0 + tile_size).min(n);
                    
                    // Process tile
                    for i in i0..i_max {
                        for k in k0..k_max {
                            let a_ik = a[i * n + k];
                            for j in j0..j_max {
                                c[i * p + j] = c[i * p + j] + a_ik * b[k * p + j];
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Tiled transpose for cache efficiency
    pub fn tiled_transpose<T: Copy>(
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
        tile_size: usize,
    ) -> Result<()> {
        if src.len() != rows * cols || dst.len() != rows * cols {
            return Err(VeritasError::DimensionMismatch);
        }
        
        for i0 in (0..rows).step_by(tile_size) {
            for j0 in (0..cols).step_by(tile_size) {
                let i_max = (i0 + tile_size).min(rows);
                let j_max = (j0 + tile_size).min(cols);
                
                for i in i0..i_max {
                    for j in j0..j_max {
                        dst[j * rows + i] = src[i * cols + j];
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate optimal tile size based on cache size
    pub fn optimal_tile_size<T>(cache_size: usize) -> usize {
        // Aim to fit 3 tiles in cache (for matmul: A tile, B tile, C tile)
        let element_size = mem::size_of::<T>();
        let tiles_in_cache = 3;
        let tile_elements = cache_size / (tiles_in_cache * element_size);
        (tile_elements as f64).sqrt() as usize
    }
}

/// Cache-aware algorithms
pub struct CacheAwareAlgorithms;

impl CacheAwareAlgorithms {
    /// Cache-friendly array sum using blocking
    pub fn blocked_sum(data: &[f32], block_size: usize) -> f32 {
        let mut total = 0.0f32;
        
        // Process complete blocks
        let num_blocks = data.len() / block_size;
        for block in 0..num_blocks {
            let start = block * block_size;
            let end = start + block_size;
            
            // Sum within block (fits in L1 cache)
            let mut block_sum = 0.0f32;
            for i in start..end {
                block_sum += data[i];
            }
            total += block_sum;
        }
        
        // Handle remainder
        let remainder_start = num_blocks * block_size;
        for i in remainder_start..data.len() {
            total += data[i];
        }
        
        total
    }
    
    /// Cache-oblivious recursive matrix transpose
    pub fn recursive_transpose<T: Copy>(
        src: &[T],
        dst: &mut [T],
        src_stride: usize,
        dst_stride: usize,
        rows: usize,
        cols: usize,
        row_offset: usize,
        col_offset: usize,
    ) {
        const THRESHOLD: usize = 16; // Base case size
        
        if rows <= THRESHOLD && cols <= THRESHOLD {
            // Base case: simple transpose
            for i in 0..rows {
                for j in 0..cols {
                    let src_idx = (row_offset + i) * src_stride + (col_offset + j);
                    let dst_idx = (col_offset + j) * dst_stride + (row_offset + i);
                    dst[dst_idx] = src[src_idx];
                }
            }
        } else if rows >= cols {
            // Split rows
            let mid = rows / 2;
            Self::recursive_transpose(
                src, dst, src_stride, dst_stride,
                mid, cols, row_offset, col_offset
            );
            Self::recursive_transpose(
                src, dst, src_stride, dst_stride,
                rows - mid, cols, row_offset + mid, col_offset
            );
        } else {
            // Split columns
            let mid = cols / 2;
            Self::recursive_transpose(
                src, dst, src_stride, dst_stride,
                rows, mid, row_offset, col_offset
            );
            Self::recursive_transpose(
                src, dst, src_stride, dst_stride,
                rows, cols - mid, row_offset, col_offset + mid
            );
        }
    }
}

/// Prefetching strategies
pub struct Prefetcher;

impl Prefetcher {
    /// Software prefetch for read
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn prefetch_read<T>(ptr: *const T, distance: usize) {
        let addr = ptr.add(distance) as *const i8;
        _mm_prefetch(addr, _MM_HINT_T0); // Prefetch to all cache levels
    }
    
    /// Software prefetch for write
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn prefetch_write<T>(ptr: *const T, distance: usize) {
        let addr = ptr.add(distance) as *const i8;
        _mm_prefetch(addr, _MM_HINT_T0);
    }
    
    /// Prefetch entire cache line
    #[inline(always)]
    pub fn prefetch_line<T>(data: &[T], index: usize) {
        if index < data.len() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                Self::prefetch_read(&data[index], 0);
            }
        }
    }
    
    /// Stride prefetching for array traversal
    pub fn stride_prefetch<T, F>(data: &[T], stride: usize, prefetch_distance: usize, mut process: F)
    where
        F: FnMut(&T),
    {
        let len = data.len();
        
        for i in (0..len).step_by(stride) {
            // Prefetch future elements
            if i + prefetch_distance < len {
                Self::prefetch_line(data, i + prefetch_distance);
            }
            
            // Process current element
            process(&data[i]);
        }
    }
}

/// Structure of Arrays (SoA) for better cache usage
pub struct SoAContainer<T1, T2, T3> {
    pub field1: Vec<T1>,
    pub field2: Vec<T2>,
    pub field3: Vec<T3>,
}

impl<T1, T2, T3> SoAContainer<T1, T2, T3> {
    pub fn new() -> Self {
        Self {
            field1: Vec::new(),
            field2: Vec::new(),
            field3: Vec::new(),
        }
    }
    
    pub fn push(&mut self, v1: T1, v2: T2, v3: T3) {
        self.field1.push(v1);
        self.field2.push(v2);
        self.field3.push(v3);
    }
    
    pub fn len(&self) -> usize {
        self.field1.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.field1.is_empty()
    }
}

/// Cache blocking for 2D convolution
pub struct BlockedConvolution;

impl BlockedConvolution {
    pub fn convolve_2d_blocked(
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        input_height: usize,
        input_width: usize,
        kernel_size: usize,
        block_size: usize,
    ) -> Result<()> {
        let output_height = input_height - kernel_size + 1;
        let output_width = input_width - kernel_size + 1;
        
        if output.len() != output_height * output_width {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Clear output
        output.fill(0.0);
        
        // Process in blocks
        for by in (0..output_height).step_by(block_size) {
            for bx in (0..output_width).step_by(block_size) {
                let by_end = (by + block_size).min(output_height);
                let bx_end = (bx + block_size).min(output_width);
                
                // Process block
                for y in by..by_end {
                    for x in bx..bx_end {
                        let mut sum = 0.0f32;
                        
                        // Apply kernel
                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let input_y = y + ky;
                                let input_x = x + kx;
                                let input_idx = input_y * input_width + input_x;
                                let kernel_idx = ky * kernel_size + kx;
                                
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                        
                        output[y * output_width + x] = sum;
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Memory access pattern analyzer
pub struct AccessPatternAnalyzer {
    pub sequential_accesses: usize,
    pub random_accesses: usize,
    pub stride_accesses: usize,
    last_address: Option<usize>,
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            sequential_accesses: 0,
            random_accesses: 0,
            stride_accesses: 0,
            last_address: None,
        }
    }
    
    pub fn record_access<T>(&mut self, ptr: *const T) {
        let addr = ptr as usize;
        
        if let Some(last) = self.last_address {
            let diff = addr.abs_diff(last);
            
            if diff == mem::size_of::<T>() {
                self.sequential_accesses += 1;
            } else if diff % mem::size_of::<T>() == 0 && diff < CACHE_LINE_SIZE {
                self.stride_accesses += 1;
            } else {
                self.random_accesses += 1;
            }
        }
        
        self.last_address = Some(addr);
    }
    
    pub fn report(&self) -> String {
        format!(
            "Access Pattern: Sequential: {}, Stride: {}, Random: {}",
            self.sequential_accesses,
            self.stride_accesses,
            self.random_accesses
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_aligned() {
        let aligned: CacheAligned<[f32; 16]> = CacheAligned::new([0.0; 16]);
        let ptr = aligned.get() as *const _ as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }
    
    #[test]
    fn test_cache_optimized_matrix() {
        let matrix: CacheOptimizedMatrix<f32> = CacheOptimizedMatrix::new(10, 10);
        assert!(matrix.padded_cols >= 10);
        assert_eq!(matrix.padded_cols % (CACHE_LINE_SIZE / mem::size_of::<f32>()), 0);
    }
    
    #[test]
    fn test_tiled_matmul() {
        let a = vec![1.0f32; 64 * 64];
        let b = vec![2.0f32; 64 * 64];
        let mut c = vec![0.0f32; 64 * 64];
        
        let tile_size = LoopTiling::optimal_tile_size::<f32>(L2_CACHE_SIZE);
        LoopTiling::tiled_matmul(&a, &b, &mut c, 64, 64, 64, tile_size).unwrap();
        
        // Check first element
        assert_eq!(c[0], 128.0); // 64 * 1.0 * 2.0
    }
    
    #[test]
    fn test_blocked_sum() {
        let data = vec![1.0f32; 10000];
        let sum = CacheAwareAlgorithms::blocked_sum(&data, 1024);
        assert_eq!(sum, 10000.0);
    }
    
    #[test]
    fn test_soa_container() {
        let mut soa: SoAContainer<f32, f32, f32> = SoAContainer::new();
        soa.push(1.0, 2.0, 3.0);
        soa.push(4.0, 5.0, 6.0);
        
        assert_eq!(soa.len(), 2);
        assert_eq!(soa.field1[0], 1.0);
        assert_eq!(soa.field2[1], 5.0);
    }
}