//! Cache-friendly data structures and algorithms.
//!
//! This module provides data structures and algorithms optimized for CPU cache
//! performance, including:
//! - Cache-blocked matrix operations
//! - Structure of Arrays (SoA) layouts
//! - Cache-friendly neural network layers
//! - Memory prefetching utilities
//! - Loop tiling and blocking strategies

use crate::{Result, VeritasError};
use crate::optimization::vectorization_hints::alignment::AlignedVec;
use std::arch::x86_64::_mm_prefetch;

/// Cache optimization configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size in bytes
    pub l1_cache_size: usize,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    /// L3 cache size in bytes
    pub l3_cache_size: usize,
    /// Cache line size in bytes
    pub cache_line_size: usize,
    /// Block size for tiled operations
    pub block_size: usize,
    /// Enable prefetching
    pub enable_prefetch: bool,
    /// Prefetch distance
    pub prefetch_distance: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024,      // 32KB L1
            l2_cache_size: 256 * 1024,     // 256KB L2
            l3_cache_size: 8 * 1024 * 1024, // 8MB L3
            cache_line_size: 64,           // 64 bytes
            block_size: 64,                // 64x64 blocks
            enable_prefetch: true,
            prefetch_distance: 8,          // Prefetch 8 cache lines ahead
        }
    }
}

impl CacheConfig {
    /// Configuration for small L1 cache (embedded systems).
    pub fn small() -> Self {
        Self {
            l1_cache_size: 16 * 1024,
            l2_cache_size: 128 * 1024,
            l3_cache_size: 1 * 1024 * 1024,
            block_size: 32,
            prefetch_distance: 4,
            ..Self::default()
        }
    }
    
    /// Configuration for medium cache (edge computing).
    pub fn medium() -> Self {
        Self {
            l1_cache_size: 32 * 1024,
            l2_cache_size: 512 * 1024,
            l3_cache_size: 4 * 1024 * 1024,
            block_size: 64,
            prefetch_distance: 8,
            ..Self::default()
        }
    }
    
    /// Configuration for large cache (server systems).
    pub fn large() -> Self {
        Self {
            l1_cache_size: 64 * 1024,
            l2_cache_size: 1024 * 1024,
            l3_cache_size: 32 * 1024 * 1024,
            block_size: 128,
            prefetch_distance: 16,
            ..Self::default()
        }
    }
}

/// Cache-optimized matrix with blocked layout.
pub struct CacheOptimizedMatrix {
    data: AlignedVec<f32>,
    rows: usize,
    cols: usize,
    block_size: usize,
    blocks_per_row: usize,
    blocks_per_col: usize,
}

impl CacheOptimizedMatrix {
    /// Create a new cache-optimized matrix.
    pub fn new(rows: usize, cols: usize, block_size: usize) -> Self {
        let blocks_per_row = (rows + block_size - 1) / block_size;
        let blocks_per_col = (cols + block_size - 1) / block_size;
        let total_elements = blocks_per_row * blocks_per_col * block_size * block_size;
        
        let mut data = AlignedVec::with_capacity(64, total_elements);
        data.resize(total_elements, 0.0);
        
        Self {
            data,
            rows,
            cols,
            block_size,
            blocks_per_row,
            blocks_per_col,
        }
    }
    
    /// Create matrix from row-major data.
    pub fn from_row_major(data: &[f32], rows: usize, cols: usize, block_size: usize) -> Self {
        let mut matrix = Self::new(rows, cols, block_size);
        matrix.copy_from_row_major(data);
        matrix
    }
    
    /// Copy data from row-major layout.
    pub fn copy_from_row_major(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.rows * self.cols);
        
        for row in 0..self.rows {
            for col in 0..self.cols {
                let value = data[row * self.cols + col];
                self.set(row, col, value);
            }
        }
    }
    
    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows && col < self.cols);
        
        let block_row = row / self.block_size;
        let block_col = col / self.block_size;
        let in_block_row = row % self.block_size;
        let in_block_col = col % self.block_size;
        
        let block_index = block_row * self.blocks_per_col + block_col;
        let block_offset = block_index * self.block_size * self.block_size;
        let element_offset = in_block_row * self.block_size + in_block_col;
        
        self.data[block_offset + element_offset]
    }
    
    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows && col < self.cols);
        
        let block_row = row / self.block_size;
        let block_col = col / self.block_size;
        let in_block_row = row % self.block_size;
        let in_block_col = col % self.block_size;
        
        let block_index = block_row * self.blocks_per_col + block_col;
        let block_offset = block_index * self.block_size * self.block_size;
        let element_offset = in_block_row * self.block_size + in_block_col;
        
        self.data[block_offset + element_offset] = value;
    }
    
    /// Perform cache-blocked matrix multiplication: C = A * B.
    pub fn multiply(&self, other: &Self, result: &mut Self) -> Result<()> {
        if self.cols != other.rows {
            return Err(VeritasError::invalid_input("Matrix dimension mismatch", "matrix_dimensions"));
        }
        
        if result.rows != self.rows || result.cols != other.cols {
            return Err(VeritasError::invalid_input("Result matrix size mismatch", "result_matrix"));
        }
        
        // Initialize result to zero
        result.data.fill(0.0);
        
        // Cache-blocked multiplication
        for i_block in (0..self.rows).step_by(self.block_size) {
            for j_block in (0..other.cols).step_by(self.block_size) {
                for k_block in (0..self.cols).step_by(self.block_size) {
                    // Process block
                    let i_end = (i_block + self.block_size).min(self.rows);
                    let j_end = (j_block + self.block_size).min(other.cols);
                    let k_end = (k_block + self.block_size).min(self.cols);
                    
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = result.get(i, j);
                            
                            for k in k_block..k_end {
                                sum += self.get(i, k) * other.get(k, j);
                            }
                            
                            result.set(i, j, sum);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get matrix dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Convert to row-major layout.
    pub fn to_row_major(&self) -> Vec<f32> {
        let mut result = vec![0.0; self.rows * self.cols];
        
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[row * self.cols + col] = self.get(row, col);
            }
        }
        
        result
    }
}

/// Structure of Arrays (SoA) layout for better cache performance.
#[derive(Debug, Clone)]
pub struct SoAData<T> {
    arrays: Vec<AlignedVec<T>>,
    capacity: usize,
    len: usize,
}

impl<T: Clone + Default> SoAData<T> {
    /// Create a new SoA with specified number of arrays.
    pub fn new(num_arrays: usize, capacity: usize) -> Self {
        let mut arrays = Vec::with_capacity(num_arrays);
        for _ in 0..num_arrays {
            let mut array = AlignedVec::with_capacity(64, capacity);
            array.resize(capacity, T::default());
            arrays.push(array);
        }
        
        Self {
            arrays,
            capacity,
            len: 0,
        }
    }
    
    /// Push a tuple of values.
    pub fn push(&mut self, values: &[T]) -> Result<()> {
        if values.len() != self.arrays.len() {
            return Err(VeritasError::invalid_input("Value count mismatch", "values"));
        }
        
        if self.len >= self.capacity {
            return Err(VeritasError::memory_error("Structure of Arrays capacity exceeded"));
        }
        
        for (i, value) in values.iter().enumerate() {
            self.arrays[i][self.len] = value.clone();
        }
        
        self.len += 1;
        Ok(())
    }
    
    /// Get a specific array.
    pub fn array(&self, index: usize) -> Option<&[T]> {
        self.arrays.get(index).map(|arr| &arr[..self.len])
    }
    
    /// Get mutable reference to a specific array.
    pub fn array_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.arrays.get_mut(index).map(|arr| &mut arr[..self.len])
    }
    
    /// Get current length.
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Clear all data.
    pub fn clear(&mut self) {
        self.len = 0;
    }
    
    /// Number of arrays.
    pub fn num_arrays(&self) -> usize {
        self.arrays.len()
    }
}

/// Cache-friendly neural network layer implementation.
pub struct CacheOptimizedLayer {
    weights: CacheOptimizedMatrix,
    biases: AlignedVec<f32>,
    config: CacheConfig,
}

impl CacheOptimizedLayer {
    /// Create a new cache-optimized layer.
    pub fn new(
        input_size: usize,
        output_size: usize,
        config: CacheConfig,
    ) -> Self {
        let weights = CacheOptimizedMatrix::new(
            output_size,
            input_size,
            config.block_size,
        );
        
        let mut biases = AlignedVec::with_capacity(64, output_size);
        biases.resize(output_size, 0.0);
        
        Self {
            weights,
            biases,
            config,
        }
    }
    
    /// Initialize weights with random values.
    pub fn initialize_weights(&mut self, weights: &[f32], biases: &[f32]) -> Result<()> {
        let (rows, cols) = self.weights.dimensions();
        
        if weights.len() != rows * cols {
            return Err(VeritasError::invalid_input("Weight matrix size mismatch", "weights"));
        }
        
        if biases.len() != self.biases.len() {
            return Err(VeritasError::invalid_input("Bias vector size mismatch", "bias"));
        }
        
        self.weights.copy_from_row_major(weights);
        self.biases.copy_from_slice(biases);
        
        Ok(())
    }
    
    /// Forward pass with cache optimization.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let (output_size, input_size) = self.weights.dimensions();
        
        if input.len() != input_size {
            return Err(VeritasError::invalid_input("Input size mismatch", "input"));
        }
        
        if output.len() != output_size {
            return Err(VeritasError::invalid_input("Output size mismatch", "output"));
        }
        
        // Cache-optimized matrix-vector multiplication
        self.matrix_vector_multiply_optimized(input, output)?;
        
        // Add biases
        for (out, &bias) in output.iter_mut().zip(self.biases.iter()) {
            *out += bias;
        }
        
        Ok(())
    }
    
    /// Cache-optimized matrix-vector multiplication.
    fn matrix_vector_multiply_optimized(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let (output_size, input_size) = self.weights.dimensions();
        
        // Initialize output
        output.fill(0.0);
        
        // Block the computation for better cache usage
        let block_size = self.config.block_size;
        
        for i_block in (0..output_size).step_by(block_size) {
            for k_block in (0..input_size).step_by(block_size) {
                let i_end = (i_block + block_size).min(output_size);
                let k_end = (k_block + block_size).min(input_size);
                
                // Prefetch next block if enabled
                if self.config.enable_prefetch && i_block + block_size < output_size {
                    self.prefetch_block(i_block + block_size, k_block);
                }
                
                // Process block
                for i in i_block..i_end {
                    let mut sum = output[i];
                    
                    for k in k_block..k_end {
                        sum += self.weights.get(i, k) * input[k];
                    }
                    
                    output[i] = sum;
                }
            }
        }
        
        Ok(())
    }
    
    /// Prefetch memory for better cache performance.
    fn prefetch_block(&self, i_block: usize, k_block: usize) {
        if !self.config.enable_prefetch {
            return;
        }
        
        // Prefetch weight matrix data
        let (output_size, _) = self.weights.dimensions();
        let block_size = self.config.block_size;
        let i_end = (i_block + block_size).min(output_size);
        
        for i in i_block..i_end {
            let addr = &self.weights.get(i, k_block) as *const f32;
            
            unsafe {
                // Prefetch for temporal locality (T0 = non-temporal, T1 = L2 cache, T2 = L1 cache)
                _mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T1);
            }
        }
    }
    
    /// Get layer dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        self.weights.dimensions()
    }
}

/// Cache-aware data prefetching utilities.
pub struct PrefetchUtils;

impl PrefetchUtils {
    /// Prefetch data for reading (temporal locality).
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_read(addr: *const u8) {
        unsafe {
            _mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    
    /// Prefetch data for writing (non-temporal).
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_write(addr: *const u8) {
        unsafe {
            _mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_NTA);
        }
    }
    
    /// No-op for non-x86 architectures.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch_read(_addr: *const u8) {
        // No-op for other architectures
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch_write(_addr: *const u8) {
        // No-op for other architectures
    }
    
    /// Prefetch array data with stride.
    pub fn prefetch_array<T>(data: &[T], stride: usize, distance: usize) {
        for i in (0..data.len()).step_by(stride) {
            if i + distance < data.len() {
                let addr = data.as_ptr().wrapping_add(i + distance) as *const u8;
                Self::prefetch_read(addr);
            }
        }
    }
}

/// Loop tiling utilities for cache optimization.
pub struct LoopTiling;

impl LoopTiling {
    /// Calculate optimal tile sizes for matrix operations.
    pub fn calculate_tile_sizes(
        m: usize,
        n: usize,
        k: usize,
        cache_size: usize,
        element_size: usize,
    ) -> (usize, usize, usize) {
        // Simple heuristic for tile sizes
        // In practice, this would be more sophisticated
        
        let available_cache = cache_size / element_size;
        let base_tile = (available_cache as f64).sqrt() as usize;
        
        let tile_m = base_tile.min(m);
        let tile_n = base_tile.min(n);
        let tile_k = (available_cache / (tile_m + tile_n)).min(k);
        
        (tile_m, tile_n, tile_k)
    }
    
    /// Execute tiled matrix multiplication.
    pub fn tiled_matmul(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
    ) {
        // Initialize result
        c.fill(0.0);
        
        for i_tile in (0..m).step_by(tile_m) {
            for j_tile in (0..n).step_by(tile_n) {
                for k_tile in (0..k).step_by(tile_k) {
                    // Process tile
                    let i_end = (i_tile + tile_m).min(m);
                    let j_end = (j_tile + tile_n).min(n);
                    let k_end = (k_tile + tile_k).min(k);
                    
                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            let mut sum = c[i * n + j];
                            
                            for kk in k_tile..k_end {
                                sum += a[i * k + kk] * b[kk * n + j];
                            }
                            
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    }
}

/// Cache-friendly algorithms for common operations.
pub struct CacheAlgorithms;

impl CacheAlgorithms {
    /// Cache-blocked convolution.
    pub fn blocked_convolution(
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        input_height: usize,
        input_width: usize,
        kernel_size: usize,
        output_height: usize,
        output_width: usize,
        block_size: usize,
    ) {
        output.fill(0.0);
        
        for out_y_block in (0..output_height).step_by(block_size) {
            for out_x_block in (0..output_width).step_by(block_size) {
                let out_y_end = (out_y_block + block_size).min(output_height);
                let out_x_end = (out_x_block + block_size).min(output_width);
                
                for out_y in out_y_block..out_y_end {
                    for out_x in out_x_block..out_x_end {
                        let mut sum = 0.0;
                        
                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let in_y = out_y + ky;
                                let in_x = out_x + kx;
                                
                                if in_y < input_height && in_x < input_width {
                                    sum += input[in_y * input_width + in_x] *
                                           kernel[ky * kernel_size + kx];
                                }
                            }
                        }
                        
                        output[out_y * output_width + out_x] = sum;
                    }
                }
            }
        }
    }
    
    /// Cache-efficient transpose.
    pub fn cache_efficient_transpose(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) {
        for i_block in (0..rows).step_by(block_size) {
            for j_block in (0..cols).step_by(block_size) {
                let i_end = (i_block + block_size).min(rows);
                let j_end = (j_block + block_size).min(cols);
                
                for i in i_block..i_end {
                    for j in j_block..j_end {
                        output[j * rows + i] = input[i * cols + j];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_optimized_matrix() {
        let mut matrix = CacheOptimizedMatrix::new(4, 4, 2);
        
        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 3.0);
        matrix.set(3, 3, 4.0);
        
        // Check values
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 2.0);
        assert_eq!(matrix.get(2, 2), 3.0);
        assert_eq!(matrix.get(3, 3), 4.0);
    }
    
    #[test]
    fn test_matrix_multiplication() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![2.0, 0.0, 1.0, 2.0];
        
        let a = CacheOptimizedMatrix::from_row_major(&a_data, 2, 2, 2);
        let b = CacheOptimizedMatrix::from_row_major(&b_data, 2, 2, 2);
        let mut c = CacheOptimizedMatrix::new(2, 2, 2);
        
        a.multiply(&b, &mut c).unwrap();
        
        let result = c.to_row_major();
        assert_eq!(result, vec![4.0, 4.0, 10.0, 8.0]);
    }
    
    #[test]
    fn test_soa_data() {
        let mut soa = SoAData::<f32>::new(3, 10);
        
        soa.push(&[1.0, 2.0, 3.0]).unwrap();
        soa.push(&[4.0, 5.0, 6.0]).unwrap();
        
        assert_eq!(soa.len(), 2);
        assert_eq!(soa.array(0).unwrap(), &[1.0, 4.0]);
        assert_eq!(soa.array(1).unwrap(), &[2.0, 5.0]);
        assert_eq!(soa.array(2).unwrap(), &[3.0, 6.0]);
    }
    
    #[test]
    fn test_cache_optimized_layer() {
        let config = CacheConfig::default();
        let mut layer = CacheOptimizedLayer::new(3, 2, config);
        
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let biases = vec![0.1, 0.2];
        
        layer.initialize_weights(&weights, &biases).unwrap();
        
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];
        
        layer.forward(&input, &mut output).unwrap();
        
        // Expected: [1*1 + 2*2 + 3*3 + 0.1, 1*4 + 2*5 + 3*6 + 0.2] = [14.1, 32.2]
        assert!((output[0] - 14.1).abs() < 1e-6);
        assert!((output[1] - 32.2).abs() < 1e-6);
    }
    
    #[test]
    fn test_loop_tiling() {
        let tile_sizes = LoopTiling::calculate_tile_sizes(100, 100, 100, 32768, 4);
        assert!(tile_sizes.0 > 0);
        assert!(tile_sizes.1 > 0);
        assert!(tile_sizes.2 > 0);
    }
}