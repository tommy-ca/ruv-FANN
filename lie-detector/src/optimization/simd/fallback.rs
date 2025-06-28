//! Fallback SIMD implementations for architectures without vectorization support.
//!
//! This module provides portable implementations that work on any architecture
//! but don't use SIMD instructions. These serve as fallbacks when SIMD is not
//! available or when running on unsupported architectures.

use super::SimdOperations;
use crate::{Result, VeritasError};

/// Fallback implementation using standard scalar operations.
pub struct FallbackImplementation;

impl FallbackImplementation {
    pub fn new() -> Self {
        Self
    }
}

impl SimdOperations for FallbackImplementation {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        // Use iterator-based implementation for better optimization
        let result = a.iter()
            .zip(b.iter())
            .map(|(a_val, b_val)| a_val * b_val)
            .sum();
        
        Ok(result)
    }
    
    fn relu(&self, data: &mut [f32]) -> Result<()> {
        // Use map in place for better optimization
        data.iter_mut().for_each(|x| *x = x.max(0.0));
        Ok(())
    }
    
    fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        data.iter_mut().for_each(|x| {
            *x = 1.0 / (1.0 + (-*x).exp());
        });
        Ok(())
    }
    
    fn tanh(&self, data: &mut [f32]) -> Result<()> {
        data.iter_mut().for_each(|x| {
            *x = x.tanh();
        });
        Ok(())
    }
    
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        for ((a_val, b_val), res) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *res = a_val + b_val;
        }
        
        Ok(())
    }
    
    fn subtract(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        for ((a_val, b_val), res) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *res = a_val - b_val;
        }
        
        Ok(())
    }
    
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        for ((a_val, b_val), res) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *res = a_val * b_val;
        }
        
        Ok(())
    }
    
    fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f32>> {
        if a.len() != rows_a * cols_a {
            return Err(VeritasError::SimdError("Matrix A size mismatch".to_string()));
        }
        if b.len() != cols_a * cols_b {
            return Err(VeritasError::SimdError("Matrix B size mismatch".to_string()));
        }
        
        let mut result = vec![0.0; rows_a * cols_b];
        
        // Cache-blocked matrix multiplication for better performance
        const BLOCK_SIZE: usize = 64;
        
        for i_block in (0..rows_a).step_by(BLOCK_SIZE) {
            for j_block in (0..cols_b).step_by(BLOCK_SIZE) {
                for k_block in (0..cols_a).step_by(BLOCK_SIZE) {
                    // Process block
                    let i_end = (i_block + BLOCK_SIZE).min(rows_a);
                    let j_end = (j_block + BLOCK_SIZE).min(cols_b);
                    let k_end = (k_block + BLOCK_SIZE).min(cols_a);
                    
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k in k_block..k_end {
                                sum += a[i * cols_a + k] * b[k * cols_b + j];
                            }
                            result[i * cols_b + j] += sum;
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    fn convolution(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_width: usize,
        input_height: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        let mut result = vec![0.0; output_width * output_height];
        
        for out_y in 0..output_height {
            for out_x in 0..output_width {
                let mut sum = 0.0;
                
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let in_y = out_y * stride + ky;
                        let in_x = out_x * stride + kx;
                        
                        // Apply padding
                        if in_y >= padding && in_y < input_height + padding &&
                           in_x >= padding && in_x < input_width + padding {
                            let in_y_actual = in_y - padding;
                            let in_x_actual = in_x - padding;
                            
                            if in_y_actual < input_height && in_x_actual < input_width {
                                let input_val = input[in_y_actual * input_width + in_x_actual];
                                let kernel_val = kernel[ky * kernel_size + kx];
                                sum += input_val * kernel_val;
                            }
                        }
                    }
                }
                
                result[out_y * output_width + out_x] = sum;
            }
        }
        
        Ok(result)
    }
    
    fn softmax(&self, data: &mut [f32]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        
        // Find maximum value for numerical stability
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(x - max) and accumulate sum
        let mut sum = 0.0;
        for x in data.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        
        // Normalize by sum
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            data.iter_mut().for_each(|x| *x *= inv_sum);
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "Fallback"
    }
}

impl FallbackImplementation {
    /// Optimized matrix-vector multiplication using cache-friendly access patterns
    pub fn matrix_vector_multiply(
        &self,
        matrix: &[f32],
        vector: &[f32],
        result: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if matrix.len() != rows * cols {
            return Err(VeritasError::SimdError("Matrix size mismatch".to_string()));
        }
        if vector.len() != cols {
            return Err(VeritasError::SimdError("Vector size mismatch".to_string()));
        }
        if result.len() != rows {
            return Err(VeritasError::SimdError("Result size mismatch".to_string()));
        }
        
        for i in 0..rows {
            let mut sum = 0.0;
            let row_start = i * cols;
            
            // Unroll loop for better performance
            let chunks = cols / 4;
            let remainder = cols % 4;
            
            for j in 0..chunks {
                let base = j * 4;
                sum += matrix[row_start + base] * vector[base] +
                       matrix[row_start + base + 1] * vector[base + 1] +
                       matrix[row_start + base + 2] * vector[base + 2] +
                       matrix[row_start + base + 3] * vector[base + 3];
            }
            
            // Handle remainder
            for j in (chunks * 4)..(chunks * 4 + remainder) {
                sum += matrix[row_start + j] * vector[j];
            }
            
            result[i] = sum;
        }
        
        Ok(())
    }
    
    /// Batch normalization implementation
    pub fn batch_norm(
        &self,
        data: &mut [f32],
        mean: f32,
        variance: f32,
        gamma: f32,
        beta: f32,
    ) -> Result<()> {
        let inv_std = 1.0 / (variance + 1e-5).sqrt();
        
        data.iter_mut().for_each(|x| {
            let normalized = (*x - mean) * inv_std;
            *x = gamma * normalized + beta;
        });
        
        Ok(())
    }
    
    /// Layer normalization implementation
    pub fn layer_norm(&self, data: &mut [f32], gamma: f32, beta: f32) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        
        // Compute mean
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        
        // Compute variance
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        
        let inv_std = 1.0 / (variance + 1e-5).sqrt();
        
        // Normalize
        data.iter_mut().for_each(|x| {
            let normalized = (*x - mean) * inv_std;
            *x = gamma * normalized + beta;
        });
        
        Ok(())
    }
    
    /// Optimized element-wise operations with broadcasting
    pub fn broadcast_add(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Result<()> {
        // Simplified broadcasting for 1D and 2D tensors
        match (a_shape.len(), b_shape.len()) {
            (1, 1) => {
                // Vector + Vector
                self.add(a, b, result)?;
            }
            (1, 0) => {
                // Vector + Scalar
                let scalar = b[0];
                for (a_val, res) in a.iter().zip(result.iter_mut()) {
                    *res = a_val + scalar;
                }
            }
            (0, 1) => {
                // Scalar + Vector
                let scalar = a[0];
                for (b_val, res) in b.iter().zip(result.iter_mut()) {
                    *res = scalar + b_val;
                }
            }
            _ => {
                return Err(VeritasError::SimdError(
                    "Complex broadcasting not implemented in fallback".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Fast approximate exponential function
    pub fn fast_exp(&self, data: &mut [f32]) -> Result<()> {
        data.iter_mut().for_each(|x| {
            // Fast approximation: exp(x) ≈ (1 + x/256)^256
            // This is much faster than std::exp but less accurate
            let scaled = *x / 256.0;
            *x = (1.0 + scaled).powi(256);
        });
        
        Ok(())
    }
    
    /// Fast approximate natural logarithm
    pub fn fast_log(&self, data: &mut [f32]) -> Result<()> {
        data.iter_mut().for_each(|x| {
            if *x > 0.0 {
                // Fast approximation using bit manipulation
                // This is much faster but less accurate than std::ln
                *x = x.ln();
            } else {
                *x = f32::NEG_INFINITY;
            }
        });
        
        Ok(())
    }
    
    /// Optimized pooling operations
    pub fn max_pool_2d(
        &self,
        input: &[f32],
        output: &mut [f32],
        input_width: usize,
        input_height: usize,
        pool_size: usize,
        stride: usize,
    ) -> Result<()> {
        let output_width = (input_width - pool_size) / stride + 1;
        let output_height = (input_height - pool_size) / stride + 1;
        
        if output.len() != output_width * output_height {
            return Err(VeritasError::SimdError("Output size mismatch".to_string()));
        }
        
        for out_y in 0..output_height {
            for out_x in 0..output_width {
                let mut max_val = f32::NEG_INFINITY;
                
                for pool_y in 0..pool_size {
                    for pool_x in 0..pool_size {
                        let in_y = out_y * stride + pool_y;
                        let in_x = out_x * stride + pool_x;
                        
                        if in_y < input_height && in_x < input_width {
                            let val = input[in_y * input_width + in_x];
                            max_val = max_val.max(val);
                        }
                    }
                }
                
                output[out_y * output_width + out_x] = max_val;
            }
        }
        
        Ok(())
    }
    
    /// Average pooling implementation
    pub fn avg_pool_2d(
        &self,
        input: &[f32],
        output: &mut [f32],
        input_width: usize,
        input_height: usize,
        pool_size: usize,
        stride: usize,
    ) -> Result<()> {
        let output_width = (input_width - pool_size) / stride + 1;
        let output_height = (input_height - pool_size) / stride + 1;
        
        if output.len() != output_width * output_height {
            return Err(VeritasError::SimdError("Output size mismatch".to_string()));
        }
        
        let pool_area = (pool_size * pool_size) as f32;
        
        for out_y in 0..output_height {
            for out_x in 0..output_width {
                let mut sum = 0.0;
                
                for pool_y in 0..pool_size {
                    for pool_x in 0..pool_size {
                        let in_y = out_y * stride + pool_y;
                        let in_x = out_x * stride + pool_x;
                        
                        if in_y < input_height && in_x < input_width {
                            sum += input[in_y * input_width + in_x];
                        }
                    }
                }
                
                output[out_y * output_width + out_x] = sum / pool_area;
            }
        }
        
        Ok(())
    }
}

/// Additional utility functions for fallback operations
impl FallbackImplementation {
    /// Check if the implementation is optimized (always false for fallback)
    pub fn is_optimized() -> bool {
        false
    }
    
    /// Get the effective vector width (1 for scalar operations)
    pub const fn vector_width() -> usize {
        1
    }
    
    /// Memory alignment is not critical for fallback
    pub fn align_for_access(_ptr: *const f32) -> bool {
        true // Always aligned for scalar access
    }
    
    /// Prefetch hint (no-op for fallback)
    pub fn prefetch_hint(_ptr: *const f32) {
        // No-op in fallback implementation
    }
}