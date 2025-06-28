//! ARM NEON SIMD implementations for AArch64 architecture.
//!
//! This module provides high-performance vectorized operations specifically
//! optimized for ARM processors with NEON SIMD support.

use super::{SimdFeatures, SimdOperations};
use crate::{Result, VeritasError};
use std::arch::aarch64::*;

/// Detect ARM NEON features.
pub fn detect_features() -> SimdFeatures {
    let mut features = SimdFeatures::default();
    
    // NEON is standard on AArch64, so we can assume it's available
    #[cfg(target_arch = "aarch64")]
    {
        features.neon = true;
    }
    
    // Additional feature detection could be added here using
    // system calls or CPUID-equivalent mechanisms for ARM
    
    features
}

/// NEON implementation for ARM processors.
pub struct NeonImplementation;

impl NeonImplementation {
    pub fn new() -> Self {
        Self
    }
}

impl SimdOperations for NeonImplementation {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        unsafe { self.dot_product_neon(a, b) }
    }
    
    fn relu(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.relu_neon(data) };
        Ok(())
    }
    
    fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.sigmoid_neon(data) };
        Ok(())
    }
    
    fn tanh(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.tanh_neon(data) };
        Ok(())
    }
    
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        unsafe { self.add_neon(a, b, result) };
        Ok(())
    }
    
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        unsafe { self.multiply_neon(a, b, result) };
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
        let mut result = vec![0.0; rows_a * cols_b];
        unsafe { self.matrix_multiply_neon(a, b, &mut result, rows_a, cols_a, cols_b) };
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
        
        unsafe {
            self.convolution_neon(
                input,
                kernel,
                &mut result,
                input_width,
                input_height,
                kernel_size,
                stride,
                padding,
            )
        };
        
        Ok(result)
    }
    
    fn softmax(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.softmax_neon(data) };
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "NEON"
    }
}

impl NeonImplementation {
    #[target_feature(enable = "neon")]
    unsafe fn dot_product_neon(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let mut sum = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            sum = vfmaq_f32(sum, va, vb); // Fused multiply-add
        }
        
        // Horizontal sum of the vector
        let sum_pairs = vpaddq_f32(sum, sum);
        let sum_final = vpadds_f32(vget_low_f32(sum_pairs));
        let mut result = sum_final;
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn relu_neon(&self, data: &mut [f32]) {
        let len = data.len();
        let zero = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let values = vld1q_f32(data.as_ptr().add(offset));
            let result = vmaxq_f32(values, zero);
            vst1q_f32(data.as_mut_ptr().add(offset), result);
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            data[i] = data[i].max(0.0);
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn sigmoid_neon(&self, data: &mut [f32]) {
        let len = data.len();
        let one = vdupq_n_f32(1.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let x = vld1q_f32(data.as_ptr().add(offset));
            
            // Approximate sigmoid using tanh: sigmoid(x) ≈ 0.5 * (1 + tanh(0.5 * x))
            let half_x = vmulq_n_f32(x, 0.5);
            let tanh_approx = self.fast_tanh_neon(half_x);
            let sigmoid = vmulq_n_f32(vaddq_f32(one, tanh_approx), 0.5);
            
            vst1q_f32(data.as_mut_ptr().add(offset), sigmoid);
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            data[i] = 1.0 / (1.0 + (-data[i]).exp());
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn tanh_neon(&self, data: &mut [f32]) {
        let len = data.len();
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let x = vld1q_f32(data.as_ptr().add(offset));
            let result = self.fast_tanh_neon(x);
            vst1q_f32(data.as_mut_ptr().add(offset), result);
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            data[i] = data[i].tanh();
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn fast_tanh_neon(&self, x: float32x4_t) -> float32x4_t {
        // Fast tanh approximation using polynomial
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        let x5 = vmulq_f32(x3, x2);
        
        // tanh(x) ≈ x - x³/3 + 2x⁵/15 (for |x| < 1.5)
        let term1 = x;
        let term2 = vmulq_n_f32(x3, -1.0 / 3.0);
        let term3 = vmulq_n_f32(x5, 2.0 / 15.0);
        
        vaddq_f32(vaddq_f32(term1, term2), term3)
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn add_neon(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let sum = vaddq_f32(va, vb);
            vst1q_f32(result.as_mut_ptr().add(offset), sum);
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            result[i] = a[i] + b[i];
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn multiply_neon(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let prod = vmulq_f32(va, vb);
            vst1q_f32(result.as_mut_ptr().add(offset), prod);
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            result[i] = a[i] * b[i];
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn matrix_multiply_neon(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) {
        // Cache-blocked matrix multiplication with NEON optimization
        const BLOCK_SIZE: usize = 32;
        
        for i_block in (0..rows_a).step_by(BLOCK_SIZE) {
            for j_block in (0..cols_b).step_by(BLOCK_SIZE) {
                for k_block in (0..cols_a).step_by(BLOCK_SIZE) {
                    // Process block
                    let i_end = (i_block + BLOCK_SIZE).min(rows_a);
                    let j_end = (j_block + BLOCK_SIZE).min(cols_b);
                    let k_end = (k_block + BLOCK_SIZE).min(cols_a);
                    
                    for i in i_block..i_end {
                        for j in (j_block..j_end).step_by(4) {
                            let j_end_chunk = (j + 4).min(j_end);
                            let mut acc = vdupq_n_f32(0.0);
                            
                            for k in k_block..k_end {
                                let a_val = vdupq_n_f32(a[i * cols_a + k]);
                                if j_end_chunk - j == 4 {
                                    let b_vals = vld1q_f32(b.as_ptr().add(k * cols_b + j));
                                    acc = vfmaq_f32(acc, a_val, b_vals);
                                } else {
                                    // Handle partial vector
                                    for jj in j..j_end_chunk {
                                        result[i * cols_b + jj] += a[i * cols_a + k] * b[k * cols_b + jj];
                                    }
                                    continue;
                                }
                            }
                            
                            if j_end_chunk - j == 4 {
                                let prev = vld1q_f32(result.as_ptr().add(i * cols_b + j));
                                let sum = vaddq_f32(prev, acc);
                                vst1q_f32(result.as_mut_ptr().add(i * cols_b + j), sum);
                            }
                        }
                    }
                }
            }
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn convolution_neon(
        &self,
        input: &[f32],
        kernel: &[f32],
        result: &mut [f32],
        input_width: usize,
        input_height: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) {
        let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        
        for out_y in 0..output_height {
            for out_x in 0..output_width {
                let mut acc = vdupq_n_f32(0.0);
                let mut scalar_acc = 0.0f32;
                
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let in_y = out_y * stride + ky;
                        let in_x = out_x * stride + kx;
                        
                        // Apply padding
                        if in_y >= padding && in_y < input_height + padding &&
                           in_x >= padding && in_x < input_width + padding {
                            let in_y_actual = in_y - padding;
                            let in_x_actual = in_x - padding;
                            let input_val = input[in_y_actual * input_width + in_x_actual];
                            let kernel_val = kernel[ky * kernel_size + kx];
                            scalar_acc += input_val * kernel_val;
                        }
                    }
                }
                
                result[out_y * output_width + out_x] = scalar_acc;
            }
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn softmax_neon(&self, data: &mut [f32]) {
        let len = data.len();
        
        // Find maximum value for numerical stability
        let mut max_val = data[0];
        for &val in data.iter() {
            if val > max_val {
                max_val = val;
            }
        }
        
        let max_vec = vdupq_n_f32(max_val);
        let mut sum = 0.0f32;
        
        // Compute exp(x - max) and accumulate sum
        let chunks = len / 4;
        let mut partial_sums = vdupq_n_f32(0.0);
        
        for i in 0..chunks {
            let offset = i * 4;
            let vals = vld1q_f32(data.as_ptr().add(offset));
            let shifted = vsubq_f32(vals, max_vec);
            
            // Approximate exp using polynomial (for better performance)
            let exp_vals = self.fast_exp_neon(shifted);
            vst1q_f32(data.as_mut_ptr().add(offset), exp_vals);
            partial_sums = vaddq_f32(partial_sums, exp_vals);
        }
        
        // Sum the partial sums
        let sum_pairs = vpaddq_f32(partial_sums, partial_sums);
        sum += vpadds_f32(vget_low_f32(sum_pairs));
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            let exp_val = (data[i] - max_val).exp();
            data[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize by sum
        let inv_sum = vdupq_n_f32(1.0 / sum);
        for i in 0..chunks {
            let offset = i * 4;
            let vals = vld1q_f32(data.as_ptr().add(offset));
            let normalized = vmulq_f32(vals, inv_sum);
            vst1q_f32(data.as_mut_ptr().add(offset), normalized);
        }
        
        for i in (chunks * 4)..len {
            data[i] /= sum;
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn fast_exp_neon(&self, x: float32x4_t) -> float32x4_t {
        // Fast exp approximation using polynomial
        // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        let one = vdupq_n_f32(1.0);
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        let x4 = vmulq_f32(x3, x);
        
        let term1 = one;
        let term2 = x;
        let term3 = vmulq_n_f32(x2, 0.5);
        let term4 = vmulq_n_f32(x3, 1.0 / 6.0);
        let term5 = vmulq_n_f32(x4, 1.0 / 24.0);
        
        vaddq_f32(
            vaddq_f32(vaddq_f32(term1, term2), vaddq_f32(term3, term4)),
            term5
        )
    }
    
    /// Optimized matrix-vector multiplication for NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn matrix_vector_multiply_neon(
        &self,
        matrix: &[f32],
        vector: &[f32],
        result: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        for i in 0..rows {
            let mut acc = vdupq_n_f32(0.0);
            let row_start = i * cols;
            
            // Process 4 elements at a time
            let chunks = cols / 4;
            for j in 0..chunks {
                let offset = j * 4;
                let mat_vals = vld1q_f32(matrix.as_ptr().add(row_start + offset));
                let vec_vals = vld1q_f32(vector.as_ptr().add(offset));
                acc = vfmaq_f32(acc, mat_vals, vec_vals);
            }
            
            // Horizontal sum
            let sum_pairs = vpaddq_f32(acc, acc);
            let mut sum = vpadds_f32(vget_low_f32(sum_pairs));
            
            // Handle remaining elements
            for j in (chunks * 4)..cols {
                sum += matrix[row_start + j] * vector[j];
            }
            
            result[i] = sum;
        }
    }
    
    /// Optimized batch normalization for NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn batch_norm_neon(
        &self,
        data: &mut [f32],
        mean: f32,
        variance: f32,
        gamma: f32,
        beta: f32,
    ) {
        let len = data.len();
        let inv_std = 1.0 / (variance + 1e-5).sqrt();
        
        let mean_vec = vdupq_n_f32(mean);
        let inv_std_vec = vdupq_n_f32(inv_std);
        let gamma_vec = vdupq_n_f32(gamma);
        let beta_vec = vdupq_n_f32(beta);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let vals = vld1q_f32(data.as_ptr().add(offset));
            
            // Normalize: (x - mean) / std
            let normalized = vmulq_f32(vsubq_f32(vals, mean_vec), inv_std_vec);
            
            // Scale and shift: gamma * normalized + beta
            let result = vfmaq_f32(beta_vec, normalized, gamma_vec);
            
            vst1q_f32(data.as_mut_ptr().add(offset), result);
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            let normalized = (data[i] - mean) * inv_std;
            data[i] = gamma * normalized + beta;
        }
    }
}

/// Additional utility functions for ARM NEON
impl NeonImplementation {
    /// Check if NEON is available on the current system
    pub fn is_available() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on AArch64
            true
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }
    
    /// Get the optimal vector width for NEON operations
    pub const fn vector_width() -> usize {
        4 // NEON processes 4 f32 values in parallel
    }
    
    /// Align data pointer for optimal NEON access
    pub fn align_for_neon(ptr: *const f32) -> bool {
        (ptr as usize) % 16 == 0
    }
}