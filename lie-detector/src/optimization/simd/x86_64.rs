//! x86_64 SIMD implementations using AVX, AVX2, and AVX512.
//!
//! This module provides high-performance vectorized operations specifically
//! optimized for x86_64 processors with various SIMD instruction sets.

use super::{SimdFeatures, SimdOperations};
use crate::{Result, VeritasError};
use std::arch::x86_64::*;

/// Detect x86_64 SIMD features using CPUID.
pub fn detect_features() -> SimdFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::CpuId;
        
        let cpuid = CpuId::new();
        let mut features = SimdFeatures::default();
        
        if let Some(feature_info) = cpuid.get_feature_info() {
            features.sse = feature_info.has_sse();
            features.sse2 = feature_info.has_sse2();
            features.sse3 = feature_info.has_sse3();
            features.ssse3 = feature_info.has_ssse3();
            features.sse41 = feature_info.has_sse41();
            features.sse42 = feature_info.has_sse42();
            features.avx = feature_info.has_avx();
            features.fma = feature_info.has_fma();
        }
        
        if let Some(extended_features) = cpuid.get_extended_feature_info() {
            features.avx2 = extended_features.has_avx2();
            features.avx512f = extended_features.has_avx512f();
            features.avx512dq = extended_features.has_avx512dq();
        }
        
        features
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        SimdFeatures::default()
    }
}

/// AVX512 implementation (highest performance).
pub struct Avx512Implementation;

impl Avx512Implementation {
    pub fn new() -> Self {
        Self
    }
}

impl SimdOperations for Avx512Implementation {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        unsafe { self.dot_product_avx512(a, b) }
    }
    
    fn relu(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.relu_avx512(data) };
        Ok(())
    }
    
    fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.sigmoid_avx512(data) };
        Ok(())
    }
    
    fn tanh(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.tanh_avx512(data) };
        Ok(())
    }
    
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        unsafe { self.add_avx512(a, b, result) };
        Ok(())
    }
    
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        unsafe { self.multiply_avx512(a, b, result) };
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
        unsafe { self.matrix_multiply_avx512(a, b, &mut result, rows_a, cols_a, cols_b) };
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
            self.convolution_avx512(
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
        unsafe { self.softmax_avx512(data) };
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AVX512"
    }
}

impl Avx512Implementation {
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_product_avx512(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let mut sum = _mm512_setzero_ps();
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        
        // Horizontal sum of the vector
        let mut result = _mm512_reduce_add_ps(sum);
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn relu_avx512(&self, data: &mut [f32]) {
        let len = data.len();
        let zero = _mm512_setzero_ps();
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let values = _mm512_loadu_ps(data.as_ptr().add(offset));
            let result = _mm512_max_ps(values, zero);
            _mm512_storeu_ps(data.as_mut_ptr().add(offset), result);
        }
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            data[i] = data[i].max(0.0);
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn sigmoid_avx512(&self, data: &mut [f32]) {
        let len = data.len();
        let one = _mm512_set1_ps(1.0);
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let x = _mm512_loadu_ps(data.as_ptr().add(offset));
            
            // Approximate sigmoid using tanh: sigmoid(x) ≈ 0.5 * (1 + tanh(0.5 * x))
            let half_x = _mm512_mul_ps(x, _mm512_set1_ps(0.5));
            let tanh_approx = self.fast_tanh_avx512(half_x);
            let sigmoid = _mm512_mul_ps(_mm512_add_ps(one, tanh_approx), _mm512_set1_ps(0.5));
            
            _mm512_storeu_ps(data.as_mut_ptr().add(offset), sigmoid);
        }
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            data[i] = 1.0 / (1.0 + (-data[i]).exp());
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn tanh_avx512(&self, data: &mut [f32]) {
        let len = data.len();
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let x = _mm512_loadu_ps(data.as_ptr().add(offset));
            let result = self.fast_tanh_avx512(x);
            _mm512_storeu_ps(data.as_mut_ptr().add(offset), result);
        }
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            data[i] = data[i].tanh();
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn fast_tanh_avx512(&self, x: __m512) -> __m512 {
        // Fast tanh approximation using polynomial
        let x2 = _mm512_mul_ps(x, x);
        let x3 = _mm512_mul_ps(x2, x);
        let x5 = _mm512_mul_ps(x3, x2);
        
        // tanh(x) ≈ x - x³/3 + 2x⁵/15 (for |x| < 1.5)
        let term1 = x;
        let term2 = _mm512_mul_ps(x3, _mm512_set1_ps(-1.0 / 3.0));
        let term3 = _mm512_mul_ps(x5, _mm512_set1_ps(2.0 / 15.0));
        
        _mm512_add_ps(_mm512_add_ps(term1, term2), term3)
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn add_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let sum = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(result.as_mut_ptr().add(offset), sum);
        }
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            result[i] = a[i] + b[i];
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn multiply_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let prod = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(result.as_mut_ptr().add(offset), prod);
        }
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            result[i] = a[i] * b[i];
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn matrix_multiply_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) {
        // Cache-blocked matrix multiplication with AVX512
        const BLOCK_SIZE: usize = 64;
        
        for i_block in (0..rows_a).step_by(BLOCK_SIZE) {
            for j_block in (0..cols_b).step_by(BLOCK_SIZE) {
                for k_block in (0..cols_a).step_by(BLOCK_SIZE) {
                    // Process block
                    let i_end = (i_block + BLOCK_SIZE).min(rows_a);
                    let j_end = (j_block + BLOCK_SIZE).min(cols_b);
                    let k_end = (k_block + BLOCK_SIZE).min(cols_a);
                    
                    for i in i_block..i_end {
                        for j in (j_block..j_end).step_by(16) {
                            let j_end_chunk = (j + 16).min(j_end);
                            let mut acc = _mm512_setzero_ps();
                            
                            for k in k_block..k_end {
                                let a_val = _mm512_set1_ps(a[i * cols_a + k]);
                                if j_end_chunk - j == 16 {
                                    let b_vals = _mm512_loadu_ps(b.as_ptr().add(k * cols_b + j));
                                    acc = _mm512_fmadd_ps(a_val, b_vals, acc);
                                } else {
                                    // Handle partial vector
                                    for jj in j..j_end_chunk {
                                        result[i * cols_b + jj] += a[i * cols_a + k] * b[k * cols_b + jj];
                                    }
                                    continue;
                                }
                            }
                            
                            if j_end_chunk - j == 16 {
                                let prev = _mm512_loadu_ps(result.as_ptr().add(i * cols_b + j));
                                let sum = _mm512_add_ps(prev, acc);
                                _mm512_storeu_ps(result.as_mut_ptr().add(i * cols_b + j), sum);
                            }
                        }
                    }
                }
            }
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn convolution_avx512(
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
                let mut acc = _mm512_setzero_ps();
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
    
    #[target_feature(enable = "avx512f")]
    unsafe fn softmax_avx512(&self, data: &mut [f32]) {
        let len = data.len();
        
        // Find maximum value for numerical stability
        let mut max_val = data[0];
        for &val in data.iter() {
            if val > max_val {
                max_val = val;
            }
        }
        
        let max_vec = _mm512_set1_ps(max_val);
        let mut sum = 0.0f32;
        
        // Compute exp(x - max) and accumulate sum
        let chunks = len / 16;
        let mut partial_sums = _mm512_setzero_ps();
        
        for i in 0..chunks {
            let offset = i * 16;
            let vals = _mm512_loadu_ps(data.as_ptr().add(offset));
            let shifted = _mm512_sub_ps(vals, max_vec);
            
            // Approximate exp using polynomial (for better performance)
            let exp_vals = self.fast_exp_avx512(shifted);
            _mm512_storeu_ps(data.as_mut_ptr().add(offset), exp_vals);
            partial_sums = _mm512_add_ps(partial_sums, exp_vals);
        }
        
        sum += _mm512_reduce_add_ps(partial_sums);
        
        // Handle remaining elements
        for i in (chunks * 16)..len {
            let exp_val = (data[i] - max_val).exp();
            data[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize by sum
        let inv_sum = _mm512_set1_ps(1.0 / sum);
        for i in 0..chunks {
            let offset = i * 16;
            let vals = _mm512_loadu_ps(data.as_ptr().add(offset));
            let normalized = _mm512_mul_ps(vals, inv_sum);
            _mm512_storeu_ps(data.as_mut_ptr().add(offset), normalized);
        }
        
        for i in (chunks * 16)..len {
            data[i] /= sum;
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn fast_exp_avx512(&self, x: __m512) -> __m512 {
        // Fast exp approximation using polynomial
        // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        let one = _mm512_set1_ps(1.0);
        let x2 = _mm512_mul_ps(x, x);
        let x3 = _mm512_mul_ps(x2, x);
        let x4 = _mm512_mul_ps(x3, x);
        
        let term1 = one;
        let term2 = x;
        let term3 = _mm512_mul_ps(x2, _mm512_set1_ps(0.5));
        let term4 = _mm512_mul_ps(x3, _mm512_set1_ps(1.0 / 6.0));
        let term5 = _mm512_mul_ps(x4, _mm512_set1_ps(1.0 / 24.0));
        
        _mm512_add_ps(
            _mm512_add_ps(_mm512_add_ps(term1, term2), _mm512_add_ps(term3, term4)),
            term5
        )
    }
}

/// AVX2 implementation.
pub struct Avx2Implementation;

impl Avx2Implementation {
    pub fn new() -> Self {
        Self
    }
}

impl SimdOperations for Avx2Implementation {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        unsafe { self.dot_product_avx2(a, b) }
    }
    
    fn relu(&self, data: &mut [f32]) -> Result<()> {
        unsafe { self.relu_avx2(data) };
        Ok(())
    }
    
    fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        // Use fallback for sigmoid in AVX2 (more complex to implement efficiently)
        for x in data.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        Ok(())
    }
    
    fn tanh(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = x.tanh();
        }
        Ok(())
    }
    
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        unsafe { self.add_avx2(a, b, result) };
        Ok(())
    }
    
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        unsafe { self.multiply_avx2(a, b, result) };
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
        unsafe { self.matrix_multiply_avx2(a, b, &mut result, rows_a, cols_a, cols_b) };
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
        // Use fallback implementation for convolution
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
                        
                        if in_y >= padding && in_y < input_height + padding &&
                           in_x >= padding && in_x < input_width + padding {
                            let in_y_actual = in_y - padding;
                            let in_x_actual = in_x - padding;
                            sum += input[in_y_actual * input_width + in_x_actual] *
                                   kernel[ky * kernel_size + kx];
                        }
                    }
                }
                result[out_y * output_width + out_x] = sum;
            }
        }
        
        Ok(result)
    }
    
    fn softmax(&self, data: &mut [f32]) -> Result<()> {
        // Use fallback implementation
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        
        for x in data.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        
        for x in data.iter_mut() {
            *x /= sum;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AVX2"
    }
}

impl Avx2Implementation {
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let mut sum = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        
        // Horizontal sum
        let high = _mm256_extractf128_ps(sum, 1);
        let low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(&self, data: &mut [f32]) {
        let len = data.len();
        let zero = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            let result = _mm256_max_ps(values, zero);
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), result);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            data[i] = data[i].max(0.0);
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let sum = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), sum);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            result[i] = a[i] + b[i];
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn multiply_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let prod = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), prod);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            result[i] = a[i] * b[i];
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn matrix_multiply_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) {
        // Simple implementation for AVX2
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
    }
}

/// Basic AVX implementation.
pub struct AvxImplementation;

impl AvxImplementation {
    pub fn new() -> Self {
        Self
    }
}

impl SimdOperations for AvxImplementation {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        // Similar to AVX2 but without FMA
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum += a_val * b_val;
        }
        Ok(sum)
    }
    
    fn relu(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = x.max(0.0);
        }
        Ok(())
    }
    
    fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        Ok(())
    }
    
    fn tanh(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = x.tanh();
        }
        Ok(())
    }
    
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        for ((a_val, b_val), res) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *res = a_val + b_val;
        }
        Ok(())
    }
    
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
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
        let mut result = vec![0.0; rows_a * cols_b];
        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..cols_a {
                    result[i * cols_b + j] += a[i * cols_a + k] * b[k * cols_b + j];
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
                        
                        if in_y >= padding && in_y < input_height + padding &&
                           in_x >= padding && in_x < input_width + padding {
                            let in_y_actual = in_y - padding;
                            let in_x_actual = in_x - padding;
                            sum += input[in_y_actual * input_width + in_x_actual] *
                                   kernel[ky * kernel_size + kx];
                        }
                    }
                }
                result[out_y * output_width + out_x] = sum;
            }
        }
        
        Ok(result)
    }
    
    fn softmax(&self, data: &mut [f32]) -> Result<()> {
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        
        for x in data.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        
        for x in data.iter_mut() {
            *x /= sum;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AVX"
    }
}

/// SSE implementation.
pub struct SseImplementation;

impl SseImplementation {
    pub fn new() -> Self {
        Self
    }
}

impl SimdOperations for SseImplementation {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum += a_val * b_val;
        }
        Ok(sum)
    }
    
    fn relu(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = x.max(0.0);
        }
        Ok(())
    }
    
    fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        Ok(())
    }
    
    fn tanh(&self, data: &mut [f32]) -> Result<()> {
        for x in data.iter_mut() {
            *x = x.tanh();
        }
        Ok(())
    }
    
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        for ((a_val, b_val), res) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *res = a_val + b_val;
        }
        Ok(())
    }
    
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
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
        let mut result = vec![0.0; rows_a * cols_b];
        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..cols_a {
                    result[i * cols_b + j] += a[i * cols_a + k] * b[k * cols_b + j];
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
                        
                        if in_y >= padding && in_y < input_height + padding &&
                           in_x >= padding && in_x < input_width + padding {
                            let in_y_actual = in_y - padding;
                            let in_x_actual = in_x - padding;
                            sum += input[in_y_actual * input_width + in_x_actual] *
                                   kernel[ky * kernel_size + kx];
                        }
                    }
                }
                result[out_y * output_width + out_x] = sum;
            }
        }
        
        Ok(result)
    }
    
    fn softmax(&self, data: &mut [f32]) -> Result<()> {
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        
        for x in data.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        
        for x in data.iter_mut() {
            *x /= sum;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "SSE"
    }
}