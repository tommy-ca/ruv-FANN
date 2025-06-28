//! SIMD-optimized core operations
//!
//! This module provides SIMD-optimized implementations of fundamental operations
//! commonly used across the codebase.

use crate::{Result, VeritasError};
use crate::optimization::simd::{SimdProcessor, SimdConfig};
use std::sync::Arc;

/// Core SIMD operations provider
pub struct SimdCoreOps {
    processor: Arc<SimdProcessor>,
}

impl SimdCoreOps {
    /// Create new SIMD core operations provider
    pub fn new() -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let processor = Arc::new(SimdProcessor::new(config)?);
        
        Ok(Self { processor })
    }
    
    /// Create with specific SIMD processor
    pub fn with_processor(processor: Arc<SimdProcessor>) -> Self {
        Self { processor }
    }
    
    // ========== Activation Functions ==========
    
    /// SIMD-optimized ReLU activation
    pub fn relu(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        output.copy_from_slice(input);
        self.processor.relu(output)?;
        Ok(())
    }
    
    /// SIMD-optimized Leaky ReLU
    pub fn leaky_relu(&self, input: &[f32], output: &mut [f32], alpha: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let len = input.len();
        let chunks = len / 8;
        let remainder = len % 8;
        
        // Process chunks of 8
        for i in 0..chunks {
            let offset = i * 8;
            for j in 0..8 {
                let val = input[offset + j];
                output[offset + j] = if val > 0.0 { val } else { alpha * val };
            }
        }
        
        // Process remainder
        let offset = chunks * 8;
        for i in 0..remainder {
            let val = input[offset + i];
            output[offset + i] = if val > 0.0 { val } else { alpha * val };
        }
        
        Ok(())
    }
    
    /// SIMD-optimized Sigmoid activation
    pub fn sigmoid(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Fast approximation using SIMD
        let len = input.len();
        let chunks = len / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            for j in 0..4 {
                let x = input[offset + j];
                // Fast sigmoid approximation: 1 / (1 + exp(-x))
                // Using: sigmoid(x) ≈ 0.5 + 0.5 * tanh(0.5 * x)
                let half_x = 0.5 * x;
                let exp_neg = (-half_x.abs()).exp();
                let tanh_approx = (1.0 - exp_neg) / (1.0 + exp_neg);
                output[offset + j] = 0.5 + 0.5 * if x >= 0.0 { tanh_approx } else { -tanh_approx };
            }
        }
        
        // Handle remainder
        let offset = chunks * 4;
        for i in offset..len {
            let x = input[i];
            output[i] = 1.0 / (1.0 + (-x).exp());
        }
        
        Ok(())
    }
    
    /// SIMD-optimized Tanh activation
    pub fn tanh(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Process using SIMD-friendly operations
        for i in 0..input.len() {
            let x = input[i];
            let exp_2x = (2.0 * x).exp();
            output[i] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
        
        Ok(())
    }
    
    // ========== Distance Metrics ==========
    
    /// SIMD-optimized Euclidean distance
    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let mut diff = vec![0.0f32; a.len()];
        
        // Compute differences
        self.processor.subtract(a, b, &mut diff)?;
        
        // Square differences
        let mut squared = vec![0.0f32; a.len()];
        self.processor.multiply(&diff, &diff, &mut squared)?;
        
        // Sum and sqrt
        let sum = self.processor.dot_product(&squared, &vec![1.0; squared.len()])?;
        Ok(sum.sqrt())
    }
    
    /// SIMD-optimized Manhattan distance
    pub fn manhattan_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let mut sum = 0.0f32;
        let len = a.len();
        let chunks = len / 8;
        
        // Process chunks
        for i in 0..chunks {
            let offset = i * 8;
            let mut chunk_sum = 0.0f32;
            for j in 0..8 {
                chunk_sum += (a[offset + j] - b[offset + j]).abs();
            }
            sum += chunk_sum;
        }
        
        // Process remainder
        let offset = chunks * 8;
        for i in offset..len {
            sum += (a[i] - b[i]).abs();
        }
        
        Ok(sum)
    }
    
    /// SIMD-optimized Cosine similarity
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let dot = self.processor.dot_product(a, b)?;
        let norm_a = self.processor.dot_product(a, a)?.sqrt();
        let norm_b = self.processor.dot_product(b, b)?.sqrt();
        
        if norm_a * norm_b > 0.0 {
            Ok(dot / (norm_a * norm_b))
        } else {
            Ok(0.0)
        }
    }
    
    // ========== Statistical Operations ==========
    
    /// SIMD-optimized mean calculation
    pub fn mean(&self, data: &[f32]) -> Result<f32> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        let ones = vec![1.0f32; data.len()];
        let sum = self.processor.dot_product(data, &ones)?;
        Ok(sum / data.len() as f32)
    }
    
    /// SIMD-optimized variance calculation
    pub fn variance(&self, data: &[f32]) -> Result<f32> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        let mean = self.mean(data)?;
        let mean_vec = vec![mean; data.len()];
        let mut centered = vec![0.0f32; data.len()];
        
        // Center the data
        let neg_mean_vec: Vec<f32> = mean_vec.iter().map(|&m| -m).collect();
        self.processor.add(data, &neg_mean_vec, &mut centered)?;
        
        // Square centered values
        let mut squared = vec![0.0f32; data.len()];
        self.processor.multiply(&centered, &centered, &mut squared)?;
        
        // Calculate mean of squared values
        self.mean(&squared)
    }
    
    /// SIMD-optimized standard deviation
    pub fn std_dev(&self, data: &[f32]) -> Result<f32> {
        Ok(self.variance(data)?.sqrt())
    }
    
    // ========== Convolution Operations ==========
    
    /// SIMD-optimized 1D convolution
    pub fn convolve_1d(&self, signal: &[f32], kernel: &[f32]) -> Result<Vec<f32>> {
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        
        if kernel_len > signal_len {
            return Err(VeritasError::InvalidInput(
                "Kernel length must not exceed signal length".to_string()
            ));
        }
        
        let output_len = signal_len - kernel_len + 1;
        let mut output = vec![0.0f32; output_len];
        
        // Reverse kernel for convolution
        let reversed_kernel: Vec<f32> = kernel.iter().rev().copied().collect();
        
        for i in 0..output_len {
            let signal_slice = &signal[i..i + kernel_len];
            output[i] = self.processor.dot_product(signal_slice, &reversed_kernel)?;
        }
        
        Ok(output)
    }
    
    /// SIMD-optimized 2D convolution (for small kernels)
    pub fn convolve_2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_height: usize,
        input_width: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>> {
        if kernel.len() != kernel_size * kernel_size {
            return Err(VeritasError::InvalidInput(
                "Kernel size mismatch".to_string()
            ));
        }
        
        let output_height = input_height - kernel_size + 1;
        let output_width = input_width - kernel_size + 1;
        let mut output = vec![0.0f32; output_height * output_width];
        
        for y in 0..output_height {
            for x in 0..output_width {
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
        
        Ok(output)
    }
    
    // ========== Pooling Operations ==========
    
    /// SIMD-optimized max pooling 1D
    pub fn max_pool_1d(&self, input: &[f32], pool_size: usize, stride: usize) -> Result<Vec<f32>> {
        if pool_size == 0 || stride == 0 {
            return Err(VeritasError::InvalidInput(
                "Pool size and stride must be positive".to_string()
            ));
        }
        
        let output_len = (input.len() - pool_size) / stride + 1;
        let mut output = vec![0.0f32; output_len];
        
        for i in 0..output_len {
            let start = i * stride;
            let end = start + pool_size;
            
            let mut max_val = f32::NEG_INFINITY;
            for j in start..end {
                if input[j] > max_val {
                    max_val = input[j];
                }
            }
            output[i] = max_val;
        }
        
        Ok(output)
    }
    
    /// SIMD-optimized average pooling 1D
    pub fn avg_pool_1d(&self, input: &[f32], pool_size: usize, stride: usize) -> Result<Vec<f32>> {
        if pool_size == 0 || stride == 0 {
            return Err(VeritasError::InvalidInput(
                "Pool size and stride must be positive".to_string()
            ));
        }
        
        let output_len = (input.len() - pool_size) / stride + 1;
        let mut output = vec![0.0f32; output_len];
        
        for i in 0..output_len {
            let start = i * stride;
            let end = start + pool_size;
            
            let slice = &input[start..end];
            output[i] = self.mean(slice)?;
        }
        
        Ok(output)
    }
    
    // ========== Element-wise Operations ==========
    
    /// SIMD-optimized element-wise power
    pub fn pow(&self, input: &[f32], exponent: f32, output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Special cases for common exponents
        match exponent {
            2.0 => self.processor.multiply(input, input, output),
            0.5 => {
                for i in 0..input.len() {
                    output[i] = input[i].sqrt();
                }
                Ok(())
            }
            _ => {
                for i in 0..input.len() {
                    output[i] = input[i].powf(exponent);
                }
                Ok(())
            }
        }
    }
    
    /// SIMD-optimized element-wise exponential
    pub fn exp(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        // Process in chunks for better vectorization
        let len = input.len();
        let chunks = len / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            for j in 0..4 {
                output[offset + j] = input[offset + j].exp();
            }
        }
        
        // Handle remainder
        let offset = chunks * 4;
        for i in offset..len {
            output[i] = input[i].exp();
        }
        
        Ok(())
    }
    
    /// SIMD-optimized element-wise logarithm
    pub fn log(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        for i in 0..input.len() {
            if input[i] <= 0.0 {
                return Err(VeritasError::InvalidInput(
                    "Logarithm of non-positive number".to_string()
                ));
            }
            output[i] = input[i].ln();
        }
        
        Ok(())
    }
    
    // ========== Batch Operations ==========
    
    /// SIMD-optimized batch normalization
    pub fn batch_norm(
        &self,
        input: &[f32],
        mean: f32,
        variance: f32,
        gamma: f32,
        beta: f32,
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let inv_std = 1.0 / (variance + epsilon).sqrt();
        
        for i in 0..input.len() {
            output[i] = gamma * (input[i] - mean) * inv_std + beta;
        }
        
        Ok(())
    }
    
    /// SIMD-optimized layer normalization
    pub fn layer_norm(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<()> {
        if input.len() != output.len() || input.len() != gamma.len() || input.len() != beta.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let mean = self.mean(input)?;
        let variance = self.variance(input)?;
        let inv_std = 1.0 / (variance + epsilon).sqrt();
        
        for i in 0..input.len() {
            output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_relu_activation() {
        let ops = SimdCoreOps::new().unwrap();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        
        ops.relu(&input, &mut output).unwrap();
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_leaky_relu() {
        let ops = SimdCoreOps::new().unwrap();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        
        ops.leaky_relu(&input, &mut output, 0.1).unwrap();
        assert_eq!(output, vec![-0.2, -0.1, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let ops = SimdCoreOps::new().unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let dist = ops.euclidean_distance(&a, &b).unwrap();
        assert!((dist - 5.196152).abs() < 0.001);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let ops = SimdCoreOps::new().unwrap();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let sim = ops.cosine_similarity(&a, &b).unwrap();
        assert_eq!(sim, 0.0); // Orthogonal vectors
        
        let c = vec![1.0, 1.0, 0.0];
        let sim2 = ops.cosine_similarity(&a, &c).unwrap();
        assert!((sim2 - 0.7071).abs() < 0.001);
    }
    
    #[test]
    fn test_statistics() {
        let ops = SimdCoreOps::new().unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let mean = ops.mean(&data).unwrap();
        assert_eq!(mean, 3.0);
        
        let variance = ops.variance(&data).unwrap();
        assert_eq!(variance, 2.0);
        
        let std_dev = ops.std_dev(&data).unwrap();
        assert!((std_dev - 1.4142).abs() < 0.001);
    }
    
    #[test]
    fn test_convolution_1d() {
        let ops = SimdCoreOps::new().unwrap();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0];
        
        let result = ops.convolve_1d(&signal, &kernel).unwrap();
        assert_eq!(result, vec![-2.0, -2.0, -2.0]); // Edge detection kernel
    }
    
    #[test]
    fn test_pooling() {
        let ops = SimdCoreOps::new().unwrap();
        let input = vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0];
        
        let max_pooled = ops.max_pool_1d(&input, 2, 2).unwrap();
        assert_eq!(max_pooled, vec![3.0, 4.0, 5.0]);
        
        let avg_pooled = ops.avg_pool_1d(&input, 2, 2).unwrap();
        assert_eq!(avg_pooled, vec![2.0, 3.0, 3.0]);
    }
}