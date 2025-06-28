//! SIMD (Single Instruction, Multiple Data) optimization module.
//!
//! This module provides high-performance vectorized operations for different CPU architectures:
//! - x86_64: AVX2, AVX512 support
//! - ARM: NEON support
//! - Fallback: Portable implementations for unsupported architectures
//!
//! The module automatically detects CPU capabilities and selects the best available
//! implementation at runtime.

use crate::{Result, VeritasError};
use serde::{Deserialize, Serialize};
use std::fmt;

// Architecture-specific implementations
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// Fallback implementation for unsupported architectures
pub mod fallback;

// Core SIMD operations
pub mod core_ops;

// Runtime SIMD selection
pub mod runtime_select;

// Re-export platform-specific optimizations
#[cfg(target_arch = "x86_64")]
pub use self::x86_64::*;

#[cfg(target_arch = "aarch64")]
pub use self::aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use self::fallback::*;

// Re-export core SIMD operations
pub use self::core_ops::SimdCoreOps;

// Re-export runtime selection functionality
pub use self::runtime_select::{
    SimdDispatcher, SimdLevel, SimdInfo, 
    get_simd_level, is_simd_available, get_simd_info
};

/// SIMD feature detection and capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdFeatures {
    /// SSE support (x86_64)
    pub sse: bool,
    /// SSE2 support (x86_64)
    pub sse2: bool,
    /// SSE3 support (x86_64)
    pub sse3: bool,
    /// SSSE3 support (x86_64)
    pub ssse3: bool,
    /// SSE4.1 support (x86_64)
    pub sse41: bool,
    /// SSE4.2 support (x86_64)
    pub sse42: bool,
    /// AVX support (x86_64)
    pub avx: bool,
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// AVX512F support (x86_64)
    pub avx512f: bool,
    /// AVX512DQ support (x86_64)
    pub avx512dq: bool,
    /// FMA support (x86_64)
    pub fma: bool,
    /// NEON support (ARM)
    pub neon: bool,
}

impl Default for SimdFeatures {
    fn default() -> Self {
        Self {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse41: false,
            sse42: false,
            avx: false,
            avx2: false,
            avx512f: false,
            avx512dq: false,
            fma: false,
            neon: false,
        }
    }
}

impl fmt::Display for SimdFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut features = Vec::new();
        
        if self.sse { features.push("SSE"); }
        if self.sse2 { features.push("SSE2"); }
        if self.sse3 { features.push("SSE3"); }
        if self.ssse3 { features.push("SSSE3"); }
        if self.sse41 { features.push("SSE4.1"); }
        if self.sse42 { features.push("SSE4.2"); }
        if self.avx { features.push("AVX"); }
        if self.avx2 { features.push("AVX2"); }
        if self.avx512f { features.push("AVX512F"); }
        if self.avx512dq { features.push("AVX512DQ"); }
        if self.fma { features.push("FMA"); }
        if self.neon { features.push("NEON"); }
        
        if features.is_empty() {
            write!(f, "No SIMD features detected")
        } else {
            write!(f, "{}", features.join(", "))
        }
    }
}

/// SIMD configuration options.
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Detected or manually set features
    pub features: SimdFeatures,
    /// Prefer specific implementation
    pub preferred_impl: SimdImplementation,
    /// Enable/disable specific optimizations
    pub optimizations: SimdOptimizations,
}

/// SIMD implementation preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdImplementation {
    /// Automatically detect and use best available
    Auto,
    /// Force use of AVX512 (x86_64)
    AVX512,
    /// Force use of AVX2 (x86_64)
    AVX2,
    /// Force use of AVX (x86_64)
    AVX,
    /// Force use of SSE (x86_64)
    SSE,
    /// Force use of NEON (ARM)
    NEON,
    /// Use fallback implementation
    Fallback,
}

/// SIMD optimization settings.
#[derive(Debug, Clone)]
pub struct SimdOptimizations {
    /// Enable loop unrolling
    pub loop_unrolling: bool,
    /// Enable prefetching
    pub prefetching: bool,
    /// Enable branch prediction hints
    pub branch_hints: bool,
    /// Use aligned memory access when possible
    pub aligned_access: bool,
}

impl Default for SimdOptimizations {
    fn default() -> Self {
        Self {
            loop_unrolling: true,
            prefetching: true,
            branch_hints: true,
            aligned_access: true,
        }
    }
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            features: detect_cpu_features(),
            preferred_impl: SimdImplementation::Auto,
            optimizations: SimdOptimizations::default(),
        }
    }
}

impl SimdConfig {
    /// Create configuration with automatic feature detection.
    pub fn auto_detect() -> Self {
        Self::default()
    }
    
    /// Create conservative configuration (basic features only).
    pub fn conservative() -> Self {
        Self {
            features: SimdFeatures::default(),
            preferred_impl: SimdImplementation::Fallback,
            optimizations: SimdOptimizations {
                loop_unrolling: false,
                prefetching: false,
                branch_hints: false,
                aligned_access: true,
            },
        }
    }
    
    /// Create aggressive configuration (use all available features).
    pub fn aggressive() -> Self {
        let mut config = Self::auto_detect();
        config.preferred_impl = SimdImplementation::Auto;
        config.optimizations = SimdOptimizations {
            loop_unrolling: true,
            prefetching: true,
            branch_hints: true,
            aligned_access: true,
        };
        config
    }
    
    /// Force specific implementation.
    pub fn with_implementation(mut self, impl_type: SimdImplementation) -> Self {
        self.preferred_impl = impl_type;
        self
    }
    
    /// Set optimization settings.
    pub fn with_optimizations(mut self, optimizations: SimdOptimizations) -> Self {
        self.optimizations = optimizations;
        self
    }
}

/// Main SIMD processor that coordinates different implementations.
pub struct SimdProcessor {
    config: SimdConfig,
    implementation: Box<dyn SimdOperations + Send + Sync>,
}

impl SimdProcessor {
    /// Create a new SIMD processor with the given configuration.
    pub fn new(config: SimdConfig) -> Result<Self> {
        let implementation = select_implementation(&config)?;
        
        Ok(Self {
            config,
            implementation,
        })
    }
    
    /// Get the current configuration.
    pub fn config(&self) -> &SimdConfig {
        &self.config
    }
    
    /// Get supported features.
    pub fn supported_features(&self) -> SimdFeatures {
        self.config.features
    }
    
    /// Perform vectorized dot product.
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        self.implementation.dot_product(a, b)
    }
    
    /// Apply ReLU activation function.
    pub fn relu(&self, data: &mut [f32]) -> Result<()> {
        self.implementation.relu(data)
    }
    
    /// Apply sigmoid activation function.
    pub fn sigmoid(&self, data: &mut [f32]) -> Result<()> {
        self.implementation.sigmoid(data)
    }
    
    /// Apply tanh activation function.
    pub fn tanh(&self, data: &mut [f32]) -> Result<()> {
        self.implementation.tanh(data)
    }
    
    /// Perform element-wise addition.
    pub fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        self.implementation.add(a, b, result)
    }
    
    /// Perform element-wise subtraction.
    pub fn subtract(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        self.implementation.subtract(a, b, result)
    }
    
    /// Perform element-wise multiplication.
    pub fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(VeritasError::SimdError("Array lengths must match".to_string()));
        }
        
        self.implementation.multiply(a, b, result)
    }
    
    /// Perform matrix multiplication.
    pub fn matrix_multiply(
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
        
        self.implementation.matrix_multiply(a, b, rows_a, cols_a, cols_b)
    }
    
    /// Perform convolution operation.
    pub fn convolution(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_width: usize,
        input_height: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        self.implementation.convolution(
            input,
            kernel,
            input_width,
            input_height,
            kernel_size,
            stride,
            padding,
        )
    }
    
    /// Compute softmax activation.
    pub fn softmax(&self, data: &mut [f32]) -> Result<()> {
        self.implementation.softmax(data)
    }
}

/// Trait defining SIMD operations that must be implemented by each backend.
pub trait SimdOperations {
    /// Vectorized dot product
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32>;
    
    /// ReLU activation function
    fn relu(&self, data: &mut [f32]) -> Result<()>;
    
    /// Sigmoid activation function
    fn sigmoid(&self, data: &mut [f32]) -> Result<()>;
    
    /// Tanh activation function
    fn tanh(&self, data: &mut [f32]) -> Result<()>;
    
    /// Element-wise addition
    fn add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()>;
    
    /// Element-wise subtraction
    fn subtract(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()>;
    
    /// Element-wise multiplication
    fn multiply(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()>;
    
    /// Matrix multiplication
    fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f32>>;
    
    /// Convolution operation
    fn convolution(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_width: usize,
        input_height: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>>;
    
    /// Softmax activation
    fn softmax(&self, data: &mut [f32]) -> Result<()>;
    
    /// Get implementation name
    fn name(&self) -> &'static str;
}

/// Detect CPU features at runtime.
pub fn detect_cpu_features() -> SimdFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::detect_features()
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::detect_features()
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdFeatures::default()
    }
}

/// Get currently supported SIMD features.
pub fn get_supported_features() -> SimdFeatures {
    detect_cpu_features()
}

/// Select the best SIMD implementation based on configuration and capabilities.
fn select_implementation(config: &SimdConfig) -> Result<Box<dyn SimdOperations + Send + Sync>> {
    match config.preferred_impl {
        SimdImplementation::Auto => {
            // Automatically select the best available implementation
            #[cfg(target_arch = "x86_64")]
            {
                if config.features.avx512f {
                    Ok(Box::new(x86_64::Avx512Implementation::new()))
                } else if config.features.avx2 {
                    Ok(Box::new(x86_64::Avx2Implementation::new()))
                } else if config.features.avx {
                    Ok(Box::new(x86_64::AvxImplementation::new()))
                } else if config.features.sse2 {
                    Ok(Box::new(x86_64::SseImplementation::new()))
                } else {
                    Ok(Box::new(fallback::FallbackImplementation::new()))
                }
            }
            
            #[cfg(target_arch = "aarch64")]
            {
                if config.features.neon {
                    Ok(Box::new(aarch64::NeonImplementation::new()))
                } else {
                    Ok(Box::new(fallback::FallbackImplementation::new()))
                }
            }
            
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                Ok(Box::new(fallback::FallbackImplementation::new()))
            }
        }
        
        SimdImplementation::AVX512 => {
            #[cfg(target_arch = "x86_64")]
            {
                if config.features.avx512f {
                    Ok(Box::new(x86_64::Avx512Implementation::new()))
                } else {
                    Err(VeritasError::SimdError("AVX512 not supported".to_string()))
                }
            }
            
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(VeritasError::SimdError("AVX512 only available on x86_64".to_string()))
            }
        }
        
        SimdImplementation::AVX2 => {
            #[cfg(target_arch = "x86_64")]
            {
                if config.features.avx2 {
                    Ok(Box::new(x86_64::Avx2Implementation::new()))
                } else {
                    Err(VeritasError::SimdError("AVX2 not supported".to_string()))
                }
            }
            
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(VeritasError::SimdError("AVX2 only available on x86_64".to_string()))
            }
        }
        
        SimdImplementation::AVX => {
            #[cfg(target_arch = "x86_64")]
            {
                if config.features.avx {
                    Ok(Box::new(x86_64::AvxImplementation::new()))
                } else {
                    Err(VeritasError::SimdError("AVX not supported".to_string()))
                }
            }
            
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(VeritasError::SimdError("AVX only available on x86_64".to_string()))
            }
        }
        
        SimdImplementation::SSE => {
            #[cfg(target_arch = "x86_64")]
            {
                if config.features.sse2 {
                    Ok(Box::new(x86_64::SseImplementation::new()))
                } else {
                    Err(VeritasError::SimdError("SSE2 not supported".to_string()))
                }
            }
            
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(VeritasError::SimdError("SSE only available on x86_64".to_string()))
            }
        }
        
        SimdImplementation::NEON => {
            #[cfg(target_arch = "aarch64")]
            {
                if config.features.neon {
                    Ok(Box::new(aarch64::NeonImplementation::new()))
                } else {
                    Err(VeritasError::SimdError("NEON not supported".to_string()))
                }
            }
            
            #[cfg(not(target_arch = "aarch64"))]
            {
                Err(VeritasError::SimdError("NEON only available on ARM".to_string()))
            }
        }
        
        SimdImplementation::Fallback => {
            Ok(Box::new(fallback::FallbackImplementation::new()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        let features = detect_cpu_features();
        // Just ensure it doesn't panic
        println!("Detected features: {}", features);
    }
    
    #[test]
    fn test_simd_config_creation() {
        let config = SimdConfig::auto_detect();
        assert_eq!(config.preferred_impl, SimdImplementation::Auto);
        
        let conservative = SimdConfig::conservative();
        assert_eq!(conservative.preferred_impl, SimdImplementation::Fallback);
    }
    
    #[test]
    fn test_simd_processor_creation() {
        let config = SimdConfig::auto_detect();
        let processor = SimdProcessor::new(config);
        assert!(processor.is_ok());
    }
    
    #[test]
    fn test_basic_operations() {
        let config = SimdConfig::auto_detect();
        let processor = SimdProcessor::new(config).unwrap();
        
        // Test dot product
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let result = processor.dot_product(&a, &b).unwrap();
        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        
        // Test ReLU
        let mut data = vec![-1.0, 0.0, 1.0, -2.0, 3.0];
        processor.relu(&mut data).unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }
}