//! Runtime SIMD implementation selection and fallback management
//!
//! This module provides runtime detection and selection of the best available
//! SIMD implementation, with automatic fallback to scalar implementations
//! when SIMD is not available.

use crate::{Result, VeritasError};
use std::sync::Once;

/// Global initialization for SIMD runtime selection
static INIT: Once = Once::new();
static mut SIMD_AVAILABLE: bool = false;
static mut SIMD_LEVEL: SimdLevel = SimdLevel::None;

/// Available SIMD levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    None = 0,
    SSE = 1,
    SSE2 = 2,
    SSE3 = 3,
    SSSE3 = 4,
    SSE41 = 5,
    SSE42 = 6,
    AVX = 7,
    AVX2 = 8,
    AVX512 = 9,
    NEON = 10,
}

impl SimdLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SimdLevel::None => "None (Scalar)",
            SimdLevel::SSE => "SSE",
            SimdLevel::SSE2 => "SSE2",
            SimdLevel::SSE3 => "SSE3",
            SimdLevel::SSSE3 => "SSSE3",
            SimdLevel::SSE41 => "SSE4.1",
            SimdLevel::SSE42 => "SSE4.2",
            SimdLevel::AVX => "AVX",
            SimdLevel::AVX2 => "AVX2",
            SimdLevel::AVX512 => "AVX512",
            SimdLevel::NEON => "NEON",
        }
    }
    
    /// Get vector width in elements
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::None => 1,
            SimdLevel::SSE | SimdLevel::SSE2 | SimdLevel::SSE3 | 
            SimdLevel::SSSE3 | SimdLevel::SSE41 | SimdLevel::SSE42 => 4,
            SimdLevel::AVX | SimdLevel::AVX2 => 8,
            SimdLevel::AVX512 => 16,
            SimdLevel::NEON => 4,
        }
    }
}

/// Initialize SIMD runtime detection
pub fn initialize() {
    INIT.call_once(|| {
        unsafe {
            detect_simd_support();
        }
    });
}

/// Detect available SIMD support
unsafe fn detect_simd_support() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        
        if is_x86_feature_detected!("avx512f") {
            SIMD_LEVEL = SimdLevel::AVX512;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("avx2") {
            SIMD_LEVEL = SimdLevel::AVX2;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("avx") {
            SIMD_LEVEL = SimdLevel::AVX;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("sse4.2") {
            SIMD_LEVEL = SimdLevel::SSE42;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("sse4.1") {
            SIMD_LEVEL = SimdLevel::SSE41;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("ssse3") {
            SIMD_LEVEL = SimdLevel::SSSE3;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("sse3") {
            SIMD_LEVEL = SimdLevel::SSE3;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("sse2") {
            SIMD_LEVEL = SimdLevel::SSE2;
            SIMD_AVAILABLE = true;
        } else if is_x86_feature_detected!("sse") {
            SIMD_LEVEL = SimdLevel::SSE;
            SIMD_AVAILABLE = true;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        
        if is_aarch64_feature_detected!("neon") {
            SIMD_LEVEL = SimdLevel::NEON;
            SIMD_AVAILABLE = true;
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SIMD_LEVEL = SimdLevel::None;
        SIMD_AVAILABLE = false;
    }
}

/// Get current SIMD level
pub fn get_simd_level() -> SimdLevel {
    initialize();
    unsafe { SIMD_LEVEL }
}

/// Check if SIMD is available
pub fn is_simd_available() -> bool {
    initialize();
    unsafe { SIMD_AVAILABLE }
}

/// Runtime dispatcher for SIMD operations
pub struct SimdDispatcher;

impl SimdDispatcher {
    /// Dispatch dot product operation
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
        if !is_simd_available() {
            return fallback::dot_product_scalar(a, b);
        }
        
        match get_simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::AVX2 | SimdLevel::AVX512 => {
                unsafe { x86_64::dot_product_avx2(a, b) }
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::SSE | SimdLevel::SSE2 | SimdLevel::SSE3 | 
            SimdLevel::SSSE3 | SimdLevel::SSE41 | SimdLevel::SSE42 => {
                unsafe { x86_64::dot_product_sse(a, b) }
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::NEON => {
                unsafe { aarch64::dot_product_neon(a, b) }
            }
            _ => fallback::dot_product_scalar(a, b),
        }
    }
    
    /// Dispatch ReLU activation
    #[inline]
    pub fn relu(data: &mut [f32]) -> Result<()> {
        if !is_simd_available() {
            return fallback::relu_scalar(data);
        }
        
        match get_simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::AVX2 | SimdLevel::AVX512 => {
                unsafe { x86_64::relu_avx2(data) }
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::SSE | SimdLevel::SSE2 | SimdLevel::SSE3 | 
            SimdLevel::SSSE3 | SimdLevel::SSE41 | SimdLevel::SSE42 => {
                unsafe { x86_64::relu_sse(data) }
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::NEON => {
                unsafe { aarch64::relu_neon(data) }
            }
            _ => fallback::relu_scalar(data),
        }
    }
    
    /// Select optimal chunk size for current SIMD level
    pub fn optimal_chunk_size() -> usize {
        get_simd_level().vector_width() * 4 // Process 4 vectors at a time
    }
    
    /// Check if pointer is aligned for SIMD access
    pub fn is_aligned(ptr: *const f32) -> bool {
        let alignment = match get_simd_level() {
            SimdLevel::AVX512 => 64,
            SimdLevel::AVX | SimdLevel::AVX2 => 32,
            SimdLevel::SSE | SimdLevel::SSE2 | SimdLevel::SSE3 | 
            SimdLevel::SSSE3 | SimdLevel::SSE41 | SimdLevel::SSE42 | 
            SimdLevel::NEON => 16,
            SimdLevel::None => 4,
        };
        
        (ptr as usize) % alignment == 0
    }
}

/// Fallback scalar implementations
mod fallback {
    use crate::{Result, VeritasError};
    
    pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        Ok(sum)
    }
    
    pub fn relu_scalar(data: &mut [f32]) -> Result<()> {
        for val in data.iter_mut() {
            *val = val.max(0.0);
        }
        Ok(())
    }
}

/// x86_64 SIMD implementations
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use crate::{Result, VeritasError};
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let len = a.len();
        let mut sum = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(&a[i * 8]);
            let b_vec = _mm256_loadu_ps(&b[i * 8]);
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, prod);
        }
        
        // Horizontal sum
        let sum_array: [f32; 8] = std::mem::transmute(sum);
        let mut result = sum_array.iter().sum::<f32>();
        
        // Handle remainder
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    #[target_feature(enable = "sse2")]
    pub unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let len = a.len();
        let mut sum = _mm_setzero_ps();
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let a_vec = _mm_loadu_ps(&a[i * 4]);
            let b_vec = _mm_loadu_ps(&b[i * 4]);
            let prod = _mm_mul_ps(a_vec, b_vec);
            sum = _mm_add_ps(sum, prod);
        }
        
        // Horizontal sum
        let sum_array: [f32; 4] = std::mem::transmute(sum);
        let mut result = sum_array.iter().sum::<f32>();
        
        // Handle remainder
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn relu_avx2(data: &mut [f32]) -> Result<()> {
        let len = data.len();
        let zero = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let val = _mm256_loadu_ps(&data[i * 8]);
            let result = _mm256_max_ps(val, zero);
            _mm256_storeu_ps(&mut data[i * 8], result);
        }
        
        // Handle remainder
        for i in (chunks * 8)..len {
            data[i] = data[i].max(0.0);
        }
        
        Ok(())
    }
    
    #[target_feature(enable = "sse")]
    pub unsafe fn relu_sse(data: &mut [f32]) -> Result<()> {
        let len = data.len();
        let zero = _mm_setzero_ps();
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let val = _mm_loadu_ps(&data[i * 4]);
            let result = _mm_max_ps(val, zero);
            _mm_storeu_ps(&mut data[i * 4], result);
        }
        
        // Handle remainder
        for i in (chunks * 4)..len {
            data[i] = data[i].max(0.0);
        }
        
        Ok(())
    }
}

/// ARM NEON SIMD implementations
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use crate::{Result, VeritasError};
    use std::arch::aarch64::*;
    
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VeritasError::DimensionMismatch);
        }
        
        let len = a.len();
        let mut sum = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let a_vec = vld1q_f32(&a[i * 4]);
            let b_vec = vld1q_f32(&b[i * 4]);
            sum = vmlaq_f32(sum, a_vec, b_vec);
        }
        
        // Horizontal sum
        let mut result = vaddvq_f32(sum);
        
        // Handle remainder
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    pub unsafe fn relu_neon(data: &mut [f32]) -> Result<()> {
        let len = data.len();
        let zero = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let val = vld1q_f32(&data[i * 4]);
            let result = vmaxq_f32(val, zero);
            vst1q_f32(&mut data[i * 4], result);
        }
        
        // Handle remainder
        for i in (chunks * 4)..len {
            data[i] = data[i].max(0.0);
        }
        
        Ok(())
    }
}

/// Get runtime information about SIMD support
pub fn get_simd_info() -> SimdInfo {
    initialize();
    
    SimdInfo {
        available: unsafe { SIMD_AVAILABLE },
        level: unsafe { SIMD_LEVEL },
        vector_width: unsafe { SIMD_LEVEL.vector_width() },
        features: get_feature_list(),
    }
}

/// SIMD information structure
#[derive(Debug, Clone)]
pub struct SimdInfo {
    pub available: bool,
    pub level: SimdLevel,
    pub vector_width: usize,
    pub features: Vec<String>,
}

impl std::fmt::Display for SimdInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SIMD Support Information:")?;
        writeln!(f, "  Available: {}", self.available)?;
        writeln!(f, "  Level: {}", self.level.name())?;
        writeln!(f, "  Vector Width: {} elements", self.vector_width)?;
        writeln!(f, "  Features: {}", self.features.join(", "))?;
        Ok(())
    }
}

/// Get list of available features
fn get_feature_list() -> Vec<String> {
    let mut features = Vec::new();
    
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse") { features.push("SSE".to_string()); }
        if std::is_x86_feature_detected!("sse2") { features.push("SSE2".to_string()); }
        if std::is_x86_feature_detected!("sse3") { features.push("SSE3".to_string()); }
        if std::is_x86_feature_detected!("ssse3") { features.push("SSSE3".to_string()); }
        if std::is_x86_feature_detected!("sse4.1") { features.push("SSE4.1".to_string()); }
        if std::is_x86_feature_detected!("sse4.2") { features.push("SSE4.2".to_string()); }
        if std::is_x86_feature_detected!("avx") { features.push("AVX".to_string()); }
        if std::is_x86_feature_detected!("avx2") { features.push("AVX2".to_string()); }
        if std::is_x86_feature_detected!("avx512f") { features.push("AVX512F".to_string()); }
        if std::is_x86_feature_detected!("fma") { features.push("FMA".to_string()); }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") { 
            features.push("NEON".to_string()); 
        }
    }
    
    if features.is_empty() {
        features.push("None (Scalar only)".to_string());
    }
    
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_detection() {
        initialize();
        let info = get_simd_info();
        println!("{}", info);
        
        // Just ensure it doesn't panic
        assert!(info.vector_width >= 1);
    }
    
    #[test]
    fn test_dispatcher_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        
        let result = SimdDispatcher::dot_product(&a, &b).unwrap();
        assert_eq!(result, 40.0);
    }
    
    #[test]
    fn test_dispatcher_relu() {
        let mut data = vec![-1.0, 0.0, 1.0, -2.0, 3.0];
        SimdDispatcher::relu(&mut data).unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }
    
    #[test]
    fn test_optimal_chunk_size() {
        let chunk_size = SimdDispatcher::optimal_chunk_size();
        assert!(chunk_size >= 4); // At least scalar * 4
    }
}