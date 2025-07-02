// SIMD optimization utilities for neural operations
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SIMDProcessor {
    simd_available: bool,
}

#[wasm_bindgen]
impl SIMDProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SIMDProcessor {
        SIMDProcessor {
            simd_available: crate::utils::detect_simd_support(),
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn simd_available(&self) -> bool {
        self.simd_available
    }
    
    #[wasm_bindgen]
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Vector lengths must match");
        }
        
        if self.simd_available {
            self.simd_vector_add(a, b)
        } else {
            self.scalar_vector_add(a, b)
        }
    }
    
    #[wasm_bindgen]
    pub fn vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Vector lengths must match");
        }
        
        if self.simd_available {
            self.simd_vector_multiply(a, b)
        } else {
            self.scalar_vector_multiply(a, b)
        }
    }
    
    #[wasm_bindgen]
    pub fn vector_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            panic!("Vector lengths must match");
        }
        
        if self.simd_available {
            self.simd_dot_product(a, b)
        } else {
            self.scalar_dot_product(a, b)
        }
    }
    
    #[wasm_bindgen]
    pub fn matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        if self.simd_available {
            self.simd_matrix_multiply(a, b, rows_a, cols_a, cols_b)
        } else {
            self.scalar_matrix_multiply(a, b, rows_a, cols_a, cols_b)
        }
    }
    
    #[wasm_bindgen]
    pub fn activation_relu(&self, input: &[f32]) -> Vec<f32> {
        if self.simd_available {
            self.simd_relu(input)
        } else {
            self.scalar_relu(input)
        }
    }
    
    #[wasm_bindgen]
    pub fn activation_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        if self.simd_available {
            self.simd_sigmoid(input)
        } else {
            self.scalar_sigmoid(input)
        }
    }
    
    // SIMD implementations (when available)
    #[cfg(target_feature = "simd128")]
    fn simd_vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        use std::arch::wasm32::*;
        
        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;
        
        unsafe {
            // Process 4 elements at a time using SIMD
            for i in 0..chunks {
                let offset = i * 4;
                let va = v128_load(&a[offset] as *const f32 as *const v128);
                let vb = v128_load(&b[offset] as *const f32 as *const v128);
                let sum = f32x4_add(va, vb);
                
                let mut temp = [0f32; 4];
                v128_store(&mut temp[0] as *mut f32 as *mut v128, sum);
                result.extend_from_slice(&temp);
            }
            
            // Handle remaining elements
            for i in (chunks * 4)..a.len() {
                result.push(a[i] + b[i]);
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.scalar_vector_add(a, b)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        use std::arch::wasm32::*;
        
        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;
        
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let va = v128_load(&a[offset] as *const f32 as *const v128);
                let vb = v128_load(&b[offset] as *const f32 as *const v128);
                let product = f32x4_mul(va, vb);
                
                let mut temp = [0f32; 4];
                v128_store(&mut temp[0] as *mut f32 as *mut v128, product);
                result.extend_from_slice(&temp);
            }
            
            for i in (chunks * 4)..a.len() {
                result.push(a[i] * b[i]);
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.scalar_vector_multiply(a, b)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::wasm32::*;
        
        let chunks = a.len() / 4;
        let mut sum = 0.0f32;
        
        unsafe {
            let mut acc = f32x4_splat(0.0);
            
            for i in 0..chunks {
                let offset = i * 4;
                let va = v128_load(&a[offset] as *const f32 as *const v128);
                let vb = v128_load(&b[offset] as *const f32 as *const v128);
                let product = f32x4_mul(va, vb);
                acc = f32x4_add(acc, product);
            }
            
            // Sum the accumulator lanes
            let mut temp = [0f32; 4];
            v128_store(&mut temp[0] as *mut f32 as *mut v128, acc);
            sum = temp.iter().sum();
            
            // Handle remaining elements
            for i in (chunks * 4)..a.len() {
                sum += a[i] * b[i];
            }
        }
        
        sum
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        self.scalar_dot_product(a, b)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_relu(&self, input: &[f32]) -> Vec<f32> {
        use std::arch::wasm32::*;
        
        let mut result = Vec::with_capacity(input.len());
        let chunks = input.len() / 4;
        
        unsafe {
            let zero = f32x4_splat(0.0);
            
            for i in 0..chunks {
                let offset = i * 4;
                let v = v128_load(&input[offset] as *const f32 as *const v128);
                let relu = f32x4_max(v, zero);
                
                let mut temp = [0f32; 4];
                v128_store(&mut temp[0] as *mut f32 as *mut v128, relu);
                result.extend_from_slice(&temp);
            }
            
            for i in (chunks * 4)..input.len() {
                result.push(input[i].max(0.0));
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_relu(&self, input: &[f32]) -> Vec<f32> {
        self.scalar_relu(input)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        // SIMD sigmoid is complex, so we'll use a fast approximation
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        self.scalar_sigmoid(input)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        // Simplified SIMD matrix multiplication
        // In production, this would use blocked algorithms for cache efficiency
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        self.scalar_matrix_multiply(a, b, rows_a, cols_a, cols_b)
    }
    
    // Scalar fallback implementations
    fn scalar_vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
    
    fn scalar_vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }
    
    fn scalar_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    fn scalar_relu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x.max(0.0)).collect()
    }
    
    fn scalar_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }
    
    fn scalar_matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
        
        result
    }
}

// Benchmark utilities
#[wasm_bindgen]
pub fn benchmark_simd_performance(size: usize) -> JsValue {
    let processor = SIMDProcessor::new();
    let timer = crate::utils::PerformanceTimer::new();
    
    // Create test data
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    
    // Benchmark vector operations
    timer.reset();
    let _ = processor.vector_multiply(&a, &b);
    let simd_time = timer.elapsed();
    
    // Force scalar operations for comparison
    let scalar_processor = SIMDProcessor { simd_available: false };
    timer.reset();
    let _ = scalar_processor.vector_multiply(&a, &b);
    let scalar_time = timer.elapsed();
    
    let speedup = if simd_time > 0.0 { scalar_time / simd_time } else { 1.0 };
    
    let results = serde_json::json!({
        "simd_available": processor.simd_available,
        "vector_size": size,
        "simd_time_ms": simd_time,
        "scalar_time_ms": scalar_time,
        "speedup_factor": speedup,
        "efficiency_percent": (speedup * 25.0).min(100.0) // Theoretical max 4x for SIMD
    });
    
    serde_wasm_bindgen::to_value(&results).unwrap()
}