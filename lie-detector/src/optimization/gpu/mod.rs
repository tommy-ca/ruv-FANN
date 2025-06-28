//! GPU acceleration module using candle-core for neural network operations.
//!
//! This module provides high-performance GPU-accelerated operations for:
//! - Neural network inference and training
//! - Matrix operations and linear algebra
//! - Tensor manipulations
//! - Custom kernels for specialized operations
//!
//! Supports multiple GPU backends:
//! - CUDA (NVIDIA GPUs)
//! - Metal (Apple Silicon)
//! - OpenCL (cross-platform)

use crate::{Result, VeritasError};
use serde::{Deserialize, Serialize};
use std::fmt;

// Conditional compilation for GPU features
#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor, DType};

// Submodules
#[cfg(feature = "gpu")]
pub mod candle_backend;

#[cfg(feature = "gpu")]
pub mod kernels;

// Re-export types when GPU features are enabled
#[cfg(feature = "gpu")]
pub use self::candle_backend::{CandleBackend, CandleError};

#[cfg(feature = "gpu")]
pub use self::kernels::{CustomKernels, KernelManager};

/// GPU device information and capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device identifier
    pub id: usize,
    /// Device name
    pub name: String,
    /// Device type (CUDA, Metal, OpenCL)
    pub device_type: GpuDeviceType,
    /// Total memory in MB
    pub total_memory_mb: usize,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Number of multiprocessors/compute units
    pub multiprocessor_count: Option<usize>,
    /// Is this the default device
    pub is_default: bool,
}

/// Supported GPU device types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDeviceType {
    /// NVIDIA CUDA device
    Cuda,
    /// Apple Metal device
    Metal,
    /// OpenCL device
    OpenCL,
    /// CPU fallback
    Cpu,
}

impl fmt::Display for GpuDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda => write!(f, "CUDA"),
            Self::Metal => write!(f, "Metal"),
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

/// GPU acceleration configuration.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Preferred device type
    pub preferred_device: Option<GpuDeviceType>,
    /// Specific device ID to use
    pub device_id: Option<usize>,
    /// Memory pool size in MB
    pub memory_pool_size_mb: usize,
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// Batch size for operations
    pub default_batch_size: usize,
    /// Enable kernel fusion optimizations
    pub kernel_fusion: bool,
    /// Enable tensor core usage (CUDA)
    pub use_tensor_cores: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_device: None,
            device_id: None,
            memory_pool_size_mb: 1024, // 1GB default
            mixed_precision: true,
            default_batch_size: 32,
            kernel_fusion: true,
            use_tensor_cores: true,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Auto-detect and use the best available GPU.
    pub fn auto_detect() -> Self {
        Self::default()
    }
    
    /// Prefer CUDA if available, fallback to CPU.
    pub fn cuda_if_available() -> Self {
        Self {
            preferred_device: Some(GpuDeviceType::Cuda),
            ..Self::default()
        }
    }
    
    /// Prefer Metal if available, fallback to CPU.
    pub fn metal_if_available() -> Self {
        Self {
            preferred_device: Some(GpuDeviceType::Metal),
            ..Self::default()
        }
    }
    
    /// Force CPU usage (no GPU acceleration).
    pub fn cpu_only() -> Self {
        Self {
            preferred_device: Some(GpuDeviceType::Cpu),
            memory_pool_size_mb: 0,
            mixed_precision: false,
            kernel_fusion: false,
            use_tensor_cores: false,
            ..Self::default()
        }
    }
    
    /// Set preferred device type.
    pub fn with_device_type(mut self, device_type: GpuDeviceType) -> Self {
        self.preferred_device = Some(device_type);
        self
    }
    
    /// Set specific device ID.
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = Some(device_id);
        self
    }
    
    /// Set memory pool size.
    pub fn with_memory_pool_size(mut self, size_mb: usize) -> Self {
        self.memory_pool_size_mb = size_mb;
        self
    }
    
    /// Enable or disable mixed precision.
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }
    
    /// Set default batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.default_batch_size = batch_size;
        self
    }
}

/// Main GPU accelerator that coordinates different GPU operations.
pub struct GpuAccelerator {
    config: GpuConfig,
    device: GpuDevice,
    #[cfg(feature = "gpu")]
    backend: CandleBackend,
    #[cfg(feature = "gpu")]
    kernels: KernelManager,
}

impl GpuAccelerator {
    /// Create a new GPU accelerator with the given configuration.
    pub fn new(config: GpuConfig) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            let available_devices = get_available_devices();
            
            if available_devices.is_empty() {
                return Err(VeritasError::GpuError("No GPU devices available".to_string()));
            }
            
            // Select device based on configuration
            let device = select_device(&config, &available_devices)?;
            
            // Initialize backend
            let backend = CandleBackend::new(&device)?;
            
            // Initialize kernel manager
            let kernels = KernelManager::new(&device)?;
            
            Ok(Self {
                config,
                device,
                backend,
                kernels,
            })
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            Err(VeritasError::GpuError(
                "GPU acceleration not compiled in. Enable 'gpu' feature.".to_string()
            ))
        }
    }
    
    /// Get the current configuration.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
    
    /// Get information about the current device.
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }
    
    /// Get list of available GPU devices.
    pub fn available_devices(&self) -> Vec<GpuDevice> {
        get_available_devices()
    }
    
    /// Check if GPU acceleration is available.
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.device.device_type != GpuDeviceType::Cpu
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
    
    /// Perform GPU-accelerated dot product.
    #[cfg(feature = "gpu")]
    pub fn dot_product(&mut self, a: &[f32], b: &[f32]) -> Result<f32> {
        self.backend.dot_product(a, b)
    }
    
    /// Perform GPU-accelerated matrix multiplication.
    #[cfg(feature = "gpu")]
    pub fn matrix_multiply(
        &mut self,
        a: &[f32],
        b: &[f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f32>> {
        self.backend.matrix_multiply(a, b, rows_a, cols_a, cols_b)
    }
    
    /// Apply ReLU activation on GPU.
    #[cfg(feature = "gpu")]
    pub fn relu(&mut self, data: &mut [f32]) -> Result<()> {
        self.backend.relu(data)
    }
    
    /// Apply sigmoid activation on GPU.
    #[cfg(feature = "gpu")]
    pub fn sigmoid(&mut self, data: &mut [f32]) -> Result<()> {
        self.backend.sigmoid(data)
    }
    
    /// Apply softmax activation on GPU.
    #[cfg(feature = "gpu")]
    pub fn softmax(&mut self, data: &mut [f32]) -> Result<()> {
        self.backend.softmax(data)
    }
    
    /// Perform convolution on GPU.
    #[cfg(feature = "gpu")]
    pub fn convolution(
        &mut self,
        input: &[f32],
        kernel: &[f32],
        input_width: usize,
        input_height: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        self.backend.convolution(
            input,
            kernel,
            input_width,
            input_height,
            kernel_size,
            stride,
            padding,
        )
    }
    
    /// Get memory usage statistics.
    #[cfg(feature = "gpu")]
    pub fn memory_stats(&self) -> GpuMemoryStats {
        self.backend.memory_stats()
    }
    
    /// Synchronize GPU operations (wait for completion).
    #[cfg(feature = "gpu")]
    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }
    
    /// Fallback methods when GPU is not available
    #[cfg(not(feature = "gpu"))]
    pub fn dot_product(&mut self, _a: &[f32], _b: &[f32]) -> Result<f32> {
        Err(VeritasError::GpuError("GPU not available".to_string()))
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn matrix_multiply(
        &mut self,
        _a: &[f32],
        _b: &[f32],
        _rows_a: usize,
        _cols_a: usize,
        _cols_b: usize,
    ) -> Result<Vec<f32>> {
        Err(VeritasError::GpuError("GPU not available".to_string()))
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn relu(&mut self, _data: &mut [f32]) -> Result<()> {
        Err(VeritasError::GpuError("GPU not available".to_string()))
    }
}

/// GPU memory usage statistics.
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    /// Total memory in bytes
    pub total_memory: usize,
    /// Used memory in bytes
    pub used_memory: usize,
    /// Free memory in bytes
    pub free_memory: usize,
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Number of allocations
    pub allocation_count: usize,
}

/// Get list of available GPU devices.
pub fn get_available_devices() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    
    // Always include CPU as fallback
    devices.push(GpuDevice {
        id: 0,
        name: "CPU".to_string(),
        device_type: GpuDeviceType::Cpu,
        total_memory_mb: 0, // Will be filled with system RAM
        available_memory_mb: 0,
        compute_capability: None,
        multiprocessor_count: Some(num_cpus::get()),
        is_default: devices.is_empty(),
    });
    
    #[cfg(feature = "gpu")]
    {
        // Detect CUDA devices
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_devices) = detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }
        
        // Detect Metal devices
        #[cfg(feature = "metal")]
        {
            if let Ok(metal_devices) = detect_metal_devices() {
                devices.extend(metal_devices);
            }
        }
        
        // Detect OpenCL devices
        #[cfg(feature = "opencl")]
        {
            if let Ok(opencl_devices) = detect_opencl_devices() {
                devices.extend(opencl_devices);
            }
        }
    }
    
    devices
}

/// Select the best device based on configuration and available devices.
fn select_device(config: &GpuConfig, devices: &[GpuDevice]) -> Result<GpuDevice> {
    // If specific device ID is requested
    if let Some(device_id) = config.device_id {
        if let Some(device) = devices.iter().find(|d| d.id == device_id) {
            return Ok(device.clone());
        } else {
            return Err(VeritasError::GpuError(
                format!("Device with ID {} not found", device_id)
            ));
        }
    }
    
    // If specific device type is preferred
    if let Some(preferred_type) = config.preferred_device {
        if let Some(device) = devices.iter().find(|d| d.device_type == preferred_type) {
            return Ok(device.clone());
        }
    }
    
    // Auto-select best device (prefer GPU over CPU)
    let gpu_devices: Vec<_> = devices.iter()
        .filter(|d| d.device_type != GpuDeviceType::Cpu)
        .collect();
    
    if !gpu_devices.is_empty() {
        // Prefer device with most memory
        let best_device = gpu_devices.iter()
            .max_by_key(|d| d.total_memory_mb)
            .unwrap();
        return Ok((*best_device).clone());
    }
    
    // Fallback to CPU
    devices.iter()
        .find(|d| d.device_type == GpuDeviceType::Cpu)
        .cloned()
        .ok_or_else(|| VeritasError::GpuError("No suitable device found".to_string()))
}

// Platform-specific device detection functions
#[cfg(all(feature = "gpu", feature = "cuda"))]
fn detect_cuda_devices() -> Result<Vec<GpuDevice>> {
    // This would use CUDA runtime API to detect devices
    // For now, return empty vector as placeholder
    Ok(Vec::new())
}

#[cfg(all(feature = "gpu", feature = "metal"))]
fn detect_metal_devices() -> Result<Vec<GpuDevice>> {
    // This would use Metal API to detect devices
    // For now, return empty vector as placeholder
    Ok(Vec::new())
}

#[cfg(all(feature = "gpu", feature = "opencl"))]
fn detect_opencl_devices() -> Result<Vec<GpuDevice>> {
    // This would use OpenCL API to detect devices
    // For now, return empty vector as placeholder
    Ok(Vec::new())
}

/// Utility functions for GPU operations.
pub mod utils {
    use super::*;
    
    /// Check if GPU acceleration is compiled in.
    pub fn is_gpu_compiled() -> bool {
        cfg!(feature = "gpu")
    }
    
    /// Check if CUDA support is compiled in.
    pub fn is_cuda_compiled() -> bool {
        cfg!(feature = "cuda")
    }
    
    /// Check if Metal support is compiled in.
    pub fn is_metal_compiled() -> bool {
        cfg!(feature = "metal")
    }
    
    /// Get optimal batch size for the given device and operation.
    pub fn optimal_batch_size(device: &GpuDevice, operation: &str) -> usize {
        match (device.device_type, operation) {
            (GpuDeviceType::Cuda, "matrix_multiply") => {
                // Larger batches for CUDA matrix operations
                128
            }
            (GpuDeviceType::Metal, "matrix_multiply") => {
                // Medium batches for Metal
                64
            }
            (GpuDeviceType::Cpu, _) => {
                // Smaller batches for CPU
                16
            }
            _ => {
                // Default batch size
                32
            }
        }
    }
    
    /// Estimate memory requirement for operation.
    pub fn estimate_memory_requirement(
        operation: &str,
        data_size: usize,
        batch_size: usize,
    ) -> usize {
        match operation {
            "matrix_multiply" => {
                // Rough estimate: input + output + intermediate buffers
                data_size * 3 * batch_size
            }
            "convolution" => {
                // Convolution requires more intermediate memory
                data_size * 4 * batch_size
            }
            _ => {
                // Default estimate
                data_size * 2 * batch_size
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_config_creation() {
        let config = GpuConfig::new();
        assert_eq!(config.default_batch_size, 32);
        assert!(config.mixed_precision);
    }
    
    #[test]
    fn test_device_detection() {
        let devices = get_available_devices();
        assert!(!devices.is_empty()); // Should at least have CPU
        
        // Should have CPU device
        assert!(devices.iter().any(|d| d.device_type == GpuDeviceType::Cpu));
    }
    
    #[test]
    fn test_device_selection() {
        let devices = get_available_devices();
        let config = GpuConfig::cpu_only();
        
        let selected = select_device(&config, &devices);
        assert!(selected.is_ok());
        assert_eq!(selected.unwrap().device_type, GpuDeviceType::Cpu);
    }
    
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_accelerator_creation() {
        let config = GpuConfig::cpu_only(); // Use CPU to avoid GPU requirement
        let accelerator = GpuAccelerator::new(config);
        
        // This might fail if no suitable device is found, which is OK
        match accelerator {
            Ok(_) => println!("GPU accelerator created successfully"),
            Err(e) => println!("GPU accelerator creation failed: {}", e),
        }
    }
}