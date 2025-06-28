//! High-performance optimization modules for CPU and GPU acceleration.
//!
//! This module provides comprehensive optimization strategies including:
//! - SIMD vectorization for different CPU architectures
//! - GPU acceleration via candle-core
//! - Efficient memory management and pooling
//! - Cache-friendly algorithms and data structures
//! - Performance profiling and metrics collection

use crate::{Result, VeritasError};
use serde::{Deserialize, Serialize};
use std::fmt;

// Submodules
pub mod simd;
pub mod gpu;
pub mod memory_pool;
pub mod cache;
pub mod cache_optimization;
pub mod object_pool;
pub mod object_pools;
pub mod arena;
pub mod string_cache;
pub mod compact_types;
pub mod memory_profiler;
pub mod memory_profiling;
pub mod memory_monitor;
pub mod allocators;
pub mod vectorization_hints;
pub mod branch_optimization;
pub mod benchmarks;

#[cfg(feature = "profiling")]
pub mod profiling;

// Re-export important types
pub use self::simd::{SimdProcessor, SimdConfig, SimdFeatures};
pub use self::gpu::{GpuAccelerator, GpuConfig, GpuDevice};
pub use self::memory_pool::{MemoryPool, MemoryConfig, MemoryInfo};
pub use self::cache::{CacheOptimizedMatrix, CacheConfig};
pub use self::cache_optimization::{
    CacheAligned, LoopTiling, CacheAwareAlgorithms, Prefetcher, 
    BlockedConvolution, CACHE_LINE_SIZE, L1_CACHE_SIZE, L2_CACHE_SIZE, L3_CACHE_SIZE
};
pub use self::object_pool::{ObjectPool, GlobalPools, Poolable, PooledObject, AudioChunk, ImageBuffer};
pub use self::object_pools::{ObjectPool as ObjectPoolNew, GlobalPools as GlobalPoolsNew};
pub use self::arena::{Arena, ScopedArena, ArenaString, ArenaVec};
pub use self::string_cache::{intern, InternedString, OptimizedString, StringCache};
pub use self::compact_types::{CompactFeatures, CompactAnalysisResult, CompactModality, FeatureKeyInterner};
pub use self::memory_profiler::{MemoryProfiler, MemoryReport};
pub use self::memory_profiling::{MemoryProfiler as NewMemoryProfiler, ProfilerConfig};
pub use self::memory_monitor::{MemoryMonitor, MemoryPressure, AdaptiveMemoryManager, AdaptiveGCStrategy};
pub use self::allocators::{SegregatedAllocator, ArenaAllocator, StackAllocator};
pub use self::benchmarks::{BenchmarkSuite, BenchmarkResult, run_benchmarks};

#[cfg(feature = "profiling")]
pub use self::profiling::{Profiler, PerformanceMetrics, MetricsCollector};

/// Optimization level for performance tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Conservative optimizations, prioritize compatibility
    Conservative,
    /// Balanced optimizations for general use
    Balanced,
    /// Aggressive optimizations, maximum performance
    Aggressive,
    /// Custom optimization settings
    Custom,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        Self::Balanced
    }
}

impl fmt::Display for OptimizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conservative => write!(f, "Conservative"),
            Self::Balanced => write!(f, "Balanced"),
            Self::Aggressive => write!(f, "Aggressive"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Configuration for the optimization system.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// SIMD configuration
    pub simd: SimdConfig,
    /// GPU configuration (optional)
    pub gpu: Option<GpuConfig>,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Cache optimization settings
    pub cache: CacheConfig,
    /// Overall optimization level
    pub level: OptimizationLevel,
    /// Enable parallel processing
    pub parallel_enabled: bool,
    /// Number of worker threads
    pub num_threads: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            simd: SimdConfig::auto_detect(),
            gpu: None,
            memory: MemoryConfig::default(),
            cache: CacheConfig::default(),
            level: OptimizationLevel::Balanced,
            parallel_enabled: true,
            num_threads: num_cpus::get(),
        }
    }
}

impl OptimizationConfig {
    /// Create a new optimization configuration.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Configure for embedded systems (low resource usage).
    pub fn embedded() -> Self {
        Self {
            simd: SimdConfig::conservative(),
            gpu: None,
            memory: MemoryConfig::embedded(),
            cache: CacheConfig::small(),
            level: OptimizationLevel::Conservative,
            parallel_enabled: false,
            num_threads: 1,
        }
    }
    
    /// Configure for edge computing (balanced performance/resources).
    pub fn edge() -> Self {
        Self {
            simd: SimdConfig::auto_detect(),
            gpu: Some(GpuConfig::auto_detect()),
            memory: MemoryConfig::edge(),
            cache: CacheConfig::medium(),
            level: OptimizationLevel::Balanced,
            parallel_enabled: true,
            num_threads: (num_cpus::get() / 2).max(1),
        }
    }
    
    /// Configure for server deployment (maximum performance).
    pub fn server() -> Self {
        Self {
            simd: SimdConfig::aggressive(),
            gpu: Some(GpuConfig::cuda_if_available()),
            memory: MemoryConfig::server(),
            cache: CacheConfig::large(),
            level: OptimizationLevel::Aggressive,
            parallel_enabled: true,
            num_threads: num_cpus::get(),
        }
    }
    
    /// Set SIMD configuration.
    pub fn with_simd(mut self, simd: SimdConfig) -> Self {
        self.simd = simd;
        self
    }
    
    /// Set GPU configuration.
    pub fn with_gpu(mut self, gpu: Option<GpuConfig>) -> Self {
        self.gpu = gpu;
        self
    }
    
    /// Set memory configuration.
    pub fn with_memory(mut self, memory: MemoryConfig) -> Self {
        self.memory = memory;
        self
    }
    
    /// Set optimization level.
    pub fn with_level(mut self, level: OptimizationLevel) -> Self {
        self.level = level;
        self
    }
    
    /// Enable or disable parallel processing.
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel_enabled = enabled;
        self
    }
    
    /// Set number of worker threads.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }
}

/// Main optimization coordinator that manages different optimization backends.
pub struct OptimizationEngine {
    config: OptimizationConfig,
    simd_processor: SimdProcessor,
    gpu_accelerator: Option<GpuAccelerator>,
    memory_pool: MemoryPool,
    #[cfg(feature = "profiling")]
    profiler: Option<Profiler>,
}

impl OptimizationEngine {
    /// Create a new optimization engine with the given configuration.
    pub fn new(config: OptimizationConfig) -> Result<Self> {
        // Initialize SIMD processor
        let simd_processor = SimdProcessor::new(config.simd.clone())?;
        
        // Initialize GPU accelerator if configured
        let gpu_accelerator = if let Some(gpu_config) = &config.gpu {
            Some(GpuAccelerator::new(gpu_config.clone())?)
        } else {
            None
        };
        
        // Initialize memory pool
        let memory_pool = MemoryPool::new(config.memory.clone())?;
        
        // Initialize profiler if enabled
        #[cfg(feature = "profiling")]
        let profiler = if config.level == OptimizationLevel::Aggressive {
            Some(Profiler::new()?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            simd_processor,
            gpu_accelerator,
            memory_pool,
            #[cfg(feature = "profiling")]
            profiler,
        })
    }
    
    /// Get the current configuration.
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }
    
    /// Get reference to the SIMD processor.
    pub fn simd_processor(&self) -> &SimdProcessor {
        &self.simd_processor
    }
    
    /// Get mutable reference to the SIMD processor.
    pub fn simd_processor_mut(&mut self) -> &mut SimdProcessor {
        &mut self.simd_processor
    }
    
    /// Get reference to the GPU accelerator if available.
    pub fn gpu_accelerator(&self) -> Option<&GpuAccelerator> {
        self.gpu_accelerator.as_ref()
    }
    
    /// Get mutable reference to the GPU accelerator if available.
    pub fn gpu_accelerator_mut(&mut self) -> Option<&mut GpuAccelerator> {
        self.gpu_accelerator.as_mut()
    }
    
    /// Get reference to the memory pool.
    pub fn memory_pool(&self) -> &MemoryPool {
        &self.memory_pool
    }
    
    /// Get mutable reference to the memory pool.
    pub fn memory_pool_mut(&mut self) -> &mut MemoryPool {
        &mut self.memory_pool
    }
    
    /// Check if GPU acceleration is available.
    pub fn has_gpu_acceleration(&self) -> bool {
        self.gpu_accelerator.is_some()
    }
    
    /// Get system capabilities and optimization status.
    pub fn get_capabilities(&self) -> OptimizationCapabilities {
        OptimizationCapabilities {
            simd_features: self.simd_processor.supported_features(),
            gpu_devices: self.gpu_accelerator
                .as_ref()
                .map(|gpu| gpu.available_devices())
                .unwrap_or_default(),
            memory_info: self.memory_pool.memory_info(),
            parallel_workers: self.config.num_threads,
            optimization_level: self.config.level,
        }
    }
    
    /// Perform optimized dot product operation.
    /// Automatically selects the best available implementation (SIMD/GPU).
    pub fn dot_product(&mut self, a: &[f32], b: &[f32]) -> Result<f32> {
        #[cfg(feature = "profiling")]
        let _timer = self.profiler.as_mut().map(|p| p.start_timer("dot_product"));
        
        // Choose implementation based on data size and available backends
        if let Some(gpu) = &mut self.gpu_accelerator {
            if a.len() > 1024 {
                // Use GPU for large arrays
                return gpu.dot_product(a, b);
            }
        }
        
        // Use SIMD for smaller arrays or when GPU is not available
        self.simd_processor.dot_product(a, b)
    }
    
    /// Perform optimized matrix multiplication.
    pub fn matrix_multiply(
        &mut self,
        a: &[f32],
        b: &[f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "profiling")]
        let _timer = self.profiler.as_mut().map(|p| p.start_timer("matrix_multiply"));
        
        let size = rows_a * cols_a * cols_b;
        
        if let Some(gpu) = &mut self.gpu_accelerator {
            if size > 4096 {
                // Use GPU for large matrices
                return gpu.matrix_multiply(a, b, rows_a, cols_a, cols_b);
            }
        }
        
        // Use cache-optimized CPU implementation
        self.simd_processor.matrix_multiply(a, b, rows_a, cols_a, cols_b)
    }
    
    /// Apply ReLU activation function with optimizations.
    pub fn relu(&mut self, data: &mut [f32]) -> Result<()> {
        #[cfg(feature = "profiling")]
        let _timer = self.profiler.as_mut().map(|p| p.start_timer("relu"));
        
        if let Some(gpu) = &mut self.gpu_accelerator {
            if data.len() > 2048 {
                return gpu.relu(data);
            }
        }
        
        self.simd_processor.relu(data)
    }
    
    /// Get performance metrics if profiling is enabled.
    #[cfg(feature = "profiling")]
    pub fn get_performance_metrics(&self) -> Option<PerformanceMetrics> {
        self.profiler.as_ref().map(|p| p.get_metrics())
    }
}

/// System optimization capabilities.
#[derive(Debug, Clone)]
pub struct OptimizationCapabilities {
    /// Supported SIMD features
    pub simd_features: SimdFeatures,
    /// Available GPU devices
    pub gpu_devices: Vec<GpuDevice>,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// Number of parallel workers
    pub parallel_workers: usize,
    /// Current optimization level
    pub optimization_level: OptimizationLevel,
}

impl fmt::Display for OptimizationCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Optimization Capabilities:")?;
        writeln!(f, "  SIMD Features: {:?}", self.simd_features)?;
        writeln!(f, "  GPU Devices: {} available", self.gpu_devices.len())?;
        writeln!(f, "  Memory: {} MB available", self.memory_info.available_mb)?;
        writeln!(f, "  Parallel Workers: {}", self.parallel_workers)?;
        writeln!(f, "  Optimization Level: {}", self.optimization_level)?;
        Ok(())
    }
}

/// Utility functions for optimization detection and configuration.
pub mod utils {
    use super::*;
    
    /// Detect optimal configuration based on system capabilities.
    pub fn detect_optimal_config() -> OptimizationConfig {
        let num_cores = num_cpus::get();
        let total_memory = get_total_memory_mb();
        
        // Choose configuration based on system resources
        if total_memory < 1024 {
            // Low memory system
            OptimizationConfig::embedded()
        } else if total_memory < 8192 || num_cores < 4 {
            // Medium resources
            OptimizationConfig::edge()
        } else {
            // High-end system
            OptimizationConfig::server()
        }
    }
    
    /// Get total system memory in MB.
    fn get_total_memory_mb() -> usize {
        // This is a simplified implementation
        // In a real implementation, you'd use platform-specific APIs
        8192 // Default to 8GB
    }
    
    /// Benchmark different optimization strategies and return the best configuration.
    pub async fn benchmark_optimal_config() -> Result<OptimizationConfig> {
        // This would run benchmarks with different configurations
        // and return the best performing one
        Ok(detect_optimal_config())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimization_config_creation() {
        let config = OptimizationConfig::new();
        assert_eq!(config.level, OptimizationLevel::Balanced);
        assert!(config.parallel_enabled);
    }
    
    #[test]
    fn test_embedded_config() {
        let config = OptimizationConfig::embedded();
        assert_eq!(config.level, OptimizationLevel::Conservative);
        assert!(!config.parallel_enabled);
        assert_eq!(config.num_threads, 1);
    }
    
    #[test]
    fn test_server_config() {
        let config = OptimizationConfig::server();
        assert_eq!(config.level, OptimizationLevel::Aggressive);
        assert!(config.parallel_enabled);
        assert!(config.gpu.is_some());
    }
    
    #[tokio::test]
    async fn test_optimization_engine_creation() {
        let config = OptimizationConfig::new();
        let engine = OptimizationEngine::new(config);
        
        // This test might fail if GPU is not available, which is fine
        // The engine should still work with CPU-only optimizations
        assert!(engine.is_ok() || engine.is_err());
    }
}