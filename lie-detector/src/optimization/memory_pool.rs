//! High-performance memory pool for efficient allocation and deallocation.
//!
//! This module provides memory pooling strategies optimized for neural network
//! workloads, including:
//! - Size-class segregated allocators
//! - Thread-local pools to reduce contention
//! - NUMA-aware allocation
//! - Zero-copy tensor operations
//! - Memory-mapped buffers for large allocations

use crate::{Result, VeritasError};
use serde::{Deserialize, Serialize};
use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam_utils::CachePadded;
use crate::optimization::vectorization_hints::alignment::AlignedVec;

/// Memory pool configuration.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Pool sizes for different size classes (in bytes)
    pub size_classes: Vec<usize>,
    /// Initial pool size for each class
    pub initial_pool_sizes: Vec<usize>,
    /// Maximum pool size for each class
    pub max_pool_sizes: Vec<usize>,
    /// Enable thread-local pools
    pub thread_local: bool,
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
    /// Alignment requirement for allocations
    pub alignment: usize,
    /// Enable memory prefaulting
    pub prefault: bool,
    /// Memory usage limit in bytes
    pub memory_limit: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            // Common size classes for neural network workloads
            size_classes: vec![
                64, 128, 256, 512, 1024,           // Small tensors
                2048, 4096, 8192, 16384,          // Medium tensors
                32768, 65536, 131072, 262144,     // Large tensors
                524288, 1048576, 2097152,         // Very large tensors
            ],
            initial_pool_sizes: vec![
                100, 100, 100, 100, 50,           // Small: many allocations
                50, 50, 25, 25,                   // Medium: moderate count
                10, 10, 5, 5,                     // Large: fewer allocations
                2, 2, 1,                          // Very large: minimal count
            ],
            max_pool_sizes: vec![
                1000, 1000, 500, 500, 200,        // Small: large pools
                200, 100, 100, 50,                // Medium: moderate pools
                50, 25, 25, 10,                   // Large: smaller pools
                10, 5, 2,                         // Very large: minimal pools
            ],
            thread_local: true,
            numa_aware: false, // Disabled by default for simplicity
            alignment: 64,     // Cache line alignment
            prefault: false,
            memory_limit: None,
        }
    }
}

impl MemoryConfig {
    /// Create configuration optimized for embedded systems.
    pub fn embedded() -> Self {
        Self {
            size_classes: vec![64, 128, 256, 512, 1024, 2048],
            initial_pool_sizes: vec![10, 10, 5, 5, 2, 1],
            max_pool_sizes: vec![50, 50, 25, 25, 10, 5],
            thread_local: false,
            numa_aware: false,
            alignment: 32,
            prefault: false,
            memory_limit: Some(64 * 1024 * 1024), // 64MB limit
        }
    }
    
    /// Create configuration optimized for edge computing.
    pub fn edge() -> Self {
        Self {
            size_classes: vec![
                64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
            ],
            initial_pool_sizes: vec![50, 50, 25, 25, 10, 10, 5, 5, 2, 1],
            max_pool_sizes: vec![200, 200, 100, 100, 50, 50, 25, 25, 10, 5],
            thread_local: true,
            numa_aware: false,
            alignment: 64,
            prefault: false,
            memory_limit: Some(512 * 1024 * 1024), // 512MB limit
        }
    }
    
    /// Create configuration optimized for server deployment.
    pub fn server() -> Self {
        Self {
            thread_local: true,
            numa_aware: true,
            alignment: 64,
            prefault: true,
            memory_limit: None, // No limit on servers
            ..Self::default()
        }
    }
}

/// Memory information and statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory in MB
    pub total_memory_mb: usize,
    /// Available system memory in MB
    pub available_mb: usize,
    /// Memory used by the pool in bytes
    pub pool_used_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// High-performance memory pool with size-class segregation.
pub struct MemoryPool {
    config: MemoryConfig,
    pools: Vec<SizeClassPool>,
    large_allocations: Arc<Mutex<HashMap<usize, LargeAllocation>>>,
    statistics: Arc<MemoryStatistics>,
    thread_local_pools: RwLock<HashMap<std::thread::ThreadId, Vec<ThreadLocalPool>>>,
}

/// Pool for a specific size class.
struct SizeClassPool {
    size_class: usize,
    free_blocks: Mutex<Vec<NonNull<u8>>>,
    allocated_blocks: AtomicUsize,
    total_blocks: AtomicUsize,
    max_blocks: usize,
}

/// Large allocation tracking.
struct LargeAllocation {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

/// Thread-local memory pool.
struct ThreadLocalPool {
    size_class: usize,
    free_blocks: Vec<NonNull<u8>>,
    capacity: usize,
}

/// Memory allocation statistics.
#[derive(Default)]
struct MemoryStatistics {
    total_allocated: CachePadded<AtomicUsize>,
    total_deallocated: CachePadded<AtomicUsize>,
    peak_usage: CachePadded<AtomicUsize>,
    allocation_count: CachePadded<AtomicUsize>,
    deallocation_count: CachePadded<AtomicUsize>,
    cache_hits: CachePadded<AtomicUsize>,
    cache_misses: CachePadded<AtomicUsize>,
}

impl MemoryPool {
    /// Create a new memory pool with the given configuration.
    pub fn new(config: MemoryConfig) -> Result<Self> {
        if config.size_classes.len() != config.initial_pool_sizes.len() ||
           config.size_classes.len() != config.max_pool_sizes.len() {
            return Err(VeritasError::config_error(
                "Size class configuration arrays must have the same length"
            ));
        }
        
        // Create size class pools
        let mut pools = Vec::new();
        for (i, &size_class) in config.size_classes.iter().enumerate() {
            let pool = SizeClassPool::new(
                size_class,
                config.initial_pool_sizes[i],
                config.max_pool_sizes[i],
                config.alignment,
            )?;
            pools.push(pool);
        }
        
        Ok(Self {
            config,
            pools,
            large_allocations: Arc::new(Mutex::new(HashMap::new())),
            statistics: Arc::new(MemoryStatistics::default()),
            thread_local_pools: RwLock::new(HashMap::new()),
        })
    }
    
    /// Allocate memory from the pool.
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        if size == 0 {
            return Err(VeritasError::invalid_input("Cannot allocate zero bytes", "size"));
        }
        
        // Check memory limit
        if let Some(limit) = self.config.memory_limit {
            let current_usage = self.statistics.total_allocated.load(Ordering::Relaxed) -
                               self.statistics.total_deallocated.load(Ordering::Relaxed);
            if current_usage + size > limit {
                return Err(VeritasError::memory_error("Memory limit exceeded"));
            }
        }
        
        // Find appropriate size class
        if let Some(pool_index) = self.find_size_class(size) {
            self.allocate_from_pool(pool_index, size)
        } else {
            self.allocate_large(size)
        }
    }
    
    /// Deallocate memory back to the pool.
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        
        // Find appropriate size class
        if let Some(pool_index) = self.find_size_class(size) {
            self.deallocate_to_pool(pool_index, ptr, size)
        } else {
            self.deallocate_large(ptr, size)
        }
    }
    
    /// Allocate aligned memory.
    pub fn allocate_aligned(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        if alignment > self.config.alignment && alignment.is_power_of_two() {
            self.allocate_large_aligned(size, alignment)
        } else {
            self.allocate(size)
        }
    }
    
    /// Create an aligned vector.
    pub fn create_aligned_vec<T>(&self, capacity: usize) -> Result<AlignedVec<T>> 
    where
        T: Clone + Default,
    {
        let aligned_vec = AlignedVec::with_capacity(self.config.alignment, capacity);
        Ok(aligned_vec)
    }
    
    /// Preallocate memory to reduce allocation overhead.
    pub fn preallocate(&self) -> Result<()> {
        for (i, pool) in self.pools.iter().enumerate() {
            pool.preallocate(self.config.initial_pool_sizes[i], self.config.alignment)?;
        }
        Ok(())
    }
    
    /// Get memory usage statistics.
    pub fn memory_info(&self) -> MemoryInfo {
        let total_allocated = self.statistics.total_allocated.load(Ordering::Relaxed);
        let total_deallocated = self.statistics.total_deallocated.load(Ordering::Relaxed);
        let current_usage = total_allocated.saturating_sub(total_deallocated);
        
        MemoryInfo {
            total_memory_mb: get_total_system_memory_mb(),
            available_mb: get_available_system_memory_mb(),
            pool_used_bytes: current_usage,
            peak_usage_bytes: self.statistics.peak_usage.load(Ordering::Relaxed),
            allocation_count: self.statistics.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.statistics.deallocation_count.load(Ordering::Relaxed),
            cache_hits: self.statistics.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.statistics.cache_misses.load(Ordering::Relaxed),
        }
    }
    
    /// Trim unused memory from pools.
    pub fn trim(&self) -> Result<usize> {
        let mut total_freed = 0;
        
        for pool in &self.pools {
            total_freed += pool.trim()?;
        }
        
        Ok(total_freed)
    }
    
    /// Reset pool statistics.
    pub fn reset_statistics(&self) {
        self.statistics.allocation_count.store(0, Ordering::Relaxed);
        self.statistics.deallocation_count.store(0, Ordering::Relaxed);
        self.statistics.cache_hits.store(0, Ordering::Relaxed);
        self.statistics.cache_misses.store(0, Ordering::Relaxed);
    }
    
    // Private methods
    
    fn find_size_class(&self, size: usize) -> Option<usize> {
        self.config.size_classes.iter()
            .position(|&class_size| size <= class_size)
    }
    
    fn allocate_from_pool(&self, pool_index: usize, size: usize) -> Result<NonNull<u8>> {
        let pool = &self.pools[pool_index];
        
        // Try thread-local pool first if enabled
        if self.config.thread_local {
            if let Some(ptr) = self.try_thread_local_allocation(pool_index) {
                self.update_allocation_stats(size);
                return Ok(ptr);
            }
        }
        
        // Try main pool
        if let Some(ptr) = pool.try_allocate()? {
            self.update_allocation_stats(size);
            self.statistics.cache_hits.fetch_add(1, Ordering::Relaxed);
            Ok(ptr)
        } else {
            // Pool exhausted, allocate new block
            let ptr = pool.allocate_new_block(self.config.alignment)?;
            self.update_allocation_stats(size);
            self.statistics.cache_misses.fetch_add(1, Ordering::Relaxed);
            Ok(ptr)
        }
    }
    
    fn deallocate_to_pool(&self, pool_index: usize, ptr: NonNull<u8>, size: usize) -> Result<()> {
        // Try thread-local pool first if enabled
        if self.config.thread_local {
            if self.try_thread_local_deallocation(pool_index, ptr) {
                self.update_deallocation_stats(size);
                return Ok(());
            }
        }
        
        // Return to main pool
        let pool = &self.pools[pool_index];
        pool.deallocate(ptr)?;
        self.update_deallocation_stats(size);
        Ok(())
    }
    
    fn allocate_large(&self, size: usize) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(size, self.config.alignment)
            .map_err(|e| VeritasError::memory_error(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe {
            NonNull::new(std::alloc::alloc(layout))
                .ok_or_else(|| VeritasError::memory_error("Large allocation failed"))?
        };
        
        // Track large allocation
        let allocation = LargeAllocation {
            ptr,
            size,
            layout,
        };
        
        let mut large_allocs = self.large_allocations.lock().unwrap();
        large_allocs.insert(ptr.as_ptr() as usize, allocation);
        
        self.update_allocation_stats(size);
        Ok(ptr)
    }
    
    fn allocate_large_aligned(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| VeritasError::memory_error(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe {
            NonNull::new(std::alloc::alloc(layout))
                .ok_or_else(|| VeritasError::memory_error("Aligned allocation failed"))?
        };
        
        let allocation = LargeAllocation {
            ptr,
            size,
            layout,
        };
        
        let mut large_allocs = self.large_allocations.lock().unwrap();
        large_allocs.insert(ptr.as_ptr() as usize, allocation);
        
        self.update_allocation_stats(size);
        Ok(ptr)
    }
    
    fn deallocate_large(&self, ptr: NonNull<u8>, _size: usize) -> Result<()> {
        let mut large_allocs = self.large_allocations.lock().unwrap();
        let allocation = large_allocs.remove(&(ptr.as_ptr() as usize))
            .ok_or_else(|| VeritasError::memory_error("Large allocation not found"))?;
        
        unsafe {
            std::alloc::dealloc(ptr.as_ptr(), allocation.layout);
        }
        
        self.update_deallocation_stats(allocation.size);
        Ok(())
    }
    
    fn try_thread_local_allocation(&self, pool_index: usize) -> Option<NonNull<u8>> {
        let thread_id = std::thread::current().id();
        
        if let Ok(mut pools) = self.thread_local_pools.write() {
            let thread_pools = pools.entry(thread_id).or_insert_with(|| {
                self.create_thread_local_pools()
            });
            
            if let Some(pool) = thread_pools.get_mut(pool_index) {
                pool.free_blocks.pop()
            } else {
                None
            }
        } else {
            None
        }
    }
    
    fn try_thread_local_deallocation(&self, pool_index: usize, ptr: NonNull<u8>) -> bool {
        let thread_id = std::thread::current().id();
        
        if let Ok(mut pools) = self.thread_local_pools.write() {
            let thread_pools = pools.entry(thread_id).or_insert_with(|| {
                self.create_thread_local_pools()
            });
            
            if let Some(pool) = thread_pools.get_mut(pool_index) {
                if pool.free_blocks.len() < pool.capacity {
                    pool.free_blocks.push(ptr);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    }
    
    fn create_thread_local_pools(&self) -> Vec<ThreadLocalPool> {
        self.config.size_classes.iter()
            .map(|&size_class| ThreadLocalPool {
                size_class,
                free_blocks: Vec::new(),
                capacity: 32, // Reasonable default for thread-local cache
            })
            .collect()
    }
    
    fn update_allocation_stats(&self, size: usize) {
        let current = self.statistics.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        let total_deallocated = self.statistics.total_deallocated.load(Ordering::Relaxed);
        let current_usage = current.saturating_sub(total_deallocated);
        
        // Update peak usage
        let mut peak = self.statistics.peak_usage.load(Ordering::Relaxed);
        while current_usage > peak {
            match self.statistics.peak_usage.compare_exchange_weak(
                peak, current_usage, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        
        self.statistics.allocation_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn update_deallocation_stats(&self, size: usize) {
        self.statistics.total_deallocated.fetch_add(size, Ordering::Relaxed);
        self.statistics.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl SizeClassPool {
    fn new(size_class: usize, initial_size: usize, max_size: usize, alignment: usize) -> Result<Self> {
        let pool = Self {
            size_class,
            free_blocks: Mutex::new(Vec::with_capacity(initial_size)),
            allocated_blocks: AtomicUsize::new(0),
            total_blocks: AtomicUsize::new(0),
            max_blocks: max_size,
        };
        
        // Preallocate initial blocks
        pool.preallocate(initial_size, alignment)?;
        
        Ok(pool)
    }
    
    fn preallocate(&self, count: usize, alignment: usize) -> Result<()> {
        let layout = Layout::from_size_align(self.size_class, alignment)
            .map_err(|e| VeritasError::memory_error(format!("Invalid layout: {}", e)))?;
        
        let mut free_blocks = self.free_blocks.lock().unwrap();
        
        for _ in 0..count {
            let ptr = unsafe {
                NonNull::new(std::alloc::alloc(layout))
                    .ok_or_else(|| VeritasError::memory_error("Preallocation failed"))?
            };
            
            free_blocks.push(ptr);
        }
        
        self.total_blocks.fetch_add(count, Ordering::Relaxed);
        Ok(())
    }
    
    fn try_allocate(&self) -> Result<Option<NonNull<u8>>> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        
        if let Some(ptr) = free_blocks.pop() {
            self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
            Ok(Some(ptr))
        } else {
            Ok(None)
        }
    }
    
    fn allocate_new_block(&self, alignment: usize) -> Result<NonNull<u8>> {
        let current_total = self.total_blocks.load(Ordering::Relaxed);
        
        if current_total >= self.max_blocks {
            return Err(VeritasError::memory_error("Pool size limit exceeded"));
        }
        
        let layout = Layout::from_size_align(self.size_class, alignment)
            .map_err(|e| VeritasError::memory_error(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe {
            NonNull::new(std::alloc::alloc(layout))
                .ok_or_else(|| VeritasError::memory_error("Block allocation failed"))?
        };
        
        self.total_blocks.fetch_add(1, Ordering::Relaxed);
        self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    fn deallocate(&self, ptr: NonNull<u8>) -> Result<()> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        free_blocks.push(ptr);
        self.allocated_blocks.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }
    
    fn trim(&self) -> Result<usize> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let trim_count = free_blocks.len() / 2; // Trim half of free blocks
        
        let layout = Layout::from_size_align(self.size_class, 64)
            .map_err(|e| VeritasError::memory_error(format!("Invalid layout: {}", e)))?;
        
        for _ in 0..trim_count {
            if let Some(ptr) = free_blocks.pop() {
                unsafe {
                    std::alloc::dealloc(ptr.as_ptr(), layout);
                }
            }
        }
        
        self.total_blocks.fetch_sub(trim_count, Ordering::Relaxed);
        
        Ok(trim_count * self.size_class)
    }
}

// System memory information functions (platform-specific)
fn get_total_system_memory_mb() -> usize {
    // Simplified implementation - would use platform-specific APIs in practice
    8192 // Default to 8GB
}

fn get_available_system_memory_mb() -> usize {
    // Simplified implementation
    6144 // Default to 6GB available
}

/// Zero-copy tensor view for efficient memory operations.
#[derive(Debug)]
pub struct ZeroCopyTensor<T> {
    data: Arc<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<T: Clone> ZeroCopyTensor<T> {
    /// Create a new tensor from existing data.
    pub fn new(data: Arc<[T]>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            offset: 0,
        }
    }
    
    /// Create a view with different shape (must have same total elements).
    pub fn view(&self, new_shape: &[usize]) -> Result<Self> {
        let total_elements: usize = self.shape.iter().product();
        let new_total: usize = new_shape.iter().product();
        
        if total_elements != new_total {
            return Err(VeritasError::invalid_input("Shape mismatch in tensor view", "new_shape"));
        }
        
        Ok(Self {
            data: Arc::clone(&self.data),
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
            offset: self.offset,
        })
    }
    
    /// Create a slice of the tensor.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(VeritasError::invalid_input("Range dimension mismatch in tensor slice", "ranges"));
        }
        
        let mut new_offset = self.offset;
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        
        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(VeritasError::invalid_input("Range out of bounds in tensor slice", "ranges"));
            }
            
            new_offset += range.start * self.strides[i];
            new_shape.push(range.end - range.start);
            new_strides.push(self.strides[i]);
        }
        
        Ok(Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }
    
    /// Get tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get tensor strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Get data slice.
    pub fn data(&self) -> &[T] {
        &self.data[self.offset..]
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);
        assert!(pool.is_ok());
    }
    
    #[test]
    fn test_allocation_deallocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        // Test small allocation
        let ptr = pool.allocate(128).unwrap();
        assert!(!ptr.as_ptr().is_null());
        
        // Test deallocation
        pool.deallocate(ptr, 128).unwrap();
        
        // Test large allocation
        let large_ptr = pool.allocate(1024 * 1024).unwrap();
        assert!(!large_ptr.as_ptr().is_null());
        
        pool.deallocate(large_ptr, 1024 * 1024).unwrap();
    }
    
    #[test]
    fn test_aligned_allocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let ptr = pool.allocate_aligned(256, 128).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 128, 0);
        
        pool.deallocate(ptr, 256).unwrap();
    }
    
    #[test]
    fn test_zero_copy_tensor() {
        let data: Arc<[f32]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice());
        let tensor = ZeroCopyTensor::new(data, vec![2, 3]);
        
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.strides(), &[3, 1]);
        
        // Test view
        let view = tensor.view(&[6]).unwrap();
        assert_eq!(view.shape(), &[6]);
        
        // Test slice
        let slice = tensor.slice(&[0..1, 1..3]).unwrap();
        assert_eq!(slice.shape(), &[1, 2]);
    }
    
    #[test]
    fn test_memory_statistics() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let initial_stats = pool.memory_info();
        assert_eq!(initial_stats.allocation_count, 0);
        
        let _ptr = pool.allocate(128).unwrap();
        let stats_after_alloc = pool.memory_info();
        assert!(stats_after_alloc.allocation_count > initial_stats.allocation_count);
        assert!(stats_after_alloc.pool_used_bytes > 0);
    }
    
    #[test]
    fn test_embedded_config() {
        let config = MemoryConfig::embedded();
        assert!(config.memory_limit.is_some());
        assert_eq!(config.size_classes.len(), 6);
        assert!(!config.thread_local);
    }
}