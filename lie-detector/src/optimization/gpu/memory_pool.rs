//! GPU memory optimization with pinned memory pools
//!
//! This module provides optimized GPU memory management including:
//! - Pinned memory pools for zero-copy transfers
//! - Unified memory management
//! - Asynchronous memory transfers
//! - Multi-stream memory operations

use crate::{Result, VeritasError};
use std::sync::{Arc, Weak};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor, DType};

/// Pinned memory allocation for efficient GPU transfers
pub struct PinnedMemory {
    ptr: *mut u8,
    size: usize,
    device_id: usize,
    is_mapped: bool,
}

// Safety: PinnedMemory manages raw pointers but ensures proper cleanup
unsafe impl Send for PinnedMemory {}
unsafe impl Sync for PinnedMemory {}

impl PinnedMemory {
    /// Allocate pinned memory
    #[cfg(feature = "cuda")]
    pub fn allocate(size: usize, device_id: usize) -> Result<Self> {
        use cuda_sys::cuda;
        
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let result = unsafe {
            cuda::cudaHostAlloc(
                &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
                size,
                cuda::cudaHostAllocPortable | cuda::cudaHostAllocMapped,
            )
        };
        
        if result != cuda::cudaSuccess {
            return Err(VeritasError::GpuError("Failed to allocate pinned memory".to_string()));
        }
        
        Ok(Self {
            ptr,
            size,
            device_id,
            is_mapped: true,
        })
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn allocate(size: usize, device_id: usize) -> Result<Self> {
        // Fallback to regular allocation
        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|e| VeritasError::MemoryError(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(VeritasError::MemoryError("Failed to allocate memory".to_string()));
        }
        
        Ok(Self {
            ptr,
            size,
            device_id,
            is_mapped: false,
        })
    }
    
    /// Get pointer to pinned memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    
    /// Get mutable pointer to pinned memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    
    /// Get size of allocation
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get slice view of memory
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.size)
    }
    
    /// Get mutable slice view of memory
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.size)
    }
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            use cuda_sys::cuda;
            unsafe {
                cuda::cudaFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            if !self.ptr.is_null() {
                let layout = std::alloc::Layout::from_size_align(self.size, 64).unwrap();
                unsafe {
                    std::alloc::dealloc(self.ptr, layout);
                }
            }
        }
    }
}

/// Pinned memory pool for efficient reuse
pub struct PinnedMemoryPool {
    pools: RwLock<HashMap<usize, Vec<Arc<PinnedMemory>>>>, // size -> available blocks
    allocated: Arc<Mutex<HashMap<usize, Weak<PinnedMemory>>>>, // track allocations
    config: PoolConfig,
    stats: Arc<PoolStats>,
}

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum total memory (bytes)
    pub max_memory: usize,
    /// Maximum blocks per size class
    pub max_blocks_per_size: usize,
    /// Size classes to preallocate
    pub size_classes: Vec<usize>,
    /// Device ID for allocations
    pub device_id: usize,
    /// Enable memory prefaulting
    pub prefault: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            max_blocks_per_size: 32,
            size_classes: vec![
                4096,           // 4KB
                16384,          // 16KB
                65536,          // 64KB
                262144,         // 256KB
                1048576,        // 1MB
                4194304,        // 4MB
                16777216,       // 16MB
                67108864,       // 64MB
            ],
            device_id: 0,
            prefault: true,
        }
    }
}

/// Pool statistics
#[derive(Debug, Default)]
struct PoolStats {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    hits: AtomicUsize,
    misses: AtomicUsize,
    current_memory: AtomicU64,
    peak_memory: AtomicU64,
}

impl PinnedMemoryPool {
    /// Create a new pinned memory pool
    pub fn new(config: PoolConfig) -> Self {
        let mut pools = HashMap::new();
        
        // Preallocate some blocks for common sizes
        for &size in &config.size_classes {
            pools.insert(size, Vec::with_capacity(4));
        }
        
        Self {
            pools: RwLock::new(pools),
            allocated: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: Arc::new(PoolStats::default()),
        }
    }
    
    /// Allocate pinned memory from pool
    pub fn allocate(&self, size: usize) -> Result<Arc<PinnedMemory>> {
        // Round up to nearest size class
        let size_class = self.find_size_class(size);
        
        // Try to get from pool
        {
            let mut pools = self.pools.write();
            if let Some(pool) = pools.get_mut(&size_class) {
                if let Some(memory) = pool.pop() {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    
                    // Track allocation
                    self.allocated.lock().insert(
                        Arc::as_ptr(&memory) as usize,
                        Arc::downgrade(&memory)
                    );
                    
                    return Ok(memory);
                }
            }
        }
        
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        
        // Check memory limit
        let current = self.stats.current_memory.load(Ordering::Relaxed) as usize;
        if current + size_class > self.config.max_memory {
            return Err(VeritasError::MemoryError("Pool memory limit exceeded".to_string()));
        }
        
        // Allocate new block
        let memory = Arc::new(PinnedMemory::allocate(size_class, self.config.device_id)?);
        
        // Prefault if configured
        if self.config.prefault {
            unsafe {
                let slice = memory.as_slice();
                // Touch pages to ensure they're mapped
                for chunk in slice.chunks(4096) {
                    std::ptr::read_volatile(&chunk[0]);
                }
            }
        }
        
        // Update stats
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        self.stats.current_memory.fetch_add(size_class as u64, Ordering::Relaxed);
        self.update_peak_memory();
        
        // Track allocation
        self.allocated.lock().insert(
            Arc::as_ptr(&memory) as usize,
            Arc::downgrade(&memory)
        );
        
        Ok(memory)
    }
    
    /// Return memory to pool
    pub fn deallocate(&self, memory: Arc<PinnedMemory>) {
        let size = memory.size();
        let ptr = Arc::as_ptr(&memory) as usize;
        
        // Remove from tracking
        self.allocated.lock().remove(&ptr);
        
        // Only one strong reference left (ours)
        if Arc::strong_count(&memory) == 1 {
            let mut pools = self.pools.write();
            let pool = pools.entry(size).or_insert_with(Vec::new);
            
            // Check pool size limit
            if pool.len() < self.config.max_blocks_per_size {
                pool.push(memory);
                self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
            } else {
                // Let it drop
                self.stats.current_memory.fetch_sub(size as u64, Ordering::Relaxed);
            }
        }
    }
    
    /// Find appropriate size class
    fn find_size_class(&self, size: usize) -> usize {
        *self.config.size_classes.iter()
            .find(|&&s| s >= size)
            .unwrap_or(&size.next_power_of_two())
    }
    
    /// Update peak memory usage
    fn update_peak_memory(&self) {
        let current = self.stats.current_memory.load(Ordering::Relaxed);
        let mut peak = self.stats.peak_memory.load(Ordering::Relaxed);
        
        while current > peak {
            match self.stats.peak_memory.compare_exchange_weak(
                peak, current, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, f64, u64, u64) {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 { hits as f64 / total as f64 } else { 0.0 };
        
        (
            self.stats.allocations.load(Ordering::Relaxed),
            self.stats.deallocations.load(Ordering::Relaxed),
            hit_rate,
            self.stats.current_memory.load(Ordering::Relaxed),
            self.stats.peak_memory.load(Ordering::Relaxed),
        )
    }
    
    /// Clear all pooled memory
    pub fn clear(&self) {
        let mut pools = self.pools.write();
        for (size, pool) in pools.iter_mut() {
            let freed = pool.len() * size;
            pool.clear();
            self.stats.current_memory.fetch_sub(freed as u64, Ordering::Relaxed);
        }
    }
}

/// GPU memory transfer manager for efficient data movement
pub struct TransferManager {
    device_id: usize,
    streams: Vec<StreamHandle>,
    current_stream: AtomicUsize,
    pinned_pool: Arc<PinnedMemoryPool>,
    transfer_queue: Arc<Mutex<TransferQueue>>,
    stats: Arc<TransferStats>,
}

/// CUDA stream handle wrapper
struct StreamHandle {
    #[cfg(feature = "cuda")]
    stream: cuda_sys::cuda::cudaStream_t,
    id: usize,
}

impl StreamHandle {
    #[cfg(feature = "cuda")]
    fn new(id: usize) -> Result<Self> {
        use cuda_sys::cuda;
        
        let mut stream: cuda::cudaStream_t = std::ptr::null_mut();
        let result = unsafe {
            cuda::cudaStreamCreateWithFlags(
                &mut stream,
                cuda::cudaStreamNonBlocking,
            )
        };
        
        if result != cuda::cudaSuccess {
            return Err(VeritasError::GpuError("Failed to create stream".to_string()));
        }
        
        Ok(Self { stream, id })
    }
    
    #[cfg(not(feature = "cuda"))]
    fn new(id: usize) -> Result<Self> {
        Ok(Self { id })
    }
}

#[cfg(feature = "cuda")]
impl Drop for StreamHandle {
    fn drop(&mut self) {
        use cuda_sys::cuda;
        unsafe {
            cuda::cudaStreamDestroy(self.stream);
        }
    }
}

/// Transfer queue for async operations
struct TransferQueue {
    pending: VecDeque<TransferRequest>,
    in_flight: HashMap<usize, TransferRequest>,
}

/// Transfer request
struct TransferRequest {
    id: usize,
    src: TransferSource,
    dst: TransferDestination,
    size: usize,
    stream_id: usize,
    callback: Option<Box<dyn FnOnce(Result<()>) + Send>>,
}

enum TransferSource {
    Host(Vec<u8>),
    HostPinned(Arc<PinnedMemory>),
    Device(*const u8),
}

enum TransferDestination {
    Host(Vec<u8>),
    HostPinned(Arc<PinnedMemory>),
    Device(*mut u8),
}

/// Transfer statistics
#[derive(Debug, Default)]
struct TransferStats {
    transfers_completed: AtomicUsize,
    transfers_failed: AtomicUsize,
    bytes_transferred: AtomicU64,
    total_time_us: AtomicU64,
}

impl TransferManager {
    /// Create a new transfer manager
    pub fn new(device_id: usize, num_streams: usize) -> Result<Self> {
        let mut streams = Vec::with_capacity(num_streams);
        for i in 0..num_streams {
            streams.push(StreamHandle::new(i)?);
        }
        
        let pool_config = PoolConfig {
            device_id,
            ..Default::default()
        };
        
        Ok(Self {
            device_id,
            streams,
            current_stream: AtomicUsize::new(0),
            pinned_pool: Arc::new(PinnedMemoryPool::new(pool_config)),
            transfer_queue: Arc::new(Mutex::new(TransferQueue {
                pending: VecDeque::new(),
                in_flight: HashMap::new(),
            })),
            stats: Arc::new(TransferStats::default()),
        })
    }
    
    /// Async host to device transfer
    pub fn host_to_device_async<F>(
        &self,
        data: Vec<u8>,
        device_ptr: *mut u8,
        callback: F,
    ) -> Result<usize>
    where
        F: FnOnce(Result<()>) + Send + 'static,
    {
        let size = data.len();
        let transfer_id = self.allocate_transfer_id();
        let stream_id = self.next_stream();
        
        // Allocate pinned memory for staging
        let pinned = self.pinned_pool.allocate(size)?;
        
        // Copy to pinned memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                pinned.as_ptr() as *mut u8,
                size
            );
        }
        
        let request = TransferRequest {
            id: transfer_id,
            src: TransferSource::HostPinned(pinned),
            dst: TransferDestination::Device(device_ptr),
            size,
            stream_id,
            callback: Some(Box::new(callback)),
        };
        
        self.enqueue_transfer(request)?;
        Ok(transfer_id)
    }
    
    /// Async device to host transfer
    pub fn device_to_host_async<F>(
        &self,
        device_ptr: *const u8,
        size: usize,
        callback: F,
    ) -> Result<usize>
    where
        F: FnOnce(Result<Vec<u8>>) + Send + 'static,
    {
        let transfer_id = self.allocate_transfer_id();
        let stream_id = self.next_stream();
        
        // Allocate pinned memory for staging
        let pinned = self.pinned_pool.allocate(size)?;
        
        let pinned_clone = pinned.clone();
        let wrapped_callback = move |result: Result<()>| {
            match result {
                Ok(()) => {
                    // Copy from pinned to regular memory
                    let mut data = vec![0u8; size];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            pinned_clone.as_ptr(),
                            data.as_mut_ptr(),
                            size
                        );
                    }
                    callback(Ok(data));
                }
                Err(e) => callback(Err(e)),
            }
        };
        
        let request = TransferRequest {
            id: transfer_id,
            src: TransferSource::Device(device_ptr),
            dst: TransferDestination::HostPinned(pinned),
            size,
            stream_id,
            callback: Some(Box::new(wrapped_callback)),
        };
        
        self.enqueue_transfer(request)?;
        Ok(transfer_id)
    }
    
    /// Enqueue a transfer request
    fn enqueue_transfer(&self, request: TransferRequest) -> Result<()> {
        let mut queue = self.transfer_queue.lock();
        queue.pending.push_back(request);
        
        // Process queue
        self.process_transfers()?;
        
        Ok(())
    }
    
    /// Process pending transfers
    fn process_transfers(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cuda_sys::cuda;
            
            let mut queue = self.transfer_queue.lock();
            
            while let Some(request) = queue.pending.pop_front() {
                let stream = &self.streams[request.stream_id];
                let start_time = std::time::Instant::now();
                
                let result = match (&request.src, &request.dst) {
                    (TransferSource::HostPinned(src), TransferDestination::Device(dst)) => {
                        unsafe {
                            let result = cuda::cudaMemcpyAsync(
                                *dst as *mut std::ffi::c_void,
                                src.as_ptr() as *const std::ffi::c_void,
                                request.size,
                                cuda::cudaMemcpyHostToDevice,
                                stream.stream,
                            );
                            
                            if result == cuda::cudaSuccess {
                                Ok(())
                            } else {
                                Err(VeritasError::GpuError("Transfer failed".to_string()))
                            }
                        }
                    }
                    (TransferSource::Device(src), TransferDestination::HostPinned(dst)) => {
                        unsafe {
                            let result = cuda::cudaMemcpyAsync(
                                dst.as_ptr() as *mut std::ffi::c_void,
                                *src as *const std::ffi::c_void,
                                request.size,
                                cuda::cudaMemcpyDeviceToHost,
                                stream.stream,
                            );
                            
                            if result == cuda::cudaSuccess {
                                Ok(())
                            } else {
                                Err(VeritasError::GpuError("Transfer failed".to_string()))
                            }
                        }
                    }
                    _ => Err(VeritasError::GpuError("Unsupported transfer type".to_string())),
                };
                
                if result.is_ok() {
                    queue.in_flight.insert(request.id, request);
                    
                    // Record stats
                    let elapsed = start_time.elapsed().as_micros() as u64;
                    self.stats.total_time_us.fetch_add(elapsed, Ordering::Relaxed);
                } else if let Some(callback) = request.callback {
                    callback(result);
                    self.stats.transfers_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get next stream in round-robin fashion
    fn next_stream(&self) -> usize {
        let current = self.current_stream.fetch_add(1, Ordering::Relaxed);
        current % self.streams.len()
    }
    
    /// Allocate unique transfer ID
    fn allocate_transfer_id(&self) -> usize {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Wait for all transfers to complete
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cuda_sys::cuda;
            
            for stream in &self.streams {
                let result = unsafe {
                    cuda::cudaStreamSynchronize(stream.stream)
                };
                
                if result != cuda::cudaSuccess {
                    return Err(VeritasError::GpuError("Stream sync failed".to_string()));
                }
            }
            
            // Process completed transfers
            let mut queue = self.transfer_queue.lock();
            let completed: Vec<_> = queue.in_flight.drain().collect();
            
            for (_, request) in completed {
                if let Some(callback) = request.callback {
                    callback(Ok(()));
                }
                
                self.stats.transfers_completed.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_transferred.fetch_add(request.size as u64, Ordering::Relaxed);
            }
        }
        
        Ok(())
    }
    
    /// Get transfer statistics
    pub fn stats(&self) -> (usize, usize, u64, f64) {
        let completed = self.stats.transfers_completed.load(Ordering::Relaxed);
        let failed = self.stats.transfers_failed.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_transferred.load(Ordering::Relaxed);
        let time_us = self.stats.total_time_us.load(Ordering::Relaxed);
        
        let throughput_gb_s = if time_us > 0 {
            (bytes as f64 / 1e9) / (time_us as f64 / 1e6)
        } else {
            0.0
        };
        
        (completed, failed, bytes, throughput_gb_s)
    }
}

/// Unified memory allocator for simplified GPU/CPU access
pub struct UnifiedMemoryAllocator {
    allocations: Arc<Mutex<HashMap<usize, UnifiedAllocation>>>,
    total_allocated: AtomicU64,
    device_id: usize,
}

struct UnifiedAllocation {
    ptr: *mut u8,
    size: usize,
    #[allow(dead_code)]
    prefetch_device: Option<usize>,
}

impl UnifiedMemoryAllocator {
    /// Create a new unified memory allocator
    pub fn new(device_id: usize) -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: AtomicU64::new(0),
            device_id,
        }
    }
    
    /// Allocate unified memory
    #[cfg(feature = "cuda")]
    pub fn allocate(&self, size: usize) -> Result<*mut u8> {
        use cuda_sys::cuda;
        
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let result = unsafe {
            cuda::cudaMallocManaged(
                &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
                size,
                cuda::cudaMemAttachGlobal,
            )
        };
        
        if result != cuda::cudaSuccess {
            return Err(VeritasError::GpuError("Failed to allocate unified memory".to_string()));
        }
        
        let allocation = UnifiedAllocation {
            ptr,
            size,
            prefetch_device: None,
        };
        
        self.allocations.lock().insert(ptr as usize, allocation);
        self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn allocate(&self, size: usize) -> Result<*mut u8> {
        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|e| VeritasError::MemoryError(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(VeritasError::MemoryError("Allocation failed".to_string()));
        }
        
        let allocation = UnifiedAllocation {
            ptr,
            size,
            prefetch_device: None,
        };
        
        self.allocations.lock().insert(ptr as usize, allocation);
        self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    /// Prefetch memory to device
    #[cfg(feature = "cuda")]
    pub fn prefetch_to_device(&self, ptr: *const u8, size: usize) -> Result<()> {
        use cuda_sys::cuda;
        
        let result = unsafe {
            cuda::cudaMemPrefetchAsync(
                ptr as *const std::ffi::c_void,
                size,
                self.device_id as i32,
                std::ptr::null_mut(),
            )
        };
        
        if result != cuda::cudaSuccess {
            return Err(VeritasError::GpuError("Prefetch failed".to_string()));
        }
        
        if let Some(alloc) = self.allocations.lock().get_mut(&(ptr as usize)) {
            alloc.prefetch_device = Some(self.device_id);
        }
        
        Ok(())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn prefetch_to_device(&self, _ptr: *const u8, _size: usize) -> Result<()> {
        Ok(()) // No-op without CUDA
    }
    
    /// Free unified memory
    pub fn free(&self, ptr: *mut u8) -> Result<()> {
        let mut allocations = self.allocations.lock();
        
        if let Some(allocation) = allocations.remove(&(ptr as usize)) {
            #[cfg(feature = "cuda")]
            {
                use cuda_sys::cuda;
                unsafe {
                    cuda::cudaFree(ptr as *mut std::ffi::c_void);
                }
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                let layout = std::alloc::Layout::from_size_align(allocation.size, 64).unwrap();
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
            }
            
            self.total_allocated.fetch_sub(allocation.size as u64, Ordering::Relaxed);
            Ok(())
        } else {
            Err(VeritasError::MemoryError("Unknown allocation".to_string()))
        }
    }
    
    /// Get total allocated memory
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }
}

/// Global pinned memory pool instance
static GLOBAL_PINNED_POOL: once_cell::sync::OnceCell<Arc<PinnedMemoryPool>> = 
    once_cell::sync::OnceCell::new();

/// Initialize global pinned memory pool
pub fn init_global_pinned_pool(config: PoolConfig) -> Result<()> {
    let pool = Arc::new(PinnedMemoryPool::new(config));
    GLOBAL_PINNED_POOL.set(pool)
        .map_err(|_| VeritasError::SystemError("Pool already initialized".to_string()))?;
    Ok(())
}

/// Get global pinned memory pool
pub fn global_pinned_pool() -> Option<&'static Arc<PinnedMemoryPool>> {
    GLOBAL_PINNED_POOL.get()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pinned_memory() {
        let mem = PinnedMemory::allocate(1024, 0).unwrap();
        assert_eq!(mem.size(), 1024);
        assert!(!mem.as_ptr().is_null());
    }
    
    #[test]
    fn test_pinned_memory_pool() {
        let config = PoolConfig {
            size_classes: vec![1024, 2048, 4096],
            ..Default::default()
        };
        
        let pool = PinnedMemoryPool::new(config);
        
        // Allocate memory
        let mem1 = pool.allocate(500).unwrap();
        assert_eq!(mem1.size(), 1024); // Rounded up to size class
        
        let mem2 = pool.allocate(1500).unwrap();
        assert_eq!(mem2.size(), 2048);
        
        // Return to pool
        pool.deallocate(mem1);
        
        // Should get from pool (cache hit)
        let mem3 = pool.allocate(500).unwrap();
        assert_eq!(mem3.size(), 1024);
        
        let (_, _, hit_rate, _, _) = pool.stats();
        assert!(hit_rate > 0.0);
    }
    
    #[test]
    fn test_unified_memory_allocator() {
        let allocator = UnifiedMemoryAllocator::new(0);
        
        let ptr = allocator.allocate(1024).unwrap();
        assert!(!ptr.is_null());
        
        assert_eq!(allocator.total_allocated(), 1024);
        
        allocator.free(ptr).unwrap();
        assert_eq!(allocator.total_allocated(), 0);
    }
}