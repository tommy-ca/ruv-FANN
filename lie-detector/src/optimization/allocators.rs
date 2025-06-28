//! Custom memory allocators optimized for veritas-nexus workloads
//!
//! This module provides specialized allocators designed to reduce memory usage
//! and improve allocation patterns for machine learning workloads.

use crate::{Result, VeritasError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use parking_lot::{Mutex, RwLock};
use crossbeam_utils::CachePadded;
use crate::optimization::vectorization_hints::alignment::AlignedVec;

/// Size-class segregated allocator for reducing fragmentation
pub struct SegregatedAllocator {
    /// Size classes: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    size_classes: [SizeClass; 10],
    /// Large allocations > 8192 bytes
    large_allocs: Mutex<HashMap<usize, LargeAllocation>>,
    /// Statistics
    stats: AllocatorStats,
}

/// Single size class in the segregated allocator
struct SizeClass {
    size: usize,
    /// Free list of blocks
    free_list: Mutex<Vec<NonNull<u8>>>,
    /// Slabs of memory for this size class
    slabs: Mutex<Vec<Slab>>,
    /// Number of allocated blocks
    allocated_count: CachePadded<AtomicUsize>,
}

/// Memory slab for a size class
struct Slab {
    ptr: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    bitmap: Vec<u64>, // Allocation bitmap
}

/// Large allocation tracking
struct LargeAllocation {
    layout: Layout,
    size: usize,
}

/// Allocator statistics
#[derive(Default)]
struct AllocatorStats {
    total_allocated: CachePadded<AtomicUsize>,
    total_freed: CachePadded<AtomicUsize>,
    allocation_count: CachePadded<AtomicUsize>,
    free_count: CachePadded<AtomicUsize>,
    fragmentation_bytes: CachePadded<AtomicUsize>,
}

impl SegregatedAllocator {
    /// Create a new segregated allocator
    pub fn new() -> Self {
        const SIZES: [usize; 10] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        
        let size_classes = SIZES.map(|size| SizeClass {
            size,
            free_list: Mutex::new(Vec::with_capacity(1024)),
            slabs: Mutex::new(Vec::new()),
            allocated_count: CachePadded::new(AtomicUsize::new(0)),
        });
        
        Self {
            size_classes,
            large_allocs: Mutex::new(HashMap::new()),
            stats: AllocatorStats::default(),
        }
    }
    
    /// Find the appropriate size class for an allocation
    fn find_size_class(&self, size: usize) -> Option<usize> {
        self.size_classes.iter().position(|sc| size <= sc.size)
    }
    
    /// Allocate from a size class
    fn allocate_from_class(&self, class_idx: usize) -> Result<NonNull<u8>> {
        let size_class = &self.size_classes[class_idx];
        
        // Try to get from free list first
        if let Some(ptr) = size_class.free_list.lock().pop() {
            size_class.allocated_count.fetch_add(1, Ordering::Relaxed);
            self.stats.allocation_count.fetch_add(1, Ordering::Relaxed);
            self.stats.total_allocated.fetch_add(size_class.size, Ordering::Relaxed);
            return Ok(ptr);
        }
        
        // Allocate new slab if needed
        self.allocate_new_slab(class_idx)
    }
    
    /// Allocate a new slab for a size class
    fn allocate_new_slab(&self, class_idx: usize) -> Result<NonNull<u8>> {
        let size_class = &self.size_classes[class_idx];
        let block_size = size_class.size;
        let blocks_per_slab = 64; // Reasonable default
        let slab_size = block_size * blocks_per_slab;
        
        let layout = Layout::from_size_align(slab_size, 64)
            .map_err(|e| VeritasError::MemoryError(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe {
            let ptr = System.alloc(layout);
            NonNull::new(ptr).ok_or_else(|| 
                VeritasError::MemoryError("Slab allocation failed".to_string()))?
        };
        
        // Initialize slab
        let bitmap_size = (blocks_per_slab + 63) / 64;
        let mut slab = Slab {
            ptr,
            layout,
            capacity: blocks_per_slab,
            bitmap: vec![0; bitmap_size],
        };
        
        // Mark first block as allocated
        slab.bitmap[0] |= 1;
        
        // Add remaining blocks to free list
        let mut free_list = size_class.free_list.lock();
        for i in 1..blocks_per_slab {
            let block_ptr = unsafe {
                NonNull::new_unchecked(ptr.as_ptr().add(i * block_size))
            };
            free_list.push(block_ptr);
        }
        
        size_class.slabs.lock().push(slab);
        size_class.allocated_count.fetch_add(1, Ordering::Relaxed);
        self.stats.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.stats.total_allocated.fetch_add(block_size, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    /// Deallocate to a size class
    fn deallocate_to_class(&self, ptr: NonNull<u8>, class_idx: usize) {
        let size_class = &self.size_classes[class_idx];
        
        size_class.free_list.lock().push(ptr);
        size_class.allocated_count.fetch_sub(1, Ordering::Relaxed);
        self.stats.free_count.fetch_add(1, Ordering::Relaxed);
        self.stats.total_freed.fetch_add(size_class.size, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for SegregatedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        
        if let Some(class_idx) = self.find_size_class(size) {
            match self.allocate_from_class(class_idx) {
                Ok(ptr) => ptr.as_ptr(),
                Err(_) => ptr::null_mut(),
            }
        } else {
            // Large allocation
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                self.large_allocs.lock().insert(ptr as usize, LargeAllocation {
                    layout,
                    size,
                });
                self.stats.allocation_count.fetch_add(1, Ordering::Relaxed);
                self.stats.total_allocated.fetch_add(size, Ordering::Relaxed);
            }
            ptr
        }
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        
        if let Some(ptr) = NonNull::new(ptr) {
            if let Some(class_idx) = self.find_size_class(size) {
                self.deallocate_to_class(ptr, class_idx);
            } else {
                // Large allocation
                if self.large_allocs.lock().remove(&(ptr.as_ptr() as usize)).is_some() {
                    System.dealloc(ptr.as_ptr(), layout);
                    self.stats.free_count.fetch_add(1, Ordering::Relaxed);
                    self.stats.total_freed.fetch_add(size, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Arena allocator for temporary allocations with bulk deallocation
pub struct ArenaAllocator {
    chunks: RwLock<Vec<ArenaChunk>>,
    current: Mutex<CurrentArena>,
    chunk_size: usize,
    stats: ArenaStats,
}

struct ArenaChunk {
    data: Vec<u8>,
    used: usize,
}

struct CurrentArena {
    chunk_idx: usize,
    offset: usize,
}

#[derive(Default)]
struct ArenaStats {
    total_allocated: AtomicUsize,
    chunks_allocated: AtomicUsize,
    high_water_mark: AtomicUsize,
}

impl ArenaAllocator {
    /// Create a new arena allocator with specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        let initial_chunk = ArenaChunk {
            data: Vec::with_capacity(chunk_size),
            used: 0,
        };
        
        Self {
            chunks: RwLock::new(vec![initial_chunk]),
            current: Mutex::new(CurrentArena { chunk_idx: 0, offset: 0 }),
            chunk_size,
            stats: ArenaStats::default(),
        }
    }
    
    /// Allocate memory from the arena
    pub fn allocate(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        let mut current = self.current.lock();
        
        // Align offset
        let aligned_offset = (current.offset + align - 1) & !(align - 1);
        let end_offset = aligned_offset + size;
        
        // Check if current chunk has space
        let chunks = self.chunks.read();
        if current.chunk_idx < chunks.len() && end_offset <= self.chunk_size {
            drop(chunks);
            
            // Update offset
            current.offset = end_offset;
            self.stats.total_allocated.fetch_add(size, Ordering::Relaxed);
            
            // Update high water mark
            let mut hwm = self.stats.high_water_mark.load(Ordering::Relaxed);
            while end_offset > hwm {
                match self.stats.high_water_mark.compare_exchange_weak(
                    hwm, end_offset, Ordering::Relaxed, Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(x) => hwm = x,
                }
            }
            
            // Get pointer from current chunk
            let chunks = self.chunks.read();
            let chunk = &chunks[current.chunk_idx];
            let ptr = unsafe {
                NonNull::new_unchecked(chunk.data.as_ptr().add(aligned_offset) as *mut u8)
            };
            
            return Ok(ptr);
        }
        
        drop(chunks);
        
        // Need new chunk
        self.allocate_new_chunk(size, align)
    }
    
    /// Allocate a new chunk
    fn allocate_new_chunk(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        let chunk_size = self.chunk_size.max(size + align);
        let mut new_chunk = ArenaChunk {
            data: Vec::with_capacity(chunk_size),
            used: 0,
        };
        
        // Ensure chunk is properly initialized
        unsafe {
            new_chunk.data.set_len(chunk_size);
        }
        
        let aligned_offset = 0;
        let ptr = unsafe {
            NonNull::new_unchecked(new_chunk.data.as_ptr().add(aligned_offset) as *mut u8)
        };
        
        // Add new chunk
        let mut chunks = self.chunks.write();
        let chunk_idx = chunks.len();
        chunks.push(new_chunk);
        
        // Update current
        let mut current = self.current.lock();
        current.chunk_idx = chunk_idx;
        current.offset = size;
        
        self.stats.chunks_allocated.fetch_add(1, Ordering::Relaxed);
        self.stats.total_allocated.fetch_add(size, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    /// Reset the arena, keeping allocated memory for reuse
    pub fn reset(&self) {
        let mut current = self.current.lock();
        current.chunk_idx = 0;
        current.offset = 0;
        
        // Reset chunk usage
        let mut chunks = self.chunks.write();
        for chunk in chunks.iter_mut() {
            chunk.used = 0;
        }
    }
    
    /// Clear the arena, releasing all memory
    pub fn clear(&self) {
        let mut chunks = self.chunks.write();
        chunks.clear();
        chunks.push(ArenaChunk {
            data: Vec::with_capacity(self.chunk_size),
            used: 0,
        });
        
        let mut current = self.current.lock();
        current.chunk_idx = 0;
        current.offset = 0;
        
        self.stats.total_allocated.store(0, Ordering::Relaxed);
        self.stats.high_water_mark.store(0, Ordering::Relaxed);
    }
    
    /// Get arena statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.stats.total_allocated.load(Ordering::Relaxed),
            self.stats.chunks_allocated.load(Ordering::Relaxed),
            self.stats.high_water_mark.load(Ordering::Relaxed),
        )
    }
}

/// Stack allocator for LIFO allocation patterns
pub struct StackAllocator {
    data: Mutex<Vec<u8>>,
    stack_ptr: AtomicUsize,
    capacity: usize,
    markers: Mutex<Vec<usize>>,
}

impl StackAllocator {
    /// Create a new stack allocator
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Mutex::new(Vec::with_capacity(capacity)),
            stack_ptr: AtomicUsize::new(0),
            capacity,
            markers: Mutex::new(Vec::new()),
        }
    }
    
    /// Allocate from the stack
    pub fn allocate(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        let current_ptr = self.stack_ptr.load(Ordering::Relaxed);
        let aligned_ptr = (current_ptr + align - 1) & !(align - 1);
        let new_ptr = aligned_ptr + size;
        
        if new_ptr > self.capacity {
            return Err(VeritasError::MemoryError("Stack allocator overflow".to_string()));
        }
        
        // Update stack pointer
        match self.stack_ptr.compare_exchange(
            current_ptr, new_ptr, Ordering::SeqCst, Ordering::Relaxed
        ) {
            Ok(_) => {
                let data = self.data.lock();
                let ptr = unsafe {
                    NonNull::new_unchecked(data.as_ptr().add(aligned_ptr) as *mut u8)
                };
                Ok(ptr)
            }
            Err(_) => {
                // Retry on contention
                self.allocate(size, align)
            }
        }
    }
    
    /// Push a marker for bulk deallocation
    pub fn push_marker(&self) {
        let current = self.stack_ptr.load(Ordering::Relaxed);
        self.markers.lock().push(current);
    }
    
    /// Pop to the last marker, freeing all allocations since then
    pub fn pop_to_marker(&self) -> Result<()> {
        let mut markers = self.markers.lock();
        if let Some(marker) = markers.pop() {
            self.stack_ptr.store(marker, Ordering::SeqCst);
            Ok(())
        } else {
            Err(VeritasError::MemoryError("No marker to pop to".to_string()))
        }
    }
    
    /// Reset the stack allocator
    pub fn reset(&self) {
        self.stack_ptr.store(0, Ordering::SeqCst);
        self.markers.lock().clear();
    }
}

/// Pool allocator optimized for fixed-size objects
pub struct FixedPoolAllocator<T> {
    pool: Vec<Mutex<Vec<Box<T>>>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_pool_size: usize,
    stats: PoolStats,
}

#[derive(Default)]
struct PoolStats {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    pool_hits: AtomicUsize,
    pool_misses: AtomicUsize,
}

impl<T: Send + 'static> FixedPoolAllocator<T> {
    /// Create a new fixed pool allocator
    pub fn new<F>(factory: F, max_pool_size: usize, num_shards: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let pool = (0..num_shards)
            .map(|_| Mutex::new(Vec::with_capacity(max_pool_size / num_shards)))
            .collect();
        
        Self {
            pool,
            factory: Box::new(factory),
            max_pool_size,
            stats: PoolStats::default(),
        }
    }
    
    /// Allocate an object from the pool
    pub fn allocate(&self) -> Box<T> {
        let shard_idx = self.get_shard_index();
        
        if let Some(obj) = self.pool[shard_idx].lock().pop() {
            self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
            self.stats.allocations.fetch_add(1, Ordering::Relaxed);
            obj
        } else {
            self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
            self.stats.allocations.fetch_add(1, Ordering::Relaxed);
            Box::new((self.factory)())
        }
    }
    
    /// Return an object to the pool
    pub fn deallocate(&self, obj: Box<T>) {
        let shard_idx = self.get_shard_index();
        let mut shard = self.pool[shard_idx].lock();
        
        if shard.len() < self.max_pool_size / self.pool.len() {
            shard.push(obj);
        }
        
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get shard index based on thread ID
    fn get_shard_index(&self) -> usize {
        let thread_id = std::thread::current().id();
        let hash = unsafe { std::mem::transmute::<_, u64>(thread_id) };
        (hash as usize) % self.pool.len()
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, f64) {
        let allocations = self.stats.allocations.load(Ordering::Relaxed);
        let hits = self.stats.pool_hits.load(Ordering::Relaxed);
        let hit_rate = if allocations > 0 {
            hits as f64 / allocations as f64
        } else {
            0.0
        };
        
        (allocations, self.stats.deallocations.load(Ordering::Relaxed), hit_rate)
    }
}

/// Global allocator manager
pub struct AllocatorManager {
    segregated: SegregatedAllocator,
    arena: RwLock<Option<ArenaAllocator>>,
    stack: RwLock<Option<StackAllocator>>,
}

impl AllocatorManager {
    /// Create a new allocator manager
    pub fn new() -> Self {
        Self {
            segregated: SegregatedAllocator::new(),
            arena: RwLock::new(None),
            stack: RwLock::new(None),
        }
    }
    
    /// Get or create arena allocator
    pub fn arena(&self, chunk_size: usize) -> &ArenaAllocator {
        let mut arena = self.arena.write();
        if arena.is_none() {
            *arena = Some(ArenaAllocator::new(chunk_size));
        }
        
        // Safe because we just ensured it exists
        unsafe { &*(&**arena.as_ref().unwrap() as *const ArenaAllocator) }
    }
    
    /// Get or create stack allocator
    pub fn stack(&self, capacity: usize) -> &StackAllocator {
        let mut stack = self.stack.write();
        if stack.is_none() {
            *stack = Some(StackAllocator::new(capacity));
        }
        
        // Safe because we just ensured it exists
        unsafe { &*(&**stack.as_ref().unwrap() as *const StackAllocator) }
    }
}

// Thread-local allocator caching
thread_local! {
    static TL_ARENA: std::cell::RefCell<Option<ArenaAllocator>> = std::cell::RefCell::new(None);
    static TL_STACK: std::cell::RefCell<Option<StackAllocator>> = std::cell::RefCell::new(None);
}

/// Get thread-local arena allocator
pub fn tl_arena(chunk_size: usize) -> std::cell::Ref<'static, ArenaAllocator> {
    TL_ARENA.with(|arena| {
        if arena.borrow().is_none() {
            *arena.borrow_mut() = Some(ArenaAllocator::new(chunk_size));
        }
        
        unsafe {
            std::mem::transmute(std::cell::Ref::map(arena.borrow(), |a| a.as_ref().unwrap()))
        }
    })
}

/// Get thread-local stack allocator
pub fn tl_stack(capacity: usize) -> std::cell::Ref<'static, StackAllocator> {
    TL_STACK.with(|stack| {
        if stack.borrow().is_none() {
            *stack.borrow_mut() = Some(StackAllocator::new(capacity));
        }
        
        unsafe {
            std::mem::transmute(std::cell::Ref::map(stack.borrow(), |s| s.as_ref().unwrap()))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_segregated_allocator() {
        let alloc = SegregatedAllocator::new();
        
        unsafe {
            // Test small allocation
            let layout = Layout::from_size_align(32, 8).unwrap();
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            
            alloc.dealloc(ptr, layout);
            
            // Test large allocation
            let large_layout = Layout::from_size_align(16384, 8).unwrap();
            let large_ptr = alloc.alloc(large_layout);
            assert!(!large_ptr.is_null());
            
            alloc.dealloc(large_ptr, large_layout);
        }
    }
    
    #[test]
    fn test_arena_allocator() {
        let arena = ArenaAllocator::new(4096);
        
        // Test multiple allocations
        let ptr1 = arena.allocate(100, 8).unwrap();
        let ptr2 = arena.allocate(200, 8).unwrap();
        let ptr3 = arena.allocate(300, 8).unwrap();
        
        assert!(!ptr1.as_ptr().is_null());
        assert!(!ptr2.as_ptr().is_null());
        assert!(!ptr3.as_ptr().is_null());
        
        // Test reset
        arena.reset();
        let (total, chunks, _) = arena.stats();
        assert_eq!(chunks, 1);
    }
    
    #[test]
    fn test_stack_allocator() {
        let stack = StackAllocator::new(1024);
        
        stack.push_marker();
        
        let ptr1 = stack.allocate(100, 8).unwrap();
        let ptr2 = stack.allocate(200, 8).unwrap();
        
        assert!(!ptr1.as_ptr().is_null());
        assert!(!ptr2.as_ptr().is_null());
        
        // Pop to marker should succeed
        stack.pop_to_marker().unwrap();
    }
    
    #[test]
    fn test_fixed_pool_allocator() {
        let pool = FixedPoolAllocator::new(|| vec![0u8; 1024], 10, 4);
        
        let obj1 = pool.allocate();
        let obj2 = pool.allocate();
        
        pool.deallocate(obj1);
        pool.deallocate(obj2);
        
        let (allocs, deallocs, _) = pool.stats();
        assert_eq!(allocs, 2);
        assert_eq!(deallocs, 2);
    }
}