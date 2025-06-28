//! Arena allocator for temporary allocations
//! 
//! This module provides arena-based allocation for temporary objects
//! created during analysis and fusion operations. Objects are allocated
//! from a contiguous memory region and freed all at once.

use crate::{Result, VeritasError};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, MaybeUninit};
use std::ptr::NonNull;
use std::sync::Arc;
use parking_lot::Mutex;

/// Arena allocator for fast temporary allocations
pub struct Arena {
    chunks: Mutex<Vec<ArenaChunk>>,
    current_chunk: Mutex<usize>,
    chunk_size: usize,
    total_allocated: Arc<Mutex<usize>>,
    stats: Arc<Mutex<ArenaStats>>,
}

/// A single chunk of memory in the arena
struct ArenaChunk {
    data: NonNull<u8>,
    layout: Layout,
    position: Cell<usize>,
    capacity: usize,
}

/// Arena allocation statistics
#[derive(Debug, Default, Clone)]
pub struct ArenaStats {
    pub total_allocations: usize,
    pub total_bytes_allocated: usize,
    pub current_chunks: usize,
    pub peak_chunks: usize,
    pub wasted_bytes: usize,
}

impl Arena {
    /// Create a new arena with specified chunk size
    pub fn new(chunk_size: usize) -> Result<Self> {
        if chunk_size < 1024 {
            return Err(VeritasError::MemoryError(
                "Arena chunk size must be at least 1024 bytes".to_string()
            ));
        }
        
        let arena = Self {
            chunks: Mutex::new(Vec::new()),
            current_chunk: Mutex::new(0),
            chunk_size,
            total_allocated: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(ArenaStats::default())),
        };
        
        // Allocate first chunk
        arena.allocate_chunk()?;
        
        Ok(arena)
    }
    
    /// Allocate memory for type T
    pub fn alloc<T>(&self, value: T) -> Result<&mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout)?;
        
        unsafe {
            let typed_ptr = ptr.cast::<T>().as_ptr();
            typed_ptr.write(value);
            Ok(&mut *typed_ptr)
        }
    }
    
    /// Allocate uninitialized memory for type T
    pub fn alloc_uninit<T>(&self) -> Result<&mut MaybeUninit<T>> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout)?;
        
        unsafe {
            let typed_ptr = ptr.cast::<MaybeUninit<T>>().as_ptr();
            Ok(&mut *typed_ptr)
        }
    }
    
    /// Allocate a slice
    pub fn alloc_slice<T: Clone>(&self, data: &[T]) -> Result<&mut [T]> {
        if data.is_empty() {
            return Ok(&mut []);
        }
        
        let layout = Layout::array::<T>(data.len())
            .map_err(|_| VeritasError::MemoryError("Invalid slice layout".to_string()))?;
        let ptr = self.alloc_raw(layout)?;
        
        unsafe {
            let typed_ptr = ptr.cast::<T>().as_ptr();
            for (i, item) in data.iter().enumerate() {
                typed_ptr.add(i).write(item.clone());
            }
            Ok(std::slice::from_raw_parts_mut(typed_ptr, data.len()))
        }
    }
    
    /// Allocate raw bytes with alignment
    pub fn alloc_raw(&self, layout: Layout) -> Result<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align();
        
        // Update statistics
        {
            let mut stats = self.stats.lock();
            stats.total_allocations += 1;
            stats.total_bytes_allocated += size;
        }
        
        // Try to allocate from current chunk
        let chunks = self.chunks.lock();
        let current_idx = *self.current_chunk.lock();
        
        if current_idx < chunks.len() {
            if let Some(ptr) = chunks[current_idx].try_alloc(size, align) {
                return Ok(ptr);
            }
        }
        drop(chunks);
        
        // Need new chunk
        self.allocate_chunk()?;
        
        // Try again with new chunk
        let chunks = self.chunks.lock();
        let current_idx = *self.current_chunk.lock();
        chunks[current_idx].try_alloc(size, align)
            .ok_or_else(|| VeritasError::MemoryError(
                format!("Allocation of {} bytes exceeds chunk size", size)
            ))
    }
    
    /// Reset the arena, freeing all allocations
    pub fn reset(&self) {
        let mut chunks = self.chunks.lock();
        
        // Reset all chunks
        for chunk in chunks.iter() {
            chunk.position.set(0);
        }
        
        // Keep only the first chunk
        if chunks.len() > 1 {
            let mut stats = self.stats.lock();
            stats.wasted_bytes += chunks[1..].iter()
                .map(|c| c.capacity - c.position.get())
                .sum::<usize>();
            
            // Deallocate extra chunks
            for chunk in chunks.drain(1..) {
                unsafe {
                    dealloc(chunk.data.as_ptr(), chunk.layout);
                }
            }
        }
        
        *self.current_chunk.lock() = 0;
        *self.total_allocated.lock() = 0;
    }
    
    /// Get arena statistics
    pub fn stats(&self) -> ArenaStats {
        self.stats.lock().clone()
    }
    
    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        *self.total_allocated.lock()
    }
    
    /// Allocate a new chunk
    fn allocate_chunk(&self) -> Result<()> {
        let layout = Layout::from_size_align(self.chunk_size, 64)
            .map_err(|_| VeritasError::MemoryError("Invalid chunk layout".to_string()))?;
        
        let data = unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(VeritasError::MemoryError(
                    format!("Failed to allocate {} byte chunk", self.chunk_size)
                ));
            }
            NonNull::new_unchecked(ptr)
        };
        
        let chunk = ArenaChunk {
            data,
            layout,
            position: Cell::new(0),
            capacity: self.chunk_size,
        };
        
        let mut chunks = self.chunks.lock();
        chunks.push(chunk);
        *self.current_chunk.lock() = chunks.len() - 1;
        
        // Update stats
        let mut stats = self.stats.lock();
        stats.current_chunks = chunks.len();
        stats.peak_chunks = stats.peak_chunks.max(chunks.len());
        
        Ok(())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let chunks = self.chunks.lock();
        for chunk in chunks.iter() {
            unsafe {
                dealloc(chunk.data.as_ptr(), chunk.layout);
            }
        }
    }
}

impl ArenaChunk {
    /// Try to allocate from this chunk
    fn try_alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let pos = self.position.get();
        
        // Align the position
        let aligned_pos = (pos + align - 1) & !(align - 1);
        let end_pos = aligned_pos + size;
        
        if end_pos > self.capacity {
            return None;
        }
        
        self.position.set(end_pos);
        
        unsafe {
            Some(NonNull::new_unchecked(
                self.data.as_ptr().add(aligned_pos)
            ))
        }
    }
}

/// Scoped arena that automatically resets when dropped
pub struct ScopedArena<'a> {
    arena: &'a Arena,
    initial_allocated: usize,
}

impl<'a> ScopedArena<'a> {
    /// Create a new scoped arena
    pub fn new(arena: &'a Arena) -> Self {
        let initial_allocated = arena.allocated_bytes();
        Self {
            arena,
            initial_allocated,
        }
    }
    
    /// Allocate in the scoped arena
    pub fn alloc<T>(&self, value: T) -> Result<&mut T> {
        self.arena.alloc(value)
    }
    
    /// Allocate slice in the scoped arena
    pub fn alloc_slice<T: Clone>(&self, data: &[T]) -> Result<&mut [T]> {
        self.arena.alloc_slice(data)
    }
}

impl<'a> Drop for ScopedArena<'a> {
    fn drop(&mut self) {
        // Could implement partial reset here if needed
        // For now, we rely on the main arena's reset
    }
}

/// Arena-allocated string
pub struct ArenaString<'a> {
    data: &'a str,
}

impl<'a> ArenaString<'a> {
    /// Create a new arena string
    pub fn new(arena: &'a Arena, s: &str) -> Result<Self> {
        let bytes = arena.alloc_slice(s.as_bytes())?;
        let data = unsafe {
            std::str::from_utf8_unchecked(bytes)
        };
        Ok(Self { data })
    }
    
    /// Get the string slice
    pub fn as_str(&self) -> &str {
        self.data
    }
}

impl<'a> std::ops::Deref for ArenaString<'a> {
    type Target = str;
    
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

/// Arena-allocated vector
pub struct ArenaVec<'a, T> {
    data: &'a mut [MaybeUninit<T>],
    len: usize,
    arena: &'a Arena,
}

impl<'a, T> ArenaVec<'a, T> {
    /// Create a new arena vector with capacity
    pub fn with_capacity(arena: &'a Arena, capacity: usize) -> Result<Self> {
        let layout = Layout::array::<MaybeUninit<T>>(capacity)
            .map_err(|_| VeritasError::MemoryError("Invalid vec layout".to_string()))?;
        let ptr = arena.alloc_raw(layout)?;
        
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                ptr.cast::<MaybeUninit<T>>().as_ptr(),
                capacity
            )
        };
        
        Ok(Self {
            data,
            len: 0,
            arena,
        })
    }
    
    /// Push an element
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len >= self.data.len() {
            return Err(VeritasError::MemoryError(
                "ArenaVec capacity exceeded".to_string()
            ));
        }
        
        self.data[self.len].write(value);
        self.len += 1;
        Ok(())
    }
    
    /// Get the slice
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const T,
                self.len
            )
        }
    }
    
    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.len
            )
        }
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Thread-local arena for reduced contention
thread_local! {
    static TL_ARENA: Arena = Arena::new(64 * 1024).expect("Failed to create thread-local arena");
}

/// Get the thread-local arena
pub fn with_tl_arena<F, R>(f: F) -> R
where
    F: FnOnce(&Arena) -> R,
{
    TL_ARENA.with(f)
}

/// Run a function with a scoped arena that resets after
pub fn with_scoped_arena<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&Arena) -> Result<R>,
{
    let arena = Arena::new(16 * 1024)?;
    let result = f(&arena)?;
    // Arena is automatically deallocated when dropped
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena_basic() {
        let arena = Arena::new(1024).unwrap();
        
        let x = arena.alloc(42i32).unwrap();
        assert_eq!(*x, 42);
        
        let arr = arena.alloc_slice(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(arr, &[1, 2, 3, 4, 5]);
        
        arena.reset();
        
        let stats = arena.stats();
        assert_eq!(stats.current_chunks, 1);
    }
    
    #[test]
    fn test_arena_string() {
        let arena = Arena::new(1024).unwrap();
        
        let s1 = ArenaString::new(&arena, "Hello").unwrap();
        let s2 = ArenaString::new(&arena, "World").unwrap();
        
        assert_eq!(s1.as_str(), "Hello");
        assert_eq!(s2.as_str(), "World");
    }
    
    #[test]
    fn test_arena_vec() {
        let arena = Arena::new(1024).unwrap();
        
        let mut vec = ArenaVec::with_capacity(&arena, 10).unwrap();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();
        
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
        assert_eq!(vec.len(), 3);
    }
    
    #[test]
    fn test_chunk_allocation() {
        let arena = Arena::new(100).unwrap();
        
        // Allocate more than one chunk worth
        for i in 0..20 {
            let _ = arena.alloc(i as i32).unwrap();
        }
        
        let stats = arena.stats();
        assert!(stats.current_chunks > 1);
    }
}