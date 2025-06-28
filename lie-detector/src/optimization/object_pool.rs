//! Object pooling for frequently allocated types
//! 
//! This module provides generic object pooling to reduce allocation overhead
//! for frequently created and destroyed objects like feature vectors, 
//! analysis results, and buffer objects.

use crate::{Result, VeritasError};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use std::sync::Arc;
use std::fmt::Debug;
use std::mem::MaybeUninit;

/// Trait for objects that can be pooled
pub trait Poolable: Send + Sync {
    /// Reset the object to a clean state for reuse
    fn reset(&mut self);
    
    /// Validate that the object is in a valid state
    fn is_valid(&self) -> bool {
        true
    }
    
    /// Create a new instance if pool is empty
    fn create_new() -> Self;
}

/// Thread-safe object pool with bounded capacity
pub struct ObjectPool<T: Poolable> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    stats: Arc<Mutex<PoolStats>>,
    capacity: usize,
}

/// Statistics for pool monitoring
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub total_created: usize,
    pub total_reused: usize,
    pub current_size: usize,
    pub peak_size: usize,
    pub failed_returns: usize,
}

impl<T: Poolable> ObjectPool<T> {
    /// Create a new object pool with specified capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Self {
            sender,
            receiver,
            stats: Arc::new(Mutex::new(PoolStats::default())),
            capacity,
        }
    }
    
    /// Get an object from the pool or create a new one
    pub fn get(&self) -> PooledObject<T> {
        let obj = match self.receiver.try_recv() {
            Ok(mut obj) => {
                self.stats.lock().total_reused += 1;
                obj.reset();
                obj
            }
            Err(_) => {
                self.stats.lock().total_created += 1;
                T::create_new()
            }
        };
        
        PooledObject {
            inner: Some(obj),
            pool: self.sender.clone(),
            stats: self.stats.clone(),
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().clone()
    }
    
    /// Clear all objects from the pool
    pub fn clear(&self) {
        while self.receiver.try_recv().is_ok() {}
        let mut stats = self.stats.lock();
        stats.current_size = 0;
    }
    
    /// Pre-populate the pool with objects
    pub fn prefill(&self, count: usize) -> Result<()> {
        for _ in 0..count.min(self.capacity) {
            let obj = T::create_new();
            if self.sender.try_send(obj).is_err() {
                break;
            }
        }
        Ok(())
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<T: Poolable> {
    inner: Option<T>,
    pool: Sender<T>,
    stats: Arc<Mutex<PoolStats>>,
}

impl<T: Poolable> std::ops::Deref for PooledObject<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl<T: Poolable> std::ops::DerefMut for PooledObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}

impl<T: Poolable> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(mut obj) = self.inner.take() {
            if obj.is_valid() {
                obj.reset();
                if let Ok(()) = self.pool.try_send(obj) {
                    let mut stats = self.stats.lock();
                    stats.current_size += 1;
                    stats.peak_size = stats.peak_size.max(stats.current_size);
                } else {
                    self.stats.lock().failed_returns += 1;
                }
            }
        }
    }
}

// Specialized pools for common types

/// Poolable vector implementation
#[derive(Clone)]
pub struct PooledVec<T> {
    pub data: Vec<T>,
    capacity: usize,
}

impl<T: Clone + Default + Send + Sync> PooledVec<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }
}

impl<T: Clone + Default + Send + Sync + 'static> Poolable for PooledVec<T> {
    fn reset(&mut self) {
        self.data.clear();
        // Shrink if significantly over capacity
        if self.data.capacity() > self.capacity * 2 {
            self.data.shrink_to(self.capacity);
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(1024) // Default capacity
    }
}

/// Poolable buffer for audio/image data
pub struct PooledBuffer {
    pub data: Vec<u8>,
    capacity: usize,
}

impl PooledBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Resize buffer without reallocation if possible
    pub fn resize_no_alloc(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.data.capacity() {
            return Err(VeritasError::memory_error(
                "Buffer resize would require reallocation"
            ));
        }
        self.data.resize(new_len, 0);
        Ok(())
    }
}

impl Poolable for PooledBuffer {
    fn reset(&mut self) {
        self.data.clear();
        // Shrink if significantly over capacity
        if self.data.capacity() > self.capacity * 2 {
            self.data.shrink_to(self.capacity);
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(1024 * 1024) // 1MB default
    }
}

/// Feature vector pool for analysis results
pub struct PooledFeatures {
    pub features: hashbrown::HashMap<String, f64>,
    pub metadata: hashbrown::HashMap<String, String>,
    capacity: usize,
}

impl PooledFeatures {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            features: hashbrown::HashMap::with_capacity(capacity),
            metadata: hashbrown::HashMap::with_capacity(capacity / 4),
            capacity,
        }
    }
}

impl Poolable for PooledFeatures {
    fn reset(&mut self) {
        self.features.clear();
        self.metadata.clear();
        
        // Shrink if significantly over capacity
        if self.features.capacity() > self.capacity * 2 {
            self.features.shrink_to(self.capacity);
            self.metadata.shrink_to(self.capacity / 4);
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(64) // Typical feature count
    }
}

/// Global object pools for the application
pub struct GlobalPools {
    pub vec_f32_pool: ObjectPool<PooledVec<f32>>,
    pub vec_f64_pool: ObjectPool<PooledVec<f64>>,
    pub buffer_pool: ObjectPool<PooledBuffer>,
    pub features_pool: ObjectPool<PooledFeatures>,
    pub audio_chunk_pool: ObjectPool<AudioChunk>,
    pub image_buffer_pool: ObjectPool<ImageBuffer>,
}

impl GlobalPools {
    pub fn new() -> Self {
        Self {
            vec_f32_pool: ObjectPool::new(256),
            vec_f64_pool: ObjectPool::new(256),
            buffer_pool: ObjectPool::new(64),
            features_pool: ObjectPool::new(128),
            audio_chunk_pool: ObjectPool::new(32),
            image_buffer_pool: ObjectPool::new(16),
        }
    }
    
    /// Prefill all pools for better startup performance
    pub fn prefill_all(&self) -> Result<()> {
        self.vec_f32_pool.prefill(64)?;
        self.vec_f64_pool.prefill(64)?;
        self.buffer_pool.prefill(16)?;
        self.features_pool.prefill(32)?;
        self.audio_chunk_pool.prefill(8)?;
        self.image_buffer_pool.prefill(4)?;
        Ok(())
    }
    
    /// Get combined statistics for all pools
    pub fn all_stats(&self) -> hashbrown::HashMap<&'static str, PoolStats> {
        let mut stats = hashbrown::HashMap::new();
        stats.insert("vec_f32", self.vec_f32_pool.stats());
        stats.insert("vec_f64", self.vec_f64_pool.stats());
        stats.insert("buffer", self.buffer_pool.stats());
        stats.insert("features", self.features_pool.stats());
        stats.insert("audio_chunk", self.audio_chunk_pool.stats());
        stats.insert("image_buffer", self.image_buffer_pool.stats());
        stats
    }
}

/// Poolable audio chunk for streaming audio processing
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub timestamp: std::time::Duration,
    capacity: usize,
}

impl AudioChunk {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            sample_rate: 16000, // Default
            timestamp: std::time::Duration::ZERO,
            capacity,
        }
    }
}

impl Poolable for AudioChunk {
    fn reset(&mut self) {
        self.samples.clear();
        self.timestamp = std::time::Duration::ZERO;
        
        if self.samples.capacity() > self.capacity * 2 {
            self.samples.shrink_to(self.capacity);
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(16384) // ~1 second at 16kHz
    }
}

/// Poolable image buffer for vision processing
pub struct ImageBuffer {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    capacity: usize,
}

impl ImageBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            width: 0,
            height: 0,
            channels: 0,
            capacity,
        }
    }
    
    /// Resize for new image dimensions
    pub fn resize_for_image(&mut self, width: u32, height: u32, channels: u32) -> Result<()> {
        let required_size = (width * height * channels) as usize;
        if required_size > self.capacity * 2 {
            return Err(VeritasError::memory_error(
                format!("Image size {} exceeds buffer capacity", required_size)
            ));
        }
        
        self.width = width;
        self.height = height;
        self.channels = channels;
        self.data.resize(required_size, 0);
        Ok(())
    }
}

impl Poolable for ImageBuffer {
    fn reset(&mut self) {
        self.data.clear();
        self.width = 0;
        self.height = 0;
        self.channels = 0;
        
        if self.data.capacity() > self.capacity * 2 {
            self.data.shrink_to(self.capacity);
        }
    }
    
    fn create_new() -> Self {
        // Default capacity for 640x480 RGB image
        Self::with_capacity(640 * 480 * 3)
    }
}

// Thread-local object pools for reduced contention
thread_local! {
    static TL_VEC_F32_POOL: ObjectPool<PooledVec<f32>> = ObjectPool::new(32);
    static TL_VEC_F64_POOL: ObjectPool<PooledVec<f64>> = ObjectPool::new(32);
    static TL_FEATURES_POOL: ObjectPool<PooledFeatures> = ObjectPool::new(16);
}

/// Get a thread-local f32 vector
pub fn get_tl_vec_f32() -> PooledObject<PooledVec<f32>> {
    TL_VEC_F32_POOL.with(|pool| pool.get())
}

/// Get a thread-local f64 vector
pub fn get_tl_vec_f64() -> PooledObject<PooledVec<f64>> {
    TL_VEC_F64_POOL.with(|pool| pool.get())
}

/// Get a thread-local features map
pub fn get_tl_features() -> PooledObject<PooledFeatures> {
    TL_FEATURES_POOL.with(|pool| pool.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_object_pool_basic() {
        let pool: ObjectPool<PooledVec<f32>> = ObjectPool::new(10);
        
        // Get object from pool
        let mut obj1 = pool.get();
        obj1.data.push(1.0);
        obj1.data.push(2.0);
        
        // Return to pool by dropping
        drop(obj1);
        
        // Get again - should be reset
        let obj2 = pool.get();
        assert!(obj2.data.is_empty());
        
        let stats = pool.stats();
        assert_eq!(stats.total_created, 1);
        assert_eq!(stats.total_reused, 1);
    }
    
    #[test]
    fn test_pool_capacity() {
        let pool: ObjectPool<PooledBuffer> = ObjectPool::new(2);
        
        let obj1 = pool.get();
        let obj2 = pool.get();
        let obj3 = pool.get();
        
        drop(obj1);
        drop(obj2);
        drop(obj3);
        
        let stats = pool.stats();
        assert_eq!(stats.current_size, 2); // Pool capacity is 2
        assert_eq!(stats.failed_returns, 1); // One object couldn't be returned
    }
    
    #[test]
    fn test_thread_local_pools() {
        let mut vec1 = get_tl_vec_f32();
        vec1.data.extend(&[1.0, 2.0, 3.0]);
        
        drop(vec1);
        
        let vec2 = get_tl_vec_f32();
        assert!(vec2.data.is_empty()); // Should be reset
    }
}