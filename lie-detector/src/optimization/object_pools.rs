//! Comprehensive object pooling for all modalities and data structures
//! 
//! This module provides optimized object pooling to reduce allocation overhead
//! for frequently created and destroyed objects across all analysis modalities.

use crate::{Result, VeritasError};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, Weak};
use std::fmt::Debug;
use std::mem::MaybeUninit;
use hashbrown::HashMap;
use ndarray::{Array1, Array2, ArrayD};
use std::borrow::Cow;

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

/// Thread-safe object pool with bounded capacity and advanced features
pub struct ObjectPool<T: Poolable> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    stats: Arc<Mutex<PoolStats>>,
    capacity: usize,
    config: PoolConfig,
}

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum pool capacity
    pub capacity: usize,
    /// Enable preallocation
    pub preallocate: bool,
    /// Preallocation size
    pub prealloc_size: usize,
    /// Enable adaptive sizing
    pub adaptive_sizing: bool,
    /// Enable metrics collection
    pub collect_metrics: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            capacity: 1024,
            preallocate: true,
            prealloc_size: 64,
            adaptive_sizing: true,
            collect_metrics: true,
        }
    }
}

/// Statistics for pool monitoring
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub total_created: usize,
    pub total_reused: usize,
    pub current_size: usize,
    pub peak_size: usize,
    pub failed_returns: usize,
    pub hit_rate: f64,
    pub avg_object_lifetime: std::time::Duration,
}

impl<T: Poolable> ObjectPool<T> {
    /// Create a new object pool with specified configuration
    pub fn new(config: PoolConfig) -> Self {
        let (sender, receiver) = bounded(config.capacity);
        let pool = Self {
            sender,
            receiver,
            stats: Arc::new(Mutex::new(PoolStats::default())),
            capacity: config.capacity,
            config: config.clone(),
        };
        
        // Preallocate if configured
        if config.preallocate {
            let _ = pool.prefill(config.prealloc_size);
        }
        
        pool
    }
    
    /// Get an object from the pool or create a new one
    pub fn get(&self) -> PooledObject<T> {
        let start_time = std::time::Instant::now();
        
        let obj = match self.receiver.try_recv() {
            Ok(mut obj) => {
                if self.config.collect_metrics {
                    let mut stats = self.stats.lock();
                    stats.total_reused += 1;
                    stats.current_size = stats.current_size.saturating_sub(1);
                    self.update_hit_rate(&mut stats);
                }
                obj.reset();
                obj
            }
            Err(_) => {
                if self.config.collect_metrics {
                    let mut stats = self.stats.lock();
                    stats.total_created += 1;
                    self.update_hit_rate(&mut stats);
                }
                T::create_new()
            }
        };
        
        PooledObject {
            inner: Some(obj),
            pool: self.sender.clone(),
            stats: self.stats.clone(),
            created_at: start_time,
            config: self.config.clone(),
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
            if self.config.collect_metrics {
                let mut stats = self.stats.lock();
                stats.current_size += 1;
                stats.peak_size = stats.peak_size.max(stats.current_size);
            }
        }
        Ok(())
    }
    
    /// Resize pool capacity dynamically
    pub fn resize(&mut self, new_capacity: usize) -> Result<()> {
        if new_capacity == self.capacity {
            return Ok(());
        }
        
        // Create new channel with new capacity
        let (new_sender, new_receiver) = bounded(new_capacity);
        
        // Transfer existing objects
        while let Ok(obj) = self.receiver.try_recv() {
            if new_sender.try_send(obj).is_err() {
                break;
            }
        }
        
        self.sender = new_sender;
        self.receiver = new_receiver;
        self.capacity = new_capacity;
        self.config.capacity = new_capacity;
        
        Ok(())
    }
    
    /// Update hit rate statistics
    fn update_hit_rate(&self, stats: &mut PoolStats) {
        let total = stats.total_created + stats.total_reused;
        if total > 0 {
            stats.hit_rate = stats.total_reused as f64 / total as f64;
        }
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<T: Poolable> {
    inner: Option<T>,
    pool: Sender<T>,
    stats: Arc<Mutex<PoolStats>>,
    created_at: std::time::Instant,
    config: PoolConfig,
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
                    if self.config.collect_metrics {
                        let mut stats = self.stats.lock();
                        stats.current_size += 1;
                        stats.peak_size = stats.peak_size.max(stats.current_size);
                        
                        // Update average lifetime
                        let lifetime = self.created_at.elapsed();
                        let total_objects = stats.total_created + stats.total_reused;
                        if total_objects > 0 {
                            let current_avg = stats.avg_object_lifetime.as_secs_f64();
                            let new_avg = (current_avg * (total_objects - 1) as f64 + lifetime.as_secs_f64()) 
                                / total_objects as f64;
                            stats.avg_object_lifetime = std::time::Duration::from_secs_f64(new_avg);
                        }
                    }
                } else if self.config.collect_metrics {
                    self.stats.lock().failed_returns += 1;
                }
            }
        }
    }
}

// Vision Modality Pools

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
            return Err(VeritasError::MemoryError(
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

/// Poolable face detection result
pub struct FaceDetectionResult {
    pub faces: Vec<FaceData>,
    pub landmarks: Vec<LandmarkData>,
    pub timestamp: std::time::Duration,
}

#[derive(Clone)]
pub struct FaceData {
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub face_id: Option<u32>,
}

#[derive(Clone)]
pub struct LandmarkData {
    pub points: Vec<[f32; 2]>,
    pub visibility: Vec<f32>,
}

impl Poolable for FaceDetectionResult {
    fn reset(&mut self) {
        self.faces.clear();
        self.landmarks.clear();
        self.timestamp = std::time::Duration::ZERO;
    }
    
    fn create_new() -> Self {
        Self {
            faces: Vec::with_capacity(10),
            landmarks: Vec::with_capacity(10),
            timestamp: std::time::Duration::ZERO,
        }
    }
}

/// Poolable micro-expression analysis result
pub struct MicroExpressionResult {
    pub expressions: HashMap<String, f32>,
    pub action_units: Vec<(u8, f32)>,
    pub temporal_features: Vec<f32>,
}

impl Poolable for MicroExpressionResult {
    fn reset(&mut self) {
        self.expressions.clear();
        self.action_units.clear();
        self.temporal_features.clear();
    }
    
    fn create_new() -> Self {
        Self {
            expressions: HashMap::with_capacity(7), // 7 basic emotions
            action_units: Vec::with_capacity(30),
            temporal_features: Vec::with_capacity(64),
        }
    }
}

// Audio Modality Pools

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

/// Poolable audio feature extraction result
pub struct AudioFeatures {
    pub mfcc: Array2<f32>,
    pub pitch: Vec<f32>,
    pub energy: Vec<f32>,
    pub voice_quality: HashMap<String, f32>,
}

impl Poolable for AudioFeatures {
    fn reset(&mut self) {
        self.mfcc = Array2::zeros((0, 0));
        self.pitch.clear();
        self.energy.clear();
        self.voice_quality.clear();
    }
    
    fn create_new() -> Self {
        Self {
            mfcc: Array2::zeros((13, 100)), // 13 MFCC coefficients, 100 frames
            pitch: Vec::with_capacity(100),
            energy: Vec::with_capacity(100),
            voice_quality: HashMap::with_capacity(10),
        }
    }
}

/// Poolable audio window for FFT processing
pub struct AudioWindow {
    pub data: Vec<f32>,
    pub fft_buffer: Vec<num_complex::Complex<f32>>,
    pub window_function: Vec<f32>,
    size: usize,
}

impl AudioWindow {
    pub fn with_size(size: usize) -> Self {
        let window_function = Self::create_window_function(size);
        Self {
            data: vec![0.0; size],
            fft_buffer: vec![num_complex::Complex::zero(); size],
            window_function,
            size,
        }
    }
    
    fn create_window_function(size: usize) -> Vec<f32> {
        // Hann window
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }
}

impl Poolable for AudioWindow {
    fn reset(&mut self) {
        self.data.fill(0.0);
        self.fft_buffer.fill(num_complex::Complex::zero());
    }
    
    fn create_new() -> Self {
        Self::with_size(1024) // Common FFT size
    }
}

// Text Modality Pools

/// Poolable tokenization buffer
pub struct TokenizationBuffer {
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_type_ids: Vec<u8>,
    capacity: usize,
}

impl TokenizationBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            token_ids: Vec::with_capacity(capacity),
            attention_mask: Vec::with_capacity(capacity),
            token_type_ids: Vec::with_capacity(capacity),
            capacity,
        }
    }
}

impl Poolable for TokenizationBuffer {
    fn reset(&mut self) {
        self.tokens.clear();
        self.token_ids.clear();
        self.attention_mask.clear();
        self.token_type_ids.clear();
        
        // Shrink if over capacity
        if self.tokens.capacity() > self.capacity * 2 {
            self.tokens.shrink_to(self.capacity);
            self.token_ids.shrink_to(self.capacity);
            self.attention_mask.shrink_to(self.capacity);
            self.token_type_ids.shrink_to(self.capacity);
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(512) // BERT max sequence length
    }
}

/// Poolable text analysis result
pub struct TextAnalysisResult {
    pub embeddings: Array2<f32>,
    pub linguistic_features: HashMap<String, f32>,
    pub deception_markers: Vec<(String, f32)>,
    pub sentiment_scores: HashMap<String, f32>,
}

impl Poolable for TextAnalysisResult {
    fn reset(&mut self) {
        self.embeddings = Array2::zeros((0, 0));
        self.linguistic_features.clear();
        self.deception_markers.clear();
        self.sentiment_scores.clear();
    }
    
    fn create_new() -> Self {
        Self {
            embeddings: Array2::zeros((512, 768)), // BERT dimensions
            linguistic_features: HashMap::with_capacity(50),
            deception_markers: Vec::with_capacity(20),
            sentiment_scores: HashMap::with_capacity(10),
        }
    }
}

// Neural Network Pools

/// Poolable tensor for neural network operations
pub struct PooledTensor {
    pub data: ArrayD<f32>,
    pub shape: Vec<usize>,
    capacity: usize,
}

impl PooledTensor {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: ArrayD::zeros(vec![1]),
            shape: vec![1],
            capacity,
        }
    }
    
    /// Reshape tensor for new dimensions
    pub fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        let total_elements: usize = shape.iter().product();
        if total_elements > self.capacity {
            return Err(VeritasError::MemoryError(
                format!("Tensor size {} exceeds capacity", total_elements)
            ));
        }
        
        self.shape = shape.clone();
        self.data = ArrayD::zeros(shape);
        Ok(())
    }
}

impl Poolable for PooledTensor {
    fn reset(&mut self) {
        self.data = ArrayD::zeros(vec![1]);
        self.shape = vec![1];
    }
    
    fn create_new() -> Self {
        Self::with_capacity(1024 * 1024) // 1M elements
    }
}

/// Poolable gradient buffer
pub struct GradientBuffer {
    pub gradients: HashMap<String, ArrayD<f32>>,
    pub optimizer_state: HashMap<String, ArrayD<f32>>,
}

impl Poolable for GradientBuffer {
    fn reset(&mut self) {
        self.gradients.clear();
        self.optimizer_state.clear();
    }
    
    fn create_new() -> Self {
        Self {
            gradients: HashMap::with_capacity(100),
            optimizer_state: HashMap::with_capacity(100),
        }
    }
}

// Fusion Modality Pools

/// Poolable multi-modal fusion buffer
pub struct FusionBuffer {
    pub vision_features: Array1<f32>,
    pub audio_features: Array1<f32>,
    pub text_features: Array1<f32>,
    pub physiological_features: Array1<f32>,
    pub attention_weights: Array2<f32>,
}

impl Poolable for FusionBuffer {
    fn reset(&mut self) {
        self.vision_features = Array1::zeros(0);
        self.audio_features = Array1::zeros(0);
        self.text_features = Array1::zeros(0);
        self.physiological_features = Array1::zeros(0);
        self.attention_weights = Array2::zeros((0, 0));
    }
    
    fn create_new() -> Self {
        Self {
            vision_features: Array1::zeros(512),
            audio_features: Array1::zeros(256),
            text_features: Array1::zeros(768),
            physiological_features: Array1::zeros(128),
            attention_weights: Array2::zeros((4, 4)), // 4 modalities
        }
    }
}

// Specialized Pools

/// Poolable vector implementation with Cow for zero-copy operations
pub struct PooledVec<T: Clone> {
    pub data: Cow<'static, [T]>,
    capacity: usize,
}

impl<T: Clone + Default + Send + Sync> PooledVec<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Cow::Owned(Vec::with_capacity(capacity)),
            capacity,
        }
    }
    
    /// Convert to owned if borrowed
    pub fn make_owned(&mut self) {
        if let Cow::Borrowed(_) = self.data {
            self.data = Cow::Owned(self.data.to_vec());
        }
    }
}

impl<T: Clone + Default + Send + Sync + 'static> Poolable for PooledVec<T> {
    fn reset(&mut self) {
        match &mut self.data {
            Cow::Owned(v) => {
                v.clear();
                if v.capacity() > self.capacity * 2 {
                    v.shrink_to(self.capacity);
                }
            }
            Cow::Borrowed(_) => {
                self.data = Cow::Owned(Vec::with_capacity(self.capacity));
            }
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(1024)
    }
}

/// Poolable string buffer with interning support
pub struct PooledString {
    pub data: String,
    pub interned: Option<Arc<str>>,
    capacity: usize,
}

impl PooledString {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: String::with_capacity(capacity),
            interned: None,
            capacity,
        }
    }
    
    /// Intern the string for reduced memory usage
    pub fn intern(&mut self) {
        if !self.data.is_empty() {
            self.interned = Some(Arc::from(self.data.as_str()));
        }
    }
}

impl Poolable for PooledString {
    fn reset(&mut self) {
        self.data.clear();
        self.interned = None;
        
        if self.data.capacity() > self.capacity * 2 {
            self.data.shrink_to(self.capacity);
        }
    }
    
    fn create_new() -> Self {
        Self::with_capacity(256)
    }
}

/// Global object pools for the application
pub struct GlobalPools {
    // Vision pools
    pub image_buffer_pool: ObjectPool<ImageBuffer>,
    pub face_detection_pool: ObjectPool<FaceDetectionResult>,
    pub micro_expression_pool: ObjectPool<MicroExpressionResult>,
    
    // Audio pools
    pub audio_chunk_pool: ObjectPool<AudioChunk>,
    pub audio_features_pool: ObjectPool<AudioFeatures>,
    pub audio_window_pool: ObjectPool<AudioWindow>,
    
    // Text pools
    pub tokenization_pool: ObjectPool<TokenizationBuffer>,
    pub text_analysis_pool: ObjectPool<TextAnalysisResult>,
    
    // Neural network pools
    pub tensor_pool: ObjectPool<PooledTensor>,
    pub gradient_pool: ObjectPool<GradientBuffer>,
    
    // Fusion pools
    pub fusion_buffer_pool: ObjectPool<FusionBuffer>,
    
    // Generic pools
    pub vec_f32_pool: ObjectPool<PooledVec<f32>>,
    pub vec_f64_pool: ObjectPool<PooledVec<f64>>,
    pub string_pool: ObjectPool<PooledString>,
}

impl GlobalPools {
    pub fn new() -> Self {
        Self {
            // Vision pools
            image_buffer_pool: ObjectPool::new(PoolConfig {
                capacity: 32,
                preallocate: true,
                prealloc_size: 8,
                ..Default::default()
            }),
            face_detection_pool: ObjectPool::new(PoolConfig {
                capacity: 64,
                preallocate: true,
                prealloc_size: 16,
                ..Default::default()
            }),
            micro_expression_pool: ObjectPool::new(PoolConfig {
                capacity: 64,
                preallocate: true,
                prealloc_size: 16,
                ..Default::default()
            }),
            
            // Audio pools
            audio_chunk_pool: ObjectPool::new(PoolConfig {
                capacity: 128,
                preallocate: true,
                prealloc_size: 32,
                ..Default::default()
            }),
            audio_features_pool: ObjectPool::new(PoolConfig {
                capacity: 64,
                preallocate: true,
                prealloc_size: 16,
                ..Default::default()
            }),
            audio_window_pool: ObjectPool::new(PoolConfig {
                capacity: 256,
                preallocate: true,
                prealloc_size: 64,
                ..Default::default()
            }),
            
            // Text pools
            tokenization_pool: ObjectPool::new(PoolConfig {
                capacity: 128,
                preallocate: true,
                prealloc_size: 32,
                ..Default::default()
            }),
            text_analysis_pool: ObjectPool::new(PoolConfig {
                capacity: 64,
                preallocate: true,
                prealloc_size: 16,
                ..Default::default()
            }),
            
            // Neural network pools
            tensor_pool: ObjectPool::new(PoolConfig {
                capacity: 256,
                preallocate: true,
                prealloc_size: 64,
                ..Default::default()
            }),
            gradient_pool: ObjectPool::new(PoolConfig {
                capacity: 128,
                preallocate: true,
                prealloc_size: 32,
                ..Default::default()
            }),
            
            // Fusion pools
            fusion_buffer_pool: ObjectPool::new(PoolConfig {
                capacity: 64,
                preallocate: true,
                prealloc_size: 16,
                ..Default::default()
            }),
            
            // Generic pools
            vec_f32_pool: ObjectPool::new(PoolConfig {
                capacity: 512,
                preallocate: true,
                prealloc_size: 128,
                ..Default::default()
            }),
            vec_f64_pool: ObjectPool::new(PoolConfig {
                capacity: 256,
                preallocate: true,
                prealloc_size: 64,
                ..Default::default()
            }),
            string_pool: ObjectPool::new(PoolConfig {
                capacity: 256,
                preallocate: true,
                prealloc_size: 64,
                ..Default::default()
            }),
        }
    }
    
    /// Prefill all pools for better startup performance
    pub fn prefill_all(&self) -> Result<()> {
        // Use smaller prefill sizes to conserve memory
        self.image_buffer_pool.prefill(4)?;
        self.face_detection_pool.prefill(8)?;
        self.micro_expression_pool.prefill(8)?;
        
        self.audio_chunk_pool.prefill(16)?;
        self.audio_features_pool.prefill(8)?;
        self.audio_window_pool.prefill(32)?;
        
        self.tokenization_pool.prefill(16)?;
        self.text_analysis_pool.prefill(8)?;
        
        self.tensor_pool.prefill(32)?;
        self.gradient_pool.prefill(16)?;
        
        self.fusion_buffer_pool.prefill(8)?;
        
        self.vec_f32_pool.prefill(64)?;
        self.vec_f64_pool.prefill(32)?;
        self.string_pool.prefill(32)?;
        
        Ok(())
    }
    
    /// Get combined statistics for all pools
    pub fn all_stats(&self) -> HashMap<&'static str, PoolStats> {
        let mut stats = HashMap::new();
        
        // Vision
        stats.insert("image_buffer", self.image_buffer_pool.stats());
        stats.insert("face_detection", self.face_detection_pool.stats());
        stats.insert("micro_expression", self.micro_expression_pool.stats());
        
        // Audio
        stats.insert("audio_chunk", self.audio_chunk_pool.stats());
        stats.insert("audio_features", self.audio_features_pool.stats());
        stats.insert("audio_window", self.audio_window_pool.stats());
        
        // Text
        stats.insert("tokenization", self.tokenization_pool.stats());
        stats.insert("text_analysis", self.text_analysis_pool.stats());
        
        // Neural
        stats.insert("tensor", self.tensor_pool.stats());
        stats.insert("gradient", self.gradient_pool.stats());
        
        // Fusion
        stats.insert("fusion_buffer", self.fusion_buffer_pool.stats());
        
        // Generic
        stats.insert("vec_f32", self.vec_f32_pool.stats());
        stats.insert("vec_f64", self.vec_f64_pool.stats());
        stats.insert("string", self.string_pool.stats());
        
        stats
    }
    
    /// Calculate total memory saved by pooling
    pub fn memory_saved(&self) -> usize {
        let stats = self.all_stats();
        stats.values()
            .map(|s| s.total_reused * 1024) // Rough estimate
            .sum()
    }
}

// Thread-local object pools for reduced contention
thread_local! {
    static TL_VEC_F32_POOL: ObjectPool<PooledVec<f32>> = ObjectPool::new(PoolConfig {
        capacity: 64,
        preallocate: false,
        ..Default::default()
    });
    
    static TL_VEC_F64_POOL: ObjectPool<PooledVec<f64>> = ObjectPool::new(PoolConfig {
        capacity: 32,
        preallocate: false,
        ..Default::default()
    });
    
    static TL_STRING_POOL: ObjectPool<PooledString> = ObjectPool::new(PoolConfig {
        capacity: 32,
        preallocate: false,
        ..Default::default()
    });
    
    static TL_TENSOR_POOL: ObjectPool<PooledTensor> = ObjectPool::new(PoolConfig {
        capacity: 16,
        preallocate: false,
        ..Default::default()
    });
}

/// Get a thread-local f32 vector
pub fn get_tl_vec_f32() -> PooledObject<PooledVec<f32>> {
    TL_VEC_F32_POOL.with(|pool| pool.get())
}

/// Get a thread-local f64 vector
pub fn get_tl_vec_f64() -> PooledObject<PooledVec<f64>> {
    TL_VEC_F64_POOL.with(|pool| pool.get())
}

/// Get a thread-local string
pub fn get_tl_string() -> PooledObject<PooledString> {
    TL_STRING_POOL.with(|pool| pool.get())
}

/// Get a thread-local tensor
pub fn get_tl_tensor() -> PooledObject<PooledTensor> {
    TL_TENSOR_POOL.with(|pool| pool.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_object_pool_basic() {
        let pool: ObjectPool<PooledVec<f32>> = ObjectPool::new(PoolConfig::default());
        
        // Get object from pool
        let mut obj1 = pool.get();
        if let Cow::Owned(v) = &mut obj1.data {
            v.push(1.0);
            v.push(2.0);
        }
        
        // Return to pool by dropping
        drop(obj1);
        
        // Get again - should be reset
        let obj2 = pool.get();
        assert!(matches!(obj2.data, Cow::Owned(ref v) if v.is_empty()));
        
        let stats = pool.stats();
        assert_eq!(stats.total_created, 1);
        assert_eq!(stats.total_reused, 1);
    }
    
    #[test]
    fn test_image_buffer_pool() {
        let pool: ObjectPool<ImageBuffer> = ObjectPool::new(PoolConfig::default());
        
        let mut img = pool.get();
        img.resize_for_image(100, 100, 3).unwrap();
        assert_eq!(img.data.len(), 30000);
        
        drop(img);
        
        let img2 = pool.get();
        assert_eq!(img2.data.len(), 0);
        assert_eq!(img2.width, 0);
    }
    
    #[test]
    fn test_thread_local_pools() {
        let mut vec1 = get_tl_vec_f32();
        vec1.make_owned();
        if let Cow::Owned(v) = &mut vec1.data {
            v.extend(&[1.0, 2.0, 3.0]);
        }
        
        drop(vec1);
        
        let vec2 = get_tl_vec_f32();
        assert!(matches!(vec2.data, Cow::Owned(ref v) if v.is_empty()));
    }
    
    #[test]
    fn test_global_pools() {
        let pools = GlobalPools::new();
        
        // Test multiple pool types
        let _img = pools.image_buffer_pool.get();
        let _audio = pools.audio_chunk_pool.get();
        let _text = pools.tokenization_pool.get();
        
        let stats = pools.all_stats();
        assert!(stats.len() > 10);
        assert!(stats["image_buffer"].total_created > 0);
    }
}