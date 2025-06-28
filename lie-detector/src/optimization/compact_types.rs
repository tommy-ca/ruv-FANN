//! Enhanced compact data structures for maximum memory efficiency
//! 
//! This module provides highly optimized, memory-efficient data structures
//! specifically designed for each modality in the lie detector system.

use crate::{Result, VeritasError};
use arrayvec::ArrayVec;
use smallvec::SmallVec;
use indexmap::IndexMap;
use hashbrown::HashMap;
use tinyvec::TinyVec;
use ordered_float::OrderedFloat;
use std::sync::Arc;
use std::borrow::Cow;
use bitvec::prelude::*;
use half::f16;

/// Compact feature storage using array-backed storage for small collections
pub type CompactFeatures = SmallVec<[(InternedFeatureKey, f32); 16]>;

/// Interned feature key for reduced memory usage (16-bit instead of string)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedFeatureKey(u16);

/// Feature key interner with a fixed-size cache
pub struct FeatureKeyInterner {
    keys: Vec<Arc<str>>,
    indices: HashMap<Arc<str>, u16>,
}

impl FeatureKeyInterner {
    /// Create a new feature key interner
    pub fn new() -> Self {
        let mut interner = Self {
            keys: Vec::with_capacity(512),
            indices: HashMap::with_capacity(512),
        };
        
        // Pre-intern common feature keys across all modalities
        let common_keys = [
            // Statistical features
            "mean", "std", "min", "max", "variance", "skewness", "kurtosis",
            "median", "q1", "q3", "iqr", "range", "energy", "entropy",
            
            // Audio features
            "pitch", "formant1", "formant2", "formant3", "zcr", "rms",
            "spectral_centroid", "spectral_rolloff", "spectral_flux",
            "mfcc0", "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5",
            "mfcc6", "mfcc7", "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12",
            "jitter", "shimmer", "hnr", "voice_breaks", "tremor",
            
            // Vision features
            "gaze_x", "gaze_y", "pupil_diameter", "blink_rate", "blink_duration",
            "au01", "au02", "au04", "au05", "au06", "au07", "au09", "au10",
            "au12", "au14", "au15", "au17", "au20", "au23", "au25", "au26",
            "head_pitch", "head_yaw", "head_roll", "face_confidence",
            
            // Text features
            "word_count", "sentence_count", "avg_word_length", "lexical_diversity",
            "sentiment_pos", "sentiment_neg", "sentiment_neu", "readability",
            "pos_verb", "pos_noun", "pos_adj", "pos_adv", "hedge_words",
            
            // Physiological features
            "heart_rate", "heart_rate_var", "skin_conductance", "temperature",
            "respiration_rate", "blood_pressure_sys", "blood_pressure_dia",
            
            // General
            "confidence", "duration", "onset", "offset", "amplitude",
            "delta", "velocity", "acceleration", "correlation", "coherence",
        ];
        
        for key in &common_keys {
            interner.intern(key);
        }
        
        interner
    }
    
    /// Intern a feature key
    pub fn intern(&mut self, key: &str) -> InternedFeatureKey {
        let arc_key: Arc<str> = key.into();
        
        if let Some(&idx) = self.indices.get(&arc_key) {
            return InternedFeatureKey(idx);
        }
        
        let idx = self.keys.len() as u16;
        if idx == u16::MAX {
            panic!("Feature key interner capacity exceeded");
        }
        
        self.indices.insert(arc_key.clone(), idx);
        self.keys.push(arc_key);
        
        InternedFeatureKey(idx)
    }
    
    /// Resolve an interned key
    pub fn resolve(&self, key: InternedFeatureKey) -> &str {
        &self.keys[key.0 as usize]
    }
    
    /// Get or intern a key
    pub fn get_or_intern(&mut self, key: &str) -> InternedFeatureKey {
        self.intern(key)
    }
}

/// Compact analysis result optimized for memory
#[derive(Debug, Clone)]
pub struct CompactAnalysisResult {
    /// Deception score (0-255, mapped from 0.0-1.0)
    pub deception_score: u8,
    /// Confidence (0-255, mapped from 0.0-1.0)
    pub confidence: u8,
    /// Compact features storage
    pub features: CompactFeatures,
    /// Modality as enum to save space
    pub modality: CompactModality,
    /// Timestamp as seconds since epoch
    pub timestamp_secs: u32,
    /// Processing duration in milliseconds
    pub duration_ms: u16,
}

impl CompactAnalysisResult {
    /// Convert deception score to float
    pub fn deception_score_f32(&self) -> f32 {
        self.deception_score as f32 / 255.0
    }
    
    /// Convert confidence to float
    pub fn confidence_f32(&self) -> f32 {
        self.confidence as f32 / 255.0
    }
    
    /// Set deception score from float
    pub fn set_deception_score(&mut self, score: f32) {
        self.deception_score = (score.clamp(0.0, 1.0) * 255.0) as u8;
    }
    
    /// Set confidence from float
    pub fn set_confidence(&mut self, conf: f32) {
        self.confidence = (conf.clamp(0.0, 1.0) * 255.0) as u8;
    }
}

/// Compact modality representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompactModality {
    Vision = 0,
    Audio = 1,
    Text = 2,
    Physiological = 3,
    Fusion = 4,
}

impl CompactModality {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CompactModality::Vision => "vision",
            CompactModality::Audio => "audio",
            CompactModality::Text => "text",
            CompactModality::Physiological => "physiological",
            CompactModality::Fusion => "fusion",
        }
    }
}

// Vision-specific compact types

/// Compact face landmark representation using 16-bit coordinates
#[derive(Debug, Clone, Copy)]
pub struct CompactLandmark {
    /// X coordinate (0-65535 mapped to image width)
    pub x: u16,
    /// Y coordinate (0-65535 mapped to image height)
    pub y: u16,
    /// Visibility/confidence (0-255)
    pub confidence: u8,
}

impl CompactLandmark {
    pub fn from_normalized(x: f32, y: f32, confidence: f32) -> Self {
        Self {
            x: (x.clamp(0.0, 1.0) * 65535.0) as u16,
            y: (y.clamp(0.0, 1.0) * 65535.0) as u16,
            confidence: (confidence.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }
    
    pub fn to_normalized(&self) -> (f32, f32, f32) {
        (
            self.x as f32 / 65535.0,
            self.y as f32 / 65535.0,
            self.confidence as f32 / 255.0,
        )
    }
}

/// Compact face data for vision modality
#[derive(Debug, Clone)]
pub struct CompactFaceData {
    /// Bounding box as normalized u16 values
    pub bbox: [u16; 4], // [x, y, width, height]
    /// Key facial landmarks (68 points compressed to 17 key points)
    pub landmarks: ArrayVec<CompactLandmark, 17>,
    /// Action units as bit flags (AU1-AU27)
    pub action_units: BitArray<u32>,
    /// Head pose angles quantized to 8-bit
    pub head_pose: [u8; 3], // [pitch, yaw, roll] mapped from -90 to 90
    /// Face ID for tracking
    pub face_id: u16,
}

/// Compact gaze data
#[derive(Debug, Clone, Copy)]
pub struct CompactGazeData {
    /// Gaze direction as normalized 16-bit values
    pub direction: [u16; 2], // [x, y] mapped to -1 to 1
    /// Pupil diameter in millimeters * 100
    pub pupil_diameter: u16,
    /// Eye openness (0-255)
    pub openness: [u8; 2], // [left, right]
}

// Audio-specific compact types

/// Compact audio frame for streaming processing
#[derive(Debug, Clone)]
pub struct CompactAudioFrame {
    /// Frame timestamp in milliseconds
    pub timestamp_ms: u32,
    /// RMS energy quantized to 16-bit
    pub energy: u16,
    /// Zero crossing rate
    pub zcr: u16,
    /// Spectral centroid
    pub spectral_centroid: u16,
    /// MFCC coefficients quantized to 8-bit
    pub mfcc: ArrayVec<i8, 13>,
    /// Voice quality metrics packed
    pub voice_quality: CompactVoiceQuality,
}

/// Compact voice quality metrics
#[derive(Debug, Clone, Copy)]
pub struct CompactVoiceQuality {
    /// Pitch in Hz / 2
    pub pitch: u8,
    /// Jitter * 1000
    pub jitter: u8,
    /// Shimmer * 100
    pub shimmer: u8,
    /// HNR in dB
    pub hnr: u8,
}

/// Compact spectrogram representation using half precision
#[derive(Debug, Clone)]
pub struct CompactSpectrogram {
    /// Time bins
    pub time_bins: u16,
    /// Frequency bins
    pub freq_bins: u16,
    /// Magnitude data in half precision
    pub data: Vec<f16>,
}

impl CompactSpectrogram {
    pub fn new(time_bins: u16, freq_bins: u16) -> Self {
        Self {
            time_bins,
            freq_bins,
            data: vec![f16::ZERO; time_bins as usize * freq_bins as usize],
        }
    }
    
    pub fn set(&mut self, time: usize, freq: usize, value: f32) {
        let idx = time * self.freq_bins as usize + freq;
        self.data[idx] = f16::from_f32(value);
    }
    
    pub fn get(&self, time: usize, freq: usize) -> f32 {
        let idx = time * self.freq_bins as usize + freq;
        self.data[idx].to_f32()
    }
}

// Text-specific compact types

/// Compact token representation
#[derive(Debug, Clone, Copy)]
pub struct CompactToken {
    /// Token ID from vocabulary
    pub id: u16,
    /// Position in text
    pub position: u16,
    /// Part of speech tag
    pub pos: u8,
    /// Dependency relation
    pub dep: u8,
}

/// Compact text features
#[derive(Debug, Clone)]
pub struct CompactTextFeatures {
    /// Token sequence
    pub tokens: SmallVec<[CompactToken; 128]>,
    /// Sentence boundaries as bit vector
    pub sentence_boundaries: BitVec,
    /// Named entities as (start, end, type)
    pub entities: SmallVec<[(u16, u16, u8); 16]>,
    /// Sentiment scores quantized
    pub sentiment: [u8; 3], // [positive, negative, neutral]
    /// Linguistic complexity metrics
    pub complexity: CompactLinguisticComplexity,
}

/// Compact linguistic complexity metrics
#[derive(Debug, Clone, Copy)]
pub struct CompactLinguisticComplexity {
    /// Type-token ratio * 255
    pub ttr: u8,
    /// Average word length * 10
    pub avg_word_len: u8,
    /// Average sentence length
    pub avg_sent_len: u8,
    /// Flesch reading ease / 2
    pub readability: u8,
}

// Physiological-specific compact types

/// Compact physiological sample
#[derive(Debug, Clone, Copy)]
pub struct CompactPhysioSample {
    /// Timestamp in milliseconds
    pub timestamp_ms: u32,
    /// Heart rate in BPM
    pub heart_rate: u8,
    /// Heart rate variability
    pub hrv: u8,
    /// Skin conductance * 100
    pub skin_conductance: u16,
    /// Temperature * 10 (e.g., 365 = 36.5°C)
    pub temperature: u16,
    /// Respiration rate
    pub respiration_rate: u8,
    /// Flags for various conditions
    pub flags: CompactPhysioFlags,
}

/// Compact physiological flags
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactPhysioFlags {
    flags: u8,
}

impl CompactPhysioFlags {
    pub const MOTION_ARTIFACT: u8 = 1 << 0;
    pub const POOR_CONTACT: u8 = 1 << 1;
    pub const BASELINE_SHIFT: u8 = 1 << 2;
    pub const ANOMALY: u8 = 1 << 3;
    
    pub fn set(&mut self, flag: u8, value: bool) {
        if value {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }
    
    pub fn is_set(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }
}

// Fusion-specific compact types

/// Compact multi-modal feature vector
#[derive(Debug, Clone)]
pub struct CompactMultiModalFeatures {
    /// Modal contributions (0-255 each)
    pub modal_weights: [u8; 4], // [vision, audio, text, physio]
    /// Fused feature vector using sparse representation
    pub features: CompactSparseVector,
    /// Cross-modal correlations
    pub correlations: CompactSymmetricMatrix<4>,
}

/// Compact symmetric matrix for small sizes
#[derive(Debug, Clone)]
pub struct CompactSymmetricMatrix<const N: usize> {
    /// Upper triangular packed storage - using fixed size to avoid const operations
    data: ArrayVec<f16, 10>, // Fixed size for common use cases
    actual_size: usize,
}

impl<const N: usize> CompactSymmetricMatrix<N> {
    pub fn new() -> Self {
        let actual_size = N * (N + 1) / 2;
        let mut data = ArrayVec::new();
        for _ in 0..actual_size {
            data.push(f16::ZERO);
        }
        Self {
            data,
            actual_size,
        }
    }
    
    fn index(i: usize, j: usize) -> usize {
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        j * (j + 1) / 2 + i
    }
    
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        self.data[Self::index(i, j)] = f16::from_f32(value);
    }
    
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[Self::index(i, j)].to_f32()
    }
}

/// Compact temporal buffer for time-series data
#[derive(Debug, Clone)]
pub struct CompactTemporalBuffer<T, const N: usize> {
    /// Ring buffer of values
    data: ArrayVec<T, N>,
    /// Current position in ring buffer
    position: usize,
    /// Whether buffer has wrapped
    wrapped: bool,
}

impl<T: Clone + Default, const N: usize> CompactTemporalBuffer<T, N> {
    /// Create a new temporal buffer
    pub fn new() -> Self {
        Self {
            data: ArrayVec::new(),
            position: 0,
            wrapped: false,
        }
    }
    
    /// Push a new value
    pub fn push(&mut self, value: T) {
        if self.data.len() < N {
            self.data.push(value);
        } else {
            self.data[self.position] = value;
            self.wrapped = true;
        }
        
        self.position = (self.position + 1) % N;
    }
    
    /// Get values in temporal order (oldest to newest)
    pub fn values(&self) -> Vec<&T> {
        let mut result = Vec::with_capacity(self.len());
        
        if self.wrapped {
            result.extend(&self.data[self.position..]);
        }
        
        result.extend(&self.data[..self.position]);
        result
    }
    
    /// Get buffer length
    pub fn len(&self) -> usize {
        if self.wrapped { N } else { self.data.len() }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.position = 0;
        self.wrapped = false;
    }
    
    /// Get most recent value
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            let idx = if self.position == 0 {
                if self.wrapped { N - 1 } else { return None }
            } else {
                self.position - 1
            };
            self.data.get(idx)
        }
    }
}

/// Compact sparse vector for features with many zeros
#[derive(Debug, Clone)]
pub struct CompactSparseVector {
    /// Non-zero indices and values
    entries: SmallVec<[(u16, f32); 32]>,
    /// Total dimension
    dimension: u16,
}

impl CompactSparseVector {
    /// Create a new sparse vector
    pub fn new(dimension: u16) -> Self {
        Self {
            entries: SmallVec::new(),
            dimension,
        }
    }
    
    /// Set a value (only stores non-zero)
    pub fn set(&mut self, index: u16, value: f32) {
        if value.abs() < f32::EPSILON {
            self.entries.retain(|(i, _)| *i != index);
        } else {
            if let Some(entry) = self.entries.iter_mut().find(|(i, _)| *i == index) {
                entry.1 = value;
            } else {
                self.entries.push((index, value));
                self.entries.sort_by_key(|(i, _)| *i);
            }
        }
    }
    
    /// Get a value
    pub fn get(&self, index: u16) -> f32 {
        self.entries.iter()
            .find(|(i, _)| *i == index)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }
    
    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
    
    /// Compute dot product with dense vector
    pub fn dot(&self, dense: &[f32]) -> f32 {
        debug_assert_eq!(dense.len(), self.dimension as usize);
        self.entries.iter()
            .map(|(i, v)| v * dense[*i as usize])
            .sum()
    }
    
    /// Convert to dense vector
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.dimension as usize];
        for &(idx, val) in &self.entries {
            dense[idx as usize] = val;
        }
        dense
    }
}

/// Compact matrix for small fixed-size matrices
#[derive(Debug, Clone)]
pub struct CompactMatrix<const R: usize, const C: usize> {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl<const R: usize, const C: usize> CompactMatrix<R, C> {
    /// Create a new zero matrix
    pub fn zeros() -> Self {
        Self { 
            data: vec![0.0; R * C],
            rows: R,
            cols: C,
        }
    }
    
    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> f32 {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col]
    }
    
    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col] = value;
    }
    
    /// Matrix multiply with vector
    pub fn mul_vec(&self, vec: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * vec[j];
            }
        }
        result
    }
}

/// Compact decision history for tracking recent decisions
pub type CompactDecisionHistory = TinyVec<[CompactDecision; 8]>;

/// Compact decision representation
#[derive(Debug, Clone, Copy)]
pub struct CompactDecision {
    /// Decision type (0=truth, 1=deception, 2=uncertain)
    pub decision: u8,
    /// Confidence (0-255)
    pub confidence: u8,
    /// Timestamp offset from base in seconds
    pub timestamp_offset: u16,
}

/// Memory usage estimator for compact types
pub struct CompactMemoryEstimator;

impl CompactMemoryEstimator {
    /// Estimate memory usage of CompactAnalysisResult
    pub fn analysis_result_size() -> usize {
        std::mem::size_of::<CompactAnalysisResult>() + 
        std::mem::size_of::<(InternedFeatureKey, f32)>() * 16
    }
    
    /// Estimate memory usage of CompactFaceData
    pub fn face_data_size() -> usize {
        std::mem::size_of::<CompactFaceData>() + 
        std::mem::size_of::<CompactLandmark>() * 17
    }
    
    /// Compare with non-compact equivalent
    pub fn savings_ratio() -> f32 {
        // Rough estimate: compact types use ~30-40% of original memory
        0.35
    }
}

/// Global feature key interner
static GLOBAL_INTERNER: once_cell::sync::Lazy<parking_lot::RwLock<FeatureKeyInterner>> = 
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(FeatureKeyInterner::new()));

/// Intern a feature key globally
pub fn intern_feature_key(key: &str) -> InternedFeatureKey {
    GLOBAL_INTERNER.write().intern(key)
}

/// Resolve an interned feature key
pub fn resolve_feature_key(key: InternedFeatureKey) -> String {
    GLOBAL_INTERNER.read().resolve(key).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compact_landmark() {
        let landmark = CompactLandmark::from_normalized(0.5, 0.75, 0.9);
        let (x, y, conf) = landmark.to_normalized();
        
        assert!((x - 0.5).abs() < 0.001);
        assert!((y - 0.75).abs() < 0.001);
        assert!((conf - 0.9).abs() < 0.01);
    }
    
    #[test]
    fn test_compact_spectrogram() {
        let mut spec = CompactSpectrogram::new(10, 20);
        spec.set(5, 10, 0.75);
        
        assert!((spec.get(5, 10) - 0.75).abs() < 0.01);
        assert_eq!(spec.get(0, 0), 0.0);
    }
    
    #[test]
    fn test_compact_symmetric_matrix() {
        let mut mat = CompactSymmetricMatrix::<3>::new();
        mat.set(0, 1, 0.5);
        mat.set(1, 2, 0.7);
        
        assert!((mat.get(0, 1) - 0.5).abs() < 0.01);
        assert!((mat.get(1, 0) - 0.5).abs() < 0.01); // Symmetric
        assert!((mat.get(1, 2) - 0.7).abs() < 0.01);
    }
    
    #[test]
    fn test_memory_savings() {
        let compact_size = CompactMemoryEstimator::analysis_result_size();
        let regular_size = 
            std::mem::size_of::<f32>() * 2 + // scores
            std::mem::size_of::<HashMap<String, f32>>() + // features
            std::mem::size_of::<String>() * 20 + // feature keys
            std::mem::size_of::<f32>() * 20 + // feature values
            std::mem::size_of::<String>() + // modality
            std::mem::size_of::<u64>() + // timestamp
            std::mem::size_of::<u64>(); // duration
        
        println!("Compact size: {} bytes", compact_size);
        println!("Regular size estimate: {} bytes", regular_size);
        println!("Savings: {:.1}%", (1.0 - compact_size as f32 / regular_size as f32) * 100.0);
        
        assert!(compact_size < regular_size / 2);
    }
}