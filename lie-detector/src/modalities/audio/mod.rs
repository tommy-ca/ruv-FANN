//! Audio Processing Module for Veritas-Nexus
//!
//! This module provides high-performance audio analysis capabilities for lie detection,
//! including voice analysis, pitch detection, stress feature extraction, and MFCC computation.
//! 
//! # Features
//! - Real-time voice activity detection
//! - Pitch and tone analysis with SIMD optimization
//! - Stress indicators (jitter, shimmer, energy)
//! - Mel-frequency cepstral coefficients (MFCC)
//! - Spectral feature extraction
//! - Voice quality metrics

use std::time::Duration;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod voice_analyzer;
pub mod pitch_detection;
pub mod stress_features;
pub mod mfcc;
pub mod simd_optimized;

pub use voice_analyzer::VoiceAnalyzer;
pub use pitch_detection::{PitchDetector, PitchFeatures};
pub use stress_features::{StressFeatures, StressDetector};
pub use mfcc::{MfccExtractor, MfccFeatures};
pub use simd_optimized::{SimdFFT, SimdMfccExtractor, SimdPitchDetector, SimdVoiceStressAnalyzer};

/// Core audio analysis trait for the modality system
pub trait AudioAnalyzer: Send + Sync {
    /// Process an audio chunk and extract features
    fn process_chunk(&mut self, chunk: &[f32], sample_rate: u32) -> Result<AudioFeatures>;
    
    /// Configure the analyzer for optimal performance
    fn configure(&mut self, config: AudioConfig) -> Result<()>;
    
    /// Get the minimum required chunk size for processing
    fn min_chunk_size(&self) -> usize;
    
    /// Check if the analyzer supports real-time processing
    fn supports_realtime(&self) -> bool;
}

/// Audio configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub chunk_size: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub enable_voice_activity_detection: bool,
    pub enable_pitch_detection: bool,
    pub enable_stress_analysis: bool,
    pub enable_mfcc: bool,
    pub num_mfcc_coefficients: usize,
    pub mel_filters: usize,
    pub min_frequency: f32,
    pub max_frequency: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            chunk_size: 1024,
            hop_length: 512,
            window_size: 1024,
            enable_voice_activity_detection: true,
            enable_pitch_detection: true,
            enable_stress_analysis: true,
            enable_mfcc: true,
            num_mfcc_coefficients: 13,
            mel_filters: 26,
            min_frequency: 80.0,
            max_frequency: 8000.0,
        }
    }
}

/// Comprehensive audio features extracted from voice data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    /// Mel-frequency cepstral coefficients
    pub mfcc: Option<MfccFeatures>,
    
    /// Pitch and fundamental frequency features
    pub pitch: Option<PitchFeatures>,
    
    /// Voice stress indicators
    pub stress: Option<StressFeatures>,
    
    /// Voice activity detection result
    pub voice_activity: VoiceActivity,
    
    /// Spectral features
    pub spectral: SpectralFeatures,
    
    /// Energy and power features
    pub energy: EnergyFeatures,
    
    /// Quality metrics
    pub quality: VoiceQuality,
    
    /// Timestamp information
    pub timestamp: Duration,
}

/// Voice activity detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceActivity {
    pub is_speech: bool,
    pub confidence: f32,
    pub energy_threshold: f32,
    pub zero_crossing_rate: f32,
    pub spectral_centroid: f32,
}

/// Spectral features for voice analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    pub spectral_centroid: f32,
    pub spectral_bandwidth: f32,
    pub spectral_rolloff: f32,
    pub spectral_flux: f32,
    pub spectral_flatness: f32,
    pub harmonic_ratio: f32,
    pub noise_ratio: f32,
}

/// Energy and power-related features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyFeatures {
    pub rms_energy: f32,
    pub log_energy: f32,
    pub total_energy: f32,
    pub energy_entropy: f32,
    pub short_time_energy: Vec<f32>,
}

/// Voice quality metrics for deception detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQuality {
    pub signal_to_noise_ratio: f32,
    pub harmonic_to_noise_ratio: f32,
    pub breathiness: f32,
    pub roughness: f32,
    pub hoarseness: f32,
    pub vocal_tremor: f32,
}

/// Audio processing error types
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),
    
    #[error("Insufficient audio data: got {got} samples, need at least {required}")]
    InsufficientData { got: usize, required: usize },
    
    #[error("FFT size must be power of 2: {0}")]
    InvalidFftSize(usize),
    
    #[error("Mel filter configuration error: {0}")]
    MelFilterError(String),
    
    #[error("Pitch detection failed: {0}")]
    PitchDetectionError(String),
    
    #[error("Voice activity detection failed: {0}")]
    VadError(String),
}

/// Utility functions for audio processing
pub mod utils {
    use super::*;
    
    /// Apply Hamming window to audio data
    pub fn apply_hamming_window(data: &mut [f32]) {
        let n = data.len();
        for (i, sample) in data.iter_mut().enumerate() {
            let window_val = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
            *sample *= window_val;
        }
    }
    
    /// Apply Hanning window to audio data
    pub fn apply_hanning_window(data: &mut [f32]) {
        let n = data.len();
        for (i, sample) in data.iter_mut().enumerate() {
            let window_val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            *sample *= window_val;
        }
    }
    
    /// Compute zero-crossing rate
    pub fn zero_crossing_rate(data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let mut crossings = 0;
        for window in data.windows(2) {
            if (window[0] >= 0.0) != (window[1] >= 0.0) {
                crossings += 1;
            }
        }
        
        crossings as f32 / (data.len() - 1) as f32
    }
    
    /// Compute RMS energy
    pub fn rms_energy(data: &[f32]) -> f32 {
        let sum_squares: f32 = data.iter().map(|x| x * x).sum();
        (sum_squares / data.len() as f32).sqrt()
    }
    
    /// Pre-emphasis filter for speech processing
    pub fn pre_emphasis(data: &mut [f32], alpha: f32) {
        for i in (1..data.len()).rev() {
            data[i] -= alpha * data[i - 1];
        }
    }
    
    /// Normalize audio data to [-1, 1] range
    pub fn normalize(data: &mut [f32]) {
        let max_val = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for sample in data {
                *sample /= max_val;
            }
        }
    }
    
    /// Check if value is power of 2
    pub fn is_power_of_two(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }
    
    /// Get next power of 2
    pub fn next_power_of_two(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        let mut power = 1;
        while power < n {
            power <<= 1;
        }
        power
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::utils::*;
    
    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.chunk_size, 1024);
        assert!(config.enable_voice_activity_detection);
    }
    
    #[test]
    fn test_hamming_window() {
        let mut data = vec![1.0; 8];
        apply_hamming_window(&mut data);
        
        // First and last samples should be close to 0.08
        assert!((data[0] - 0.08).abs() < 0.01);
        assert!((data[7] - 0.08).abs() < 0.01);
        
        // Middle sample should be close to 1.0
        assert!((data[4] - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_zero_crossing_rate() {
        let data = vec![1.0, -1.0, 1.0, -1.0];
        let zcr = zero_crossing_rate(&data);
        assert_eq!(zcr, 1.0); // 3 crossings in 3 intervals
    }
    
    #[test]
    fn test_rms_energy() {
        let data = vec![1.0, -1.0, 1.0, -1.0];
        let energy = rms_energy(&data);
        assert_eq!(energy, 1.0);
    }
    
    #[test]
    fn test_normalize() {
        let mut data = vec![2.0, -4.0, 1.0];
        normalize(&mut data);
        assert_eq!(data, vec![0.5, -1.0, 0.25]);
    }
    
    #[test]
    fn test_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(1023));
    }
    
    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(1000), 1024);
    }
}

/// Comprehensive audio analyzer that integrates all audio processing components
pub struct ComprehensiveAudioAnalyzer {
    config: AudioConfig,
    voice_analyzer: VoiceAnalyzer,
    pitch_detector: PitchDetector,
    stress_detector: StressDetector,
    mfcc_extractor: MfccExtractor,
    frame_count: usize,
}

impl ComprehensiveAudioAnalyzer {
    /// Create new comprehensive audio analyzer
    pub fn new(config: AudioConfig) -> Result<Self> {
        let voice_analyzer = VoiceAnalyzer::with_config(config.clone())?;
        let pitch_detector = PitchDetector::new(&config)?;
        let stress_detector = StressDetector::new(&config)?;
        let mfcc_extractor = MfccExtractor::new(&config)?;
        
        Ok(Self {
            config,
            voice_analyzer,
            pitch_detector,
            stress_detector,
            mfcc_extractor,
            frame_count: 0,
        })
    }
    
    /// Process audio chunk with all enabled features
    pub fn process_comprehensive(&mut self, chunk: &[f32]) -> Result<AudioFeatures> {
        // Basic voice analysis (includes VAD, spectral, energy, quality)
        let mut features = self.voice_analyzer.analyze_chunk(chunk)?;
        
        // Add timestamp
        features.timestamp = Duration::from_millis(
            (self.frame_count * self.config.hop_length * 1000) / self.config.sample_rate as usize
        );
        
        // Pitch detection if enabled and voice activity detected
        if self.config.enable_pitch_detection && features.voice_activity.is_speech {
            match self.pitch_detector.detect_pitch(chunk) {
                Ok(pitch_features) => {
                    features.pitch = Some(pitch_features);
                }
                Err(e) => {
                    // Log error but continue processing
                    tracing::warn!("Pitch detection failed: {}", e);
                }
            }
        }
        
        // MFCC extraction if enabled
        if self.config.enable_mfcc {
            match self.mfcc_extractor.extract_features(chunk) {
                Ok(mfcc_features) => {
                    features.mfcc = Some(mfcc_features);
                }
                Err(e) => {
                    tracing::warn!("MFCC extraction failed: {}", e);
                }
            }
        }
        
        // Stress analysis if enabled and we have pitch information
        if self.config.enable_stress_analysis {
            let fundamental_frequency = features.pitch
                .as_ref()
                .map(|p| p.fundamental_frequency)
                .unwrap_or(0.0);
            
            match self.stress_detector.analyze_stress(
                chunk, 
                fundamental_frequency, 
                &features.voice_activity
            ) {
                Ok(stress_features) => {
                    features.stress = Some(stress_features);
                }
                Err(e) => {
                    tracing::warn!("Stress analysis failed: {}", e);
                }
            }
        }
        
        self.frame_count += 1;
        
        Ok(features)
    }
    
    /// Get configuration
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }
    
    /// Reset internal state
    pub fn reset(&mut self) -> Result<()> {
        self.voice_analyzer = VoiceAnalyzer::with_config(self.config.clone())?;
        self.pitch_detector = PitchDetector::new(&self.config)?;
        self.stress_detector = StressDetector::new(&self.config)?;
        self.mfcc_extractor = MfccExtractor::new(&self.config)?;
        self.frame_count = 0;
        Ok(())
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        ProcessingStats {
            frames_processed: self.frame_count,
            sample_rate: self.config.sample_rate,
            chunk_size: self.config.chunk_size,
            total_duration_ms: (self.frame_count * self.config.hop_length * 1000) / self.config.sample_rate as usize,
        }
    }
}

impl AudioAnalyzer for ComprehensiveAudioAnalyzer {
    fn process_chunk(&mut self, chunk: &[f32], sample_rate: u32) -> Result<AudioFeatures> {
        if sample_rate != self.config.sample_rate {
            return Err(AudioError::InvalidSampleRate(sample_rate).into());
        }
        
        self.process_comprehensive(chunk)
    }
    
    fn configure(&mut self, config: AudioConfig) -> Result<()> {
        self.config = config.clone();
        self.voice_analyzer.configure(config.clone())?;
        self.pitch_detector = PitchDetector::new(&config)?;
        self.stress_detector = StressDetector::new(&config)?;
        self.mfcc_extractor = MfccExtractor::new(&config)?;
        self.frame_count = 0;
        Ok(())
    }
    
    fn min_chunk_size(&self) -> usize {
        self.config.chunk_size
    }
    
    fn supports_realtime(&self) -> bool {
        true
    }
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub frames_processed: usize,
    pub sample_rate: u32,
    pub chunk_size: usize,
    pub total_duration_ms: usize,
}

/// Optimized audio analyzer factory
pub struct AudioAnalyzerFactory;

impl AudioAnalyzerFactory {
    /// Create optimized audio analyzer based on target platform and requirements
    pub fn create_analyzer(config: AudioConfig) -> Result<Box<dyn AudioAnalyzer>> {
        // Choose implementation based on configuration and available features
        
        #[cfg(feature = "gpu")]
        if config.sample_rate >= 44100 && config.chunk_size >= 2048 {
            // Use GPU-accelerated version for high sample rates
            return Self::create_gpu_analyzer(config);
        }
        
        #[cfg(feature = "simd-avx2")]
        if is_x86_feature_detected!("avx2") {
            // Use SIMD-optimized version
            return Self::create_simd_analyzer(config);
        }
        
        // Default CPU implementation
        let analyzer = ComprehensiveAudioAnalyzer::new(config)?;
        Ok(Box::new(analyzer))
    }
    
    /// Create analyzer optimized for embedded systems
    pub fn create_embedded_analyzer(config: AudioConfig) -> Result<Box<dyn AudioAnalyzer>> {
        // Use minimal configuration for embedded systems
        let embedded_config = AudioConfig {
            sample_rate: config.sample_rate.min(16000),
            chunk_size: config.chunk_size.min(512),
            enable_mfcc: false,  // Disable computationally expensive features
            num_mfcc_coefficients: 8,  // Reduce MFCC count
            mel_filters: 16,     // Reduce mel filters
            ..config
        };
        
        let analyzer = ComprehensiveAudioAnalyzer::new(embedded_config)?;
        Ok(Box::new(analyzer))
    }
    
    /// Create analyzer for real-time applications
    pub fn create_realtime_analyzer(config: AudioConfig) -> Result<Box<dyn AudioAnalyzer>> {
        // Optimize for low latency
        let realtime_config = AudioConfig {
            chunk_size: config.chunk_size.min(256),  // Small chunks for low latency
            hop_length: config.hop_length.min(128),
            window_size: config.window_size.min(512),
            ..config
        };
        
        let analyzer = ComprehensiveAudioAnalyzer::new(realtime_config)?;
        Ok(Box::new(analyzer))
    }
    
    #[cfg(feature = "gpu")]
    fn create_gpu_analyzer(config: AudioConfig) -> Result<Box<dyn AudioAnalyzer>> {
        // GPU-accelerated implementation would go here
        // For now, fallback to CPU
        let analyzer = ComprehensiveAudioAnalyzer::new(config)?;
        Ok(Box::new(analyzer))
    }
    
    #[cfg(feature = "simd-avx2")]
    fn create_simd_analyzer(config: AudioConfig) -> Result<Box<dyn AudioAnalyzer>> {
        // SIMD-optimized implementation would go here
        // For now, use the regular implementation which has some SIMD optimizations
        let analyzer = ComprehensiveAudioAnalyzer::new(config)?;
        Ok(Box::new(analyzer))
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_comprehensive_analyzer_creation() {
        let config = AudioConfig::default();
        let analyzer = ComprehensiveAudioAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }
    
    #[test]
    fn test_comprehensive_audio_processing() {
        let config = AudioConfig::default();
        let mut analyzer = ComprehensiveAudioAnalyzer::new(config).unwrap();
        
        // Generate test audio (440 Hz sine wave)
        let sample_rate = analyzer.config().sample_rate as f32;
        let chunk_size = analyzer.config().chunk_size;
        let frequency = 440.0;
        
        let chunk: Vec<f32> = (0..chunk_size)
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin())
            .collect();
        
        let result = analyzer.process_comprehensive(&chunk);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        
        // Check that basic features are present
        assert!(features.voice_activity.is_speech);
        assert!(features.energy.rms_energy > 0.0);
        assert!(features.spectral.spectral_centroid >= 0.0);
        
        // Check that optional features are computed when enabled
        if analyzer.config().enable_pitch_detection {
            assert!(features.pitch.is_some());
        }
        
        if analyzer.config().enable_mfcc {
            assert!(features.mfcc.is_some());
        }
        
        if analyzer.config().enable_stress_analysis {
            assert!(features.stress.is_some());
        }
    }
    
    #[test]
    fn test_analyzer_factory() {
        let config = AudioConfig::default();
        
        // Test default analyzer
        let analyzer = AudioAnalyzerFactory::create_analyzer(config.clone());
        assert!(analyzer.is_ok());
        
        // Test embedded analyzer
        let embedded = AudioAnalyzerFactory::create_embedded_analyzer(config.clone());
        assert!(embedded.is_ok());
        
        // Test realtime analyzer
        let realtime = AudioAnalyzerFactory::create_realtime_analyzer(config);
        assert!(realtime.is_ok());
    }
    
    #[test]
    fn test_analyzer_configuration() {
        let mut config = AudioConfig::default();
        config.enable_pitch_detection = false;
        config.enable_mfcc = false;
        config.enable_stress_analysis = false;
        
        let mut analyzer = ComprehensiveAudioAnalyzer::new(config.clone()).unwrap();
        
        // Process chunk with limited features
        let chunk = vec![0.5; config.chunk_size];
        let features = analyzer.process_comprehensive(&chunk).unwrap();
        
        assert!(features.pitch.is_none());
        assert!(features.mfcc.is_none());
        assert!(features.stress.is_none());
        
        // Reconfigure to enable all features
        config.enable_pitch_detection = true;
        config.enable_mfcc = true;
        config.enable_stress_analysis = true;
        
        analyzer.configure(config.clone()).unwrap();
        
        // Generate speech-like signal
        let speech_chunk: Vec<f32> = (0..config.chunk_size)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / config.sample_rate as f32).sin())
            .collect();
        
        let features = analyzer.process_comprehensive(&speech_chunk).unwrap();
        
        // Features should now be present
        assert!(features.pitch.is_some());
        assert!(features.mfcc.is_some());
        assert!(features.stress.is_some());
    }
    
    #[test]
    fn test_processing_stats() {
        let config = AudioConfig::default();
        let mut analyzer = ComprehensiveAudioAnalyzer::new(config).unwrap();
        
        let chunk = vec![0.1; analyzer.config().chunk_size];
        
        // Process multiple chunks
        for _ in 0..5 {
            let _ = analyzer.process_comprehensive(&chunk);
        }
        
        let stats = analyzer.get_stats();
        assert_eq!(stats.frames_processed, 5);
        assert_eq!(stats.sample_rate, analyzer.config().sample_rate);
        assert!(stats.total_duration_ms > 0);
    }
    
    #[test]
    fn test_reset_functionality() {
        let config = AudioConfig::default();
        let mut analyzer = ComprehensiveAudioAnalyzer::new(config).unwrap();
        
        let chunk = vec![0.1; analyzer.config().chunk_size];
        
        // Process some chunks
        for _ in 0..3 {
            let _ = analyzer.process_comprehensive(&chunk);
        }
        
        assert_eq!(analyzer.get_stats().frames_processed, 3);
        
        // Reset analyzer
        analyzer.reset().unwrap();
        
        assert_eq!(analyzer.get_stats().frames_processed, 0);
    }
}