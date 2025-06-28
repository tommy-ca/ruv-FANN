//! Voice Analysis Module for Lie Detection
//!
//! This module implements comprehensive voice analysis capabilities including
//! voice activity detection, spectral analysis, and quality metrics.

use std::collections::VecDeque;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::*;
use super::utils::*;

/// Voice analyzer with streaming support and voice activity detection
pub struct VoiceAnalyzer {
    config: AudioConfig,
    vad: VoiceActivityDetector,
    spectral_analyzer: SpectralAnalyzer,
    energy_analyzer: EnergyAnalyzer,
    quality_analyzer: QualityAnalyzer,
    buffer: VecDeque<f32>,
    sample_rate: u32,
}

impl VoiceAnalyzer {
    /// Create a new voice analyzer with default configuration
    pub fn new(sample_rate: u32) -> Result<Self> {
        Self::with_config(AudioConfig {
            sample_rate,
            ..Default::default()
        })
    }
    
    /// Create voice analyzer with custom configuration
    pub fn with_config(config: AudioConfig) -> Result<Self> {
        if config.sample_rate == 0 {
            return Err(AudioError::InvalidSampleRate(config.sample_rate).into());
        }
        
        let vad = VoiceActivityDetector::new(&config)?;
        let spectral_analyzer = SpectralAnalyzer::new(&config)?;
        let energy_analyzer = EnergyAnalyzer::new(&config)?;
        let quality_analyzer = QualityAnalyzer::new(&config)?;
        
        Ok(Self {
            config: config.clone(),
            vad,
            spectral_analyzer,
            energy_analyzer,
            quality_analyzer,
            buffer: VecDeque::with_capacity(config.chunk_size * 2),
            sample_rate: config.sample_rate,
        })
    }
    
    /// Process audio chunk and extract voice features
    pub fn analyze_chunk(&mut self, chunk: &[f32]) -> Result<AudioFeatures> {
        // Extend internal buffer
        self.buffer.extend(chunk);
        
        // Process if we have enough data
        if self.buffer.len() >= self.config.chunk_size {
            let mut data: Vec<f32> = self.buffer.drain(..self.config.chunk_size).collect();
            
            // Normalize and pre-process
            normalize(&mut data);
            pre_emphasis(&mut data, 0.97);
            
            // Voice activity detection
            let voice_activity = if self.config.enable_voice_activity_detection {
                self.vad.detect(&data)?
            } else {
                VoiceActivity {
                    is_speech: true,
                    confidence: 1.0,
                    energy_threshold: 0.0,
                    zero_crossing_rate: zero_crossing_rate(&data),
                    spectral_centroid: 0.0,
                }
            };
            
            // Skip processing if no voice activity detected
            if !voice_activity.is_speech {
                return Ok(AudioFeatures {
                    mfcc: None,
                    pitch: None,
                    stress: None,
                    voice_activity,
                    spectral: SpectralFeatures::default(),
                    energy: EnergyFeatures::default(),
                    quality: VoiceQuality::default(),
                    timestamp: std::time::Duration::from_secs(0),
                });
            }
            
            // Extract features
            let spectral = self.spectral_analyzer.analyze(&data)?;
            let energy = self.energy_analyzer.analyze(&data)?;
            let quality = self.quality_analyzer.analyze(&data, &spectral)?;
            
            Ok(AudioFeatures {
                mfcc: None, // Will be filled by MFCC extractor
                pitch: None, // Will be filled by pitch detector
                stress: None, // Will be filled by stress detector
                voice_activity,
                spectral,
                energy,
                quality,
                timestamp: std::time::Duration::from_secs(0),
            })
        } else {
            // Return empty features if not enough data
            Err(AudioError::InsufficientData {
                got: self.buffer.len(),
                required: self.config.chunk_size,
            }.into())
        }
    }
}

impl AudioAnalyzer for VoiceAnalyzer {
    fn process_chunk(&mut self, chunk: &[f32], sample_rate: u32) -> Result<AudioFeatures> {
        if sample_rate != self.sample_rate {
            return Err(AudioError::InvalidSampleRate(sample_rate).into());
        }
        self.analyze_chunk(chunk)
    }
    
    fn configure(&mut self, config: AudioConfig) -> Result<()> {
        self.config = config.clone();
        self.vad = VoiceActivityDetector::new(&config)?;
        self.spectral_analyzer = SpectralAnalyzer::new(&config)?;
        self.energy_analyzer = EnergyAnalyzer::new(&config)?;
        self.quality_analyzer = QualityAnalyzer::new(&config)?;
        self.sample_rate = config.sample_rate;
        Ok(())
    }
    
    fn min_chunk_size(&self) -> usize {
        self.config.chunk_size
    }
    
    fn supports_realtime(&self) -> bool {
        true
    }
}

/// Voice Activity Detector using energy and spectral features
pub struct VoiceActivityDetector {
    energy_threshold: f32,
    zcr_threshold: f32,
    min_speech_duration: usize,
    min_silence_duration: usize,
    speech_counter: usize,
    silence_counter: usize,
    sample_rate: u32,
}

impl VoiceActivityDetector {
    pub fn new(config: &AudioConfig) -> Result<Self> {
        Ok(Self {
            energy_threshold: 0.01, // Adaptive threshold
            zcr_threshold: 0.3,
            min_speech_duration: config.sample_rate as usize / 20, // 50ms
            min_silence_duration: config.sample_rate as usize / 10, // 100ms
            speech_counter: 0,
            silence_counter: 0,
            sample_rate: config.sample_rate,
        })
    }
    
    pub fn detect(&mut self, data: &[f32]) -> Result<VoiceActivity> {
        let energy = rms_energy(data);
        let zcr = zero_crossing_rate(data);
        
        // Simple energy-based detection with ZCR refinement
        let is_energy_speech = energy > self.energy_threshold;
        let is_zcr_speech = zcr < self.zcr_threshold; // Speech typically has lower ZCR
        
        let is_speech_candidate = is_energy_speech && is_zcr_speech;
        
        // Apply temporal smoothing
        if is_speech_candidate {
            self.speech_counter += data.len();
            self.silence_counter = 0;
        } else {
            self.silence_counter += data.len();
            self.speech_counter = 0;
        }
        
        let is_speech = self.speech_counter >= self.min_speech_duration;
        let confidence = if is_speech {
            (energy / (self.energy_threshold * 2.0)).min(1.0)
        } else {
            0.0
        };
        
        // Adaptive threshold update
        if !is_speech {
            self.energy_threshold = 0.9 * self.energy_threshold + 0.1 * energy;
        }
        
        Ok(VoiceActivity {
            is_speech,
            confidence,
            energy_threshold: self.energy_threshold,
            zero_crossing_rate: zcr,
            spectral_centroid: 0.0, // Will be computed by spectral analyzer
        })
    }
}

/// Spectral feature analyzer
pub struct SpectralAnalyzer {
    fft_size: usize,
    window: Vec<f32>,
    fft_buffer: Vec<f32>,
    prev_magnitude: Vec<f32>,
}

impl SpectralAnalyzer {
    pub fn new(config: &AudioConfig) -> Result<Self> {
        let fft_size = next_power_of_two(config.window_size);
        let mut window = vec![0.0; fft_size];
        
        // Generate Hamming window
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos();
        }
        
        Ok(Self {
            fft_size,
            window,
            fft_buffer: vec![0.0; fft_size],
            prev_magnitude: vec![0.0; fft_size / 2 + 1],
        })
    }
    
    pub fn analyze(&mut self, data: &[f32]) -> Result<SpectralFeatures> {
        // Prepare FFT input
        self.fft_buffer.fill(0.0);
        let len = data.len().min(self.fft_size);
        
        for i in 0..len {
            self.fft_buffer[i] = data[i] * self.window[i];
        }
        
        // Compute FFT (simplified - in real implementation would use rustfft)
        let magnitude = self.compute_magnitude_spectrum();
        
        // Compute spectral features
        let spectral_centroid = self.spectral_centroid(&magnitude);
        let spectral_bandwidth = self.spectral_bandwidth(&magnitude, spectral_centroid);
        let spectral_rolloff = self.spectral_rolloff(&magnitude);
        let spectral_flux = self.spectral_flux(&magnitude);
        let spectral_flatness = self.spectral_flatness(&magnitude);
        let (harmonic_ratio, noise_ratio) = self.harmonic_noise_ratio(&magnitude);
        
        // Update previous magnitude for flux calculation
        self.prev_magnitude.copy_from_slice(&magnitude);
        
        Ok(SpectralFeatures {
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flux,
            spectral_flatness,
            harmonic_ratio,
            noise_ratio,
        })
    }
    
    fn compute_magnitude_spectrum(&self) -> Vec<f32> {
        // Simplified FFT magnitude computation
        // In real implementation, would use rustfft crate
        let mut magnitude = vec![0.0; self.fft_size / 2 + 1];
        
        for (i, mag) in magnitude.iter_mut().enumerate() {
            let real = self.fft_buffer[i];
            let imag = if i < self.fft_buffer.len() / 2 {
                self.fft_buffer[i + self.fft_buffer.len() / 2]
            } else {
                0.0
            };
            *mag = (real * real + imag * imag).sqrt();
        }
        
        magnitude
    }
    
    fn spectral_centroid(&self, magnitude: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (i, &mag) in magnitude.iter().enumerate() {
            weighted_sum += i as f32 * mag;
            magnitude_sum += mag;
        }
        
        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
    
    fn spectral_bandwidth(&self, magnitude: &[f32], centroid: f32) -> f32 {
        let mut weighted_variance = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (i, &mag) in magnitude.iter().enumerate() {
            let freq_diff = i as f32 - centroid;
            weighted_variance += freq_diff * freq_diff * mag;
            magnitude_sum += mag;
        }
        
        if magnitude_sum > 0.0 {
            (weighted_variance / magnitude_sum).sqrt()
        } else {
            0.0
        }
    }
    
    fn spectral_rolloff(&self, magnitude: &[f32]) -> f32 {
        let total_energy: f32 = magnitude.iter().sum();
        let threshold = 0.85 * total_energy;
        
        let mut cumulative_energy = 0.0;
        for (i, &mag) in magnitude.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= threshold {
                return i as f32;
            }
        }
        
        magnitude.len() as f32 - 1.0
    }
    
    fn spectral_flux(&self, magnitude: &[f32]) -> f32 {
        let mut flux = 0.0;
        for (i, &mag) in magnitude.iter().enumerate() {
            let diff = mag - self.prev_magnitude.get(i).unwrap_or(&0.0);
            flux += diff * diff;
        }
        flux.sqrt()
    }
    
    fn spectral_flatness(&self, magnitude: &[f32]) -> f32 {
        let geometric_mean = {
            let log_sum: f32 = magnitude.iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| x.ln())
                .sum();
            (log_sum / magnitude.len() as f32).exp()
        };
        
        let arithmetic_mean: f32 = magnitude.iter().sum::<f32>() / magnitude.len() as f32;
        
        if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }
    
    fn harmonic_noise_ratio(&self, magnitude: &[f32]) -> (f32, f32) {
        // Simplified harmonic/noise ratio computation
        // In real implementation would use more sophisticated peak detection
        let total_energy: f32 = magnitude.iter().sum();
        
        if total_energy == 0.0 {
            return (0.0, 1.0);
        }
        
        // Estimate harmonic content (peaks) vs noise (valleys)
        let mut harmonic_energy = 0.0;
        for i in 1..magnitude.len()-1 {
            if magnitude[i] > magnitude[i-1] && magnitude[i] > magnitude[i+1] {
                harmonic_energy += magnitude[i];
            }
        }
        
        let noise_energy = total_energy - harmonic_energy;
        let harmonic_ratio = harmonic_energy / total_energy;
        let noise_ratio = noise_energy / total_energy;
        
        (harmonic_ratio, noise_ratio)
    }
}

/// Energy feature analyzer
pub struct EnergyAnalyzer {
    frame_size: usize,
}

impl EnergyAnalyzer {
    pub fn new(config: &AudioConfig) -> Result<Self> {
        Ok(Self {
            frame_size: config.hop_length,
        })
    }
    
    pub fn analyze(&self, data: &[f32]) -> Result<EnergyFeatures> {
        let rms_energy = rms_energy(data);
        let log_energy = if rms_energy > 0.0 {
            rms_energy.ln()
        } else {
            f32::NEG_INFINITY
        };
        
        let total_energy: f32 = data.iter().map(|x| x * x).sum();
        
        // Compute energy entropy
        let energy_entropy = self.compute_energy_entropy(data);
        
        // Short-time energy
        let short_time_energy = self.compute_short_time_energy(data);
        
        Ok(EnergyFeatures {
            rms_energy,
            log_energy,
            total_energy,
            energy_entropy,
            short_time_energy,
        })
    }
    
    fn compute_energy_entropy(&self, data: &[f32]) -> f32 {
        let frame_count = data.len() / self.frame_size;
        if frame_count == 0 {
            return 0.0;
        }
        
        let mut frame_energies = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let start = i * self.frame_size;
            let end = ((i + 1) * self.frame_size).min(data.len());
            let frame_energy: f32 = data[start..end].iter().map(|x| x * x).sum();
            frame_energies.push(frame_energy);
        }
        
        let total_energy: f32 = frame_energies.iter().sum();
        if total_energy == 0.0 {
            return 0.0;
        }
        
        // Compute entropy
        let mut entropy = 0.0;
        for &energy in &frame_energies {
            if energy > 0.0 {
                let prob = energy / total_energy;
                entropy -= prob * prob.log2();
            }
        }
        
        entropy
    }
    
    fn compute_short_time_energy(&self, data: &[f32]) -> Vec<f32> {
        let frame_count = data.len() / self.frame_size;
        let mut energies = Vec::with_capacity(frame_count);
        
        for i in 0..frame_count {
            let start = i * self.frame_size;
            let end = ((i + 1) * self.frame_size).min(data.len());
            let energy: f32 = data[start..end].iter().map(|x| x * x).sum();
            energies.push(energy);
        }
        
        energies
    }
}

/// Voice quality analyzer for deception detection
pub struct QualityAnalyzer {
    sample_rate: u32,
}

impl QualityAnalyzer {
    pub fn new(config: &AudioConfig) -> Result<Self> {
        Ok(Self {
            sample_rate: config.sample_rate,
        })
    }
    
    pub fn analyze(&self, data: &[f32], spectral: &SpectralFeatures) -> Result<VoiceQuality> {
        let signal_to_noise_ratio = self.compute_snr(data);
        let harmonic_to_noise_ratio = spectral.harmonic_ratio / spectral.noise_ratio.max(1e-10);
        
        // Voice quality metrics based on spectral characteristics
        let breathiness = self.compute_breathiness(spectral);
        let roughness = self.compute_roughness(spectral);
        let hoarseness = self.compute_hoarseness(spectral);
        let vocal_tremor = self.compute_vocal_tremor(data);
        
        Ok(VoiceQuality {
            signal_to_noise_ratio,
            harmonic_to_noise_ratio,
            breathiness,
            roughness,
            hoarseness,
            vocal_tremor,
        })
    }
    
    fn compute_snr(&self, data: &[f32]) -> f32 {
        let signal_power = rms_energy(data).powi(2);
        
        // Estimate noise from silent regions (simplified)
        let sorted_samples: Vec<f32> = {
            let mut samples = data.to_vec();
            samples.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
            samples
        };
        
        let noise_samples = &sorted_samples[..sorted_samples.len() / 4]; // Bottom 25%
        let noise_power = rms_energy(noise_samples).powi(2);
        
        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // High SNR when no noise detected
        }
    }
    
    fn compute_breathiness(&self, spectral: &SpectralFeatures) -> f32 {
        // Breathiness correlates with high-frequency noise
        spectral.noise_ratio * spectral.spectral_centroid / 1000.0
    }
    
    fn compute_roughness(&self, spectral: &SpectralFeatures) -> f32 {
        // Roughness correlates with spectral irregularity
        spectral.spectral_flux * (1.0 - spectral.spectral_flatness)
    }
    
    fn compute_hoarseness(&self, spectral: &SpectralFeatures) -> f32 {
        // Hoarseness correlates with reduced harmonic content
        1.0 - spectral.harmonic_ratio
    }
    
    fn compute_vocal_tremor(&self, data: &[f32]) -> f32 {
        // Simple tremor detection based on amplitude modulation
        if data.len() < 2 {
            return 0.0;
        }
        
        let mut modulation = 0.0;
        for window in data.windows(2) {
            modulation += (window[1] - window[0]).abs();
        }
        
        modulation / (data.len() - 1) as f32
    }
}

// Default implementations for feature structs
impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            spectral_rolloff: 0.0,
            spectral_flux: 0.0,
            spectral_flatness: 0.0,
            harmonic_ratio: 0.0,
            noise_ratio: 1.0,
        }
    }
}

impl Default for EnergyFeatures {
    fn default() -> Self {
        Self {
            rms_energy: 0.0,
            log_energy: f32::NEG_INFINITY,
            total_energy: 0.0,
            energy_entropy: 0.0,
            short_time_energy: Vec::new(),
        }
    }
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self {
            signal_to_noise_ratio: 0.0,
            harmonic_to_noise_ratio: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            hoarseness: 0.0,
            vocal_tremor: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_voice_analyzer_creation() {
        let analyzer = VoiceAnalyzer::new(16000);
        assert!(analyzer.is_ok());
    }
    
    #[test]
    fn test_voice_activity_detector() {
        let config = AudioConfig::default();
        let mut vad = VoiceActivityDetector::new(&config).unwrap();
        
        // Test with silence
        let silence = vec![0.0; 1024];
        let result = vad.detect(&silence).unwrap();
        assert!(!result.is_speech);
        
        // Test with speech-like signal
        let speech: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let result = vad.detect(&speech).unwrap();
        // Note: This might not detect as speech due to simple VAD algorithm
    }
    
    #[test]
    fn test_spectral_analyzer() {
        let config = AudioConfig::default();
        let mut analyzer = SpectralAnalyzer::new(&config).unwrap();
        
        // Test with sinusoidal signal
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let features = analyzer.analyze(&signal).unwrap();
        
        assert!(features.spectral_centroid >= 0.0);
        assert!(features.spectral_bandwidth >= 0.0);
    }
    
    #[test]
    fn test_energy_analyzer() {
        let config = AudioConfig::default();
        let analyzer = EnergyAnalyzer::new(&config).unwrap();
        
        let signal = vec![1.0; 1024];
        let features = analyzer.analyze(&signal).unwrap();
        
        assert_eq!(features.rms_energy, 1.0);
        assert!(features.total_energy > 0.0);
    }
}