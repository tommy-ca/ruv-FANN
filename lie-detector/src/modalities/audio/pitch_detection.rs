//! Pitch Detection Module with SIMD Optimization
//!
//! This module implements high-performance pitch detection algorithms including
//! autocorrelation, YIN algorithm, and harmonic analysis for lie detection.

use std::cmp::Ordering;
use std::collections::VecDeque;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::*;
use super::utils::*;

/// Pitch detection features for voice analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchFeatures {
    /// Fundamental frequency in Hz
    pub fundamental_frequency: f32,
    
    /// Pitch confidence (0.0 to 1.0)
    pub confidence: f32,
    
    /// Pitch stability over time
    pub stability: f32,
    
    /// Voicing probability
    pub voicing_probability: f32,
    
    /// Harmonic structure analysis
    pub harmonics: Vec<Harmonic>,
    
    /// Pitch range metrics
    pub pitch_range: PitchRange,
    
    /// Intonation features
    pub intonation: IntonationFeatures,
}

/// Individual harmonic component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Harmonic {
    pub frequency: f32,
    pub amplitude: f32,
    pub phase: f32,
}

/// Pitch range characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchRange {
    pub min_frequency: f32,
    pub max_frequency: f32,
    pub mean_frequency: f32,
    pub frequency_variance: f32,
}

/// Intonation and prosodic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationFeatures {
    pub pitch_slope: f32,
    pub pitch_contour_entropy: f32,
    pub stress_pattern: Vec<f32>,
    pub rhythm_regularity: f32,
}

/// High-performance pitch detector with multiple algorithms
pub struct PitchDetector {
    sample_rate: u32,
    min_frequency: f32,
    max_frequency: f32,
    autocorr_buffer: Vec<f32>,
    yin_buffer: Vec<f32>,
    window_size: usize,
    hop_length: usize,
    history: PitchHistory,
}

impl PitchDetector {
    /// Create new pitch detector with configuration
    pub fn new(config: &AudioConfig) -> Result<Self> {
        let min_frequency = config.min_frequency;
        let max_frequency = config.max_frequency.min(config.sample_rate as f32 / 2.0);
        
        if min_frequency >= max_frequency {
            return Err(AudioError::PitchDetectionError(
                "Invalid frequency range".to_string()
            ).into());
        }
        
        let window_size = config.window_size;
        let hop_length = config.hop_length;
        
        Ok(Self {
            sample_rate: config.sample_rate,
            min_frequency,
            max_frequency,
            autocorr_buffer: vec![0.0; window_size],
            yin_buffer: vec![0.0; window_size / 2],
            window_size,
            hop_length,
            history: PitchHistory::new(100), // Keep 100 frames of history
        })
    }
    
    /// Detect pitch using ensemble of algorithms
    pub fn detect_pitch(&mut self, data: &[f32]) -> Result<PitchFeatures> {
        if data.len() < self.window_size {
            return Err(AudioError::InsufficientData {
                got: data.len(),
                required: self.window_size,
            }.into());
        }
        
        // Autocorrelation-based detection
        let autocorr_result = self.autocorrelation_pitch(data)?;
        
        // YIN algorithm for better accuracy
        let yin_result = self.yin_pitch(data)?;
        
        // Harmonic product spectrum
        let hps_result = self.harmonic_product_spectrum(data)?;
        
        // Combine results with weighted voting
        let fundamental_frequency = self.combine_pitch_estimates(
            autocorr_result.frequency,
            yin_result.frequency,
            hps_result.frequency,
            autocorr_result.confidence,
            yin_result.confidence,
            hps_result.confidence,
        );
        
        // Compute overall confidence
        let confidence = (autocorr_result.confidence + 
                         yin_result.confidence + 
                         hps_result.confidence) / 3.0;
        
        // Analyze harmonics
        let harmonics = self.analyze_harmonics(data, fundamental_frequency)?;
        
        // Update history and compute stability
        self.history.add_pitch(fundamental_frequency, confidence);
        let stability = self.history.compute_stability();
        
        // Compute voicing probability
        let voicing_probability = self.compute_voicing_probability(data, fundamental_frequency)?;
        
        // Analyze pitch range and intonation
        let pitch_range = self.history.compute_pitch_range();
        let intonation = self.analyze_intonation()?;
        
        Ok(PitchFeatures {
            fundamental_frequency,
            confidence,
            stability,
            voicing_probability,
            harmonics,
            pitch_range,
            intonation,
        })
    }
    
    /// Autocorrelation-based pitch detection with SIMD optimization
    fn autocorrelation_pitch(&mut self, data: &[f32]) -> Result<PitchResult> {
        let max_lag = (self.sample_rate as f32 / self.min_frequency) as usize;
        let min_lag = (self.sample_rate as f32 / self.max_frequency) as usize;
        
        // Compute autocorrelation with SIMD
        self.autocorr_buffer.fill(0.0);
        
        #[cfg(feature = "simd-avx2")]
        {
            self.autocorrelation_simd_avx2(data, max_lag)?;
        }
        
        #[cfg(not(feature = "simd-avx2"))]
        {
            self.autocorrelation_scalar(data, max_lag)?;
        }
        
        // Find peak in valid range
        let mut max_value = 0.0;
        let mut best_lag = min_lag;
        
        for lag in min_lag..max_lag.min(self.autocorr_buffer.len()) {
            if self.autocorr_buffer[lag] > max_value {
                max_value = self.autocorr_buffer[lag];
                best_lag = lag;
            }
        }
        
        // Parabolic interpolation for sub-sample accuracy
        let frequency = if max_value > 0.0 {
            let interpolated_lag = self.parabolic_interpolation(
                &self.autocorr_buffer,
                best_lag,
            );
            self.sample_rate as f32 / interpolated_lag
        } else {
            0.0
        };
        
        let confidence = max_value / self.autocorr_buffer[0].max(1e-10);
        
        Ok(PitchResult { frequency, confidence })
    }
    
    /// YIN algorithm implementation
    fn yin_pitch(&mut self, data: &[f32]) -> Result<PitchResult> {
        let max_lag = self.yin_buffer.len();
        
        // Compute difference function
        for lag in 0..max_lag {
            let mut sum = 0.0;
            for i in 0..data.len().saturating_sub(lag) {
                let diff = data[i] - data[i + lag];
                sum += diff * diff;
            }
            self.yin_buffer[lag] = sum;
        }
        
        // Compute cumulative mean normalized difference
        let mut cumulative_sum = 0.0;
        for lag in 1..max_lag {
            cumulative_sum += self.yin_buffer[lag];
            self.yin_buffer[lag] *= lag as f32 / cumulative_sum;
        }
        
        // Find minimum below threshold in valid range
        let min_lag = (self.sample_rate as f32 / self.max_frequency) as usize;
        let max_search_lag = (self.sample_rate as f32 / self.min_frequency) as usize;
        let threshold = 0.1;
        
        let mut best_lag = min_lag;
        let mut min_value = f32::INFINITY;
        
        for lag in min_lag..max_search_lag.min(max_lag) {
            if self.yin_buffer[lag] < threshold && self.yin_buffer[lag] < min_value {
                min_value = self.yin_buffer[lag];
                best_lag = lag;
            }
        }
        
        let frequency = if min_value < threshold {
            let interpolated_lag = self.parabolic_interpolation_min(
                &self.yin_buffer,
                best_lag,
            );
            self.sample_rate as f32 / interpolated_lag
        } else {
            0.0
        };
        
        let confidence = if min_value < threshold {
            1.0 - min_value
        } else {
            0.0
        };
        
        Ok(PitchResult { frequency, confidence })
    }
    
    /// Harmonic Product Spectrum method
    fn harmonic_product_spectrum(&self, data: &[f32]) -> Result<PitchResult> {
        // Simplified HPS implementation
        // In real implementation would use FFT
        let max_lag = (self.sample_rate as f32 / self.min_frequency) as usize;
        let min_lag = (self.sample_rate as f32 / self.max_frequency) as usize;
        
        let mut hps_scores = vec![0.0; max_lag];
        
        // Compute autocorrelation as proxy for spectrum
        for lag in min_lag..max_lag {
            let mut sum = 0.0;
            for i in 0..data.len().saturating_sub(lag) {
                sum += data[i] * data[i + lag];
            }
            
            // Multiply harmonics (simplified)
            let mut product = sum.abs();
            for harmonic in 2..=5 {
                let harmonic_lag = lag * harmonic;
                if harmonic_lag < data.len() {
                    let mut harmonic_sum = 0.0;
                    for i in 0..data.len().saturating_sub(harmonic_lag) {
                        harmonic_sum += data[i] * data[i + harmonic_lag];
                    }
                    product *= harmonic_sum.abs().sqrt();
                }
            }
            
            hps_scores[lag] = product;
        }
        
        // Find maximum
        let mut max_value = 0.0;
        let mut best_lag = min_lag;
        
        for lag in min_lag..max_lag {
            if hps_scores[lag] > max_value {
                max_value = hps_scores[lag];
                best_lag = lag;
            }
        }
        
        let frequency = if max_value > 0.0 {
            self.sample_rate as f32 / best_lag as f32
        } else {
            0.0
        };
        
        let confidence = if max_value > 0.0 { 0.8 } else { 0.0 };
        
        Ok(PitchResult { frequency, confidence })
    }
    
    /// SIMD-optimized autocorrelation for AVX2
    #[cfg(feature = "simd-avx2")]
    fn autocorrelation_simd_avx2(&mut self, data: &[f32], max_lag: usize) -> Result<()> {
        use std::arch::x86_64::*;
        
        unsafe {
            for lag in 0..max_lag.min(self.autocorr_buffer.len()) {
                let mut sum = _mm256_setzero_ps();
                let len = data.len().saturating_sub(lag);
                let chunks = len / 8;
                
                for i in 0..chunks {
                    let idx = i * 8;
                    let a = _mm256_loadu_ps(data.as_ptr().add(idx));
                    let b = _mm256_loadu_ps(data.as_ptr().add(idx + lag));
                    sum = _mm256_fmadd_ps(a, b, sum);
                }
                
                // Horizontal sum
                let sum_scalar = self.hsum_ps_avx2(sum);
                
                // Handle remainder
                let remainder_start = chunks * 8;
                let mut remainder_sum = 0.0f32;
                for i in remainder_start..len {
                    remainder_sum += data[i] * data[i + lag];
                }
                
                self.autocorr_buffer[lag] = sum_scalar + remainder_sum;
            }
        }
        
        Ok(())
    }
    
    #[cfg(feature = "simd-avx2")]
    unsafe fn hsum_ps_avx2(&self, v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        
        let hiQuad = _mm256_extractf128_ps(v, 1);
        let loQuad = _mm256_castps256_ps128(v);
        let sumQuad = _mm_add_ps(loQuad, hiQuad);
        let loDual = sumQuad;
        let hiDual = _mm_movehl_ps(sumQuad, sumQuad);
        let sumDual = _mm_add_ps(loDual, hiDual);
        let lo = sumDual;
        let hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
        let sum = _mm_add_ss(lo, hi);
        _mm_cvtss_f32(sum)
    }
    
    /// Scalar autocorrelation fallback
    fn autocorrelation_scalar(&mut self, data: &[f32], max_lag: usize) -> Result<()> {
        for lag in 0..max_lag.min(self.autocorr_buffer.len()) {
            let mut sum = 0.0;
            for i in 0..data.len().saturating_sub(lag) {
                sum += data[i] * data[i + lag];
            }
            self.autocorr_buffer[lag] = sum;
        }
        Ok(())
    }
    
    /// Parabolic interpolation for sub-sample accuracy
    fn parabolic_interpolation(&self, data: &[f32], peak_idx: usize) -> f32 {
        if peak_idx == 0 || peak_idx >= data.len() - 1 {
            return peak_idx as f32;
        }
        
        let y1 = data[peak_idx - 1];
        let y2 = data[peak_idx];
        let y3 = data[peak_idx + 1];
        
        let a = (y1 - 2.0 * y2 + y3) / 2.0;
        if a.abs() < 1e-10 {
            return peak_idx as f32;
        }
        
        let x0 = -(y3 - y1) / (2.0 * a);
        peak_idx as f32 + x0
    }
    
    /// Parabolic interpolation for minimum finding
    fn parabolic_interpolation_min(&self, data: &[f32], min_idx: usize) -> f32 {
        if min_idx == 0 || min_idx >= data.len() - 1 {
            return min_idx as f32;
        }
        
        let y1 = data[min_idx - 1];
        let y2 = data[min_idx];
        let y3 = data[min_idx + 1];
        
        let a = (y1 - 2.0 * y2 + y3) / 2.0;
        if a.abs() < 1e-10 {
            return min_idx as f32;
        }
        
        let x0 = (y1 - y3) / (2.0 * a);
        min_idx as f32 + x0
    }
    
    /// Combine multiple pitch estimates
    fn combine_pitch_estimates(
        &self,
        f1: f32, f2: f32, f3: f32,
        c1: f32, c2: f32, c3: f32,
    ) -> f32 {
        let total_confidence = c1 + c2 + c3;
        if total_confidence == 0.0 {
            return 0.0;
        }
        
        (f1 * c1 + f2 * c2 + f3 * c3) / total_confidence
    }
    
    /// Analyze harmonic structure
    fn analyze_harmonics(&self, data: &[f32], fundamental: f32) -> Result<Vec<Harmonic>> {
        if fundamental <= 0.0 {
            return Ok(Vec::new());
        }
        
        let mut harmonics = Vec::new();
        
        // Detect first 5 harmonics
        for harmonic_num in 1..=5 {
            let target_freq = fundamental * harmonic_num as f32;
            if target_freq > self.sample_rate as f32 / 2.0 {
                break;
            }
            
            // Simple amplitude estimation using autocorrelation
            let lag = (self.sample_rate as f32 / target_freq) as usize;
            if lag < data.len() {
                let mut amplitude = 0.0;
                for i in 0..data.len().saturating_sub(lag) {
                    amplitude += data[i] * data[i + lag];
                }
                amplitude = amplitude.abs() / (data.len() - lag) as f32;
                
                harmonics.push(Harmonic {
                    frequency: target_freq,
                    amplitude,
                    phase: 0.0, // Phase computation would require FFT
                });
            }
        }
        
        Ok(harmonics)
    }
    
    /// Compute voicing probability
    fn compute_voicing_probability(&self, data: &[f32], fundamental: f32) -> Result<f32> {
        if fundamental <= 0.0 {
            return Ok(0.0);
        }
        
        // Compute harmonic-to-noise ratio as voicing indicator
        let period_samples = (self.sample_rate as f32 / fundamental) as usize;
        if period_samples >= data.len() {
            return Ok(0.0);
        }
        
        let periods = data.len() / period_samples;
        if periods < 2 {
            return Ok(0.0);
        }
        
        // Measure periodicity
        let mut correlation_sum = 0.0;
        let mut total_energy = 0.0;
        
        for period in 0..periods - 1 {
            let start1 = period * period_samples;
            let start2 = (period + 1) * period_samples;
            let len = period_samples.min(data.len() - start2);
            
            let mut correlation = 0.0;
            let mut energy1 = 0.0;
            let mut energy2 = 0.0;
            
            for i in 0..len {
                let s1 = data[start1 + i];
                let s2 = data[start2 + i];
                correlation += s1 * s2;
                energy1 += s1 * s1;
                energy2 += s2 * s2;
            }
            
            let norm = (energy1 * energy2).sqrt();
            if norm > 0.0 {
                correlation_sum += correlation / norm;
                total_energy += 1.0;
            }
        }
        
        Ok(if total_energy > 0.0 {
            (correlation_sum / total_energy).max(0.0).min(1.0)
        } else {
            0.0
        })
    }
    
    /// Analyze intonation patterns
    fn analyze_intonation(&self) -> Result<IntonationFeatures> {
        let pitch_history = self.history.get_recent_pitches(50); // Last 50 frames
        
        if pitch_history.len() < 2 {
            return Ok(IntonationFeatures {
                pitch_slope: 0.0,
                pitch_contour_entropy: 0.0,
                stress_pattern: Vec::new(),
                rhythm_regularity: 0.0,
            });
        }
        
        // Compute pitch slope using linear regression
        let pitch_slope = self.compute_pitch_slope(&pitch_history);
        
        // Compute contour entropy
        let pitch_contour_entropy = self.compute_contour_entropy(&pitch_history);
        
        // Detect stress patterns (simplified)
        let stress_pattern = self.detect_stress_pattern(&pitch_history);
        
        // Measure rhythm regularity
        let rhythm_regularity = self.compute_rhythm_regularity(&pitch_history);
        
        Ok(IntonationFeatures {
            pitch_slope,
            pitch_contour_entropy,
            stress_pattern,
            rhythm_regularity,
        })
    }
    
    fn compute_pitch_slope(&self, pitches: &[f32]) -> f32 {
        if pitches.len() < 2 {
            return 0.0;
        }
        
        let n = pitches.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = pitches.iter().sum::<f32>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &y) in pitches.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean) * (x - x_mean);
        }
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn compute_contour_entropy(&self, pitches: &[f32]) -> f32 {
        if pitches.len() < 2 {
            return 0.0;
        }
        
        // Compute pitch differences
        let differences: Vec<f32> = pitches.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        // Quantize differences into bins
        let bins = 10;
        let mut histogram = vec![0; bins];
        
        if let (Some(&min_diff), Some(&max_diff)) = (
            differences.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            differences.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
        ) {
            let range = max_diff - min_diff;
            if range > 0.0 {
                for &diff in &differences {
                    let bin = ((diff - min_diff) / range * (bins - 1) as f32) as usize;
                    histogram[bin.min(bins - 1)] += 1;
                }
            }
        }
        
        // Compute entropy
        let total = differences.len() as f32;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0 {
                let prob = count as f32 / total;
                entropy -= prob * prob.log2();
            }
        }
        
        entropy
    }
    
    fn detect_stress_pattern(&self, _pitches: &[f32]) -> Vec<f32> {
        // Simplified stress detection - would need more sophisticated analysis
        Vec::new()
    }
    
    fn compute_rhythm_regularity(&self, pitches: &[f32]) -> f32 {
        if pitches.len() < 3 {
            return 0.0;
        }
        
        // Measure regularity of pitch changes
        let differences: Vec<f32> = pitches.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        
        let mean_diff = differences.iter().sum::<f32>() / differences.len() as f32;
        let variance = differences.iter()
            .map(|&x| (x - mean_diff).powi(2))
            .sum::<f32>() / differences.len() as f32;
        
        // Regularity is inverse of coefficient of variation
        if mean_diff > 0.0 {
            1.0 / (1.0 + variance.sqrt() / mean_diff)
        } else {
            1.0
        }
    }
}

/// Internal pitch detection result
struct PitchResult {
    frequency: f32,
    confidence: f32,
}

/// Pitch history for temporal analysis
struct PitchHistory {
    pitches: VecDeque<f32>,
    confidences: VecDeque<f32>,
    max_history: usize,
}

impl PitchHistory {
    fn new(max_history: usize) -> Self {
        Self {
            pitches: VecDeque::with_capacity(max_history),
            confidences: VecDeque::with_capacity(max_history),
            max_history,
        }
    }
    
    fn add_pitch(&mut self, pitch: f32, confidence: f32) {
        if self.pitches.len() >= self.max_history {
            self.pitches.pop_front();
            self.confidences.pop_front();
        }
        
        self.pitches.push_back(pitch);
        self.confidences.push_back(confidence);
    }
    
    fn compute_stability(&self) -> f32 {
        if self.pitches.len() < 2 {
            return 0.0;
        }
        
        // Compute coefficient of variation
        let valid_pitches: Vec<f32> = self.pitches.iter()
            .zip(self.confidences.iter())
            .filter(|(_, &conf)| conf > 0.5)
            .map(|(&pitch, _)| pitch)
            .filter(|&p| p > 0.0)
            .collect();
        
        if valid_pitches.len() < 2 {
            return 0.0;
        }
        
        let mean = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;
        let variance = valid_pitches.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / valid_pitches.len() as f32;
        
        let cv = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            1.0
        };
        
        // Stability is inverse of coefficient of variation
        1.0 / (1.0 + cv)
    }
    
    fn compute_pitch_range(&self) -> PitchRange {
        let valid_pitches: Vec<f32> = self.pitches.iter()
            .zip(self.confidences.iter())
            .filter(|(_, &conf)| conf > 0.5)
            .map(|(&pitch, _)| pitch)
            .filter(|&p| p > 0.0)
            .collect();
        
        if valid_pitches.is_empty() {
            return PitchRange {
                min_frequency: 0.0,
                max_frequency: 0.0,
                mean_frequency: 0.0,
                frequency_variance: 0.0,
            };
        }
        
        let min_frequency = valid_pitches.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        
        let max_frequency = valid_pitches.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        
        let mean_frequency = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;
        
        let frequency_variance = valid_pitches.iter()
            .map(|&x| (x - mean_frequency).powi(2))
            .sum::<f32>() / valid_pitches.len() as f32;
        
        PitchRange {
            min_frequency,
            max_frequency,
            mean_frequency,
            frequency_variance,
        }
    }
    
    fn get_recent_pitches(&self, count: usize) -> Vec<f32> {
        self.pitches.iter()
            .rev()
            .take(count)
            .rev()
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pitch_detector_creation() {
        let config = AudioConfig::default();
        let detector = PitchDetector::new(&config);
        assert!(detector.is_ok());
    }
    
    #[test]
    fn test_pitch_detection_sinusoid() {
        let config = AudioConfig::default();
        let mut detector = PitchDetector::new(&config).unwrap();
        
        // Generate 440 Hz sinusoid
        let frequency = 440.0;
        let duration = 0.1; // 100ms
        let sample_rate = config.sample_rate as f32;
        let samples = (duration * sample_rate) as usize;
        
        let signal: Vec<f32> = (0..samples)
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin())
            .collect();
        
        let result = detector.detect_pitch(&signal);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        // Allow some tolerance for frequency detection
        assert!((features.fundamental_frequency - frequency).abs() < 50.0);
    }
    
    #[test]
    fn test_pitch_history() {
        let mut history = PitchHistory::new(5);
        
        // Add some pitches
        for i in 0..3 {
            history.add_pitch(440.0 + i as f32, 0.8);
        }
        
        assert_eq!(history.pitches.len(), 3);
        
        let stability = history.compute_stability();
        assert!(stability > 0.0);
        assert!(stability <= 1.0);
        
        let range = history.compute_pitch_range();
        assert!(range.min_frequency > 0.0);
        assert!(range.max_frequency >= range.min_frequency);
    }
    
    #[test]
    fn test_parabolic_interpolation() {
        let config = AudioConfig::default();
        let detector = PitchDetector::new(&config).unwrap();
        
        let data = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let interpolated = detector.parabolic_interpolation(&data, 2);
        
        // Peak at index 2 should remain at 2.0
        assert!((interpolated - 2.0).abs() < 0.1);
    }
}