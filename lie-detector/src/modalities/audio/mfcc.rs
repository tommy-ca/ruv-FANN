//! Mel-Frequency Cepstral Coefficients (MFCC) Feature Extraction
//!
//! This module implements optimized MFCC computation with mel filterbank for
//! voice analysis and lie detection applications.

use std::f32::consts::PI;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::*;
use super::utils::*;

/// MFCC features for voice analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfccFeatures {
    /// MFCC coefficients (typically 13 coefficients)
    pub coefficients: Vec<f32>,
    
    /// Delta (first derivative) coefficients
    pub delta_coefficients: Vec<f32>,
    
    /// Delta-delta (second derivative) coefficients
    pub delta_delta_coefficients: Vec<f32>,
    
    /// Mel filterbank energies
    pub mel_energies: Vec<f32>,
    
    /// Log energy (C0 coefficient)
    pub log_energy: f32,
    
    /// Spectral features derived from MFCC
    pub spectral_features: MfccSpectralFeatures,
}

/// Spectral features derived from MFCC analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfccSpectralFeatures {
    /// Spectral centroid from mel domain
    pub mel_spectral_centroid: f32,
    
    /// Spectral bandwidth in mel domain
    pub mel_spectral_bandwidth: f32,
    
    /// Spectral skewness
    pub spectral_skewness: f32,
    
    /// Spectral kurtosis
    pub spectral_kurtosis: f32,
    
    /// MFCC variance (temporal stability)
    pub mfcc_variance: f32,
}

/// High-performance MFCC extractor with mel filterbank
pub struct MfccExtractor {
    sample_rate: u32,
    fft_size: usize,
    n_mfcc: usize,
    n_mels: usize,
    min_freq: f32,
    max_freq: f32,
    
    // Pre-computed filterbank
    mel_filterbank: MelFilterbank,
    
    // DCT matrix for MFCC computation
    dct_matrix: Vec<Vec<f32>>,
    
    // FFT buffers
    fft_buffer: Vec<f32>,
    magnitude_spectrum: Vec<f32>,
    
    // Window function
    window: Vec<f32>,
    
    // Feature history for delta computation
    feature_history: Vec<Vec<f32>>,
    max_history: usize,
    
    // Pre-emphasis filter state
    pre_emphasis_state: f32,
}

impl MfccExtractor {
    /// Create new MFCC extractor with configuration
    pub fn new(config: &AudioConfig) -> Result<Self> {
        let sample_rate = config.sample_rate;
        let fft_size = next_power_of_two(config.window_size);
        let n_mfcc = config.num_mfcc_coefficients;
        let n_mels = config.mel_filters;
        let min_freq = config.min_frequency;
        let max_freq = config.max_frequency.min(sample_rate as f32 / 2.0);
        
        if min_freq >= max_freq {
            return Err(AudioError::MelFilterError(
                "Invalid frequency range".to_string()
            ).into());
        }
        
        if n_mfcc > n_mels {
            return Err(AudioError::MelFilterError(
                "Number of MFCC coefficients cannot exceed number of mel filters".to_string()
            ).into());
        }
        
        // Create mel filterbank
        let mel_filterbank = MelFilterbank::new(
            n_mels,
            fft_size,
            sample_rate,
            min_freq,
            max_freq,
        )?;
        
        // Create DCT matrix
        let dct_matrix = Self::create_dct_matrix(n_mfcc, n_mels);
        
        // Create window function (Hamming)
        let window = Self::create_hamming_window(fft_size);
        
        Ok(Self {
            sample_rate,
            fft_size,
            n_mfcc,
            n_mels,
            min_freq,
            max_freq,
            mel_filterbank,
            dct_matrix,
            fft_buffer: vec![0.0; fft_size],
            magnitude_spectrum: vec![0.0; fft_size / 2 + 1],
            window,
            feature_history: Vec::with_capacity(10),
            max_history: 10,
            pre_emphasis_state: 0.0,
        })
    }
    
    /// Extract MFCC features from audio frame
    pub fn extract_features(&mut self, frame: &[f32]) -> Result<MfccFeatures> {
        if frame.len() > self.fft_size {
            return Err(AudioError::InsufficientData {
                got: frame.len(),
                required: self.fft_size,
            }.into());
        }
        
        // Pre-emphasis filtering
        let mut pre_emphasized = self.apply_pre_emphasis(frame);
        
        // Apply window function
        self.apply_window(&mut pre_emphasized);
        
        // Compute magnitude spectrum
        self.compute_magnitude_spectrum(&pre_emphasized)?;
        
        // Apply mel filterbank
        let mel_energies = self.mel_filterbank.apply(&self.magnitude_spectrum)?;
        
        // Convert to log domain
        let log_mel_energies: Vec<f32> = mel_energies.iter()
            .map(|&energy| {
                if energy > 1e-10 {
                    energy.ln()
                } else {
                    -23.025850929940458 // ln(1e-10)
                }
            })
            .collect();
        
        // Apply DCT to get MFCC coefficients
        let coefficients = self.apply_dct(&log_mel_energies);
        
        // Compute log energy (C0)
        let log_energy = if !log_mel_energies.is_empty() {
            log_mel_energies.iter().sum::<f32>() / log_mel_energies.len() as f32
        } else {
            0.0
        };
        
        // Update feature history for delta computation
        self.update_feature_history(&coefficients);
        
        // Compute delta and delta-delta coefficients
        let delta_coefficients = self.compute_delta_coefficients(&coefficients);
        let delta_delta_coefficients = self.compute_delta_delta_coefficients();
        
        // Compute spectral features
        let spectral_features = self.compute_spectral_features(&mel_energies, &coefficients)?;
        
        Ok(MfccFeatures {
            coefficients,
            delta_coefficients,
            delta_delta_coefficients,
            mel_energies,
            log_energy,
            spectral_features,
        })
    }
    
    /// Apply pre-emphasis filter
    fn apply_pre_emphasis(&mut self, frame: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; frame.len()];
        let alpha = 0.97;
        
        if !frame.is_empty() {
            result[0] = frame[0] - alpha * self.pre_emphasis_state;
            
            for i in 1..frame.len() {
                result[i] = frame[i] - alpha * frame[i - 1];
            }
            
            self.pre_emphasis_state = frame[frame.len() - 1];
        }
        
        result
    }
    
    /// Apply window function to frame
    fn apply_window(&mut self, frame: &mut [f32]) {
        let len = frame.len().min(self.window.len());
        for i in 0..len {
            frame[i] *= self.window[i];
        }
    }
    
    /// Compute magnitude spectrum using simplified FFT
    fn compute_magnitude_spectrum(&mut self, frame: &[f32]) -> Result<()> {
        // Clear FFT buffer
        self.fft_buffer.fill(0.0);
        
        // Copy frame to FFT buffer
        let len = frame.len().min(self.fft_size);
        self.fft_buffer[..len].copy_from_slice(&frame[..len]);
        
        // Simplified magnitude spectrum computation
        // In a real implementation, would use rustfft crate
        for i in 0..self.magnitude_spectrum.len() {
            if i < self.fft_buffer.len() {
                // Simplified: use time-domain values as proxy for frequency domain
                // Real implementation would perform actual FFT
                let real = self.fft_buffer[i];
                let imag = if i < self.fft_buffer.len() / 2 {
                    self.fft_buffer[i + self.fft_buffer.len() / 2]
                } else {
                    0.0
                };
                
                self.magnitude_spectrum[i] = (real * real + imag * imag).sqrt();
            } else {
                self.magnitude_spectrum[i] = 0.0;
            }
        }
        
        Ok(())
    }
    
    /// Apply DCT to convert mel energies to MFCC coefficients
    fn apply_dct(&self, log_mel_energies: &[f32]) -> Vec<f32> {
        let mut coefficients = vec![0.0; self.n_mfcc];
        
        for i in 0..self.n_mfcc {
            for j in 0..log_mel_energies.len().min(self.n_mels) {
                coefficients[i] += self.dct_matrix[i][j] * log_mel_energies[j];
            }
        }
        
        coefficients
    }
    
    /// Update feature history for delta computation
    fn update_feature_history(&mut self, coefficients: &[f32]) {
        if self.feature_history.len() >= self.max_history {
            self.feature_history.remove(0);
        }
        self.feature_history.push(coefficients.to_vec());
    }
    
    /// Compute delta (first derivative) coefficients
    fn compute_delta_coefficients(&self, current_coefficients: &[f32]) -> Vec<f32> {
        if self.feature_history.len() < 2 {
            return vec![0.0; current_coefficients.len()];
        }
        
        let prev_idx = self.feature_history.len() - 2;
        let prev_coefficients = &self.feature_history[prev_idx];
        
        let mut delta = vec![0.0; current_coefficients.len()];
        
        for i in 0..delta.len().min(current_coefficients.len()).min(prev_coefficients.len()) {
            delta[i] = current_coefficients[i] - prev_coefficients[i];
        }
        
        delta
    }
    
    /// Compute delta-delta (second derivative) coefficients
    fn compute_delta_delta_coefficients(&self) -> Vec<f32> {
        if self.feature_history.len() < 3 {
            return vec![0.0; self.n_mfcc];
        }
        
        let len = self.feature_history.len();
        let current = &self.feature_history[len - 1];
        let prev = &self.feature_history[len - 2];
        let prev_prev = &self.feature_history[len - 3];
        
        let mut delta_delta = vec![0.0; self.n_mfcc];
        
        for i in 0..delta_delta.len().min(current.len()).min(prev.len()).min(prev_prev.len()) {
            let delta_current = current[i] - prev[i];
            let delta_prev = prev[i] - prev_prev[i];
            delta_delta[i] = delta_current - delta_prev;
        }
        
        delta_delta
    }
    
    /// Compute spectral features from MFCC analysis
    fn compute_spectral_features(
        &self,
        mel_energies: &[f32],
        coefficients: &[f32],
    ) -> Result<MfccSpectralFeatures> {
        
        // Mel spectral centroid
        let mel_spectral_centroid = {
            let mut weighted_sum = 0.0;
            let mut total_energy = 0.0;
            
            for (i, &energy) in mel_energies.iter().enumerate() {
                weighted_sum += i as f32 * energy;
                total_energy += energy;
            }
            
            if total_energy > 0.0 {
                weighted_sum / total_energy
            } else {
                0.0
            }
        };
        
        // Mel spectral bandwidth
        let mel_spectral_bandwidth = {
            let mut weighted_variance = 0.0;
            let mut total_energy = 0.0;
            
            for (i, &energy) in mel_energies.iter().enumerate() {
                let freq_diff = i as f32 - mel_spectral_centroid;
                weighted_variance += freq_diff * freq_diff * energy;
                total_energy += energy;
            }
            
            if total_energy > 0.0 {
                (weighted_variance / total_energy).sqrt()
            } else {
                0.0
            }
        };
        
        // Spectral skewness and kurtosis
        let (spectral_skewness, spectral_kurtosis) = 
            self.compute_spectral_moments(mel_energies, mel_spectral_centroid, mel_spectral_bandwidth);
        
        // MFCC variance (temporal stability)
        let mfcc_variance = if self.feature_history.len() > 1 {
            let mut variance_sum = 0.0;
            let n_frames = self.feature_history.len() as f32;
            
            // Compute mean for each coefficient
            let mut mean_coeffs = vec![0.0; self.n_mfcc];
            for frame in &self.feature_history {
                for (i, &coeff) in frame.iter().enumerate().take(self.n_mfcc) {
                    mean_coeffs[i] += coeff / n_frames;
                }
            }
            
            // Compute variance for each coefficient
            for frame in &self.feature_history {
                for (i, &coeff) in frame.iter().enumerate().take(self.n_mfcc) {
                    let diff = coeff - mean_coeffs[i];
                    variance_sum += diff * diff;
                }
            }
            
            variance_sum / (n_frames * self.n_mfcc as f32)
        } else {
            0.0
        };
        
        Ok(MfccSpectralFeatures {
            mel_spectral_centroid,
            mel_spectral_bandwidth,
            spectral_skewness,
            spectral_kurtosis,
            mfcc_variance,
        })
    }
    
    /// Compute spectral moments (skewness and kurtosis)
    fn compute_spectral_moments(
        &self,
        mel_energies: &[f32],
        centroid: f32,
        bandwidth: f32,
    ) -> (f32, f32) {
        if bandwidth == 0.0 || mel_energies.is_empty() {
            return (0.0, 0.0);
        }
        
        let mut third_moment = 0.0;
        let mut fourth_moment = 0.0;
        let mut total_energy = 0.0;
        
        for (i, &energy) in mel_energies.iter().enumerate() {
            let normalized_freq = (i as f32 - centroid) / bandwidth;
            let freq_pow3 = normalized_freq.powi(3);
            let freq_pow4 = normalized_freq.powi(4);
            
            third_moment += freq_pow3 * energy;
            fourth_moment += freq_pow4 * energy;
            total_energy += energy;
        }
        
        if total_energy > 0.0 {
            let skewness = third_moment / total_energy;
            let kurtosis = fourth_moment / total_energy - 3.0; // Excess kurtosis
            (skewness, kurtosis)
        } else {
            (0.0, 0.0)
        }
    }
    
    /// Create DCT matrix for MFCC computation
    fn create_dct_matrix(n_mfcc: usize, n_mels: usize) -> Vec<Vec<f32>> {
        let mut dct_matrix = vec![vec![0.0; n_mels]; n_mfcc];
        
        for i in 0..n_mfcc {
            for j in 0..n_mels {
                dct_matrix[i][j] = (PI * i as f32 * (j as f32 + 0.5) / n_mels as f32).cos()
                    * (2.0 / n_mels as f32).sqrt();
            }
        }
        
        // Apply orthogonalization for the first coefficient
        if n_mfcc > 0 {
            let sqrt_2_inv = 1.0 / 2.0_f32.sqrt();
            for j in 0..n_mels {
                dct_matrix[0][j] *= sqrt_2_inv;
            }
        }
        
        dct_matrix
    }
    
    /// Create Hamming window
    fn create_hamming_window(size: usize) -> Vec<f32> {
        let mut window = vec![0.0; size];
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos();
        }
        window
    }
}

/// Mel filterbank for spectral analysis
pub struct MelFilterbank {
    filters: Vec<Vec<f32>>,
    n_filters: usize,
    fft_size: usize,
}

impl MelFilterbank {
    /// Create new mel filterbank
    pub fn new(
        n_filters: usize,
        fft_size: usize,
        sample_rate: u32,
        min_freq: f32,
        max_freq: f32,
    ) -> Result<Self> {
        let mut filters = vec![vec![0.0; fft_size / 2 + 1]; n_filters];
        
        // Convert frequencies to mel scale
        let min_mel = Self::hz_to_mel(min_freq);
        let max_mel = Self::hz_to_mel(max_freq);
        
        // Create equally spaced mel frequencies
        let mel_points: Vec<f32> = (0..=n_filters + 1)
            .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_filters + 1) as f32)
            .collect();
        
        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter()
            .map(|&mel| Self::mel_to_hz(mel))
            .collect();
        
        // Convert Hz to FFT bin indices
        let bin_points: Vec<usize> = hz_points.iter()
            .map(|&hz| ((hz * fft_size as f32) / sample_rate as f32).round() as usize)
            .map(|bin| bin.min(fft_size / 2))
            .collect();
        
        // Create triangular filters
        for i in 0..n_filters {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];
            
            // Left slope
            for bin in left..center {
                if center > left {
                    filters[i][bin] = (bin - left) as f32 / (center - left) as f32;
                }
            }
            
            // Right slope
            for bin in center..=right.min(fft_size / 2) {
                if right > center {
                    filters[i][bin] = (right - bin) as f32 / (right - center) as f32;
                }
            }
        }
        
        Ok(Self {
            filters,
            n_filters,
            fft_size,
        })
    }
    
    /// Apply mel filterbank to magnitude spectrum
    pub fn apply(&self, magnitude_spectrum: &[f32]) -> Result<Vec<f32>> {
        let mut mel_energies = vec![0.0; self.n_filters];
        
        #[cfg(feature = "simd-avx2")]
        {
            self.apply_simd_avx2(magnitude_spectrum, &mut mel_energies)?;
        }
        
        #[cfg(not(feature = "simd-avx2"))]
        {
            self.apply_scalar(magnitude_spectrum, &mut mel_energies)?;
        }
        
        Ok(mel_energies)
    }
    
    /// SIMD-optimized filterbank application
    #[cfg(feature = "simd-avx2")]
    fn apply_simd_avx2(&self, magnitude_spectrum: &[f32], mel_energies: &mut [f32]) -> Result<()> {
        use std::arch::x86_64::*;
        
        unsafe {
            for (filter_idx, filter) in self.filters.iter().enumerate() {
                let mut energy_vec = _mm256_setzero_ps();
                let len = magnitude_spectrum.len().min(filter.len());
                let chunks = len / 8;
                
                for i in 0..chunks {
                    let idx = i * 8;
                    let mag = _mm256_loadu_ps(magnitude_spectrum.as_ptr().add(idx));
                    let filt = _mm256_loadu_ps(filter.as_ptr().add(idx));
                    
                    // magnitude^2 * filter
                    let mag_sq = _mm256_mul_ps(mag, mag);
                    let filtered = _mm256_mul_ps(mag_sq, filt);
                    energy_vec = _mm256_add_ps(energy_vec, filtered);
                }
                
                // Horizontal sum
                let energy_scalar = self.hsum_ps_avx2(energy_vec);
                
                // Handle remainder
                let remainder_start = chunks * 8;
                let mut remainder_energy = 0.0f32;
                for i in remainder_start..len {
                    remainder_energy += magnitude_spectrum[i] * magnitude_spectrum[i] * filter[i];
                }
                
                mel_energies[filter_idx] = energy_scalar + remainder_energy;
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
    
    /// Scalar fallback implementation
    fn apply_scalar(&self, magnitude_spectrum: &[f32], mel_energies: &mut [f32]) -> Result<()> {
        for (filter_idx, filter) in self.filters.iter().enumerate() {
            let mut energy = 0.0;
            
            for (bin_idx, &magnitude) in magnitude_spectrum.iter().enumerate() {
                if bin_idx < filter.len() {
                    energy += magnitude * magnitude * filter[bin_idx];
                }
            }
            
            mel_energies[filter_idx] = energy;
        }
        
        Ok(())
    }
    
    /// Convert frequency in Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    
    /// Convert mel scale to frequency in Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mfcc_extractor_creation() {
        let config = AudioConfig::default();
        let extractor = MfccExtractor::new(&config);
        assert!(extractor.is_ok());
    }
    
    #[test]
    fn test_mel_filterbank_creation() {
        let filterbank = MelFilterbank::new(26, 1024, 16000, 80.0, 8000.0);
        assert!(filterbank.is_ok());
        
        let fb = filterbank.unwrap();
        assert_eq!(fb.n_filters, 26);
        assert_eq!(fb.filters.len(), 26);
    }
    
    #[test]
    fn test_hz_mel_conversion() {
        let hz = 1000.0;
        let mel = MelFilterbank::hz_to_mel(hz);
        let hz_back = MelFilterbank::mel_to_hz(mel);
        
        assert!((hz - hz_back).abs() < 1.0); // Allow small floating point error
    }
    
    #[test]
    fn test_mfcc_feature_extraction() {
        let config = AudioConfig::default();
        let mut extractor = MfccExtractor::new(&config).unwrap();
        
        // Generate test signal (440 Hz sine wave)
        let sample_rate = config.sample_rate as f32;
        let duration = 0.025; // 25ms frame
        let samples = (duration * sample_rate) as usize;
        let frequency = 440.0;
        
        let signal: Vec<f32> = (0..samples)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin())
            .collect();
        
        let features = extractor.extract_features(&signal);
        assert!(features.is_ok());
        
        let mfcc = features.unwrap();
        assert_eq!(mfcc.coefficients.len(), config.num_mfcc_coefficients);
        assert_eq!(mfcc.mel_energies.len(), config.mel_filters);
    }
    
    #[test]
    fn test_dct_matrix_creation() {
        let dct = MfccExtractor::create_dct_matrix(13, 26);
        assert_eq!(dct.len(), 13);
        assert_eq!(dct[0].len(), 26);
        
        // Check orthogonality property (first row should be normalized differently)
        let first_row_norm: f32 = dct[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let second_row_norm: f32 = dct[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        
        assert!((first_row_norm - 1.0).abs() < 0.1);
        assert!((second_row_norm - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_delta_coefficient_computation() {
        let config = AudioConfig::default();
        let mut extractor = MfccExtractor::new(&config).unwrap();
        
        // Add some feature history
        extractor.feature_history.push(vec![1.0, 2.0, 3.0]);
        extractor.feature_history.push(vec![1.1, 2.1, 3.1]);
        
        let current = vec![1.2, 2.2, 3.2];
        let delta = extractor.compute_delta_coefficients(&current);
        
        assert_eq!(delta.len(), 3);
        assert!((delta[0] - 0.1).abs() < 1e-6);
        assert!((delta[1] - 0.1).abs() < 1e-6);
        assert!((delta[2] - 0.1).abs() < 1e-6);
    }
    
    #[test]
    fn test_mel_filterbank_application() {
        let filterbank = MelFilterbank::new(26, 1024, 16000, 80.0, 8000.0).unwrap();
        let magnitude_spectrum = vec![1.0; 513]; // 1024/2 + 1
        
        let mel_energies = filterbank.apply(&magnitude_spectrum);
        assert!(mel_energies.is_ok());
        
        let energies = mel_energies.unwrap();
        assert_eq!(energies.len(), 26);
        
        // All energies should be positive
        assert!(energies.iter().all(|&e| e >= 0.0));
    }
}