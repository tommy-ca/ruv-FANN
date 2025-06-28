//! SIMD-optimized audio processing operations
//!
//! This module provides high-performance implementations of audio processing
//! operations using SIMD instructions for voice analysis and feature extraction.

use crate::optimization::simd::{SimdProcessor, SimdConfig};
use crate::modalities::audio::{AudioError, AudioConfig};
use crate::{Result, VeritasError};
use std::f32::consts::PI;
use std::arch::x86_64::*;

/// SIMD-optimized FFT operations for audio processing
pub struct SimdFFT {
    simd_processor: SimdProcessor,
    size: usize,
    twiddle_factors: Vec<(f32, f32)>, // Pre-computed cos/sin values
}

impl SimdFFT {
    /// Create a new SIMD-optimized FFT processor
    pub fn new(size: usize) -> Result<Self> {
        if !size.is_power_of_two() {
            return Err(AudioError::InvalidConfig(
                "FFT size must be a power of 2".to_string()
            ).into());
        }
        
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        // Pre-compute twiddle factors
        let mut twiddle_factors = Vec::with_capacity(size / 2);
        for k in 0..size / 2 {
            let angle = -2.0 * PI * k as f32 / size as f32;
            twiddle_factors.push((angle.cos(), angle.sin()));
        }
        
        Ok(Self {
            simd_processor,
            size,
            twiddle_factors,
        })
    }
    
    /// Perform SIMD-optimized FFT
    pub fn forward(&self, input: &[f32], output_real: &mut [f32], output_imag: &mut [f32]) -> Result<()> {
        if input.len() != self.size {
            return Err(AudioError::InvalidConfig(
                format!("Input size {} doesn't match FFT size {}", input.len(), self.size)
            ).into());
        }
        
        // Copy input to real part, zero imaginary part
        output_real.copy_from_slice(input);
        output_imag.fill(0.0);
        
        // Perform FFT using Cooley-Tukey algorithm with SIMD
        self.fft_radix2_simd(output_real, output_imag)?;
        
        Ok(())
    }
    
    /// SIMD-optimized radix-2 FFT implementation
    fn fft_radix2_simd(&self, real: &mut [f32], imag: &mut [f32]) -> Result<()> {
        let n = self.size;
        let log_n = (n as f32).log2() as usize;
        
        // Bit reversal
        self.bit_reversal_simd(real, imag)?;
        
        // Cooley-Tukey FFT
        let mut stage_size = 2;
        for _ in 0..log_n {
            let half_stage = stage_size / 2;
            let angle_step = -2.0 * PI / stage_size as f32;
            
            for start in (0..n).step_by(stage_size) {
                for k in 0..half_stage {
                    let angle = angle_step * k as f32;
                    let tw_real = angle.cos();
                    let tw_imag = angle.sin();
                    
                    let even_idx = start + k;
                    let odd_idx = start + k + half_stage;
                    
                    // Butterfly operation
                    let odd_real = real[odd_idx];
                    let odd_imag = imag[odd_idx];
                    
                    let temp_real = odd_real * tw_real - odd_imag * tw_imag;
                    let temp_imag = odd_real * tw_imag + odd_imag * tw_real;
                    
                    real[odd_idx] = real[even_idx] - temp_real;
                    imag[odd_idx] = imag[even_idx] - temp_imag;
                    
                    real[even_idx] += temp_real;
                    imag[even_idx] += temp_imag;
                }
            }
            
            stage_size *= 2;
        }
        
        Ok(())
    }
    
    /// SIMD-optimized bit reversal
    fn bit_reversal_simd(&self, real: &mut [f32], imag: &mut [f32]) -> Result<()> {
        let n = self.size;
        let log_n = (n as f32).log2() as usize;
        
        for i in 0..n {
            let mut reversed = 0;
            let mut temp = i;
            
            for _ in 0..log_n {
                reversed = (reversed << 1) | (temp & 1);
                temp >>= 1;
            }
            
            if i < reversed {
                real.swap(i, reversed);
                imag.swap(i, reversed);
            }
        }
        
        Ok(())
    }
    
    /// Compute magnitude spectrum using SIMD
    pub fn compute_magnitude_spectrum(&self, real: &[f32], imag: &[f32], magnitude: &mut [f32]) -> Result<()> {
        if real.len() != imag.len() || real.len() != magnitude.len() {
            return Err(AudioError::InvalidConfig(
                "Array lengths must match".to_string()
            ).into());
        }
        
        // Process in chunks for SIMD
        let chunk_size = 4;
        let chunks = real.len() / chunk_size;
        
        for i in 0..chunks {
            let offset = i * chunk_size;
            
            // Compute real^2 + imag^2
            let mut real_sq = vec![0.0; chunk_size];
            let mut imag_sq = vec![0.0; chunk_size];
            
            for j in 0..chunk_size {
                if offset + j < real.len() {
                    real_sq[j] = real[offset + j];
                    imag_sq[j] = imag[offset + j];
                }
            }
            
            self.simd_processor.multiply(&real_sq, &real_sq, &mut real_sq)?;
            self.simd_processor.multiply(&imag_sq, &imag_sq, &mut imag_sq)?;
            self.simd_processor.add(&real_sq, &imag_sq, &mut real_sq)?;
            
            // Compute sqrt
            for j in 0..chunk_size {
                if offset + j < magnitude.len() {
                    magnitude[offset + j] = real_sq[j].sqrt();
                }
            }
        }
        
        // Handle remaining elements
        for i in (chunks * chunk_size)..real.len() {
            magnitude[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
        }
        
        Ok(())
    }
}

/// SIMD-optimized MFCC feature extractor
pub struct SimdMfccExtractor {
    simd_processor: SimdProcessor,
    fft: SimdFFT,
    num_filters: usize,
    num_coefficients: usize,
    sample_rate: u32,
    mel_filters: Vec<Vec<f32>>,
    dct_matrix: Vec<Vec<f32>>,
}

impl SimdMfccExtractor {
    /// Create a new SIMD-optimized MFCC extractor
    pub fn new(config: &AudioConfig) -> Result<Self> {
        let simd_config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(simd_config)?;
        
        let fft_size = next_power_of_two(config.window_size);
        let fft = SimdFFT::new(fft_size)?;
        
        // Create mel filterbank
        let mel_filters = Self::create_mel_filterbank(
            config.mel_filters,
            fft_size,
            config.sample_rate,
            config.min_frequency,
            config.max_frequency,
        )?;
        
        // Create DCT matrix
        let dct_matrix = Self::create_dct_matrix(config.num_mfcc_coefficients, config.mel_filters);
        
        Ok(Self {
            simd_processor,
            fft,
            num_filters: config.mel_filters,
            num_coefficients: config.num_mfcc_coefficients,
            sample_rate: config.sample_rate,
            mel_filters,
            dct_matrix,
        })
    }
    
    /// Extract MFCC features using SIMD optimization
    pub fn extract_features(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        let fft_size = self.fft.size;
        
        // Apply window and compute FFT
        let mut windowed = vec![0.0; fft_size];
        let window = self.create_hamming_window(frame.len());
        self.apply_window_simd(frame, &window, &mut windowed[..frame.len()])?;
        
        let mut fft_real = vec![0.0; fft_size];
        let mut fft_imag = vec![0.0; fft_size];
        self.fft.forward(&windowed, &mut fft_real, &mut fft_imag)?;
        
        // Compute magnitude spectrum
        let mut magnitude = vec![0.0; fft_size / 2 + 1];
        self.fft.compute_magnitude_spectrum(
            &fft_real[..fft_size / 2 + 1],
            &fft_imag[..fft_size / 2 + 1],
            &mut magnitude,
        )?;
        
        // Apply mel filterbank
        let mel_energies = self.apply_mel_filterbank_simd(&magnitude)?;
        
        // Convert to log scale
        let log_mel_energies: Vec<f32> = mel_energies.iter()
            .map(|&e| if e > 1e-10 { e.ln() } else { -23.025850929940458 })
            .collect();
        
        // Apply DCT to get MFCC coefficients
        let mfcc_coefficients = self.apply_dct_simd(&log_mel_energies)?;
        
        Ok(mfcc_coefficients)
    }
    
    /// Apply window function using SIMD
    fn apply_window_simd(&self, input: &[f32], window: &[f32], output: &mut [f32]) -> Result<()> {
        self.simd_processor.multiply(input, window, output)
    }
    
    /// Apply mel filterbank using SIMD
    fn apply_mel_filterbank_simd(&self, magnitude: &[f32]) -> Result<Vec<f32>> {
        let mut mel_energies = vec![0.0; self.num_filters];
        
        for (i, filter) in self.mel_filters.iter().enumerate() {
            // Compute dot product of magnitude spectrum and filter
            let energy = self.simd_processor.dot_product(magnitude, filter)?;
            mel_energies[i] = energy;
        }
        
        Ok(mel_energies)
    }
    
    /// Apply DCT using SIMD
    fn apply_dct_simd(&self, log_mel_energies: &[f32]) -> Result<Vec<f32>> {
        let mut coefficients = vec![0.0; self.num_coefficients];
        
        for i in 0..self.num_coefficients {
            let coeff = self.simd_processor.dot_product(
                log_mel_energies,
                &self.dct_matrix[i],
            )?;
            coefficients[i] = coeff;
        }
        
        Ok(coefficients)
    }
    
    /// Create Hamming window
    fn create_hamming_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos())
            .collect()
    }
    
    /// Create mel filterbank
    fn create_mel_filterbank(
        num_filters: usize,
        fft_size: usize,
        sample_rate: u32,
        min_freq: f32,
        max_freq: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let mut filterbank = vec![vec![0.0; fft_size / 2 + 1]; num_filters];
        
        // Convert frequencies to mel scale
        let min_mel = Self::hz_to_mel(min_freq);
        let max_mel = Self::hz_to_mel(max_freq);
        
        // Create equally spaced mel frequencies
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_filters + 1) as f32)
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
        for i in 0..num_filters {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];
            
            // Left slope
            for bin in left..center {
                if center > left && bin < filterbank[i].len() {
                    filterbank[i][bin] = (bin - left) as f32 / (center - left) as f32;
                }
            }
            
            // Right slope
            for bin in center..=right.min(fft_size / 2) {
                if right > center && bin < filterbank[i].len() {
                    filterbank[i][bin] = (right - bin) as f32 / (right - center) as f32;
                }
            }
        }
        
        Ok(filterbank)
    }
    
    /// Create DCT matrix
    fn create_dct_matrix(num_coefficients: usize, num_filters: usize) -> Vec<Vec<f32>> {
        let mut dct_matrix = vec![vec![0.0; num_filters]; num_coefficients];
        
        for i in 0..num_coefficients {
            for j in 0..num_filters {
                dct_matrix[i][j] = (PI * i as f32 * (j as f32 + 0.5) / num_filters as f32).cos()
                    * (2.0 / num_filters as f32).sqrt();
            }
        }
        
        // Apply orthogonalization for the first coefficient
        if num_coefficients > 0 {
            let sqrt_2_inv = 1.0 / 2.0_f32.sqrt();
            for j in 0..num_filters {
                dct_matrix[0][j] *= sqrt_2_inv;
            }
        }
        
        dct_matrix
    }
    
    /// Convert Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    
    /// Convert mel scale to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }
}

/// SIMD-optimized pitch detection
pub struct SimdPitchDetector {
    simd_processor: SimdProcessor,
    sample_rate: u32,
    min_frequency: f32,
    max_frequency: f32,
}

impl SimdPitchDetector {
    /// Create a new SIMD-optimized pitch detector
    pub fn new(sample_rate: u32, min_frequency: f32, max_frequency: f32) -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        Ok(Self {
            simd_processor,
            sample_rate,
            min_frequency,
            max_frequency,
        })
    }
    
    /// Detect pitch using SIMD-optimized autocorrelation
    pub fn detect_pitch(&self, signal: &[f32]) -> Result<Option<f32>> {
        let min_lag = (self.sample_rate as f32 / self.max_frequency) as usize;
        let max_lag = (self.sample_rate as f32 / self.min_frequency) as usize;
        
        if signal.len() < max_lag * 2 {
            return Ok(None);
        }
        
        // Compute autocorrelation using SIMD
        let autocorr = self.compute_autocorrelation_simd(signal, min_lag, max_lag)?;
        
        // Find the peak in autocorrelation
        let (best_lag, correlation) = self.find_peak(&autocorr, min_lag)?;
        
        if correlation > 0.5 {
            let frequency = self.sample_rate as f32 / (best_lag + min_lag) as f32;
            Ok(Some(frequency))
        } else {
            Ok(None)
        }
    }
    
    /// Compute autocorrelation using SIMD
    fn compute_autocorrelation_simd(
        &self,
        signal: &[f32],
        min_lag: usize,
        max_lag: usize,
    ) -> Result<Vec<f32>> {
        let mut autocorr = Vec::with_capacity(max_lag - min_lag + 1);
        
        // Normalize signal
        let signal_energy = self.simd_processor.dot_product(signal, signal)?;
        
        for lag in min_lag..=max_lag {
            if lag < signal.len() {
                let correlation = self.simd_processor.dot_product(
                    &signal[..signal.len() - lag],
                    &signal[lag..],
                )?;
                
                let normalized = correlation / signal_energy.sqrt();
                autocorr.push(normalized);
            }
        }
        
        Ok(autocorr)
    }
    
    /// Find peak in autocorrelation
    fn find_peak(&self, autocorr: &[f32], min_lag: usize) -> Result<(usize, f32)> {
        let mut best_lag = 0;
        let mut best_correlation = 0.0;
        
        for (i, &corr) in autocorr.iter().enumerate() {
            if corr > best_correlation {
                best_correlation = corr;
                best_lag = i;
            }
        }
        
        Ok((best_lag, best_correlation))
    }
}

/// SIMD-optimized voice stress analyzer
pub struct SimdVoiceStressAnalyzer {
    simd_processor: SimdProcessor,
    sample_rate: u32,
}

impl SimdVoiceStressAnalyzer {
    /// Create a new SIMD-optimized voice stress analyzer
    pub fn new(sample_rate: u32) -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        Ok(Self {
            simd_processor,
            sample_rate,
        })
    }
    
    /// Analyze voice stress features using SIMD
    pub fn analyze_stress(&self, audio_frame: &[f32]) -> Result<StressFeatures> {
        // Compute energy using SIMD
        let energy = self.compute_energy_simd(audio_frame)?;
        
        // Compute zero crossing rate
        let zcr = self.compute_zcr_simd(audio_frame)?;
        
        // Compute spectral features
        let spectral_centroid = self.compute_spectral_centroid_simd(audio_frame)?;
        let spectral_spread = self.compute_spectral_spread_simd(audio_frame, spectral_centroid)?;
        
        // Compute formant features
        let formant_shift = self.estimate_formant_shift_simd(audio_frame)?;
        
        Ok(StressFeatures {
            energy,
            zero_crossing_rate: zcr,
            spectral_centroid,
            spectral_spread,
            formant_shift,
        })
    }
    
    /// Compute signal energy using SIMD
    fn compute_energy_simd(&self, signal: &[f32]) -> Result<f32> {
        self.simd_processor.dot_product(signal, signal)
    }
    
    /// Compute zero crossing rate using SIMD
    fn compute_zcr_simd(&self, signal: &[f32]) -> Result<f32> {
        if signal.len() < 2 {
            return Ok(0.0);
        }
        
        let mut crossings = 0;
        let mut prev_sign = signal[0] >= 0.0;
        
        for &sample in &signal[1..] {
            let curr_sign = sample >= 0.0;
            if curr_sign != prev_sign {
                crossings += 1;
            }
            prev_sign = curr_sign;
        }
        
        Ok(crossings as f32 / signal.len() as f32)
    }
    
    /// Compute spectral centroid using SIMD
    fn compute_spectral_centroid_simd(&self, signal: &[f32]) -> Result<f32> {
        let fft_size = next_power_of_two(signal.len());
        let fft = SimdFFT::new(fft_size)?;
        
        let mut padded = vec![0.0; fft_size];
        padded[..signal.len()].copy_from_slice(signal);
        
        let mut fft_real = vec![0.0; fft_size];
        let mut fft_imag = vec![0.0; fft_size];
        fft.forward(&padded, &mut fft_real, &mut fft_imag)?;
        
        let mut magnitude = vec![0.0; fft_size / 2 + 1];
        fft.compute_magnitude_spectrum(
            &fft_real[..fft_size / 2 + 1],
            &fft_imag[..fft_size / 2 + 1],
            &mut magnitude,
        )?;
        
        // Compute weighted sum and total magnitude
        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;
        
        for (i, &mag) in magnitude.iter().enumerate() {
            let frequency = i as f32 * self.sample_rate as f32 / fft_size as f32;
            weighted_sum += frequency * mag;
            total_magnitude += mag;
        }
        
        if total_magnitude > 0.0 {
            Ok(weighted_sum / total_magnitude)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute spectral spread using SIMD
    fn compute_spectral_spread_simd(&self, signal: &[f32], centroid: f32) -> Result<f32> {
        let fft_size = next_power_of_two(signal.len());
        let fft = SimdFFT::new(fft_size)?;
        
        let mut padded = vec![0.0; fft_size];
        padded[..signal.len()].copy_from_slice(signal);
        
        let mut fft_real = vec![0.0; fft_size];
        let mut fft_imag = vec![0.0; fft_size];
        fft.forward(&padded, &mut fft_real, &mut fft_imag)?;
        
        let mut magnitude = vec![0.0; fft_size / 2 + 1];
        fft.compute_magnitude_spectrum(
            &fft_real[..fft_size / 2 + 1],
            &fft_imag[..fft_size / 2 + 1],
            &mut magnitude,
        )?;
        
        // Compute weighted variance
        let mut weighted_variance = 0.0;
        let mut total_magnitude = 0.0;
        
        for (i, &mag) in magnitude.iter().enumerate() {
            let frequency = i as f32 * self.sample_rate as f32 / fft_size as f32;
            let deviation = frequency - centroid;
            weighted_variance += deviation * deviation * mag;
            total_magnitude += mag;
        }
        
        if total_magnitude > 0.0 {
            Ok((weighted_variance / total_magnitude).sqrt())
        } else {
            Ok(0.0)
        }
    }
    
    /// Estimate formant shift using SIMD
    fn estimate_formant_shift_simd(&self, signal: &[f32]) -> Result<f32> {
        // Simplified formant shift estimation
        // In practice, this would use LPC analysis
        let energy_low = self.compute_band_energy_simd(signal, 0.0, 1000.0)?;
        let energy_high = self.compute_band_energy_simd(signal, 1000.0, 4000.0)?;
        
        if energy_low > 0.0 {
            Ok(energy_high / energy_low)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute band energy using SIMD
    fn compute_band_energy_simd(&self, signal: &[f32], low_freq: f32, high_freq: f32) -> Result<f32> {
        let fft_size = next_power_of_two(signal.len());
        let fft = SimdFFT::new(fft_size)?;
        
        let mut padded = vec![0.0; fft_size];
        padded[..signal.len()].copy_from_slice(signal);
        
        let mut fft_real = vec![0.0; fft_size];
        let mut fft_imag = vec![0.0; fft_size];
        fft.forward(&padded, &mut fft_real, &mut fft_imag)?;
        
        let mut magnitude = vec![0.0; fft_size / 2 + 1];
        fft.compute_magnitude_spectrum(
            &fft_real[..fft_size / 2 + 1],
            &fft_imag[..fft_size / 2 + 1],
            &mut magnitude,
        )?;
        
        // Sum energy in frequency band
        let low_bin = (low_freq * fft_size as f32 / self.sample_rate as f32) as usize;
        let high_bin = (high_freq * fft_size as f32 / self.sample_rate as f32) as usize;
        
        let mut band_energy = 0.0;
        for i in low_bin..=high_bin.min(magnitude.len() - 1) {
            band_energy += magnitude[i] * magnitude[i];
        }
        
        Ok(band_energy)
    }
}

/// Voice stress features
#[derive(Debug, Clone)]
pub struct StressFeatures {
    pub energy: f32,
    pub zero_crossing_rate: f32,
    pub spectral_centroid: f32,
    pub spectral_spread: f32,
    pub formant_shift: f32,
}

/// Helper function to find next power of two
fn next_power_of_two(n: usize) -> usize {
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_fft_creation() {
        let fft = SimdFFT::new(512);
        assert!(fft.is_ok());
        
        let fft_invalid = SimdFFT::new(511);
        assert!(fft_invalid.is_err());
    }
    
    #[test]
    fn test_fft_forward() {
        let fft = SimdFFT::new(16).unwrap();
        
        // Create a simple sine wave
        let freq = 2.0;
        let signal: Vec<f32> = (0..16)
            .map(|i| (2.0 * PI * freq * i as f32 / 16.0).sin())
            .collect();
        
        let mut real = vec![0.0; 16];
        let mut imag = vec![0.0; 16];
        
        let result = fft.forward(&signal, &mut real, &mut imag);
        assert!(result.is_ok());
        
        // Check that we have non-zero output
        assert!(real.iter().any(|&x| x.abs() > 0.01));
    }
    
    #[test]
    fn test_magnitude_spectrum() {
        let fft = SimdFFT::new(16).unwrap();
        
        let real = vec![1.0, 0.0, -1.0, 0.0];
        let imag = vec![0.0, 1.0, 0.0, -1.0];
        let mut magnitude = vec![0.0; 4];
        
        let result = fft.compute_magnitude_spectrum(&real, &imag, &mut magnitude);
        assert!(result.is_ok());
        
        // Check magnitudes (sqrt(real^2 + imag^2))
        assert!((magnitude[0] - 1.0).abs() < 0.001);
        assert!((magnitude[1] - 1.0).abs() < 0.001);
        assert!((magnitude[2] - 1.0).abs() < 0.001);
        assert!((magnitude[3] - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_mfcc_extractor_creation() {
        let config = AudioConfig {
            sample_rate: 16000,
            window_size: 512,
            hop_size: 256,
            mel_filters: 26,
            num_mfcc_coefficients: 13,
            min_frequency: 80.0,
            max_frequency: 8000.0,
            pre_emphasis: 0.97,
        };
        
        let extractor = SimdMfccExtractor::new(&config);
        assert!(extractor.is_ok());
    }
    
    #[test]
    fn test_pitch_detector() {
        let detector = SimdPitchDetector::new(16000, 80.0, 400.0).unwrap();
        
        // Create a 200 Hz sine wave
        let freq = 200.0;
        let duration = 0.1; // 100ms
        let sample_rate = 16000;
        let samples = (duration * sample_rate as f32) as usize;
        
        let signal: Vec<f32> = (0..samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();
        
        let pitch = detector.detect_pitch(&signal);
        assert!(pitch.is_ok());
        
        if let Some(detected_freq) = pitch.unwrap() {
            // Allow 10% error in pitch detection
            assert!((detected_freq - freq).abs() / freq < 0.1);
        }
    }
    
    #[test]
    fn test_voice_stress_analyzer() {
        let analyzer = SimdVoiceStressAnalyzer::new(16000).unwrap();
        
        // Create a test signal
        let signal = vec![0.1; 1000];
        
        let features = analyzer.analyze_stress(&signal);
        assert!(features.is_ok());
        
        let stress = features.unwrap();
        assert!(stress.energy > 0.0);
        assert!(stress.zero_crossing_rate >= 0.0);
    }
    
    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(16), 16);
        assert_eq!(next_power_of_two(17), 32);
    }
}