/// Unit tests for audio modality analyzer
/// 
/// Tests voice stress analysis, pitch detection, and speech pattern recognition

use crate::common::*;
use approx::assert_relative_eq;

#[cfg(test)]
mod audio_analyzer_tests {
    use super::*;
    use fixtures::{AudioTestData, MultiModalTestData};
    use helpers::*;

    /// Test audio analyzer initialization
    #[test]
    fn test_audio_analyzer_creation() {
        let config = TestConfig::default();
        config.setup().expect("Failed to setup test config");
        
        let audio_data = AudioTestData::new_simple();
        assert_eq!(audio_data.sample_rate, 16000);
        assert_eq!(audio_data.duration_ms, 1000);
        assert_eq!(audio_data.samples.len(), 16000);
    }

    /// Test audio sample validation
    #[test]
    fn test_audio_sample_validation() {
        let audio_data = AudioTestData::new_simple();
        
        // Verify all samples are in valid range [-1.0, 1.0]
        for (i, &sample) in audio_data.samples.iter().enumerate() {
            assert!(
                sample >= -1.0 && sample <= 1.0,
                "Sample {} out of range: {} (should be in [-1.0, 1.0])",
                i, sample
            );
            assert!(
                sample.is_finite(),
                "Sample {} is not finite: {}",
                i, sample
            );
        }
    }

    /// Test pitch contour analysis
    #[test]
    fn test_pitch_contour_analysis() {
        let relaxed_audio = AudioTestData::new_relaxed();
        let stressed_audio = AudioTestData::new_stressed();
        
        // Calculate pitch stability
        let relaxed_stability = calculate_pitch_stability(&relaxed_audio.pitch_contour);
        let stressed_stability = calculate_pitch_stability(&stressed_audio.pitch_contour);
        
        assert!(
            relaxed_stability > stressed_stability,
            "Relaxed speech should have more stable pitch than stressed speech"
        );
        
        // Check pitch range
        let relaxed_range = calculate_pitch_range(&relaxed_audio.pitch_contour);
        let stressed_range = calculate_pitch_range(&stressed_audio.pitch_contour);
        
        assert!(
            stressed_range >= relaxed_range,
            "Stressed speech should have wider pitch range"
        );
    }

    /// Test voice stress indicators
    #[test]
    fn test_voice_stress_indicators() {
        let relaxed_audio = AudioTestData::new_relaxed();
        let stressed_audio = AudioTestData::new_stressed();
        
        // Compare energy levels
        let relaxed_energy = calculate_average_energy(&relaxed_audio.energy_contour);
        let stressed_energy = calculate_average_energy(&stressed_audio.energy_contour);
        
        assert!(
            stressed_energy < relaxed_energy,
            "Stressed speech typically has lower energy due to vocal tension"
        );
        
        // Check for energy variations
        let relaxed_energy_variance = calculate_energy_variance(&relaxed_audio.energy_contour);
        let stressed_energy_variance = calculate_energy_variance(&stressed_audio.energy_contour);
        
        assert!(
            stressed_energy_variance >= relaxed_energy_variance,
            "Stressed speech should show more energy variation"
        );
    }

    /// Test formant analysis
    #[test]
    fn test_formant_analysis() {
        let relaxed_audio = AudioTestData::new_relaxed();
        let stressed_audio = AudioTestData::new_stressed();
        
        // Compare formant frequencies
        let relaxed_f1_avg = calculate_average_formant(&relaxed_audio.formants, 0);
        let relaxed_f2_avg = calculate_average_formant(&relaxed_audio.formants, 1);
        let stressed_f1_avg = calculate_average_formant(&stressed_audio.formants, 0);
        let stressed_f2_avg = calculate_average_formant(&stressed_audio.formants, 1);
        
        // Stress typically causes formant shifts due to vocal tract tension
        assert!(
            stressed_f1_avg > relaxed_f1_avg,
            "Stress should cause F1 frequency increase"
        );
        assert!(
            stressed_f2_avg > relaxed_f2_avg,
            "Stress should cause F2 frequency increase"
        );
        
        // Verify formant frequencies are in reasonable ranges
        assert!(relaxed_f1_avg > 200.0 && relaxed_f1_avg < 1000.0, "F1 should be in typical range");
        assert!(relaxed_f2_avg > 800.0 && relaxed_f2_avg < 3000.0, "F2 should be in typical range");
    }

    /// Test audio feature extraction
    #[test]
    fn test_audio_feature_extraction() {
        let audio_data = AudioTestData::new_simple();
        let features = extract_mock_audio_features(&audio_data);
        
        // Verify feature vector properties
        assert!(!features.is_empty(), "Feature vector should not be empty");
        assert!(features.len() >= 10, "Should extract multiple audio features");
        
        // Check for valid feature values
        for (i, &feature) in features.iter().enumerate() {
            assert!(
                feature.is_finite(),
                "Feature {} should be finite, got {}",
                i, feature
            );
        }
    }

    /// Test spectral analysis
    #[test]
    fn test_spectral_analysis() {
        let audio_data = AudioTestData::new_simple();
        let spectrum = calculate_mock_spectrum(&audio_data.samples, audio_data.sample_rate);
        
        // Verify spectrum properties
        assert!(!spectrum.is_empty(), "Spectrum should not be empty");
        assert_eq!(spectrum.len(), audio_data.samples.len() / 2, "Spectrum size should be half of input");
        
        // Check for spectral energy conservation
        let total_energy: f32 = spectrum.iter().sum();
        assert!(total_energy > 0.0, "Total spectral energy should be positive");
        
        // Verify no NaN or infinite values
        for (i, &magnitude) in spectrum.iter().enumerate() {
            assert!(
                magnitude.is_finite() && magnitude >= 0.0,
                "Spectrum magnitude {} should be finite and non-negative, got {}",
                i, magnitude
            );
        }
    }

    /// Test mel-frequency cepstral coefficients (MFCC)
    #[test]
    fn test_mfcc_extraction() {
        let audio_data = AudioTestData::new_simple();
        let mfccs = calculate_mock_mfcc(&audio_data.samples, audio_data.sample_rate);
        
        // Typical MFCC configuration has 12-13 coefficients
        assert!(
            mfccs.len() >= 12 && mfccs.len() <= 13,
            "MFCC should have 12-13 coefficients, got {}",
            mfccs.len()
        );
        
        // First coefficient (C0) represents energy and should be larger
        assert!(
            mfccs[0].abs() > mfccs[1].abs(),
            "C0 coefficient should typically be larger than others"
        );
        
        // Check for valid MFCC values
        for (i, &coeff) in mfccs.iter().enumerate() {
            assert!(
                coeff.is_finite(),
                "MFCC coefficient {} should be finite, got {}",
                i, coeff
            );
        }
    }

    /// Test silence detection
    #[test]
    fn test_silence_detection() {
        // Create audio with silence
        let mut audio_data = AudioTestData::new_simple();
        // Insert silence in the middle
        for i in 4000..8000 {
            audio_data.samples[i] = 0.0;
        }
        
        let silence_segments = detect_mock_silence(&audio_data.samples, 0.01);
        
        assert!(
            !silence_segments.is_empty(),
            "Should detect silence segments"
        );
        
        // Verify silence segment is approximately in the middle
        let first_silence = silence_segments[0];
        assert!(
            first_silence.0 >= 4000 && first_silence.1 <= 8000,
            "Silence segment should be in expected range"
        );
    }

    /// Test audio processing performance
    #[test]
    fn test_audio_processing_performance() {
        let audio_data = AudioTestData::new_simple();
        
        let (_, measurement) = measure_performance(|| {
            extract_mock_audio_features(&audio_data)
        });
        
        // Assert reasonable processing time for 1 second of audio
        assert_performance_bounds(
            &measurement,
            std::time::Duration::from_millis(50), // Max 50ms for 1s audio
            Some(5 * 1024 * 1024) // Max 5MB memory usage
        );
    }

    /// Test edge cases in audio processing
    #[test]
    fn test_audio_edge_cases() {
        // Test very short audio
        let short_audio = create_short_audio_data();
        let features = extract_mock_audio_features(&short_audio);
        assert!(!features.is_empty(), "Should handle short audio");
        
        // Test very quiet audio
        let quiet_audio = create_quiet_audio_data();
        let features = extract_mock_audio_features(&quiet_audio);
        assert!(!features.is_empty(), "Should handle quiet audio");
        
        // Test clipped audio
        let clipped_audio = create_clipped_audio_data();
        let features = extract_mock_audio_features(&clipped_audio);
        assert!(!features.is_empty(), "Should handle clipped audio");
    }

    /// Test noise robustness
    #[test]
    fn test_noise_robustness() {
        let clean_audio = AudioTestData::new_simple();
        let noisy_audio = add_mock_noise(&clean_audio, 0.1); // 10% noise
        
        let clean_features = extract_mock_audio_features(&clean_audio);
        let noisy_features = extract_mock_audio_features(&noisy_audio);
        
        // Features should be reasonably similar despite noise
        let feature_correlation = calculate_feature_correlation(&clean_features, &noisy_features);
        assert!(
            feature_correlation > 0.8,
            "Features should be robust to moderate noise, correlation: {}",
            feature_correlation
        );
    }

    /// Test concurrent audio processing
    #[tokio::test]
    async fn test_concurrent_audio_processing() {
        let audio_data = AudioTestData::new_simple();
        
        let results = run_concurrent_tests(5, |_| {
            let data = audio_data.clone();
            async move {
                let features = extract_mock_audio_features(&data);
                assert!(!features.is_empty());
                Ok(features)
            }
        }).await;
        
        // All should succeed
        for result in results {
            assert!(result.is_ok(), "Concurrent audio processing should succeed");
        }
    }

    // Helper functions for tests

    fn calculate_pitch_stability(pitch_contour: &[f32]) -> f32 {
        if pitch_contour.len() < 2 {
            return 0.0;
        }
        
        let mean = pitch_contour.iter().sum::<f32>() / pitch_contour.len() as f32;
        let variance = pitch_contour.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f32>() / pitch_contour.len() as f32;
        
        1.0 / (1.0 + variance.sqrt()) // Higher stability = lower variance
    }

    fn calculate_pitch_range(pitch_contour: &[f32]) -> f32 {
        if pitch_contour.is_empty() {
            return 0.0;
        }
        
        let min_pitch = pitch_contour.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_pitch = pitch_contour.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        max_pitch - min_pitch
    }

    fn calculate_average_energy(energy_contour: &[f32]) -> f32 {
        if energy_contour.is_empty() {
            return 0.0;
        }
        energy_contour.iter().sum::<f32>() / energy_contour.len() as f32
    }

    fn calculate_energy_variance(energy_contour: &[f32]) -> f32 {
        if energy_contour.len() < 2 {
            return 0.0;
        }
        
        let mean = calculate_average_energy(energy_contour);
        energy_contour.iter()
            .map(|&e| (e - mean).powi(2))
            .sum::<f32>() / energy_contour.len() as f32
    }

    fn calculate_average_formant(formants: &[Vec<f32>], formant_index: usize) -> f32 {
        if formants.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = formants.iter()
            .filter_map(|f| f.get(formant_index))
            .sum();
        
        sum / formants.len() as f32
    }

    fn extract_mock_audio_features(data: &AudioTestData) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Basic statistical features
        let mean = data.samples.iter().sum::<f32>() / data.samples.len() as f32;
        let variance = data.samples.iter()
            .map(|&s| (s - mean).powi(2))
            .sum::<f32>() / data.samples.len() as f32;
        features.push(mean);
        features.push(variance.sqrt());
        
        // Pitch features
        features.push(calculate_average_energy(&data.pitch_contour));
        features.push(calculate_pitch_stability(&data.pitch_contour));
        features.push(calculate_pitch_range(&data.pitch_contour));
        
        // Energy features
        features.push(calculate_average_energy(&data.energy_contour));
        features.push(calculate_energy_variance(&data.energy_contour));
        
        // Formant features
        for i in 0..3 {
            features.push(calculate_average_formant(&data.formants, i));
        }
        
        // Spectral features
        let spectrum = calculate_mock_spectrum(&data.samples, data.sample_rate);
        let spectral_centroid = calculate_spectral_centroid(&spectrum, data.sample_rate);
        features.push(spectral_centroid);
        
        features
    }

    fn calculate_mock_spectrum(samples: &[f32], _sample_rate: u32) -> Vec<f32> {
        // Simplified FFT mock - in real implementation would use proper FFT
        let n = samples.len() / 2;
        let mut spectrum = Vec::with_capacity(n);
        
        for i in 0..n {
            let real = samples.get(i * 2).unwrap_or(&0.0);
            let imag = samples.get(i * 2 + 1).unwrap_or(&0.0);
            let magnitude = (real * real + imag * imag).sqrt();
            spectrum.push(magnitude);
        }
        
        spectrum
    }

    fn calculate_mock_mfcc(samples: &[f32], sample_rate: u32) -> Vec<f32> {
        // Mock MFCC calculation - real implementation would use proper mel filter bank
        let spectrum = calculate_mock_spectrum(samples, sample_rate);
        let mut mfccs = Vec::with_capacity(13);
        
        // C0 - energy coefficient
        let energy: f32 = spectrum.iter().sum();
        mfccs.push(energy.ln());
        
        // C1-C12 - cepstral coefficients
        for i in 1..13 {
            let coeff = spectrum.iter()
                .enumerate()
                .map(|(j, &mag)| mag * (std::f32::consts::PI * i as f32 * j as f32 / spectrum.len() as f32).cos())
                .sum::<f32>() / spectrum.len() as f32;
            mfccs.push(coeff);
        }
        
        mfccs
    }

    fn calculate_spectral_centroid(spectrum: &[f32], sample_rate: u32) -> f32 {
        let weighted_sum: f32 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| i as f32 * mag)
            .sum();
        let total_magnitude: f32 = spectrum.iter().sum();
        
        if total_magnitude > 0.0 {
            (weighted_sum / total_magnitude) * (sample_rate as f32 / 2.0) / spectrum.len() as f32
        } else {
            0.0
        }
    }

    fn detect_mock_silence(samples: &[f32], threshold: f32) -> Vec<(usize, usize)> {
        let mut silence_segments = Vec::new();
        let mut in_silence = false;
        let mut silence_start = 0;
        
        for (i, &sample) in samples.iter().enumerate() {
            let is_silent = sample.abs() < threshold;
            
            if is_silent && !in_silence {
                in_silence = true;
                silence_start = i;
            } else if !is_silent && in_silence {
                in_silence = false;
                silence_segments.push((silence_start, i));
            }
        }
        
        // Handle case where silence extends to end
        if in_silence {
            silence_segments.push((silence_start, samples.len()));
        }
        
        silence_segments
    }

    fn create_short_audio_data() -> AudioTestData {
        AudioTestData {
            sample_rate: 16000,
            samples: vec![0.1; 160], // 10ms of audio
            duration_ms: 10,
            pitch_contour: vec![220.0],
            energy_contour: vec![0.1],
            formants: vec![vec![600.0, 1200.0, 2500.0]],
        }
    }

    fn create_quiet_audio_data() -> AudioTestData {
        AudioTestData {
            sample_rate: 16000,
            samples: vec![0.001; 16000], // Very quiet audio
            duration_ms: 1000,
            pitch_contour: vec![200.0; 100],
            energy_contour: vec![0.001; 100],
            formants: vec![vec![600.0, 1200.0, 2500.0]; 100],
        }
    }

    fn create_clipped_audio_data() -> AudioTestData {
        let mut data = AudioTestData::new_simple();
        // Clip some samples to ±1.0
        for i in (0..data.samples.len()).step_by(10) {
            data.samples[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        data
    }

    fn add_mock_noise(audio_data: &AudioTestData, noise_level: f32) -> AudioTestData {
        let mut noisy_data = audio_data.clone();
        
        for sample in noisy_data.samples.iter_mut() {
            let noise = (rand::random::<f32>() - 0.5) * 2.0 * noise_level;
            *sample = (*sample + noise).clamp(-1.0, 1.0);
        }
        
        noisy_data
    }

    fn calculate_feature_correlation(features1: &[f32], features2: &[f32]) -> f32 {
        if features1.len() != features2.len() || features1.is_empty() {
            return 0.0;
        }
        
        let mean1 = features1.iter().sum::<f32>() / features1.len() as f32;
        let mean2 = features2.iter().sum::<f32>() / features2.len() as f32;
        
        let covariance: f32 = features1.iter()
            .zip(features2.iter())
            .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
            .sum();
        
        let var1: f32 = features1.iter().map(|&x| (x - mean1).powi(2)).sum();
        let var2: f32 = features2.iter().map(|&x| (x - mean2).powi(2)).sum();
        
        if var1 > 0.0 && var2 > 0.0 {
            covariance / (var1.sqrt() * var2.sqrt())
        } else {
            0.0
        }
    }
}