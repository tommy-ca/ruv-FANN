/// Unit tests for physiological modality analyzer
/// 
/// Tests heart rate analysis, skin conductance processing, and physiological signal anomaly detection

use crate::common::*;

#[cfg(test)]
mod physiological_analyzer_tests {
    use super::*;
    use fixtures::{PhysiologicalTestData, MultiModalTestData};
    use helpers::*;

    /// Test physiological analyzer initialization
    #[test]
    fn test_physiological_analyzer_creation() {
        let config = TestConfig::default();
        config.setup().expect("Failed to setup test config");
        
        let physio_data = PhysiologicalTestData::new_baseline();
        assert!(!physio_data.heart_rate_bpm.is_empty());
        assert!(!physio_data.skin_conductance.is_empty());
        assert!(!physio_data.blood_pressure.is_empty());
        assert!(physio_data.sampling_rate_hz > 0.0);
    }

    /// Test heart rate analysis
    #[test]
    fn test_heart_rate_analysis() {
        let baseline_data = PhysiologicalTestData::new_baseline();
        let stressed_data = PhysiologicalTestData::new_stressed();
        
        // Calculate average heart rates
        let baseline_hr_avg = calculate_average_heart_rate(&baseline_data.heart_rate_bpm);
        let stressed_hr_avg = calculate_average_heart_rate(&stressed_data.heart_rate_bpm);
        
        assert!(
            stressed_hr_avg > baseline_hr_avg,
            "Stressed heart rate should be higher than baseline"
        );
        
        // Verify reasonable ranges
        assert!(
            baseline_hr_avg >= 60.0 && baseline_hr_avg <= 100.0,
            "Baseline heart rate should be in normal resting range"
        );
        assert!(
            stressed_hr_avg >= 80.0 && stressed_hr_avg <= 180.0,
            "Stressed heart rate should be in elevated range"
        );
    }

    /// Test heart rate variability (HRV)
    #[test]
    fn test_heart_rate_variability() {
        let baseline_data = PhysiologicalTestData::new_baseline();
        let stressed_data = PhysiologicalTestData::new_stressed();
        
        let baseline_hrv = calculate_hrv(&baseline_data.heart_rate_bpm);
        let stressed_hrv = calculate_hrv(&stressed_data.heart_rate_bpm);
        
        // HRV typically decreases under stress
        assert!(
            baseline_hrv > stressed_hrv,
            "Baseline HRV should be higher than stressed HRV (stress reduces variability)"
        );
        
        // Verify HRV values are reasonable
        assert!(baseline_hrv > 0.0, "HRV should be positive");
        assert!(stressed_hrv > 0.0, "HRV should be positive");
    }

    /// Test skin conductance analysis
    #[test]
    fn test_skin_conductance_analysis() {
        let baseline_data = PhysiologicalTestData::new_baseline();
        let stressed_data = PhysiologicalTestData::new_stressed();
        
        let baseline_sc_avg = calculate_average_skin_conductance(&baseline_data.skin_conductance);
        let stressed_sc_avg = calculate_average_skin_conductance(&stressed_data.skin_conductance);
        
        assert!(
            stressed_sc_avg > baseline_sc_avg,
            "Stressed skin conductance should be higher than baseline"
        );
        
        // Test skin conductance response (SCR) detection
        let baseline_scr_count = detect_skin_conductance_responses(&baseline_data.skin_conductance, 0.1);
        let stressed_scr_count = detect_skin_conductance_responses(&stressed_data.skin_conductance, 0.1);
        
        assert!(
            stressed_scr_count >= baseline_scr_count,
            "Stressed state should show more SCR events"
        );
    }

    /// Test blood pressure analysis
    #[test]
    fn test_blood_pressure_analysis() {
        let baseline_data = PhysiologicalTestData::new_baseline();
        let stressed_data = PhysiologicalTestData::new_stressed();
        
        let baseline_systolic = calculate_average_systolic(&baseline_data.blood_pressure);
        let baseline_diastolic = calculate_average_diastolic(&baseline_data.blood_pressure);
        let stressed_systolic = calculate_average_systolic(&stressed_data.blood_pressure);
        let stressed_diastolic = calculate_average_diastolic(&stressed_data.blood_pressure);
        
        assert!(
            stressed_systolic > baseline_systolic,
            "Stressed systolic pressure should be higher"
        );
        assert!(
            stressed_diastolic > baseline_diastolic,
            "Stressed diastolic pressure should be higher"
        );
        
        // Verify reasonable blood pressure ranges
        assert!(
            baseline_systolic >= 90.0 && baseline_systolic <= 140.0,
            "Baseline systolic should be in normal range"
        );
        assert!(
            baseline_diastolic >= 60.0 && baseline_diastolic <= 90.0,
            "Baseline diastolic should be in normal range"
        );
    }

    /// Test respiration rate analysis
    #[test]
    fn test_respiration_analysis() {
        let baseline_data = PhysiologicalTestData::new_baseline();
        let stressed_data = PhysiologicalTestData::new_stressed();
        
        let baseline_resp_avg = calculate_average_respiration(&baseline_data.respiration_rate);
        let stressed_resp_avg = calculate_average_respiration(&stressed_data.respiration_rate);
        
        assert!(
            stressed_resp_avg > baseline_resp_avg,
            "Stressed respiration rate should be higher"
        );
        
        // Verify reasonable respiration ranges
        assert!(
            baseline_resp_avg >= 12.0 && baseline_resp_avg <= 20.0,
            "Baseline respiration should be in normal range"
        );
        assert!(
            stressed_resp_avg >= 16.0 && stressed_resp_avg <= 30.0,
            "Stressed respiration should be in elevated range"
        );
    }

    /// Test signal preprocessing
    #[test]
    fn test_signal_preprocessing() {
        let mut noisy_data = PhysiologicalTestData::new_baseline();
        
        // Add noise to heart rate signal
        for hr in noisy_data.heart_rate_bpm.iter_mut() {
            *hr += (rand::random::<f32>() - 0.5) * 10.0; // ±5 BPM noise
        }
        
        let filtered_hr = apply_lowpass_filter(&noisy_data.heart_rate_bpm, 0.1);
        
        // Filtered signal should be smoother
        let original_variance = calculate_signal_variance(&noisy_data.heart_rate_bpm);
        let filtered_variance = calculate_signal_variance(&filtered_hr);
        
        assert!(
            filtered_variance < original_variance,
            "Filtered signal should have lower variance"
        );
    }

    /// Test anomaly detection
    #[test]
    fn test_anomaly_detection() {
        let mut normal_data = PhysiologicalTestData::new_baseline();
        
        // Inject anomalies
        normal_data.heart_rate_bpm[100] = 200.0; // Tachycardia
        normal_data.heart_rate_bpm[200] = 40.0;  // Bradycardia
        normal_data.skin_conductance[150] = 50.0; // High SCL spike
        
        let hr_anomalies = detect_heart_rate_anomalies(&normal_data.heart_rate_bpm);
        let sc_anomalies = detect_skin_conductance_anomalies(&normal_data.skin_conductance);
        
        assert!(
            hr_anomalies.len() >= 2,
            "Should detect heart rate anomalies"
        );
        assert!(
            sc_anomalies.len() >= 1,
            "Should detect skin conductance anomalies"
        );
        
        // Verify anomaly indices
        assert!(hr_anomalies.contains(&100), "Should detect tachycardia at index 100");
        assert!(hr_anomalies.contains(&200), "Should detect bradycardia at index 200");
        assert!(sc_anomalies.contains(&150), "Should detect SCL spike at index 150");
    }

    /// Test feature extraction
    #[test]
    fn test_physiological_feature_extraction() {
        let physio_data = PhysiologicalTestData::new_baseline();
        let features = extract_physiological_features(&physio_data);
        
        // Verify feature vector properties
        assert!(!features.is_empty(), "Feature vector should not be empty");
        assert!(features.len() >= 15, "Should extract multiple physiological features");
        
        // Check for valid feature values
        for (i, &feature) in features.iter().enumerate() {
            assert!(
                feature.is_finite(),
                "Feature {} should be finite, got {}",
                i, feature
            );
            assert!(
                feature >= 0.0,
                "Physiological features should be non-negative, got {} at index {}",
                feature, i
            );
        }
    }

    /// Test signal quality assessment
    #[test]
    fn test_signal_quality_assessment() {
        let good_data = PhysiologicalTestData::new_baseline();
        let mut poor_data = PhysiologicalTestData::new_baseline();
        
        // Corrupt some data to simulate poor signal quality
        for i in (0..poor_data.heart_rate_bpm.len()).step_by(10) {
            poor_data.heart_rate_bpm[i] = f32::NAN;
        }
        
        let good_quality = assess_signal_quality(&good_data.heart_rate_bpm);
        let poor_quality = assess_signal_quality(&poor_data.heart_rate_bpm);
        
        assert!(
            good_quality > poor_quality,
            "Good data should have higher quality score"
        );
        assert!(
            good_quality >= 0.8,
            "Good data should have high quality score"
        );
        assert!(
            poor_quality <= 0.5,
            "Poor data should have low quality score"
        );
    }

    /// Test physiological stress index calculation
    #[test]
    fn test_stress_index_calculation() {
        let baseline_data = PhysiologicalTestData::new_baseline();
        let stressed_data = PhysiologicalTestData::new_stressed();
        
        let baseline_stress_index = calculate_stress_index(&baseline_data);
        let stressed_stress_index = calculate_stress_index(&stressed_data);
        
        assert!(
            stressed_stress_index > baseline_stress_index,
            "Stressed state should have higher stress index"
        );
        
        // Verify stress index is in reasonable range [0, 1]
        assert!(
            baseline_stress_index >= 0.0 && baseline_stress_index <= 1.0,
            "Baseline stress index should be in [0, 1]"
        );
        assert!(
            stressed_stress_index >= 0.0 && stressed_stress_index <= 1.0,
            "Stressed stress index should be in [0, 1]"
        );
    }

    /// Test performance of physiological processing
    #[test]
    fn test_physiological_processing_performance() {
        let physio_data = PhysiologicalTestData::new_baseline();
        
        let (_, measurement) = measure_performance(|| {
            extract_physiological_features(&physio_data)
        });
        
        // Assert reasonable processing time
        assert_performance_bounds(
            &measurement,
            std::time::Duration::from_millis(50), // Max 50ms for 5 minutes of data
            Some(5 * 1024 * 1024) // Max 5MB memory usage
        );
    }

    /// Test edge cases in physiological processing
    #[test]
    fn test_physiological_edge_cases() {
        // Test with minimal data
        let minimal_data = create_minimal_physiological_data();
        let features = extract_physiological_features(&minimal_data);
        assert!(!features.is_empty(), "Should handle minimal data");
        
        // Test with extreme values
        let extreme_data = create_extreme_physiological_data();
        let features = extract_physiological_features(&extreme_data);
        assert!(!features.is_empty(), "Should handle extreme values");
        
        // Test with missing data (NaN values)
        let missing_data = create_missing_physiological_data();
        let features = extract_physiological_features(&missing_data);
        assert!(!features.is_empty(), "Should handle missing data gracefully");
    }

    /// Test concurrent physiological processing
    #[tokio::test]
    async fn test_concurrent_physiological_processing() {
        let physio_data = PhysiologicalTestData::new_baseline();
        
        let results = run_concurrent_tests(5, |_| {
            let data = physio_data.clone();
            async move {
                let features = extract_physiological_features(&data);
                assert!(!features.is_empty());
                Ok(features)
            }
        }).await;
        
        // All should succeed
        for result in results {
            assert!(result.is_ok(), "Concurrent physiological processing should succeed");
        }
    }

    // Helper functions for tests

    fn calculate_average_heart_rate(heart_rate: &[f32]) -> f32 {
        if heart_rate.is_empty() {
            return 0.0;
        }
        heart_rate.iter().sum::<f32>() / heart_rate.len() as f32
    }

    fn calculate_hrv(heart_rate: &[f32]) -> f32 {
        if heart_rate.len() < 2 {
            return 0.0;
        }
        
        // Calculate successive differences
        let differences: Vec<f32> = heart_rate.windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();
        
        // RMSSD (Root Mean Square of Successive Differences)
        let mean_sq_diff = differences.iter()
            .map(|&diff| diff * diff)
            .sum::<f32>() / differences.len() as f32;
        
        mean_sq_diff.sqrt()
    }

    fn calculate_average_skin_conductance(skin_conductance: &[f32]) -> f32 {
        if skin_conductance.is_empty() {
            return 0.0;
        }
        skin_conductance.iter().sum::<f32>() / skin_conductance.len() as f32
    }

    fn detect_skin_conductance_responses(skin_conductance: &[f32], threshold: f32) -> usize {
        if skin_conductance.len() < 10 {
            return 0;
        }
        
        let mut responses = 0;
        let window_size = 5;
        
        for i in window_size..skin_conductance.len() - window_size {
            let baseline = skin_conductance[i - window_size..i].iter().sum::<f32>() / window_size as f32;
            let current = skin_conductance[i];
            
            if current - baseline > threshold {
                responses += 1;
            }
        }
        
        responses
    }

    fn calculate_average_systolic(blood_pressure: &[(f32, f32)]) -> f32 {
        if blood_pressure.is_empty() {
            return 0.0;
        }
        blood_pressure.iter().map(|&(systolic, _)| systolic).sum::<f32>() / blood_pressure.len() as f32
    }

    fn calculate_average_diastolic(blood_pressure: &[(f32, f32)]) -> f32 {
        if blood_pressure.is_empty() {
            return 0.0;
        }
        blood_pressure.iter().map(|&(_, diastolic)| diastolic).sum::<f32>() / blood_pressure.len() as f32
    }

    fn calculate_average_respiration(respiration_rate: &[f32]) -> f32 {
        if respiration_rate.is_empty() {
            return 0.0;
        }
        respiration_rate.iter().sum::<f32>() / respiration_rate.len() as f32
    }

    fn apply_lowpass_filter(signal: &[f32], alpha: f32) -> Vec<f32> {
        if signal.is_empty() {
            return Vec::new();
        }
        
        let mut filtered = Vec::with_capacity(signal.len());
        filtered.push(signal[0]);
        
        for i in 1..signal.len() {
            let filtered_value = alpha * signal[i] + (1.0 - alpha) * filtered[i - 1];
            filtered.push(filtered_value);
        }
        
        filtered
    }

    fn calculate_signal_variance(signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        signal.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / signal.len() as f32
    }

    fn detect_heart_rate_anomalies(heart_rate: &[f32]) -> Vec<usize> {
        let mut anomalies = Vec::new();
        
        for (i, &hr) in heart_rate.iter().enumerate() {
            if hr.is_nan() || hr < 30.0 || hr > 220.0 {
                anomalies.push(i);
            }
        }
        
        anomalies
    }

    fn detect_skin_conductance_anomalies(skin_conductance: &[f32]) -> Vec<usize> {
        let mut anomalies = Vec::new();
        let mean = calculate_average_skin_conductance(skin_conductance);
        let std_dev = calculate_signal_variance(skin_conductance).sqrt();
        let threshold = mean + 3.0 * std_dev; // 3-sigma rule
        
        for (i, &sc) in skin_conductance.iter().enumerate() {
            if sc.is_nan() || sc > threshold || sc < 0.0 {
                anomalies.push(i);
            }
        }
        
        anomalies
    }

    fn extract_physiological_features(data: &PhysiologicalTestData) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Heart rate features
        features.push(calculate_average_heart_rate(&data.heart_rate_bpm));
        features.push(calculate_hrv(&data.heart_rate_bpm));
        features.push(calculate_signal_variance(&data.heart_rate_bpm));
        
        // Skin conductance features
        features.push(calculate_average_skin_conductance(&data.skin_conductance));
        features.push(calculate_signal_variance(&data.skin_conductance));
        features.push(detect_skin_conductance_responses(&data.skin_conductance, 0.1) as f32);
        
        // Blood pressure features
        features.push(calculate_average_systolic(&data.blood_pressure));
        features.push(calculate_average_diastolic(&data.blood_pressure));
        
        let systolic_values: Vec<f32> = data.blood_pressure.iter().map(|&(s, _)| s).collect();
        let diastolic_values: Vec<f32> = data.blood_pressure.iter().map(|&(_, d)| d).collect();
        features.push(calculate_signal_variance(&systolic_values));
        features.push(calculate_signal_variance(&diastolic_values));
        
        // Respiration features
        features.push(calculate_average_respiration(&data.respiration_rate));
        features.push(calculate_signal_variance(&data.respiration_rate));
        
        // Cross-signal features
        let hr_resp_correlation = calculate_signal_correlation(&data.heart_rate_bpm, &data.respiration_rate);
        features.push(hr_resp_correlation);
        
        // Signal quality features
        features.push(assess_signal_quality(&data.heart_rate_bpm));
        features.push(assess_signal_quality(&data.skin_conductance));
        
        features
    }

    fn assess_signal_quality(signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        
        let nan_count = signal.iter().filter(|&&x| x.is_nan()).count();
        let infinite_count = signal.iter().filter(|&&x| x.is_infinite()).count();
        let total_count = signal.len();
        
        let quality = 1.0 - (nan_count + infinite_count) as f32 / total_count as f32;
        quality.max(0.0).min(1.0)
    }

    fn calculate_stress_index(data: &PhysiologicalTestData) -> f32 {
        let hr_avg = calculate_average_heart_rate(&data.heart_rate_bpm);
        let sc_avg = calculate_average_skin_conductance(&data.skin_conductance);
        let resp_avg = calculate_average_respiration(&data.respiration_rate);
        let systolic_avg = calculate_average_systolic(&data.blood_pressure);
        
        // Normalize and combine features for stress index
        let hr_norm = ((hr_avg - 70.0) / 50.0).max(0.0).min(1.0);
        let sc_norm = ((sc_avg - 2.0) / 10.0).max(0.0).min(1.0);
        let resp_norm = ((resp_avg - 16.0) / 10.0).max(0.0).min(1.0);
        let bp_norm = ((systolic_avg - 120.0) / 40.0).max(0.0).min(1.0);
        
        (hr_norm * 0.3 + sc_norm * 0.3 + resp_norm * 0.2 + bp_norm * 0.2)
    }

    fn calculate_signal_correlation(signal1: &[f32], signal2: &[f32]) -> f32 {
        let min_len = signal1.len().min(signal2.len());
        if min_len < 2 {
            return 0.0;
        }
        
        let sig1 = &signal1[..min_len];
        let sig2 = &signal2[..min_len];
        
        let mean1 = sig1.iter().sum::<f32>() / min_len as f32;
        let mean2 = sig2.iter().sum::<f32>() / min_len as f32;
        
        let covariance: f32 = sig1.iter()
            .zip(sig2.iter())
            .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
            .sum();
        
        let var1: f32 = sig1.iter().map(|&x| (x - mean1).powi(2)).sum();
        let var2: f32 = sig2.iter().map(|&x| (x - mean2).powi(2)).sum();
        
        if var1 > 0.0 && var2 > 0.0 {
            covariance / (var1.sqrt() * var2.sqrt())
        } else {
            0.0
        }
    }

    fn create_minimal_physiological_data() -> PhysiologicalTestData {
        PhysiologicalTestData {
            heart_rate_bpm: vec![72.0; 10],
            skin_conductance: vec![2.0; 10],
            blood_pressure: vec![(120.0, 80.0); 10],
            respiration_rate: vec![16.0; 10],
            sampling_rate_hz: 1.0,
        }
    }

    fn create_extreme_physiological_data() -> PhysiologicalTestData {
        PhysiologicalTestData {
            heart_rate_bpm: vec![200.0, 30.0, 150.0, 45.0, 180.0],
            skin_conductance: vec![50.0, 0.1, 25.0, 0.5, 40.0],
            blood_pressure: vec![(200.0, 120.0), (80.0, 40.0), (160.0, 100.0)],
            respiration_rate: vec![40.0, 8.0, 30.0, 10.0, 35.0],
            sampling_rate_hz: 1.0,
        }
    }

    fn create_missing_physiological_data() -> PhysiologicalTestData {
        PhysiologicalTestData {
            heart_rate_bpm: vec![72.0, f32::NAN, 75.0, f32::NAN, 70.0],
            skin_conductance: vec![2.0, 2.1, f32::NAN, 1.9, 2.2],
            blood_pressure: vec![(120.0, 80.0), (f32::NAN, 82.0), (122.0, f32::NAN)],
            respiration_rate: vec![16.0, f32::NAN, 17.0, 15.0, f32::NAN],
            sampling_rate_hz: 1.0,
        }
    }
}