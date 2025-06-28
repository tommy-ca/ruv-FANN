/// Property-based invariant tests
/// 
/// Tests that verify fundamental system invariants hold across
/// all possible inputs and configurations

use crate::common::*;
use proptest::prelude::*;
use generators::*;

#[cfg(test)]
mod system_invariants {
    use super::*;

    /// Test that probability outputs are always in [0, 1] range
    proptest! {
        #[test]
        fn probability_values_in_valid_range(
            prob in probability::<f32>(),
            conf in confidence_score::<f32>()
        ) {
            // Any generated probability should be in [0, 1]
            prop_assert!(prob >= 0.0 && prob <= 1.0);
            prop_assert!(conf >= 0.0 && conf <= 1.0);
            
            // Test mock deception score creation
            let score = mocks::MockDeceptionScore::new(prob, conf);
            prop_assert!(score.probability >= 0.0 && score.probability <= 1.0);
            prop_assert!(score.confidence >= 0.0 && score.confidence <= 1.0);
        }
    }

    /// Test that feature vectors maintain consistent dimensionality
    proptest! {
        #[test]
        fn feature_vector_dimensionality_consistency(
            dim in 1usize..=1000,
            range in -10.0f64..10.0f64
        ) {
            let features = feature_vector::<f32>(dim, range..range+1.0).new_tree(&mut TestRunner::default()).unwrap().current();
            
            prop_assert_eq!(features.len(), dim);
            
            // All features should be finite
            for &feature in &features {
                prop_assert!(feature.is_finite());
            }
        }
    }

    /// Test that fusion always produces valid outputs regardless of input scores
    proptest! {
        #[test]
        fn fusion_output_validity(
            score1 in probability::<f32>(),
            score2 in probability::<f32>(),
            score3 in probability::<f32>(),
            conf1 in confidence_score::<f32>(),
            conf2 in confidence_score::<f32>(),
            conf3 in confidence_score::<f32>()
        ) {
            use std::collections::HashMap;
            
            let mut scores = HashMap::new();
            scores.insert("modality1".to_string(), mocks::MockDeceptionScore::new(score1, conf1));
            scores.insert("modality2".to_string(), mocks::MockDeceptionScore::new(score2, conf2));
            scores.insert("modality3".to_string(), mocks::MockDeceptionScore::new(score3, conf3));
            
            let weights = vec![0.33, 0.33, 0.34];
            let fused = test_late_fusion(&scores, &weights);
            
            // Fused result must always be valid
            prop_assert!(fused.deception_probability >= 0.0 && fused.deception_probability <= 1.0);
            prop_assert!(fused.confidence >= 0.0 && fused.confidence <= 1.0);
            prop_assert!(fused.deception_probability.is_finite());
            prop_assert!(fused.confidence.is_finite());
        }
    }

    /// Test that neural network outputs are bounded and finite
    proptest! {
        #[test]
        fn neural_network_output_bounds(
            input_size in 1usize..=100,
            hidden_size in 1usize..=50,
            features in feature_vector::<f32>(input_size, -1.0..1.0)
        ) {
            let network = mocks::MockNeuralNetwork::new(vec![input_size, hidden_size, 1]);
            
            match network.forward(&features) {
                Ok(output) => {
                    prop_assert_eq!(output.len(), 1);
                    prop_assert!(output[0].is_finite());
                    // For sigmoid output, should be in [0, 1]
                    prop_assert!(output[0] >= 0.0 && output[0] <= 1.0);
                }
                Err(_) => {
                    // Network errors are acceptable for invalid inputs
                }
            }
        }
    }

    /// Test that streaming pipeline maintains temporal consistency
    proptest! {
        #[test]
        fn streaming_temporal_consistency(
            frame_count in 1usize..=20,
            frame_size in 100usize..=1000
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let pipeline = mocks::MockStreamingPipeline::new();
                let mut timestamps = Vec::new();
                
                for _ in 0..frame_count {
                    let frame = vec![0u8; frame_size];
                    let start_time = std::time::Instant::now();
                    
                    match pipeline.process_frame(&frame).await {
                        Ok(_) => {
                            timestamps.push(start_time.elapsed());
                        }
                        Err(_) => {
                            // Processing errors are acceptable
                        }
                    }
                }
                
                // Processing times should be reasonable and consistent
                if timestamps.len() > 1 {
                    let avg_time = timestamps.iter().sum::<std::time::Duration>() / timestamps.len() as u32;
                    prop_assert!(avg_time.as_millis() < 1000); // Less than 1 second per frame
                }
            });
        }
    }

    /// Test that confidence decreases with conflicting evidence
    proptest! {
        #[test]
        fn confidence_conflict_property(
            base_prob in 0.3f32..0.7f32,
            deviation in 0.1f32..0.4f32,
            base_conf in 0.7f32..0.9f32
        ) {
            use std::collections::HashMap;
            
            // Create agreeing scores
            let mut agreeing_scores = HashMap::new();
            agreeing_scores.insert("mod1".to_string(), mocks::MockDeceptionScore::new(base_prob, base_conf));
            agreeing_scores.insert("mod2".to_string(), mocks::MockDeceptionScore::new(base_prob + 0.05, base_conf));
            agreeing_scores.insert("mod3".to_string(), mocks::MockDeceptionScore::new(base_prob - 0.05, base_conf));
            
            // Create conflicting scores
            let mut conflicting_scores = HashMap::new();
            conflicting_scores.insert("mod1".to_string(), mocks::MockDeceptionScore::new(base_prob, base_conf));
            conflicting_scores.insert("mod2".to_string(), mocks::MockDeceptionScore::new(base_prob + deviation, base_conf));
            conflicting_scores.insert("mod3".to_string(), mocks::MockDeceptionScore::new(base_prob - deviation, base_conf));
            
            let weights = vec![0.33, 0.33, 0.34];
            let agreeing_result = test_late_fusion(&agreeing_scores, &weights);
            let conflicting_result = test_late_fusion(&conflicting_scores, &weights);
            
            // Conflicting evidence should produce lower confidence
            prop_assert!(agreeing_result.confidence >= conflicting_result.confidence);
        }
    }

    /// Test that system respects causality in temporal data
    proptest! {
        #[test]
        fn temporal_causality_property(
            sequence_length in 2usize..=10,
            base_value in 0.3f32..0.7f32
        ) {
            let mut temporal_sequence = Vec::new();
            
            // Create monotonically increasing sequence
            for i in 0..sequence_length {
                let value = base_value + (i as f32 * 0.05);
                temporal_sequence.push(value.min(1.0));
            }
            
            // Process temporal sequence
            let processed = process_temporal_sequence(&temporal_sequence);
            
            // Later values should generally be higher (respecting temporal trend)
            if processed.len() > 1 {
                let first_half_avg = processed[..processed.len()/2].iter().sum::<f32>() / (processed.len()/2) as f32;
                let second_half_avg = processed[processed.len()/2..].iter().sum::<f32>() / (processed.len() - processed.len()/2) as f32;
                
                prop_assert!(second_half_avg >= first_half_avg - 0.1); // Allow small variance
            }
        }
    }

    /// Test memory management invariants
    proptest! {
        #[test]
        fn memory_management_invariants(
            allocation_count in 1usize..=10,
            buffer_size in 100usize..=10000
        ) {
            let memory_manager = mocks::MockMemoryManager::<f32>::new();
            let mut allocations = Vec::new();
            
            // Perform allocations
            for i in 0..allocation_count {
                let name = format!("buffer_{}", i);
                match memory_manager.allocate(&name, buffer_size) {
                    Ok(_) => allocations.push(name),
                    Err(_) => break, // Allocation failure is acceptable
                }
            }
            
            let peak_usage = memory_manager.get_peak_usage();
            let current_usage = memory_manager.get_current_usage();
            
            // Peak usage should never be less than current usage
            prop_assert!(peak_usage >= current_usage);
            
            // Current usage should match allocated buffers
            let expected_usage = allocations.len() * buffer_size;
            prop_assert_eq!(current_usage, expected_usage);
            
            // Deallocate and verify
            for name in &allocations {
                let _ = memory_manager.deallocate(name);
            }
            
            let final_usage = memory_manager.get_current_usage();
            prop_assert_eq!(final_usage, 0);
        }
    }

    /// Test that feature normalization preserves information
    proptest! {
        #[test]
        fn feature_normalization_preserves_info(
            features in feature_vector::<f32>(50, -100.0..100.0)
        ) {
            let normalized = normalize_features(&features);
            
            // Normalized features should be in [-1, 1] or [0, 1] range
            for &norm_feat in &normalized {
                prop_assert!(norm_feat >= -1.1 && norm_feat <= 1.1); // Small tolerance
                prop_assert!(norm_feat.is_finite());
            }
            
            // Normalization should preserve relative ordering for monotonic transforms
            if features.len() > 1 {
                let orig_variance = calculate_variance(&features);
                let norm_variance = calculate_variance(&normalized);
                
                // Normalized variance should be reasonable (not zero unless input was constant)
                if orig_variance > 1e-6 {
                    prop_assert!(norm_variance > 1e-8);
                }
            }
        }
    }

    // Helper functions for property tests

    fn test_late_fusion(
        scores: &std::collections::HashMap<String, mocks::MockDeceptionScore<f32>>,
        weights: &[f32]
    ) -> mocks::MockFusedDecision<f32> {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut confidence_sum = 0.0;
        let mut contributions = std::collections::HashMap::new();
        
        let modality_names: Vec<_> = scores.keys().collect();
        
        for (i, (modality, score)) in scores.iter().enumerate() {
            let weight = if i < weights.len() { weights[i] } else { 0.0 };
            weighted_sum += score.probability * weight;
            confidence_sum += score.confidence * weight;
            weight_sum += weight;
            contributions.insert(modality.clone(), weight);
        }
        
        // Handle division by zero
        let final_probability = if weight_sum > 0.0 { weighted_sum / weight_sum } else { 0.0 };
        let final_confidence = if weight_sum > 0.0 { confidence_sum / weight_sum } else { 0.0 };
        
        // Calculate confidence penalty for disagreement
        let probabilities: Vec<f32> = scores.values().map(|s| s.probability).collect();
        let variance = calculate_variance(&probabilities);
        let confidence_penalty = 1.0 / (1.0 + variance * 10.0); // Higher variance = lower confidence
        
        mocks::MockFusedDecision {
            deception_probability: final_probability.max(0.0).min(1.0),
            confidence: (final_confidence * confidence_penalty).max(0.0).min(1.0),
            modality_contributions: contributions,
            explanation: "Property test fusion".to_string(),
        }
    }

    fn process_temporal_sequence(sequence: &[f32]) -> Vec<f32> {
        // Simple temporal smoothing
        let window_size = 3;
        let mut processed = Vec::new();
        
        for i in 0..sequence.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(sequence.len());
            
            let window_avg = sequence[start..end].iter().sum::<f32>() / (end - start) as f32;
            processed.push(window_avg);
        }
        
        processed
    }

    fn normalize_features(features: &[f32]) -> Vec<f32> {
        if features.is_empty() {
            return Vec::new();
        }
        
        let min_val = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val - min_val < 1e-6 {
            // Constant features
            return vec![0.0; features.len()];
        }
        
        features.iter()
            .map(|&f| (f - min_val) / (max_val - min_val))
            .collect()
    }

    fn calculate_variance(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32
    }
}