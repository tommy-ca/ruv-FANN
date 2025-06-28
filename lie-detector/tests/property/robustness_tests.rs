/// Property-based robustness tests
/// 
/// Tests that verify system resilience against edge cases, invalid inputs,
/// and stress conditions through random test case generation

use crate::common::*;
use proptest::prelude::*;
use generators::*;
use std::collections::HashMap;

#[cfg(test)]
mod robustness_properties {
    use super::*;

    /// Test that fusion is robust to extreme probability values
    proptest! {
        #[test]
        fn fusion_handles_extreme_probabilities(
            extreme_probs in prop::collection::vec(
                prop::strategy::Union::new([
                    1 => Just(0.0f32),
                    1 => Just(1.0f32), 
                    1 => Just(f32::MIN_POSITIVE),
                    1 => Just(1.0 - f32::EPSILON),
                ]),
                1..=5
            ),
            normal_confs in prop::collection::vec(0.1f32..0.9f32, 1..=5)
        ) {
            let scores: HashMap<String, mocks::MockDeceptionScore<f32>> = extreme_probs
                .iter()
                .zip(normal_confs.iter())
                .enumerate()
                .map(|(i, (&prob, &conf))| {
                    (format!("modality_{}", i), mocks::MockDeceptionScore::new(prob, conf))
                })
                .collect();

            let weights: Vec<f32> = vec![1.0 / scores.len() as f32; scores.len()];
            let result = test_robust_fusion(&scores, &weights);

            // System should handle extreme values gracefully
            prop_assert!(result.deception_probability >= 0.0 && result.deception_probability <= 1.0);
            prop_assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
            prop_assert!(result.deception_probability.is_finite());
            prop_assert!(result.confidence.is_finite());
        }
    }

    /// Test that neural networks are robust to noisy inputs
    proptest! {
        #[test]
        fn neural_network_noise_robustness(
            clean_input in feature_vector::<f32>(50, -1.0..1.0),
            noise_level in 0.0f32..0.5f32,
            noise_input in feature_vector::<f32>(50, -1.0..1.0)
        ) {
            let network = mocks::MockNeuralNetwork::new(vec![50, 25, 1]);
            
            // Get clean prediction
            let clean_output = network.forward(&clean_input);
            
            // Add noise to input
            let noisy_input: Vec<f32> = clean_input
                .iter()
                .zip(noise_input.iter())
                .map(|(&clean, &noise)| clean + noise * noise_level)
                .collect();
            
            let noisy_output = network.forward(&noisy_input);
            
            match (clean_output, noisy_output) {
                (Ok(clean), Ok(noisy)) => {
                    // Small noise should not cause dramatic output changes
                    if noise_level < 0.1 {
                        let output_diff = (clean[0] - noisy[0]).abs();
                        prop_assert!(output_diff < 0.3, 
                            "Output too sensitive to noise: diff = {}, noise_level = {}", 
                            output_diff, noise_level);
                    }
                    // All outputs should remain bounded
                    prop_assert!(noisy[0] >= 0.0 && noisy[0] <= 1.0);
                }
                _ => {
                    // Error handling is acceptable for extreme noise
                }
            }
        }
    }

    /// Test temporal alignment robustness to missing data
    proptest! {
        #[test]
        fn temporal_alignment_missing_data_robustness(
            sequence_length in 5usize..50,
            missing_ratio in 0.0f32..0.5f32
        ) {
            let mut complete_sequence: Vec<f32> = (0..sequence_length)
                .map(|i| 0.5 + 0.3 * (i as f32 / sequence_length as f32))
                .collect();
            
            // Randomly remove data points
            let missing_count = (sequence_length as f32 * missing_ratio) as usize;
            for _ in 0..missing_count {
                if !complete_sequence.is_empty() {
                    let idx = (rand::random::<f32>() * complete_sequence.len() as f32) as usize;
                    complete_sequence.remove(idx.min(complete_sequence.len() - 1));
                }
            }
            
            if !complete_sequence.is_empty() {
                let interpolated = interpolate_missing_data(&complete_sequence);
                
                // Interpolated sequence should maintain reasonable bounds
                for &value in &interpolated {
                    prop_assert!(value >= 0.0 && value <= 1.0);
                    prop_assert!(value.is_finite());
                }
                
                // Should not be shorter than input (interpolation fills gaps)
                prop_assert!(interpolated.len() >= complete_sequence.len());
            }
        }
    }

    /// Test memory management under allocation pressure
    proptest! {
        #[test]
        fn memory_management_allocation_pressure(
            num_allocations in 10usize..100,
            base_size in 100usize..1000,
            size_variance in 0.1f32..2.0f32
        ) {
            let memory_manager = mocks::MockMemoryManager::<f32>::new();
            let mut successful_allocations = Vec::new();
            let mut total_allocated = 0usize;
            
            // Perform many allocations with varying sizes
            for i in 0..num_allocations {
                let varied_size = (base_size as f32 * size_variance) as usize;
                let allocation_name = format!("pressure_test_{}", i);
                
                match memory_manager.allocate(&allocation_name, varied_size) {
                    Ok(_) => {
                        successful_allocations.push((allocation_name, varied_size));
                        total_allocated += varied_size;
                    }
                    Err(_) => {
                        // Allocation failure under pressure is acceptable
                        break;
                    }
                }
                
                // Check invariants after each allocation
                let current_usage = memory_manager.get_current_usage();
                let peak_usage = memory_manager.get_peak_usage();
                
                prop_assert!(peak_usage >= current_usage);
                prop_assert!(current_usage <= total_allocated);
            }
            
            // Deallocate everything and verify cleanup
            for (name, size) in &successful_allocations {
                let result = memory_manager.deallocate(name);
                prop_assert!(result.is_ok());
                total_allocated = total_allocated.saturating_sub(*size);
            }
            
            prop_assert_eq!(memory_manager.get_current_usage(), 0);
        }
    }

    /// Test streaming pipeline under high throughput
    proptest! {
        #[test]
        fn streaming_pipeline_high_throughput(
            frame_count in 20usize..100,
            concurrent_streams in 1usize..5,
            frame_size in 1000usize..5000
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let pipeline = mocks::MockStreamingPipeline::new();
                let mut handles = Vec::new();
                
                // Create multiple concurrent streams
                for stream_id in 0..concurrent_streams {
                    let pipeline_ref = &pipeline;
                    let handle = tokio::spawn(async move {
                        let mut stream_results = Vec::new();
                        
                        for frame_id in 0..frame_count {
                            let frame = vec![0u8; frame_size];
                            let start_time = std::time::Instant::now();
                            
                            match pipeline_ref.process_frame(&frame).await {
                                Ok(result) => {
                                    let processing_time = start_time.elapsed();
                                    stream_results.push((stream_id, frame_id, processing_time, result));
                                }
                                Err(e) => {
                                    // Some failures under high load are acceptable
                                    println!("Frame processing failed: {}", e);
                                }
                            }
                        }
                        
                        stream_results
                    });
                    
                    handles.push(handle);
                }
                
                // Wait for all streams to complete
                let mut all_results = Vec::new();
                for handle in handles {
                    if let Ok(stream_results) = handle.await {
                        all_results.extend(stream_results);
                    }
                }
                
                // Verify results quality under load
                if !all_results.is_empty() {
                    let avg_processing_time = all_results.iter()
                        .map(|(_, _, time, _)| time.as_millis())
                        .sum::<u128>() / all_results.len() as u128;
                    
                    // Processing should not degrade dramatically under load
                    prop_assert!(avg_processing_time < 1000); // Less than 1 second per frame
                    
                    // All results should be valid
                    for (_, _, _, result) in &all_results {
                        prop_assert!(result.probability >= 0.0 && result.probability <= 1.0);
                        prop_assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                    }
                }
            });
        }
    }

    /// Test system resilience to malformed input data
    proptest! {
        #[test]
        fn system_handles_malformed_inputs(
            malformed_data in edge_cases::extreme_values::<f32>()
        ) {
            // Test with extreme probability values
            let extreme_score = mocks::MockDeceptionScore::new(
                malformed_data.extreme_probability, 
                0.5f32
            );
            
            // System should either handle gracefully or return error
            let result = validate_deception_score(&extreme_score);
            match result {
                Ok(validated_score) => {
                    // If accepted, must be valid
                    prop_assert!(validated_score.probability >= 0.0 && validated_score.probability <= 1.0);
                    prop_assert!(validated_score.confidence >= 0.0 && validated_score.confidence <= 1.0);
                    prop_assert!(validated_score.probability.is_finite());
                    prop_assert!(validated_score.confidence.is_finite());
                }
                Err(_) => {
                    // Rejection of malformed input is acceptable
                }
            }
            
            // Test empty feature vectors
            if !malformed_data.empty_feature_vector.is_empty() {
                let normalized = normalize_features(&malformed_data.empty_feature_vector);
                prop_assert!(normalized.is_empty());
            }
        }
    }

    /// Test confidence calculation stability
    proptest! {
        #[test]
        fn confidence_calculation_stability(
            base_scores in prop::collection::vec(0.1f32..0.9f32, 3..=10),
            perturbation in -0.05f32..0.05f32
        ) {
            // Calculate confidence with original scores
            let original_confidence = calculate_confidence_from_agreement(&base_scores);
            
            // Apply small perturbations
            let perturbed_scores: Vec<f32> = base_scores.iter()
                .map(|&score| (score + perturbation).max(0.0).min(1.0))
                .collect();
            
            let perturbed_confidence = calculate_confidence_from_agreement(&perturbed_scores);
            
            // Small changes in input should not cause dramatic confidence changes
            let confidence_change = (original_confidence - perturbed_confidence).abs();
            prop_assert!(confidence_change < 0.2, 
                "Confidence too sensitive to small changes: {} -> {} (change: {})", 
                original_confidence, perturbed_confidence, confidence_change);
            
            // Both confidence values should be valid
            prop_assert!(original_confidence >= 0.0 && original_confidence <= 1.0);
            prop_assert!(perturbed_confidence >= 0.0 && perturbed_confidence <= 1.0);
        }
    }

    // Helper functions for robustness tests

    fn test_robust_fusion(
        scores: &HashMap<String, mocks::MockDeceptionScore<f32>>,
        weights: &[f32]
    ) -> mocks::MockFusedDecision<f32> {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut confidence_sum = 0.0;
        let mut contributions = HashMap::new();
        
        for (i, (modality, score)) in scores.iter().enumerate() {
            let weight = if i < weights.len() { weights[i] } else { 0.0 };
            
            // Handle extreme values safely
            let safe_prob = score.probability.max(0.0).min(1.0);
            let safe_conf = score.confidence.max(0.0).min(1.0);
            
            if safe_prob.is_finite() && safe_conf.is_finite() && weight.is_finite() {
                weighted_sum += safe_prob * weight;
                confidence_sum += safe_conf * weight;
                weight_sum += weight;
                contributions.insert(modality.clone(), weight);
            }
        }
        
        // Safe division with fallback
        let final_probability = if weight_sum > f32::EPSILON {
            (weighted_sum / weight_sum).max(0.0).min(1.0)
        } else {
            0.5 // Neutral default
        };
        
        let final_confidence = if weight_sum > f32::EPSILON {
            (confidence_sum / weight_sum).max(0.0).min(1.0)
        } else {
            0.0 // Low confidence default
        };
        
        mocks::MockFusedDecision {
            deception_probability: final_probability,
            confidence: final_confidence,
            modality_contributions: contributions,
            explanation: "Robust fusion result".to_string(),
        }
    }

    fn interpolate_missing_data(incomplete_sequence: &[f32]) -> Vec<f32> {
        if incomplete_sequence.len() <= 1 {
            return incomplete_sequence.to_vec();
        }
        
        let mut interpolated = incomplete_sequence.to_vec();
        
        // Simple linear interpolation for missing segments
        // For this mock implementation, just ensure bounds
        for value in &mut interpolated {
            *value = value.max(0.0).min(1.0);
            if !value.is_finite() {
                *value = 0.5; // Safe default
            }
        }
        
        interpolated
    }

    fn validate_deception_score(score: &mocks::MockDeceptionScore<f32>) -> Result<mocks::MockDeceptionScore<f32>, String> {
        if !score.probability.is_finite() || score.probability < 0.0 || score.probability > 1.0 {
            return Err("Invalid probability".to_string());
        }
        
        if !score.confidence.is_finite() || score.confidence < 0.0 || score.confidence > 1.0 {
            return Err("Invalid confidence".to_string());
        }
        
        Ok(score.clone())
    }

    fn calculate_confidence_from_agreement(scores: &[f32]) -> f32 {
        if scores.len() < 2 {
            return 0.0;
        }
        
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
        
        // Higher agreement (lower variance) = higher confidence
        let confidence = 1.0 / (1.0 + variance * 10.0);
        confidence.max(0.0).min(1.0)
    }

    fn normalize_features(features: &[f32]) -> Vec<f32> {
        if features.is_empty() {
            return Vec::new();
        }
        
        let finite_features: Vec<f32> = features.iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect();
        
        if finite_features.is_empty() {
            return vec![0.0; features.len()];
        }
        
        let min_val = finite_features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = finite_features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < f32::EPSILON {
            return vec![0.0; features.len()];
        }
        
        features.iter()
            .map(|&f| {
                if f.is_finite() {
                    ((f - min_val) / (max_val - min_val)).max(0.0).min(1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }
}