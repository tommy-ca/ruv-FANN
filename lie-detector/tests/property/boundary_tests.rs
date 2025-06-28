/// Property-based boundary tests
/// 
/// Tests that verify system behavior at boundary conditions, edge cases,
/// and transition points in the input space

use crate::common::*;
use proptest::prelude::*;
use generators::*;
use std::collections::HashMap;

#[cfg(test)]
mod boundary_properties {
    use super::*;

    /// Test behavior at probability boundaries (0.0, 1.0)
    proptest! {
        #[test]
        fn probability_boundary_behavior(
            boundary_prob in prop::strategy::Union::new([
                1 => Just(0.0f32),
                1 => Just(1.0f32),
                1 => Just(f32::EPSILON),
                1 => Just(1.0 - f32::EPSILON),
            ]),
            normal_conf in 0.1f32..0.9f32
        ) {
            let score = mocks::MockDeceptionScore::new(boundary_prob, normal_conf);
            
            // Test fusion at boundaries
            let mut scores = HashMap::new();
            scores.insert("boundary_test".to_string(), score);
            scores.insert("normal_test".to_string(), mocks::MockDeceptionScore::new(0.5, 0.7));
            
            let weights = vec![0.5, 0.5];
            let fused = test_boundary_fusion(&scores, &weights);
            
            // Result should be well-behaved at boundaries
            prop_assert!(fused.deception_probability >= 0.0 && fused.deception_probability <= 1.0);
            prop_assert!(fused.confidence >= 0.0 && fused.confidence <= 1.0);
            prop_assert!(fused.deception_probability.is_finite());
            prop_assert!(fused.confidence.is_finite());
            
            // Extreme values should influence the result appropriately
            if boundary_prob == 0.0 {
                prop_assert!(fused.deception_probability <= 0.6);
            } else if boundary_prob == 1.0 {
                prop_assert!(fused.deception_probability >= 0.4);
            }
        }
    }

    /// Test neural network behavior with minimal/maximal inputs
    proptest! {
        #[test]
        fn neural_network_input_boundaries(
            network_size in 1usize..=10,
            boundary_value in prop::strategy::Union::new([
                1 => Just(-1.0f32),
                1 => Just(1.0f32),
                1 => Just(0.0f32),
                1 => Just(f32::MIN),
                1 => Just(f32::MAX),
            ])
        ) {
            let network = mocks::MockNeuralNetwork::new(vec![network_size, network_size / 2 + 1, 1]);
            
            // Test with boundary values
            let boundary_input = vec![boundary_value; network_size];
            let result = network.forward(&boundary_input);
            
            match result {
                Ok(output) => {
                    // Output should be bounded and finite even with extreme inputs
                    prop_assert_eq!(output.len(), 1);
                    prop_assert!(output[0].is_finite() || boundary_value.is_infinite());
                    if output[0].is_finite() {
                        prop_assert!(output[0] >= 0.0 && output[0] <= 1.0);
                    }
                }
                Err(_) => {
                    // Rejection of extreme inputs is acceptable
                    prop_assert!(boundary_value.is_infinite() || boundary_value.is_nan());
                }
            }
        }
    }

    /// Test temporal alignment at sequence boundaries
    proptest! {
        #[test]
        fn temporal_sequence_boundaries(
            sequence_length in 1usize..=3, // Test very short sequences
            boundary_start in prop::strategy::Union::new([
                1 => Just(0.0f32),
                1 => Just(1.0f32),
            ]),
            boundary_end in prop::strategy::Union::new([
                1 => Just(0.0f32),
                1 => Just(1.0f32),
            ])
        ) {
            let mut sequence = vec![0.5f32; sequence_length];
            
            // Set boundary values
            if !sequence.is_empty() {
                sequence[0] = boundary_start;
                if sequence.len() > 1 {
                    sequence[sequence.len() - 1] = boundary_end;
                }
            }
            
            let aligned = align_temporal_sequence(&sequence);
            
            // Alignment should preserve sequence length
            prop_assert_eq!(aligned.len(), sequence.len());
            
            // All values should remain in bounds
            for &value in &aligned {
                prop_assert!(value >= 0.0 && value <= 1.0);
                prop_assert!(value.is_finite());
            }
            
            // Boundary values should be preserved or reasonably modified
            if !aligned.is_empty() {
                prop_assert!((aligned[0] - boundary_start).abs() <= 0.1);
                if aligned.len() > 1 {
                    prop_assert!((aligned[aligned.len() - 1] - boundary_end).abs() <= 0.1);
                }
            }
        }
    }

    /// Test confidence calculation at extreme agreement/disagreement
    proptest! {
        #[test]
        fn confidence_extreme_agreement(
            modality_count in 2usize..=5,
            agreement_type in prop::strategy::Union::new([
                1 => Just("perfect_agreement"),
                1 => Just("perfect_disagreement"), 
                1 => Just("single_outlier"),
            ])
        ) {
            let scores = match agreement_type {
                "perfect_agreement" => {
                    // All modalities agree perfectly
                    vec![0.7f32; modality_count]
                }
                "perfect_disagreement" => {
                    // Maximum disagreement
                    (0..modality_count)
                        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
                        .collect()
                }
                "single_outlier" => {
                    // One extreme outlier, others agree
                    let mut scores = vec![0.5f32; modality_count];
                    scores[0] = 1.0;
                    scores
                }
                _ => vec![0.5f32; modality_count]
            };
            
            let confidence = calculate_confidence_score(&scores);
            
            // Confidence should be in valid range
            prop_assert!(confidence >= 0.0 && confidence <= 1.0);
            prop_assert!(confidence.is_finite());
            
            // Perfect agreement should yield high confidence
            if agreement_type == "perfect_agreement" {
                prop_assert!(confidence >= 0.8);
            }
            
            // Perfect disagreement should yield low confidence
            if agreement_type == "perfect_disagreement" {
                prop_assert!(confidence <= 0.3);
            }
        }
    }

    /// Test feature normalization boundaries
    proptest! {
        #[test]
        fn feature_normalization_boundaries(
            feature_count in 1usize..=10,
            boundary_type in prop::strategy::Union::new([
                1 => Just("all_same"),
                1 => Just("extreme_range"),
                1 => Just("single_extreme"),
            ])
        ) {
            let features = match boundary_type {
                "all_same" => {
                    // All features identical
                    vec![0.5f32; feature_count]
                }
                "extreme_range" => {
                    // Maximum possible range
                    (0..feature_count)
                        .map(|i| if i % 2 == 0 { f32::MIN } else { f32::MAX })
                        .collect()
                }
                "single_extreme" => {
                    // One extreme value among normals
                    let mut features = vec![0.5f32; feature_count];
                    if !features.is_empty() {
                        features[0] = 1000.0;
                    }
                    features
                }
                _ => vec![0.5f32; feature_count]
            };
            
            let normalized = normalize_features_safe(&features);
            
            // Normalized features should be in [0, 1] range
            for &feature in &normalized {
                if feature.is_finite() {
                    prop_assert!(feature >= 0.0 && feature <= 1.0);
                }
            }
            
            // Length should be preserved
            prop_assert_eq!(normalized.len(), features.len());
            
            // Special cases should be handled appropriately
            if boundary_type == "all_same" && !features.is_empty() {
                // All same values should normalize to same output
                let first_value = normalized[0];
                for &value in &normalized {
                    prop_assert!((value - first_value).abs() < f32::EPSILON);
                }
            }
        }
    }

    /// Test streaming pipeline at rate boundaries
    proptest! {
        #[test]
        fn streaming_rate_boundaries(
            frame_rate in prop::strategy::Union::new([
                1 => Just(1u64),      // Very slow (1 FPS)
                1 => Just(1000u64),   // Very fast (1000 FPS equivalent latency)
            ]),
            frame_count in 1usize..=5
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let latency_ms = 1000 / frame_rate; // Convert FPS to latency
                let pipeline = mocks::MockStreamingPipeline::new().with_latency(latency_ms);
                
                let mut processing_times = Vec::new();
                let mut successful_frames = 0;
                
                for _ in 0..frame_count {
                    let frame = vec![0u8; 1024];
                    let start = std::time::Instant::now();
                    
                    match pipeline.process_frame(&frame).await {
                        Ok(result) => {
                            let duration = start.elapsed();
                            processing_times.push(duration);
                            successful_frames += 1;
                            
                            // Results should be valid regardless of rate
                            prop_assert!(result.probability >= 0.0 && result.probability <= 1.0);
                            prop_assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                        }
                        Err(_) => {
                            // Some failures at extreme rates are acceptable
                        }
                    }
                }
                
                // At least some frames should process successfully
                prop_assert!(successful_frames > 0);
                
                if !processing_times.is_empty() {
                    let avg_time = processing_times.iter().sum::<std::time::Duration>() 
                        / processing_times.len() as u32;
                    
                    // Processing time should be reasonable relative to target rate
                    if frame_rate <= 10 {
                        // Slow rates: processing should be much faster than frame period
                        prop_assert!(avg_time.as_millis() < 500);
                    }
                    // Fast rates: some delay is expected and acceptable
                }
            });
        }
    }

    /// Test memory allocation at size boundaries
    proptest! {
        #[test]
        fn memory_allocation_size_boundaries(
            allocation_size in prop::strategy::Union::new([
                1 => Just(0usize),
                1 => Just(1usize),
                1 => Just(usize::MAX / 1000000), // Large but not MAX
            ])
        ) {
            let memory_manager = mocks::MockMemoryManager::<f32>::new();
            let allocation_name = "boundary_test";
            
            let result = memory_manager.allocate(allocation_name, allocation_size);
            
            match result {
                Ok(buffer) => {
                    // Successful allocation should have correct size
                    prop_assert_eq!(buffer.len(), allocation_size);
                    
                    // All values should be initialized
                    for &value in &buffer {
                        prop_assert!(value.is_finite());
                    }
                    
                    // Memory tracking should be accurate
                    prop_assert_eq!(memory_manager.get_current_usage(), allocation_size);
                    
                    // Cleanup should work
                    let dealloc_result = memory_manager.deallocate(allocation_name);
                    prop_assert!(dealloc_result.is_ok());
                    prop_assert_eq!(memory_manager.get_current_usage(), 0);
                }
                Err(_) => {
                    // Rejection of extreme sizes is acceptable
                    prop_assert!(allocation_size == 0 || allocation_size > 1000000);
                }
            }
        }
    }

    /// Test fusion strategy with single modality (boundary case)
    proptest! {
        #[test]
        fn single_modality_fusion(
            single_prob in 0.0f32..1.0f32,
            single_conf in 0.0f32..1.0f32
        ) {
            let mut scores = HashMap::new();
            scores.insert("only_modality".to_string(), mocks::MockDeceptionScore::new(single_prob, single_conf));
            
            let weights = vec![1.0];
            let fused = test_boundary_fusion(&scores, &weights);
            
            // Single modality fusion should preserve the input
            prop_assert!((fused.deception_probability - single_prob).abs() < 0.01);
            prop_assert!((fused.confidence - single_conf).abs() < 0.01);
            prop_assert!(fused.deception_probability >= 0.0 && fused.deception_probability <= 1.0);
            prop_assert!(fused.confidence >= 0.0 && fused.confidence <= 1.0);
        }
    }

    // Helper functions for boundary tests

    fn test_boundary_fusion(
        scores: &HashMap<String, mocks::MockDeceptionScore<f32>>,
        weights: &[f32]
    ) -> mocks::MockFusedDecision<f32> {
        if scores.is_empty() {
            return mocks::MockFusedDecision {
                deception_probability: 0.5,
                confidence: 0.0,
                modality_contributions: HashMap::new(),
                explanation: "Empty input".to_string(),
            };
        }
        
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut confidence_sum = 0.0;
        let mut contributions = HashMap::new();
        
        for (i, (modality, score)) in scores.iter().enumerate() {
            let weight = if i < weights.len() { weights[i] } else { 1.0 / scores.len() as f32 };
            
            weighted_sum += score.probability * weight;
            confidence_sum += score.confidence * weight;
            weight_sum += weight;
            contributions.insert(modality.clone(), weight);
        }
        
        let final_probability = if weight_sum > 0.0 {
            (weighted_sum / weight_sum).max(0.0).min(1.0)
        } else {
            0.5
        };
        
        let final_confidence = if weight_sum > 0.0 {
            (confidence_sum / weight_sum).max(0.0).min(1.0)
        } else {
            0.0
        };
        
        mocks::MockFusedDecision {
            deception_probability: final_probability,
            confidence: final_confidence,
            modality_contributions: contributions,
            explanation: "Boundary fusion result".to_string(),
        }
    }

    fn align_temporal_sequence(sequence: &[f32]) -> Vec<f32> {
        if sequence.is_empty() {
            return Vec::new();
        }
        
        // Simple temporal alignment - ensure all values are in bounds
        sequence.iter()
            .map(|&value| value.max(0.0).min(1.0))
            .collect()
    }

    fn calculate_confidence_score(scores: &[f32]) -> f32 {
        if scores.len() < 2 {
            return if scores.is_empty() { 0.0 } else { 0.5 };
        }
        
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
        
        // Convert variance to confidence (lower variance = higher confidence)
        let confidence = (-variance * 5.0).exp(); // Exponential decay
        confidence.max(0.0).min(1.0)
    }

    fn normalize_features_safe(features: &[f32]) -> Vec<f32> {
        if features.is_empty() {
            return Vec::new();
        }
        
        // Filter out non-finite values for min/max calculation
        let finite_values: Vec<f32> = features.iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect();
        
        if finite_values.is_empty() {
            // All values are non-finite, return zeros
            return vec![0.0; features.len()];
        }
        
        let min_val = finite_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = finite_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let range = max_val - min_val;
        
        if range < f32::EPSILON {
            // All finite values are the same
            return features.iter()
                .map(|&x| if x.is_finite() { 0.0 } else { 0.0 })
                .collect();
        }
        
        features.iter()
            .map(|&x| {
                if x.is_finite() {
                    ((x - min_val) / range).max(0.0).min(1.0)
                } else {
                    0.0 // Replace non-finite with 0
                }
            })
            .collect()
    }
}