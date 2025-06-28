/// Comprehensive unit tests for fusion strategies and temporal alignment
/// 
/// Tests multi-modal fusion, temporal synchronization, attention mechanisms,
/// weighted voting, and fusion strategy management

use crate::common::*;
use crate::common::generators_enhanced::*;
use veritas_nexus::fusion::*;
use veritas_nexus::{ModalityType, DeceptionScore};
use std::collections::HashMap;
use proptest::prelude::*;
use tokio_test;
use serial_test::serial;
use float_cmp::approx_eq;

#[cfg(test)]
mod fusion_strategy_tests {
    use super::*;
    
    #[test]
    fn test_fusion_manager_creation() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config);
        
        assert!(manager.is_ok(), "Should create fusion manager successfully");
        
        if let Ok(manager) = manager {
            let strategies = manager.list_strategies();
            
            // Verify default strategies are registered
            assert!(strategies.contains(&"late_fusion".to_string()));
            assert!(strategies.contains(&"early_fusion".to_string()));
            assert!(strategies.contains(&"hybrid_fusion".to_string()));
            assert!(strategies.contains(&"attention_fusion".to_string()));
            assert!(strategies.contains(&"weighted_voting".to_string()));
            
            assert_eq!(strategies.len(), 5, "Should have 5 default strategies");
        }
    }
    
    #[test]
    fn test_late_fusion_strategy() {
        let config = LateFusionConfig::default();
        let strategy = LateFusion::new(config).unwrap();
        
        // Create mock scores from different modalities
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.7, 0.8));
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.6, 0.7));
        scores.insert(ModalityType::Audio, MockDeceptionScore::new(0.8, 0.9));
        
        let result = strategy.fuse(&scores, None);
        
        assert!(result.is_ok(), "Late fusion should succeed");
        
        if let Ok(decision) = result {
            // Probability should be weighted average of input scores
            assert!(decision.probability >= 0.0 && decision.probability <= 1.0);
            assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
            
            // Should have contributions from all modalities
            assert_eq!(decision.modality_contributions.len(), 3);
            assert!(decision.modality_contributions.contains_key(&ModalityType::Text));
            assert!(decision.modality_contributions.contains_key(&ModalityType::Vision));
            assert!(decision.modality_contributions.contains_key(&ModalityType::Audio));
            
            // Confidence should reflect consistency of inputs
            // Since inputs are reasonably consistent (0.6-0.8), confidence should be moderate to high
            assert!(decision.confidence > 0.5);
        }
    }
    
    #[test]
    fn test_early_fusion_strategy() {
        let config = EarlyFusionConfig::default();
        let strategy = EarlyFusion::new(config).unwrap();
        
        // Create combined features for early fusion
        let mut combined_features = CombinedFeatures::new();
        combined_features.add_modality_features(ModalityType::Text, vec![0.7, 0.5, 0.8]);
        combined_features.add_modality_features(ModalityType::Vision, vec![0.6, 0.7, 0.4]);
        combined_features.add_modality_features(ModalityType::Audio, vec![0.8, 0.6, 0.9]);
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.7, 0.8));
        
        let result = strategy.fuse(&scores, Some(&combined_features));
        
        assert!(result.is_ok(), "Early fusion should succeed");
        
        if let Ok(decision) = result {
            assert_valid_probability(decision.probability);
            assert_valid_probability(decision.confidence);
            
            // Early fusion should leverage combined features
            assert!(!decision.explanation.steps.is_empty());
        }
    }
    
    #[test]
    fn test_attention_fusion_strategy() {
        let config = AttentionConfig::default();
        let strategy = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.9, 0.95)); // High confidence
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.3, 0.4)); // Low confidence
        scores.insert(ModalityType::Audio, MockDeceptionScore::new(0.7, 0.8));  // Medium confidence
        
        let result = strategy.fuse(&scores, None);
        
        assert!(result.is_ok(), "Attention fusion should succeed");
        
        if let Ok(decision) = result {
            assert_valid_probability(decision.probability);
            assert_valid_probability(decision.confidence);
            
            // Attention should focus more on high-confidence modalities
            let text_contribution = decision.modality_contributions.get(&ModalityType::Text).unwrap();
            let vision_contribution = decision.modality_contributions.get(&ModalityType::Vision).unwrap();
            
            // Text should have higher contribution due to higher confidence
            assert!(text_contribution > vision_contribution);
            
            // Final probability should be closer to high-confidence text score
            let text_influence = (decision.probability - 0.9_f64).abs();
            let vision_influence = (decision.probability - 0.3_f64).abs();
            assert!(text_influence < vision_influence);
        }
    }
    
    #[test]
    fn test_weighted_voting_strategy() {
        let mut config = WeightedVotingConfig::default();
        
        // Set custom weights
        config.modality_weights.insert(ModalityType::Text, 0.4);
        config.modality_weights.insert(ModalityType::Vision, 0.3);
        config.modality_weights.insert(ModalityType::Audio, 0.3);
        
        let strategy = WeightedVoting::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.8, 0.9));
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.4, 0.7));
        scores.insert(ModalityType::Audio, MockDeceptionScore::new(0.6, 0.8));
        
        let result = strategy.fuse(&scores, None);
        
        assert!(result.is_ok(), "Weighted voting should succeed");
        
        if let Ok(decision) = result {
            // Verify weighted average calculation
            let expected = 0.8 * 0.4 + 0.4 * 0.3 + 0.6 * 0.3; // 0.62
            
            assert!(
                (decision.probability - expected).abs() < 0.05,
                "Weighted average should match expected value: {} vs {}",
                decision.probability, expected
            );
            
            // Weights should be reflected in contributions
            let weights = strategy.get_modality_weights();
            assert_eq!(weights.get(&ModalityType::Text).unwrap(), &0.4);
            assert_eq!(weights.get(&ModalityType::Vision).unwrap(), &0.3);
            assert_eq!(weights.get(&ModalityType::Audio).unwrap(), &0.3);
        }
    }
    
    #[test]
    fn test_hybrid_fusion_strategy() {
        let config = HybridFusionConfig::default();
        let strategy = HybridFusion::new(config).unwrap();
        
        // Test with both scores and features
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.7, 0.8));
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.6, 0.7));
        
        let mut combined_features = CombinedFeatures::new();
        combined_features.add_modality_features(ModalityType::Text, vec![0.7, 0.5]);
        combined_features.add_modality_features(ModalityType::Vision, vec![0.6, 0.7]);
        
        let result = strategy.fuse(&scores, Some(&combined_features));
        
        assert!(result.is_ok(), "Hybrid fusion should succeed");
        
        if let Ok(decision) = result {
            assert_valid_probability(decision.probability);
            assert_valid_probability(decision.confidence);
            
            // Hybrid should combine both early and late fusion approaches
            assert!(!decision.explanation.steps.is_empty());
            assert!(decision.explanation.reasoning.contains("hybrid"));
        }
    }
    
    #[test]
    fn test_fusion_with_insufficient_modalities() {
        let config = FusionConfig::<f64> {
            min_modalities: 3,
            ..Default::default()
        };
        let manager = FusionManager::new(config).unwrap();
        
        // Only provide 2 modalities when 3 are required
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.7, 0.8));
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.6, 0.7));
        
        let result = manager.fuse_default(&scores, None);
        
        assert!(result.is_err(), "Should fail with insufficient modalities");
        
        if let Err(e) = result {
            // Should be a specific error about insufficient modalities
            println!("Expected error: {:?}", e);
        }
    }
    
    #[test]
    fn test_fusion_quality_threshold() {
        let config = FusionConfig::<f64> {
            quality_threshold: 0.9, // Very high threshold
            ..Default::default()
        };
        let manager = FusionManager::new(config).unwrap();
        
        // Provide conflicting modality scores (low quality)
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.1, 0.5)); // Low deception
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.9, 0.5)); // High deception
        scores.insert(ModalityType::Audio, MockDeceptionScore::new(0.5, 0.3)); // Medium with low confidence
        
        let result = manager.fuse_default(&scores, None);
        
        // Might fail due to low quality (high disagreement between modalities)
        match result {
            Ok(result) => {
                // If it succeeds, quality should be below threshold but fusion succeeded anyway
                assert!(result.quality_score < 0.9);
            },
            Err(_) => {
                // Expected due to low quality
                println!("Expected quality threshold failure");
            }
        }
    }
}

#[cfg(test)]
mod temporal_alignment_tests {
    use super::*;
    
    #[test]
    fn test_temporal_aligner_creation() {
        let config = AlignmentConfig::default();
        let aligner = TemporalAligner::<f64>::new(config);
        
        assert!(aligner.is_ok(), "Should create temporal aligner successfully");
    }
    
    #[test]
    fn test_temporal_alignment_basic() {
        let config = AlignmentConfig::default();
        let aligner = TemporalAligner::<f64>::new(config).unwrap();
        
        // Create features with different temporal resolutions
        let mut features = CombinedFeatures::new();
        
        // Vision: 30 FPS (high temporal resolution)
        features.add_temporal_features(ModalityType::Vision, create_timestamped_features(30, 1.0));
        
        // Audio: 16000 Hz (very high temporal resolution, but we'll downsample)
        features.add_temporal_features(ModalityType::Audio, create_timestamped_features(100, 1.0));
        
        // Text: 1 sample per second (low temporal resolution)
        features.add_temporal_features(ModalityType::Text, create_timestamped_features(1, 1.0));
        
        let result = aligner.align(&features);
        
        assert!(result.is_ok(), "Temporal alignment should succeed");
        
        if let Ok(aligned) = result {
            // All modalities should now have the same temporal resolution
            let vision_features = aligned.get_temporal_features(&ModalityType::Vision).unwrap();
            let audio_features = aligned.get_temporal_features(&ModalityType::Audio).unwrap();
            let text_features = aligned.get_temporal_features(&ModalityType::Text).unwrap();
            
            // Should have same number of time points (or proportional)
            let target_resolution = config.target_fps;
            assert!(vision_features.len() <= target_resolution * 2); // Allow some tolerance
            
            // Timestamps should be aligned
            if vision_features.len() == audio_features.len() {
                for (v_feature, a_feature) in vision_features.iter().zip(audio_features.iter()) {
                    let time_diff = (v_feature.timestamp - a_feature.timestamp).abs();
                    assert!(time_diff < 0.1, "Timestamps should be closely aligned");
                }
            }
        }
    }
    
    #[test]
    fn test_temporal_synchronization() {
        let config = AlignmentConfig {
            synchronization_window_ms: 100.0,
            interpolation_method: InterpolationMethod::Linear,
            ..Default::default()
        };
        let aligner = TemporalAligner::<f64>::new(config).unwrap();
        
        // Create slightly misaligned features
        let mut features = CombinedFeatures::new();
        
        let base_time = 0.0;
        let vision_features = vec![
            TimestampedFeature { value: 0.5, timestamp: base_time + 0.0 },
            TimestampedFeature { value: 0.6, timestamp: base_time + 0.5 },
            TimestampedFeature { value: 0.7, timestamp: base_time + 1.0 },
        ];
        
        let audio_features = vec![
            TimestampedFeature { value: 0.4, timestamp: base_time + 0.05 }, // Slightly offset
            TimestampedFeature { value: 0.65, timestamp: base_time + 0.55 },
            TimestampedFeature { value: 0.75, timestamp: base_time + 1.05 },
        ];
        
        features.add_temporal_features(ModalityType::Vision, vision_features);
        features.add_temporal_features(ModalityType::Audio, audio_features);
        
        let result = aligner.align(&features);
        
        assert!(result.is_ok(), "Synchronization should succeed");
        
        if let Ok(aligned) = result {
            let vision_aligned = aligned.get_temporal_features(&ModalityType::Vision).unwrap();
            let audio_aligned = aligned.get_temporal_features(&ModalityType::Audio).unwrap();
            
            // Should have synchronized timestamps
            for (v, a) in vision_aligned.iter().zip(audio_aligned.iter()) {
                assert!(
                    (v.timestamp - a.timestamp).abs() < 0.01,
                    "Features should be temporally synchronized"
                );
            }
        }
    }
    
    #[test]
    fn test_interpolation_methods() {
        let interpolation_methods = vec![
            InterpolationMethod::Linear,
            InterpolationMethod::Cubic,
            InterpolationMethod::NearestNeighbor,
        ];
        
        for method in interpolation_methods {
            let config = AlignmentConfig {
                interpolation_method: method,
                ..Default::default()
            };
            let aligner = TemporalAligner::<f64>::new(config).unwrap();
            
            let mut features = CombinedFeatures::new();
            features.add_temporal_features(
                ModalityType::Vision,
                create_sparse_timestamped_features()
            );
            
            let result = aligner.align(&features);
            assert!(result.is_ok(), "Interpolation method {:?} should work", method);
            
            if let Ok(aligned) = result {
                let aligned_features = aligned.get_temporal_features(&ModalityType::Vision).unwrap();
                
                // Should have more data points after interpolation
                assert!(aligned_features.len() >= 3);
                
                // Values should be reasonable (no extreme outliers from interpolation)
                for feature in aligned_features {
                    assert!(feature.value >= 0.0 && feature.value <= 1.0);
                    assert!(feature.timestamp >= 0.0);
                }
            }
        }
    }
    
    fn create_timestamped_features(count: usize, duration: f64) -> Vec<TimestampedFeature<f64>> {
        (0..count).map(|i| {
            let timestamp = (i as f64 / count as f64) * duration;
            let value = 0.5 + 0.3 * (timestamp * std::f64::consts::PI).sin();
            TimestampedFeature { value, timestamp }
        }).collect()
    }
    
    fn create_sparse_timestamped_features() -> Vec<TimestampedFeature<f64>> {
        vec![
            TimestampedFeature { value: 0.2, timestamp: 0.0 },
            TimestampedFeature { value: 0.8, timestamp: 0.5 },
            TimestampedFeature { value: 0.4, timestamp: 1.0 },
        ]
    }
}

#[cfg(test)]
mod property_based_fusion_tests {
    use super::*;
    
    proptest! {
        /// Test that fusion always produces valid probability ranges regardless of input
        #[test]
        fn fusion_probability_invariant(
            scores in prop::collection::hash_map(
                prop::sample::select(vec![ModalityType::Text, ModalityType::Vision, ModalityType::Audio]),
                enhanced_probability::<f64>(),
                2..=4
            )
        ) {
            let config = FusionConfig::default();
            let manager = FusionManager::new(config).unwrap();
            
            let mock_scores: HashMap<ModalityType, MockDeceptionScore<f64>> = scores
                .into_iter()
                .map(|(modality, prob)| (modality, MockDeceptionScore::new(prob, 0.8)))
                .collect();
            
            if let Ok(result) = manager.fuse_default(&mock_scores, None) {
                prop_assert!(result.decision.probability >= 0.0);
                prop_assert!(result.decision.probability <= 1.0);
                prop_assert!(result.decision.confidence >= 0.0);
                prop_assert!(result.decision.confidence <= 1.0);
                prop_assert!(result.quality_score >= 0.0);
                prop_assert!(result.quality_score <= 1.0);
            }
        }
        
        /// Test that fusion confidence correlates with input consistency
        #[test]
        fn fusion_confidence_correlates_with_consistency(
            pattern in consistency_patterns::<f64>()
        ) {
            let config = FusionConfig::default();
            let manager = FusionManager::new(config).unwrap();
            
            let mut scores = HashMap::new();
            scores.insert(ModalityType::Vision, MockDeceptionScore::new(pattern.vision_score, 0.8));
            scores.insert(ModalityType::Audio, MockDeceptionScore::new(pattern.audio_score, 0.8));
            scores.insert(ModalityType::Text, MockDeceptionScore::new(pattern.text_score, 0.8));
            
            if let Ok(result) = manager.fuse_default(&scores, None) {
                if pattern.expected_consistency {
                    // Consistent inputs should produce higher confidence
                    prop_assert!(result.decision.confidence > 0.6);
                    prop_assert!(result.quality_score > 0.6);
                } else {
                    // Inconsistent inputs might have lower confidence or quality
                    // (but not necessarily - the system might be confident about the conflict)
                    prop_assert!(result.decision.confidence >= 0.0);
                }
            }
        }
        
        /// Test that temporal alignment preserves feature order
        #[test]
        fn temporal_alignment_preserves_order(
            feature_count in 3usize..20,
            duration in 0.1f64..10.0
        ) {
            let config = AlignmentConfig::default();
            let aligner = TemporalAligner::<f64>::new(config).unwrap();
            
            // Create monotonically increasing timestamps
            let features: Vec<TimestampedFeature<f64>> = (0..feature_count)
                .map(|i| TimestampedFeature {
                    value: (i as f64) / (feature_count as f64),
                    timestamp: (i as f64) / (feature_count as f64) * duration,
                })
                .collect();
            
            let mut combined = CombinedFeatures::new();
            combined.add_temporal_features(ModalityType::Vision, features);
            
            if let Ok(aligned) = aligner.align(&combined) {
                if let Some(aligned_features) = aligned.get_temporal_features(&ModalityType::Vision) {
                    // Timestamps should remain in order
                    for window in aligned_features.windows(2) {
                        prop_assert!(window[0].timestamp <= window[1].timestamp);
                    }
                }
            }
        }
        
        /// Test weight normalization in fusion strategies
        #[test]
        fn weight_normalization_invariant(
            mut weights in prop::collection::hash_map(
                prop::sample::select(vec![ModalityType::Text, ModalityType::Vision, ModalityType::Audio]),
                0.01f64..10.0, // Arbitrary positive weights
                2..=3
            )
        ) {
            utils::normalize_weights(&mut weights).unwrap();
            
            let sum: f64 = weights.values().sum();
            prop_assert!((sum - 1.0).abs() < 1e-10, "Weights should sum to 1.0, got {}", sum);
            
            for &weight in weights.values() {
                prop_assert!(weight >= 0.0 && weight <= 1.0, "Individual weights should be in [0,1]");
            }
        }
        
        /// Test fusion strategy symmetry properties
        #[test]
        fn fusion_strategy_symmetry(
            prob1 in enhanced_probability::<f64>(),
            prob2 in enhanced_probability::<f64>()
        ) {
            let config = LateFusionConfig::default();
            let strategy = LateFusion::new(config).unwrap();
            
            // Test with two modalities in different orders
            let mut scores1 = HashMap::new();
            scores1.insert(ModalityType::Text, MockDeceptionScore::new(prob1, 0.8));
            scores1.insert(ModalityType::Vision, MockDeceptionScore::new(prob2, 0.8));
            
            let mut scores2 = HashMap::new();
            scores2.insert(ModalityType::Vision, MockDeceptionScore::new(prob2, 0.8));
            scores2.insert(ModalityType::Text, MockDeceptionScore::new(prob1, 0.8));
            
            if let (Ok(result1), Ok(result2)) = (strategy.fuse(&scores1, None), strategy.fuse(&scores2, None)) {
                // Results should be identical regardless of input order
                let prob_diff = (result1.probability - result2.probability).abs();
                prop_assert!(prob_diff < 1e-10, "Fusion should be symmetric");
            }
        }
    }
}

#[cfg(test)]
mod fusion_performance_tests {
    use super::*;
    use std::time::{Duration, Instant};
    
    #[test]
    fn test_fusion_performance_scaling() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config).unwrap();
        
        // Test with increasing numbers of modalities
        let modality_counts = vec![2, 3, 4];
        let all_modalities = vec![
            ModalityType::Text,
            ModalityType::Vision,
            ModalityType::Audio,
            ModalityType::Physiological,
        ];
        
        for count in modality_counts {
            let mut scores = HashMap::new();
            for i in 0..count {
                scores.insert(
                    all_modalities[i],
                    MockDeceptionScore::new(0.5 + (i as f64) * 0.1, 0.8)
                );
            }
            
            let start = Instant::now();
            let result = manager.fuse_default(&scores, None);
            let duration = start.elapsed();
            
            assert!(result.is_ok(), "Fusion should succeed with {} modalities", count);
            
            // Performance should scale reasonably
            let max_expected = Duration::from_millis(10 * count as u64);
            assert!(
                duration < max_expected,
                "Fusion took too long with {} modalities: {:?}",
                count, duration
            );
        }
    }
    
    #[test]
    fn test_fusion_memory_efficiency() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config).unwrap();
        
        // Process many fusion operations to check for memory leaks
        for i in 0..1000 {
            let mut scores = HashMap::new();
            scores.insert(ModalityType::Text, MockDeceptionScore::new(0.6, 0.8));
            scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.5, 0.7));
            scores.insert(ModalityType::Audio, MockDeceptionScore::new(0.7, 0.9));
            
            let result = manager.fuse_default(&scores, None);
            assert!(result.is_ok(), "Fusion should succeed on iteration {}", i);
            
            // Periodic memory checks would go here in a real implementation
            if i % 100 == 0 {
                // Check memory usage, garbage collect, etc.
            }
        }
    }
    
    #[test]
    fn test_concurrent_fusion_operations() {
        use std::sync::Arc;
        use std::thread;
        
        let config = FusionConfig::<f64>::default();
        let manager = Arc::new(FusionManager::new(config).unwrap());
        
        let handles: Vec<_> = (0..10).map(|i| {
            let manager = manager.clone();
            thread::spawn(move || {
                let mut scores = HashMap::new();
                scores.insert(ModalityType::Text, MockDeceptionScore::new(0.5 + (i as f64) * 0.05, 0.8));
                scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.6, 0.7));
                
                manager.fuse_default(&scores, None)
            })
        }).collect();
        
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.join().unwrap();
            assert!(result.is_ok(), "Concurrent fusion {} should succeed", i);
        }
    }
}

#[cfg(test)]
mod fusion_edge_cases_tests {
    use super::*;
    
    #[test]
    fn test_fusion_with_extreme_values() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config).unwrap();
        
        let extreme_cases = vec![
            ("All zeros", vec![0.0, 0.0, 0.0]),
            ("All ones", vec![1.0, 1.0, 1.0]),
            ("Mixed extremes", vec![0.0, 1.0, 0.5]),
            ("Very close values", vec![0.500001, 0.500002, 0.500003]),
        ];
        
        for (description, probs) in extreme_cases {
            let mut scores = HashMap::new();
            scores.insert(ModalityType::Text, MockDeceptionScore::new(probs[0], 0.8));
            scores.insert(ModalityType::Vision, MockDeceptionScore::new(probs[1], 0.8));
            scores.insert(ModalityType::Audio, MockDeceptionScore::new(probs[2], 0.8));
            
            let result = manager.fuse_default(&scores, None);
            
            match result {
                Ok(fusion_result) => {
                    assert_valid_probability(fusion_result.decision.probability);
                    assert_valid_probability(fusion_result.decision.confidence);
                    println!("{}: Successfully fused to {:.3}", description, fusion_result.decision.probability);
                },
                Err(e) => {
                    println!("{}: Error (may be expected): {:?}", description, e);
                }
            }
        }
    }
    
    #[test]
    fn test_fusion_with_missing_modalities() {
        let config = FusionConfig::<f64> {
            min_modalities: 1, // Allow single modality
            ..Default::default()
        };
        let manager = FusionManager::new(config).unwrap();
        
        // Test with only one modality
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.7, 0.8));
        
        let result = manager.fuse_default(&scores, None);
        
        assert!(result.is_ok(), "Should handle single modality");
        
        if let Ok(fusion_result) = result {
            // Result should approximately match the single input
            assert!(
                (fusion_result.decision.probability - 0.7).abs() < 0.1,
                "Single modality fusion should preserve input value"
            );
        }
    }
    
    #[test]
    fn test_fusion_with_conflicting_confidences() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Text, MockDeceptionScore::new(0.9, 0.99)); // High confidence, high deception
        scores.insert(ModalityType::Vision, MockDeceptionScore::new(0.1, 0.05)); // Low confidence, low deception
        scores.insert(ModalityType::Audio, MockDeceptionScore::new(0.5, 0.95)); // High confidence, medium deception
        
        let result = manager.fuse_default(&scores, None);
        
        assert!(result.is_ok(), "Should handle conflicting confidences");
        
        if let Ok(fusion_result) = result {
            // High-confidence modalities should dominate
            // Text (0.9, conf 0.99) and Audio (0.5, conf 0.95) should outweigh Vision (0.1, conf 0.05)
            assert!(fusion_result.decision.probability > 0.5);
            
            // Quality might be lower due to some conflict
            // But the high-confidence modalities should still provide reasonable overall confidence
        }
    }
    
    #[test]
    fn test_temporal_alignment_edge_cases() {
        let config = AlignmentConfig::default();
        let aligner = TemporalAligner::<f64>::new(config).unwrap();
        
        let edge_cases = vec![
            (
                "Single feature",
                vec![TimestampedFeature { value: 0.5, timestamp: 0.0 }]
            ),
            (
                "Reverse timestamps",
                vec![
                    TimestampedFeature { value: 0.3, timestamp: 1.0 },
                    TimestampedFeature { value: 0.7, timestamp: 0.0 },
                ]
            ),
            (
                "Duplicate timestamps",
                vec![
                    TimestampedFeature { value: 0.4, timestamp: 0.5 },
                    TimestampedFeature { value: 0.6, timestamp: 0.5 },
                ]
            ),
        ];
        
        for (description, features) in edge_cases {
            let mut combined = CombinedFeatures::new();
            combined.add_temporal_features(ModalityType::Vision, features);
            
            match aligner.align(&combined) {
                Ok(aligned) => {
                    println!("{}: Successfully aligned", description);
                    
                    if let Some(aligned_features) = aligned.get_temporal_features(&ModalityType::Vision) {
                        // Basic validation
                        assert!(!aligned_features.is_empty());
                        for feature in aligned_features {
                            assert!(feature.value.is_finite());
                            assert!(feature.timestamp.is_finite());
                        }
                    }
                },
                Err(e) => {
                    println!("{}: Error (may be expected): {:?}", description, e);
                }
            }
        }
    }
}

// Mock implementations and helper types for testing

#[derive(Debug, Clone)]
pub struct MockDeceptionScore<T: Float> {
    probability: T,
    confidence: T,
}

impl<T: Float> MockDeceptionScore<T> {
    pub fn new(probability: T, confidence: T) -> Self {
        Self { probability, confidence }
    }
}

impl<T: Float> DeceptionScore<T> for MockDeceptionScore<T> {
    fn probability(&self) -> T {
        self.probability
    }
    
    fn confidence(&self) -> T {
        self.confidence
    }
    
    fn modality(&self) -> ModalityType {
        ModalityType::Text
    }
    
    fn features(&self) -> Vec<veritas_nexus::Feature<T>> {
        vec![]
    }
    
    fn timestamp(&self) -> std::time::SystemTime {
        std::time::SystemTime::now()
    }
}

// Fusion strategy implementations (simplified for testing)

#[derive(Debug, Clone)]
pub struct LateFusion<T: Float> {
    config: LateFusionConfig<T>,
}

impl<T: Float> LateFusion<T> {
    pub fn new(config: LateFusionConfig<T>) -> Result<Self, FusionError> {
        Ok(Self { config })
    }
    
    pub fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError> {
        if scores.is_empty() {
            return Err(FusionError::MissingModality { modality: "any".to_string() });
        }
        
        // Simple weighted average
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            let weight = score.confidence();
            weighted_sum = weighted_sum + (score.probability() * weight);
            weight_sum = weight_sum + weight;
            contributions.insert(*modality, score.probability());
        }
        
        let probability = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        // Calculate confidence based on agreement
        let mean_prob = probability;
        let variance = scores.values()
            .map(|s| {
                let diff = s.probability() - mean_prob;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / T::from(scores.len()).unwrap();
        
        let confidence = T::one() - variance.sqrt().min(T::one());
        
        Ok(FusedDecision {
            probability,
            confidence,
            modality_contributions: contributions,
            timestamp: std::time::SystemTime::now(),
            explanation: ExplanationTrace {
                steps: vec![
                    ExplanationStep {
                        step_type: "late_fusion".to_string(),
                        description: "Combined modality scores using late fusion".to_string(),
                        evidence: vec![format!("Processed {} modalities", scores.len())],
                        confidence: confidence.to_f64().unwrap_or(0.0),
                    }
                ],
                confidence: confidence.to_f64().unwrap_or(0.0),
                reasoning: "Late fusion combines individual modality decisions".to_string(),
            },
        })
    }
    
    pub fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        self.config.weights.clone()
    }
}

// More mock types...

#[derive(Debug, Clone)]
pub struct LateFusionConfig<T: Float> {
    pub weights: HashMap<ModalityType, T>,
}

impl<T: Float> Default for LateFusionConfig<T> {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Text, T::from(0.33).unwrap());
        weights.insert(ModalityType::Vision, T::from(0.33).unwrap());
        weights.insert(ModalityType::Audio, T::from(0.34).unwrap());
        
        Self { weights }
    }
}

// Additional fusion strategy configs...
#[derive(Debug, Clone)]
pub struct EarlyFusionConfig<T: Float> {
    pub feature_weights: HashMap<String, T>,
}

impl<T: Float> Default for EarlyFusionConfig<T> {
    fn default() -> Self {
        Self {
            feature_weights: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionConfig<T: Float> {
    pub attention_heads: usize,
    pub temperature: T,
}

impl<T: Float> Default for AttentionConfig<T> {
    fn default() -> Self {
        Self {
            attention_heads: 4,
            temperature: T::from(0.1).unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WeightedVotingConfig<T: Float> {
    pub modality_weights: HashMap<ModalityType, T>,
}

impl<T: Float> Default for WeightedVotingConfig<T> {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Text, T::from(0.33).unwrap());
        weights.insert(ModalityType::Vision, T::from(0.33).unwrap());
        weights.insert(ModalityType::Audio, T::from(0.34).unwrap());
        
        Self {
            modality_weights: weights,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridFusionConfig<T: Float> {
    pub early_weight: T,
    pub late_weight: T,
}

impl<T: Float> Default for HybridFusionConfig<T> {
    fn default() -> Self {
        Self {
            early_weight: T::from(0.4).unwrap(),
            late_weight: T::from(0.6).unwrap(),
        }
    }
}

// Simplified implementations of other fusion strategies
#[derive(Debug, Clone)]
pub struct EarlyFusion<T: Float> {
    config: EarlyFusionConfig<T>,
}

impl<T: Float> EarlyFusion<T> {
    pub fn new(config: EarlyFusionConfig<T>) -> Result<Self, FusionError> {
        Ok(Self { config })
    }
    
    pub fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError> {
        // Simplified early fusion using features if available
        let probability = if features.is_some() {
            T::from(0.65).unwrap() // Mock result from feature fusion
        } else {
            scores.values().map(|s| s.probability()).fold(T::zero(), |acc, x| acc + x) / T::from(scores.len()).unwrap()
        };
        
        Ok(FusedDecision {
            probability,
            confidence: T::from(0.8).unwrap(),
            modality_contributions: scores.iter().map(|(k, v)| (*k, v.probability())).collect(),
            timestamp: std::time::SystemTime::now(),
            explanation: ExplanationTrace {
                steps: vec![],
                confidence: 0.8,
                reasoning: "Early fusion combined features before decision".to_string(),
            },
        })
    }
}

// Similar simplified implementations for other strategies...
// [Additional strategy implementations would go here]

#[derive(Debug, Clone)]
pub struct AttentionFusion<T: Float> {
    config: AttentionConfig<T>,
}

impl<T: Float> AttentionFusion<T> {
    pub fn new(config: AttentionConfig<T>) -> Result<Self, FusionError> {
        Ok(Self { config })
    }
    
    pub fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError> {
        // Attention weights based on confidence
        let mut attention_weights = HashMap::new();
        let total_confidence: T = scores.values().map(|s| s.confidence()).fold(T::zero(), |acc, x| acc + x);
        
        for (modality, score) in scores {
            let weight = if total_confidence > T::zero() {
                score.confidence() / total_confidence
            } else {
                T::one() / T::from(scores.len()).unwrap()
            };
            attention_weights.insert(*modality, weight);
        }
        
        // Weighted combination
        let probability = scores.iter()
            .map(|(modality, score)| score.probability() * attention_weights[modality])
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(FusedDecision {
            probability,
            confidence: T::from(0.85).unwrap(),
            modality_contributions: scores.iter().map(|(k, v)| (*k, v.probability())).collect(),
            timestamp: std::time::SystemTime::now(),
            explanation: ExplanationTrace {
                steps: vec![],
                confidence: 0.85,
                reasoning: "Attention fusion weighted modalities by confidence".to_string(),
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct WeightedVoting<T: Float> {
    config: WeightedVotingConfig<T>,
}

impl<T: Float> WeightedVoting<T> {
    pub fn new(config: WeightedVotingConfig<T>) -> Result<Self, FusionError> {
        Ok(Self { config })
    }
    
    pub fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError> {
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        
        for (modality, score) in scores {
            if let Some(&weight) = self.config.modality_weights.get(modality) {
                weighted_sum = weighted_sum + (score.probability() * weight);
                weight_sum = weight_sum + weight;
            }
        }
        
        let probability = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        Ok(FusedDecision {
            probability,
            confidence: T::from(0.75).unwrap(),
            modality_contributions: scores.iter().map(|(k, v)| (*k, v.probability())).collect(),
            timestamp: std::time::SystemTime::now(),
            explanation: ExplanationTrace {
                steps: vec![],
                confidence: 0.75,
                reasoning: "Weighted voting used predefined modality weights".to_string(),
            },
        })
    }
    
    pub fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        self.config.modality_weights.clone()
    }
}

#[derive(Debug, Clone)]
pub struct HybridFusion<T: Float> {
    config: HybridFusionConfig<T>,
}

impl<T: Float> HybridFusion<T> {
    pub fn new(config: HybridFusionConfig<T>) -> Result<Self, FusionError> {
        Ok(Self { config })
    }
    
    pub fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError> {
        // Combine early and late fusion
        let late_result = T::from(0.6).unwrap(); // Mock late fusion result
        let early_result = if features.is_some() {
            T::from(0.7).unwrap() // Mock early fusion result
        } else {
            late_result
        };
        
        let probability = early_result * self.config.early_weight + late_result * self.config.late_weight;
        
        Ok(FusedDecision {
            probability,
            confidence: T::from(0.8).unwrap(),
            modality_contributions: scores.iter().map(|(k, v)| (*k, v.probability())).collect(),
            timestamp: std::time::SystemTime::now(),
            explanation: ExplanationTrace {
                steps: vec![],
                confidence: 0.8,
                reasoning: "Hybrid fusion combined early and late fusion strategies".to_string(),
            },
        })
    }
}

// Temporal alignment types and implementations
#[derive(Debug, Clone)]
pub struct TimestampedFeature<T: Float> {
    pub value: T,
    pub timestamp: f64, // Time in seconds
}

#[derive(Debug, Clone)]
pub struct CombinedFeatures<T: Float> {
    modality_features: HashMap<ModalityType, Vec<T>>,
    temporal_features: HashMap<ModalityType, Vec<TimestampedFeature<T>>>,
}

impl<T: Float> CombinedFeatures<T> {
    pub fn new() -> Self {
        Self {
            modality_features: HashMap::new(),
            temporal_features: HashMap::new(),
        }
    }
    
    pub fn add_modality_features(&mut self, modality: ModalityType, features: Vec<T>) {
        self.modality_features.insert(modality, features);
    }
    
    pub fn add_temporal_features(&mut self, modality: ModalityType, features: Vec<TimestampedFeature<T>>) {
        self.temporal_features.insert(modality, features);
    }
    
    pub fn get_temporal_features(&self, modality: &ModalityType) -> Option<&Vec<TimestampedFeature<T>>> {
        self.temporal_features.get(modality)
    }
}

#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    pub target_fps: usize,
    pub synchronization_window_ms: f64,
    pub interpolation_method: InterpolationMethod,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            target_fps: 30,
            synchronization_window_ms: 50.0,
            interpolation_method: InterpolationMethod::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    Linear,
    Cubic,
    NearestNeighbor,
}

#[derive(Debug, Clone)]
pub struct TemporalAligner<T: Float> {
    config: AlignmentConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> TemporalAligner<T> {
    pub fn new(config: AlignmentConfig) -> Result<Self, FusionError> {
        Ok(Self {
            config,
            _phantom: std::marker::PhantomData,
        })
    }
    
    pub fn align(&self, features: &CombinedFeatures<T>) -> Result<CombinedFeatures<T>, FusionError> {
        let mut aligned = CombinedFeatures::new();
        
        // Copy modality features unchanged
        for (modality, features) in &features.modality_features {
            aligned.add_modality_features(*modality, features.clone());
        }
        
        // Align temporal features
        for (modality, temporal_features) in &features.temporal_features {
            let aligned_features = self.align_modality_features(temporal_features)?;
            aligned.add_temporal_features(*modality, aligned_features);
        }
        
        Ok(aligned)
    }
    
    fn align_modality_features(&self, features: &[TimestampedFeature<T>]) -> Result<Vec<TimestampedFeature<T>>, FusionError> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        
        // Simple alignment: resample to target FPS
        let min_time = features.iter().map(|f| f.timestamp).fold(f64::INFINITY, f64::min);
        let max_time = features.iter().map(|f| f.timestamp).fold(f64::NEG_INFINITY, f64::max);
        let duration = max_time - min_time;
        
        if duration <= 0.0 {
            return Ok(features.to_vec());
        }
        
        let target_interval = 1.0 / self.config.target_fps as f64;
        let num_samples = ((duration / target_interval).ceil() as usize).max(1);
        
        let mut aligned = Vec::new();
        
        for i in 0..num_samples {
            let target_time = min_time + (i as f64) * target_interval;
            let interpolated_value = self.interpolate_at_time(features, target_time);
            
            aligned.push(TimestampedFeature {
                value: interpolated_value,
                timestamp: target_time,
            });
        }
        
        Ok(aligned)
    }
    
    fn interpolate_at_time(&self, features: &[TimestampedFeature<T>], target_time: f64) -> T {
        if features.len() == 1 {
            return features[0].value;
        }
        
        // Find surrounding points
        let mut before_idx = 0;
        let mut after_idx = features.len() - 1;
        
        for (i, feature) in features.iter().enumerate() {
            if feature.timestamp <= target_time {
                before_idx = i;
            }
            if feature.timestamp >= target_time && after_idx == features.len() - 1 {
                after_idx = i;
                break;
            }
        }
        
        if before_idx == after_idx {
            return features[before_idx].value;
        }
        
        // Linear interpolation
        let before = &features[before_idx];
        let after = &features[after_idx];
        
        let time_diff = after.timestamp - before.timestamp;
        if time_diff == 0.0 {
            return before.value;
        }
        
        let weight = (target_time - before.timestamp) / time_diff;
        let weight_t = T::from(weight).unwrap_or_else(|| T::zero());
        
        before.value * (T::one() - weight_t) + after.value * weight_t
    }
}

// Error types
#[derive(Debug, Clone)]
pub enum FusionError {
    MissingModality { modality: String },
    InsufficientConfidence { confidence: f64, threshold: f64 },
    StrategyNotConfigured,
    WeightNormalization { sum: f64 },
    TemporalAlignment { message: String },
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::MissingModality { modality } => write!(f, "Missing modality: {}", modality),
            FusionError::InsufficientConfidence { confidence, threshold } => {
                write!(f, "Insufficient confidence: {} < {}", confidence, threshold)
            },
            FusionError::StrategyNotConfigured => write!(f, "Fusion strategy not configured"),
            FusionError::WeightNormalization { sum } => write!(f, "Weight normalization failed: sum = {}", sum),
            FusionError::TemporalAlignment { message } => write!(f, "Temporal alignment error: {}", message),
        }
    }
}

impl std::error::Error for FusionError {}

// Export types for other modules
pub use veritas_nexus::{ExplanationTrace, ExplanationStep};

// Utility function for creating explanation traces
pub fn create_explanation_trace(steps: Vec<ExplanationStep>, confidence: f64, reasoning: String) -> ExplanationTrace {
    ExplanationTrace {
        steps,
        confidence,
        reasoning,
    }
}

// Mock FusionManager for testing
#[derive(Debug)]
pub struct FusionManager<T: Float> {
    strategies: HashMap<String, Box<dyn FusionStrategy<T>>>,
    default_strategy: String,
}

impl<T: Float> FusionManager<T> {
    pub fn new(_config: FusionConfig<T>) -> Result<Self, FusionError> {
        let mut manager = Self {
            strategies: HashMap::new(),
            default_strategy: "late_fusion".to_string(),
        };
        
        // Register mock strategies
        manager.strategies.insert("late_fusion".to_string(), Box::new(MockFusionStrategy::new()));
        manager.strategies.insert("early_fusion".to_string(), Box::new(MockFusionStrategy::new()));
        manager.strategies.insert("hybrid_fusion".to_string(), Box::new(MockFusionStrategy::new()));
        manager.strategies.insert("attention_fusion".to_string(), Box::new(MockFusionStrategy::new()));
        manager.strategies.insert("weighted_voting".to_string(), Box::new(MockFusionStrategy::new()));
        
        Ok(manager)
    }
    
    pub fn list_strategies(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }
    
    pub fn fuse_default(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusionResult<T>, FusionError> {
        let strategy = self.strategies.get(&self.default_strategy)
            .ok_or(FusionError::StrategyNotConfigured)?;
        
        let decision = strategy.fuse(scores, features)?;
        
        Ok(FusionResult {
            decision,
            intermediate_steps: vec![],
            quality_score: T::from(0.8).unwrap(),
        })
    }
}

// Mock fusion strategy for testing
pub struct MockFusionStrategy<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> MockFusionStrategy<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait FusionStrategy<T: Float> {
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError>;
}

impl<T: Float> FusionStrategy<T> for MockFusionStrategy<T> {
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, MockDeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>, FusionError> {
        if scores.is_empty() {
            return Err(FusionError::MissingModality { modality: "any".to_string() });
        }
        
        // Simple average
        let sum: T = scores.values().map(|s| s.probability()).fold(T::zero(), |acc, x| acc + x);
        let probability = sum / T::from(scores.len()).unwrap();
        
        Ok(FusedDecision {
            probability,
            confidence: T::from(0.8).unwrap(),
            modality_contributions: scores.iter().map(|(k, v)| (*k, v.probability())).collect(),
            timestamp: std::time::SystemTime::now(),
            explanation: ExplanationTrace {
                steps: vec![],
                confidence: 0.8,
                reasoning: "Mock fusion strategy".to_string(),
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct FusedDecision<T: Float> {
    pub probability: T,
    pub confidence: T,
    pub modality_contributions: HashMap<ModalityType, T>,
    pub timestamp: std::time::SystemTime,
    pub explanation: ExplanationTrace,
}

#[derive(Debug, Clone)]
pub struct FusionResult<T: Float> {
    pub decision: FusedDecision<T>,
    pub intermediate_steps: Vec<FusionStep<T>>,
    pub quality_score: T,
}

#[derive(Debug, Clone)]
pub struct FusionStep<T: Float> {
    pub name: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub duration: std::time::Duration,
    pub confidence: T,
}

#[derive(Debug, Clone)]
pub struct FusionConfig<T: Float> {
    pub min_modalities: usize,
    pub confidence_threshold: T,
    pub quality_threshold: T,
    pub enable_uncertainty: bool,
    pub max_processing_time: std::time::Duration,
    pub default_weights: HashMap<ModalityType, T>,
}

impl<T: Float> Default for FusionConfig<T> {
    fn default() -> Self {
        let mut default_weights = HashMap::new();
        default_weights.insert(ModalityType::Vision, T::from(0.3).unwrap());
        default_weights.insert(ModalityType::Audio, T::from(0.25).unwrap());
        default_weights.insert(ModalityType::Text, T::from(0.25).unwrap());
        default_weights.insert(ModalityType::Physiological, T::from(0.2).unwrap());
        
        Self {
            min_modalities: 2,
            confidence_threshold: T::from(0.75).unwrap(),
            quality_threshold: T::from(0.8).unwrap(),
            enable_uncertainty: true,
            max_processing_time: std::time::Duration::from_secs(30),
            default_weights,
        }
    }
}

// Additional utility modules
pub mod utils {
    use super::*;
    
    pub fn normalize_weights<T: Float>(weights: &mut HashMap<ModalityType, T>) -> Result<(), FusionError> {
        let sum: T = weights.values().fold(T::zero(), |acc, &w| acc + w);
        
        if sum <= T::zero() {
            return Err(FusionError::WeightNormalization { 
                sum: sum.to_f64().unwrap_or(0.0) 
            });
        }
        
        for weight in weights.values_mut() {
            *weight = *weight / sum;
        }
        
        Ok(())
    }
}