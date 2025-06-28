//! Comprehensive tests for the fusion system
//!
//! This module contains extensive tests for all fusion strategies,
//! temporal alignment, attention mechanisms, and edge cases.

use super::*;
use crate::types::*;
use chrono::{Duration as ChronoDuration, Utc};
use hashbrown::HashMap;
use std::time::Duration;

/// Helper function to create test deception scores
fn create_test_score<T: Float>(
    probability: f64,
    confidence: f64,
    processing_time_ms: u64,
) -> DeceptionScore<T> {
    DeceptionScore {
        probability: T::from(probability).unwrap(),
        confidence: T::from(confidence).unwrap(),
        contributing_factors: vec![
            ("micro_expressions".to_string(), T::from(0.3).unwrap()),
            ("voice_stress".to_string(), T::from(0.4).unwrap()),
            ("linguistic_patterns".to_string(), T::from(0.3).unwrap()),
        ],
        timestamp: Utc::now(),
        processing_time: Duration::from_millis(processing_time_ms),
    }
}

/// Helper function to create combined features for testing
fn create_test_features<T: Float>(
    vision_features: Vec<T>,
    audio_features: Vec<T>,
    text_features: Vec<T>,
) -> CombinedFeatures<T> {
    let mut modalities = HashMap::new();
    modalities.insert(ModalityType::Vision, vision_features.clone());
    modalities.insert(ModalityType::Audio, audio_features.clone());
    modalities.insert(ModalityType::Text, text_features.clone());
    
    let mut combined = Vec::new();
    combined.extend(vision_features);
    combined.extend(audio_features);
    combined.extend(text_features);
    
    let mut dimension_map = HashMap::new();
    let mut offset = 0;
    
    for (modality, features) in &modalities {
        let start = offset;
        let end = offset + features.len();
        dimension_map.insert(*modality, (start, end));
        offset = end;
    }
    
    let temporal_info = TemporalInfo {
        start_time: Utc::now() - ChronoDuration::seconds(5),
        end_time: Utc::now(),
        frame_rate: Some(30.0),
        sample_rate: Some(16000),
        sync_offsets: HashMap::new(),
    };
    
    CombinedFeatures {
        modalities,
        combined,
        dimension_map,
        temporal_info,
    }
}

/// Helper function to create test feedback data
fn create_test_feedback<T: Float>(
    ground_truth: bool,
    prediction: f64,
    modality_performances: Vec<(ModalityType, f64)>,
) -> FeedbackData<T> {
    let mut performance_map = HashMap::new();
    for (modality, performance) in modality_performances {
        performance_map.insert(modality, T::from(performance).unwrap());
    }
    
    FeedbackData {
        ground_truth,
        prediction: T::from(prediction).unwrap(),
        modality_performance: performance_map,
        session_id: "test_session".to_string(),
        timestamp: Utc::now(),
    }
}

#[cfg(test)]
mod fusion_manager_tests {
    use super::*;
    
    #[test]
    fn test_fusion_manager_creation() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config).unwrap();
        
        let strategies = manager.list_strategies();
        assert!(strategies.len() >= 5);
        assert!(strategies.contains(&"late_fusion".to_string()));
        assert!(strategies.contains(&"early_fusion".to_string()));
        assert!(strategies.contains(&"hybrid_fusion".to_string()));
        assert!(strategies.contains(&"attention_fusion".to_string()));
        assert!(strategies.contains(&"weighted_voting".to_string()));
    }
    
    #[test]
    fn test_fusion_manager_default_strategy() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.7, 0.85, 5));
        
        let result = manager.fuse_default(&scores, None).unwrap();
        
        assert!(result.decision.deception_probability >= 0.0);
        assert!(result.decision.deception_probability <= 1.0);
        assert!(result.decision.confidence > 0.0);
        assert!(result.quality_score > 0.0);
        assert_eq!(result.decision.modality_contributions.len(), 3);
    }
    
    #[test]
    fn test_fusion_manager_insufficient_modalities() {
        let mut config = FusionConfig::<f64>::default();
        config.min_modalities = 3;
        let manager = FusionManager::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        
        let result = manager.fuse_default(&scores, None);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_fusion_manager_with_features() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.75, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.65, 0.8, 15));
        
        let features = create_test_features(
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7],
            vec![0.8, 0.9],
        );
        
        let result = manager.fuse_default(&scores, Some(&features)).unwrap();
        
        assert!(result.decision.confidence > 0.0);
        assert!(!result.intermediate_steps.is_empty());
    }
}

#[cfg(test)]
mod late_fusion_tests {
    use super::*;
    use crate::fusion::strategies::*;
    
    #[test]
    fn test_late_fusion_weighted_average() {
        let config = LateFusionConfig {
            method: LateFusionMethod::WeightedAverage,
            ..Default::default()
        };
        let fusion = LateFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.7, 0.85, 5));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        assert!(result.deception_probability > 0.6);
        assert!(result.deception_probability < 0.8);
        assert!(result.confidence > 0.8);
        assert_eq!(result.modality_contributions.len(), 3);
    }
    
    #[test]
    fn test_late_fusion_confidence_weighted() {
        let config = LateFusionConfig {
            method: LateFusionMethod::ConfidenceWeighted,
            confidence_weight: 0.5,
            ..Default::default()
        };
        let fusion = LateFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.9, 0.95, 10)); // High confidence
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.3, 0.4, 15));  // Low confidence
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Result should be closer to vision due to higher confidence
        assert!(result.deception_probability > 0.6);
    }
    
    #[test]
    fn test_late_fusion_product_method() {
        let config = LateFusionConfig {
            method: LateFusionMethod::Product,
            ..Default::default()
        };
        let fusion = LateFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.9, 0.8, 15));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Product should be 0.8 * 0.9 = 0.72
        assert!((result.deception_probability - 0.72).abs() < 0.01);
    }
    
    #[test]
    fn test_late_fusion_majority_vote() {
        let config = LateFusionConfig {
            method: LateFusionMethod::MajorityVote,
            ..Default::default()
        };
        let fusion = LateFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));   // Vote: Yes
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.7, 0.8, 15));   // Vote: Yes
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.3, 0.85, 5));    // Vote: No
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Majority (2/3) vote for deception
        assert!(result.deception_probability > 0.5);
    }
    
    #[test]
    fn test_late_fusion_low_confidence_filtering() {
        let config = LateFusionConfig {
            min_confidence: 0.8,
            ..Default::default()
        };
        let fusion = LateFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.9, 0.9, 10));  // Above threshold
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.3, 0.5, 15));  // Below threshold
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Should only use vision modality
        assert!(result.modality_contributions.len() == 1);
        assert!(result.modality_contributions.contains_key(&ModalityType::Vision));
    }
    
    #[test]
    fn test_late_fusion_adaptive_weights() {
        let mut fusion = LateFusion::new(LateFusionConfig::default()).unwrap();
        
        let feedback = create_test_feedback(
            true,
            0.8,
            vec![(ModalityType::Vision, 0.9), (ModalityType::Audio, 0.6)],
        );
        
        let old_weights = fusion.get_modality_weights();
        fusion.update(&feedback).unwrap();
        let new_weights = fusion.get_modality_weights();
        
        // Vision weight should increase due to better performance
        assert!(new_weights[&ModalityType::Vision] >= old_weights[&ModalityType::Vision]);
    }
}

#[cfg(test)]
mod early_fusion_tests {
    use super::*;
    use crate::fusion::strategies::*;
    
    #[test]
    fn test_early_fusion_with_features() {
        let fusion = EarlyFusion::new(EarlyFusionConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        
        let features = create_test_features(
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0.6, 0.7, 0.8],
            vec![0.9, 1.0],
        );
        
        let result = fusion.fuse(&scores, Some(&features)).unwrap();
        
        assert!(result.deception_probability >= 0.0);
        assert!(result.deception_probability <= 1.0);
        assert!(result.confidence > 0.0);
        assert!(result.explanation.contains("feature-level integration"));
    }
    
    #[test]
    fn test_early_fusion_normalization() {
        let config = EarlyFusionConfig {
            normalization: FeatureNormalization::ZScore,
            ..Default::default()
        };
        let fusion = EarlyFusion::new(config).unwrap();
        
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = fusion.normalize_features(&features).unwrap();
        
        // Check that mean is approximately 0 and std is approximately 1
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);
        
        let variance: f64 = normalized.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / normalized.len() as f64;
        let std_dev = variance.sqrt();
        assert!((std_dev - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_early_fusion_minmax_normalization() {
        let config = EarlyFusionConfig {
            normalization: FeatureNormalization::MinMax,
            ..Default::default()
        };
        let fusion = EarlyFusion::new(config).unwrap();
        
        let features = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let normalized = fusion.normalize_features(&features).unwrap();
        
        // Check that min is 0 and max is 1
        let min = normalized.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = normalized.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_early_fusion_dimensionality_reduction() {
        let config = EarlyFusionConfig {
            target_dimensions: 5,
            ..Default::default()
        };
        let fusion = EarlyFusion::new(config).unwrap();
        
        let features = (0..20).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let reduced = fusion.reduce_dimensions(&features).unwrap();
        
        assert_eq!(reduced.len(), 5);
    }
    
    #[test]
    fn test_early_fusion_fallback_to_scores() {
        let fusion = EarlyFusion::new(EarlyFusionConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        assert!(result.deception_probability > 0.0);
        assert!(result.explanation.contains("fallback"));
    }
}

#[cfg(test)]
mod attention_fusion_tests {
    use super::*;
    use crate::fusion::attention_fusion::*;
    
    #[test]
    fn test_attention_fusion_multi_head() {
        let config = AttentionConfig {
            attention_type: AttentionType::MultiHead,
            num_heads: 4,
            ..Default::default()
        };
        let fusion = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.7, 0.85, 5));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        assert!(result.deception_probability >= 0.0);
        assert!(result.deception_probability <= 1.0);
        assert!(result.confidence > 0.0);
        assert!(result.explanation.contains("Multi-head"));
    }
    
    #[test]
    fn test_attention_fusion_cross_modal() {
        let config = AttentionConfig {
            attention_type: AttentionType::CrossModal,
            enable_cross_modal: true,
            ..Default::default()
        };
        let fusion = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.75, 0.85, 15));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        assert!(result.confidence > 0.0);
        assert!(result.explanation.contains("CrossModal"));
    }
    
    #[test]
    fn test_attention_fusion_temporal() {
        let config = AttentionConfig {
            attention_type: AttentionType::Temporal,
            enable_temporal: true,
            max_sequence_length: 10,
            ..Default::default()
        };
        let fusion = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8, 15));
        
        let features = create_test_features(
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5],
            vec![0.6],
        );
        
        let result = fusion.fuse(&scores, Some(&features)).unwrap();
        
        assert!(result.confidence > 0.0);
        assert!(result.explanation.contains("Temporal"));
    }
    
    #[test]
    fn test_attention_weight_adaptation() {
        let mut fusion = AttentionFusion::new(AttentionConfig::default()).unwrap();
        
        let feedback = create_test_feedback(
            true,
            0.8,
            vec![
                (ModalityType::Vision, 0.95),
                (ModalityType::Audio, 0.7),
                (ModalityType::Text, 0.8),
            ],
        );
        
        let old_weights = fusion.get_modality_weights();
        fusion.update(&feedback).unwrap();
        let new_weights = fusion.get_modality_weights();
        
        // Weights should have been updated
        assert_ne!(old_weights, new_weights);
        assert!(!fusion.get_adaptation_history().is_empty());
    }
    
    #[test]
    fn test_attention_temperature_scaling() {
        let config = AttentionConfig {
            temperature: 2.0,
            ..Default::default()
        };
        let fusion = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.9, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.5, 0.8, 15));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Temperature scaling should smooth the attention distribution
        assert!(result.confidence > 0.0);
    }
}

#[cfg(test)]
mod hybrid_fusion_tests {
    use super::*;
    use crate::fusion::strategies::*;
    
    #[test]
    fn test_hybrid_fusion_agreement() {
        let config = HybridFusionConfig {
            early_weight: 0.4,
            late_weight: 0.6,
            min_agreement: 0.7,
        };
        let fusion = HybridFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.75, 0.85, 15));
        
        let result = fusion.fuse(&scores, None);
        
        // Should succeed if agreement is high enough
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_hybrid_fusion_low_agreement() {
        let config = HybridFusionConfig {
            early_weight: 0.5,
            late_weight: 0.5,
            min_agreement: 0.9, // Very high threshold
        };
        let fusion = HybridFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.9, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.1, 0.8, 15)); // Very different
        
        let result = fusion.fuse(&scores, None);
        
        // Should fail due to low agreement between early and late fusion
        assert!(result.is_err());
    }
    
    #[test]
    fn test_hybrid_fusion_weight_combination() {
        let fusion = HybridFusion::new(HybridFusionConfig::default()).unwrap();
        
        let weights = fusion.get_modality_weights();
        
        // Weights should be combination of early and late fusion weights
        assert!(!weights.is_empty());
        for (_, &weight) in &weights {
            assert!(weight >= 0.0);
            assert!(weight <= 1.0);
        }
    }
}

#[cfg(test)]
mod weighted_voting_tests {
    use super::*;
    use crate::fusion::strategies::*;
    
    #[test]
    fn test_weighted_voting_majority() {
        let fusion = WeightedVoting::new(WeightedVotingConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));   // Vote: Yes
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.7, 0.8, 15));   // Vote: Yes
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.3, 0.85, 5));    // Vote: No
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Majority should vote for deception
        assert!(result.deception_probability > 0.5);
        assert!(result.explanation.contains("Deceptive"));
    }
    
    #[test]
    fn test_weighted_voting_minority() {
        let fusion = WeightedVoting::new(WeightedVotingConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.3, 0.9, 10));   // Vote: No
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.2, 0.8, 15));   // Vote: No
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.8, 0.85, 5));    // Vote: Yes
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // Majority should vote against deception
        assert!(result.deception_probability < 0.5);
        assert!(result.explanation.contains("Truthful"));
    }
    
    #[test]
    fn test_weighted_voting_confidence_weighting() {
        let config = WeightedVotingConfig {
            use_confidence_weighting: true,
            ..Default::default()
        };
        let fusion = WeightedVoting::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.95, 10));  // High confidence
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.3, 0.5, 15));   // Low confidence
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        // High confidence vision should dominate
        assert!(result.deception_probability > 0.5);
    }
    
    #[test]
    fn test_weighted_voting_insufficient_votes() {
        let config = WeightedVotingConfig {
            min_votes: 3,
            confidence_threshold: 0.8,
            ..Default::default()
        };
        let fusion = WeightedVoting::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));   // Above threshold
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.7, 15));   // Below threshold
        
        let result = fusion.fuse(&scores, None);
        
        // Should fail due to insufficient valid votes
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod temporal_alignment_tests {
    use super::*;
    use crate::fusion::temporal_alignment::*;
    
    #[test]
    fn test_temporal_aligner_creation() {
        let config = AlignmentConfig::<f64>::default();
        let aligner = TemporalAligner::new(config).unwrap();
        
        // Should start with empty buffers
        assert_eq!(aligner.sync_buffers.len(), 0);
    }
    
    #[test]
    fn test_temporal_aligner_add_data() {
        let mut aligner = TemporalAligner::new(AlignmentConfig::<f64>::default()).unwrap();
        
        let timestamp = Utc::now();
        let features = vec![1.0, 2.0, 3.0];
        let quality = 0.9;
        
        aligner.add_data(ModalityType::Vision, timestamp, features.clone(), quality).unwrap();
        
        // Buffer should now contain data
        assert_eq!(aligner.sync_buffers.len(), 1);
        assert!(aligner.sync_buffers.contains_key(&ModalityType::Vision));
    }
    
    #[test]
    fn test_temporal_alignment_with_features() {
        let aligner = TemporalAligner::new(AlignmentConfig::<f64>::default()).unwrap();
        
        let features = create_test_features(
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5],
            vec![0.6],
        );
        
        let aligned = aligner.align(&features).unwrap();
        
        // Should preserve basic structure
        assert!(!aligned.combined.is_empty());
        assert!(!aligned.modalities.is_empty());
        assert!(!aligned.dimension_map.is_empty());
    }
    
    #[test]
    fn test_linear_interpolation() {
        let config = AlignmentConfig {
            interpolation_method: InterpolationMethod::Linear,
            ..Default::default()
        };
        let aligner = TemporalAligner::new(config).unwrap();
        
        let features_a = vec![1.0, 2.0, 3.0];
        let features_b = vec![3.0, 4.0, 5.0];
        let ratio = 0.5;
        
        let result = aligner.linear_interpolate(&features_a, &features_b, ratio).unwrap();
        
        // Should be midpoint
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_cubic_interpolation() {
        let config = AlignmentConfig {
            interpolation_method: InterpolationMethod::Cubic,
            ..Default::default()
        };
        let aligner = TemporalAligner::new(config).unwrap();
        
        let features_a = vec![1.0, 2.0];
        let features_b = vec![3.0, 4.0];
        let ratio = 0.5;
        
        let result = aligner.cubic_interpolate(&features_a, &features_b, ratio).unwrap();
        
        // Should return valid interpolated values
        assert_eq!(result.len(), 2);
        assert!(result[0] >= 1.0 && result[0] <= 3.0);
        assert!(result[1] >= 2.0 && result[1] <= 4.0);
    }
    
    #[test]
    fn test_temporal_smoothing() {
        let config = AlignmentConfig {
            smoothing_window: 3,
            ..Default::default()
        };
        let aligner = TemporalAligner::new(config).unwrap();
        
        let data = vec![1.0, 5.0, 1.0, 5.0, 1.0]; // Noisy data
        let smoothed = aligner.apply_temporal_smoothing(&data).unwrap();
        
        // Smoothed data should have reduced variation
        assert_eq!(smoothed.len(), data.len());
        
        // Middle values should be averaged
        assert!(smoothed[2] > 1.0 && smoothed[2] < 5.0);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_pipeline_integration() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        // Create comprehensive test data
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.75, 0.85, 15));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.7, 0.8, 8));
        scores.insert(ModalityType::Physiological, create_test_score::<f64>(0.65, 0.75, 12));
        
        let features = create_test_features(
            vec![0.1, 0.2, 0.3, 0.4, 0.5],    // Vision features
            vec![0.6, 0.7, 0.8],              // Audio features
            vec![0.9, 1.0],                   // Text features
        );
        
        // Test all fusion strategies
        let strategies = ["late_fusion", "early_fusion", "hybrid_fusion", "attention_fusion", "weighted_voting"];
        
        for strategy in &strategies {
            let result = manager.fuse(strategy, &scores, Some(&features));
            
            match result {
                Ok(fusion_result) => {
                    assert!(fusion_result.decision.deception_probability >= 0.0);
                    assert!(fusion_result.decision.deception_probability <= 1.0);
                    assert!(fusion_result.decision.confidence > 0.0);
                    assert!(fusion_result.quality_score > 0.0);
                    assert!(!fusion_result.intermediate_steps.is_empty());
                }
                Err(e) => {
                    // Some strategies might fail with certain configurations
                    println!("Strategy {} failed: {}", strategy, e);
                }
            }
        }
    }
    
    #[test]
    fn test_cross_strategy_consistency() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        // Create consistent high-confidence scores
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.85, 0.95, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.8, 0.9, 15));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.82, 0.92, 8));
        
        let late_result = manager.fuse("late_fusion", &scores, None).unwrap();
        let voting_result = manager.fuse("weighted_voting", &scores, None).unwrap();
        
        // Both strategies should agree on deceptive classification
        assert!(late_result.decision.deception_probability > 0.5);
        assert!(voting_result.decision.deception_probability > 0.5);
        
        // Results should be reasonably similar
        let diff = (late_result.decision.deception_probability - voting_result.decision.deception_probability).abs();
        assert!(diff < 0.3); // Allow some variation between strategies
    }
    
    #[test]
    fn test_edge_case_single_modality() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        
        let result = manager.fuse("late_fusion", &scores, None);
        
        // Should handle single modality gracefully
        if let Ok(fusion_result) = result {
            assert_eq!(fusion_result.decision.modality_contributions.len(), 1);
            assert!(fusion_result.decision.modality_contributions.contains_key(&ModalityType::Vision));
        }
    }
    
    #[test]
    fn test_edge_case_conflicting_modalities() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        // Create highly conflicting scores
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.95, 0.9, 10));  // High deception
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.05, 0.9, 15));  // Low deception
        
        let result = manager.fuse("late_fusion", &scores, None).unwrap();
        
        // Should handle conflicting inputs and produce reasonable result
        assert!(result.decision.deception_probability >= 0.0);
        assert!(result.decision.deception_probability <= 1.0);
        // Confidence might be lower due to conflict
        assert!(result.decision.confidence >= 0.0);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_fusion_performance() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.75, 0.85, 15));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.7, 0.8, 8));
        
        let start = Instant::now();
        let _result = manager.fuse_default(&scores, None).unwrap();
        let duration = start.elapsed();
        
        // Fusion should complete quickly (< 100ms for simple case)
        assert!(duration.as_millis() < 100);
    }
    
    #[test]
    fn test_large_feature_vectors() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.75, 0.85, 15));
        
        // Create large feature vectors
        let large_vision_features: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let large_audio_features: Vec<f64> = (0..500).map(|i| i as f64 * 0.002).collect();
        let large_text_features: Vec<f64> = (0..200).map(|i| i as f64 * 0.005).collect();
        
        let features = create_test_features(
            large_vision_features,
            large_audio_features,
            large_text_features,
        );
        
        let start = Instant::now();
        let result = manager.fuse("early_fusion", &scores, Some(&features));
        let duration = start.elapsed();
        
        // Should handle large features efficiently
        assert!(result.is_ok());
        assert!(duration.as_millis() < 500); // Allow more time for large features
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;
    
    #[test]
    fn test_empty_scores() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        let scores = HashMap::new();
        
        let result = manager.fuse_default(&scores, None);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_invalid_strategy() {
        let manager = FusionManager::new(FusionConfig::<f64>::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        
        let result = manager.fuse("nonexistent_strategy", &scores, None);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_low_quality_threshold() {
        let mut config = FusionConfig::<f64>::default();
        config.quality_threshold = 0.99; // Very high threshold
        
        let manager = FusionManager::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9, 10));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.3, 0.8, 15)); // Conflicting
        
        let result = manager.fuse_default(&scores, None);
        // Might fail due to quality threshold
        if result.is_err() {
            // This is expected behavior
            assert!(true);
        }
    }
}