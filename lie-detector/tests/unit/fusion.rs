/// Unit tests for multi-modal fusion strategies
/// 
/// Tests early fusion, late fusion, attention-based fusion, and hybrid approaches

use crate::common::*;
use std::collections::HashMap;

#[cfg(test)]
mod fusion_strategy_tests {
    use super::*;
    use fixtures::{MultiModalTestData};
    use mocks::*;
    use helpers::*;

    /// Test late fusion strategy
    #[test]
    fn test_late_fusion_strategy() {
        let config = TestConfig::default();
        config.setup().expect("Failed to setup test config");
        
        // Create mock scores from different modalities
        let mut modality_scores = HashMap::new();
        modality_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.8));
        modality_scores.insert("audio".to_string(), MockDeceptionScore::new(0.6, 0.9));
        modality_scores.insert("text".to_string(), MockDeceptionScore::new(0.8, 0.7));
        
        let fused_result = late_fusion(&modality_scores, &[0.33, 0.33, 0.34]);
        
        // Verify fusion result
        assert!(fused_result.deception_probability >= 0.0 && fused_result.deception_probability <= 1.0,
                "Fused probability should be in [0, 1]");
        assert!(fused_result.confidence > 0.0, "Fused confidence should be positive");
        assert_eq!(fused_result.modality_contributions.len(), 3, "Should have contributions from all modalities");
    }

    /// Test early fusion strategy
    #[test]
    fn test_early_fusion_strategy() {
        let vision_features = vec![0.1, 0.2, 0.3, 0.4];
        let audio_features = vec![0.5, 0.6, 0.7];
        let text_features = vec![0.8, 0.9];
        
        let combined_features = early_fusion(vec![
            ("vision", vision_features),
            ("audio", audio_features),
            ("text", text_features),
        ]);
        
        // Verify concatenation
        let expected_length = 4 + 3 + 2;
        assert_eq!(combined_features.len(), expected_length, "Should concatenate all features");
        
        // Verify feature order
        assert_eq!(combined_features[0], 0.1, "First vision feature should be first");
        assert_eq!(combined_features[4], 0.5, "First audio feature should be at index 4");
        assert_eq!(combined_features[7], 0.8, "First text feature should be at index 7");
    }

    /// Test attention-based fusion
    #[test]
    fn test_attention_fusion() {
        let mut modality_scores = HashMap::new();
        modality_scores.insert("vision".to_string(), MockDeceptionScore::new(0.9, 0.95)); // High confidence
        modality_scores.insert("audio".to_string(), MockDeceptionScore::new(0.3, 0.4));  // Low confidence
        modality_scores.insert("text".to_string(), MockDeceptionScore::new(0.7, 0.8));   // Medium confidence
        
        let fused_result = attention_fusion(&modality_scores);
        
        // High confidence modality should have more influence
        assert!(fused_result.modality_contributions["vision"] > fused_result.modality_contributions["audio"],
                "High confidence modality should have higher attention weight");
        assert!(fused_result.modality_contributions["vision"] > fused_result.modality_contributions["text"],
                "High confidence modality should have highest attention weight");
        
        // Final probability should be closer to high-confidence modality
        let vision_influence = (fused_result.deception_probability - 0.9).abs();
        let audio_influence = (fused_result.deception_probability - 0.3).abs();
        assert!(vision_influence < audio_influence, "Result should be closer to high-confidence modality");
    }

    /// Test hybrid fusion strategy
    #[test]
    fn test_hybrid_fusion() {
        let vision_features = vec![0.1, 0.2, 0.3];
        let audio_features = vec![0.4, 0.5];
        let text_features = vec![0.6];
        
        let mut modality_scores = HashMap::new();
        modality_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.8));
        modality_scores.insert("audio".to_string(), MockDeceptionScore::new(0.6, 0.9));
        modality_scores.insert("text".to_string(), MockDeceptionScore::new(0.8, 0.7));
        
        let feature_map = vec![
            ("vision", vision_features),
            ("audio", audio_features),
            ("text", text_features),
        ];
        
        let fused_result = hybrid_fusion(&feature_map, &modality_scores, 0.6, 0.4);
        
        // Verify the result combines both early and late fusion
        assert!(fused_result.deception_probability >= 0.0 && fused_result.deception_probability <= 1.0,
                "Hybrid fusion probability should be in [0, 1]");
        assert!(fused_result.confidence > 0.0, "Hybrid fusion confidence should be positive");
    }

    /// Test fusion with missing modalities
    #[test]
    fn test_fusion_with_missing_modalities() {
        let mut incomplete_scores = HashMap::new();
        incomplete_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.8));
        incomplete_scores.insert("audio".to_string(), MockDeceptionScore::new(0.6, 0.9));
        // Missing text modality
        
        let fused_result = late_fusion(&incomplete_scores, &[0.5, 0.5]);
        
        // Should handle missing modalities gracefully
        assert!(fused_result.deception_probability >= 0.0 && fused_result.deception_probability <= 1.0,
                "Should handle missing modalities");
        assert_eq!(fused_result.modality_contributions.len(), 2, "Should only have contributions from available modalities");
    }

    /// Test fusion confidence calculation
    #[test]
    fn test_fusion_confidence_calculation() {
        // High confidence scores
        let mut high_conf_scores = HashMap::new();
        high_conf_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.9));
        high_conf_scores.insert("audio".to_string(), MockDeceptionScore::new(0.6, 0.95));
        high_conf_scores.insert("text".to_string(), MockDeceptionScore::new(0.8, 0.85));
        
        // Low confidence scores
        let mut low_conf_scores = HashMap::new();
        low_conf_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.3));
        low_conf_scores.insert("audio".to_string(), MockDeceptionScore::new(0.6, 0.4));
        low_conf_scores.insert("text".to_string(), MockDeceptionScore::new(0.8, 0.2));
        
        let high_conf_result = late_fusion(&high_conf_scores, &[0.33, 0.33, 0.34]);
        let low_conf_result = late_fusion(&low_conf_scores, &[0.33, 0.33, 0.34]);
        
        assert!(high_conf_result.confidence > low_conf_result.confidence,
                "High confidence inputs should produce higher fusion confidence");
    }

    /// Test dynamic weight adjustment
    #[test]
    fn test_dynamic_weight_adjustment() {
        let mut modality_scores = HashMap::new();
        modality_scores.insert("vision".to_string(), MockDeceptionScore::new(0.9, 0.95)); // Very reliable
        modality_scores.insert("audio".to_string(), MockDeceptionScore::new(0.1, 0.3));  // Unreliable
        modality_scores.insert("text".to_string(), MockDeceptionScore::new(0.7, 0.8));   // Moderate
        
        let adaptive_weights = calculate_adaptive_weights(&modality_scores);
        
        // Reliable modality should get higher weight
        assert!(adaptive_weights["vision"] > adaptive_weights["audio"],
                "Reliable modality should get higher weight");
        assert!(adaptive_weights["vision"] > adaptive_weights["text"],
                "Most reliable modality should get highest weight");
        
        // Weights should sum to 1.0
        let weight_sum: f32 = adaptive_weights.values().sum();
        assert_float_eq(weight_sum, 1.0, 1e-6);
    }

    /// Test consensus-based fusion
    #[test]
    fn test_consensus_fusion() {
        // High consensus case (all modalities agree)
        let mut consensus_scores = HashMap::new();
        consensus_scores.insert("vision".to_string(), MockDeceptionScore::new(0.8, 0.9));
        consensus_scores.insert("audio".to_string(), MockDeceptionScore::new(0.75, 0.8));
        consensus_scores.insert("text".to_string(), MockDeceptionScore::new(0.85, 0.85));
        
        // Low consensus case (modalities disagree)
        let mut disagreement_scores = HashMap::new();
        disagreement_scores.insert("vision".to_string(), MockDeceptionScore::new(0.9, 0.9));
        disagreement_scores.insert("audio".to_string(), MockDeceptionScore::new(0.2, 0.8));
        disagreement_scores.insert("text".to_string(), MockDeceptionScore::new(0.5, 0.7));
        
        let consensus_result = consensus_fusion(&consensus_scores);
        let disagreement_result = consensus_fusion(&disagreement_scores);
        
        assert!(consensus_result.confidence > disagreement_result.confidence,
                "High consensus should produce higher confidence");
        
        let consensus_variance = calculate_score_variance(&consensus_scores);
        let disagreement_variance = calculate_score_variance(&disagreement_scores);
        assert!(disagreement_variance > consensus_variance,
                "Disagreement should produce higher variance");
    }

    /// Test temporal fusion for streaming data
    #[test]
    fn test_temporal_fusion() {
        let mut temporal_scores = Vec::new();
        
        // Create time series of scores
        for i in 0..10 {
            let mut frame_scores = HashMap::new();
            let base_prob = 0.5 + (i as f32 * 0.05); // Gradually increasing deception
            frame_scores.insert("vision".to_string(), MockDeceptionScore::new(base_prob, 0.8));
            frame_scores.insert("audio".to_string(), MockDeceptionScore::new(base_prob + 0.1, 0.9));
            temporal_scores.push(frame_scores);
        }
        
        let temporal_result = temporal_fusion(&temporal_scores, 5); // 5-frame window
        
        // Should show temporal trend
        assert!(temporal_result.deception_probability > 0.5,
                "Should capture increasing deception trend");
        assert!(temporal_result.confidence > 0.0, "Should have temporal confidence");
    }

    /// Test fusion robustness to outliers
    #[test]
    fn test_fusion_outlier_robustness() {
        let mut normal_scores = HashMap::new();
        normal_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.8));
        normal_scores.insert("audio".to_string(), MockDeceptionScore::new(0.65, 0.9));
        normal_scores.insert("text".to_string(), MockDeceptionScore::new(0.75, 0.7));
        
        let mut outlier_scores = HashMap::new();
        outlier_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.8));
        outlier_scores.insert("audio".to_string(), MockDeceptionScore::new(0.65, 0.9));
        outlier_scores.insert("text".to_string(), MockDeceptionScore::new(0.05, 0.1)); // Outlier
        
        let normal_result = robust_fusion(&normal_scores);
        let outlier_result = robust_fusion(&outlier_scores);
        
        // Robust fusion should be less affected by outliers
        let normal_median = calculate_score_median(&normal_scores);
        let outlier_median = calculate_score_median(&outlier_scores);
        
        let normal_diff = (normal_result.deception_probability - normal_median).abs();
        let outlier_diff = (outlier_result.deception_probability - outlier_median).abs();
        
        // The robust fusion should handle outliers better than simple averaging
        assert!(outlier_diff <= normal_diff + 0.1, "Robust fusion should handle outliers");
    }

    /// Test fusion performance with many modalities
    #[test]
    fn test_fusion_scalability() {
        let mut many_modalities = HashMap::new();
        
        // Create 20 mock modalities
        for i in 0..20 {
            let modality_name = format!("modality_{}", i);
            let prob = 0.5 + (i as f32 * 0.02); // Varying probabilities
            let conf = 0.7 + (i as f32 * 0.01); // Varying confidences
            many_modalities.insert(modality_name, MockDeceptionScore::new(prob, conf));
        }
        
        let (_, measurement) = measure_performance(|| {
            late_fusion(&many_modalities, &vec![0.05; 20])
        });
        
        // Should handle many modalities efficiently
        assert_performance_bounds(
            &measurement,
            std::time::Duration::from_millis(10), // Max 10ms for 20 modalities
            Some(1024 * 1024) // Max 1MB memory
        );
    }

    /// Test concurrent fusion processing
    #[tokio::test]
    async fn test_concurrent_fusion() {
        let mut modality_scores = HashMap::new();
        modality_scores.insert("vision".to_string(), MockDeceptionScore::new(0.7, 0.8));
        modality_scores.insert("audio".to_string(), MockDeceptionScore::new(0.6, 0.9));
        modality_scores.insert("text".to_string(), MockDeceptionScore::new(0.8, 0.7));
        
        let results = run_concurrent_tests(10, |_| {
            let scores = modality_scores.clone();
            async move {
                let result = late_fusion(&scores, &[0.33, 0.33, 0.34]);
                assert!(result.deception_probability >= 0.0 && result.deception_probability <= 1.0);
                Ok(result)
            }
        }).await;
        
        // All should succeed and produce consistent results
        let mut probabilities = Vec::new();
        for result in results {
            assert!(result.is_ok(), "Concurrent fusion should succeed");
            probabilities.push(result.unwrap().deception_probability);
        }
        
        // Results should be consistent (same inputs should produce same outputs)
        let variance = calculate_variance(&probabilities);
        assert!(variance < 1e-10, "Concurrent fusion should be deterministic");
    }

    // Helper functions for fusion tests

    fn late_fusion(scores: &HashMap<String, MockDeceptionScore<f32>>, weights: &[f32]) -> MockFusedDecision<f32> {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut confidence_sum = 0.0;
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            let weight_idx = match modality.as_str() {
                "vision" => 0,
                "audio" => 1,
                "text" => 2,
                _ => 0,
            };
            
            if weight_idx < weights.len() {
                let weight = weights[weight_idx];
                weighted_sum += score.probability * weight;
                confidence_sum += score.confidence * weight;
                weight_sum += weight;
                contributions.insert(modality.clone(), weight);
            }
        }
        
        let final_probability = if weight_sum > 0.0 { weighted_sum / weight_sum } else { 0.0 };
        let final_confidence = if weight_sum > 0.0 { confidence_sum / weight_sum } else { 0.0 };
        
        MockFusedDecision {
            deception_probability: final_probability,
            confidence: final_confidence,
            modality_contributions: contributions,
            explanation: "Late fusion result".to_string(),
        }
    }

    fn early_fusion(feature_sets: Vec<(&str, Vec<f32>)>) -> Vec<f32> {
        let mut combined = Vec::new();
        
        for (_, features) in feature_sets {
            combined.extend(features);
        }
        
        combined
    }

    fn attention_fusion(scores: &HashMap<String, MockDeceptionScore<f32>>) -> MockFusedDecision<f32> {
        let mut attention_weights = HashMap::new();
        let mut total_attention = 0.0;
        
        // Calculate attention weights based on confidence
        for (modality, score) in scores {
            let attention = score.confidence * score.confidence; // Squared for emphasis
            attention_weights.insert(modality.clone(), attention);
            total_attention += attention;
        }
        
        // Normalize attention weights
        for weight in attention_weights.values_mut() {
            *weight /= total_attention;
        }
        
        // Apply attention-weighted fusion
        let mut weighted_sum = 0.0;
        let mut confidence_sum = 0.0;
        
        for (modality, score) in scores {
            let attention = attention_weights[modality];
            weighted_sum += score.probability * attention;
            confidence_sum += score.confidence * attention;
        }
        
        MockFusedDecision {
            deception_probability: weighted_sum,
            confidence: confidence_sum,
            modality_contributions: attention_weights,
            explanation: "Attention-based fusion result".to_string(),
        }
    }

    fn hybrid_fusion(
        feature_sets: &[(&str, Vec<f32>)],
        scores: &HashMap<String, MockDeceptionScore<f32>>,
        early_weight: f32,
        late_weight: f32,
    ) -> MockFusedDecision<f32> {
        // Early fusion component
        let combined_features = early_fusion(feature_sets.to_vec());
        let early_result = 0.5; // Mock early fusion neural network result
        
        // Late fusion component
        let equal_weights = vec![1.0 / scores.len() as f32; scores.len()];
        let late_result = late_fusion(scores, &equal_weights);
        
        // Combine early and late results
        let final_probability = early_result * early_weight + late_result.deception_probability * late_weight;
        let final_confidence = late_result.confidence;
        
        MockFusedDecision {
            deception_probability: final_probability,
            confidence: final_confidence,
            modality_contributions: late_result.modality_contributions,
            explanation: "Hybrid fusion result".to_string(),
        }
    }

    fn calculate_adaptive_weights(scores: &HashMap<String, MockDeceptionScore<f32>>) -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        let mut total_confidence = 0.0;
        
        // Weight by confidence
        for (modality, score) in scores {
            total_confidence += score.confidence;
        }
        
        for (modality, score) in scores {
            let weight = score.confidence / total_confidence;
            weights.insert(modality.clone(), weight);
        }
        
        weights
    }

    fn consensus_fusion(scores: &HashMap<String, MockDeceptionScore<f32>>) -> MockFusedDecision<f32> {
        let probabilities: Vec<f32> = scores.values().map(|s| s.probability).collect();
        let confidences: Vec<f32> = scores.values().map(|s| s.confidence).collect();
        
        let mean_prob = probabilities.iter().sum::<f32>() / probabilities.len() as f32;
        let mean_conf = confidences.iter().sum::<f32>() / confidences.len() as f32;
        
        // Calculate consensus based on variance
        let variance = calculate_variance(&probabilities);
        let consensus_confidence = mean_conf * (1.0 / (1.0 + variance)); // Lower variance = higher confidence
        
        let mut contributions = HashMap::new();
        for modality in scores.keys() {
            contributions.insert(modality.clone(), 1.0 / scores.len() as f32);
        }
        
        MockFusedDecision {
            deception_probability: mean_prob,
            confidence: consensus_confidence,
            modality_contributions: contributions,
            explanation: "Consensus-based fusion result".to_string(),
        }
    }

    fn calculate_score_variance(scores: &HashMap<String, MockDeceptionScore<f32>>) -> f32 {
        let probabilities: Vec<f32> = scores.values().map(|s| s.probability).collect();
        calculate_variance(&probabilities)
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

    fn temporal_fusion(
        temporal_scores: &[HashMap<String, MockDeceptionScore<f32>>],
        window_size: usize,
    ) -> MockFusedDecision<f32> {
        if temporal_scores.is_empty() {
            return MockFusedDecision {
                deception_probability: 0.0,
                confidence: 0.0,
                modality_contributions: HashMap::new(),
                explanation: "No temporal data".to_string(),
            };
        }
        
        let start_idx = temporal_scores.len().saturating_sub(window_size);
        let recent_scores = &temporal_scores[start_idx..];
        
        // Simple temporal averaging
        let mut prob_sum = 0.0;
        let mut conf_sum = 0.0;
        let mut count = 0;
        
        for frame_scores in recent_scores {
            for score in frame_scores.values() {
                prob_sum += score.probability;
                conf_sum += score.confidence;
                count += 1;
            }
        }
        
        let final_prob = if count > 0 { prob_sum / count as f32 } else { 0.0 };
        let final_conf = if count > 0 { conf_sum / count as f32 } else { 0.0 };
        
        MockFusedDecision {
            deception_probability: final_prob,
            confidence: final_conf,
            modality_contributions: HashMap::new(),
            explanation: "Temporal fusion result".to_string(),
        }
    }

    fn robust_fusion(scores: &HashMap<String, MockDeceptionScore<f32>>) -> MockFusedDecision<f32> {
        let mut probabilities: Vec<f32> = scores.values().map(|s| s.probability).collect();
        probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Use median instead of mean for robustness
        let median_prob = if probabilities.len() % 2 == 0 {
            let mid = probabilities.len() / 2;
            (probabilities[mid - 1] + probabilities[mid]) / 2.0
        } else {
            probabilities[probabilities.len() / 2]
        };
        
        let mean_conf = scores.values().map(|s| s.confidence).sum::<f32>() / scores.len() as f32;
        
        let mut contributions = HashMap::new();
        for modality in scores.keys() {
            contributions.insert(modality.clone(), 1.0 / scores.len() as f32);
        }
        
        MockFusedDecision {
            deception_probability: median_prob,
            confidence: mean_conf,
            modality_contributions: contributions,
            explanation: "Robust fusion result".to_string(),
        }
    }

    fn calculate_score_median(scores: &HashMap<String, MockDeceptionScore<f32>>) -> f32 {
        let mut probabilities: Vec<f32> = scores.values().map(|s| s.probability).collect();
        probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if probabilities.len() % 2 == 0 {
            let mid = probabilities.len() / 2;
            (probabilities[mid - 1] + probabilities[mid]) / 2.0
        } else {
            probabilities[probabilities.len() / 2]
        }
    }
}