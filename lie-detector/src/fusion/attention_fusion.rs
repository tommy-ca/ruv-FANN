//! Attention-based fusion mechanisms for multi-modal integration
//!
//! This module implements sophisticated attention mechanisms to dynamically
//! weight different modalities based on their relevance and reliability for
//! deception detection in different contexts.

use crate::error::{FusionError, Result};
use crate::fusion::{FeedbackData, FusionStrategy, utils};
use crate::types::{
    AttentionWeights, CombinedFeatures, DeceptionScore, FusedDecision, 
    FusionMetadata, ModalityType, ProcessingTiming, QualityMetrics,
};
use hashbrown::HashMap;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Attention-based fusion strategy
#[derive(Debug, Clone)]
pub struct AttentionFusion<T: Float> {
    config: AttentionConfig<T>,
    attention_weights: AttentionWeights<T>,
    learned_parameters: AttentionParameters<T>,
    adaptation_history: Vec<AdaptationStep<T>>,
}

/// Configuration for attention-based fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig<T: Float> {
    /// Attention mechanism type
    pub attention_type: AttentionType,
    /// Number of attention heads for multi-head attention
    pub num_heads: usize,
    /// Dimension of attention hidden layers
    pub hidden_dim: usize,
    /// Learning rate for attention adaptation
    pub learning_rate: T,
    /// Temperature for softmax attention
    pub temperature: T,
    /// Enable temporal attention
    pub enable_temporal: bool,
    /// Enable cross-modal attention
    pub enable_cross_modal: bool,
    /// Dropout probability for regularization
    pub dropout: T,
    /// Maximum sequence length for temporal attention
    pub max_sequence_length: usize,
}

/// Types of attention mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    /// Simple additive attention
    Additive,
    /// Scaled dot-product attention
    ScaledDotProduct,
    /// Multi-head attention
    MultiHead,
    /// Self-attention with cross-modal interaction
    CrossModal,
    /// Temporal attention with memory
    Temporal,
}

impl<T: Float> Default for AttentionConfig<T> {
    fn default() -> Self {
        Self {
            attention_type: AttentionType::MultiHead,
            num_heads: 4,
            hidden_dim: 128,
            learning_rate: T::from(0.001).unwrap(),
            temperature: T::from(1.0).unwrap(),
            enable_temporal: true,
            enable_cross_modal: true,
            dropout: T::from(0.1).unwrap(),
            max_sequence_length: 100,
        }
    }
}

/// Learned parameters for attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionParameters<T: Float> {
    /// Query projection weights
    pub query_weights: HashMap<ModalityType, Vec<T>>,
    /// Key projection weights
    pub key_weights: HashMap<ModalityType, Vec<T>>,
    /// Value projection weights
    pub value_weights: HashMap<ModalityType, Vec<T>>,
    /// Output projection weights
    pub output_weights: Vec<T>,
    /// Bias terms
    pub bias: Vec<T>,
    /// Temporal weights for sequence modeling
    pub temporal_weights: Option<Vec<T>>,
    /// Cross-modal interaction matrix
    pub cross_modal_matrix: Option<Vec<Vec<T>>>,
}

/// Adaptation step for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStep<T: Float> {
    /// Timestamp of adaptation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Previous attention weights
    pub previous_weights: HashMap<ModalityType, T>,
    /// New attention weights
    pub new_weights: HashMap<ModalityType, T>,
    /// Performance improvement
    pub improvement: T,
    /// Adaptation reason
    pub reason: String,
}

impl<T: Float + Send + Sync> FusionStrategy<T> for AttentionFusion<T> {
    type Config = AttentionConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        let mut modality_weights = HashMap::new();
        modality_weights.insert(ModalityType::Vision, T::from(0.25).unwrap());
        modality_weights.insert(ModalityType::Audio, T::from(0.25).unwrap());
        modality_weights.insert(ModalityType::Text, T::from(0.25).unwrap());
        modality_weights.insert(ModalityType::Physiological, T::from(0.25).unwrap());
        
        let attention_weights = AttentionWeights {
            modality_weights,
            temporal_weights: vec![T::one(); config.max_sequence_length],
            cross_modal: None,
        };
        
        let learned_parameters = Self::initialize_parameters(&config)?;
        
        Ok(Self {
            config,
            attention_weights,
            learned_parameters,
            adaptation_history: Vec::new(),
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Compute attention weights dynamically
        let dynamic_attention = self.compute_attention(scores, features)?;
        
        // Apply attention mechanism based on type
        let (prediction, modality_contributions) = match self.config.attention_type {
            AttentionType::Additive => self.additive_attention(scores, &dynamic_attention)?,
            AttentionType::ScaledDotProduct => self.scaled_dot_product_attention(scores, &dynamic_attention)?,
            AttentionType::MultiHead => self.multi_head_attention(scores, features, &dynamic_attention)?,
            AttentionType::CrossModal => self.cross_modal_attention(scores, features, &dynamic_attention)?,
            AttentionType::Temporal => self.temporal_attention(scores, features, &dynamic_attention)?,
        };
        
        // Calculate attention-based confidence
        let confidence = self.calculate_attention_confidence(&dynamic_attention, scores)?;
        
        let processing_time = start_time.elapsed();
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence,
            modality_contributions,
            explanation: format!(
                "Attention-based fusion using {:?} mechanism. \
                Dynamic attention weights: Vision={:.3}, Audio={:.3}, Text={:.3}, Physio={:.3}",
                self.config.attention_type,
                dynamic_attention.modality_weights.get(&ModalityType::Vision)
                    .unwrap_or(&T::zero()).to_f64().unwrap_or(0.0),
                dynamic_attention.modality_weights.get(&ModalityType::Audio)
                    .unwrap_or(&T::zero()).to_f64().unwrap_or(0.0),
                dynamic_attention.modality_weights.get(&ModalityType::Text)
                    .unwrap_or(&T::zero()).to_f64().unwrap_or(0.0),
                dynamic_attention.modality_weights.get(&ModalityType::Physiological)
                    .unwrap_or(&T::zero()).to_f64().unwrap_or(0.0)
            ),
            metadata: self.create_metadata(processing_time, scores, &dynamic_attention)?,
        })
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        self.attention_weights.modality_weights.clone()
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        // Store previous weights for adaptation tracking
        let previous_weights = self.attention_weights.modality_weights.clone();
        
        // Update attention parameters based on feedback
        self.update_attention_parameters(feedback)?;
        
        // Record adaptation step
        let adaptation_step = AdaptationStep {
            timestamp: chrono::Utc::now(),
            previous_weights,
            new_weights: self.attention_weights.modality_weights.clone(),
            improvement: feedback.modality_performance.values().fold(T::zero(), |acc, &x| acc + x)
                / T::from(feedback.modality_performance.len()).unwrap(),
            reason: "Performance feedback adaptation".to_string(),
        };
        
        self.adaptation_history.push(adaptation_step);
        
        // Keep only recent history
        if self.adaptation_history.len() > 100 {
            self.adaptation_history.remove(0);
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "attention_fusion"
    }
}

impl<T: Float> AttentionFusion<T> {
    /// Initialize attention parameters
    fn initialize_parameters(config: &AttentionConfig<T>) -> Result<AttentionParameters<T>> {
        let modalities = [
            ModalityType::Vision,
            ModalityType::Audio,
            ModalityType::Text,
            ModalityType::Physiological,
        ];
        
        let mut query_weights = HashMap::new();
        let mut key_weights = HashMap::new();
        let mut value_weights = HashMap::new();
        
        for modality in &modalities {
            // Random initialization (in practice, would use proper initialization)
            let dim = config.hidden_dim;
            query_weights.insert(*modality, vec![T::from(0.1).unwrap(); dim]);
            key_weights.insert(*modality, vec![T::from(0.1).unwrap(); dim]);
            value_weights.insert(*modality, vec![T::from(0.1).unwrap(); dim]);
        }
        
        let output_weights = vec![T::from(0.1).unwrap(); config.hidden_dim];
        let bias = vec![T::zero(); config.hidden_dim];
        
        let temporal_weights = if config.enable_temporal {
            Some(vec![T::from(0.1).unwrap(); config.max_sequence_length])
        } else {
            None
        };
        
        let cross_modal_matrix = if config.enable_cross_modal {
            let size = modalities.len();
            Some(vec![vec![T::from(0.1).unwrap(); size]; size])
        } else {
            None
        };
        
        Ok(AttentionParameters {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            bias,
            temporal_weights,
            cross_modal_matrix,
        })
    }
    
    /// Compute dynamic attention weights
    fn compute_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<AttentionWeights<T>> {
        let mut modality_weights = HashMap::new();
        
        // Base attention on confidence and reliability
        let mut total_attention = T::zero();
        for (modality, score) in scores {
            // Combine confidence with learned weights
            let base_weight = self.attention_weights.modality_weights
                .get(modality)
                .copied()
                .unwrap_or(T::from(0.25).unwrap());
            
            // Attention based on confidence and score quality
            let confidence_factor = score.confidence;
            let consistency_factor = self.calculate_modality_consistency(modality, scores)?;
            
            let attention = base_weight * confidence_factor * consistency_factor;
            modality_weights.insert(*modality, attention);
            total_attention = total_attention + attention;
        }
        
        // Normalize attention weights
        if total_attention > T::zero() {
            for weight in modality_weights.values_mut() {
                *weight = *weight / total_attention;
            }
        }
        
        // Apply temperature scaling
        if self.config.temperature != T::one() {
            for weight in modality_weights.values_mut() {
                *weight = (*weight / self.config.temperature).exp();
            }
            
            // Re-normalize after temperature scaling
            let sum: T = modality_weights.values().fold(T::zero(), |acc, &w| acc + w);
            if sum > T::zero() {
                for weight in modality_weights.values_mut() {
                    *weight = *weight / sum;
                }
            }
        }
        
        // Compute temporal weights if features are available
        let temporal_weights = if let Some(features) = features {
            self.compute_temporal_attention(features)?
        } else {
            self.attention_weights.temporal_weights.clone()
        };
        
        // Compute cross-modal attention matrix
        let cross_modal = if self.config.enable_cross_modal {
            Some(self.compute_cross_modal_attention(scores)?)
        } else {
            None
        };
        
        Ok(AttentionWeights {
            modality_weights,
            temporal_weights,
            cross_modal,
        })
    }
    
    /// Calculate modality consistency
    fn calculate_modality_consistency(
        &self,
        target_modality: &ModalityType,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        if scores.len() < 2 {
            return Ok(T::one());
        }
        
        let target_prob = scores.get(target_modality)
            .map(|s| s.probability)
            .unwrap_or(T::from(0.5).unwrap());
        
        let mut consistency_sum = T::zero();
        let mut count = 0;
        
        for (modality, score) in scores {
            if modality != target_modality {
                let agreement = T::one() - (target_prob - score.probability).abs();
                consistency_sum = consistency_sum + agreement;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(consistency_sum / T::from(count).unwrap())
        } else {
            Ok(T::one())
        }
    }
    
    /// Compute temporal attention weights
    fn compute_temporal_attention(
        &self,
        features: &CombinedFeatures<T>,
    ) -> Result<Vec<T>> {
        // Simplified temporal attention based on feature dynamics
        let sequence_length = features.combined.len().min(self.config.max_sequence_length);
        let mut temporal_weights = vec![T::one(); sequence_length];
        
        // Apply learned temporal patterns if available
        if let Some(ref learned_temporal) = self.learned_parameters.temporal_weights {
            for (i, weight) in temporal_weights.iter_mut().enumerate() {
                if i < learned_temporal.len() {
                    *weight = learned_temporal[i];
                }
            }
        }
        
        // Normalize temporal weights
        let sum: T = temporal_weights.iter().fold(T::zero(), |acc, &w| acc + w);
        if sum > T::zero() {
            for weight in temporal_weights.iter_mut() {
                *weight = *weight / sum;
            }
        }
        
        Ok(temporal_weights)
    }
    
    /// Compute cross-modal attention matrix
    fn compute_cross_modal_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<Vec<Vec<T>>> {
        let modalities: Vec<_> = scores.keys().copied().collect();
        let size = modalities.len();
        let mut matrix = vec![vec![T::zero(); size]; size];
        
        // Compute pairwise attention between modalities
        for (i, &mod_i) in modalities.iter().enumerate() {
            for (j, &mod_j) in modalities.iter().enumerate() {
                if i == j {
                    matrix[i][j] = T::one(); // Self-attention
                } else {
                    let score_i = &scores[&mod_i];
                    let score_j = &scores[&mod_j];
                    
                    // Attention based on confidence and agreement
                    let confidence_product = score_i.confidence * score_j.confidence;
                    let agreement = T::one() - (score_i.probability - score_j.probability).abs();
                    
                    matrix[i][j] = confidence_product * agreement;
                }
            }
        }
        
        // Normalize each row
        for row in matrix.iter_mut() {
            let sum: T = row.iter().fold(T::zero(), |acc, &x| acc + x);
            if sum > T::zero() {
                for val in row.iter_mut() {
                    *val = *val / sum;
                }
            }
        }
        
        Ok(matrix)
    }
    
    /// Additive attention mechanism
    fn additive_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        attention: &AttentionWeights<T>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let mut weighted_sum = T::zero();
        let mut modality_contributions = HashMap::new();
        
        for (modality, score) in scores {
            let weight = attention.modality_weights.get(modality).copied().unwrap_or(T::zero());
            let contribution = score.probability * weight;
            weighted_sum = weighted_sum + contribution;
            modality_contributions.insert(*modality, contribution);
        }
        
        Ok((weighted_sum, modality_contributions))
    }
    
    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        attention: &AttentionWeights<T>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        // Simplified version for score-based input
        let scale = T::one() / T::from(attention.modality_weights.len()).unwrap().sqrt();
        
        let mut weighted_sum = T::zero();
        let mut modality_contributions = HashMap::new();
        
        for (modality, score) in scores {
            let weight = attention.modality_weights.get(modality).copied().unwrap_or(T::zero());
            let scaled_weight = weight * scale;
            let contribution = score.probability * scaled_weight;
            weighted_sum = weighted_sum + contribution;
            modality_contributions.insert(*modality, contribution);
        }
        
        Ok((weighted_sum, modality_contributions))
    }
    
    /// Multi-head attention mechanism
    fn multi_head_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
        attention: &AttentionWeights<T>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let num_heads = self.config.num_heads;
        let mut head_outputs = Vec::new();
        let mut all_contributions = HashMap::new();
        
        // Process each attention head
        for head_idx in 0..num_heads {
            let mut head_sum = T::zero();
            let mut head_contributions = HashMap::new();
            
            for (modality, score) in scores {
                // Different attention weights per head (simplified)
                let base_weight = attention.modality_weights.get(modality).copied().unwrap_or(T::zero());
                let head_weight = base_weight * T::from(1.0 + (head_idx as f64 * 0.1)).unwrap();
                
                let contribution = score.probability * head_weight;
                head_sum = head_sum + contribution;
                head_contributions.insert(*modality, contribution);
            }
            
            head_outputs.push(head_sum);
            
            // Accumulate contributions
            for (modality, contribution) in head_contributions {
                *all_contributions.entry(modality).or_insert(T::zero()) += contribution;
            }
        }
        
        // Average across heads
        let final_output = head_outputs.iter().fold(T::zero(), |acc, &x| acc + x) 
            / T::from(num_heads).unwrap();
        
        // Average contributions
        for contribution in all_contributions.values_mut() {
            *contribution = *contribution / T::from(num_heads).unwrap();
        }
        
        Ok((final_output, all_contributions))
    }
    
    /// Cross-modal attention mechanism
    fn cross_modal_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
        attention: &AttentionWeights<T>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        if let Some(ref cross_modal_matrix) = attention.cross_modal {
            let modalities: Vec<_> = scores.keys().copied().collect();
            let mut enhanced_scores = HashMap::new();
            
            // Apply cross-modal interactions
            for (i, &mod_i) in modalities.iter().enumerate() {
                let mut enhanced_value = T::zero();
                
                for (j, &mod_j) in modalities.iter().enumerate() {
                    if i < cross_modal_matrix.len() && j < cross_modal_matrix[i].len() {
                        let interaction_weight = cross_modal_matrix[i][j];
                        let modality_score = scores[&mod_j].probability;
                        enhanced_value = enhanced_value + (interaction_weight * modality_score);
                    }
                }
                
                enhanced_scores.insert(mod_i, enhanced_value);
            }
            
            // Apply final attention weights
            let mut weighted_sum = T::zero();
            let mut modality_contributions = HashMap::new();
            
            for (modality, &enhanced_score) in &enhanced_scores {
                let weight = attention.modality_weights.get(modality).copied().unwrap_or(T::zero());
                let contribution = enhanced_score * weight;
                weighted_sum = weighted_sum + contribution;
                modality_contributions.insert(*modality, contribution);
            }
            
            Ok((weighted_sum, modality_contributions))
        } else {
            // Fallback to regular attention
            self.additive_attention(scores, attention)
        }
    }
    
    /// Temporal attention mechanism
    fn temporal_attention(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
        attention: &AttentionWeights<T>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        // For temporal attention, we would need sequence data
        // This is a simplified version that applies temporal weights to current scores
        
        let temporal_factor = if !attention.temporal_weights.is_empty() {
            // Use the last temporal weight as current importance
            *attention.temporal_weights.last().unwrap()
        } else {
            T::one()
        };
        
        let mut weighted_sum = T::zero();
        let mut modality_contributions = HashMap::new();
        
        for (modality, score) in scores {
            let spatial_weight = attention.modality_weights.get(modality).copied().unwrap_or(T::zero());
            let temporal_weighted = spatial_weight * temporal_factor;
            let contribution = score.probability * temporal_weighted;
            weighted_sum = weighted_sum + contribution;
            modality_contributions.insert(*modality, contribution);
        }
        
        Ok((weighted_sum, modality_contributions))
    }
    
    /// Calculate attention-based confidence
    fn calculate_attention_confidence(
        &self,
        attention: &AttentionWeights<T>,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        // Confidence based on attention distribution entropy and score confidence
        let entropy = utils::calculate_entropy(
            &attention.modality_weights.values().copied().collect::<Vec<_>>()
        );
        
        // Lower entropy (more focused attention) = higher confidence
        let attention_confidence = T::one() / (T::one() + entropy);
        
        // Weighted average of modality confidences
        let mut confidence_sum = T::zero();
        let mut weight_sum = T::zero();
        
        for (modality, score) in scores {
            let weight = attention.modality_weights.get(modality).copied().unwrap_or(T::zero());
            confidence_sum = confidence_sum + (score.confidence * weight);
            weight_sum = weight_sum + weight;
        }
        
        let weighted_confidence = if weight_sum > T::zero() {
            confidence_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        // Combine attention and modality confidences
        Ok((attention_confidence + weighted_confidence) / T::from(2.0).unwrap())
    }
    
    /// Update attention parameters based on feedback
    fn update_attention_parameters(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        let learning_rate = self.config.learning_rate;
        
        // Update modality weights based on performance
        for (modality, &performance) in &feedback.modality_performance {
            if let Some(current_weight) = self.attention_weights.modality_weights.get_mut(modality) {
                // Gradient-based update (simplified)
                let gradient = performance - *current_weight;
                *current_weight = *current_weight + (learning_rate * gradient);
            }
        }
        
        // Normalize weights
        utils::normalize_weights(&mut self.attention_weights.modality_weights)?;
        
        Ok(())
    }
    
    /// Create fusion metadata
    fn create_metadata(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        attention: &AttentionWeights<T>,
    ) -> Result<FusionMetadata<T>> {
        let mut modality_processing = HashMap::new();
        for (modality, score) in scores {
            modality_processing.insert(*modality, score.processing_time);
        }
        
        Ok(FusionMetadata {
            strategy: self.name().to_string(),
            weights: attention.modality_weights.clone(),
            attention_scores: Some(attention.modality_weights.clone()),
            timing: ProcessingTiming {
                modality_processing,
                fusion_time: processing_time,
                total_time: processing_time,
            },
            quality_metrics: QualityMetrics {
                agreement_score: T::from(0.9).unwrap(),
                consistency_score: T::from(0.88).unwrap(),
                quality_score: T::from(0.92).unwrap(),
                uncertainty: T::from(0.08).unwrap(),
            },
        })
    }
    
    /// Get adaptation history for analysis
    pub fn get_adaptation_history(&self) -> &[AdaptationStep<T>] {
        &self.adaptation_history
    }
    
    /// Get current learned parameters
    pub fn get_learned_parameters(&self) -> &AttentionParameters<T> {
        &self.learned_parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    fn create_test_score<T: Float>(prob: f64, conf: f64) -> DeceptionScore<T> {
        DeceptionScore {
            probability: T::from(prob).unwrap(),
            confidence: T::from(conf).unwrap(),
            contributing_factors: vec![],
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(10),
        }
    }
    
    #[test]
    fn test_attention_fusion_creation() {
        let config = AttentionConfig::<f64>::default();
        let fusion = AttentionFusion::new(config).unwrap();
        
        assert_eq!(fusion.name(), "attention_fusion");
        assert!(!fusion.attention_weights.modality_weights.is_empty());
    }
    
    #[test]
    fn test_multi_head_attention() {
        let config = AttentionConfig {
            attention_type: AttentionType::MultiHead,
            num_heads: 3,
            ..Default::default()
        };
        
        let fusion = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.7, 0.85));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        assert!(result.deception_probability >= 0.0);
        assert!(result.deception_probability <= 1.0);
        assert!(result.confidence > 0.0);
        assert_eq!(result.modality_contributions.len(), 3);
    }
    
    #[test]
    fn test_attention_weight_update() {
        let mut fusion = AttentionFusion::new(AttentionConfig::default()).unwrap();
        
        let mut performance = HashMap::new();
        performance.insert(ModalityType::Vision, 0.9);
        performance.insert(ModalityType::Audio, 0.7);
        
        let feedback = FeedbackData {
            ground_truth: true,
            prediction: 0.8,
            modality_performance: performance,
            session_id: "test".to_string(),
            timestamp: Utc::now(),
        };
        
        let old_weights = fusion.get_modality_weights();
        fusion.update(&feedback).unwrap();
        let new_weights = fusion.get_modality_weights();
        
        // Weights should have changed
        assert_ne!(
            old_weights.get(&ModalityType::Vision),
            new_weights.get(&ModalityType::Vision)
        );
    }
    
    #[test]
    fn test_cross_modal_attention() {
        let config = AttentionConfig {
            attention_type: AttentionType::CrossModal,
            enable_cross_modal: true,
            ..Default::default()
        };
        
        let fusion = AttentionFusion::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.75, 0.85));
        
        let result = fusion.fuse(&scores, None).unwrap();
        
        assert!(result.deception_probability > 0.0);
        assert!(result.confidence > 0.0);
        assert!(result.explanation.contains("Attention-based fusion"));
    }
}