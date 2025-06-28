//! Multi-modal fusion system for combining different modality analyses
//!
//! This module provides various fusion strategies for combining deception scores
//! from different modalities (vision, audio, text, physiological) into a unified
//! decision with explainable reasoning.

use crate::error::{Result, VeritasError, FusionError};
use crate::types::{
    AttentionWeights, CombinedFeatures, DeceptionScore, FusedDecision, ModalityType, VotingResult,
};
use hashbrown::HashMap;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod strategies;
pub mod attention_fusion;
pub mod temporal_alignment;
pub mod simd_optimized;

#[cfg(test)]
mod tests;

// Re-export key types and traits
pub use attention_fusion::{AttentionFusion, AttentionConfig};
pub use strategies::{EarlyFusion, LateFusion, HybridFusion, WeightedVoting};
pub use temporal_alignment::{TemporalAligner, AlignmentConfig};
pub use simd_optimized::{SimdFusionOps, SimdAttentionFusion, SimdFusionMetrics, FusionQualityMetrics};

/// Core trait for fusion strategies
pub trait FusionStrategy<T: Float + Send + Sync>: Send + Sync {
    /// Configuration type for this fusion strategy
    type Config: Clone + Send + Sync;
    
    /// Create a new fusion strategy with configuration
    fn new(config: Self::Config) -> Result<Self>
    where
        Self: Sized;
    
    /// Fuse multiple modality scores into a unified decision
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>>;
    
    /// Get the weight/importance of each modality
    fn get_modality_weights(&self) -> HashMap<ModalityType, T>;
    
    /// Update fusion parameters based on feedback
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()>;
    
    /// Get strategy name for logging/debugging
    fn name(&self) -> &'static str;
    
    /// Validate that required modalities are available
    fn validate_inputs(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<()> {
        if scores.is_empty() {
            return Err(FusionError::MissingModality {
                modality: "any".to_string(),
            });
        }
        Ok(())
    }
}

/// Feedback data for updating fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackData<T: Float> {
    /// Ground truth label (true for deceptive)
    pub ground_truth: bool,
    /// Predicted probability
    pub prediction: T,
    /// Individual modality performances
    pub modality_performance: HashMap<ModalityType, T>,
    /// Session identifier
    pub session_id: String,
    /// Feedback timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Result of fusion operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult<T: Float> {
    /// Fused decision
    pub decision: FusedDecision<T>,
    /// Intermediate fusion steps for debugging
    pub intermediate_steps: Vec<FusionStep<T>>,
    /// Quality assessment
    pub quality_score: T,
}

/// Individual step in fusion process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStep<T: Float> {
    /// Step name
    pub name: String,
    /// Input data
    pub inputs: Vec<String>,
    /// Output data
    pub output: String,
    /// Processing time
    pub duration: std::time::Duration,
    /// Confidence in this step
    pub confidence: T,
}

/// Fusion configuration shared across strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig<T: Float> {
    /// Minimum number of modalities required
    pub min_modalities: usize,
    /// Confidence threshold for decision
    pub confidence_threshold: T,
    /// Quality threshold for accepting results
    pub quality_threshold: T,
    /// Enable uncertainty estimation
    pub enable_uncertainty: bool,
    /// Maximum processing time before timeout
    pub max_processing_time: std::time::Duration,
    /// Default modality weights
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

/// Manager for multiple fusion strategies
#[derive(Debug)]
pub struct FusionManager<T: Float + Send + Sync> {
    /// Available fusion strategies
    strategies: HashMap<String, Arc<dyn FusionStrategy<T>>>,
    /// Default strategy name
    default_strategy: String,
    /// Temporal aligner
    temporal_aligner: TemporalAligner<T>,
    /// Configuration
    config: FusionConfig<T>,
}

impl<T: Float + Send + Sync> FusionManager<T> {
    /// Create a new fusion manager
    pub fn new(config: FusionConfig<T>) -> Result<Self> {
        let mut manager = Self {
            strategies: HashMap::new(),
            default_strategy: "late_fusion".to_string(),
            temporal_aligner: TemporalAligner::new(AlignmentConfig::default())?,
            config,
        };
        
        // Register default strategies
        manager.register_default_strategies()?;
        
        Ok(manager)
    }
    
    /// Register default fusion strategies
    fn register_default_strategies(&mut self) -> Result<()> {
        // Late fusion strategy
        let late_fusion = Arc::new(LateFusion::new(Default::default())?);
        self.strategies.insert("late_fusion".to_string(), late_fusion);
        
        // Early fusion strategy
        let early_fusion = Arc::new(EarlyFusion::new(Default::default())?);
        self.strategies.insert("early_fusion".to_string(), early_fusion);
        
        // Hybrid fusion strategy
        let hybrid_fusion = Arc::new(HybridFusion::new(Default::default())?);
        self.strategies.insert("hybrid_fusion".to_string(), hybrid_fusion);
        
        // Attention-based fusion
        let attention_fusion = Arc::new(AttentionFusion::new(AttentionConfig::default())?);
        self.strategies.insert("attention_fusion".to_string(), attention_fusion);
        
        // Weighted voting
        let weighted_voting = Arc::new(WeightedVoting::new(Default::default())?);
        self.strategies.insert("weighted_voting".to_string(), weighted_voting);
        
        Ok(())
    }
    
    /// Register a custom fusion strategy
    pub fn register_strategy(
        &mut self,
        name: String,
        strategy: Arc<dyn FusionStrategy<T>>,
    ) {
        self.strategies.insert(name, strategy);
    }
    
    /// Perform fusion using specified strategy
    pub fn fuse(
        &self,
        strategy_name: &str,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusionResult<T>> {
        let start_time = std::time::Instant::now();
        
        // Get the fusion strategy
        let strategy = self.strategies.get(strategy_name)
            .ok_or_else(|| FusionError::StrategyNotConfigured)?;
        
        // Validate inputs
        strategy.validate_inputs(scores)?;
        
        // Check minimum modalities requirement
        if scores.len() < self.config.min_modalities {
            return Err(FusionError::InsufficientConfidence {
                confidence: scores.len() as f64,
                threshold: self.config.min_modalities as f64,
            });
        }
        
        // Perform temporal alignment if features are provided
        let aligned_features = if let Some(features) = features {
            Some(self.temporal_aligner.align(features)?)
        } else {
            None
        };
        
        // Perform fusion
        let decision = strategy.fuse(scores, aligned_features.as_ref())?;
        
        // Calculate quality score
        let quality_score = self.calculate_quality_score(&decision, scores)?;
        
        // Check quality threshold
        if quality_score < self.config.quality_threshold {
            return Err(FusionError::InsufficientConfidence {
                confidence: quality_score.to_f64().unwrap(),
                threshold: self.config.quality_threshold.to_f64().unwrap(),
            });
        }
        
        let processing_time = start_time.elapsed();
        
        // Build result with intermediate steps
        let intermediate_steps = vec![
            FusionStep {
                name: "input_validation".to_string(),
                inputs: scores.keys().map(|k| k.to_string()).collect(),
                output: "validated".to_string(),
                duration: std::time::Duration::from_millis(1),
                confidence: T::from(1.0).unwrap(),
            },
            FusionStep {
                name: "temporal_alignment".to_string(),
                inputs: vec!["features".to_string()],
                output: "aligned_features".to_string(),
                duration: std::time::Duration::from_millis(5),
                confidence: T::from(0.95).unwrap(),
            },
            FusionStep {
                name: strategy.name().to_string(),
                inputs: vec!["scores".to_string(), "features".to_string()],
                output: "fused_decision".to_string(),
                duration: processing_time,
                confidence: decision.confidence,
            },
        ];
        
        Ok(FusionResult {
            decision,
            intermediate_steps,
            quality_score,
        })
    }
    
    /// Perform fusion using default strategy
    pub fn fuse_default(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusionResult<T>> {
        self.fuse(&self.default_strategy, scores, features)
    }
    
    /// Calculate quality score for fusion result
    fn calculate_quality_score(
        &self,
        decision: &FusedDecision<T>,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        // Inter-modality agreement
        let probabilities: Vec<T> = scores.values().map(|s| s.probability).collect();
        let mean_prob = probabilities.iter().fold(T::zero(), |acc, &x| acc + x) 
            / T::from(probabilities.len()).unwrap();
        
        let variance = probabilities.iter()
            .map(|&p| {
                let diff = p - mean_prob;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / T::from(probabilities.len()).unwrap();
        
        let agreement = T::one() - variance.sqrt();
        
        // Confidence-weighted quality
        let confidence_weight = decision.confidence;
        
        // Combine metrics
        let quality = (agreement * T::from(0.6).unwrap()) + (confidence_weight * T::from(0.4).unwrap());
        
        Ok(quality.max(T::zero()).min(T::one()))
    }
    
    /// List available strategies
    pub fn list_strategies(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }
    
    /// Set default strategy
    pub fn set_default_strategy(&mut self, name: String) -> Result<()> {
        if !self.strategies.contains_key(&name) {
            return Err(FusionError::StrategyNotConfigured);
        }
        self.default_strategy = name;
        Ok(())
    }
}

/// Utility functions for fusion operations
pub mod utils {
    use super::*;
    
    /// Normalize weights to sum to 1.0
    pub fn normalize_weights<T: Float>(weights: &mut HashMap<ModalityType, T>) -> Result<()> {
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
    
    /// Calculate weighted average of scores
    pub fn weighted_average<T: Float>(
        scores: &HashMap<ModalityType, T>,
        weights: &HashMap<ModalityType, T>,
    ) -> T {
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        
        for (modality, &score) in scores {
            if let Some(&weight) = weights.get(modality) {
                weighted_sum = weighted_sum + (score * weight);
                weight_sum = weight_sum + weight;
            }
        }
        
        if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::zero()
        }
    }
    
    /// Calculate entropy of score distribution
    pub fn calculate_entropy<T: Float>(scores: &[T]) -> T {
        let mut entropy = T::zero();
        let sum: T = scores.iter().fold(T::zero(), |acc, &x| acc + x);
        
        if sum <= T::zero() {
            return T::zero();
        }
        
        for &score in scores {
            if score > T::zero() {
                let prob = score / sum;
                entropy = entropy - (prob * prob.ln());
            }
        }
        
        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    #[test]
    fn test_fusion_manager_creation() {
        let config = FusionConfig::<f64>::default();
        let manager = FusionManager::new(config).unwrap();
        
        let strategies = manager.list_strategies();
        assert!(strategies.contains(&"late_fusion".to_string()));
        assert!(strategies.contains(&"early_fusion".to_string()));
        assert!(strategies.contains(&"attention_fusion".to_string()));
    }
    
    #[test]
    fn test_normalize_weights() {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Vision, 0.6);
        weights.insert(ModalityType::Audio, 0.8);
        
        utils::normalize_weights(&mut weights).unwrap();
        
        let sum: f64 = weights.values().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_weighted_average() {
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, 0.8);
        scores.insert(ModalityType::Audio, 0.6);
        
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Vision, 0.7);
        weights.insert(ModalityType::Audio, 0.3);
        
        let avg = utils::weighted_average(&scores, &weights);
        let expected = (0.8 * 0.7 + 0.6 * 0.3) / (0.7 + 0.3);
        assert!((avg - expected).abs() < 1e-10);
    }
}