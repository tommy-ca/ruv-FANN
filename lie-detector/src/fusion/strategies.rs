//! Fusion strategies for combining multi-modal deception scores
//!
//! This module implements various fusion strategies:
//! - Early Fusion: Combine features before neural network processing
//! - Late Fusion: Combine decisions after individual modality processing
//! - Hybrid Fusion: Combine both early and late fusion approaches
//! - Weighted Voting: Democratic voting with confidence weighting

use crate::error::{Result, VeritasError, FusionError};
use crate::fusion::{FeedbackData, FusionStrategy, utils};
use crate::types::{
    CombinedFeatures, DeceptionScore, FusedDecision, FusionMetadata, ModalityType,
    ProcessingTiming, QualityMetrics, VotingResult,
};
use hashbrown::HashMap;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Early fusion strategy - combines features before processing
#[derive(Debug, Clone)]
pub struct EarlyFusion<T: Float> {
    config: EarlyFusionConfig<T>,
    weights: HashMap<ModalityType, T>,
    feature_dim_map: HashMap<ModalityType, (usize, usize)>,
}

/// Configuration for early fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyFusionConfig<T: Float> {
    /// Dimensionality reduction target
    pub target_dimensions: usize,
    /// Feature normalization method
    pub normalization: FeatureNormalization,
    /// Modality weights for feature combination
    pub modality_weights: HashMap<ModalityType, T>,
    /// Enable feature selection
    pub enable_feature_selection: bool,
    /// Maximum feature correlation threshold
    pub max_correlation: T,
}

impl<T: Float> Default for EarlyFusionConfig<T> {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Vision, T::from(0.3).unwrap());
        weights.insert(ModalityType::Audio, T::from(0.25).unwrap());
        weights.insert(ModalityType::Text, T::from(0.25).unwrap());
        weights.insert(ModalityType::Physiological, T::from(0.2).unwrap());
        
        Self {
            target_dimensions: 256,
            normalization: FeatureNormalization::ZScore,
            modality_weights: weights,
            enable_feature_selection: true,
            max_correlation: T::from(0.85).unwrap(),
        }
    }
}

/// Feature normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureNormalization {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// L2 normalization
    L2,
}

impl<T: Float + Send + Sync> FusionStrategy<T> for EarlyFusion<T> {
    type Config = EarlyFusionConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        let weights = config.modality_weights.clone();
        
        Ok(Self {
            config,
            weights,
            feature_dim_map: HashMap::new(),
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        if let Some(combined_features) = features {
            // Early fusion: work with combined features
            self.fuse_features(combined_features, scores)
        } else {
            // Fallback to score-based fusion
            self.fuse_scores(scores)
        }
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        self.weights.clone()
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        // Update weights based on modality performance
        for (modality, &performance) in &feedback.modality_performance {
            if let Some(current_weight) = self.weights.get_mut(modality) {
                // Adaptive weight update using exponential moving average
                let alpha = T::from(0.1).unwrap(); // Learning rate
                *current_weight = (*current_weight * (T::one() - alpha)) + (performance * alpha);
            }
        }
        
        // Normalize weights
        utils::normalize_weights(&mut self.weights)?;
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "early_fusion"
    }
}

impl<T: Float> EarlyFusion<T> {
    fn fuse_features(
        &self,
        combined_features: &CombinedFeatures<T>,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Extract and normalize features
        let normalized_features = self.normalize_features(&combined_features.combined)?;
        
        // Apply dimensionality reduction if needed
        let reduced_features = if normalized_features.len() > self.config.target_dimensions {
            self.reduce_dimensions(&normalized_features)?
        } else {
            normalized_features
        };
        
        // Generate fused prediction using simplified neural network
        let prediction = self.predict_from_features(&reduced_features)?;
        
        // Calculate confidence based on feature consistency
        let confidence = self.calculate_feature_confidence(&reduced_features, scores)?;
        
        // Generate modality contributions
        let mut modality_contributions = HashMap::new();
        for (modality, &weight) in &self.weights {
            if scores.contains_key(modality) {
                modality_contributions.insert(*modality, weight * prediction);
            }
        }
        
        let processing_time = start_time.elapsed();
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence,
            modality_contributions,
            explanation: format!(
                "Early fusion combined {} modalities using feature-level integration. \
                Confidence based on inter-feature consistency: {:.3}",
                scores.len(),
                confidence.to_f64().unwrap_or(0.0)
            ),
            metadata: self.create_metadata(processing_time, scores)?,
        })
    }
    
    fn fuse_scores(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<FusedDecision<T>> {
        // Fallback: weighted average of scores
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        let mut modality_contributions = HashMap::new();
        
        for (modality, score) in scores {
            if let Some(&weight) = self.weights.get(modality) {
                let contribution = score.probability * weight;
                weighted_sum = weighted_sum + contribution;
                weight_sum = weight_sum + weight;
                modality_contributions.insert(*modality, contribution);
            }
        }
        
        let prediction = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        let confidence = self.calculate_score_confidence(scores)?;
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence,
            modality_contributions,
            explanation: format!(
                "Early fusion fallback using score-level combination of {} modalities",
                scores.len()
            ),
            metadata: self.create_metadata(Duration::from_millis(1), scores)?,
        })
    }
    
    fn normalize_features(&self, features: &[T]) -> Result<Vec<T>> {
        match self.config.normalization {
            FeatureNormalization::None => Ok(features.to_vec()),
            FeatureNormalization::MinMax => self.normalize_minmax(features),
            FeatureNormalization::ZScore => self.normalize_zscore(features),
            FeatureNormalization::L2 => self.normalize_l2(features),
        }
    }
    
    fn normalize_minmax(&self, features: &[T]) -> Result<Vec<T>> {
        let min_val = features.iter().fold(T::infinity(), |acc, &x| acc.min(x));
        let max_val = features.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        let range = max_val - min_val;
        
        if range <= T::zero() {
            return Ok(features.to_vec());
        }
        
        Ok(features
            .iter()
            .map(|&x| (x - min_val) / range)
            .collect())
    }
    
    fn normalize_zscore(&self, features: &[T]) -> Result<Vec<T>> {
        let n = T::from(features.len()).unwrap();
        let mean = features.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        
        let variance = features.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / n;
        
        let std_dev = variance.sqrt();
        
        if std_dev <= T::zero() {
            return Ok(features.to_vec());
        }
        
        Ok(features
            .iter()
            .map(|&x| (x - mean) / std_dev)
            .collect())
    }
    
    fn normalize_l2(&self, features: &[T]) -> Result<Vec<T>> {
        let norm = features.iter()
            .fold(T::zero(), |acc, &x| acc + (x * x))
            .sqrt();
        
        if norm <= T::zero() {
            return Ok(features.to_vec());
        }
        
        Ok(features
            .iter()
            .map(|&x| x / norm)
            .collect())
    }
    
    fn reduce_dimensions(&self, features: &[T]) -> Result<Vec<T>> {
        // Simple dimensionality reduction using stride sampling
        let stride = features.len() / self.config.target_dimensions;
        let stride = stride.max(1);
        
        Ok(features
            .iter()
            .step_by(stride)
            .take(self.config.target_dimensions)
            .copied()
            .collect())
    }
    
    fn predict_from_features(&self, features: &[T]) -> Result<T> {
        // Simplified prediction using linear combination
        let sum = features.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / T::from(features.len()).unwrap();
        
        // Apply sigmoid activation
        let sigmoid_input = mean - T::from(0.5).unwrap();
        let prediction = T::one() / (T::one() + (-sigmoid_input).exp());
        
        Ok(prediction)
    }
    
    fn calculate_feature_confidence(
        &self,
        features: &[T],
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        // Confidence based on feature variance and score agreement
        let feature_variance = self.calculate_variance(features);
        let score_agreement = self.calculate_score_agreement(scores)?;
        
        // Lower variance and higher agreement = higher confidence
        let feature_conf = T::one() / (T::one() + feature_variance);
        let combined_conf = (feature_conf + score_agreement) / T::from(2.0).unwrap();
        
        Ok(combined_conf.max(T::zero()).min(T::one()))
    }
    
    fn calculate_score_confidence(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        self.calculate_score_agreement(scores)
    }
    
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        
        let n = T::from(values.len()).unwrap();
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        
        values.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / n
    }
    
    fn calculate_score_agreement(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        if scores.len() < 2 {
            return Ok(T::one());
        }
        
        let probabilities: Vec<T> = scores.values().map(|s| s.probability).collect();
        let variance = self.calculate_variance(&probabilities);
        
        // Higher variance = lower agreement
        Ok(T::one() / (T::one() + variance))
    }
    
    fn create_metadata(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<FusionMetadata<T>> {
        let mut modality_processing = HashMap::new();
        for (modality, score) in scores {
            modality_processing.insert(*modality, score.processing_time);
        }
        
        Ok(FusionMetadata {
            strategy: self.name().to_string(),
            weights: self.weights.clone(),
            attention_scores: None,
            timing: ProcessingTiming {
                modality_processing,
                fusion_time: processing_time,
                total_time: processing_time,
            },
            quality_metrics: QualityMetrics {
                agreement_score: self.calculate_score_agreement(scores)?,
                consistency_score: T::from(0.8).unwrap(), // Placeholder
                quality_score: T::from(0.85).unwrap(),   // Placeholder
                uncertainty: T::from(0.1).unwrap(),      // Placeholder
            },
        })
    }
}

/// Late fusion strategy - combines decisions after processing
#[derive(Debug, Clone)]
pub struct LateFusion<T: Float> {
    config: LateFusionConfig<T>,
    weights: HashMap<ModalityType, T>,
}

/// Configuration for late fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateFusionConfig<T: Float> {
    /// Fusion method
    pub method: LateFusionMethod,
    /// Confidence weighting factor
    pub confidence_weight: T,
    /// Minimum confidence threshold per modality
    pub min_confidence: T,
    /// Enable adaptive weighting
    pub adaptive_weights: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LateFusionMethod {
    /// Simple weighted average
    WeightedAverage,
    /// Product of probabilities
    Product,
    /// Maximum probability
    Maximum,
    /// Majority voting
    MajorityVote,
    /// Confidence-weighted average
    ConfidenceWeighted,
}

impl<T: Float> Default for LateFusionConfig<T> {
    fn default() -> Self {
        Self {
            method: LateFusionMethod::ConfidenceWeighted,
            confidence_weight: T::from(0.3).unwrap(),
            min_confidence: T::from(0.5).unwrap(),
            adaptive_weights: true,
        }
    }
}

impl<T: Float + Send + Sync> FusionStrategy<T> for LateFusion<T> {
    type Config = LateFusionConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Vision, T::from(0.3).unwrap());
        weights.insert(ModalityType::Audio, T::from(0.25).unwrap());
        weights.insert(ModalityType::Text, T::from(0.25).unwrap());
        weights.insert(ModalityType::Physiological, T::from(0.2).unwrap());
        
        Ok(Self {
            config,
            weights,
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Filter scores by confidence threshold
        let filtered_scores: HashMap<_, _> = scores
            .iter()
            .filter(|(_, score)| score.confidence >= self.config.min_confidence)
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        
        if filtered_scores.is_empty() {
            return Err(FusionError::InsufficientConfidence {
                confidence: 0.0,
                threshold: self.config.min_confidence.to_f64().unwrap_or(0.5),
            });
        }
        
        let (prediction, modality_contributions) = match self.config.method {
            LateFusionMethod::WeightedAverage => self.weighted_average(&filtered_scores)?,
            LateFusionMethod::Product => self.product_fusion(&filtered_scores)?,
            LateFusionMethod::Maximum => self.maximum_fusion(&filtered_scores)?,
            LateFusionMethod::MajorityVote => self.majority_vote(&filtered_scores)?,
            LateFusionMethod::ConfidenceWeighted => self.confidence_weighted(&filtered_scores)?,
        };
        
        let confidence = self.calculate_fusion_confidence(&filtered_scores)?;
        let processing_time = start_time.elapsed();
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence,
            modality_contributions,
            explanation: format!(
                "Late fusion using {:?} method with {} modalities. \
                Average confidence: {:.3}",
                self.config.method,
                filtered_scores.len(),
                confidence.to_f64().unwrap_or(0.0)
            ),
            metadata: self.create_metadata(processing_time, &filtered_scores)?,
        })
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        self.weights.clone()
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        if self.config.adaptive_weights {
            // Update weights based on performance feedback
            for (modality, &performance) in &feedback.modality_performance {
                if let Some(current_weight) = self.weights.get_mut(modality) {
                    let alpha = T::from(0.05).unwrap(); // Conservative learning rate
                    *current_weight = (*current_weight * (T::one() - alpha)) + (performance * alpha);
                }
            }
            utils::normalize_weights(&mut self.weights)?;
        }
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "late_fusion"
    }
}

impl<T: Float> LateFusion<T> {
    fn weighted_average(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            if let Some(&weight) = self.weights.get(modality) {
                let contribution = score.probability * weight;
                weighted_sum = weighted_sum + contribution;
                weight_sum = weight_sum + weight;
                contributions.insert(*modality, contribution);
            }
        }
        
        let prediction = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        Ok((prediction, contributions))
    }
    
    fn product_fusion(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let mut product = T::one();
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            product = product * score.probability;
            contributions.insert(*modality, score.probability);
        }
        
        Ok((product, contributions))
    }
    
    fn maximum_fusion(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let mut max_prob = T::zero();
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            if score.probability > max_prob {
                max_prob = score.probability;
            }
            contributions.insert(*modality, score.probability);
        }
        
        Ok((max_prob, contributions))
    }
    
    fn majority_vote(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let threshold = T::from(0.5).unwrap();
        let mut positive_votes = 0;
        let mut total_votes = 0;
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            total_votes += 1;
            let vote = if score.probability > threshold { T::one() } else { T::zero() };
            if vote > T::zero() {
                positive_votes += 1;
            }
            contributions.insert(*modality, vote);
        }
        
        let prediction = if positive_votes > total_votes / 2 {
            T::one()
        } else {
            T::zero()
        };
        
        Ok((prediction, contributions))
    }
    
    fn confidence_weighted(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<(T, HashMap<ModalityType, T>)> {
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        let mut contributions = HashMap::new();
        
        for (modality, score) in scores {
            let base_weight = self.weights.get(modality).copied().unwrap_or(T::one());
            let confidence_factor = T::one() + (score.confidence * self.config.confidence_weight);
            let total_weight = base_weight * confidence_factor;
            
            let contribution = score.probability * total_weight;
            weighted_sum = weighted_sum + contribution;
            weight_sum = weight_sum + total_weight;
            contributions.insert(*modality, contribution);
        }
        
        let prediction = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        Ok((prediction, contributions))
    }
    
    fn calculate_fusion_confidence(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        if scores.is_empty() {
            return Ok(T::zero());
        }
        
        // Average confidence weighted by modality weights
        let mut confidence_sum = T::zero();
        let mut weight_sum = T::zero();
        
        for (modality, score) in scores {
            let weight = self.weights.get(modality).copied().unwrap_or(T::one());
            confidence_sum = confidence_sum + (score.confidence * weight);
            weight_sum = weight_sum + weight;
        }
        
        Ok(if weight_sum > T::zero() {
            confidence_sum / weight_sum
        } else {
            T::zero()
        })
    }
    
    fn create_metadata(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<FusionMetadata<T>> {
        let mut modality_processing = HashMap::new();
        for (modality, score) in scores {
            modality_processing.insert(*modality, score.processing_time);
        }
        
        Ok(FusionMetadata {
            strategy: self.name().to_string(),
            weights: self.weights.clone(),
            attention_scores: None,
            timing: ProcessingTiming {
                modality_processing,
                fusion_time: processing_time,
                total_time: processing_time,
            },
            quality_metrics: QualityMetrics {
                agreement_score: T::from(0.85).unwrap(), // Placeholder
                consistency_score: T::from(0.8).unwrap(), // Placeholder
                quality_score: T::from(0.82).unwrap(),   // Placeholder
                uncertainty: T::from(0.15).unwrap(),     // Placeholder
            },
        })
    }
}

/// Hybrid fusion combining early and late fusion
#[derive(Debug, Clone)]
pub struct HybridFusion<T: Float> {
    early_fusion: EarlyFusion<T>,
    late_fusion: LateFusion<T>,
    config: HybridFusionConfig<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridFusionConfig<T: Float> {
    /// Weight for early fusion component
    pub early_weight: T,
    /// Weight for late fusion component
    pub late_weight: T,
    /// Minimum agreement threshold between fusion methods
    pub min_agreement: T,
}

impl<T: Float> Default for HybridFusionConfig<T> {
    fn default() -> Self {
        Self {
            early_weight: T::from(0.4).unwrap(),
            late_weight: T::from(0.6).unwrap(),
            min_agreement: T::from(0.7).unwrap(),
        }
    }
}

impl<T: Float + Send + Sync> FusionStrategy<T> for HybridFusion<T> {
    type Config = HybridFusionConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        let early_fusion = EarlyFusion::new(EarlyFusionConfig::default())?;
        let late_fusion = LateFusion::new(LateFusionConfig::default())?;
        
        Ok(Self {
            early_fusion,
            late_fusion,
            config,
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Get predictions from both fusion methods
        let early_result = self.early_fusion.fuse(scores, features)?;
        let late_result = self.late_fusion.fuse(scores, features)?;
        
        // Check agreement between methods
        let agreement = self.calculate_agreement(
            early_result.deception_probability,
            late_result.deception_probability,
        );
        
        if agreement < self.config.min_agreement {
            return Err(FusionError::InsufficientConfidence {
                confidence: agreement.to_f64().unwrap_or(0.0),
                threshold: self.config.min_agreement.to_f64().unwrap_or(0.7),
            });
        }
        
        // Combine predictions
        let combined_prediction = (early_result.deception_probability * self.config.early_weight)
            + (late_result.deception_probability * self.config.late_weight);
        
        let combined_confidence = (early_result.confidence * self.config.early_weight)
            + (late_result.confidence * self.config.late_weight);
        
        // Combine modality contributions
        let mut modality_contributions = HashMap::new();
        for modality in scores.keys() {
            let early_contrib = early_result.modality_contributions.get(modality).copied().unwrap_or(T::zero());
            let late_contrib = late_result.modality_contributions.get(modality).copied().unwrap_or(T::zero());
            let combined_contrib = (early_contrib * self.config.early_weight)
                + (late_contrib * self.config.late_weight);
            modality_contributions.insert(*modality, combined_contrib);
        }
        
        let processing_time = start_time.elapsed();
        
        Ok(FusedDecision {
            deception_probability: combined_prediction,
            confidence: combined_confidence,
            modality_contributions,
            explanation: format!(
                "Hybrid fusion combining early (weight: {:.2}) and late (weight: {:.2}) fusion. \
                Agreement: {:.3}, Combined confidence: {:.3}",
                self.config.early_weight.to_f64().unwrap_or(0.0),
                self.config.late_weight.to_f64().unwrap_or(0.0),
                agreement.to_f64().unwrap_or(0.0),
                combined_confidence.to_f64().unwrap_or(0.0)
            ),
            metadata: self.create_metadata(processing_time, scores, agreement)?,
        })
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        // Combine weights from both fusion methods
        let early_weights = self.early_fusion.get_modality_weights();
        let late_weights = self.late_fusion.get_modality_weights();
        
        let mut combined_weights = HashMap::new();
        for modality in [ModalityType::Vision, ModalityType::Audio, ModalityType::Text, ModalityType::Physiological] {
            let early_weight = early_weights.get(&modality).copied().unwrap_or(T::zero());
            let late_weight = late_weights.get(&modality).copied().unwrap_or(T::zero());
            let combined = (early_weight * self.config.early_weight) + (late_weight * self.config.late_weight);
            combined_weights.insert(modality, combined);
        }
        
        combined_weights
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        self.early_fusion.update(feedback)?;
        self.late_fusion.update(feedback)?;
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "hybrid_fusion"
    }
}

impl<T: Float> HybridFusion<T> {
    fn calculate_agreement(&self, prediction1: T, prediction2: T) -> T {
        let diff = (prediction1 - prediction2).abs();
        T::one() - diff
    }
    
    fn create_metadata(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        agreement: T,
    ) -> Result<FusionMetadata<T>> {
        let mut modality_processing = HashMap::new();
        for (modality, score) in scores {
            modality_processing.insert(*modality, score.processing_time);
        }
        
        Ok(FusionMetadata {
            strategy: self.name().to_string(),
            weights: self.get_modality_weights(),
            attention_scores: None,
            timing: ProcessingTiming {
                modality_processing,
                fusion_time: processing_time,
                total_time: processing_time,
            },
            quality_metrics: QualityMetrics {
                agreement_score: agreement,
                consistency_score: T::from(0.85).unwrap(),
                quality_score: T::from(0.87).unwrap(),
                uncertainty: T::from(0.12).unwrap(),
            },
        })
    }
}

/// Weighted voting fusion strategy
#[derive(Debug, Clone)]
pub struct WeightedVoting<T: Float> {
    config: WeightedVotingConfig<T>,
    weights: HashMap<ModalityType, T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedVotingConfig<T: Float> {
    /// Voting threshold (0.5 for majority)
    pub voting_threshold: T,
    /// Confidence threshold for considering a vote
    pub confidence_threshold: T,
    /// Enable confidence weighting
    pub use_confidence_weighting: bool,
    /// Minimum number of votes required
    pub min_votes: usize,
}

impl<T: Float> Default for WeightedVotingConfig<T> {
    fn default() -> Self {
        Self {
            voting_threshold: T::from(0.5).unwrap(),
            confidence_threshold: T::from(0.6).unwrap(),
            use_confidence_weighting: true,
            min_votes: 2,
        }
    }
}

impl<T: Float + Send + Sync> FusionStrategy<T> for WeightedVoting<T> {
    type Config = WeightedVotingConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Vision, T::from(0.3).unwrap());
        weights.insert(ModalityType::Audio, T::from(0.25).unwrap());
        weights.insert(ModalityType::Text, T::from(0.25).unwrap());
        weights.insert(ModalityType::Physiological, T::from(0.2).unwrap());
        
        Ok(Self {
            config,
            weights,
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Filter by confidence threshold
        let valid_scores: HashMap<_, _> = scores
            .iter()
            .filter(|(_, score)| score.confidence >= self.config.confidence_threshold)
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        
        if valid_scores.len() < self.config.min_votes {
            return Err(FusionError::InsufficientConfidence {
                confidence: valid_scores.len() as f64,
                threshold: self.config.min_votes as f64,
            });
        }
        
        let voting_result = self.perform_voting(&valid_scores)?;
        
        let prediction = if voting_result.decision {
            T::from(0.8).unwrap() // High confidence for positive vote
        } else {
            T::from(0.2).unwrap() // Low confidence for negative vote
        };
        
        let processing_time = start_time.elapsed();
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence: voting_result.consensus_confidence,
            modality_contributions: voting_result.vote_confidence,
            explanation: format!(
                "Weighted voting with {} valid votes. Decision: {}, Consensus: {:.3}",
                valid_scores.len(),
                if voting_result.decision { "Deceptive" } else { "Truthful" },
                voting_result.consensus_confidence.to_f64().unwrap_or(0.0)
            ),
            metadata: self.create_metadata(processing_time, &valid_scores)?,
        })
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        self.weights.clone()
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        // Update weights based on voting accuracy
        for (modality, &performance) in &feedback.modality_performance {
            if let Some(current_weight) = self.weights.get_mut(modality) {
                let alpha = T::from(0.08).unwrap();
                *current_weight = (*current_weight * (T::one() - alpha)) + (performance * alpha);
            }
        }
        utils::normalize_weights(&mut self.weights)?;
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "weighted_voting"
    }
}

impl<T: Float> WeightedVoting<T> {
    fn perform_voting(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<VotingResult<T>> {
        let mut weighted_positive = T::zero();
        let mut weighted_total = T::zero();
        let mut votes = HashMap::new();
        let mut vote_confidence = HashMap::new();
        
        for (modality, score) in scores {
            let base_weight = self.weights.get(modality).copied().unwrap_or(T::one());
            
            let weight = if self.config.use_confidence_weighting {
                base_weight * score.confidence
            } else {
                base_weight
            };
            
            let vote = score.probability > self.config.voting_threshold;
            votes.insert(*modality, vote);
            vote_confidence.insert(*modality, score.confidence);
            
            if vote {
                weighted_positive = weighted_positive + weight;
            }
            weighted_total = weighted_total + weight;
        }
        
        let decision = weighted_positive > (weighted_total / T::from(2.0).unwrap());
        let consensus_confidence = if weighted_total > T::zero() {
            if decision {
                weighted_positive / weighted_total
            } else {
                (weighted_total - weighted_positive) / weighted_total
            }
        } else {
            T::zero()
        };
        
        Ok(VotingResult {
            decision,
            votes,
            vote_confidence,
            consensus_confidence,
        })
    }
    
    fn create_metadata(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<FusionMetadata<T>> {
        let mut modality_processing = HashMap::new();
        for (modality, score) in scores {
            modality_processing.insert(*modality, score.processing_time);
        }
        
        Ok(FusionMetadata {
            strategy: self.name().to_string(),
            weights: self.weights.clone(),
            attention_scores: None,
            timing: ProcessingTiming {
                modality_processing,
                fusion_time: processing_time,
                total_time: processing_time,
            },
            quality_metrics: QualityMetrics {
                agreement_score: T::from(0.9).unwrap(),  // High for voting
                consistency_score: T::from(0.85).unwrap(),
                quality_score: T::from(0.88).unwrap(),
                uncertainty: T::from(0.1).unwrap(),
            },
        })
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
    fn test_late_fusion_weighted_average() {
        let fusion = LateFusion::new(LateFusionConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.6, 0.8));
        
        let result = fusion.fuse(&scores, None).unwrap();
        assert!(result.deception_probability > 0.0);
        assert!(result.deception_probability < 1.0);
        assert!(result.confidence > 0.0);
    }
    
    #[test]
    fn test_weighted_voting() {
        let voting = WeightedVoting::new(WeightedVotingConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.8, 0.9));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.7, 0.8));
        scores.insert(ModalityType::Text, create_test_score::<f64>(0.6, 0.85));
        
        let result = voting.fuse(&scores, None).unwrap();
        assert!(result.confidence > 0.0);
        assert_eq!(result.modality_contributions.len(), 3);
    }
    
    #[test]
    fn test_hybrid_fusion() {
        let hybrid = HybridFusion::new(HybridFusionConfig::default()).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(ModalityType::Vision, create_test_score::<f64>(0.75, 0.9));
        scores.insert(ModalityType::Audio, create_test_score::<f64>(0.65, 0.8));
        
        let result = hybrid.fuse(&scores, None).unwrap();
        assert!(result.deception_probability > 0.0);
        assert!(result.confidence > 0.0);
    }
}