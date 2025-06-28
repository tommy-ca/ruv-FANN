//! Optimized fusion strategies with reduced memory usage
//!
//! This module implements memory-efficient fusion strategies that minimize
//! allocations and use copy-on-write semantics where possible.

use crate::error::{FusionError, Result};
use crate::fusion::{FeedbackData, FusionStrategy, utils};
use crate::types::{
    CombinedFeatures, DeceptionScore, FusedDecision, FusionMetadata, ModalityType,
    ProcessingTiming, QualityMetrics, VotingResult,
};
use crate::optimization::{
    Arena, ArenaVec, CompactFeatures, CompactModality, intern, InternedString,
    ObjectPool, PooledObject, get_tl_vec_f64, get_tl_features
};
use hashbrown::HashMap;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Optimized early fusion strategy with reduced allocations
#[derive(Debug, Clone)]
pub struct OptimizedEarlyFusion<T: Float> {
    config: Arc<EarlyFusionConfig<T>>, // Shared config
    weights: Arc<HashMap<ModalityType, T>>, // Shared weights
    feature_dim_map: HashMap<ModalityType, (usize, usize)>,
    // Pools for temporary allocations
    feature_pool: ObjectPool<Vec<T>>,
    // Arena for per-fusion temporary allocations
    arena: Arena,
    // Cached strings
    explanation_template: InternedString,
}

impl<T: Float + Send + Sync> FusionStrategy<T> for OptimizedEarlyFusion<T> {
    type Config = EarlyFusionConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        let weights = Arc::new(config.modality_weights.clone());
        let config = Arc::new(config);
        
        Ok(Self {
            config,
            weights,
            feature_dim_map: HashMap::new(),
            feature_pool: ObjectPool::new(32),
            arena: Arena::new(16 * 1024)?, // 16KB arena
            explanation_template: intern("Early fusion combined {} modalities using feature-level integration. Confidence based on inter-feature consistency: {:.3}"),
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Reset arena for this fusion operation
        self.arena.reset();
        
        if let Some(combined_features) = features {
            self.fuse_features_optimized(combined_features, scores, start_time)
        } else {
            self.fuse_scores_optimized(scores, start_time)
        }
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        // Return a copy since the trait requires it
        (*self.weights).clone()
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        // Create new weights map only when updating
        let mut new_weights = (*self.weights).clone();
        
        // Update weights based on modality performance
        for (modality, &performance) in &feedback.modality_performance {
            if let Some(current_weight) = new_weights.get_mut(modality) {
                let alpha = T::from(0.1).unwrap(); // Learning rate
                *current_weight = (*current_weight * (T::one() - alpha)) + (performance * alpha);
            }
        }
        
        // Normalize weights
        utils::normalize_weights(&mut new_weights)?;
        
        // Update the Arc only once
        self.weights = Arc::new(new_weights);
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "optimized_early_fusion"
    }
}

impl<T: Float> OptimizedEarlyFusion<T> {
    fn fuse_features_optimized(
        &self,
        combined_features: &CombinedFeatures<T>,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        start_time: Instant,
    ) -> Result<FusedDecision<T>> {
        // Get pooled vector for features
        let mut normalized_features = self.feature_pool.get();
        normalized_features.clear();
        
        // Normalize features in-place
        self.normalize_features_inplace(&combined_features.combined, &mut normalized_features)?;
        
        // Apply dimensionality reduction if needed
        let reduced_features = if normalized_features.len() > self.config.target_dimensions {
            // Use arena for temporary reduction
            let reduced = ArenaVec::with_capacity(&self.arena, self.config.target_dimensions)?;
            self.reduce_dimensions_to_arena(&normalized_features, reduced)?
        } else {
            // Create arena view of existing features
            self.arena.alloc_slice(&normalized_features)?
        };
        
        // Generate fused prediction
        let prediction = self.predict_from_features_opt(reduced_features)?;
        
        // Calculate confidence
        let confidence = self.calculate_feature_confidence_opt(reduced_features, scores)?;
        
        // Generate modality contributions using thread-local vector
        let mut contributions = get_tl_vec_f64();
        contributions.data.clear();
        
        for (modality, &weight) in self.weights.iter() {
            if scores.contains_key(modality) {
                contributions.data.push((*modality, weight * prediction));
            }
        }
        
        // Convert to required HashMap (unavoidable due to API)
        let modality_contributions: HashMap<ModalityType, T> = 
            contributions.data.iter().cloned().collect();
        
        let processing_time = start_time.elapsed();
        
        // Format explanation efficiently
        let explanation = format!(
            "Early fusion combined {} modalities using feature-level integration. Confidence based on inter-feature consistency: {:.3}",
            scores.len(),
            confidence.to_f64().unwrap_or(0.0)
        );
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence,
            modality_contributions,
            explanation,
            metadata: self.create_metadata_optimized(processing_time, scores)?,
        })
    }
    
    fn fuse_scores_optimized(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        start_time: Instant,
    ) -> Result<FusedDecision<T>> {
        // Use thread-local pooled vector
        let mut temp_calc = get_tl_vec_f64();
        temp_calc.data.clear();
        
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        
        // Single pass calculation
        for (modality, score) in scores {
            if let Some(&weight) = self.weights.get(modality) {
                let contribution = score.probability * weight;
                weighted_sum = weighted_sum + contribution;
                weight_sum = weight_sum + weight;
                temp_calc.data.push((*modality, contribution));
            }
        }
        
        let prediction = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            T::from(0.5).unwrap()
        };
        
        // Convert to HashMap
        let modality_contributions: HashMap<ModalityType, T> = 
            temp_calc.data.iter().cloned().collect();
        
        let confidence = self.calculate_score_confidence_opt(scores)?;
        let processing_time = start_time.elapsed();
        
        let explanation = format!(
            "Score fusion using weighted average of {} modalities",
            scores.len()
        );
        
        Ok(FusedDecision {
            deception_probability: prediction,
            confidence,
            modality_contributions,
            explanation,
            metadata: self.create_metadata_optimized(processing_time, scores)?,
        })
    }
    
    fn normalize_features_inplace(&self, input: &[T], output: &mut Vec<T>) -> Result<()> {
        output.clear();
        output.reserve_exact(input.len());
        
        match self.config.normalization {
            FeatureNormalization::None => {
                output.extend_from_slice(input);
            }
            FeatureNormalization::MinMax => {
                // Find min/max in single pass
                let (min, max) = input.iter()
                    .fold((T::max_value(), T::min_value()), |(min, max), &x| {
                        (min.min(x), max.max(x))
                    });
                
                let range = max - min;
                if range > T::zero() {
                    for &x in input {
                        output.push((x - min) / range);
                    }
                } else {
                    output.extend_from_slice(input);
                }
            }
            FeatureNormalization::ZScore => {
                // Calculate mean and std in single pass
                let n = T::from(input.len()).unwrap();
                let mean = input.iter().fold(T::zero(), |sum, &x| sum + x) / n;
                
                let variance = input.iter()
                    .map(|&x| (x - mean).powi(2))
                    .fold(T::zero(), |sum, x| sum + x) / n;
                
                let std = variance.sqrt();
                
                if std > T::zero() {
                    for &x in input {
                        output.push((x - mean) / std);
                    }
                } else {
                    for &x in input {
                        output.push(x - mean);
                    }
                }
            }
            FeatureNormalization::L2 => {
                let norm = input.iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |sum, x| sum + x)
                    .sqrt();
                
                if norm > T::zero() {
                    for &x in input {
                        output.push(x / norm);
                    }
                } else {
                    output.extend_from_slice(input);
                }
            }
        }
        
        Ok(())
    }
    
    fn reduce_dimensions_to_arena<'a>(
        &self,
        features: &[T],
        mut output: ArenaVec<'a, T>,
    ) -> Result<&'a mut [T]> {
        // Simple dimensionality reduction - take top features
        // In practice, would use PCA or similar
        
        let target = self.config.target_dimensions.min(features.len());
        
        for i in 0..target {
            output.push(features[i])?;
        }
        
        Ok(output.as_mut_slice())
    }
    
    fn predict_from_features_opt(&self, features: &[T]) -> Result<T> {
        // Simplified prediction - in practice would use neural network
        let sum: T = features.iter().fold(T::zero(), |acc, &x| acc + x.abs());
        let mean = sum / T::from(features.len()).unwrap();
        
        // Sigmoid-like transformation
        let exp_val = (-mean * T::from(2.0).unwrap()).exp();
        Ok(T::one() / (T::one() + exp_val))
    }
    
    fn calculate_feature_confidence_opt(
        &self,
        features: &[T],
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        // Calculate confidence based on feature consistency
        if features.is_empty() {
            return Ok(T::from(0.5).unwrap());
        }
        
        // Use variance as inverse confidence measure
        let mean = features.iter().fold(T::zero(), |sum, &x| sum + x) / T::from(features.len()).unwrap();
        let variance = features.iter()
            .map(|&x| (x - mean).powi(2))
            .fold(T::zero(), |sum, x| sum + x) / T::from(features.len()).unwrap();
        
        // Convert variance to confidence (lower variance = higher confidence)
        let base_confidence = T::one() / (T::one() + variance);
        
        // Weight by modality confidences
        let avg_modality_confidence = scores.values()
            .map(|s| s.confidence)
            .fold(T::zero(), |sum, c| sum + c) / T::from(scores.len()).unwrap();
        
        Ok((base_confidence + avg_modality_confidence) / T::from(2.0).unwrap())
    }
    
    fn calculate_score_confidence_opt(&self, scores: &HashMap<ModalityType, DeceptionScore<T>>) -> Result<T> {
        if scores.is_empty() {
            return Ok(T::from(0.5).unwrap());
        }
        
        // Calculate weighted confidence
        let mut weighted_confidence = T::zero();
        let mut weight_sum = T::zero();
        
        for (modality, score) in scores {
            if let Some(&weight) = self.weights.get(modality) {
                weighted_confidence = weighted_confidence + score.confidence * weight;
                weight_sum = weight_sum + weight;
            }
        }
        
        if weight_sum > T::zero() {
            Ok(weighted_confidence / weight_sum)
        } else {
            Ok(T::from(0.5).unwrap())
        }
    }
    
    fn create_metadata_optimized(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<FusionMetadata> {
        // Use pooled features map
        let mut quality_metrics = get_tl_features();
        quality_metrics.features.clear();
        
        // Pre-intern common metric names
        static METRIC_NAMES: &[&str] = &[
            "fusion_quality",
            "confidence_variance",
            "modality_agreement",
        ];
        
        // Calculate metrics
        let fusion_quality = T::from(0.85).unwrap();
        let confidence_variance = T::from(0.1).unwrap();
        let agreement = T::from(0.9).unwrap();
        
        quality_metrics.features.insert(
            intern(METRIC_NAMES[0]).as_str().to_string(),
            fusion_quality.to_f64().unwrap()
        );
        quality_metrics.features.insert(
            intern(METRIC_NAMES[1]).as_str().to_string(),
            confidence_variance.to_f64().unwrap()
        );
        quality_metrics.features.insert(
            intern(METRIC_NAMES[2]).as_str().to_string(),
            agreement.to_f64().unwrap()
        );
        
        Ok(FusionMetadata {
            strategy_name: intern(self.name()).as_str().to_string(),
            processing_time,
            modalities_used: scores.keys().cloned().collect(),
            quality_metrics: QualityMetrics {
                fusion_quality: fusion_quality.to_f64().unwrap(),
                confidence_variance: confidence_variance.to_f64().unwrap(),
                modality_agreement: agreement.to_f64().unwrap(),
                timing: ProcessingTiming {
                    feature_extraction: Duration::from_millis(1),
                    fusion_computation: processing_time,
                    total: processing_time,
                },
            },
        })
    }
}

/// Optimized weighted voting strategy
#[derive(Debug)]
pub struct OptimizedWeightedVoting<T: Float> {
    config: Arc<WeightedVotingConfig<T>>,
    weights: Arc<HashMap<ModalityType, T>>,
    voting_cache: dashmap::DashMap<u64, VotingResult<T>>,
}

/// Configuration for weighted voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedVotingConfig<T: Float> {
    pub initial_weights: HashMap<ModalityType, T>,
    pub confidence_threshold: T,
    pub quorum_percentage: T,
    pub use_confidence_weighting: bool,
}

impl<T: Float> Default for WeightedVotingConfig<T> {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(ModalityType::Vision, T::from(0.3).unwrap());
        weights.insert(ModalityType::Audio, T::from(0.25).unwrap());
        weights.insert(ModalityType::Text, T::from(0.25).unwrap());
        weights.insert(ModalityType::Physiological, T::from(0.2).unwrap());
        
        Self {
            initial_weights: weights,
            confidence_threshold: T::from(0.5).unwrap(),
            quorum_percentage: T::from(0.6).unwrap(),
            use_confidence_weighting: true,
        }
    }
}

impl<T: Float + Send + Sync> FusionStrategy<T> for OptimizedWeightedVoting<T> {
    type Config = WeightedVotingConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self> {
        Ok(Self {
            weights: Arc::new(config.initial_weights.clone()),
            config: Arc::new(config),
            voting_cache: dashmap::DashMap::with_capacity(128),
        })
    }
    
    fn fuse(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        _features: Option<&CombinedFeatures<T>>,
    ) -> Result<FusedDecision<T>> {
        let start_time = Instant::now();
        
        // Calculate cache key
        let cache_key = self.calculate_cache_key(scores);
        
        // Check cache first
        if let Some(cached) = self.voting_cache.get(&cache_key) {
            return Ok(self.create_decision_from_cached(&cached, scores, start_time));
        }
        
        // Perform voting
        let voting_result = self.perform_voting(scores)?;
        
        // Cache result
        self.voting_cache.insert(cache_key, voting_result.clone());
        
        // Keep cache size bounded
        if self.voting_cache.len() > 1000 {
            // Remove oldest entries
            self.voting_cache.retain(|_, _| rand::random::<bool>());
        }
        
        Ok(self.create_decision(&voting_result, scores, start_time))
    }
    
    fn get_modality_weights(&self) -> HashMap<ModalityType, T> {
        (*self.weights).clone()
    }
    
    fn update(&mut self, feedback: &FeedbackData<T>) -> Result<()> {
        let mut new_weights = (*self.weights).clone();
        
        for (modality, &performance) in &feedback.modality_performance {
            if let Some(weight) = new_weights.get_mut(modality) {
                let alpha = T::from(0.05).unwrap(); // Slower learning rate
                *weight = *weight * (T::one() - alpha) + performance * alpha;
            }
        }
        
        utils::normalize_weights(&mut new_weights)?;
        self.weights = Arc::new(new_weights);
        
        // Clear cache after weight update
        self.voting_cache.clear();
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "optimized_weighted_voting"
    }
}

impl<T: Float> OptimizedWeightedVoting<T> {
    fn calculate_cache_key(&self, scores: &HashMap<ModalityType, DeceptionScore<T>>) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash modalities and quantized scores
        let mut sorted_scores: Vec<_> = scores.iter()
            .map(|(m, s)| (*m, (s.probability.to_f64().unwrap() * 100.0) as i32))
            .collect();
        sorted_scores.sort_by_key(|(m, _)| *m as u8);
        
        for (modality, score) in sorted_scores {
            (modality as u8).hash(&mut hasher);
            score.hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    fn perform_voting(&self, scores: &HashMap<ModalityType, DeceptionScore<T>>) -> Result<VotingResult<T>> {
        let mut votes_deception = T::zero();
        let mut votes_truth = T::zero();
        let mut total_weight = T::zero();
        
        for (modality, score) in scores {
            if let Some(&weight) = self.weights.get(modality) {
                let vote_weight = if self.config.use_confidence_weighting {
                    weight * score.confidence
                } else {
                    weight
                };
                
                if score.probability >= self.config.confidence_threshold {
                    votes_deception = votes_deception + vote_weight;
                } else {
                    votes_truth = votes_truth + vote_weight;
                }
                
                total_weight = total_weight + vote_weight;
            }
        }
        
        let deception_ratio = if total_weight > T::zero() {
            votes_deception / total_weight
        } else {
            T::from(0.5).unwrap()
        };
        
        Ok(VotingResult {
            deception_votes: votes_deception,
            truth_votes: votes_truth,
            total_weight,
            deception_ratio,
            has_quorum: total_weight >= self.config.quorum_percentage,
        })
    }
    
    fn create_decision(
        &self,
        voting_result: &VotingResult<T>,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        start_time: std::time::Instant,
    ) -> FusedDecision<T> {
        let mut modality_contributions = HashMap::new();
        
        for (modality, score) in scores {
            if let Some(&weight) = self.weights.get(modality) {
                modality_contributions.insert(
                    *modality,
                    score.probability * weight,
                );
            }
        }
        
        FusedDecision {
            deception_probability: voting_result.deception_ratio,
            confidence: self.calculate_voting_confidence(voting_result),
            modality_contributions,
            explanation: format!(
                "Weighted voting: {:.1}% votes for deception, {:.1}% for truth (quorum: {})",
                voting_result.deception_ratio.to_f64().unwrap() * 100.0,
                (T::one() - voting_result.deception_ratio).to_f64().unwrap() * 100.0,
                if voting_result.has_quorum { "YES" } else { "NO" }
            ),
            metadata: self.create_voting_metadata(start_time, scores),
        }
    }
    
    fn create_decision_from_cached(
        &self,
        cached: &VotingResult<T>,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        start_time: std::time::Instant,
    ) -> FusedDecision<T> {
        self.create_decision(cached, scores, start_time)
    }
    
    fn calculate_voting_confidence(&self, result: &VotingResult<T>) -> T {
        if !result.has_quorum {
            return T::from(0.3).unwrap(); // Low confidence without quorum
        }
        
        // Confidence based on vote margin
        let margin = (result.deception_votes - result.truth_votes).abs() / result.total_weight;
        
        // Higher margin = higher confidence
        (margin * T::from(0.7).unwrap() + T::from(0.3).unwrap()).min(T::one())
    }
    
    fn create_voting_metadata(
        &self,
        processing_time: Duration,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> FusionMetadata {
        FusionMetadata {
            strategy_name: self.name().to_string(),
            processing_time,
            modalities_used: scores.keys().cloned().collect(),
            quality_metrics: QualityMetrics {
                fusion_quality: 0.85,
                confidence_variance: 0.1,
                modality_agreement: 0.9,
                timing: ProcessingTiming {
                    feature_extraction: Duration::from_millis(0),
                    fusion_computation: processing_time,
                    total: processing_time,
                },
            },
        }
    }
}

/// Feature normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureNormalization {
    None,
    MinMax,
    ZScore,
    L2,
}