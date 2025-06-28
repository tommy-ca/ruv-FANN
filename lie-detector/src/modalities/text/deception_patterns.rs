//! Deception pattern detection module for analyzing linguistic cues that may indicate deception

use crate::{Result, VeritasError, Feature};
use crate::types::*;
use crate::modalities::LinguisticFeatures;
use num_traits::Float;
use std::collections::HashMap;

/// Feature weights for different aspects of deception detection
#[derive(Debug, Clone)]
pub struct FeatureWeights<T: Float> {
    /// Weight for linguistic features
    pub linguistic_weight: T,
    /// Weight for temporal features
    pub temporal_weight: T,
    /// Weight for emotional features
    pub emotional_weight: T,
    /// Weight for syntactic features
    pub syntactic_weight: T,
    /// Weight for semantic features
    pub semantic_weight: T,
    /// Weight for pragmatic features
    pub pragmatic_weight: T,
}

impl<T: Float> Default for FeatureWeights<T> {
    fn default() -> Self {
        Self {
            linguistic_weight: T::from(1.0).unwrap(),
            temporal_weight: T::from(0.8).unwrap(),
            emotional_weight: T::from(0.9).unwrap(),
            syntactic_weight: T::from(0.7).unwrap(),
            semantic_weight: T::from(0.85).unwrap(),
            pragmatic_weight: T::from(0.75).unwrap(),
        }
    }
}

/// Detected deception patterns with scores
#[derive(Debug, Clone)]
pub struct DeceptionPatterns<T: Float> {
    /// Hedging frequency score
    pub hedging_frequency: T,
    /// Uncertainty markers score
    pub uncertainty_markers: T,
    /// Temporal references score
    pub temporal_references: T,
    /// Self-reference patterns score
    pub self_references: T,
    /// Detail level score
    pub detail_level: T,
    /// Emotional consistency score
    pub emotional_consistency: T,
    /// Certainty markers score
    pub certainty_markers: T,
    /// Overall pattern score
    pub overall_score: T,
    /// Individual pattern confidence scores
    pub pattern_confidence: HashMap<String, T>,
}

/// Deception pattern detector that analyzes linguistic features for deception indicators
pub struct DeceptionPatternDetector<T: Float> {
    weights: FeatureWeights<T>,
    thresholds: DeceptionThresholds<T>,
    pattern_weights: HashMap<String, T>,
    _phantom: std::marker::PhantomData<T>,
}

/// Thresholds for different deception indicators
#[derive(Debug, Clone)]
struct DeceptionThresholds<T: Float> {
    hedging_threshold: T,
    uncertainty_threshold: T,
    temporal_inconsistency_threshold: T,
    self_reference_threshold: T,
    detail_level_threshold: T,
    emotional_inconsistency_threshold: T,
}

impl<T: Float> Default for DeceptionThresholds<T> {
    fn default() -> Self {
        Self {
            hedging_threshold: T::from(0.1).unwrap(),
            uncertainty_threshold: T::from(0.15).unwrap(),
            temporal_inconsistency_threshold: T::from(0.2).unwrap(),
            self_reference_threshold: T::from(0.3).unwrap(),
            detail_level_threshold: T::from(0.1).unwrap(),
            emotional_inconsistency_threshold: T::from(0.25).unwrap(),
        }
    }
}

impl<T: Float> DeceptionPatternDetector<T> {
    /// Create a new deception pattern detector
    pub fn new(weights: &FeatureWeights<T>) -> Result<Self> {
        let pattern_weights = Self::initialize_pattern_weights()?;
        
        Ok(Self {
            weights: weights.clone(),
            thresholds: DeceptionThresholds::default(),
            pattern_weights,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Analyze patterns in linguistic features to detect deception indicators
    pub fn analyze_patterns(&self, features: &LinguisticFeatures<T>) -> Result<DeceptionPatterns<T>> {
        // Extract relevant features for deception analysis
        let hedging_frequency = self.calculate_hedging_frequency(features)?;
        let negation_patterns = self.calculate_negation_patterns(features)?;
        let temporal_references = self.calculate_temporal_inconsistency(features)?;
        let self_references = self.calculate_self_reference_patterns(features)?;
        let certainty_markers = self.calculate_certainty_markers(features)?;
        let detail_level = self.calculate_detail_level(features)?;
        let emotional_consistency = self.calculate_emotional_consistency(features)?;
        
        Ok(DeceptionPatterns {
            hedging_frequency,
            negation_patterns,
            temporal_references,
            self_references,
            certainty_markers,
            detail_level,
            emotional_consistency,
            response_latency: None, // Would need timing data
        })
    }
    
    /// Calculate overall deception probability based on patterns and features
    pub fn calculate_probability(&self, features: &LinguisticFeatures<T>, patterns: &DeceptionPatterns<T>) -> Result<T> {
        let mut score = T::zero();
        let mut total_weight = T::zero();
        
        // Weight different pattern indicators
        let indicators = [
            (patterns.hedging_frequency, self.weights.uncertainty_markers, "hedging"),
            (patterns.negation_patterns, self.weights.syntactic_patterns, "negation"),
            (patterns.temporal_references, self.weights.temporal_patterns, "temporal"),
            (patterns.self_references, self.weights.syntactic_patterns, "self_ref"),
            (patterns.certainty_markers, self.weights.uncertainty_markers, "certainty"),
            (patterns.detail_level, self.weights.linguistic_complexity, "detail"),
            (patterns.emotional_consistency, self.weights.emotional_indicators, "emotion"),
        ];
        
        for (value, weight, pattern_name) in indicators {
            let pattern_weight = self.pattern_weights.get(pattern_name).cloned().unwrap_or(T::one());
            let weighted_contribution = value * weight * pattern_weight;
            score = score + weighted_contribution;
            total_weight = total_weight + weight * pattern_weight;
        }
        
        // Normalize to probability range [0, 1]
        let raw_probability = if total_weight > T::zero() {
            score / total_weight
        } else {
            T::from(0.5).unwrap() // Neutral probability if no weights
        };
        
        // Apply sigmoid function to ensure [0,1] range
        let sigmoid_prob = self.sigmoid(raw_probability);
        
        // Apply additional scoring based on linguistic complexity
        let complexity_modifier = self.calculate_complexity_modifier(features)?;
        let final_prob = sigmoid_prob * complexity_modifier;
        
        Ok(final_prob.max(T::zero()).min(T::one()))
    }
    
    /// Calculate confidence in the deception prediction
    pub fn calculate_confidence(&self, features: &LinguisticFeatures<T>, patterns: &DeceptionPatterns<T>) -> Result<T> {
        // Confidence based on consistency of indicators
        let indicators = [
            patterns.hedging_frequency,
            patterns.negation_patterns,
            patterns.temporal_references,
            patterns.self_references,
            patterns.certainty_markers,
            patterns.detail_level,
            patterns.emotional_consistency,
        ];
        
        // Calculate variance of indicators (lower variance = higher confidence)
        let mean = indicators.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(indicators.len()).unwrap();
        let variance = indicators.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(indicators.len()).unwrap();
        
        // Higher variance = lower confidence
        let consistency_confidence = T::one() / (T::one() + variance);
        
        // Feature richness confidence (more features = higher confidence)
        let feature_count = T::from(features.feature_count()).unwrap();
        let richness_confidence = feature_count / (feature_count + T::from(10.0).unwrap());
        
        // Combine confidences
        let overall_confidence = (consistency_confidence + richness_confidence) / T::from(2.0).unwrap();
        
        Ok(overall_confidence.max(T::from(0.1).unwrap()).min(T::one()))
    }
    
    /// Get feature contributions for explainability
    pub fn get_feature_contributions(&self, features: &LinguisticFeatures<T>) -> Result<Vec<Feature<T>>> {
        let mut contributions = Vec::new();
        
        // Calculate contributions from different feature types
        self.add_lexical_contributions(&mut contributions, features)?;
        self.add_syntactic_contributions(&mut contributions, features)?;
        self.add_semantic_contributions(&mut contributions, features)?;
        self.add_pragmatic_contributions(&mut contributions, features)?;
        self.add_discourse_contributions(&mut contributions, features)?;
        
        // Sort by absolute contribution value
        contributions.sort_by(|a, b| {
            let a_abs = if a.value >= T::zero() { a.value } else { T::zero() - a.value };
            let b_abs = if b.value >= T::zero() { b.value } else { T::zero() - b.value };
            b_abs.partial_cmp(&a_abs).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(contributions)
    }
    
    /// Update pattern weights based on feedback
    pub fn update_weights(&mut self, feedback: &HashMap<String, T>) {
        for (pattern, weight) in feedback {
            if let Some(current_weight) = self.pattern_weights.get_mut(pattern) {
                *current_weight = *weight;
            }
        }
    }
    
    /// Get current pattern weights
    pub fn get_pattern_weights(&self) -> &HashMap<String, T> {
        &self.pattern_weights
    }
    
    // Private helper methods
    
    fn initialize_pattern_weights() -> Result<HashMap<String, T>> {
        let mut weights = HashMap::new();
        
        // Research-based weights for different deception indicators
        weights.insert("hedging".to_string(), T::from(0.8).unwrap());
        weights.insert("negation".to_string(), T::from(0.6).unwrap());
        weights.insert("temporal".to_string(), T::from(0.9).unwrap());
        weights.insert("self_ref".to_string(), T::from(0.7).unwrap());
        weights.insert("certainty".to_string(), T::from(0.75).unwrap());
        weights.insert("detail".to_string(), T::from(0.85).unwrap());
        weights.insert("emotion".to_string(), T::from(0.8).unwrap());
        
        Ok(weights)
    }
    
    fn calculate_hedging_frequency(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Hedging patterns from lexical features
        if features.lexical_features.len() >= 5 {
            // Index 4 should be uncertainty markers from linguistic analyzer
            let hedging_count = features.lexical_features[4];
            let total_words = features.lexical_features[0]; // word count
            
            if total_words > T::zero() {
                Ok(hedging_count / total_words)
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn calculate_negation_patterns(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Negation patterns from lexical features
        if features.lexical_features.len() >= 8 {
            // Index 7 should be negation patterns from linguistic analyzer
            let negation_count = features.lexical_features[7];
            let total_words = features.lexical_features[0];
            
            if total_words > T::zero() {
                Ok(negation_count / total_words)
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn calculate_temporal_inconsistency(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Temporal patterns from discourse features
        if !features.discourse_features.is_empty() {
            // Temporal markers from discourse features
            let temporal_markers = features.discourse_features[features.discourse_features.len() - 1];
            let sentence_count = if !features.syntactic_features.is_empty() {
                features.syntactic_features[0]
            } else {
                T::one()
            };
            
            // Higher ratio of temporal markers might indicate temporal inconsistency
            if sentence_count > T::zero() {
                let temporal_ratio = temporal_markers / sentence_count;
                // Inconsistency increases with too many or too few temporal markers
                let optimal_ratio = T::from(0.2).unwrap();
                let deviation = if temporal_ratio > optimal_ratio {
                    temporal_ratio - optimal_ratio
                } else {
                    optimal_ratio - temporal_ratio
                };
                Ok(deviation)
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn calculate_self_reference_patterns(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Self-reference patterns from lexical features
        if features.lexical_features.len() >= 9 {
            // Index 8 should be self references from linguistic analyzer
            let self_ref_count = features.lexical_features[8];
            let total_words = features.lexical_features[0];
            
            if total_words > T::zero() {
                let self_ref_ratio = self_ref_count / total_words;
                // Deceptive individuals often use fewer self-references
                let baseline_ratio = T::from(0.1).unwrap();
                if self_ref_ratio < baseline_ratio {
                    Ok(baseline_ratio - self_ref_ratio)
                } else {
                    Ok(T::zero())
                }
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn calculate_certainty_markers(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Certainty markers from lexical features
        if features.lexical_features.len() >= 7 {
            // Index 6 should be certainty markers from linguistic analyzer
            let certainty_count = features.lexical_features[6];
            let total_words = features.lexical_features[0];
            
            if total_words > T::zero() {
                let certainty_ratio = certainty_count / total_words;
                // Excessive certainty can indicate overcompensation
                let threshold = T::from(0.05).unwrap();
                if certainty_ratio > threshold {
                    Ok(certainty_ratio - threshold)
                } else {
                    Ok(T::zero())
                }
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn calculate_detail_level(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Detail level based on lexical diversity and complexity
        if features.lexical_features.len() >= 4 {
            let type_token_ratio = features.lexical_features[3]; // lexical diversity
            let avg_word_length = features.lexical_features[2];
            
            // Deceptive text often has lower detail/complexity
            let detail_score = (type_token_ratio + avg_word_length / T::from(10.0).unwrap()) / T::from(2.0).unwrap();
            let baseline_detail = T::from(0.5).unwrap();
            
            if detail_score < baseline_detail {
                Ok(baseline_detail - detail_score)
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn calculate_emotional_consistency(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Emotional consistency from semantic features
        if features.semantic_features.len() >= 3 {
            // Index 2 should be sentiment variance from linguistic analyzer
            let sentiment_variance = features.semantic_features[2];
            
            // Higher variance indicates emotional inconsistency
            let threshold = T::from(0.1).unwrap();
            if sentiment_variance > threshold {
                Ok(sentiment_variance - threshold)
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
    
    fn sigmoid(&self, x: T) -> T {
        // Sigmoid function: 1 / (1 + e^(-x))
        let one = T::one();
        let exp_neg_x = (-x).exp();
        one / (one + exp_neg_x)
    }
    
    fn calculate_complexity_modifier(&self, features: &LinguisticFeatures<T>) -> Result<T> {
        // Modifier based on linguistic complexity
        if features.lexical_features.len() >= 4 {
            let type_token_ratio = features.lexical_features[3];
            let avg_word_length = features.lexical_features[2];
            
            // Lower complexity might indicate deception
            let complexity_score = (type_token_ratio + avg_word_length / T::from(10.0).unwrap()) / T::from(2.0).unwrap();
            
            // Modifier ranges from 0.8 to 1.2
            let modifier = T::from(0.8).unwrap() + (T::one() - complexity_score) * T::from(0.4).unwrap();
            Ok(modifier.max(T::from(0.8).unwrap()).min(T::from(1.2).unwrap()))
        } else {
            Ok(T::one())
        }
    }
    
    fn add_lexical_contributions(&self, contributions: &mut Vec<Feature<T>>, features: &LinguisticFeatures<T>) -> Result<()> {
        let lexical_names = [
            "word_count", "char_count", "avg_word_length", "type_token_ratio",
            "uncertainty_markers", "hedging_patterns", "certainty_markers", 
            "negation_patterns", "self_references"
        ];
        
        for (i, &value) in features.lexical_features.iter().enumerate() {
            if i < lexical_names.len() {
                let weight = match lexical_names[i] {
                    "uncertainty_markers" => self.weights.uncertainty_markers,
                    "hedging_patterns" => self.weights.uncertainty_markers,
                    "certainty_markers" => self.weights.uncertainty_markers,
                    "negation_patterns" => self.weights.syntactic_patterns,
                    "self_references" => self.weights.syntactic_patterns,
                    _ => T::from(0.5).unwrap(),
                };
                
                contributions.push(Feature {
                    name: lexical_names[i].to_string(),
                    value,
                    weight,
                    description: format!("Lexical feature: {}", lexical_names[i]),
                });
            }
        }
        
        Ok(())
    }
    
    fn add_syntactic_contributions(&self, contributions: &mut Vec<Feature<T>>, features: &LinguisticFeatures<T>) -> Result<()> {
        let syntactic_names = [
            "sentence_count", "avg_sentence_length", "noun_ratio", 
            "verb_ratio", "adjective_ratio", "adverb_ratio", "pronoun_ratio"
        ];
        
        for (i, &value) in features.syntactic_features.iter().enumerate() {
            if i < syntactic_names.len() {
                contributions.push(Feature {
                    name: syntactic_names[i].to_string(),
                    value,
                    weight: self.weights.syntactic_patterns,
                    description: format!("Syntactic feature: {}", syntactic_names[i]),
                });
            }
        }
        
        Ok(())
    }
    
    fn add_semantic_contributions(&self, contributions: &mut Vec<Feature<T>>, features: &LinguisticFeatures<T>) -> Result<()> {
        let semantic_names = ["semantic_density", "avg_sentiment", "sentiment_variance"];
        
        for (i, &value) in features.semantic_features.iter().enumerate() {
            if i < semantic_names.len() {
                contributions.push(Feature {
                    name: semantic_names[i].to_string(),
                    value,
                    weight: self.weights.semantic_coherence,
                    description: format!("Semantic feature: {}", semantic_names[i]),
                });
            }
        }
        
        Ok(())
    }
    
    fn add_pragmatic_contributions(&self, contributions: &mut Vec<Feature<T>>, features: &LinguisticFeatures<T>) -> Result<()> {
        let pragmatic_names = ["uncertainty_ratio", "question_count", "exclamation_count"];
        
        for (i, &value) in features.pragmatic_features.iter().enumerate() {
            if i < pragmatic_names.len() {
                contributions.push(Feature {
                    name: pragmatic_names[i].to_string(),
                    value,
                    weight: self.weights.uncertainty_markers,
                    description: format!("Pragmatic feature: {}", pragmatic_names[i]),
                });
            }
        }
        
        Ok(())
    }
    
    fn add_discourse_contributions(&self, contributions: &mut Vec<Feature<T>>, features: &LinguisticFeatures<T>) -> Result<()> {
        let discourse_names = ["connective_count", "temporal_markers"];
        
        for (i, &value) in features.discourse_features.iter().enumerate() {
            if i < discourse_names.len() {
                contributions.push(Feature {
                    name: discourse_names[i].to_string(),
                    value,
                    weight: self.weights.temporal_patterns,
                    description: format!("Discourse feature: {}", discourse_names[i]),
                });
            }
        }
        
        Ok(())
    }
}

/// Advanced deception detection algorithms
impl<T: Float> DeceptionPatternDetector<T> {
    /// Detect specific deception strategies
    pub fn detect_deception_strategies(&self, patterns: &DeceptionPatterns<T>) -> Vec<DeceptionStrategy<T>> {
        let mut strategies = Vec::new();
        
        // Hedging strategy
        if patterns.hedging_frequency > self.thresholds.hedging_threshold {
            strategies.push(DeceptionStrategy {
                strategy_type: "hedging".to_string(),
                confidence: patterns.hedging_frequency,
                description: "Excessive use of hedging language to avoid commitment".to_string(),
                indicators: vec!["maybe", "perhaps", "kind of", "sort of"].iter().map(|s| s.to_string()).collect(),
            });
        }
        
        // Temporal evasion
        if patterns.temporal_references > self.thresholds.temporal_inconsistency_threshold {
            strategies.push(DeceptionStrategy {
                strategy_type: "temporal_evasion".to_string(),
                confidence: patterns.temporal_references,
                description: "Inconsistent or evasive temporal references".to_string(),
                indicators: vec!["vague time references", "timeline inconsistencies"].iter().map(|s| s.to_string()).collect(),
            });
        }
        
        // Emotional detachment
        if patterns.self_references < self.thresholds.self_reference_threshold {
            strategies.push(DeceptionStrategy {
                strategy_type: "emotional_detachment".to_string(),
                confidence: self.thresholds.self_reference_threshold - patterns.self_references,
                description: "Reduced self-referencing indicating emotional detachment".to_string(),
                indicators: vec!["fewer personal pronouns", "distant language"].iter().map(|s| s.to_string()).collect(),
            });
        }
        
        // Overcompensation
        if patterns.certainty_markers > self.thresholds.uncertainty_threshold {
            strategies.push(DeceptionStrategy {
                strategy_type: "overcompensation".to_string(),
                confidence: patterns.certainty_markers,
                description: "Excessive certainty markers indicating overcompensation".to_string(),
                indicators: vec!["definitely", "absolutely", "certainly"].iter().map(|s| s.to_string()).collect(),
            });
        }
        
        strategies
    }
    
    /// Calculate deception risk score
    pub fn calculate_risk_score(&self, patterns: &DeceptionPatterns<T>) -> T {
        let risk_factors = [
            (patterns.hedging_frequency, T::from(0.9).unwrap()),
            (patterns.temporal_references, T::from(1.0).unwrap()),
            (patterns.emotional_consistency, T::from(0.8).unwrap()),
            (patterns.detail_level, T::from(0.7).unwrap()),
        ];
        
        let mut total_risk = T::zero();
        let mut total_weight = T::zero();
        
        for (factor, weight) in risk_factors {
            total_risk = total_risk + factor * weight;
            total_weight = total_weight + weight;
        }
        
        if total_weight > T::zero() {
            total_risk / total_weight
        } else {
            T::from(0.5).unwrap()
        }
    }
}

/// Deception strategy detection result
#[derive(Debug, Clone)]
pub struct DeceptionStrategy<T: Float> {
    pub strategy_type: String,
    pub confidence: T,
    pub description: String,
    pub indicators: Vec<String>,
}

/// Statistical analysis of deception patterns
impl<T: Float> DeceptionPatternDetector<T> {
    /// Perform statistical significance testing on patterns
    pub fn test_statistical_significance(&self, patterns: &[DeceptionPatterns<T>]) -> StatisticalResult<T> {
        if patterns.len() < 2 {
            return StatisticalResult {
                is_significant: false,
                p_value: T::one(),
                effect_size: T::zero(),
                confidence_interval: (T::zero(), T::zero()),
            };
        }
        
        // Simplified statistical test - in production use proper statistical libraries
        let means: Vec<T> = vec![
            patterns.iter().map(|p| p.hedging_frequency).fold(T::zero(), |acc, x| acc + x) / T::from(patterns.len()).unwrap(),
            patterns.iter().map(|p| p.temporal_references).fold(T::zero(), |acc, x| acc + x) / T::from(patterns.len()).unwrap(),
            patterns.iter().map(|p| p.emotional_consistency).fold(T::zero(), |acc, x| acc + x) / T::from(patterns.len()).unwrap(),
        ];
        
        let overall_mean = means.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(means.len()).unwrap();
        let variance = means.iter()
            .map(|&x| (x - overall_mean) * (x - overall_mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(means.len()).unwrap();
        
        let effect_size = variance.sqrt();
        let p_value = T::from(0.05).unwrap(); // Placeholder
        let is_significant = p_value < T::from(0.05).unwrap();
        
        StatisticalResult {
            is_significant,
            p_value,
            effect_size,
            confidence_interval: (overall_mean - effect_size, overall_mean + effect_size),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatisticalResult<T: Float> {
    pub is_significant: bool,
    pub p_value: T,
    pub effect_size: T,
    pub confidence_interval: (T, T),
}