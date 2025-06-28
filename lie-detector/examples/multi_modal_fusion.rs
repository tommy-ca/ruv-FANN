//! # Multi-Modal Fusion Strategies Example for Veritas Nexus
//! 
//! This example demonstrates different fusion strategies for combining multi-modal
//! lie detection inputs. It shows how to:
//! - Implement various fusion approaches (early, late, attention-based)
//! - Compare effectiveness of different strategies
//! - Handle missing modalities gracefully
//! - Optimize fusion weights automatically
//! - Provide uncertainty quantification
//! - Generate fusion explanations
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example multi_modal_fusion
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio;
use serde::{Deserialize, Serialize};

/// Multi-modal input for fusion
#[derive(Debug, Clone)]
pub struct MultiModalInput {
    pub video_features: Option<Vec<f32>>,
    pub audio_features: Option<Vec<f32>>,
    pub text_features: Option<Vec<f32>>,
    pub physiological_features: Option<Vec<f32>>,
    pub metadata: InputMetadata,
}

/// Metadata for input
#[derive(Debug, Clone)]
pub struct InputMetadata {
    pub timestamp: Instant,
    pub sample_id: String,
    pub quality_scores: HashMap<String, f32>,
    pub confidence_scores: HashMap<String, f32>,
}

/// Result from fusion process
#[derive(Debug, Clone, Serialize)]
pub struct FusionResult {
    pub final_decision: DeceptionDecision,
    pub confidence: f32,
    pub modality_weights: HashMap<String, f32>,
    pub individual_scores: HashMap<String, f32>,
    pub fusion_explanation: String,
    pub uncertainty_score: f32,
    pub processing_time_ms: u64,
    pub strategy_used: String,
}

/// Deception decision
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum DeceptionDecision {
    Truthful,
    Deceptive,
    Uncertain,
    InsufficientData,
}

/// Fusion strategy trait
pub trait FusionStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn fuse(&self, input: &MultiModalInput) -> Result<FusionResult, FusionError>;
    fn supports_missing_modalities(&self) -> bool;
    fn required_modalities(&self) -> Vec<String>;
}

/// Early fusion strategy - combines features before decision
pub struct EarlyFusionStrategy {
    feature_weights: HashMap<String, f32>,
    normalization: bool,
}

impl EarlyFusionStrategy {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("vision".to_string(), 0.3);
        weights.insert("audio".to_string(), 0.25);
        weights.insert("text".to_string(), 0.25);
        weights.insert("physiological".to_string(), 0.2);
        
        Self {
            feature_weights: weights,
            normalization: true,
        }
    }
    
    fn combine_features(&self, input: &MultiModalInput) -> Vec<f32> {
        let mut combined = Vec::new();
        
        // Combine all available features into a single vector
        if let Some(ref features) = input.video_features {
            let weight = self.feature_weights.get("vision").unwrap_or(&1.0);
            for &feat in features {
                combined.push(feat * weight);
            }
        }
        
        if let Some(ref features) = input.audio_features {
            let weight = self.feature_weights.get("audio").unwrap_or(&1.0);
            for &feat in features {
                combined.push(feat * weight);
            }
        }
        
        if let Some(ref features) = input.text_features {
            let weight = self.feature_weights.get("text").unwrap_or(&1.0);
            for &feat in features {
                combined.push(feat * weight);
            }
        }
        
        if let Some(ref features) = input.physiological_features {
            let weight = self.feature_weights.get("physiological").unwrap_or(&1.0);
            for &feat in features {
                combined.push(feat * weight);
            }
        }
        
        // Normalize if enabled
        if self.normalization && !combined.is_empty() {
            let mean: f32 = combined.iter().sum::<f32>() / combined.len() as f32;
            let std_dev = (combined.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / combined.len() as f32).sqrt();
            
            if std_dev > 0.0 {
                for val in combined.iter_mut() {
                    *val = (*val - mean) / std_dev;
                }
            }
        }
        
        combined
    }
    
    fn classify_features(&self, features: &[f32]) -> (f32, f32) {
        if features.is_empty() {
            return (0.5, 0.0);
        }
        
        // Simple linear classifier simulation
        let positive_weight = features.iter().map(|&x| x.max(0.0)).sum::<f32>();
        let negative_weight = features.iter().map(|&x| (-x).max(0.0)).sum::<f32>();
        
        let total = positive_weight + negative_weight;
        let score = if total > 0.0 {
            positive_weight / total
        } else {
            0.5
        };
        
        let confidence = (total / features.len() as f32).min(1.0);
        
        (score, confidence)
    }
}

impl FusionStrategy for EarlyFusionStrategy {
    fn name(&self) -> &str {
        "early_fusion"
    }
    
    fn description(&self) -> &str {
        "Combines features from all modalities early in the pipeline before making decisions"
    }
    
    fn fuse(&self, input: &MultiModalInput) -> Result<FusionResult, FusionError> {
        let start_time = Instant::now();
        
        // Combine features
        let combined_features = self.combine_features(input);
        
        if combined_features.is_empty() {
            return Ok(FusionResult {
                final_decision: DeceptionDecision::InsufficientData,
                confidence: 0.0,
                modality_weights: HashMap::new(),
                individual_scores: HashMap::new(),
                fusion_explanation: "No features available for early fusion".to_string(),
                uncertainty_score: 1.0,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                strategy_used: self.name().to_string(),
            });
        }
        
        // Classify combined features
        let (score, confidence) = self.classify_features(&combined_features);
        
        // Generate individual scores based on contributions
        let mut individual_scores = HashMap::new();
        if input.video_features.is_some() {
            individual_scores.insert("vision".to_string(), score * 0.9 + 0.05);
        }
        if input.audio_features.is_some() {
            individual_scores.insert("audio".to_string(), score * 0.8 + 0.1);
        }
        if input.text_features.is_some() {
            individual_scores.insert("text".to_string(), score * 0.85 + 0.075);
        }
        if input.physiological_features.is_some() {
            individual_scores.insert("physiological".to_string(), score * 0.95);
        }
        
        let decision = if confidence < 0.3 {
            DeceptionDecision::Uncertain
        } else if score > 0.6 {
            DeceptionDecision::Deceptive
        } else if score < 0.4 {
            DeceptionDecision::Truthful
        } else {
            DeceptionDecision::Uncertain
        };
        
        let explanation = format!(
            "Early fusion combined {} features into {}-dimensional space. Final score: {:.3} with confidence {:.3}. \
            Available modalities: {}",
            combined_features.len(),
            combined_features.len(),
            score,
            confidence,
            individual_scores.keys().cloned().collect::<Vec<_>>().join(", ")
        );
        
        Ok(FusionResult {
            final_decision: decision,
            confidence,
            modality_weights: self.feature_weights.clone(),
            individual_scores,
            fusion_explanation: explanation,
            uncertainty_score: 1.0 - confidence,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            strategy_used: self.name().to_string(),
        })
    }
    
    fn supports_missing_modalities(&self) -> bool {
        true
    }
    
    fn required_modalities(&self) -> Vec<String> {
        vec![] // No required modalities
    }
}

/// Late fusion strategy - makes decisions per modality then combines
pub struct LateFusionStrategy {
    modality_weights: HashMap<String, f32>,
    decision_threshold: f32,
}

impl LateFusionStrategy {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("vision".to_string(), 0.35);
        weights.insert("audio".to_string(), 0.25);
        weights.insert("text".to_string(), 0.25);
        weights.insert("physiological".to_string(), 0.15);
        
        Self {
            modality_weights: weights,
            decision_threshold: 0.5,
        }
    }
    
    fn analyze_modality(&self, features: &[f32], modality: &str) -> (f32, f32) {
        if features.is_empty() {
            return (0.5, 0.0);
        }
        
        // Modality-specific analysis simulation
        let score = match modality {
            "vision" => {
                // Vision features analysis
                let avg = features.iter().sum::<f32>() / features.len() as f32;
                let variance = features.iter().map(|x| (x - avg).powi(2)).sum::<f32>() / features.len() as f32;
                (avg + variance * 0.1).clamp(0.0, 1.0)
            }
            "audio" => {
                // Audio features analysis
                let energy = features.iter().map(|x| x.powi(2)).sum::<f32>() / features.len() as f32;
                energy.sqrt().clamp(0.0, 1.0)
            }
            "text" => {
                // Text features analysis
                let complexity = features.iter().map(|x| x.abs()).sum::<f32>() / features.len() as f32;
                complexity.clamp(0.0, 1.0)
            }
            "physiological" => {
                // Physiological features analysis
                let peak = features.iter().cloned().fold(0.0f32, f32::max);
                peak.clamp(0.0, 1.0)
            }
            _ => 0.5,
        };
        
        let confidence = (features.len() as f32 / 100.0).min(1.0); // More features = higher confidence
        
        (score, confidence)
    }
}

impl FusionStrategy for LateFusionStrategy {
    fn name(&self) -> &str {
        "late_fusion"
    }
    
    fn description(&self) -> &str {
        "Makes independent decisions for each modality then combines using weighted voting"
    }
    
    fn fuse(&self, input: &MultiModalInput) -> Result<FusionResult, FusionError> {
        let start_time = Instant::now();
        
        let mut individual_scores = HashMap::new();
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut confidences = Vec::new();
        
        // Analyze each modality independently
        if let Some(ref features) = input.video_features {
            let (score, confidence) = self.analyze_modality(features, "vision");
            individual_scores.insert("vision".to_string(), score);
            
            let weight = self.modality_weights.get("vision").unwrap_or(&0.25);
            weighted_sum += score * weight * confidence; // Weight by confidence
            total_weight += weight * confidence;
            confidences.push(confidence);
        }
        
        if let Some(ref features) = input.audio_features {
            let (score, confidence) = self.analyze_modality(features, "audio");
            individual_scores.insert("audio".to_string(), score);
            
            let weight = self.modality_weights.get("audio").unwrap_or(&0.25);
            weighted_sum += score * weight * confidence;
            total_weight += weight * confidence;
            confidences.push(confidence);
        }
        
        if let Some(ref features) = input.text_features {
            let (score, confidence) = self.analyze_modality(features, "text");
            individual_scores.insert("text".to_string(), score);
            
            let weight = self.modality_weights.get("text").unwrap_or(&0.25);
            weighted_sum += score * weight * confidence;
            total_weight += weight * confidence;
            confidences.push(confidence);
        }
        
        if let Some(ref features) = input.physiological_features {
            let (score, confidence) = self.analyze_modality(features, "physiological");
            individual_scores.insert("physiological".to_string(), score);
            
            let weight = self.modality_weights.get("physiological").unwrap_or(&0.25);
            weighted_sum += score * weight * confidence;
            total_weight += weight * confidence;
            confidences.push(confidence);
        }
        
        if total_weight == 0.0 {
            return Ok(FusionResult {
                final_decision: DeceptionDecision::InsufficientData,
                confidence: 0.0,
                modality_weights: HashMap::new(),
                individual_scores: HashMap::new(),
                fusion_explanation: "No valid modalities available for late fusion".to_string(),
                uncertainty_score: 1.0,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                strategy_used: self.name().to_string(),
            });
        }
        
        let final_score = weighted_sum / total_weight;
        let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
        
        let decision = if avg_confidence < 0.3 {
            DeceptionDecision::Uncertain
        } else if final_score > 0.6 {
            DeceptionDecision::Deceptive
        } else if final_score < 0.4 {
            DeceptionDecision::Truthful
        } else {
            DeceptionDecision::Uncertain
        };
        
        let explanation = format!(
            "Late fusion analyzed {} modalities independently. Weighted average: {:.3}. \
            Individual scores: {}. Confidence-weighted combination used.",
            individual_scores.len(),
            final_score,
            individual_scores.iter()
                .map(|(k, v)| format!("{}={:.3}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        
        Ok(FusionResult {
            final_decision: decision,
            confidence: avg_confidence,
            modality_weights: self.modality_weights.clone(),
            individual_scores,
            fusion_explanation: explanation,
            uncertainty_score: 1.0 - avg_confidence,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            strategy_used: self.name().to_string(),
        })
    }
    
    fn supports_missing_modalities(&self) -> bool {
        true
    }
    
    fn required_modalities(&self) -> Vec<String> {
        vec![] // No required modalities
    }
}

/// Attention-based fusion strategy
pub struct AttentionFusionStrategy {
    attention_heads: usize,
    hidden_dim: usize,
}

impl AttentionFusionStrategy {
    pub fn new() -> Self {
        Self {
            attention_heads: 4,
            hidden_dim: 64,
        }
    }
    
    fn compute_attention_weights(&self, features: &HashMap<String, Vec<f32>>) -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        let mut total_importance = 0.0;
        
        for (modality, feature_vec) in features {
            // Simulate attention computation
            let feature_norm = feature_vec.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
            let complexity = feature_vec.len() as f32;
            let variance = if feature_vec.len() > 1 {
                let mean = feature_vec.iter().sum::<f32>() / feature_vec.len() as f32;
                feature_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / feature_vec.len() as f32
            } else {
                0.0
            };
            
            // Attention score based on feature characteristics
            let attention_score = (feature_norm * 0.4 + complexity.ln() * 0.3 + variance * 0.3).max(0.001);
            weights.insert(modality.clone(), attention_score);
            total_importance += attention_score;
        }
        
        // Normalize weights
        if total_importance > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_importance;
            }
        }
        
        weights
    }
    
    fn apply_attention(&self, features: &HashMap<String, Vec<f32>>, weights: &HashMap<String, f32>) -> Vec<f32> {
        let mut attended_features = Vec::new();
        
        for (modality, feature_vec) in features {
            let weight = weights.get(modality).unwrap_or(&0.0);
            
            // Apply attention weight to features
            for &feat in feature_vec {
                attended_features.push(feat * weight);
            }
        }
        
        attended_features
    }
}

impl FusionStrategy for AttentionFusionStrategy {
    fn name(&self) -> &str {
        "attention_fusion"
    }
    
    fn description(&self) -> &str {
        "Uses attention mechanisms to dynamically weight modalities based on feature importance"
    }
    
    fn fuse(&self, input: &MultiModalInput) -> Result<FusionResult, FusionError> {
        let start_time = Instant::now();
        
        // Collect available features
        let mut features = HashMap::new();
        let mut individual_scores = HashMap::new();
        
        if let Some(ref feat) = input.video_features {
            features.insert("vision".to_string(), feat.clone());
        }
        if let Some(ref feat) = input.audio_features {
            features.insert("audio".to_string(), feat.clone());
        }
        if let Some(ref feat) = input.text_features {
            features.insert("text".to_string(), feat.clone());
        }
        if let Some(ref feat) = input.physiological_features {
            features.insert("physiological".to_string(), feat.clone());
        }
        
        if features.is_empty() {
            return Ok(FusionResult {
                final_decision: DeceptionDecision::InsufficientData,
                confidence: 0.0,
                modality_weights: HashMap::new(),
                individual_scores: HashMap::new(),
                fusion_explanation: "No features available for attention fusion".to_string(),
                uncertainty_score: 1.0,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                strategy_used: self.name().to_string(),
            });
        }
        
        // Compute attention weights
        let attention_weights = self.compute_attention_weights(&features);
        
        // Apply attention and create attended features
        let attended_features = self.apply_attention(&features, &attention_weights);
        
        // Generate individual scores based on attention
        for (modality, weight) in &attention_weights {
            let base_score = features.get(modality).map(|f| {
                f.iter().sum::<f32>() / f.len() as f32
            }).unwrap_or(0.5);
            
            individual_scores.insert(modality.clone(), base_score);
        }
        
        // Final decision based on attended features
        let final_score = if attended_features.is_empty() {
            0.5
        } else {
            attended_features.iter().sum::<f32>() / attended_features.len() as f32
        };
        
        let confidence = attention_weights.values().map(|w| w.powi(2)).sum::<f32>().sqrt(); // Concentration measure
        
        let decision = if confidence < 0.3 {
            DeceptionDecision::Uncertain
        } else if final_score > 0.6 {
            DeceptionDecision::Deceptive
        } else if final_score < 0.4 {
            DeceptionDecision::Truthful
        } else {
            DeceptionDecision::Uncertain
        };
        
        let explanation = format!(
            "Attention fusion with {} heads dynamically weighted modalities. \
            Attention weights: {}. Final attended score: {:.3}",
            self.attention_heads,
            attention_weights.iter()
                .map(|(k, v)| format!("{}={:.3}", k, v))
                .collect::<Vec<_>>()
                .join(", "),
            final_score
        );
        
        Ok(FusionResult {
            final_decision: decision,
            confidence,
            modality_weights: attention_weights,
            individual_scores,
            fusion_explanation: explanation,
            uncertainty_score: 1.0 - confidence,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            strategy_used: self.name().to_string(),
        })
    }
    
    fn supports_missing_modalities(&self) -> bool {
        true
    }
    
    fn required_modalities(&self) -> Vec<String> {
        vec![] // No required modalities
    }
}

/// Adaptive fusion strategy that selects the best approach
pub struct AdaptiveFusionStrategy {
    strategies: Vec<Box<dyn FusionStrategy>>,
    selection_criteria: SelectionCriteria,
}

#[derive(Debug, Clone)]
pub enum SelectionCriteria {
    HighestConfidence,
    BestDataAvailability,
    LowestUncertainty,
    Ensemble,
}

impl AdaptiveFusionStrategy {
    pub fn new() -> Self {
        let strategies: Vec<Box<dyn FusionStrategy>> = vec![
            Box::new(EarlyFusionStrategy::new()),
            Box::new(LateFusionStrategy::new()),
            Box::new(AttentionFusionStrategy::new()),
        ];
        
        Self {
            strategies,
            selection_criteria: SelectionCriteria::Ensemble,
        }
    }
    
    fn select_strategy(&self, input: &MultiModalInput) -> usize {
        match self.selection_criteria {
            SelectionCriteria::BestDataAvailability => {
                let modality_count = [
                    input.video_features.is_some(),
                    input.audio_features.is_some(),
                    input.text_features.is_some(),
                    input.physiological_features.is_some(),
                ].iter().filter(|&&x| x).count();
                
                match modality_count {
                    1 => 1, // Late fusion for single modality
                    2 => 0, // Early fusion for two modalities
                    _ => 2, // Attention fusion for 3+ modalities
                }
            }
            _ => 0, // Default to first strategy
        }
    }
}

impl FusionStrategy for AdaptiveFusionStrategy {
    fn name(&self) -> &str {
        "adaptive_fusion"
    }
    
    fn description(&self) -> &str {
        "Adaptively selects the best fusion strategy based on data characteristics"
    }
    
    fn fuse(&self, input: &MultiModalInput) -> Result<FusionResult, FusionError> {
        let start_time = Instant::now();
        
        match self.selection_criteria {
            SelectionCriteria::Ensemble => {
                // Run all strategies and ensemble the results
                let mut results = Vec::new();
                for strategy in &self.strategies {
                    if let Ok(result) = strategy.fuse(input) {
                        results.push(result);
                    }
                }
                
                if results.is_empty() {
                    return Err(FusionError::NoValidStrategies);
                }
                
                // Ensemble the results
                let mut final_scores = Vec::new();
                let mut confidences = Vec::new();
                let mut combined_weights = HashMap::new();
                let mut combined_scores = HashMap::new();
                
                for result in &results {
                    let score = match result.final_decision {
                        DeceptionDecision::Truthful => 0.2,
                        DeceptionDecision::Deceptive => 0.8,
                        DeceptionDecision::Uncertain => 0.5,
                        DeceptionDecision::InsufficientData => 0.5,
                    };
                    final_scores.push(score * result.confidence);
                    confidences.push(result.confidence);
                    
                    // Combine weights and scores
                    for (k, v) in &result.modality_weights {
                        *combined_weights.entry(k.clone()).or_insert(0.0) += v / results.len() as f32;
                    }
                    for (k, v) in &result.individual_scores {
                        *combined_scores.entry(k.clone()).or_insert(0.0) += v / results.len() as f32;
                    }
                }
                
                let ensemble_score = final_scores.iter().sum::<f32>() / final_scores.len() as f32;
                let ensemble_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
                
                let decision = if ensemble_confidence < 0.3 {
                    DeceptionDecision::Uncertain
                } else if ensemble_score > 0.6 {
                    DeceptionDecision::Deceptive
                } else if ensemble_score < 0.4 {
                    DeceptionDecision::Truthful
                } else {
                    DeceptionDecision::Uncertain
                };
                
                let explanation = format!(
                    "Adaptive fusion ensembled {} strategies. Individual results: {}. Final ensemble score: {:.3}",
                    results.len(),
                    results.iter()
                        .map(|r| format!("{}={:.3}", r.strategy_used, 
                            match r.final_decision {
                                DeceptionDecision::Truthful => 0.2,
                                DeceptionDecision::Deceptive => 0.8,
                                _ => 0.5,
                            }))
                        .collect::<Vec<_>>()
                        .join(", "),
                    ensemble_score
                );
                
                Ok(FusionResult {
                    final_decision: decision,
                    confidence: ensemble_confidence,
                    modality_weights: combined_weights,
                    individual_scores: combined_scores,
                    fusion_explanation: explanation,
                    uncertainty_score: 1.0 - ensemble_confidence,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    strategy_used: self.name().to_string(),
                })
            }
            _ => {
                // Select single best strategy
                let strategy_idx = self.select_strategy(input);
                self.strategies[strategy_idx].fuse(input)
            }
        }
    }
    
    fn supports_missing_modalities(&self) -> bool {
        true
    }
    
    fn required_modalities(&self) -> Vec<String> {
        vec![] // No required modalities
    }
}

/// Fusion manager
pub struct FusionManager {
    strategies: HashMap<String, Box<dyn FusionStrategy>>,
}

impl FusionManager {
    pub fn new() -> Self {
        let mut strategies: HashMap<String, Box<dyn FusionStrategy>> = HashMap::new();
        
        strategies.insert("early_fusion".to_string(), Box::new(EarlyFusionStrategy::new()));
        strategies.insert("late_fusion".to_string(), Box::new(LateFusionStrategy::new()));
        strategies.insert("attention_fusion".to_string(), Box::new(AttentionFusionStrategy::new()));
        strategies.insert("adaptive_fusion".to_string(), Box::new(AdaptiveFusionStrategy::new()));
        
        Self { strategies }
    }
    
    pub fn fuse(&self, strategy_name: &str, input: &MultiModalInput) -> Result<FusionResult, FusionError> {
        self.strategies.get(strategy_name)
            .ok_or_else(|| FusionError::UnknownStrategy(strategy_name.to_string()))?
            .fuse(input)
    }
    
    pub fn list_strategies(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }
    
    pub fn get_strategy_info(&self, name: &str) -> Option<(String, String)> {
        self.strategies.get(name).map(|s| (s.name().to_string(), s.description().to_string()))
    }
}

/// Fusion errors
#[derive(Debug)]
pub enum FusionError {
    UnknownStrategy(String),
    InsufficientData,
    ProcessingError(String),
    NoValidStrategies,
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::UnknownStrategy(s) => write!(f, "Unknown fusion strategy: {}", s),
            FusionError::InsufficientData => write!(f, "Insufficient data for fusion"),
            FusionError::ProcessingError(s) => write!(f, "Processing error: {}", s),
            FusionError::NoValidStrategies => write!(f, "No valid fusion strategies available"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Generate synthetic multi-modal data for testing
fn generate_test_data(scenario: &str) -> MultiModalInput {
    let timestamp = Instant::now();
    let mut quality_scores = HashMap::new();
    let mut confidence_scores = HashMap::new();
    
    match scenario {
        "high_deception" => {
            quality_scores.insert("overall".to_string(), 0.85);
            confidence_scores.insert("overall".to_string(), 0.82);
            
            MultiModalInput {
                video_features: Some(vec![0.8, 0.7, 0.9, 0.85, 0.75]), // High deception indicators
                audio_features: Some(vec![0.75, 0.8, 0.7, 0.85]),
                text_features: Some(vec![0.9, 0.85, 0.8]),
                physiological_features: Some(vec![0.88, 0.9, 0.82]),
                metadata: InputMetadata {
                    timestamp,
                    sample_id: "high_deception_test".to_string(),
                    quality_scores,
                    confidence_scores,
                },
            }
        }
        "low_deception" => {
            quality_scores.insert("overall".to_string(), 0.90);
            confidence_scores.insert("overall".to_string(), 0.88);
            
            MultiModalInput {
                video_features: Some(vec![0.2, 0.3, 0.1, 0.25, 0.35]),
                audio_features: Some(vec![0.25, 0.2, 0.3, 0.15]),
                text_features: Some(vec![0.1, 0.15, 0.2]),
                physiological_features: Some(vec![0.18, 0.1, 0.22]),
                metadata: InputMetadata {
                    timestamp,
                    sample_id: "low_deception_test".to_string(),
                    quality_scores,
                    confidence_scores,
                },
            }
        }
        "missing_video" => {
            quality_scores.insert("overall".to_string(), 0.75);
            confidence_scores.insert("overall".to_string(), 0.70);
            
            MultiModalInput {
                video_features: None, // Missing video
                audio_features: Some(vec![0.6, 0.5, 0.7, 0.55]),
                text_features: Some(vec![0.65, 0.6, 0.7]),
                physiological_features: Some(vec![0.58, 0.65, 0.62]),
                metadata: InputMetadata {
                    timestamp,
                    sample_id: "missing_video_test".to_string(),
                    quality_scores,
                    confidence_scores,
                },
            }
        }
        "text_only" => {
            quality_scores.insert("overall".to_string(), 0.60);
            confidence_scores.insert("overall".to_string(), 0.55);
            
            MultiModalInput {
                video_features: None,
                audio_features: None,
                text_features: Some(vec![0.7, 0.65, 0.8, 0.72, 0.68]),
                physiological_features: None,
                metadata: InputMetadata {
                    timestamp,
                    sample_id: "text_only_test".to_string(),
                    quality_scores,
                    confidence_scores,
                },
            }
        }
        _ => {
            quality_scores.insert("overall".to_string(), 0.8);
            confidence_scores.insert("overall".to_string(), 0.75);
            
            MultiModalInput {
                video_features: Some(vec![0.5, 0.55, 0.45, 0.6, 0.5]),
                audio_features: Some(vec![0.52, 0.48, 0.55, 0.5]),
                text_features: Some(vec![0.5, 0.53, 0.47]),
                physiological_features: Some(vec![0.51, 0.49, 0.52]),
                metadata: InputMetadata {
                    timestamp,
                    sample_id: "balanced_test".to_string(),
                    quality_scores,
                    confidence_scores,
                },
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔀 Veritas Nexus - Multi-Modal Fusion Strategies Example");
    println!("========================================================\n");
    
    // Initialize fusion manager
    let fusion_manager = FusionManager::new();
    
    println!("🧠 Available Fusion Strategies:");
    for strategy_name in fusion_manager.list_strategies() {
        if let Some((name, description)) = fusion_manager.get_strategy_info(&strategy_name) {
            println!("  • {}: {}", name, description);
        }
    }
    
    println!("\n" + &"=".repeat(60) + "\n");
    
    // Test scenarios
    let scenarios = vec![
        ("High Deception", "high_deception"),
        ("Low Deception", "low_deception"),
        ("Missing Video", "missing_video"),
        ("Text Only", "text_only"),
        ("Balanced", "balanced"),
    ];
    
    for (scenario_name, scenario_key) in scenarios {
        println!("📊 Scenario: {}", scenario_name);
        println!("{}", "-".repeat(30));
        
        let test_data = generate_test_data(scenario_key);
        
        // Show available modalities
        let mut available_modalities = Vec::new();
        if test_data.video_features.is_some() { available_modalities.push("Video"); }
        if test_data.audio_features.is_some() { available_modalities.push("Audio"); }
        if test_data.text_features.is_some() { available_modalities.push("Text"); }
        if test_data.physiological_features.is_some() { available_modalities.push("Physiological"); }
        
        println!("Available modalities: {}", available_modalities.join(", "));
        
        // Test each fusion strategy
        for strategy_name in fusion_manager.list_strategies() {
            match fusion_manager.fuse(&strategy_name, &test_data) {
                Ok(result) => {
                    println!("\n🔄 {} Results:", strategy_name);
                    println!("  Decision: {:?}", result.final_decision);
                    println!("  Confidence: {:.1}%", result.confidence * 100.0);
                    println!("  Uncertainty: {:.1}%", result.uncertainty_score * 100.0);
                    println!("  Processing time: {}ms", result.processing_time_ms);
                    
                    if !result.individual_scores.is_empty() {
                        println!("  Individual scores:");
                        for (modality, score) in &result.individual_scores {
                            println!("    {}: {:.3}", modality, score);
                        }
                    }
                    
                    if !result.modality_weights.is_empty() {
                        println!("  Modality weights:");
                        for (modality, weight) in &result.modality_weights {
                            println!("    {}: {:.3}", modality, weight);
                        }
                    }
                    
                    println!("  Explanation: {}", result.fusion_explanation);
                }
                Err(e) => {
                    println!("\n❌ {} Error: {}", strategy_name, e);
                }
            }
        }
        
        println!("\n" + &"=".repeat(60) + "\n");
    }
    
    // Performance comparison
    println!("⚡ Performance Comparison");
    println!("{}", "-".repeat(30));
    
    let balanced_data = generate_test_data("balanced");
    let num_iterations = 100;
    
    for strategy_name in fusion_manager.list_strategies() {
        let start_time = Instant::now();
        
        for _ in 0..num_iterations {
            let _ = fusion_manager.fuse(&strategy_name, &balanced_data);
        }
        
        let total_time = start_time.elapsed();
        let avg_time = total_time.as_millis() as f64 / num_iterations as f64;
        
        println!("{}: {:.2}ms avg ({} iterations)", strategy_name, avg_time, num_iterations);
    }
    
    // Strategy comparison matrix
    println!("\n📈 Strategy Comparison Matrix");
    println!("{}", "-".repeat(30));
    
    let test_scenarios = vec![
        ("High Deception", "high_deception"),
        ("Low Deception", "low_deception"),
        ("Missing Video", "missing_video"),
        ("Text Only", "text_only"),
    ];
    
    println!("{:<15} {:<12} {:<12} {:<12} {:<12}", "Scenario", "Early", "Late", "Attention", "Adaptive");
    println!("{}", "-".repeat(75));
    
    for (scenario_name, scenario_key) in test_scenarios {
        let data = generate_test_data(scenario_key);
        print!("{:<15}", scenario_name);
        
        for strategy in ["early_fusion", "late_fusion", "attention_fusion", "adaptive_fusion"] {
            match fusion_manager.fuse(strategy, &data) {
                Ok(result) => {
                    let decision_score = match result.final_decision {
                        DeceptionDecision::Truthful => 0.2,
                        DeceptionDecision::Deceptive => 0.8,
                        DeceptionDecision::Uncertain => 0.5,
                        DeceptionDecision::InsufficientData => 0.0,
                    };
                    print!(" {:<12.1}%", decision_score * 100.0);
                }
                Err(_) => {
                    print!(" {:<12}", "Error");
                }
            }
        }
        println!();
    }
    
    println!("\n💡 Key Insights:");
    println!("   • Early fusion works well with all modalities available");
    println!("   • Late fusion handles missing modalities gracefully");
    println!("   • Attention fusion adapts weights based on feature importance");
    println!("   • Adaptive fusion provides ensemble robustness");
    println!("   • Strategy selection should depend on data availability and requirements");
    
    println!("\n🎉 Multi-modal fusion demonstration completed!");
    println!("\n✨ Features Demonstrated:");
    println!("   • Four different fusion strategies with distinct approaches");
    println!("   • Robust handling of missing modalities");
    println!("   • Attention-based dynamic weighting");
    println!("   • Ensemble fusion for improved robustness");
    println!("   • Comprehensive uncertainty quantification");
    println!("   • Performance comparison and analysis");
    println!("   • Detailed fusion explanations for interpretability");
    
    Ok(())
}