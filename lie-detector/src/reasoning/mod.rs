//! Neuro-symbolic reasoning for lie detection
//!
//! This module implements a hybrid reasoning system that combines neural network
//! processing with symbolic logical reasoning for explainable and robust
//! deception detection decisions.

pub mod neuro_symbolic;
pub mod rule_engine;
pub mod knowledge_base;

pub use neuro_symbolic::*;
pub use rule_engine::*;
pub use knowledge_base::*;

use crate::error::Result;
use crate::types::*;
use num_traits::Float;
use std::collections::HashMap;

/// Core trait for neuro-symbolic reasoning systems
pub trait NeuroSymbolicReasoner<T: Float>: Send + Sync {
    /// Apply symbolic rules to neural network output
    async fn apply_rules(&self, neural_output: &NeuralOutput<T>) -> Result<SymbolicOutput>;
    
    /// Merge neural and symbolic outputs into a final decision
    async fn merge(&self, neural: &NeuralOutput<T>, symbolic: &SymbolicOutput) -> Result<Decision<T>>;
    
    /// Generate explanation for reasoning process
    fn generate_explanation(&self, neural: &NeuralOutput<T>, symbolic: &SymbolicOutput, decision: &Decision<T>) -> ExplanationTrace;
    
    /// Update knowledge base with new evidence
    fn update_knowledge(&mut self, evidence: Evidence) -> Result<()>;
}

/// Evidence for updating knowledge base
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Evidence identifier
    pub id: uuid::Uuid,
    /// Evidence content
    pub content: String,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Confidence in evidence
    pub confidence: f64,
    /// Source of evidence
    pub source: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of evidence
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceType {
    /// Behavioral observation
    Behavioral,
    /// Linguistic pattern
    Linguistic,
    /// Physiological response
    Physiological,
    /// Contextual information
    Contextual,
    /// Expert knowledge
    Expert,
    /// Ground truth
    GroundTruth,
}

/// Neural network output representation
#[derive(Debug, Clone)]
pub struct NeuralOutput<T: Float> {
    /// Raw prediction scores
    pub raw_scores: Vec<T>,
    /// Probability distribution
    pub probabilities: Vec<T>,
    /// Feature activations
    pub features: Vec<Feature<T>>,
    /// Attention weights
    pub attention_weights: Option<Vec<T>>,
    /// Layer activations
    pub layer_activations: HashMap<String, Vec<T>>,
    /// Confidence score
    pub confidence: T,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Symbolic reasoning output
#[derive(Debug, Clone)]
pub struct SymbolicOutput {
    /// Applied rules
    pub rules_applied: Vec<String>,
    /// Logical conclusions
    pub conclusions: Vec<Conclusion>,
    /// Confidence in symbolic reasoning
    pub confidence: f64,
    /// Reasoning chain
    pub reasoning_chain: Vec<ReasoningStep>,
    /// Explanations for each conclusion
    pub explanations: Vec<String>,
}

/// Logical conclusion from symbolic reasoning
#[derive(Debug, Clone)]
pub struct Conclusion {
    /// Conclusion statement
    pub statement: String,
    /// Conclusion type
    pub conclusion_type: ConclusionType,
    /// Confidence level
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Refuting evidence
    pub counter_evidence: Vec<String>,
}

/// Types of conclusions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConclusionType {
    /// Direct inference
    Direct,
    /// Abductive reasoning
    Abductive,
    /// Inductive reasoning
    Inductive,
    /// Deductive reasoning
    Deductive,
    /// Probabilistic inference
    Probabilistic,
}

/// Reasoning step in symbolic processing
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step identifier
    pub id: uuid::Uuid,
    /// Step description
    pub description: String,
    /// Input premises
    pub premises: Vec<String>,
    /// Applied rule
    pub rule: Option<String>,
    /// Output conclusion
    pub conclusion: String,
    /// Step confidence
    pub confidence: f64,
}

/// Configuration for neuro-symbolic reasoning
#[derive(Debug, Clone)]
pub struct NeuroSymbolicConfig<T: Float> {
    /// Neural network weight in final decision
    pub neural_weight: T,
    /// Symbolic reasoning weight in final decision
    pub symbolic_weight: T,
    /// Minimum confidence threshold for decisions
    pub min_confidence: T,
    /// Maximum reasoning depth
    pub max_reasoning_depth: usize,
    /// Enable explanation generation
    pub enable_explanations: bool,
    /// Rule application strategy
    pub rule_strategy: RuleApplicationStrategy,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
}

/// Strategies for applying symbolic rules
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleApplicationStrategy {
    /// Apply all applicable rules
    Exhaustive,
    /// Apply rules by priority order
    Priority,
    /// Apply most confident rules first
    Confidence,
    /// Apply rules with highest expected utility
    Utility,
}

/// Strategies for resolving conflicts between neural and symbolic outputs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolutionStrategy {
    /// Trust neural output more
    Neural,
    /// Trust symbolic output more
    Symbolic,
    /// Weighted average based on confidence
    Weighted,
    /// Use ensemble voting
    Ensemble,
    /// Defer to human judgment
    Human,
}

impl<T: Float> Default for NeuroSymbolicConfig<T> {
    fn default() -> Self {
        Self {
            neural_weight: T::from(0.6).unwrap(),
            symbolic_weight: T::from(0.4).unwrap(),
            min_confidence: T::from(0.3).unwrap(),
            max_reasoning_depth: 10,
            enable_explanations: true,
            rule_strategy: RuleApplicationStrategy::Priority,
            conflict_resolution: ConflictResolutionStrategy::Weighted,
        }
    }
}

/// Utility functions for neuro-symbolic integration
pub mod utils {
    use super::*;
    
    /// Convert neural features to symbolic predicates
    pub fn neural_to_symbolic<T: Float>(features: &[Feature<T>]) -> Vec<String> {
        features.iter()
            .filter(|f| f.weight > T::from(0.5).unwrap())
            .map(|f| format!("{}({})", f.name, f.value.to_f64().unwrap()))
            .collect()
    }
    
    /// Calculate confidence weighted score
    pub fn weighted_confidence_score<T: Float>(
        neural_score: T,
        neural_confidence: T,
        symbolic_score: f64,
        symbolic_confidence: f64,
        neural_weight: T,
        symbolic_weight: T,
    ) -> T {
        let neural_contrib = neural_score * neural_confidence * neural_weight;
        let symbolic_contrib = T::from(symbolic_score * symbolic_confidence).unwrap() * symbolic_weight;
        let total_weight = neural_confidence * neural_weight + T::from(symbolic_confidence).unwrap() * symbolic_weight;
        
        if total_weight > T::zero() {
            (neural_contrib + symbolic_contrib) / total_weight
        } else {
            T::from(0.5).unwrap() // Default neutral score
        }
    }
    
    /// Extract symbolic facts from observations
    pub fn extract_symbolic_facts<T: Float>(observations: &Observations<T>) -> Vec<String> {
        let mut facts = Vec::new();
        
        // Vision facts
        if let Some(vision) = &observations.vision {
            if vision.face_detected {
                facts.push("face_detected(true)".to_string());
                
                if !vision.micro_expressions.is_empty() {
                    for expr in &vision.micro_expressions {
                        facts.push(format!("micro_expression({})", expr));
                    }
                }
                
                if !vision.gaze_patterns.is_empty() {
                    for pattern in &vision.gaze_patterns {
                        facts.push(format!("gaze_pattern({})", pattern));
                    }
                }
            }
        }
        
        // Audio facts
        if let Some(audio) = &observations.audio {
            let voice_quality = audio.voice_quality.to_f64().unwrap_or(0.0);
            facts.push(format!("voice_quality({:.2})", voice_quality));
            
            let speaking_rate = audio.speaking_rate.to_f64().unwrap_or(0.0);
            if speaking_rate > 180.0 {
                facts.push("speaking_rate(fast)".to_string());
            } else if speaking_rate < 120.0 {
                facts.push("speaking_rate(slow)".to_string());
            } else {
                facts.push("speaking_rate(normal)".to_string());
            }
            
            for indicator in &audio.stress_indicators {
                facts.push(format!("stress_indicator({})", indicator));
            }
        }
        
        // Text facts
        if let Some(text) = &observations.text {
            if text.sentiment_score < -0.3 {
                facts.push("sentiment(negative)".to_string());
            } else if text.sentiment_score > 0.3 {
                facts.push("sentiment(positive)".to_string());
            } else {
                facts.push("sentiment(neutral)".to_string());
            }
            
            for indicator in &text.deception_indicators {
                facts.push(format!("deception_indicator({})", indicator));
            }
            
            // Simple linguistic analysis
            if text.content.contains("not") || text.content.contains("never") {
                facts.push("contains_negation(true)".to_string());
            }
            
            if text.content.contains("I think") || text.content.contains("maybe") || text.content.contains("perhaps") {
                facts.push("contains_hedging(true)".to_string());
            }
        }
        
        // Physiological facts
        if let Some(physio) = &observations.physiological {
            let stress_level = physio.stress_level.to_f64().unwrap_or(0.0);
            if stress_level > 0.7 {
                facts.push("stress_level(high)".to_string());
            } else if stress_level > 0.4 {
                facts.push("stress_level(medium)".to_string());
            } else {
                facts.push("stress_level(low)".to_string());
            }
            
            let arousal = physio.arousal_level.to_f64().unwrap_or(0.0);
            if arousal > 0.6 {
                facts.push("arousal(high)".to_string());
            } else {
                facts.push("arousal(normal)".to_string());
            }
        }
        
        facts
    }
    
    /// Validate reasoning chain for logical consistency
    pub fn validate_reasoning_chain(chain: &[ReasoningStep]) -> bool {
        if chain.is_empty() {
            return true;
        }
        
        // Check for circular reasoning
        let conclusions: Vec<&str> = chain.iter().map(|s| s.conclusion.as_str()).collect();
        for (i, step) in chain.iter().enumerate() {
            for premise in &step.premises {
                if conclusions[..i].contains(&premise.as_str()) {
                    // This is actually valid - using previous conclusions as premises
                    continue;
                }
                if conclusions[i+1..].contains(&premise.as_str()) {
                    // Circular reasoning detected
                    return false;
                }
            }
        }
        
        // Check confidence consistency (optional)
        let avg_confidence: f64 = chain.iter().map(|s| s.confidence).sum::<f64>() / chain.len() as f64;
        
        // Reasoning chain is valid if no circular dependencies and reasonable confidence
        avg_confidence > 0.1
    }
}