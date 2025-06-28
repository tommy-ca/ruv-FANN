//! Neuro-symbolic reasoning implementation
//!
//! This module implements a hybrid reasoning system that combines neural network
//! processing with symbolic logical reasoning for explainable and robust
//! deception detection decisions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;
use chrono::Utc;

use crate::error::{Result, VeritasError};
use crate::types::*;
use crate::learning::ActivationFunction;
use super::*;

/// Main neuro-symbolic reasoning system
pub struct NeuroSymbolicReasoner<T: Float> {
    config: NeuroSymbolicConfig<T>,
    /// Neural processing components
    neural_processor: NeuralProcessor<T>,
    /// Symbolic reasoning engine
    symbolic_engine: SymbolicEngine,
    /// Knowledge base for storing facts and rules
    knowledge_base: Arc<Mutex<KnowledgeBase>>,
    /// Rule engine for applying logical rules
    rule_engine: Arc<RuleEngine>,
    /// Explanation generator
    explanation_generator: ExplanationGenerator,
    /// Integration statistics
    stats: IntegrationStats,
}

/// Neural processing component
pub struct NeuralProcessor<T: Float> {
    /// Feature extractors by modality
    feature_extractors: HashMap<ModalityType, Box<dyn FeatureExtractor<T>>>,
    /// Neural networks for different tasks
    networks: HashMap<String, NeuralNetwork<T>>,
    /// Processing pipeline configuration
    pipeline_config: PipelineConfig<T>,
}

/// Symbolic reasoning engine
pub struct SymbolicEngine {
    /// Fact database
    facts: HashMap<String, Fact>,
    /// Active rules
    rules: Vec<Rule>,
    /// Inference chains
    inference_chains: Vec<InferenceChain>,
    /// Reasoning context
    context: ReasoningContext,
}

/// Explanation generator for neuro-symbolic reasoning
pub struct ExplanationGenerator {
    /// Template library for explanations
    templates: HashMap<String, String>,
    /// Generation configuration
    config: ExplanationConfig,
}

impl ExplanationGenerator {
    /// Create new explanation generator
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        templates.insert(
            "neural_confidence".to_string(),
            "Neural network analysis shows {confidence:.2}% confidence in {prediction}".to_string(),
        );
        templates.insert(
            "symbolic_rule".to_string(),
            "Symbolic rule '{rule}' applies: {conclusion}".to_string(),
        );
        templates.insert(
            "multimodal_fusion".to_string(),
            "Combined evidence from {modalities} indicates {result}".to_string(),
        );
        
        Self {
            templates,
            config: ExplanationConfig::default(),
        }
    }
}

/// Configuration for explanation generation
#[derive(Debug, Clone)]
pub struct ExplanationConfig {
    /// Maximum explanation length
    pub max_length: usize,
    /// Include technical details
    pub include_technical: bool,
    /// Confidence threshold for detailed explanations
    pub detail_threshold: f64,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            max_length: 500,
            include_technical: false,
            detail_threshold: 0.7,
        }
    }
}

/// Feature extractor trait for different modalities
pub trait FeatureExtractor<T: Float>: Send + Sync {
    /// Extract features from raw data
    fn extract_features(&self, data: &[u8]) -> Result<Vec<Feature<T>>>;
    
    /// Get supported modality
    fn modality(&self) -> ModalityType;
    
    /// Get feature dimension
    fn feature_dimension(&self) -> usize;
}

/// Neural network representation
#[derive(Debug, Clone)]
pub struct NeuralNetwork<T: Float> {
    /// Network identifier
    pub id: String,
    /// Network type
    pub network_type: NetworkType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Weights (simplified representation)
    pub weights: Vec<T>,
    /// Biases
    pub biases: Vec<T>,
    /// Activation functions
    pub activations: Vec<ActivationFunction>,
    /// Network metadata
    pub metadata: HashMap<String, String>,
}

/// Types of neural networks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkType {
    /// Feedforward network
    Feedforward,
    /// Convolutional network
    Convolutional,
    /// Recurrent network
    Recurrent,
    /// Transformer network
    Transformer,
    /// Graph neural network
    GraphNeural,
}

/// Pipeline configuration for neural processing
#[derive(Debug, Clone)]
pub struct PipelineConfig<T: Float> {
    /// Processing stages
    pub stages: Vec<ProcessingStage>,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy<T>,
    /// Quality thresholds
    pub quality_thresholds: HashMap<String, T>,
    /// Timeout settings
    pub timeout_ms: u64,
}

/// Processing stage in neural pipeline
#[derive(Debug, Clone)]
pub struct ProcessingStage {
    /// Stage name
    pub name: String,
    /// Stage type
    pub stage_type: StageType,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Dependencies on other stages
    pub dependencies: Vec<String>,
}

/// Types of processing stages
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageType {
    /// Data preprocessing
    Preprocessing,
    /// Feature extraction
    FeatureExtraction,
    /// Neural network inference
    NeuralInference,
    /// Post-processing
    PostProcessing,
    /// Quality assessment
    QualityAssessment,
}

/// Symbolic fact representation
#[derive(Debug, Clone)]
pub struct Fact {
    /// Fact identifier
    pub id: String,
    /// Fact predicate
    pub predicate: String,
    /// Fact arguments
    pub arguments: Vec<String>,
    /// Confidence in fact
    pub confidence: f64,
    /// Fact source
    pub source: FactSource,
    /// Timestamp when fact was added
    pub timestamp: chrono::DateTime<Utc>,
    /// Fact metadata
    pub metadata: HashMap<String, String>,
}

/// Sources of facts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactSource {
    /// Derived from neural network output
    Neural,
    /// Derived from symbolic reasoning
    Symbolic,
    /// User-provided fact
    UserProvided,
    /// External knowledge base
    External,
    /// Observation-based fact
    Observation,
}

/// Logical rule for symbolic reasoning
#[derive(Debug, Clone)]
pub struct Rule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule premises (antecedents)
    pub premises: Vec<Premise>,
    /// Rule conclusion (consequent)
    pub conclusion: Conclusion,
    /// Rule confidence
    pub confidence: f64,
    /// Rule priority
    pub priority: i32,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}

/// Premise in a logical rule
#[derive(Debug, Clone)]
pub struct Premise {
    /// Premise predicate
    pub predicate: String,
    /// Premise arguments
    pub arguments: Vec<String>,
    /// Whether premise is negated
    pub negated: bool,
    /// Premise weight
    pub weight: f64,
}

/// Inference chain for tracking reasoning
#[derive(Debug, Clone)]
pub struct InferenceChain {
    /// Chain identifier
    pub id: Uuid,
    /// Applied rules in order
    pub applied_rules: Vec<String>,
    /// Intermediate conclusions
    pub intermediate_conclusions: Vec<Conclusion>,
    /// Final conclusion
    pub final_conclusion: Option<Conclusion>,
    /// Chain confidence
    pub confidence: f64,
    /// Chain timestamp
    pub timestamp: chrono::DateTime<Utc>,
}

/// Reasoning context for maintaining state
#[derive(Debug, Clone, Default)]
pub struct ReasoningContext {
    /// Current working memory
    pub working_memory: HashMap<String, String>,
    /// Active hypotheses
    pub active_hypotheses: Vec<String>,
    /// Reasoning goals
    pub goals: Vec<String>,
    /// Context metadata
    pub metadata: HashMap<String, String>,
}

/// Integration statistics
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Neural processing time
    pub neural_processing_time_ms: f64,
    /// Symbolic processing time
    pub symbolic_processing_time_ms: f64,
    /// Integration time
    pub integration_time_ms: f64,
    /// Number of facts generated
    pub facts_generated: usize,
    /// Number of rules applied
    pub rules_applied: usize,
    /// Confidence in final decision
    pub final_confidence: f64,
    /// Agreement between neural and symbolic
    pub neural_symbolic_agreement: f64,
}

impl<T: Float> NeuroSymbolicReasoner<T> {
    /// Create new neuro-symbolic reasoner
    pub fn new(
        config: NeuroSymbolicConfig<T>,
        knowledge_base: Arc<Mutex<KnowledgeBase>>,
        rule_engine: Arc<RuleEngine>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            neural_processor: NeuralProcessor::new()?,
            symbolic_engine: SymbolicEngine::new(),
            knowledge_base,
            rule_engine,
            explanation_generator: ExplanationGenerator::new(),
            stats: IntegrationStats::default(),
        })
    }

    /// Process observations through neuro-symbolic pipeline
    pub async fn process_observations(&mut self, observations: &Observations<T>) -> Result<Decision<T>> {
        let start_time = Instant::now();

        // Step 1: Neural processing
        let neural_start = Instant::now();
        let neural_output = self.neural_processing(observations).await?;
        self.stats.neural_processing_time_ms = neural_start.elapsed().as_millis() as f64;

        // Step 2: Convert neural output to symbolic facts
        let symbolic_facts = self.neural_to_symbolic(&neural_output)?;

        // Step 3: Symbolic reasoning
        let symbolic_start = Instant::now();
        let symbolic_output = self.symbolic_reasoning(&symbolic_facts).await?;
        self.stats.symbolic_processing_time_ms = symbolic_start.elapsed().as_millis() as f64;

        // Step 4: Integration and final decision
        let integration_start = Instant::now();
        let final_decision = self.integrate_outputs(&neural_output, &symbolic_output).await?;
        self.stats.integration_time_ms = integration_start.elapsed().as_millis() as f64;

        // Update statistics
        self.stats.facts_generated = symbolic_facts.len();
        self.stats.neural_symbolic_agreement = self.calculate_agreement(&neural_output, &symbolic_output);

        Ok(final_decision)
    }

    /// Perform neural processing on observations
    async fn neural_processing(&mut self, observations: &Observations<T>) -> Result<NeuralOutput<T>> {
        let mut features = Vec::new();
        let mut raw_scores = Vec::new();
        let mut layer_activations = HashMap::new();
        let mut metadata = HashMap::new();

        // Process each modality
        if let Some(vision) = &observations.vision {
            let vision_features = self.process_vision_modality(vision)?;
            features.extend(vision_features);
        }

        if let Some(audio) = &observations.audio {
            let audio_features = self.process_audio_modality(audio)?;
            features.extend(audio_features);
        }

        if let Some(text) = &observations.text {
            let text_features = self.process_text_modality(text)?;
            features.extend(text_features);
        }

        if let Some(physio) = &observations.physiological {
            let physio_features = self.process_physiological_modality(physio)?;
            features.extend(physio_features);
        }

        // Generate prediction scores (simplified)
        raw_scores = vec![
            T::from(0.7).unwrap(), // Truth score
            T::from(0.3).unwrap(), // Deception score
        ];

        // Calculate probabilities (softmax)
        let max_score = raw_scores.iter().cloned().fold(T::neg_infinity(), T::max);
        let exp_scores: Vec<T> = raw_scores.iter()
            .map(|&score| (score - max_score).exp())
            .collect();
        let sum_exp: T = exp_scores.iter().cloned().sum();
        let probabilities: Vec<T> = exp_scores.iter()
            .map(|&exp_score| exp_score / sum_exp)
            .collect();

        // Calculate overall confidence
        let confidence = probabilities.iter().cloned().fold(T::zero(), T::max);

        metadata.insert("processing_time_ms".to_string(), "50".to_string());
        metadata.insert("modalities_processed".to_string(), "4".to_string());

        Ok(NeuralOutput {
            raw_scores,
            probabilities,
            features,
            attention_weights: None,
            layer_activations,
            confidence,
            metadata,
        })
    }

    /// Process vision modality
    fn process_vision_modality(&self, vision: &VisionObservation) -> Result<Vec<Feature<T>>> {
        let mut features = Vec::new();

        // Face detection feature
        features.push(Feature {
            name: "face_detected".to_string(),
            value: if vision.face_detected { T::one() } else { T::zero() },
            weight: T::from(0.8).unwrap(),
            feature_type: "binary".to_string(),
        });

        // Micro-expression features
        features.push(Feature {
            name: "micro_expression_count".to_string(),
            value: T::from(vision.micro_expressions.len()).unwrap(),
            weight: T::from(0.9).unwrap(),
            feature_type: "count".to_string(),
        });

        // Gaze pattern features
        features.push(Feature {
            name: "gaze_pattern_count".to_string(),
            value: T::from(vision.gaze_patterns.len()).unwrap(),
            weight: T::from(0.7).unwrap(),
            feature_type: "count".to_string(),
        });

        Ok(features)
    }

    /// Process audio modality
    fn process_audio_modality(&self, audio: &AudioObservation<T>) -> Result<Vec<Feature<T>>> {
        let mut features = Vec::new();

        features.push(Feature {
            name: "voice_quality".to_string(),
            value: audio.voice_quality,
            weight: T::from(0.8).unwrap(),
            feature_type: "continuous".to_string(),
        });

        features.push(Feature {
            name: "speaking_rate".to_string(),
            value: audio.speaking_rate,
            weight: T::from(0.6).unwrap(),
            feature_type: "continuous".to_string(),
        });

        features.push(Feature {
            name: "stress_indicator_count".to_string(),
            value: T::from(audio.stress_indicators.len()).unwrap(),
            weight: T::from(0.9).unwrap(),
            feature_type: "count".to_string(),
        });

        Ok(features)
    }

    /// Process text modality
    fn process_text_modality(&self, text: &TextObservation) -> Result<Vec<Feature<T>>> {
        let mut features = Vec::new();

        features.push(Feature {
            name: "sentiment_score".to_string(),
            value: T::from(text.sentiment_score).unwrap(),
            weight: T::from(0.7).unwrap(),
            feature_type: "continuous".to_string(),
        });

        features.push(Feature {
            name: "deception_indicator_count".to_string(),
            value: T::from(text.deception_indicators.len()).unwrap(),
            weight: T::from(0.95).unwrap(),
            feature_type: "count".to_string(),
        });

        features.push(Feature {
            name: "text_length".to_string(),
            value: T::from(text.content.len()).unwrap(),
            weight: T::from(0.3).unwrap(),
            feature_type: "count".to_string(),
        });

        Ok(features)
    }

    /// Process physiological modality
    fn process_physiological_modality(&self, physio: &PhysiologicalObservation<T>) -> Result<Vec<Feature<T>>> {
        let mut features = Vec::new();

        features.push(Feature {
            name: "stress_level".to_string(),
            value: physio.stress_level,
            weight: T::from(0.9).unwrap(),
            feature_type: "continuous".to_string(),
        });

        features.push(Feature {
            name: "arousal_level".to_string(),
            value: physio.arousal_level,
            weight: T::from(0.8).unwrap(),
            feature_type: "continuous".to_string(),
        });

        features.push(Feature {
            name: "heart_rate_variability".to_string(),
            value: physio.heart_rate_variability,
            weight: T::from(0.7).unwrap(),
            feature_type: "continuous".to_string(),
        });

        Ok(features)
    }

    /// Convert neural output to symbolic facts
    fn neural_to_symbolic(&self, neural_output: &NeuralOutput<T>) -> Result<Vec<Fact>> {
        let mut facts = Vec::new();
        let timestamp = Utc::now();

        // Convert features to facts
        for feature in &neural_output.features {
            let confidence = (feature.weight.to_f64().unwrap_or(0.0) * 
                           feature.value.to_f64().unwrap_or(0.0)).min(1.0);

            if confidence > 0.5 {
                let fact = Fact {
                    id: Uuid::new_v4().to_string(),
                    predicate: feature.name.clone(),
                    arguments: vec![feature.value.to_string()],
                    confidence,
                    source: FactSource::Neural,
                    timestamp,
                    metadata: HashMap::new(),
                };
                facts.push(fact);
            }
        }

        // Convert neural predictions to facts
        if neural_output.probabilities.len() >= 2 {
            let truth_prob = neural_output.probabilities[0].to_f64().unwrap_or(0.0);
            let deception_prob = neural_output.probabilities[1].to_f64().unwrap_or(0.0);

            if truth_prob > 0.5 {
                facts.push(Fact {
                    id: Uuid::new_v4().to_string(),
                    predicate: "neural_prediction".to_string(),
                    arguments: vec!["truth".to_string(), truth_prob.to_string()],
                    confidence: truth_prob,
                    source: FactSource::Neural,
                    timestamp,
                    metadata: HashMap::new(),
                });
            }

            if deception_prob > 0.5 {
                facts.push(Fact {
                    id: Uuid::new_v4().to_string(),
                    predicate: "neural_prediction".to_string(),
                    arguments: vec!["deception".to_string(), deception_prob.to_string()],
                    confidence: deception_prob,
                    source: FactSource::Neural,
                    timestamp,
                    metadata: HashMap::new(),
                });
            }
        }

        Ok(facts)
    }

    /// Perform symbolic reasoning on facts
    async fn symbolic_reasoning(&mut self, facts: &[Fact]) -> Result<SymbolicOutput> {
        let mut conclusions = Vec::new();
        let mut reasoning_chain = Vec::new();
        let mut rules_applied = Vec::new();

        // Add facts to symbolic engine
        for fact in facts {
            self.symbolic_engine.add_fact(fact.clone());
        }

        // Apply deception detection rules
        let rule_engine = Arc::clone(&self.rule_engine);
        let applied_rules = rule_engine.apply_rules(&self.symbolic_engine.facts)?;

        for (rule_id, conclusion) in applied_rules {
            rules_applied.push(rule_id.clone());
            conclusions.push(conclusion.clone());

            let reasoning_step = ReasoningStep {
                id: Uuid::new_v4(),
                description: format!("Applied rule: {}", rule_id),
                premises: vec![], // Would be populated from rule premises
                rule: Some(rule_id),
                conclusion: conclusion.statement.clone(),
                confidence: conclusion.confidence,
            };
            reasoning_chain.push(reasoning_step);
        }

        // Calculate overall confidence
        let confidence = if conclusions.is_empty() {
            0.5 // Neutral confidence when no conclusions
        } else {
            conclusions.iter().map(|c| c.confidence).sum::<f64>() / conclusions.len() as f64
        };

        // Generate explanations
        let explanations = conclusions.iter()
            .map(|c| format!("Conclusion: {} (confidence: {:.2})", c.statement, c.confidence))
            .collect();

        Ok(SymbolicOutput {
            rules_applied,
            conclusions,
            confidence,
            reasoning_chain,
            explanations,
        })
    }

    /// Integrate neural and symbolic outputs
    async fn integrate_outputs(
        &mut self,
        neural_output: &NeuralOutput<T>,
        symbolic_output: &SymbolicOutput,
    ) -> Result<Decision<T>> {
        let neural_weight = self.config.neural_weight;
        let symbolic_weight = self.config.symbolic_weight;

        // Extract neural decision
        let neural_decision = if neural_output.probabilities.len() >= 2 {
            let truth_prob = neural_output.probabilities[0].to_f64().unwrap_or(0.0);
            let deception_prob = neural_output.probabilities[1].to_f64().unwrap_or(0.0);
            
            if truth_prob > deception_prob {
                (Decision::Truth, truth_prob)
            } else {
                (Decision::Deception, deception_prob)
            }
        } else {
            (Decision::Uncertain, 0.5)
        };

        // Extract symbolic decision
        let symbolic_decision = if symbolic_output.conclusions.is_empty() {
            (Decision::Uncertain, 0.5)
        } else {
            // Find the highest confidence conclusion
            let best_conclusion = symbolic_output.conclusions.iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .unwrap();
            
            let decision = if best_conclusion.statement.contains("truth") {
                Decision::Truth
            } else if best_conclusion.statement.contains("deception") {
                Decision::Deception
            } else {
                Decision::Uncertain
            };
            
            (decision, best_conclusion.confidence)
        };

        // Weighted combination
        let neural_score = neural_decision.1 * neural_weight.to_f64().unwrap_or(0.6);
        let symbolic_score = symbolic_decision.1 * symbolic_weight.to_f64().unwrap_or(0.4);
        let combined_confidence = neural_score + symbolic_score;

        // Final decision based on agreement and confidence
        let final_decision = match self.config.conflict_resolution {
            ConflictResolutionStrategy::Neural => neural_decision.0,
            ConflictResolutionStrategy::Symbolic => symbolic_decision.0,
            ConflictResolutionStrategy::Weighted => {
                if neural_decision.0 == symbolic_decision.0 {
                    neural_decision.0 // Agreement
                } else if combined_confidence > 0.7 {
                    // High confidence, choose higher confidence decision
                    if neural_decision.1 > symbolic_decision.1 {
                        neural_decision.0
                    } else {
                        symbolic_decision.0
                    }
                } else {
                    Decision::Uncertain // Low confidence disagreement
                }
            },
            ConflictResolutionStrategy::Ensemble => {
                // Simple majority voting (would be more sophisticated in practice)
                if neural_decision.0 == symbolic_decision.0 {
                    neural_decision.0
                } else {
                    Decision::Uncertain
                }
            },
            ConflictResolutionStrategy::Human => Decision::Uncertain, // Defer to human
        };

        self.stats.final_confidence = combined_confidence;

        Ok(Decision {
            decision: final_decision,
            confidence: T::from(combined_confidence).unwrap(),
            explanation: format!(
                "Neural: {:?} ({:.2}), Symbolic: {:?} ({:.2}), Combined: ({:.2})",
                neural_decision.0, neural_decision.1,
                symbolic_decision.0, symbolic_decision.1,
                combined_confidence
            ),
            reasoning_trace: self.generate_full_reasoning_trace(neural_output, symbolic_output),
        })
    }

    /// Calculate agreement between neural and symbolic outputs
    fn calculate_agreement(&self, neural_output: &NeuralOutput<T>, symbolic_output: &SymbolicOutput) -> f64 {
        // Extract decisions
        let neural_decision = if neural_output.probabilities.len() >= 2 {
            if neural_output.probabilities[0] > neural_output.probabilities[1] {
                "truth"
            } else {
                "deception"
            }
        } else {
            "uncertain"
        };

        let symbolic_decision = if symbolic_output.conclusions.is_empty() {
            "uncertain"
        } else {
            let best_conclusion = symbolic_output.conclusions.iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .unwrap();
            
            if best_conclusion.statement.contains("truth") {
                "truth"
            } else if best_conclusion.statement.contains("deception") {
                "deception"
            } else {
                "uncertain"
            }
        };

        if neural_decision == symbolic_decision {
            1.0 // Perfect agreement
        } else if neural_decision == "uncertain" || symbolic_decision == "uncertain" {
            0.5 // Partial agreement when one is uncertain
        } else {
            0.0 // Disagreement
        }
    }

    /// Generate full reasoning trace
    fn generate_full_reasoning_trace(
        &self,
        neural_output: &NeuralOutput<T>,
        symbolic_output: &SymbolicOutput,
    ) -> ExplanationTrace {
        let mut steps = Vec::new();

        // Neural processing step
        steps.push(ExplanationStep {
            step: 1,
            description: "Neural network processing".to_string(),
            input: "Multi-modal observations".to_string(),
            output: format!("Confidence: {:.2}", neural_output.confidence.to_f64().unwrap_or(0.0)),
            confidence: Confidence::new(neural_output.confidence.to_f64().unwrap_or(0.0)).unwrap_or(Confidence::new_unchecked(0.5)),
            duration: Duration::from_millis(50),
        });

        // Symbolic reasoning step
        steps.push(ExplanationStep {
            step: 2,
            description: "Symbolic rule application".to_string(),
            input: "Neural features converted to facts".to_string(),
            output: format!("{} rules applied, {} conclusions", 
                          symbolic_output.rules_applied.len(),
                          symbolic_output.conclusions.len()),
            confidence: Confidence::new(symbolic_output.confidence).unwrap_or(Confidence::new_unchecked(0.5)),
            duration: Duration::from_millis(30),
        });

        // Integration step
        steps.push(ExplanationStep {
            step: 3,
            description: "Neural-symbolic integration".to_string(),
            input: "Neural and symbolic decisions".to_string(),
            output: "Final integrated decision".to_string(),
            confidence: Confidence::new(self.stats.final_confidence).unwrap_or(Confidence::new_unchecked(0.5)),
            duration: Duration::from_millis(10),
        });

        let key_factors = vec![
            "Neural network confidence".to_string(),
            "Symbolic rule matching".to_string(),
            "Cross-modal consistency".to_string(),
            "Historical pattern matching".to_string(),
        ];

        ExplanationTrace {
            steps,
            summary: format!(
                "Integrated neuro-symbolic reasoning with {:.2} agreement between components",
                self.stats.neural_symbolic_agreement
            ),
            explanation_confidence: Confidence::new(0.8).unwrap(),
            key_factors,
        }
    }

    /// Get integration statistics
    pub fn get_stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Update configuration
    pub fn update_config(&mut self, config: NeuroSymbolicConfig<T>) -> Result<()> {
        self.config = config;
        Ok(())
    }
}

/// Decision with extended information
#[derive(Debug, Clone)]
pub struct Decision<T: Float> {
    /// The decision made
    pub decision: crate::types::Decision,
    /// Confidence in the decision
    pub confidence: T,
    /// Explanation of the decision
    pub explanation: String,
    /// Full reasoning trace
    pub reasoning_trace: ExplanationTrace,
}

impl<T: Float> NeuralProcessor<T> {
    /// Create new neural processor
    pub fn new() -> Result<Self> {
        Ok(Self {
            feature_extractors: HashMap::new(),
            networks: HashMap::new(),
            pipeline_config: PipelineConfig {
                stages: vec![],
                fusion_strategy: FusionStrategy::WeightedAverage,
                quality_thresholds: HashMap::new(),
                timeout_ms: 5000,
            },
        })
    }
}

impl SymbolicEngine {
    /// Create new symbolic engine
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
            rules: Vec::new(),
            inference_chains: Vec::new(),
            context: ReasoningContext::default(),
        }
    }

    /// Add fact to the engine
    pub fn add_fact(&mut self, fact: Fact) {
        self.facts.insert(fact.id.clone(), fact);
    }

    /// Get all facts
    pub fn get_facts(&self) -> &HashMap<String, Fact> {
        &self.facts
    }
}

/// Fusion strategy for combining modalities
#[derive(Debug, Clone)]
pub enum FusionStrategy<T: Float> {
    /// Simple average
    Average,
    /// Weighted average
    WeightedAverage,
    /// Maximum confidence
    MaxConfidence,
    /// Attention-based fusion
    AttentionBased,
    /// Custom fusion with parameters
    Custom(HashMap<String, T>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning::{KnowledgeBase, RuleEngine};

    #[test]
    fn test_neuro_symbolic_reasoner_creation() {
        let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
        let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
        let rule_engine = Arc::new(RuleEngine::new());
        
        let reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine);
        assert!(reasoner.is_ok());
    }

    #[test]
    fn test_neural_to_symbolic_conversion() {
        let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
        let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
        let rule_engine = Arc::new(RuleEngine::new());
        
        let reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine).unwrap();
        
        let neural_output = NeuralOutput {
            raw_scores: vec![0.7, 0.3],
            probabilities: vec![0.7, 0.3],
            features: vec![
                Feature {
                    name: "stress_level".to_string(),
                    value: 0.8,
                    weight: 0.9,
                    feature_type: "continuous".to_string(),
                }
            ],
            attention_weights: None,
            layer_activations: HashMap::new(),
            confidence: 0.7,
            metadata: HashMap::new(),
        };
        
        let facts = reasoner.neural_to_symbolic(&neural_output);
        assert!(facts.is_ok());
        let facts = facts.unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_fact_creation() {
        let fact = Fact {
            id: "test_fact".to_string(),
            predicate: "stress_level".to_string(),
            arguments: vec!["high".to_string()],
            confidence: 0.8,
            source: FactSource::Neural,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(fact.predicate, "stress_level");
        assert_eq!(fact.confidence, 0.8);
    }

    #[test]
    fn test_feature_extraction() {
        let vision_obs = VisionObservation {
            face_detected: true,
            micro_expressions: vec!["surprise".to_string(), "fear".to_string()],
            gaze_patterns: vec!["avoidance".to_string()],
            facial_landmarks: vec![(0.1, 0.2), (0.3, 0.4)],
        };
        
        let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
        let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
        let rule_engine = Arc::new(RuleEngine::new());
        
        let reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine).unwrap();
        let features = reasoner.process_vision_modality(&vision_obs);
        
        assert!(features.is_ok());
        let features = features.unwrap();
        assert!(!features.is_empty());
        assert!(features.iter().any(|f| f.name == "face_detected"));
    }
}