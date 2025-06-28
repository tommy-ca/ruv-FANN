//! # Veritas Nexus: Multi-Modal Lie Detection System
//!
//! A cutting-edge Rust implementation of a multi-modal lie detection system that combines
//! state-of-the-art neural processing with explainable AI techniques.
//!
//! ## Features
//!
//! - **Multi-Modal Analysis**: Vision, audio, text, and physiological signal processing
//! - **Blazing Performance**: CPU-optimized with optional GPU acceleration
//! - **Explainable AI**: ReAct reasoning framework with complete decision traces
//! - **Self-Improving**: GSPO reinforcement learning for continuous improvement
//! - **Ethical Design**: Privacy-preserving, bias-aware, human-in-the-loop

pub mod error;
pub mod types;
pub mod modalities;
pub mod prelude;

pub use error::{VeritasError, Result};
pub use types::*;

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use async_trait::async_trait;

/// Core trait for all modality analyzers
#[async_trait]
pub trait ModalityAnalyzer<T: Float>: Send + Sync {
    type Input: Send + Sync;
    type Output: DeceptionScore<T> + Send + Sync;
    type Config: Send + Sync;
    
    /// Analyze input data for deception indicators
    async fn analyze(&self, input: &Self::Input) -> Result<Self::Output>;
    
    /// Get the current confidence level of the analyzer
    fn confidence(&self) -> T;
    
    /// Generate an explanation trace for the analysis
    fn explain(&self) -> ExplanationTrace;
    
    /// Get the analyzer's configuration
    fn config(&self) -> &Self::Config;
}

/// Trait for deception scores from any modality
pub trait DeceptionScore<T: Float>: Debug + Send + Sync {
    /// Get the deception probability (0.0 to 1.0)
    fn probability(&self) -> T;
    
    /// Get the confidence of this score (0.0 to 1.0)
    fn confidence(&self) -> T;
    
    /// Get the modality that generated this score
    fn modality(&self) -> ModalityType;
    
    /// Get detailed features that contributed to this score
    fn features(&self) -> Vec<Feature<T>>;
    
    /// Get the timestamp of when this score was generated
    fn timestamp(&self) -> std::time::SystemTime;
}

/// Types of supported modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    Text,
    Vision,
    Audio,
    Physiological,
}

/// A feature that contributed to a deception score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature<T: Float> {
    pub name: String,
    pub value: T,
    pub weight: T,
    pub description: String,
}

/// Explanation trace for analysis decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationTrace {
    pub steps: Vec<ExplanationStep>,
    pub confidence: f64,
    pub reasoning: String,
}

/// A single step in an explanation trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationStep {
    pub step_type: String,
    pub description: String,
    pub evidence: Vec<String>,
    pub confidence: f64,
}

/// Fusion strategy trait for combining multiple modality scores
#[async_trait] 
pub trait FusionStrategy<T: Float>: Send + Sync {
    /// Fuse multiple scores into a single decision
    async fn fuse(&self, scores: &[Box<dyn DeceptionScore<T>>]) -> Result<FusedScore<T>>;
    
    /// Get the current weights for each modality
    fn weights(&self) -> &[T];
    
    /// Update weights based on feedback
    fn update_weights(&mut self, feedback: &Feedback<T>);
}

/// A fused score from multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedScore<T: Float> {
    pub probability: T,
    pub confidence: T,
    pub modality_contributions: std::collections::HashMap<ModalityType, T>,
    pub timestamp: std::time::SystemTime,
    pub explanation: ExplanationTrace,
}

impl<T: Float> DeceptionScore<T> for FusedScore<T> {
    fn probability(&self) -> T {
        self.probability
    }
    
    fn confidence(&self) -> T {
        self.confidence
    }
    
    fn modality(&self) -> ModalityType {
        ModalityType::Text // Fusion is considered a meta-modality
    }
    
    fn features(&self) -> Vec<Feature<T>> {
        vec![] // Fused scores don't have individual features
    }
    
    fn timestamp(&self) -> std::time::SystemTime {
        self.timestamp
    }
}

/// Feedback for updating fusion weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback<T: Float> {
    pub ground_truth: bool,
    pub predicted_probability: T,
    pub modality_accuracies: std::collections::HashMap<ModalityType, T>,
}

/// ReAct agent trait for reasoning and action
#[async_trait]
pub trait ReactAgent<T: Float>: Send + Sync {
    /// Observe the current environment/inputs
    async fn observe(&mut self, observations: Observations<T>) -> Result<()>;
    
    /// Think about the observations and plan actions
    async fn think(&mut self) -> Result<Thoughts>;
    
    /// Take action based on thoughts
    async fn act(&mut self) -> Result<Action<T>>;
    
    /// Generate explanation for the reasoning process
    fn explain(&self) -> ReasoningTrace;
}

/// Simple decision type for actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Truth,
    Deception,
    Uncertain,
}

/// Observations for the ReAct agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observations<T: Float> {
    pub id: uuid::Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub vision: Option<VisionObservation>,
    pub audio: Option<AudioObservation<T>>,
    pub text: Option<TextObservation>,
    pub physiological: Option<PhysiologicalObservation<T>>,
    pub context: ObservationContext,
}

impl<T: Float> Observations<T> {
    /// Check if observations contain a specific modality
    pub fn has_modality(&self, modality: ModalityType) -> bool {
        match modality {
            ModalityType::Vision => self.vision.is_some(),
            ModalityType::Audio => self.audio.is_some(),
            ModalityType::Text => self.text.is_some(),
            ModalityType::Physiological => self.physiological.is_some(),
        }
    }
}

/// Vision-based observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionObservation {
    pub face_detected: bool,
    pub micro_expressions: Vec<String>,
    pub gaze_patterns: Vec<String>,
    pub facial_landmarks: Vec<(f32, f32)>,
}

/// Audio-based observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioObservation<T: Float> {
    pub pitch_variations: Vec<T>,
    pub stress_indicators: Vec<String>,
    pub voice_quality: T,
    pub speaking_rate: T,
}

/// Text-based observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextObservation {
    pub content: String,
    pub linguistic_features: Vec<String>,
    pub sentiment_score: f64,
    pub deception_indicators: Vec<String>,
}

/// Physiological observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologicalObservation<T: Float> {
    pub stress_level: T,
    pub arousal_level: T,
    pub heart_rate_variability: T,
}

/// Context for observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationContext {
    pub environment: String,
    pub subject_id: Option<String>,
    pub session_id: Option<String>,
    pub interviewer_id: Option<String>,
}

/// Individual thought with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub id: uuid::Uuid,
    pub content: String,
    pub reasoning_type: ReasoningType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub confidence: f64,
}

/// Types of reasoning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningType {
    Observation,
    Pattern,
    Comparative,
    Causal,
    Hypothesis,
    Evidence,
    Synthesis,
}

/// Collection of thoughts from reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thoughts {
    pub thoughts: Vec<Thought>,
    pub generation_time: std::time::Duration,
}

impl Thoughts {
    /// Create new empty thoughts collection
    pub fn new() -> Self {
        Self {
            thoughts: Vec::new(),
            generation_time: std::time::Duration::from_millis(0),
        }
    }

    /// Add a thought to the collection
    pub fn add_thought(&mut self, content: String, reasoning_type: ReasoningType) {
        let thought = Thought {
            id: uuid::Uuid::new_v4(),
            content,
            reasoning_type,
            timestamp: chrono::Utc::now(),
            confidence: 0.7, // Default confidence
        };
        self.thoughts.push(thought);
    }

    /// Get average confidence across all thoughts
    pub fn average_confidence(&self) -> f64 {
        if self.thoughts.is_empty() {
            0.0
        } else {
            self.thoughts.iter().map(|t| t.confidence).sum::<f64>() / self.thoughts.len() as f64
        }
    }

    /// For compatibility with existing code
    pub fn confidence(&self) -> f64 {
        self.average_confidence()
    }

    /// For compatibility - get hypothesis-like thoughts
    pub fn hypotheses(&self) -> Vec<String> {
        self.thoughts.iter()
            .filter(|t| t.reasoning_type == ReasoningType::Hypothesis)
            .map(|t| t.content.clone())
            .collect()
    }
}

/// Actions that can be taken by the ReAct agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action<T: Float> {
    pub id: uuid::Uuid,
    pub action_type: ActionType,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    pub expected_outcome: String,
    pub confidence: T,
    pub explanation: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub decision: Option<Decision>,
}

/// Types of actions the agent can take
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    AnalyzeText,
    AnalyzeModality,
    RequestMoreData,
    MakeDecision,
    SeekHumanInput,
    UpdateModel,
}

/// Step in reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub id: uuid::Uuid,
    pub step_type: ReasoningStepType,
    pub input: String,
    pub output: String,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_time: std::time::Duration,
}

/// Types of reasoning steps
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningStepType {
    Observe,
    Think,
    Act,
    Explain,
}

/// Enhanced reasoning trace from the ReAct agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    pub steps: Vec<ReasoningStep>,
    pub total_time: std::time::Duration,
}

impl ReasoningTrace {
    /// Create new empty reasoning trace
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_time: std::time::Duration::from_millis(0),
        }
    }

    /// Add a reasoning step
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.total_time += step.execution_time;
        self.steps.push(step);
    }
}

/// Configuration types for ReAct agents
#[derive(Debug, Clone)]
pub struct DetectorConfig<T: Float> {
    pub memory_config: MemoryConfig<T>,
    pub reasoning_config: ReasoningConfig<T>,
    pub action_config: ActionConfig<T>,
}

impl<T: Float> Default for DetectorConfig<T> {
    fn default() -> Self {
        Self {
            memory_config: MemoryConfig::default(),
            reasoning_config: ReasoningConfig::default(),
            action_config: ActionConfig::default(),
        }
    }
}

/// Memory system configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig<T: Float> {
    pub short_term_capacity: usize,
    pub long_term_capacity: usize,
    pub episodic_capacity: usize,
    pub similarity_threshold: T,
    pub decay_rate: T,
}

impl<T: Float> Default for MemoryConfig<T> {
    fn default() -> Self {
        Self {
            short_term_capacity: 100,
            long_term_capacity: 10000,
            episodic_capacity: 1000,
            similarity_threshold: T::from(0.7).unwrap(),
            decay_rate: T::from(0.01).unwrap(),
        }
    }
}

/// Reasoning engine configuration
#[derive(Debug, Clone)]
pub struct ReasoningConfig<T: Float> {
    pub temperature: T,
    pub max_thoughts: usize,
    pub min_confidence: T,
    pub reasoning_timeout_ms: u64,
}

impl<T: Float> Default for ReasoningConfig<T> {
    fn default() -> Self {
        Self {
            temperature: T::from(0.7).unwrap(),
            max_thoughts: 20,
            min_confidence: T::from(0.3).unwrap(),
            reasoning_timeout_ms: 10000,
        }
    }
}

/// Action engine configuration 
#[derive(Debug, Clone)]
pub struct ActionConfig<T: Float> {
    pub default_strategy: String,
    pub max_actions: usize,
    pub min_confidence: T,
    pub action_timeout_ms: u64,
    pub enable_validation: bool,
    pub max_history_size: usize,
    pub strategy_params: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
}

impl<T: Float> Default for ActionConfig<T> {
    fn default() -> Self {
        Self {
            default_strategy: "multimodal_weighted".to_string(),
            max_actions: 10,
            min_confidence: T::from(0.3).unwrap(),
            action_timeout_ms: 5000,
            enable_validation: true,
            max_history_size: 1000,
            strategy_params: std::collections::HashMap::new(),
        }
    }
}

/// Neuro-symbolic reasoning trait
#[async_trait]
pub trait NeuroSymbolicReasoner<T: Float>: Send + Sync {
    /// Apply symbolic rules to neural network output
    async fn apply_rules(&self, neural_output: &NeuralOutput<T>) -> Result<SymbolicOutput>;
    
    /// Merge neural and symbolic outputs into a final decision
    async fn merge(&self, neural: &NeuralOutput<T>, symbolic: &SymbolicOutput) -> Result<Decision<T>>;
}

/// Output from neural network processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralOutput<T: Float> {
    pub raw_scores: Vec<T>,
    pub probabilities: Vec<T>,
    pub features: Vec<Feature<T>>,
    pub confidence: T,
}

/// Output from symbolic reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicOutput {
    pub rules_applied: Vec<String>,
    pub conclusions: Vec<String>,
    pub confidence: f64,
    pub explanations: Vec<String>,
}


