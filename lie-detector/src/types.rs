//! Core type definitions for the Veritas Nexus system.
//!
//! This module contains the fundamental types, traits, and interfaces used
//! throughout the lie detection system. These types provide the foundation
//! for modality processing, fusion strategies, and reasoning engines.
//!
//! # Type Categories
//!
//! - **Numeric Types**: Type aliases for floating point operations
//! - **Time Types**: Timestamp and duration representations
//! - **Score Types**: Confidence and deception scoring types
//! - **Trait Definitions**: Core traits for analyzers and strategies
//! - **Input/Output Types**: Data structures for system interfaces
//! - **Metadata Types**: System information and diagnostics
//!
//! # Examples
//!
//! ```rust
//! use veritas_nexus::types::*;
//!
//! // Working with confidence scores
//! let confidence = Confidence::new(0.85)?;
//! assert!(confidence.is_high());
//!
//! // Creating timestamps
//! let timestamp = Timestamp::now();
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, VeritasError};

// ================================================================================================
// NUMERIC TYPE ALIASES
// ================================================================================================

/// Single-precision floating point number type alias
pub type Float32 = f32;

/// Double-precision floating point number type alias  
pub type Float64 = f64;

/// Default floating point type used throughout the system
pub type DefaultFloat = Float64;

/// Matrix type for numerical computations
pub type Matrix<T> = DMatrix<T>;

/// Vector type for numerical computations
pub type Vector<T> = DVector<T>;

// ================================================================================================
// TIME AND DURATION TYPES
// ================================================================================================

/// Timestamp representation with nanosecond precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp {
    /// Nanoseconds since Unix epoch
    nanos: u64,
}

impl Timestamp {
    /// Creates a new timestamp from the current system time
    pub fn now() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0));
        Self {
            nanos: now.as_nanos() as u64,
        }
    }

    /// Creates a timestamp from nanoseconds since Unix epoch
    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Creates a timestamp from milliseconds since Unix epoch
    pub fn from_millis(millis: u64) -> Self {
        Self {
            nanos: millis * 1_000_000,
        }
    }

    /// Creates a timestamp from seconds since Unix epoch
    pub fn from_secs(secs: u64) -> Self {
        Self {
            nanos: secs * 1_000_000_000,
        }
    }

    /// Returns nanoseconds since Unix epoch
    pub fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Returns milliseconds since Unix epoch
    pub fn as_millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Returns seconds since Unix epoch
    pub fn as_secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Converts to chrono DateTime
    pub fn to_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_nanos(self.nanos as i64)
    }

    /// Calculates duration since another timestamp
    pub fn duration_since(&self, other: Timestamp) -> Duration {
        if self.nanos >= other.nanos {
            Duration::from_nanos(self.nanos - other.nanos)
        } else {
            Duration::from_nanos(0)
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_datetime().format("%Y-%m-%d %H:%M:%S%.3f UTC"))
    }
}

// ================================================================================================
// SCORE AND CONFIDENCE TYPES
// ================================================================================================

/// Confidence score with validation and utility methods
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence {
    /// Confidence value between 0.0 and 1.0
    value: f64,
}

impl Confidence {
    /// Creates a new confidence score
    ///
    /// # Errors
    /// Returns error if value is not between 0.0 and 1.0
    pub fn new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(VeritasError::invalid_input(
                format!("Confidence must be between 0.0 and 1.0, got {}", value),
                "confidence",
            ));
        }
        Ok(Self { value })
    }

    /// Creates a confidence score without validation (for internal use)
    pub(crate) fn new_unchecked(value: f64) -> Self {
        Self { value }
    }

    /// Returns the confidence value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Returns true if confidence is high (>= 0.8)
    pub fn is_high(&self) -> bool {
        self.value >= 0.8
    }

    /// Returns true if confidence is medium (0.5 <= confidence < 0.8)
    pub fn is_medium(&self) -> bool {
        (0.5..0.8).contains(&self.value)
    }

    /// Returns true if confidence is low (< 0.5)
    pub fn is_low(&self) -> bool {
        self.value < 0.5
    }

    /// Returns confidence as percentage
    pub fn as_percentage(&self) -> f64 {
        self.value * 100.0
    }

    /// Combines with another confidence using weighted average
    pub fn combine_with(&self, other: Confidence, weight: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(VeritasError::invalid_input(
                "Weight must be between 0.0 and 1.0",
                "weight",
            ));
        }
        let combined = self.value * weight + other.value * (1.0 - weight);
        Ok(Self::new_unchecked(combined))
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.as_percentage())
    }
}

/// Deception probability score
pub type DeceptionScore = Confidence;

// ================================================================================================
// CORE TRAIT DEFINITIONS
// ================================================================================================

/// Core trait for all modality analyzers
///
/// This trait defines the interface that all modality processors must implement.
/// It supports both synchronous and asynchronous processing modes.
pub trait ModalityAnalyzer<T: Float>: Send + Sync {
    /// Input data type for this analyzer
    type Input;
    /// Output type containing deception analysis
    type Output: DeceptionAnalysis<T>;
    /// Configuration type for this analyzer
    type Config: Clone + Send + Sync;

    /// Analyzes input data and returns deception analysis
    ///
    /// # Errors
    /// Returns error if analysis fails due to invalid input or processing errors
    fn analyze(&self, input: &Self::Input) -> Result<Self::Output>;

    /// Returns the current confidence in this analyzer's capabilities
    fn confidence(&self) -> Confidence;

    /// Generates explanation for the most recent analysis
    fn explain(&self) -> ExplanationTrace;

    /// Returns analyzer metadata
    fn metadata(&self) -> AnalyzerMetadata;

    /// Validates input data before processing
    fn validate_input(&self, input: &Self::Input) -> Result<()>;

    /// Returns supported input formats or constraints
    fn supported_formats(&self) -> Vec<String>;
}

/// Trait for async modality analyzers
#[async_trait::async_trait]
pub trait AsyncModalityAnalyzer<T: Float>: Send + Sync {
    /// Input data type for this analyzer
    type Input: Send + Sync;
    /// Output type containing deception analysis
    type Output: DeceptionAnalysis<T> + Send;
    /// Configuration type for this analyzer
    type Config: Clone + Send + Sync;

    /// Analyzes input data asynchronously
    async fn analyze_async(&self, input: &Self::Input) -> Result<Self::Output>;

    /// Returns the current confidence in this analyzer's capabilities
    async fn confidence_async(&self) -> Confidence;

    /// Generates explanation for the most recent analysis
    async fn explain_async(&self) -> ExplanationTrace;
}

/// Trait for deception analysis results
pub trait DeceptionAnalysis<T: Float>: Send + Sync + fmt::Debug {
    /// Returns the deception score
    fn deception_score(&self) -> DeceptionScore;

    /// Returns confidence in this analysis
    fn confidence(&self) -> Confidence;

    /// Returns detailed features extracted during analysis
    fn features(&self) -> &HashMap<String, T>;

    /// Returns temporal information if available
    fn temporal_info(&self) -> Option<TemporalInfo>;

    /// Returns analysis metadata
    fn metadata(&self) -> &AnalysisMetadata;

    /// Converts to a standardized result format
    fn to_standard_result(&self) -> StandardAnalysisResult<T>;
}

/// Trait for fusion strategies that combine multiple modality results
pub trait FusionStrategy<T: Float>: Send + Sync {
    /// Input type - typically a collection of analysis results
    type Input;
    /// Output type - fused analysis result
    type Output: DeceptionAnalysis<T>;

    /// Fuses multiple modality results into a single decision
    fn fuse(&self, inputs: &Self::Input) -> Result<Self::Output>;

    /// Returns current fusion weights for each modality
    fn weights(&self) -> &HashMap<String, T>;

    /// Updates fusion weights based on feedback
    fn update_weights(&mut self, feedback: &FusionFeedback<T>) -> Result<()>;

    /// Returns fusion strategy metadata
    fn strategy_info(&self) -> FusionStrategyInfo;
}

/// Trait for ReAct (Reasoning and Acting) agents
pub trait ReactAgent<T: Float>: Send + Sync {
    /// Observation data type
    type Observation;
    /// Thought/reasoning type
    type Thought;
    /// Action type
    type Action;

    /// Process new observations
    fn observe(&mut self, observations: Self::Observation) -> Result<()>;

    /// Generate thoughts/reasoning based on current state
    fn think(&mut self) -> Result<Self::Thought>;

    /// Take action based on reasoning
    fn act(&mut self) -> Result<Self::Action>;

    /// Generate explanation of reasoning process
    fn explain(&self) -> ReasoningTrace;

    /// Reset agent state
    fn reset(&mut self);

    /// Returns agent configuration and capabilities
    fn agent_info(&self) -> AgentInfo;
}

// ================================================================================================
// DATA STRUCTURES
// ================================================================================================

/// Standard analysis result format used across all modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardAnalysisResult<T: Float> {
    /// Unique identifier for this analysis
    pub id: Uuid,
    /// Timestamp when analysis was performed
    pub timestamp: Timestamp,
    /// Deception score
    pub deception_score: DeceptionScore,
    /// Confidence in the analysis
    pub confidence: Confidence,
    /// Extracted features
    pub features: HashMap<String, T>,
    /// Modality that produced this result
    pub modality: String,
    /// Processing duration
    pub duration: Duration,
    /// Additional metadata
    pub metadata: AnalysisMetadata,
}

/// Metadata associated with an analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Version of the analyzer used
    pub analyzer_version: String,
    /// Model version if applicable
    pub model_version: Option<String>,
    /// Processing parameters used
    pub parameters: HashMap<String, String>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Warnings or notes
    pub warnings: Vec<String>,
}

/// Metadata for modality analyzers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerMetadata {
    /// Analyzer name
    pub name: String,
    /// Version
    pub version: String,
    /// Supported input types
    pub input_types: Vec<String>,
    /// Performance characteristics
    pub performance: PerformanceInfo,
    /// Capabilities and limitations
    pub capabilities: Vec<String>,
    /// Required resources
    pub requirements: ResourceRequirements,
}

/// Performance characteristics of an analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInfo {
    /// Typical processing time per input
    pub avg_processing_time: Duration,
    /// Memory usage estimate
    pub memory_usage_mb: f64,
    /// Throughput (inputs per second)
    pub throughput: f64,
    /// Accuracy metrics if available
    pub accuracy: Option<f64>,
}

/// Resource requirements for an analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum memory required (MB)
    pub min_memory_mb: f64,
    /// Recommended memory (MB)
    pub recommended_memory_mb: f64,
    /// CPU cores recommended
    pub cpu_cores: Option<u32>,
    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,
    /// External dependencies
    pub dependencies: Vec<String>,
}

/// GPU requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum VRAM (GB)
    pub min_vram_gb: f64,
    /// Compute capability required
    pub compute_capability: Option<String>,
    /// Supported GPU types
    pub supported_types: Vec<String>,
}

/// Temporal information for time-series analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    /// Start time of the analyzed segment
    pub start_time: Timestamp,
    /// End time of the analyzed segment
    pub end_time: Timestamp,
    /// Sample rate if applicable
    pub sample_rate: Option<f64>,
    /// Frame rate for video data
    pub frame_rate: Option<f64>,
    /// Temporal features
    pub temporal_features: HashMap<String, f64>,
}

/// Explanation trace for analysis transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationTrace {
    /// Step-by-step explanations
    pub steps: Vec<ExplanationStep>,
    /// Overall reasoning summary
    pub summary: String,
    /// Confidence in the explanation
    pub explanation_confidence: Confidence,
    /// Key factors that influenced the decision
    pub key_factors: Vec<String>,
}

/// Individual step in an explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationStep {
    /// Step number
    pub step: u32,
    /// Description of what happened
    pub description: String,
    /// Input data for this step
    pub input: String,
    /// Output or result of this step
    pub output: String,
    /// Confidence in this step
    pub confidence: Confidence,
    /// Time taken for this step
    pub duration: Duration,
}


/// Fusion feedback for updating strategy weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionFeedback<T: Float> {
    /// Ground truth label if available
    pub ground_truth: Option<bool>,
    /// User feedback on the decision
    pub user_feedback: Option<UserFeedback>,
    /// Performance metrics
    pub metrics: HashMap<String, T>,
    /// Timestamp of feedback
    pub timestamp: Timestamp,
}

/// User feedback on system decisions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum UserFeedback {
    /// User agrees with the decision
    Agree,
    /// User disagrees with the decision
    Disagree,
    /// User is uncertain about the decision
    #[default]
    Uncertain,
    /// User provides custom feedback
    Custom(String),
}

/// Information about fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStrategyInfo {
    /// Strategy name
    pub name: String,
    /// Strategy version
    pub version: String,
    /// Description of the strategy
    pub description: String,
    /// Supported modalities
    pub supported_modalities: Vec<String>,
    /// Strategy parameters
    pub parameters: HashMap<String, String>,
}

/// Information about ReAct agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent name
    pub name: String,
    /// Agent version
    pub version: String,
    /// Agent capabilities
    pub capabilities: Vec<String>,
    /// Supported observation types
    pub observation_types: Vec<String>,
    /// Supported action types
    pub action_types: Vec<String>,
}

// ================================================================================================
// UTILITY TYPES
// ================================================================================================

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult<T> {
    /// Individual results
    pub results: Vec<T>,
    /// Overall batch statistics
    pub statistics: BatchStatistics,
    /// Processing metadata
    pub metadata: BatchMetadata,
}

/// Statistics for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// Total items processed
    pub total_items: usize,
    /// Successfully processed items
    pub successful_items: usize,
    /// Failed items
    pub failed_items: usize,
    /// Average processing time per item
    pub avg_time_per_item: Duration,
    /// Total processing time
    pub total_time: Duration,
}

/// Metadata for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetadata {
    /// Batch identifier
    pub batch_id: Uuid,
    /// Timestamp when batch started
    pub start_time: Timestamp,
    /// Timestamp when batch completed
    pub end_time: Timestamp,
    /// Processing configuration used
    pub config: HashMap<String, String>,
}

/// System health and status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// Overall system health
    pub health: HealthStatus,
    /// Active components
    pub active_components: Vec<ComponentStatus>,
    /// Resource utilization
    pub resource_usage: ResourceUsage,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// System uptime
    pub uptime: Duration,
    /// Last update timestamp
    pub last_updated: Timestamp,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HealthStatus {
    /// System is operating normally
    Healthy,
    /// System has minor issues but is functional
    Degraded,
    /// System has significant issues
    Unhealthy,
    /// System status is unknown
    #[default]
    Unknown,
}

/// Status of individual system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    /// Component name
    pub name: String,
    /// Component health
    pub health: HealthStatus,
    /// Last error if any
    pub last_error: Option<String>,
    /// Component metrics
    pub metrics: HashMap<String, f64>,
    /// Last health check timestamp
    pub last_check: Timestamp,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// GPU utilization if available
    pub gpu_percent: Option<f64>,
    /// GPU memory usage if available
    pub gpu_memory_mb: Option<f64>,
    /// Disk usage in MB
    pub disk_usage_mb: f64,
    /// Network throughput in MB/s
    pub network_mbps: f64,
}

// ================================================================================================
// CONSTANTS AND DEFAULTS
// ================================================================================================

/// Default confidence threshold for high-confidence decisions
pub const DEFAULT_HIGH_CONFIDENCE_THRESHOLD: f64 = 0.8;

/// Default confidence threshold for low-confidence decisions
pub const DEFAULT_LOW_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// Default timeout for analysis operations (seconds)
pub const DEFAULT_ANALYSIS_TIMEOUT_SECS: u64 = 30;

/// Default batch size for processing
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Default memory pool size (MB)
pub const DEFAULT_MEMORY_POOL_SIZE_MB: usize = 512;

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

/// Creates a new UUID v4
pub fn new_id() -> Uuid {
    Uuid::new_v4()
}

/// Creates a timestamp from the current time
pub fn now() -> Timestamp {
    Timestamp::now()
}

/// Validates that a value is within a specified range
pub fn validate_range<T: PartialOrd + fmt::Display>(
    value: T,
    min: T,
    max: T,
    param_name: &str,
) -> Result<()> {
    if value < min || value > max {
        return Err(VeritasError::invalid_input(
            format!("{} must be between {} and {}, got {}", param_name, min, max, value),
            param_name,
        ));
    }
    Ok(())
}

/// Validates that a collection is not empty
pub fn validate_not_empty<T>(collection: &[T], param_name: &str) -> Result<()> {
    if collection.is_empty() {
        return Err(VeritasError::invalid_input(
            format!("{} cannot be empty", param_name),
            param_name,
        ));
    }
    Ok(())
}

// ================================================================================================
// REACT AGENT TYPES
// ================================================================================================

/// Multi-modal observations for ReAct agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observations<T: Float> {
    /// Unique observation ID
    pub id: Uuid,
    /// Timestamp when observations were made
    pub timestamp: DateTime<Utc>,
    /// Vision/visual observations
    pub vision: Option<VisionObservation>,
    /// Audio observations
    pub audio: Option<AudioObservation<T>>,
    /// Text observations
    pub text: Option<TextObservation>,
    /// Physiological observations
    pub physiological: Option<PhysiologicalObservation<T>>,
    /// Context information
    pub context: ObservationContext,
}

impl<T: Float> Observations<T> {
    /// Check if a specific modality is present
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
    /// Whether a face was detected
    pub face_detected: bool,
    /// Micro-expressions detected
    pub micro_expressions: Vec<String>,
    /// Gaze patterns
    pub gaze_patterns: Vec<String>,
    /// Facial landmarks
    pub facial_landmarks: Vec<(f64, f64)>,
}

/// Audio-based observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioObservation<T: Float> {
    /// Pitch variations
    pub pitch_variations: Vec<T>,
    /// Stress indicators in voice
    pub stress_indicators: Vec<String>,
    /// Overall voice quality score
    pub voice_quality: T,
    /// Speaking rate (words per minute)
    pub speaking_rate: T,
}

/// Text-based observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextObservation {
    /// Raw text content
    pub content: String,
    /// Linguistic features extracted
    pub linguistic_features: Vec<String>,
    /// Sentiment score (-1.0 to 1.0)
    pub sentiment_score: f64,
    /// Deception indicators found
    pub deception_indicators: Vec<String>,
}

/// Physiological observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologicalObservation<T: Float> {
    /// Stress level (0.0 to 1.0)
    pub stress_level: T,
    /// Arousal level (0.0 to 1.0)
    pub arousal_level: T,
    /// Heart rate variability
    pub heart_rate_variability: T,
}

/// Context for observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationContext {
    /// Environment description
    pub environment: String,
    /// Subject identifier
    pub subject_id: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Interviewer identifier
    pub interviewer_id: Option<String>,
}

/// Structured thoughts generated by reasoning engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thoughts {
    /// Individual thought entries
    pub thoughts: Vec<Thought>,
    /// Generated hypotheses
    pub hypotheses: Vec<String>,
    /// Overall confidence in thoughts
    pub confidence: f64,
    /// Time taken to generate thoughts
    pub generation_time: Duration,
}

impl Thoughts {
    /// Create new empty thoughts
    pub fn new() -> Self {
        Self {
            thoughts: Vec::new(),
            hypotheses: Vec::new(),
            confidence: 0.0,
            generation_time: Duration::from_millis(0),
        }
    }

    /// Add a thought
    pub fn add_thought(&mut self, content: String, reasoning_type: ReasoningType) {
        let thought = Thought {
            id: Uuid::new_v4(),
            content,
            reasoning_type,
            timestamp: Utc::now(),
            confidence: 0.8, // Default confidence
        };
        self.thoughts.push(thought);
    }
}

/// Individual thought entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// Unique thought ID
    pub id: Uuid,
    /// Thought content
    pub content: String,
    /// Type of reasoning used
    pub reasoning_type: ReasoningType,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Confidence in this thought
    pub confidence: f64,
}

/// Types of reasoning
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ReasoningType {
    /// Observational reasoning
    #[default]
    Observation,
    /// Pattern recognition
    Pattern,
    /// Comparative analysis
    Comparative,
    /// Causal reasoning
    Causal,
    /// Hypothesis generation
    Hypothesis,
    /// Evidence evaluation
    Evidence,
    /// Synthesis
    Synthesis,
}

/// Action that can be taken by a ReAct agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action<T: Float> {
    /// Action ID
    pub id: Uuid,
    /// Type of action
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Expected outcome
    pub expected_outcome: String,
    /// Confidence in this action
    pub confidence: T,
    /// Explanation for taking this action
    pub explanation: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Decision made (if applicable)
    pub decision: Option<Decision>,
}

/// Types of actions agents can take
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ActionType {
    /// Make a deception detection decision
    #[default]
    MakeDecision,
    /// Analyze a specific modality
    AnalyzeModality,
    /// Request more data
    RequestMoreData,
    /// Update internal model
    UpdateModel,
    /// Explain reasoning
    ExplainReasoning,
}

/// Decision outcomes for deception detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Decision {
    /// Truth detected
    Truth,
    /// Deception detected
    Deception,
    /// Uncertain/unclear
    #[default]
    Uncertain,
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Decision::Truth => write!(f, "Truth"),
            Decision::Deception => write!(f, "Deception"),
            Decision::Uncertain => write!(f, "Uncertain"),
        }
    }
}

/// Modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ModalityType {
    /// Visual/vision modality
    Vision,
    /// Audio modality
    Audio,
    /// Text modality
    #[default]
    Text,
    /// Physiological modality
    Physiological,
}

/// Feature extracted from data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature<T: Float> {
    /// Feature name
    pub name: String,
    /// Feature value
    pub value: T,
    /// Feature weight/importance
    pub weight: T,
    /// Feature type
    pub feature_type: String,
}

/// Feature vector containing multiple features
pub type FeatureVector<T> = Vec<Feature<T>>;

/// Reasoning trace for explainability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Individual reasoning steps
    pub steps: Vec<ReasoningStep>,
    /// Total time for reasoning
    pub total_time: Duration,
    /// Overall confidence
    pub confidence: f64,
    /// Final summary
    pub summary: String,
}

impl ReasoningTrace {
    /// Create new empty reasoning trace
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_time: Duration::from_millis(0),
            confidence: 0.0,
            summary: String::new(),
        }
    }

    /// Add a reasoning step
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.total_time += step.execution_time;
        self.steps.push(step);
    }
}

/// Individual reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step ID
    pub id: Uuid,
    /// Type of reasoning step
    pub step_type: ReasoningStepType,
    /// Input to this step
    pub input: String,
    /// Output from this step
    pub output: String,
    /// Confidence in this step
    pub confidence: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Execution time
    pub execution_time: Duration,
}

/// Types of reasoning steps
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ReasoningStepType {
    /// Observation step
    #[default]
    Observe,
    /// Thinking/reasoning step
    Think,
    /// Action step
    Act,
    /// Explanation step
    Explain,
}

// ================================================================================================
// CONFIGURATION TYPES
// ================================================================================================

/// Configuration for the detector system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig<T: Float> {
    /// Memory system configuration
    pub memory_config: MemoryConfig<T>,
    /// Reasoning engine configuration
    pub reasoning_config: ReasoningConfig<T>,
    /// Action engine configuration
    pub action_config: ActionConfig<T>,
    /// General system configuration
    pub system_config: SystemConfig<T>,
}

impl<T: Float> Default for DetectorConfig<T> {
    fn default() -> Self {
        Self {
            memory_config: MemoryConfig::default(),
            reasoning_config: ReasoningConfig::default(),
            action_config: ActionConfig::default(),
            system_config: SystemConfig::default(),
        }
    }
}

/// Configuration for memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig<T: Float> {
    /// Short-term memory capacity
    pub short_term_capacity: usize,
    /// Long-term memory capacity
    pub long_term_capacity: usize,
    /// Episodic memory capacity
    pub episodic_capacity: usize,
    /// Similarity threshold for retrieval
    pub similarity_threshold: T,
    /// Decay rate for temporal forgetting
    pub decay_rate: T,
    /// Consolidation threshold
    pub consolidation_threshold: T,
}

impl<T: Float> Default for MemoryConfig<T> {
    fn default() -> Self {
        Self {
            short_term_capacity: 100,
            long_term_capacity: 1000,
            episodic_capacity: 500,
            similarity_threshold: T::from(0.7).unwrap(),
            decay_rate: T::from(0.01).unwrap(),
            consolidation_threshold: T::from(0.8).unwrap(),
        }
    }
}

/// Configuration for reasoning engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig<T: Float> {
    /// Temperature for thought generation
    pub temperature: T,
    /// Maximum reasoning depth
    pub max_depth: usize,
    /// Confidence threshold for decisions
    pub confidence_threshold: T,
    /// Enable causal reasoning
    pub enable_causal_reasoning: bool,
    /// Enable temporal reasoning
    pub enable_temporal_reasoning: bool,
}

impl<T: Float> Default for ReasoningConfig<T> {
    fn default() -> Self {
        Self {
            temperature: T::from(0.7).unwrap(),
            max_depth: 10,
            confidence_threshold: T::from(0.5).unwrap(),
            enable_causal_reasoning: true,
            enable_temporal_reasoning: true,
        }
    }
}

/// Configuration for action engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConfig<T: Float> {
    /// Default action selection strategy
    pub default_strategy: String,
    /// Maximum actions to consider
    pub max_actions: usize,
    /// Minimum confidence for actions
    pub min_confidence: T,
    /// Action timeout
    pub action_timeout_ms: u64,
    /// Enable action validation
    pub enable_validation: bool,
}

impl<T: Float> Default for ActionConfig<T> {
    fn default() -> Self {
        Self {
            default_strategy: "multimodal_weighted".to_string(),
            max_actions: 10,
            min_confidence: T::from(0.3).unwrap(),
            action_timeout_ms: 5000,
            enable_validation: true,
        }
    }
}

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig<T: Float> {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Memory pool size
    pub memory_pool_mb: usize,
    /// Log level
    pub log_level: String,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval_ms: u64,
}

impl<T: Float> Default for SystemConfig<T> {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            worker_threads: 4,
            memory_pool_mb: 512,
            log_level: "info".to_string(),
            enable_metrics: true,
            metrics_interval_ms: 1000,
        }
    }
}

/// Usage statistics for memory systems
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Short-term memory utilization (0.0 to 1.0)
    pub short_term_utilization: f64,
    /// Long-term memory utilization (0.0 to 1.0)
    pub long_term_utilization: f64,
    /// Episodic memory utilization (0.0 to 1.0)
    pub episodic_utilization: f64,
    /// Total memory entries
    pub total_entries: usize,
}

// ================================================================================================
// FUSION-RELATED TYPES
// ================================================================================================

/// Attention weights for fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights<T: Float> {
    /// Weights for each modality
    pub weights: std::collections::HashMap<ModalityType, T>,
    /// Overall attention score
    pub attention_score: T,
}

impl<T: Float> Default for AttentionWeights<T> {
    fn default() -> Self {
        Self {
            weights: std::collections::HashMap::new(),
            attention_score: T::zero(),
        }
    }
}

/// Combined features from multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedFeatures<T: Float> {
    /// Features by modality
    pub modality_features: std::collections::HashMap<ModalityType, Vec<T>>,
    /// Combined feature vector
    pub combined_vector: Vec<T>,
}

impl<T: Float> Default for CombinedFeatures<T> {
    fn default() -> Self {
        Self {
            modality_features: std::collections::HashMap::new(),
            combined_vector: Vec::new(),
        }
    }
}

/// Fused decision from multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedDecision<T: Float> {
    /// Final decision
    pub decision: Decision,
    /// Decision confidence
    pub confidence: T,
    /// Contribution from each modality
    pub modality_contributions: std::collections::HashMap<ModalityType, T>,
    /// Fusion method used
    pub fusion_method: String,
}

impl<T: Float> Default for FusedDecision<T> {
    fn default() -> Self {
        Self {
            decision: Decision::default(),
            confidence: T::zero(),
            modality_contributions: std::collections::HashMap::new(),
            fusion_method: "default".to_string(),
        }
    }
}

/// Result from voting-based fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingResult<T: Float> {
    /// Vote counts for each decision
    pub vote_counts: std::collections::HashMap<Decision, usize>,
    /// Weighted votes if applicable
    pub weighted_votes: std::collections::HashMap<Decision, T>,
    /// Final decision based on voting
    pub final_decision: Decision,
    /// Confidence in the voting result
    pub confidence: T,
}

impl<T: Float> Default for VotingResult<T> {
    fn default() -> Self {
        Self {
            vote_counts: std::collections::HashMap::new(),
            weighted_votes: std::collections::HashMap::new(),
            final_decision: Decision::default(),
            confidence: T::zero(),
        }
    }
}

/// Metadata for fusion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    /// Fusion strategy used
    pub strategy: String,
    /// Modalities involved
    pub modalities: Vec<ModalityType>,
    /// Quality of input data
    pub input_quality: std::collections::HashMap<ModalityType, f64>,
    /// Processing parameters
    pub parameters: std::collections::HashMap<String, String>,
}

impl Default for FusionMetadata {
    fn default() -> Self {
        Self {
            strategy: "default".to_string(),
            modalities: Vec::new(),
            input_quality: std::collections::HashMap::new(),
            parameters: std::collections::HashMap::new(),
        }
    }
}

/// Processing timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTiming {
    /// Start time
    pub start_time: std::time::SystemTime,
    /// End time
    pub end_time: std::time::SystemTime,
    /// Duration
    pub duration: Duration,
    /// Breakdown by stage
    pub stage_timings: std::collections::HashMap<String, Duration>,
}

impl Default for ProcessingTiming {
    fn default() -> Self {
        let now = std::time::SystemTime::now();
        Self {
            start_time: now,
            end_time: now,
            duration: Duration::from_millis(0),
            stage_timings: std::collections::HashMap::new(),
        }
    }
}

/// Quality metrics for analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Data quality score
    pub data_quality: f64,
    /// Model confidence
    pub model_confidence: f64,
    /// Completeness of analysis
    pub completeness: f64,
    /// Individual metric scores
    pub metrics: std::collections::HashMap<String, f64>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            data_quality: 0.0,
            model_confidence: 0.0,
            completeness: 0.0,
            metrics: std::collections::HashMap::new(),
        }
    }
}

/// Performance metrics for analysis and processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Processing time metrics
    pub timing: ProcessingTiming,
    /// Memory usage metrics
    pub memory: MemoryUsage,
    /// Throughput metrics
    pub throughput: f64,
    /// Accuracy metrics
    pub accuracy: f64,
    /// Additional custom metrics
    pub custom_metrics: std::collections::HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            timing: ProcessingTiming::default(),
            memory: MemoryUsage::default(),
            throughput: 0.0,
            accuracy: 0.0,
            custom_metrics: std::collections::HashMap::new(),
        }
    }
}

/// Cache entry for storing computed results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cached value
    pub value: T,
    /// Timestamp when cached
    pub cached_at: std::time::SystemTime,
    /// Time-to-live for cache entry
    pub ttl: Duration,
    /// Access count
    pub access_count: usize,
    /// Last access time
    pub last_accessed: std::time::SystemTime,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(value: T, ttl: Duration) -> Self {
        let now = std::time::SystemTime::now();
        Self {
            value,
            cached_at: now,
            ttl,
            access_count: 0,
            last_accessed: now,
        }
    }
    
    /// Check if entry is expired
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed().unwrap_or(Duration::MAX) > self.ttl
    }
    
    /// Update access tracking
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now();
    }
}

impl<T: Clone> CacheEntry<T> {
    /// Get value and update access tracking
    pub fn get(&mut self) -> T {
        self.mark_accessed();
        self.value.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_creation() {
        let conf = Confidence::new(0.75).unwrap();
        assert_eq!(conf.value(), 0.75);
        assert!(conf.is_medium());
        assert!(!conf.is_high());
        assert!(!conf.is_low());
    }

    #[test]
    fn test_confidence_validation() {
        assert!(Confidence::new(-0.1).is_err());
        assert!(Confidence::new(1.1).is_err());
        assert!(Confidence::new(0.0).is_ok());
        assert!(Confidence::new(1.0).is_ok());
    }

    #[test]
    fn test_timestamp_creation() {
        let ts1 = Timestamp::now();
        let ts2 = Timestamp::from_secs(1000);
        assert!(ts1.as_nanos() > ts2.as_nanos());
    }

    #[test]
    fn test_timestamp_duration() {
        let ts1 = Timestamp::from_secs(1000);
        let ts2 = Timestamp::from_secs(1010);
        let duration = ts2.duration_since(ts1);
        assert_eq!(duration.as_secs(), 10);
    }

    #[test]
    fn test_validation_helpers() {
        assert!(validate_range(5, 0, 10, "test").is_ok());
        assert!(validate_range(-1, 0, 10, "test").is_err());
        
        let empty_vec: Vec<i32> = vec![];
        assert!(validate_not_empty(&empty_vec, "test").is_err());
        assert!(validate_not_empty(&[1, 2, 3], "test").is_ok());
    }

    #[test]
    fn test_confidence_combination() {
        let conf1 = Confidence::new(0.8).unwrap();
        let conf2 = Confidence::new(0.6).unwrap();
        let combined = conf1.combine_with(conf2, 0.7).unwrap();
        assert!((combined.value() - 0.74).abs() < 0.01);
    }
}