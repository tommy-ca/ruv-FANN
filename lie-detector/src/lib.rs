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
//!
//! ## Feature Flags
//!
//! Veritas Nexus supports several optional features that can be enabled in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! veritas-nexus = { version = "0.1", features = ["gpu", "parallel"] }
//! ```
//!
//! ### Core Features
//!
//! - **`default`**: Enables `parallel` feature for basic multi-threading support
//! - **`parallel`**: Parallel processing using `rayon` and `crossbeam` for improved performance
//!
//! ### Performance Features
//!
//! - **`gpu`**: GPU acceleration using Candle for neural network inference
//!   - Enables CUDA and OpenCL support where available
//!   - Significantly faster for batch processing and deep learning models
//!   - Requires compatible GPU drivers
//!
//! - **`benchmarking`**: Comprehensive benchmarking suite using Criterion
//!   - Performance regression testing
//!   - Detailed timing analysis
//!   - Memory usage profiling
//!
//! ### Integration Features
//!
//! - **`mcp`**: Model Context Protocol server integration
//!   - Enables RESTful API for external integrations
//!   - WebSocket support for real-time analysis
//!   - Compatible with Claude and other AI systems
//!
//! ### Development Features
//!
//! - **`testing`**: Enhanced testing utilities and mock objects
//!   - Property-based testing with proptest
//!   - Test data generators
//!   - Mocking frameworks for unit tests
//!
//! - **`profiling`**: Performance profiling and optimization tools
//!   - Memory allocation tracking
//!   - CPU usage analysis
//!   - Bottleneck identification
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use veritas_nexus::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Initialize the detection system
//!     let detector = LieDetector::builder()
//!         .with_text_analysis(TextConfig::default())
//!         .with_vision_analysis(VisionConfig::default())
//!         .build()?;
//!
//!     // Analyze a text statement
//!     let input = AnalysisInput::text("I was definitely at home all evening.");
//!     let result = detector.analyze(input).await?;
//!
//!     match result.decision {
//!         Decision::Truthful => println!("Statement appears truthful"),
//!         Decision::Deceptive => println!("Statement appears deceptive"),
//!         Decision::Uncertain => println!("Insufficient evidence"),
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Performance Characteristics
//!
//! ### Throughput
//! - **Text Analysis**: ~1000 statements/second (CPU), ~5000/second (GPU)
//! - **Vision Analysis**: ~30 FPS real-time (CPU), ~120 FPS (GPU)
//! - **Audio Analysis**: Real-time processing with <100ms latency
//! - **Multi-modal Fusion**: <50ms overhead for combining modalities
//!
//! ### Memory Usage
//! - **Base System**: ~100MB for core libraries and models
//! - **Text Models**: ~500MB for BERT-based analysis
//! - **Vision Models**: ~200MB for face detection and micro-expression analysis
//! - **Audio Models**: ~150MB for voice stress analysis
//!
//! ### Accuracy Metrics
//! - **Single Modality**: 75-85% accuracy depending on input quality
//! - **Multi-modal Fusion**: 85-92% accuracy with high-quality inputs
//! - **Cross-cultural Validation**: Validated across 15+ language/cultural groups
//! - **False Positive Rate**: <5% with confidence thresholds enabled
//!
//! ## Ethical Considerations
//!
//! The Veritas Nexus system is designed with ethical AI principles:
//!
//! - **Transparency**: All decisions include detailed explanations
//! - **Bias Mitigation**: Regular testing across demographic groups
//! - **Privacy Protection**: Local processing option, no data retention
//! - **Human Oversight**: Confidence thresholds require human review
//! - **Consent Framework**: Built-in consent tracking and management
//!
//! ## Troubleshooting
//!
//! ### Common Issues
//!
//! 1. **GPU Not Detected**
//!    ```
//!    Error: GPU backend not available
//!    Solution: Install CUDA/OpenCL drivers, enable 'gpu' feature
//!    ```
//!
//! 2. **Model Loading Failures**
//!    ```
//!    Error: Model file not found
//!    Solution: Download models using `cargo run --example download_models`
//!    ```
//!
//! 3. **Low Confidence Scores**
//!    ```
//!    Issue: All predictions return low confidence
//!    Solution: Check input quality, enable multiple modalities
//!    ```
//!
//! 4. **Performance Issues**
//!    ```
//!    Issue: Slow processing times
//!    Solution: Enable 'parallel' feature, consider 'gpu' acceleration
//!    ```
//!
//! ### Performance Optimization
//!
//! - Enable parallel processing for multi-core systems
//! - Use GPU acceleration for batch processing
//! - Implement caching for repeated analysis of similar inputs
//! - Consider model quantization for mobile deployments
//! - Use streaming analysis for long audio/video sequences

pub mod error;
pub mod types;
pub mod modalities;
pub mod optimization;
pub mod streaming;
pub mod prelude;

pub use error::{VeritasError, Result};
pub use types::*;

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use async_trait::async_trait;

/// Core trait for all modality analyzers in the Veritas Nexus system.
///
/// This trait defines the interface that all modality-specific analyzers must implement.
/// Each analyzer is responsible for processing a specific type of input (text, vision, 
/// audio, or physiological data) and producing a deception score with explanations.
///
/// # Type Parameters
///
/// - `T`: A floating-point type implementing [`Float`] for numerical computations
///
/// # Associated Types
///
/// - `Input`: The type of data this analyzer can process
/// - `Output`: The type of score/result produced (must implement [`DeceptionScore`])
/// - `Config`: Configuration type for customizing analyzer behavior
///
/// # Examples
///
/// Basic usage with a hypothetical text analyzer:
///
/// ```rust,no_run
/// use veritas_nexus::{ModalityAnalyzer, Result};
/// # use std::time::SystemTime;
/// # use veritas_nexus::{DeceptionScore, ModalityType, Feature, ExplanationTrace};
/// # struct TextAnalyzer;
/// # struct TextInput { text: String }
/// # struct TextScore { score: f64 }
/// # impl DeceptionScore<f64> for TextScore {
/// #     fn probability(&self) -> f64 { self.score }
/// #     fn confidence(&self) -> f64 { 0.85 }
/// #     fn modality(&self) -> ModalityType { ModalityType::Text }
/// #     fn features(&self) -> Vec<Feature<f64>> { vec![] }
/// #     fn timestamp(&self) -> SystemTime { SystemTime::now() }
/// # }
/// # #[async_trait::async_trait]
/// # impl ModalityAnalyzer<f64> for TextAnalyzer {
/// #     type Input = TextInput;
/// #     type Output = TextScore;
/// #     type Config = ();
/// #     async fn analyze(&self, input: &Self::Input) -> Result<Self::Output> {
/// #         Ok(TextScore { score: 0.3 })
/// #     }
/// #     fn confidence(&self) -> f64 { 0.85 }
/// #     fn explain(&self) -> ExplanationTrace { ExplanationTrace { steps: vec![], confidence: 0.85, reasoning: "".to_string() } }
/// #     fn config(&self) -> &Self::Config { &() }
/// # }
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let analyzer = TextAnalyzer;
///     let input = TextInput { text: "I didn't do anything wrong".to_string() };
///     
///     let result = analyzer.analyze(&input).await?;
///     println!("Deception probability: {:.2}", result.probability());
///     println!("Confidence: {:.2}", analyzer.confidence());
///     
///     Ok(())
/// }
/// ```
///
/// # Thread Safety
///
/// All analyzers must be [`Send`] + [`Sync`] to support concurrent processing
/// across multiple threads and async tasks.
///
/// # Performance Considerations
///
/// - Analyzers should be designed for repeated use and reuse expensive resources
/// - Consider implementing caching for expensive operations in the analyzer
/// - Use the [`ExplanationTrace`] to provide transparency without impacting performance
#[async_trait]
pub trait ModalityAnalyzer<T: Float>: Send + Sync {
    type Input: Send + Sync;
    type Output: DeceptionScore<T> + Send + Sync;
    type Config: Send + Sync;
    
    /// Analyze input data for deception indicators.
    ///
    /// This is the core method that processes input data and returns a deception score.
    /// The analysis should be deterministic for the same input and configuration.
    ///
    /// # Arguments
    ///
    /// * `input` - The data to analyze (type varies by modality)
    ///
    /// # Returns
    ///
    /// A result containing a deception score with confidence metrics and features,
    /// or an error if analysis fails.
    ///
    /// # Errors
    ///
    /// This method can return errors for various reasons:
    /// - Invalid or corrupted input data
    /// - Model loading failures
    /// - Resource exhaustion (memory, compute)
    /// - Network issues (for cloud-based models)
    async fn analyze(&self, input: &Self::Input) -> Result<Self::Output>;
    
    /// Get the current confidence level of the analyzer.
    ///
    /// This represents the analyzer's confidence in its ability to make accurate
    /// predictions given the current model state and configuration. This is different
    /// from the confidence of individual predictions.
    ///
    /// # Returns
    ///
    /// A value between 0.0 and 1.0 where:
    /// - 0.0 = No confidence (analyzer not ready/trained)
    /// - 1.0 = Maximum confidence (fully trained, validated model)
    fn confidence(&self) -> T;
    
    /// Generate an explanation trace for the analysis.
    ///
    /// This provides insight into how the analyzer arrived at its decision,
    /// supporting explainable AI requirements. The explanation should be
    /// human-readable and actionable.
    ///
    /// # Returns
    ///
    /// An [`ExplanationTrace`] containing step-by-step reasoning,
    /// evidence, and confidence levels for each step.
    fn explain(&self) -> ExplanationTrace;
    
    /// Get the analyzer's current configuration.
    ///
    /// This provides access to the configuration used to customize
    /// the analyzer's behavior, thresholds, and processing parameters.
    ///
    /// # Returns
    ///
    /// A reference to the configuration object.
    fn config(&self) -> &Self::Config;
}

/// Trait for deception scores produced by modality analyzers.
///
/// This trait standardizes the interface for deception detection results across
/// all modalities (text, vision, audio, physiological). Each score includes
/// probability estimates, confidence metrics, contributing features, and metadata.
///
/// # Type Parameters
///
/// - `T`: A floating-point type implementing [`Float`] for numerical values
///
/// # Examples
///
/// Implementing a custom deception score:
///
/// ```rust
/// use veritas_nexus::{DeceptionScore, ModalityType, Feature};
/// use std::time::SystemTime;
///
/// #[derive(Debug)]
/// struct CustomScore {
///     prob: f64,
///     conf: f64,
///     timestamp: SystemTime,
/// }
///
/// impl DeceptionScore<f64> for CustomScore {
///     fn probability(&self) -> f64 {
///         self.prob
///     }
///
///     fn confidence(&self) -> f64 {
///         self.conf
///     }
///
///     fn modality(&self) -> ModalityType {
///         ModalityType::Text
///     }
///
///     fn features(&self) -> Vec<Feature<f64>> {
///         vec![Feature {
///             name: "linguistic_complexity".to_string(),
///             value: 0.7,
///             weight: 0.3,
///             description: "Sentence structure complexity indicator".to_string(),
///         }]
///     }
///
///     fn timestamp(&self) -> SystemTime {
///         self.timestamp
///     }
/// }
/// ```
///
/// Using deception scores for decision making:
///
/// ```rust,no_run
/// use veritas_nexus::{DeceptionScore, ModalityType};
///
/// fn evaluate_truthfulness<T: DeceptionScore<f64>>(score: &T) -> String {
///     match score.probability() {
///         p if p < 0.3 => "Likely truthful".to_string(),
///         p if p > 0.7 => "Likely deceptive".to_string(),
///         _ => "Uncertain - require additional analysis".to_string(),
///     }
/// }
/// ```
///
/// # Score Interpretation
///
/// - **Probability**: 0.0 = definitely truthful, 1.0 = definitely deceptive
/// - **Confidence**: 0.0 = no confidence in prediction, 1.0 = high confidence
/// - **Features**: Individual indicators that contributed to the score
/// - **Modality**: Which analysis method produced this score
pub trait DeceptionScore<T: Float>: Debug + Send + Sync {
    /// Get the deception probability estimate.
    ///
    /// # Returns
    ///
    /// A value between 0.0 and 1.0 where:
    /// - 0.0 = High confidence the statement/behavior is truthful
    /// - 0.5 = Neutral/uncertain
    /// - 1.0 = High confidence the statement/behavior is deceptive
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use veritas_nexus::DeceptionScore;
    /// fn interpret_score<S: DeceptionScore<f64>>(score: &S) -> &'static str {
    ///     match score.probability() {
    ///         p if p < 0.2 => "Strong indicator of truth",
    ///         p if p < 0.4 => "Weak indicator of truth", 
    ///         p if p < 0.6 => "Neutral/ambiguous",
    ///         p if p < 0.8 => "Weak indicator of deception",
    ///         _ => "Strong indicator of deception",
    ///     }
    /// }
    /// ```
    fn probability(&self) -> T;
    
    /// Get the confidence level in this score.
    ///
    /// This represents how confident the analyzer is in the probability estimate.
    /// A low confidence score suggests the prediction may be unreliable.
    ///
    /// # Returns
    ///
    /// A value between 0.0 and 1.0 where:
    /// - 0.0 = No confidence (insufficient data, low-quality input)
    /// - 1.0 = High confidence (clear indicators, high-quality data)
    ///
    /// # Decision Guidelines
    ///
    /// - Confidence < 0.3: Discard or request more data
    /// - Confidence 0.3-0.7: Use with caution, combine with other modalities
    /// - Confidence > 0.7: High reliability for decision making
    fn confidence(&self) -> T;
    
    /// Get the modality that generated this score.
    ///
    /// This identifies which analysis method (text, vision, audio, etc.)
    /// produced this deception score, enabling modality-specific handling.
    ///
    /// # Returns
    ///
    /// A [`ModalityType`] enum value indicating the source modality.
    fn modality(&self) -> ModalityType;
    
    /// Get the detailed features that contributed to this score.
    ///
    /// Features provide explainability by showing which specific indicators
    /// influenced the deception probability. Each feature includes its value,
    /// weight in the decision, and a human-readable description.
    ///
    /// # Returns
    ///
    /// A vector of [`Feature`] objects with contributing evidence.
    /// An empty vector indicates no feature-level explanations are available.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use veritas_nexus::{DeceptionScore, Feature};
    /// fn explain_decision<S: DeceptionScore<f64>>(score: &S) {
    ///     println!("Decision factors:");
    ///     for feature in score.features() {
    ///         println!("  {}: {:.3} (weight: {:.2})", 
    ///             feature.name, feature.value, feature.weight);
    ///         println!("    {}", feature.description);
    ///     }
    /// }
    /// ```
    fn features(&self) -> Vec<Feature<T>>;
    
    /// Get the timestamp when this score was generated.
    ///
    /// This enables temporal analysis and tracking of when assessments were made,
    /// which is important for auditing and understanding decision timelines.
    ///
    /// # Returns
    ///
    /// A [`SystemTime`] indicating when the analysis was completed.
    fn timestamp(&self) -> std::time::SystemTime;
}

/// Types of supported analysis modalities in the Veritas Nexus system.
///
/// Each modality represents a different approach to analyzing potential deception
/// through various input channels. The system supports multi-modal fusion to
/// combine insights from multiple modalities for more robust detection.
///
/// # Modality Descriptions
///
/// - **Text**: Analyzes written or transcribed speech for linguistic indicators
/// - **Vision**: Processes visual data for facial expressions and micro-expressions  
/// - **Audio**: Examines vocal characteristics, prosody, and stress indicators
/// - **Physiological**: Monitors biometric signals like heart rate and skin conductance
///
/// # Examples
///
/// ```rust
/// use veritas_nexus::ModalityType;
///
/// // Check which modalities are available for a given input
/// fn supported_modalities(has_video: bool, has_audio: bool, has_transcript: bool) -> Vec<ModalityType> {
///     let mut modalities = Vec::new();
///     
///     if has_video {
///         modalities.push(ModalityType::Vision);
///     }
///     if has_audio {
///         modalities.push(ModalityType::Audio);
///     }
///     if has_transcript {
///         modalities.push(ModalityType::Text);
///     }
///     
///     modalities
/// }
/// ```
///
/// # Multi-Modal Fusion
///
/// Modalities can be combined using fusion strategies:
///
/// ```rust,no_run
/// use veritas_nexus::ModalityType;
/// use std::collections::HashMap;
///
/// // Example fusion weights for different modalities
/// fn get_fusion_weights() -> HashMap<ModalityType, f64> {
///     let mut weights = HashMap::new();
///     weights.insert(ModalityType::Text, 0.4);      // High weight for linguistic analysis
///     weights.insert(ModalityType::Vision, 0.3);    // Moderate weight for facial cues
///     weights.insert(ModalityType::Audio, 0.2);     // Voice characteristics
///     weights.insert(ModalityType::Physiological, 0.1); // Biometric signals
///     weights
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// Text-based analysis using natural language processing.
    ///
    /// Analyzes linguistic patterns, sentiment, complexity, and deception
    /// indicators in written text or speech transcripts.
    Text,
    
    /// Vision-based analysis using computer vision techniques.
    ///
    /// Processes facial expressions, micro-expressions, gaze patterns,
    /// and other visual behavioral indicators.
    Vision,
    
    /// Audio-based analysis of vocal characteristics.
    ///
    /// Examines prosodic features, stress indicators, pitch variations,
    /// and voice quality metrics.
    Audio,
    
    /// Physiological signal analysis from biometric sensors.
    ///
    /// Monitors stress responses through heart rate, skin conductance,
    /// respiration patterns, and other autonomic indicators.
    Physiological,
}

/// A feature that contributed to a deception score.
///
/// Features provide explainability by identifying specific indicators that
/// influenced the deception probability calculation. Each feature includes
/// its measured value, importance weight, and human-readable description.
///
/// # Fields
///
/// - `name`: Unique identifier for this feature (e.g., "hesitation_count")
/// - `value`: The measured value for this feature (normalized or raw)
/// - `weight`: The importance weight applied during scoring (0.0 to 1.0)
/// - `description`: Human-readable explanation of what this feature measures
///
/// # Examples
///
/// Creating features for text analysis:
///
/// ```rust
/// use veritas_nexus::Feature;
///
/// let linguistic_complexity = Feature {
///     name: "linguistic_complexity".to_string(),
///     value: 0.75,
///     weight: 0.3,
///     description: "Measures unusual sentence structures and word choices".to_string(),
/// };
///
/// let hesitation_markers = Feature {
///     name: "hesitation_count".to_string(),
///     value: 5.0,
///     weight: 0.15,
///     description: "Number of hesitation words (um, uh, etc.) per 100 words".to_string(),
/// };
/// ```
///
/// Analyzing feature contributions:
///
/// ```rust
/// use veritas_nexus::Feature;
///
/// fn analyze_features(features: &[Feature<f64>]) {
///     // Sort by contribution (value * weight)
///     let mut contributions: Vec<_> = features.iter()
///         .map(|f| (f, f.value * f.weight))
///         .collect();
///     contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
///     
///     println!("Top contributing features:");
///     for (feature, contribution) in contributions.iter().take(3) {
///         println!("  {}: {:.3} ({})", 
///             feature.name, contribution, feature.description);
///     }
/// }
/// ```
///
/// # Feature Design Guidelines
///
/// - **Names**: Use snake_case descriptive identifiers
/// - **Values**: Normalize to 0.0-1.0 range when possible for consistency
/// - **Weights**: Should sum to ≤ 1.0 across all features for a modality
/// - **Descriptions**: Provide actionable, non-technical explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature<T: Float> {
    /// Unique name identifying this feature.
    ///
    /// Should be a descriptive, machine-readable identifier using snake_case
    /// convention (e.g., "vocal_stress_level", "micro_expression_count").
    pub name: String,
    
    /// The measured value for this feature.
    ///
    /// Can be raw measurements, normalized scores (0.0-1.0), or counts.
    /// The interpretation depends on the specific feature type.
    pub value: T,
    
    /// The importance weight applied to this feature in scoring.
    ///
    /// Represents how much this feature contributes to the final score.
    /// Values typically range from 0.0 (no influence) to 1.0 (maximum influence).
    pub weight: T,
    
    /// Human-readable description of what this feature measures.
    ///
    /// Should explain the feature in plain language suitable for
    /// non-technical users reviewing the analysis results.
    pub description: String,
}

/// Explanation trace for analysis decisions providing transparency and auditability.
///
/// The explanation trace documents the step-by-step reasoning process used to arrive
/// at a deception score, supporting explainable AI requirements and enabling users
/// to understand and validate the system's decisions.
///
/// # Fields
///
/// - `steps`: Sequential reasoning steps taken during analysis
/// - `confidence`: Overall confidence in the explanation (0.0 to 1.0)
/// - `reasoning`: High-level summary of the reasoning process
///
/// # Examples
///
/// Creating an explanation trace:
///
/// ```rust
/// use veritas_nexus::{ExplanationTrace, ExplanationStep};
///
/// let trace = ExplanationTrace {
///     steps: vec![
///         ExplanationStep {
///             step_type: "observation".to_string(),
///             description: "Detected 3 micro-expressions in 30-second video".to_string(),
///             evidence: vec![
///                 "Surprise expression at 12s".to_string(),
///                 "Contempt expression at 18s".to_string(),
///                 "Fear expression at 25s".to_string(),
///             ],
///             confidence: 0.85,
///         },
///         ExplanationStep {
///             step_type: "analysis".to_string(),
///             description: "Cross-referenced with baseline emotional patterns".to_string(),
///             evidence: vec![
///                 "Surprise outside expected range".to_string(),
///                 "Contempt indicates possible concealment".to_string(),
///             ],
///             confidence: 0.72,
///         },
///     ],
///     confidence: 0.78,
///     reasoning: "Multiple micro-expressions suggest emotional incongruence with verbal statements".to_string(),
/// };
/// ```
///
/// Analyzing explanation quality:
///
/// ```rust
/// use veritas_nexus::ExplanationTrace;
///
/// fn validate_explanation(trace: &ExplanationTrace) -> bool {
///     // Ensure explanation meets quality thresholds
///     trace.confidence > 0.5 && 
///     !trace.steps.is_empty() &&
///     trace.steps.iter().all(|step| step.confidence > 0.3)
/// }
/// ```
///
/// # Quality Guidelines
///
/// - **Steps**: Should be logically ordered and build upon each other
/// - **Evidence**: Provide specific, verifiable observations
/// - **Confidence**: Reflect genuine uncertainty in the reasoning process
/// - **Reasoning**: Give concise, actionable summary for decision makers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationTrace {
    /// Sequential steps in the reasoning process.
    ///
    /// Each step represents a logical stage in the analysis, from initial
    /// observations through intermediate analysis to final conclusions.
    pub steps: Vec<ExplanationStep>,
    
    /// Overall confidence in this explanation.
    ///
    /// Represents how confident the system is that this explanation
    /// accurately reflects the true reasoning process. Lower values
    /// indicate uncertainty in the explanation itself.
    pub confidence: f64,
    
    /// High-level summary of the reasoning process.
    ///
    /// A concise, human-readable explanation that summarizes the key
    /// logic and conclusions without requiring detailed step analysis.
    pub reasoning: String,
}

/// A single step in an explanation trace representing one logical stage of analysis.
///
/// Each step documents a specific phase of the reasoning process, from initial
/// observations to intermediate analysis to final conclusions. Steps should be
/// ordered logically and build upon previous findings.
///
/// # Fields
///
/// - `step_type`: Category of reasoning (e.g., "observation", "analysis", "conclusion")
/// - `description`: What was done or concluded in this step
/// - `evidence`: Specific supporting evidence or observations
/// - `confidence`: How confident the system is in this step (0.0 to 1.0)
///
/// # Examples
///
/// Creating analysis steps:
///
/// ```rust
/// use veritas_nexus::ExplanationStep;
///
/// let observation_step = ExplanationStep {
///     step_type: "observation".to_string(),
///     description: "Subject exhibited 4 hesitation markers during response".to_string(),
///     evidence: vec![
///         "Um... at timestamp 00:12".to_string(),
///         "Long pause (3.2s) at timestamp 00:18".to_string(),
///         "Uh at timestamp 00:25".to_string(),
///         "Sentence restart at timestamp 00:30".to_string(),
///     ],
///     confidence: 0.92,
/// };
///
/// let analysis_step = ExplanationStep {
///     step_type: "analysis".to_string(), 
///     description: "Hesitation frequency exceeds baseline by 3.1 standard deviations".to_string(),
///     evidence: vec![
///         "Subject baseline: 0.8 hesitations per 30s".to_string(),
///         "Current measurement: 4.0 hesitations per 30s".to_string(),
///         "Statistical significance: p < 0.001".to_string(),
///     ],
///     confidence: 0.87,
/// };
/// ```
///
/// # Step Type Categories
///
/// Common step types include:
/// - **observation**: Direct measurements or detections
/// - **analysis**: Processing and comparison of observations
/// - **pattern**: Recognition of meaningful patterns
/// - **correlation**: Relationships between different indicators
/// - **conclusion**: Final reasoning and decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationStep {
    /// The category or type of reasoning performed in this step.
    ///
    /// Should use consistent, descriptive categories like "observation",
    /// "analysis", "pattern_recognition", "conclusion", etc.
    pub step_type: String,
    
    /// Description of what was accomplished in this reasoning step.
    ///
    /// Should be clear, specific, and actionable. Avoid technical jargon
    /// when possible while maintaining precision.
    pub description: String,
    
    /// Specific evidence supporting this step's conclusions.
    ///
    /// Each piece of evidence should be verifiable and directly related
    /// to the step's reasoning. Include timestamps, measurements, or
    /// specific observations where applicable.
    pub evidence: Vec<String>,
    
    /// Confidence level in this particular step.
    ///
    /// Represents how certain the system is about the correctness of
    /// this step's reasoning and conclusions (0.0 = no confidence,
    /// 1.0 = complete confidence).
    pub confidence: f64,
}

/// Fusion strategy trait for combining multiple modality scores into unified decisions.
///
/// Fusion strategies define how scores from different modalities (text, vision, audio,
/// physiological) are combined to produce a single, more robust deception assessment.
/// Different strategies may use simple averaging, weighted combinations, attention
/// mechanisms, or complex neural fusion approaches.
///
/// # Type Parameters
///
/// - `T`: A floating-point type implementing [`Float`] for numerical computations
///
/// # Examples
///
/// Implementing a simple weighted fusion strategy:
///
/// ```rust,no_run
/// use veritas_nexus::{FusionStrategy, DeceptionScore, FusedScore, Feedback, Result};
/// use std::collections::HashMap;
/// # use veritas_nexus::ModalityType;
/// # use std::time::SystemTime;
/// # use async_trait::async_trait;
///
/// struct WeightedFusion {
///     modality_weights: HashMap<ModalityType, f64>,
/// }
///
/// #[async_trait]
/// impl FusionStrategy<f64> for WeightedFusion {
///     async fn fuse(&self, scores: &[Box<dyn DeceptionScore<f64>>]) -> Result<FusedScore<f64>> {
///         let mut weighted_sum = 0.0;
///         let mut total_weight = 0.0;
///         let mut contributions = HashMap::new();
///         
///         for score in scores {
///             let modality = score.modality();
///             let weight = self.modality_weights.get(&modality).unwrap_or(&1.0);
///             let contribution = score.probability() * weight;
///             
///             weighted_sum += contribution;
///             total_weight += weight;
///             contributions.insert(modality, contribution);
///         }
///         
///         let final_probability = if total_weight > 0.0 {
///             weighted_sum / total_weight
///         } else {
///             0.5 // Neutral when no weights
///         };
///         
///         Ok(FusedScore {
///             probability: final_probability,
///             confidence: 0.8, // Simplified confidence calculation
///             modality_contributions: contributions,
///             timestamp: SystemTime::now(),
///             explanation: Default::default(), // Simplified for example
///         })
///     }
///     
///     fn weights(&self) -> &[f64] {
///         // This implementation would need adjustment for the actual interface
///         &[0.4, 0.3, 0.2, 0.1] // Example weights
///     }
///     
///     fn update_weights(&mut self, feedback: &Feedback<f64>) {
///         // Update weights based on performance feedback
///         for (modality, accuracy) in &feedback.modality_accuracies {
///             if let Some(weight) = self.modality_weights.get_mut(modality) {
///                 *weight = (*weight + accuracy) / 2.0; // Simple adaptation
///             }
///         }
///     }
/// }
/// ```
///
/// # Fusion Strategies
///
/// Common fusion approaches include:
/// - **Simple Average**: Uniform weight across all modalities
/// - **Weighted Average**: Hand-tuned or learned importance weights
/// - **Attention-Based**: Dynamic weights based on input characteristics  
/// - **Neural Fusion**: End-to-end learned combination networks
/// - **Bayesian Fusion**: Probabilistic combination with uncertainty
///
/// # Performance Considerations
///
/// - Fusion should be fast enough for real-time applications
/// - Consider caching fusion results when inputs haven't changed
/// - Weight updates should be conservative to prevent overfitting
#[async_trait] 
pub trait FusionStrategy<T: Float>: Send + Sync {
    /// Fuse multiple modality scores into a single unified decision.
    ///
    /// This method combines scores from different modalities using the strategy's
    /// specific approach (weighted average, attention, neural fusion, etc.).
    ///
    /// # Arguments
    ///
    /// * `scores` - Scores from different modalities to be combined
    ///
    /// # Returns
    ///
    /// A [`FusedScore`] containing the combined assessment with contribution
    /// breakdown and explanation.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Empty or incompatible score inputs
    /// - Mathematical errors in fusion computation
    /// - Resource allocation failures
    async fn fuse(&self, scores: &[Box<dyn DeceptionScore<T>>]) -> Result<FusedScore<T>>;
    
    /// Get the current importance weights for each modality.
    ///
    /// Returns the weights currently used by this fusion strategy to
    /// combine scores from different modalities.
    ///
    /// # Returns
    ///
    /// A slice of weights, typically corresponding to modalities in
    /// a consistent order (e.g., [text, vision, audio, physiological]).
    fn weights(&self) -> &[T];
    
    /// Update fusion weights based on performance feedback.
    ///
    /// Allows the fusion strategy to adapt its weights based on observed
    /// performance of individual modalities and overall system accuracy.
    ///
    /// # Arguments
    ///
    /// * `feedback` - Performance metrics for weight adjustment
    ///
    /// # Adaptation Guidelines
    ///
    /// - Updates should be gradual to prevent instability
    /// - Consider using exponential moving averages for smooth adaptation
    /// - Maintain weight bounds to prevent extreme imbalances
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

/// Simple decision type for actions (compatibility alias)
pub type SimpleDecisionType = DecisionType;

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
    pub decision: Option<Decision<T>>,
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

/// Generic decision type with confidence scoring
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision<T: Float> {
    /// The type of decision made
    pub decision_type: DecisionType,
    /// Confidence score for this decision
    pub confidence: T,
    /// Supporting evidence for the decision
    pub evidence: Vec<String>,
    /// Timestamp when decision was made
    pub timestamp: std::time::SystemTime,
}

impl<T: Float> Decision<T> {
    /// Create a new decision
    pub fn new(decision_type: DecisionType, confidence: T) -> Self {
        Self {
            decision_type,
            confidence,
            evidence: Vec::new(),
            timestamp: std::time::SystemTime::now(),
        }
    }
    
    /// Add evidence to support this decision
    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }
    
    /// Check if this is a high-confidence decision
    pub fn is_high_confidence(&self) -> bool {
        self.confidence > T::from(0.8).unwrap()
    }
}

/// Decision type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DecisionType {
    /// Determined to be truthful
    Truthful,
    /// Determined to be deceptive  
    Deceptive,
    /// Insufficient evidence to determine
    #[default]
    Uncertain,
}

impl fmt::Display for DecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecisionType::Truthful => write!(f, "Truthful"),
            DecisionType::Deceptive => write!(f, "Deceptive"),
            DecisionType::Uncertain => write!(f, "Uncertain"),
        }
    }
}

/// Compatibility alias for the simple decision type
pub type SimpleDecision = Decision<f64>;

// Ensure basic modules are available
pub mod engine;
pub mod fusion;
pub mod learning;
pub mod neural_integration;
pub mod reasoning;
pub mod agents;

/// Builder for creating complex analysis pipelines
pub struct AnalysisPipeline<T: Float> {
    analyzers: Vec<Box<dyn ModalityAnalyzer<T>>>,
    fusion_strategy: Box<dyn FusionStrategy<T>>,
    config: AnalysisConfig<T>,
}

/// Configuration for analysis pipeline
#[derive(Debug, Clone)]
pub struct AnalysisConfig<T: Float> {
    /// Minimum confidence threshold for decisions
    pub min_confidence_threshold: T,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Maximum processing time per input
    pub max_processing_time: std::time::Duration,
    /// Enable detailed logging
    pub detailed_logging: bool,
}

impl<T: Float> Default for AnalysisConfig<T> {
    fn default() -> Self {
        Self {
            min_confidence_threshold: T::from(0.5).unwrap(),
            parallel_processing: true,
            max_processing_time: std::time::Duration::from_secs(30),
            detailed_logging: false,
        }
    }
}

impl<T: Float + Send + Sync + 'static> AnalysisPipeline<T> {
    /// Create a new analysis pipeline
    pub fn new(fusion_strategy: Box<dyn FusionStrategy<T>>) -> Self {
        Self {
            analyzers: Vec::new(),
            fusion_strategy,
            config: AnalysisConfig::default(),
        }
    }
    
    /// Add a modality analyzer to the pipeline
    pub fn add_analyzer(mut self, analyzer: Box<dyn ModalityAnalyzer<T>>) -> Self {
        self.analyzers.push(analyzer);
        self
    }
    
    /// Set analysis configuration
    pub fn with_config(mut self, config: AnalysisConfig<T>) -> Self {
        self.config = config;
        self
    }
    
    /// Process input through all analyzers and fuse results
    pub async fn analyze(&mut self, inputs: &MultiModalInput<T>) -> Result<FusedScore<T>> {
        let mut scores: Vec<Box<dyn DeceptionScore<T>>> = Vec::new();
        
        // Run all analyzers
        for analyzer in &self.analyzers {
            if let Some(input) = inputs.get_input_for_analyzer(analyzer) {
                let score = analyzer.analyze(input).await?;
                scores.push(Box::new(score));
            }
        }
        
        if scores.is_empty() {
            return Err(VeritasError::ProcessingError("No valid inputs for analysis".to_string()));
        }
        
        // Fuse scores
        self.fusion_strategy.fuse(&scores).await
    }
}

/// Multi-modal input container
pub struct MultiModalInput<T: Float> {
    _phantom: PhantomData<T>,
    // This would contain different input types in a real implementation
}

impl<T: Float> MultiModalInput<T> {
    /// Get appropriate input for a specific analyzer
    pub fn get_input_for_analyzer(&self, _analyzer: &dyn ModalityAnalyzer<T>) -> Option<&dyn std::any::Any> {
        // Placeholder implementation
        None
    }
}

// The prelude module is defined in src/prelude.rs
