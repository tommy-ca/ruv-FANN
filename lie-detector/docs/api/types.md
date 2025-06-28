# Core Types

This document describes the core data types and structures used throughout Veritas Nexus.

## Main Types

### `LieDetector`

The main interface for lie detection analysis.

```rust
pub struct LieDetector {
    // Internal fields are private
}

impl LieDetector {
    /// Create a new lie detector with default configuration
    pub async fn new() -> Result<Self>;
    
    /// Create a builder for custom configuration
    pub fn builder() -> LieDetectorBuilder;
    
    /// Analyze a single input
    pub async fn analyze(&self, input: &AnalysisInput) -> Result<AnalysisResult>;
    
    /// Analyze multiple inputs in batch
    pub async fn analyze_batch(&self, inputs: &[AnalysisInput]) -> Result<Vec<AnalysisResult>>;
    
    /// Analyze with custom configuration
    pub async fn analyze_with_config(
        &self, 
        input: &AnalysisInput, 
        config: &AnalysisConfig
    ) -> Result<AnalysisResult>;
}
```

#### Usage Example

```rust
let detector = LieDetector::new().await?;
let result = detector.analyze(&input).await?;
```

### `LieDetectorBuilder`

Builder pattern for configuring `LieDetector` instances.

```rust
pub struct LieDetectorBuilder {
    // Configuration fields
}

impl LieDetectorBuilder {
    /// Set vision analysis configuration
    pub fn with_vision(self, config: VisionConfig) -> Self;
    
    /// Set audio analysis configuration
    pub fn with_audio(self, config: AudioConfig) -> Self;
    
    /// Set text analysis configuration
    pub fn with_text(self, config: TextConfig) -> Self;
    
    /// Set fusion strategy
    pub fn with_fusion_strategy(self, strategy: FusionStrategy) -> Self;
    
    /// Enable GPU acceleration
    pub fn with_gpu_acceleration(self, enabled: bool) -> Self;
    
    /// Set batch size for processing
    pub fn with_batch_size(self, size: usize) -> Self;
    
    /// Set memory limit in MB
    pub fn with_memory_limit_mb(self, limit: usize) -> Self;
    
    /// Build the configured detector
    pub async fn build(self) -> Result<LieDetector>;
}
```

#### Usage Example

```rust
let detector = LieDetector::builder()
    .with_gpu_acceleration(true)
    .with_batch_size(32)
    .with_fusion_strategy(FusionStrategy::AttentionBased)
    .build()
    .await?;
```

### `AnalysisInput`

Input data structure for lie detection analysis.

```rust
#[derive(Debug, Clone)]
pub struct AnalysisInput {
    pub video_path: Option<PathBuf>,
    pub audio_path: Option<PathBuf>,
    pub text: Option<String>,
    pub physiological_data: Option<Vec<f32>>,
    pub metadata: InputMetadata,
}

impl AnalysisInput {
    /// Create a new empty input
    pub fn new() -> Self;
    
    /// Add video input from file path
    pub fn with_video_path<P: Into<PathBuf>>(self, path: P) -> Self;
    
    /// Add video input from raw frames
    pub fn with_video_frames(self, frames: Vec<VideoFrame>) -> Self;
    
    /// Add audio input from file path
    pub fn with_audio_path<P: Into<PathBuf>>(self, path: P) -> Self;
    
    /// Add audio input from raw data
    pub fn with_audio_data(self, data: AudioData) -> Self;
    
    /// Add text input
    pub fn with_text<S: Into<String>>(self, text: S) -> Self;
    
    /// Add physiological data
    pub fn with_physiological_data(self, data: Vec<f32>) -> Self;
    
    /// Add metadata
    pub fn with_metadata(self, metadata: InputMetadata) -> Self;
    
    /// Validate that the input has at least one modality
    pub fn validate(&self) -> Result<()>;
}
```

#### Usage Example

```rust
let input = AnalysisInput::new()
    .with_text("I was definitely not there at that time")
    .with_video_path("interview.mp4")
    .with_audio_path("interview.wav")
    .with_physiological_data(vec![72.5, 73.1, 74.2]);
```

### `AnalysisResult`

Result structure containing analysis outcomes and metadata.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    pub decision: DeceptionDecision,
    pub confidence: f32,
    pub modality_scores: ModalityScores,
    pub reasoning_trace: Vec<String>,
    pub processing_time: Duration,
    pub explanation: Option<String>,
    pub uncertainty: UncertaintyAnalysis,
    pub metadata: ResultMetadata,
}

impl AnalysisResult {
    /// Get the final deception probability (0.0 = truthful, 1.0 = deceptive)
    pub fn deception_probability(&self) -> f32;
    
    /// Check if the result indicates deception
    pub fn is_deceptive(&self) -> bool;
    
    /// Check if the result indicates truthfulness
    pub fn is_truthful(&self) -> bool;
    
    /// Check if the result is uncertain
    pub fn is_uncertain(&self) -> bool;
    
    /// Get confidence as a percentage (0-100)
    pub fn confidence_percentage(&self) -> f32;
    
    /// Export result to JSON
    pub fn to_json(&self) -> Result<String>;
    
    /// Export result to structured text
    pub fn to_text(&self) -> String;
}
```

#### Usage Example

```rust
let result = detector.analyze(&input).await?;

println!("Decision: {:?}", result.decision);
println!("Confidence: {:.1}%", result.confidence_percentage());
println!("Processing time: {}ms", result.processing_time.as_millis());

if result.is_deceptive() {
    println!("Deception detected!");
}
```

### `DeceptionDecision`

Enumeration of possible analysis decisions.

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeceptionDecision {
    /// Subject appears to be telling the truth
    Truthful {
        probability: f32,
    },
    
    /// Subject appears to be engaging in deception
    Deceptive {
        probability: f32,
    },
    
    /// Analysis is uncertain due to conflicting evidence
    Uncertain {
        conflicting_evidence: Vec<String>,
    },
    
    /// Insufficient data to make a determination
    InsufficientData {
        missing_modalities: Vec<String>,
    },
}

impl DeceptionDecision {
    /// Get the confidence level of this decision
    pub fn confidence(&self) -> f32;
    
    /// Get a human-readable description
    pub fn description(&self) -> String;
    
    /// Check if this represents a high-confidence decision
    pub fn is_high_confidence(&self) -> bool;
}
```

#### Usage Example

```rust
match result.decision {
    DeceptionDecision::Truthful { probability } => {
        println!("Subject is truthful (probability: {:.1}%)", probability * 100.0);
    }
    DeceptionDecision::Deceptive { probability } => {
        println!("Deception detected (probability: {:.1}%)", probability * 100.0);
    }
    DeceptionDecision::Uncertain { conflicting_evidence } => {
        println!("Uncertain result due to: {:?}", conflicting_evidence);
    }
    DeceptionDecision::InsufficientData { missing_modalities } => {
        println!("Need more data: {:?}", missing_modalities);
    }
}
```

### `ModalityScores`

Scores and confidence measures for each input modality.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ModalityScores {
    pub vision: Option<ModalityScore>,
    pub audio: Option<ModalityScore>,
    pub text: Option<ModalityScore>,
    pub physiological: Option<ModalityScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModalityScore {
    pub score: f32,           // 0.0 = truthful, 1.0 = deceptive
    pub confidence: f32,      // 0.0 = low confidence, 1.0 = high confidence
    pub features: Vec<FeatureContribution>,
    pub processing_time: Duration,
    pub quality: f32,         // Input quality assessment
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureContribution {
    pub name: String,
    pub value: f32,
    pub weight: f32,
    pub description: String,
}
```

#### Usage Example

```rust
let scores = &result.modality_scores;

if let Some(vision) = &scores.vision {
    println!("Vision score: {:.3} (confidence: {:.1}%)", 
             vision.score, vision.confidence * 100.0);
    
    for feature in &vision.features {
        println!("  {}: {:.3} (weight: {:.3})", 
                 feature.name, feature.value, feature.weight);
    }
}
```

### `UncertaintyAnalysis`

Detailed uncertainty quantification for the analysis.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct UncertaintyAnalysis {
    pub total_uncertainty: f32,
    pub epistemic_uncertainty: f32,    // Model uncertainty
    pub aleatoric_uncertainty: f32,    // Data uncertainty
    pub confidence_interval: (f32, f32),
    pub uncertainty_sources: Vec<UncertaintySource>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UncertaintySource {
    pub source: String,
    pub contribution: f32,
    pub description: String,
}
```

#### Usage Example

```rust
let uncertainty = &result.uncertainty;

println!("Total uncertainty: {:.1}%", uncertainty.total_uncertainty * 100.0);
println!("Confidence interval: {:.1}% - {:.1}%", 
         uncertainty.confidence_interval.0 * 100.0,
         uncertainty.confidence_interval.1 * 100.0);

for source in &uncertainty.uncertainty_sources {
    println!("Uncertainty from {}: {:.1}% - {}", 
             source.source, 
             source.contribution * 100.0, 
             source.description);
}
```

## Configuration Types

### `VisionConfig`

Configuration for vision-based analysis.

```rust
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub enable_face_detection: bool,
    pub enable_micro_expressions: bool,
    pub enable_eye_tracking: bool,
    pub model_precision: ModelPrecision,
    pub frame_rate: Option<f32>,
    pub resolution: Option<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub enum ModelPrecision {
    Fast,      // Lower precision, faster processing
    Balanced,  // Good balance of speed and accuracy
    Accurate,  // Highest precision, slower processing
}
```

### `AudioConfig`

Configuration for audio-based analysis.

```rust
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub enable_pitch_analysis: bool,
    pub enable_stress_detection: bool,
    pub enable_voice_quality: bool,
    pub chunk_size_ms: u32,
    pub noise_reduction: bool,
}
```

### `TextConfig`

Configuration for text-based analysis.

```rust
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub model_type: TextModel,
    pub enable_linguistic_analysis: bool,
    pub enable_sentiment_analysis: bool,
    pub language: Language,
    pub max_sequence_length: usize,
}

#[derive(Debug, Clone)]
pub enum TextModel {
    Bert,
    RoBerta,
    DistilBert,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Auto,  // Automatic detection
}
```

## Metadata Types

### `InputMetadata`

Metadata associated with analysis input.

```rust
#[derive(Debug, Clone)]
pub struct InputMetadata {
    pub timestamp: Option<SystemTime>,
    pub session_id: Option<String>,
    pub subject_id: Option<String>,
    pub context: Option<String>,
    pub environment: Environment,
    pub custom_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum Environment {
    Controlled,    // Laboratory setting
    Field,         // Real-world environment
    Simulated,     // Testing/simulation
}
```

### `ResultMetadata`

Metadata associated with analysis results.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ResultMetadata {
    pub analysis_id: String,
    pub timestamp: SystemTime,
    pub version: String,
    pub model_versions: HashMap<String, String>,
    pub processing_node: Option<String>,
    pub custom_fields: HashMap<String, serde_json::Value>,
}
```

## Error Types

### `VeritasError`

Main error type for the Veritas Nexus system.

```rust
#[derive(Debug, thiserror::Error)]
pub enum VeritasError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Timeout error: operation took longer than {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },
    
    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),
}
```

#### Usage Example

```rust
match detector.analyze(&input).await {
    Ok(result) => {
        // Handle successful result
    }
    Err(VeritasError::InvalidInput(msg)) => {
        eprintln!("Invalid input provided: {}", msg);
        // Handle input validation error
    }
    Err(VeritasError::ModelError(msg)) => {
        eprintln!("Model processing failed: {}", msg);
        // Handle model errors (e.g., model not loaded)
    }
    Err(VeritasError::TimeoutError { timeout_ms }) => {
        eprintln!("Analysis timed out after {}ms", timeout_ms);
        // Handle timeout scenarios
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
        // Handle other errors
    }
}
```

## Type Aliases

Common type aliases used throughout the API:

```rust
/// Standard result type for the library
pub type Result<T> = std::result::Result<T, VeritasError>;

/// Feature vector type
pub type FeatureVector = Vec<f32>;

/// Confidence score type (0.0 to 1.0)
pub type Confidence = f32;

/// Probability type (0.0 to 1.0) 
pub type Probability = f32;

/// Timestamp type
pub type Timestamp = std::time::SystemTime;
```

## Constants

Important constants used in the API:

```rust
/// Default confidence threshold for making decisions
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Maximum supported video resolution
pub const MAX_VIDEO_RESOLUTION: (u32, u32) = (1920, 1080);

/// Default audio sample rate
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Maximum text length for analysis
pub const MAX_TEXT_LENGTH: usize = 10000;

/// Default processing timeout in milliseconds
pub const DEFAULT_TIMEOUT_MS: u64 = 30000;
```

## Best Practices

### Error Handling

Always handle potential errors appropriately:

```rust
// Good: Explicit error handling
match detector.analyze(&input).await {
    Ok(result) => process_result(result),
    Err(e) => handle_error(e),
}

// Better: Use the ? operator for propagation
let result = detector.analyze(&input).await?;
process_result(result);
```

### Resource Management

The library handles resource management automatically, but for optimal performance:

```rust
// Reuse detector instances when possible
let detector = LieDetector::new().await?;

for input in inputs {
    let result = detector.analyze(&input).await?;
    // Process result
}

// Use batch processing for multiple inputs
let results = detector.analyze_batch(&inputs).await?;
```

### Configuration

Use appropriate configurations for your use case:

```rust
// For real-time applications
let detector = LieDetector::builder()
    .with_model_precision(ModelPrecision::Fast)
    .with_memory_limit_mb(512)
    .build()
    .await?;

// For high-accuracy analysis
let detector = LieDetector::builder()
    .with_model_precision(ModelPrecision::Accurate)
    .with_gpu_acceleration(true)
    .build()
    .await?;
```