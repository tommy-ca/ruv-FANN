//! Error types for the Veritas Nexus lie detection system.
//!
//! This module provides a comprehensive error handling system that covers all
//! potential failure modes across the different components of the system.
//!
//! # Error Categories
//!
//! - **Input/Validation Errors**: Invalid input data or configuration
//! - **Modality Errors**: Failures in specific modality processors (vision, audio, text, etc.)
//! - **Fusion Errors**: Failures in combining modality results
//! - **Reasoning Errors**: Failures in the AI reasoning engine
//! - **I/O Errors**: File system and network-related failures
//! - **Performance Errors**: Resource exhaustion or performance bottlenecks
//! - **Serialization Errors**: Data format and encoding issues
//!
//! # Examples
//!
//! ```rust
//! use veritas_nexus::error::{VeritasError, Result};
//!
//! fn process_video(path: &str) -> Result<()> {
//!     if path.is_empty() {
//!         return Err(VeritasError::InvalidInput {
//!             message: "Video path cannot be empty".to_string(),
//!             parameter: "video_path".to_string(),
//!         });
//!     }
//!     Ok(())
//! }
//! ```

use std::fmt;
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Main error type for the Veritas Nexus system.
///
/// This enum covers all possible error conditions that can occur during
/// lie detection analysis, from input validation to neural network processing.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum VeritasError {
    /// Invalid input data or parameters
    #[error("Invalid input: {message} (parameter: {parameter})")]
    InvalidInput {
        /// Description of what's invalid
        message: String,
        /// Name of the invalid parameter
        parameter: String,
    },

    /// Configuration-related errors
    #[error("Configuration error: {message}")]
    Configuration {
        /// Description of the configuration issue
        message: String,
    },

    /// Vision processing errors
    #[error("Vision processing error: {message}")]
    Vision {
        /// Description of the vision error
        message: String,
        /// Optional source error
        #[source]
        #[serde(skip)]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Audio processing errors
    #[error("Audio processing error: {message}")]
    Audio {
        /// Description of the audio error
        message: String,
        /// Optional source error
        #[source]
        #[serde(skip)]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Text processing errors
    #[error("Text processing error: {message}")]
    Text {
        /// Description of the text error
        message: String,
        /// Optional source error
        #[source]
        #[serde(skip)]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Physiological signal processing errors
    #[error("Physiological processing error: {message}")]
    Physiological {
        /// Description of the physiological error
        message: String,
        /// Optional source error
        #[source]
        #[serde(skip)]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Fusion strategy errors
    #[error("Fusion error: {message}")]
    Fusion {
        /// Description of the fusion error
        message: String,
        /// Which modalities were being fused
        modalities: Vec<String>,
    },

    /// ReAct agent reasoning errors
    #[error("Reasoning error: {message}")]
    Reasoning {
        /// Description of the reasoning error
        message: String,
        /// Current reasoning step
        step: Option<String>,
    },

    /// Neural network processing errors
    #[error("Neural network error: {message}")]
    NeuralNetwork {
        /// Description of the neural network error
        message: String,
        /// Network component that failed
        component: String,
    },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    Memory {
        /// Description of the memory error
        message: String,
        /// Requested size in bytes
        requested_size: Option<usize>,
    },

    /// GPU/hardware acceleration errors
    #[error("GPU error: {message}")]
    Gpu {
        /// Description of the GPU error
        message: String,
        /// GPU device identifier
        device_id: Option<u32>,
    },

    /// File I/O errors
    #[error("I/O error: {message}")]
    Io {
        /// Description of the I/O error
        message: String,
        /// File path involved
        path: Option<String>,
        /// Underlying I/O error
        #[source]
        #[serde(skip)]
        source: Option<std::io::Error>,
    },

    /// Network-related errors
    #[error("Network error: {message}")]
    Network {
        /// Description of the network error
        message: String,
        /// URL or endpoint involved
        endpoint: Option<String>,
    },

    /// Serialization and deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Database operation errors
    #[error("Database error: {message}")]
    Database {
        /// Description of the database error
        message: String,
        /// Query or operation that failed
        operation: Option<String>,
    },

    /// MCP server errors
    #[error("MCP error: {message}")]
    Mcp {
        /// Description of the MCP error
        message: String,
        /// Tool or resource involved
        component: Option<String>,
    },

    /// Timeout errors
    #[error("Timeout error: operation '{operation}' exceeded {duration_ms}ms")]
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Timeout duration in milliseconds
        duration_ms: u64,
    },

    /// Resource exhaustion errors
    #[error("Resource exhausted: {resource} - {message}")]
    ResourceExhausted {
        /// Type of resource exhausted
        resource: String,
        /// Description of the exhaustion
        message: String,
    },

    /// Concurrency and threading errors
    #[error("Concurrency error: {message}")]
    Concurrency {
        /// Description of the concurrency error
        message: String,
    },

    /// Model loading and inference errors
    #[error("Model error: {message}")]
    Model {
        /// Description of the model error
        message: String,
        /// Model name or path
        model_name: Option<String>,
    },

    /// Training and learning errors
    #[error("Training error: {message}")]
    Training {
        /// Description of the training error
        message: String,
        /// Training epoch or iteration
        epoch: Option<u32>,
    },

    /// Feature extraction errors
    #[error("Feature extraction error: {message}")]
    FeatureExtraction {
        /// Description of the feature extraction error
        message: String,
        /// Modality being processed
        modality: String,
    },

    /// Validation and testing errors
    #[error("Validation error: {message}")]
    Validation {
        /// Description of the validation error
        message: String,
        /// Validation metric that failed
        metric: Option<String>,
    },

    /// Internal system errors (should be rare)
    #[error("Internal error: {message}")]
    Internal {
        /// Description of the internal error
        message: String,
        /// Location in code where error occurred
        location: Option<String>,
    },

    /// External dependency errors
    #[error("External dependency error: {dependency} - {message}")]
    ExternalDependency {
        /// Name of the external dependency
        dependency: String,
        /// Description of the error
        message: String,
    },

    /// Version compatibility errors
    #[error("Version compatibility error: {message}")]
    VersionCompatibility {
        /// Description of the compatibility issue
        message: String,
        /// Expected version
        expected: Option<String>,
        /// Actual version
        actual: Option<String>,
    },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {operation} - {current}/{limit} requests in {window_ms}ms")]
    RateLimitExceeded {
        /// Operation being rate limited
        operation: String,
        /// Current request count
        current: u64,
        /// Maximum allowed requests
        limit: u64,
        /// Time window in milliseconds
        window_ms: u64,
        /// Time until reset
        reset_in_ms: u64,
    },

    /// Backpressure errors when system is overloaded
    #[error("Backpressure triggered: {message} - queue size: {queue_size}, latency: {latency_ms}ms")]
    Backpressure {
        /// Description of backpressure condition
        message: String,
        /// Current queue size
        queue_size: usize,
        /// Current latency in milliseconds
        latency_ms: u64,
        /// Recommended retry delay
        retry_after_ms: u64,
    },

    /// Circuit breaker errors when a service is temporarily unavailable
    #[error("Circuit breaker {name} is {state}: {reason}")]
    CircuitBreaker {
        /// Circuit breaker name
        name: String,
        /// Current state (open/half-open/closed)
        state: String,
        /// Reason for current state
        reason: String,
        /// Time until next retry attempt
        retry_after_ms: u64,
    },

    /// Resource pool exhaustion
    #[error("Resource pool exhausted: {pool_type} - {used}/{total} resources in use")]
    ResourcePoolExhausted {
        /// Type of resource pool
        pool_type: String,
        /// Resources currently in use
        used: usize,
        /// Total available resources
        total: usize,
        /// Wait time for resource availability
        wait_time_ms: u64,
    },

    /// Graceful degradation when modalities fail
    #[error("Modality degradation: {failed_modalities:?} failed, using {active_modalities:?}")]
    ModalityDegradation {
        /// List of failed modalities
        failed_modalities: Vec<String>,
        /// List of still active modalities
        active_modalities: Vec<String>,
        /// Impact on overall confidence
        confidence_impact: f64,
    },

    /// Data quality issues
    #[error("Data quality issue: {metric} is {value}, threshold is {threshold}")]
    DataQuality {
        /// Quality metric name
        metric: String,
        /// Actual metric value
        value: f64,
        /// Quality threshold
        threshold: f64,
        /// Severity level
        severity: DataQualitySeverity,
    },

    /// Edge case handling for extreme or unusual values
    #[error("Edge case detected: {case_type} - {description}")]
    EdgeCase {
        /// Type of edge case
        case_type: String,
        /// Description of the specific case
        description: String,
        /// Whether the case can be handled gracefully
        can_handle: bool,
        /// Suggested fallback action
        fallback_action: Option<String>,
    },

    /// Recovery operation failures
    #[error("Recovery failed: {operation} attempt {attempt}/{max_attempts} - {reason}")]
    RecoveryFailed {
        /// Recovery operation type
        operation: String,
        /// Current attempt number
        attempt: u32,
        /// Maximum attempts allowed
        max_attempts: u32,
        /// Reason for failure
        reason: String,
        /// Whether to retry
        should_retry: bool,
    },

    /// Malformed input data
    #[error("Malformed input: {input_type} - {validation_errors:?}")]
    MalformedInput {
        /// Type of input that was malformed
        input_type: String,
        /// List of validation errors
        validation_errors: Vec<String>,
        /// Partial data that could be salvaged
        salvageable_fields: Vec<String>,
    },

    /// System health degradation
    #[error("System health degraded: {component} health is {status} - {metrics:?}")]
    HealthDegraded {
        /// Component with degraded health
        component: String,
        /// Current health status
        status: String,
        /// Health metrics
        metrics: std::collections::HashMap<String, f64>,
        /// Remediation actions taken
        actions_taken: Vec<String>,
    },
    
    /// Streaming/processing errors
    #[error("Stream error: {0}")]
    StreamError(String),
    
    /// I/O error variant (alias for consistency)
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Fusion error variant (alias for consistency)
    #[error("Fusion error: {0}")]
    FusionError(String),
}

/// Data quality severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataQualitySeverity {
    /// Low severity - data is usable but may affect accuracy
    Low,
    /// Medium severity - data quality issues that should be addressed
    Medium,
    /// High severity - data quality issues that significantly impact results
    High,
    /// Critical severity - data is unusable or dangerous to use
    Critical,
}

impl fmt::Display for DataQualitySeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Result type alias for Veritas Nexus operations.
pub type Result<T> = std::result::Result<T, VeritasError>;

/// Type alias for fusion errors to maintain backward compatibility
pub type FusionError = VeritasError;

impl VeritasError {
    /// Creates a new input validation error.
    pub fn invalid_input(message: impl Into<String>, parameter: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
            parameter: parameter.into(),
        }
    }

    /// Creates a new configuration error.
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Creates a new vision processing error.
    pub fn vision_error(message: impl Into<String>) -> Self {
        Self::Vision {
            message: message.into(),
            source: None,
        }
    }

    /// Creates a new vision processing error with source.
    pub fn vision_error_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Vision {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Creates a new audio processing error.
    pub fn audio_error(message: impl Into<String>) -> Self {
        Self::Audio {
            message: message.into(),
            source: None,
        }
    }

    /// Creates a new text processing error.
    pub fn text_error(message: impl Into<String>) -> Self {
        Self::Text {
            message: message.into(),
            source: None,
        }
    }

    /// Creates a new fusion error.
    pub fn fusion_error(message: impl Into<String>, modalities: Vec<String>) -> Self {
        Self::Fusion {
            message: message.into(),
            modalities,
        }
    }

    /// Creates a new reasoning error.
    pub fn reasoning_error(message: impl Into<String>) -> Self {
        Self::Reasoning {
            message: message.into(),
            step: None,
        }
    }

    /// Creates a new reasoning error with step information.
    pub fn reasoning_error_with_step(
        message: impl Into<String>,
        step: impl Into<String>,
    ) -> Self {
        Self::Reasoning {
            message: message.into(),
            step: Some(step.into()),
        }
    }

    /// Creates a new neural network error.
    pub fn neural_network_error(
        message: impl Into<String>,
        component: impl Into<String>,
    ) -> Self {
        Self::NeuralNetwork {
            message: message.into(),
            component: component.into(),
        }
    }

    /// Creates a new memory error.
    pub fn memory_error(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
            requested_size: None,
        }
    }

    /// Creates a new memory error with size information.
    pub fn memory_error_with_size(message: impl Into<String>, size: usize) -> Self {
        Self::Memory {
            message: message.into(),
            requested_size: Some(size),
        }
    }

    /// Creates a new I/O error.
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::Io {
            message: message.into(),
            path: None,
            source: None,
        }
    }

    /// Creates a new I/O error with path and source.
    pub fn io_error_with_path(
        message: impl Into<String>,
        path: impl Into<String>,
        source: std::io::Error,
    ) -> Self {
        Self::Io {
            message: message.into(),
            path: Some(path.into()),
            source: Some(source),
        }
    }

    /// Creates a new timeout error.
    pub fn timeout_error(operation: impl Into<String>, duration_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration_ms,
        }
    }

    /// Creates a new internal error with location.
    pub fn internal_error_with_location(
        message: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self::Internal {
            message: message.into(),
            location: Some(location.into()),
        }
    }

    /// Creates a new rate limit exceeded error.
    pub fn rate_limit_exceeded(
        operation: impl Into<String>,
        current: u64,
        limit: u64,
        window_ms: u64,
        reset_in_ms: u64,
    ) -> Self {
        Self::RateLimitExceeded {
            operation: operation.into(),
            current,
            limit,
            window_ms,
            reset_in_ms,
        }
    }

    /// Creates a new backpressure error.
    pub fn backpressure(
        message: impl Into<String>,
        queue_size: usize,
        latency_ms: u64,
        retry_after_ms: u64,
    ) -> Self {
        Self::Backpressure {
            message: message.into(),
            queue_size,
            latency_ms,
            retry_after_ms,
        }
    }

    /// Creates a new circuit breaker error.
    pub fn circuit_breaker(
        name: impl Into<String>,
        state: impl Into<String>,
        reason: impl Into<String>,
        retry_after_ms: u64,
    ) -> Self {
        Self::CircuitBreaker {
            name: name.into(),
            state: state.into(),
            reason: reason.into(),
            retry_after_ms,
        }
    }

    /// Creates a new resource pool exhausted error.
    pub fn resource_pool_exhausted(
        pool_type: impl Into<String>,
        used: usize,
        total: usize,
        wait_time_ms: u64,
    ) -> Self {
        Self::ResourcePoolExhausted {
            pool_type: pool_type.into(),
            used,
            total,
            wait_time_ms,
        }
    }

    /// Creates a new modality degradation error.
    pub fn modality_degradation(
        failed_modalities: Vec<String>,
        active_modalities: Vec<String>,
        confidence_impact: f64,
    ) -> Self {
        Self::ModalityDegradation {
            failed_modalities,
            active_modalities,
            confidence_impact,
        }
    }

    /// Creates a new data quality error.
    pub fn data_quality(
        metric: impl Into<String>,
        value: f64,
        threshold: f64,
        severity: DataQualitySeverity,
    ) -> Self {
        Self::DataQuality {
            metric: metric.into(),
            value,
            threshold,
            severity,
        }
    }

    /// Creates a new edge case error.
    pub fn edge_case(
        case_type: impl Into<String>,
        description: impl Into<String>,
        can_handle: bool,
        fallback_action: Option<String>,
    ) -> Self {
        Self::EdgeCase {
            case_type: case_type.into(),
            description: description.into(),
            can_handle,
            fallback_action,
        }
    }

    /// Creates a new recovery failed error.
    pub fn recovery_failed(
        operation: impl Into<String>,
        attempt: u32,
        max_attempts: u32,
        reason: impl Into<String>,
        should_retry: bool,
    ) -> Self {
        Self::RecoveryFailed {
            operation: operation.into(),
            attempt,
            max_attempts,
            reason: reason.into(),
            should_retry,
        }
    }

    /// Creates a new malformed input error.
    pub fn malformed_input(
        input_type: impl Into<String>,
        validation_errors: Vec<String>,
        salvageable_fields: Vec<String>,
    ) -> Self {
        Self::MalformedInput {
            input_type: input_type.into(),
            validation_errors,
            salvageable_fields,
        }
    }

    /// Creates a new health degraded error.
    pub fn health_degraded(
        component: impl Into<String>,
        status: impl Into<String>,
        metrics: std::collections::HashMap<String, f64>,
        actions_taken: Vec<String>,
    ) -> Self {
        Self::HealthDegraded {
            component: component.into(),
            status: status.into(),
            metrics,
            actions_taken,
        }
    }

    /// Returns true if this is a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. }
                | Self::Network { .. }
                | Self::Memory { .. }
                | Self::ResourceExhausted { .. }
                | Self::Concurrency { .. }
                | Self::RateLimitExceeded { .. }
                | Self::Backpressure { .. }
                | Self::CircuitBreaker { .. }
                | Self::ResourcePoolExhausted { .. }
                | Self::ModalityDegradation { .. }
                | Self::RecoveryFailed { should_retry: true, .. }
                | Self::DataQuality { severity: DataQualitySeverity::Low | DataQualitySeverity::Medium, .. }
                | Self::EdgeCase { can_handle: true, .. }
                | Self::StreamError(_)
                | Self::IoError(_)
        )
    }

    /// Returns true if this error is related to user input.
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidInput { .. } 
                | Self::Configuration { .. } 
                | Self::Validation { .. }
                | Self::MalformedInput { .. }
                | Self::DataQuality { severity: DataQualitySeverity::Critical, .. }
        )
    }

    /// Returns true if this error is a system/internal error.
    pub fn is_system_error(&self) -> bool {
        matches!(
            self,
            Self::Internal { .. }
                | Self::Memory { .. }
                | Self::ResourceExhausted { .. }
                | Self::ExternalDependency { .. }
                | Self::ResourcePoolExhausted { .. }
                | Self::CircuitBreaker { .. }
                | Self::HealthDegraded { .. }
                | Self::RecoveryFailed { .. }
        )
    }

    /// Returns the error category as a string.
    pub fn category(&self) -> &'static str {
        match self {
            Self::InvalidInput { .. } | Self::Configuration { .. } | Self::Validation { .. } 
            | Self::MalformedInput { .. } => {
                "user_input"
            }
            Self::Vision { .. }
            | Self::Audio { .. }
            | Self::Text { .. }
            | Self::Physiological { .. } 
            | Self::ModalityDegradation { .. } => "modality_processing",
            Self::Fusion { .. } | Self::FusionError(_) => "fusion",
            Self::Reasoning { .. } => "reasoning",
            Self::NeuralNetwork { .. } | Self::Model { .. } | Self::Training { .. } => {
                "machine_learning"
            }
            Self::Memory { .. } | Self::Gpu { .. } | Self::ResourceExhausted { .. }
            | Self::ResourcePoolExhausted { .. } => "resources",
            Self::Io { .. } | Self::Network { .. } | Self::Database { .. } | Self::IoError(_) => "io",
            Self::Serialization(_) => "serialization",
            Self::Mcp { .. } => "mcp",
            Self::Timeout { .. } | Self::Concurrency { .. } | Self::StreamError(_) => "concurrency",
            Self::FeatureExtraction { .. } => "feature_extraction",
            Self::Internal { .. } | Self::ExternalDependency { .. } | Self::RecoveryFailed { .. }
            | Self::HealthDegraded { .. } => "system",
            Self::VersionCompatibility { .. } => "compatibility",
            Self::RateLimitExceeded { .. } | Self::Backpressure { .. } => "rate_limiting",
            Self::CircuitBreaker { .. } => "circuit_breaker",
            Self::DataQuality { .. } => "data_quality",
            Self::EdgeCase { .. } => "edge_case",
        }
    }

    /// Returns true if this error should trigger immediate alerts
    pub fn requires_immediate_alert(&self) -> bool {
        matches!(
            self,
            Self::DataQuality { severity: DataQualitySeverity::Critical, .. }
                | Self::HealthDegraded { .. }
                | Self::Memory { .. }
                | Self::ResourcePoolExhausted { .. }
                | Self::CircuitBreaker { .. }
                | Self::Internal { .. }
        )
    }

    /// Returns the suggested retry delay in milliseconds for recoverable errors
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            Self::RateLimitExceeded { reset_in_ms, .. } => Some(*reset_in_ms),
            Self::Backpressure { retry_after_ms, .. } => Some(*retry_after_ms),
            Self::CircuitBreaker { retry_after_ms, .. } => Some(*retry_after_ms),
            Self::ResourcePoolExhausted { wait_time_ms, .. } => Some(*wait_time_ms),
            Self::Timeout { .. } => Some(1000), // 1 second default
            Self::Network { .. } => Some(5000), // 5 seconds for network issues
            Self::Concurrency { .. } => Some(100), // 100ms for concurrency issues
            Self::StreamError(_) => Some(500), // 500ms for stream errors
            Self::IoError(_) => Some(1000), // 1 second for I/O errors
            Self::FusionError(_) => Some(200), // 200ms for fusion retry
            _ => None,
        }
    }

    /// Returns the severity level of this error (0-10 scale)
    pub fn severity_level(&self) -> u8 {
        match self {
            // Critical errors (9-10)
            Self::Internal { .. } | Self::Memory { .. } => 10,
            Self::DataQuality { severity: DataQualitySeverity::Critical, .. } => 9,
            Self::HealthDegraded { .. } => 9,

            // High severity errors (7-8)
            Self::ResourcePoolExhausted { .. } => 8,
            Self::CircuitBreaker { .. } => 7,
            Self::RecoveryFailed { should_retry: false, .. } => 7,

            // Medium severity errors (4-6)
            Self::DataQuality { severity: DataQualitySeverity::High, .. } => 6,
            Self::ModalityDegradation { .. } => 5,
            Self::Backpressure { .. } => 5,
            Self::Timeout { .. } => 4,

            // Low severity errors (1-3)
            Self::RateLimitExceeded { .. } => 3,
            Self::DataQuality { severity: DataQualitySeverity::Medium, .. } => 3,
            Self::EdgeCase { can_handle: true, .. } => 2,
            Self::DataQuality { severity: DataQualitySeverity::Low, .. } => 1,

            // I/O and streaming errors
            Self::StreamError(_) => 4,
            Self::IoError(_) => 4,
            Self::FusionError(_) => 5,
            
            // Default for other errors
            _ => 5,
        }
    }

    /// Returns the recommended action for handling this error
    pub fn recommended_action(&self) -> ErrorAction {
        match self {
            Self::RateLimitExceeded { .. } => ErrorAction::RetryAfterDelay,
            Self::Backpressure { .. } => ErrorAction::ReduceLoad,
            Self::CircuitBreaker { .. } => ErrorAction::RetryAfterDelay,
            Self::ResourcePoolExhausted { .. } => ErrorAction::WaitForResources,
            Self::ModalityDegradation { .. } => ErrorAction::ContinueWithDegradedService,
            Self::DataQuality { severity: DataQualitySeverity::Critical, .. } => ErrorAction::RejectInput,
            Self::DataQuality { .. } => ErrorAction::WarnAndContinue,
            Self::EdgeCase { can_handle: true, .. } => ErrorAction::UseFailsafe,
            Self::EdgeCase { can_handle: false, .. } => ErrorAction::RejectInput,
            Self::RecoveryFailed { should_retry: true, .. } => ErrorAction::RetryWithBackoff,
            Self::RecoveryFailed { should_retry: false, .. } => ErrorAction::Alert,
            Self::HealthDegraded { .. } => ErrorAction::Alert,
            Self::Timeout { .. } => ErrorAction::RetryWithBackoff,
            Self::Network { .. } => ErrorAction::RetryWithBackoff,
            Self::Memory { .. } => ErrorAction::Alert,
            Self::Internal { .. } => ErrorAction::Alert,
            Self::StreamError(_) => ErrorAction::RetryWithBackoff,
            Self::IoError(_) => ErrorAction::RetryWithBackoff,
            Self::FusionError(_) => ErrorAction::RetryAfterDelay,
            _ => ErrorAction::Log,
        }
    }

    /// Checks if this error can be handled gracefully without interrupting the main process
    pub fn can_handle_gracefully(&self) -> bool {
        matches!(
            self,
            Self::ModalityDegradation { .. }
                | Self::DataQuality { severity: DataQualitySeverity::Low | DataQualitySeverity::Medium, .. }
                | Self::EdgeCase { can_handle: true, .. }
                | Self::RateLimitExceeded { .. }
                | Self::Backpressure { .. }
                | Self::StreamError(_)
        )
    }
}

/// Recommended actions for error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorAction {
    /// Log the error and continue
    Log,
    /// Retry the operation after a delay
    RetryAfterDelay,
    /// Retry with exponential backoff
    RetryWithBackoff,
    /// Wait for resources to become available
    WaitForResources,
    /// Reduce system load
    ReduceLoad,
    /// Continue with degraded service
    ContinueWithDegradedService,
    /// Use a failsafe/fallback mechanism
    UseFailsafe,
    /// Reject the input and request new input
    RejectInput,
    /// Warn the user but continue processing
    WarnAndContinue,
    /// Trigger immediate alert to operators
    Alert,
}

impl fmt::Display for ErrorAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Log => write!(f, "Log"),
            Self::RetryAfterDelay => write!(f, "Retry after delay"),
            Self::RetryWithBackoff => write!(f, "Retry with backoff"),
            Self::WaitForResources => write!(f, "Wait for resources"),
            Self::ReduceLoad => write!(f, "Reduce load"),
            Self::ContinueWithDegradedService => write!(f, "Continue with degraded service"),
            Self::UseFailsafe => write!(f, "Use failsafe"),
            Self::RejectInput => write!(f, "Reject input"),
            Self::WarnAndContinue => write!(f, "Warn and continue"),
            Self::Alert => write!(f, "Alert"),
        }
    }
}

// Implement conversion from common error types
impl From<std::io::Error> for VeritasError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            path: None,
            source: Some(err),
        }
    }
}

impl From<serde_json::Error> for VeritasError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<uuid::Error> for VeritasError {
    fn from(err: uuid::Error) -> Self {
        Self::Serialization(format!("UUID error: {}", err))
    }
}

impl From<tokio::time::error::Elapsed> for VeritasError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        Self::Timeout {
            operation: "unknown".to_string(),
            duration_ms: 0, // Cannot determine duration from Elapsed error
        }
    }
}

// Conversion from anyhow::Error for compatibility
impl From<anyhow::Error> for VeritasError {
    fn from(err: anyhow::Error) -> Self {
        // Try to downcast to known error types first
        if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
            return Self::Io {
                message: io_err.to_string(),
                path: None,
                source: Some(io_err.kind().into()),
            };
        }
        
        if let Some(json_err) = err.downcast_ref::<serde_json::Error>() {
            return Self::Serialization(json_err.to_string());
        }

        // Fallback to internal error
        Self::Internal {
            message: err.to_string(),
            location: None,
        }
    }
}

impl VeritasError {
    /// Creates an unsupported language error (helper for text modality)
    pub fn unsupported_language(language: impl Into<String>) -> Self {
        Self::invalid_input(
            format!("Unsupported language: {}", language.into()),
            "language"
        )
    }
    
    /// Creates a new stream error.
    pub fn stream_error(message: impl Into<String>) -> Self {
        Self::StreamError(message.into())
    }
    
    /// Creates a new fusion error.
    pub fn fusion_error_simple(message: impl Into<String>) -> Self {
        Self::FusionError(message.into())
    }
}

// Helper function to create a Result with context
pub fn with_context<T, F>(result: Result<T>, context: F) -> Result<T>
where
    F: FnOnce() -> String,
{
    result.map_err(|mut err| {
        // Add context to the error message where applicable
        match &mut err {
            VeritasError::InvalidInput { message, .. } => {
                *message = format!("{}: {}", context(), message);
            }
            VeritasError::Configuration { message } => {
                *message = format!("{}: {}", context(), message);
            }
            VeritasError::Vision { message, .. } => {
                *message = format!("{}: {}", context(), message);
            }
            VeritasError::Audio { message, .. } => {
                *message = format!("{}: {}", context(), message);
            }
            VeritasError::Text { message, .. } => {
                *message = format!("{}: {}", context(), message);
            }
            VeritasError::Fusion { message, .. } => {
                *message = format!("{}: {}", context(), message);
            }
            VeritasError::Reasoning { message, .. } => {
                *message = format!("{}: {}", context(), message);
            }
            _ => {} // Other error types don't need context modification
        }
        err
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = VeritasError::invalid_input("Invalid parameter", "test_param");
        assert!(matches!(err, VeritasError::InvalidInput { .. }));
        assert!(err.is_user_error());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_categorization() {
        let vision_err = VeritasError::vision_error("Test vision error");
        assert_eq!(vision_err.category(), "modality_processing");

        let timeout_err = VeritasError::timeout_error("test_op", 5000);
        assert_eq!(timeout_err.category(), "concurrency");
        assert!(timeout_err.is_recoverable());
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let veritas_err: VeritasError = io_err.into();
        assert!(matches!(veritas_err, VeritasError::Io { .. }));
    }

    #[test]
    fn test_error_serialization() {
        let err = VeritasError::invalid_input("Test message", "test_param");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("Test message"));
    }

    #[test]
    fn test_with_context() {
        let result: Result<()> = Err(VeritasError::vision_error("Original error"));
        let result_with_context = with_context(result, || "During video processing".to_string());
        
        match result_with_context.unwrap_err() {
            VeritasError::Vision { message, .. } => {
                assert!(message.contains("During video processing"));
                assert!(message.contains("Original error"));
            }
            _ => panic!("Expected Vision error"),
        }
    }
}