//! Convenient re-exports for common Veritas Nexus functionality.
//!
//! This module provides a curated set of the most commonly used types and traits
//! from across the Veritas Nexus crate. Import this module to get started quickly:
//!
//! ```rust
//! use veritas_nexus::prelude::*;
//! ```
//!
//! # What's Included
//!
//! ## Core Types
//! - [`LieDetector`] - Main entry point for lie detection
//! - [`LieDetectorBuilder`] - Builder for configuring the detector
//! - [`AnalysisInput`] - Input data for analysis
//! - [`AnalysisResult`] - Result of analysis
//! - [`Decision`] - Final decision classification
//! - [`ModalityScores`] - Individual modality scores
//!
//! ## Configuration Types
//! - [`VisionConfig`] - Vision analysis configuration
//! - [`AudioConfig`] - Audio analysis configuration
//! - [`TextConfig`] - Text analysis configuration
//! - [`ReActConfig`] - Reasoning engine configuration
//!
//! ## Error Handling
//! - [`Result`] - Standard result type with [`VeritasError`]
//! - [`VeritasError`] - Main error type for the crate
//! - [`Context`] - Error context extension trait
//!
//! ## Common Traits
//! - [`Float`] - Trait for floating point operations
//! - [`Serialize`] / [`Deserialize`] - Serde serialization traits
//! - [`Send`] / [`Sync`] - Thread safety markers
//!
//! ## Core Type Aliases
//! - [`Float32`] / [`Float64`] - Floating point type aliases
//! - [`Timestamp`] - Time representation
//! - [`Confidence`] - Confidence score type
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```rust,no_run
//! use veritas_nexus::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let detector = LieDetector::builder()
//!         .with_vision(VisionConfig::default())
//!         .build()?;
//!
//!     let result = detector.analyze(AnalysisInput {
//!         video_path: Some("interview.mp4"),
//!         audio_path: Some("interview.wav"),
//!         transcript: Some("I didn't do anything wrong"),
//!         physiological: None,
//!     }).await?;
//!
//!     match result.decision {
//!         Decision::Truthful => println!("Statement appears truthful"),
//!         Decision::Deceptive => println!("Statement appears deceptive"),
//!         Decision::Uncertain => println!("Insufficient data for decision"),
//!     }
//!
//!     Ok(())
//! }
//! ```

// Re-export the main public API from the crate root
// Commenting out non-existent types temporarily
// pub use crate::{
//     AnalysisInput, AnalysisResult, AudioConfig, Decision, LieDetector, LieDetectorBuilder,
//     ModalityScores, ReActConfig, TextConfig, VisionConfig, DESCRIPTION, NAME, VERSION,
// };

// Re-export error types
pub use crate::error::{Result, VeritasError};

// Re-export core types
pub use crate::types::{Confidence, Float32, Float64, Timestamp};

// Re-export commonly used external dependencies
pub use anyhow::Context;
pub use num_traits::Float;
pub use serde::{Deserialize, Serialize};
pub use uuid::Uuid;

// Re-export modality traits (when implemented)
// #[cfg(feature = "default")]
// pub use crate::modalities::{ModalityAnalyzer, ModalityResult};

// Re-export fusion types (when implemented)
// #[cfg(feature = "default")]
// pub use crate::fusion::{FusionResult, FusionStrategy};

// Re-export agent types (when implemented)
// #[cfg(feature = "default")]
// pub use crate::agents::{ReactAgent, ReasoningTrace};

// Re-export streaming types (when implemented)
// #[cfg(feature = "default")]
// pub use crate::streaming::{StreamingPipeline, StreamingResult};

// Re-export MCP types when MCP feature is enabled
#[cfg(feature = "mcp")]
pub use crate::mcp::{McpServer, McpTool};

// Re-export optimization types for performance-critical code
#[cfg(feature = "cpu-optimized")]
pub use crate::optimization::{MemoryPool, SimdOps};

// Re-export GPU types when GPU features are enabled
#[cfg(feature = "gpu")]
pub use crate::optimization::gpu::{GpuDevice, GpuMemoryPool};

// Re-export profiling utilities when profiling is enabled
#[cfg(feature = "profiling")]
pub use crate::utils::profiling::{ProfileScope, Profiler};

// Re-export common standard library types for convenience
pub use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

// Re-export common async types
pub use tokio::{sync::mpsc, task::JoinHandle, time::sleep};

/// Type alias for the standard Result type used throughout the crate.
///
/// This is equivalent to `std::result::Result<T, VeritasError>`.
pub type VeritasResult<T> = std::result::Result<T, VeritasError>;

// Common type aliases for working with numerical data
/// Single-precision floating point number
pub type f32 = std::primitive::f32;
/// Double-precision floating point number  
pub type f64 = std::primitive::f64;

// Vector and matrix type aliases for numerical computing
pub use nalgebra::{DMatrix, DVector, Matrix4, Vector3, Vector4};
pub use ndarray::{Array1, Array2, Array3, ArrayD};

/// Re-export commonly used mathematical constants
pub mod math {
    pub use std::f64::consts::{E, PI, TAU};
    
    /// Golden ratio (φ)
    pub const PHI: f64 = 1.618033988749894;
    
    /// Square root of 2
    pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
    
    /// Natural logarithm of 2
    pub const LN_2: f64 = std::f64::consts::LN_2;
}

/// Re-export common time utilities
pub mod time {
    pub use chrono::{DateTime, Duration, NaiveDateTime, Utc};
    pub use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};
    
    /// Get current UTC timestamp
    pub fn now_utc() -> DateTime<Utc> {
        Utc::now()
    }
    
    /// Convert milliseconds to Duration
    pub fn millis(ms: u64) -> StdDuration {
        StdDuration::from_millis(ms)
    }
    
    /// Convert seconds to Duration
    pub fn seconds(secs: u64) -> StdDuration {
        StdDuration::from_secs(secs)
    }
}

/// Re-export common async utilities
pub mod async_utils {
    pub use futures::{
        future::{join, join_all, select, try_join, try_join_all},
        stream::{Stream, StreamExt},
        Future, FutureExt,
    };
    pub use tokio::{
        spawn,
        sync::{broadcast, mpsc, oneshot, watch},
        time::{interval, sleep, timeout},
    };
    
    /// Type alias for async result
    pub type AsyncResult<T> = std::pin::Pin<Box<dyn Future<Output = super::VeritasResult<T>> + Send>>;
}

/// Re-export common serialization utilities
pub mod serde_utils {
    pub use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};
    pub use serde_json::{from_str, to_string, to_string_pretty, to_vec, Value};
    
    /// Serialize to JSON string with pretty formatting
    pub fn to_json_pretty<T: Serialize>(value: &T) -> crate::error::Result<String> {
        to_string_pretty(value).map_err(|e| crate::error::VeritasError::Serialization(e.to_string()))
    }
    
    /// Deserialize from JSON string
    pub fn from_json<T: DeserializeOwned>(json: &str) -> crate::error::Result<T> {
        from_str(json).map_err(|e| crate::error::VeritasError::Serialization(e.to_string()))
    }
}

/// Re-export logging and tracing utilities
pub mod logging {
    pub use tracing::{debug, error, info, instrument, span, trace, warn, Level, Span};
    pub use tracing_subscriber::{
        filter::EnvFilter, fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt,
        Registry,
    };
    
    /// Initialize default logging for the application
    pub fn init_default_logging() {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_span_events(FmtSpan::CLOSE),
            )
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
            .init();
    }
}

// Conditional re-exports based on feature flags

/// GPU-specific utilities (only available with 'gpu' feature)
#[cfg(feature = "gpu")]
pub mod gpu {
    pub use candle_core::{Device, Tensor};
    pub use candle_nn::Module;
}

/// Computer vision utilities (only available with 'vision' feature)
#[cfg(feature = "vision")]
pub mod vision {
    pub use image::{ImageBuffer, Rgb, RgbImage};
    pub use imageproc::contours::Contour;
}

/// Web and WASM utilities (only available with 'web' feature)
#[cfg(feature = "web")]
pub mod web {
    pub use js_sys::*;
    pub use wasm_bindgen::prelude::*;
    pub use web_sys::*;
}

/// Testing utilities (only available with 'testing' feature)
#[cfg(feature = "testing")]
pub mod testing {
    pub use mockall::{mock, predicate::*};
    pub use proptest::prelude::*;
    pub use tokio_test::{assert_err, assert_ok, assert_pending, assert_ready};
    
    /// Create a temporary directory for testing
    pub fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("Failed to create temp directory")
    }
}

// Placeholder exports for traits that will be implemented in later phases
// These will be uncommented as the corresponding modules are implemented

// pub use crate::modalities::vision::VisionAnalyzer;
// pub use crate::modalities::audio::AudioAnalyzer;
// pub use crate::modalities::text::TextAnalyzer;
// pub use crate::modalities::physiological::PhysiologicalAnalyzer;
// pub use crate::fusion::AttentionFusion;
// pub use crate::agents::ReActAgent;
// pub use crate::learning::GspoTrainer;
// pub use crate::reasoning::NeuroSymbolicReasoner;