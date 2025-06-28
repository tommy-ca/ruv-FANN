//! # Veritas Nexus - Multi-Modal Lie Detection System
//!
//! A comprehensive lie detection system built on ruv-FANN neural networks,
//! featuring multi-modal analysis, explainable AI, and high-performance processing.
//!
//! ## Features
//!
//! - **Multi-Modal Analysis**: Vision, audio, text, and physiological signals
//! - **Advanced Fusion**: Early, late, hybrid, and attention-based fusion strategies
//! - **Explainable AI**: ReAct reasoning with complete decision traces
//! - **High Performance**: CPU-optimized with optional GPU acceleration
//! - **Temporal Alignment**: Synchronization across modalities
//! - **Neural Integration**: Built on ruv-FANN neural network foundation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use veritas_nexus::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let detector = LieDetector::builder()
//!         .with_vision(VisionConfig::default())
//!         .with_audio(AudioConfig::default())
//!         .with_text(TextConfig::default())
//!         .build()?;
//!
//!     let result = detector.analyze(AnalysisInput {
//!         video_path: Some("interview.mp4"),
//!         audio_path: Some("interview.wav"),
//!         transcript: Some("I did not take the money"),
//!         physiological: None,
//!     }).await?;
//!
//!     println!("Deception probability: {:.2}%", result.probability * 100.0);
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod error;
pub mod types;
pub mod fusion;
pub mod neural_integration;

/// Prelude module containing commonly used types and traits
pub mod prelude {
    pub use crate::error::*;
    pub use crate::types::*;
    pub use crate::fusion::*;
    pub use crate::neural_integration::*;
}

// Re-export key types at the crate root
pub use error::{VeritasError, Result};
pub use types::{ModalityType, DeceptionScore, AnalysisInput, AnalysisResult};
pub use fusion::{FusionStrategy, FusionResult, FusionManager};
pub use neural_integration::{NeuralFusion, NeuralFusionConfig};