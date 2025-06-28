//! Multi-modal analysis modules
//!
//! This module contains analyzers for different types of input modalities:
//! - Text: Linguistic analysis, BERT embeddings, deception patterns
//! - Vision: Facial analysis, micro-expressions, face detection
//! - Audio: Voice analysis, prosodic features 
//! - Physiological: Biometric signals (future)

pub mod text;
pub mod vision;
pub mod audio;

// Re-export main text components
pub use text::{
    TextAnalyzer, TextAnalyzerConfig, TextScore, TextInput, 
    Language, PreprocessingConfig, LinguisticFeatures, 
    SentimentResult, NamedEntity, EntityType, EmotionScore, AnalyzedToken,
    PosTag, ComplexityMetrics, TemporalPattern, CognitiveLoadIndicators,
    SemanticCoherence
};

// Re-export main vision components
pub use vision::{
    VisionAnalyzer, VisionConfig, VisionInput, VisionFeatures, VisionError,
    FaceAnalyzer, MicroExpressionDetector, MicroExpressionType,
    BoundingBox, VisionDeceptionScore, VisionExplanation
};

#[cfg(feature = "gpu")]
pub use vision::GpuVisionProcessor;

// Re-export main audio components
pub use audio::{
    AudioAnalyzer, AudioConfig, AudioFeatures, AudioError,
    VoiceAnalyzer, PitchDetector, PitchFeatures, StressDetector, StressFeatures,
    MfccExtractor, MfccFeatures, ComprehensiveAudioAnalyzer, AudioAnalyzerFactory,
    VoiceActivity, SpectralFeatures, EnergyFeatures, VoiceQuality,
    ProcessingStats
};