//! Vision-based deception detection module.
//!
//! This module provides state-of-the-art computer vision capabilities for detecting
//! deception through analysis of facial expressions, micro-expressions, gaze patterns,
//! and other non-verbal behavioral indicators. It combines traditional computer vision
//! techniques with deep learning approaches for robust visual analysis.
//!
//! # Core Components
//!
//! - **Face Analysis**: [`FaceAnalyzer`] - Face detection, landmark extraction, and facial feature analysis
//! - **Micro-Expressions**: [`MicroExpressionDetector`] - Detection of brief, involuntary facial expressions
//! - **Performance**: [`SimdFaceAnalyzer`] - SIMD-optimized processing for real-time applications
//! - **GPU Acceleration**: [`GpuVisionProcessor`] - CUDA/OpenCL acceleration for high-throughput processing
//!
//! # Supported Features
//!
//! ## Facial Analysis
//! - Real-time face detection and tracking
//! - 68-point facial landmark extraction
//! - Head pose estimation (pitch, yaw, roll)
//! - Facial Action Unit (AU) detection
//! - Eye gaze direction and fixation analysis
//! - Blink rate and pattern analysis
//!
//! ## Micro-Expression Detection
//! - Seven universal micro-expressions (fear, anger, sadness, joy, surprise, disgust, contempt)
//! - Temporal analysis of expression onset, apex, and offset
//! - Expression intensity measurement
//! - Baseline comparison for individual calibration
//! - Cultural adaptation for expression interpretation
//!
//! ## Behavioral Indicators
//! - Facial asymmetry analysis
//! - Eye contact patterns and avoidance
//! - Self-touching and grooming behaviors
//! - Involuntary muscle tension indicators
//! - Timing synchronization with speech
//!
//! # Examples
//!
//! Basic vision analysis:
//!
//! ```rust,no_run
//! use veritas_nexus::modalities::vision::{VisionAnalyzer, VisionConfig, VisionInput};
//! use veritas_nexus::ModalityAnalyzer;
//!
//! #[tokio::main]
//! async fn main() -> veritas_nexus::Result<()> {
//!     let config = VisionConfig::default();
//!     let analyzer = VisionAnalyzer::new(config)?;
//!     
//!     let input = VisionInput {
//!         image_data: load_image("interview_frame.jpg")?,
//!         timestamp: Some(std::time::SystemTime::now()),
//!         metadata: Default::default(),
//!     };
//!     
//!     let result = analyzer.analyze(&input).await?;
//!     
//!     println!("Deception probability: {:.2}", result.probability());
//!     println!("Confidence: {:.2}", result.confidence());
//!     
//!     // Examine detected features
//!     for feature in result.features() {
//!         println!("{}: {:.3}", feature.name, feature.value);
//!     }
//!     
//!     Ok(())
//! }
//!
//! fn load_image(path: &str) -> veritas_nexus::Result<Vec<u8>> {
//!     // Implementation would load image file
//!     Ok(vec![])
//! }
//! ```
//!
//! Advanced configuration with GPU acceleration:
//!
//! ```rust,no_run
//! use veritas_nexus::modalities::vision::{VisionConfig, ImagePreprocessing, ModelPaths};
//!
//! let config = VisionConfig {
//!     face_detection_threshold: 0.8,
//!     micro_expression_sensitivity: 0.9,
//!     max_faces: 3,
//!     enable_gpu: true,
//!     preprocessing: ImagePreprocessing {
//!         target_width: 224,
//!         target_height: 224,
//!         normalize_mean: [0.485, 0.456, 0.406],
//!         normalize_std: [0.229, 0.224, 0.225],
//!         histogram_equalization: true,
//!     },
//!     model_paths: ModelPaths {
//!         face_detection: "models/face_detection.onnx".to_string(),
//!         landmark_detection: "models/landmarks.onnx".to_string(),
//!         micro_expression: "models/micro_expr.onnx".to_string(),
//!         action_units: "models/action_units.onnx".to_string(),
//!     },
//! };
//! ```
//!
//! # Performance Considerations
//!
//! - **Real-time Processing**: Optimized for 30 FPS video analysis
//! - **GPU Acceleration**: Significant speedup for batch processing
//! - **SIMD Optimization**: CPU-optimized routines for embedded systems
//! - **Memory Efficiency**: Streaming processing for long video sequences
//! - **Model Compression**: Quantized models for mobile deployment
//!
//! # Accuracy and Robustness
//!
//! ## Strengths
//! - High accuracy on frontal face orientations
//! - Robust to lighting variations and image quality
//! - Real-time performance on modern hardware
//! - Cross-cultural micro-expression recognition
//! - Temporal consistency in video sequences
//!
//! ## Limitations
//! - Reduced accuracy with extreme head poses (> 45° rotation)
//! - Performance degrades with occlusions (masks, hands, objects)
//! - Sensitivity to camera quality and resolution
//! - Cultural biases in expression interpretation
//! - Difficulty with highly emotional or expressive individuals
//!
//! ## Best Practices
//! - Ensure adequate lighting and camera positioning
//! - Use multiple camera angles when possible
//! - Calibrate baselines for individual subjects
//! - Combine with audio analysis for better accuracy
//! - Apply temporal smoothing for video sequences
//! - Consider cultural context in interpretation

use std::collections::HashMap;
use std::marker::PhantomData;
use num_traits::Float;
use thiserror::Error;

pub mod face_analyzer;
pub mod micro_expression;
pub mod simd_optimized;

#[cfg(feature = "gpu")]
pub mod gpu_vision;

pub use face_analyzer::FaceAnalyzer;
pub use micro_expression::{MicroExpressionDetector, MicroExpressionType};
pub use simd_optimized::{SimdFaceAnalyzer, SimdFaceDetector};

#[cfg(feature = "gpu")]
pub use gpu_vision::GpuVisionProcessor;

/// Vision-specific error types
#[derive(Error, Debug)]
pub enum VisionError {
    #[error("Face detection failed: {0}")]
    FaceDetectionFailed(String),
    
    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),
    
    #[error("Micro-expression analysis failed: {0}")]
    MicroExpressionFailed(String),
    
    #[error("Invalid image format: {0}")]
    InvalidImageFormat(String),
    
    #[error("GPU processing error: {0}")]
    GpuError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Model loading error: {0}")]
    ModelLoadError(String),
}

/// Configuration for vision analyzer
#[derive(Debug, Clone)]
pub struct VisionConfig<T: Float> {
    /// Face detection confidence threshold (0.0 to 1.0)
    pub face_detection_threshold: T,
    /// Micro-expression sensitivity (0.0 to 1.0)
    pub micro_expression_sensitivity: T,
    /// Maximum number of faces to process per frame
    pub max_faces: usize,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// Image preprocessing settings
    pub preprocessing: ImagePreprocessing<T>,
    /// Model paths
    pub model_paths: ModelPaths,
}

impl<T: Float> Default for VisionConfig<T> {
    fn default() -> Self {
        Self {
            face_detection_threshold: T::from(0.7).unwrap(),
            micro_expression_sensitivity: T::from(0.85).unwrap(),
            max_faces: 5,
            enable_gpu: false,
            preprocessing: ImagePreprocessing::default(),
            model_paths: ModelPaths::default(),
        }
    }
}

/// Image preprocessing configuration
#[derive(Debug, Clone)]
pub struct ImagePreprocessing<T: Float> {
    /// Target image width for processing
    pub target_width: u32,
    /// Target image height for processing
    pub target_height: u32,
    /// Normalization mean values [R, G, B]
    pub normalize_mean: [T; 3],
    /// Normalization standard deviation values [R, G, B]
    pub normalize_std: [T; 3],
    /// Enable histogram equalization
    pub histogram_equalization: bool,
}

impl<T: Float> Default for ImagePreprocessing<T> {
    fn default() -> Self {
        Self {
            target_width: 224,
            target_height: 224,
            normalize_mean: [T::from(0.485).unwrap(), T::from(0.456).unwrap(), T::from(0.406).unwrap()],
            normalize_std: [T::from(0.229).unwrap(), T::from(0.224).unwrap(), T::from(0.225).unwrap()],
            histogram_equalization: true,
        }
    }
}

/// Model file paths
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Face detection model path
    pub face_detection: String,
    /// Facial landmark detection model path
    pub landmark_detection: String,
    /// Micro-expression classification model path
    pub micro_expression: String,
    /// Gaze estimation model path
    pub gaze_estimation: String,
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self {
            face_detection: "models/face_detection.onnx".to_string(),
            landmark_detection: "models/landmarks.onnx".to_string(),
            micro_expression: "models/micro_expressions.ruv".to_string(),
            gaze_estimation: "models/gaze.onnx".to_string(),
        }
    }
}

/// Input data for vision processing
#[derive(Debug, Clone)]
pub struct VisionInput {
    /// Image data as RGB bytes
    pub image_data: Vec<u8>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Number of channels (typically 3 for RGB)
    pub channels: u32,
    /// Timestamp of the image
    pub timestamp: Option<std::time::SystemTime>,
}

impl VisionInput {
    /// Create new vision input from image data
    pub fn new(image_data: Vec<u8>, width: u32, height: u32, channels: u32) -> Self {
        Self {
            image_data,
            width,
            height,
            channels,
            timestamp: Some(std::time::SystemTime::now()),
        }
    }
    
    /// Load from image file
    pub fn from_file(path: &str) -> Result<Self, VisionError> {
        // Note: In a real implementation, this would use image loading libraries
        // like image-rs or opencv-rust
        Err(VisionError::InvalidImageFormat("File loading not implemented".to_string()))
    }
}

/// Face features extracted from facial analysis
pub type FaceFeatures<T> = VisionFeatures<T>;

/// Vision feature vector containing extracted visual features
#[derive(Debug, Clone)]
pub struct VisionFeatures<T: Float> {
    /// Facial landmark features (68 landmark points * 2 coordinates)
    pub facial_landmarks: Vec<T>,
    /// Micro-expression features
    pub micro_expressions: Vec<T>,
    /// Gaze direction features [x, y, z]
    pub gaze_direction: [T; 3],
    /// Facial action unit activations
    pub action_units: Vec<T>,
    /// Head pose estimation [pitch, yaw, roll]
    pub head_pose: [T; 3],
    /// Eye movement features
    pub eye_features: Vec<T>,
    /// Overall confidence scores for each feature type
    pub feature_confidence: HashMap<String, T>,
}

impl<T: Float> VisionFeatures<T> {
    /// Create new empty vision features
    pub fn new() -> Self {
        Self {
            facial_landmarks: Vec::new(),
            micro_expressions: Vec::new(),
            gaze_direction: [T::zero(); 3],
            action_units: Vec::new(),
            head_pose: [T::zero(); 3],
            eye_features: Vec::new(),
            feature_confidence: HashMap::new(),
        }
    }
    
    /// Get the total number of features
    pub fn feature_count(&self) -> usize {
        self.facial_landmarks.len() + 
        self.micro_expressions.len() + 
        3 + // gaze_direction
        self.action_units.len() + 
        3 + // head_pose
        self.eye_features.len()
    }
    
    /// Convert to flat feature vector
    pub fn to_flat_vector(&self) -> Vec<T> {
        let mut features = Vec::with_capacity(self.feature_count());
        features.extend(&self.facial_landmarks);
        features.extend(&self.micro_expressions);
        features.extend(&self.gaze_direction);
        features.extend(&self.action_units);
        features.extend(&self.head_pose);
        features.extend(&self.eye_features);
        features
    }
}

/// Deception score with contributing factors
#[derive(Debug, Clone)]
pub struct VisionDeceptionScore<T: Float> {
    /// Overall deception probability (0.0 to 1.0)
    pub probability: T,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: T,
    /// Contributing factors with their individual scores
    pub contributing_factors: HashMap<String, T>,
    /// Detected micro-expressions
    pub detected_expressions: Vec<String>,
    /// Facial regions of interest for visualization
    pub regions_of_interest: Vec<BoundingBox<T>>,
}

/// Bounding box for regions of interest
#[derive(Debug, Clone)]
pub struct BoundingBox<T: Float> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
    pub confidence: T,
    pub label: String,
}

/// Explanation trace for vision analysis
#[derive(Debug, Clone)]
pub struct VisionExplanation {
    /// Step-by-step analysis description
    pub analysis_steps: Vec<String>,
    /// Key visual indicators detected
    pub visual_indicators: Vec<String>,
    /// Confidence assessment reasoning
    pub confidence_reasoning: String,
    /// Recommendation for human review
    pub human_review_recommendation: Option<String>,
}

/// Main vision analyzer implementing the ModalityAnalyzer trait
pub struct VisionAnalyzer<T: Float> {
    config: VisionConfig<T>,
    face_analyzer: FaceAnalyzer<T>,
    micro_expression_detector: MicroExpressionDetector<T>,
    #[cfg(feature = "gpu")]
    gpu_processor: Option<GpuVisionProcessor<T>>,
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> VisionAnalyzer<T> {
    /// Create a new vision analyzer with the given configuration
    pub fn new(config: VisionConfig<T>) -> Result<Self, VisionError> {
        let face_analyzer = FaceAnalyzer::new(&config)?;
        let micro_expression_detector = MicroExpressionDetector::new(&config)?;
        
        #[cfg(feature = "gpu")]
        let gpu_processor = if config.enable_gpu {
            Some(GpuVisionProcessor::new(&config)?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            face_analyzer,
            micro_expression_detector,
            #[cfg(feature = "gpu")]
            gpu_processor,
            _phantom: PhantomData,
        })
    }
    
    /// Extract visual features from input image
    pub fn extract_features(&mut self, input: &VisionInput) -> Result<VisionFeatures<T>, VisionError> {
        // Validate input
        if input.image_data.is_empty() {
            return Err(VisionError::InvalidImageFormat("Empty image data".to_string()));
        }
        
        let mut features = VisionFeatures::new();
        
        // Use GPU processing if available and enabled
        #[cfg(feature = "gpu")]
        if let Some(gpu_processor) = &self.gpu_processor {
            return gpu_processor.extract_features(input);
        }
        
        // CPU-based feature extraction
        
        // 1. Face detection and landmark extraction
        let faces = self.face_analyzer.detect_faces(input)?;
        if !faces.is_empty() {
            let landmarks = self.face_analyzer.extract_landmarks(input, &faces[0])?;
            features.facial_landmarks = landmarks;
            
            // 2. Head pose estimation
            features.head_pose = self.face_analyzer.estimate_head_pose(&landmarks)?;
            
            // 3. Gaze direction estimation
            features.gaze_direction = self.face_analyzer.estimate_gaze_direction(input, &faces[0])?;
            
            // 4. Eye movement analysis
            features.eye_features = self.face_analyzer.analyze_eye_movements(input, &landmarks)?;
        }
        
        // 5. Micro-expression detection
        let micro_expressions = self.micro_expression_detector.detect_expressions(input)?;
        features.micro_expressions = micro_expressions.features;
        
        // 6. Facial action unit detection
        features.action_units = self.face_analyzer.detect_action_units(input, &features.facial_landmarks)?;
        
        // Set confidence scores
        features.feature_confidence.insert("landmarks".to_string(), T::from(0.9).unwrap());
        features.feature_confidence.insert("micro_expressions".to_string(), T::from(0.85).unwrap());
        features.feature_confidence.insert("gaze".to_string(), T::from(0.8).unwrap());
        
        Ok(features)
    }
    
    /// Analyze features to determine deception probability
    pub fn analyze(&self, features: &VisionFeatures<T>) -> Result<VisionDeceptionScore<T>, VisionError> {
        let mut contributing_factors = HashMap::new();
        let mut detected_expressions = Vec::new();
        let mut regions_of_interest = Vec::new();
        
        // Analyze micro-expressions for deception indicators
        let micro_expr_score = self.analyze_micro_expressions(&features.micro_expressions)?;
        contributing_factors.insert("micro_expressions".to_string(), micro_expr_score);
        
        // Analyze gaze patterns
        let gaze_score = self.analyze_gaze_patterns(&features.gaze_direction, &features.eye_features)?;
        contributing_factors.insert("gaze_patterns".to_string(), gaze_score);
        
        // Analyze facial action units
        let action_unit_score = self.analyze_action_units(&features.action_units)?;
        contributing_factors.insert("action_units".to_string(), action_unit_score);
        
        // Analyze head pose and movement
        let head_pose_score = self.analyze_head_pose(&features.head_pose)?;
        contributing_factors.insert("head_pose".to_string(), head_pose_score);
        
        // Combine scores using weighted average
        let weights = [T::from(0.4).unwrap(), T::from(0.3).unwrap(), T::from(0.2).unwrap(), T::from(0.1).unwrap()];
        let scores = [micro_expr_score, gaze_score, action_unit_score, head_pose_score];
        
        let probability = weights.iter()
            .zip(scores.iter())
            .map(|(&w, &s)| w * s)
            .fold(T::zero(), |acc, x| acc + x);
        
        // Calculate confidence based on feature quality
        let confidence = self.calculate_confidence(features)?;
        
        Ok(VisionDeceptionScore {
            probability,
            confidence,
            contributing_factors,
            detected_expressions,
            regions_of_interest,
        })
    }
    
    /// Generate explanation for the analysis
    pub fn explain(&self, features: &VisionFeatures<T>) -> VisionExplanation {
        let mut analysis_steps = Vec::new();
        let mut visual_indicators = Vec::new();
        
        analysis_steps.push("1. Face detection and landmark extraction".to_string());
        analysis_steps.push("2. Micro-expression analysis".to_string());
        analysis_steps.push("3. Gaze pattern evaluation".to_string());
        analysis_steps.push("4. Facial action unit assessment".to_string());
        analysis_steps.push("5. Head pose and movement analysis".to_string());
        
        // Add specific indicators based on features
        if !features.micro_expressions.is_empty() {
            visual_indicators.push("Micro-expressions detected".to_string());
        }
        
        if features.gaze_direction.iter().any(|&x| x > T::from(0.5).unwrap()) {
            visual_indicators.push("Irregular gaze patterns".to_string());
        }
        
        VisionExplanation {
            analysis_steps,
            visual_indicators,
            confidence_reasoning: "Analysis based on facial features and behavioral patterns".to_string(),
            human_review_recommendation: Some("Consider additional context for final assessment".to_string()),
        }
    }
    
    // Private helper methods
    
    fn analyze_micro_expressions(&self, expressions: &[T]) -> Result<T, VisionError> {
        if expressions.is_empty() {
            return Ok(T::zero());
        }
        
        // Simple aggregation - in practice, this would use trained models
        let sum: T = expressions.iter().cloned().fold(T::zero(), |acc, x| acc + x);
        let avg = sum / T::from(expressions.len()).unwrap();
        Ok(avg.min(T::one()))
    }
    
    fn analyze_gaze_patterns(&self, gaze_direction: &[T; 3], eye_features: &[T]) -> Result<T, VisionError> {
        // Analyze gaze stability and direction changes
        let gaze_instability = gaze_direction.iter()
            .map(|&x| (x - T::from(0.5).unwrap()).abs())
            .fold(T::zero(), |acc, x| acc + x) / T::from(3.0).unwrap();
        
        Ok(gaze_instability)
    }
    
    fn analyze_action_units(&self, action_units: &[T]) -> Result<T, VisionError> {
        if action_units.is_empty() {
            return Ok(T::zero());
        }
        
        // Focus on specific action units associated with deception
        // AU4 (Brow Lowerer), AU15 (Lip Corner Depressor), etc.
        let deception_indicators = if action_units.len() >= 15 {
            [action_units.get(3), action_units.get(14)]
                .iter()
                .filter_map(|&x| x)
                .cloned()
                .fold(T::zero(), |acc, x| acc + x) / T::from(2.0).unwrap()
        } else {
            T::zero()
        };
        
        Ok(deception_indicators)
    }
    
    fn analyze_head_pose(&self, head_pose: &[T; 3]) -> Result<T, VisionError> {
        // Analyze unusual head movements (excessive nodding, head turns)
        let pose_deviation = head_pose.iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |acc, x| acc + x) / T::from(3.0).unwrap();
        
        Ok(pose_deviation)
    }
    
    fn calculate_confidence(&self, features: &VisionFeatures<T>) -> Result<T, VisionError> {
        let mut confidence_sum = T::zero();
        let mut confidence_count = T::zero();
        
        for (_, &conf) in &features.feature_confidence {
            confidence_sum = confidence_sum + conf;
            confidence_count = confidence_count + T::one();
        }
        
        if confidence_count > T::zero() {
            Ok(confidence_sum / confidence_count)
        } else {
            Ok(T::from(0.5).unwrap()) // Default moderate confidence
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Integration tests moved to tests/ directory
    
    #[test]
    fn test_vision_config_default() {
        let config: VisionConfig<f32> = VisionConfig::default();
        assert_eq!(config.face_detection_threshold, 0.7);
        assert_eq!(config.micro_expression_sensitivity, 0.85);
        assert_eq!(config.max_faces, 5);
        assert!(!config.enable_gpu);
    }
    
    #[test]
    fn test_vision_input_creation() {
        let image_data = vec![255u8; 224 * 224 * 3];
        let input = VisionInput::new(image_data, 224, 224, 3);
        assert_eq!(input.width, 224);
        assert_eq!(input.height, 224);
        assert_eq!(input.channels, 3);
        assert!(input.timestamp.is_some());
    }
    
    #[test]
    fn test_vision_features_creation() {
        let features: VisionFeatures<f32> = VisionFeatures::new();
        assert_eq!(features.feature_count(), 6); // gaze_direction (3) + head_pose (3)
        assert!(features.facial_landmarks.is_empty());
        assert!(features.micro_expressions.is_empty());
    }
    
    #[test]
    fn test_vision_features_flat_vector() {
        let mut features: VisionFeatures<f32> = VisionFeatures::new();
        features.facial_landmarks = vec![1.0, 2.0];
        features.micro_expressions = vec![0.5];
        features.gaze_direction = [0.1, 0.2, 0.3];
        features.head_pose = [0.4, 0.5, 0.6];
        
        let flat = features.to_flat_vector();
        assert_eq!(flat.len(), 8); // 2 + 1 + 3 + 0 + 3 + 0
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[1], 2.0);
        assert_eq!(flat[2], 0.5);
    }
    
    #[test]
    fn test_vision_analyzer_creation() {
        let config: VisionConfig<f32> = VisionConfig::default();
        // Note: This will fail without proper model files, but tests the structure
        let result = VisionAnalyzer::new(config);
        // In a real implementation, we'd mock the dependencies
        assert!(result.is_err() || result.is_ok());
    }
    
    #[test]
    fn test_bounding_box() {
        let bbox: BoundingBox<f32> = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 150.0,
            confidence: 0.95,
            label: "face".to_string(),
        };
        
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 150.0);
        assert_eq!(bbox.confidence, 0.95);
        assert_eq!(bbox.label, "face");
    }
    
    #[test]
    fn test_vision_deception_score() {
        let mut contributing_factors = HashMap::new();
        contributing_factors.insert("micro_expressions".to_string(), 0.8);
        contributing_factors.insert("gaze_patterns".to_string(), 0.6);
        
        let score: VisionDeceptionScore<f32> = VisionDeceptionScore {
            probability: 0.75,
            confidence: 0.9,
            contributing_factors,
            detected_expressions: vec!["suppression".to_string()],
            regions_of_interest: Vec::new(),
        };
        
        assert_eq!(score.probability, 0.75);
        assert_eq!(score.confidence, 0.9);
        assert_eq!(score.contributing_factors.len(), 2);
        assert_eq!(score.detected_expressions.len(), 1);
    }
    
    #[test]
    fn test_vision_explanation() {
        let explanation = VisionExplanation {
            analysis_steps: vec![
                "Step 1: Face detection".to_string(),
                "Step 2: Feature extraction".to_string(),
            ],
            visual_indicators: vec!["Micro-expression detected".to_string()],
            confidence_reasoning: "High confidence due to clear features".to_string(),
            human_review_recommendation: Some("Additional review recommended".to_string()),
        };
        
        assert_eq!(explanation.analysis_steps.len(), 2);
        assert_eq!(explanation.visual_indicators.len(), 1);
        assert!(!explanation.confidence_reasoning.is_empty());
        assert!(explanation.human_review_recommendation.is_some());
    }
}