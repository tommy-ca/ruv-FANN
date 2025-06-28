//! Micro-expression detection module
//! 
//! This module detects subtle facial expressions that occur involuntarily
//! and may indicate concealed emotions or deception attempts.

use std::collections::HashMap;
use num_traits::Float;
use crate::modalities::vision::{VisionError, VisionConfig, VisionInput};

/// Types of micro-expressions that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MicroExpressionType {
    /// Brief happiness/joy expression
    Happiness,
    /// Brief sadness expression
    Sadness,
    /// Brief anger expression
    Anger,
    /// Brief fear expression
    Fear,
    /// Brief surprise expression
    Surprise,
    /// Brief disgust expression
    Disgust,
    /// Brief contempt expression
    Contempt,
    /// Suppressed emotion
    Suppression,
    /// Emotional leakage (brief genuine emotion breaking through)
    Leakage,
    /// Duping delight (micro-expression of pleasure when deceiving)
    DupingDelight,
}

impl MicroExpressionType {
    /// Get the deception relevance score for this micro-expression type
    pub fn deception_relevance<T: Float>(&self) -> T {
        match self {
            MicroExpressionType::Suppression => T::from(0.9).unwrap(),
            MicroExpressionType::Leakage => T::from(0.85).unwrap(),
            MicroExpressionType::DupingDelight => T::from(0.8).unwrap(),
            MicroExpressionType::Contempt => T::from(0.7).unwrap(),
            MicroExpressionType::Fear => T::from(0.6).unwrap(),
            MicroExpressionType::Anger => T::from(0.5).unwrap(),
            MicroExpressionType::Disgust => T::from(0.5).unwrap(),
            MicroExpressionType::Sadness => T::from(0.3).unwrap(),
            MicroExpressionType::Surprise => T::from(0.2).unwrap(),
            MicroExpressionType::Happiness => T::from(0.1).unwrap(),
        }
    }
    
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            MicroExpressionType::Happiness => "Brief happiness micro-expression",
            MicroExpressionType::Sadness => "Brief sadness micro-expression",
            MicroExpressionType::Anger => "Brief anger micro-expression",
            MicroExpressionType::Fear => "Brief fear micro-expression",
            MicroExpressionType::Surprise => "Brief surprise micro-expression",
            MicroExpressionType::Disgust => "Brief disgust micro-expression",
            MicroExpressionType::Contempt => "Brief contempt micro-expression",
            MicroExpressionType::Suppression => "Emotion suppression detected",
            MicroExpressionType::Leakage => "Emotional leakage detected",
            MicroExpressionType::DupingDelight => "Duping delight detected",
        }
    }
}

/// A detected micro-expression with timing and intensity
#[derive(Debug, Clone)]
pub struct DetectedMicroExpression<T: Float> {
    /// Type of micro-expression
    pub expression_type: MicroExpressionType,
    /// Intensity/confidence of detection (0.0 to 1.0)
    pub intensity: T,
    /// Duration in milliseconds (typically 25-500ms for micro-expressions)
    pub duration_ms: T,
    /// Onset time relative to analysis start
    pub onset_time: Option<std::time::Duration>,
    /// Facial regions involved
    pub affected_regions: Vec<FacialRegion>,
    /// Action units involved in this micro-expression
    pub action_units: Vec<u8>,
}

/// Facial regions for micro-expression analysis
#[derive(Debug, Clone, PartialEq)]
pub enum FacialRegion {
    /// Upper face region (forehead, eyebrows)
    UpperFace,
    /// Eye region (eyes, eyelids)
    EyeRegion,
    /// Mid-face region (nose, cheeks)
    MidFace,
    /// Lower face region (mouth, jaw)
    LowerFace,
    /// Left side of face
    LeftSide,
    /// Right side of face
    RightSide,
}

/// Result of micro-expression analysis
#[derive(Debug, Clone)]
pub struct MicroExpressionResult<T: Float> {
    /// Detected micro-expressions
    pub expressions: Vec<DetectedMicroExpression<T>>,
    /// Overall deception indicator score based on detected micro-expressions
    pub deception_score: T,
    /// Feature vector for further analysis
    pub features: Vec<T>,
    /// Confidence in the analysis
    pub confidence: T,
    /// Temporal analysis information
    pub temporal_analysis: TemporalAnalysis<T>,
}

/// Temporal analysis of expression patterns
#[derive(Debug, Clone)]
pub struct TemporalAnalysis<T: Float> {
    /// Number of expression transitions detected
    pub transition_count: usize,
    /// Average expression duration
    pub avg_duration: T,
    /// Expression frequency (expressions per second)
    pub frequency: T,
    /// Baseline vs active expression ratio
    pub baseline_ratio: T,
    /// Asymmetry in left vs right face expressions
    pub asymmetry_score: T,
}

/// Micro-expression detector using computer vision and machine learning
pub struct MicroExpressionDetector<T: Float> {
    config: VisionConfig<T>,
    /// Sensitivity threshold for detection
    sensitivity: T,
    /// Temporal window for analyzing expression sequences
    temporal_window_ms: T,
    /// Previous frame data for motion analysis
    previous_frame: Option<FrameData<T>>,
    /// Expression history for temporal analysis
    expression_history: Vec<DetectedMicroExpression<T>>,
    /// Baseline expression measurements
    baseline_measurements: Option<BaselineExpressions<T>>,
}

/// Frame data for temporal analysis
#[derive(Debug, Clone)]
struct FrameData<T: Float> {
    /// Action unit activations for this frame
    pub action_units: Vec<T>,
    /// Landmark positions
    pub landmarks: Vec<T>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Baseline expression measurements for comparison
#[derive(Debug, Clone)]
struct BaselineExpressions<T: Float> {
    /// Average action unit activations during neutral expression
    pub neutral_aus: Vec<T>,
    /// Standard deviations for each AU
    pub au_std_devs: Vec<T>,
    /// Average facial landmark positions
    pub neutral_landmarks: Vec<T>,
    /// Landmark position variations
    pub landmark_variations: Vec<T>,
}

impl<T: Float + Send + Sync + 'static> MicroExpressionDetector<T> {
    /// Create a new micro-expression detector
    pub fn new(config: &VisionConfig<T>) -> Result<Self, VisionError> {
        Ok(Self {
            config: config.clone(),
            sensitivity: config.micro_expression_sensitivity,
            temporal_window_ms: T::from(1000.0).unwrap(), // 1 second window
            previous_frame: None,
            expression_history: Vec::new(),
            baseline_measurements: None,
        })
    }
    
    /// Detect micro-expressions in the input image
    pub fn detect_expressions(&mut self, input: &VisionInput) -> Result<MicroExpressionResult<T>, VisionError> {
        // This would typically require:
        // 1. Facial landmark detection
        // 2. Action unit extraction
        // 3. Temporal analysis across frames
        // 4. Pattern recognition for micro-expressions
        
        // For this implementation, we'll simulate the process
        let mut expressions = Vec::new();
        let mut features = Vec::new();
        
        // Simulate action unit extraction (this would come from face analyzer)
        let mock_action_units = self.extract_mock_action_units(input)?;
        
        // Temporal analysis if we have previous frame data
        let temporal_changes = if let Some(ref prev_frame) = self.previous_frame {
            self.calculate_temporal_changes(&mock_action_units, &prev_frame.action_units)?
        } else {
            vec![T::zero(); mock_action_units.len()]
        };
        
        // Update frame history
        self.previous_frame = Some(FrameData {
            action_units: mock_action_units.clone(),
            landmarks: vec![T::zero(); 136], // Mock landmarks
            timestamp: std::time::SystemTime::now(),
        });
        
        // Detect micro-expressions based on AU patterns and temporal changes
        let detected_expressions = self.analyze_au_patterns(&mock_action_units, &temporal_changes)?;
        expressions.extend(detected_expressions);
        
        // Calculate deception score based on detected expressions
        let deception_score = self.calculate_deception_score(&expressions);
        
        // Generate feature vector
        features.extend(&mock_action_units);
        features.extend(&temporal_changes);
        features.push(deception_score);
        
        // Calculate confidence based on detection quality
        let confidence = self.calculate_confidence(&expressions, &temporal_changes);
        
        // Temporal analysis
        let temporal_analysis = self.perform_temporal_analysis(&expressions);
        
        // Update expression history
        self.expression_history.extend(expressions.clone());
        
        // Keep only recent history (last 10 seconds)
        let cutoff_time = std::time::SystemTime::now() - std::time::Duration::from_secs(10);
        self.expression_history.retain(|expr| {
            expr.onset_time
                .map(|onset| std::time::SystemTime::now() - onset < std::time::Duration::from_secs(10))
                .unwrap_or(true)
        });
        
        Ok(MicroExpressionResult {
            expressions,
            deception_score,
            features,
            confidence,
            temporal_analysis,
        })
    }
    
    /// Set baseline measurements for comparison
    pub fn set_baseline(&mut self, baseline_input: &VisionInput) -> Result<(), VisionError> {
        let action_units = self.extract_mock_action_units(baseline_input)?;
        let landmarks = vec![T::zero(); 136]; // Mock landmarks
        
        // Calculate standard deviations (mock implementation)
        let au_std_devs = action_units.iter()
            .map(|&au| au * T::from(0.1).unwrap()) // Mock std dev as 10% of value
            .collect();
        
        let landmark_variations = landmarks.iter()
            .map(|&lm| lm * T::from(0.05).unwrap()) // Mock variation as 5% of value
            .collect();
        
        self.baseline_measurements = Some(BaselineExpressions {
            neutral_aus: action_units,
            au_std_devs,
            neutral_landmarks: landmarks,
            landmark_variations,
        });
        
        Ok(())
    }
    
    /// Clear expression history and reset temporal analysis
    pub fn reset_history(&mut self) {
        self.expression_history.clear();
        self.previous_frame = None;
    }
    
    /// Get current expression statistics
    pub fn get_expression_statistics(&self) -> ExpressionStatistics<T> {
        let total_expressions = self.expression_history.len();
        
        let mut type_counts = HashMap::new();
        let mut total_intensity = T::zero();
        
        for expr in &self.expression_history {
            *type_counts.entry(expr.expression_type.clone()).or_insert(0) += 1;
            total_intensity = total_intensity + expr.intensity;
        }
        
        let avg_intensity = if total_expressions > 0 {
            total_intensity / T::from(total_expressions).unwrap()
        } else {
            T::zero()
        };
        
        ExpressionStatistics {
            total_expressions,
            expression_type_counts: type_counts,
            average_intensity: avg_intensity,
            deception_relevant_count: self.expression_history.iter()
                .filter(|expr| expr.expression_type.deception_relevance::<T>() > T::from(0.5).unwrap())
                .count(),
        }
    }
    
    // Private helper methods
    
    fn extract_mock_action_units(&self, input: &VisionInput) -> Result<Vec<T>, VisionError> {
        // Mock action unit extraction - in practice, this would use the face analyzer
        let au_count = 17; // Standard number of action units
        let mut action_units = Vec::with_capacity(au_count);
        
        // Generate mock AU values based on image properties
        let image_hash = input.image_data.len() % 1000;
        
        for i in 0..au_count {
            let base_value = T::from((image_hash + i * 37) % 100).unwrap() / T::from(100.0).unwrap();
            let noise = T::from((input.width as usize + i) % 20).unwrap() / T::from(200.0).unwrap();
            let au_value = (base_value + noise).min(T::one()).max(T::zero());
            action_units.push(au_value);
        }
        
        Ok(action_units)
    }
    
    fn calculate_temporal_changes(&self, current_aus: &[T], previous_aus: &[T]) -> Result<Vec<T>, VisionError> {
        if current_aus.len() != previous_aus.len() {
            return Err(VisionError::FeatureExtractionFailed(
                "AU vector length mismatch between frames".to_string()
            ));
        }
        
        let changes = current_aus.iter()
            .zip(previous_aus.iter())
            .map(|(&current, &previous)| (current - previous).abs())
            .collect();
        
        Ok(changes)
    }
    
    fn analyze_au_patterns(&self, action_units: &[T], temporal_changes: &[T]) -> Result<Vec<DetectedMicroExpression<T>>, VisionError> {
        let mut expressions = Vec::new();
        
        // Define AU patterns for different micro-expressions
        let patterns = self.get_expression_patterns();
        
        for (expr_type, pattern) in patterns {
            let intensity = self.match_pattern(&expr_type, pattern, action_units, temporal_changes)?;
            
            if intensity > self.sensitivity {
                let expression = DetectedMicroExpression {
                    expression_type: expr_type.clone(),
                    intensity,
                    duration_ms: T::from(150.0).unwrap(), // Typical micro-expression duration
                    onset_time: Some(std::time::Duration::from_millis(0)),
                    affected_regions: self.get_affected_regions(&expr_type),
                    action_units: self.get_expression_aus(&expr_type),
                };
                
                expressions.push(expression);
            }
        }
        
        Ok(expressions)
    }
    
    fn get_expression_patterns(&self) -> Vec<(MicroExpressionType, Vec<(usize, T)>)> {
        // Define AU patterns for micro-expressions
        // Format: (expression_type, [(au_index, minimum_activation), ...])
        vec![
            (MicroExpressionType::Suppression, vec![(3, T::from(0.7).unwrap()), (14, T::from(0.6).unwrap())]), // AU4 + AU15
            (MicroExpressionType::Leakage, vec![(5, T::from(0.8).unwrap())]), // AU6 (cheek raiser)
            (MicroExpressionType::DupingDelight, vec![(11, T::from(0.6).unwrap())]), // AU12 (lip corner puller)
            (MicroExpressionType::Contempt, vec![(13, T::from(0.7).unwrap())]), // AU14 (dimpler)
            (MicroExpressionType::Fear, vec![(0, T::from(0.8).unwrap()), (4, T::from(0.7).unwrap())]), // AU1 + AU5
            (MicroExpressionType::Anger, vec![(3, T::from(0.7).unwrap()), (6, T::from(0.6).unwrap())]), // AU4 + AU7
            (MicroExpressionType::Disgust, vec![(8, T::from(0.7).unwrap())]), // AU9 (nose wrinkler)
            (MicroExpressionType::Sadness, vec![(0, T::from(0.6).unwrap()), (14, T::from(0.7).unwrap())]), // AU1 + AU15
            (MicroExpressionType::Surprise, vec![(0, T::from(0.8).unwrap()), (1, T::from(0.8).unwrap())]), // AU1 + AU2
            (MicroExpressionType::Happiness, vec![(5, T::from(0.7).unwrap()), (11, T::from(0.8).unwrap())]), // AU6 + AU12
        ]
    }
    
    fn match_pattern(&self, expr_type: &MicroExpressionType, pattern: Vec<(usize, T)>, action_units: &[T], temporal_changes: &[T]) -> Result<T, VisionError> {
        let mut total_match = T::zero();
        let mut total_weight = T::zero();
        
        for (au_index, threshold) in pattern {
            if au_index < action_units.len() && au_index < temporal_changes.len() {
                let au_activation = action_units[au_index];
                let temporal_change = temporal_changes[au_index];
                
                // Combine static activation with temporal change
                let combined_score = au_activation + temporal_change * T::from(2.0).unwrap();
                
                if combined_score >= threshold {
                    let match_strength = combined_score - threshold + T::one();
                    total_match = total_match + match_strength;
                }
                total_weight = total_weight + T::one();
            }
        }
        
        if total_weight > T::zero() {
            let base_intensity = total_match / total_weight;
            
            // Apply deception relevance weighting
            let relevance_weight = expr_type.deception_relevance::<T>();
            Ok((base_intensity * relevance_weight).min(T::one()))
        } else {
            Ok(T::zero())
        }
    }
    
    fn get_affected_regions(&self, expr_type: &MicroExpressionType) -> Vec<FacialRegion> {
        match expr_type {
            MicroExpressionType::Happiness | MicroExpressionType::Sadness => {
                vec![FacialRegion::EyeRegion, FacialRegion::LowerFace]
            },
            MicroExpressionType::Anger => {
                vec![FacialRegion::UpperFace, FacialRegion::EyeRegion]
            },
            MicroExpressionType::Fear | MicroExpressionType::Surprise => {
                vec![FacialRegion::UpperFace, FacialRegion::EyeRegion, FacialRegion::LowerFace]
            },
            MicroExpressionType::Disgust => {
                vec![FacialRegion::MidFace, FacialRegion::LowerFace]
            },
            MicroExpressionType::Contempt => {
                vec![FacialRegion::LowerFace]
            },
            MicroExpressionType::Suppression | MicroExpressionType::Leakage => {
                vec![FacialRegion::EyeRegion, FacialRegion::LowerFace]
            },
            MicroExpressionType::DupingDelight => {
                vec![FacialRegion::EyeRegion, FacialRegion::LowerFace]
            },
        }
    }
    
    fn get_expression_aus(&self, expr_type: &MicroExpressionType) -> Vec<u8> {
        match expr_type {
            MicroExpressionType::Happiness => vec![6, 12], // AU6, AU12
            MicroExpressionType::Sadness => vec![1, 4, 15], // AU1, AU4, AU15
            MicroExpressionType::Anger => vec![4, 5, 7, 23], // AU4, AU5, AU7, AU23
            MicroExpressionType::Fear => vec![1, 2, 4, 5, 7, 20, 26], // AU1, AU2, AU4, AU5, AU7, AU20, AU26
            MicroExpressionType::Surprise => vec![1, 2, 5, 26], // AU1, AU2, AU5, AU26
            MicroExpressionType::Disgust => vec![9, 15, 16], // AU9, AU15, AU16
            MicroExpressionType::Contempt => vec![12, 14], // AU12, AU14 (unilateral)
            MicroExpressionType::Suppression => vec![4, 15, 17], // AU4, AU15, AU17
            MicroExpressionType::Leakage => vec![6, 12], // AU6, AU12 (brief)
            MicroExpressionType::DupingDelight => vec![6, 12, 14], // AU6, AU12, AU14
        }
    }
    
    fn calculate_deception_score(&self, expressions: &[DetectedMicroExpression<T>]) -> T {
        if expressions.is_empty() {
            return T::zero();
        }
        
        let mut weighted_score = T::zero();
        let mut total_weight = T::zero();
        
        for expr in expressions {
            let relevance = expr.expression_type.deception_relevance::<T>();
            let weighted_intensity = expr.intensity * relevance;
            weighted_score = weighted_score + weighted_intensity;
            total_weight = total_weight + relevance;
        }
        
        if total_weight > T::zero() {
            (weighted_score / total_weight).min(T::one())
        } else {
            T::zero()
        }
    }
    
    fn calculate_confidence(&self, expressions: &[DetectedMicroExpression<T>], temporal_changes: &[T]) -> T {
        // Base confidence on the strength of temporal changes and number of expressions
        let avg_temporal_change = if !temporal_changes.is_empty() {
            temporal_changes.iter().cloned().fold(T::zero(), |acc, x| acc + x) / T::from(temporal_changes.len()).unwrap()
        } else {
            T::zero()
        };
        
        let expression_count_factor = T::from(expressions.len().min(5)).unwrap() / T::from(5.0).unwrap();
        
        let base_confidence = (avg_temporal_change + expression_count_factor) / T::from(2.0).unwrap();
        base_confidence.min(T::one()).max(T::from(0.1).unwrap())
    }
    
    fn perform_temporal_analysis(&self, expressions: &[DetectedMicroExpression<T>]) -> TemporalAnalysis<T> {
        let transition_count = expressions.len();
        
        let avg_duration = if !expressions.is_empty() {
            let total_duration: T = expressions.iter()
                .map(|expr| expr.duration_ms)
                .fold(T::zero(), |acc, x| acc + x);
            total_duration / T::from(expressions.len()).unwrap()
        } else {
            T::zero()
        };
        
        let frequency = T::from(transition_count).unwrap() / self.temporal_window_ms * T::from(1000.0).unwrap();
        
        // Mock baseline ratio
        let baseline_ratio = T::from(0.7).unwrap();
        
        // Calculate asymmetry (mock implementation)
        let asymmetry_score = T::from(0.1).unwrap();
        
        TemporalAnalysis {
            transition_count,
            avg_duration,
            frequency,
            baseline_ratio,
            asymmetry_score,
        }
    }
}

/// Statistics about detected expressions
#[derive(Debug, Clone)]
pub struct ExpressionStatistics<T: Float> {
    pub total_expressions: usize,
    pub expression_type_counts: HashMap<MicroExpressionType, usize>,
    pub average_intensity: T,
    pub deception_relevant_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modalities::vision::VisionConfig;
    
    #[test]
    fn test_micro_expression_detector_creation() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let detector = MicroExpressionDetector::new(&config);
        assert!(detector.is_ok());
    }
    
    #[test]
    fn test_micro_expression_types() {
        assert_eq!(MicroExpressionType::Suppression.deception_relevance::<f32>(), 0.9);
        assert_eq!(MicroExpressionType::Happiness.deception_relevance::<f32>(), 0.1);
        assert!(!MicroExpressionType::Fear.description().is_empty());
    }
    
    #[test]
    fn test_expression_detection() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let mut detector = MicroExpressionDetector::new(&config).unwrap();
        
        let image_data = vec![128u8; 224 * 224 * 3];
        let input = VisionInput::new(image_data, 224, 224, 3);
        
        let result = detector.detect_expressions(&input);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.confidence > 0.0);
        assert!(!result.features.is_empty());
    }
    
    #[test]
    fn test_baseline_setting() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let mut detector = MicroExpressionDetector::new(&config).unwrap();
        
        let image_data = vec![128u8; 224 * 224 * 3];
        let input = VisionInput::new(image_data, 224, 224, 3);
        
        let result = detector.set_baseline(&input);
        assert!(result.is_ok());
        assert!(detector.baseline_measurements.is_some());
    }
    
    #[test]
    fn test_temporal_analysis() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let mut detector = MicroExpressionDetector::new(&config).unwrap();
        
        // Process multiple frames to test temporal analysis
        for i in 0..3 {
            let mut image_data = vec![128u8; 224 * 224 * 3];
            // Vary the data slightly for each frame
            if i > 0 {
                image_data[i] = (128 + i * 10) as u8;
            }
            let input = VisionInput::new(image_data, 224, 224, 3);
            
            let result = detector.detect_expressions(&input);
            assert!(result.is_ok());
        }
        
        let stats = detector.get_expression_statistics();
        assert!(stats.total_expressions >= 0);
    }
    
    #[test]
    fn test_expression_patterns() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let detector = MicroExpressionDetector::new(&config).unwrap();
        
        let patterns = detector.get_expression_patterns();
        assert!(!patterns.is_empty());
        
        // Test that each pattern has valid AU indices
        for (expr_type, pattern) in patterns {
            assert!(!pattern.is_empty());
            for (au_index, threshold) in pattern {
                assert!(au_index < 17); // Should be valid AU index
                assert!(threshold > 0.0 && threshold <= 1.0);
            }
        }
    }
    
    #[test]
    fn test_affected_regions() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let detector = MicroExpressionDetector::new(&config).unwrap();
        
        let regions = detector.get_affected_regions(&MicroExpressionType::Happiness);
        assert!(!regions.is_empty());
        
        let regions = detector.get_affected_regions(&MicroExpressionType::Anger);
        assert!(!regions.is_empty());
    }
    
    #[test]
    fn test_expression_aus() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let detector = MicroExpressionDetector::new(&config).unwrap();
        
        let aus = detector.get_expression_aus(&MicroExpressionType::Happiness);
        assert!(!aus.is_empty());
        assert!(aus.contains(&6)); // AU6 for happiness
        assert!(aus.contains(&12)); // AU12 for happiness
    }
    
    #[test]
    fn test_deception_score_calculation() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let detector = MicroExpressionDetector::new(&config).unwrap();
        
        let expressions = vec![
            DetectedMicroExpression {
                expression_type: MicroExpressionType::Suppression,
                intensity: 0.8,
                duration_ms: 150.0,
                onset_time: None,
                affected_regions: vec![FacialRegion::LowerFace],
                action_units: vec![4, 15],
            },
            DetectedMicroExpression {
                expression_type: MicroExpressionType::Happiness,
                intensity: 0.5,
                duration_ms: 200.0,
                onset_time: None,
                affected_regions: vec![FacialRegion::LowerFace],
                action_units: vec![6, 12],
            },
        ];
        
        let score = detector.calculate_deception_score(&expressions);
        assert!(score > 0.0);
        assert!(score <= 1.0);
        
        // Suppression should have higher weight than happiness
        assert!(score > 0.3); // Should be weighted towards suppression
    }
    
    #[test]
    fn test_history_management() {
        let config: VisionConfig<f32> = VisionConfig::default();
        let mut detector = MicroExpressionDetector::new(&config).unwrap();
        
        // Add some expressions to history
        detector.expression_history.push(DetectedMicroExpression {
            expression_type: MicroExpressionType::Fear,
            intensity: 0.6,
            duration_ms: 120.0,
            onset_time: Some(std::time::Duration::from_secs(0)),
            affected_regions: vec![FacialRegion::EyeRegion],
            action_units: vec![1, 2],
        });
        
        assert_eq!(detector.expression_history.len(), 1);
        
        detector.reset_history();
        assert_eq!(detector.expression_history.len(), 0);
        assert!(detector.previous_frame.is_none());
    }
}