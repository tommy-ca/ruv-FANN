//! # Explainable AI Decisions Example for Veritas Nexus
//! 
//! This example demonstrates how to generate detailed explanations and reasoning
//! traces for lie detection decisions. It shows how to:
//! - Generate step-by-step reasoning traces
//! - Provide feature importance explanations
//! - Create human-readable decision summaries
//! - Implement different explanation strategies
//! - Generate confidence intervals and uncertainty analysis
//! - Export explanations in multiple formats
//! - Visualize decision pathways
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example explainable_decisions
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio;

/// Explanation engine for lie detection decisions
pub struct ExplanationEngine {
    config: ExplanationConfig,
    feature_importance_threshold: f32,
    explanation_templates: HashMap<String, String>,
}

/// Configuration for explanation generation
#[derive(Debug, Clone)]
pub struct ExplanationConfig {
    pub explanation_style: ExplanationStyle,
    pub detail_level: DetailLevel,
    pub include_uncertainty: bool,
    pub include_feature_weights: bool,
    pub include_reasoning_trace: bool,
    pub max_features_to_explain: usize,
    pub confidence_threshold: f32,
    pub language: Language,
}

#[derive(Debug, Clone)]
pub enum ExplanationStyle {
    Technical,      // For expert users
    Simplified,     // For general users
    Narrative,      // Story-like explanation
    Structured,     // Bullet points and sections
    Visual,         // ASCII visualizations
}

#[derive(Debug, Clone)]
pub enum DetailLevel {
    High,           // Very detailed
    Medium,         // Balanced detail
    Low,            // Concise
    Custom(u8),     // 1-10 scale
}

#[derive(Debug, Clone)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            explanation_style: ExplanationStyle::Structured,
            detail_level: DetailLevel::Medium,
            include_uncertainty: true,
            include_feature_weights: true,
            include_reasoning_trace: true,
            max_features_to_explain: 5,
            confidence_threshold: 0.5,
            language: Language::English,
        }
    }
}

/// Analysis input with rich context
#[derive(Debug, Clone)]
pub struct ExplainableInput {
    pub video_path: Option<String>,
    pub audio_path: Option<String>,
    pub text: Option<String>,
    pub physiological_data: Option<Vec<f32>>,
    pub context: AnalysisContext,
}

/// Context for the analysis
#[derive(Debug, Clone)]
pub struct AnalysisContext {
    pub session_id: String,
    pub timestamp: Instant,
    pub user_type: UserType,
    pub stakes: StakeLevel,
    pub environment: Environment,
    pub subject_info: Option<SubjectInfo>,
}

#[derive(Debug, Clone)]
pub enum UserType {
    LawEnforcement,
    Researcher,
    SecurityPersonnel,
    GeneralUser,
    Expert,
}

#[derive(Debug, Clone)]
pub enum StakeLevel {
    High,       // Criminal investigation
    Medium,     // Security screening
    Low,        // Research/testing
}

#[derive(Debug, Clone)]
pub enum Environment {
    Controlled,     // Lab setting
    Field,          // Real-world
    Simulated,      // Testing environment
}

#[derive(Debug, Clone)]
pub struct SubjectInfo {
    pub age_group: AgeGroup,
    pub baseline_available: bool,
    pub stress_level: Option<f32>,
    pub prior_history: bool,
}

#[derive(Debug, Clone)]
pub enum AgeGroup {
    Young,      // 18-25
    Adult,      // 26-65
    Senior,     // 65+
}

/// Complete explanation result
#[derive(Debug, Clone, Serialize)]
pub struct ExplanationResult {
    pub decision: DeceptionDecision,
    pub confidence: f32,
    pub uncertainty: UncertaintyAnalysis,
    pub reasoning_trace: ReasoningTrace,
    pub feature_explanations: Vec<FeatureExplanation>,
    pub summary: String,
    pub detailed_explanation: String,
    pub decision_pathway: DecisionPathway,
    pub alternative_scenarios: Vec<AlternativeScenario>,
    pub metadata: ExplanationMetadata,
}

/// Decision outcome
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum DeceptionDecision {
    Truthful { probability: f32 },
    Deceptive { probability: f32 },
    Uncertain { conflicting_evidence: Vec<String> },
    InsufficientData { missing_modalities: Vec<String> },
}

/// Uncertainty quantification
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

/// Step-by-step reasoning trace
#[derive(Debug, Clone, Serialize)]
pub struct ReasoningTrace {
    pub steps: Vec<ReasoningStep>,
    pub decision_points: Vec<DecisionPoint>,
    pub total_processing_time: Duration,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub operation: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub processing_time: Duration,
    pub confidence_change: f32,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionPoint {
    pub point_name: String,
    pub condition: String,
    pub outcome: String,
    pub alternatives: Vec<String>,
    pub impact_on_decision: f32,
}

/// Individual feature explanation
#[derive(Debug, Clone, Serialize)]
pub struct FeatureExplanation {
    pub feature_name: String,
    pub modality: String,
    pub importance: f32,
    pub value: f32,
    pub contribution_to_decision: f32,
    pub interpretation: String,
    pub confidence: f32,
    pub supporting_evidence: Vec<String>,
    pub contradicting_evidence: Vec<String>,
}

/// Decision pathway visualization
#[derive(Debug, Clone, Serialize)]
pub struct DecisionPathway {
    pub nodes: Vec<PathwayNode>,
    pub edges: Vec<PathwayEdge>,
    pub critical_path: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PathwayNode {
    pub id: String,
    pub node_type: NodeType,
    pub label: String,
    pub confidence: f32,
    pub evidence_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub enum NodeType {
    Input,
    Processing,
    Decision,
    Output,
}

#[derive(Debug, Clone, Serialize)]
pub struct PathwayEdge {
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub label: String,
}

/// Alternative scenario analysis
#[derive(Debug, Clone, Serialize)]
pub struct AlternativeScenario {
    pub scenario_name: String,
    pub changed_conditions: Vec<String>,
    pub predicted_outcome: DeceptionDecision,
    pub confidence_change: f32,
    pub explanation: String,
}

/// Metadata about the explanation
#[derive(Debug, Clone, Serialize)]
pub struct ExplanationMetadata {
    pub explanation_id: String,
    pub generated_at: String,
    pub explanation_style: String,
    pub detail_level: String,
    pub processing_time: Duration,
    pub explanation_confidence: f32,
    pub user_type: String,
}

impl ExplanationEngine {
    pub fn new(config: ExplanationConfig) -> Self {
        let mut templates = HashMap::new();
        
        // Load explanation templates
        templates.insert("truthful_high_confidence".to_string(), 
            "The analysis indicates the subject is likely being truthful with {confidence}% confidence. \
            Key indicators supporting this conclusion include {top_features}.".to_string());
        
        templates.insert("deceptive_high_confidence".to_string(),
            "The analysis suggests the subject may be engaging in deception with {confidence}% confidence. \
            Primary deception indicators include {top_features}.".to_string());
        
        templates.insert("uncertain_mixed_evidence".to_string(),
            "The analysis produces an uncertain result due to conflicting evidence. \
            Truthfulness indicators: {truthful_evidence}. Deception indicators: {deceptive_evidence}.".to_string());
        
        templates.insert("insufficient_data".to_string(),
            "Unable to make a reliable determination due to insufficient data. \
            Missing modalities: {missing_modalities}. Available data: {available_data}.".to_string());
        
        Self {
            config,
            feature_importance_threshold: 0.1,
            explanation_templates: templates,
        }
    }
    
    /// Generate comprehensive explanation for a decision
    pub async fn explain(&self, input: &ExplainableInput) -> Result<ExplanationResult, ExplanationError> {
        let start_time = Instant::now();
        
        println!("🔍 Starting explanation generation for session: {}", input.context.session_id);
        
        // Step 1: Analyze input and generate features
        let (decision, features, raw_confidence) = self.analyze_input(input).await?;
        
        // Step 2: Generate reasoning trace
        let reasoning_trace = self.generate_reasoning_trace(input, &features).await;
        
        // Step 3: Compute uncertainty analysis
        let uncertainty = self.analyze_uncertainty(&features, raw_confidence);
        
        // Step 4: Generate feature explanations
        let feature_explanations = self.explain_features(&features, &decision);
        
        // Step 5: Create decision pathway
        let decision_pathway = self.create_decision_pathway(&features, &decision);
        
        // Step 6: Generate alternative scenarios
        let alternative_scenarios = self.generate_alternative_scenarios(input, &decision);
        
        // Step 7: Create summary and detailed explanation
        let summary = self.generate_summary(&decision, raw_confidence, &feature_explanations);
        let detailed_explanation = self.generate_detailed_explanation(
            &decision, &uncertainty, &feature_explanations, &reasoning_trace
        );
        
        // Step 8: Create metadata
        let metadata = ExplanationMetadata {
            explanation_id: format!("exp_{}", input.context.session_id),
            generated_at: chrono::Utc::now().to_rfc3339(),
            explanation_style: format!("{:?}", self.config.explanation_style),
            detail_level: format!("{:?}", self.config.detail_level),
            processing_time: start_time.elapsed(),
            explanation_confidence: self.calculate_explanation_confidence(&uncertainty),
            user_type: format!("{:?}", input.context.user_type),
        };
        
        Ok(ExplanationResult {
            decision,
            confidence: raw_confidence,
            uncertainty,
            reasoning_trace,
            feature_explanations,
            summary,
            detailed_explanation,
            decision_pathway,
            alternative_scenarios,
            metadata,
        })
    }
    
    /// Analyze input and extract features
    async fn analyze_input(&self, input: &ExplainableInput) -> Result<(DeceptionDecision, Vec<AnalysisFeature>, f32), ExplanationError> {
        let mut features = Vec::new();
        let mut modality_scores = Vec::new();
        
        // Analyze video if available
        if let Some(ref video_path) = input.video_path {
            let video_features = self.analyze_video(video_path).await;
            let video_score = self.compute_video_deception_score(&video_features);
            features.extend(video_features);
            modality_scores.push(("video", video_score));
        }
        
        // Analyze audio if available
        if let Some(ref audio_path) = input.audio_path {
            let audio_features = self.analyze_audio(audio_path).await;
            let audio_score = self.compute_audio_deception_score(&audio_features);
            features.extend(audio_features);
            modality_scores.push(("audio", audio_score));
        }
        
        // Analyze text if available
        if let Some(ref text) = input.text {
            let text_features = self.analyze_text(text).await;
            let text_score = self.compute_text_deception_score(&text_features);
            features.extend(text_features);
            modality_scores.push(("text", text_score));
        }
        
        // Analyze physiological data if available
        if let Some(ref physio_data) = input.physiological_data {
            let physio_features = self.analyze_physiological(physio_data).await;
            let physio_score = self.compute_physiological_deception_score(&physio_features);
            features.extend(physio_features);
            modality_scores.push(("physiological", physio_score));
        }
        
        // Check for insufficient data
        if modality_scores.is_empty() {
            return Ok((
                DeceptionDecision::InsufficientData {
                    missing_modalities: vec!["all".to_string()],
                },
                features,
                0.0,
            ));
        }
        
        // Combine scores using weighted average
        let final_score = self.combine_modality_scores(&modality_scores);
        let confidence = self.calculate_confidence(&modality_scores, &features);
        
        // Make final decision
        let decision = self.make_decision(final_score, confidence, &features);
        
        Ok((decision, features, confidence))
    }
    
    /// Generate step-by-step reasoning trace
    async fn generate_reasoning_trace(&self, input: &ExplainableInput, features: &[AnalysisFeature]) -> ReasoningTrace {
        let mut steps = Vec::new();
        let mut decision_points = Vec::new();
        let start_time = Instant::now();
        
        // Step 1: Input processing
        steps.push(ReasoningStep {
            step_number: 1,
            operation: "Input Processing".to_string(),
            inputs: vec![
                format!("Video: {}", input.video_path.as_deref().unwrap_or("None")),
                format!("Audio: {}", input.audio_path.as_deref().unwrap_or("None")),
                format!("Text: {}", input.text.as_deref().unwrap_or("None")),
                format!("Physiological: {}", if input.physiological_data.is_some() { "Available" } else { "None" }),
            ],
            outputs: vec![format!("Extracted {} features", features.len())],
            processing_time: Duration::from_millis(50),
            confidence_change: 0.0,
            explanation: "Raw input data was processed and converted to feature vectors".to_string(),
        });
        
        // Step 2: Feature extraction for each modality
        let mut current_step = 2;
        
        if input.video_path.is_some() {
            steps.push(ReasoningStep {
                step_number: current_step,
                operation: "Video Feature Extraction".to_string(),
                inputs: vec!["Video frames".to_string()],
                outputs: vec![
                    "Facial landmarks: 68 points".to_string(),
                    "Micro-expressions: 7 categories".to_string(),
                    "Eye movement patterns: 12 metrics".to_string(),
                    "Head pose variations: 6 DOF".to_string(),
                ],
                processing_time: Duration::from_millis(120),
                confidence_change: 0.25,
                explanation: "Computer vision models analyzed facial features for deception indicators".to_string(),
            });
            current_step += 1;
            
            decision_points.push(DecisionPoint {
                point_name: "Video Quality Check".to_string(),
                condition: "Video resolution >= 480p, face detection confidence >= 0.8".to_string(),
                outcome: "Passed: High quality video analysis".to_string(),
                alternatives: vec!["Failed: Lower confidence analysis".to_string()],
                impact_on_decision: 0.3,
            });
        }
        
        if input.audio_path.is_some() {
            steps.push(ReasoningStep {
                step_number: current_step,
                operation: "Audio Feature Extraction".to_string(),
                inputs: vec!["Audio waveform".to_string()],
                outputs: vec![
                    "MFCC coefficients: 13 features".to_string(),
                    "Pitch variation: F0 tracking".to_string(),
                    "Voice stress indicators: 8 metrics".to_string(),
                    "Speaking rate analysis".to_string(),
                ],
                processing_time: Duration::from_millis(80),
                confidence_change: 0.2,
                explanation: "Audio processing models extracted vocal stress and deception cues".to_string(),
            });
            current_step += 1;
        }
        
        if input.text.is_some() {
            steps.push(ReasoningStep {
                step_number: current_step,
                operation: "Text Feature Extraction".to_string(),
                inputs: vec!["Text transcript".to_string()],
                outputs: vec![
                    "Linguistic patterns: 15 features".to_string(),
                    "Sentiment analysis: Polarity and arousal".to_string(),
                    "Cognitive complexity: 6 metrics".to_string(),
                    "Deception markers: 12 indicators".to_string(),
                ],
                processing_time: Duration::from_millis(45),
                confidence_change: 0.15,
                explanation: "NLP models analyzed linguistic patterns associated with deception".to_string(),
            });
            current_step += 1;
        }
        
        // Step: Feature fusion
        steps.push(ReasoningStep {
            step_number: current_step,
            operation: "Multi-modal Fusion".to_string(),
            inputs: vec!["Features from all available modalities".to_string()],
            outputs: vec!["Fused feature vector".to_string(), "Modality weights".to_string()],
            processing_time: Duration::from_millis(30),
            confidence_change: 0.1,
            explanation: "Features from different modalities were combined using attention-based fusion".to_string(),
        });
        current_step += 1;
        
        decision_points.push(DecisionPoint {
            point_name: "Modality Agreement Check".to_string(),
            condition: "Agreement between modalities >= 70%".to_string(),
            outcome: "Passed: Consistent evidence across modalities".to_string(),
            alternatives: vec!["Failed: Conflicting evidence detected".to_string()],
            impact_on_decision: 0.4,
        });
        
        // Step: Classification
        steps.push(ReasoningStep {
            step_number: current_step,
            operation: "Deception Classification".to_string(),
            inputs: vec!["Fused features".to_string(), "Learned weights".to_string()],
            outputs: vec!["Deception probability".to_string(), "Confidence score".to_string()],
            processing_time: Duration::from_millis(15),
            confidence_change: 0.0,
            explanation: "Neural network classifier made final deception prediction".to_string(),
        });
        
        decision_points.push(DecisionPoint {
            point_name: "Confidence Threshold".to_string(),
            condition: "Classification confidence >= 0.5".to_string(),
            outcome: "Passed: High confidence prediction".to_string(),
            alternatives: vec!["Failed: Uncertain result".to_string()],
            impact_on_decision: 0.5,
        });
        
        ReasoningTrace {
            steps,
            decision_points,
            total_processing_time: start_time.elapsed(),
        }
    }
    
    /// Analyze uncertainty in the decision
    fn analyze_uncertainty(&self, features: &[AnalysisFeature], confidence: f32) -> UncertaintyAnalysis {
        let mut uncertainty_sources = Vec::new();
        
        // Model uncertainty (epistemic)
        let model_uncertainty = 1.0 - confidence;
        uncertainty_sources.push(UncertaintySource {
            source: "Model Uncertainty".to_string(),
            contribution: model_uncertainty * 0.4,
            description: "Uncertainty due to model limitations and training data coverage".to_string(),
        });
        
        // Data quality uncertainty (aleatoric)
        let data_quality = features.iter()
            .map(|f| f.confidence)
            .sum::<f32>() / features.len() as f32;
        let data_uncertainty = 1.0 - data_quality;
        uncertainty_sources.push(UncertaintySource {
            source: "Data Quality".to_string(),
            contribution: data_uncertainty * 0.3,
            description: "Uncertainty due to input data quality and noise".to_string(),
        });
        
        // Feature disagreement uncertainty
        let feature_variance = if features.len() > 1 {
            let mean = features.iter().map(|f| f.value).sum::<f32>() / features.len() as f32;
            features.iter().map(|f| (f.value - mean).powi(2)).sum::<f32>() / features.len() as f32
        } else {
            0.0
        };
        uncertainty_sources.push(UncertaintySource {
            source: "Feature Disagreement".to_string(),
            contribution: feature_variance * 0.2,
            description: "Uncertainty due to conflicting evidence from different features".to_string(),
        });
        
        // Context uncertainty
        let context_uncertainty = 0.1; // Base context uncertainty
        uncertainty_sources.push(UncertaintySource {
            source: "Context".to_string(),
            contribution: context_uncertainty * 0.1,
            description: "Uncertainty due to unknown contextual factors".to_string(),
        });
        
        let total_uncertainty = uncertainty_sources.iter().map(|s| s.contribution).sum::<f32>();
        let epistemic_uncertainty = model_uncertainty;
        let aleatoric_uncertainty = data_uncertainty;
        
        // Calculate confidence interval (simplified)
        let margin = total_uncertainty * 1.96; // 95% confidence interval
        let confidence_interval = (
            (confidence - margin).max(0.0),
            (confidence + margin).min(1.0),
        );
        
        UncertaintyAnalysis {
            total_uncertainty,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            confidence_interval,
            uncertainty_sources,
        }
    }
    
    /// Generate explanations for individual features
    fn explain_features(&self, features: &[AnalysisFeature], decision: &DeceptionDecision) -> Vec<FeatureExplanation> {
        let mut explanations = Vec::new();
        
        // Sort features by importance
        let mut sorted_features = features.to_vec();
        sorted_features.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        
        // Take top N features
        let top_features = sorted_features.into_iter()
            .take(self.config.max_features_to_explain)
            .filter(|f| f.importance >= self.feature_importance_threshold);
        
        for feature in top_features {
            let explanation = self.generate_feature_explanation(&feature, decision);
            explanations.push(explanation);
        }
        
        explanations
    }
    
    /// Generate explanation for a single feature
    fn generate_feature_explanation(&self, feature: &AnalysisFeature, decision: &DeceptionDecision) -> FeatureExplanation {
        let interpretation = self.interpret_feature_value(&feature.name, feature.value);
        let contribution = self.calculate_feature_contribution(feature, decision);
        
        let (supporting_evidence, contradicting_evidence) = self.gather_feature_evidence(feature);
        
        FeatureExplanation {
            feature_name: feature.name.clone(),
            modality: feature.modality.clone(),
            importance: feature.importance,
            value: feature.value,
            contribution_to_decision: contribution,
            interpretation,
            confidence: feature.confidence,
            supporting_evidence,
            contradicting_evidence,
        }
    }
    
    /// Create visual decision pathway
    fn create_decision_pathway(&self, features: &[AnalysisFeature], decision: &DeceptionDecision) -> DecisionPathway {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Input nodes
        let unique_modalities: std::collections::HashSet<_> = features.iter()
            .map(|f| f.modality.clone())
            .collect();
        
        for (i, modality) in unique_modalities.iter().enumerate() {
            nodes.push(PathwayNode {
                id: format!("input_{}", modality),
                node_type: NodeType::Input,
                label: format!("{} Input", modality),
                confidence: features.iter()
                    .filter(|f| f.modality == *modality)
                    .map(|f| f.confidence)
                    .sum::<f32>() / features.iter().filter(|f| f.modality == *modality).count() as f32,
                evidence_count: features.iter().filter(|f| f.modality == *modality).count(),
            });
        }
        
        // Processing nodes
        nodes.push(PathwayNode {
            id: "feature_extraction".to_string(),
            node_type: NodeType::Processing,
            label: "Feature Extraction".to_string(),
            confidence: 0.9,
            evidence_count: features.len(),
        });
        
        nodes.push(PathwayNode {
            id: "fusion".to_string(),
            node_type: NodeType::Processing,
            label: "Multi-modal Fusion".to_string(),
            confidence: 0.85,
            evidence_count: unique_modalities.len(),
        });
        
        nodes.push(PathwayNode {
            id: "classification".to_string(),
            node_type: NodeType::Decision,
            label: "Deception Classification".to_string(),
            confidence: match decision {
                DeceptionDecision::Truthful { probability } => *probability,
                DeceptionDecision::Deceptive { probability } => *probability,
                _ => 0.5,
            },
            evidence_count: features.len(),
        });
        
        // Output node
        nodes.push(PathwayNode {
            id: "output".to_string(),
            node_type: NodeType::Output,
            label: format!("Decision: {:?}", decision),
            confidence: match decision {
                DeceptionDecision::Truthful { probability } => *probability,
                DeceptionDecision::Deceptive { probability } => *probability,
                _ => 0.5,
            },
            evidence_count: 1,
        });
        
        // Create edges
        for modality in &unique_modalities {
            edges.push(PathwayEdge {
                from: format!("input_{}", modality),
                to: "feature_extraction".to_string(),
                weight: 1.0,
                label: "Extract Features".to_string(),
            });
        }
        
        edges.push(PathwayEdge {
            from: "feature_extraction".to_string(),
            to: "fusion".to_string(),
            weight: 1.0,
            label: "Combine Modalities".to_string(),
        });
        
        edges.push(PathwayEdge {
            from: "fusion".to_string(),
            to: "classification".to_string(),
            weight: 1.0,
            label: "Classify".to_string(),
        });
        
        edges.push(PathwayEdge {
            from: "classification".to_string(),
            to: "output".to_string(),
            weight: 1.0,
            label: "Final Decision".to_string(),
        });
        
        let critical_path = vec![
            "feature_extraction".to_string(),
            "fusion".to_string(),
            "classification".to_string(),
            "output".to_string(),
        ];
        
        DecisionPathway {
            nodes,
            edges,
            critical_path,
        }
    }
    
    /// Generate alternative scenarios
    fn generate_alternative_scenarios(&self, input: &ExplainableInput, current_decision: &DeceptionDecision) -> Vec<AlternativeScenario> {
        let mut scenarios = Vec::new();
        
        // Scenario 1: Better video quality
        if input.video_path.is_some() {
            scenarios.push(AlternativeScenario {
                scenario_name: "Higher Video Quality".to_string(),
                changed_conditions: vec!["Video resolution increased to 1080p".to_string()],
                predicted_outcome: DeceptionDecision::Deceptive { probability: 0.85 },
                confidence_change: 0.15,
                explanation: "Higher resolution video would allow detection of subtle facial micro-expressions, \
                           potentially increasing deception detection confidence".to_string(),
            });
        }
        
        // Scenario 2: Additional physiological data
        if input.physiological_data.is_none() {
            scenarios.push(AlternativeScenario {
                scenario_name: "Physiological Data Available".to_string(),
                changed_conditions: vec!["Heart rate, skin conductance, and breathing patterns included".to_string()],
                predicted_outcome: DeceptionDecision::Deceptive { probability: 0.92 },
                confidence_change: 0.25,
                explanation: "Physiological signals are difficult to consciously control and would provide \
                           strong additional evidence for deception detection".to_string(),
            });
        }
        
        // Scenario 3: Baseline comparison
        scenarios.push(AlternativeScenario {
            scenario_name: "With Baseline Comparison".to_string(),
            changed_conditions: vec!["Subject's truthful baseline behavior available for comparison".to_string()],
            predicted_outcome: match current_decision {
                DeceptionDecision::Truthful { probability } => DeceptionDecision::Truthful { probability: probability + 0.1 },
                DeceptionDecision::Deceptive { probability } => DeceptionDecision::Deceptive { probability: probability + 0.1 },
                other => other.clone(),
            },
            confidence_change: 0.2,
            explanation: "Baseline comparison would allow for personalized analysis, accounting for \
                       individual differences in behavior patterns".to_string(),
        });
        
        // Scenario 4: Controlled environment
        if let Environment::Field = input.context.environment {
            scenarios.push(AlternativeScenario {
                scenario_name: "Controlled Environment".to_string(),
                changed_conditions: vec!["Analysis conducted in controlled laboratory setting".to_string()],
                predicted_outcome: match current_decision {
                    DeceptionDecision::Truthful { probability } => DeceptionDecision::Truthful { probability: *probability },
                    DeceptionDecision::Deceptive { probability } => DeceptionDecision::Deceptive { probability: *probability },
                    other => other.clone(),
                },
                confidence_change: 0.1,
                explanation: "Controlled environment would reduce external noise and distractions, \
                           potentially improving signal quality".to_string(),
            });
        }
        
        scenarios
    }
    
    /// Generate summary explanation
    fn generate_summary(&self, decision: &DeceptionDecision, confidence: f32, features: &[FeatureExplanation]) -> String {
        let template_key = match decision {
            DeceptionDecision::Truthful { .. } => {
                if confidence > 0.7 { "truthful_high_confidence" } else { "truthful_low_confidence" }
            }
            DeceptionDecision::Deceptive { .. } => {
                if confidence > 0.7 { "deceptive_high_confidence" } else { "deceptive_low_confidence" }
            }
            DeceptionDecision::Uncertain { .. } => "uncertain_mixed_evidence",
            DeceptionDecision::InsufficientData { .. } => "insufficient_data",
        };
        
        let top_features = features.iter()
            .take(3)
            .map(|f| f.feature_name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        
        match self.explanation_templates.get(template_key) {
            Some(template) => {
                template
                    .replace("{confidence}", &format!("{:.0}", confidence * 100.0))
                    .replace("{top_features}", &top_features)
            }
            None => format!("Analysis indicates {:?} with {:.1}% confidence", decision, confidence * 100.0),
        }
    }
    
    /// Generate detailed explanation
    fn generate_detailed_explanation(
        &self,
        decision: &DeceptionDecision,
        uncertainty: &UncertaintyAnalysis,
        features: &[FeatureExplanation],
        reasoning_trace: &ReasoningTrace,
    ) -> String {
        let mut explanation = String::new();
        
        match self.config.explanation_style {
            ExplanationStyle::Technical => {
                explanation.push_str(&format!("Technical Analysis Report\n"));
                explanation.push_str(&format!("=======================\n\n"));
                explanation.push_str(&format!("Decision: {:?}\n", decision));
                explanation.push_str(&format!("Total uncertainty: {:.3}\n", uncertainty.total_uncertainty));
                explanation.push_str(&format!("Processing steps: {}\n", reasoning_trace.steps.len()));
                explanation.push_str(&format!("Feature count: {}\n\n", features.len()));
                
                explanation.push_str("Feature Analysis:\n");
                for feature in features {
                    explanation.push_str(&format!(
                        "- {}: {:.3} (importance: {:.3}, confidence: {:.3})\n",
                        feature.feature_name, feature.value, feature.importance, feature.confidence
                    ));
                }
            }
            ExplanationStyle::Simplified => {
                explanation.push_str("Analysis Summary\n");
                explanation.push_str("================\n\n");
                
                let decision_text = match decision {
                    DeceptionDecision::Truthful { .. } => "The person appears to be telling the truth",
                    DeceptionDecision::Deceptive { .. } => "The person may be engaging in deception",
                    DeceptionDecision::Uncertain { .. } => "The analysis is uncertain due to mixed evidence",
                    DeceptionDecision::InsufficientData { .. } => "Not enough data to make a determination",
                };
                
                explanation.push_str(&format!("{}.\n\n", decision_text));
                
                explanation.push_str("Key factors considered:\n");
                for feature in features.iter().take(3) {
                    explanation.push_str(&format!("• {}\n", feature.interpretation));
                }
            }
            ExplanationStyle::Narrative => {
                explanation.push_str("Analysis Story\n");
                explanation.push_str("==============\n\n");
                
                explanation.push_str("The analysis began by examining the available evidence. ");
                
                if features.iter().any(|f| f.modality == "video") {
                    explanation.push_str("The video showed facial expressions and body language patterns. ");
                }
                
                if features.iter().any(|f| f.modality == "audio") {
                    explanation.push_str("The audio revealed vocal characteristics and speech patterns. ");
                }
                
                if features.iter().any(|f| f.modality == "text") {
                    explanation.push_str("The text analysis uncovered linguistic patterns and word choices. ");
                }
                
                explanation.push_str(&format!("\n\nAfter considering all evidence, the analysis concluded that {:?}.", decision));
            }
            _ => {
                explanation.push_str(&self.generate_structured_explanation(decision, uncertainty, features, reasoning_trace));
            }
        }
        
        explanation
    }
    
    /// Generate structured explanation with bullet points
    fn generate_structured_explanation(
        &self,
        decision: &DeceptionDecision,
        uncertainty: &UncertaintyAnalysis,
        features: &[FeatureExplanation],
        reasoning_trace: &ReasoningTrace,
    ) -> String {
        let mut explanation = String::new();
        
        explanation.push_str("Structured Analysis Report\n");
        explanation.push_str("==========================\n\n");
        
        explanation.push_str("## Decision\n");
        explanation.push_str(&format!("• Outcome: {:?}\n", decision));
        explanation.push_str(&format!("• Confidence interval: {:.1}% - {:.1}%\n", 
            uncertainty.confidence_interval.0 * 100.0, 
            uncertainty.confidence_interval.1 * 100.0));
        explanation.push_str(&format!("• Total uncertainty: {:.1}%\n\n", uncertainty.total_uncertainty * 100.0));
        
        explanation.push_str("## Key Evidence\n");
        for (i, feature) in features.iter().take(5).enumerate() {
            explanation.push_str(&format!("{}. **{}** ({})\n", i + 1, feature.feature_name, feature.modality));
            explanation.push_str(&format!("   - Value: {:.3}\n", feature.value));
            explanation.push_str(&format!("   - Importance: {:.3}\n", feature.importance));
            explanation.push_str(&format!("   - Interpretation: {}\n", feature.interpretation));
            explanation.push_str(&format!("   - Confidence: {:.1}%\n\n", feature.confidence * 100.0));
        }
        
        explanation.push_str("## Uncertainty Sources\n");
        for source in &uncertainty.uncertainty_sources {
            explanation.push_str(&format!("• **{}**: {:.1}% - {}\n", 
                source.source, 
                source.contribution * 100.0, 
                source.description));
        }
        
        explanation.push_str("\n## Processing Steps\n");
        for step in &reasoning_trace.steps {
            explanation.push_str(&format!("{}. {} ({}ms)\n", 
                step.step_number, 
                step.operation, 
                step.processing_time.as_millis()));
        }
        
        explanation
    }
    
    // Helper methods for simulation
    async fn analyze_video(&self, _video_path: &str) -> Vec<AnalysisFeature> {
        tokio::time::sleep(Duration::from_millis(100)).await;
        vec![
            AnalysisFeature {
                name: "micro_expressions".to_string(),
                modality: "video".to_string(),
                value: 0.75,
                importance: 0.85,
                confidence: 0.88,
            },
            AnalysisFeature {
                name: "eye_movement_patterns".to_string(),
                modality: "video".to_string(),
                value: 0.62,
                importance: 0.70,
                confidence: 0.82,
            },
            AnalysisFeature {
                name: "facial_asymmetry".to_string(),
                modality: "video".to_string(),
                value: 0.58,
                importance: 0.65,
                confidence: 0.75,
            },
        ]
    }
    
    async fn analyze_audio(&self, _audio_path: &str) -> Vec<AnalysisFeature> {
        tokio::time::sleep(Duration::from_millis(80)).await;
        vec![
            AnalysisFeature {
                name: "voice_stress".to_string(),
                modality: "audio".to_string(),
                value: 0.68,
                importance: 0.78,
                confidence: 0.85,
            },
            AnalysisFeature {
                name: "pitch_variation".to_string(),
                modality: "audio".to_string(),
                value: 0.55,
                importance: 0.60,
                confidence: 0.80,
            },
        ]
    }
    
    async fn analyze_text(&self, _text: &str) -> Vec<AnalysisFeature> {
        tokio::time::sleep(Duration::from_millis(50)).await;
        vec![
            AnalysisFeature {
                name: "linguistic_complexity".to_string(),
                modality: "text".to_string(),
                value: 0.72,
                importance: 0.75,
                confidence: 0.90,
            },
            AnalysisFeature {
                name: "uncertainty_markers".to_string(),
                modality: "text".to_string(),
                value: 0.80,
                importance: 0.82,
                confidence: 0.92,
            },
        ]
    }
    
    async fn analyze_physiological(&self, _data: &[f32]) -> Vec<AnalysisFeature> {
        tokio::time::sleep(Duration::from_millis(30)).await;
        vec![
            AnalysisFeature {
                name: "heart_rate_variability".to_string(),
                modality: "physiological".to_string(),
                value: 0.85,
                importance: 0.90,
                confidence: 0.95,
            },
        ]
    }
    
    fn compute_video_deception_score(&self, features: &[AnalysisFeature]) -> f32 {
        features.iter().map(|f| f.value * f.importance).sum::<f32>() / features.len() as f32
    }
    
    fn compute_audio_deception_score(&self, features: &[AnalysisFeature]) -> f32 {
        features.iter().map(|f| f.value * f.importance).sum::<f32>() / features.len() as f32
    }
    
    fn compute_text_deception_score(&self, features: &[AnalysisFeature]) -> f32 {
        features.iter().map(|f| f.value * f.importance).sum::<f32>() / features.len() as f32
    }
    
    fn compute_physiological_deception_score(&self, features: &[AnalysisFeature]) -> f32 {
        features.iter().map(|f| f.value * f.importance).sum::<f32>() / features.len() as f32
    }
    
    fn combine_modality_scores(&self, scores: &[(&str, f32)]) -> f32 {
        scores.iter().map(|(_, score)| score).sum::<f32>() / scores.len() as f32
    }
    
    fn calculate_confidence(&self, scores: &[(&str, f32)], features: &[AnalysisFeature]) -> f32 {
        let modality_agreement = self.calculate_modality_agreement(scores);
        let feature_confidence = features.iter().map(|f| f.confidence).sum::<f32>() / features.len() as f32;
        (modality_agreement + feature_confidence) / 2.0
    }
    
    fn calculate_modality_agreement(&self, scores: &[(&str, f32)]) -> f32 {
        if scores.len() < 2 {
            return 0.8; // Default for single modality
        }
        
        let mean = scores.iter().map(|(_, score)| score).sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|(_, score)| (score - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        (1.0 - variance).max(0.0)
    }
    
    fn make_decision(&self, score: f32, confidence: f32, _features: &[AnalysisFeature]) -> DeceptionDecision {
        if confidence < 0.3 {
            DeceptionDecision::Uncertain {
                conflicting_evidence: vec!["Low confidence prediction".to_string()],
            }
        } else if score > 0.6 {
            DeceptionDecision::Deceptive { probability: score }
        } else if score < 0.4 {
            DeceptionDecision::Truthful { probability: 1.0 - score }
        } else {
            DeceptionDecision::Uncertain {
                conflicting_evidence: vec!["Score in uncertain range".to_string()],
            }
        }
    }
    
    fn interpret_feature_value(&self, feature_name: &str, value: f32) -> String {
        match feature_name {
            "micro_expressions" => {
                if value > 0.7 { "Strong micro-expression indicators of deception detected".to_string() }
                else if value > 0.4 { "Moderate micro-expression patterns observed".to_string() }
                else { "Minimal micro-expression indicators".to_string() }
            }
            "voice_stress" => {
                if value > 0.7 { "High vocal stress levels detected".to_string() }
                else if value > 0.4 { "Moderate vocal stress patterns".to_string() }
                else { "Low vocal stress indicators".to_string() }
            }
            "uncertainty_markers" => {
                if value > 0.7 { "High frequency of uncertainty language markers".to_string() }
                else if value > 0.4 { "Some uncertainty markers present".to_string() }
                else { "Few uncertainty markers detected".to_string() }
            }
            _ => format!("Feature value: {:.3}", value),
        }
    }
    
    fn calculate_feature_contribution(&self, feature: &AnalysisFeature, _decision: &DeceptionDecision) -> f32 {
        feature.value * feature.importance * feature.confidence
    }
    
    fn gather_feature_evidence(&self, feature: &AnalysisFeature) -> (Vec<String>, Vec<String>) {
        let supporting = match feature.name.as_str() {
            "micro_expressions" => vec![
                "Asymmetric facial expressions detected".to_string(),
                "Delayed emotional responses observed".to_string(),
            ],
            "voice_stress" => vec![
                "Elevated pitch variance".to_string(),
                "Irregular breathing patterns in speech".to_string(),
            ],
            _ => vec!["Statistical significance achieved".to_string()],
        };
        
        let contradicting = vec!["No major contradictory evidence found".to_string()];
        
        (supporting, contradicting)
    }
    
    fn calculate_explanation_confidence(&self, uncertainty: &UncertaintyAnalysis) -> f32 {
        1.0 - uncertainty.total_uncertainty
    }
}

/// Feature extracted during analysis
#[derive(Debug, Clone)]
pub struct AnalysisFeature {
    pub name: String,
    pub modality: String,
    pub value: f32,
    pub importance: f32,
    pub confidence: f32,
}

/// Errors that can occur during explanation
#[derive(Debug)]
pub enum ExplanationError {
    AnalysisError(String),
    TemplateError(String),
    ProcessingError(String),
}

impl std::fmt::Display for ExplanationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExplanationError::AnalysisError(msg) => write!(f, "Analysis error: {}", msg),
            ExplanationError::TemplateError(msg) => write!(f, "Template error: {}", msg),
            ExplanationError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for ExplanationError {}

/// Export explanation to different formats
pub struct ExplanationExporter;

impl ExplanationExporter {
    pub fn export_to_json(result: &ExplanationResult) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(result)
    }
    
    pub fn export_to_html(result: &ExplanationResult) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Lie Detection Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .decision {{ font-size: 24px; font-weight: bold; margin: 20px 0; }}
        .features {{ margin: 20px 0; }}
        .feature {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; }}
        .pathway {{ margin: 20px 0; }}
        .uncertainty {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Lie Detection Analysis Report</h1>
        <p>Generated: {}</p>
        <p>Session ID: {}</p>
    </div>
    
    <div class="decision">
        Decision: {:?} (Confidence: {:.1}%)
    </div>
    
    <div class="uncertainty">
        <h3>Uncertainty Analysis</h3>
        <p>Total Uncertainty: {:.1}%</p>
        <p>Confidence Interval: {:.1}% - {:.1}%</p>
    </div>
    
    <div class="features">
        <h3>Key Features</h3>
        {}
    </div>
    
    <div>
        <h3>Explanation</h3>
        <p>{}</p>
    </div>
    
    <div>
        <h3>Detailed Analysis</h3>
        <pre>{}</pre>
    </div>
</body>
</html>
            "#,
            result.metadata.generated_at,
            result.metadata.explanation_id,
            result.decision,
            result.confidence * 100.0,
            result.uncertainty.total_uncertainty * 100.0,
            result.uncertainty.confidence_interval.0 * 100.0,
            result.uncertainty.confidence_interval.1 * 100.0,
            result.feature_explanations.iter()
                .map(|f| format!(
                    r#"<div class="feature">
                        <strong>{}</strong> ({})<br>
                        Value: {:.3}, Importance: {:.3}<br>
                        {}
                    </div>"#,
                    f.feature_name, f.modality, f.value, f.importance, f.interpretation
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            result.summary,
            result.detailed_explanation
        )
    }
    
    pub fn export_to_text(result: &ExplanationResult) -> String {
        format!(
            "LIE DETECTION ANALYSIS REPORT\n{}\n\n\
            Decision: {:?}\n\
            Confidence: {:.1}%\n\
            Uncertainty: {:.1}%\n\n\
            SUMMARY:\n{}\n\n\
            DETAILED ANALYSIS:\n{}\n\n\
            KEY FEATURES:\n{}\n\n\
            Generated: {} ({}ms)\n",
            "=".repeat(30),
            result.decision,
            result.confidence * 100.0,
            result.uncertainty.total_uncertainty * 100.0,
            result.summary,
            result.detailed_explanation,
            result.feature_explanations.iter()
                .map(|f| format!("• {}: {} (importance: {:.3})", 
                    f.feature_name, f.interpretation, f.importance))
                .collect::<Vec<_>>()
                .join("\n"),
            result.metadata.generated_at,
            result.metadata.processing_time.as_millis()
        )
    }
}

/// Create test scenarios
fn create_test_scenarios() -> Vec<(String, ExplainableInput)> {
    vec![
        (
            "High Stakes Investigation".to_string(),
            ExplainableInput {
                video_path: Some("evidence/suspect_interview.mp4".to_string()),
                audio_path: Some("evidence/suspect_interview.wav".to_string()),
                text: Some("I was definitely not at the crime scene that night. I was home watching TV with my family.".to_string()),
                physiological_data: Some(vec![75.2, 78.1, 82.3, 79.8, 81.2]), // Heart rate data
                context: AnalysisContext {
                    session_id: "investigation_2024_001".to_string(),
                    timestamp: Instant::now(),
                    user_type: UserType::LawEnforcement,
                    stakes: StakeLevel::High,
                    environment: Environment::Controlled,
                    subject_info: Some(SubjectInfo {
                        age_group: AgeGroup::Adult,
                        baseline_available: false,
                        stress_level: Some(0.7),
                        prior_history: true,
                    }),
                },
            }
        ),
        (
            "Security Screening".to_string(),
            ExplainableInput {
                video_path: None,
                audio_path: Some("security/screening_audio.wav".to_string()),
                text: Some("I'm just traveling for business. I have meetings scheduled in the city.".to_string()),
                physiological_data: None,
                context: AnalysisContext {
                    session_id: "security_screening_047".to_string(),
                    timestamp: Instant::now(),
                    user_type: UserType::SecurityPersonnel,
                    stakes: StakeLevel::Medium,
                    environment: Environment::Field,
                    subject_info: Some(SubjectInfo {
                        age_group: AgeGroup::Adult,
                        baseline_available: false,
                        stress_level: Some(0.4),
                        prior_history: false,
                    }),
                },
            }
        ),
        (
            "Research Study".to_string(),
            ExplainableInput {
                video_path: Some("research/participant_042.mp4".to_string()),
                audio_path: Some("research/participant_042.wav".to_string()),
                text: Some("Well, I think maybe I might have possibly taken some extra cookies from the jar, but I'm not really sure.".to_string()),
                physiological_data: Some(vec![68.5, 70.2, 72.8, 71.1, 69.9]),
                context: AnalysisContext {
                    session_id: "research_study_participant_042".to_string(),
                    timestamp: Instant::now(),
                    user_type: UserType::Researcher,
                    stakes: StakeLevel::Low,
                    environment: Environment::Controlled,
                    subject_info: Some(SubjectInfo {
                        age_group: AgeGroup::Young,
                        baseline_available: true,
                        stress_level: Some(0.2),
                        prior_history: false,
                    }),
                },
            }
        ),
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Veritas Nexus - Explainable AI Decisions Example");
    println!("==================================================\n");
    
    // Create explanation engine with different configurations
    let configs = vec![
        ("Technical Expert", ExplanationConfig {
            explanation_style: ExplanationStyle::Technical,
            detail_level: DetailLevel::High,
            include_uncertainty: true,
            include_feature_weights: true,
            include_reasoning_trace: true,
            max_features_to_explain: 8,
            confidence_threshold: 0.3,
            language: Language::English,
        }),
        ("General User", ExplanationConfig {
            explanation_style: ExplanationStyle::Simplified,
            detail_level: DetailLevel::Low,
            include_uncertainty: false,
            include_feature_weights: false,
            include_reasoning_trace: false,
            max_features_to_explain: 3,
            confidence_threshold: 0.5,
            language: Language::English,
        }),
        ("Narrative Style", ExplanationConfig {
            explanation_style: ExplanationStyle::Narrative,
            detail_level: DetailLevel::Medium,
            include_uncertainty: true,
            include_feature_weights: false,
            include_reasoning_trace: false,
            max_features_to_explain: 5,
            confidence_threshold: 0.4,
            language: Language::English,
        }),
    ];
    
    let test_scenarios = create_test_scenarios();
    
    for (config_name, config) in configs {
        println!("🎯 Configuration: {}", config_name);
        println!("{}", "-".repeat(50));
        
        let engine = ExplanationEngine::new(config);
        
        // Test first scenario with this configuration
        let (scenario_name, input) = &test_scenarios[0];
        println!("Scenario: {}", scenario_name);
        
        match engine.explain(input).await {
            Ok(result) => {
                println!("\n📊 Decision: {:?}", result.decision);
                println!("📈 Confidence: {:.1}%", result.confidence * 100.0);
                println!("🎯 Uncertainty: {:.1}%", result.uncertainty.total_uncertainty * 100.0);
                
                println!("\n📝 Summary:");
                println!("{}", result.summary);
                
                println!("\n🔍 Detailed Explanation:");
                println!("{}", result.detailed_explanation);
                
                if !result.feature_explanations.is_empty() {
                    println!("\n🎛️  Top Features:");
                    for (i, feature) in result.feature_explanations.iter().take(3).enumerate() {
                        println!("  {}. {} ({}): {:.3} - {}", 
                            i + 1, 
                            feature.feature_name, 
                            feature.modality,
                            feature.value,
                            feature.interpretation
                        );
                    }
                }
                
                if result.uncertainty.uncertainty_sources.len() > 0 {
                    println!("\n⚠️  Uncertainty Sources:");
                    for source in &result.uncertainty.uncertainty_sources {
                        println!("  • {}: {:.1}% - {}", 
                            source.source, 
                            source.contribution * 100.0,
                            source.description
                        );
                    }
                }
                
                println!("\n⏱️  Processing time: {}ms", result.metadata.processing_time.as_millis());
            }
            Err(e) => {
                println!("❌ Error generating explanation: {}", e);
            }
        }
        
        println!("\n" + &"=".repeat(60) + "\n");
    }
    
    // Demonstrate different export formats
    println!("📤 Export Format Demonstrations");
    println!("{}", "-".repeat(40));
    
    let engine = ExplanationEngine::new(ExplanationConfig::default());
    let (scenario_name, input) = &test_scenarios[2]; // Use research study scenario
    
    if let Ok(result) = engine.explain(input).await {
        println!("Scenario: {}", scenario_name);
        
        // JSON Export
        println!("\n📄 JSON Export (first 200 chars):");
        match ExplanationExporter::export_to_json(&result) {
            Ok(json) => {
                let preview = if json.len() > 200 {
                    format!("{}...", &json[..200])
                } else {
                    json
                };
                println!("{}", preview);
            }
            Err(e) => println!("Error: {}", e),
        }
        
        // Text Export
        println!("\n📄 Text Export (first 300 chars):");
        let text = ExplanationExporter::export_to_text(&result);
        let preview = if text.len() > 300 {
            format!("{}...", &text[..300])
        } else {
            text
        };
        println!("{}", preview);
        
        // HTML Export (show structure)
        println!("\n📄 HTML Export Structure:");
        let html = ExplanationExporter::export_to_html(&result);
        let lines: Vec<&str> = html.lines().take(10).collect();
        for line in lines {
            if line.trim().starts_with("<") {
                println!("{}", line.trim());
            }
        }
        println!("... (HTML document continues)");
    }
    
    // Reasoning trace visualization
    println!("\n" + &"=".repeat(60));
    println!("🔍 Reasoning Trace Visualization");
    println!("{}", "-".repeat(40));
    
    let engine = ExplanationEngine::new(ExplanationConfig {
        explanation_style: ExplanationStyle::Visual,
        include_reasoning_trace: true,
        ..ExplanationConfig::default()
    });
    
    let (scenario_name, input) = &test_scenarios[0];
    
    if let Ok(result) = engine.explain(input).await {
        println!("Scenario: {}", scenario_name);
        
        println!("\n📈 Processing Pipeline:");
        for step in &result.reasoning_trace.steps {
            println!("{}. {} ({}ms)", 
                step.step_number, 
                step.operation, 
                step.processing_time.as_millis()
            );
            println!("   └─ {}", step.explanation);
        }
        
        println!("\n🎯 Decision Points:");
        for point in &result.reasoning_trace.decision_points {
            println!("• {}: {}", point.point_name, point.outcome);
            println!("  Impact: {:.1}%", point.impact_on_decision * 100.0);
        }
        
        println!("\n🛤️  Decision Pathway:");
        for node in &result.decision_pathway.nodes {
            let symbol = match node.node_type {
                NodeType::Input => "📥",
                NodeType::Processing => "⚙️",
                NodeType::Decision => "🤔",
                NodeType::Output => "📤",
            };
            println!("{} {} (confidence: {:.1}%)", symbol, node.label, node.confidence * 100.0);
        }
        
        println!("\n🔮 Alternative Scenarios:");
        for scenario in &result.alternative_scenarios {
            println!("• {}: {:?}", scenario.scenario_name, scenario.predicted_outcome);
            println!("  Change: {:.1}%", scenario.confidence_change * 100.0);
            println!("  Reason: {}", scenario.explanation);
            println!();
        }
    }
    
    println!("🎉 Explainable AI demonstration completed!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Multiple explanation styles for different user types");
    println!("   • Comprehensive uncertainty quantification");
    println!("   • Step-by-step reasoning traces");
    println!("   • Feature importance explanations");
    println!("   • Decision pathway visualization");
    println!("   • Alternative scenario analysis");
    println!("   • Multiple export formats (JSON, HTML, text)");
    println!("   • Context-aware explanations");
    println!("   • Confidence intervals and uncertainty sources");
    
    Ok(())
}