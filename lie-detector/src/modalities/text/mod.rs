//! Text modality analysis for deception detection.
//!
//! This module provides comprehensive natural language processing capabilities
//! for detecting deception through linguistic analysis. It combines traditional
//! linguistic features with modern deep learning approaches to identify patterns
//! indicative of deceptive communication.
//!
//! # Core Components
//!
//! - **Text Analysis**: [`TextAnalyzer`] - Main entry point for text-based deception detection
//! - **Linguistic Processing**: [`LinguisticAnalyzer`] - Extracts grammatical and stylistic features
//! - **Deep Learning**: [`BertIntegration`] - BERT-based semantic embeddings and analysis
//! - **Pattern Detection**: [`DeceptionPatternDetector`] - Specialized deception indicator recognition
//! - **Performance**: [`SimdTextAnalyzer`] - SIMD-optimized processing for high-throughput scenarios
//!
//! # Supported Features
//!
//! ## Linguistic Analysis
//! - Lexical diversity and complexity metrics
//! - Syntactic patterns and grammatical structures
//! - Discourse markers and hedging language
//! - Emotional expression and sentiment analysis
//! - Named entity recognition and contextual analysis
//!
//! ## Deception Indicators
//! - Hesitation markers and verbal fillers
//! - Cognitive load indicators (complexity, contradictions)
//! - Temporal inconsistencies in narratives
//! - Distancing language and responsibility attribution
//! - Specificity and detail levels
//!
//! ## Multi-Language Support
//! - Automatic language detection
//! - Language-specific linguistic features
//! - Cross-lingual semantic embeddings
//! - Cultural and linguistic bias mitigation
//!
//! # Examples
//!
//! Basic text analysis:
//!
//! ```rust,no_run
//! use veritas_nexus::modalities::text::{TextAnalyzer, TextAnalyzerConfig, TextInput};
//! use veritas_nexus::ModalityAnalyzer;
//!
//! #[tokio::main]
//! async fn main() -> veritas_nexus::Result<()> {
//!     let config = TextAnalyzerConfig::default();
//!     let analyzer = TextAnalyzer::new(config)?;
//!     
//!     let input = TextInput {
//!         text: "I was definitely at home all evening and didn't go anywhere.".to_string(),
//!         context: Default::default(),
//!         language: None, // Auto-detect
//!         timestamp: None,
//!     };
//!     
//!     let result = analyzer.analyze(&input).await?;
//!     
//!     println!("Deception probability: {:.2}", result.probability());
//!     println!("Confidence: {:.2}", result.confidence());
//!     
//!     // Examine contributing features
//!     for feature in result.features() {
//!         println!("{}: {:.3} (weight: {:.2})", 
//!             feature.name, feature.value, feature.weight);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! Advanced configuration:
//!
//! ```rust,no_run
//! use veritas_nexus::modalities::text::{
//!     TextAnalyzerConfig, BertConfig, PreprocessingConfig, FeatureWeights
//! };
//!
//! let config = TextAnalyzerConfig {
//!     bert_config: BertConfig {
//!         model_name: "distilbert-base-uncased".to_string(),
//!         max_sequence_length: 512,
//!         use_gpu: false,
//!     },
//!     preprocessing_config: PreprocessingConfig {
//!         normalize_unicode: true,
//!         remove_extra_whitespace: true,
//!         preserve_case: false,
//!         language_detection: true,
//!     },
//!     feature_weights: FeatureWeights {
//!         linguistic_complexity: 0.25,
//!         hesitation_markers: 0.20,
//!         emotional_indicators: 0.15,
//!         semantic_coherence: 0.25,
//!         deception_patterns: 0.15,
//!     },
//!     enable_sentiment: true,
//!     enable_ner: true,
//!     enable_caching: true,
//!     max_cache_size: 5000,
//!     timeout_ms: 10000,
//!     confidence_threshold: 0.6,
//!     enable_parallel: true,
//! };
//! ```
//!
//! # Performance Considerations
//!
//! - **Caching**: Enable result caching for repeated analysis of similar texts
//! - **Parallel Processing**: Use parallel feature extraction for large documents
//! - **SIMD Optimization**: Consider [`SimdTextAnalyzer`] for high-throughput scenarios
//! - **Model Size**: Balance accuracy vs. speed when choosing BERT model variants
//! - **Preprocessing**: Efficient text normalization reduces downstream processing costs
//!
//! # Accuracy and Limitations
//!
//! ## Strengths
//! - High accuracy on structured deceptive statements
//! - Robust to variations in writing style and domain
//! - Language-agnostic core features
//! - Explainable feature-based scoring
//!
//! ## Limitations
//! - Reduced accuracy on very short texts (< 50 words)
//! - Cultural and demographic biases in linguistic patterns
//! - Difficulty with creative or metaphorical language
//! - Performance degrades with poor grammar or spelling
//!
//! ## Best Practices
//! - Combine with other modalities for robust detection
//! - Use confidence thresholds to filter uncertain predictions
//! - Consider context and domain when interpreting results
//! - Regularly retrain and validate models on diverse datasets

mod linguistic_analyzer;
mod bert_integration;
mod deception_patterns;
mod simd_optimized;

pub use linguistic_analyzer::{
    LinguisticAnalyzer, Language, PreprocessingConfig, LinguisticFeatures, 
    SentimentResult, NamedEntity, EntityType, EmotionScore, AnalyzedToken,
    PosTag, ComplexityMetrics, TemporalPattern, CognitiveLoadIndicators,
    SemanticCoherence
};
pub use bert_integration::{BertIntegration, BertConfig, BertEmbedding};
pub use deception_patterns::{DeceptionPatternDetector, FeatureWeights, DeceptionPatterns};
pub use simd_optimized::{SimdTextAnalyzer, SimdPatternMatcher, SimdTextVectorizer};

use crate::{
    ModalityAnalyzer, DeceptionScore, ModalityType, Feature, ExplanationTrace, 
    ExplanationStep, Result, VeritasError,
};
use crate::types::*;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use async_trait::async_trait;

/// Configuration for the text analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAnalyzerConfig {
    /// BERT model configuration
    pub bert_config: BertConfig,
    
    /// Text preprocessing configuration
    pub preprocessing_config: PreprocessingConfig,
    
    /// Feature weights for different aspects
    pub feature_weights: FeatureWeights<f64>,
    
    /// Enable sentiment analysis
    pub enable_sentiment: bool,
    
    /// Enable named entity recognition
    pub enable_ner: bool,
    
    /// Enable caching of results
    pub enable_caching: bool,
    
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    
    /// Timeout for processing in milliseconds
    pub timeout_ms: u64,
    
    /// Minimum confidence threshold for predictions
    pub confidence_threshold: f64,
    
    /// Enable parallel processing where possible
    pub enable_parallel: bool,
}

impl Default for TextAnalyzerConfig {
    fn default() -> Self {
        Self {
            bert_config: BertConfig::default(),
            preprocessing_config: PreprocessingConfig::default(),
            feature_weights: FeatureWeights::default(),
            enable_sentiment: true,
            enable_ner: true,
            enable_caching: true,
            max_cache_size: 1000,
            timeout_ms: 30000, // 30 seconds
            confidence_threshold: 0.5,
            enable_parallel: true,
        }
    }
}

/// Text input for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInput {
    /// The text to analyze
    pub text: String,
    
    /// Optional context metadata
    pub context: HashMap<String, serde_json::Value>,
    
    /// Optional language hint (auto-detected if not provided)
    pub language: Option<Language>,
    
    /// Optional timestamp for temporal analysis
    pub timestamp: Option<SystemTime>,
    
    /// Optional speaker/author identifier
    pub speaker_id: Option<String>,
}

impl TextInput {
    /// Create a new text input from a string
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            context: HashMap::new(),
            language: None,
            timestamp: None,
            speaker_id: None,
        }
    }
    
    /// Add context metadata
    pub fn with_context(mut self, key: String, value: serde_json::Value) -> Self {
        self.context.insert(key, value);
        self
    }
    
    /// Set the language
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }
    
    /// Set the timestamp
    pub fn with_timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = Some(timestamp);
        self
    }
    
    /// Set the speaker ID
    pub fn with_speaker_id<S: Into<String>>(mut self, speaker_id: S) -> Self {
        self.speaker_id = Some(speaker_id.into());
        self
    }
}

/// Text analysis result with deception score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextScore<T: Float> {
    /// Overall deception probability (0.0 to 1.0)
    pub probability: T,
    
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: T,
    
    /// Detected language
    pub language: Language,
    
    /// Linguistic features extracted
    pub linguistic_features: LinguisticFeatures<T>,
    
    /// BERT embeddings
    pub bert_embeddings: Option<BertEmbedding<T>>,
    
    /// Sentiment analysis result
    pub sentiment: Option<SentimentResult<T>>,
    
    /// Deception patterns detected
    pub deception_patterns: DeceptionPatterns<T>,
    
    /// Complexity metrics
    pub complexity: ComplexityMetrics<T>,
    
    /// Named entities found
    pub named_entities: Vec<NamedEntity>,
    
    /// Analyzed tokens
    pub tokens: Vec<AnalyzedToken>,
    
    /// Temporal patterns
    pub temporal_patterns: TemporalPattern<T>,
    
    /// Cognitive load indicators
    pub cognitive_load: CognitiveLoadIndicators<T>,
    
    /// Semantic coherence metrics
    pub semantic_coherence: SemanticCoherence<T>,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Timestamp of analysis
    pub timestamp: SystemTime,
    
    /// Individual feature contributions
    pub feature_contributions: Vec<Feature<T>>,
}

impl<T: Float> DeceptionScore<T> for TextScore<T> {
    fn probability(&self) -> T {
        self.probability
    }
    
    fn confidence(&self) -> T {
        self.confidence
    }
    
    fn modality(&self) -> ModalityType {
        ModalityType::Text
    }
    
    fn features(&self) -> Vec<Feature<T>> {
        self.feature_contributions.clone()
    }
    
    fn timestamp(&self) -> SystemTime {
        self.timestamp
    }
}

/// Main text analyzer that implements the ModalityAnalyzer trait
pub struct TextAnalyzer<T: Float> {
    config: TextAnalyzerConfig,
    linguistic_analyzer: LinguisticAnalyzer<T>,
    bert_integration: BertIntegration<T>,
    deception_detector: DeceptionPatternDetector<T>,
    cache: std::sync::RwLock<HashMap<u64, CacheEntry<T>>>,
    performance_tracker: std::sync::RwLock<PerformanceMetrics>,
}

impl<T: Float> TextAnalyzer<T> {
    /// Create a new text analyzer with the given configuration
    pub fn new(config: TextAnalyzerConfig) -> Result<Self> {
        let linguistic_analyzer = LinguisticAnalyzer::new(&config.preprocessing_config)?;
        let bert_integration = BertIntegration::new(&config.bert_config)?;
        let deception_detector = DeceptionPatternDetector::new(&config.feature_weights)?;
        
        Ok(Self {
            config,
            linguistic_analyzer,
            bert_integration,
            deception_detector,
            cache: std::sync::RwLock::new(HashMap::new()),
            performance_tracker: std::sync::RwLock::new(PerformanceMetrics {
                processing_time_ms: 0,
                feature_extraction_time_ms: 0,
                bert_inference_time_ms: 0,
                total_tokens: 0,
                cache_hits: 0,
                cache_misses: 0,
            }),
        })
    }
    
    /// Create a text analyzer with default configuration
    pub fn default() -> Result<Self> {
        Self::new(TextAnalyzerConfig::default())
    }
    
    /// Get performance statistics
    pub fn performance_stats(&self) -> PerformanceMetrics {
        self.performance_tracker.read().unwrap().clone()
    }
    
    /// Clear the analysis cache
    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().unwrap();
        (cache.len(), self.config.max_cache_size)
    }
    
    /// Detect language of the input text
    pub fn detect_language(&self, text: &str) -> Result<Language> {
        self.linguistic_analyzer.detect_language(text)
    }
    
    /// Preprocess text according to configuration
    pub fn preprocess_text(&self, text: &str, language: Language) -> Result<String> {
        self.linguistic_analyzer.preprocess(text, language, &self.config.preprocessing_config)
    }
    
    /// Extract linguistic features from text
    pub async fn extract_features(&self, text: &str, language: Language) -> Result<LinguisticFeatures<T>> {
        self.linguistic_analyzer.extract_features(text, language).await
    }
    
    /// Generate BERT embeddings
    pub async fn generate_embeddings(&self, text: &str) -> Result<BertEmbedding<T>> {
        self.bert_integration.encode(text).await
    }
    
    /// Detect deception patterns
    pub fn detect_patterns(&self, features: &LinguisticFeatures<T>) -> Result<DeceptionPatterns<T>> {
        self.deception_detector.analyze_patterns(features)
    }
    
    /// Generate explanation for the analysis
    fn generate_explanation(&self, score: &TextScore<T>) -> ExplanationTrace {
        let mut steps = Vec::new();
        
        // Language detection step
        steps.push(ExplanationStep {
            step_type: "language_detection".to_string(),
            description: format!("Detected language: {:?}", score.language),
            evidence: vec![format!("Language confidence: {:.2}", 0.95)], // TODO: actual confidence
            confidence: 0.95,
        });
        
        // Feature extraction step
        steps.push(ExplanationStep {
            step_type: "feature_extraction".to_string(),
            description: format!("Extracted {} linguistic features", score.linguistic_features.feature_count()),
            evidence: vec![
                format!("Lexical features: {}", score.linguistic_features.lexical_features.len()),
                format!("Syntactic features: {}", score.linguistic_features.syntactic_features.len()),
                format!("Semantic features: {}", score.linguistic_features.semantic_features.len()),
            ],
            confidence: score.confidence.to_f64().unwrap_or(0.0),
        });
        
        // Deception analysis step
        let deception_confidence = if score.probability > T::from(0.7).unwrap() {
            0.9
        } else if score.probability > T::from(0.3).unwrap() {
            0.7
        } else {
            0.5
        };
        
        steps.push(ExplanationStep {
            step_type: "deception_analysis".to_string(),
            description: "Analyzed text for deception indicators".to_string(),
            evidence: vec![
                format!("Hedging frequency: {:.3}", score.deception_patterns.hedging_frequency.to_f64().unwrap_or(0.0)),
                format!("Temporal inconsistency: {:.3}", score.deception_patterns.temporal_references.to_f64().unwrap_or(0.0)),
                format!("Certainty markers: {:.3}", score.deception_patterns.certainty_markers.to_f64().unwrap_or(0.0)),
            ],
            confidence: deception_confidence,
        });
        
        ExplanationTrace {
            steps,
            confidence: score.confidence.to_f64().unwrap_or(0.0),
            reasoning: format!(
                "Based on linguistic analysis, this text has a {:.1}% probability of containing deception. \
                Key indicators include linguistic complexity, temporal patterns, and emotional consistency.",
                score.probability.to_f64().unwrap_or(0.0) * 100.0
            ),
        }
    }
}

#[async_trait]
impl<T: Float + Send + Sync + 'static> ModalityAnalyzer<T> for TextAnalyzer<T>
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    type Input = TextInput;
    type Output = TextScore<T>;
    type Config = TextAnalyzerConfig;
    
    async fn analyze(&self, input: &Self::Input) -> Result<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let text_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            input.text.hash(&mut hasher);
            hasher.finish()
        };
        
        if self.config.enable_caching {
            if let Ok(cache) = self.cache.read() {
                if let Some(cached) = cache.get(&text_hash) {
                    // Update cache hit counter
                    if let Ok(mut tracker) = self.performance_tracker.write() {
                        tracker.cache_hits += 1;
                    }
                    
                    // Return cached result (simplified - would need proper conversion)
                    // For now, we'll continue with analysis
                }
            }
        }
        
        // Update cache miss counter
        if let Ok(mut tracker) = self.performance_tracker.write() {
            tracker.cache_misses += 1;
        }
        
        // Detect language
        let language = if let Some(lang) = input.language {
            lang
        } else {
            self.detect_language(&input.text)?
        };
        
        // Validate language support
        if !language.is_supported() && self.config.bert_config.model_name.contains("uncased") {
            return Err(VeritasError::unsupported_language(format!("{:?}", language)));
        }
        
        // Preprocess text
        let preprocessed_text = self.preprocess_text(&input.text, language)?;
        
        // Extract linguistic features
        let feature_start = std::time::Instant::now();
        let linguistic_features = self.extract_features(&preprocessed_text, language).await?;
        let feature_time = feature_start.elapsed().as_millis() as u64;
        
        // Generate BERT embeddings
        let bert_start = std::time::Instant::now();
        let bert_embeddings = if self.config.bert_config.model_name.is_empty() {
            None
        } else {
            Some(self.generate_embeddings(&preprocessed_text).await?)
        };
        let bert_time = bert_start.elapsed().as_millis() as u64;
        
        // Detect deception patterns
        let deception_patterns = self.detect_patterns(&linguistic_features)?;
        
        // Calculate overall deception probability
        let probability = self.deception_detector.calculate_probability(&linguistic_features, &deception_patterns)?;
        
        // Calculate confidence based on feature consistency
        let confidence = self.deception_detector.calculate_confidence(&linguistic_features, &deception_patterns)?;
        
        // Generate additional analysis
        let sentiment = if self.config.enable_sentiment {
            Some(self.linguistic_analyzer.analyze_sentiment(&input.text, language)?)
        } else {
            None
        };
        
        let named_entities = if self.config.enable_ner {
            self.linguistic_analyzer.extract_entities(&input.text, language)?
        } else {
            Vec::new()
        };
        
        let tokens = self.linguistic_analyzer.tokenize_and_analyze(&input.text, language)?;
        let complexity = self.linguistic_analyzer.calculate_complexity(&input.text)?;
        let temporal_patterns = self.linguistic_analyzer.analyze_temporal_patterns(&tokens)?;
        let cognitive_load = self.linguistic_analyzer.analyze_cognitive_load(&tokens)?;
        let semantic_coherence = self.linguistic_analyzer.analyze_semantic_coherence(&input.text, language)?;
        
        // Generate feature contributions for explainability
        let feature_contributions = self.deception_detector.get_feature_contributions(&linguistic_features)?;
        
        let total_time = start_time.elapsed().as_millis() as u64;
        
        // Update performance metrics
        if let Ok(mut tracker) = self.performance_tracker.write() {
            tracker.processing_time_ms = total_time;
            tracker.feature_extraction_time_ms = feature_time;
            tracker.bert_inference_time_ms = bert_time;
            tracker.total_tokens = tokens.len();
        }
        
        let performance = PerformanceMetrics {
            processing_time_ms: total_time,
            feature_extraction_time_ms: feature_time,
            bert_inference_time_ms: bert_time,
            total_tokens: tokens.len(),
            cache_hits: 0,
            cache_misses: 1,
        };
        
        let score = TextScore {
            probability,
            confidence,
            language,
            linguistic_features,
            bert_embeddings,
            sentiment,
            deception_patterns,
            complexity,
            named_entities,
            tokens,
            temporal_patterns,
            cognitive_load,
            semantic_coherence,
            performance,
            timestamp: SystemTime::now(),
            feature_contributions,
        };
        
        // Cache the result if enabled
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                if cache.len() >= self.config.max_cache_size {
                    // Remove oldest entry (simplified LRU)
                    if let Some(oldest_key) = cache.keys().next().cloned() {
                        cache.remove(&oldest_key);
                    }
                }
                
                cache.insert(text_hash, CacheEntry {
                    text_hash,
                    features: score.linguistic_features.clone(),
                    timestamp: SystemTime::now(),
                    language,
                });
            }
        }
        
        Ok(score)
    }
    
    fn confidence(&self) -> T {
        T::from(0.85).unwrap() // Base analyzer confidence
    }
    
    fn explain(&self) -> ExplanationTrace {
        ExplanationTrace {
            steps: vec![
                ExplanationStep {
                    step_type: "initialization".to_string(),
                    description: "Text analyzer initialized with configuration".to_string(),
                    evidence: vec![
                        format!("BERT model: {}", self.config.bert_config.model_name),
                        format!("Sentiment analysis: {}", self.config.enable_sentiment),
                        format!("NER enabled: {}", self.config.enable_ner),
                    ],
                    confidence: 1.0,
                }
            ],
            confidence: self.confidence().to_f64().unwrap_or(0.85),
            reasoning: "Text analyzer ready to process linguistic input for deception detection".to_string(),
        }
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
}