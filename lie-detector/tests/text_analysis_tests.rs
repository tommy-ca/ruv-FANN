//! Comprehensive tests for text analysis functionality

use veritas_nexus::prelude::*;
use veritas_nexus::modalities::text::*;
use tokio_test;

#[tokio::test]
async fn test_text_analyzer_creation() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config);
    
    assert!(analyzer.is_ok(), "TextAnalyzer creation should succeed");
    
    let analyzer = analyzer.unwrap();
    assert_eq!(analyzer.confidence(), 0.85);
}

#[tokio::test]
async fn test_basic_text_analysis() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new("I definitely did not take the money from the office yesterday.");
    
    let result = analyzer.analyze(&input).await;
    assert!(result.is_ok(), "Text analysis should succeed");
    
    let score = result.unwrap();
    assert_eq!(score.modality(), ModalityType::Text);
    assert!(score.probability() >= 0.0 && score.probability() <= 1.0);
    assert!(score.confidence() >= 0.0 && score.confidence() <= 1.0);
    assert_eq!(score.language, Language::English);
}

#[tokio::test]
async fn test_deceptive_text_patterns() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    // Text with multiple deception indicators
    let deceptive_input = TextInput::new(
        "Well, I kind of think maybe I might have possibly seen something like that, \
         but I'm not really sure, you know, because it was sort of dark and I wasn't \
         really paying attention, if you know what I mean."
    );
    
    let result = analyzer.analyze(&deceptive_input).await.expect("Analysis should succeed");
    
    // Should detect hedging patterns
    assert!(result.deception_patterns.hedging_frequency > 0.0);
    assert!(result.deception_patterns.uncertainty_markers > 0.0);
    
    // Compare with truthful text
    let truthful_input = TextInput::new(
        "I saw John take the money from the drawer at 3:15 PM yesterday. \
         He was wearing a blue shirt and left through the back door."
    );
    
    let truthful_result = analyzer.analyze(&truthful_input).await.expect("Analysis should succeed");
    
    // Deceptive text should have higher deception probability
    assert!(result.probability() > truthful_result.probability());
}

#[tokio::test]
async fn test_language_detection() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    // English text
    let english_input = TextInput::new("Hello, how are you today?");
    let result = analyzer.analyze(&english_input).await.expect("Analysis should succeed");
    assert_eq!(result.language, Language::English);
    
    // Spanish text
    let spanish_input = TextInput::new("Hola, ¿cómo estás hoy?");
    let result = analyzer.analyze(&spanish_input).await.expect("Analysis should succeed");
    assert_eq!(result.language, Language::Spanish);
}

#[tokio::test]
async fn test_sentiment_analysis() {
    let config = TextAnalyzerConfig {
        enable_sentiment: true,
        ..Default::default()
    };
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    // Positive sentiment
    let positive_input = TextInput::new("I am very happy and excited about this wonderful opportunity!");
    let result = analyzer.analyze(&positive_input).await.expect("Analysis should succeed");
    
    if let Some(sentiment) = &result.sentiment {
        assert_eq!(sentiment.dominant_sentiment(), SentimentLabel::Positive);
        assert!(sentiment.positive > sentiment.negative);
    }
    
    // Negative sentiment
    let negative_input = TextInput::new("I hate this terrible situation and feel awful about everything.");
    let result = analyzer.analyze(&negative_input).await.expect("Analysis should succeed");
    
    if let Some(sentiment) = &result.sentiment {
        assert_eq!(sentiment.dominant_sentiment(), SentimentLabel::Negative);
        assert!(sentiment.negative > sentiment.positive);
    }
}

#[tokio::test]
async fn test_named_entity_recognition() {
    let config = TextAnalyzerConfig {
        enable_ner: true,
        ..Default::default()
    };
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new(
        "John Smith took $500 from Acme Corp on January 15, 2024 at 3:30 PM."
    );
    
    let result = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Should detect various entity types
    let entity_types: std::collections::HashSet<EntityType> = result.named_entities
        .iter()
        .map(|e| e.entity_type)
        .collect();
    
    // Should find money, date, and potentially person/organization
    assert!(entity_types.contains(&EntityType::Money));
    assert!(entity_types.contains(&EntityType::Date));
}

#[tokio::test]
async fn test_complexity_metrics() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    // Simple text
    let simple_input = TextInput::new("I did it. Yes. No doubt.");
    let simple_result = analyzer.analyze(&simple_input).await.expect("Analysis should succeed");
    
    // Complex text
    let complex_input = TextInput::new(
        "The multifaceted implications of the aforementioned circumstances \
         necessitate a comprehensive evaluation of the underlying epistemological \
         frameworks that fundamentally govern our understanding of the \
         phenomenological manifestations inherent in this particular situation."
    );
    let complex_result = analyzer.analyze(&complex_input).await.expect("Analysis should succeed");
    
    // Complex text should have higher complexity metrics
    assert!(complex_result.complexity.flesch_kincaid_grade > simple_result.complexity.flesch_kincaid_grade);
    assert!(complex_result.complexity.average_word_length > simple_result.complexity.average_word_length);
}

#[tokio::test]
async fn test_temporal_analysis() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new(
        "Yesterday I was there, but today I am here. Tomorrow I will be somewhere else. \
         Before that happened, I had already left. After everything ended, I returned."
    );
    
    let result = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Should detect temporal patterns
    assert!(result.temporal_patterns.past_references > 0.0);
    assert!(result.temporal_patterns.present_references > 0.0);
    assert!(result.temporal_patterns.future_references > 0.0);
    assert!(result.temporal_patterns.timeline_clarity > 0.0);
}

#[tokio::test]
async fn test_cognitive_load_indicators() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new(
        "Um, well, you know, I was like, actually, basically thinking that, \
         you know, maybe it was, like, I mean, you know what I mean?"
    );
    
    let result = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Should detect high cognitive load
    assert!(result.cognitive_load.hesitation_markers > 0.0);
    assert!(result.cognitive_load.filler_words > 0.0);
    assert!(result.cognitive_load.processing_effort > 0.0);
}

#[tokio::test]
async fn test_feature_contributions() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new("I definitely never took anything from anywhere at any time.");
    
    let result = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Should have feature contributions for explainability
    assert!(!result.feature_contributions.is_empty());
    
    // Features should have names, values, and weights
    for feature in &result.feature_contributions {
        assert!(!feature.name.is_empty());
        assert!(!feature.description.is_empty());
        assert!(feature.weight >= 0.0);
    }
}

#[tokio::test]
async fn test_performance_metrics() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new("This is a test sentence for performance measurement.");
    
    let result = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Should track performance metrics
    assert!(result.performance.processing_time_ms > 0);
    assert!(result.performance.total_tokens > 0);
    assert_eq!(result.performance.cache_misses, 1); // First analysis should be cache miss
    
    // Get overall performance stats
    let stats = analyzer.performance_stats();
    assert!(stats.processing_time_ms > 0);
}

#[tokio::test]
async fn test_caching_functionality() {
    let config = TextAnalyzerConfig {
        enable_caching: true,
        max_cache_size: 10,
        ..Default::default()
    };
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let input = TextInput::new("This text will be cached.");
    
    // First analysis
    let _result1 = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Second analysis of same text should potentially use cache
    let _result2 = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // Check cache stats
    let (cache_size, max_size) = analyzer.cache_stats();
    assert!(cache_size <= max_size);
    
    // Clear cache
    analyzer.clear_cache();
    let (cache_size_after_clear, _) = analyzer.cache_stats();
    assert_eq!(cache_size_after_clear, 0);
}

#[tokio::test]
async fn test_empty_input_handling() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    let empty_input = TextInput::new("");
    let result = analyzer.analyze(&empty_input).await;
    
    // Should handle empty input gracefully
    assert!(result.is_err());
    
    let whitespace_input = TextInput::new("   \n\t   ");
    let result = analyzer.analyze(&whitespace_input).await;
    
    // Should handle whitespace-only input
    assert!(result.is_err());
}

#[tokio::test]
async fn test_configuration_validation() {
    // Test default configuration
    let default_config = TextAnalyzerConfig::default();
    assert!(default_config.confidence_threshold > 0.0);
    assert!(default_config.timeout_ms > 0);
    assert!(default_config.max_cache_size > 0);
    
    // Test custom configuration
    let custom_config = TextAnalyzerConfig {
        enable_sentiment: false,
        enable_ner: false,
        enable_caching: false,
        confidence_threshold: 0.8,
        ..Default::default()
    };
    
    let analyzer = TextAnalyzer::<f64>::new(custom_config);
    assert!(analyzer.is_ok());
}

#[tokio::test]
async fn test_explanation_trace() {
    let config = TextAnalyzerConfig::default();
    let analyzer = TextAnalyzer::<f64>::new(config).expect("Failed to create analyzer");
    
    // Test analyzer explanation
    let explanation = analyzer.explain();
    assert!(!explanation.steps.is_empty());
    assert!(!explanation.reasoning.is_empty());
    assert!(explanation.confidence > 0.0);
    
    // Test analysis explanation
    let input = TextInput::new("I think maybe I might have possibly seen something.");
    let result = analyzer.analyze(&input).await.expect("Analysis should succeed");
    
    // The result should contain explanation traces through feature contributions
    assert!(!result.feature_contributions.is_empty());
}

#[cfg(test)]
mod linguistic_analyzer_tests {
    use super::*;
    use veritas_nexus::modalities::text::LinguisticAnalyzer;
    
    #[test]
    fn test_language_detection() {
        let config = PreprocessingConfig::default();
        let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
        
        assert_eq!(analyzer.detect_language("Hello world").unwrap(), Language::English);
        assert_eq!(analyzer.detect_language("Hola mundo").unwrap(), Language::Spanish);
        assert_eq!(analyzer.detect_language("Bonjour monde").unwrap(), Language::French);
        assert_eq!(analyzer.detect_language("").unwrap(), Language::Unknown);
    }
    
    #[test]
    fn test_text_preprocessing() {
        let config = PreprocessingConfig {
            normalize_unicode: true,
            lowercase: true,
            remove_punctuation: true,
            ..Default::default()
        };
        let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
        
        let input = "Hello, World! This is a TEST.";
        let result = analyzer.preprocess(input, Language::English, &config).expect("Preprocessing should succeed");
        
        assert_eq!(result, "hello world this is a test");
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let config = PreprocessingConfig::default();
        let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
        
        let text = "This is a simple test sentence for feature extraction.";
        let features = analyzer.extract_features(text, Language::English).await.expect("Feature extraction should succeed");
        
        assert!(features.feature_count() > 0);
        assert!(!features.lexical_features.is_empty());
        assert!(!features.syntactic_features.is_empty());
        assert!(!features.semantic_features.is_empty());
        assert!(!features.feature_names.is_empty());
    }
}

#[cfg(test)]
mod deception_patterns_tests {
    use super::*;
    use veritas_nexus::modalities::text::DeceptionPatternDetector;
    
    #[test]
    fn test_pattern_detector_creation() {
        let weights = FeatureWeights::<f64>::default();
        let detector = DeceptionPatternDetector::new(&weights);
        
        assert!(detector.is_ok());
    }
    
    #[test]
    fn test_probability_calculation() {
        let weights = FeatureWeights::<f64>::default();
        let detector = DeceptionPatternDetector::new(&weights).expect("Failed to create detector");
        
        // Create mock features
        let features = LinguisticFeatures {
            lexical_features: vec![10.0, 50.0, 5.0, 0.8, 2.0, 1.0, 0.5, 1.0, 2.0], // word_count, char_count, avg_word_len, ttr, uncertainty, hedging, certainty, negation, self_ref
            syntactic_features: vec![3.0, 3.3, 0.3, 0.2, 0.1, 0.05, 0.15], // sent_count, avg_sent_len, noun_ratio, verb_ratio, adj_ratio, adv_ratio, pronoun_ratio
            semantic_features: vec![0.7, 0.1, 0.2], // semantic_density, avg_sentiment, sentiment_variance
            pragmatic_features: vec![0.1, 1.0, 0.0], // uncertainty_ratio, question_count, exclamation_count
            discourse_features: vec![2.0, 3.0], // connective_count, temporal_markers
            feature_names: vec!["test".to_string()],
        };
        
        let patterns = detector.analyze_patterns(&features).expect("Pattern analysis should succeed");
        let probability = detector.calculate_probability(&features, &patterns).expect("Probability calculation should succeed");
        
        assert!(probability >= 0.0 && probability <= 1.0);
        
        let confidence = detector.calculate_confidence(&features, &patterns).expect("Confidence calculation should succeed");
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}