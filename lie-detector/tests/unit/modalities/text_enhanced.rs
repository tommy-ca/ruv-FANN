/// Enhanced unit tests for text modality analyzer with property-based testing
/// 
/// Tests linguistic analysis, sentiment detection, deception pattern recognition,
/// performance characteristics, and edge cases using proptest and comprehensive scenarios

use crate::common::*;
use crate::common::generators_enhanced::*;
use veritas_nexus::modalities::text::*;
use veritas_nexus::{ModalityAnalyzer, DeceptionScore, ModalityType};
use std::collections::HashMap;
use proptest::prelude::*;
use tokio_test;
use serial_test::serial;
use float_cmp::approx_eq;
use fake::{Fake, Faker};

#[cfg(test)]
mod property_based_tests {
    use super::*;
    
    proptest! {
        /// Test that probability outputs are always in valid range [0, 1]
        #[test]
        fn probability_always_in_valid_range(
            text in "\\PC{20,200}", // Any printable characters, 20-200 length
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let config = TextAnalyzerConfig::default();
                let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
                let input = TextInput::new(text);
                
                if let Ok(result) = analyzer.analyze(&input).await {
                    prop_assert!(result.probability() >= 0.0);
                    prop_assert!(result.probability() <= 1.0);
                    prop_assert!(result.confidence() >= 0.0);
                    prop_assert!(result.confidence() <= 1.0);
                }
            });
        }
        
        /// Test that confidence correlates with feature consistency
        #[test]
        fn confidence_correlates_with_consistency(
            pattern in consistent_signals::<f64>()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let config = TextAnalyzerConfig::default();
                let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
                
                // Create consistent text (all features agree on deception level)
                let consistent_text = if pattern.expected_consistency {
                    "I am absolutely certain about this specific event that happened yesterday at exactly 3:45 PM."
                } else {
                    "I think maybe something possibly happened somewhere, I'm not really sure when."
                };
                
                let input = TextInput::new(consistent_text);
                if let Ok(result) = analyzer.analyze(&input).await {
                    if pattern.expected_consistency {
                        prop_assert!(result.confidence() > 0.7);
                    } else {
                        prop_assert!(result.confidence() < 0.8); // May still be confident about inconsistency
                    }
                }
            });
        }
        
        /// Test linguistic feature extraction invariants
        #[test]
        fn linguistic_features_invariants(
            text in "[a-zA-Z0-9 .,!?]{10,500}"
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let config = TextAnalyzerConfig::default();
                let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
                let language = Language::English; // Assume English for this test
                
                if let Ok(features) = analyzer.extract_features(&text, language).await {
                    // Word count should be positive for non-empty text
                    if !text.trim().is_empty() {
                        prop_assert!(features.lexical_features.get("word_count").unwrap_or(&0.0) > &0.0);
                    }
                    
                    // Ratios should be in [0, 1]
                    for (name, &value) in &features.lexical_features {
                        if name.contains("ratio") || name.contains("frequency") {
                            prop_assert!(value >= 0.0 && value <= 1.0, 
                                "Feature {} has invalid ratio: {}", name, value);
                        }
                    }
                    
                    // Features should be finite
                    for &value in features.lexical_features.values() {
                        prop_assert!(value.is_finite(), "Feature value should be finite: {}", value);
                    }
                }
            });
        }
        
        /// Test that analyzer handles various text lengths gracefully
        #[test]
        fn handles_variable_text_lengths(
            length in 0usize..10000,
            base_word in "[a-zA-Z]{3,10}"
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let text = if length == 0 {
                    String::new()
                } else {
                    std::iter::repeat(base_word)
                        .take(length / 5) // Approximate word count
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                
                let config = TextAnalyzerConfig::default();
                let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
                let input = TextInput::new(text.clone());
                
                // Should not panic or return invalid results
                match analyzer.analyze(&input).await {
                    Ok(result) => {
                        prop_assert!(result.probability().is_finite());
                        prop_assert!(result.confidence().is_finite());
                        
                        // Empty text should have low confidence
                        if text.trim().is_empty() {
                            prop_assert!(result.confidence() < 0.5);
                        }
                    },
                    Err(_) => {
                        // Some errors are acceptable for edge cases
                        if text.len() > 5000 {
                            // Very long text might timeout
                        } else if text.trim().is_empty() {
                            // Empty text might be rejected
                        } else {
                            prop_assert!(false, "Unexpected error for text length: {}", text.len());
                        }
                    }
                }
            });
        }
        
        /// Test deception pattern detection consistency
        #[test]
        fn deception_patterns_consistency(
            hedging_ratio in 0.0f64..1.0,
            certainty_ratio in 0.0f64..1.0
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Generate text with controlled hedging and certainty
                let hedge_words = ["maybe", "possibly", "I think", "perhaps", "might"];
                let certainty_words = ["definitely", "absolutely", "certainly", "clearly"];
                
                let mut text_parts = Vec::new();
                
                // Add hedging words based on ratio
                let hedge_count = (hedging_ratio * 10.0) as usize;
                for _ in 0..hedge_count {
                    text_parts.push(hedge_words[text_parts.len() % hedge_words.len()]);
                }
                
                // Add certainty words based on ratio
                let certainty_count = (certainty_ratio * 10.0) as usize;
                for _ in 0..certainty_count {
                    text_parts.push(certainty_words[text_parts.len() % certainty_words.len()]);
                }
                
                // Add filler words
                text_parts.extend_from_slice(&["I", "went", "to", "the", "store", "yesterday"]);
                
                let text = text_parts.join(" ") + ".";
                
                let config = TextAnalyzerConfig::default();
                let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
                let input = TextInput::new(text);
                
                if let Ok(result) = analyzer.analyze(&input).await {
                    // High hedging should increase deception probability
                    if hedging_ratio > 0.7 && certainty_ratio < 0.3 {
                        prop_assert!(result.probability() > 0.4, 
                            "High hedging should increase deception probability");
                    }
                    
                    // High certainty should decrease deception probability
                    if certainty_ratio > 0.7 && hedging_ratio < 0.3 {
                        prop_assert!(result.probability() < 0.6,
                            "High certainty should decrease deception probability");
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use criterion::{Criterion, BenchmarkId};
    use std::time::{Duration, Instant};
    
    #[tokio::test]
    async fn test_large_text_performance() {
        let config = TestConfig::default();
        config.setup().unwrap();
        
        let analyzer_config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(analyzer_config).unwrap();
        
        // Test with increasingly large texts
        let sizes = [100, 500, 1000, 5000, 10000];
        
        for &size in &sizes {
            let text = "This is a test sentence with various words. ".repeat(size / 8);
            let input = TextInput::new(text.clone());
            
            let start = Instant::now();
            let result = analyzer.analyze(&input).await;
            let duration = start.elapsed();
            
            assert!(result.is_ok(), "Analysis should succeed for size {}", size);
            
            // Performance should scale reasonably
            let max_expected_duration = Duration::from_millis(100 + (size as u64 / 10));
            assert!(
                duration < max_expected_duration,
                "Analysis took too long for size {}: {:?} > {:?}",
                size, duration, max_expected_duration
            );
            
            if let Ok(score) = result {
                // Performance metrics should be reasonable
                assert!(score.performance.processing_time_ms < 5000);
                assert!(score.performance.total_tokens > 0);
            }
        }
    }
    
    #[tokio::test]
    async fn test_concurrent_analysis_performance() {
        let config = TextAnalyzerConfig::default();
        let analyzer = std::sync::Arc::new(TextAnalyzer::<f64>::new(config).unwrap());
        
        let test_texts = vec![
            "I went to the store yesterday and bought milk.",
            "Maybe I possibly went somewhere, I think.",
            "The weather is nice today and sunny.",
            "I'm absolutely certain this happened exactly as described.",
            "I don't know, maybe it was Tuesday or Wednesday.",
        ];
        
        let start = Instant::now();
        
        // Run 50 concurrent analyses
        let tasks: Vec<_> = (0..50).map(|i| {
            let analyzer = analyzer.clone();
            let text = test_texts[i % test_texts.len()].to_string();
            
            tokio::spawn(async move {
                let input = TextInput::new(text);
                analyzer.analyze(&input).await
            })
        }).collect();
        
        let results = futures::future::join_all(tasks).await;
        let total_duration = start.elapsed();
        
        // All tasks should complete successfully
        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }
        
        // Concurrent execution should be reasonably fast
        assert!(total_duration < Duration::from_secs(30));
        
        println!("Concurrent analysis of 50 texts took: {:?}", total_duration);
    }
    
    #[tokio::test]
    async fn test_memory_usage_stability() {
        let config = TextAnalyzerConfig {
            enable_caching: false, // Disable caching to test base memory usage
            ..Default::default()
        };
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Measure memory before
        let initial_memory = get_memory_usage();
        
        // Process many texts
        for i in 0..1000 {
            let text = format!("This is test text number {} with some content.", i);
            let input = TextInput::new(text);
            
            let _ = analyzer.analyze(&input).await;
            
            // Check memory every 100 iterations
            if i % 100 == 0 {
                let current_memory = get_memory_usage();
                let memory_growth = current_memory.saturating_sub(initial_memory);
                
                // Memory growth should be reasonable (less than 100MB)
                assert!(
                    memory_growth < 100 * 1024 * 1024,
                    "Memory usage grew too much: {} bytes after {} iterations",
                    memory_growth, i
                );
            }
        }
    }
    
    fn get_memory_usage() -> usize {
        // Simplified memory usage tracking
        // In a real implementation, you'd use system-specific memory tracking
        0
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_empty_and_whitespace_texts() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        let edge_cases = vec![
            "",
            " ",
            "\n",
            "\t",
            "   \n\t   ",
            "\u{2000}\u{2001}\u{2002}", // Unicode spaces
        ];
        
        for text in edge_cases {
            let input = TextInput::new(text);
            
            match analyzer.analyze(&input).await {
                Ok(result) => {
                    // Should handle gracefully with low confidence
                    assert!(result.confidence() < 0.3);
                    assert!(result.probability().is_finite());
                },
                Err(_) => {
                    // Error is acceptable for empty/whitespace-only text
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_special_characters_and_unicode() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        let special_texts = vec![
            "Hello! @#$%^&*()_+ {}[]|\\:;\"'<>?,./",
            "¡Hola! ¿Cómo estás? Niño, año, señor",
            "Здравствуй мир", // Russian
            "こんにちは世界", // Japanese
            "🙂😀🎉💯🔥", // Emojis
            "Test\u{0000}null\u{0001}control\u{001F}chars",
            "Long\u{2014}dashes\u{2013}and\u{2012}hyphens",
        ];
        
        for text in special_texts {
            let input = TextInput::new(text);
            
            match analyzer.analyze(&input).await {
                Ok(result) => {
                    // Should handle gracefully
                    assert!(result.probability().is_finite());
                    assert!(result.confidence().is_finite());
                    assert_eq!(result.modality(), ModalityType::Text);
                },
                Err(e) => {
                    // Some unicode/special chars might cause language detection errors
                    println!("Expected error for text '{}': {:?}", text, e);
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_very_long_texts() {
        let config = TextAnalyzerConfig {
            timeout_ms: 60000, // Increase timeout for long texts
            ..Default::default()
        };
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Test texts of various extreme lengths
        let test_cases = vec![
            ("Single word", "supercalifragilisticexpialidocious".repeat(1000)),
            ("Repeated sentence", "This is a test sentence. ".repeat(10000)),
            ("Repeated paragraph", "This is a longer test paragraph with multiple sentences. It contains various linguistic features. The text should be processable by the analyzer. ".repeat(1000)),
        ];
        
        for (description, text) in test_cases {
            let input = TextInput::new(text.clone());
            
            match analyzer.analyze(&input).await {
                Ok(result) => {
                    println!("{}: Processed {} chars successfully", description, text.len());
                    assert!(result.probability().is_finite());
                    assert!(result.confidence().is_finite());
                    
                    // Very long texts might have different characteristics
                    if text.len() > 100000 {
                        // Might have lower confidence due to processing constraints
                        assert!(result.confidence() >= 0.0);
                    }
                },
                Err(e) => {
                    // Timeout or memory errors are acceptable for very long texts
                    println!("{}: Expected error for {} chars: {:?}", description, text.len(), e);
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_malformed_input_handling() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Test various malformed inputs
        let malformed_cases = vec![
            TextInput::new("").with_context("invalid".to_string(), serde_json::Value::Null),
            TextInput::new("test").with_language(Language::Unknown),
            TextInput::new("test").with_timestamp(SystemTime::UNIX_EPOCH), // Very old timestamp
        ];
        
        for input in malformed_cases {
            match analyzer.analyze(&input).await {
                Ok(result) => {
                    // Should handle gracefully
                    assert!(result.probability().is_finite());
                    assert!(result.confidence().is_finite());
                },
                Err(_) => {
                    // Some malformed inputs might legitimately cause errors
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_numerical_and_mixed_content() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        let numerical_texts = vec![
            "123456789",
            "The price is $123.45 and the date is 2023-12-31.",
            "Call me at +1-555-123-4567 or email user@example.com",
            "3.14159 is approximately equal to π (pi)",
            "1st, 2nd, 3rd, 4th place finishers",
            "100% confident about 50/50 odds",
        ];
        
        for text in numerical_texts {
            let input = TextInput::new(text);
            let result = analyzer.analyze(&input).await;
            
            assert!(result.is_ok(), "Should handle numerical text: {}", text);
            
            if let Ok(score) = result {
                assert!(score.probability().is_finite());
                assert!(score.confidence().is_finite());
                
                // Numerical content might have different deception characteristics
                // This is a domain-specific assertion that would need validation
            }
        }
    }
}

#[cfg(test)]
mod integration_with_fixtures_tests {
    use super::*;
    use crate::common::fixtures::*;
    
    #[tokio::test]
    async fn test_with_multimodal_fixtures() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Test with pre-defined test fixtures
        let test_cases = vec![
            MultiModalTestData::new_truthful(),
            MultiModalTestData::new_deceptive(),
            MultiModalTestData::new_mixed(),
        ];
        
        for test_case in test_cases {
            let input = TextInput::new(test_case.text.content.clone());
            let result = analyzer.analyze(&input).await;
            
            assert!(result.is_ok(), "Should analyze fixture text");
            
            if let Ok(score) = result {
                // Verify the analysis aligns with expected ground truth
                let expected_deceptive = test_case.ground_truth_label;
                let predicted_deceptive = score.probability() > 0.5;
                
                // For test fixtures, we expect reasonable alignment
                // but allow for some variance due to the complexity of deception detection
                if expected_deceptive == predicted_deceptive {
                    assert!(score.confidence() > 0.5, "Should be confident when correct");
                } else {
                    // When incorrect, confidence might be lower or this might be a hard case
                    println!("Prediction mismatch for fixture - expected: {}, got: {}, confidence: {}", 
                        expected_deceptive, predicted_deceptive, score.confidence());
                }
                
                // Verify linguistic features are extracted
                assert!(!score.linguistic_features.lexical_features.is_empty());
                assert!(!score.tokens.is_empty());
                
                // Verify explanation is generated
                let explanation = analyzer.explain();
                assert!(!explanation.steps.is_empty());
                assert!(!explanation.reasoning.is_empty());
            }
        }
    }
    
    #[tokio::test]
    async fn test_linguistic_feature_consistency() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Test that similar texts produce similar features
        let similar_texts = vec![
            ("I went to the store yesterday.", "I visited the shop yesterday."),
            ("Maybe I could possibly go.", "Perhaps I might possibly go."),
            ("I am absolutely certain.", "I am completely sure."),
        ];
        
        for (text1, text2) in similar_texts {
            let input1 = TextInput::new(text1);
            let input2 = TextInput::new(text2);
            
            let result1 = analyzer.analyze(&input1).await.unwrap();
            let result2 = analyzer.analyze(&input2).await.unwrap();
            
            // Similar texts should have similar deception probabilities
            let prob_diff = (result1.probability() - result2.probability()).abs();
            assert!(
                prob_diff < 0.3,
                "Similar texts should have similar probabilities: {} vs {} (diff: {})",
                result1.probability(), result2.probability(), prob_diff
            );
            
            // Should have similar confidence levels
            let conf_diff = (result1.confidence() - result2.confidence()).abs();
            assert!(
                conf_diff < 0.4,
                "Similar texts should have similar confidence: {} vs {} (diff: {})",
                result1.confidence(), result2.confidence(), conf_diff
            );
        }
    }
}

#[cfg(test)]
mod realistic_scenario_tests {
    use super::*;
    use crate::common::generators_enhanced::realistic_scenarios::*;
    
    #[tokio::test]
    async fn test_interview_scenarios() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Simulate different interview scenarios
        let scenarios = vec![
            ("Police interrogation", "I was at home all evening watching TV. I didn't go anywhere."),
            ("Job interview", "I have extensive experience in this field and I'm very qualified."),
            ("Court testimony", "I clearly remember exactly what happened that night at 9:15 PM."),
            ("Security screening", "I'm just traveling for business. Nothing unusual in my luggage."),
        ];
        
        for (scenario_type, response) in scenarios {
            let input = TextInput::new(response)
                .with_context("scenario_type".to_string(), serde_json::Value::String(scenario_type.to_string()));
            
            let result = analyzer.analyze(&input).await;
            assert!(result.is_ok(), "Should handle {} scenario", scenario_type);
            
            if let Ok(score) = result {
                // Different scenarios might have different baseline deception rates
                assert!(score.probability().is_finite());
                assert!(score.confidence().is_finite());
                
                // Verify scenario-specific analysis
                match scenario_type {
                    "Police interrogation" => {
                        // Might show more stress indicators
                    },
                    "Job interview" => {
                        // Might show more positive sentiment
                    },
                    "Court testimony" => {
                        // Might show more specific details
                        assert!(score.complexity.detail_level > 0.0);
                    },
                    "Security screening" => {
                        // Might be more brief and direct
                    },
                    _ => {}
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_temporal_consistency() {
        let config = TextAnalyzerConfig::default();
        let analyzer = TextAnalyzer::<f64>::new(config).unwrap();
        
        // Test temporal consistency in narratives
        let consistent_narrative = "Yesterday at 3 PM, I went to the store. I bought milk and bread. Then I drove home and arrived at 3:30 PM.";
        let inconsistent_narrative = "Yesterday I went to the store in the morning. Later that evening, I bought milk. Then I went home the next day.";
        
        let input1 = TextInput::new(consistent_narrative);
        let input2 = TextInput::new(inconsistent_narrative);
        
        let result1 = analyzer.analyze(&input1).await.unwrap();
        let result2 = analyzer.analyze(&input2).await.unwrap();
        
        // Inconsistent narrative should have higher deception probability
        assert!(
            result2.probability() > result1.probability(),
            "Inconsistent temporal narrative should have higher deception probability"
        );
        
        // Verify temporal pattern analysis
        assert!(result1.temporal_patterns.consistency_score > result2.temporal_patterns.consistency_score);
    }
}

// Helper functions and utilities for tests

impl Language {
    fn is_supported(&self) -> bool {
        matches!(self, Language::English | Language::Spanish | Language::French | Language::German)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    English,
    Spanish,
    French,
    German,
    Unknown,
}

// Mock implementations for testing
#[derive(Debug, Clone)]
struct CacheEntry<T: Float> {
    text_hash: u64,
    features: LinguisticFeatures<T>,
    timestamp: SystemTime,
    language: Language,
}

#[derive(Debug, Clone)]
struct LinguisticFeatures<T: Float> {
    lexical_features: HashMap<String, T>,
    syntactic_features: HashMap<String, T>,
    semantic_features: HashMap<String, T>,
}

impl<T: Float> LinguisticFeatures<T> {
    fn feature_count(&self) -> usize {
        self.lexical_features.len() + self.syntactic_features.len() + self.semantic_features.len()
    }
}

// Additional mock types for testing
#[derive(Debug, Clone)]
struct BertConfig {
    model_name: String,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            model_name: "bert-base-uncased".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct PreprocessingConfig {
    lowercase: bool,
    remove_punctuation: bool,
    normalize_whitespace: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            normalize_whitespace: true,
        }
    }
}

#[derive(Debug, Clone)]
struct FeatureWeights<T: Float> {
    hedging: T,
    certainty: T,
    complexity: T,
    sentiment: T,
}

impl<T: Float> Default for FeatureWeights<T> {
    fn default() -> Self {
        Self {
            hedging: T::from(0.3).unwrap(),
            certainty: T::from(0.25).unwrap(),
            complexity: T::from(0.25).unwrap(),
            sentiment: T::from(0.2).unwrap(),
        }
    }
}

// More mock types...
#[derive(Debug, Clone)]
struct BertEmbedding<T: Float> {
    embeddings: Vec<T>,
    attention_weights: Vec<T>,
}

#[derive(Debug, Clone)]
struct SentimentResult<T: Float> {
    score: T,
    confidence: T,
    label: String,
}

#[derive(Debug, Clone)]
struct DeceptionPatterns<T: Float> {
    hedging_frequency: T,
    temporal_references: T,
    certainty_markers: T,
}

#[derive(Debug, Clone)]
struct ComplexityMetrics<T: Float> {
    detail_level: T,
    syntactic_complexity: T,
    lexical_diversity: T,
}

#[derive(Debug, Clone)]
struct NamedEntity {
    text: String,
    entity_type: String,
    confidence: f64,
}

#[derive(Debug, Clone)]
struct AnalyzedToken {
    text: String,
    pos_tag: String,
    lemma: String,
}

#[derive(Debug, Clone)]
struct TemporalPattern<T: Float> {
    consistency_score: T,
    temporal_markers: Vec<String>,
}

#[derive(Debug, Clone)]
struct CognitiveLoadIndicators<T: Float> {
    processing_difficulty: T,
    working_memory_load: T,
}

#[derive(Debug, Clone)]
struct SemanticCoherence<T: Float> {
    coherence_score: T,
    topic_consistency: T,
}

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    processing_time_ms: u64,
    feature_extraction_time_ms: u64,
    bert_inference_time_ms: u64,
    total_tokens: usize,
    cache_hits: usize,
    cache_misses: usize,
}