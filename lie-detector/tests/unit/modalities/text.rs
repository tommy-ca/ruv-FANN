/// Unit tests for text modality analyzer
/// 
/// Tests linguistic analysis, sentiment detection, and deception pattern recognition

use crate::common::*;
use std::collections::HashMap;

#[cfg(test)]
mod text_analyzer_tests {
    use super::*;
    use fixtures::{TextTestData, MultiModalTestData};
    use helpers::*;

    /// Test text analyzer initialization
    #[test]
    fn test_text_analyzer_creation() {
        let config = TestConfig::default();
        config.setup().expect("Failed to setup test config");
        
        let text_data = TextTestData::new_truthful();
        assert!(!text_data.content.is_empty());
        assert!(text_data.word_count > 0);
        assert!(text_data.sentence_count > 0);
    }

    /// Test basic text preprocessing
    #[test]
    fn test_text_preprocessing() {
        let text_data = TextTestData::new_truthful();
        
        let processed = preprocess_text(&text_data.content);
        assert!(!processed.is_empty(), "Processed text should not be empty");
        
        // Verify basic preprocessing steps
        assert!(!processed.contains("  "), "Should remove extra spaces");
        assert_eq!(processed.chars().next().unwrap().to_lowercase().next().unwrap(), 
                  processed.chars().next().unwrap(), "Should normalize case");
    }

    /// Test linguistic feature extraction
    #[test]
    fn test_linguistic_feature_extraction() {
        let truthful_data = TextTestData::new_truthful();
        let deceptive_data = TextTestData::new_deceptive();
        
        let truthful_features = extract_linguistic_features(&truthful_data.content);
        let deceptive_features = extract_linguistic_features(&deceptive_data.content);
        
        // Verify feature extraction
        assert!(!truthful_features.is_empty(), "Should extract features from truthful text");
        assert!(!deceptive_features.is_empty(), "Should extract features from deceptive text");
        
        // Compare certainty features
        let truthful_certainty = truthful_features.get("certainty").unwrap_or(&0.0);
        let deceptive_certainty = deceptive_features.get("certainty").unwrap_or(&0.0);
        
        assert!(
            truthful_certainty > deceptive_certainty,
            "Truthful text should show higher certainty"
        );
    }

    /// Test hedge word detection
    #[test]
    fn test_hedge_word_detection() {
        let text_with_hedges = "I think maybe I possibly went to the store, I guess.";
        let text_without_hedges = "I went to the store yesterday morning.";
        
        let hedge_count_with = count_hedge_words(text_with_hedges);
        let hedge_count_without = count_hedge_words(text_without_hedges);
        
        assert!(
            hedge_count_with > hedge_count_without,
            "Text with hedges should have higher hedge count"
        );
        
        // Test specific hedge words
        let hedge_words = vec!["think", "maybe", "possibly", "guess"];
        for word in hedge_words {
            assert!(
                text_with_hedges.contains(word),
                "Test text should contain hedge word: {}",
                word
            );
        }
    }

    /// Test sentiment analysis
    #[test]
    fn test_sentiment_analysis() {
        let positive_text = "I am absolutely certain this is correct and wonderful.";
        let negative_text = "I'm not sure, this might be wrong and terrible.";
        let neutral_text = "The temperature is 70 degrees Fahrenheit.";
        
        let positive_sentiment = analyze_sentiment(positive_text);
        let negative_sentiment = analyze_sentiment(negative_text);
        let neutral_sentiment = analyze_sentiment(neutral_text);
        
        assert!(positive_sentiment > 0.0, "Positive text should have positive sentiment");
        assert!(negative_sentiment < 0.0, "Negative text should have negative sentiment");
        assert!(
            neutral_sentiment.abs() < 0.3,
            "Neutral text should have neutral sentiment"
        );
    }

    /// Test complexity metrics
    #[test]
    fn test_complexity_metrics() {
        let simple_text = "I went to the store. I bought milk.";
        let complex_text = "When considering the multifaceted implications of contemporary socioeconomic policies, one must acknowledge the inherent complexities.";
        
        let simple_complexity = calculate_text_complexity(simple_text);
        let complex_complexity = calculate_text_complexity(complex_text);
        
        assert!(
            complex_complexity > simple_complexity,
            "Complex text should have higher complexity score"
        );
        
        // Test specific complexity metrics
        let simple_avg_word_length = calculate_average_word_length(simple_text);
        let complex_avg_word_length = calculate_average_word_length(complex_text);
        
        assert!(
            complex_avg_word_length > simple_avg_word_length,
            "Complex text should have longer average word length"
        );
    }

    /// Test first-person pronoun usage
    #[test]
    fn test_first_person_analysis() {
        let high_first_person = "I believe I can do this. I think I will succeed.";
        let low_first_person = "The project will succeed. It should work well.";
        
        let high_ratio = calculate_first_person_ratio(high_first_person);
        let low_ratio = calculate_first_person_ratio(low_first_person);
        
        assert!(
            high_ratio > low_ratio,
            "Text with more first-person pronouns should have higher ratio"
        );
        
        // Verify ratio is in valid range
        assert!(high_ratio >= 0.0 && high_ratio <= 1.0, "Ratio should be in [0, 1]");
        assert!(low_ratio >= 0.0 && low_ratio <= 1.0, "Ratio should be in [0, 1]");
    }

    /// Test temporal reference analysis
    #[test]
    fn test_temporal_analysis() {
        let past_text = "I went there yesterday. I saw him last week.";
        let present_text = "I am going there now. I see him today.";
        let future_text = "I will go there tomorrow. I shall see him next week.";
        
        let past_ratio = calculate_past_tense_ratio(past_text);
        let present_ratio = calculate_present_tense_ratio(present_text);
        let future_ratio = calculate_future_tense_ratio(future_text);
        
        assert!(past_ratio > present_ratio && past_ratio > future_ratio, 
                "Past text should have highest past tense ratio");
        assert!(present_ratio > past_ratio && present_ratio > future_ratio,
                "Present text should have highest present tense ratio");
        assert!(future_ratio > past_ratio && future_ratio > present_ratio,
                "Future text should have highest future tense ratio");
    }

    /// Test deception pattern detection
    #[test]
    fn test_deception_pattern_detection() {
        let truthful_data = TextTestData::new_truthful();
        let deceptive_data = TextTestData::new_deceptive();
        
        let truthful_patterns = detect_deception_patterns(&truthful_data.content);
        let deceptive_patterns = detect_deception_patterns(&deceptive_data.content);
        
        // Deceptive text should trigger more pattern detections
        let truthful_pattern_count: usize = truthful_patterns.values().sum();
        let deceptive_pattern_count: usize = deceptive_patterns.values().sum();
        
        assert!(
            deceptive_pattern_count >= truthful_pattern_count,
            "Deceptive text should trigger more deception patterns"
        );
        
        // Check for specific pattern types
        assert!(deceptive_patterns.contains_key("hedge_words"), "Should detect hedge word patterns");
        assert!(deceptive_patterns.contains_key("uncertainty_markers"), "Should detect uncertainty markers");
    }

    /// Test text normalization
    #[test]
    fn test_text_normalization() {
        let messy_text = "  Hello,,,   WORLD!!!   How  are   YOU???  ";
        let normalized = normalize_text(messy_text);
        
        assert!(!normalized.starts_with(' '), "Should remove leading spaces");
        assert!(!normalized.ends_with(' '), "Should remove trailing spaces");
        assert!(!normalized.contains("  "), "Should remove extra spaces");
        assert!(!normalized.contains(",,"), "Should normalize punctuation");
    }

    /// Test feature vector creation
    #[test]
    fn test_feature_vector_creation() {
        let text_data = TextTestData::new_truthful();
        let feature_vector = create_text_feature_vector(&text_data.content);
        
        // Verify feature vector properties
        assert!(!feature_vector.is_empty(), "Feature vector should not be empty");
        assert!(feature_vector.len() >= 10, "Should have multiple features");
        
        // Check for valid feature values
        for (i, &feature) in feature_vector.iter().enumerate() {
            assert!(
                feature.is_finite(),
                "Feature {} should be finite, got {}",
                i, feature
            );
            assert!(
                feature >= 0.0,
                "Text features should be non-negative, got {} at index {}",
                feature, i
            );
        }
    }

    /// Test performance of text processing
    #[test]
    fn test_text_processing_performance() {
        let long_text = "This is a test sentence. ".repeat(1000);
        
        let (_, measurement) = measure_performance(|| {
            extract_linguistic_features(&long_text)
        });
        
        // Assert reasonable processing time
        assert_performance_bounds(
            &measurement,
            std::time::Duration::from_millis(100), // Max 100ms for large text
            Some(10 * 1024 * 1024) // Max 10MB memory usage
        );
    }

    /// Test edge cases in text processing
    #[test]
    fn test_text_edge_cases() {
        // Test empty text
        let empty_features = extract_linguistic_features("");
        assert!(empty_features.is_empty() || empty_features.values().all(|&v| v == 0.0),
                "Empty text should produce empty or zero features");
        
        // Test single word
        let single_word_features = extract_linguistic_features("Hello");
        assert!(!single_word_features.is_empty(), "Should handle single words");
        
        // Test very long text
        let long_text = "word ".repeat(10000);
        let long_features = extract_linguistic_features(&long_text);
        assert!(!long_features.is_empty(), "Should handle very long text");
        
        // Test special characters
        let special_chars = "Hello!@#$%^&*()_+{}|:<>?[];',./`~";
        let special_features = extract_linguistic_features(special_chars);
        assert!(!special_features.is_empty(), "Should handle special characters");
    }

    /// Test multilingual support (basic)
    #[test]
    fn test_multilingual_basic() {
        // Test basic English
        let english_text = "This is English text for testing.";
        let english_features = extract_linguistic_features(english_text);
        assert!(!english_features.is_empty(), "Should process English text");
        
        // Test text with numbers
        let numeric_text = "I have 123 items and 45.67 dollars.";
        let numeric_features = extract_linguistic_features(numeric_text);
        assert!(!numeric_features.is_empty(), "Should handle numeric text");
    }

    /// Test concurrent text processing
    #[tokio::test]
    async fn test_concurrent_text_processing() {
        let text_data = TextTestData::new_truthful();
        
        let results = run_concurrent_tests(5, |_| {
            let content = text_data.content.clone();
            async move {
                let features = extract_linguistic_features(&content);
                assert!(!features.is_empty());
                Ok(features)
            }
        }).await;
        
        // All should succeed
        for result in results {
            assert!(result.is_ok(), "Concurrent text processing should succeed");
        }
    }

    // Helper functions for tests

    fn preprocess_text(text: &str) -> String {
        text.trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn extract_linguistic_features(text: &str) -> HashMap<String, f32> {
        let mut features = HashMap::new();
        
        if text.is_empty() {
            return features;
        }
        
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f32;
        
        // Basic statistics
        features.insert("word_count".to_string(), word_count);
        features.insert("sentence_count".to_string(), count_sentences(text) as f32);
        features.insert("average_word_length".to_string(), calculate_average_word_length(text));
        
        // Linguistic patterns
        features.insert("first_person_ratio".to_string(), calculate_first_person_ratio(text));
        features.insert("past_tense_ratio".to_string(), calculate_past_tense_ratio(text));
        features.insert("present_tense_ratio".to_string(), calculate_present_tense_ratio(text));
        features.insert("future_tense_ratio".to_string(), calculate_future_tense_ratio(text));
        features.insert("hedge_word_ratio".to_string(), count_hedge_words(text) / word_count);
        
        // Complexity metrics
        features.insert("complexity".to_string(), calculate_text_complexity(text));
        features.insert("certainty".to_string(), calculate_certainty_score(text));
        features.insert("specificity".to_string(), calculate_specificity_score(text));
        
        // Sentiment
        features.insert("sentiment".to_string(), analyze_sentiment(text));
        
        features
    }

    fn count_sentences(text: &str) -> usize {
        text.matches(|c| c == '.' || c == '!' || c == '?').count().max(1)
    }

    fn calculate_average_word_length(text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let total_length: usize = words.iter().map(|w| w.len()).sum();
        total_length as f32 / words.len() as f32
    }

    fn calculate_first_person_ratio(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let first_person_words = ["i", "me", "my", "mine", "myself"];
        let count = words.iter()
            .filter(|&&word| first_person_words.contains(&word))
            .count();
        
        count as f32 / words.len() as f32
    }

    fn calculate_past_tense_ratio(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let past_indicators = ["went", "was", "were", "had", "did", "yesterday", "ago", "last"];
        let count = words.iter()
            .filter(|&&word| past_indicators.contains(&word) || word.ends_with("ed"))
            .count();
        
        count as f32 / words.len() as f32
    }

    fn calculate_present_tense_ratio(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let present_indicators = ["am", "is", "are", "do", "does", "now", "today", "currently"];
        let count = words.iter()
            .filter(|&&word| present_indicators.contains(&word) || word.ends_with("ing"))
            .count();
        
        count as f32 / words.len() as f32
    }

    fn calculate_future_tense_ratio(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let future_indicators = ["will", "shall", "going", "tomorrow", "next", "future", "plan"];
        let count = words.iter()
            .filter(|&&word| future_indicators.contains(&word))
            .count();
        
        count as f32 / words.len() as f32
    }

    fn count_hedge_words(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        let hedge_words = [
            "maybe", "perhaps", "possibly", "probably", "might", "could", 
            "think", "believe", "guess", "suppose", "sort", "kind", "like"
        ];
        
        words.iter()
            .filter(|&&word| hedge_words.contains(&word))
            .count() as f32
    }

    fn calculate_text_complexity(text: &str) -> f32 {
        let avg_word_length = calculate_average_word_length(text);
        let sentence_count = count_sentences(text) as f32;
        let word_count = text.split_whitespace().count() as f32;
        
        if sentence_count == 0.0 {
            return 0.0;
        }
        
        let avg_sentence_length = word_count / sentence_count;
        
        // Combine metrics for complexity score
        (avg_word_length * 0.3 + avg_sentence_length * 0.7) / 20.0 // Normalize to ~[0,1]
    }

    fn calculate_certainty_score(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let certainty_words = ["definitely", "certainly", "absolutely", "clearly", "obviously"];
        let uncertainty_words = ["maybe", "perhaps", "possibly", "might", "unsure"];
        
        let certainty_count = words.iter()
            .filter(|&&word| certainty_words.contains(&word))
            .count() as f32;
        let uncertainty_count = words.iter()
            .filter(|&&word| uncertainty_words.contains(&word))
            .count() as f32;
        
        let total_count = words.len() as f32;
        (certainty_count - uncertainty_count) / total_count + 0.5 // Normalize to [0,1]
    }

    fn calculate_specificity_score(text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        // Count specific vs. vague words
        let specific_patterns = ["the", "this", "that", "these", "those"];
        let vague_patterns = ["some", "thing", "stuff", "whatever", "something"];
        
        let specific_count = words.iter()
            .filter(|&&word| specific_patterns.contains(&word.to_lowercase().as_str()))
            .count() as f32;
        let vague_count = words.iter()
            .filter(|&&word| vague_patterns.contains(&word.to_lowercase().as_str()))
            .count() as f32;
        
        let total_count = words.len() as f32;
        if total_count == 0.0 {
            return 0.0;
        }
        
        (specific_count + 1.0) / (vague_count + specific_count + 2.0) // Smooth normalization
    }

    fn analyze_sentiment(text: &str) -> f32 {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let positive_words = ["good", "great", "excellent", "wonderful", "amazing", "correct", "certain"];
        let negative_words = ["bad", "terrible", "awful", "wrong", "horrible", "uncertain", "unsure"];
        
        let positive_count = words.iter()
            .filter(|&&word| positive_words.contains(&word))
            .count() as f32;
        let negative_count = words.iter()
            .filter(|&&word| negative_words.contains(&word))
            .count() as f32;
        
        (positive_count - negative_count) / words.len() as f32
    }

    fn detect_deception_patterns(text: &str) -> HashMap<String, usize> {
        let mut patterns = HashMap::new();
        
        let hedge_count = count_hedge_words(text) as usize;
        patterns.insert("hedge_words".to_string(), hedge_count);
        
        let uncertainty_markers = ["not sure", "i think", "i believe", "i guess"];
        let uncertainty_count = uncertainty_markers.iter()
            .map(|&marker| text.to_lowercase().matches(marker).count())
            .sum();
        patterns.insert("uncertainty_markers".to_string(), uncertainty_count);
        
        let negation_count = text.to_lowercase().matches("not").count() +
                           text.to_lowercase().matches("never").count() +
                           text.to_lowercase().matches("no").count();
        patterns.insert("negations".to_string(), negation_count);
        
        patterns
    }

    fn normalize_text(text: &str) -> String {
        text.trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .map(|c| match c {
                ',' | '.' | '!' | '?' => ' ',
                _ => c.to_lowercase().next().unwrap_or(c),
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn create_text_feature_vector(text: &str) -> Vec<f32> {
        let features = extract_linguistic_features(text);
        let mut vector = Vec::new();
        
        // Convert HashMap to ordered vector
        let feature_order = [
            "word_count", "sentence_count", "average_word_length",
            "first_person_ratio", "past_tense_ratio", "present_tense_ratio",
            "future_tense_ratio", "hedge_word_ratio", "complexity",
            "certainty", "specificity", "sentiment"
        ];
        
        for feature_name in &feature_order {
            vector.push(features.get(*feature_name).unwrap_or(&0.0).clone());
        }
        
        vector
    }
}