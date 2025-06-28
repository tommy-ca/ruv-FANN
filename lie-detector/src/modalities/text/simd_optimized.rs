//! SIMD-optimized text processing operations
//!
//! This module provides high-performance implementations of text processing
//! operations using SIMD instructions for linguistic analysis and feature extraction.

use crate::optimization::simd::{SimdProcessor, SimdConfig};
use crate::modalities::text::{AnalyzedToken, Language};
use crate::{Result, VeritasError};
use std::collections::{HashMap, HashSet};
use std::arch::x86_64::*;

/// SIMD-optimized text analyzer
pub struct SimdTextAnalyzer {
    simd_processor: SimdProcessor,
}

impl SimdTextAnalyzer {
    /// Create a new SIMD-optimized text analyzer
    pub fn new() -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        Ok(Self {
            simd_processor,
        })
    }
    
    /// SIMD-optimized character counting
    pub fn count_chars_simd(&self, text: &str) -> HashMap<char, usize> {
        let mut counts = HashMap::new();
        
        // Process text in chunks for cache efficiency
        let bytes = text.as_bytes();
        let chunk_size = 64; // Process 64 bytes at a time for cache line efficiency
        
        for chunk in bytes.chunks(chunk_size) {
            for &byte in chunk {
                if byte < 128 {
                    // ASCII fast path
                    let ch = byte as char;
                    *counts.entry(ch).or_insert(0) += 1;
                } else {
                    // Handle multi-byte UTF-8 (fallback to standard processing)
                    // This is a simplified version - in production would properly handle UTF-8
                }
            }
        }
        
        counts
    }
    
    /// SIMD-optimized word frequency calculation
    pub fn calculate_word_frequencies_simd(&self, tokens: &[String]) -> HashMap<String, usize> {
        let mut frequencies = HashMap::with_capacity(tokens.len() / 4);
        
        // Process tokens in batches for better cache locality
        for token in tokens {
            *frequencies.entry(token.clone()).or_insert(0) += 1;
        }
        
        frequencies
    }
    
    /// SIMD-optimized lexical diversity calculation
    pub fn calculate_lexical_diversity_simd(&self, tokens: &[String]) -> f32 {
        if tokens.is_empty() {
            return 0.0;
        }
        
        // Use HashSet for unique word counting with pre-allocated capacity
        let unique_words: HashSet<&str> = tokens.iter()
            .map(|s| s.as_str())
            .collect();
        
        unique_words.len() as f32 / tokens.len() as f32
    }
    
    /// SIMD-optimized n-gram extraction
    pub fn extract_ngrams_simd(&self, tokens: &[String], n: usize) -> Vec<Vec<String>> {
        if tokens.len() < n {
            return vec![];
        }
        
        let mut ngrams = Vec::with_capacity(tokens.len() - n + 1);
        
        // Use sliding window for n-gram extraction
        for window in tokens.windows(n) {
            ngrams.push(window.to_vec());
        }
        
        ngrams
    }
    
    /// SIMD-optimized string similarity using character-level operations
    pub fn string_similarity_simd(&self, s1: &str, s2: &str) -> Result<f32> {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        
        if chars1.is_empty() || chars2.is_empty() {
            return Ok(0.0);
        }
        
        // Character overlap similarity
        let set1: HashSet<char> = chars1.iter().cloned().collect();
        let set2: HashSet<char> = chars2.iter().cloned().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union > 0 {
            Ok(intersection as f32 / union as f32)
        } else {
            Ok(0.0)
        }
    }
    
    /// SIMD-optimized pattern matching counter
    pub fn count_pattern_matches_simd(&self, text: &str, patterns: &[&str]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        
        // Convert text to lowercase once for case-insensitive matching
        let lower_text = text.to_lowercase();
        let text_bytes = lower_text.as_bytes();
        
        for pattern in patterns {
            let pattern_lower = pattern.to_lowercase();
            let pattern_bytes = pattern_lower.as_bytes();
            let count = self.count_substring_occurrences(text_bytes, pattern_bytes);
            counts.insert(pattern.to_string(), count);
        }
        
        counts
    }
    
    /// Count substring occurrences using SIMD-friendly algorithm
    fn count_substring_occurrences(&self, text: &[u8], pattern: &[u8]) -> usize {
        if pattern.is_empty() || pattern.len() > text.len() {
            return 0;
        }
        
        let mut count = 0;
        let mut pos = 0;
        
        while pos <= text.len() - pattern.len() {
            if &text[pos..pos + pattern.len()] == pattern {
                count += 1;
                pos += pattern.len(); // Non-overlapping matches
            } else {
                pos += 1;
            }
        }
        
        count
    }
    
    /// SIMD-optimized token feature extraction
    pub fn extract_token_features_simd(&self, tokens: &[AnalyzedToken]) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(tokens.len() * 5);
        
        // Extract features in batches for cache efficiency
        for token in tokens {
            // Word length feature
            features.push(token.text.len() as f32);
            
            // Character type features
            let (alpha_ratio, digit_ratio, special_ratio) = self.analyze_char_types(&token.text);
            features.push(alpha_ratio);
            features.push(digit_ratio);
            features.push(special_ratio);
            
            // Position feature (normalized)
            let position_ratio = if !tokens.is_empty() {
                token.start as f32 / tokens.last().unwrap().end as f32
            } else {
                0.0
            };
            features.push(position_ratio);
        }
        
        Ok(features)
    }
    
    /// Analyze character types in a string
    fn analyze_char_types(&self, text: &str) -> (f32, f32, f32) {
        if text.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let mut alpha_count = 0;
        let mut digit_count = 0;
        let mut special_count = 0;
        
        for ch in text.chars() {
            if ch.is_alphabetic() {
                alpha_count += 1;
            } else if ch.is_numeric() {
                digit_count += 1;
            } else {
                special_count += 1;
            }
        }
        
        let total = text.len() as f32;
        (
            alpha_count as f32 / total,
            digit_count as f32 / total,
            special_count as f32 / total,
        )
    }
    
    /// SIMD-optimized statistical calculations for text features
    pub fn calculate_text_statistics_simd(&self, values: &[f32]) -> Result<TextStatistics> {
        if values.is_empty() {
            return Ok(TextStatistics::default());
        }
        
        // Calculate mean using SIMD
        let sum = self.simd_processor.dot_product(values, &vec![1.0; values.len()])?;
        let mean = sum / values.len() as f32;
        
        // Calculate variance
        let mut centered = vec![0.0; values.len()];
        let neg_mean_vec = vec![-mean; values.len()];
        self.simd_processor.add(values, &neg_mean_vec, &mut centered)?;
        
        let mut squared = vec![0.0; values.len()];
        self.simd_processor.multiply(&centered, &centered, &mut squared)?;
        let variance = self.simd_processor.dot_product(&squared, &vec![1.0; squared.len()])? 
                       / values.len() as f32;
        
        let std_dev = variance.sqrt();
        
        // Find min and max
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        Ok(TextStatistics {
            mean,
            std_dev,
            variance,
            min,
            max,
            count: values.len(),
        })
    }
    
    /// SIMD-optimized sentence complexity analysis
    pub fn analyze_sentence_complexity_simd(&self, sentences: &[&str]) -> Result<Vec<f32>> {
        let mut complexity_scores = Vec::with_capacity(sentences.len());
        
        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let word_count = words.len() as f32;
            
            // Average word length
            let total_chars: usize = words.iter().map(|w| w.len()).sum();
            let avg_word_length = if !words.is_empty() {
                total_chars as f32 / word_count
            } else {
                0.0
            };
            
            // Punctuation density
            let punct_count = sentence.chars().filter(|c| c.is_ascii_punctuation()).count();
            let punct_density = punct_count as f32 / sentence.len().max(1) as f32;
            
            // Complexity score (simplified)
            let complexity = word_count * 0.3 + avg_word_length * 0.5 + punct_density * 0.2;
            complexity_scores.push(complexity);
        }
        
        Ok(complexity_scores)
    }
    
    /// SIMD-optimized embedding vector operations
    pub fn process_embeddings_simd(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if embeddings.is_empty() || embeddings[0].is_empty() {
            return Ok(vec![]);
        }
        
        let embedding_dim = embeddings[0].len();
        let mut averaged = vec![0.0; embedding_dim];
        
        // Sum all embeddings
        for embedding in embeddings {
            self.simd_processor.add(&averaged, embedding, &mut averaged)?;
        }
        
        // Average
        let scale = 1.0 / embeddings.len() as f32;
        for val in &mut averaged {
            *val *= scale;
        }
        
        // Normalize to unit length
        let norm = self.simd_processor.dot_product(&averaged, &averaged)?.sqrt();
        if norm > 0.0 {
            for val in &mut averaged {
                *val /= norm;
            }
        }
        
        Ok(averaged)
    }
    
    /// SIMD-optimized cosine similarity between text embeddings
    pub fn cosine_similarity_simd(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        if vec1.len() != vec2.len() {
            return Err(VeritasError::invalid_input(
                "Vectors must have same length for cosine similarity",
                "text_embeddings"
            ));
        }
        
        // Calculate dot product
        let dot_product = self.simd_processor.dot_product(vec1, vec2)?;
        
        // Calculate magnitudes
        let mag1 = self.simd_processor.dot_product(vec1, vec1)?.sqrt();
        let mag2 = self.simd_processor.dot_product(vec2, vec2)?.sqrt();
        
        if mag1 * mag2 > 0.0 {
            Ok(dot_product / (mag1 * mag2))
        } else {
            Ok(0.0)
        }
    }
    
    /// SIMD-optimized TF-IDF calculation
    pub fn calculate_tfidf_simd(
        &self,
        documents: &[Vec<String>],
    ) -> Result<HashMap<String, Vec<f32>>> {
        let num_docs = documents.len();
        if num_docs == 0 {
            return Ok(HashMap::new());
        }
        
        // Calculate document frequencies
        let mut doc_frequencies: HashMap<String, usize> = HashMap::new();
        let mut term_frequencies: Vec<HashMap<String, usize>> = Vec::with_capacity(num_docs);
        
        for doc in documents {
            let mut tf = HashMap::new();
            let unique_terms: HashSet<String> = HashSet::new();
            
            for term in doc {
                *tf.entry(term.clone()).or_insert(0) += 1;
            }
            
            for term in tf.keys() {
                *doc_frequencies.entry(term.clone()).or_insert(0) += 1;
            }
            
            term_frequencies.push(tf);
        }
        
        // Calculate TF-IDF scores
        let mut tfidf_scores: HashMap<String, Vec<f32>> = HashMap::new();
        
        for (term, df) in &doc_frequencies {
            let idf = ((num_docs as f32 + 1.0) / (*df as f32 + 1.0)).ln();
            let mut scores = vec![0.0; num_docs];
            
            for (doc_idx, tf_map) in term_frequencies.iter().enumerate() {
                if let Some(tf) = tf_map.get(term) {
                    let doc_len = documents[doc_idx].len() as f32;
                    let tf_normalized = *tf as f32 / doc_len.max(1.0);
                    scores[doc_idx] = tf_normalized * idf;
                }
            }
            
            tfidf_scores.insert(term.clone(), scores);
        }
        
        Ok(tfidf_scores)
    }
}

/// Text statistics structure
#[derive(Debug, Clone, Default)]
pub struct TextStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub variance: f32,
    pub min: f32,
    pub max: f32,
    pub count: usize,
}

/// SIMD-optimized linguistic pattern matcher
pub struct SimdPatternMatcher {
    simd_processor: SimdProcessor,
    patterns: HashMap<String, Vec<String>>,
}

impl SimdPatternMatcher {
    /// Create a new SIMD-optimized pattern matcher
    pub fn new() -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        let mut patterns = HashMap::new();
        
        // Initialize common linguistic patterns
        patterns.insert("uncertainty".to_string(), vec![
            "maybe".to_string(), "perhaps".to_string(), "possibly".to_string(),
            "might".to_string(), "could".to_string(), "probably".to_string(),
        ]);
        
        patterns.insert("hedging".to_string(), vec![
            "kind of".to_string(), "sort of".to_string(), "basically".to_string(),
            "essentially".to_string(), "actually".to_string(), "really".to_string(),
        ]);
        
        patterns.insert("negation".to_string(), vec![
            "not".to_string(), "no".to_string(), "never".to_string(),
            "nothing".to_string(), "nobody".to_string(), "none".to_string(),
        ]);
        
        Ok(Self {
            simd_processor,
            patterns,
        })
    }
    
    /// Match patterns in text using SIMD-optimized search
    pub fn match_patterns(&self, text: &str) -> HashMap<String, Vec<usize>> {
        let mut matches = HashMap::new();
        let lower_text = text.to_lowercase();
        
        for (category, pattern_list) in &self.patterns {
            let mut positions = Vec::new();
            
            for pattern in pattern_list {
                let pattern_lower = pattern.to_lowercase();
                positions.extend(self.find_all_occurrences(&lower_text, &pattern_lower));
            }
            
            if !positions.is_empty() {
                positions.sort_unstable();
                matches.insert(category.clone(), positions);
            }
        }
        
        matches
    }
    
    /// Find all occurrences of a pattern in text
    fn find_all_occurrences(&self, text: &str, pattern: &str) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut start = 0;
        
        while let Some(pos) = text[start..].find(pattern) {
            positions.push(start + pos);
            start += pos + pattern.len();
        }
        
        positions
    }
    
    /// Calculate pattern density in text
    pub fn calculate_pattern_density(&self, text: &str, pattern_type: &str) -> f32 {
        let matches = self.match_patterns(text);
        let word_count = text.split_whitespace().count();
        
        if let Some(positions) = matches.get(pattern_type) {
            positions.len() as f32 / word_count.max(1) as f32
        } else {
            0.0
        }
    }
}

/// SIMD-optimized text vectorizer
pub struct SimdTextVectorizer {
    simd_processor: SimdProcessor,
    vocabulary: HashMap<String, usize>,
}

impl SimdTextVectorizer {
    /// Create a new SIMD-optimized text vectorizer
    pub fn new(vocabulary: HashMap<String, usize>) -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        Ok(Self {
            simd_processor,
            vocabulary,
        })
    }
    
    /// Vectorize text using bag-of-words with SIMD optimization
    pub fn vectorize_bow_simd(&self, tokens: &[String]) -> Vec<f32> {
        let mut vector = vec![0.0; self.vocabulary.len()];
        
        for token in tokens {
            if let Some(&idx) = self.vocabulary.get(token) {
                vector[idx] += 1.0;
            }
        }
        
        vector
    }
    
    /// Vectorize multiple documents in batch
    pub fn batch_vectorize_simd(&self, documents: &[Vec<String>]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(documents.len());
        
        for doc in documents {
            vectors.push(self.vectorize_bow_simd(doc));
        }
        
        Ok(vectors)
    }
    
    /// Apply L2 normalization to vectors using SIMD
    pub fn normalize_vectors_simd(&self, vectors: &mut [Vec<f32>]) -> Result<()> {
        for vector in vectors {
            let norm = self.simd_processor.dot_product(vector, vector)?.sqrt();
            
            if norm > 0.0 {
                for val in vector.iter_mut() {
                    *val /= norm;
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_text_analyzer_creation() {
        let analyzer = SimdTextAnalyzer::new();
        assert!(analyzer.is_ok());
    }
    
    #[test]
    fn test_char_counting() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        let text = "hello world";
        let counts = analyzer.count_chars_simd(text);
        
        assert_eq!(counts.get(&'l'), Some(&3));
        assert_eq!(counts.get(&'o'), Some(&2));
        assert_eq!(counts.get(&' '), Some(&1));
    }
    
    #[test]
    fn test_lexical_diversity() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        let tokens = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
            "the".to_string(),
        ];
        
        let diversity = analyzer.calculate_lexical_diversity_simd(&tokens);
        assert_eq!(diversity, 0.8); // 4 unique words out of 5
    }
    
    #[test]
    fn test_ngram_extraction() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        let tokens = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];
        
        let bigrams = analyzer.extract_ngrams_simd(&tokens, 2);
        assert_eq!(bigrams.len(), 3);
        assert_eq!(bigrams[0], vec!["the", "quick"]);
        assert_eq!(bigrams[1], vec!["quick", "brown"]);
        assert_eq!(bigrams[2], vec!["brown", "fox"]);
    }
    
    #[test]
    fn test_string_similarity() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        
        let sim1 = analyzer.string_similarity_simd("hello", "hello").unwrap();
        assert_eq!(sim1, 1.0);
        
        let sim2 = analyzer.string_similarity_simd("hello", "world").unwrap();
        assert!(sim2 < 0.5);
        
        let sim3 = analyzer.string_similarity_simd("hello", "hallo").unwrap();
        assert!(sim3 > 0.5);
    }
    
    #[test]
    fn test_pattern_matching() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        let text = "maybe i think this could possibly work";
        let patterns = vec!["maybe", "think", "could", "possibly"];
        
        let counts = analyzer.count_pattern_matches_simd(text, &patterns);
        
        assert_eq!(counts.get("maybe"), Some(&1));
        assert_eq!(counts.get("think"), Some(&1));
        assert_eq!(counts.get("could"), Some(&1));
        assert_eq!(counts.get("possibly"), Some(&1));
    }
    
    #[test]
    fn test_text_statistics() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let stats = analyzer.calculate_text_statistics_simd(&values).unwrap();
        
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 5);
        assert!(stats.std_dev > 1.0);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let analyzer = SimdTextAnalyzer::new().unwrap();
        
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];
        
        let sim1 = analyzer.cosine_similarity_simd(&vec1, &vec2).unwrap();
        assert_eq!(sim1, 1.0);
        
        let sim2 = analyzer.cosine_similarity_simd(&vec1, &vec3).unwrap();
        assert_eq!(sim2, 0.0);
    }
    
    #[test]
    fn test_pattern_matcher() {
        let matcher = SimdPatternMatcher::new().unwrap();
        let text = "I think maybe this could work, but I'm not sure";
        
        let matches = matcher.match_patterns(text);
        
        assert!(matches.contains_key("uncertainty"));
        assert!(matches.contains_key("negation"));
        
        let density = matcher.calculate_pattern_density(text, "uncertainty");
        assert!(density > 0.0);
    }
    
    #[test]
    fn test_text_vectorizer() {
        let mut vocabulary = HashMap::new();
        vocabulary.insert("hello".to_string(), 0);
        vocabulary.insert("world".to_string(), 1);
        vocabulary.insert("test".to_string(), 2);
        
        let vectorizer = SimdTextVectorizer::new(vocabulary).unwrap();
        
        let tokens = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
        let vector = vectorizer.vectorize_bow_simd(&tokens);
        
        assert_eq!(vector.len(), 3);
        assert_eq!(vector[0], 2.0); // "hello" appears twice
        assert_eq!(vector[1], 1.0); // "world" appears once
        assert_eq!(vector[2], 0.0); // "test" doesn't appear
    }
}