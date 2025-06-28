//! Linguistic analysis module for extracting textual features relevant to deception detection

use crate::{Result, VeritasError};
use crate::types::*;
use num_traits::Float;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;
use whatlang::{detect, Lang};
use std::collections::{HashMap, HashSet};
use once_cell::sync::Lazy;

/// Language enumeration for linguistic analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Russian,
    Chinese,
    Japanese,
    Korean,
    Arabic,
    Other,
}

/// Entity types for named entity recognition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Money,
    Date,
    Time,
    Number,
    Percentage,
    Other,
}

/// Configuration for preprocessing text
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub normalize_unicode: bool,
    pub filter_stopwords: bool,
}

/// Linguistic analyzer for text processing and feature extraction
pub struct LinguisticAnalyzer<T: Float> {
    stopwords: HashMap<Language, HashSet<String>>,
    uncertainty_markers: Regex,
    hedging_patterns: Regex,
    temporal_patterns: Regex,
    self_reference_patterns: Regex,
    certainty_markers: Regex,
    negation_patterns: Regex,
    emotion_words: HashMap<String, f64>,
    _phantom: std::marker::PhantomData<T>,
}

// Regex patterns compiled once
static UNCERTAINTY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(maybe|perhaps|possibly|might|could|probably|likely|seem|appear|think|believe|guess|suppose)\b").unwrap()
});

static HEDGING_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(kind of|sort of|basically|essentially|actually|really|quite|rather|somewhat|fairly|pretty)\b").unwrap()
});

static TEMPORAL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(yesterday|today|tomorrow|last week|next week|ago|before|after|then|now|when|while|during|since)\b").unwrap()
});

static SELF_REF_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b").unwrap()
});

static CERTAINTY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(definitely|certainly|absolutely|surely|clearly|obviously|undoubtedly|always|never|exactly)\b").unwrap()
});

static NEGATION_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(not|no|never|nothing|nobody|none|neither|nor|cannot|can't|won't|wouldn't|shouldn't|couldn't|didn't|don't|doesn't|isn't|aren't|wasn't|weren't)\b").unwrap()
});

/// Linguistic features extracted from text
#[derive(Debug, Clone)]
pub struct LinguisticFeatures<T: Float> {
    /// Word count
    pub word_count: usize,
    /// Sentence count
    pub sentence_count: usize,
    /// Average word length
    pub avg_word_length: T,
    /// Average sentence length
    pub avg_sentence_length: T,
    /// Lexical diversity (unique words / total words)
    pub lexical_diversity: T,
    /// Readability score
    pub readability_score: T,
    /// Uncertainty markers count
    pub uncertainty_markers: usize,
    /// Hedging patterns count
    pub hedging_patterns: usize,
    /// Self-reference count
    pub self_references: usize,
    /// Negation count
    pub negations: usize,
    /// Emotion word count
    pub emotion_words: usize,
    /// Named entities
    pub named_entities: Vec<NamedEntity>,
    /// Feature vector for ML models
    pub feature_vector: Vec<T>,
    /// Lexical features
    pub lexical_features: Vec<T>,
    /// Syntactic features
    pub syntactic_features: Vec<T>,
    /// Semantic features
    pub semantic_features: Vec<T>,
}

impl<T: Float> LinguisticFeatures<T> {
    /// Get total feature count
    pub fn feature_count(&self) -> usize {
        self.feature_vector.len()
    }
}

/// Named entity extracted from text
#[derive(Debug, Clone)]
pub struct NamedEntity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentResult<T: Float> {
    /// Overall sentiment score (-1.0 to 1.0)
    pub sentiment_score: T,
    /// Confidence in sentiment analysis
    pub confidence: T,
    /// Positive sentiment strength
    pub positive_strength: T,
    /// Negative sentiment strength
    pub negative_strength: T,
    /// Neutral sentiment strength
    pub neutral_strength: T,
    /// Emotion analysis
    pub emotions: Vec<EmotionScore<T>>,
}

/// Emotion score for specific emotion
#[derive(Debug, Clone)]
pub struct EmotionScore<T: Float> {
    /// Emotion name
    pub emotion: String,
    /// Intensity score (0.0 to 1.0)
    pub intensity: T,
    /// Confidence in detection
    pub confidence: T,
}

/// Analyzed token with linguistic properties
#[derive(Debug, Clone)]
pub struct AnalyzedToken {
    /// Token text
    pub text: String,
    /// Lemmatized form
    pub lemma: String,
    /// Part-of-speech tag
    pub pos_tag: PosTag,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Is this a stopword?
    pub is_stopword: bool,
    /// Token frequency
    pub frequency: usize,
}

/// Part-of-speech tags
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosTag {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Pronoun,
    Preposition,
    Conjunction,
    Interjection,
    Determiner,
    Unknown,
}

/// Text complexity metrics
#[derive(Debug, Clone)]
pub struct ComplexityMetrics<T: Float> {
    /// Flesch reading ease
    pub flesch_reading_ease: T,
    /// Flesch-Kincaid grade level
    pub flesch_kincaid_grade: T,
    /// Average syllables per word
    pub avg_syllables_per_word: T,
    /// Type-token ratio
    pub type_token_ratio: T,
    /// Syntactic complexity
    pub syntactic_complexity: T,
}

/// Temporal pattern analysis
#[derive(Debug, Clone)]
pub struct TemporalPattern<T: Float> {
    /// Temporal reference frequency
    pub temporal_references: T,
    /// Tense consistency score
    pub tense_consistency: T,
    /// Time expression complexity
    pub time_complexity: T,
}

/// Cognitive load indicators
#[derive(Debug, Clone)]
pub struct CognitiveLoadIndicators<T: Float> {
    /// Mental effort score
    pub mental_effort: T,
    /// Processing difficulty
    pub processing_difficulty: T,
    /// Cognitive complexity
    pub cognitive_complexity: T,
}

/// Semantic coherence metrics
#[derive(Debug, Clone)]
pub struct SemanticCoherence<T: Float> {
    /// Coherence score
    pub coherence_score: T,
    /// Topic consistency
    pub topic_consistency: T,
    /// Semantic similarity
    pub semantic_similarity: T,
}

impl<T: Float> LinguisticAnalyzer<T> {
    /// Create a new linguistic analyzer
    pub fn new(config: &PreprocessingConfig) -> Result<Self> {
        let stopwords = Self::load_stopwords()?;
        let emotion_words = Self::load_emotion_lexicon()?;
        
        Ok(Self {
            stopwords,
            uncertainty_markers: UNCERTAINTY_REGEX.clone(),
            hedging_patterns: HEDGING_REGEX.clone(),
            temporal_patterns: TEMPORAL_REGEX.clone(),
            self_reference_patterns: SELF_REF_REGEX.clone(),
            certainty_markers: CERTAINTY_REGEX.clone(),
            negation_patterns: NEGATION_REGEX.clone(),
            emotion_words,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Detect the language of the input text
    pub fn detect_language(&self, text: &str) -> Result<Language> {
        if text.trim().is_empty() {
            return Ok(Language::Unknown);
        }
        
        match detect(text) {
            Some(info) => {
                let lang = match info.lang() {
                    Lang::Eng => Language::English,
                    Lang::Spa => Language::Spanish,
                    Lang::Fra => Language::French,
                    Lang::Deu => Language::German,
                    Lang::Ita => Language::Italian,
                    Lang::Por => Language::Portuguese,
                    Lang::Rus => Language::Russian,
                    Lang::Cmn => Language::Chinese,
                    Lang::Jpn => Language::Japanese,
                    Lang::Kor => Language::Korean,
                    Lang::Ara => Language::Arabic,
                    _ => Language::Unknown,
                };
                Ok(lang)
            }
            None => Ok(Language::Unknown),
        }
    }
    
    /// Preprocess text according to configuration
    pub fn preprocess(&self, text: &str, language: Language, config: &PreprocessingConfig) -> Result<String> {
        let mut processed = text.to_string();
        
        // Unicode normalization
        if config.normalize_unicode {
            processed = processed.nfc().collect();
        }
        
        // Remove emoji if requested
        if config.remove_emoji {
            processed = self.remove_emoji(&processed);
        }
        
        // Convert to lowercase
        if config.lowercase {
            processed = processed.to_lowercase();
        }
        
        // Remove punctuation (but keep sentence structure)
        if config.remove_punctuation {
            processed = self.remove_punctuation(&processed);
        }
        
        // Remove stopwords
        if config.remove_stopwords {
            processed = self.remove_stopwords(&processed, language)?;
        }
        
        // Stem words (basic stemming)
        if config.stem_words {
            processed = self.stem_words(&processed, language)?;
        }
        
        // Truncate if too long
        if let Some(max_len) = config.max_length {
            if processed.len() > max_len {
                processed = processed.chars().take(max_len).collect();
            }
        }
        
        Ok(processed)
    }
    
    /// Extract comprehensive linguistic features
    pub async fn extract_features(&self, text: &str, language: Language) -> Result<LinguisticFeatures<T>> {
        if text.trim().is_empty() {
            return Err(VeritasError::invalid_input("Empty text provided"));
        }
        
        let tokens = self.tokenize_and_analyze(text, language)?;
        let word_count = tokens.len();
        let sentence_count = self.count_sentences(text);
        
        // Extract different types of features
        let lexical_features = self.extract_lexical_features(text, &tokens)?;
        let syntactic_features = self.extract_syntactic_features(text, &tokens)?;
        let semantic_features = self.extract_semantic_features(text, &tokens, language).await?;
        let pragmatic_features = self.extract_pragmatic_features(text, &tokens)?;
        let discourse_features = self.extract_discourse_features(text, &tokens)?;
        
        // Generate feature names for explainability
        let feature_names = self.generate_feature_names();
        
        Ok(LinguisticFeatures {
            lexical_features,
            syntactic_features,
            semantic_features,
            pragmatic_features,
            discourse_features,
            feature_names,
        })
    }
    
    /// Analyze sentiment of the text
    pub fn analyze_sentiment(&self, text: &str, language: Language) -> Result<SentimentResult<T>> {
        if !language.is_supported() {
            return Err(VeritasError::unsupported_language(format!("{:?}", language)));
        }
        
        let tokens = text.to_lowercase()
            .split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>();
        
        let mut positive_score = 0.0;
        let mut negative_score = 0.0;
        let mut total_words = 0;
        
        for token in &tokens {
            if let Some(&polarity) = self.emotion_words.get(*token) {
                total_words += 1;
                if polarity > 0.0 {
                    positive_score += polarity;
                } else {
                    negative_score += polarity.abs();
                }
            }
        }
        
        let neutral_score = tokens.len() as f64 - total_words as f64;
        let total = positive_score + negative_score + neutral_score;
        
        let (pos, neg, neu) = if total > 0.0 {
            (positive_score / total, negative_score / total, neutral_score / total)
        } else {
            (0.0, 0.0, 1.0)
        };
        
        let compound = (positive_score - negative_score) / (positive_score + negative_score + 1.0);
        let confidence = if total_words > 0 { (total_words as f64 / tokens.len() as f64).min(1.0) } else { 0.0 };
        
        Ok(SentimentResult {
            positive: T::from(pos).unwrap(),
            negative: T::from(neg).unwrap(),
            neutral: T::from(neu).unwrap(),
            compound: T::from(compound).unwrap(),
            confidence: T::from(confidence).unwrap(),
        })
    }
    
    /// Extract named entities (simplified implementation)
    pub fn extract_entities(&self, text: &str, language: Language) -> Result<Vec<NamedEntity>> {
        if !language.is_supported() {
            return Ok(Vec::new());
        }
        
        let mut entities = Vec::new();
        
        // Simple pattern-based NER (in production, use a proper NER model)
        let patterns = vec![
            (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", EntityType::Person),
            (r"\b[A-Z][a-z]+ Inc\.\b|\b[A-Z][a-z]+ Corp\.\b|\b[A-Z][a-z]+ LLC\b", EntityType::Organization),
            (r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b", EntityType::Money),
            (r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", EntityType::Date),
            (r"\b\d+%\b", EntityType::Percentage),
        ];
        
        for (pattern, entity_type) in patterns {
            let regex = Regex::new(pattern).map_err(|e| VeritasError::preprocessing(e.to_string()))?;
            
            for mat in regex.find_iter(text) {
                entities.push(NamedEntity {
                    text: mat.as_str().to_string(),
                    entity_type,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.8, // Static confidence for simple patterns
                    deception_relevance: self.calculate_entity_deception_relevance(entity_type),
                });
            }
        }
        
        Ok(entities)
    }
    
    /// Tokenize and analyze each token
    pub fn tokenize_and_analyze(&self, text: &str, language: Language) -> Result<Vec<AnalyzedToken>> {
        let mut tokens = Vec::new();
        let mut start = 0;
        
        for word in text.split_whitespace() {
            let word_start = start;
            let word_end = start + word.len();
            start = word_end + 1;
            
            let cleaned = word.trim_matches(|c: char| !c.is_alphabetic());
            if cleaned.is_empty() {
                continue;
            }
            
            let lemma = self.simple_lemmatize(cleaned, language)?;
            let pos_tag = self.simple_pos_tag(cleaned);
            let is_stopword = self.is_stopword(cleaned, language);
            let sentiment_polarity = self.emotion_words.get(cleaned).cloned().unwrap_or(0.0);
            let uncertainty_score = if self.uncertainty_markers.is_match(cleaned) { 1.0 } else { 0.0 };
            
            tokens.push(AnalyzedToken {
                text: cleaned.to_string(),
                lemma,
                pos_tag,
                start: word_start,
                end: word_end,
                is_stopword,
                sentiment_polarity,
                uncertainty_score,
            });
        }
        
        Ok(tokens)
    }
    
    /// Calculate text complexity metrics
    pub fn calculate_complexity(&self, text: &str) -> Result<ComplexityMetrics<T>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let sentences = self.count_sentences(text);
        let syllables = self.count_syllables(text);
        
        let word_count = words.len();
        let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let unique_word_ratio = if word_count > 0 {
            unique_words.len() as f64 / word_count as f64
        } else {
            0.0
        };
        
        let avg_sentence_length = if sentences > 0 {
            word_count as f64 / sentences as f64
        } else {
            0.0
        };
        
        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };
        
        // Flesch-Kincaid Grade Level
        let fk_grade = if sentences > 0 && word_count > 0 {
            (0.39 * avg_sentence_length) + (11.8 * (syllables as f64 / word_count as f64)) - 15.59
        } else {
            0.0
        };
        
        // Flesch Reading Ease
        let flesch_ease = if sentences > 0 && word_count > 0 {
            206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllables as f64 / word_count as f64))
        } else {
            0.0
        };
        
        Ok(ComplexityMetrics {
            flesch_kincaid_grade: T::from(fk_grade.max(0.0)).unwrap(),
            flesch_reading_ease: T::from(flesch_ease.max(0.0)).unwrap(),
            average_sentence_length: T::from(avg_sentence_length).unwrap(),
            average_word_length: T::from(avg_word_length).unwrap(),
            syllable_count: syllables,
            word_count,
            sentence_count: sentences,
            unique_word_ratio: T::from(unique_word_ratio).unwrap(),
        })
    }
    
    /// Analyze temporal patterns in text
    pub fn analyze_temporal_patterns(&self, tokens: &[AnalyzedToken]) -> Result<TemporalPattern<T>> {
        let mut past_count = 0;
        let mut present_count = 0;
        let mut future_count = 0;
        
        let past_indicators = ["was", "were", "had", "did", "ago", "before", "yesterday", "last"];
        let present_indicators = ["is", "are", "am", "have", "has", "do", "does", "now", "today"];
        let future_indicators = ["will", "shall", "going", "tomorrow", "next", "later", "soon"];
        
        for token in tokens {
            let word = token.text.to_lowercase();
            if past_indicators.contains(&word.as_str()) {
                past_count += 1;
            } else if present_indicators.contains(&word.as_str()) {
                present_count += 1;
            } else if future_indicators.contains(&word.as_str()) {
                future_count += 1;
            }
        }
        
        let total = past_count + present_count + future_count;
        let (past_ratio, present_ratio, future_ratio) = if total > 0 {
            (past_count as f64 / total as f64,
             present_count as f64 / total as f64,
             future_count as f64 / total as f64)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        // Calculate temporal consistency (higher when one tense dominates)
        let max_ratio = past_ratio.max(present_ratio).max(future_ratio);
        let temporal_consistency = max_ratio;
        
        // Timeline clarity (higher when temporal markers are present)
        let timeline_clarity = if tokens.len() > 0 {
            total as f64 / tokens.len() as f64
        } else {
            0.0
        };
        
        Ok(TemporalPattern {
            past_references: T::from(past_ratio).unwrap(),
            present_references: T::from(present_ratio).unwrap(),
            future_references: T::from(future_ratio).unwrap(),
            temporal_consistency: T::from(temporal_consistency).unwrap(),
            timeline_clarity: T::from(timeline_clarity).unwrap(),
        })
    }
    
    /// Analyze cognitive load indicators
    pub fn analyze_cognitive_load(&self, tokens: &[AnalyzedToken]) -> Result<CognitiveLoadIndicators<T>> {
        let hesitation_markers = ["um", "uh", "er", "ah", "well", "you know"];
        let filler_words = ["like", "you know", "I mean", "actually", "basically"];
        
        let mut hesitation_count = 0;
        let mut filler_count = 0;
        let mut repetition_count = 0;
        
        // Count hesitation markers and fillers
        for token in tokens {
            let word = token.text.to_lowercase();
            if hesitation_markers.contains(&word.as_str()) {
                hesitation_count += 1;
            }
            if filler_words.contains(&word.as_str()) {
                filler_count += 1;
            }
        }
        
        // Simple repetition detection
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            let word = token.text.to_lowercase();
            *word_counts.entry(word).or_insert(0) += 1;
        }
        
        repetition_count = word_counts.values().filter(|&&count| count > 1).sum::<usize>();
        
        let token_count = tokens.len() as f64;
        
        Ok(CognitiveLoadIndicators {
            hesitation_markers: T::from(if token_count > 0.0 { hesitation_count as f64 / token_count } else { 0.0 }).unwrap(),
            correction_frequency: T::from(0.0).unwrap(), // Would need more sophisticated analysis
            filler_words: T::from(if token_count > 0.0 { filler_count as f64 / token_count } else { 0.0 }).unwrap(),
            incomplete_sentences: T::from(0.0).unwrap(), // Would need parsing
            repetition_rate: T::from(if token_count > 0.0 { repetition_count as f64 / token_count } else { 0.0 }).unwrap(),
            processing_effort: T::from(if token_count > 0.0 { (hesitation_count + filler_count) as f64 / token_count } else { 0.0 }).unwrap(),
        })
    }
    
    /// Analyze semantic coherence
    pub fn analyze_semantic_coherence(&self, text: &str, language: Language) -> Result<SemanticCoherence<T>> {
        // Simplified implementation - in production would use embeddings
        let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // Topic consistency (simplified as lexical overlap between sentences)
        let topic_consistency = if sentences.len() > 1 {
            self.calculate_lexical_overlap(&sentences)?
        } else {
            1.0
        };
        
        // Discourse flow (simplified as sentence length variance)
        let discourse_flow = self.calculate_discourse_flow(&sentences)?;
        
        // Logical structure (simplified)
        let logical_structure = self.calculate_logical_structure(text)?;
        
        // Narrative coherence (placeholder)
        let narrative_coherence = (topic_consistency + discourse_flow + logical_structure) / 3.0;
        
        Ok(SemanticCoherence {
            topic_consistency: T::from(topic_consistency).unwrap(),
            semantic_similarity: T::from(topic_consistency).unwrap(), // Simplified
            discourse_flow: T::from(discourse_flow).unwrap(),
            logical_structure: T::from(logical_structure).unwrap(),
            narrative_coherence: T::from(narrative_coherence).unwrap(),
        })
    }
    
    // Private helper methods
    
    fn load_stopwords() -> Result<HashMap<Language, HashSet<String>>> {
        let mut stopwords = HashMap::new();
        
        // English stopwords (simplified set)
        let english_stops: HashSet<String> = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its",
            "of", "on", "that", "the", "to", "was", "were", "will", "with", "the", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "she", "do", "how", "their", "if", "up", "out", "many", "then",
            "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "two", "more",
            "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find", "long",
            "down", "day", "did", "get", "come", "made", "may", "part"
        ].iter().map(|s| s.to_string()).collect();
        
        stopwords.insert(Language::English, english_stops);
        
        // Add other languages as needed
        stopwords.insert(Language::Spanish, HashSet::new());
        stopwords.insert(Language::French, HashSet::new());
        
        Ok(stopwords)
    }
    
    fn load_emotion_lexicon() -> Result<HashMap<String, f64>> {
        let mut lexicon = HashMap::new();
        
        // Simplified emotion lexicon (positive values = positive sentiment)
        let positive_words = [
            ("good", 0.8), ("great", 0.9), ("excellent", 1.0), ("amazing", 0.9), ("wonderful", 0.9),
            ("happy", 0.8), ("joy", 0.9), ("love", 0.9), ("beautiful", 0.8), ("awesome", 0.9),
            ("fantastic", 0.9), ("brilliant", 0.8), ("perfect", 1.0), ("success", 0.8), ("win", 0.7),
        ];
        
        let negative_words = [
            ("bad", -0.8), ("terrible", -0.9), ("awful", -1.0), ("horrible", -0.9), ("hate", -0.9),
            ("sad", -0.8), ("angry", -0.8), ("fear", -0.7), ("ugly", -0.8), ("fail", -0.8),
            ("wrong", -0.7), ("problem", -0.6), ("issue", -0.6), ("concern", -0.5), ("worry", -0.7),
        ];
        
        for (word, score) in positive_words {
            lexicon.insert(word.to_string(), score);
        }
        
        for (word, score) in negative_words {
            lexicon.insert(word.to_string(), score);
        }
        
        Ok(lexicon)
    }
    
    fn remove_emoji(&self, text: &str) -> String {
        // Simple emoji removal (Unicode ranges)
        text.chars()
            .filter(|c| {
                let code = *c as u32;
                !(0x1F600..=0x1F64F).contains(&code) && // Emoticons
                !(0x1F300..=0x1F5FF).contains(&code) && // Misc Symbols
                !(0x1F680..=0x1F6FF).contains(&code) && // Transport
                !(0x2600..=0x26FF).contains(&code)     // Misc symbols
            })
            .collect()
    }
    
    fn remove_punctuation(&self, text: &str) -> String {
        text.chars()
            .map(|c| if c.is_alphabetic() || c.is_whitespace() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    fn remove_stopwords(&self, text: &str, language: Language) -> Result<String> {
        if let Some(stops) = self.stopwords.get(&language) {
            let words: Vec<&str> = text.split_whitespace()
                .filter(|word| !stops.contains(&word.to_lowercase()))
                .collect();
            Ok(words.join(" "))
        } else {
            Ok(text.to_string())
        }
    }
    
    fn stem_words(&self, text: &str, _language: Language) -> Result<String> {
        // Simplified stemming - in production use proper stemmer
        let words: Vec<String> = text.split_whitespace()
            .map(|word| {
                let lower = word.to_lowercase();
                if lower.ends_with("ing") && lower.len() > 4 {
                    lower[..lower.len()-3].to_string()
                } else if lower.ends_with("ed") && lower.len() > 3 {
                    lower[..lower.len()-2].to_string()
                } else if lower.ends_with("s") && lower.len() > 2 {
                    lower[..lower.len()-1].to_string()
                } else {
                    lower
                }
            })
            .collect();
        Ok(words.join(" "))
    }
    
    fn count_sentences(&self, text: &str) -> usize {
        text.matches(|c| c == '.' || c == '!' || c == '?').count().max(1)
    }
    
    fn count_syllables(&self, text: &str) -> usize {
        // Simplified syllable counting
        text.split_whitespace()
            .map(|word| {
                let vowels = word.to_lowercase()
                    .chars()
                    .filter(|c| "aeiou".contains(*c))
                    .count();
                vowels.max(1)
            })
            .sum()
    }
    
    fn is_stopword(&self, word: &str, language: Language) -> bool {
        self.stopwords.get(&language)
            .map(|stops| stops.contains(&word.to_lowercase()))
            .unwrap_or(false)
    }
    
    fn simple_lemmatize(&self, word: &str, _language: Language) -> Result<String> {
        // Simplified lemmatization
        let lower = word.to_lowercase();
        if lower.ends_with("ing") && lower.len() > 4 {
            Ok(lower[..lower.len()-3].to_string())
        } else if lower.ends_with("ed") && lower.len() > 3 {
            Ok(lower[..lower.len()-2].to_string())
        } else if lower.ends_with("s") && lower.len() > 2 {
            Ok(lower[..lower.len()-1].to_string())
        } else {
            Ok(lower)
        }
    }
    
    fn simple_pos_tag(&self, word: &str) -> PosTag {
        // Very simplified POS tagging
        let lower = word.to_lowercase();
        if ["the", "a", "an"].contains(&lower.as_str()) {
            PosTag::Determiner
        } else if lower.ends_with("ing") || lower.ends_with("ed") {
            PosTag::Verb
        } else if lower.ends_with("ly") {
            PosTag::Adverb
        } else if ["i", "you", "he", "she", "it", "we", "they"].contains(&lower.as_str()) {
            PosTag::Pronoun
        } else {
            PosTag::Unknown
        }
    }
    
    fn calculate_entity_deception_relevance(&self, entity_type: EntityType) -> f64 {
        match entity_type {
            EntityType::Person => 0.8,      // Names can be fabricated
            EntityType::Organization => 0.7, // Company names can be fake
            EntityType::Money => 0.9,        // Amounts often exaggerated in deception
            EntityType::Date => 0.8,         // Dates can be falsified
            EntityType::Time => 0.7,         // Times can be approximate
            EntityType::Location => 0.8,     // Places can be fabricated
            EntityType::Number => 0.6,       // Numbers moderately relevant
            EntityType::Percentage => 0.7,   // Percentages can be misleading
            EntityType::Other => 0.3,        // Low relevance for unknown types
        }
    }
    
    fn extract_lexical_features(&self, text: &str, tokens: &[AnalyzedToken]) -> Result<Vec<T>> {
        let mut features = Vec::new();
        
        // Basic lexical statistics
        features.push(T::from(tokens.len()).unwrap()); // Word count
        features.push(T::from(text.chars().count()).unwrap()); // Character count
        
        // Average word length
        let avg_word_len = if !tokens.is_empty() {
            tokens.iter().map(|t| t.text.len()).sum::<usize>() as f64 / tokens.len() as f64
        } else {
            0.0
        };
        features.push(T::from(avg_word_len).unwrap());
        
        // Type-token ratio (lexical diversity)
        let unique_words: HashSet<String> = tokens.iter().map(|t| t.text.to_lowercase()).collect();
        let ttr = if !tokens.is_empty() {
            unique_words.len() as f64 / tokens.len() as f64
        } else {
            0.0
        };
        features.push(T::from(ttr).unwrap());
        
        // Pattern matching features
        features.push(T::from(self.uncertainty_markers.find_iter(text).count()).unwrap());
        features.push(T::from(self.hedging_patterns.find_iter(text).count()).unwrap());
        features.push(T::from(self.certainty_markers.find_iter(text).count()).unwrap());
        features.push(T::from(self.negation_patterns.find_iter(text).count()).unwrap());
        features.push(T::from(self.self_reference_patterns.find_iter(text).count()).unwrap());
        
        Ok(features)
    }
    
    fn extract_syntactic_features(&self, text: &str, tokens: &[AnalyzedToken]) -> Result<Vec<T>> {
        let mut features = Vec::new();
        
        // Sentence count
        let sentence_count = self.count_sentences(text);
        features.push(T::from(sentence_count).unwrap());
        
        // Average sentence length
        let avg_sent_len = if sentence_count > 0 {
            tokens.len() as f64 / sentence_count as f64
        } else {
            0.0
        };
        features.push(T::from(avg_sent_len).unwrap());
        
        // POS tag distributions
        let pos_counts = tokens.iter().fold(HashMap::new(), |mut acc, token| {
            *acc.entry(token.pos_tag).or_insert(0) += 1;
            acc
        });
        
        let total_tokens = tokens.len() as f64;
        for pos in [PosTag::Noun, PosTag::Verb, PosTag::Adjective, PosTag::Adverb, PosTag::Pronoun] {
            let count = pos_counts.get(&pos).unwrap_or(&0);
            let ratio = if total_tokens > 0.0 { *count as f64 / total_tokens } else { 0.0 };
            features.push(T::from(ratio).unwrap());
        }
        
        Ok(features)
    }
    
    async fn extract_semantic_features(&self, text: &str, tokens: &[AnalyzedToken], _language: Language) -> Result<Vec<T>> {
        let mut features = Vec::new();
        
        // Semantic density (simplified as content word ratio)
        let content_words = tokens.iter().filter(|t| !t.is_stopword).count();
        let semantic_density = if !tokens.is_empty() {
            content_words as f64 / tokens.len() as f64
        } else {
            0.0
        };
        features.push(T::from(semantic_density).unwrap());
        
        // Average sentiment polarity
        let avg_sentiment = if !tokens.is_empty() {
            tokens.iter().map(|t| t.sentiment_polarity).sum::<f64>() / tokens.len() as f64
        } else {
            0.0
        };
        features.push(T::from(avg_sentiment).unwrap());
        
        // Sentiment variance
        let sentiment_variance = if !tokens.is_empty() {
            let mean = avg_sentiment;
            let variance = tokens.iter()
                .map(|t| (t.sentiment_polarity - mean).powi(2))
                .sum::<f64>() / tokens.len() as f64;
            variance
        } else {
            0.0
        };
        features.push(T::from(sentiment_variance).unwrap());
        
        Ok(features)
    }
    
    fn extract_pragmatic_features(&self, text: &str, tokens: &[AnalyzedToken]) -> Result<Vec<T>> {
        let mut features = Vec::new();
        
        // Uncertainty markers ratio
        let uncertainty_count = tokens.iter().filter(|t| t.uncertainty_score > 0.0).count();
        let uncertainty_ratio = if !tokens.is_empty() {
            uncertainty_count as f64 / tokens.len() as f64
        } else {
            0.0
        };
        features.push(T::from(uncertainty_ratio).unwrap());
        
        // Question marks (interrogative sentences)
        let question_count = text.matches('?').count();
        features.push(T::from(question_count).unwrap());
        
        // Exclamation marks (emphatic sentences)
        let exclamation_count = text.matches('!').count();
        features.push(T::from(exclamation_count).unwrap());
        
        Ok(features)
    }
    
    fn extract_discourse_features(&self, text: &str, tokens: &[AnalyzedToken]) -> Result<Vec<T>> {
        let mut features = Vec::new();
        
        // Discourse connectives
        let connectives = ["but", "however", "therefore", "because", "although", "since", "while"];
        let connective_count = tokens.iter()
            .filter(|t| connectives.contains(&t.text.to_lowercase().as_str()))
            .count();
        features.push(T::from(connective_count).unwrap());
        
        // Temporal discourse markers
        let temporal_count = self.temporal_patterns.find_iter(text).count();
        features.push(T::from(temporal_count).unwrap());
        
        Ok(features)
    }
    
    fn generate_feature_names(&self) -> Vec<String> {
        vec![
            // Lexical features
            "word_count".to_string(),
            "char_count".to_string(),
            "avg_word_length".to_string(),
            "type_token_ratio".to_string(),
            "uncertainty_markers".to_string(),
            "hedging_patterns".to_string(),
            "certainty_markers".to_string(),
            "negation_patterns".to_string(),
            "self_references".to_string(),
            
            // Syntactic features
            "sentence_count".to_string(),
            "avg_sentence_length".to_string(),
            "noun_ratio".to_string(),
            "verb_ratio".to_string(),
            "adjective_ratio".to_string(),
            "adverb_ratio".to_string(),
            "pronoun_ratio".to_string(),
            
            // Semantic features
            "semantic_density".to_string(),
            "avg_sentiment".to_string(),
            "sentiment_variance".to_string(),
            
            // Pragmatic features
            "uncertainty_ratio".to_string(),
            "question_count".to_string(),
            "exclamation_count".to_string(),
            
            // Discourse features
            "connective_count".to_string(),
            "temporal_markers".to_string(),
        ]
    }
    
    fn calculate_lexical_overlap(&self, sentences: &[&str]) -> Result<f64> {
        if sentences.len() < 2 {
            return Ok(1.0);
        }
        
        let mut overlaps = Vec::new();
        
        for i in 0..sentences.len()-1 {
            let words1: HashSet<String> = sentences[i].split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            let words2: HashSet<String> = sentences[i+1].split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            
            let intersection = words1.intersection(&words2).count();
            let union = words1.union(&words2).count();
            
            let overlap = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };
            
            overlaps.push(overlap);
        }
        
        Ok(overlaps.iter().sum::<f64>() / overlaps.len() as f64)
    }
    
    fn calculate_discourse_flow(&self, sentences: &[&str]) -> Result<f64> {
        if sentences.len() < 2 {
            return Ok(1.0);
        }
        
        let lengths: Vec<usize> = sentences.iter()
            .map(|s| s.split_whitespace().count())
            .collect();
        
        let mean_length = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let variance = lengths.iter()
            .map(|&len| (len as f64 - mean_length).powi(2))
            .sum::<f64>() / lengths.len() as f64;
        
        // Lower variance indicates better flow
        let flow_score = 1.0 / (1.0 + variance);
        Ok(flow_score)
    }
    
    fn calculate_logical_structure(&self, text: &str) -> Result<f64> {
        // Simplified logical structure based on discourse markers
        let logical_markers = ["first", "second", "third", "finally", "in conclusion", "therefore", "however", "moreover"];
        
        let marker_count = logical_markers.iter()
            .map(|marker| text.to_lowercase().matches(marker).count())
            .sum::<usize>();
        
        let sentence_count = self.count_sentences(text);
        let structure_score = if sentence_count > 0 {
            (marker_count as f64 / sentence_count as f64).min(1.0)
        } else {
            0.0
        };
        
        Ok(structure_score)
    }
}