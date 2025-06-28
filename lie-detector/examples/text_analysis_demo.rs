//! Demonstration of text analysis capabilities for deception detection

use veritas_nexus::prelude::*;
use veritas_nexus::modalities::text::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("=== Veritas Nexus Text Analysis Demo ===\n");
    
    // Create analyzer with custom configuration
    let config = TextAnalyzerConfig {
        bert_config: BertConfig {
            model_name: "bert-base-uncased".to_string(),
            max_sequence_length: 256,
            pooling_strategy: PoolingStrategy::ClsToken,
            use_attention_mask: true,
            device: "cpu".to_string(),
        },
        enable_sentiment: true,
        enable_ner: true,
        enable_caching: true,
        confidence_threshold: 0.6,
        ..Default::default()
    };
    
    let analyzer = TextAnalyzer::<f64>::new(config)?;
    
    // Demo 1: Basic deception detection
    println!("📝 Demo 1: Basic Deception Detection");
    println!("=====================================");
    
    let test_cases = vec![
        ("Truthful", "I took the money from the drawer at 3 PM yesterday. I needed it for groceries and planned to return it today."),
        ("Deceptive", "Well, I kind of think maybe I might have possibly seen something like that, but I'm not really sure, you know?"),
        ("Neutral", "The weather is nice today. I had lunch at noon."),
    ];
    
    for (label, text) in test_cases {
        println!("\n🔍 Analyzing {} text:", label);
        println!("Text: \"{}\"", text);
        
        let input = TextInput::new(text)
            .with_context("demo_type".to_string(), serde_json::Value::String("basic".to_string()))
            .with_timestamp(std::time::SystemTime::now());
        
        let result = analyzer.analyze(&input).await?;
        
        println!("📊 Results:");
        println!("  • Deception Probability: {:.2}%", result.probability() * 100.0);
        println!("  • Confidence: {:.2}%", result.confidence() * 100.0);
        println!("  • Language: {:?}", result.language);
        
        if let Some(sentiment) = &result.sentiment {
            println!("  • Sentiment: {:?} (confidence: {:.2})", 
                sentiment.dominant_sentiment(), 
                sentiment.confidence
            );
        }
        
        // Show top deception indicators
        println!("  • Top Indicators:");
        for (i, feature) in result.feature_contributions.iter().take(3).enumerate() {
            println!("    {}. {} (weight: {:.3})", i+1, feature.name, feature.weight);
        }
        
        println!("  • Processing Time: {}ms", result.performance.processing_time_ms);
    }
    
    // Demo 2: Detailed linguistic analysis
    println!("\n\n📊 Demo 2: Detailed Linguistic Analysis");
    println!("=======================================");
    
    let complex_text = "I definitely never took anything from anywhere at any time. \
                       Maybe someone else might have possibly done something, but I wasn't there. \
                       Yesterday I was working, today I'm here, tomorrow I'll be somewhere else.";
    
    let input = TextInput::new(complex_text)
        .with_language(Language::English)
        .with_speaker_id("speaker_001".to_string());
    
    let result = analyzer.analyze(&input).await?;
    
    println!("Text: \"{}\"", complex_text);
    println!("\n📈 Comprehensive Analysis:");
    
    // Deception patterns
    println!("\n🎯 Deception Patterns:");
    println!("  • Hedging Frequency: {:.3}", result.deception_patterns.hedging_frequency);
    println!("  • Negation Patterns: {:.3}", result.deception_patterns.negation_patterns);
    println!("  • Temporal References: {:.3}", result.deception_patterns.temporal_references);
    println!("  • Self References: {:.3}", result.deception_patterns.self_references);
    println!("  • Certainty Markers: {:.3}", result.deception_patterns.certainty_markers);
    println!("  • Detail Level: {:.3}", result.deception_patterns.detail_level);
    println!("  • Emotional Consistency: {:.3}", result.deception_patterns.emotional_consistency);
    
    // Complexity metrics
    println!("\n📚 Text Complexity:");
    println!("  • Flesch-Kincaid Grade: {:.1}", result.complexity.flesch_kincaid_grade);
    println!("  • Reading Ease: {:.1}", result.complexity.flesch_reading_ease);
    println!("  • Avg Sentence Length: {:.1} words", result.complexity.average_sentence_length);
    println!("  • Avg Word Length: {:.1} chars", result.complexity.average_word_length);
    println!("  • Unique Word Ratio: {:.3}", result.complexity.unique_word_ratio);
    
    // Temporal analysis
    println!("\n⏰ Temporal Patterns:");
    println!("  • Past References: {:.3}", result.temporal_patterns.past_references);
    println!("  • Present References: {:.3}", result.temporal_patterns.present_references);
    println!("  • Future References: {:.3}", result.temporal_patterns.future_references);
    println!("  • Timeline Clarity: {:.3}", result.temporal_patterns.timeline_clarity);
    
    // Cognitive load
    println!("\n🧠 Cognitive Load Indicators:");
    println!("  • Hesitation Markers: {:.3}", result.cognitive_load.hesitation_markers);
    println!("  • Filler Words: {:.3}", result.cognitive_load.filler_words);
    println!("  • Repetition Rate: {:.3}", result.cognitive_load.repetition_rate);
    println!("  • Processing Effort: {:.3}", result.cognitive_load.processing_effort);
    
    // Named entities
    if !result.named_entities.is_empty() {
        println!("\n🏷️  Named Entities:");
        for entity in &result.named_entities {
            println!("  • {} ({:?}) - relevance: {:.2}", 
                entity.text, 
                entity.entity_type, 
                entity.deception_relevance
            );
        }
    }
    
    // Demo 3: Batch analysis
    println!("\n\n📦 Demo 3: Batch Analysis");
    println!("=========================");
    
    let batch_texts = vec![
        "I absolutely never did anything wrong.",
        "Maybe I was there, but I don't really remember.",
        "I took the money because I needed it for my family.",
        "Well, you know, it's kind of complicated to explain.",
        "The incident occurred at approximately 2:30 PM on Tuesday.",
    ];
    
    println!("Analyzing {} texts in batch...", batch_texts.len());
    
    let start_time = std::time::Instant::now();
    let mut batch_results = Vec::new();
    
    for (i, text) in batch_texts.iter().enumerate() {
        let input = TextInput::new(*text)
            .with_context("batch_id".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
        
        let result = analyzer.analyze(&input).await?;
        batch_results.push((text, result));
    }
    
    let batch_time = start_time.elapsed();
    
    println!("\n📋 Batch Results:");
    for (i, (text, result)) in batch_results.iter().enumerate() {
        println!("  {}. Prob: {:.2}% | Conf: {:.2}% | \"{}\"", 
            i+1, 
            result.probability() * 100.0, 
            result.confidence() * 100.0,
            &text[..text.len().min(50)]
        );
    }
    
    println!("\n⚡ Performance Summary:");
    println!("  • Total Batch Time: {}ms", batch_time.as_millis());
    println!("  • Average per Text: {:.1}ms", batch_time.as_millis() as f64 / batch_texts.len() as f64);
    
    // Demo 4: Multi-language support
    println!("\n\n🌍 Demo 4: Multi-language Support");
    println!("=================================");
    
    let multilingual_texts = vec![
        ("English", "I definitely did not take the money."),
        ("Spanish", "Definitivamente no tomé el dinero."),
        ("French", "Je n'ai définitivement pas pris l'argent."),
    ];
    
    for (lang_name, text) in multilingual_texts {
        println!("\n🔍 Analyzing {} text:", lang_name);
        println!("Text: \"{}\"", text);
        
        let input = TextInput::new(text);
        
        match analyzer.analyze(&input).await {
            Ok(result) => {
                println!("  • Detected Language: {:?}", result.language);
                println!("  • Deception Probability: {:.2}%", result.probability() * 100.0);
                println!("  • Analysis Successful: ✅");
            }
            Err(e) => {
                println!("  • Error: {} ❌", e);
                println!("  • Language may not be fully supported");
            }
        }
    }
    
    // Demo 5: Performance and caching
    println!("\n\n⚡ Demo 5: Performance and Caching");
    println!("==================================");
    
    let test_text = "This text will be analyzed multiple times to demonstrate caching.";
    let input = TextInput::new(test_text);
    
    // First analysis (cache miss)
    let start = std::time::Instant::now();
    let _result1 = analyzer.analyze(&input).await?;
    let first_time = start.elapsed();
    
    // Second analysis (potential cache hit)
    let start = std::time::Instant::now();
    let _result2 = analyzer.analyze(&input).await?;
    let second_time = start.elapsed();
    
    println!("Performance comparison:");
    println!("  • First analysis: {}ms", first_time.as_millis());
    println!("  • Second analysis: {}ms", second_time.as_millis());
    
    let (cache_size, max_cache_size) = analyzer.cache_stats();
    println!("  • Cache usage: {}/{}", cache_size, max_cache_size);
    
    let stats = analyzer.performance_stats();
    println!("  • Total cache hits: {}", stats.cache_hits);
    println!("  • Total cache misses: {}", stats.cache_misses);
    println!("  • Cache hit ratio: {:.2}%", stats.cache_hit_ratio() * 100.0);
    
    // Demo 6: Explainability
    println!("\n\n🔍 Demo 6: Explainable AI");
    println!("=========================");
    
    let suspicious_text = "Well, I kind of think maybe I might have possibly been somewhere around that area, \
                          but I'm not really sure, you know, because it was sort of dark and confusing.";
    
    let input = TextInput::new(suspicious_text);
    let result = analyzer.analyze(&input).await?;
    
    println!("Text: \"{}\"", suspicious_text);
    println!("\n🧐 Explanation:");
    println!("Overall Assessment: {:.2}% probability of deception (confidence: {:.2}%)", 
        result.probability() * 100.0, 
        result.confidence() * 100.0
    );
    
    println!("\n📋 Key Contributing Factors:");
    for (i, feature) in result.feature_contributions.iter().take(5).enumerate() {
        println!("  {}. {} = {:.3} (weight: {:.3})", 
            i+1, 
            feature.name, 
            feature.value, 
            feature.weight
        );
        println!("     {}", feature.description);
    }
    
    println!("\n🔬 Detailed Analysis:");
    println!("This text shows several indicators commonly associated with deception:");
    println!("  • High hedging frequency suggests uncertainty or evasion");
    println!("  • Multiple qualifiers ('kind of', 'maybe', 'possibly') indicate lack of commitment");
    println!("  • Vague spatial references ('around that area') reduce precision");
    println!("  • Blame shifting to external factors ('dark and confusing')");
    
    println!("\n✅ Analysis Complete!");
    println!("The Veritas Nexus text analysis system provides comprehensive linguistic");
    println!("analysis for deception detection with full explainability and performance metrics.");
    
    Ok(())
}

/// Helper function to create sample data for demonstrations
#[allow(dead_code)]
fn create_sample_data() -> Vec<(String, bool)> {
    // (text, is_deceptive)
    vec![
        ("I took the money because I needed it urgently.".to_string(), false),
        ("I didn't take any money, I was never even there.".to_string(), true),
        ("Maybe I was around that area, but I don't really remember.".to_string(), true),
        ("I clearly saw John take the money at 3 PM.".to_string(), false),
        ("Well, you know, it's kind of complicated to explain.".to_string(), true),
        ("The incident occurred exactly as I described.".to_string(), false),
        ("I might have possibly seen something like that.".to_string(), true),
        ("I was working in my office during that time.".to_string(), false),
    ]
}

/// Demonstrate custom feature weighting
#[allow(dead_code)]
async fn demo_custom_weighting() -> Result<()> {
    println!("\n🎛️  Custom Feature Weighting Demo");
    println!("=================================");
    
    // Create analyzer with custom weights
    let custom_weights = FeatureWeights {
        linguistic_complexity: 1.2,
        emotional_indicators: 0.8,
        temporal_patterns: 1.5,
        semantic_coherence: 1.0,
        syntactic_patterns: 0.9,
        sentiment_consistency: 1.1,
        uncertainty_markers: 1.8,
        cognitive_load: 1.3,
    };
    
    let config = TextAnalyzerConfig {
        feature_weights: custom_weights,
        ..Default::default()
    };
    
    let analyzer = TextAnalyzer::<f64>::new(config)?;
    
    let test_text = "I'm not really sure if I maybe possibly might have seen something.";
    let input = TextInput::new(test_text);
    let result = analyzer.analyze(&input).await?;
    
    println!("With custom weights emphasizing uncertainty markers:");
    println!("  • Deception Probability: {:.2}%", result.probability() * 100.0);
    println!("  • Uncertainty weight was increased to 1.8 (from 1.0)");
    
    Ok(())
}