//! Benchmarks for text analysis performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use veritas_nexus::prelude::*;
use veritas_nexus::modalities::text::*;

fn create_analyzer() -> TextAnalyzer<f64> {
    let config = TextAnalyzerConfig::default();
    TextAnalyzer::new(config).expect("Failed to create analyzer")
}

fn create_test_texts() -> Vec<String> {
    vec![
        "I definitely did not take the money.".to_string(),
        "Well, I kind of think maybe I might have possibly seen something like that.".to_string(),
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.".to_string(),
        "I am very happy and excited about this wonderful opportunity to work with such an amazing team of brilliant professionals.".to_string(),
        "Yesterday I was there, but today I am here. Tomorrow I will be somewhere else. Before that happened, I had already left.".to_string(),
        "Um, well, you know, I was like, actually, basically thinking that, you know, maybe it was, like, I mean, you know what I mean?".to_string(),
        "The multifaceted implications of the aforementioned circumstances necessitate a comprehensive evaluation of the underlying frameworks.".to_string(),
    ]
}

fn bench_single_text_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let analyzer = create_analyzer();
    let test_text = "I definitely never took anything from anywhere at any time.";
    
    c.bench_function("single_text_analysis", |b| {
        b.to_async(&rt).iter(|| async {
            let input = TextInput::new(black_box(test_text));
            let result = analyzer.analyze(&input).await;
            black_box(result)
        })
    });
}

fn bench_batch_text_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let analyzer = create_analyzer();
    let test_texts = create_test_texts();
    
    for batch_size in [1, 5, 10, 20].iter() {
        c.bench_with_input(
            BenchmarkId::new("batch_text_analysis", batch_size),
            batch_size,
            |b, &size| {
                let texts = &test_texts[..size.min(test_texts.len())];
                b.to_async(&rt).iter(|| async {
                    let mut results = Vec::new();
                    for text in texts {
                        let input = TextInput::new(black_box(text));
                        let result = analyzer.analyze(&input).await;
                        results.push(result);
                    }
                    black_box(results)
                })
            },
        );
    }
}

fn bench_feature_extraction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    let test_text = "This is a comprehensive test sentence for benchmarking feature extraction performance with multiple linguistic indicators.";
    
    c.bench_function("feature_extraction", |b| {
        b.to_async(&rt).iter(|| async {
            let result = analyzer.extract_features(black_box(test_text), Language::English).await;
            black_box(result)
        })
    });
}

fn bench_deception_pattern_detection(c: &mut Criterion) {
    let weights = FeatureWeights::<f64>::default();
    let detector = DeceptionPatternDetector::new(&weights).expect("Failed to create detector");
    
    // Create mock features for benchmarking
    let features = LinguisticFeatures {
        lexical_features: vec![15.0, 75.0, 5.2, 0.75, 3.0, 2.0, 1.0, 2.0, 3.0],
        syntactic_features: vec![4.0, 3.75, 0.35, 0.25, 0.15, 0.08, 0.17],
        semantic_features: vec![0.68, 0.15, 0.22],
        pragmatic_features: vec![0.12, 2.0, 1.0],
        discourse_features: vec![3.0, 4.0],
        feature_names: vec!["benchmark".to_string()],
    };
    
    c.bench_function("deception_pattern_detection", |b| {
        b.iter(|| {
            let patterns = detector.analyze_patterns(black_box(&features)).unwrap();
            let probability = detector.calculate_probability(black_box(&features), &patterns).unwrap();
            let confidence = detector.calculate_confidence(black_box(&features), &patterns).unwrap();
            black_box((patterns, probability, confidence))
        })
    });
}

fn bench_text_preprocessing(c: &mut Criterion) {
    let config = PreprocessingConfig {
        normalize_unicode: true,
        lowercase: true,
        remove_punctuation: true,
        remove_stopwords: true,
        stem_words: true,
        ..Default::default()
    };
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let test_texts = create_test_texts();
    
    for (i, text) in test_texts.iter().enumerate() {
        c.bench_with_input(
            BenchmarkId::new("text_preprocessing", i),
            text,
            |b, text| {
                b.iter(|| {
                    let result = analyzer.preprocess(black_box(text), Language::English, &config);
                    black_box(result)
                })
            },
        );
    }
}

fn bench_language_detection(c: &mut Criterion) {
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let multilingual_texts = vec![
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Guten Tag, wie geht es Ihnen heute?",
        "Ciao, come stai oggi?",
        "Привет, как дела сегодня?",
        "你好，你今天怎么样？",
        "こんにちは、今日はいかがですか？",
    ];
    
    c.bench_function("language_detection", |b| {
        b.iter(|| {
            for text in &multilingual_texts {
                let result = analyzer.detect_language(black_box(text));
                black_box(result);
            }
        })
    });
}

fn bench_sentiment_analysis(c: &mut Criterion) {
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let sentiment_texts = vec![
        "I am extremely happy and excited about this wonderful opportunity!",
        "This is the worst thing that has ever happened to me, I hate everything.",
        "The weather is okay today, nothing special.",
        "I love this amazing product, it's absolutely fantastic and perfect!",
        "I'm disappointed and frustrated with this terrible service.",
    ];
    
    c.bench_function("sentiment_analysis", |b| {
        b.iter(|| {
            for text in &sentiment_texts {
                let result = analyzer.analyze_sentiment(black_box(text), Language::English);
                black_box(result);
            }
        })
    });
}

fn bench_complexity_calculation(c: &mut Criterion) {
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let complexity_texts = vec![
        "I did it.", // Very simple
        "The dog ran quickly through the park.", // Simple
        "The comprehensive analysis of multifaceted problems requires sophisticated approaches.", // Complex
        "The epistemological implications of phenomenological manifestations necessitate hermeneutic interpretation.", // Very complex
    ];
    
    c.bench_function("complexity_calculation", |b| {
        b.iter(|| {
            for text in &complexity_texts {
                let result = analyzer.calculate_complexity(black_box(text));
                black_box(result);
            }
        })
    });
}

fn bench_named_entity_extraction(c: &mut Criterion) {
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let entity_text = "John Smith from Acme Corp took $500 on January 15, 2024 at 3:30 PM from the New York office.";
    
    c.bench_function("named_entity_extraction", |b| {
        b.iter(|| {
            let result = analyzer.extract_entities(black_box(entity_text), Language::English);
            black_box(result)
        })
    });
}

fn bench_temporal_analysis(c: &mut Criterion) {
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let temporal_text = "Yesterday I was there, today I am here, tomorrow I will be elsewhere. Before that happened, after everything ended.";
    let tokens = analyzer.tokenize_and_analyze(temporal_text, Language::English).expect("Tokenization should succeed");
    
    c.bench_function("temporal_analysis", |b| {
        b.iter(|| {
            let result = analyzer.analyze_temporal_patterns(black_box(&tokens));
            black_box(result)
        })
    });
}

fn bench_cognitive_load_analysis(c: &mut Criterion) {
    let config = PreprocessingConfig::default();
    let analyzer = LinguisticAnalyzer::<f64>::new(&config).expect("Failed to create analyzer");
    
    let cognitive_text = "Um, well, you know, I was like, actually, basically thinking that, you know, maybe it was, like, I mean, you know what I mean?";
    let tokens = analyzer.tokenize_and_analyze(cognitive_text, Language::English).expect("Tokenization should succeed");
    
    c.bench_function("cognitive_load_analysis", |b| {
        b.iter(|| {
            let result = analyzer.analyze_cognitive_load(black_box(&tokens));
            black_box(result)
        })
    });
}

fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("memory_usage_large_text", |b| {
        b.to_async(&rt).iter(|| async {
            let analyzer = create_analyzer();
            
            // Large text to test memory usage
            let large_text = "This is a test sentence. ".repeat(1000);
            let input = TextInput::new(black_box(large_text));
            
            let result = analyzer.analyze(&input).await;
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    bench_single_text_analysis,
    bench_batch_text_analysis,
    bench_feature_extraction,
    bench_deception_pattern_detection,
    bench_text_preprocessing,
    bench_language_detection,
    bench_sentiment_analysis,
    bench_complexity_calculation,
    bench_named_entity_extraction,
    bench_temporal_analysis,
    bench_cognitive_load_analysis,
    bench_memory_usage
);

criterion_main!(benches);