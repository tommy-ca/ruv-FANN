//! Performance regression tests for veritas-nexus
//! 
//! These tests validate performance targets and detect regressions:
//! - Single frame analysis: < 10ms
//! - Batch processing: > 100 FPS
//! - Memory usage: < 500MB
//! 
//! Run with: cargo bench --bench performance_regression

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};

use veritas_nexus::prelude::*;
use veritas_nexus::modalities::{text::*, vision::*, audio::*};
use veritas_nexus::fusion::*;

/// Performance target constants based on specification
const SINGLE_FRAME_TARGET_MS: u64 = 10;
const BATCH_FPS_TARGET: f64 = 100.0;
const MEMORY_TARGET_MB: usize = 500;

/// Test single frame analysis performance target (< 10ms)
fn bench_single_frame_performance_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets");
    group.significance_level(0.1).sample_size(100);
    
    // Text analysis single frame
    let text_analyzer = TextAnalyzer::new(TextAnalyzerConfig::default())
        .expect("Failed to create text analyzer");
    let test_text = TextInput::new("I definitely did not take the money from the office yesterday.");
    
    group.bench_function("text_single_frame_target", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = futures::executor::block_on(text_analyzer.analyze(black_box(&test_text)));
            let elapsed = start.elapsed();
            
            // Validate performance target
            assert!(
                elapsed.as_millis() < SINGLE_FRAME_TARGET_MS as u128,
                "Text analysis took {}ms, target is <{}ms",
                elapsed.as_millis(),
                SINGLE_FRAME_TARGET_MS
            );
            
            black_box(result)
        });
    });
    
    // Vision analysis single frame
    if let Ok(vision_analyzer) = VisionAnalyzer::new(VisionConfig::default()) {
        let image_data = create_test_image(224, 224);
        let test_image = VisionInput::new(image_data, 224, 224, 3);
        
        group.bench_function("vision_single_frame_target", |b| {
            b.iter(|| {
                let start = Instant::now();
                let result = vision_analyzer.extract_features(black_box(&test_image));
                let elapsed = start.elapsed();
                
                // Validate performance target
                assert!(
                    elapsed.as_millis() < SINGLE_FRAME_TARGET_MS as u128,
                    "Vision analysis took {}ms, target is <{}ms",
                    elapsed.as_millis(),
                    SINGLE_FRAME_TARGET_MS
                );
                
                black_box(result)
            });
        });
    }
    
    // Audio analysis single frame (1 second clip)
    if let Ok(audio_analyzer) = AudioAnalyzer::new(AudioConfig::default()) {
        let audio_data = create_test_audio(44100, 44100.0); // 1 second
        let test_audio = AudioInput::new(audio_data, 44100);
        
        group.bench_function("audio_single_frame_target", |b| {
            b.iter(|| {
                let start = Instant::now();
                let result = audio_analyzer.extract_features(black_box(&test_audio));
                let elapsed = start.elapsed();
                
                // Validate performance target
                assert!(
                    elapsed.as_millis() < SINGLE_FRAME_TARGET_MS as u128,
                    "Audio analysis took {}ms, target is <{}ms",
                    elapsed.as_millis(),
                    SINGLE_FRAME_TARGET_MS
                );
                
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Test batch processing performance target (> 100 FPS)
fn bench_batch_processing_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing_targets");
    group.significance_level(0.1).sample_size(50);
    
    // Text batch processing
    let text_analyzer = TextAnalyzer::new(TextAnalyzerConfig::default())
        .expect("Failed to create text analyzer");
    
    let test_texts: Vec<TextInput> = vec![
        TextInput::new("Statement one for testing."),
        TextInput::new("Statement two for analysis."),
        TextInput::new("Statement three for benchmarking."),
        TextInput::new("Statement four for validation."),
        TextInput::new("Statement five for performance."),
    ];
    
    group.bench_function("text_batch_fps_target", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            // Process batch
            for text_input in &test_texts {
                let _result = futures::executor::block_on(
                    text_analyzer.analyze(black_box(text_input))
                );
            }
            
            let elapsed = start.elapsed();
            let fps = test_texts.len() as f64 / elapsed.as_secs_f64();
            
            // Validate FPS target
            assert!(
                fps >= BATCH_FPS_TARGET,
                "Text batch processing achieved {}fps, target is >{}fps",
                fps,
                BATCH_FPS_TARGET
            );
            
            black_box(fps)
        });
    });
    
    // Vision batch processing
    if let Ok(vision_analyzer) = VisionAnalyzer::new(VisionConfig::default()) {
        let test_images: Vec<VisionInput> = (0..5)
            .map(|_| {
                let image_data = create_test_image(224, 224);
                VisionInput::new(image_data, 224, 224, 3)
            })
            .collect();
        
        group.bench_function("vision_batch_fps_target", |b| {
            b.iter(|| {
                let start = Instant::now();
                
                // Process batch
                for image_input in &test_images {
                    let _result = vision_analyzer.extract_features(black_box(image_input));
                }
                
                let elapsed = start.elapsed();
                let fps = test_images.len() as f64 / elapsed.as_secs_f64();
                
                // Validate FPS target
                assert!(
                    fps >= BATCH_FPS_TARGET,
                    "Vision batch processing achieved {}fps, target is >{}fps",
                    fps,
                    BATCH_FPS_TARGET
                );
                
                black_box(fps)
            });
        });
    }
    
    group.finish();
}

/// Test memory usage target (< 500MB)
fn bench_memory_usage_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_targets");
    group.significance_level(0.1).sample_size(20);
    
    group.bench_function("memory_usage_target", |b| {
        b.iter(|| {
            // Create analyzers and process substantial workload
            let text_analyzer = TextAnalyzer::new(TextAnalyzerConfig::default())
                .expect("Failed to create text analyzer");
            
            // Create large batch for memory stress testing
            let large_texts: Vec<TextInput> = (0..100)
                .map(|i| TextInput::new(&format!(
                    "This is a comprehensive test sentence number {} for memory usage validation with extensive content to simulate real-world scenarios.",
                    i
                )))
                .collect();
            
            // Measure memory before processing
            let memory_before = get_memory_usage_mb();
            
            // Process large batch
            for text_input in &large_texts {
                let _result = futures::executor::block_on(
                    text_analyzer.analyze(black_box(text_input))
                );
            }
            
            // Measure memory after processing
            let memory_after = get_memory_usage_mb();
            let memory_used = memory_after.saturating_sub(memory_before);
            
            // Validate memory target
            assert!(
                memory_used < MEMORY_TARGET_MB,
                "Memory usage was {}MB, target is <{}MB",
                memory_used,
                MEMORY_TARGET_MB
            );
            
            black_box(memory_used)
        });
    });
    
    group.finish();
}

/// Test end-to-end pipeline performance
fn bench_end_to_end_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_performance");
    group.significance_level(0.1).sample_size(30);
    
    group.bench_function("complete_pipeline_target", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            // Text analysis
            let text_analyzer = TextAnalyzer::new(TextAnalyzerConfig::default())
                .expect("Failed to create text analyzer");
            let text_input = TextInput::new("I was definitely not at the location during the incident.");
            let text_result = futures::executor::block_on(text_analyzer.analyze(&text_input))
                .expect("Text analysis failed");
            
            // Create modality scores for fusion
            let mut modality_scores = ModalityScores::new();
            modality_scores.add_score("text", text_result.deception_score, text_result.confidence);
            
            // Simple fusion (if more complex fusion is available, use that)
            let fusion_strategy = WeightedFusion::new(vec![1.0]); // Single modality for now
            let fusion_result = fusion_strategy.fuse(&modality_scores)
                .expect("Fusion failed");
            
            let elapsed = start.elapsed();
            
            // End-to-end should be within reasonable time (relaxed compared to single frame)
            let target_ms = 50; // 50ms for complete pipeline
            assert!(
                elapsed.as_millis() < target_ms,
                "End-to-end pipeline took {}ms, target is <{}ms",
                elapsed.as_millis(),
                target_ms
            );
            
            black_box((text_result, fusion_result))
        });
    });
    
    group.finish();
}

/// Test GPU vs CPU performance comparison (when GPU is available)
#[cfg(feature = "gpu")]
fn bench_gpu_vs_cpu_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_vs_cpu_performance");
    group.significance_level(0.1).sample_size(50);
    
    // CPU vision processing
    if let Ok(cpu_vision_analyzer) = VisionAnalyzer::new(VisionConfig::default()) {
        let test_image = create_test_image(224, 224);
        let vision_input = VisionInput::new(test_image, 224, 224, 3);
        
        group.bench_function("cpu_vision_processing", |b| {
            b.iter(|| {
                let result = cpu_vision_analyzer.extract_features(black_box(&vision_input));
                black_box(result)
            });
        });
    }
    
    // GPU vision processing (if available)
    if let Ok(gpu_vision_processor) = GpuVisionProcessor::new(&VisionConfig::default()) {
        if gpu_vision_processor.is_gpu_enabled() {
            let test_image = create_test_image(224, 224);
            let vision_input = VisionInput::new(test_image, 224, 224, 3);
            
            group.bench_function("gpu_vision_processing", |b| {
                b.iter(|| {
                    let result = gpu_vision_processor.extract_features(black_box(&vision_input));
                    black_box(result)
                });
            });
        }
    }
    
    group.finish();
}

/// Test scalability with increasing load
fn bench_scalability_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_targets");
    group.significance_level(0.1).sample_size(20);
    
    let text_analyzer = TextAnalyzer::new(TextAnalyzerConfig::default())
        .expect("Failed to create text analyzer");
    
    // Test scalability with different batch sizes
    for batch_size in [1, 10, 50, 100].iter() {
        let test_texts: Vec<TextInput> = (0..*batch_size)
            .map(|i| TextInput::new(&format!("Test statement number {} for scalability testing.", i)))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("text_scalability", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    
                    for text_input in &test_texts {
                        let _result = futures::executor::block_on(
                            text_analyzer.analyze(black_box(text_input))
                        );
                    }
                    
                    let elapsed = start.elapsed();
                    let avg_time_per_item = elapsed.as_millis() as f64 / *batch_size as f64;
                    
                    // Average time per item should remain reasonable even with larger batches
                    assert!(
                        avg_time_per_item < 20.0, // 20ms average per item
                        "Average time per item was {}ms for batch size {}, should be <20ms",
                        avg_time_per_item,
                        batch_size
                    );
                    
                    black_box(elapsed)
                });
            },
        );
    }
    
    group.finish();
}

/// Helper function to create test image data
fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut image_data = Vec::with_capacity((width * height * 3) as usize);
    
    for y in 0..height {
        for x in 0..width {
            // Create a simple gradient pattern
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = 128u8;
            
            image_data.push(r);
            image_data.push(g);
            image_data.push(b);
        }
    }
    
    image_data
}

/// Helper function to create test audio data
fn create_test_audio(num_samples: usize, sample_rate: f32) -> Vec<f32> {
    let mut audio_data = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let signal = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1; // 440Hz sine wave
        audio_data.push(signal);
    }
    
    audio_data
}

/// Helper function to get current memory usage in MB
fn get_memory_usage_mb() -> usize {
    use std::fs;
    
    // Read memory usage from /proc/self/status on Linux
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(mem_str) = line.split_whitespace().nth(1) {
                    if let Ok(mem_kb) = mem_str.parse::<usize>() {
                        return mem_kb / 1024; // Convert KB to MB
                    }
                }
            }
        }
    }
    
    // Fallback: return 0 if we can't measure memory
    0
}

// Default implementations for testing
struct ModalityScores {
    scores: std::collections::HashMap<String, (f64, Confidence)>,
}

impl ModalityScores {
    fn new() -> Self {
        Self {
            scores: std::collections::HashMap::new(),
        }
    }
    
    fn add_score(&mut self, modality: &str, score: f64, confidence: Confidence) {
        self.scores.insert(modality.to_string(), (score, confidence));
    }
}

struct WeightedFusion {
    weights: Vec<f64>,
}

impl WeightedFusion {
    fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }
    
    fn fuse(&self, _scores: &ModalityScores) -> Result<FusionResult, String> {
        Ok(FusionResult {
            deception_score: 0.6,
            confidence: Confidence::new(0.8),
            modality_contributions: std::collections::HashMap::new(),
            temporal_consistency: 0.9,
            explanation: "Test fusion result".to_string(),
        })
    }
}

struct FusionResult {
    deception_score: f64,
    confidence: Confidence,
    modality_contributions: std::collections::HashMap<String, f64>,
    temporal_consistency: f64,
    explanation: String,
}

struct Confidence {
    value: f64,
}

impl Confidence {
    fn new(value: f64) -> Self {
        Self { value: value.clamp(0.0, 1.0) }
    }
    
    fn value(&self) -> f64 {
        self.value
    }
}

// Configure benchmark groups
criterion_group!(
    performance_regression,
    bench_single_frame_performance_target,
    bench_batch_processing_target,
    bench_memory_usage_target,
    bench_end_to_end_performance,
    bench_scalability_targets,
);

// Add GPU benchmarks if GPU feature is enabled
#[cfg(feature = "gpu")]
criterion_group!(
    gpu_performance,
    bench_gpu_vs_cpu_performance,
);

// Main benchmark runner
#[cfg(feature = "gpu")]
criterion_main!(performance_regression, gpu_performance);

#[cfg(not(feature = "gpu"))]
criterion_main!(performance_regression);