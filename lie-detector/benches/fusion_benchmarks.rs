//! Benchmarks for multi-modal fusion performance
//! 
//! Run with: cargo bench --bench fusion_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

// Import fusion module components
use veritas_nexus::fusion::{
    FusionStrategy, AttentionFusion, TemporalAlignment, FusionResult,
    WeightedFusion, AdaptiveFusion, HierarchicalFusion,
};
use veritas_nexus::types::{ModalityScores, Confidence, Timestamp};

/// Benchmark weighted fusion strategy
fn bench_weighted_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_fusion");
    
    let fusion_strategy = WeightedFusion::new(vec![0.4, 0.35, 0.25]); // Text, Vision, Audio weights
    
    for num_modalities in [2, 3, 4, 5].iter() {
        let modality_scores = create_test_modality_scores(*num_modalities);
        
        group.throughput(Throughput::Elements(*num_modalities as u64));
        group.bench_with_input(
            BenchmarkId::new("fuse_scores", num_modalities),
            num_modalities,
            |b, _| {
                b.iter(|| {
                    black_box(fusion_strategy.fuse(black_box(&modality_scores)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark attention-based fusion
fn bench_attention_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_fusion");
    
    let config = AttentionFusionConfig::default();
    let mut fusion_strategy = match AttentionFusion::new(&config) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Skipping attention fusion benchmark - strategy creation failed");
            return;
        }
    };
    
    for num_modalities in [2, 3, 4, 5].iter() {
        let modality_scores = create_test_modality_scores(*num_modalities);
        
        group.throughput(Throughput::Elements(*num_modalities as u64));
        group.bench_with_input(
            BenchmarkId::new("attention_fuse", num_modalities),
            num_modalities,
            |b, _| {
                b.iter(|| {
                    black_box(fusion_strategy.fuse_with_attention(black_box(&modality_scores)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark adaptive fusion strategy
fn bench_adaptive_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_fusion");
    
    let config = AdaptiveFusionConfig::default();
    let mut fusion_strategy = match AdaptiveFusion::new(&config) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Skipping adaptive fusion benchmark - strategy creation failed");
            return;
        }
    };
    
    for num_samples in [10, 50, 100, 200].iter() {
        // Create multiple samples for adaptation
        let mut all_scores = Vec::new();
        for _ in 0..*num_samples {
            all_scores.push(create_test_modality_scores(3));
        }
        
        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("adaptive_fuse", num_samples),
            num_samples,
            |b, _| {
                b.iter(|| {
                    for scores in &all_scores {
                        black_box(fusion_strategy.fuse_adaptive(black_box(scores)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark hierarchical fusion
fn bench_hierarchical_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_fusion");
    
    let config = HierarchicalFusionConfig::default();
    let fusion_strategy = match HierarchicalFusion::new(&config) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Skipping hierarchical fusion benchmark - strategy creation failed");
            return;
        }
    };
    
    for hierarchy_depth in [2, 3, 4].iter() {
        let modality_scores = create_hierarchical_modality_scores(*hierarchy_depth);
        
        group.throughput(Throughput::Elements(*hierarchy_depth as u64));
        group.bench_with_input(
            BenchmarkId::new("hierarchical_fuse", hierarchy_depth),
            hierarchy_depth,
            |b, _| {
                b.iter(|| {
                    black_box(fusion_strategy.fuse_hierarchical(black_box(&modality_scores)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark temporal alignment
fn bench_temporal_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_alignment");
    
    let config = TemporalAlignmentConfig::default();
    let mut aligner = match TemporalAlignment::new(&config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping temporal alignment benchmark - aligner creation failed");
            return;
        }
    };
    
    for sequence_length in [10, 50, 100, 200].iter() {
        let temporal_sequences = create_temporal_sequences(*sequence_length);
        
        group.throughput(Throughput::Elements(*sequence_length as u64));
        group.bench_with_input(
            BenchmarkId::new("align_sequences", sequence_length),
            sequence_length,
            |b, _| {
                b.iter(|| {
                    black_box(aligner.align_sequences(black_box(&temporal_sequences)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark temporal window fusion
fn bench_temporal_window_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_window_fusion");
    
    let config = TemporalAlignmentConfig::default();
    let mut aligner = match TemporalAlignment::new(&config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping temporal window fusion benchmark - aligner creation failed");
            return;
        }
    };
    
    for window_size in [5, 10, 20, 50].iter() {
        let temporal_sequences = create_temporal_sequences(100); // Fixed sequence length
        
        group.throughput(Throughput::Elements(*window_size as u64));
        group.bench_with_input(
            BenchmarkId::new("window_fusion", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    black_box(aligner.fuse_temporal_window(
                        black_box(&temporal_sequences),
                        *window_size
                    ))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark confidence aggregation
fn bench_confidence_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("confidence_aggregation");
    
    let fusion_strategy = WeightedFusion::new(vec![0.4, 0.35, 0.25]);
    
    for num_scores in [10, 50, 100, 500].iter() {
        let mut all_scores = Vec::new();
        for _ in 0..*num_scores {
            all_scores.push(create_test_modality_scores(3));
        }
        
        group.throughput(Throughput::Elements(*num_scores as u64));
        group.bench_with_input(
            BenchmarkId::new("aggregate_confidence", num_scores),
            num_scores,
            |b, _| {
                b.iter(|| {
                    let results: Vec<FusionResult> = all_scores
                        .iter()
                        .map(|scores| fusion_strategy.fuse(scores).unwrap())
                        .collect();
                    black_box(fusion_strategy.aggregate_confidence(black_box(&results)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark ensemble fusion
fn bench_ensemble_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_fusion");
    
    // Create multiple fusion strategies
    let strategies: Vec<Box<dyn FusionStrategy>> = vec![
        Box::new(WeightedFusion::new(vec![0.4, 0.35, 0.25])),
        Box::new(WeightedFusion::new(vec![0.33, 0.33, 0.34])),
        Box::new(WeightedFusion::new(vec![0.5, 0.3, 0.2])),
    ];
    
    for num_strategies in [2, 3, 5].iter() {
        let selected_strategies = &strategies[0..*num_strategies.min(&strategies.len())];
        let modality_scores = create_test_modality_scores(3);
        
        group.throughput(Throughput::Elements(*num_strategies as u64));
        group.bench_with_input(
            BenchmarkId::new("ensemble_fuse", num_strategies),
            num_strategies,
            |b, _| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for strategy in selected_strategies {
                        if let Ok(result) = strategy.fuse(black_box(&modality_scores)) {
                            results.push(result);
                        }
                    }
                    // Ensemble the results
                    black_box(ensemble_results(black_box(&results)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark real-time fusion performance
fn bench_realtime_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("realtime_fusion");
    group.measurement_time(Duration::from_secs(10));
    
    let fusion_strategy = WeightedFusion::new(vec![0.4, 0.35, 0.25]);
    
    // Simulate real-time scenario with strict timing requirements
    let target_fps = vec![30, 60, 120]; // Target frame rates
    
    for fps in target_fps.iter() {
        let modality_scores = create_test_modality_scores(3);
        let target_duration = Duration::from_nanos(1_000_000_000 / fps); // Target duration per frame
        
        group.bench_with_input(
            BenchmarkId::new("realtime_fuse", fps),
            fps,
            |b, _| {
                b.iter(|| {
                    let start = std::time::Instant::now();
                    let result = fusion_strategy.fuse(black_box(&modality_scores));
                    let elapsed = start.elapsed();
                    
                    // Verify we meet real-time requirements
                    assert!(elapsed < target_duration, 
                           "Fusion took {} ns, target was {} ns", 
                           elapsed.as_nanos(), target_duration.as_nanos());
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage during fusion
fn bench_fusion_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_memory_usage");
    
    let fusion_strategy = WeightedFusion::new(vec![0.4, 0.35, 0.25]);
    
    // Test with large batches to stress memory usage
    for batch_size in [100, 500, 1000].iter() {
        let mut all_scores = Vec::new();
        for _ in 0..*batch_size {
            all_scores.push(create_test_modality_scores(3));
        }
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_fusion", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<FusionResult> = all_scores
                        .iter()
                        .map(|scores| fusion_strategy.fuse(black_box(scores)).unwrap())
                        .collect();
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

/// Create test modality scores for benchmarking
fn create_test_modality_scores(num_modalities: usize) -> ModalityScores {
    let mut scores = ModalityScores::new();
    
    for i in 0..num_modalities {
        let modality_name = match i {
            0 => "text",
            1 => "vision", 
            2 => "audio",
            3 => "physiological",
            _ => &format!("modality_{}", i),
        };
        
        let score = 0.3 + (i as f64 * 0.2); // Varying scores
        let confidence = Confidence::new(0.7 + (i as f64 * 0.1));
        
        scores.add_score(modality_name, score, confidence);
    }
    
    scores
}

/// Create hierarchical modality scores for testing
fn create_hierarchical_modality_scores(depth: usize) -> Vec<ModalityScores> {
    let mut hierarchical_scores = Vec::new();
    
    for level in 0..depth {
        let mut scores = ModalityScores::new();
        
        // Each level has different modalities or sub-features
        for i in 0..3 {
            let modality_name = format!("level_{}_{}", level, i);
            let score = 0.2 + (level as f64 * 0.2) + (i as f64 * 0.1);
            let confidence = Confidence::new(0.6 + (level as f64 * 0.1));
            
            scores.add_score(&modality_name, score, confidence);
        }
        
        hierarchical_scores.push(scores);
    }
    
    hierarchical_scores
}

/// Create temporal sequences for alignment testing
fn create_temporal_sequences(length: usize) -> Vec<(Timestamp, ModalityScores)> {
    let mut sequences = Vec::new();
    let base_timestamp = Timestamp::now();
    
    for i in 0..length {
        let timestamp = base_timestamp + Duration::from_millis(i as u64 * 33); // ~30 FPS
        let scores = create_test_modality_scores(3);
        sequences.push((timestamp, scores));
    }
    
    sequences
}

/// Simple ensemble function for testing
fn ensemble_results(results: &[FusionResult]) -> FusionResult {
    if results.is_empty() {
        return FusionResult::default();
    }
    
    let average_score = results.iter().map(|r| r.deception_score).sum::<f64>() / results.len() as f64;
    let average_confidence = results.iter().map(|r| r.confidence.value()).sum::<f64>() / results.len() as f64;
    
    FusionResult {
        deception_score: average_score,
        confidence: Confidence::new(average_confidence),
        modality_contributions: results[0].modality_contributions.clone(), // Simplified
        temporal_consistency: results.iter().map(|r| r.temporal_consistency).sum::<f64>() / results.len() as f64,
        explanation: "Ensemble result".to_string(),
    }
}

// Default configurations for testing
struct AttentionFusionConfig {
    attention_dim: usize,
    num_heads: usize,
}

impl Default for AttentionFusionConfig {
    fn default() -> Self {
        Self {
            attention_dim: 64,
            num_heads: 8,
        }
    }
}

struct AdaptiveFusionConfig {
    learning_rate: f64,
    adaptation_window: usize,
}

impl Default for AdaptiveFusionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            adaptation_window: 50,
        }
    }
}

struct HierarchicalFusionConfig {
    levels: usize,
    aggregation_method: String,
}

impl Default for HierarchicalFusionConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            aggregation_method: "weighted_average".to_string(),
        }
    }
}

struct TemporalAlignmentConfig {
    window_size: usize,
    alignment_threshold: f64,
}

impl Default for TemporalAlignmentConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            alignment_threshold: 0.1,
        }
    }
}

// Configure benchmark groups
criterion_group!(
    benches,
    bench_weighted_fusion,
    bench_attention_fusion,
    bench_adaptive_fusion,
    bench_hierarchical_fusion,
    bench_temporal_alignment,
    bench_temporal_window_fusion,
    bench_confidence_aggregation,
    bench_ensemble_fusion,
    bench_realtime_fusion,
    bench_fusion_memory_usage
);

criterion_main!(benches);