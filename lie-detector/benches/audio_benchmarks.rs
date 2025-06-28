//! Benchmarks for audio analysis performance
//! 
//! Run with: cargo bench --bench audio_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

// Import the audio module components
use veritas_nexus::modalities::audio::{
    AudioAnalyzer, AudioConfig, AudioInput, VoiceAnalyzer, StressFeatureExtractor,
    MfccExtractor, PitchDetector,
};

/// Benchmark MFCC feature extraction for different sample rates
fn bench_mfcc_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("mfcc_extraction");
    
    for sample_rate in [16000, 22050, 44100].iter() {
        let duration_seconds = 1.0;
        let num_samples = (*sample_rate as f64 * duration_seconds) as usize;
        let audio_data = create_benchmark_audio(num_samples, *sample_rate as f32);
        
        let config = AudioConfig::default();
        let extractor = match MfccExtractor::new(&config) {
            Ok(e) => e,
            Err(_) => {
                eprintln!("Skipping MFCC benchmark for {} Hz - extractor creation failed", sample_rate);
                continue;
            }
        };
        
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("extract_mfcc", sample_rate),
            sample_rate,
            |b, _| {
                b.iter(|| {
                    black_box(extractor.extract_features(black_box(&audio_data)))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark pitch detection across different frequencies
fn bench_pitch_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pitch_detection");
    
    let sample_rate = 44100.0;
    let duration = 1.0;
    let num_samples = (sample_rate * duration) as usize;
    
    for fundamental_freq in [80.0, 120.0, 200.0, 400.0].iter() {
        let audio_data = create_sine_wave(num_samples, sample_rate, *fundamental_freq);
        
        let config = AudioConfig::default();
        let detector = match PitchDetector::new(&config) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping pitch detection benchmark for {} Hz - detector creation failed", fundamental_freq);
                continue;
            }
        };
        
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("detect_pitch", fundamental_freq as u32),
            fundamental_freq,
            |b, _| {
                b.iter(|| {
                    black_box(detector.detect_pitch(black_box(&audio_data), sample_rate))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark stress feature extraction
fn bench_stress_feature_extraction(c: &mut Criterion) {
    let config = AudioConfig::default();
    let extractor = match StressFeatureExtractor::new(&config) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping stress feature benchmark - extractor creation failed");
            return;
        }
    };
    
    let mut group = c.benchmark_group("stress_feature_extraction");
    
    for duration in [0.5, 1.0, 2.0, 5.0].iter() {
        let sample_rate = 44100.0;
        let num_samples = (sample_rate * duration) as usize;
        let audio_data = create_benchmark_audio(num_samples, sample_rate);
        
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("extract_stress_features", (duration * 1000.0) as u32),
            duration,
            |b, _| {
                b.iter(|| {
                    black_box(extractor.extract_features(black_box(&audio_data), sample_rate))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark voice activity detection
fn bench_voice_activity_detection(c: &mut Criterion) {
    let config = AudioConfig::default();
    let analyzer = match VoiceAnalyzer::new(&config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping voice activity detection benchmark - analyzer creation failed");
            return;
        }
    };
    
    let mut group = c.benchmark_group("voice_activity_detection");
    
    let sample_rate = 44100.0;
    let duration = 2.0;
    let num_samples = (sample_rate * duration) as usize;
    
    // Create audio with periods of silence and speech
    let mut audio_data = vec![0.0; num_samples];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        if t > 0.5 && t < 1.5 {
            // Add voice-like signal
            audio_data[i] = (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.1
                + (2.0 * std::f32::consts::PI * 600.0 * t).sin() * 0.05
                + (2.0 * std::f32::consts::PI * 1200.0 * t).sin() * 0.02;
        }
    }
    
    group.bench_function("detect_voice_activity", |b| {
        b.iter(|| {
            black_box(analyzer.detect_voice_activity(black_box(&audio_data), sample_rate))
        });
    });
    
    group.finish();
}

/// Benchmark complete audio analysis pipeline
fn bench_complete_audio_analysis(c: &mut Criterion) {
    let config = AudioConfig::default();
    let analyzer = match AudioAnalyzer::new(config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping complete audio analysis benchmark - analyzer creation failed");
            return;
        }
    };
    
    let mut group = c.benchmark_group("complete_audio_analysis");
    
    for duration in [1.0, 3.0, 5.0].iter() {
        let sample_rate = 44100.0;
        let num_samples = (sample_rate * duration) as usize;
        let audio_data = create_benchmark_audio(num_samples, sample_rate);
        let input = AudioInput::new(audio_data, sample_rate as u32);
        
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("extract_features", (duration * 1000.0) as u32),
            duration,
            |b, _| {
                b.iter(|| {
                    if let Ok(features) = analyzer.extract_features(black_box(&input)) {
                        black_box(features);
                    }
                });
            },
        );
        
        // Benchmark analysis step separately if feature extraction succeeds
        if let Ok(features) = analyzer.extract_features(&input) {
            group.bench_with_input(
                BenchmarkId::new("analyze_features", (duration * 1000.0) as u32),
                duration,
                |b, _| {
                    b.iter(|| {
                        black_box(analyzer.analyze(black_box(&features)))
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark spectral analysis
fn bench_spectral_analysis(c: &mut Criterion) {
    let config = AudioConfig::default();
    let analyzer = match AudioAnalyzer::new(config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping spectral analysis benchmark - analyzer creation failed");
            return;
        }
    };
    
    let mut group = c.benchmark_group("spectral_analysis");
    
    let sample_rate = 44100.0;
    let duration = 1.0;
    let num_samples = (sample_rate * duration) as usize;
    
    for window_size in [512, 1024, 2048, 4096].iter() {
        let audio_data = create_benchmark_audio(num_samples, sample_rate);
        
        group.throughput(Throughput::Elements(*window_size as u64));
        group.bench_with_input(
            BenchmarkId::new("compute_spectrum", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    black_box(analyzer.compute_spectrum(black_box(&audio_data), *window_size))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch audio processing
fn bench_batch_audio_processing(c: &mut Criterion) {
    let config = AudioConfig::default();
    let analyzer = match AudioAnalyzer::new(config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping batch processing benchmark - analyzer creation failed");
            return;
        }
    };
    
    let mut group = c.benchmark_group("batch_audio_processing");
    
    for batch_size in [1, 4, 8, 16].iter() {
        let mut inputs = Vec::new();
        for _ in 0..*batch_size {
            let audio_data = create_benchmark_audio(44100, 44100.0); // 1 second
            let input = AudioInput::new(audio_data, 44100);
            inputs.push(input);
        }
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("process_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    for input in &inputs {
                        if let Ok(features) = analyzer.extract_features(black_box(input)) {
                            black_box(analyzer.analyze(black_box(&features)));
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage for large audio files
fn bench_memory_usage(c: &mut Criterion) {
    let config = AudioConfig::default();
    let analyzer = match AudioAnalyzer::new(config) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("Skipping memory usage benchmark - analyzer creation failed");
            return;
        }
    };
    
    let mut group = c.benchmark_group("memory_usage");
    
    // Test with large audio files (up to 10 seconds)
    for duration in [5.0, 10.0].iter() {
        let sample_rate = 44100.0;
        let num_samples = (sample_rate * duration) as usize;
        let audio_data = create_benchmark_audio(num_samples, sample_rate);
        let input = AudioInput::new(audio_data, sample_rate as u32);
        
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("large_file_processing", (duration * 1000.0) as u32),
            duration,
            |b, _| {
                b.iter(|| {
                    if let Ok(features) = analyzer.extract_features(black_box(&input)) {
                        black_box(analyzer.analyze(black_box(&features)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Create benchmark audio data with realistic characteristics
fn create_benchmark_audio(num_samples: usize, sample_rate: f32) -> Vec<f32> {
    let mut audio_data = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        
        // Create a complex waveform that resembles human speech
        let fundamental = 150.0; // Base frequency
        let signal = (2.0 * std::f32::consts::PI * fundamental * t).sin() * 0.3
            + (2.0 * std::f32::consts::PI * fundamental * 2.0 * t).sin() * 0.15
            + (2.0 * std::f32::consts::PI * fundamental * 3.0 * t).sin() * 0.1
            + (2.0 * std::f32::consts::PI * fundamental * 4.0 * t).sin() * 0.05;
        
        // Add some noise for realism
        let noise = ((i * 7 + i * i) % 1000) as f32 / 1000.0 - 0.5;
        let final_signal = signal + noise * 0.02;
        
        // Apply some envelope to make it more speech-like
        let envelope = (t * 2.0).sin().abs();
        
        audio_data.push(final_signal * envelope);
    }
    
    audio_data
}

/// Create a pure sine wave for pitch detection testing
fn create_sine_wave(num_samples: usize, sample_rate: f32, frequency: f32) -> Vec<f32> {
    let mut audio_data = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let signal = (2.0 * std::f32::consts::PI * frequency * t).sin();
        audio_data.push(signal);
    }
    
    audio_data
}

// Configure benchmark groups
criterion_group!(
    benches,
    bench_mfcc_extraction,
    bench_pitch_detection,
    bench_stress_feature_extraction,
    bench_voice_activity_detection,
    bench_complete_audio_analysis,
    bench_spectral_analysis,
    bench_batch_audio_processing,
    bench_memory_usage
);

criterion_main!(benches);