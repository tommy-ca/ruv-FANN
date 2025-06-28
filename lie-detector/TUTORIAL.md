# Veritas Nexus Tutorial

Welcome to the comprehensive Veritas Nexus tutorial! This hands-on guide will take you from basic concepts to advanced usage through practical examples.

## 📚 Tutorial Overview

This tutorial is structured as a progressive learning journey:

1. **Basic Analysis** - Simple text and single-modality analysis
2. **Multi-Modal Fusion** - Combining vision, audio, and text
3. **Real-Time Processing** - Streaming analysis with live data
4. **Batch Processing** - Large-scale analysis workflows
5. **Advanced Features** - Custom analyzers and ensemble methods
6. **Production Deployment** - Scalable, production-ready systems

Each section builds on the previous one, so we recommend following them in order.

## 🎯 Prerequisites

Before starting, ensure you have:
- Rust 1.70+ installed
- Veritas Nexus added to your project (see [Getting Started](GETTING_STARTED.md))
- Basic familiarity with Rust and async programming

## 📖 Tutorial 1: Basic Analysis

Let's start with the fundamentals of lie detection analysis.

### 1.1 Your First Text Analysis

```rust
// src/tutorial1_basic.rs
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Initialize the detector
    println!("🔍 Tutorial 1: Basic Text Analysis");
    println!("===================================\n");
    
    let detector = LieDetector::builder()
        .with_text(TextConfig::default())
        .build()
        .await?;
    
    println!("✅ Detector initialized for text analysis");
    
    // Step 2: Analyze different types of statements
    let test_cases = vec![
        ("truthful", "I went to the store this morning to buy groceries."),
        ("deceptive", "I definitely did not eat the last piece of cake, I swear on my life."),
        ("uncertain", "I think maybe I was at home around that time."),
        ("complex", "The meeting was productive and we discussed various strategic initiatives."),
    ];
    
    for (category, statement) in test_cases {
        println!("\n📝 Analyzing {} statement:", category);
        println!("   \"{}\"", statement);
        
        let input = AnalysisInput {
            video_path: None,
            audio_path: None,
            transcript: Some(statement.to_string()),
            physiological_data: None,
        };
        
        let result = detector.analyze(input).await?;
        
        println!("   Decision: {:?}", result.decision);
        println!("   Confidence: {:.1}%", result.confidence * 100.0);
        
        if let Some(text_score) = result.modality_scores.text_score {
            println!("   Deception score: {:.3}", text_score);
        }
        
        // Show key reasoning steps
        println!("   Key reasoning:");
        for step in result.reasoning_trace.iter().take(3) {
            println!("     • {}", step);
        }
    }
    
    Ok(())
}
```

**Expected Output:**
```
🔍 Tutorial 1: Basic Text Analysis
===================================

✅ Detector initialized for text analysis

📝 Analyzing truthful statement:
   "I went to the store this morning to buy groceries."
   Decision: TruthTelling
   Confidence: 67.3%
   Deception score: 0.234
   Key reasoning:
     • Processing text input...
     • Simple sentence structure detected
     • No hedging language found
```

### 1.2 Understanding Configuration Options

```rust
// src/tutorial1_config.rs
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Tutorial 1.2: Configuration Options");
    println!("======================================\n");
    
    // Basic configuration
    let basic_config = TextConfig::default();
    
    // Advanced configuration
    let advanced_config = TextConfig {
        model_type: TextModel::RoBerta, // More accurate than default DistilBert
        enable_linguistic_analysis: true,
        enable_sentiment_analysis: true,
        language: "en".to_string(),
    };
    
    // Performance-optimized configuration
    let fast_config = TextConfig {
        model_type: TextModel::DistilBert, // Fastest model
        enable_linguistic_analysis: false, // Disable for speed
        enable_sentiment_analysis: false,
        language: "en".to_string(),
    };
    
    let test_text = "I absolutely did not see anyone take the money.";
    
    // Compare different configurations
    for (name, config) in vec![
        ("Basic", basic_config),
        ("Advanced", advanced_config),
        ("Fast", fast_config),
    ] {
        println!("🧪 Testing {} configuration:", name);
        
        let detector = LieDetector::builder()
            .with_text(config)
            .build()
            .await?;
        
        let start_time = std::time::Instant::now();
        
        let input = AnalysisInput {
            video_path: None,
            audio_path: None,
            transcript: Some(test_text.to_string()),
            physiological_data: None,
        };
        
        let result = detector.analyze(input).await?;
        let processing_time = start_time.elapsed();
        
        println!("   Decision: {:?}", result.decision);
        println!("   Confidence: {:.1}%", result.confidence * 100.0);
        println!("   Processing time: {}ms", processing_time.as_millis());
        println!("   Score: {:.3}\n", result.modality_scores.text_score.unwrap_or(0.0));
    }
    
    Ok(())
}
```

## 🎬 Tutorial 2: Multi-Modal Fusion

Now let's combine multiple data sources for more accurate detection.

### 2.1 Vision and Audio Analysis

```rust
// src/tutorial2_multimodal.rs
use veritas_nexus::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Tutorial 2: Multi-Modal Analysis");
    println!("==================================\n");
    
    // Configure all modalities
    let detector = LieDetector::builder()
        .with_vision(VisionConfig {
            enable_face_detection: true,
            enable_micro_expressions: true,
            enable_eye_tracking: false,
            model_precision: ModelPrecision::Balanced,
        })
        .with_audio(AudioConfig {
            sample_rate: 16000,
            enable_pitch_analysis: true,
            enable_stress_detection: true,
            enable_voice_quality: true,
        })
        .with_text(TextConfig::bert_based())
        .with_fusion_strategy(FusionStrategy::AdaptiveWeight)
        .build()
        .await?;
    
    println!("✅ Multi-modal detector initialized");
    
    // Simulate different combinations of available data
    let scenarios = vec![
        ("Text Only", create_text_only_input()),
        ("Audio + Text", create_audio_text_input()),
        ("Vision + Text", create_vision_text_input()),
        ("Full Multi-modal", create_full_multimodal_input()),
    ];
    
    for (scenario_name, input) in scenarios {
        println!("\n🎯 Scenario: {}", scenario_name);
        println!("   Available modalities:");
        
        if input.video_path.is_some() {
            println!("     👁️  Vision data");
        }
        if input.audio_path.is_some() {
            println!("     🔊 Audio data");
        }
        if input.transcript.is_some() {
            println!("     📝 Text data");
        }
        if input.physiological_data.is_some() {
            println!("     💓 Physiological data");
        }
        
        let result = detector.analyze(input).await?;
        
        println!("   Final Decision: {:?}", result.decision);
        println!("   Confidence: {:.1}%", result.confidence * 100.0);
        
        // Show individual modality contributions
        let scores = &result.modality_scores;
        if let Some(vision) = scores.vision_score {
            println!("   Vision score: {:.3}", vision);
        }
        if let Some(audio) = scores.audio_score {
            println!("   Audio score: {:.3}", audio);
        }
        if let Some(text) = scores.text_score {
            println!("   Text score: {:.3}", text);
        }
        if let Some(physio) = scores.physiological_score {
            println!("   Physiological score: {:.3}", physio);
        }
    }
    
    Ok(())
}

fn create_text_only_input() -> AnalysisInput {
    AnalysisInput {
        video_path: None,
        audio_path: None,
        transcript: Some("I was not anywhere near the building that night.".to_string()),
        physiological_data: None,
    }
}

fn create_audio_text_input() -> AnalysisInput {
    AnalysisInput {
        video_path: None,
        audio_path: Some("examples/data/sample_audio.wav".to_string()),
        transcript: Some("I was not anywhere near the building that night.".to_string()),
        physiological_data: None,
    }
}

fn create_vision_text_input() -> AnalysisInput {
    AnalysisInput {
        video_path: Some("examples/data/sample_video.mp4".to_string()),
        audio_path: None,
        transcript: Some("I was not anywhere near the building that night.".to_string()),
        physiological_data: None,
    }
}

fn create_full_multimodal_input() -> AnalysisInput {
    AnalysisInput {
        video_path: Some("examples/data/sample_video.mp4".to_string()),
        audio_path: Some("examples/data/sample_audio.wav".to_string()),
        transcript: Some("I was not anywhere near the building that night.".to_string()),
        physiological_data: Some(vec![72.5, 73.8, 75.1, 77.2, 79.8]), // Heart rate progression
    }
}
```

### 2.2 Understanding Fusion Strategies

```rust
// src/tutorial2_fusion.rs
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔀 Tutorial 2.2: Fusion Strategies");
    println!("==================================\n");
    
    let input = AnalysisInput {
        video_path: None, // Simulated for this tutorial
        audio_path: None, // Simulated for this tutorial
        transcript: Some("I swear I didn't do anything wrong.".to_string()),
        physiological_data: Some(vec![68.0, 72.0, 78.0, 84.0]), // Increasing heart rate
    };
    
    // Test different fusion strategies
    let fusion_strategies = vec![
        ("Equal Weight", FusionStrategy::EqualWeight),
        ("Adaptive Weight", FusionStrategy::AdaptiveWeight),
        ("Attention Based", FusionStrategy::AttentionBased),
        ("Context Aware", FusionStrategy::ContextAware),
    ];
    
    for (strategy_name, strategy) in fusion_strategies {
        println!("🧮 Testing {} fusion:", strategy_name);
        
        let detector = LieDetector::builder()
            .with_vision(VisionConfig::default())
            .with_audio(AudioConfig::default())
            .with_text(TextConfig::default())
            .with_fusion_strategy(strategy)
            .build()
            .await?;
        
        let result = detector.analyze(input.clone()).await?;
        
        println!("   Decision: {:?}", result.decision);
        println!("   Confidence: {:.1}%", result.confidence * 100.0);
        
        // Explain fusion strategy behavior
        match strategy_name {
            "Equal Weight" => {
                println!("   Strategy: All modalities weighted equally");
            },
            "Adaptive Weight" => {
                println!("   Strategy: Weights adapt based on individual modality confidence");
            },
            "Attention Based" => {
                println!("   Strategy: Neural attention mechanism determines optimal weights");
            },
            "Context Aware" => {
                println!("   Strategy: Weights consider situational context and environment");
            },
            _ => {},
        }
        
        println!();
    }
    
    Ok(())
}
```

## 🌊 Tutorial 3: Real-Time Processing

Learn to process live data streams for real-time analysis.

### 3.1 Basic Streaming Setup

```rust
// src/tutorial3_streaming.rs
use veritas_nexus::streaming::*;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Tutorial 3: Real-Time Streaming");
    println!("==================================\n");
    
    // Configure streaming with realistic settings
    let config = StreamingConfig {
        target_fps: 15.0, // Reasonable for most applications
        audio_chunk_size_ms: 100, // 100ms audio chunks
        sync_window_ms: 200, // 200ms synchronization window
        max_latency_ms: 300, // Maximum acceptable latency
        buffer_size: 64, // Ring buffer size
        enable_adaptive_quality: true,
    };
    
    println!("⚙️  Streaming Configuration:");
    println!("   Target FPS: {}", config.target_fps);
    println!("   Audio chunk size: {}ms", config.audio_chunk_size_ms);
    println!("   Sync window: {}ms", config.sync_window_ms);
    println!("   Max latency: {}ms", config.max_latency_ms);
    println!("   Buffer size: {}", config.buffer_size);
    
    // Create streaming pipeline
    let pipeline = StreamingPipeline::new(config);
    pipeline.start().await?;
    
    println!("\n🚀 Pipeline started successfully!");
    
    // Create data simulator for this tutorial
    let mut simulator = DataSimulator::new();
    
    // Simulate 15 seconds of streaming
    let duration = Duration::from_secs(15);
    let start_time = std::time::Instant::now();
    
    println!("📊 Starting simulation for {}s...\n", duration.as_secs());
    
    // Spawn data generation tasks
    let pipeline_clone = std::sync::Arc::new(pipeline);
    
    // Video frame generation
    let video_pipeline = pipeline_clone.clone();
    tokio::spawn(async move {
        let mut video_interval = interval(Duration::from_millis(67)); // ~15 FPS
        let mut frame_count = 0;
        
        while start_time.elapsed() < duration {
            video_interval.tick().await;
            frame_count += 1;
            
            let frame = simulator.generate_video_frame();
            video_pipeline.add_video_frame(frame).await;
            
            if frame_count % 30 == 0 { // Print every 2 seconds
                println!("📷 Generated {} video frames", frame_count);
            }
        }
    });
    
    // Audio chunk generation
    let audio_pipeline = pipeline_clone.clone();
    tokio::spawn(async move {
        let mut audio_interval = interval(Duration::from_millis(100)); // 10 chunks/sec
        let mut chunk_count = 0;
        
        while start_time.elapsed() < duration {
            audio_interval.tick().await;
            chunk_count += 1;
            
            let chunk = simulator.generate_audio_chunk();
            audio_pipeline.add_audio_chunk(chunk).await;
            
            if chunk_count % 30 == 0 { // Print every 3 seconds
                println!("🔊 Generated {} audio chunks", chunk_count);
            }
        }
    });
    
    // Text segment generation (less frequent)
    let text_pipeline = pipeline_clone.clone();
    tokio::spawn(async move {
        let mut text_interval = interval(Duration::from_secs(3)); // Every 3 seconds
        let mut segment_count = 0;
        
        while start_time.elapsed() < duration {
            text_interval.tick().await;
            segment_count += 1;
            
            if let Some(segment) = simulator.generate_text_segment() {
                text_pipeline.add_text_segment(segment).await;
                println!("📝 Generated text segment: \"{}\"", 
                    segment.text.chars().take(30).collect::<String>());
            }
        }
    });
    
    // Result monitoring
    let mut result_count = 0;
    let mut total_latency = 0u64;
    
    while start_time.elapsed() < duration {
        if let Some(result) = pipeline_clone.try_get_result().await {
            result_count += 1;
            total_latency += result.processing_latency_ms;
            
            println!("🔍 Result #{}: {:?} (score: {:.3}, confidence: {:.1}%, {}ms)",
                result_count,
                result.decision,
                result.deception_score,
                result.confidence * 100.0,
                result.processing_latency_ms
            );
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // Stop pipeline and show statistics
    pipeline_clone.stop();
    
    let avg_latency = if result_count > 0 {
        total_latency / result_count as u64
    } else {
        0
    };
    
    println!("\n📈 Final Statistics:");
    println!("   Total results: {}", result_count);
    println!("   Average latency: {}ms", avg_latency);
    println!("   Results per second: {:.1}", result_count as f64 / duration.as_secs_f64());
    
    let sync_stats = pipeline_clone.get_sync_stats().await;
    println!("   Buffer utilization:");
    println!("     Video: {}/64", sync_stats.video_buffer_size);
    println!("     Audio: {}/64", sync_stats.audio_buffer_size);
    println!("     Text: {}/64", sync_stats.text_buffer_size);
    
    Ok(())
}

// Data simulator for the tutorial
struct DataSimulator {
    frame_counter: u64,
    audio_counter: u64,
    text_segments: Vec<String>,
    text_index: usize,
}

impl DataSimulator {
    fn new() -> Self {
        Self {
            frame_counter: 0,
            audio_counter: 0,
            text_segments: vec![
                "I was at home all evening".to_string(),
                "No I never saw that document".to_string(),
                "I think maybe I was there".to_string(),
                "I definitely did not take anything".to_string(),
                "Perhaps someone else did it".to_string(),
            ],
            text_index: 0,
        }
    }
    
    fn generate_video_frame(&mut self) -> VideoFrame {
        self.frame_counter += 1;
        
        // Generate dummy RGB frame
        let width = 640;
        let height = 480;
        let mut data = vec![0u8; (width * height * 3) as usize];
        
        // Simple pattern that changes over time
        for i in 0..data.len() {
            data[i] = ((i + self.frame_counter as usize) % 256) as u8;
        }
        
        VideoFrame {
            data,
            width,
            height,
            timestamp: std::time::Instant::now(),
            format: VideoFormat::Rgb24,
        }
    }
    
    fn generate_audio_chunk(&mut self) -> AudioChunk {
        self.audio_counter += 1;
        
        let sample_rate = 16000;
        let chunk_duration_ms = 100;
        let samples_count = (sample_rate * chunk_duration_ms / 1000) as usize;
        
        // Generate sine wave with some variation
        let mut samples = Vec::with_capacity(samples_count);
        let frequency = 440.0 + (self.audio_counter as f32 * 10.0) % 200.0;
        
        for i in 0..samples_count {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.1;
            samples.push(sample);
        }
        
        AudioChunk {
            samples,
            sample_rate,
            channels: 1,
            timestamp: std::time::Instant::now(),
        }
    }
    
    fn generate_text_segment(&mut self) -> Option<TextSegment> {
        if self.text_index < self.text_segments.len() {
            let text = self.text_segments[self.text_index].clone();
            self.text_index += 1;
            
            let now = std::time::Instant::now();
            Some(TextSegment {
                text,
                confidence: 0.85,
                start_time: now,
                end_time: now + Duration::from_secs(2),
                is_final: true,
            })
        } else {
            None
        }
    }
}
```

### 3.2 Advanced Streaming with Custom Sources

```rust
// src/tutorial3_custom_streaming.rs
use veritas_nexus::streaming::*;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Tutorial 3.2: Custom Streaming Sources");
    println!("==========================================\n");
    
    // Create custom streaming pipeline with file-based sources
    let pipeline = CustomStreamingPipeline::new().await?;
    
    // Set up file-based video source
    let video_files = vec![
        "examples/data/interview_01.mp4",
        "examples/data/interview_02.mp4",
        "examples/data/interview_03.mp4",
    ];
    
    // Set up file-based audio source
    let audio_files = vec![
        "examples/data/interview_01.wav",
        "examples/data/interview_02.wav",
        "examples/data/interview_03.wav",
    ];
    
    println!("📁 Processing {} video files and {} audio files", 
        video_files.len(), audio_files.len());
    
    // Process files sequentially with streaming pipeline
    for (i, (video_file, audio_file)) in video_files.iter().zip(audio_files.iter()).enumerate() {
        println!("\n🎬 Processing file pair {} of {}", i + 1, video_files.len());
        println!("   Video: {}", video_file);
        println!("   Audio: {}", audio_file);
        
        // Configure file-based sources
        let file_config = FileStreamingConfig {
            video_path: video_file.to_string(),
            audio_path: audio_file.to_string(),
            playback_speed: 1.0, // Real-time playback
            loop_playback: false,
            sync_tolerance_ms: 50,
        };
        
        // Start processing this file pair
        let results = pipeline.process_files(file_config).await?;
        
        println!("   📊 Results: {} decisions", results.len());
        
        // Show summary statistics
        let mut decisions_count = std::collections::HashMap::new();
        let mut total_confidence = 0.0;
        
        for result in &results {
            *decisions_count.entry(result.decision.clone()).or_insert(0) += 1;
            total_confidence += result.confidence;
        }
        
        let avg_confidence = total_confidence / results.len() as f32;
        
        println!("   📈 Summary:");
        for (decision, count) in decisions_count {
            let percentage = (count as f32 / results.len() as f32) * 100.0;
            println!("     {:?}: {} ({:.1}%)", decision, count, percentage);
        }
        println!("     Average confidence: {:.1}%", avg_confidence * 100.0);
        
        // Show temporal progression of decisions
        println!("   ⏱️  Decision timeline:");
        for (i, result) in results.iter().enumerate().step_by(5) { // Show every 5th result
            println!("     {}s: {:?} ({:.1}%)", 
                i * 2, // Assuming 2-second intervals
                result.decision, 
                result.confidence * 100.0
            );
        }
    }
    
    Ok(())
}

// Custom streaming pipeline for file processing
struct CustomStreamingPipeline {
    // Implementation details would go here
}

impl CustomStreamingPipeline {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize custom pipeline
        Ok(Self {})
    }
    
    async fn process_files(&self, config: FileStreamingConfig) -> Result<Vec<RealTimeResult>, Box<dyn std::error::Error>> {
        // Simulate file processing
        let mut results = Vec::new();
        
        // In a real implementation, this would:
        // 1. Open video and audio files
        // 2. Extract frames and audio chunks with timestamps
        // 3. Feed them through the streaming pipeline
        // 4. Collect and return results
        
        // For the tutorial, generate some example results
        for i in 0..30 { // Simulate 30 results (60 seconds at 2-second intervals)
            let timestamp = std::time::Instant::now();
            
            let result = RealTimeResult {
                deception_score: 0.3 + (i as f32 * 0.02) % 0.7, // Varying score
                confidence: 0.6 + (i as f32 * 0.01) % 0.4, // Varying confidence
                decision: if i % 5 == 0 {
                    RealtimeDecision::Deceptive
                } else if i % 7 == 0 {
                    RealtimeDecision::Uncertain
                } else {
                    RealtimeDecision::TruthTelling
                },
                modality_contributions: ModalityContributions {
                    vision_weight: 0.4,
                    audio_weight: 0.35,
                    text_weight: 0.25,
                    vision_score: Some(0.5 + (i as f32 * 0.01) % 0.3),
                    audio_score: Some(0.4 + (i as f32 * 0.015) % 0.4),
                    text_score: Some(0.3 + (i as f32 * 0.02) % 0.5),
                },
                timestamp,
                processing_latency_ms: 50 + (i % 30) as u64, // Varying latency
            };
            
            results.push(result);
        }
        
        Ok(results)
    }
}

#[derive(Debug)]
struct FileStreamingConfig {
    video_path: String,
    audio_path: String,
    playback_speed: f32,
    loop_playback: bool,
    sync_tolerance_ms: u32,
}
```

## 📊 Tutorial 4: Batch Processing

Learn to process large datasets efficiently.

### 4.1 Basic Batch Processing

```rust
// src/tutorial4_batch.rs
use veritas_nexus::batch::*;
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Tutorial 4: Batch Processing");
    println!("===============================\n");
    
    // Configure batch processor
    let batch_config = BatchConfig {
        batch_size: 16,
        max_concurrent_batches: 4,
        enable_progress_reporting: true,
        checkpoint_interval: 50,
        error_handling: ErrorHandling::ContinueOnError,
        output_format: OutputFormat::JsonLines,
    };
    
    println!("⚙️  Batch Configuration:");
    println!("   Batch size: {}", batch_config.batch_size);
    println!("   Max concurrent batches: {}", batch_config.max_concurrent_batches);
    println!("   Checkpoint interval: {}", batch_config.checkpoint_interval);
    println!("   Error handling: {:?}", batch_config.error_handling);
    
    // Create batch processor
    let batch_processor = BatchProcessor::new(batch_config);
    
    // Create sample dataset
    let dataset = create_sample_dataset().await?;
    println!("\n📁 Created sample dataset with {} items", dataset.len());
    
    // Process the dataset
    println!("\n🚀 Starting batch processing...");
    
    let start_time = std::time::Instant::now();
    
    let results = batch_processor
        .process_dataset(&dataset)
        .with_progress_callback(|processed, total| {
            let progress = (processed as f32 / total as f32) * 100.0;
            println!("   Progress: {}/{} ({:.1}%)", processed, total, progress);
        })
        .await?;
    
    let total_time = start_time.elapsed();
    
    println!("\n✅ Batch processing completed!");
    println!("   Total time: {:.2}s", total_time.as_secs_f64());
    println!("   Items processed: {}", results.len());
    println!("   Throughput: {:.1} items/second", 
        results.len() as f64 / total_time.as_secs_f64());
    
    // Analyze results
    analyze_batch_results(&results).await?;
    
    // Save results
    let output_file = "batch_results.jsonl";
    batch_processor.save_results(&results, output_file).await?;
    println!("\n💾 Results saved to {}", output_file);
    
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct DatasetItem {
    id: String,
    text: String,
    video_path: Option<String>,
    audio_path: Option<String>,
    ground_truth: Option<bool>, // True for deceptive, false for truthful
}

async fn create_sample_dataset() -> Result<Vec<DatasetItem>, Box<dyn std::error::Error>> {
    let mut dataset = Vec::new();
    
    // Truthful statements
    let truthful_statements = vec![
        "I went to the store and bought groceries this morning.",
        "The meeting started at 2 PM and lasted for about an hour.",
        "I have worked at this company for three years.",
        "My favorite color is blue and I enjoy reading books.",
        "I graduated from university in 2018 with a degree in computer science.",
    ];
    
    for (i, statement) in truthful_statements.iter().enumerate() {
        dataset.push(DatasetItem {
            id: format!("truthful_{:03}", i + 1),
            text: statement.to_string(),
            video_path: None,
            audio_path: None,
            ground_truth: Some(false), // False = truthful
        });
    }
    
    // Deceptive statements
    let deceptive_statements = vec![
        "I definitely did not eat the last piece of cake in the refrigerator.",
        "I have never seen that document before in my entire life.",
        "I was absolutely at home sleeping during the time of the incident.",
        "I would never lie about something as important as this.",
        "I swear on my mother's grave that I am telling the complete truth.",
    ];
    
    for (i, statement) in deceptive_statements.iter().enumerate() {
        dataset.push(DatasetItem {
            id: format!("deceptive_{:03}", i + 1),
            text: statement.to_string(),
            video_path: None,
            audio_path: None,
            ground_truth: Some(true), // True = deceptive
        });
    }
    
    // Ambiguous statements
    let ambiguous_statements = vec![
        "I think I might have been there around that time.",
        "Perhaps someone else could have done it.",
        "I believe the situation was more complicated than it appears.",
        "It's possible that I misunderstood what happened.",
        "I'm not entirely sure about all the details.",
    ];
    
    for (i, statement) in ambiguous_statements.iter().enumerate() {
        dataset.push(DatasetItem {
            id: format!("ambiguous_{:03}", i + 1),
            text: statement.to_string(),
            video_path: None,
            audio_path: None,
            ground_truth: None, // Unknown truth value
        });
    }
    
    Ok(dataset)
}

async fn analyze_batch_results(results: &[BatchResult]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Batch Results Analysis:");
    println!("   ========================");
    
    // Count decisions
    let mut decision_counts = std::collections::HashMap::new();
    let mut confidence_sum = 0.0;
    let mut processing_time_sum = 0u64;
    
    for result in results {
        *decision_counts.entry(result.decision.clone()).or_insert(0) += 1;
        confidence_sum += result.confidence;
        processing_time_sum += result.processing_time_ms;
    }
    
    // Print decision distribution
    println!("\n📈 Decision Distribution:");
    for (decision, count) in &decision_counts {
        let percentage = (*count as f32 / results.len() as f32) * 100.0;
        println!("   {:?}: {} ({:.1}%)", decision, count, percentage);
    }
    
    // Print average metrics
    let avg_confidence = confidence_sum / results.len() as f32;
    let avg_processing_time = processing_time_sum / results.len() as u64;
    
    println!("\n📊 Average Metrics:");
    println!("   Confidence: {:.1}%", avg_confidence * 100.0);
    println!("   Processing time: {}ms", avg_processing_time);
    
    // Analyze confidence distribution
    let mut high_confidence = 0;
    let mut medium_confidence = 0;
    let mut low_confidence = 0;
    
    for result in results {
        if result.confidence >= 0.8 {
            high_confidence += 1;
        } else if result.confidence >= 0.5 {
            medium_confidence += 1;
        } else {
            low_confidence += 1;
        }
    }
    
    println!("\n🎯 Confidence Distribution:");
    println!("   High (≥80%): {} ({:.1}%)", high_confidence, 
        (high_confidence as f32 / results.len() as f32) * 100.0);
    println!("   Medium (50-79%): {} ({:.1}%)", medium_confidence,
        (medium_confidence as f32 / results.len() as f32) * 100.0);
    println!("   Low (<50%): {} ({:.1}%)", low_confidence,
        (low_confidence as f32 / results.len() as f32) * 100.0);
    
    Ok(())
}

#[derive(Debug)]
struct BatchResult {
    id: String,
    decision: Decision,
    confidence: f32,
    deception_score: f32,
    processing_time_ms: u64,
}

#[derive(Debug, Clone)]
enum Decision {
    TruthTelling,
    Deceptive,
    Uncertain,
}

// Mock batch processor for the tutorial
struct BatchProcessor {
    config: BatchConfig,
}

impl BatchProcessor {
    fn new(config: BatchConfig) -> Self {
        Self { config }
    }
    
    async fn process_dataset(&self, dataset: &[DatasetItem]) -> BatchProcessorWithCallback {
        BatchProcessorWithCallback {
            processor: self,
            dataset,
            callback: None,
        }
    }
    
    async fn save_results(&self, results: &[BatchResult], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(filename)?;
        
        for result in results {
            let json = serde_json::to_string(result)?;
            writeln!(file, "{}", json)?;
        }
        
        Ok(())
    }
}

struct BatchProcessorWithCallback<'a> {
    processor: &'a BatchProcessor,
    dataset: &'a [DatasetItem],
    callback: Option<Box<dyn Fn(usize, usize) + 'a>>,
}

impl<'a> BatchProcessorWithCallback<'a> {
    fn with_progress_callback<F>(mut self, callback: F) -> Self 
    where F: Fn(usize, usize) + 'a {
        self.callback = Some(Box::new(callback));
        self
    }
    
    async fn await(self) -> Result<Vec<BatchResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        let total = self.dataset.len();
        
        for (i, item) in self.dataset.iter().enumerate() {
            // Simulate processing time
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            
            // Simulate analysis based on text content
            let deception_score = calculate_mock_deception_score(&item.text);
            let confidence = 0.6 + (deception_score * 0.4);
            let decision = if confidence < 0.3 {
                Decision::Uncertain
            } else if deception_score > 0.6 {
                Decision::Deceptive
            } else {
                Decision::TruthTelling
            };
            
            let result = BatchResult {
                id: item.id.clone(),
                decision,
                confidence,
                deception_score,
                processing_time_ms: 45 + (i % 20) as u64, // Simulated processing time
            };
            
            results.push(result);
            
            // Call progress callback
            if let Some(ref callback) = self.callback {
                callback(i + 1, total);
            }
        }
        
        Ok(results)
    }
}

fn calculate_mock_deception_score(text: &str) -> f32 {
    // Simple mock scoring based on linguistic patterns
    let text_lower = text.to_lowercase();
    let mut score = 0.3; // Base score
    
    // Increase score for certain deceptive indicators
    if text_lower.contains("definitely") || text_lower.contains("absolutely") {
        score += 0.2;
    }
    if text_lower.contains("never") || text_lower.contains("always") {
        score += 0.15;
    }
    if text_lower.contains("swear") || text_lower.contains("promise") {
        score += 0.25;
    }
    if text_lower.contains("entire life") || text_lower.contains("mother's grave") {
        score += 0.3;
    }
    
    // Decrease score for uncertainty markers
    if text_lower.contains("think") || text_lower.contains("maybe") {
        score -= 0.1;
    }
    if text_lower.contains("perhaps") || text_lower.contains("possible") {
        score -= 0.15;
    }
    
    score.clamp(0.0, 1.0)
}

#[derive(Debug)]
struct BatchConfig {
    batch_size: usize,
    max_concurrent_batches: usize,
    enable_progress_reporting: bool,
    checkpoint_interval: usize,
    error_handling: ErrorHandling,
    output_format: OutputFormat,
}

#[derive(Debug)]
enum ErrorHandling {
    ContinueOnError,
    StopOnError,
}

#[derive(Debug)]
enum OutputFormat {
    JsonLines,
    Csv,
    Parquet,
}
```

## 🎓 Tutorial 5: Advanced Features

Explore cutting-edge capabilities of Veritas Nexus.

### 5.1 Custom Analyzers and Ensemble Methods

```rust
// src/tutorial5_advanced.rs
use veritas_nexus::prelude::*;
use async_trait::async_trait;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Tutorial 5: Advanced Features");
    println!("================================\n");
    
    // Part 1: Custom Analyzer
    println!("🔧 Part 1: Creating a Custom Analyzer");
    demonstrate_custom_analyzer().await?;
    
    println!("\n" + &"=".repeat(50) + "\n");
    
    // Part 2: Ensemble Methods
    println!("🎭 Part 2: Ensemble Methods");
    demonstrate_ensemble_methods().await?;
    
    println!("\n" + &"=".repeat(50) + "\n");
    
    // Part 3: Active Learning
    println!("🧠 Part 3: Active Learning");
    demonstrate_active_learning().await?;
    
    Ok(())
}

async fn demonstrate_custom_analyzer() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom analyzer that analyzes sentence complexity
    let custom_analyzer = SentenceComplexityAnalyzer::new();
    
    let test_sentences = vec![
        "I did it.",
        "I absolutely did not do anything wrong at all.",
        "The complex interplay of various factors led to a situation that, while unfortunate, was completely beyond my control.",
        "Perhaps, if I'm being entirely honest, there might have been some misunderstanding.",
    ];
    
    println!("📝 Testing custom sentence complexity analyzer:");
    
    for (i, sentence) in test_sentences.iter().enumerate() {
        println!("\n   Sentence {}: \"{}\"", i + 1, sentence);
        
        let result = custom_analyzer.analyze(sentence).await?;
        println!("      Complexity score: {:.3}", result.complexity_score);
        println!("      Deception indicator: {:.3}", result.deception_score);
        println!("      Confidence: {:.1}%", result.confidence * 100.0);
        
        // Show reasoning
        for reason in &result.reasoning {
            println!("      • {}", reason);
        }
    }
    
    Ok(())
}

// Custom analyzer implementation
struct SentenceComplexityAnalyzer;

impl SentenceComplexityAnalyzer {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl SentenceComplexityAnalyzer {
    async fn analyze(&self, text: &str) -> Result<ComplexityAnalysisResult, Box<dyn std::error::Error>> {
        // Analyze sentence complexity
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split(&['.', '!', '?'][..]).filter(|s| !s.trim().is_empty()).count();
        let avg_words_per_sentence = if sentence_count > 0 {
            word_count as f32 / sentence_count as f32
        } else {
            0.0
        };
        
        // Count complex words (3+ syllables approximated by length)
        let complex_words = text.split_whitespace()
            .filter(|word| word.len() > 8)
            .count();
        
        let complex_word_ratio = complex_words as f32 / word_count as f32;
        
        // Calculate complexity score (0-1)
        let complexity_score = (avg_words_per_sentence / 30.0 + complex_word_ratio).min(1.0);
        
        // Deception hypothesis: Very simple or very complex sentences might indicate deception
        let deception_score = if complexity_score < 0.2 {
            0.7 // Very simple might be evasive
        } else if complexity_score > 0.8 {
            0.75 // Very complex might be obfuscation
        } else {
            0.3 // Normal complexity
        };
        
        let confidence = if word_count < 5 {
            0.4 // Low confidence for very short text
        } else {
            0.8
        };
        
        let mut reasoning = Vec::new();
        reasoning.push(format!("Analyzed {} words in {} sentences", word_count, sentence_count));
        reasoning.push(format!("Average words per sentence: {:.1}", avg_words_per_sentence));
        reasoning.push(format!("Complex words ratio: {:.1}%", complex_word_ratio * 100.0));
        
        if complexity_score < 0.2 {
            reasoning.push("Very simple sentence structure detected - possible evasion".to_string());
        } else if complexity_score > 0.8 {
            reasoning.push("Highly complex sentence structure - possible obfuscation".to_string());
        } else {
            reasoning.push("Normal sentence complexity observed".to_string());
        }
        
        Ok(ComplexityAnalysisResult {
            complexity_score,
            deception_score,
            confidence,
            reasoning,
        })
    }
}

#[derive(Debug)]
struct ComplexityAnalysisResult {
    complexity_score: f32,
    deception_score: f32,
    confidence: f32,
    reasoning: Vec<String>,
}

async fn demonstrate_ensemble_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Creating ensemble of different detection strategies:");
    
    // Create multiple detectors with different configurations
    let conservative_detector = LieDetector::builder()
        .with_text(TextConfig {
            model_type: TextModel::Bert,
            enable_linguistic_analysis: true,
            enable_sentiment_analysis: false,
            language: "en".to_string(),
        })
        .with_fusion_strategy(FusionStrategy::EqualWeight)
        .build()
        .await?;
    
    let aggressive_detector = LieDetector::builder()
        .with_text(TextConfig {
            model_type: TextModel::RoBerta,
            enable_linguistic_analysis: true,
            enable_sentiment_analysis: true,
            language: "en".to_string(),
        })
        .with_fusion_strategy(FusionStrategy::AdaptiveWeight)
        .build()
        .await?;
    
    let fast_detector = LieDetector::builder()
        .with_text(TextConfig {
            model_type: TextModel::DistilBert,
            enable_linguistic_analysis: false,
            enable_sentiment_analysis: false,
            language: "en".to_string(),
        })
        .with_fusion_strategy(FusionStrategy::EqualWeight)
        .build()
        .await?;
    
    let test_statements = vec![
        "I was definitely at home all evening.",
        "I think maybe I saw something suspicious.",
        "The situation was more complex than it appears.",
    ];
    
    for (i, statement) in test_statements.iter().enumerate() {
        println!("\n   Statement {}: \"{}\"", i + 1, statement);
        
        let input = AnalysisInput {
            video_path: None,
            audio_path: None,
            transcript: Some(statement.to_string()),
            physiological_data: None,
        };
        
        // Get predictions from each detector
        let conservative_result = conservative_detector.analyze(input.clone()).await?;
        let aggressive_result = aggressive_detector.analyze(input.clone()).await?;
        let fast_result = fast_detector.analyze(input.clone()).await?;
        
        println!("      Conservative: {:?} ({:.1}%)", 
            conservative_result.decision, conservative_result.confidence * 100.0);
        println!("      Aggressive: {:?} ({:.1}%)", 
            aggressive_result.decision, aggressive_result.confidence * 100.0);
        println!("      Fast: {:?} ({:.1}%)", 
            fast_result.decision, fast_result.confidence * 100.0);
        
        // Simple ensemble voting
        let scores = vec![
            conservative_result.modality_scores.text_score.unwrap_or(0.5),
            aggressive_result.modality_scores.text_score.unwrap_or(0.5),
            fast_result.modality_scores.text_score.unwrap_or(0.5),
        ];
        
        let ensemble_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let ensemble_confidence = (conservative_result.confidence + 
                                 aggressive_result.confidence + 
                                 fast_result.confidence) / 3.0;
        
        let ensemble_decision = if ensemble_confidence < 0.3 {
            "Uncertain"
        } else if ensemble_score > 0.6 {
            "Deceptive"
        } else {
            "TruthTelling"
        };
        
        println!("      Ensemble: {} ({:.1}%) [score: {:.3}]", 
            ensemble_decision, ensemble_confidence * 100.0, ensemble_score);
    }
    
    Ok(())
}

async fn demonstrate_active_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Demonstrating active learning workflow:");
    
    // Simulate an active learning scenario
    let mut model_accuracy = 0.75; // Starting accuracy
    let target_accuracy = 0.90;
    let mut iteration = 0;
    
    println!("   Starting model accuracy: {:.1}%", model_accuracy * 100.0);
    println!("   Target accuracy: {:.1}%", target_accuracy * 100.0);
    
    // Simulate unlabeled data pool
    let mut unlabeled_samples = vec![
        "I was definitely not there that night.",
        "The document might have been misplaced.",
        "I absolutely guarantee this is the truth.",
        "Perhaps someone else was responsible.",
        "I would never lie about something this important.",
        "The situation was somewhat confusing.",
        "I have no recollection of that event.",
        "This is exactly what happened.",
    ];
    
    while model_accuracy < target_accuracy && !unlabeled_samples.is_empty() && iteration < 5 {
        iteration += 1;
        println!("\n   🔄 Active Learning Iteration {}", iteration);
        
        // Simulate uncertainty-based sample selection
        let uncertain_sample = unlabeled_samples.remove(0); // Take first sample
        println!("      Selected uncertain sample: \"{}\"", uncertain_sample);
        
        // Simulate human labeling
        let simulated_label = if uncertain_sample.contains("definitely") || 
                               uncertain_sample.contains("absolutely") {
            "Deceptive"
        } else if uncertain_sample.contains("perhaps") || 
                  uncertain_sample.contains("might") {
            "Uncertain"
        } else {
            "Truthful"
        };
        
        println!("      Human label: {}", simulated_label);
        
        // Simulate model retraining and improvement
        let accuracy_improvement = rand::random::<f32>() * 0.05; // 0-5% improvement
        model_accuracy += accuracy_improvement;
        
        println!("      Model retrained. New accuracy: {:.1}%", model_accuracy * 100.0);
        println!("      Remaining unlabeled samples: {}", unlabeled_samples.len());
        
        if model_accuracy >= target_accuracy {
            println!("      🎉 Target accuracy reached!");
            break;
        }
    }
    
    println!("\n   📊 Active Learning Summary:");
    println!("      Iterations: {}", iteration);
    println!("      Final accuracy: {:.1}%", model_accuracy * 100.0);
    println!("      Samples labeled: {}", iteration);
    println!("      Improvement: {:.1}%", (model_accuracy - 0.75) * 100.0);
    
    Ok(())
}
```

## 🚀 Tutorial 6: Production Deployment

Learn to deploy Veritas Nexus in production environments.

### 6.1 Production Configuration and Monitoring

```rust
// src/tutorial6_production.rs
use veritas_nexus::prelude::*;
use veritas_nexus::monitoring::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Tutorial 6: Production Deployment");
    println!("====================================\n");
    
    // Part 1: Production Configuration
    println!("⚙️  Part 1: Production Configuration");
    demonstrate_production_config().await?;
    
    println!("\n" + &"=".repeat(50) + "\n");
    
    // Part 2: Monitoring and Observability
    println!("📊 Part 2: Monitoring and Observability");
    demonstrate_monitoring().await?;
    
    println!("\n" + &"=".repeat(50) + "\n");
    
    // Part 3: Error Handling and Resilience
    println!("🛡️  Part 3: Error Handling and Resilience");
    demonstrate_resilience().await?;
    
    Ok(())
}

async fn demonstrate_production_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 Creating production-ready configuration:");
    
    // Production configuration with optimizations
    let production_config = ProductionConfig {
        // Performance settings
        enable_gpu: true,
        gpu_memory_limit_mb: 4096,
        cpu_threads: num_cpus::get(),
        batch_size: 32,
        enable_async_processing: true,
        
        // Memory management
        memory_pool_size_mb: 1024,
        enable_memory_mapping: true,
        gc_threshold_mb: 512,
        
        // Caching
        enable_result_caching: true,
        cache_size_mb: 256,
        cache_ttl_seconds: 3600,
        
        // Reliability
        max_retries: 3,
        timeout_seconds: 30,
        circuit_breaker_threshold: 10,
        
        // Monitoring
        enable_metrics: true,
        metrics_port: 9090,
        enable_health_checks: true,
        health_check_interval_seconds: 30,
        
        // Security
        enable_tls: true,
        cert_path: "/etc/ssl/certs/veritas.crt".to_string(),
        key_path: "/etc/ssl/private/veritas.key".to_string(),
        api_rate_limit_per_minute: 1000,
    };
    
    println!("   ✅ GPU acceleration: {}", production_config.enable_gpu);
    println!("   ✅ Memory pool: {}MB", production_config.memory_pool_size_mb);
    println!("   ✅ Result caching: {}", production_config.enable_result_caching);
    println!("   ✅ TLS encryption: {}", production_config.enable_tls);
    println!("   ✅ Rate limiting: {} req/min", production_config.api_rate_limit_per_minute);
    
    // Create production detector
    let detector = LieDetector::builder()
        .with_production_config(production_config)
        .with_vision(VisionConfig {
            enable_face_detection: true,
            enable_micro_expressions: true,
            enable_eye_tracking: false, // Disabled for performance
            model_precision: ModelPrecision::Balanced,
        })
        .with_audio(AudioConfig {
            sample_rate: 16000, // Standard quality for production
            enable_pitch_analysis: true,
            enable_stress_detection: true,
            enable_voice_quality: false, // Disabled for performance
        })
        .with_text(TextConfig {
            model_type: TextModel::RoBerta, // Best accuracy
            enable_linguistic_analysis: true,
            enable_sentiment_analysis: true,
            language: "en".to_string(),
        })
        .with_fusion_strategy(FusionStrategy::AttentionBased)
        .build()
        .await?;
    
    println!("   🚀 Production detector initialized successfully");
    
    // Test with sample production workload
    println!("\n   🧪 Testing production workload:");
    
    let test_inputs = vec![
        "I was definitely not involved in that incident.",
        "The meeting went exactly as planned.",
        "I have no knowledge of those events.",
    ];
    
    for (i, text) in test_inputs.iter().enumerate() {
        let start_time = std::time::Instant::now();
        
        let input = AnalysisInput {
            video_path: None,
            audio_path: None,
            transcript: Some(text.to_string()),
            physiological_data: None,
        };
        
        let result = detector.analyze(input).await?;
        let latency = start_time.elapsed();
        
        println!("      Request {}: {:?} ({:.1}%) - {}ms", 
            i + 1, result.decision, result.confidence * 100.0, latency.as_millis());
    }
    
    Ok(())
}

async fn demonstrate_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Setting up production monitoring:");
    
    // Create metrics collector
    let metrics = Arc::new(MetricsCollector::new());
    
    // Start metrics server
    let metrics_server = MetricsServer::new(metrics.clone())
        .with_port(9090)
        .with_path("/metrics");
    
    println!("   ✅ Metrics server started on port 9090");
    
    // Create health checker
    let health_checker = HealthChecker::new()
        .with_checks(vec![
            HealthCheck::MemoryUsage { threshold_mb: 4096 },
            HealthCheck::DiskSpace { threshold_percent: 90.0 },
            HealthCheck::ModelAvailability,
            HealthCheck::DatabaseConnection,
        ])
        .with_interval(Duration::from_secs(30));
    
    println!("   ✅ Health checker configured");
    
    // Simulate monitoring data collection
    println!("\n   📈 Simulating monitoring data:");
    
    for i in 1..=10 {
        // Simulate request metrics
        let latency = 50 + (i * 10) + rand::random::<u64>() % 50;
        let success = rand::random::<f32>() > 0.05; // 95% success rate
        
        metrics.record_request(RequestMetrics {
            latency_ms: latency,
            success,
            endpoint: "/analyze".to_string(),
            timestamp: chrono::Utc::now(),
        });
        
        // Simulate system metrics
        let cpu_usage = 30.0 + (i as f32 * 2.0) + rand::random::<f32>() * 20.0;
        let memory_usage = 1024 + (i * 50) + rand::random::<u64>() % 200;
        let gpu_usage = 40.0 + rand::random::<f32>() * 30.0;
        
        metrics.record_system(SystemMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: memory_usage,
            gpu_usage_percent: gpu_usage,
            timestamp: chrono::Utc::now(),
        });
        
        println!("      Sample {}: {}ms latency, {:.1}% CPU, {}MB RAM, {:.1}% GPU", 
            i, latency, cpu_usage, memory_usage, gpu_usage);
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // Generate monitoring report
    let report = metrics.generate_report().await;
    
    println!("\n   📊 Monitoring Report:");
    println!("      Average latency: {:.1}ms", report.avg_latency_ms);
    println!("      Success rate: {:.1}%", report.success_rate * 100.0);
    println!("      Peak CPU usage: {:.1}%", report.peak_cpu_usage);
    println!("      Peak memory usage: {}MB", report.peak_memory_usage);
    println!("      Average GPU usage: {:.1}%", report.avg_gpu_usage);
    
    // Check alerts
    if report.avg_latency_ms > 200.0 {
        println!("      🚨 ALERT: High latency detected!");
    }
    if report.success_rate < 0.95 {
        println!("      🚨 ALERT: Low success rate detected!");
    }
    if report.peak_memory_usage > 3000 {
        println!("      🚨 ALERT: High memory usage detected!");
    }
    
    Ok(())
}

async fn demonstrate_resilience() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  Demonstrating error handling and resilience:");
    
    // Create resilient detector with fault tolerance
    let resilient_detector = ResilientDetector::new()
        .with_circuit_breaker(CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_timeout_seconds: 30,
            half_open_max_requests: 3,
        })
        .with_retry_policy(RetryPolicy {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        })
        .with_timeout(Duration::from_secs(30))
        .with_fallback_strategy(FallbackStrategy::CachedResults)
        .build()
        .await?;
    
    println!("   ✅ Resilient detector configured");
    
    // Simulate various error scenarios
    let error_scenarios = vec![
        ("Network timeout", ErrorType::NetworkTimeout),
        ("Model unavailable", ErrorType::ModelUnavailable),
        ("Memory exhausted", ErrorType::MemoryExhausted),
        ("Invalid input", ErrorType::InvalidInput),
        ("GPU error", ErrorType::GpuError),
    ];
    
    println!("\n   🧪 Testing error scenarios:");
    
    for (scenario_name, error_type) in error_scenarios {
        println!("      Testing: {}", scenario_name);
        
        let input = AnalysisInput {
            video_path: None,
            audio_path: None,
            transcript: Some("Test input for error scenario".to_string()),
            physiological_data: None,
        };
        
        // Simulate error injection
        let result = resilient_detector.analyze_with_error_injection(input, error_type).await;
        
        match result {
            Ok(analysis_result) => {
                println!("         ✅ Recovered: {:?} ({:.1}%)", 
                    analysis_result.decision, analysis_result.confidence * 100.0);
            },
            Err(e) => {
                println!("         ❌ Failed: {}", e);
            },
        }
        
        // Check circuit breaker state
        let cb_state = resilient_detector.circuit_breaker_state().await;
        println!("         Circuit breaker: {:?}", cb_state);
        
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    
    // Demonstrate graceful degradation
    println!("\n   📉 Testing graceful degradation:");
    
    let degraded_input = AnalysisInput {
        video_path: Some("unavailable_video.mp4".to_string()), // This will fail
        audio_path: None,
        transcript: Some("I was not involved in any wrongdoing.".to_string()),
        physiological_data: None,
    };
    
    let degraded_result = resilient_detector.analyze_with_degradation(degraded_input).await?;
    
    println!("      Degraded analysis result:");
    println!("         Decision: {:?}", degraded_result.decision);
    println!("         Confidence: {:.1}%", degraded_result.confidence * 100.0);
    println!("         Available modalities: {:?}", degraded_result.available_modalities);
    println!("         Degradation reason: {}", degraded_result.degradation_reason);
    
    Ok(())
}

// Mock structs for the tutorial
#[derive(Debug)]
struct ProductionConfig {
    enable_gpu: bool,
    gpu_memory_limit_mb: u64,
    cpu_threads: usize,
    batch_size: usize,
    enable_async_processing: bool,
    memory_pool_size_mb: u64,
    enable_memory_mapping: bool,
    gc_threshold_mb: u64,
    enable_result_caching: bool,
    cache_size_mb: u64,
    cache_ttl_seconds: u64,
    max_retries: usize,
    timeout_seconds: u64,
    circuit_breaker_threshold: usize,
    enable_metrics: bool,
    metrics_port: u16,
    enable_health_checks: bool,
    health_check_interval_seconds: u64,
    enable_tls: bool,
    cert_path: String,
    key_path: String,
    api_rate_limit_per_minute: u64,
}

struct MetricsCollector {
    // Implementation would store metrics
}

impl MetricsCollector {
    fn new() -> Self {
        Self {}
    }
    
    fn record_request(&self, _metrics: RequestMetrics) {
        // Record request metrics
    }
    
    fn record_system(&self, _metrics: SystemMetrics) {
        // Record system metrics
    }
    
    async fn generate_report(&self) -> MonitoringReport {
        // Generate monitoring report
        MonitoringReport {
            avg_latency_ms: 125.5,
            success_rate: 0.98,
            peak_cpu_usage: 85.2,
            peak_memory_usage: 2048,
            avg_gpu_usage: 65.3,
        }
    }
}

#[derive(Debug)]
struct RequestMetrics {
    latency_ms: u64,
    success: bool,
    endpoint: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct SystemMetrics {
    cpu_usage_percent: f32,
    memory_usage_mb: u64,
    gpu_usage_percent: f32,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct MonitoringReport {
    avg_latency_ms: f64,
    success_rate: f64,
    peak_cpu_usage: f32,
    peak_memory_usage: u64,
    avg_gpu_usage: f32,
}

struct MetricsServer {
    _metrics: Arc<MetricsCollector>,
    _port: u16,
    _path: String,
}

impl MetricsServer {
    fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            _metrics: metrics,
            _port: 9090,
            _path: "/metrics".to_string(),
        }
    }
    
    fn with_port(mut self, port: u16) -> Self {
        self._port = port;
        self
    }
    
    fn with_path(mut self, path: &str) -> Self {
        self._path = path.to_string();
        self
    }
}

struct HealthChecker;

impl HealthChecker {
    fn new() -> Self {
        Self
    }
    
    fn with_checks(self, _checks: Vec<HealthCheck>) -> Self {
        self
    }
    
    fn with_interval(self, _interval: Duration) -> Self {
        self
    }
}

#[derive(Debug)]
enum HealthCheck {
    MemoryUsage { threshold_mb: u64 },
    DiskSpace { threshold_percent: f32 },
    ModelAvailability,
    DatabaseConnection,
}

struct ResilientDetector;

impl ResilientDetector {
    fn new() -> Self {
        Self
    }
    
    fn with_circuit_breaker(self, _config: CircuitBreakerConfig) -> Self {
        self
    }
    
    fn with_retry_policy(self, _policy: RetryPolicy) -> Self {
        self
    }
    
    fn with_timeout(self, _timeout: Duration) -> Self {
        self
    }
    
    fn with_fallback_strategy(self, _strategy: FallbackStrategy) -> Self {
        self
    }
    
    async fn build(self) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(self)
    }
    
    async fn analyze_with_error_injection(&self, _input: AnalysisInput, error_type: ErrorType) -> Result<AnalysisResult, Box<dyn std::error::Error>> {
        // Simulate error scenarios
        match error_type {
            ErrorType::NetworkTimeout => {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Err("Network timeout - retrying with cached data".into())
            },
            ErrorType::ModelUnavailable => {
                Ok(AnalysisResult {
                    decision: DeceptionDecision::Uncertain,
                    confidence: 0.3,
                    modality_scores: ModalityScores {
                        vision_score: None,
                        audio_score: None,
                        text_score: Some(0.5),
                        physiological_score: None,
                    },
                    reasoning_trace: vec!["Fallback analysis used due to model unavailability".to_string()],
                    processing_time_ms: 50,
                })
            },
            _ => Err("Simulated error".into()),
        }
    }
    
    async fn circuit_breaker_state(&self) -> CircuitBreakerState {
        CircuitBreakerState::Closed
    }
    
    async fn analyze_with_degradation(&self, _input: AnalysisInput) -> Result<DegradedAnalysisResult, Box<dyn std::error::Error>> {
        Ok(DegradedAnalysisResult {
            decision: DeceptionDecision::TruthTelling,
            confidence: 0.7,
            available_modalities: vec!["text".to_string()],
            degradation_reason: "Video analysis unavailable".to_string(),
        })
    }
}

#[derive(Debug)]
struct CircuitBreakerConfig {
    failure_threshold: usize,
    recovery_timeout_seconds: u64,
    half_open_max_requests: usize,
}

#[derive(Debug)]
struct RetryPolicy {
    max_retries: usize,
    base_delay_ms: u64,
    max_delay_ms: u64,
    backoff_multiplier: f32,
}

#[derive(Debug)]
enum FallbackStrategy {
    CachedResults,
    SimplifiedModel,
    DefaultResponse,
}

#[derive(Debug)]
enum ErrorType {
    NetworkTimeout,
    ModelUnavailable,
    MemoryExhausted,
    InvalidInput,
    GpuError,
}

#[derive(Debug)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
struct DegradedAnalysisResult {
    decision: DeceptionDecision,
    confidence: f32,
    available_modalities: Vec<String>,
    degradation_reason: String,
}
```

## 🎉 Conclusion

Congratulations! You've completed the comprehensive Veritas Nexus tutorial. You've learned:

1. **Basic Analysis** - Text analysis and configuration options
2. **Multi-Modal Fusion** - Combining vision, audio, and text data
3. **Real-Time Processing** - Streaming analysis with live data sources
4. **Batch Processing** - Large-scale analysis workflows
5. **Advanced Features** - Custom analyzers, ensemble methods, and active learning
6. **Production Deployment** - Scalable, monitored, and resilient systems

## 🚀 Next Steps

Now that you've mastered the fundamentals, explore these advanced topics:

- **[Performance Guide](PERFORMANCE_GUIDE.md)** - Optimize for your specific use case
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Deploy to cloud and edge environments
- **[API Documentation](https://docs.rs/veritas-nexus)** - Deep dive into all available features
- **[Contributing Guide](CONTRIBUTING.md)** - Help improve Veritas Nexus

## 💡 Best Practices Recap

- Always validate input data quality
- Use appropriate fusion strategies for your use case
- Implement comprehensive error handling
- Monitor system performance in production
- Respect ethical guidelines and user privacy
- Maintain human oversight in decision-making

Happy analyzing! 🔍

---

*Need help? Check our [Troubleshooting Guide](TROUBLESHOOTING.md) or join our community discussions.*