# Veritas Nexus User Guide

This comprehensive guide covers all aspects of using Veritas Nexus for multi-modal lie detection, from basic concepts to advanced usage patterns.

## 📋 Table of Contents

1. [Core Concepts](#-core-concepts)
2. [System Architecture](#-system-architecture)
3. [Configuration](#-configuration)
4. [Modality Analyzers](#-modality-analyzers)
5. [Fusion Strategies](#-fusion-strategies)
6. [Real-Time Processing](#-real-time-processing)
7. [Batch Processing](#-batch-processing)
8. [Explainable AI](#-explainable-ai)
9. [Performance Optimization](#-performance-optimization)
10. [MCP Integration](#-mcp-integration)
11. [Advanced Features](#-advanced-features)
12. [Best Practices](#-best-practices)

## 🧠 Core Concepts

### Deception Detection Fundamentals

Veritas Nexus operates on the principle that deception creates detectable patterns across multiple modalities:

- **Cognitive Load**: Lying requires more mental effort than truth-telling
- **Emotional Stress**: Deception often triggers emotional responses
- **Behavioral Inconsistencies**: Misalignment between verbal and non-verbal cues
- **Physiological Changes**: Measurable changes in heart rate, breathing, etc.

### Multi-Modal Approach

```rust
use veritas_nexus::prelude::*;

// The system analyzes four primary modalities
#[derive(Debug)]
pub enum ModalityType {
    Vision,        // Facial expressions, eye movements, micro-expressions
    Audio,         // Voice stress, pitch variations, speaking patterns
    Text,          // Linguistic patterns, sentiment, complexity
    Physiological, // Heart rate, skin conductance, breathing
}
```

### Decision Framework

Each analysis produces:
- **Deception Score**: 0.0 (truthful) to 1.0 (deceptive)
- **Confidence Level**: How certain the system is about its assessment
- **Decision**: Final classification based on score and confidence
- **Reasoning Trace**: Step-by-step explanation of the decision process

## 🏗️ System Architecture

### Processing Pipeline

```
Input Data → Modality Analysis → Feature Extraction → Fusion → Decision
     ↓              ↓                    ↓            ↓        ↓
Validation →   Preprocessing →     Normalization → Weighting → Explanation
```

### Core Components

```rust
use veritas_nexus::prelude::*;

// Main detector interface
let detector = LieDetector::builder()
    .with_vision(VisionConfig::default())
    .with_audio(AudioConfig::default())
    .with_text(TextConfig::default())
    .with_fusion_strategy(FusionStrategy::AdaptiveWeight)
    .build()
    .await?;

// Analysis input structure
let input = AnalysisInput {
    video_path: Some("path/to/video.mp4".to_string()),
    audio_path: Some("path/to/audio.wav".to_string()),
    transcript: Some("Text to analyze".to_string()),
    physiological_data: Some(vec![72.5, 73.1, 74.2]), // Optional biometric data
};

// Perform analysis
let result = detector.analyze(input).await?;
```

## ⚙️ Configuration

### Basic Configuration

```rust
use veritas_nexus::config::*;

// Vision configuration
let vision_config = VisionConfig {
    enable_face_detection: true,
    enable_micro_expressions: true,
    enable_eye_tracking: false,
    model_precision: ModelPrecision::Balanced,
    confidence_threshold: 0.7,
    max_processing_time_ms: 1000,
};

// Audio configuration
let audio_config = AudioConfig {
    sample_rate: 16000,
    enable_pitch_analysis: true,
    enable_stress_detection: true,
    enable_voice_quality: true,
    noise_reduction: true,
    normalization: AudioNormalization::RMS,
};

// Text configuration
let text_config = TextConfig {
    model_type: TextModel::Bert,
    enable_linguistic_analysis: true,
    enable_sentiment_analysis: true,
    language: "en".to_string(),
    max_sequence_length: 512,
    batch_size: 16,
};
```

### Advanced Configuration

```rust
use veritas_nexus::advanced::*;

// Performance configuration
let perf_config = PerformanceConfig {
    num_threads: 8,
    enable_simd: true,
    memory_pool_size_mb: 512,
    cache_size: 1000,
    enable_profiling: false,
    optimization_level: OptimizationLevel::Aggressive,
};

// GPU configuration
let gpu_config = GpuConfig {
    enable_gpu: true,
    device_id: 0,
    memory_limit_mb: 4096,
    batch_size: 32,
    fp16_inference: true,
    async_execution: true,
};

// Build detector with advanced configuration
let detector = LieDetector::builder()
    .with_vision(vision_config)
    .with_audio(audio_config)
    .with_text(text_config)
    .with_performance_config(perf_config)
    .with_gpu_config(gpu_config)
    .build()
    .await?;
```

## 🎯 Modality Analyzers

### Vision Analysis

The vision analyzer processes facial expressions, micro-expressions, and eye movements.

```rust
use veritas_nexus::modalities::vision::*;

// Detailed vision configuration
let vision_config = VisionConfig {
    // Face detection settings
    enable_face_detection: true,
    face_detection_threshold: 0.8,
    max_faces: 1, // Primary subject only
    
    // Micro-expression analysis
    enable_micro_expressions: true,
    micro_expression_window_ms: 500,
    emotion_categories: vec![
        EmotionType::Fear,
        EmotionType::Surprise,
        EmotionType::Disgust,
        EmotionType::Contempt,
    ],
    
    // Eye tracking
    enable_eye_tracking: false, // Requires specialized hardware
    gaze_analysis: true,
    blink_analysis: true,
    
    // Performance settings
    model_precision: ModelPrecision::Balanced,
    frame_skip: 2, // Process every 2nd frame for performance
    roi_optimization: true, // Focus on face region
};

// Vision-specific analysis
let vision_analyzer = VisionAnalyzer::new(vision_config)?;
let video_frame = load_video_frame("frame.jpg")?;
let vision_result = vision_analyzer.analyze(&video_frame).await?;

println!("Vision Score: {:.3}", vision_result.deception_score);
println!("Detected emotions: {:?}", vision_result.emotions);
println!("Micro-expressions: {:?}", vision_result.micro_expressions);
```

### Audio Analysis

The audio analyzer examines voice stress patterns, pitch variations, and speaking characteristics.

```rust
use veritas_nexus::modalities::audio::*;

// Detailed audio configuration
let audio_config = AudioConfig {
    // Basic settings
    sample_rate: 16000,
    channels: 1, // Mono audio recommended
    
    // Feature extraction
    enable_pitch_analysis: true,
    pitch_range: PitchRange::Normal, // Normal, High, Low
    enable_stress_detection: true,
    stress_sensitivity: StressSensitivity::Medium,
    
    // Voice quality analysis
    enable_voice_quality: true,
    jitter_analysis: true,
    shimmer_analysis: true,
    harmonic_noise_ratio: true,
    
    // Speech patterns
    enable_pause_analysis: true,
    enable_rate_analysis: true,
    enable_volume_analysis: true,
    
    // Preprocessing
    noise_reduction: true,
    normalization: AudioNormalization::RMS,
    high_pass_filter: Some(80.0), // Remove low-frequency noise
    low_pass_filter: Some(8000.0), // Focus on speech frequencies
};

// Audio-specific analysis
let audio_analyzer = AudioAnalyzer::new(audio_config)?;
let audio_chunk = load_audio_chunk("audio.wav")?;
let audio_result = audio_analyzer.analyze(&audio_chunk).await?;

println!("Audio Score: {:.3}", audio_result.deception_score);
println!("Stress indicators: {:?}", audio_result.stress_features);
println!("Voice quality: {:.3}", audio_result.voice_quality_score);
```

### Text Analysis

The text analyzer uses advanced NLP techniques to detect deceptive language patterns.

```rust
use veritas_nexus::modalities::text::*;

// Detailed text configuration
let text_config = TextConfig {
    // Model selection
    model_type: TextModel::RoBerta, // Bert, RoBerta, DistilBert
    model_size: ModelSize::Base, // Base, Large
    
    // Language settings
    language: "en".to_string(),
    tokenization: TokenizationStrategy::WordPiece,
    max_sequence_length: 512,
    
    // Analysis features
    enable_linguistic_analysis: true,
    enable_sentiment_analysis: true,
    enable_complexity_analysis: true,
    enable_certainty_analysis: true,
    
    // Deception indicators
    detect_hedging: true, // "maybe", "perhaps", "I think"
    detect_distancing: true, // Third person references
    detect_overstatement: true, // Excessive emphasis
    detect_inconsistency: true, // Logical contradictions
    
    // Performance
    batch_size: 16,
    enable_caching: true,
    cache_size: 1000,
};

// Text-specific analysis
let text_analyzer = TextAnalyzer::new(text_config)?;
let transcript = "I definitely did not take any money from the register.";
let text_result = text_analyzer.analyze(transcript).await?;

println!("Text Score: {:.3}", text_result.deception_score);
println!("Linguistic features: {:?}", text_result.linguistic_features);
println!("Sentiment: {:?}", text_result.sentiment);
println!("Certainty markers: {:?}", text_result.certainty_indicators);
```

### Physiological Analysis

The physiological analyzer processes biometric data for stress and arousal indicators.

```rust
use veritas_nexus::modalities::physiological::*;

// Physiological data configuration
let physio_config = PhysiologicalConfig {
    // Data types
    enable_heart_rate: true,
    enable_skin_conductance: true,
    enable_breathing_rate: false, // Not always available
    enable_blood_pressure: false, // Requires specific sensors
    
    // Analysis settings
    baseline_duration_seconds: 30.0,
    analysis_window_seconds: 10.0,
    smoothing_factor: 0.1,
    
    // Thresholds
    stress_threshold: 0.7,
    arousal_threshold: 0.6,
    
    // Preprocessing
    enable_artifact_removal: true,
    enable_normalization: true,
    sampling_rate: 100.0, // Hz
};

// Physiological analysis
let physio_analyzer = PhysiologicalAnalyzer::new(physio_config)?;

// Sample data: heart rate measurements over time
let heart_rate_data = vec![72.5, 73.1, 74.2, 75.8, 77.1, 78.9, 80.2];
let skin_conductance_data = vec![2.1, 2.3, 2.4, 2.7, 2.9, 3.1, 3.3];

let physio_data = PhysiologicalData {
    heart_rate: Some(heart_rate_data),
    skin_conductance: Some(skin_conductance_data),
    breathing_rate: None,
    blood_pressure: None,
    timestamp: chrono::Utc::now(),
};

let physio_result = physio_analyzer.analyze(&physio_data).await?;

println!("Physiological Score: {:.3}", physio_result.deception_score);
println!("Stress level: {:.3}", physio_result.stress_level);
println!("Arousal level: {:.3}", physio_result.arousal_level);
```

## 🔀 Fusion Strategies

Fusion strategies combine evidence from multiple modalities into a final decision.

### Equal Weight Fusion

```rust
use veritas_nexus::fusion::*;

// Simple equal weighting
let fusion_strategy = FusionStrategy::EqualWeight;

// All modalities contribute equally
let detector = LieDetector::builder()
    .with_fusion_strategy(fusion_strategy)
    .build()
    .await?;
```

### Adaptive Weight Fusion

```rust
// Adaptive weighting based on modality reliability
let fusion_config = AdaptiveWeightConfig {
    initial_weights: vec![0.3, 0.3, 0.2, 0.2], // Vision, Audio, Text, Physio
    learning_rate: 0.01,
    adaptation_window: 100, // Adapt based on last 100 samples
    min_weight: 0.05,
    max_weight: 0.7,
};

let fusion_strategy = FusionStrategy::AdaptiveWeight(fusion_config);
```

### Attention-Based Fusion

```rust
// Attention mechanism learns optimal weights
let attention_config = AttentionFusionConfig {
    attention_heads: 4,
    hidden_dimension: 128,
    dropout_rate: 0.1,
    temperature: 1.0,
    enable_self_attention: true,
    enable_cross_attention: true,
};

let fusion_strategy = FusionStrategy::AttentionBased(attention_config);
```

### Context-Aware Fusion

```rust
// Fusion weights adapt based on context
let context_config = ContextAwareFusionConfig {
    context_features: vec![
        ContextFeature::EnvironmentType,
        ContextFeature::SubjectAge,
        ContextFeature::InterviewType,
        ContextFeature::TimeOfDay,
    ],
    context_embedding_dim: 64,
    fusion_layer_sizes: vec![256, 128, 64],
    enable_meta_learning: true,
};

let fusion_strategy = FusionStrategy::ContextAware(context_config);
```

## 🌊 Real-Time Processing

### Streaming Pipeline Setup

```rust
use veritas_nexus::streaming::*;

// Configure streaming pipeline
let streaming_config = StreamingConfig {
    target_fps: 30.0,
    audio_chunk_size_ms: 100,
    sync_window_ms: 200,
    max_latency_ms: 300,
    buffer_size: 128,
    enable_adaptive_quality: true,
    quality_degradation_threshold: 0.8,
};

// Create and configure pipeline
let pipeline = StreamingPipeline::builder()
    .with_config(streaming_config)
    .with_video_source(VideoSource::Camera(0)) // Use camera 0
    .with_audio_source(AudioSource::Microphone(None)) // Default microphone
    .with_text_source(TextSource::SpeechToText(SttConfig::default()))
    .with_output_handler(|result| {
        println!("Real-time result: {:?} ({:.1}%)", 
            result.decision, 
            result.confidence * 100.0);
    })
    .build()?;

// Start streaming
pipeline.start().await?;

// Monitor for 60 seconds
tokio::time::sleep(Duration::from_secs(60)).await;

// Stop streaming
pipeline.stop().await?;
```

### Custom Data Sources

```rust
use veritas_nexus::streaming::sources::*;

// Custom video source
struct CustomVideoSource {
    // Your video source implementation
}

impl VideoSourceTrait for CustomVideoSource {
    async fn next_frame(&mut self) -> Result<Option<VideoFrame>> {
        // Your implementation
        todo!()
    }
}

// Custom audio source
struct CustomAudioSource {
    // Your audio source implementation
}

impl AudioSourceTrait for CustomAudioSource {
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
        // Your implementation
        todo!()
    }
}

// Use custom sources
let pipeline = StreamingPipeline::builder()
    .with_video_source(VideoSource::Custom(Box::new(CustomVideoSource::new())))
    .with_audio_source(AudioSource::Custom(Box::new(CustomAudioSource::new())))
    .build()?;
```

### Real-Time Configuration Tuning

```rust
// Performance-optimized streaming
let fast_config = StreamingConfig {
    target_fps: 15.0, // Reduced FPS for speed
    audio_chunk_size_ms: 200, // Larger chunks
    sync_window_ms: 300,
    max_latency_ms: 200,
    buffer_size: 64,
    enable_adaptive_quality: true,
    quality_degradation_threshold: 0.6, // More aggressive quality reduction
};

// Quality-optimized streaming
let quality_config = StreamingConfig {
    target_fps: 60.0, // High FPS for detail
    audio_chunk_size_ms: 50, // Smaller chunks for precision
    sync_window_ms: 100,
    max_latency_ms: 500,
    buffer_size: 256,
    enable_adaptive_quality: false, // Maintain consistent quality
    quality_degradation_threshold: 0.9,
};
```

## 📊 Batch Processing

### Large-Scale Analysis

```rust
use veritas_nexus::batch::*;

// Configure batch processor
let batch_config = BatchConfig {
    batch_size: 32,
    max_concurrent_batches: 4,
    enable_progress_reporting: true,
    checkpoint_interval: 100, // Save progress every 100 items
    error_handling: ErrorHandling::ContinueOnError,
    output_format: OutputFormat::JsonLines,
};

let batch_processor = BatchProcessor::new(batch_config);

// Process multiple files
let input_files = vec![
    "interview1.mp4",
    "interview2.mp4", 
    "interview3.mp4",
    // ... more files
];

// Process with progress tracking
let results = batch_processor
    .process_files(&input_files)
    .with_progress_callback(|processed, total| {
        println!("Progress: {}/{} ({:.1}%)", processed, total, 
            (processed as f32 / total as f32) * 100.0);
    })
    .await?;

// Save results
batch_processor.save_results(&results, "batch_results.jsonl").await?;
```

### Database Integration

```rust
use veritas_nexus::batch::database::*;

// Database configuration
let db_config = DatabaseConfig {
    connection_string: "postgresql://user:pass@localhost/veritas".to_string(),
    max_connections: 10,
    timeout_seconds: 30,
    batch_insert_size: 1000,
};

let db_processor = DatabaseBatchProcessor::new(db_config).await?;

// Process data from database
let query = "SELECT id, video_path, audio_path, transcript FROM interviews WHERE processed = false";
let results = db_processor.process_query(query).await?;

// Update database with results
for result in results {
    db_processor.update_result(&result).await?;
}
```

### Distributed Processing

```rust
use veritas_nexus::distributed::*;

// Distributed processing configuration
let distributed_config = DistributedConfig {
    coordinator_address: "coordinator:8080".to_string(),
    worker_id: "worker-1".to_string(),
    max_tasks_per_worker: 10,
    heartbeat_interval_seconds: 30,
    enable_fault_tolerance: true,
};

// Start worker node
let worker = DistributedWorker::new(distributed_config).await?;
worker.start().await?;

// On coordinator node
let coordinator = DistributedCoordinator::new().await?;
let job = DistributedJob {
    id: "job-123".to_string(),
    input_files: input_files,
    config: batch_config,
};

let results = coordinator.submit_job(job).await?;
```

## 🔍 Explainable AI

### Understanding Reasoning Traces

```rust
use veritas_nexus::explainable::*;

// Analyze with detailed explanations
let input = AnalysisInput::from_text("I didn't take the money, I swear!");
let result = detector.analyze(input).await?;

// Access reasoning trace
for step in &result.reasoning_trace {
    match &step.step_type {
        ReasoningStepType::Observe => {
            println!("📊 Observation: {}", step.description);
        },
        ReasoningStepType::Think => {
            println!("🤔 Thought: {}", step.description);
        },
        ReasoningStepType::Act => {
            println!("⚡ Action: {}", step.description);
        },
        ReasoningStepType::Explain => {
            println!("💡 Explanation: {}", step.description);
        },
    }
}

// Generate detailed explanation
let explanation = ExplanationGenerator::new()
    .with_target_audience(TargetAudience::Expert)
    .with_detail_level(DetailLevel::Comprehensive)
    .generate(&result)?;

println!("Detailed Explanation:\n{}", explanation.formatted_text);
```

### Feature Attribution

```rust
use veritas_nexus::attribution::*;

// Analyze feature importance
let attributor = FeatureAttributor::new(AttributionMethod::LIME);
let attribution_result = attributor.analyze(&detector, &input).await?;

// Print feature contributions
for (feature, importance) in &attribution_result.feature_importances {
    println!("{}: {:.4}", feature, importance);
}

// Visualize attributions (if visualization features enabled)
#[cfg(feature = "visualization")]
{
    let visualizer = AttributionVisualizer::new();
    let chart = visualizer.create_importance_chart(&attribution_result)?;
    chart.save("feature_importance.png")?;
}
```

### Counterfactual Analysis

```rust
use veritas_nexus::counterfactual::*;

// Generate counterfactual explanations
let counterfactual_generator = CounterfactualGenerator::new()
    .with_max_changes(3)
    .with_distance_metric(DistanceMetric::Euclidean);

let counterfactuals = counterfactual_generator
    .generate(&detector, &input)
    .await?;

for cf in &counterfactuals {
    println!("If {} was changed from {:.3} to {:.3}, decision would be: {:?}",
        cf.feature_name, cf.original_value, cf.modified_value, cf.new_decision);
}
```

## ⚡ Performance Optimization

### CPU Optimization

```rust
use veritas_nexus::optimization::*;

// Enable CPU optimizations
let cpu_config = CpuOptimizationConfig {
    enable_simd: true,
    simd_instruction_set: SIMDInstructionSet::Auto, // Auto-detect best available
    num_threads: num_cpus::get(),
    thread_affinity: ThreadAffinity::Auto,
    enable_vectorization: true,
    memory_alignment: MemoryAlignment::SIMD, // 32-byte alignment for SIMD
};

let detector = LieDetector::builder()
    .with_cpu_optimization(cpu_config)
    .build()
    .await?;
```

### Memory Management

```rust
use veritas_nexus::memory::*;

// Configure memory management
let memory_config = MemoryConfig {
    // Object pooling
    enable_object_pooling: true,
    small_object_pool_size: 1000,
    medium_object_pool_size: 100,
    large_object_pool_size: 10,
    
    // Memory mapping
    enable_memory_mapping: true,
    mmap_threshold_bytes: 1024 * 1024, // 1MB
    
    // Garbage collection
    gc_strategy: GCStrategy::Incremental,
    gc_threshold_mb: 100,
    
    // Caching
    enable_result_caching: true,
    cache_size_mb: 256,
    cache_ttl_seconds: 3600,
};

let detector = LieDetector::builder()
    .with_memory_config(memory_config)
    .build()
    .await?;
```

### GPU Acceleration

```rust
use veritas_nexus::gpu::*;

// Configure GPU acceleration
let gpu_config = GpuConfig {
    enable_gpu: true,
    device_id: 0,
    memory_limit_mb: 4096,
    
    // Performance settings
    batch_size: 32,
    fp16_inference: true,
    tensor_cores: true,
    async_execution: true,
    
    // Memory management
    enable_memory_pool: true,
    pool_size_mb: 2048,
    enable_unified_memory: false,
    
    // Multi-GPU settings
    enable_multi_gpu: false,
    gpu_ids: vec![0], // Single GPU
    data_parallel: false,
};

let detector = LieDetector::builder()
    .with_gpu_config(gpu_config)
    .build()
    .await?;
```

## 🔗 MCP Integration

### Server Setup

```rust
use veritas_nexus::mcp::*;

// Configure MCP server
let server_config = McpServerConfig {
    host: "0.0.0.0".to_string(),
    port: 8080,
    enable_tls: true,
    cert_path: Some("cert.pem".to_string()),
    key_path: Some("key.pem".to_string()),
    
    // Authentication
    auth_provider: Some(AuthProvider::ApiKey {
        keys: vec!["api-key-1".to_string(), "api-key-2".to_string()],
    }),
    
    // Rate limiting
    enable_rate_limiting: true,
    requests_per_minute: 60,
    burst_size: 10,
    
    // Monitoring
    enable_metrics: true,
    metrics_endpoint: "/metrics".to_string(),
    enable_health_check: true,
    health_endpoint: "/health".to_string(),
};

// Start MCP server
let server = McpServer::new(server_config).await?;
server.start().await?;

println!("MCP server running at https://localhost:8080");
```

### Tool Registration

```rust
use veritas_nexus::mcp::tools::*;

// Register analysis tool
let analysis_tool = AnalysisTool::new()
    .with_name("analyze_deception")
    .with_description("Analyze multi-modal input for deception indicators")
    .with_parameters(vec![
        Parameter::new("video_path").optional(),
        Parameter::new("audio_path").optional(),
        Parameter::new("text").optional(),
        Parameter::new("config").optional(),
    ]);

server.register_tool(analysis_tool).await?;

// Register batch processing tool
let batch_tool = BatchProcessingTool::new()
    .with_name("batch_analyze")
    .with_description("Process multiple files in batch")
    .with_parameters(vec![
        Parameter::new("input_files").required(),
        Parameter::new("output_format").optional(),
        Parameter::new("batch_size").optional(),
    ]);

server.register_tool(batch_tool).await?;

// Register training tool
let training_tool = TrainingTool::new()
    .with_name("train_model")
    .with_description("Train or fine-tune detection models")
    .with_parameters(vec![
        Parameter::new("training_data").required(),
        Parameter::new("model_type").required(),
        Parameter::new("epochs").optional(),
        Parameter::new("learning_rate").optional(),
    ]);

server.register_tool(training_tool).await?;
```

### Resource Management

```rust
use veritas_nexus::mcp::resources::*;

// Register model resources
let model_resource = ModelResource::new()
    .with_uri("models://vision/face_detection")
    .with_name("Face Detection Model")
    .with_description("Pre-trained face detection model")
    .with_mime_type("application/octet-stream");

server.register_resource(model_resource).await?;

// Register data resources  
let data_resource = DataResource::new()
    .with_uri("data://samples/interview_dataset")
    .with_name("Interview Dataset")
    .with_description("Labeled interview data for training")
    .with_mime_type("application/json");

server.register_resource(data_resource).await?;
```

## 🔧 Advanced Features

### Custom Modality Analyzers

```rust
use veritas_nexus::modalities::custom::*;

// Implement custom analyzer
struct CustomAnalyzer {
    config: CustomConfig,
}

#[async_trait]
impl ModalityAnalyzer<f32> for CustomAnalyzer {
    type Input = CustomInput;
    type Output = CustomOutput;
    type Config = CustomConfig;
    
    async fn analyze(&self, input: &Self::Input) -> Result<Self::Output> {
        // Your custom analysis logic
        todo!()
    }
    
    fn confidence(&self) -> f32 {
        // Return confidence level
        todo!()
    }
    
    fn explain(&self) -> ExplanationTrace {
        // Generate explanation
        todo!()
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// Register custom analyzer
let detector = LieDetector::builder()
    .with_custom_analyzer("custom_modality", CustomAnalyzer::new())
    .build()
    .await?;
```

### Active Learning

```rust
use veritas_nexus::learning::*;

// Configure active learning
let active_learning_config = ActiveLearningConfig {
    uncertainty_strategy: UncertaintyStrategy::LeastConfident,
    batch_size: 10,
    labeling_budget: 1000,
    retraining_frequency: RetrainingFrequency::AfterBatch,
    stopping_criteria: StoppingCriteria::PerformanceThreshold(0.95),
};

// Setup active learning loop
let active_learner = ActiveLearner::new(active_learning_config);
let mut detector = LieDetector::new().await?;

loop {
    // Get uncertain samples
    let uncertain_samples = active_learner
        .select_samples(&detector, &unlabeled_data)
        .await?;
    
    if uncertain_samples.is_empty() {
        break; // No more uncertain samples
    }
    
    // Get labels (from human annotators)
    let labels = get_labels_from_annotators(&uncertain_samples).await?;
    
    // Retrain model
    detector = detector.retrain_with_samples(&uncertain_samples, &labels).await?;
    
    // Evaluate performance
    let performance = evaluate_model(&detector, &test_data).await?;
    println!("Model performance: {:.3}", performance.accuracy);
    
    if performance.accuracy > 0.95 {
        break; // Stopping criteria met
    }
}
```

### Ensemble Methods

```rust
use veritas_nexus::ensemble::*;

// Create ensemble of detectors
let ensemble_config = EnsembleConfig {
    voting_strategy: VotingStrategy::WeightedAverage,
    diversity_threshold: 0.1,
    max_ensemble_size: 5,
    selection_strategy: SelectionStrategy::DiversityBased,
};

let ensemble = EnsembleDetector::builder()
    .with_config(ensemble_config)
    .add_detector("detector_1", detector_1)
    .add_detector("detector_2", detector_2)
    .add_detector("detector_3", detector_3)
    .build()?;

// Analyze with ensemble
let result = ensemble.analyze(input).await?;
println!("Ensemble decision: {:?} ({:.1}%)", 
    result.decision, result.confidence * 100.0);

// Get individual detector contributions
for (name, contribution) in &result.detector_contributions {
    println!("{}: {:.3}", name, contribution);
}
```

## 📈 Best Practices

### Data Quality

```rust
use veritas_nexus::validation::*;

// Validate input data quality
let validator = DataValidator::new()
    .with_video_requirements(VideoRequirements {
        min_resolution: (640, 480),
        max_resolution: (1920, 1080),
        min_fps: 15.0,
        max_fps: 60.0,
        required_codecs: vec!["h264", "vp9"],
    })
    .with_audio_requirements(AudioRequirements {
        min_sample_rate: 8000,
        max_sample_rate: 48000,
        required_channels: 1,
        min_duration_seconds: 5.0,
        max_duration_seconds: 3600.0,
    })
    .with_text_requirements(TextRequirements {
        min_length: 10,
        max_length: 10000,
        required_languages: vec!["en"],
        encoding: TextEncoding::UTF8,
    });

// Validate before analysis
let validation_result = validator.validate(&input).await?;
if !validation_result.is_valid() {
    for error in &validation_result.errors {
        eprintln!("Validation error: {}", error);
    }
    return Err("Input validation failed".into());
}
```

### Error Handling

```rust
use veritas_nexus::error::*;

// Comprehensive error handling
async fn robust_analysis(detector: &LieDetector, input: AnalysisInput) -> Result<AnalysisResult> {
    let result = match detector.analyze(input).await {
        Ok(result) => result,
        Err(VeritasError::InvalidInput(msg)) => {
            eprintln!("Input error: {}", msg);
            return Err("Please check your input data".into());
        },
        Err(VeritasError::ModelLoadError(msg)) => {
            eprintln!("Model error: {}", msg);
            return Err("Model loading failed, please restart".into());
        },
        Err(VeritasError::ProcessingError(msg)) => {
            eprintln!("Processing error: {}", msg);
            // Try with reduced quality settings
            let fallback_result = detector.analyze_with_fallback(input).await?;
            return Ok(fallback_result);
        },
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            return Err("Analysis failed".into());
        },
    };
    
    // Validate result quality
    if result.confidence < 0.3 {
        eprintln!("Warning: Low confidence result ({:.1}%)", result.confidence * 100.0);
    }
    
    Ok(result)
}
```

### Performance Monitoring

```rust
use veritas_nexus::monitoring::*;

// Setup performance monitoring
let monitor = PerformanceMonitor::new()
    .with_metrics(vec![
        Metric::Latency,
        Metric::Throughput,
        Metric::MemoryUsage,
        Metric::GpuUtilization,
        Metric::Accuracy,
    ])
    .with_alert_thresholds(AlertThresholds {
        max_latency_ms: 1000,
        min_throughput_fps: 10.0,
        max_memory_usage_mb: 4096,
        min_accuracy: 0.8,
    })
    .with_reporting_interval(Duration::from_minutes(5));

// Start monitoring
monitor.start().await?;

// Register alert handler
monitor.on_alert(|alert| {
    match alert.metric {
        Metric::Latency => {
            println!("High latency detected: {}ms", alert.value);
            // Take corrective action
        },
        Metric::Accuracy => {
            println!("Low accuracy detected: {:.3}", alert.value);
            // Trigger model retraining
        },
        _ => {},
    }
});
```

### Configuration Management

```rust
use veritas_nexus::config::*;

// Environment-specific configurations
let config = match std::env::var("VERITAS_ENV").as_deref() {
    Ok("production") => Config::production(),
    Ok("staging") => Config::staging(),
    Ok("development") => Config::development(),
    _ => Config::default(),
};

// Load from file
let config = Config::from_file("veritas_config.toml")?;

// Override with environment variables
let config = config.with_env_overrides();

// Validate configuration
config.validate()?;

// Apply configuration
let detector = LieDetector::with_config(config).await?;
```

---

This user guide provides comprehensive coverage of all Veritas Nexus features and capabilities. For additional help, see the [Troubleshooting Guide](TROUBLESHOOTING.md) or consult the [API documentation](https://docs.rs/veritas-nexus).