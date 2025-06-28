# API Reference

This section provides comprehensive API documentation for all public interfaces in Veritas Nexus.

## Module Overview

### Core Modules

| Module | Description | Key Types |
|--------|-------------|-----------|
| [`engine`](engine.md) | Main analysis engine and coordination | `LieDetector`, `AnalysisEngine` |
| [`types`](types.md) | Core data types and structures | `AnalysisInput`, `AnalysisResult`, `DeceptionDecision` |
| [`fusion`](fusion.md) | Multi-modal fusion strategies | `FusionStrategy`, `FusionManager`, `FusionResult` |
| [`mcp`](mcp.md) | Model Context Protocol integration | `McpServer`, `McpTool`, `McpResource` |

### Modality Modules

| Module | Description | Key Types |
|--------|-------------|-----------|
| [`modalities::text`](modalities/text.md) | Text analysis and NLP | `TextAnalyzer`, `TextInput`, `TextResult` |
| [`modalities::vision`](modalities/vision.md) | Computer vision analysis | `VisionAnalyzer`, `VideoFrame`, `VisionFeatures` |
| [`modalities::audio`](modalities/audio.md) | Audio processing and analysis | `AudioAnalyzer`, `AudioChunk`, `AudioFeatures` |
| [`modalities::physiological`](modalities/physiological.md) | Physiological signal processing | `PhysiologicalAnalyzer`, `PhysioData` |

### Utility Modules

| Module | Description | Key Types |
|--------|-------------|-----------|
| [`optimization`](optimization.md) | Performance optimization | `GpuAcceleration`, `CacheManager`, `MemoryPool` |
| [`streaming`](streaming.md) | Real-time processing | `StreamingPipeline`, `RingBuffer`, `TemporalSync` |
| [`reasoning`](reasoning.md) | Explainable AI and reasoning | `ExplanationEngine`, `ReasoningTrace` |
| [`training`](training.md) | Model training and fine-tuning | `CascadeTrainer`, `TrainingConfig` |

## Quick Reference

### Essential Imports

```rust
// Core functionality
use veritas_nexus::prelude::*;

// Specific modules
use veritas_nexus::{
    engine::{LieDetector, AnalysisEngine},
    types::{AnalysisInput, AnalysisResult, DeceptionDecision},
    fusion::{FusionStrategy, FusionManager},
    modalities::{
        text::{TextAnalyzer, TextInput},
        vision::{VisionAnalyzer, VideoFrame},
        audio::{AudioAnalyzer, AudioChunk},
    },
};
```

### Common Patterns

#### Basic Analysis

```rust
use veritas_nexus::prelude::*;

async fn analyze_sample() -> Result<()> {
    let detector = LieDetector::new().await?;
    
    let input = AnalysisInput::new()
        .with_text("Sample text to analyze")
        .with_video_path("video.mp4");
    
    let result = detector.analyze(&input).await?;
    
    match result.decision {
        DeceptionDecision::Truthful => println!("Subject appears truthful"),
        DeceptionDecision::Deceptive => println!("Deception detected"),
        DeceptionDecision::Uncertain => println!("Analysis uncertain"),
        DeceptionDecision::InsufficientData => println!("Need more data"),
    }
    
    Ok(())
}
```

#### Batch Processing

```rust
use veritas_nexus::prelude::*;

async fn batch_analysis(samples: Vec<AnalysisInput>) -> Result<Vec<AnalysisResult>> {
    let detector = LieDetector::builder()
        .with_batch_size(32)
        .with_gpu_acceleration(true)
        .build()
        .await?;
    
    let mut results = Vec::new();
    
    for batch in samples.chunks(32) {
        let batch_results = detector.analyze_batch(batch).await?;
        results.extend(batch_results);
    }
    
    Ok(results)
}
```

#### Real-time Streaming

```rust
use veritas_nexus::prelude::*;
use veritas_nexus::streaming::StreamingPipeline;

async fn streaming_analysis() -> Result<()> {
    let config = StreamingConfig {
        target_fps: 30.0,
        sync_window_ms: 200,
        max_latency_ms: 500,
        ..Default::default()
    };
    
    let pipeline = StreamingPipeline::new(config);
    pipeline.start().await?;
    
    // Process streaming data
    loop {
        if let Some(result) = pipeline.try_get_result().await {
            println!("Real-time result: {:?}", result.decision);
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}
```

#### Custom Fusion Strategy

```rust
use veritas_nexus::fusion::{FusionStrategy, FusionResult};

struct CustomFusion {
    weights: HashMap<ModalityType, f32>,
}

impl FusionStrategy for CustomFusion {
    fn fuse(&self, inputs: &MultiModalInput) -> Result<FusionResult> {
        // Custom fusion logic here
        todo!()
    }
}

async fn custom_fusion_example() -> Result<()> {
    let detector = LieDetector::builder()
        .with_fusion_strategy(Box::new(CustomFusion {
            weights: [
                (ModalityType::Vision, 0.4),
                (ModalityType::Audio, 0.3),
                (ModalityType::Text, 0.3),
            ].into_iter().collect(),
        }))
        .build()
        .await?;
    
    // Use detector with custom fusion
    Ok(())
}
```

## Error Handling

All APIs use the standard Rust `Result<T, E>` pattern for error handling:

```rust
use veritas_nexus::{Result, VeritasError};

async fn handle_errors() -> Result<()> {
    let detector = LieDetector::new().await?;
    
    let input = AnalysisInput::new()
        .with_text("Sample text");
    
    match detector.analyze(&input).await {
        Ok(result) => {
            println!("Analysis successful: {:?}", result.decision);
        }
        Err(VeritasError::InvalidInput(msg)) => {
            eprintln!("Invalid input: {}", msg);
        }
        Err(VeritasError::ModelError(msg)) => {
            eprintln!("Model error: {}", msg);
        }
        Err(VeritasError::ProcessingError(msg)) => {
            eprintln!("Processing error: {}", msg);
        }
        Err(e) => {
            eprintln!("Other error: {}", e);
        }
    }
    
    Ok(())
}
```

## Configuration

### Global Configuration

```rust
use veritas_nexus::config::{GlobalConfig, LogLevel};

let config = GlobalConfig {
    log_level: LogLevel::Info,
    enable_metrics: true,
    cache_size: 1000,
    max_concurrent_analyses: 10,
    gpu_device_id: Some(0),
    model_path: "/path/to/models".into(),
    temp_dir: "/tmp/veritas".into(),
};

veritas_nexus::init_with_config(config).await?;
```

### Per-Analysis Configuration

```rust
use veritas_nexus::engine::AnalysisConfig;

let analysis_config = AnalysisConfig {
    timeout_ms: 30000,
    quality_threshold: 0.7,
    enable_explanations: true,
    fusion_strategy: "attention".to_string(),
    max_processing_time_ms: 10000,
};

let result = detector
    .analyze_with_config(&input, &analysis_config)
    .await?;
```

## Performance Considerations

### Memory Management

```rust
// Configure memory limits
let detector = LieDetector::builder()
    .with_memory_limit_mb(2048)
    .with_cache_size(500)
    .build()
    .await?;
```

### GPU Acceleration

```rust
// Enable GPU acceleration
let detector = LieDetector::builder()
    .with_gpu_acceleration(true)
    .with_gpu_device(0)
    .build()
    .await?;
```

### Batch Optimization

```rust
// Optimize for batch processing
let detector = LieDetector::builder()
    .with_batch_size(64)
    .with_parallel_processing(true)
    .with_prefetch_enabled(true)
    .build()
    .await?;
```

## Thread Safety

All public APIs are thread-safe and can be used concurrently:

```rust
use std::sync::Arc;
use tokio::task;

async fn concurrent_analysis() -> Result<()> {
    let detector = Arc::new(LieDetector::new().await?);
    
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let detector_clone = Arc::clone(&detector);
        let handle = task::spawn(async move {
            let input = AnalysisInput::new()
                .with_text(&format!("Sample text {}", i));
            
            detector_clone.analyze(&input).await
        });
        handles.push(handle);
    }
    
    // Wait for all analyses to complete
    for handle in handles {
        let result = handle.await??;
        println!("Result: {:?}", result.decision);
    }
    
    Ok(())
}
```

## Feature Flags

Veritas Nexus supports optional features that can be enabled in `Cargo.toml`:

```toml
[dependencies]
veritas-nexus = { version = "0.1.0", features = [
    "gpu-acceleration",    # Enable GPU processing
    "mcp-server",         # Enable MCP server functionality  
    "streaming",          # Enable real-time streaming
    "training",           # Enable model training
    "visualization",      # Enable result visualization
    "metrics",           # Enable performance metrics
] }
```

## Next Steps

- Review individual module documentation for detailed API information
- Check the [examples](../../examples/) for practical usage patterns
- See the [performance guide](../performance.md) for optimization tips
- Read the [architecture overview](../architecture.md) for system design details