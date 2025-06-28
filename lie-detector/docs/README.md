# Veritas Nexus Documentation

Welcome to the comprehensive documentation for Veritas Nexus, an advanced multi-modal lie detection system built with Rust.

## Table of Contents

- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Architecture](#architecture)
- [Performance](#performance)
- [Deployment](#deployment)

## Getting Started

### Quick Start

```rust
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the lie detector
    let detector = LieDetector::builder()
        .with_default_models()
        .build()
        .await?;
    
    // Analyze input
    let input = AnalysisInput::new()
        .with_text("I was definitely not there")
        .with_video_path("interview.mp4")
        .with_audio_path("interview.wav");
    
    let result = detector.analyze(&input).await?;
    
    println!("Decision: {:?}", result.decision);
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    
    Ok(())
}
```

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
veritas-nexus = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## API Reference

### Core Modules

- **[Engine](api/engine.md)** - Main analysis engine and coordination
- **[Modalities](api/modalities.md)** - Individual modality processors
  - [Text Analysis](api/modalities/text.md)
  - [Vision Analysis](api/modalities/vision.md)
  - [Audio Analysis](api/modalities/audio.md)
  - [Physiological Analysis](api/modalities/physiological.md)
- **[Fusion](api/fusion.md)** - Multi-modal fusion strategies
- **[MCP](api/mcp.md)** - Model Context Protocol integration
- **[Training](api/training.md)** - Model training and optimization

### Key Types

- [`LieDetector`](api/types.md#liedetector) - Main detection interface
- [`AnalysisInput`](api/types.md#analysisinput) - Input data structure
- [`AnalysisResult`](api/types.md#analysisresult) - Output results
- [`FusionStrategy`](api/types.md#fusionstrategy) - Fusion configuration
- [`DeceptionDecision`](api/types.md#deceptiondecision) - Final decisions

## Examples

See the [examples directory](../examples/) for comprehensive examples:

1. **[Basic Detection](../examples/basic_detection.rs)** - Simple usage patterns
2. **[Real-time Analysis](../examples/real_time_analysis.rs)** - Streaming processing
3. **[Batch Processing](../examples/batch_processing.rs)** - Large-scale analysis
4. **[Multi-modal Fusion](../examples/multi_modal_fusion.rs)** - Advanced fusion
5. **[Explainable Decisions](../examples/explainable_decisions.rs)** - AI explanations
6. **[MCP Server](../examples/mcp_server.rs)** - Protocol integration
7. **[Cascade Training](../examples/cascade_training.rs)** - Model training

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Veritas Nexus Architecture               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Vision    │  │    Audio    │  │    Text     │         │
│  │  Processor  │  │  Processor  │  │  Processor  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                  ┌─────────────┐                           │
│                  │   Fusion    │                           │
│                  │   Engine    │                           │
│                  └─────────────┘                           │
│                           │                                │
│                  ┌─────────────┐                           │
│                  │  Decision   │                           │
│                  │   Engine    │                           │
│                  └─────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Modality Processors**: Specialized analysis for each input type
2. **Fusion Engine**: Combines evidence from multiple modalities
3. **Decision Engine**: Makes final determinations with confidence scores
4. **Explanation Engine**: Provides reasoning traces and interpretability

## Performance

### Benchmarks

| Operation | Single Sample | Batch (100 samples) | GPU Acceleration |
|-----------|---------------|---------------------|------------------|
| Text Analysis | ~50ms | ~2.5s | N/A |
| Vision Analysis | ~150ms | ~8s | ~3s |
| Audio Analysis | ~80ms | ~4s | ~2s |
| Full Multi-modal | ~200ms | ~12s | ~5s |

### Memory Usage

- **Base System**: ~100MB
- **Per Sample**: ~5-10MB
- **Batch Processing**: Configurable limits
- **GPU Models**: +500MB-2GB

### Optimization Tips

1. Use GPU acceleration for large batches
2. Configure appropriate batch sizes for your memory constraints
3. Enable caching for repeated analyses
4. Use streaming for real-time applications

## Deployment

### Production Configuration

```rust
use veritas_nexus::prelude::*;

let config = ProductionConfig {
    enable_gpu: true,
    batch_size: 32,
    memory_limit_mb: 4096,
    cache_size: 1000,
    log_level: LogLevel::Info,
    metrics_enabled: true,
};

let detector = LieDetector::with_config(config).await?;
```

### Docker Deployment

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates
COPY --from=builder /app/target/release/veritas-nexus /usr/local/bin/
EXPOSE 8080
CMD ["veritas-nexus", "--port", "8080"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: veritas-nexus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: veritas-nexus
  template:
    metadata:
      labels:
        app: veritas-nexus
    spec:
      containers:
      - name: veritas-nexus
        image: veritas-nexus:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.