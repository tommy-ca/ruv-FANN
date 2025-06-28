# Getting Started with Veritas Nexus

Welcome to Veritas Nexus! This guide will walk you through everything you need to know to get up and running with multi-modal lie detection in minutes.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- Rust 1.70 or later
- 4GB RAM
- 2GB free disk space
- x86_64 or ARM64 processor

**Recommended for Best Performance:**
- Rust 1.75+ 
- 8GB+ RAM
- 10GB+ free disk space
- Modern CPU with SIMD support (AVX2/NEON)
- NVIDIA GPU with CUDA 11.8+ (optional)

### Supported Platforms

- ✅ **Linux** (Ubuntu 20.04+, Debian 11+, RHEL 8+)
- ✅ **macOS** (10.15+, including Apple Silicon)
- ✅ **Windows** (Windows 10+, WSL2 recommended)

## 🚀 Quick Installation

### Option 1: Using Cargo (Recommended)

```bash
# Create a new Rust project
cargo new my-veritas-project
cd my-veritas-project

# Add veritas-nexus to your dependencies
cargo add veritas-nexus tokio --features full
```

Add to your `Cargo.toml`:
```toml
[dependencies]
veritas-nexus = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/veritas-nexus.git
cd veritas-nexus

# Build the project
cargo build --release

# Run tests to verify installation
cargo test
```

### Option 3: Pre-built Binaries

Download pre-built binaries from our [releases page](https://github.com/yourusername/veritas-nexus/releases):

```bash
# Linux/macOS
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/veritas-nexus-linux.tar.gz | tar xz
sudo mv veritas-nexus /usr/local/bin/

# Windows (PowerShell)
Invoke-WebRequest -Uri https://github.com/yourusername/veritas-nexus/releases/latest/download/veritas-nexus-windows.zip -OutFile veritas-nexus.zip
Expand-Archive veritas-nexus.zip -DestinationPath .
```

## 🔧 Feature Configuration

Choose the features you need by adding them to your `Cargo.toml`:

```toml
[dependencies]
veritas-nexus = { 
    version = "0.1.0", 
    features = [
        "parallel",      # Multi-threading support (recommended)
        "gpu",          # GPU acceleration (optional)
        "mcp",          # MCP server integration (optional)
        "simd-avx2",    # Advanced SIMD optimizations (x86_64 only)
    ]
}
```

### Feature Guide

| Feature | Description | When to Use |
|---------|-------------|-------------|
| `default` | Basic CPU processing | Always included |
| `parallel` | Multi-threading with Rayon | Recommended for all users |
| `gpu` | GPU acceleration via Candle | Large batches, real-time processing |
| `cuda` | NVIDIA CUDA support | NVIDIA GPU users |
| `metal` | Apple Metal support | Apple Silicon users |
| `mcp` | Model Context Protocol | Server deployments |
| `simd-avx2` | Advanced SIMD | x86_64 CPUs with AVX2 |
| `benchmarking` | Performance testing tools | Development and optimization |

## 🎯 Your First Analysis

Let's create your first lie detection analysis! Create a new file `src/main.rs`:

```rust
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Welcome to Veritas Nexus!");
    
    // Step 1: Initialize the detector
    println!("⚙️ Initializing lie detector...");
    let detector = LieDetector::builder()
        .with_vision(VisionConfig::default())
        .with_audio(AudioConfig::default())
        .with_text(TextConfig::default())
        .build()
        .await?;
    
    println!("✅ Detector initialized successfully!");
    
    // Step 2: Prepare your input
    let input = AnalysisInput {
        video_path: None, // We'll start with text-only
        audio_path: None,
        transcript: Some(
            "I definitely did not take any money from the register. \
             I was at home all evening watching TV with my family."
                .to_string()
        ),
        physiological_data: None,
    };
    
    // Step 3: Run the analysis
    println!("🔍 Analyzing input...");
    let result = detector.analyze(input).await?;
    
    // Step 4: Interpret the results
    println!("\n📊 Analysis Results:");
    println!("━━━━━━━━━━━━━━━━━━━━");
    println!("Decision: {:?}", result.decision);
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    println!("Processing time: {}ms", result.processing_time_ms);
    
    if let Some(text_score) = result.modality_scores.text_score {
        println!("Text deception score: {:.3}", text_score);
    }
    
    println!("\n🧠 Reasoning Trace:");
    for (i, step) in result.reasoning_trace.iter().enumerate() {
        println!("  {}. {}", i + 1, step);
    }
    
    Ok(())
}
```

Run your first analysis:

```bash
cargo run
```

You should see output like:
```
🔍 Welcome to Veritas Nexus!
⚙️ Initializing lie detector...
✅ Detector initialized successfully!
🔍 Analyzing input...

📊 Analysis Results:
━━━━━━━━━━━━━━━━━━━━
Decision: Uncertain
Confidence: 45.2%
Processing time: 127ms
Text deception score: 0.584

🧠 Reasoning Trace:
  1. Processing text input...
  2. Text analysis complete. Score: 0.584
  3. Fusing multi-modal scores...
  4. Final decision: Uncertain (confidence: 45.2%)
```

## 📁 Working with Files

### Analyzing Video and Audio Files

Create sample media files or use the provided test data:

```rust
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let detector = LieDetector::builder()
        .with_vision(VisionConfig {
            enable_face_detection: true,
            enable_micro_expressions: true,
            enable_eye_tracking: false,
            model_precision: ModelPrecision::Balanced,
        })
        .with_audio(AudioConfig::high_quality())
        .with_text(TextConfig::bert_based())
        .build()
        .await?;
    
    // Multi-modal analysis with files
    let input = AnalysisInput {
        video_path: Some("examples/data/sample_interview.mp4".to_string()),
        audio_path: Some("examples/data/sample_interview.wav".to_string()),
        transcript: Some("I have never seen that document before".to_string()),
        physiological_data: None,
    };
    
    let result = detector.analyze(input).await?;
    
    println!("Multi-modal analysis:");
    println!("Decision: {:?} ({:.1}%)", result.decision, result.confidence * 100.0);
    
    // Show contribution from each modality
    if let Some(vision) = result.modality_scores.vision_score {
        println!("👁️  Vision contribution: {:.3}", vision);
    }
    if let Some(audio) = result.modality_scores.audio_score {
        println!("🔊 Audio contribution: {:.3}", audio);
    }
    if let Some(text) = result.modality_scores.text_score {
        println!("📝 Text contribution: {:.3}", text);
    }
    
    Ok(())
}
```

### Using Test Data

Veritas Nexus comes with sample data for testing:

```bash
# Download sample data
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/sample-data.zip -o sample-data.zip
unzip sample-data.zip -d examples/data/

# Or use the provided examples
cargo run --example basic_detection
cargo run --example multi_modal_fusion
```

## 🌊 Real-Time Streaming

For real-time analysis from camera and microphone:

```rust
use veritas_nexus::streaming::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure streaming pipeline
    let config = StreamingConfig {
        target_fps: 15.0,
        audio_chunk_size_ms: 100,
        sync_window_ms: 200,
        max_latency_ms: 300,
        buffer_size: 64,
        enable_adaptive_quality: true,
    };
    
    // Create and start pipeline
    let pipeline = StreamingPipeline::new(config);
    pipeline.start().await?;
    
    println!("🎬 Real-time analysis started. Press Ctrl+C to stop.");
    
    // Process results
    while let Some(result) = pipeline.try_get_result().await {
        println!("Live: {:?} ({:.1}%) - {}ms latency",
            result.decision,
            result.confidence * 100.0,
            result.processing_latency_ms
        );
    }
    
    Ok(())
}
```

## 🔧 Configuration Options

### Basic Configuration

```rust
use veritas_nexus::prelude::*;

// Vision configuration
let vision_config = VisionConfig {
    enable_face_detection: true,
    enable_micro_expressions: true,
    enable_eye_tracking: false,
    model_precision: ModelPrecision::Balanced, // Fast, Balanced, or Accurate
};

// Audio configuration  
let audio_config = AudioConfig {
    sample_rate: 16000,
    enable_pitch_analysis: true,
    enable_stress_detection: true,
    enable_voice_quality: true,
};

// Text configuration
let text_config = TextConfig {
    model_type: TextModel::Bert, // Bert, RoBerta, or DistilBert
    enable_linguistic_analysis: true,
    enable_sentiment_analysis: true,
    language: "en".to_string(),
};

let detector = LieDetector::builder()
    .with_vision(vision_config)
    .with_audio(audio_config)
    .with_text(text_config)
    .with_fusion_strategy(FusionStrategy::AdaptiveWeight)
    .build()
    .await?;
```

### GPU Configuration

```rust
use veritas_nexus::gpu::*;

// Enable GPU acceleration
let gpu_config = GpuConfig {
    enable_gpu: true,
    device_id: 0, // Use first GPU
    memory_limit_mb: 2048,
    batch_size: 16,
};

let detector = LieDetector::builder()
    .with_gpu_config(gpu_config)
    .build()
    .await?;
```

### Performance Tuning

```rust
use veritas_nexus::optimization::*;

// Performance configuration
let perf_config = PerformanceConfig {
    num_threads: 4,
    enable_simd: true,
    memory_pool_size: 1024 * 1024 * 100, // 100MB
    cache_size: 1000,
    enable_profiling: false,
};

let detector = LieDetector::builder()
    .with_performance_config(perf_config)
    .build()
    .await?;
```

## 🚨 Common Issues & Solutions

### Issue: Slow Compilation

**Solution**: Enable parallel compilation and use faster linker:

```bash
# ~/.cargo/config.toml
[build]
jobs = 8

[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

### Issue: GPU Not Detected

**Solution**: Verify CUDA installation and enable GPU features:

```bash
# Check CUDA
nvidia-smi

# Enable GPU features
cargo build --features "gpu,cuda"
```

### Issue: Model Loading Errors

**Solution**: Ensure model files are available:

```bash
# Download required models
mkdir -p ~/.veritas-nexus/models
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/models.tar.gz | tar xz -C ~/.veritas-nexus/models/
```

### Issue: Permission Denied for Camera/Microphone

**Solution**: Grant necessary permissions:

```bash
# Linux - add user to video/audio groups
sudo usermod -a -G video,audio $USER

# macOS - grant permissions in System Preferences
# Windows - check privacy settings
```

## 📚 Next Steps

Now that you have Veritas Nexus up and running, explore these resources:

1. **[Tutorial](TUTORIAL.md)** - Step-by-step examples from basic to advanced
2. **[User Guide](USER_GUIDE.md)** - Comprehensive usage patterns and best practices
3. **[Examples](examples/)** - Ready-to-run code examples
4. **[API Documentation](https://docs.rs/veritas-nexus)** - Complete API reference
5. **[Performance Guide](PERFORMANCE_GUIDE.md)** - Optimization tips and benchmarking

### Learning Path

1. ✅ **Getting Started** (You are here!)
2. 📝 **Tutorial** - Learn core concepts with hands-on examples
3. 📖 **User Guide** - Master advanced features and patterns
4. ⚡ **Performance Guide** - Optimize for your specific use case
5. 🚀 **Deployment Guide** - Deploy to production environments

## 🆘 Getting Help

- **Documentation**: [docs.rs/veritas-nexus](https://docs.rs/veritas-nexus)
- **Examples**: Check the [`examples/`](examples/) directory
- **Issues**: [GitHub Issues](https://github.com/yourusername/veritas-nexus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/veritas-nexus/discussions)
- **Discord**: [Join our community](https://discord.gg/veritas-nexus)

## 🎉 What's Next?

Congratulations! You've successfully set up Veritas Nexus and run your first lie detection analysis. Here are some exciting things to try next:

- **Experiment with different fusion strategies** - Try `AttentionBased` or `ContextAware` fusion
- **Test with real audio/video files** - Use your own interview recordings
- **Explore the streaming API** - Build real-time applications
- **Dive into the reasoning traces** - Understand how decisions are made
- **Try GPU acceleration** - Experience the performance boost
- **Set up an MCP server** - Enable remote API access

Ready to dive deeper? Head over to our [Tutorial](TUTORIAL.md) for hands-on examples that will take you from beginner to expert!

---

*Need help? Don't hesitate to reach out to our community. We're here to help you succeed with Veritas Nexus!*