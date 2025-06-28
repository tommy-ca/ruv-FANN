# Veritas Nexus Examples

This directory contains comprehensive examples demonstrating the capabilities of the Veritas Nexus lie detection system. Each example showcases different aspects of the system, from basic usage to advanced features like real-time processing, MCP server integration, and explainable AI.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Example Overview](#example-overview)
3. [Basic Examples](#basic-examples)
4. [Advanced Examples](#advanced-examples)
5. [Integration Examples](#integration-examples)
6. [Data and Testing](#data-and-testing)
7. [Usage Instructions](#usage-instructions)
8. [Performance Notes](#performance-notes)
9. [Contributing](#contributing)

## Quick Start

To run any example:

```bash
# Clone and navigate to the project
cd lie-detector

# Run a basic example
cargo run --example basic_detection

# Run with release optimization for better performance
cargo run --release --example real_time_analysis
```

## Example Overview

| Example | Complexity | Description | Key Features |
|---------|------------|-------------|--------------|
| [basic_detection.rs](#basic-detection) | ⭐ | Simple lie detection usage | Multi-modal input, builder pattern, error handling |
| [text_analysis_demo.rs](#text-analysis) | ⭐⭐ | Text-only analysis capabilities | NLP features, batch processing, multi-language |
| [basic_fusion.rs](#basic-fusion) | ⭐⭐ | Multi-modal fusion basics | Fusion strategies, feature combination |
| [batch_processing.rs](#batch-processing) | ⭐⭐⭐ | Large-scale data processing | Parallel processing, GPU acceleration, export formats |
| [real_time_analysis.rs](#real-time-analysis) | ⭐⭐⭐ | Streaming lie detection | Ring buffers, temporal sync, performance monitoring |
| [multi_modal_fusion.rs](#multi-modal-fusion) | ⭐⭐⭐⭐ | Advanced fusion strategies | Attention mechanisms, adaptive fusion, uncertainty |
| [explainable_decisions.rs](#explainable-decisions) | ⭐⭐⭐⭐ | AI explainability and reasoning | Reasoning traces, feature importance, multiple formats |
| [cascade_training.rs](#cascade-training) | ⭐⭐⭐⭐⭐ | Model training pipeline | Progressive training, transfer learning, validation |
| [mcp_server.rs](#mcp-server) | ⭐⭐⭐⭐⭐ | MCP protocol integration | Server setup, tool registration, resource management |

## Basic Examples

### basic_detection.rs
**Purpose**: Demonstrates the simplest usage of the Veritas Nexus system.

**Key Features**:
- Initialize lie detector with default/custom configuration
- Analyze multi-modal input (video, audio, text, physiological)
- Interpret results and confidence scores
- Access basic reasoning traces
- Error handling for invalid inputs

**Usage**:
```bash
cargo run --example basic_detection
```

**What You'll Learn**:
- How to set up a basic lie detector
- Different configuration options available
- How to interpret analysis results
- Basic error handling patterns

### text_analysis_demo.rs
**Purpose**: Deep dive into text-based lie detection capabilities.

**Key Features**:
- BERT-based linguistic analysis
- Sentiment and emotion detection
- Named entity recognition
- Batch text processing
- Multi-language support
- Performance and caching optimization

**Usage**:
```bash
cargo run --example text_analysis_demo
```

**What You'll Learn**:
- Text preprocessing and feature extraction
- Linguistic deception patterns
- Batch processing for multiple texts
- Caching strategies for improved performance

### basic_fusion.rs
**Purpose**: Introduction to multi-modal fusion concepts.

**Key Features**:
- Multiple fusion strategies
- Modality weight optimization
- Feature combination techniques
- Temporal alignment basics

**Usage**:
```bash
cargo run --example basic_fusion
```

**What You'll Learn**:
- How different modalities are combined
- Fusion strategy selection
- Feature weight optimization
- Handling missing modalities

## Advanced Examples

### batch_processing.rs
**Purpose**: Efficient processing of large datasets with various optimization strategies.

**Key Features**:
- File and database data sources
- Parallel and GPU-accelerated processing
- Memory optimization and limits
- Progress monitoring with ETA
- Multiple output formats (JSON, CSV, database)
- Error handling strategies (skip, retry, quarantine)

**Usage**:
```bash
cargo run --example batch_processing
```

**What You'll Learn**:
- How to process thousands of samples efficiently
- GPU acceleration benefits and setup
- Memory management for large datasets
- Progress tracking and performance monitoring
- Different data source integrations

**Sample Output**:
```
📦 Veritas Nexus - Batch Processing Example
==========================================

📁 Example 1: Processing from CSV file
-------------------------------------
    Progress: 150/200 samples (75.0%) - 12.3 samples/sec - ETA: 4.1s
✅ File processing completed:
    Total samples: 200
    Successful: 198
    Failed: 2
    Total time: 16.24s
    Throughput: 12.3 samples/sec
    Memory usage: 256.7MB
```

### real_time_analysis.rs
**Purpose**: Real-time streaming lie detection with temporal synchronization.

**Key Features**:
- Multi-modal streaming pipeline
- Ring buffer management for memory efficiency
- Temporal synchronization across modalities
- Real-time feedback with configurable latency
- Performance monitoring and statistics
- Adaptive quality based on system load

**Usage**:
```bash
cargo run --example real_time_analysis
```

**What You'll Learn**:
- Setting up streaming data pipelines
- Handling temporal misalignment between modalities
- Memory-efficient data structures for streaming
- Real-time performance optimization
- System monitoring and diagnostics

**Sample Output**:
```
🎬 Veritas Nexus - Real-Time Streaming Analysis
==============================================

⚙️  Configuration:
   Target FPS: 10
   Audio chunk size: 100ms
   Sync window: 200ms
   Max latency: 300ms

🔍 Result #1: Deceptive (score: 0.687, confidence: 78.3%, latency: 145ms)
   Contributions: 👁️ 0.72 🔊 0.54 📝 0.81 

📈 Pipeline Statistics:
   Frames processed: 84
   Audio chunks processed: 92
   Average latency: 142.3ms
```

### multi_modal_fusion.rs
**Purpose**: Comprehensive demonstration of advanced fusion strategies.

**Key Features**:
- Early fusion (feature-level combination)
- Late fusion (decision-level combination)
- Attention-based fusion (dynamic weighting)
- Adaptive fusion (strategy selection)
- Uncertainty quantification
- Missing modality handling
- Performance comparison across strategies

**Usage**:
```bash
cargo run --example multi_modal_fusion
```

**What You'll Learn**:
- Different fusion approaches and their trade-offs
- When to use each fusion strategy
- Attention mechanisms for dynamic weighting
- Ensemble methods for improved robustness
- Uncertainty estimation and confidence intervals

**Sample Output**:
```
🔀 Veritas Nexus - Multi-Modal Fusion Strategies Example
========================================================

📊 Scenario: High Deception
----------------------------
Available modalities: Video, Audio, Text, Physiological

🔄 early_fusion Results:
  Decision: Deceptive
  Confidence: 78.2%
  Uncertainty: 21.8%
  Processing time: 15ms
  Explanation: Early fusion combined 14 features into 14-dimensional space...

🔄 attention_fusion Results:
  Decision: Deceptive
  Confidence: 85.7%
  Uncertainty: 14.3%
  Modality weights:
    vision: 0.412
    audio: 0.248
    text: 0.201
    physiological: 0.139
```

### explainable_decisions.rs
**Purpose**: Comprehensive AI explainability and decision transparency.

**Key Features**:
- Multiple explanation styles (technical, simplified, narrative)
- Step-by-step reasoning traces
- Feature importance analysis
- Uncertainty source identification
- Decision pathway visualization
- Alternative scenario analysis
- Export to multiple formats (JSON, HTML, text)

**Usage**:
```bash
cargo run --example explainable_decisions
```

**What You'll Learn**:
- How to generate human-readable explanations
- Different explanation styles for different audiences
- Reasoning trace construction
- Uncertainty analysis and interpretation
- Decision pathway visualization techniques

**Sample Output**:
```
🧠 Veritas Nexus - Explainable AI Decisions Example
==================================================

🎯 Configuration: Technical Expert
--------------------------------------------------
📊 Decision: Deceptive { probability: 0.78 }
📈 Confidence: 78.2%
🎯 Uncertainty: 21.8%

📝 Summary:
The analysis suggests the subject may be engaging in deception with 78% confidence. 
Primary deception indicators include micro_expressions, voice_stress, uncertainty_markers.

🎛️  Top Features:
  1. micro_expressions (video): 0.750 - Strong micro-expression indicators of deception detected
  2. uncertainty_markers (text): 0.800 - High frequency of uncertainty language markers
  3. voice_stress (audio): 0.680 - High vocal stress levels detected

⚠️  Uncertainty Sources:
  • Model Uncertainty: 8.7% - Uncertainty due to model limitations and training data coverage
  • Data Quality: 6.2% - Uncertainty due to input data quality and noise
  • Feature Disagreement: 4.8% - Uncertainty due to conflicting evidence from different features
```

## Integration Examples

### cascade_training.rs
**Purpose**: Complete model training pipeline with progressive complexity.

**Key Features**:
- Multi-stage training pipeline (vision → audio → text → fusion)
- Transfer learning and fine-tuning
- Dependency resolution and topological sorting
- Progress monitoring and validation metrics
- Early stopping and learning rate scheduling
- Model saving and checkpoint management

**Usage**:
```bash
cargo run --example cascade_training
```

**What You'll Learn**:
- How to set up complex training pipelines
- Progressive training strategies
- Model dependency management
- Training monitoring and optimization
- Validation and early stopping techniques

**Sample Output**:
```
🏗️  Veritas Nexus - Cascade Training Example
===========================================

📋 Training Configuration:
  Total stages: 5
  Global learning rate: 0.001
  Max epochs per stage: 20
  Device: Auto

📚 Training Stages:
  1. vision_pretraining (VisionPretraining)
  2. audio_pretraining (AudioPretraining)
  3. text_pretraining (TextPretraining)
  4. vision_finetuning (VisionFinetuning)
     Dependencies: ["vision_pretraining"]
  5. late_fusion (LateFusion)
     Dependencies: ["vision_finetuning", "audio_pretraining", "text_pretraining"]

📚 Training stage: vision_pretraining (VisionPretraining)
  Configuration:
    Architecture: ResNet { layers: 50 }
    Learning rate: 0.001
    Batch size: 32
    Max epochs: 20
    Loss function: CrossEntropy
    Optimizer: Adam { beta1: 0.9, beta2: 0.999, eps: 0.000000001 }
    Epoch 1/20: loss=0.8234, acc=0.620, val_loss=0.9057, val_acc=0.570 (2.3s)
    ...
```

### mcp_server.rs
**Purpose**: Model Context Protocol (MCP) server implementation for integration with external systems.

**Key Features**:
- Complete MCP server setup
- Tool registration (analyze_deception, manage_models, monitor_server)
- Resource management for models and data
- Session management and state tracking
- Event streaming for real-time updates
- Performance monitoring and health checks

**Usage**:
```bash
cargo run --example mcp_server
```

**What You'll Learn**:
- How to expose lie detection as MCP tools
- Resource management and model loading
- Session handling and state management
- Real-time event streaming
- Performance monitoring and health checks

**Sample Output**:
```
🏗️  Veritas Nexus MCP Server Example
===================================

🚀 Starting Veritas Nexus MCP Server
=====================================
Name: veritas-nexus-server
Version: 0.1.0
Max concurrent sessions: 10
Model cache size: 5
Streaming enabled: true

📋 Available Tools:
  • analyze_deception: Analyze multi-modal inputs for deception detection
  • manage_models: Load, unload, and get information about available models
  • monitor_server: Get server status, performance metrics, and health information

🧠 Available Models:
  • default_vision: Default Vision Model (125.0MB, 88.7% accuracy)
  • default_audio: Default Audio Model (87.0MB, 82.3% accuracy)
  • default_text: Default Text Model (342.0MB, 79.4% accuracy)

📞 Simulated client connection received
🔧 Tool call: analyze_deception
✅ Analysis completed:
   Decision: uncertain
   Confidence: 62.7%
   Processing time: 156ms
```

## Data and Testing

### Example Data Structure

The examples use both simulated and real data formats. Here's the expected structure:

```
lie-detector/
├── examples/
│   ├── data/
│   │   ├── samples.csv
│   │   ├── video/
│   │   │   ├── interview_1.mp4
│   │   │   └── interview_2.mp4
│   │   ├── audio/
│   │   │   ├── interview_1.wav
│   │   │   └── interview_2.wav
│   │   └── physiological/
│   │       ├── session_1_hr.csv
│   │       └── session_1_gsr.csv
│   └── output/
│       ├── results.json
│       ├── analysis_report.html
│       └── batch_results.csv
```

### Sample Data Format

**CSV Input Format** (`samples.csv`):
```csv
id,video_path,audio_path,text,label,confidence
sample_1,video/interview_1.mp4,audio/interview_1.wav,"I was definitely not there",deceptive,0.85
sample_2,video/interview_2.mp4,audio/interview_2.wav,"I clearly remember being home",truthful,0.92
```

**JSON Input Format**:
```json
{
  "samples": [
    {
      "id": "sample_001",
      "video_path": "video/interview.mp4",
      "audio_path": "audio/interview.wav",
      "text": "I have never seen that document before",
      "metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "environment": "controlled",
        "subject_age": 34
      }
    }
  ]
}
```

## Usage Instructions

### Prerequisites

1. **Rust Installation**: Ensure you have Rust 1.70+ installed
2. **Dependencies**: All dependencies are specified in `Cargo.toml`
3. **Optional GPU Support**: For GPU acceleration examples, ensure CUDA is installed

### Running Examples

```bash
# Basic examples (fast execution)
cargo run --example basic_detection
cargo run --example text_analysis_demo
cargo run --example basic_fusion

# Advanced examples (may take longer)
cargo run --release --example batch_processing
cargo run --release --example real_time_analysis
cargo run --release --example multi_modal_fusion

# Complex examples (longer execution, more output)
cargo run --release --example explainable_decisions
cargo run --release --example cascade_training
cargo run --release --example mcp_server
```

### Common Command Line Options

Most examples support environment variables for configuration:

```bash
# Enable debug logging
RUST_LOG=debug cargo run --example real_time_analysis

# Set custom data directory
DATA_DIR=/path/to/data cargo run --example batch_processing

# Configure GPU device
CUDA_DEVICE=1 cargo run --example batch_processing

# Set output directory
OUTPUT_DIR=/path/to/output cargo run --example explainable_decisions
```

### Example-Specific Configuration

**batch_processing.rs**:
```bash
# High-performance configuration
cargo run --release --example batch_processing -- --batch-size 64 --parallel --gpu-device 0

# Memory-optimized configuration
cargo run --example batch_processing -- --batch-size 16 --memory-limit 2048
```

**real_time_analysis.rs**:
```bash
# Low-latency configuration
cargo run --release --example real_time_analysis -- --target-fps 60 --max-latency 100

# High-quality configuration
cargo run --example real_time_analysis -- --target-fps 30 --sync-window 500
```

## Performance Notes

### Expected Performance

| Example | Typical Runtime | Memory Usage | GPU Acceleration |
|---------|----------------|--------------|------------------|
| basic_detection | < 1 second | 50MB | No |
| text_analysis_demo | 2-5 seconds | 100MB | Optional |
| batch_processing | 10-60 seconds | 200-500MB | Yes |
| real_time_analysis | Continuous | 100-200MB | Yes |
| multi_modal_fusion | 5-10 seconds | 150MB | Optional |
| explainable_decisions | 3-8 seconds | 120MB | No |
| cascade_training | 2-10 minutes | 500MB-2GB | Yes |
| mcp_server | Continuous | 200MB | Yes |

### Optimization Tips

1. **Use Release Mode**: Always use `--release` for performance-critical examples
2. **GPU Acceleration**: Examples that support GPU will automatically detect and use available devices
3. **Memory Management**: Large dataset examples include memory limit configuration
4. **Parallel Processing**: Most examples support parallel execution for better throughput

### Troubleshooting

**Common Issues**:

1. **Out of Memory**: Reduce batch size or enable memory limits
   ```bash
   cargo run --example batch_processing -- --batch-size 8 --memory-limit 1024
   ```

2. **GPU Not Detected**: Check CUDA installation and device availability
   ```bash
   nvidia-smi  # Check GPU status
   ```

3. **Slow Performance**: Ensure release mode and consider GPU acceleration
   ```bash
   cargo run --release --example batch_processing -- --gpu-device 0
   ```

4. **File Not Found**: Check data directory structure and paths
   ```bash
   ls -la examples/data/  # Verify data files exist
   ```

## Advanced Configuration

### Custom Model Configuration

Many examples support custom model configuration through configuration files:

**config.toml**:
```toml
[model]
vision_model = "resnet50"
audio_model = "wav2vec2"
text_model = "bert-base-uncased"

[fusion]
strategy = "attention"
weights = { vision = 0.4, audio = 0.3, text = 0.3 }

[performance]
batch_size = 32
use_gpu = true
memory_limit_mb = 2048
```

### Environment Variables

```bash
# Model configuration
export VERITAS_MODEL_PATH=/path/to/models
export VERITAS_CACHE_DIR=/path/to/cache

# Performance settings
export VERITAS_BATCH_SIZE=64
export VERITAS_GPU_DEVICE=0
export VERITAS_MEMORY_LIMIT=4096

# Logging
export RUST_LOG=veritas_nexus=info
export VERITAS_LOG_LEVEL=debug
```

## Contributing

To add new examples:

1. Create a new `.rs` file in the `examples/` directory
2. Follow the existing documentation pattern with comprehensive comments
3. Include error handling and progress reporting
4. Add performance metrics and timing information
5. Update this README with the new example
6. Ensure the example compiles and runs successfully

### Example Template

```rust
//! # New Example for Veritas Nexus
//! 
//! This example demonstrates [specific functionality].
//! It shows how to:
//! - [Key feature 1]
//! - [Key feature 2]
//! - [Key feature 3]
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example new_example
//! ```

use std::time::Instant;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Veritas Nexus - New Example");
    println!("===============================\n");
    
    let start_time = Instant::now();
    
    // Example implementation here
    
    println!("✅ Example completed in {:.2}s", start_time.elapsed().as_secs_f32());
    println!("\n💡 Key Features Demonstrated:");
    println!("   • [Feature 1]");
    println!("   • [Feature 2]");
    println!("   • [Feature 3]");
    
    Ok(())
}
```

## API Reference

For detailed API documentation, see the [API Documentation](../docs/api/) directory or generate documentation with:

```bash
cargo doc --open
```

This will open comprehensive API documentation in your browser with detailed information about all types, functions, and examples.

---

For questions or issues with the examples, please check the [main documentation](../docs/) or open an issue in the project repository.