# Vision Modality Module

This module provides computer vision capabilities for detecting deception through facial expressions, micro-expressions, gaze patterns, and other visual cues.

## Overview

The vision modality module is designed to analyze visual input for signs of deception by processing:

- **Facial Expressions**: Detection and analysis of facial expressions and micro-expressions
- **Gaze Patterns**: Eye movement and gaze direction analysis  
- **Facial Landmarks**: 68-point facial landmark detection and tracking
- **Action Units**: Facial Action Unit (AU) detection based on the Facial Action Coding System (FACS)
- **Head Pose**: Head position and movement analysis

## Architecture

### Core Components

1. **VisionAnalyzer**: Main analyzer implementing the ModalityAnalyzer trait
2. **FaceAnalyzer**: Face detection, landmark extraction, and facial feature analysis
3. **MicroExpressionDetector**: Micro-expression detection and temporal analysis
4. **GpuVisionProcessor**: Optional GPU acceleration using Candle framework

### Key Features

- **Real-time Processing**: Optimized for real-time video analysis
- **Temporal Analysis**: Tracks changes across multiple frames for micro-expression detection
- **GPU Acceleration**: Optional GPU processing for improved performance
- **Explainable Results**: Provides detailed explanations for detection decisions
- **Modular Design**: Each component can be used independently

## Usage

### Basic Usage

```rust
use veritas_nexus::modalities::vision::{VisionAnalyzer, VisionConfig, VisionInput};

// Create analyzer with default configuration
let config = VisionConfig::default();
let analyzer = VisionAnalyzer::new(config)?;

// Process image
let image_data = load_image_data("path/to/image.jpg")?;
let input = VisionInput::new(image_data, width, height, 3);

// Extract features
let features = analyzer.extract_features(&input)?;

// Analyze for deception
let score = analyzer.analyze(&features)?;
println!("Deception probability: {:.2}%", score.probability * 100.0);

// Get explanation
let explanation = analyzer.explain(&features);
println!("Analysis: {}", explanation.confidence_reasoning);
```

### Configuration

```rust
let mut config = VisionConfig::default();
config.face_detection_threshold = 0.8;  // Higher sensitivity
config.micro_expression_sensitivity = 0.9;
config.enable_gpu = true;  // Enable GPU acceleration
config.max_faces = 3;  // Process up to 3 faces

let analyzer = VisionAnalyzer::new(config)?;
```

### Micro-expression Detection

```rust
let mut detector = MicroExpressionDetector::new(&config)?;

// Set baseline for comparison
detector.set_baseline(&baseline_input)?;

// Process frames for temporal analysis
for frame in video_frames {
    let result = detector.detect_expressions(&frame)?;
    
    for expression in result.expressions {
        println!("Detected: {} (intensity: {:.2})", 
                expression.expression_type.description(),
                expression.intensity);
    }
}
```

### GPU Acceleration

```rust
#[cfg(feature = "gpu")]
{
    let processor = GpuVisionProcessor::new(&config)?;
    
    if processor.is_gpu_enabled() {
        let features = processor.extract_features(&input)?;
        println!("GPU processing enabled");
    }
}
```

## Feature Types

### Micro-expressions

The module detects various micro-expression types with different deception relevance scores:

- **Suppression** (0.9): Emotion suppression detected
- **Leakage** (0.85): Emotional leakage detected  
- **Duping Delight** (0.8): Pleasure from deceiving
- **Contempt** (0.7): Brief contempt expression
- **Fear** (0.6): Brief fear expression
- **Anger** (0.5): Brief anger expression
- **Disgust** (0.5): Brief disgust expression
- **Sadness** (0.3): Brief sadness expression
- **Surprise** (0.2): Brief surprise expression
- **Happiness** (0.1): Brief happiness expression

### Facial Action Units

Based on the Facial Action Coding System (FACS), the module detects 17 standard action units:

- AU1: Inner Brow Raiser
- AU2: Outer Brow Raiser
- AU4: Brow Lowerer
- AU5: Upper Lid Raiser
- AU6: Cheek Raiser
- AU7: Lid Tightener
- AU9: Nose Wrinkler
- AU10: Upper Lip Raiser
- AU12: Lip Corner Puller
- AU15: Lip Corner Depressor
- AU17: Chin Raiser
- AU20: Lip Stretcher
- AU25: Lips Part
- AU26: Jaw Drop
- AU28: Lip Suck
- AU43: Eyes Closed
- AU45: Blink

## Performance

### Benchmarks

Run benchmarks with:
```bash
cargo bench --bench vision_benchmarks
```

Available benchmarks:
- Face detection performance across image sizes
- Landmark extraction speed
- Micro-expression detection throughput
- Complete vision analysis pipeline
- GPU vs CPU performance comparison
- Memory usage patterns

### Optimization Tips

1. **Image Size**: Use 224x224 for optimal performance vs accuracy balance
2. **GPU**: Enable GPU acceleration for batch processing
3. **Temporal Window**: Adjust temporal window size based on video frame rate
4. **Sensitivity**: Lower sensitivity for faster processing, higher for better detection

## Testing

### Unit Tests

Each module includes comprehensive unit tests:

```bash
cargo test modalities::vision
```

### Integration Tests

Full pipeline tests are available:

```bash
cargo test modalities::vision::tests::integration_tests
```

### Test Data

The module includes utilities for generating test data:
- `create_test_image()`: Creates synthetic test images
- `create_face_like_image()`: Creates images with face-like patterns

## Error Handling

The module uses a comprehensive error hierarchy:

```rust
pub enum VisionError {
    FaceDetectionFailed(String),
    FeatureExtractionFailed(String),
    MicroExpressionFailed(String),
    InvalidImageFormat(String),
    GpuError(String),
    ConfigError(String),
    ModelLoadError(String),
}
```

## Dependencies

Core dependencies:
- `num-traits`: Numerical trait abstractions
- `thiserror`: Error handling

Optional dependencies:
- `candle-core`: GPU acceleration (feature: "gpu")
- `candle-nn`: Neural network operations (feature: "gpu")

## Model Requirements

The vision module expects pre-trained models for:
- Face detection (ONNX format)
- Facial landmark detection (ONNX format)  
- Micro-expression classification (ruv-FANN format)
- Action unit detection (ONNX format)

Default model paths can be configured via `ModelPaths` in the configuration.

## Future Enhancements

Planned improvements:
- Real-time video stream processing
- Advanced temporal modeling with LSTMs
- 3D face analysis
- Multi-face tracking and identification
- Integration with other modalities (audio, text)
- Model fine-tuning capabilities

## Contributing

When contributing to this module:
1. Add unit tests for new functionality
2. Update benchmarks for performance-critical changes
3. Ensure GPU code is properly feature-gated
4. Add documentation for new public APIs
5. Test with various image sizes and formats