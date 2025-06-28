# MCP Interface Specification for Veritas-Core Lie Detection System

## Executive Summary

This document specifies the Model Context Protocol (MCP) interface for the Veritas-Core lie detection crate. The MCP server provides a comprehensive API for managing multi-modal lie detection models, including training, inference, monitoring, and configuration. The system leverages the official Rust MCP SDK to expose tools, resources, and real-time capabilities for neural network-based deception detection.

## Architecture Overview

### Core Components

1. **MCP Server** - Main server process built with the Rust MCP SDK
2. **Model Manager** - Handles loading, training, and inference of neural models
3. **Data Pipeline** - Processes multi-modal inputs (vision, audio, text, physiological)
4. **ReAct Agent** - Reasoning and action framework for interpretable decisions
5. **Monitor Service** - Real-time performance and resource monitoring
6. **Storage Backend** - Model weights, training data, and results persistence

### Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (veritas-mcp)                  │
├─────────────────────────────────────────────────────────────┤
│  Tools Layer           Resources Layer      Events Layer     │
│  ┌─────────────┐      ┌─────────────┐     ┌──────────────┐ │
│  │Model Tools  │      │Model Weights│     │Training Events│ │
│  │Train Tools  │      │Datasets     │     │Inference Logs │ │
│  │Infer Tools  │      │Configs      │     │Monitor Stream │ │
│  │Monitor Tools│      │Results      │     │Alert Events   │ │
│  └─────────────┘      └─────────────┘     └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Core Services Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │Model Manager│  │Data Pipeline│  │ReAct Agent Engine │ │
│  │  - Vision   │  │ - Preprocess│  │ - Reasoning       │ │
│  │  - Audio    │  │ - Transform │  │ - Decision Making │ │
│  │  - Text     │  │ - Validate  │  │ - Explanation     │ │
│  │  - Physio   │  │ - Augment   │  │ - Self-Play (GSPO)│ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## MCP Tool Definitions

### 1. Model Management Tools

#### `model/list`
Lists all available models and their status.

```json
{
  "name": "model/list",
  "description": "List all available lie detection models",
  "inputSchema": {
    "type": "object",
    "properties": {
      "filter": {
        "type": "string",
        "enum": ["all", "trained", "untrained", "active"],
        "default": "all"
      }
    }
  }
}
```

#### `model/load`
Load a specific model into memory for inference.

```json
{
  "name": "model/load",
  "description": "Load a model for inference",
  "inputSchema": {
    "type": "object",
    "required": ["model_id"],
    "properties": {
      "model_id": {
        "type": "string",
        "description": "Unique model identifier"
      },
      "device": {
        "type": "string",
        "enum": ["cpu", "cuda", "mps"],
        "default": "cpu"
      },
      "precision": {
        "type": "string",
        "enum": ["fp32", "fp16", "int8"],
        "default": "fp32"
      }
    }
  }
}
```

#### `model/create`
Create a new model with specified architecture.

```json
{
  "name": "model/create",
  "description": "Create a new lie detection model",
  "inputSchema": {
    "type": "object",
    "required": ["name", "modalities"],
    "properties": {
      "name": {
        "type": "string",
        "description": "Model name"
      },
      "modalities": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["vision", "audio", "text", "physiological"]
        }
      },
      "architecture": {
        "type": "object",
        "properties": {
          "vision_model": {
            "type": "string",
            "enum": ["cnn", "resnet", "efficientnet", "custom"]
          },
          "audio_model": {
            "type": "string",
            "enum": ["lstm", "gru", "wav2vec", "custom"]
          },
          "text_model": {
            "type": "string",
            "enum": ["bert", "roberta", "gpt", "custom"]
          },
          "fusion_strategy": {
            "type": "string",
            "enum": ["late", "early", "attention", "learned"]
          }
        }
      }
    }
  }
}
```

### 2. Training Tools

#### `training/start`
Start training a model on specified dataset.

```json
{
  "name": "training/start",
  "description": "Start model training",
  "inputSchema": {
    "type": "object",
    "required": ["model_id", "dataset_id"],
    "properties": {
      "model_id": {
        "type": "string"
      },
      "dataset_id": {
        "type": "string"
      },
      "config": {
        "type": "object",
        "properties": {
          "epochs": {
            "type": "integer",
            "default": 100
          },
          "batch_size": {
            "type": "integer",
            "default": 32
          },
          "learning_rate": {
            "type": "number",
            "default": 0.001
          },
          "optimizer": {
            "type": "string",
            "enum": ["adam", "sgd", "rmsprop", "adamw"],
            "default": "adam"
          },
          "enable_gspo": {
            "type": "boolean",
            "default": false,
            "description": "Enable Generative Self-Play Optimization"
          },
          "validation_split": {
            "type": "number",
            "default": 0.2
          }
        }
      }
    }
  }
}
```

#### `training/status`
Get current training status and metrics.

```json
{
  "name": "training/status",
  "description": "Get training status and metrics",
  "inputSchema": {
    "type": "object",
    "required": ["training_id"],
    "properties": {
      "training_id": {
        "type": "string"
      },
      "include_metrics": {
        "type": "boolean",
        "default": true
      }
    }
  }
}
```

#### `training/stop`
Stop an ongoing training session.

```json
{
  "name": "training/stop",
  "description": "Stop training session",
  "inputSchema": {
    "type": "object",
    "required": ["training_id"],
    "properties": {
      "training_id": {
        "type": "string"
      },
      "save_checkpoint": {
        "type": "boolean",
        "default": true
      }
    }
  }
}
```

### 3. Inference Tools

#### `inference/analyze`
Perform lie detection analysis on multi-modal input.

```json
{
  "name": "inference/analyze",
  "description": "Analyze input for deception",
  "inputSchema": {
    "type": "object",
    "required": ["model_id"],
    "properties": {
      "model_id": {
        "type": "string"
      },
      "inputs": {
        "type": "object",
        "properties": {
          "video_path": {
            "type": "string",
            "description": "Path to video file or stream URL"
          },
          "audio_path": {
            "type": "string",
            "description": "Path to audio file or stream URL"
          },
          "text": {
            "type": "string",
            "description": "Text transcript or statement"
          },
          "physiological_data": {
            "type": "object",
            "description": "Physiological sensor data"
          }
        }
      },
      "options": {
        "type": "object",
        "properties": {
          "use_react": {
            "type": "boolean",
            "default": true,
            "description": "Use ReAct reasoning framework"
          },
          "explain": {
            "type": "boolean",
            "default": true,
            "description": "Generate explanation for decision"
          },
          "confidence_threshold": {
            "type": "number",
            "default": 0.7,
            "description": "Minimum confidence for decision"
          }
        }
      }
    }
  }
}
```

#### `inference/stream`
Start real-time streaming analysis.

```json
{
  "name": "inference/stream",
  "description": "Start real-time streaming analysis",
  "inputSchema": {
    "type": "object",
    "required": ["model_id", "stream_config"],
    "properties": {
      "model_id": {
        "type": "string"
      },
      "stream_config": {
        "type": "object",
        "properties": {
          "video_stream": {
            "type": "string",
            "description": "Video stream URL or device"
          },
          "audio_stream": {
            "type": "string",
            "description": "Audio stream URL or device"
          },
          "window_size": {
            "type": "integer",
            "default": 5,
            "description": "Analysis window in seconds"
          },
          "update_interval": {
            "type": "integer",
            "default": 1,
            "description": "Update interval in seconds"
          }
        }
      }
    }
  }
}
```

### 4. Monitoring Tools

#### `monitor/metrics`
Get current system metrics and performance data.

```json
{
  "name": "monitor/metrics",
  "description": "Get system metrics",
  "inputSchema": {
    "type": "object",
    "properties": {
      "include": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["cpu", "memory", "gpu", "inference_time", "accuracy"]
        },
        "default": ["cpu", "memory", "inference_time"]
      }
    }
  }
}
```

#### `monitor/alerts`
Configure performance and accuracy alerts.

```json
{
  "name": "monitor/alerts",
  "description": "Configure monitoring alerts",
  "inputSchema": {
    "type": "object",
    "properties": {
      "alerts": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "metric": {
              "type": "string"
            },
            "threshold": {
              "type": "number"
            },
            "condition": {
              "type": "string",
              "enum": ["above", "below", "equals"]
            },
            "action": {
              "type": "string",
              "enum": ["log", "webhook", "email"]
            }
          }
        }
      }
    }
  }
}
```

### 5. Configuration Tools

#### `config/get`
Get current configuration settings.

```json
{
  "name": "config/get",
  "description": "Get configuration settings",
  "inputSchema": {
    "type": "object",
    "properties": {
      "section": {
        "type": "string",
        "enum": ["all", "model", "training", "inference", "monitoring"]
      }
    }
  }
}
```

#### `config/set`
Update configuration settings.

```json
{
  "name": "config/set",
  "description": "Update configuration settings",
  "inputSchema": {
    "type": "object",
    "required": ["settings"],
    "properties": {
      "settings": {
        "type": "object",
        "description": "Configuration key-value pairs"
      },
      "persist": {
        "type": "boolean",
        "default": true
      }
    }
  }
}
```

## MCP Resource Definitions

### 1. Model Resources

#### `models/{model_id}/weights`
Access model weight files.

```json
{
  "uri": "models/{model_id}/weights",
  "name": "Model Weights",
  "mimeType": "application/octet-stream",
  "description": "Trained model weight files"
}
```

#### `models/{model_id}/config`
Model configuration and metadata.

```json
{
  "uri": "models/{model_id}/config",
  "name": "Model Configuration",
  "mimeType": "application/json",
  "description": "Model architecture and training configuration"
}
```

### 2. Dataset Resources

#### `datasets/{dataset_id}/manifest`
Dataset metadata and structure.

```json
{
  "uri": "datasets/{dataset_id}/manifest",
  "name": "Dataset Manifest",
  "mimeType": "application/json",
  "description": "Dataset structure and metadata"
}
```

#### `datasets/{dataset_id}/samples`
Access dataset samples.

```json
{
  "uri": "datasets/{dataset_id}/samples",
  "name": "Dataset Samples",
  "mimeType": "application/json",
  "description": "Sample data for preview and validation"
}
```

### 3. Results Resources

#### `results/{analysis_id}/report`
Detailed analysis report.

```json
{
  "uri": "results/{analysis_id}/report",
  "name": "Analysis Report",
  "mimeType": "application/json",
  "description": "Detailed deception analysis results with explanations"
}
```

#### `results/{analysis_id}/media`
Annotated media files from analysis.

```json
{
  "uri": "results/{analysis_id}/media",
  "name": "Annotated Media",
  "mimeType": "multipart/mixed",
  "description": "Media files with analysis annotations"
}
```

## Event Streaming

### Event Types

1. **Training Events**
   - `training.started`
   - `training.epoch_completed`
   - `training.metric_update`
   - `training.completed`
   - `training.failed`

2. **Inference Events**
   - `inference.started`
   - `inference.result`
   - `inference.stream_update`
   - `inference.completed`

3. **System Events**
   - `system.resource_alert`
   - `system.model_loaded`
   - `system.error`

### Event Schema

```json
{
  "type": "object",
  "properties": {
    "event_type": {
      "type": "string"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "data": {
      "type": "object"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "model_id": {
          "type": "string"
        },
        "session_id": {
          "type": "string"
        }
      }
    }
  }
}
```

## API Endpoints

### HTTP REST Endpoints

```
POST   /api/v1/models
GET    /api/v1/models
GET    /api/v1/models/{model_id}
DELETE /api/v1/models/{model_id}

POST   /api/v1/training/start
GET    /api/v1/training/{training_id}/status
POST   /api/v1/training/{training_id}/stop

POST   /api/v1/inference/analyze
POST   /api/v1/inference/stream/start
POST   /api/v1/inference/stream/{stream_id}/stop

GET    /api/v1/monitor/metrics
POST   /api/v1/monitor/alerts

GET    /api/v1/config
PUT    /api/v1/config
```

### WebSocket Endpoints

```
ws://localhost:3000/ws/events        # Event streaming
ws://localhost:3000/ws/inference     # Real-time inference
ws://localhost:3000/ws/monitoring    # Live metrics
```

## Security and Authentication

### Authentication Methods

1. **API Key Authentication**
   ```
   Authorization: Bearer <api_key>
   ```

2. **mTLS (Mutual TLS)**
   - Client certificate validation
   - Server certificate pinning

3. **OAuth 2.0 (Optional)**
   - For enterprise deployments
   - Integration with identity providers

### Authorization Levels

1. **Read-Only** - View models, results, and metrics
2. **Analyst** - Perform inference and analysis
3. **Trainer** - Train and manage models
4. **Admin** - Full system configuration

### Data Protection

1. **Encryption**
   - TLS 1.3 for all communications
   - AES-256 for data at rest
   - Key rotation every 90 days

2. **Privacy Controls**
   - PII redaction in logs
   - Configurable data retention
   - GDPR compliance features

## CLI Tool Integration

### Installation

```bash
cargo install veritas-cli
```

### Basic Commands

```bash
# Model management
veritas model list
veritas model create --name "detector-v1" --modalities vision,audio,text
veritas model load detector-v1 --device cuda

# Training
veritas train start --model detector-v1 --dataset strawberry-phi
veritas train status --id training-123
veritas train stop --id training-123

# Inference
veritas analyze --model detector-v1 --video interview.mp4 --audio interview.wav
veritas stream --model detector-v1 --camera 0 --microphone default

# Monitoring
veritas monitor metrics --live
veritas monitor alerts add --metric accuracy --below 0.8 --webhook https://alerts.example.com

# Configuration
veritas config get
veritas config set inference.confidence_threshold 0.75
```

### Interactive Mode

```bash
veritas repl
> load model detector-v1
Model loaded successfully
> analyze text "I did not take the money"
Analyzing...
Result: TRUTHFUL (confidence: 0.82)
Reasoning: Text analysis shows consistent language patterns...
>
```

## Example Usage Patterns

### 1. Basic Analysis

```rust
use veritas_mcp::{Client, AnalysisRequest};

async fn analyze_statement() -> Result<()> {
    let client = Client::connect("localhost:3000").await?;
    
    let request = AnalysisRequest {
        model_id: "detector-v1",
        inputs: Inputs {
            text: Some("I was at home all evening"),
            video_path: Some("/path/to/video.mp4"),
            ..Default::default()
        },
        options: AnalysisOptions {
            use_react: true,
            explain: true,
            ..Default::default()
        },
    };
    
    let result = client.analyze(request).await?;
    println!("Decision: {:?}", result.decision);
    println!("Confidence: {:.2}", result.confidence);
    println!("Explanation: {}", result.explanation);
    
    Ok(())
}
```

### 2. Real-Time Streaming

```rust
use veritas_mcp::{StreamClient, StreamConfig};

async fn stream_analysis() -> Result<()> {
    let client = StreamClient::connect("ws://localhost:3000/ws/inference").await?;
    
    let config = StreamConfig {
        model_id: "detector-v1",
        video_stream: Some("rtsp://camera.local/stream"),
        audio_stream: Some("device:microphone:0"),
        window_size: 5,
        update_interval: 1,
    };
    
    let mut stream = client.start_stream(config).await?;
    
    while let Some(update) = stream.next().await {
        println!("Time: {}, Deception Score: {:.2}", 
                 update.timestamp, update.deception_score);
        
        if update.deception_score > 0.8 {
            println!("HIGH DECEPTION DETECTED!");
            println!("Reasoning: {}", update.reasoning);
        }
    }
    
    Ok(())
}
```

### 3. Model Training

```rust
use veritas_mcp::{TrainingClient, TrainingConfig};

async fn train_model() -> Result<()> {
    let client = TrainingClient::connect("localhost:3000").await?;
    
    let config = TrainingConfig {
        model_id: "detector-v2",
        dataset_id: "strawberry-phi",
        epochs: 100,
        batch_size: 32,
        learning_rate: 0.001,
        enable_gspo: true,
        ..Default::default()
    };
    
    let training_id = client.start_training(config).await?;
    
    // Monitor training progress
    let mut events = client.subscribe_events(training_id).await?;
    
    while let Some(event) = events.next().await {
        match event {
            TrainingEvent::EpochCompleted { epoch, metrics } => {
                println!("Epoch {}: loss={:.4}, accuracy={:.4}", 
                         epoch, metrics.loss, metrics.accuracy);
            }
            TrainingEvent::Completed { final_metrics } => {
                println!("Training completed! Final accuracy: {:.4}", 
                         final_metrics.accuracy);
                break;
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-4)
- [ ] MCP server scaffold with Rust SDK
- [ ] Basic model management tools
- [ ] Simple inference pipeline
- [ ] CLI tool foundation

### Phase 2: Training & Models (Weeks 5-8)
- [ ] Training pipeline implementation
- [ ] Multi-modal model integration
- [ ] GSPO self-play optimization
- [ ] Model persistence and versioning

### Phase 3: Advanced Features (Weeks 9-12)
- [ ] ReAct reasoning engine
- [ ] Real-time streaming analysis
- [ ] Explainability tools (LIME/SHAP)
- [ ] Performance optimization

### Phase 4: Production Ready (Weeks 13-16)
- [ ] Security and authentication
- [ ] Monitoring and alerting
- [ ] Documentation and examples
- [ ] Performance benchmarks

## Conclusion

This MCP interface specification provides a comprehensive framework for building a production-ready lie detection system. The design emphasizes modularity, scalability, and ethical considerations while providing powerful tools for multi-modal deception analysis. The use of the Rust MCP SDK ensures type safety, performance, and reliability for mission-critical applications.

## Appendix: Error Codes

| Code | Name | Description |
|------|------|-------------|
| 1001 | MODEL_NOT_FOUND | Specified model does not exist |
| 1002 | MODEL_LOAD_FAILED | Failed to load model into memory |
| 2001 | TRAINING_FAILED | Training process encountered an error |
| 2002 | INVALID_DATASET | Dataset format or content is invalid |
| 3001 | INFERENCE_TIMEOUT | Inference exceeded time limit |
| 3002 | INSUFFICIENT_INPUT | Not enough modalities for analysis |
| 4001 | AUTH_FAILED | Authentication credentials invalid |
| 4002 | PERMISSION_DENIED | Insufficient permissions for operation |
| 5001 | RESOURCE_EXHAUSTED | System resources exceeded |
| 5002 | RATE_LIMIT_EXCEEDED | API rate limit reached |