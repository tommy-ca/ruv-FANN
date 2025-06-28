# Example Data

This directory contains sample data files for testing and demonstrating the Veritas Nexus lie detection system.

## File Structure

```
data/
├── README.md                    # This file
├── samples.csv                  # CSV format sample data
├── test_texts.json             # Text analysis test cases
├── batch_input.jsonl           # JSON Lines format for batch processing
├── physiological_sample.json   # Sample physiological data
├── config_template.toml        # Configuration template
└── video/                      # Video files (placeholder)
    └── README.md
└── audio/                      # Audio files (placeholder)
    └── README.md
```

## Data Formats

### CSV Format (`samples.csv`)

Standard CSV format with the following columns:
- `id`: Unique sample identifier
- `video_path`: Path to video file (optional)
- `audio_path`: Path to audio file (optional)
- `text`: Text content to analyze
- `label`: Expected classification (truthful/deceptive/uncertain)
- `confidence`: Ground truth confidence (0.0-1.0)
- `environment`: Recording environment (controlled/field)
- `subject_age`: Subject age
- `baseline_available`: Whether baseline data exists

### JSON Text Tests (`test_texts.json`)

Structured JSON with test cases for text analysis:
```json
{
  "test_cases": [
    {
      "id": "test_001",
      "text": "Sample text to analyze",
      "expected_label": "truthful|deceptive|uncertain",
      "reasoning": "Explanation of expected result",
      "language": "english|spanish|french",
      "complexity": "low|medium|high"
    }
  ],
  "metadata": {
    "total_samples": 15,
    "languages": ["english", "spanish", "french"],
    "label_distribution": {...}
  }
}
```

### JSON Lines Batch Input (`batch_input.jsonl`)

One JSON object per line for streaming/batch processing:
```json
{"id": "batch_001", "text": "Sample text", "video_path": null, "audio_path": "audio.wav", "metadata": {...}}
```

### Physiological Data (`physiological_sample.json`)

Comprehensive physiological signal data:
```json
{
  "session_id": "physio_001",
  "data": {
    "heart_rate": {
      "unit": "bpm",
      "baseline": 72.5,
      "values": [72.3, 72.8, ...],
      "timestamps": ["2024-01-15T14:30:00Z", ...]
    },
    "skin_conductance": {...},
    "breathing_rate": {...},
    "blood_pressure": {...}
  },
  "annotations": [...],
  "analysis": {...}
}
```

## Usage Examples

### Loading CSV Data

```rust
use std::fs::File;
use csv::Reader;

let file = File::open("examples/data/samples.csv")?;
let mut reader = Reader::from_reader(file);

for result in reader.records() {
    let record = result?;
    let id = &record[0];
    let text = &record[3];
    // Process record...
}
```

### Loading JSON Test Cases

```rust
use serde_json;
use std::fs;

let content = fs::read_to_string("examples/data/test_texts.json")?;
let test_data: serde_json::Value = serde_json::from_str(&content)?;

for test_case in test_data["test_cases"].as_array().unwrap() {
    let text = test_case["text"].as_str().unwrap();
    let expected = test_case["expected_label"].as_str().unwrap();
    // Process test case...
}
```

### Processing JSON Lines

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

let file = File::open("examples/data/batch_input.jsonl")?;
let reader = BufReader::new(file);

for line in reader.lines() {
    let json_str = line?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;
    // Process JSON object...
}
```

### Loading Physiological Data

```rust
use serde_json;
use std::fs;

let content = fs::read_to_string("examples/data/physiological_sample.json")?;
let physio_data: serde_json::Value = serde_json::from_str(&content)?;

let heart_rate = physio_data["data"]["heart_rate"]["values"]
    .as_array()
    .unwrap()
    .iter()
    .map(|v| v.as_f64().unwrap() as f32)
    .collect::<Vec<f32>>();
```

## Data Generation

This sample data is synthetically generated for demonstration purposes. For real applications:

1. **Video Data**: Should be MP4 files with clear facial visibility
2. **Audio Data**: Should be WAV files with clear speech at 16kHz+ sample rate
3. **Text Data**: Should be interview transcripts or statements
4. **Physiological Data**: Should be from calibrated sensors at appropriate sample rates

## Creating Custom Data

### Video Requirements
- Format: MP4, AVI, or MOV
- Resolution: 480p minimum, 1080p recommended
- Frame rate: 24-30 FPS
- Duration: 30 seconds to 10 minutes
- Quality: Clear facial visibility, good lighting

### Audio Requirements
- Format: WAV, MP3, or FLAC
- Sample rate: 16kHz minimum, 44.1kHz recommended
- Channels: Mono or stereo
- Duration: Should match video if both present
- Quality: Clear speech, minimal background noise

### Text Requirements
- Format: Plain text, UTF-8 encoded
- Length: 10-1000 words typically
- Content: Natural speech or written statements
- Language: Specify language for optimal processing

### Physiological Requirements
- Sensors: Heart rate, skin conductance, breathing recommended
- Sample rate: 100Hz minimum, 1kHz recommended
- Duration: At least 1 minute for baseline establishment
- Calibration: Use calibrated sensors with known baselines

## Data Privacy and Ethics

When working with real data:

1. **Consent**: Ensure proper informed consent from all subjects
2. **Anonymization**: Remove or encrypt personally identifiable information
3. **Storage**: Use secure storage with appropriate access controls
4. **Retention**: Follow data retention policies and legal requirements
5. **Ethics**: Ensure research complies with institutional review board requirements

## Performance Considerations

### File Sizes
- CSV: Efficient for tabular data, good for < 100MB
- JSON: Human-readable but larger, good for < 50MB
- JSON Lines: Streamable, good for any size
- Binary formats: Most efficient for large datasets

### Processing Tips
- Use batch processing for large datasets
- Enable caching for repeated analyses
- Consider GPU acceleration for video/audio processing
- Use appropriate batch sizes based on available memory

## Example Usage in Code

See the following examples for practical usage:

- `basic_detection.rs`: Simple single-file analysis
- `batch_processing.rs`: Large-scale data processing
- `text_analysis_demo.rs`: Text-focused analysis
- `real_time_analysis.rs`: Streaming data processing

## Extending the Dataset

To add new test cases:

1. Follow the existing format conventions
2. Include appropriate metadata
3. Validate data quality and format
4. Add corresponding documentation
5. Test with existing examples to ensure compatibility

For questions about data formats or creating custom datasets, see the main documentation or example files.