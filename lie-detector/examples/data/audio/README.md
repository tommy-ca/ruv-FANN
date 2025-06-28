# Audio Data Directory

This directory should contain audio files for testing the Veritas Nexus audio analysis capabilities.

## Expected Files

Based on the sample data, the following audio files are referenced:

- `interview_001.wav` - Deceptive subject with vocal stress
- `interview_002.wav` - Truthful subject with normal speech patterns
- `interview_003.wav` - Uncertain response with hesitation
- `interview_005.wav` - Clear, confident truthful statement
- `interview_007.wav` - Strong denial with stress indicators
- `interview_008.wav` - Factual witness testimony
- `interview_009.wav` - Field recording with background noise
- `interview_011.wav` - Confident alibi statement
- `interview_012.wav` - Evasive response patterns
- `interview_013.wav` - Very confident truthful response
- `interview_015.wav` - Honest admission of guilt
- `interview_017.wav` - Truthful with evidence
- `interview_019.wav` - Formal denial statement
- `interview_020.wav` - Detailed truthful account
- `batch_001.wav` through `batch_010.wav` - Batch processing samples

## File Format Requirements

### Supported Formats
- **WAV** (recommended) - Uncompressed, high quality
- **FLAC** - Lossless compression
- **MP3** - Lossy compression (acceptable for most use cases)
- **M4A/AAC** - Good compression with reasonable quality
- **OGG** - Open source alternative

### Technical Specifications
- **Sample Rate**: Minimum 16kHz, recommended 44.1kHz or 48kHz
- **Bit Depth**: 16-bit minimum, 24-bit recommended
- **Channels**: Mono or stereo (mono preferred for speech)
- **Duration**: 30 seconds to 10 minutes typical
- **Bitrate**: 128kbps minimum for MP3, higher for better quality

### Audio Quality Requirements
- **Clear Speech**: Minimal background noise
- **Consistent Volume**: Avoid clipping or very low levels
- **No Distortion**: Clean, undistorted audio
- **Minimal Echo**: Avoid reverberant environments
- **Single Speaker**: Focus on one speaker at a time

## Creating Test Audio

For demonstration purposes, you can:

1. **Record speech samples**: Using microphone or phone
2. **Extract from videos**: Use existing video content
3. **Text-to-speech**: Generate synthetic speech
4. **Public domain**: Use appropriate license content

### Recording Guidelines

```bash
# Using ffmpeg to extract audio from video
ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output_audio.wav

# Convert to standard format
ffmpeg -i input_audio.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output_audio.wav

# Reduce noise (requires sox)
sox input_audio.wav output_audio.wav noisered noise_profile.txt 0.21

# Normalize audio levels
ffmpeg -i input_audio.wav -af "volume=0.5" output_audio.wav
```

## Audio Analysis Features

The Veritas Nexus audio analysis examines:

### Voice Stress Indicators
- **Fundamental Frequency (F0)**: Pitch variations and instability
- **Jitter**: Cycle-to-cycle variations in fundamental frequency
- **Shimmer**: Cycle-to-cycle variations in amplitude
- **Harmonics-to-Noise Ratio**: Voice quality assessment

### Speech Patterns
- **Speaking Rate**: Words per minute variations
- **Pause Patterns**: Hesitations, filled pauses ("um", "uh")
- **Volume Changes**: Sudden increases or decreases
- **Articulation**: Clarity and precision of speech

### Temporal Analysis
- **Response Latency**: Time to begin responding
- **Speech Rhythm**: Regular vs. irregular timing
- **Silence Duration**: Length of pauses between words
- **Overlap Patterns**: Interruptions or speech overlaps

### Spectral Features
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Speech characteristics
- **Spectral Centroid**: Brightness of the sound
- **Zero Crossing Rate**: Signal complexity measure
- **Energy Distribution**: Frequency band analysis

## Sample Audio Descriptions

Since actual audio files are not included, here are descriptions of what each referenced audio would contain:

- **interview_001.wav**: Nervous speech with pitch variations and hesitations
- **interview_002.wav**: Calm, steady speech with normal vocal characteristics
- **interview_003.wav**: Uncertain tone with many qualifiers and hesitations
- **interview_005.wav**: Confident, clear speech with stable vocal patterns
- **interview_007.wav**: Elevated stress levels with pitch instability
- **interview_008.wav**: Professional, factual delivery
- **interview_009.wav**: Field recording with some background noise
- **interview_011.wav**: Confident delivery with supporting details
- **interview_012.wav**: Evasive speech patterns with voice changes
- **interview_013.wav**: Very confident, clear articulation
- **interview_015.wav**: Emotional but honest admission
- **interview_017.wav**: Detailed, confident explanation
- **interview_019.wav**: Formal, controlled denial
- **interview_020.wav**: Natural, conversational truthful account

## Audio Processing Pipeline

The system processes audio through these stages:

1. **Preprocessing**: Noise reduction, normalization, segmentation
2. **Feature Extraction**: MFCC, pitch, energy, spectral features
3. **Voice Activity Detection**: Separate speech from silence
4. **Stress Analysis**: Voice stress and emotion indicators
5. **Pattern Recognition**: Deception-related speech patterns
6. **Temporal Analysis**: Timing and rhythm assessment

## Performance Considerations

### File Size vs. Quality
- **WAV files**: Larger but highest quality
- **FLAC**: Good compression with lossless quality
- **MP3**: Smaller files but some quality loss
- **Sample rate**: Higher rates provide more information but larger files

### Processing Speed
- **Duration**: Longer audio takes more time to process
- **Sample rate**: Higher rates require more computation
- **Complexity**: Multiple speakers or noise increase processing time
- **Real-time**: Streaming processing requires optimized parameters

## Data Privacy and Ethics

**IMPORTANT**: When working with real audio data:

1. **Consent**: Obtain explicit consent from all speakers
2. **Anonymization**: Consider voice modification for sensitive content
3. **Storage**: Use secure storage with encryption if needed
4. **Legal Compliance**: Follow wiretapping and recording laws
5. **Ethics Review**: Ensure compliance with research ethics requirements

## Testing Without Audio Files

The examples are designed to work even without actual audio files:

1. **Simulation Mode**: Examples will simulate audio analysis results
2. **Text-Only Analysis**: Many examples can run with just text input
3. **Mock Data**: Synthetic audio features are generated for testing
4. **Error Handling**: Graceful handling of missing audio files

## Audio Quality Assessment

The system can assess audio quality and adjust analysis accordingly:

- **Signal-to-Noise Ratio**: Quality of speech vs. background noise
- **Clipping Detection**: Identify overloaded or distorted audio
- **Frequency Response**: Ensure adequate frequency range
- **Channel Consistency**: For stereo recordings, check channel balance

## Troubleshooting Common Issues

### Low Quality Results
- Check sample rate (minimum 16kHz)
- Ensure clear speech without background noise
- Verify audio is not clipped or distorted
- Confirm single speaker focus

### Processing Errors
- Verify file format is supported
- Check file is not corrupted
- Ensure adequate audio duration (minimum 10 seconds)
- Confirm file permissions are correct

### Performance Issues
- Use appropriate sample rates for your use case
- Consider mono instead of stereo for speech
- Optimize file sizes for batch processing
- Enable GPU acceleration if available

To test with actual audio files, simply place them in this directory with the correct filenames as referenced in the sample data.