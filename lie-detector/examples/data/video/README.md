# Video Data Directory

This directory should contain video files for testing the Veritas Nexus vision analysis capabilities.

## Expected Files

Based on the sample data, the following video files are referenced:

- `interview_001.mp4` - Interview with deceptive subject
- `interview_002.mp4` - Interview with truthful subject  
- `interview_004.mp4` - Field interview video
- `interview_005.mp4` - Controlled environment interview
- `interview_007.mp4` - High-confidence deception case
- `interview_008.mp4` - Truthful witness statement
- `interview_010.mp4` - Denial with deceptive indicators
- `interview_011.mp4` - Truthful alibi with evidence
- `interview_013.mp4` - Strong truthful statement
- `interview_014.mp4` - Uncertain/ambiguous case
- `interview_015.mp4` - Admission of guilt
- `interview_017.mp4` - Truthful with corroborating evidence
- `interview_018.mp4` - Evasive deceptive response
- `interview_020.mp4` - Clear truthful statement
- `batch_002.mp4` - Batch processing sample
- `batch_004.mp4` - Batch processing sample
- `batch_005.mp4` - Batch processing sample
- `batch_007.mp4` - Batch processing sample
- `batch_008.mp4` - Batch processing sample
- `batch_010.mp4` - Batch processing sample

## File Format Requirements

### Supported Formats
- **MP4** (recommended) - H.264 video codec
- **AVI** - Various codecs supported
- **MOV** - QuickTime format
- **MKV** - Matroska container

### Technical Specifications
- **Resolution**: Minimum 480p (854×480), recommended 720p (1280×720) or higher
- **Frame Rate**: 24-30 FPS recommended
- **Duration**: 30 seconds to 10 minutes typical
- **Bitrate**: 1-5 Mbps for good quality
- **Audio**: Optional, if present should be synchronized

### Content Requirements
- **Facial Visibility**: Subject's face should be clearly visible
- **Lighting**: Adequate lighting for facial feature detection
- **Angle**: Front-facing or slight angle preferred
- **Stability**: Minimal camera shake or movement
- **Background**: Preferably simple, non-distracting background

## Creating Test Videos

For demonstration purposes, you can:

1. **Use webcam recordings**: Simple talking head videos
2. **Screen recordings**: Of video calls or interviews
3. **Stock footage**: Ensure licensing allows use
4. **Synthetic data**: AI-generated talking head videos

### Recording Guidelines

```bash
# Using ffmpeg to convert video to suitable format
ffmpeg -i input_video.mov -c:v libx264 -preset medium -crf 23 -c:a aac -ac 2 -ar 44100 output_video.mp4

# Extract frames for analysis
ffmpeg -i input_video.mp4 -vf fps=1 frame_%04d.png

# Resize video to standard resolution
ffmpeg -i input_video.mp4 -vf scale=1280:720 -c:a copy resized_video.mp4
```

## Data Privacy and Ethics

**IMPORTANT**: When working with real video data:

1. **Consent**: Obtain explicit consent from all subjects
2. **Anonymization**: Consider face blurring for sensitive content
3. **Storage**: Use secure storage with proper access controls
4. **Legal Compliance**: Follow local privacy laws and regulations
5. **Ethics Review**: Ensure compliance with institutional review boards

## Sample Video Descriptions

Since actual video files are not included in this repository, here are descriptions of what each referenced video would contain:

- **interview_001.mp4**: Subject denying involvement, showing micro-expressions of stress
- **interview_002.mp4**: Subject providing truthful account with consistent facial expressions
- **interview_004.mp4**: Field interview with some environmental noise and lighting challenges
- **interview_005.mp4**: Clean, controlled environment interview with clear facial features
- **interview_007.mp4**: Strong denial with visible signs of deception
- **interview_008.mp4**: Witness providing factual testimony
- **interview_010.mp4**: Deflection and blame-shifting behavior
- **interview_011.mp4**: Confident, truthful statement with supporting evidence
- **interview_013.mp4**: Very confident truthful response
- **interview_014.mp4**: Genuinely uncertain response
- **interview_015.mp4**: Honest admission of wrongdoing
- **interview_017.mp4**: Truthful alibi with photographic evidence
- **interview_018.mp4**: Evasive, unclear responses
- **interview_020.mp4**: Clear, detailed truthful statement

## Vision Analysis Features

The Veritas Nexus vision analysis looks for:

- **Facial Micro-expressions**: Brief, involuntary facial expressions
- **Eye Movement Patterns**: Gaze direction, blink rate, eye contact
- **Head Pose Changes**: Head orientation and movement patterns
- **Facial Asymmetry**: Asymmetric expressions or movements
- **Timing Analysis**: Delays in facial responses
- **Baseline Comparison**: Deviation from individual normal behavior

## Performance Considerations

- **File Size**: Larger files take longer to process
- **Resolution**: Higher resolution provides more detail but requires more processing power
- **Duration**: Longer videos provide more data but increase processing time
- **Compression**: Too much compression can reduce analysis accuracy

## Testing Without Video Files

The examples are designed to work even without actual video files:

1. **Simulation Mode**: Examples will simulate video analysis results
2. **Text-Only Analysis**: Many examples can run with just text input
3. **Mock Data**: Synthetic vision features are generated for testing
4. **Error Handling**: Graceful handling of missing video files

To test with actual video files, simply place them in this directory with the correct filenames as referenced in the sample data.