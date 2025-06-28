# Contributing to Veritas Nexus

Thank you for your interest in contributing to Veritas Nexus! This guide will help you get started with contributing to our multi-modal lie detection system.

## 📋 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [Getting Started](#-getting-started)
3. [Development Environment](#-development-environment)
4. [Project Structure](#-project-structure)
5. [Development Workflow](#-development-workflow)
6. [Coding Standards](#-coding-standards)
7. [Testing Guidelines](#-testing-guidelines)
8. [Documentation](#-documentation)
9. [Submitting Changes](#-submitting-changes)
10. [Review Process](#-review-process)
11. [Community Guidelines](#-community-guidelines)
12. [Recognition](#-recognition)

## 📜 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Our Pledge

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome people of all backgrounds and experience levels
- **Be constructive**: Focus on helping each other learn and grow
- **Be ethical**: Consider the ethical implications of AI and deception detection

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or trolling
- Publishing private information without consent
- Contributions that promote harmful or unethical uses of the technology

## 🚀 Getting Started

### Ways to Contribute

- **Code contributions**: Bug fixes, new features, optimizations
- **Documentation**: Improve guides, tutorials, and API documentation  
- **Testing**: Write tests, report bugs, test on different platforms
- **Research**: Contribute algorithms, datasets, or academic insights
- **Community**: Help others in discussions, review pull requests
- **Ethics**: Help ensure responsible AI development

### Before You Start

1. **Check existing issues**: Look for related issues or discussions
2. **Read the documentation**: Familiarize yourself with the project
3. **Understand the ethics**: Review our [Ethics Guide](ETHICS.md)
4. **Join the community**: Connect with us on Discord or Discussions

## 🔧 Development Environment

### Prerequisites

```bash
# Install Rust (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version  # Should be 1.70+
cargo --version

# Install additional tools
cargo install cargo-watch
cargo install cargo-tarpaulin  # For coverage
cargo install cargo-audit      # For security audits
cargo install cargo-deny       # For license checking
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    clang \
    lldb

# macOS
brew install cmake pkg-config openssl

# For GPU development (optional)
# Follow CUDA installation guide for your platform
```

### Clone and Setup

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/yourusername/veritas-nexus.git
cd veritas-nexus

# Add upstream remote
git remote add upstream https://github.com/original-owner/veritas-nexus.git

# Install pre-commit hooks
cargo install cargo-husky
```

### Environment Configuration

```bash
# Create .env file for development
cat > .env << 'EOF'
RUST_LOG=veritas_nexus=debug
VERITAS_ENV=development
VERITAS_MODELS_PATH=./models
EOF

# Download development models
mkdir -p models
# Download smaller models for development
curl -L https://github.com/veritas-nexus/models/releases/latest/download/dev-models.tar.gz | tar xz -C models/
```

### IDE Setup

#### VS Code

```json
// .vscode/settings.json
{
    "rust-analyzer.cargo.features": ["parallel", "gpu"],
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.checkOnSave.extraArgs": ["--", "-D", "warnings"],
    "rust-analyzer.rustfmt.extraArgs": ["+nightly"],
    "files.associations": {
        "*.rs": "rust"
    }
}
```

#### Recommended Extensions

- rust-analyzer
- CodeLLDB (for debugging)
- Better TOML
- GitLens

## 🏗️ Project Structure

```
veritas-nexus/
├── src/
│   ├── lib.rs              # Main library entry point
│   ├── error.rs            # Error types and handling
│   ├── types.rs            # Core type definitions
│   ├── prelude.rs          # Common imports
│   ├── modalities/         # Individual analysis modules
│   │   ├── vision/         # Computer vision analysis
│   │   ├── audio/          # Audio/speech analysis
│   │   ├── text/           # Natural language processing
│   │   └── physiological/  # Biometric signal processing
│   ├── fusion/             # Multi-modal fusion strategies
│   ├── agents/             # ReAct reasoning agents
│   ├── learning/           # Machine learning algorithms
│   ├── optimization/       # Performance optimizations
│   ├── streaming/          # Real-time processing
│   ├── mcp/                # Model Context Protocol
│   └── utils/              # Utility functions
├── tests/                  # Test suites
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── property/           # Property-based tests
├── benches/                # Benchmark suites
├── examples/               # Usage examples
├── docs/                   # Documentation
└── tools/                  # Development tools
```

### Module Responsibilities

- **modalities/**: Implement individual analysis techniques
- **fusion/**: Combine evidence from multiple modalities
- **agents/**: High-level reasoning and decision making
- **learning/**: Training and adaptation algorithms
- **optimization/**: Performance and efficiency improvements
- **streaming/**: Real-time processing capabilities
- **mcp/**: External integration and management

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-number-description
```

### 2. Development Cycle

```bash
# Make your changes
# ...

# Run tests frequently
cargo test

# Run with watch for continuous testing
cargo watch -x test

# Check formatting
cargo fmt --check

# Run linting
cargo clippy -- -D warnings

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture
```

### 3. Code Quality Checks

```bash
# Full quality check before committing
make check  # Or run manually:

cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo audit
cargo deny check
```

### 4. Performance Testing

```bash
# Run benchmarks
cargo bench

# Profile specific functionality
cargo bench --bench vision_benchmarks

# Memory usage analysis
cargo test --features profiling
```

## 📏 Coding Standards

### Rust Style Guide

We follow the official [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/) with these additions:

#### Code Formatting

```bash
# Use rustfmt with nightly for all features
cargo +nightly fmt

# Configuration in rustfmt.toml
edition = "2021"
max_width = 100
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Max"
```

#### Naming Conventions

```rust
// Use descriptive names
struct VisionAnalyzer {  // Not: VA or Analyzer
    face_detector: FaceDetector,  // Not: fd or detector
}

// Constants in SCREAMING_SNAKE_CASE
const MAX_BATCH_SIZE: usize = 32;
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.7;

// Functions and variables in snake_case
fn analyze_deception_patterns(input_text: &str) -> Result<DeceptionScore> {
    // Implementation
}
```

#### Error Handling

```rust
// Use Result for fallible operations
fn load_model(path: &str) -> Result<Model, VeritasError> {
    // Implementation
}

// Create specific error types
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Invalid input format: {0}")]
    InvalidFormat(String),
    
    #[error("Model not found: {path}")]
    ModelNotFound { path: String },
    
    #[error("Processing failed: {source}")]
    ProcessingFailed {
        #[from]
        source: std::io::Error,
    },
}
```

#### Documentation

```rust
/// Analyzes text for deception indicators using linguistic patterns.
///
/// This function processes natural language text and extracts features
/// that may indicate deceptive communication patterns. It uses advanced
/// NLP techniques including BERT embeddings and statistical analysis.
///
/// # Arguments
///
/// * `text` - The input text to analyze (must be non-empty)
/// * `config` - Configuration parameters for analysis
///
/// # Returns
///
/// Returns a `DeceptionScore` containing the analysis results, including
/// confidence scores and feature attributions.
///
/// # Errors
///
/// Returns `AnalysisError::InvalidFormat` if the text is empty or contains
/// only whitespace.
///
/// # Examples
///
/// ```rust
/// use veritas_nexus::text::analyze_text;
/// use veritas_nexus::TextConfig;
///
/// let config = TextConfig::default();
/// let score = analyze_text("I was definitely not there", &config)?;
/// assert!(score.probability > 0.0);
/// ```
pub fn analyze_text(text: &str, config: &TextConfig) -> Result<DeceptionScore> {
    // Implementation
}
```

#### Performance Guidelines

```rust
// Prefer `&str` over `String` for parameters
fn process_text(text: &str) -> Result<ProcessedText> {  // Good
    // vs fn process_text(text: String) -> Result<ProcessedText>  // Bad
}

// Use appropriate data structures
use std::collections::HashMap;  // For key-value lookups
use std::collections::BTreeMap; // For ordered keys
use smallvec::SmallVec;         // For small vectors

// Avoid unnecessary allocations
fn process_items(items: &[Item]) -> Vec<ProcessedItem> {
    items.iter()
        .filter(|item| item.is_valid())  // No allocation
        .map(|item| item.process())      // Transform
        .collect()                       // Single allocation
}
```

#### Async/Await Guidelines

```rust
// Use async when doing I/O or CPU-intensive work
async fn analyze_batch(items: &[AnalysisInput]) -> Result<Vec<AnalysisResult>> {
    // Process items in parallel
    let futures: Vec<_> = items.iter()
        .map(|item| analyze_single(item))
        .collect();
    
    futures::future::try_join_all(futures).await
}

// Use proper error propagation
async fn load_and_analyze(path: &str) -> Result<AnalysisResult> {
    let data = load_file(path).await?;  // ? operator
    let result = analyze_data(&data).await?;
    Ok(result)
}
```

## 🧪 Testing Guidelines

### Test Organization

```rust
// Unit tests in the same file
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Test implementation
    }
    
    #[tokio::test]
    async fn test_async_functionality() {
        // Async test implementation
    }
}

// Integration tests in tests/ directory
// tests/integration/vision_tests.rs
use veritas_nexus::prelude::*;

#[tokio::test]
async fn test_vision_analysis_integration() {
    let analyzer = VisionAnalyzer::new().await.unwrap();
    // Test implementation
}
```

### Test Categories

#### Unit Tests
```rust
#[test]
fn test_deception_score_calculation() {
    let features = vec![
        Feature { name: "hesitation".to_string(), value: 0.8, weight: 0.3 },
        Feature { name: "contradiction".to_string(), value: 0.6, weight: 0.7 },
    ];
    
    let score = calculate_deception_score(&features);
    assert!((score - 0.68).abs() < 0.01);
}
```

#### Property-Based Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_deception_score_bounds(
        features in prop::collection::vec(
            (prop::string::string_regex("[a-z]+").unwrap(), 0.0f32..1.0, 0.0f32..1.0),
            1..10
        )
    ) {
        let features: Vec<Feature> = features.into_iter()
            .map(|(name, value, weight)| Feature { name, value, weight })
            .collect();
        
        let score = calculate_deception_score(&features);
        prop_assert!(score >= 0.0 && score <= 1.0);
    }
}
```

#### Integration Tests
```rust
#[tokio::test]
async fn test_end_to_end_analysis() {
    let detector = LieDetector::builder()
        .with_test_config()
        .build()
        .await
        .unwrap();
    
    let input = create_test_input();
    let result = detector.analyze(input).await.unwrap();
    
    assert!(result.confidence > 0.0);
    assert!(result.confidence <= 1.0);
    assert!(!result.reasoning_trace.is_empty());
}
```

#### Benchmark Tests
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_text_analysis(c: &mut Criterion) {
    let analyzer = TextAnalyzer::new();
    let text = "Sample text for benchmarking performance";
    
    c.bench_function("text_analysis", |b| {
        b.iter(|| analyzer.analyze(black_box(text)))
    });
}

criterion_group!(benches, bench_text_analysis);
criterion_main!(benches);
```

### Test Data Management

```rust
// Create test utilities
pub mod test_utils {
    use super::*;
    
    pub fn create_test_video_frame() -> VideoFrame {
        VideoFrame {
            data: vec![0u8; 640 * 480 * 3],
            width: 640,
            height: 480,
            timestamp: std::time::Instant::now(),
            format: VideoFormat::Rgb24,
        }
    }
    
    pub fn create_test_analysis_input() -> AnalysisInput {
        AnalysisInput {
            video_path: Some("tests/data/sample_video.mp4".to_string()),
            audio_path: Some("tests/data/sample_audio.wav".to_string()),
            transcript: Some("Test transcript for analysis".to_string()),
            physiological_data: Some(vec![72.0, 73.5, 75.0]),
        }
    }
}
```

### Test Coverage

```bash
# Generate coverage report
cargo tarpaulin --out Html

# Aim for >80% coverage for new code
# Focus on critical paths and error handling
```

## 📚 Documentation

### Documentation Types

#### Code Documentation
```rust
//! Module-level documentation goes here.
//! 
//! This module provides functionality for analyzing facial expressions
//! and micro-expressions in video data for deception detection.

/// Struct documentation describes the purpose and usage.
pub struct FaceAnalyzer {
    /// The underlying face detection model.
    detector: FaceDetector,
    /// Configuration parameters for analysis.
    config: FaceAnalysisConfig,
}

impl FaceAnalyzer {
    /// Creates a new face analyzer with the specified configuration.
    /// 
    /// # Arguments
    /// 
    /// * `config` - Configuration parameters
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let config = FaceAnalysisConfig::default();
    /// let analyzer = FaceAnalyzer::new(config)?;
    /// ```
    pub fn new(config: FaceAnalysisConfig) -> Result<Self> {
        // Implementation
    }
}
```

#### README Files
- Keep READMEs concise but informative
- Include practical examples
- Update when adding new features

#### User Guides
- Step-by-step instructions
- Real-world examples
- Troubleshooting sections

### Documentation Standards

```rust
// Use present tense
/// Analyzes the input video for facial expressions.  // Good
/// Will analyze the input video for facial expressions.  // Bad

// Be specific about behavior
/// Returns `None` if no face is detected in the frame.  // Good
/// May return `None` in some cases.  // Bad

// Include error conditions
/// # Errors
/// 
/// Returns `AnalysisError::InvalidFormat` if the video format is not supported.
/// Returns `AnalysisError::ProcessingFailed` if face detection fails.
```

## 📤 Submitting Changes

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>[optional scope]: <description>

# Types:
feat:     # New feature
fix:      # Bug fix
docs:     # Documentation changes
style:    # Code style changes (formatting, etc.)
refactor: # Code refactoring
perf:     # Performance improvements
test:     # Adding or updating tests
chore:    # Maintenance tasks

# Examples:
git commit -m "feat(vision): add micro-expression detection"
git commit -m "fix(audio): resolve memory leak in pitch detection"
git commit -m "docs: update installation guide for GPU setup"
git commit -m "perf(fusion): optimize attention mechanism computation"
```

### Pull Request Process

#### 1. Prepare Your Branch

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run quality checks
make check

# Update documentation if needed
cargo doc --no-deps --open
```

#### 2. Create Pull Request

```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Benchmarks run (if performance-related)

## Performance Impact
Describe any performance implications of the changes.

## Ethical Considerations
Address any ethical implications, especially for new detection methods.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] No new compiler warnings
```

#### 3. PR Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Write clear descriptions**: Explain what and why
- **Include tests**: Ensure good test coverage
- **Update documentation**: Keep docs in sync
- **Be responsive**: Address review feedback promptly

### Breaking Changes

For breaking changes:

1. **Discuss first**: Open an issue to discuss the change
2. **Migration guide**: Provide clear migration instructions
3. **Deprecation period**: When possible, deprecate before removing
4. **Version bump**: Follow semantic versioning

## 🔍 Review Process

### Review Criteria

Reviews focus on:

- **Correctness**: Does the code work as intended?
- **Performance**: Are there performance implications?
- **Security**: Are there security vulnerabilities?
- **Ethics**: Does it align with ethical guidelines?
- **Maintainability**: Is the code easy to understand and maintain?
- **Testing**: Is there adequate test coverage?

### Review Workflow

1. **Automated checks**: CI must pass
2. **Peer review**: At least one approving review required
3. **Maintainer review**: Core maintainer approval for significant changes
4. **Ethics review**: Additional review for sensitive changes

### As a Reviewer

- **Be constructive**: Focus on helping improve the code
- **Be specific**: Provide actionable feedback
- **Be timely**: Respond within 2-3 business days
- **Ask questions**: If something is unclear, ask for clarification

### As an Author

- **Be responsive**: Address feedback promptly
- **Be open**: Consider different perspectives
- **Be patient**: Review takes time
- **Learn**: Use feedback to improve future contributions

## 👥 Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat and community support
- **Email**: Security issues and private matters

### Best Practices

- **Search first**: Check if your question has been asked before
- **Be clear**: Provide context and specific details
- **Be patient**: Maintainers are volunteers
- **Help others**: Answer questions when you can
- **Stay on topic**: Keep discussions focused

### Mentorship

We encourage experienced contributors to mentor newcomers:

- **Good first issues**: We label beginner-friendly issues
- **Pair programming**: Offer to work together on features
- **Code reviews**: Provide detailed, educational feedback
- **Documentation**: Help improve onboarding materials

## 🏆 Recognition

We appreciate all contributions and recognize them in several ways:

### Contributors

- **Contributors file**: All contributors are listed
- **Release notes**: Significant contributions are highlighted
- **Social media**: We share and celebrate contributions

### Maintainers

Long-term contributors may be invited to become maintainers, with additional responsibilities:

- **Code review**: Help review pull requests
- **Issue triage**: Help organize and prioritize issues
- **Release management**: Assist with releases
- **Community building**: Help grow the community

### Hall of Fame

Exceptional contributors are recognized in our Hall of Fame for:

- **Significant features**: Major functionality additions
- **Performance improvements**: Substantial optimizations
- **Research contributions**: Novel algorithms or insights
- **Community leadership**: Helping build and nurture the community

## 📞 Getting Help

### For Contributors

- **Discord**: Join our contributor channel
- **Office hours**: Weekly video calls for discussion
- **Documentation**: Comprehensive guides and tutorials
- **Mentorship**: Connect with experienced contributors

### For Maintainers

- **Maintainer guide**: Detailed processes and procedures
- **Team meetings**: Regular sync meetings
- **Decision records**: Document important decisions
- **Tooling**: Automated tools for common tasks

## 🚀 Future Roadmap

Stay informed about project direction:

- **Roadmap document**: High-level vision and goals
- **Milestone planning**: Quarterly planning sessions
- **RFC process**: Request for Comments on major changes
- **Community input**: Regular feedback collection

---

Thank you for contributing to Veritas Nexus! Together, we're building responsible and effective AI for deception detection. Every contribution, no matter how small, makes a difference.

For questions about contributing, please reach out:
- **Email**: contributors@veritas-nexus.ai
- **Discord**: [Contributor Channel](https://discord.gg/veritas-nexus-contributors)
- **GitHub**: [@veritas-nexus-maintainers](https://github.com/orgs/veritas-nexus/teams/maintainers)