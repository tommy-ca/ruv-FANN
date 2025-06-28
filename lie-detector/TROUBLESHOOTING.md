# Veritas Nexus Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using Veritas Nexus. Solutions are organized by category and include step-by-step instructions.

## 📋 Table of Contents

1. [Installation Issues](#-installation-issues)
2. [Model Loading Problems](#-model-loading-problems)
3. [GPU and Hardware Issues](#-gpu-and-hardware-issues)
4. [Performance Problems](#-performance-problems)
5. [Memory Issues](#-memory-issues)
6. [Network and Connectivity](#-network-and-connectivity)
7. [Input/Output Errors](#-inputoutput-errors)
8. [Analysis and Accuracy Issues](#-analysis-and-accuracy-issues)
9. [Deployment Problems](#-deployment-problems)
10. [Logging and Debugging](#-logging-and-debugging)
11. [Platform-Specific Issues](#-platform-specific-issues)
12. [Getting Help](#-getting-help)

## 🛠️ Installation Issues

### Problem: Compilation Fails with Missing Dependencies

**Symptoms:**
```
error: failed to run custom build command for `openssl-sys v0.9.87`
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install pkg-config libssl-dev build-essential cmake

# RHEL/CentOS/Fedora
sudo yum install openssl-devel pkgconfig gcc gcc-c++ cmake
# or
sudo dnf install openssl-devel pkgconfig gcc gcc-c++ cmake

# macOS
brew install openssl cmake pkg-config
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"

# Then retry compilation
cargo clean
cargo build --release
```

### Problem: Rust Version Too Old

**Symptoms:**
```
error: package `veritas-nexus v0.1.0` cannot be built because it requires rustc 1.70 or newer
```

**Solution:**
```bash
# Update Rust to latest stable
rustup update stable
rustup default stable

# Verify version
rustc --version
# Should show 1.70 or newer

# Clean and rebuild
cargo clean
cargo build --release
```

### Problem: Feature Flag Errors

**Symptoms:**
```
error: failed to select a version for the requirement `candle-core = "^0.4"`
```

**Solution:**
```bash
# Check your Cargo.toml features
[dependencies]
veritas-nexus = { version = "0.1.0", features = ["gpu", "parallel"] }

# For systems without GPU support
[dependencies]
veritas-nexus = { version = "0.1.0", features = ["parallel"] }

# Build with specific features
cargo build --release --features "parallel,simd-avx2" --no-default-features
```

### Problem: Cross-Compilation Issues

**Symptoms:**
```
error: linking with `cc` failed: exit status: 1
```

**Solution:**
```bash
# Install cross-compilation target
rustup target add x86_64-unknown-linux-gnu

# Install cross-compilation tools
sudo apt install gcc-multilib

# For ARM64 targets
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu

# Set environment variables
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

# Cross-compile
cargo build --target aarch64-unknown-linux-gnu --release
```

## 🤖 Model Loading Problems

### Problem: Model Files Not Found

**Symptoms:**
```
Error: ModelLoadError("Vision model not found at /opt/veritas-nexus/models/vision_model.onnx")
```

**Solution:**
```bash
# Check if model directory exists
ls -la /opt/veritas-nexus/models/

# Download models if missing
mkdir -p /opt/veritas-nexus/models
cd /opt/veritas-nexus/models

# Download from releases
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/models.tar.gz | tar xz

# Or set custom model path
export VERITAS_MODELS_PATH="/path/to/your/models"

# Verify permissions
sudo chown -R $USER:$USER /opt/veritas-nexus/models
chmod -R 644 /opt/veritas-nexus/models/*
```

### Problem: Model Version Incompatibility

**Symptoms:**
```
Error: ModelLoadError("Model version 2.1 not compatible with runtime version 2.0")
```

**Solution:**
```bash
# Check model versions
veritas-nexus model info --path /opt/veritas-nexus/models/

# Download compatible models
veritas-nexus model download --version 2.0

# Or update Veritas Nexus
cargo install veritas-nexus --force
```

### Problem: Corrupted Model Files

**Symptoms:**
```
Error: ModelLoadError("Failed to parse model: invalid magic number")
```

**Solution:**
```bash
# Verify file integrity
sha256sum /opt/veritas-nexus/models/vision_model.onnx
# Compare with expected checksums from releases

# Re-download if corrupted
rm /opt/veritas-nexus/models/vision_model.onnx
curl -L -o /opt/veritas-nexus/models/vision_model.onnx \
  https://github.com/yourusername/veritas-nexus/releases/latest/download/vision_model.onnx

# Verify download
sha256sum /opt/veritas-nexus/models/vision_model.onnx
```

### Problem: Out of Memory During Model Loading

**Symptoms:**
```
Error: ModelLoadError("Failed to allocate 2.1GB for model weights")
```

**Solution:**
```rust
// Use memory-mapped models for large files
let config = ModelConfig {
    enable_memory_mapping: true,
    lazy_loading: true,
    quantization: ModelQuantization::Int8, // Reduce memory usage
};

let detector = LieDetector::builder()
    .with_model_config(config)
    .build()
    .await?;
```

## 🎮 GPU and Hardware Issues

### Problem: CUDA Not Detected

**Symptoms:**
```
Error: GpuError("CUDA runtime not found")
```

**Solution:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install CUDA toolkit if missing
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Restart and test
sudo reboot
nvidia-smi
```

### Problem: GPU Out of Memory

**Symptoms:**
```
Error: GpuError("CUDA out of memory. Tried to allocate 2.00GB")
```

**Solution:**
```rust
// Reduce GPU memory usage
let gpu_config = GpuConfig {
    memory_limit_mb: 4096, // Limit to 4GB
    batch_size: 8,         // Smaller batches
    fp16_inference: true,  // Half precision
    enable_memory_pool: true,
    pool_size_mb: 2048,
};

let detector = LieDetector::builder()
    .with_gpu_config(gpu_config)
    .build()
    .await?;
```

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Clear GPU memory cache
python3 -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"
```

### Problem: GPU Driver Issues

**Symptoms:**
```
Error: GpuError("NVIDIA driver version 470.57.02 is not supported")
```

**Solution:**
```bash
# Check driver version
nvidia-smi

# Update NVIDIA drivers (Ubuntu)
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-525  # Use latest stable

# For manual installation
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.60.11/NVIDIA-Linux-x86_64-525.60.11.run
sudo chmod +x NVIDIA-Linux-x86_64-525.60.11.run
sudo ./NVIDIA-Linux-x86_64-525.60.11.run

# Reboot after installation
sudo reboot
```

### Problem: Multiple GPU Selection Issues

**Symptoms:**
```
Error: GpuError("Failed to select GPU device 1: device not found")
```

**Solution:**
```bash
# List available GPUs
nvidia-smi -L

# Check GPU status
nvidia-smi -q -d MEMORY,UTILIZATION

# Set specific GPU in code
let gpu_config = GpuConfig {
    device_id: 0, // Use first GPU
    enable_multi_gpu: false,
    // ... other config
};

# Or set via environment variable
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0    # Use only GPU 0
```

## ⚡ Performance Problems

### Problem: Slow Inference Speed

**Symptoms:**
- Analysis takes > 5 seconds per sample
- High CPU usage without GPU utilization

**Solution:**
```rust
// Enable all performance optimizations
let detector = LieDetector::builder()
    .with_performance_config(PerformanceConfig {
        enable_simd: true,
        num_threads: num_cpus::get(),
        enable_gpu: true,
        batch_processing: true,
        cache_size_mb: 512,
    })
    .build()
    .await?;
```

```bash
# Check CPU features
lscpu | grep -E "(avx|fma|sse)"

# Compile with optimizations
export RUSTFLAGS="-C target-cpu=native"
cargo build --release

# Enable huge pages for better memory performance
echo 'vm.nr_hugepages=1024' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Problem: High Memory Usage

**Symptoms:**
```
Process killed by OOM killer
Memory usage constantly increasing
```

**Solution:**
```rust
// Configure memory limits
let memory_config = MemoryConfig {
    pool_size_mb: 1024,
    cache_size_mb: 256,
    gc_threshold_mb: 512,
    enable_memory_mapping: true,
    lazy_loading: true,
};

let detector = LieDetector::builder()
    .with_memory_config(memory_config)
    .build()
    .await?;
```

```bash
# Monitor memory usage
htop
watch -n 1 'free -h'

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./target/release/veritas-nexus

# Set memory limits
ulimit -m 4194304  # 4GB limit
```

### Problem: Poor Throughput

**Symptoms:**
- Low requests per second
- High latency under load

**Solution:**
```rust
// Optimize for throughput
let config = ThroughputConfig {
    batch_size: 32,
    max_concurrent_requests: 100,
    connection_pool_size: 50,
    enable_async_processing: true,
    queue_size: 1000,
};

// Use batch processing
let inputs = vec![input1, input2, input3, /* ... */];
let results = detector.analyze_batch(&inputs).await?;
```

```bash
# Tune system for high throughput
echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Increase file descriptor limits
echo 'fs.file-max = 100000' | sudo tee -a /etc/sysctl.conf
echo '* soft nofile 65535' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65535' | sudo tee -a /etc/security/limits.conf
```

## 💾 Memory Issues

### Problem: Memory Leaks

**Symptoms:**
```
Memory usage increases over time
Process eventually killed by OOM
```

**Diagnostic Steps:**
```bash
# Monitor memory over time
while true; do
    echo "$(date): $(ps -o pid,vsz,rss,comm -p $(pgrep veritas-nexus))"
    sleep 60
done

# Use memory profiler
cargo install cargo-profiler
cargo profiler callgrind --bin veritas-nexus

# Check for leaked file descriptors
lsof -p $(pgrep veritas-nexus) | wc -l
watch -n 1 'lsof -p $(pgrep veritas-nexus) | wc -l'
```

**Solution:**
```rust
// Implement proper cleanup
impl Drop for LieDetector {
    fn drop(&mut self) {
        // Cleanup GPU memory
        if let Some(gpu_context) = &mut self.gpu_context {
            gpu_context.cleanup();
        }
        
        // Clear caches
        self.cache.clear();
        
        // Release model resources
        self.models.clear();
    }
}

// Use memory pools to avoid frequent allocation
let pool = MemoryPool::new(1024 * 1024 * 100); // 100MB pool
let buffer = pool.allocate(size);
// buffer automatically returned to pool when dropped
```

### Problem: Stack Overflow

**Symptoms:**
```
thread 'main' has overflowed its stack
Segmentation fault (core dumped)
```

**Solution:**
```bash
# Increase stack size
export RUST_MIN_STACK=8388608  # 8MB stack

# Or set in code
std::thread::Builder::new()
    .stack_size(8 * 1024 * 1024)  // 8MB
    .spawn(|| {
        // Your code here
    })
    .unwrap();
```

```rust
// Avoid deep recursion
// Instead of recursive implementation:
fn recursive_process(data: &[Data]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    process_item(&data[0])?;
    recursive_process(&data[1..])  // Deep recursion!
}

// Use iterative approach:
fn iterative_process(data: &[Data]) -> Result<()> {
    for item in data {
        process_item(item)?;
    }
    Ok(())
}
```

## 🌐 Network and Connectivity

### Problem: Connection Timeouts

**Symptoms:**
```
Error: NetworkError("Connection timed out after 30s")
```

**Solution:**
```rust
// Increase timeout values
let client_config = ClientConfig {
    connection_timeout: Duration::from_secs(60),
    request_timeout: Duration::from_secs(120),
    max_retries: 5,
    retry_delay: Duration::from_secs(2),
};

// Use connection pooling
let client = HttpClient::builder()
    .pool_max_idle_per_host(20)
    .pool_idle_timeout(Duration::from_secs(60))
    .tcp_keepalive(Duration::from_secs(30))
    .build()?;
```

```bash
# Check network connectivity
ping google.com
curl -I https://api.veritas-nexus.example.com/health

# Check DNS resolution
nslookup api.veritas-nexus.example.com
dig api.veritas-nexus.example.com

# Test with increased timeout
curl --connect-timeout 60 --max-time 120 https://api.veritas-nexus.example.com/health
```

### Problem: SSL/TLS Certificate Issues

**Symptoms:**
```
Error: TlsError("certificate verify failed: unable to get local issuer certificate")
```

**Solution:**
```bash
# Update certificate store
sudo apt update
sudo apt install ca-certificates

# For custom certificates
sudo cp your-cert.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates

# Skip verification (development only!)
export RUSTLS_TLS_INSECURE=1  # NOT for production!
```

```rust
// Configure TLS in code
let tls_config = TlsConfig {
    verify_certificates: true,
    ca_cert_file: Some("/path/to/ca-certificates.crt".to_string()),
    client_cert_file: None,
    client_key_file: None,
};

let client = HttpClient::builder()
    .tls_config(tls_config)
    .build()?;
```

### Problem: Rate Limiting

**Symptoms:**
```
Error: RateLimitError("Rate limit exceeded: 429 Too Many Requests")
```

**Solution:**
```rust
// Implement exponential backoff
use tokio::time::{sleep, Duration};

async fn retry_with_backoff<F, T, E>(mut f: F, max_retries: usize) -> Result<T, E>
where
    F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, E>>>>,
{
    let mut delay = Duration::from_millis(100);
    
    for attempt in 0..max_retries {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt == max_retries - 1 => return Err(e),
            Err(_) => {
                sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
        }
    }
    
    unreachable!()
}

// Use request queuing
let rate_limiter = RateLimiter::new(100, Duration::from_secs(60)); // 100 requests per minute
rate_limiter.acquire().await;
let result = api_call().await;
```

## 📁 Input/Output Errors

### Problem: File Not Found

**Symptoms:**
```
Error: IoError("No such file or directory: 'video.mp4'")
```

**Solution:**
```bash
# Check file exists and permissions
ls -la video.mp4
file video.mp4  # Check file type

# Check file permissions
chmod 644 video.mp4

# Verify file path
realpath video.mp4

# Check disk space
df -h
```

```rust
// Validate file before processing
use std::path::Path;

fn validate_input_file(path: &str) -> Result<()> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(format!("File not found: {}", path.display()).into());
    }
    
    if !path.is_file() {
        return Err(format!("Path is not a file: {}", path.display()).into());
    }
    
    let metadata = path.metadata()?;
    if metadata.len() == 0 {
        return Err(format!("File is empty: {}", path.display()).into());
    }
    
    Ok(())
}
```

### Problem: Unsupported File Format

**Symptoms:**
```
Error: FormatError("Unsupported video format: .avi")
```

**Solution:**
```bash
# Check supported formats
veritas-nexus formats --list

# Convert to supported format
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# For audio files
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

```rust
// Add format validation
fn validate_video_format(path: &str) -> Result<()> {
    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    match extension.to_lowercase().as_str() {
        "mp4" | "avi" | "mov" | "mkv" => Ok(()),
        _ => Err(format!("Unsupported video format: .{}", extension).into()),
    }
}
```

### Problem: Corrupted Media Files

**Symptoms:**
```
Error: DecodeError("Failed to decode video frame at timestamp 5.2s")
```

**Solution:**
```bash
# Check file integrity
ffprobe -v error -show_format -show_streams video.mp4

# Repair corrupted video
ffmpeg -i corrupted_video.mp4 -c copy -movflags faststart repaired_video.mp4

# Extract valid portion
ffmpeg -i corrupted_video.mp4 -t 00:05:00 -c copy first_5_minutes.mp4
```

```rust
// Add error recovery
async fn robust_video_analysis(path: &str) -> Result<AnalysisResult> {
    match analyze_video(path).await {
        Ok(result) => Ok(result),
        Err(DecodeError(_)) => {
            // Try with reduced quality
            let repaired_path = repair_video(path).await?;
            analyze_video(&repaired_path).await
        },
        Err(e) => Err(e),
    }
}
```

## 🔍 Analysis and Accuracy Issues

### Problem: Low Confidence Scores

**Symptoms:**
- All results show confidence < 30%
- Inconsistent results for same input

**Solution:**
```rust
// Check model calibration
let calibration_data = load_calibration_dataset()?;
let detector = LieDetector::builder()
    .with_calibration_data(calibration_data)
    .build()
    .await?;

// Enable uncertainty quantification
let config = AnalysisConfig {
    enable_uncertainty_estimation: true,
    monte_carlo_samples: 100,
    confidence_threshold: 0.7,
};
```

```bash
# Check input data quality
veritas-nexus validate --input video.mp4 --verbose

# Run diagnostics
veritas-nexus diagnose --input video.mp4 --output report.json
```

### Problem: Biased Results

**Symptoms:**
- Consistently higher deception scores for certain demographics
- Model performs poorly on non-English speakers

**Solution:**
```rust
// Enable bias monitoring
let bias_monitor = BiasMonitor::new()
    .with_protected_attributes(&["age", "gender", "ethnicity"])
    .with_fairness_metrics(&[FairnessMetric::EqualOpportunity]);

let detector = LieDetector::builder()
    .with_bias_monitor(bias_monitor)
    .build()
    .await?;

// Use ensemble to reduce bias
let ensemble = EnsembleDetector::builder()
    .add_detector("primary", primary_detector)
    .add_detector("bias_aware", bias_aware_detector)
    .add_detector("diverse", diverse_detector)
    .with_voting_strategy(VotingStrategy::WeightedFairness)
    .build()?;
```

### Problem: Unexpected Results

**Symptoms:**
- Truth labeled as deception
- Obviously deceptive statements marked as truthful

**Solution:**
```rust
// Enable detailed logging
let detector = LieDetector::builder()
    .with_logging_config(LoggingConfig {
        level: LogLevel::Debug,
        enable_feature_logging: true,
        enable_decision_trace: true,
    })
    .build()
    .await?;

// Get detailed explanation
let result = detector.analyze_with_explanation(input).await?;
for step in &result.reasoning_trace {
    println!("Step: {}", step.description);
    println!("Evidence: {:?}", step.evidence);
    println!("Confidence: {:.2}", step.confidence);
}
```

## 🚀 Deployment Problems

### Problem: Docker Build Failures

**Symptoms:**
```
ERROR: failed to solve: process "/bin/sh -c cargo build --release" did not complete successfully
```

**Solution:**
```dockerfile
# Use multi-stage build with proper dependencies
FROM rust:1.75-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy source and build
WORKDIR /app
COPY . .
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/veritas-nexus /usr/local/bin/
```

```bash
# Build with more memory
docker build --memory=8g --cpus=4 -t veritas-nexus .

# Use BuildKit for better caching
export DOCKER_BUILDKIT=1
docker build -t veritas-nexus .
```

### Problem: Kubernetes Pod Crashes

**Symptoms:**
```
Pod veritas-nexus-xxx is in CrashLoopBackOff state
OOMKilled
```

**Solution:**
```yaml
# Increase resource limits
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: veritas-nexus
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        # Add health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 10
```

```bash
# Debug pod issues
kubectl describe pod veritas-nexus-xxx
kubectl logs veritas-nexus-xxx --previous

# Check node resources
kubectl top nodes
kubectl describe node worker-node-1
```

### Problem: Load Balancer Issues

**Symptoms:**
```
502 Bad Gateway
Service Unavailable
```

**Solution:**
```bash
# Check service endpoints
kubectl get endpoints veritas-nexus-service

# Test service directly
kubectl port-forward service/veritas-nexus-service 8080:8080
curl http://localhost:8080/health

# Check ingress
kubectl describe ingress veritas-nexus-ingress
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

```yaml
# Fix service configuration
apiVersion: v1
kind: Service
metadata:
  name: veritas-nexus-service
spec:
  selector:
    app: veritas-nexus  # Must match pod labels
  ports:
  - port: 8080
    targetPort: 8080    # Must match container port
    protocol: TCP
```

## 📊 Logging and Debugging

### Enable Debug Logging

```rust
// In your application
use tracing::{info, debug, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn init_logging() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "veritas_nexus=debug".into())
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
}

// Add detailed logging to analysis
async fn analyze_with_logging(input: AnalysisInput) -> Result<AnalysisResult> {
    debug!("Starting analysis with input: {:?}", input);
    
    let start_time = std::time::Instant::now();
    
    // Vision processing
    let vision_result = if let Some(video_path) = &input.video_path {
        info!("Processing video: {}", video_path);
        let result = analyze_video(video_path).await?;
        debug!("Vision analysis result: score={:.3}", result.score);
        Some(result)
    } else {
        None
    };
    
    // Continue with other modalities...
    
    let duration = start_time.elapsed();
    info!("Analysis completed in {:?}", duration);
    
    Ok(final_result)
}
```

```bash
# Set log level
export RUST_LOG=veritas_nexus=debug,tower_http=debug

# Enable specific module logging
export RUST_LOG=veritas_nexus::vision=trace,veritas_nexus::audio=debug

# Log to file
veritas-nexus server --config config.toml 2>&1 | tee veritas.log

# Structured logging with JSON
export RUST_LOG_FORMAT=json
```

### Performance Profiling

```bash
# CPU profiling with flamegraph
cargo install flamegraph
sudo cargo flamegraph --bin veritas-nexus

# Memory profiling
cargo install cargo-instruments
cargo instruments -t "Time Profiler" --bin veritas-nexus

# Heap profiling
export CARGO_PROFILE_RELEASE_DEBUG=true
cargo build --release
valgrind --tool=massif ./target/release/veritas-nexus
```

### Debugging with GDB

```bash
# Build with debug symbols
cargo build --release
strip -g target/release/veritas-nexus  # Remove debug symbols
cargo build  # Debug build with symbols

# Run with GDB
gdb target/debug/veritas-nexus
(gdb) run --config config.toml
(gdb) bt  # Backtrace when crashed
(gdb) info registers
(gdb) x/10i $pc  # Examine instructions
```

## 🖥️ Platform-Specific Issues

### Windows Issues

**Problem: Long path issues**
```
Error: path too long
```

**Solution:**
```cmd
# Enable long paths in Windows 10/11
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

# Use short paths
subst V: C:\very\long\path\to\veritas\nexus
cd V:\
```

**Problem: Missing Visual Studio Build Tools**
```
Error: Microsoft C++ build tools not found
```

**Solution:**
```cmd
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Or install via chocolatey
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
```

### macOS Issues

**Problem: Code signing issues**
```
Error: "veritas-nexus" cannot be opened because the developer cannot be verified
```

**Solution:**
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine veritas-nexus

# Or allow in Security & Privacy settings
# System Preferences > Security & Privacy > General > Allow anyway
```

**Problem: ARM64 compatibility**
```
Error: cannot find library
```

**Solution:**
```bash
# Install Rosetta 2 for x86_64 compatibility
softwareupdate --install-rosetta

# Build native ARM64 version
rustup target add aarch64-apple-darwin
cargo build --target aarch64-apple-darwin --release
```

### Linux Distribution Issues

**Problem: Old glibc version**
```
Error: version `GLIBC_2.28' not found
```

**Solution:**
```bash
# Check glibc version
ldd --version

# Build with older glibc (use older container)
docker run --rm -v "$PWD":/usr/src/myapp -w /usr/src/myapp \
  rust:1.75-slim-bullseye \
  cargo build --release

# Or use musl for static linking
rustup target add x86_64-unknown-linux-musl
cargo build --target x86_64-unknown-linux-musl --release
```

## 🆘 Getting Help

### Collecting Diagnostic Information

```bash
#!/bin/bash
# collect_diagnostics.sh

echo "=== System Information ==="
uname -a
lscpu
free -h
df -h

echo "=== Rust Environment ==="
rustc --version
cargo --version

echo "=== GPU Information ==="
nvidia-smi || echo "NVIDIA GPU not available"
lspci | grep -i vga

echo "=== Veritas Nexus Version ==="
veritas-nexus --version

echo "=== Configuration ==="
cat config.toml

echo "=== Recent Logs ==="
tail -n 100 /var/log/veritas-nexus/app.log

echo "=== Running Processes ==="
ps aux | grep veritas

echo "=== Network Connectivity ==="
ping -c 3 google.com
curl -I https://api.veritas-nexus.example.com/health

echo "=== File Permissions ==="
ls -la /opt/veritas-nexus/models/
```

### Creating Minimal Reproduction

```rust
// minimal_repro.rs
use veritas_nexus::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimal code that reproduces the issue
    let detector = LieDetector::builder()
        .with_text(TextConfig::default())
        .build()
        .await?;
    
    let input = AnalysisInput {
        video_path: None,
        audio_path: None,
        transcript: Some("Test statement".to_string()),
        physiological_data: None,
    };
    
    let result = detector.analyze(input).await?;
    println!("Result: {:?}", result);
    
    Ok(())
}
```

### Reporting Issues

When reporting issues, please include:

1. **Environment information** (run `collect_diagnostics.sh`)
2. **Minimal reproduction case**
3. **Expected vs actual behavior**
4. **Error messages and stack traces**
5. **Configuration files** (remove sensitive information)
6. **Steps to reproduce**

### Support Channels

- **GitHub Issues**: [https://github.com/yourusername/veritas-nexus/issues](https://github.com/yourusername/veritas-nexus/issues)
- **Discussions**: [https://github.com/yourusername/veritas-nexus/discussions](https://github.com/yourusername/veritas-nexus/discussions)
- **Discord**: [Community Server](https://discord.gg/veritas-nexus)
- **Email**: support@veritas-nexus.ai

### Emergency Procedures

For critical production issues:

1. **Immediate actions**:
   - Scale down to minimum replicas
   - Check system resources
   - Review recent changes

2. **Rollback procedure**:
   ```bash
   # Kubernetes rollback
   kubectl rollout undo deployment/veritas-nexus
   
   # Docker rollback
   docker-compose down
   docker-compose up -d --scale veritas-nexus=1
   ```

3. **Contact support**:
   - Use priority support channel
   - Include incident ID
   - Provide diagnostic information

---

This troubleshooting guide covers the most common issues encountered with Veritas Nexus. If you encounter an issue not covered here, please check our [GitHub Issues](https://github.com/yourusername/veritas-nexus/issues) or create a new issue with detailed information.