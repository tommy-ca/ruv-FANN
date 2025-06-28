# Veritas Nexus Performance Guide

This guide provides comprehensive performance optimization strategies, benchmarking methodologies, and tuning recommendations for Veritas Nexus deployments.

## 📋 Table of Contents

1. [Performance Overview](#-performance-overview)
2. [Benchmarking](#-benchmarking)
3. [CPU Optimization](#-cpu-optimization)
4. [GPU Acceleration](#-gpu-acceleration)
5. [Memory Optimization](#-memory-optimization)
6. [I/O and Storage](#-io-and-storage)
7. [Network Optimization](#-network-optimization)
8. [Model Optimization](#-model-optimization)
9. [Streaming Performance](#-streaming-performance)
10. [Monitoring and Profiling](#-monitoring-and-profiling)
11. [Troubleshooting Performance Issues](#-troubleshooting-performance-issues)
12. [Production Tuning](#-production-tuning)

## 📊 Performance Overview

### Performance Targets

| Metric | Target | Measurement Context |
|--------|--------|-------------------|
| **Single Analysis Latency** | < 200ms | P95, CPU-only, text+audio |
| **GPU Analysis Latency** | < 100ms | P95, with GPU acceleration |
| **Throughput** | > 100 RPS | Sustained load, multi-modal |
| **Memory Usage** | < 2GB | Per instance, loaded models |
| **GPU Memory** | < 4GB | Including model weights |
| **Startup Time** | < 30s | Cold start with model loading |
| **Model Load Time** | < 10s | Individual model initialization |

### Performance Characteristics

```rust
// Performance measurement example
use veritas_nexus::performance::*;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let detector = LieDetector::builder()
        .with_performance_monitoring(true)
        .build()
        .await?;
    
    let input = AnalysisInput {
        video_path: Some("test_video.mp4".to_string()),
        audio_path: Some("test_audio.wav".to_string()),
        transcript: Some("Sample text for analysis".to_string()),
        physiological_data: None,
    };
    
    // Warm up the system
    for _ in 0..5 {
        let _ = detector.analyze(input.clone()).await?;
    }
    
    // Measure performance
    let start = Instant::now();
    let result = detector.analyze(input).await?;
    let duration = start.elapsed();
    
    println!("Analysis completed in: {:?}", duration);
    println!("Decision: {:?}", result.decision);
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    
    // Get detailed performance metrics
    let metrics = detector.get_performance_metrics().await;
    println!("Detailed metrics: {:#?}", metrics);
    
    Ok(())
}
```

## 🔍 Benchmarking

### Comprehensive Benchmark Suite

```rust
// benches/comprehensive_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use veritas_nexus::prelude::*;
use tokio::runtime::Runtime;

fn create_test_inputs() -> Vec<AnalysisInput> {
    vec![
        // Text only
        AnalysisInput {
            video_path: None,
            audio_path: None,
            transcript: Some("I was definitely not involved in that incident.".to_string()),
            physiological_data: None,
        },
        // Audio + Text
        AnalysisInput {
            video_path: None,
            audio_path: Some("test_audio.wav".to_string()),
            transcript: Some("I was definitely not involved in that incident.".to_string()),
            physiological_data: None,
        },
        // Full multi-modal
        AnalysisInput {
            video_path: Some("test_video.mp4".to_string()),
            audio_path: Some("test_audio.wav".to_string()),
            transcript: Some("I was definitely not involved in that incident.".to_string()),
            physiological_data: Some(vec![72.0, 73.5, 75.0, 77.2, 79.8]),
        },
    ]
}

fn bench_single_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let detector = rt.block_on(async {
        LieDetector::builder()
            .with_performance_config(PerformanceConfig::benchmark())
            .build()
            .await
            .unwrap()
    });
    
    let inputs = create_test_inputs();
    
    let mut group = c.benchmark_group("single_analysis");
    group.sample_size(100);
    
    for (i, input) in inputs.iter().enumerate() {
        let modalities = match (input.video_path.is_some(), input.audio_path.is_some()) {
            (true, true) => "video_audio_text",
            (false, true) => "audio_text",
            _ => "text_only",
        };
        
        group.bench_with_input(
            BenchmarkId::new("analysis", modalities),
            input,
            |b, input| {
                b.to_async(&rt).iter(|| async {
                    detector.analyze(black_box(input.clone())).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let detector = rt.block_on(async {
        LieDetector::builder()
            .with_batch_config(BatchConfig {
                batch_size: 32,
                enable_parallel: true,
                ..Default::default()
            })
            .build()
            .await
            .unwrap()
    });
    
    let mut group = c.benchmark_group("batch_processing");
    
    for batch_size in [1, 4, 8, 16, 32].iter() {
        let inputs: Vec<_> = (0..*batch_size)
            .map(|i| AnalysisInput {
                video_path: None,
                audio_path: None,
                transcript: Some(format!("Test statement number {}", i)),
                physiological_data: None,
            })
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &inputs,
            |b, inputs| {
                b.to_async(&rt).iter(|| async {
                    detector.analyze_batch(black_box(inputs.clone())).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_streaming_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("streaming_throughput");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    for fps in [10, 15, 30, 60].iter() {
        group.bench_with_input(
            BenchmarkId::new("fps", fps),
            fps,
            |b, fps| {
                b.to_async(&rt).iter_custom(|iters| async move {
                    let pipeline = StreamingPipeline::builder()
                        .with_target_fps(*fps as f32)
                        .build()
                        .unwrap();
                    
                    let start = std::time::Instant::now();
                    
                    for _ in 0..iters {
                        // Simulate frame processing
                        pipeline.process_frame(create_test_frame()).await.unwrap();
                    }
                    
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    
    group.bench_function("detector_creation", |b| {
        b.to_async(&rt).iter(|| async {
            let detector = LieDetector::builder()
                .build()
                .await
                .unwrap();
            
            // Prevent optimization
            black_box(detector);
        });
    });
    
    group.bench_function("model_loading", |b| {
        b.to_async(&rt).iter(|| async {
            let model_manager = ModelManager::new();
            let models = model_manager.load_all_models().await.unwrap();
            black_box(models);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_analysis,
    bench_batch_processing,
    bench_streaming_throughput,
    bench_memory_usage
);

criterion_main!(benches);
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench single_analysis

# Generate detailed reports
cargo bench -- --output-format html

# Profile memory usage
cargo bench --features profiling

# Compare with baseline
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main
```

### Benchmark Results Analysis

```bash
# Generate performance report
cat > analyze_benchmarks.py << 'EOF'
import json
import pandas as pd
import matplotlib.pyplot as plt

def analyze_benchmark_results():
    # Load benchmark data
    with open('target/criterion/single_analysis/base/estimates.json') as f:
        data = json.load(f)
    
    # Analyze latency distribution
    mean_latency = data['mean']['point_estimate'] / 1_000_000  # Convert to ms
    std_dev = data['std_dev']['point_estimate'] / 1_000_000
    
    print(f"Mean latency: {mean_latency:.2f}ms")
    print(f"Standard deviation: {std_dev:.2f}ms")
    print(f"95th percentile: {mean_latency + 1.96 * std_dev:.2f}ms")
    
    # Compare against targets
    if mean_latency < 200:
        print("✅ Latency target met")
    else:
        print("❌ Latency target exceeded")

if __name__ == "__main__":
    analyze_benchmark_results()
EOF

python analyze_benchmarks.py
```

## 🚀 CPU Optimization

### SIMD Vectorization

```rust
// Enable SIMD optimizations
use veritas_nexus::optimization::simd::*;

#[cfg(target_arch = "x86_64")]
pub mod x86_optimizations {
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn optimized_dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let chunks = len / 8;
        
        let mut sum = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        
        // Horizontal sum
        let high = _mm256_extractf128_ps(sum, 1);
        let low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(high, low);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let result = _mm_add_ss(sums, shuf2);
        
        let mut final_sum = _mm_cvtss_f32(result);
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            final_sum += a[i] * b[i];
        }
        
        final_sum
    }
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn optimized_softmax(input: &mut [f32]) {
        let len = input.len();
        let chunks = len / 8;
        
        // Find maximum value
        let mut max_val = f32::NEG_INFINITY;
        for &val in input.iter() {
            max_val = max_val.max(val);
        }
        
        let max_vec = _mm256_set1_ps(max_val);
        
        // Subtract max and compute exp
        for i in 0..chunks {
            let vals = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let shifted = _mm256_sub_ps(vals, max_vec);
            // Note: Would need custom exp implementation or use approx
            _mm256_storeu_ps(input.as_mut_ptr().add(i * 8), shifted);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            input[i] = (input[i] - max_val).exp();
        }
        
        // Normalize
        let sum: f32 = input.iter().sum();
        let sum_vec = _mm256_set1_ps(sum);
        
        for i in 0..chunks {
            let vals = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let normalized = _mm256_div_ps(vals, sum_vec);
            _mm256_storeu_ps(input.as_mut_ptr().add(i * 8), normalized);
        }
        
        for i in (chunks * 8)..len {
            input[i] /= sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod arm_optimizations {
    use std::arch::aarch64::*;
    
    #[target_feature(enable = "neon")]
    pub unsafe fn optimized_dot_product(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        
        let mut sum = vdupq_n_f32(0.0);
        
        for i in 0..chunks {
            let a_vec = vld1q_f32(a.as_ptr().add(i * 4));
            let b_vec = vld1q_f32(b.as_ptr().add(i * 4));
            sum = vfmaq_f32(sum, a_vec, b_vec);
        }
        
        // Horizontal sum
        let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        let final_sum = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(final_sum, 0);
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }
        
        result
    }
}
```

### CPU Configuration

```toml
# Cargo.toml optimization flags
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
debug = false
overflow-checks = false

[profile.release.package."*"]
opt-level = 3

# Enable target CPU features
[target.x86_64-unknown-linux-gnu]
rustflags = [
    "-C", "target-cpu=native",
    "-C", "target-feature=+avx2,+fma",
]

[target.aarch64-unknown-linux-gnu]
rustflags = [
    "-C", "target-cpu=native",
    "-C", "target-feature=+neon",
]
```

### Thread Pool Optimization

```rust
use veritas_nexus::threading::*;

// Configure thread pool for optimal performance
let thread_config = ThreadConfig {
    num_worker_threads: num_cpus::get(),
    stack_size: 2 * 1024 * 1024, // 2MB stack
    thread_affinity: ThreadAffinity::Spread, // Spread across cores
    priority: ThreadPriority::High,
    enable_work_stealing: true,
};

let detector = LieDetector::builder()
    .with_thread_config(thread_config)
    .build()
    .await?;

// Parallel processing configuration
let parallel_config = ParallelConfig {
    chunk_size: 1000, // Items per chunk
    max_parallelism: num_cpus::get(),
    enable_rayon: true,
    enable_async_parallel: true,
};
```

## 🎮 GPU Acceleration

### GPU Configuration

```rust
use veritas_nexus::gpu::*;

// Optimal GPU configuration
let gpu_config = GpuConfig {
    enable_gpu: true,
    device_id: 0,
    memory_limit_mb: 6144, // Leave 2GB for system
    
    // Performance settings
    batch_size: 32, // Optimal for most GPUs
    fp16_inference: true, // Halve memory usage
    tensor_cores: true, // Enable if available
    async_execution: true,
    
    // Memory management
    enable_memory_pool: true,
    pool_size_mb: 4096,
    enable_unified_memory: false, // Usually slower
    
    // Multi-GPU settings
    enable_multi_gpu: true,
    gpu_ids: vec![0, 1], // Use multiple GPUs
    data_parallel: true,
    model_parallel: false, // For very large models
    
    // Optimization flags
    enable_cudnn_autotuning: true,
    workspace_size_mb: 512,
    allow_growth: true, // Allocate memory as needed
};

let detector = LieDetector::builder()
    .with_gpu_config(gpu_config)
    .build()
    .await?;
```

### GPU Memory Optimization

```rust
use veritas_nexus::gpu::memory::*;

// Memory pool management
struct GpuMemoryPool {
    device: Device,
    pools: HashMap<usize, VecDeque<CudaBuffer>>,
    stats: MemoryStats,
}

impl GpuMemoryPool {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            pools: HashMap::new(),
            stats: MemoryStats::default(),
        }
    }
    
    pub fn allocate(&mut self, size: usize) -> Result<CudaBuffer> {
        // Round up to nearest power of 2 for efficient pooling
        let pool_size = size.next_power_of_two();
        
        if let Some(pool) = self.pools.get_mut(&pool_size) {
            if let Some(buffer) = pool.pop_front() {
                self.stats.pool_hits += 1;
                return Ok(buffer);
            }
        }
        
        // Allocate new buffer
        let buffer = CudaBuffer::allocate(&self.device, pool_size)?;
        self.stats.allocations += 1;
        self.stats.total_allocated += pool_size;
        
        Ok(buffer)
    }
    
    pub fn deallocate(&mut self, buffer: CudaBuffer) {
        let size = buffer.size();
        let pool_size = size.next_power_of_two();
        
        self.pools.entry(pool_size)
            .or_insert_with(VecDeque::new)
            .push_back(buffer);
        
        self.stats.deallocations += 1;
    }
    
    pub fn cleanup(&mut self) {
        // Remove unused buffers from pools
        for (_, pool) in self.pools.iter_mut() {
            pool.retain(|buffer| buffer.last_used().elapsed() < Duration::from_secs(60));
        }
    }
}
```

### GPU Profiling

```rust
use veritas_nexus::profiling::gpu::*;

async fn profile_gpu_performance() -> Result<GpuProfile> {
    let profiler = GpuProfiler::new()?;
    
    profiler.start_recording("model_inference")?;
    
    // Your GPU operations here
    let result = model.forward(&input).await?;
    
    profiler.stop_recording("model_inference")?;
    
    let profile = profiler.get_profile()?;
    
    println!("GPU Kernel Times:");
    for kernel in &profile.kernels {
        println!("  {}: {:.2}ms", kernel.name, kernel.duration_ms);
    }
    
    println!("Memory Usage:");
    println!("  Peak: {}MB", profile.peak_memory_mb);
    println!("  Average: {}MB", profile.avg_memory_mb);
    
    Ok(profile)
}
```

## 💾 Memory Optimization

### Memory Pool Architecture

```rust
use veritas_nexus::memory::*;

// Hierarchical memory pools
pub struct MemoryManager {
    small_pool: ObjectPool<SmallBuffer>,  // < 1KB
    medium_pool: ObjectPool<MediumBuffer>, // 1KB - 1MB  
    large_pool: ObjectPool<LargeBuffer>,   // > 1MB
    string_pool: StringPool,
    tensor_pool: TensorPool,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            small_pool: ObjectPool::new(1000, || SmallBuffer::new(1024)),
            medium_pool: ObjectPool::new(100, || MediumBuffer::new(1024 * 1024)),
            large_pool: ObjectPool::new(10, || LargeBuffer::new(10 * 1024 * 1024)),
            string_pool: StringPool::new(10000),
            tensor_pool: TensorPool::new(100),
        }
    }
    
    pub fn allocate_buffer(&self, size: usize) -> Box<dyn Buffer> {
        match size {
            0..=1024 => self.small_pool.get(),
            1025..=1048576 => self.medium_pool.get(),
            _ => self.large_pool.get(),
        }
    }
    
    pub fn allocate_string(&self, capacity: usize) -> PooledString {
        self.string_pool.get_with_capacity(capacity)
    }
    
    pub fn allocate_tensor(&self, shape: &[usize]) -> PooledTensor {
        self.tensor_pool.get_with_shape(shape)
    }
}

// Memory-mapped file support for large models
pub struct MemoryMappedModel {
    mapping: memmap2::Mmap,
    metadata: ModelMetadata,
}

impl MemoryMappedModel {
    pub fn load(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mapping = unsafe { memmap2::Mmap::map(&file)? };
        
        // Parse metadata from header
        let metadata = ModelMetadata::from_bytes(&mapping[0..1024])?;
        
        Ok(Self { mapping, metadata })
    }
    
    pub fn get_weights(&self, layer_id: usize) -> &[f32] {
        let offset = self.metadata.layer_offsets[layer_id];
        let size = self.metadata.layer_sizes[layer_id];
        
        unsafe {
            std::slice::from_raw_parts(
                self.mapping.as_ptr().add(offset) as *const f32,
                size / 4, // f32 is 4 bytes
            )
        }
    }
}
```

### Memory Monitoring

```rust
use veritas_nexus::monitoring::memory::*;

// Real-time memory monitoring
pub struct MemoryMonitor {
    peak_usage: AtomicUsize,
    current_usage: AtomicUsize,
    allocation_count: AtomicU64,
    alerts: Vec<MemoryAlert>,
}

impl MemoryMonitor {
    pub fn record_allocation(&self, size: usize) {
        let new_usage = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        // Update peak usage
        self.peak_usage.fetch_max(new_usage, Ordering::Relaxed);
        
        // Check for alerts
        if new_usage > 2 * 1024 * 1024 * 1024 { // 2GB threshold
            self.trigger_alert(MemoryAlert::HighUsage { current: new_usage });
        }
    }
    
    pub fn record_deallocation(&self, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
        }
    }
}

// Custom allocator for tracking
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

pub struct TrackingAllocator<A: GlobalAlloc> {
    inner: A,
    monitor: MemoryMonitor,
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            self.monitor.record_allocation(layout.size());
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.monitor.record_deallocation(layout.size());
        self.inner.dealloc(ptr, layout);
    }
}
```

## 💿 I/O and Storage

### Asynchronous I/O Optimization

```rust
use veritas_nexus::io::*;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};

// Optimized file loading
pub struct AsyncFileLoader {
    buffer_size: usize,
    max_concurrent_loads: usize,
    semaphore: Semaphore,
}

impl AsyncFileLoader {
    pub fn new() -> Self {
        Self {
            buffer_size: 64 * 1024, // 64KB buffer
            max_concurrent_loads: 10,
            semaphore: Semaphore::new(10),
        }
    }
    
    pub async fn load_video(&self, path: &str) -> Result<VideoData> {
        let _permit = self.semaphore.acquire().await?;
        
        let file = File::open(path).await?;
        let mut reader = BufReader::with_capacity(self.buffer_size, file);
        
        // Use memory mapping for large files
        let metadata = reader.get_ref().metadata().await?;
        if metadata.len() > 100 * 1024 * 1024 { // 100MB threshold
            return self.load_video_mmap(path).await;
        }
        
        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        reader.read_to_end(&mut buffer).await?;
        
        VideoData::from_bytes(buffer)
    }
    
    async fn load_video_mmap(&self, path: &str) -> Result<VideoData> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        
        VideoData::from_mmap(mmap)
    }
    
    pub async fn load_batch(&self, paths: Vec<String>) -> Result<Vec<VideoData>> {
        let futures: Vec<_> = paths.into_iter()
            .map(|path| self.load_video(&path))
            .collect();
        
        // Process in chunks to avoid overwhelming the system
        let chunk_size = self.max_concurrent_loads;
        let mut results = Vec::new();
        
        for chunk in futures.chunks(chunk_size) {
            let chunk_results = futures::future::try_join_all(chunk).await?;
            results.extend(chunk_results);
        }
        
        Ok(results)
    }
}
```

### Storage Optimization

```rust
use veritas_nexus::storage::*;

// Cached storage layer
pub struct CachedStorage {
    cache: Arc<DashMap<String, Arc<CachedItem>>>,
    storage: Box<dyn Storage>,
    cache_size_limit: usize,
    ttl: Duration,
}

impl CachedStorage {
    pub fn new(storage: Box<dyn Storage>) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            storage,
            cache_size_limit: 1024 * 1024 * 1024, // 1GB cache
            ttl: Duration::from_secs(3600), // 1 hour TTL
        }
    }
    
    pub async fn get(&self, key: &str) -> Result<Arc<CachedItem>> {
        // Check cache first
        if let Some(item) = self.cache.get(key) {
            if item.created_at.elapsed() < self.ttl {
                return Ok(item.clone());
            } else {
                // Item expired, remove from cache
                self.cache.remove(key);
            }
        }
        
        // Load from storage
        let data = self.storage.get(key).await?;
        let item = Arc::new(CachedItem {
            data,
            created_at: Instant::now(),
            access_count: AtomicU64::new(1),
        });
        
        // Add to cache with size checking
        self.maybe_evict().await;
        self.cache.insert(key.to_string(), item.clone());
        
        Ok(item)
    }
    
    async fn maybe_evict(&self) {
        let current_size: usize = self.cache.iter()
            .map(|entry| entry.value().data.len())
            .sum();
        
        if current_size > self.cache_size_limit {
            // LRU eviction
            let mut items: Vec<_> = self.cache.iter()
                .map(|entry| (entry.key().clone(), entry.value().created_at))
                .collect();
            
            items.sort_by_key(|(_, created_at)| *created_at);
            
            // Remove oldest 25%
            let remove_count = items.len() / 4;
            for (key, _) in items.into_iter().take(remove_count) {
                self.cache.remove(&key);
            }
        }
    }
}

#[derive(Debug)]
pub struct CachedItem {
    pub data: Vec<u8>,
    pub created_at: Instant,
    pub access_count: AtomicU64,
}
```

## 🌐 Network Optimization

### Connection Pooling

```rust
use veritas_nexus::network::*;

// HTTP client with connection pooling
pub struct OptimizedHttpClient {
    client: reqwest::Client,
    connection_pool_size: usize,
    timeout: Duration,
    retry_config: RetryConfig,
}

impl OptimizedHttpClient {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(20)
            .pool_idle_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true)
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            client,
            connection_pool_size: 20,
            timeout: Duration::from_secs(30),
            retry_config: RetryConfig::default(),
        }
    }
    
    pub async fn upload_with_retry(&self, url: &str, data: Vec<u8>) -> Result<Response> {
        let mut attempts = 0;
        let max_attempts = self.retry_config.max_attempts;
        
        loop {
            attempts += 1;
            
            let request = self.client
                .post(url)
                .body(data.clone())
                .header("Content-Type", "application/octet-stream");
            
            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    return Ok(response);
                }
                Ok(response) if attempts >= max_attempts => {
                    return Err(format!("HTTP error after {} attempts: {}", 
                        attempts, response.status()).into());
                }
                Err(e) if attempts >= max_attempts => {
                    return Err(format!("Network error after {} attempts: {}", 
                        attempts, e).into());
                }
                _ => {
                    // Exponential backoff
                    let delay = Duration::from_millis(
                        self.retry_config.base_delay_ms * (2_u64.pow(attempts - 1))
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}
```

### Batch API Optimization

```rust
use veritas_nexus::api::batch::*;

// Batched API requests
pub struct BatchApiClient {
    client: OptimizedHttpClient,
    batch_size: usize,
    max_wait_time: Duration,
    pending_requests: Arc<Mutex<Vec<PendingRequest>>>,
}

impl BatchApiClient {
    pub async fn analyze_async(&self, input: AnalysisInput) -> Result<AnalysisResult> {
        let (tx, rx) = oneshot::channel();
        
        let request = PendingRequest {
            input,
            response_tx: tx,
            created_at: Instant::now(),
        };
        
        // Add to batch
        {
            let mut pending = self.pending_requests.lock().await;
            pending.push(request);
            
            // Check if we should flush the batch
            if pending.len() >= self.batch_size {
                self.flush_batch(&mut pending).await?;
            }
        }
        
        // Wait for response
        rx.await?
    }
    
    async fn flush_batch(&self, pending: &mut Vec<PendingRequest>) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        
        let batch: Vec<_> = pending.drain(..).collect();
        let inputs: Vec<_> = batch.iter().map(|req| &req.input).collect();
        
        // Send batch request
        let results = self.send_batch_request(&inputs).await?;
        
        // Send responses back to waiting tasks
        for (request, result) in batch.into_iter().zip(results.into_iter()) {
            let _ = request.response_tx.send(result);
        }
        
        Ok(())
    }
    
    async fn send_batch_request(&self, inputs: &[&AnalysisInput]) -> Result<Vec<AnalysisResult>> {
        let payload = BatchRequest { inputs: inputs.to_vec() };
        let json = serde_json::to_vec(&payload)?;
        
        let response = self.client
            .upload_with_retry("/api/batch/analyze", json)
            .await?;
        
        let batch_response: BatchResponse = response.json().await?;
        Ok(batch_response.results)
    }
}
```

## 🤖 Model Optimization

### Model Quantization

```rust
use veritas_nexus::models::quantization::*;

// 8-bit quantization for faster inference
pub struct QuantizedModel {
    weights: Vec<i8>,
    scale: f32,
    zero_point: i8,
    original_model: Box<dyn Model>,
}

impl QuantizedModel {
    pub fn from_model(model: Box<dyn Model>) -> Result<Self> {
        let (weights, scale, zero_point) = quantize_weights(&model)?;
        
        Ok(Self {
            weights,
            scale,
            zero_point,
            original_model: model,
        })
    }
    
    pub async fn inference(&self, input: &Tensor) -> Result<Tensor> {
        // Dequantize on-the-fly for computation
        let float_weights = self.dequantize_weights();
        
        // Perform inference with quantized weights
        self.forward_quantized(input, &float_weights).await
    }
    
    fn dequantize_weights(&self) -> Vec<f32> {
        self.weights.iter()
            .map(|&w| self.scale * (w as f32 - self.zero_point as f32))
            .collect()
    }
}

fn quantize_weights(model: &dyn Model) -> Result<(Vec<i8>, f32, i8)> {
    let weights = model.get_weights();
    
    // Calculate quantization parameters
    let min_val = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let scale = (max_val - min_val) / 255.0;
    let zero_point = (-min_val / scale).round() as i8;
    
    // Quantize weights
    let quantized: Vec<i8> = weights.iter()
        .map(|&w| ((w / scale).round() as i32 + zero_point as i32).clamp(-128, 127) as i8)
        .collect();
    
    Ok((quantized, scale, zero_point))
}
```

### Model Pruning

```rust
use veritas_nexus::models::pruning::*;

// Structured pruning for smaller models
pub struct PrunedModel {
    original_model: Box<dyn Model>,
    pruning_mask: Vec<bool>,
    compression_ratio: f32,
}

impl PrunedModel {
    pub fn prune_by_magnitude(model: Box<dyn Model>, sparsity: f32) -> Result<Self> {
        let weights = model.get_weights();
        let threshold = calculate_magnitude_threshold(&weights, sparsity);
        
        let pruning_mask: Vec<bool> = weights.iter()
            .map(|&w| w.abs() > threshold)
            .collect();
        
        let compression_ratio = pruning_mask.iter()
            .filter(|&&keep| keep)
            .count() as f32 / weights.len() as f32;
        
        Ok(Self {
            original_model: model,
            pruning_mask,
            compression_ratio,
        })
    }
    
    pub async fn inference(&self, input: &Tensor) -> Result<Tensor> {
        // Apply pruning mask during inference
        let weights = self.original_model.get_weights();
        let pruned_weights: Vec<f32> = weights.iter()
            .zip(self.pruning_mask.iter())
            .map(|(&w, &keep)| if keep { w } else { 0.0 })
            .collect();
        
        self.forward_with_weights(input, &pruned_weights).await
    }
}

fn calculate_magnitude_threshold(weights: &[f32], sparsity: f32) -> f32 {
    let mut sorted_weights: Vec<f32> = weights.iter()
        .map(|&w| w.abs())
        .collect();
    sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let threshold_index = (weights.len() as f32 * sparsity) as usize;
    sorted_weights[threshold_index.min(sorted_weights.len() - 1)]
}
```

### Model Distillation

```rust
use veritas_nexus::models::distillation::*;

// Knowledge distillation for smaller student models
pub struct DistilledModel {
    student_model: Box<dyn Model>,
    teacher_model: Box<dyn Model>,
    temperature: f32,
    alpha: f32, // Weight between hard and soft targets
}

impl DistilledModel {
    pub async fn train_student(
        teacher: Box<dyn Model>,
        student_architecture: ModelArchitecture,
        training_data: &[TrainingExample],
    ) -> Result<Self> {
        let mut student = create_model(student_architecture)?;
        let temperature = 4.0;
        let alpha = 0.7;
        
        for epoch in 0..100 {
            for batch in training_data.chunks(32) {
                // Get teacher predictions (soft targets)
                let teacher_logits = teacher.forward_batch(batch).await?;
                let soft_targets = softmax_with_temperature(&teacher_logits, temperature);
                
                // Get student predictions
                let student_logits = student.forward_batch(batch).await?;
                
                // Calculate distillation loss
                let distillation_loss = kl_divergence(&student_logits, &soft_targets, temperature);
                let hard_loss = cross_entropy(&student_logits, &batch.labels);
                
                let total_loss = alpha * distillation_loss + (1.0 - alpha) * hard_loss;
                
                // Backpropagation
                student.backward(total_loss).await?;
            }
            
            if epoch % 10 == 0 {
                let accuracy = evaluate_model(&student, &validation_data).await?;
                println!("Epoch {}: Accuracy = {:.2}%", epoch, accuracy * 100.0);
            }
        }
        
        Ok(Self {
            student_model: student,
            teacher_model: teacher,
            temperature,
            alpha,
        })
    }
}

fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_logits: Vec<f32> = logits.iter()
        .map(|&x| x / temperature)
        .collect();
    
    softmax(&scaled_logits)
}

fn kl_divergence(predictions: &[f32], targets: &[f32], temperature: f32) -> f32 {
    let pred_soft = softmax_with_temperature(predictions, temperature);
    
    pred_soft.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| t * (t.ln() - p.ln()))
        .sum::<f32>() * temperature * temperature
}
```

## 🌊 Streaming Performance

### Low-Latency Streaming

```rust
use veritas_nexus::streaming::optimized::*;

pub struct LowLatencyStreaming {
    frame_buffer: RingBuffer<VideoFrame>,
    audio_buffer: RingBuffer<AudioChunk>,
    processing_pipeline: Pipeline,
    latency_target: Duration,
}

impl LowLatencyStreaming {
    pub fn new(latency_target: Duration) -> Self {
        let buffer_size = calculate_buffer_size(latency_target);
        
        Self {
            frame_buffer: RingBuffer::new(buffer_size),
            audio_buffer: RingBuffer::new(buffer_size * 2), // Audio needs more samples
            processing_pipeline: Pipeline::new_optimized(),
            latency_target,
        }
    }
    
    pub async fn process_frame(&mut self, frame: VideoFrame) -> Result<Option<AnalysisResult>> {
        // Add frame to buffer
        self.frame_buffer.push(frame);
        
        // Check if we have enough data for analysis
        if self.has_sufficient_data() {
            let batch = self.extract_analysis_batch()?;
            
            // Process with timeout to meet latency target
            let result = timeout(
                self.latency_target,
                self.processing_pipeline.process(batch)
            ).await;
            
            match result {
                Ok(Ok(analysis)) => Ok(Some(analysis)),
                Ok(Err(e)) => Err(e),
                Err(_) => {
                    // Timeout - return partial result or skip
                    println!("Analysis timeout, skipping frame");
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }
    
    fn has_sufficient_data(&self) -> bool {
        self.frame_buffer.len() >= 3 && // At least 3 frames
        self.audio_buffer.len() >= 10    // At least 1 second of audio at 10 chunks/sec
    }
    
    fn extract_analysis_batch(&mut self) -> Result<AnalysisBatch> {
        let frames = self.frame_buffer.drain_recent(3);
        let audio = self.audio_buffer.drain_recent(10);
        
        Ok(AnalysisBatch { frames, audio, text: None })
    }
}

fn calculate_buffer_size(latency_target: Duration) -> usize {
    // Buffer size based on latency target and processing requirements
    let target_ms = latency_target.as_millis() as usize;
    let fps = 30; // Target FPS
    
    (target_ms * fps / 1000).max(5) // Minimum 5 frames
}
```

### Adaptive Quality Control

```rust
use veritas_nexus::streaming::adaptive::*;

pub struct AdaptiveQualityController {
    current_quality: QualityLevel,
    latency_history: VecDeque<Duration>,
    cpu_usage_history: VecDeque<f32>,
    target_latency: Duration,
    adjustment_interval: Duration,
    last_adjustment: Instant,
}

impl AdaptiveQualityController {
    pub fn new(target_latency: Duration) -> Self {
        Self {
            current_quality: QualityLevel::Medium,
            latency_history: VecDeque::with_capacity(60), // 60 samples
            cpu_usage_history: VecDeque::with_capacity(60),
            target_latency,
            adjustment_interval: Duration::from_secs(5),
            last_adjustment: Instant::now(),
        }
    }
    
    pub fn record_performance(&mut self, latency: Duration, cpu_usage: f32) {
        self.latency_history.push_back(latency);
        self.cpu_usage_history.push_back(cpu_usage);
        
        // Keep only recent history
        if self.latency_history.len() > 60 {
            self.latency_history.pop_front();
        }
        if self.cpu_usage_history.len() > 60 {
            self.cpu_usage_history.pop_front();
        }
        
        // Check if adjustment is needed
        if self.last_adjustment.elapsed() >= self.adjustment_interval {
            self.maybe_adjust_quality();
            self.last_adjustment = Instant::now();
        }
    }
    
    fn maybe_adjust_quality(&mut self) {
        if self.latency_history.len() < 10 {
            return; // Need more data
        }
        
        let avg_latency = self.latency_history.iter().sum::<Duration>() / self.latency_history.len() as u32;
        let avg_cpu = self.cpu_usage_history.iter().sum::<f32>() / self.cpu_usage_history.len() as f32;
        
        let latency_ratio = avg_latency.as_millis() as f32 / self.target_latency.as_millis() as f32;
        
        if latency_ratio > 1.2 || avg_cpu > 0.8 {
            // System overloaded, decrease quality
            self.current_quality = match self.current_quality {
                QualityLevel::High => QualityLevel::Medium,
                QualityLevel::Medium => QualityLevel::Low,
                QualityLevel::Low => QualityLevel::Low,
            };
            println!("Decreased quality to {:?}", self.current_quality);
        } else if latency_ratio < 0.8 && avg_cpu < 0.6 {
            // System underutilized, can increase quality
            self.current_quality = match self.current_quality {
                QualityLevel::Low => QualityLevel::Medium,
                QualityLevel::Medium => QualityLevel::High,
                QualityLevel::High => QualityLevel::High,
            };
            println!("Increased quality to {:?}", self.current_quality);
        }
    }
    
    pub fn get_current_quality(&self) -> QualityLevel {
        self.current_quality
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QualityLevel {
    Low,    // Reduced resolution, simpler models
    Medium, // Standard settings
    High,   // Full resolution, best models
}

impl QualityLevel {
    pub fn get_config(&self) -> QualityConfig {
        match self {
            QualityLevel::Low => QualityConfig {
                video_resolution: (320, 240),
                model_precision: ModelPrecision::Fast,
                batch_size: 1,
                enable_micro_expressions: false,
            },
            QualityLevel::Medium => QualityConfig {
                video_resolution: (640, 480),
                model_precision: ModelPrecision::Balanced,
                batch_size: 4,
                enable_micro_expressions: true,
            },
            QualityLevel::High => QualityConfig {
                video_resolution: (1280, 720),
                model_precision: ModelPrecision::Accurate,
                batch_size: 8,
                enable_micro_expressions: true,
            },
        }
    }
}
```

## 📊 Monitoring and Profiling

### Performance Metrics Collection

```rust
use veritas_nexus::metrics::*;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub throughput_rps: f64,
    pub memory_usage_mb: usize,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub error_rate: f32,
    pub cache_hit_rate: f32,
}

pub struct MetricsCollector {
    latency_histogram: Histogram,
    throughput_counter: Counter,
    memory_gauge: Gauge,
    cpu_gauge: Gauge,
    gpu_gauge: Gauge,
    error_counter: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            latency_histogram: Histogram::new(),
            throughput_counter: Counter::new(),
            memory_gauge: Gauge::new(),
            cpu_gauge: Gauge::new(),
            gpu_gauge: Gauge::new(),
            error_counter: Counter::new(),
            cache_hits: Counter::new(),
            cache_misses: Counter::new(),
        }
    }
    
    pub fn record_latency(&self, duration: Duration) {
        self.latency_histogram.record(duration.as_millis() as u64);
    }
    
    pub fn record_request(&self) {
        self.throughput_counter.increment();
    }
    
    pub fn record_error(&self) {
        self.error_counter.increment();
    }
    
    pub fn record_cache_hit(&self, hit: bool) {
        if hit {
            self.cache_hits.increment();
        } else {
            self.cache_misses.increment();
        }
    }
    
    pub fn get_current_metrics(&self) -> PerformanceMetrics {
        let total_cache = self.cache_hits.value() + self.cache_misses.value();
        let cache_hit_rate = if total_cache > 0 {
            self.cache_hits.value() as f32 / total_cache as f32
        } else {
            0.0
        };
        
        PerformanceMetrics {
            latency_p50: Duration::from_millis(self.latency_histogram.percentile(50.0)),
            latency_p95: Duration::from_millis(self.latency_histogram.percentile(95.0)),
            latency_p99: Duration::from_millis(self.latency_histogram.percentile(99.0)),
            throughput_rps: self.throughput_counter.rate_per_second(),
            memory_usage_mb: get_memory_usage_mb(),
            cpu_usage_percent: get_cpu_usage_percent(),
            gpu_usage_percent: get_gpu_usage_percent(),
            error_rate: self.error_counter.rate_per_second() / self.throughput_counter.rate_per_second(),
            cache_hit_rate,
        }
    }
}
```

### Profiling Integration

```rust
use veritas_nexus::profiling::*;

// CPU profiling with flame graphs
pub struct CpuProfiler {
    guard: Option<pprof::ProfilerGuard<'static>>,
}

impl CpuProfiler {
    pub fn start() -> Result<Self> {
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000) // 1000Hz sampling
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()?;
        
        Ok(Self {
            guard: Some(guard),
        })
    }
    
    pub fn stop_and_report(&mut self, output_path: &str) -> Result<()> {
        if let Some(guard) = self.guard.take() {
            let report = guard.report().build()?;
            
            // Generate flame graph
            let file = std::fs::File::create(format!("{}.svg", output_path))?;
            report.flamegraph(file)?;
            
            // Generate pprof profile
            let mut file = std::fs::File::create(format!("{}.pb", output_path))?;
            let profile = report.pprof()?;
            
            let mut content = Vec::new();
            profile.encode(&mut content)?;
            std::io::Write::write_all(&mut file, &content)?;
            
            println!("Profile saved to {}.svg and {}.pb", output_path, output_path);
        }
        
        Ok(())
    }
}

// Memory profiling
pub struct MemoryProfiler {
    peak_usage: usize,
    allocation_count: u64,
    hotspots: HashMap<String, AllocationInfo>,
}

impl MemoryProfiler {
    pub fn profile_function<F, R>(&mut self, name: &str, f: F) -> R 
    where F: FnOnce() -> R {
        let start_memory = get_memory_usage();
        let start_time = Instant::now();
        
        let result = f();
        
        let end_memory = get_memory_usage();
        let duration = start_time.elapsed();
        
        let allocation_info = AllocationInfo {
            memory_delta: end_memory.saturating_sub(start_memory),
            duration,
            call_count: 1,
        };
        
        self.hotspots.entry(name.to_string())
            .and_modify(|info| {
                info.memory_delta += allocation_info.memory_delta;
                info.duration += allocation_info.duration;
                info.call_count += 1;
            })
            .or_insert(allocation_info);
        
        result
    }
    
    pub fn generate_report(&self) -> MemoryReport {
        let mut hotspots: Vec<_> = self.hotspots.iter().collect();
        hotspots.sort_by_key(|(_, info)| std::cmp::Reverse(info.memory_delta));
        
        MemoryReport {
            peak_usage: self.peak_usage,
            total_allocations: self.allocation_count,
            top_allocators: hotspots.into_iter()
                .take(10)
                .map(|(name, info)| (name.clone(), info.clone()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub memory_delta: usize,
    pub duration: Duration,
    pub call_count: u64,
}

#[derive(Debug)]
pub struct MemoryReport {
    pub peak_usage: usize,
    pub total_allocations: u64,
    pub top_allocators: Vec<(String, AllocationInfo)>,
}
```

## 🚨 Troubleshooting Performance Issues

### Common Performance Problems

#### High Latency Issues

```bash
# Check system resources
htop
nvidia-smi  # For GPU usage
iostat -x 1  # For I/O usage

# Check network latency
ping target-server
traceroute target-server

# Profile application
cargo flamegraph --bin veritas-nexus
```

#### Memory Leaks

```rust
// Memory leak detection
use veritas_nexus::debugging::*;

async fn detect_memory_leaks() {
    let mut tracker = MemoryTracker::new();
    
    for i in 0..1000 {
        tracker.record_allocation(format!("iteration_{}", i), 1024);
        
        // Simulate some work
        let detector = LieDetector::new().await.unwrap();
        let result = detector.analyze(test_input()).await.unwrap();
        
        tracker.record_deallocation(format!("iteration_{}", i));
        
        if i % 100 == 0 {
            let report = tracker.generate_report();
            if report.potential_leaks.len() > 0 {
                println!("Potential memory leaks detected:");
                for leak in &report.potential_leaks {
                    println!("  {}: {} bytes", leak.location, leak.size);
                }
            }
        }
    }
}
```

#### CPU Bottlenecks

```rust
// CPU bottleneck identification
use veritas_nexus::profiling::cpu::*;

pub fn profile_cpu_hotspots() -> Result<CpuProfile> {
    let mut profiler = CpuProfiler::new();
    
    profiler.start_profiling()?;
    
    // Run your workload here
    let detector = LieDetector::new().await?;
    for _ in 0..100 {
        detector.analyze(test_input()).await?;
    }
    
    let profile = profiler.stop_profiling()?;
    
    // Analyze hotspots
    for hotspot in &profile.hotspots {
        if hotspot.cpu_time_percent > 10.0 {
            println!("CPU hotspot: {} ({}%)", 
                hotspot.function_name, 
                hotspot.cpu_time_percent);
        }
    }
    
    Ok(profile)
}
```

### Performance Optimization Checklist

```checklist
- [ ] Enable all relevant CPU features (AVX2, FMA, etc.)
- [ ] Use appropriate batch sizes for your workload
- [ ] Enable GPU acceleration if available
- [ ] Configure memory pools appropriately
- [ ] Use memory mapping for large files
- [ ] Enable result caching where beneficial
- [ ] Monitor and tune garbage collection
- [ ] Profile regularly to identify bottlenecks
- [ ] Use connection pooling for network requests
- [ ] Implement proper error handling and retries
- [ ] Configure appropriate timeouts
- [ ] Monitor system resources continuously
```

## 🔧 Production Tuning

### Environment-Specific Configurations

```rust
// Production configuration template
pub fn create_production_config() -> ProductionConfig {
    ProductionConfig {
        // CPU settings
        cpu_threads: num_cpus::get(),
        enable_simd: true,
        cpu_affinity: CpuAffinity::Spread,
        
        // Memory settings
        memory_pool_size_mb: 2048,
        cache_size_mb: 512,
        enable_memory_mapping: true,
        gc_strategy: GcStrategy::Incremental,
        
        // GPU settings
        enable_gpu: true,
        gpu_memory_limit_mb: 6144,
        gpu_batch_size: 32,
        enable_fp16: true,
        
        // Network settings
        connection_pool_size: 100,
        request_timeout_seconds: 30,
        max_retries: 3,
        
        // Monitoring
        enable_metrics: true,
        metrics_interval_seconds: 10,
        enable_profiling: false, // Disable in production
        
        // Security
        enable_tls: true,
        require_authentication: true,
        rate_limit_per_minute: 1000,
    }
}

// Development configuration
pub fn create_development_config() -> DevelopmentConfig {
    DevelopmentConfig {
        // More relaxed settings for development
        cpu_threads: 4,
        memory_pool_size_mb: 512,
        cache_size_mb: 128,
        enable_gpu: false, // May not be available
        enable_profiling: true,
        enable_debug_logging: true,
        hot_reload: true,
    }
}
```

### Load Testing

```bash
#!/bin/bash
# load_test.sh

ENDPOINT="http://localhost:8080/api/analyze"
CONCURRENT_USERS=50
DURATION=300  # 5 minutes
RAMP_UP=60    # 1 minute ramp-up

# Install required tools
pip install locust

# Create load test script
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between
import json
import random

class VeritasUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Authentication if required
        self.api_key = "your-api-key"
    
    @task(3)
    def analyze_text(self):
        payload = {
            "video_path": None,
            "audio_path": None,
            "transcript": f"Test statement {random.randint(1, 1000)}",
            "physiological_data": None
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        with self.client.post("/api/analyze", 
                             json=payload, 
                             headers=headers,
                             catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "decision" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def analyze_multimodal(self):
        payload = {
            "video_path": "sample_video.mp4",
            "audio_path": "sample_audio.wav",
            "transcript": f"Complex statement {random.randint(1, 100)}",
            "physiological_data": [72.0, 73.5, 75.0, 77.2]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.client.post("/api/analyze", json=payload, headers=headers)
EOF

# Run load test
locust -f locustfile.py --host=$ENDPOINT \
    --users=$CONCURRENT_USERS \
    --spawn-rate=10 \
    --run-time=${DURATION}s \
    --html=load_test_report.html
```

### Continuous Performance Monitoring

```yaml
# prometheus-rules.yml
groups:
  - name: veritas-nexus-performance
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(veritas_request_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: LowThroughput
        expr: rate(veritas_requests_total[5m]) < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low request throughput"
          description: "Request rate is {{ $value }} requests/second"
      
      - alert: HighErrorRate
        expr: rate(veritas_requests_total{status!~"2.."}[5m]) / rate(veritas_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: HighMemoryUsage
        expr: veritas_memory_usage_bytes / veritas_memory_limit_bytes > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
      
      - alert: GPUMemoryExhaustion
        expr: veritas_gpu_memory_usage_bytes / veritas_gpu_memory_total_bytes > 0.95
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory nearly exhausted"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
```

---

This performance guide provides comprehensive strategies for optimizing Veritas Nexus across all system components. Regular benchmarking and monitoring are essential for maintaining optimal performance in production environments. For specific performance issues, consult the [Troubleshooting Guide](TROUBLESHOOTING.md).