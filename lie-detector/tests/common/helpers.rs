/// Test helper functions and utilities
/// 
/// This module provides common helper functions for setting up tests,
/// assertions, performance measurements, and test data validation

use num_traits::Float;
use std::time::{Duration, Instant};
use std::path::Path;
use super::{TestConfig, FLOAT_TOLERANCE};

/// Performance measurement helper
pub struct PerformanceMeasurement {
    pub duration: Duration,
    pub memory_usage: Option<usize>,
    pub cpu_usage: Option<f64>,
}

/// Measure execution time and optionally memory usage
pub fn measure_performance<F, R>(f: F) -> (R, PerformanceMeasurement)
where
    F: FnOnce() -> R,
{
    let start_time = Instant::now();
    let start_memory = get_memory_usage();
    
    let result = f();
    
    let duration = start_time.elapsed();
    let end_memory = get_memory_usage();
    
    let measurement = PerformanceMeasurement {
        duration,
        memory_usage: if let (Some(start), Some(end)) = (start_memory, end_memory) {
            Some(end.saturating_sub(start))
        } else {
            None
        },
        cpu_usage: None, // Could be extended to measure CPU usage
    };
    
    (result, measurement)
}

/// Async version of performance measurement
pub async fn measure_performance_async<F, Fut, R>(f: F) -> (R, PerformanceMeasurement)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let start_time = Instant::now();
    let start_memory = get_memory_usage();
    
    let result = f().await;
    
    let duration = start_time.elapsed();
    let end_memory = get_memory_usage();
    
    let measurement = PerformanceMeasurement {
        duration,
        memory_usage: if let (Some(start), Some(end)) = (start_memory, end_memory) {
            Some(end.saturating_sub(start))
        } else {
            None
        },
        cpu_usage: None,
    };
    
    (result, measurement)
}

/// Get current memory usage (platform-specific implementation)
fn get_memory_usage() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return Some(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    None // Fallback for other platforms or if reading fails
}

/// Assert that execution time is within acceptable bounds
pub fn assert_performance_bounds(
    measurement: &PerformanceMeasurement,
    max_duration: Duration,
    max_memory: Option<usize>,
) {
    assert!(
        measurement.duration <= max_duration,
        "Execution took too long: {:?} > {:?}",
        measurement.duration,
        max_duration
    );
    
    if let (Some(used), Some(max)) = (measurement.memory_usage, max_memory) {
        assert!(
            used <= max,
            "Memory usage too high: {} bytes > {} bytes",
            used,
            max
        );
    }
}

/// Batch assertion for multiple floating point values
pub fn assert_float_vec_eq<T: Float + std::fmt::Debug>(
    actual: &[T],
    expected: &[T],
    tolerance: T,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Vector lengths don't match: {} vs {}",
        actual.len(),
        expected.len()
    );
    
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = if *a > *e { *a - *e } else { *e - *a };
        assert!(
            diff < tolerance,
            "Vectors differ at index {}: {:?} != {:?} (diff: {:?}, tolerance: {:?})",
            i, a, e, diff, tolerance
        );
    }
}

/// Assert that a probability distribution is valid
pub fn assert_valid_distribution<T: Float + std::fmt::Debug>(probabilities: &[T]) {
    // Each probability should be in [0, 1]
    for (i, &prob) in probabilities.iter().enumerate() {
        assert!(
            prob >= T::zero() && prob <= T::one(),
            "Invalid probability at index {}: {:?}",
            i, prob
        );
    }
    
    // Sum should be approximately 1.0 (if it's meant to be a distribution)
    let sum = probabilities.iter().fold(T::zero(), |acc, &x| acc + x);
    let tolerance = T::from(FLOAT_TOLERANCE).unwrap_or_else(|| T::from(1e-6).unwrap());
    let one = T::one();
    let diff = if sum > one { sum - one } else { one - sum };
    
    assert!(
        diff < tolerance || probabilities.len() == 1, // Single probability doesn't need to sum to 1
        "Probabilities don't sum to 1.0: sum = {:?}",
        sum
    );
}

/// Create temporary directory for test files
pub fn create_temp_dir(test_name: &str) -> Result<std::path::PathBuf, std::io::Error> {
    let temp_dir = std::env::temp_dir().join(format!("veritas-nexus-test-{}", test_name));
    std::fs::create_dir_all(&temp_dir)?;
    Ok(temp_dir)
}

/// Clean up temporary test files
pub fn cleanup_temp_dir(dir: &Path) -> Result<(), std::io::Error> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}

/// Set up test environment with logging and directories
pub fn setup_test_environment(test_name: &str) -> Result<TestConfig, Box<dyn std::error::Error>> {
    let mut config = TestConfig::default();
    
    // Create test-specific temporary directory
    let test_dir = create_temp_dir(test_name)?;
    config.temp_dir = test_dir.clone();
    config.data_dir = test_dir.join("data");
    config.model_dir = test_dir.join("models");
    
    // Initialize the configuration
    config.setup()?;
    
    Ok(config)
}

/// Teardown test environment
pub fn teardown_test_environment(config: &TestConfig) -> Result<(), std::io::Error> {
    cleanup_temp_dir(&config.temp_dir)
}

/// Generate test data file with specified content
pub fn create_test_file(
    path: &Path,
    content: &[u8],
) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)
}

/// Load binary test data from file
pub fn load_test_data(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    std::fs::read(path)
}

/// Create test image data in various formats
pub fn create_test_image_data(width: usize, height: usize, channels: usize) -> Vec<u8> {
    let size = width * height * channels;
    let mut data = Vec::with_capacity(size);
    
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                // Create a simple pattern
                let value = ((x + y + c) % 256) as u8;
                data.push(value);
            }
        }
    }
    
    data
}

/// Create test audio data (sine wave)
pub fn create_test_audio_data(
    sample_rate: u32,
    duration_ms: u32,
    frequency: f32,
) -> Vec<f32> {
    let samples = (sample_rate * duration_ms / 1000) as usize;
    let mut data = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        data.push(sample);
    }
    
    data
}

/// Validate neural network output format
pub fn validate_network_output<T: Float + std::fmt::Debug>(
    output: &[T],
    expected_size: usize,
) -> Result<(), String> {
    if output.len() != expected_size {
        return Err(format!(
            "Output size mismatch: expected {}, got {}",
            expected_size,
            output.len()
        ));
    }
    
    // Check for NaN or infinite values
    for (i, &value) in output.iter().enumerate() {
        if value.is_nan() {
            return Err(format!("NaN value at index {}", i));
        }
        if value.is_infinite() {
            return Err(format!("Infinite value at index {}", i));
        }
    }
    
    Ok(())
}

/// Statistical analysis helper for test results
pub struct StatisticalAnalysis<T: Float> {
    pub mean: T,
    pub std_dev: T,
    pub min: T,
    pub max: T,
    pub median: T,
}

impl<T: Float + std::fmt::Debug + Clone> StatisticalAnalysis<T> {
    pub fn new(data: &[T]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let len = T::from(data.len()).unwrap();
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / len;
        
        let variance = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / len;
        
        let std_dev = variance.sqrt();
        
        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        
        let median = if sorted_data.len() % 2 == 0 {
            let mid = sorted_data.len() / 2;
            (sorted_data[mid - 1] + sorted_data[mid]) / T::from(2).unwrap()
        } else {
            sorted_data[sorted_data.len() / 2]
        };
        
        Some(Self {
            mean,
            std_dev,
            min,
            max,
            median,
        })
    }
}

/// Assert statistical properties of data
pub fn assert_statistical_bounds<T: Float + std::fmt::Debug + Clone>(
    data: &[T],
    expected_mean: T,
    max_std_dev: T,
    tolerance: T,
) {
    let analysis = StatisticalAnalysis::new(data)
        .expect("Cannot analyze empty data");
    
    // Check mean
    let mean_diff = if analysis.mean > expected_mean {
        analysis.mean - expected_mean
    } else {
        expected_mean - analysis.mean
    };
    
    assert!(
        mean_diff < tolerance,
        "Mean outside tolerance: {:?} vs {:?} (diff: {:?}, tolerance: {:?})",
        analysis.mean, expected_mean, mean_diff, tolerance
    );
    
    // Check standard deviation
    assert!(
        analysis.std_dev <= max_std_dev,
        "Standard deviation too high: {:?} > {:?}",
        analysis.std_dev, max_std_dev
    );
}

/// Concurrent test execution helper
pub async fn run_concurrent_tests<F, Fut, R>(
    test_count: usize,
    test_fn: F,
) -> Vec<Result<R, Box<dyn std::error::Error + Send + Sync>>>
where
    F: Fn(usize) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<R, Box<dyn std::error::Error + Send + Sync>>> + Send,
    R: Send + 'static,
{
    use tokio::task;
    
    let mut handles = Vec::new();
    
    for i in 0..test_count {
        let handle = task::spawn(test_fn(i));
        handles.push(handle);
    }
    
    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => results.push(Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>)),
        }
    }
    
    results
}

/// Macro for creating test cases with automatic setup/teardown
#[macro_export]
macro_rules! test_case {
    ($test_name:ident, $test_fn:expr) => {
        #[tokio::test]
        async fn $test_name() {
            let config = setup_test_environment(stringify!($test_name))
                .expect("Failed to setup test environment");
            
            let result = $test_fn(&config).await;
            
            teardown_test_environment(&config)
                .expect("Failed to teardown test environment");
            
            result.expect("Test failed");
        }
    };
}

/// Macro for creating benchmarked test cases
#[macro_export]
macro_rules! benchmark_test {
    ($test_name:ident, $test_fn:expr, $max_duration:expr) => {
        #[tokio::test]
        async fn $test_name() {
            let config = setup_test_environment(stringify!($test_name))
                .expect("Failed to setup test environment");
            
            let (result, measurement) = measure_performance_async(|| $test_fn(&config)).await;
            
            assert_performance_bounds(&measurement, $max_duration, None);
            
            teardown_test_environment(&config)
                .expect("Failed to teardown test environment");
            
            result.expect("Benchmark test failed");
        }
    };
}