/// Common test utilities and fixtures for veritas-nexus
/// 
/// This module provides shared functionality for all test types including:
/// - Mock data generators
/// - Test fixtures and sample data
/// - Common assertions and helpers
/// - Test configuration utilities

pub mod fixtures;
pub mod generators;
pub mod generators_enhanced;
pub mod helpers;
pub mod mocks;

use num_traits::Float;
use std::path::PathBuf;

/// Test configuration for consistent test setup
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub enable_gpu: bool,
    pub enable_parallel: bool,
    pub log_level: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("tests/data"),
            model_dir: PathBuf::from("tests/models"),
            temp_dir: PathBuf::from("target/tmp"),
            enable_gpu: false,
            enable_parallel: true,
            log_level: "debug".to_string(),
        }
    }
}

impl TestConfig {
    /// Initialize test environment with this configuration
    pub fn setup(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Create directories if they don't exist
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.model_dir)?;
        std::fs::create_dir_all(&self.temp_dir)?;
        
        // Initialize logging for tests
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", &self.log_level);
        }
        
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .try_init();
        
        Ok(())
    }
}

/// Standard tolerance for floating point comparisons in tests
pub const FLOAT_TOLERANCE: f64 = 1e-6;
pub const FLOAT_TOLERANCE_F32: f32 = 1e-6;

/// Assert floating point values are approximately equal
pub fn assert_float_eq<T: Float + std::fmt::Debug>(a: T, b: T, tolerance: T) {
    let diff = if a > b { a - b } else { b - a };
    assert!(
        diff < tolerance,
        "Float values not equal: {:?} != {:?} (diff: {:?}, tolerance: {:?})",
        a, b, diff, tolerance
    );
}

/// Assert probability is in valid range [0, 1]
pub fn assert_valid_probability<T: Float + std::fmt::Debug>(prob: T) {
    assert!(
        prob >= T::zero() && prob <= T::one(),
        "Invalid probability: {:?} (must be in [0, 1])",
        prob
    );
}

/// Initialize test logging
pub fn init_test_logging() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("debug")
        .try_init();
}