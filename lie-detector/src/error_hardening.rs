//! Error hardening utilities and improved error handling
//!
//! This module provides enhanced error handling utilities and safer versions
//! of operations that could potentially panic in edge cases.

use crate::error::{VeritasError, Result, DataQualitySeverity};
use crate::types::*;
use num_traits::Float;
use std::collections::HashMap;
use std::time::Duration;

/// Safe division with error handling for floating point operations
pub fn safe_divide<T: Float>(numerator: T, denominator: T, context: &str) -> Result<T> {
    if denominator == T::zero() {
        return Err(VeritasError::edge_case(
            "division_by_zero",
            format!("Division by zero in {}", context),
            false,
            Some("Use fallback value or skip calculation".to_string()),
        ));
    }
    
    if denominator.is_nan() || numerator.is_nan() {
        return Err(VeritasError::edge_case(
            "nan_values",
            format!("NaN value encountered in division: {}", context),
            false,
            Some("Validate input data for NaN values".to_string()),
        ));
    }
    
    if denominator.is_infinite() || numerator.is_infinite() {
        return Err(VeritasError::edge_case(
            "infinite_values",
            format!("Infinite value encountered in division: {}", context),
            false,
            Some("Check for overflow conditions".to_string()),
        ));
    }
    
    let result = numerator / denominator;
    
    if result.is_nan() || result.is_infinite() {
        return Err(VeritasError::edge_case(
            "invalid_result",
            format!("Division resulted in invalid value: {}", context),
            false,
            Some("Check for numerical instability".to_string()),
        ));
    }
    
    Ok(result)
}

/// Safe square root with validation
pub fn safe_sqrt<T: Float>(value: T, context: &str) -> Result<T> {
    if value < T::zero() {
        return Err(VeritasError::edge_case(
            "negative_sqrt",
            format!("Attempted square root of negative value in {}: {}", context, value),
            false,
            Some("Use absolute value or skip calculation".to_string()),
        ));
    }
    
    if value.is_nan() {
        return Err(VeritasError::edge_case(
            "nan_sqrt",
            format!("Attempted square root of NaN in {}", context),
            false,
            Some("Validate input data".to_string()),
        ));
    }
    
    if value.is_infinite() {
        return Err(VeritasError::edge_case(
            "infinite_sqrt",
            format!("Attempted square root of infinite value in {}", context),
            true,
            Some("Result will be infinite".to_string()),
        ));
    }
    
    Ok(value.sqrt())
}

/// Safe logarithm with validation
pub fn safe_ln<T: Float>(value: T, context: &str) -> Result<T> {
    if value <= T::zero() {
        return Err(VeritasError::edge_case(
            "invalid_ln",
            format!("Attempted logarithm of non-positive value in {}: {}", context, value),
            false,
            Some("Use small positive value or skip calculation".to_string()),
        ));
    }
    
    if value.is_nan() {
        return Err(VeritasError::edge_case(
            "nan_ln",
            format!("Attempted logarithm of NaN in {}", context),
            false,
            Some("Validate input data".to_string()),
        ));
    }
    
    if value.is_infinite() {
        return Err(VeritasError::edge_case(
            "infinite_ln",
            format!("Attempted logarithm of infinite value in {}", context),
            true,
            Some("Result will be infinite".to_string()),
        ));
    }
    
    Ok(value.ln())
}

/// Safe mean calculation with empty collection handling
pub fn safe_mean<T: Float>(values: &[T], context: &str) -> Result<T> {
    if values.is_empty() {
        return Err(VeritasError::edge_case(
            "empty_collection",
            format!("Cannot calculate mean of empty collection in {}", context),
            false,
            Some("Use default value or skip calculation".to_string()),
        ));
    }
    
    // Check for invalid values
    for (i, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return Err(VeritasError::data_quality(
                "nan_values",
                i as f64,
                values.len() as f64,
                DataQualitySeverity::Critical,
            ));
        }
        
        if value.is_infinite() {
            return Err(VeritasError::data_quality(
                "infinite_values",
                value.to_f64().unwrap_or(f64::INFINITY),
                1.0,
                DataQualitySeverity::High,
            ));
        }
    }
    
    let sum = values.iter().fold(T::zero(), |acc, &x| acc + x);
    let count = T::from(values.len()).unwrap();
    
    safe_divide(sum, count, &format!("mean calculation in {}", context))
}

/// Safe variance calculation
pub fn safe_variance<T: Float>(values: &[T], context: &str) -> Result<T> {
    if values.len() < 2 {
        return Err(VeritasError::edge_case(
            "insufficient_data",
            format!("Cannot calculate variance with less than 2 values in {}", context),
            false,
            Some("Use default variance or skip calculation".to_string()),
        ));
    }
    
    let mean = safe_mean(values, context)?;
    
    let squared_diffs: Result<Vec<T>> = values.iter()
        .map(|&x| {
            let diff = x - mean;
            Ok(diff * diff)
        })
        .collect();
    
    let squared_diffs = squared_diffs?;
    let variance_sum = squared_diffs.iter().fold(T::zero(), |acc, &x| acc + x);
    let n_minus_1 = T::from(values.len() - 1).unwrap();
    
    safe_divide(variance_sum, n_minus_1, &format!("variance calculation in {}", context))
}

/// Safe standard deviation calculation
pub fn safe_std_dev<T: Float>(values: &[T], context: &str) -> Result<T> {
    let variance = safe_variance(values, context)?;
    safe_sqrt(variance, &format!("standard deviation in {}", context))
}

/// Safe array indexing with bounds checking
pub fn safe_index<T>(array: &[T], index: usize, context: &str) -> Result<&T> {
    array.get(index).ok_or_else(|| {
        VeritasError::edge_case(
            "index_out_of_bounds",
            format!("Index {} out of bounds for array of length {} in {}", 
                   index, array.len(), context),
            false,
            Some("Check array bounds before indexing".to_string()),
        )
    })
}

/// Safe mutable array indexing with bounds checking
pub fn safe_index_mut<T>(array: &mut [T], index: usize, context: &str) -> Result<&mut T> {
    let len = array.len();
    array.get_mut(index).ok_or_else(|| {
        VeritasError::edge_case(
            "index_out_of_bounds",
            format!("Index {} out of bounds for array of length {} in {}", 
                   index, len, context),
            false,
            Some("Check array bounds before indexing".to_string()),
        )
    })
}

/// Safe matrix access with bounds checking
pub fn safe_matrix_get<T>(matrix: &[Vec<T>], row: usize, col: usize, context: &str) -> Result<&T> {
    let matrix_row = matrix.get(row).ok_or_else(|| {
        VeritasError::edge_case(
            "row_out_of_bounds",
            format!("Row {} out of bounds for matrix with {} rows in {}", 
                   row, matrix.len(), context),
            false,
            Some("Check matrix dimensions before access".to_string()),
        )
    })?;
    
    matrix_row.get(col).ok_or_else(|| {
        VeritasError::edge_case(
            "column_out_of_bounds",
            format!("Column {} out of bounds for row with {} columns in {}", 
                   col, matrix_row.len(), context),
            false,
            Some("Check matrix dimensions before access".to_string()),
        )
    })
}

/// Safe percentage calculation
pub fn safe_percentage<T: Float>(part: T, whole: T, context: &str) -> Result<T> {
    if whole == T::zero() {
        return Err(VeritasError::edge_case(
            "zero_denominator",
            format!("Cannot calculate percentage with zero total in {}", context),
            false,
            Some("Use 0% or skip calculation".to_string()),
        ));
    }
    
    let ratio = safe_divide(part, whole, context)?;
    Ok(ratio * T::from(100.0).unwrap())
}

/// Validate floating point value
pub fn validate_float<T: Float>(value: T, name: &str, context: &str) -> Result<T> {
    if value.is_nan() {
        return Err(VeritasError::data_quality(
            "nan_value",
            0.0,
            1.0,
            DataQualitySeverity::Critical,
        ));
    }
    
    if value.is_infinite() {
        return Err(VeritasError::data_quality(
            "infinite_value",
            value.to_f64().unwrap_or(f64::INFINITY),
            1.0,
            DataQualitySeverity::High,
        ));
    }
    
    Ok(value)
}

/// Validate array dimensions
pub fn validate_dimensions(width: u32, height: u32, channels: u32, context: &str) -> Result<()> {
    if width == 0 || height == 0 || channels == 0 {
        return Err(VeritasError::invalid_input(
            format!("Dimensions cannot be zero in {}: {}x{}x{}", context, width, height, channels),
            "dimensions",
        ));
    }
    
    // Check for overflow
    let total_size = width as u64 * height as u64 * channels as u64;
    if total_size > u32::MAX as u64 {
        return Err(VeritasError::edge_case(
            "dimension_overflow",
            format!("Dimensions too large in {}: {}x{}x{}", context, width, height, channels),
            false,
            Some("Reduce image dimensions or process in chunks".to_string()),
        ));
    }
    
    // Check for reasonable limits (prevent excessive memory usage)
    const MAX_DIMENSION: u32 = 10000;
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(VeritasError::edge_case(
            "excessive_dimensions",
            format!("Dimensions exceed reasonable limits in {}: {}x{}", context, width, height),
            false,
            Some("Consider resizing input or processing in tiles".to_string()),
        ));
    }
    
    Ok(())
}

/// Validate audio configuration
pub fn validate_audio_config(sample_rate: u32, chunk_size: usize, context: &str) -> Result<()> {
    if sample_rate == 0 {
        return Err(VeritasError::invalid_input(
            format!("Sample rate cannot be zero in {}", context),
            "sample_rate",
        ));
    }
    
    if sample_rate < 8000 {
        return Err(VeritasError::data_quality(
            "low_sample_rate",
            sample_rate as f64,
            8000.0,
            DataQualitySeverity::Medium,
        ));
    }
    
    if sample_rate > 192000 {
        return Err(VeritasError::edge_case(
            "excessive_sample_rate",
            format!("Sample rate {} exceeds reasonable limit in {}", sample_rate, context),
            false,
            Some("Use standard sample rates (8kHz-192kHz)".to_string()),
        ));
    }
    
    if chunk_size == 0 {
        return Err(VeritasError::invalid_input(
            format!("Chunk size cannot be zero in {}", context),
            "chunk_size",
        ));
    }
    
    if chunk_size > 1_000_000 {
        return Err(VeritasError::edge_case(
            "excessive_chunk_size",
            format!("Chunk size {} too large in {}", chunk_size, context),
            false,
            Some("Use smaller chunk sizes for real-time processing".to_string()),
        ));
    }
    
    Ok(())
}

/// Validate text input
pub fn validate_text_input(text: &str, context: &str) -> Result<()> {
    if text.is_empty() {
        return Err(VeritasError::invalid_input(
            format!("Text cannot be empty in {}", context),
            "text",
        ));
    }
    
    if text.trim().is_empty() {
        return Err(VeritasError::data_quality(
            "whitespace_only",
            text.len() as f64,
            1.0,
            DataQualitySeverity::Medium,
        ));
    }
    
    const MAX_TEXT_LENGTH: usize = 1_000_000; // 1MB
    if text.len() > MAX_TEXT_LENGTH {
        return Err(VeritasError::edge_case(
            "text_too_long",
            format!("Text length {} exceeds limit in {}", text.len(), context),
            false,
            Some("Split text into smaller chunks".to_string()),
        ));
    }
    
    // Check for excessive control characters
    let control_char_count = text.chars()
        .filter(|c| c.is_control() && !matches!(*c, '\t' | '\n' | '\r'))
        .count();
    
    let control_char_ratio = control_char_count as f64 / text.len() as f64;
    if control_char_ratio > 0.1 {
        return Err(VeritasError::data_quality(
            "excessive_control_chars",
            control_char_ratio,
            0.1,
            DataQualitySeverity::Medium,
        ));
    }
    
    Ok(())
}

/// Validate probability value
pub fn validate_probability<T: Float>(prob: T, context: &str) -> Result<T> {
    let prob = validate_float(prob, "probability", context)?;
    
    if prob < T::zero() || prob > T::one() {
        return Err(VeritasError::invalid_input(
            format!("Probability {} out of range [0,1] in {}", 
                   prob.to_f64().unwrap_or(0.0), context),
            "probability",
        ));
    }
    
    Ok(prob)
}

/// Validate confidence value
pub fn validate_confidence<T: Float>(conf: T, context: &str) -> Result<T> {
    let conf = validate_float(conf, "confidence", context)?;
    
    if conf < T::zero() || conf > T::one() {
        return Err(VeritasError::invalid_input(
            format!("Confidence {} out of range [0,1] in {}", 
                   conf.to_f64().unwrap_or(0.0), context),
            "confidence",
        ));
    }
    
    Ok(conf)
}

/// Safe weight normalization
pub fn safe_normalize_weights<T: Float>(weights: &mut HashMap<ModalityType, T>, context: &str) -> Result<()> {
    if weights.is_empty() {
        return Err(VeritasError::invalid_input(
            format!("Cannot normalize empty weights in {}", context),
            "weights",
        ));
    }
    
    // Validate all weights first
    for (modality, &weight) in weights.iter() {
        let weight = validate_float(weight, &format!("{:?}_weight", modality), context)?;
        if weight < T::zero() {
            return Err(VeritasError::invalid_input(
                format!("Weight for {:?} cannot be negative in {}", modality, context),
                "weights",
            ));
        }
    }
    
    let total: T = weights.values().fold(T::zero(), |acc, &x| acc + x);
    
    if total == T::zero() {
        return Err(VeritasError::edge_case(
            "zero_weight_sum",
            format!("Sum of weights is zero in {}", context),
            false,
            Some("Ensure at least one weight is positive".to_string()),
        ));
    }
    
    // Normalize weights
    for weight in weights.values_mut() {
        *weight = *weight / total;
    }
    
    Ok(())
}

/// Timeout wrapper for operations
pub async fn with_timeout<F, T>(
    future: F,
    timeout_duration: Duration,
    operation_name: &str,
) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    match tokio::time::timeout(timeout_duration, future).await {
        Ok(result) => Ok(result),
        Err(_) => Err(VeritasError::timeout_error(
            operation_name,
            timeout_duration.as_millis() as u64,
        )),
    }
}

/// Memory allocation guard
pub fn check_memory_allocation(size: usize, context: &str) -> Result<()> {
    const MAX_ALLOCATION_MB: usize = 1024; // 1GB limit
    const BYTES_PER_MB: usize = 1024 * 1024;
    
    if size > MAX_ALLOCATION_MB * BYTES_PER_MB {
        return Err(VeritasError::memory_error_with_size(
            format!("Allocation size {} exceeds limit in {}", size, context),
            size,
        ));
    }
    
    Ok(())
}

/// Recovery utilities for common error scenarios
pub mod recovery {
    use super::*;
    
    /// Attempt to recover from division by zero
    pub fn recover_division_by_zero<T: Float>(
        numerator: T,
        fallback: T,
        context: &str,
    ) -> Result<T> {
        if numerator == T::zero() {
            Ok(T::zero())
        } else {
            Ok(fallback)
        }
    }
    
    /// Attempt to recover from empty collection
    pub fn recover_empty_collection<T: Clone>(fallback: T) -> T {
        fallback
    }
    
    /// Attempt to recover from invalid float values
    pub fn recover_invalid_float<T: Float>(fallback: T) -> T {
        fallback
    }
    
    /// Create a degraded result when modalities fail
    pub fn create_degraded_result<T: Float>(
        available_modalities: &[ModalityType],
        failed_modalities: &[ModalityType],
        partial_confidence: T,
    ) -> Result<()> {
        if available_modalities.is_empty() {
            return Err(VeritasError::edge_case(
                "all_modalities_failed",
                "All modalities failed, cannot create degraded result".to_string(),
                false,
                Some("Check system configuration and inputs".to_string()),
            ));
        }
        
        let confidence_impact = T::from(failed_modalities.len() as f64 / 
                                       (available_modalities.len() + failed_modalities.len()) as f64)
                                .unwrap();
        
        if confidence_impact > T::from(0.5).unwrap() {
            return Err(VeritasError::modality_degradation(
                failed_modalities.iter().map(|m| format!("{:?}", m)).collect(),
                available_modalities.iter().map(|m| format!("{:?}", m)).collect(),
                confidence_impact.to_f64().unwrap_or(0.5),
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_divide() {
        // Normal case
        assert!(safe_divide(10.0, 2.0, "test").is_ok());
        assert_eq!(safe_divide(10.0, 2.0, "test").unwrap(), 5.0);
        
        // Division by zero
        assert!(safe_divide(10.0, 0.0, "test").is_err());
        
        // NaN values
        assert!(safe_divide(f64::NAN, 2.0, "test").is_err());
        assert!(safe_divide(10.0, f64::NAN, "test").is_err());
        
        // Infinite values
        assert!(safe_divide(f64::INFINITY, 2.0, "test").is_err());
        assert!(safe_divide(10.0, f64::INFINITY, "test").is_err());
    }
    
    #[test]
    fn test_safe_mean() {
        // Normal case
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(safe_mean(&values, "test").is_ok());
        assert_eq!(safe_mean(&values, "test").unwrap(), 3.0);
        
        // Empty collection
        let empty: Vec<f64> = vec![];
        assert!(safe_mean(&empty, "test").is_err());
        
        // NaN values
        let nan_values = vec![1.0, f64::NAN, 3.0];
        assert!(safe_mean(&nan_values, "test").is_err());
    }
    
    #[test]
    fn test_validate_dimensions() {
        // Valid dimensions
        assert!(validate_dimensions(100, 100, 3, "test").is_ok());
        
        // Zero dimensions
        assert!(validate_dimensions(0, 100, 3, "test").is_err());
        assert!(validate_dimensions(100, 0, 3, "test").is_err());
        assert!(validate_dimensions(100, 100, 0, "test").is_err());
        
        // Oversized dimensions
        assert!(validate_dimensions(20000, 20000, 3, "test").is_err());
    }
    
    #[test]
    fn test_validate_text_input() {
        // Valid text
        assert!(validate_text_input("Hello world", "test").is_ok());
        
        // Empty text
        assert!(validate_text_input("", "test").is_err());
        
        // Whitespace only
        assert!(validate_text_input("   \t\n  ", "test").is_err());
        
        // Too long
        let long_text = "a".repeat(2_000_000);
        assert!(validate_text_input(&long_text, "test").is_err());
    }
}