//! SIMD-optimized fusion operations
//!
//! This module provides high-performance implementations of fusion operations
//! using SIMD instructions for combining multi-modal features and scores.

use crate::optimization::simd::{SimdProcessor, SimdConfig};
use crate::types::{ModalityType, DeceptionScore, CombinedFeatures};
use crate::{Result, VeritasError};
use hashbrown::HashMap;
use num_traits::Float;
use std::time::Duration;

/// SIMD-optimized fusion operations
pub struct SimdFusionOps {
    simd_processor: SimdProcessor,
}

impl SimdFusionOps {
    /// Create a new SIMD-optimized fusion operations processor
    pub fn new() -> Result<Self> {
        let config = SimdConfig::auto_detect();
        let simd_processor = SimdProcessor::new(config)?;
        
        Ok(Self {
            simd_processor,
        })
    }
    
    /// SIMD-optimized weighted average calculation
    pub fn weighted_average<T: Float>(
        &self,
        values: &[T],
        weights: &[T],
    ) -> Result<T> {
        if values.len() != weights.len() {
            return Err(VeritasError::invalid_input(
                "Values and weights must have same length",
                "values_and_weights"
            ));
        }
        
        if values.is_empty() {
            return Ok(T::zero());
        }
        
        // Convert to f32 for SIMD processing
        let values_f32: Vec<f32> = values.iter()
            .map(|&v| v.to_f32().unwrap_or(0.0))
            .collect();
        let weights_f32: Vec<f32> = weights.iter()
            .map(|&w| w.to_f32().unwrap_or(0.0))
            .collect();
        
        // Calculate weighted sum using SIMD
        let mut weighted_values = vec![0.0f32; values.len()];
        self.simd_processor.multiply(&values_f32, &weights_f32, &mut weighted_values)?;
        
        let weighted_sum = self.simd_processor.dot_product(&weighted_values, &vec![1.0; weighted_values.len()])?;
        let weight_sum = self.simd_processor.dot_product(&weights_f32, &vec![1.0; weights_f32.len()])?;
        
        if weight_sum > 0.0 {
            Ok(T::from(weighted_sum / weight_sum).unwrap())
        } else {
            Ok(T::zero())
        }
    }
    
    /// SIMD-optimized variance calculation
    pub fn calculate_variance<T: Float>(&self, values: &[T]) -> Result<T> {
        if values.is_empty() {
            return Ok(T::zero());
        }
        
        // Convert to f32 for SIMD
        let values_f32: Vec<f32> = values.iter()
            .map(|&v| v.to_f32().unwrap_or(0.0))
            .collect();
        
        // Calculate mean using SIMD
        let ones = vec![1.0f32; values.len()];
        let sum = self.simd_processor.dot_product(&values_f32, &ones)?;
        let mean = sum / values.len() as f32;
        
        // Calculate variance
        let mean_vec = vec![mean; values.len()];
        let mut centered = vec![0.0f32; values.len()];
        let neg_mean_vec: Vec<f32> = mean_vec.iter().map(|&m| -m).collect();
        self.simd_processor.add(&values_f32, &neg_mean_vec, &mut centered)?;
        
        let mut squared = vec![0.0f32; values.len()];
        self.simd_processor.multiply(&centered, &centered, &mut squared)?;
        
        let variance = self.simd_processor.dot_product(&squared, &ones)? / values.len() as f32;
        
        Ok(T::from(variance).unwrap())
    }
    
    /// SIMD-optimized standard deviation calculation
    pub fn calculate_std_dev<T: Float>(&self, values: &[T]) -> Result<T> {
        let variance = self.calculate_variance(values)?;
        Ok(variance.sqrt())
    }
    
    /// SIMD-optimized normalization (MinMax)
    pub fn normalize_minmax<T: Float>(&self, features: &[T]) -> Result<Vec<T>> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        
        let min_val = features.iter().fold(T::infinity(), |acc, &x| acc.min(x));
        let max_val = features.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        let range = max_val - min_val;
        
        if range <= T::zero() {
            return Ok(features.to_vec());
        }
        
        // Convert to f32 for SIMD
        let features_f32: Vec<f32> = features.iter()
            .map(|&f| f.to_f32().unwrap_or(0.0))
            .collect();
        
        let min_f32 = min_val.to_f32().unwrap_or(0.0);
        let range_f32 = range.to_f32().unwrap_or(1.0);
        
        let min_vec = vec![min_f32; features.len()];
        let mut normalized = vec![0.0f32; features.len()];
        
        // Subtract minimum
        let neg_min_vec: Vec<f32> = min_vec.iter().map(|&m| -m).collect();
        self.simd_processor.add(&features_f32, &neg_min_vec, &mut normalized)?;
        
        // Divide by range
        for val in &mut normalized {
            *val /= range_f32;
        }
        
        Ok(normalized.into_iter()
            .map(|v| T::from(v).unwrap())
            .collect())
    }
    
    /// SIMD-optimized Z-score normalization
    pub fn normalize_zscore<T: Float>(&self, features: &[T]) -> Result<Vec<T>> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        
        let mean = self.calculate_mean(features)?;
        let std_dev = self.calculate_std_dev(features)?;
        
        if std_dev <= T::zero() {
            return Ok(features.to_vec());
        }
        
        // Convert to f32 for SIMD
        let features_f32: Vec<f32> = features.iter()
            .map(|&f| f.to_f32().unwrap_or(0.0))
            .collect();
        
        let mean_f32 = mean.to_f32().unwrap_or(0.0);
        let std_dev_f32 = std_dev.to_f32().unwrap_or(1.0);
        
        let mean_vec = vec![mean_f32; features.len()];
        let mut normalized = vec![0.0f32; features.len()];
        
        // Subtract mean
        let neg_mean_vec: Vec<f32> = mean_vec.iter().map(|&m| -m).collect();
        self.simd_processor.add(&features_f32, &neg_mean_vec, &mut normalized)?;
        
        // Divide by standard deviation
        for val in &mut normalized {
            *val /= std_dev_f32;
        }
        
        Ok(normalized.into_iter()
            .map(|v| T::from(v).unwrap())
            .collect())
    }
    
    /// SIMD-optimized L2 normalization
    pub fn normalize_l2<T: Float>(&self, features: &[T]) -> Result<Vec<T>> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        
        // Convert to f32 for SIMD
        let features_f32: Vec<f32> = features.iter()
            .map(|&f| f.to_f32().unwrap_or(0.0))
            .collect();
        
        // Calculate L2 norm
        let norm = self.simd_processor.dot_product(&features_f32, &features_f32)?.sqrt();
        
        if norm <= 0.0 {
            return Ok(features.to_vec());
        }
        
        // Normalize
        let normalized: Vec<f32> = features_f32.iter()
            .map(|&v| v / norm)
            .collect();
        
        Ok(normalized.into_iter()
            .map(|v| T::from(v).unwrap())
            .collect())
    }
    
    /// SIMD-optimized mean calculation
    pub fn calculate_mean<T: Float>(&self, values: &[T]) -> Result<T> {
        if values.is_empty() {
            return Ok(T::zero());
        }
        
        let values_f32: Vec<f32> = values.iter()
            .map(|&v| v.to_f32().unwrap_or(0.0))
            .collect();
        
        let ones = vec![1.0f32; values.len()];
        let sum = self.simd_processor.dot_product(&values_f32, &ones)?;
        
        Ok(T::from(sum / values.len() as f32).unwrap())
    }
    
    /// SIMD-optimized feature combination
    pub fn combine_features<T: Float>(
        &self,
        features_map: &HashMap<ModalityType, Vec<T>>,
        weights: &HashMap<ModalityType, T>,
    ) -> Result<Vec<T>> {
        // Find the feature dimension
        let feature_dim = features_map.values()
            .next()
            .map(|v| v.len())
            .unwrap_or(0);
        
        if feature_dim == 0 {
            return Ok(vec![]);
        }
        
        let mut combined = vec![T::zero(); feature_dim];
        let mut weight_sum = T::zero();
        
        for (modality, features) in features_map {
            if let Some(&weight) = weights.get(modality) {
                // Convert to f32 for SIMD
                let features_f32: Vec<f32> = features.iter()
                    .map(|&f| f.to_f32().unwrap_or(0.0))
                    .collect();
                let weight_f32 = weight.to_f32().unwrap_or(0.0);
                
                // Scale features by weight
                let mut scaled = vec![0.0f32; feature_dim];
                let weight_vec = vec![weight_f32; feature_dim];
                self.simd_processor.multiply(&features_f32, &weight_vec, &mut scaled)?;
                
                // Add to combined
                let combined_f32: Vec<f32> = combined.iter()
                    .map(|&c| c.to_f32().unwrap_or(0.0))
                    .collect();
                let mut result = vec![0.0f32; feature_dim];
                self.simd_processor.add(&combined_f32, &scaled, &mut result)?;
                
                combined = result.into_iter()
                    .map(|v| T::from(v).unwrap())
                    .collect();
                
                weight_sum = weight_sum + weight;
            }
        }
        
        // Normalize by weight sum
        if weight_sum > T::zero() {
            for val in &mut combined {
                *val = *val / weight_sum;
            }
        }
        
        Ok(combined)
    }
    
    /// SIMD-optimized correlation calculation
    pub fn calculate_correlation<T: Float>(&self, x: &[T], y: &[T]) -> Result<T> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(T::zero());
        }
        
        // Convert to f32 for SIMD
        let x_f32: Vec<f32> = x.iter().map(|&v| v.to_f32().unwrap_or(0.0)).collect();
        let y_f32: Vec<f32> = y.iter().map(|&v| v.to_f32().unwrap_or(0.0)).collect();
        
        // Calculate means
        let x_mean = self.simd_processor.dot_product(&x_f32, &vec![1.0; x.len()])? / x.len() as f32;
        let y_mean = self.simd_processor.dot_product(&y_f32, &vec![1.0; y.len()])? / y.len() as f32;
        
        // Center the data
        let mut x_centered = vec![0.0f32; x.len()];
        let mut y_centered = vec![0.0f32; y.len()];
        
        let x_mean_vec = vec![-x_mean; x.len()];
        let y_mean_vec = vec![-y_mean; y.len()];
        
        self.simd_processor.add(&x_f32, &x_mean_vec, &mut x_centered)?;
        self.simd_processor.add(&y_f32, &y_mean_vec, &mut y_centered)?;
        
        // Calculate covariance
        let mut xy_product = vec![0.0f32; x.len()];
        self.simd_processor.multiply(&x_centered, &y_centered, &mut xy_product)?;
        let covariance = self.simd_processor.dot_product(&xy_product, &vec![1.0; xy_product.len()])?;
        
        // Calculate standard deviations
        let mut x_squared = vec![0.0f32; x.len()];
        let mut y_squared = vec![0.0f32; y.len()];
        
        self.simd_processor.multiply(&x_centered, &x_centered, &mut x_squared)?;
        self.simd_processor.multiply(&y_centered, &y_centered, &mut y_squared)?;
        
        let x_variance = self.simd_processor.dot_product(&x_squared, &vec![1.0; x_squared.len()])?;
        let y_variance = self.simd_processor.dot_product(&y_squared, &vec![1.0; y_squared.len()])?;
        
        let denominator = (x_variance * y_variance).sqrt();
        
        if denominator > 0.0 {
            Ok(T::from(covariance / denominator).unwrap())
        } else {
            Ok(T::zero())
        }
    }
    
    /// SIMD-optimized score agreement calculation
    pub fn calculate_score_agreement<T: Float>(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
    ) -> Result<T> {
        if scores.len() < 2 {
            return Ok(T::one());
        }
        
        let probabilities: Vec<T> = scores.values()
            .map(|s| s.probability)
            .collect();
        
        let variance = self.calculate_variance(&probabilities)?;
        
        // Higher variance = lower agreement
        Ok(T::one() / (T::one() + variance))
    }
    
    /// SIMD-optimized matrix multiplication for fusion networks
    pub fn matrix_multiply<T: Float>(
        &self,
        a: &[T],
        b: &[T],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<T>> {
        if a.len() != rows_a * cols_a || b.len() != cols_a * cols_b {
            return Err(VeritasError::invalid_input(
                "Matrix dimensions mismatch",
                "matrix_dimensions"
            ));
        }
        
        // Convert to f32 for SIMD
        let a_f32: Vec<f32> = a.iter().map(|&v| v.to_f32().unwrap_or(0.0)).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v.to_f32().unwrap_or(0.0)).collect();
        
        let result_f32 = self.simd_processor.matrix_multiply(
            &a_f32,
            &b_f32,
            rows_a,
            cols_a,
            cols_b,
        )?;
        
        Ok(result_f32.into_iter()
            .map(|v| T::from(v).unwrap())
            .collect())
    }
    
    /// SIMD-optimized softmax for attention mechanisms
    pub fn softmax<T: Float>(&self, scores: &mut [T]) -> Result<()> {
        if scores.is_empty() {
            return Ok(());
        }
        
        // Convert to f32 for SIMD
        let mut scores_f32: Vec<f32> = scores.iter()
            .map(|&s| s.to_f32().unwrap_or(0.0))
            .collect();
        
        self.simd_processor.softmax(&mut scores_f32)?;
        
        // Convert back
        for (i, &val) in scores_f32.iter().enumerate() {
            scores[i] = T::from(val).unwrap();
        }
        
        Ok(())
    }
}

/// SIMD-optimized attention fusion
pub struct SimdAttentionFusion<T: Float> {
    simd_ops: SimdFusionOps,
    hidden_dim: usize,
    num_heads: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> SimdAttentionFusion<T> {
    /// Create a new SIMD-optimized attention fusion
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        let simd_ops = SimdFusionOps::new()?;
        
        Ok(Self {
            simd_ops,
            hidden_dim,
            num_heads,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Apply multi-head attention using SIMD
    pub fn apply_attention(
        &self,
        queries: &[T],
        keys: &[T],
        values: &[T],
        seq_len: usize,
    ) -> Result<Vec<T>> {
        let head_dim = self.hidden_dim / self.num_heads;
        
        // Simplified attention mechanism
        // Q, K, V are already projected
        
        // Calculate attention scores: Q @ K^T / sqrt(d_k)
        let scale = T::from((head_dim as f64).sqrt()).unwrap();
        
        // For each head
        let mut all_heads_output = Vec::new();
        
        for head in 0..self.num_heads {
            let head_start = head * head_dim * seq_len;
            let head_end = (head + 1) * head_dim * seq_len;
            
            // Extract head-specific Q, K, V
            let q_head = &queries[head_start..head_end];
            let k_head = &keys[head_start..head_end];
            let v_head = &values[head_start..head_end];
            
            // Compute attention scores
            let scores = self.simd_ops.matrix_multiply(
                q_head,
                k_head,
                seq_len,
                head_dim,
                seq_len,
            )?;
            
            // Scale scores
            let mut scaled_scores: Vec<T> = scores.iter()
                .map(|&s| s / scale)
                .collect();
            
            // Apply softmax
            self.simd_ops.softmax(&mut scaled_scores)?;
            
            // Apply attention to values
            let head_output = self.simd_ops.matrix_multiply(
                &scaled_scores,
                v_head,
                seq_len,
                seq_len,
                head_dim,
            )?;
            
            all_heads_output.extend(head_output);
        }
        
        Ok(all_heads_output)
    }
}

/// SIMD-optimized fusion metrics calculator
pub struct SimdFusionMetrics {
    simd_ops: SimdFusionOps,
}

impl SimdFusionMetrics {
    /// Create a new SIMD-optimized metrics calculator
    pub fn new() -> Result<Self> {
        let simd_ops = SimdFusionOps::new()?;
        Ok(Self { simd_ops })
    }
    
    /// Calculate comprehensive fusion quality metrics
    pub fn calculate_quality_metrics<T: Float>(
        &self,
        scores: &HashMap<ModalityType, DeceptionScore<T>>,
        fused_score: T,
    ) -> Result<FusionQualityMetrics<T>> {
        let agreement_score = self.simd_ops.calculate_score_agreement(scores)?;
        
        // Calculate consistency
        let probabilities: Vec<T> = scores.values()
            .map(|s| s.probability)
            .collect();
        let mean_prob = self.simd_ops.calculate_mean(&probabilities)?;
        let consistency_score = T::one() - (fused_score - mean_prob).abs();
        
        // Calculate confidence variance
        let confidences: Vec<T> = scores.values()
            .map(|s| s.confidence)
            .collect();
        let confidence_variance = self.simd_ops.calculate_variance(&confidences)?;
        
        // Calculate uncertainty
        let uncertainty = if !probabilities.is_empty() {
            let entropy = probabilities.iter()
                .map(|&p| {
                    let p_f64 = p.to_f64().unwrap_or(0.0);
                    if p_f64 > 0.0 && p_f64 < 1.0 {
                        -p_f64 * p_f64.ln() - (1.0 - p_f64) * (1.0 - p_f64).ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>() / probabilities.len() as f64;
            T::from(entropy).unwrap()
        } else {
            T::zero()
        };
        
        Ok(FusionQualityMetrics {
            agreement_score,
            consistency_score,
            confidence_variance,
            uncertainty,
            quality_score: (agreement_score + consistency_score) / T::from(2.0).unwrap(),
        })
    }
}

/// Fusion quality metrics
#[derive(Debug, Clone)]
pub struct FusionQualityMetrics<T: Float> {
    pub agreement_score: T,
    pub consistency_score: T,
    pub confidence_variance: T,
    pub uncertainty: T,
    pub quality_score: T,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_fusion_ops_creation() {
        let fusion_ops = SimdFusionOps::new();
        assert!(fusion_ops.is_ok());
    }
    
    #[test]
    fn test_weighted_average() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        let values = vec![0.8, 0.6, 0.7];
        let weights = vec![0.5, 0.3, 0.2];
        
        let result = fusion_ops.weighted_average(&values, &weights);
        assert!(result.is_ok());
        
        let avg = result.unwrap();
        assert!(avg > 0.6 && avg < 0.8);
    }
    
    #[test]
    fn test_variance_calculation() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = fusion_ops.calculate_variance(&values).unwrap();
        
        // Variance should be 2.0 for these values
        assert!((variance - 2.0).abs() < 0.01);
    }
    
    #[test]
    fn test_normalization_minmax() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = fusion_ops.normalize_minmax(&features).unwrap();
        
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[4], 1.0);
        assert!((normalized[2] - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_normalization_zscore() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = fusion_ops.normalize_zscore(&features).unwrap();
        
        // Mean should be approximately 0
        let mean = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 0.01);
        
        // Standard deviation should be approximately 1
        let variance: f64 = normalized.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / normalized.len() as f64;
        let std_dev = variance.sqrt();
        assert!((std_dev - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_normalization_l2() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        let features = vec![3.0, 4.0]; // 3-4-5 triangle
        let normalized = fusion_ops.normalize_l2(&features).unwrap();
        
        // L2 norm should be 1
        let norm = (normalized[0].powi(2) + normalized[1].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_correlation_calculation() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = fusion_ops.calculate_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.01);
        
        // Perfect negative correlation
        let z = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = fusion_ops.calculate_correlation(&x, &z).unwrap();
        assert!((corr_neg + 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_matrix_multiply() {
        let fusion_ops = SimdFusionOps::new().unwrap();
        
        // 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = fusion_ops.matrix_multiply(&a, &b, 2, 2, 2).unwrap();
        
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result[0], 19.0);
        assert_eq!(result[1], 22.0);
        assert_eq!(result[2], 43.0);
        assert_eq!(result[3], 50.0);
    }
    
    #[test]
    fn test_attention_fusion() {
        let attention = SimdAttentionFusion::<f64>::new(64, 4).unwrap();
        
        let seq_len = 4;
        let hidden_dim = 64;
        
        // Create dummy Q, K, V
        let queries = vec![0.1; seq_len * hidden_dim];
        let keys = vec![0.2; seq_len * hidden_dim];
        let values = vec![0.3; seq_len * hidden_dim];
        
        let result = attention.apply_attention(&queries, &keys, &values, seq_len);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), seq_len * hidden_dim);
    }
    
    #[test]
    fn test_fusion_metrics() {
        let metrics_calc = SimdFusionMetrics::new().unwrap();
        
        let mut scores = HashMap::new();
        
        // Create mock deception scores
        use std::time::Duration;
        use chrono::Utc;
        
        let score1 = TestDeceptionScore {
            probability: 0.7,
            confidence: 0.8,
        };
        let score2 = TestDeceptionScore {
            probability: 0.75,
            confidence: 0.85,
        };
        
        scores.insert(ModalityType::Vision, score1);
        scores.insert(ModalityType::Audio, score2);
        
        let metrics = metrics_calc.calculate_quality_metrics(&scores, 0.72);
        assert!(metrics.is_ok());
        
        let quality = metrics.unwrap();
        assert!(quality.agreement_score > 0.0);
        assert!(quality.consistency_score > 0.0);
        assert!(quality.quality_score > 0.0);
    }
    
    // Test helper struct
    #[derive(Clone)]
    struct TestDeceptionScore<T: Float> {
        probability: T,
        confidence: T,
    }
    
    impl<T: Float> DeceptionScore<T> for TestDeceptionScore<T> {
        fn probability(&self) -> T {
            self.probability
        }
        
        fn confidence(&self) -> T {
            self.confidence
        }
        
        fn contributing_factors(&self) -> Vec<(String, T)> {
            vec![]
        }
        
        fn timestamp(&self) -> chrono::DateTime<chrono::Utc> {
            chrono::Utc::now()
        }
        
        fn processing_time(&self) -> Duration {
            Duration::from_millis(10)
        }
    }
}