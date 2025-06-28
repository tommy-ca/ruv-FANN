//! Temporal alignment for synchronizing multi-modal data streams
//!
//! This module provides sophisticated temporal alignment algorithms to synchronize
//! different modality data streams that may have different sampling rates, delays,
//! and temporal characteristics for accurate fusion.

use crate::error::{FusionError, Result};
use crate::types::{CombinedFeatures, ModalityType, TemporalInfo};
use chrono::{DateTime, Utc};
use hashbrown::HashMap;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

/// Temporal aligner for multi-modal synchronization
#[derive(Debug, Clone)]
pub struct TemporalAligner<T: Float> {
    config: AlignmentConfig<T>,
    sync_buffers: HashMap<ModalityType, SyncBuffer<T>>,
    alignment_history: VecDeque<AlignmentResult<T>>,
    drift_compensation: HashMap<ModalityType, DriftCompensator<T>>,
}

/// Configuration for temporal alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig<T: Float> {
    /// Maximum allowed time offset between modalities (in seconds)
    pub max_time_offset: T,
    /// Minimum overlap required for synchronization (0.0 to 1.0)
    pub min_overlap_ratio: T,
    /// Interpolation method for resampling
    pub interpolation_method: InterpolationMethod,
    /// Window size for temporal smoothing
    pub smoothing_window: usize,
    /// Enable adaptive sync offset learning
    pub adaptive_sync: bool,
    /// Maximum buffer size per modality
    pub max_buffer_size: usize,
    /// Quality threshold for accepting alignment
    pub quality_threshold: T,
    /// Enable drift compensation
    pub enable_drift_compensation: bool,
}

/// Interpolation methods for temporal resampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    Cubic,
    /// Nearest neighbor
    Nearest,
    /// Sinc interpolation for audio
    Sinc,
    /// Gaussian kernel interpolation
    Gaussian,
}

impl<T: Float> Default for AlignmentConfig<T> {
    fn default() -> Self {
        Self {
            max_time_offset: T::from(0.5).unwrap(), // 500ms
            min_overlap_ratio: T::from(0.8).unwrap(),
            interpolation_method: InterpolationMethod::Linear,
            smoothing_window: 5,
            adaptive_sync: true,
            max_buffer_size: 1000,
            quality_threshold: T::from(0.85).unwrap(),
            enable_drift_compensation: true,
        }
    }
}

/// Synchronization buffer for a single modality
#[derive(Debug, Clone)]
pub struct SyncBuffer<T: Float> {
    /// Temporal data points
    data: VecDeque<TemporalDataPoint<T>>,
    /// Modality type
    modality: ModalityType,
    /// Estimated sampling rate
    sampling_rate: Option<T>,
    /// Last synchronization timestamp
    last_sync: Option<DateTime<Utc>>,
    /// Buffer statistics
    stats: BufferStats<T>,
}

/// Individual temporal data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDataPoint<T: Float> {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Feature vector
    pub features: Vec<T>,
    /// Quality score for this data point
    pub quality: T,
    /// Original index in the sequence
    pub original_index: usize,
}

/// Buffer statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats<T: Float> {
    /// Average sampling interval
    pub avg_interval: T,
    /// Variance in sampling intervals
    pub interval_variance: T,
    /// Number of data points
    pub count: usize,
    /// Quality statistics
    pub avg_quality: T,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Result of temporal alignment operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult<T: Float> {
    /// Aligned features
    pub aligned_features: CombinedFeatures<T>,
    /// Synchronization quality score
    pub sync_quality: T,
    /// Time offsets applied per modality
    pub applied_offsets: HashMap<ModalityType, Duration>,
    /// Interpolation statistics
    pub interpolation_stats: InterpolationStats<T>,
    /// Temporal window used
    pub temporal_window: (DateTime<Utc>, DateTime<Utc>),
}

/// Statistics about interpolation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationStats<T: Float> {
    /// Number of points interpolated per modality
    pub interpolated_points: HashMap<ModalityType, usize>,
    /// Interpolation error estimates
    pub interpolation_errors: HashMap<ModalityType, T>,
    /// Resampling ratios applied
    pub resampling_ratios: HashMap<ModalityType, T>,
}

/// Drift compensation for long-term synchronization
#[derive(Debug, Clone)]
pub struct DriftCompensator<T: Float> {
    /// Estimated drift rate (seconds per second)
    drift_rate: T,
    /// Confidence in drift estimate
    confidence: T,
    /// History of timing measurements
    timing_history: VecDeque<TimingMeasurement<T>>,
    /// Last compensation applied
    last_compensation: DateTime<Utc>,
}

/// Timing measurement for drift estimation
#[derive(Debug, Clone)]
pub struct TimingMeasurement<T: Float> {
    /// Measurement timestamp
    timestamp: DateTime<Utc>,
    /// Observed offset from expected time
    offset: T,
    /// Quality of this measurement
    quality: T,
}

impl<T: Float + Send + Sync> TemporalAligner<T> {
    /// Create a new temporal aligner
    pub fn new(config: AlignmentConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            sync_buffers: HashMap::new(),
            alignment_history: VecDeque::new(),
            drift_compensation: HashMap::new(),
        })
    }
    
    /// Add data to a modality buffer
    pub fn add_data(
        &mut self,
        modality: ModalityType,
        timestamp: DateTime<Utc>,
        features: Vec<T>,
        quality: T,
    ) -> Result<()> {
        let data_point = TemporalDataPoint {
            timestamp,
            features,
            quality,
            original_index: 0, // Will be set when buffering
        };
        
        let buffer = self.sync_buffers.entry(modality)
            .or_insert_with(|| SyncBuffer::new(modality));
        
        buffer.add_data_point(data_point, self.config.max_buffer_size)?;
        
        // Update drift compensation if enabled
        if self.config.enable_drift_compensation {
            self.update_drift_compensation(modality, timestamp)?;
        }
        
        Ok(())
    }
    
    /// Align features across modalities
    pub fn align(&self, features: &CombinedFeatures<T>) -> Result<CombinedFeatures<T>> {
        // Extract temporal window from features
        let temporal_window = self.determine_temporal_window(features)?;
        
        // Collect synchronized data for each modality
        let mut aligned_modalities = HashMap::new();
        let mut applied_offsets = HashMap::new();
        let mut interpolation_stats = InterpolationStats {
            interpolated_points: HashMap::new(),
            interpolation_errors: HashMap::new(),
            resampling_ratios: HashMap::new(),
        };
        
        // Find the reference modality (highest quality/sampling rate)
        let reference_modality = self.find_reference_modality(features)?;
        
        // Align each modality to the reference
        for (modality, modality_features) in &features.modalities {
            if let Some(buffer) = self.sync_buffers.get(modality) {
                let (aligned_data, offset, interp_stats) = self.align_modality_to_reference(
                    *modality,
                    buffer,
                    &temporal_window,
                    reference_modality,
                )?;
                
                aligned_modalities.insert(*modality, aligned_data);
                applied_offsets.insert(*modality, offset);
                interpolation_stats.interpolated_points.insert(*modality, interp_stats.point_count);
                interpolation_stats.interpolation_errors.insert(*modality, interp_stats.error);
                interpolation_stats.resampling_ratios.insert(*modality, interp_stats.resampling_ratio);
            } else {
                // Fallback: use original features if no buffer available
                aligned_modalities.insert(*modality, modality_features.clone());
                applied_offsets.insert(*modality, Duration::from_millis(0));
            }
        }
        
        // Combine aligned features
        let combined = self.combine_aligned_features(&aligned_modalities)?;
        
        // Calculate dimension mapping
        let dimension_map = self.calculate_dimension_map(&aligned_modalities);
        
        // Create temporal info for aligned data
        let temporal_info = TemporalInfo {
            start_time: temporal_window.0,
            end_time: temporal_window.1,
            frame_rate: self.estimate_frame_rate(&aligned_modalities),
            sample_rate: self.estimate_sample_rate(&aligned_modalities),
            sync_offsets: applied_offsets.iter()
                .map(|(k, &v)| (*k, v))
                .collect(),
        };
        
        Ok(CombinedFeatures {
            modalities: aligned_modalities,
            combined,
            dimension_map,
            temporal_info,
        })
    }
    
    /// Determine optimal temporal window for alignment
    fn determine_temporal_window(
        &self,
        features: &CombinedFeatures<T>,
    ) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
        let start_time = features.temporal_info.start_time;
        let end_time = features.temporal_info.end_time;
        
        // Find overlapping time window across all modalities
        let mut common_start = start_time;
        let mut common_end = end_time;
        
        for (modality, _) in &features.modalities {
            if let Some(buffer) = self.sync_buffers.get(modality) {
                if let Some(buffer_start) = buffer.get_start_time() {
                    common_start = common_start.max(buffer_start);
                }
                if let Some(buffer_end) = buffer.get_end_time() {
                    common_end = common_end.min(buffer_end);
                }
            }
        }
        
        // Check if we have sufficient overlap
        let total_duration = (end_time - start_time).num_milliseconds() as f64 / 1000.0;
        let overlap_duration = (common_end - common_start).num_milliseconds() as f64 / 1000.0;
        let overlap_ratio = if total_duration > 0.0 {
            overlap_duration / total_duration
        } else {
            0.0
        };
        
        if T::from(overlap_ratio).unwrap() < self.config.min_overlap_ratio {
            return Err(FusionError::TemporalAlignment {
                reason: format!(
                    "Insufficient temporal overlap: {:.2}% < {:.2}%",
                    overlap_ratio * 100.0,
                    self.config.min_overlap_ratio.to_f64().unwrap_or(80.0)
                ),
            });
        }
        
        Ok((common_start, common_end))
    }
    
    /// Find the reference modality for alignment
    fn find_reference_modality(
        &self,
        features: &CombinedFeatures<T>,
    ) -> Result<ModalityType> {
        let mut best_modality = ModalityType::Vision; // Default
        let mut best_score = T::zero();
        
        for (modality, _) in &features.modalities {
            let score = if let Some(buffer) = self.sync_buffers.get(modality) {
                // Score based on sampling rate, data quality, and stability
                let sampling_score = buffer.sampling_rate.unwrap_or(T::one());
                let quality_score = buffer.stats.avg_quality;
                let stability_score = T::one() / (T::one() + buffer.stats.interval_variance);
                
                sampling_score * quality_score * stability_score
            } else {
                T::from(0.5).unwrap() // Default score
            };
            
            if score > best_score {
                best_score = score;
                best_modality = *modality;
            }
        }
        
        Ok(best_modality)
    }
    
    /// Align a specific modality to the reference
    fn align_modality_to_reference(
        &self,
        modality: ModalityType,
        buffer: &SyncBuffer<T>,
        temporal_window: &(DateTime<Utc>, DateTime<Utc>),
        reference_modality: ModalityType,
    ) -> Result<(Vec<T>, Duration, ModalityInterpolationStats<T>)> {
        // Extract data within temporal window
        let windowed_data = buffer.get_data_in_window(temporal_window.0, temporal_window.1)?;
        
        if windowed_data.is_empty() {
            return Err(FusionError::TemporalAlignment {
                reason: format!("No data available for {} in temporal window", modality),
            });
        }
        
        // Apply drift compensation if available
        let compensated_data = if self.config.enable_drift_compensation {
            self.apply_drift_compensation(modality, windowed_data)?
        } else {
            windowed_data
        };
        
        // Resample to target sampling rate
        let target_sampling_rate = self.get_target_sampling_rate(reference_modality)?;
        let (resampled_data, interp_stats) = self.resample_data(
            &compensated_data,
            target_sampling_rate,
            temporal_window,
        )?;
        
        // Apply temporal smoothing if configured
        let smoothed_data = if self.config.smoothing_window > 1 {
            self.apply_temporal_smoothing(&resampled_data)?
        } else {
            resampled_data
        };
        
        // Calculate applied time offset
        let offset = self.calculate_applied_offset(modality, temporal_window)?;
        
        Ok((smoothed_data, offset, interp_stats))
    }
    
    /// Resample data to target sampling rate
    fn resample_data(
        &self,
        data: &[TemporalDataPoint<T>],
        target_rate: T,
        temporal_window: &(DateTime<Utc>, DateTime<Utc>),
    ) -> Result<(Vec<T>, ModalityInterpolationStats<T>)> {
        if data.is_empty() {
            return Ok((vec![], ModalityInterpolationStats::default()));
        }
        
        let window_duration = (temporal_window.1 - temporal_window.0).num_milliseconds() as f64 / 1000.0;
        let target_samples = (T::from(window_duration).unwrap() * target_rate).to_usize().unwrap_or(1);
        
        let mut resampled = Vec::new();
        let mut interpolation_error = T::zero();
        let original_rate = T::from(data.len()).unwrap() / T::from(window_duration).unwrap();
        
        for i in 0..target_samples {
            let target_time_ratio = T::from(i).unwrap() / T::from(target_samples).unwrap();
            let target_timestamp = temporal_window.0 + chrono::Duration::milliseconds(
                (target_time_ratio.to_f64().unwrap() * window_duration * 1000.0) as i64
            );
            
            let interpolated_value = self.interpolate_at_timestamp(data, target_timestamp)?;
            resampled.extend(interpolated_value.features);
            
            // Estimate interpolation error (simplified)
            interpolation_error = interpolation_error + (interpolated_value.quality - T::one()).abs();
        }
        
        let avg_error = if target_samples > 0 {
            interpolation_error / T::from(target_samples).unwrap()
        } else {
            T::zero()
        };
        
        let stats = ModalityInterpolationStats {
            point_count: target_samples,
            error: avg_error,
            resampling_ratio: target_rate / original_rate,
        };
        
        Ok((resampled, stats))
    }
    
    /// Interpolate feature value at specific timestamp
    fn interpolate_at_timestamp(
        &self,
        data: &[TemporalDataPoint<T>],
        target_timestamp: DateTime<Utc>,
    ) -> Result<TemporalDataPoint<T>> {
        if data.is_empty() {
            return Err(FusionError::TemporalAlignment {
                reason: "No data available for interpolation".to_string(),
            });
        }
        
        // Find surrounding data points
        let mut before_idx = None;
        let mut after_idx = None;
        
        for (i, point) in data.iter().enumerate() {
            if point.timestamp <= target_timestamp {
                before_idx = Some(i);
            }
            if point.timestamp >= target_timestamp && after_idx.is_none() {
                after_idx = Some(i);
                break;
            }
        }
        
        let interpolated = match (before_idx, after_idx) {
            (Some(before), Some(after)) if before != after => {
                // Interpolate between two points
                self.interpolate_between_points(&data[before], &data[after], target_timestamp)?
            }
            (Some(idx), _) | (_, Some(idx)) => {
                // Use nearest point
                data[idx].clone()
            }
            _ => {
                return Err(FusionError::TemporalAlignment {
                    reason: "Cannot find interpolation points".to_string(),
                });
            }
        };
        
        Ok(TemporalDataPoint {
            timestamp: target_timestamp,
            features: interpolated.features,
            quality: interpolated.quality,
            original_index: 0,
        })
    }
    
    /// Interpolate between two data points
    fn interpolate_between_points(
        &self,
        point_a: &TemporalDataPoint<T>,
        point_b: &TemporalDataPoint<T>,
        target_timestamp: DateTime<Utc>,
    ) -> Result<TemporalDataPoint<T>> {
        let duration_ab = (point_b.timestamp - point_a.timestamp).num_milliseconds() as f64;
        let duration_at = (target_timestamp - point_a.timestamp).num_milliseconds() as f64;
        
        if duration_ab <= 0.0 {
            return Ok(point_a.clone());
        }
        
        let interpolation_ratio = T::from(duration_at / duration_ab).unwrap();
        
        let interpolated_features = match self.config.interpolation_method {
            InterpolationMethod::Linear => {
                self.linear_interpolate(&point_a.features, &point_b.features, interpolation_ratio)?
            }
            InterpolationMethod::Nearest => {
                if interpolation_ratio < T::from(0.5).unwrap() {
                    point_a.features.clone()
                } else {
                    point_b.features.clone()
                }
            }
            InterpolationMethod::Cubic => {
                // Simplified cubic interpolation (would need more points for true cubic)
                self.cubic_interpolate(&point_a.features, &point_b.features, interpolation_ratio)?
            }
            InterpolationMethod::Sinc => {
                // Simplified sinc interpolation
                self.sinc_interpolate(&point_a.features, &point_b.features, interpolation_ratio)?
            }
            InterpolationMethod::Gaussian => {
                // Gaussian kernel interpolation
                self.gaussian_interpolate(&point_a.features, &point_b.features, interpolation_ratio)?
            }
        };
        
        let interpolated_quality = point_a.quality + 
            (point_b.quality - point_a.quality) * interpolation_ratio;
        
        Ok(TemporalDataPoint {
            timestamp: target_timestamp,
            features: interpolated_features,
            quality: interpolated_quality,
            original_index: 0,
        })
    }
    
    /// Linear interpolation between feature vectors
    fn linear_interpolate(
        &self,
        features_a: &[T],
        features_b: &[T],
        ratio: T,
    ) -> Result<Vec<T>> {
        if features_a.len() != features_b.len() {
            return Err(FusionError::TemporalAlignment {
                reason: "Feature dimension mismatch in interpolation".to_string(),
            });
        }
        
        Ok(features_a.iter()
            .zip(features_b.iter())
            .map(|(&a, &b)| a + (b - a) * ratio)
            .collect())
    }
    
    /// Cubic interpolation (simplified)
    fn cubic_interpolate(
        &self,
        features_a: &[T],
        features_b: &[T],
        ratio: T,
    ) -> Result<Vec<T>> {
        // Simplified cubic interpolation using Hermite spline
        let ratio_squared = ratio * ratio;
        let ratio_cubed = ratio_squared * ratio;
        
        let h00 = T::from(2.0).unwrap() * ratio_cubed - T::from(3.0).unwrap() * ratio_squared + T::one();
        let h10 = ratio_cubed - T::from(2.0).unwrap() * ratio_squared + ratio;
        let h01 = T::from(-2.0).unwrap() * ratio_cubed + T::from(3.0).unwrap() * ratio_squared;
        let h11 = ratio_cubed - ratio_squared;
        
        Ok(features_a.iter()
            .zip(features_b.iter())
            .map(|(&a, &b)| {
                let tangent_a = T::zero(); // Simplified: no tangent information
                let tangent_b = T::zero();
                h00 * a + h10 * tangent_a + h01 * b + h11 * tangent_b
            })
            .collect())
    }
    
    /// Sinc interpolation for audio signals
    fn sinc_interpolate(
        &self,
        features_a: &[T],
        features_b: &[T],
        ratio: T,
    ) -> Result<Vec<T>> {
        // Simplified sinc interpolation (windowed sinc)
        let sinc_value = if ratio == T::zero() {
            T::one()
        } else {
            let pi_ratio = T::from(std::f64::consts::PI).unwrap() * ratio;
            pi_ratio.sin() / pi_ratio
        };
        
        Ok(features_a.iter()
            .zip(features_b.iter())
            .map(|(&a, &b)| a * (T::one() - sinc_value) + b * sinc_value)
            .collect())
    }
    
    /// Gaussian kernel interpolation
    fn gaussian_interpolate(
        &self,
        features_a: &[T],
        features_b: &[T],
        ratio: T,
    ) -> Result<Vec<T>> {
        let sigma = T::from(0.5).unwrap();
        let gaussian_weight = (-(ratio - T::from(0.5).unwrap()).powi(2) / (T::from(2.0).unwrap() * sigma * sigma)).exp();
        
        Ok(features_a.iter()
            .zip(features_b.iter())
            .map(|(&a, &b)| a * (T::one() - gaussian_weight) + b * gaussian_weight)
            .collect())
    }
    
    /// Apply temporal smoothing to reduce noise
    fn apply_temporal_smoothing(&self, data: &[T]) -> Result<Vec<T>> {
        if data.len() < self.config.smoothing_window {
            return Ok(data.to_vec());
        }
        
        let window_size = self.config.smoothing_window;
        let half_window = window_size / 2;
        let mut smoothed = Vec::with_capacity(data.len());
        
        for i in 0..data.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(data.len());
            
            let window_sum = data[start..end].iter().fold(T::zero(), |acc, &x| acc + x);
            let window_avg = window_sum / T::from(end - start).unwrap();
            
            smoothed.push(window_avg);
        }
        
        Ok(smoothed)
    }
    
    /// Combine aligned features from all modalities
    fn combine_aligned_features(
        &self,
        aligned_modalities: &HashMap<ModalityType, Vec<T>>,
    ) -> Result<Vec<T>> {
        let mut combined = Vec::new();
        
        // Concatenate features in a consistent order
        let modality_order = [
            ModalityType::Vision,
            ModalityType::Audio,
            ModalityType::Text,
            ModalityType::Physiological,
        ];
        
        for modality in &modality_order {
            if let Some(features) = aligned_modalities.get(modality) {
                combined.extend_from_slice(features);
            }
        }
        
        Ok(combined)
    }
    
    /// Calculate dimension mapping for combined features
    fn calculate_dimension_map(
        &self,
        aligned_modalities: &HashMap<ModalityType, Vec<T>>,
    ) -> HashMap<ModalityType, (usize, usize)> {
        let mut dimension_map = HashMap::new();
        let mut current_offset = 0;
        
        let modality_order = [
            ModalityType::Vision,
            ModalityType::Audio,
            ModalityType::Text,
            ModalityType::Physiological,
        ];
        
        for modality in &modality_order {
            if let Some(features) = aligned_modalities.get(modality) {
                let start = current_offset;
                let end = current_offset + features.len();
                dimension_map.insert(*modality, (start, end));
                current_offset = end;
            }
        }
        
        dimension_map
    }
    
    /// Update drift compensation for a modality
    fn update_drift_compensation(
        &mut self,
        modality: ModalityType,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        // Implementation would track timing discrepancies and estimate drift
        // This is a simplified version
        
        if !self.drift_compensation.contains_key(&modality) {
            self.drift_compensation.insert(modality, DriftCompensator::new());
        }
        
        // Update drift compensator (implementation details omitted for brevity)
        Ok(())
    }
    
    /// Apply drift compensation to data
    fn apply_drift_compensation(
        &self,
        modality: ModalityType,
        data: Vec<TemporalDataPoint<T>>,
    ) -> Result<Vec<TemporalDataPoint<T>>> {
        if let Some(_compensator) = self.drift_compensation.get(&modality) {
            // Apply drift correction (simplified implementation)
            Ok(data)
        } else {
            Ok(data)
        }
    }
    
    /// Get target sampling rate for reference modality
    fn get_target_sampling_rate(&self, reference_modality: ModalityType) -> Result<T> {
        if let Some(buffer) = self.sync_buffers.get(&reference_modality) {
            Ok(buffer.sampling_rate.unwrap_or(T::from(30.0).unwrap())) // Default 30 Hz
        } else {
            Ok(T::from(30.0).unwrap())
        }
    }
    
    /// Calculate applied time offset
    fn calculate_applied_offset(
        &self,
        modality: ModalityType,
        temporal_window: &(DateTime<Utc>, DateTime<Utc>),
    ) -> Result<Duration> {
        // Implementation would calculate actual offset applied
        Ok(Duration::from_millis(0))
    }
    
    /// Estimate frame rate from aligned modalities
    fn estimate_frame_rate(&self, aligned_modalities: &HashMap<ModalityType, Vec<T>>) -> Option<f64> {
        // Implementation would estimate frame rate based on video modality
        if aligned_modalities.contains_key(&ModalityType::Vision) {
            Some(30.0) // Default video frame rate
        } else {
            None
        }
    }
    
    /// Estimate sample rate from aligned modalities
    fn estimate_sample_rate(&self, aligned_modalities: &HashMap<ModalityType, Vec<T>>) -> Option<u32> {
        // Implementation would estimate sample rate based on audio modality
        if aligned_modalities.contains_key(&ModalityType::Audio) {
            Some(16000) // Default audio sample rate
        } else {
            None
        }
    }
}

/// Statistics for individual modality interpolation
#[derive(Debug, Clone, Default)]
pub struct ModalityInterpolationStats<T: Float> {
    pub point_count: usize,
    pub error: T,
    pub resampling_ratio: T,
}

impl<T: Float> SyncBuffer<T> {
    fn new(modality: ModalityType) -> Self {
        Self {
            data: VecDeque::new(),
            modality,
            sampling_rate: None,
            last_sync: None,
            stats: BufferStats {
                avg_interval: T::zero(),
                interval_variance: T::zero(),
                count: 0,
                avg_quality: T::one(),
                last_update: Utc::now(),
            },
        }
    }
    
    fn add_data_point(&mut self, mut data_point: TemporalDataPoint<T>, max_size: usize) -> Result<()> {
        data_point.original_index = self.data.len();
        
        // Update statistics
        if let Some(last_point) = self.data.back() {
            let interval = (data_point.timestamp - last_point.timestamp).num_milliseconds() as f64 / 1000.0;
            let interval_t = T::from(interval).unwrap();
            
            // Update average interval and variance
            let new_count = self.stats.count + 1;
            let old_avg = self.stats.avg_interval;
            self.stats.avg_interval = (old_avg * T::from(self.stats.count).unwrap() + interval_t) / T::from(new_count).unwrap();
            
            // Update variance (simplified)
            let diff = interval_t - self.stats.avg_interval;
            self.stats.interval_variance = (self.stats.interval_variance * T::from(self.stats.count).unwrap() + diff * diff) / T::from(new_count).unwrap();
        }
        
        self.stats.count += 1;
        self.stats.avg_quality = (self.stats.avg_quality * T::from(self.stats.count - 1).unwrap() + data_point.quality) / T::from(self.stats.count).unwrap();
        self.stats.last_update = Utc::now();
        
        // Estimate sampling rate
        if self.data.len() > 1 {
            let rate = T::one() / self.stats.avg_interval;
            self.sampling_rate = Some(rate);
        }
        
        self.data.push_back(data_point);
        
        // Maintain buffer size
        while self.data.len() > max_size {
            self.data.pop_front();
        }
        
        Ok(())
    }
    
    fn get_data_in_window(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<TemporalDataPoint<T>>> {
        Ok(self.data.iter()
            .filter(|point| point.timestamp >= start && point.timestamp <= end)
            .cloned()
            .collect())
    }
    
    fn get_start_time(&self) -> Option<DateTime<Utc>> {
        self.data.front().map(|point| point.timestamp)
    }
    
    fn get_end_time(&self) -> Option<DateTime<Utc>> {
        self.data.back().map(|point| point.timestamp)
    }
}

impl<T: Float> DriftCompensator<T> {
    fn new() -> Self {
        Self {
            drift_rate: T::zero(),
            confidence: T::zero(),
            timing_history: VecDeque::new(),
            last_compensation: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporal_aligner_creation() {
        let config = AlignmentConfig::<f64>::default();
        let aligner = TemporalAligner::new(config).unwrap();
        assert_eq!(aligner.sync_buffers.len(), 0);
    }
    
    #[test]
    fn test_sync_buffer_operations() {
        let mut buffer = SyncBuffer::new(ModalityType::Vision);
        
        let data_point = TemporalDataPoint {
            timestamp: Utc::now(),
            features: vec![1.0, 2.0, 3.0],
            quality: 0.9,
            original_index: 0,
        };
        
        buffer.add_data_point(data_point, 100).unwrap();
        assert_eq!(buffer.data.len(), 1);
        assert_eq!(buffer.stats.count, 1);
    }
    
    #[test]
    fn test_linear_interpolation() {
        let config = AlignmentConfig::<f64>::default();
        let aligner = TemporalAligner::new(config).unwrap();
        
        let features_a = vec![1.0, 2.0, 3.0];
        let features_b = vec![3.0, 4.0, 5.0];
        let ratio = 0.5;
        
        let result = aligner.linear_interpolate(&features_a, &features_b, ratio).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }
}