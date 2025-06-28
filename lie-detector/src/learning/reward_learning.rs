//! Reward learning and signal processing
//!
//! This module implements reward learning mechanisms for training agents
//! through various feedback signals.

use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Result;
use crate::types::*;
use super::*;

/// Reward learning system for processing feedback signals
pub struct RewardLearningSystem<T: Float> {
    /// Configuration
    pub config: RewardConfig<T>,
    /// Reward models by source
    pub reward_models: HashMap<RewardSource, RewardModel<T>>,
    /// Signal processors
    pub processors: Vec<Box<dyn SignalProcessor<T>>>,
    /// Learning history
    pub learning_history: Vec<RewardLearningEvent<T>>,
}

/// Configuration for reward learning
#[derive(Debug, Clone)]
pub struct RewardConfig<T: Float> {
    /// Learning rate for reward updates
    pub learning_rate: T,
    /// Discount factor for future rewards
    pub discount_factor: T,
    /// Sources to learn from
    pub enabled_sources: Vec<RewardSource>,
    /// Signal processing parameters
    pub signal_params: SignalParameters<T>,
}

/// Parameters for signal processing
#[derive(Debug, Clone)]
pub struct SignalParameters<T: Float> {
    /// Noise filtering threshold
    pub noise_threshold: T,
    /// Signal smoothing factor
    pub smoothing_factor: T,
    /// Outlier detection sensitivity
    pub outlier_sensitivity: T,
    /// Temporal window size
    pub temporal_window: usize,
}

/// Reward model for specific source
#[derive(Debug, Clone)]
pub struct RewardModel<T: Float> {
    /// Model identifier
    pub id: Uuid,
    /// Source type
    pub source: RewardSource,
    /// Model parameters
    pub parameters: HashMap<String, T>,
    /// Model performance metrics
    pub metrics: RewardModelMetrics<T>,
    /// Update count
    pub updates: usize,
}

/// Metrics for reward models
#[derive(Debug, Clone)]
pub struct RewardModelMetrics<T: Float> {
    /// Prediction accuracy
    pub accuracy: T,
    /// Signal correlation
    pub correlation: T,
    /// Stability measure
    pub stability: T,
    /// Calibration quality
    pub calibration: T,
}

/// Trait for signal processors
pub trait SignalProcessor<T: Float>: Send + Sync {
    /// Process raw signal
    fn process_signal(&self, signal: &RewardSignal<T>) -> Result<ProcessedSignal<T>>;
    
    /// Get processor name
    fn name(&self) -> &str;
    
    /// Update processor parameters
    fn update_parameters(&mut self, params: HashMap<String, T>);
}

/// Processed reward signal
#[derive(Debug, Clone)]
pub struct ProcessedSignal<T: Float> {
    /// Original signal
    pub original: RewardSignal<T>,
    /// Processed value
    pub processed_value: T,
    /// Signal quality score
    pub quality: T,
    /// Confidence in processing
    pub confidence: T,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Event in reward learning history
#[derive(Debug, Clone)]
pub struct RewardLearningEvent<T: Float> {
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Signal that was processed
    pub signal: RewardSignal<T>,
    /// Model update
    pub model_update: ModelUpdate<T>,
    /// Learning outcome
    pub outcome: LearningOutcome<T>,
}

/// Model update information
#[derive(Debug, Clone)]
pub struct ModelUpdate<T: Float> {
    /// Updated model ID
    pub model_id: Uuid,
    /// Parameters changed
    pub parameter_changes: HashMap<String, T>,
    /// Update magnitude
    pub magnitude: T,
    /// Update type
    pub update_type: UpdateType,
}

/// Types of model updates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateType {
    /// Gradient-based update
    Gradient,
    /// Bayesian update
    Bayesian,
    /// Evolutionary update
    Evolutionary,
    /// Rule-based update
    RuleBased,
}

/// Outcome of learning update
#[derive(Debug, Clone)]
pub struct LearningOutcome<T: Float> {
    /// Improvement in model performance
    pub improvement: T,
    /// New model accuracy
    pub new_accuracy: T,
    /// Convergence indicator
    pub convergence: T,
    /// Additional metrics
    pub metrics: HashMap<String, T>,
}

/// Noise filter processor
pub struct NoiseFilter<T: Float> {
    /// Filtering parameters
    pub params: NoiseFilterParams<T>,
    /// Signal history for filtering
    pub history: std::collections::VecDeque<RewardSignal<T>>,
}

/// Parameters for noise filtering
#[derive(Debug, Clone)]
pub struct NoiseFilterParams<T: Float> {
    /// Filter window size
    pub window_size: usize,
    /// Noise threshold
    pub threshold: T,
    /// Filter type
    pub filter_type: FilterType,
}

/// Types of noise filters
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterType {
    /// Moving average filter
    MovingAverage,
    /// Median filter
    Median,
    /// Kalman filter
    Kalman,
    /// Outlier removal
    OutlierRemoval,
}

impl<T: Float> NoiseFilter<T> {
    pub fn new(params: NoiseFilterParams<T>) -> Self {
        Self {
            params,
            history: std::collections::VecDeque::with_capacity(params.window_size),
        }
    }
}

impl<T: Float> SignalProcessor<T> for NoiseFilter<T> {
    fn process_signal(&self, signal: &RewardSignal<T>) -> Result<ProcessedSignal<T>> {
        let processed_value = match self.params.filter_type {
            FilterType::MovingAverage => self.apply_moving_average(signal),
            FilterType::Median => self.apply_median_filter(signal),
            FilterType::OutlierRemoval => self.apply_outlier_removal(signal),
            _ => signal.value, // Fallback
        };
        
        // Calculate quality based on filtering
        let quality = if (processed_value - signal.value).abs() < self.params.threshold {
            T::from(0.9).unwrap()
        } else {
            T::from(0.6).unwrap()
        };
        
        Ok(ProcessedSignal {
            original: signal.clone(),
            processed_value,
            quality,
            confidence: T::from(0.85).unwrap(),
            metadata: HashMap::from([
                ("filter_type".to_string(), format!("{:?}", self.params.filter_type)),
                ("window_size".to_string(), self.params.window_size.to_string()),
            ]),
        })
    }
    
    fn name(&self) -> &str {
        "noise_filter"
    }
    
    fn update_parameters(&mut self, params: HashMap<String, T>) {
        if let Some(&threshold) = params.get("threshold") {
            self.params.threshold = threshold;
        }
        if let Some(&window_size) = params.get("window_size") {
            let new_size = window_size.to_usize().unwrap_or(self.params.window_size);
            self.params.window_size = new_size;
            // Resize history buffer
            while self.history.len() > new_size {
                self.history.pop_front();
            }
        }
    }
}

impl<T: Float> NoiseFilter<T> {
    fn apply_moving_average(&self, signal: &RewardSignal<T>) -> T {
        if self.history.is_empty() {
            return signal.value;
        }
        
        let sum: f64 = self.history.iter()
            .map(|s| s.value.to_f64().unwrap_or(0.0))
            .sum::<f64>() + signal.value.to_f64().unwrap_or(0.0);
        
        let count = self.history.len() + 1;
        T::from(sum / count as f64).unwrap_or(signal.value)
    }
    
    fn apply_median_filter(&self, signal: &RewardSignal<T>) -> T {
        let mut values: Vec<f64> = self.history.iter()
            .map(|s| s.value.to_f64().unwrap_or(0.0))
            .collect();
        values.push(signal.value.to_f64().unwrap_or(0.0));
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let median = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };
        
        T::from(median).unwrap_or(signal.value)
    }
    
    fn apply_outlier_removal(&self, signal: &RewardSignal<T>) -> T {
        if self.history.len() < 3 {
            return signal.value;
        }
        
        let values: Vec<f64> = self.history.iter()
            .map(|s| s.value.to_f64().unwrap_or(0.0))
            .collect();
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        let signal_value = signal.value.to_f64().unwrap_or(0.0);
        let z_score = (signal_value - mean).abs() / std_dev;
        
        // If outlier (z-score > 2), use mean instead
        if z_score > 2.0 {
            T::from(mean).unwrap_or(signal.value)
        } else {
            signal.value
        }
    }
}

/// Signal smoothing processor
pub struct SignalSmoother<T: Float> {
    /// Smoothing parameters
    pub params: SmoothingParams<T>,
    /// Previous smoothed value
    pub previous_value: Option<T>,
}

/// Parameters for signal smoothing
#[derive(Debug, Clone)]
pub struct SmoothingParams<T: Float> {
    /// Smoothing factor (0.0 to 1.0)
    pub alpha: T,
    /// Trend smoothing factor
    pub beta: T,
    /// Adaptive smoothing
    pub adaptive: bool,
}

impl<T: Float> SignalSmoother<T> {
    pub fn new(params: SmoothingParams<T>) -> Self {
        Self {
            params,
            previous_value: None,
        }
    }
}

impl<T: Float> SignalProcessor<T> for SignalSmoother<T> {
    fn process_signal(&self, signal: &RewardSignal<T>) -> Result<ProcessedSignal<T>> {
        let processed_value = if let Some(prev) = self.previous_value {
            // Exponential smoothing
            let alpha = if self.params.adaptive {
                // Adapt alpha based on signal confidence
                self.params.alpha * signal.confidence
            } else {
                self.params.alpha
            };
            
            prev * (T::one() - alpha) + signal.value * alpha
        } else {
            signal.value
        };
        
        Ok(ProcessedSignal {
            original: signal.clone(),
            processed_value,
            quality: T::from(0.9).unwrap(),
            confidence: T::from(0.9).unwrap(),
            metadata: HashMap::from([
                ("smoother".to_string(), "exponential".to_string()),
                ("alpha".to_string(), self.params.alpha.to_f64().unwrap().to_string()),
            ]),
        })
    }
    
    fn name(&self) -> &str {
        "signal_smoother"
    }
    
    fn update_parameters(&mut self, params: HashMap<String, T>) {
        if let Some(&alpha) = params.get("alpha") {
            self.params.alpha = alpha;
        }
        if let Some(&beta) = params.get("beta") {
            self.params.beta = beta;
        }
    }
}

impl<T: Float> RewardLearningSystem<T> {
    /// Create new reward learning system
    pub fn new(config: RewardConfig<T>) -> Self {
        let mut processors: Vec<Box<dyn SignalProcessor<T>>> = Vec::new();
        
        // Add default processors
        processors.push(Box::new(NoiseFilter::new(NoiseFilterParams {
            window_size: 5,
            threshold: config.signal_params.noise_threshold,
            filter_type: FilterType::MovingAverage,
        })));
        
        processors.push(Box::new(SignalSmoother::new(SmoothingParams {
            alpha: config.signal_params.smoothing_factor,
            beta: T::from(0.3).unwrap(),
            adaptive: true,
        })));
        
        // Initialize reward models for each enabled source
        let mut reward_models = HashMap::new();
        for source in &config.enabled_sources {
            let model = RewardModel {
                id: Uuid::new_v4(),
                source: source.clone(),
                parameters: HashMap::from([
                    ("weight".to_string(), T::from(1.0).unwrap()),
                    ("bias".to_string(), T::zero()),
                ]),
                metrics: RewardModelMetrics {
                    accuracy: T::from(0.5).unwrap(),
                    correlation: T::from(0.5).unwrap(),
                    stability: T::from(0.8).unwrap(),
                    calibration: T::from(0.7).unwrap(),
                },
                updates: 0,
            };
            reward_models.insert(source.clone(), model);
        }
        
        Self {
            config,
            reward_models,
            processors,
            learning_history: Vec::new(),
        }
    }
    
    /// Process and learn from reward signal
    pub fn learn_from_signal(&mut self, signal: RewardSignal<T>) -> Result<LearningOutcome<T>> {
        // Process signal through all processors
        let mut processed_signal = ProcessedSignal {
            original: signal.clone(),
            processed_value: signal.value,
            quality: T::from(1.0).unwrap(),
            confidence: signal.confidence,
            metadata: HashMap::new(),
        };
        
        for processor in &self.processors {
            let new_processed = processor.process_signal(&processed_signal.original)?;
            processed_signal = new_processed;
        }
        
        // Update corresponding reward model
        let outcome = if let Some(model) = self.reward_models.get_mut(&signal.source) {
            self.update_model(model, &processed_signal)?
        } else {
            // Create new model for unknown source
            let mut new_model = RewardModel {
                id: Uuid::new_v4(),
                source: signal.source.clone(),
                parameters: HashMap::new(),
                metrics: RewardModelMetrics {
                    accuracy: T::from(0.5).unwrap(),
                    correlation: T::from(0.5).unwrap(),
                    stability: T::from(0.8).unwrap(),
                    calibration: T::from(0.7).unwrap(),
                },
                updates: 0,
            };
            
            let outcome = self.update_model(&mut new_model, &processed_signal)?;
            self.reward_models.insert(signal.source, new_model);
            outcome
        };
        
        // Record learning event
        let event = RewardLearningEvent {
            timestamp: chrono::Utc::now(),
            signal,
            model_update: ModelUpdate {
                model_id: Uuid::new_v4(), // Would use actual model ID
                parameter_changes: HashMap::new(),
                magnitude: T::from(0.1).unwrap(),
                update_type: UpdateType::Gradient,
            },
            outcome: outcome.clone(),
        };
        
        self.learning_history.push(event);
        
        Ok(outcome)
    }
    
    /// Update reward model based on processed signal
    fn update_model(&mut self, model: &mut RewardModel<T>, signal: &ProcessedSignal<T>) -> Result<LearningOutcome<T>> {
        let old_accuracy = model.metrics.accuracy;
        
        // Simple model update (in practice would be more sophisticated)
        let learning_rate = self.config.learning_rate;
        
        // Update model weight based on signal quality
        if let Some(weight) = model.parameters.get_mut("weight") {
            let update = signal.quality * learning_rate;
            *weight = *weight + update;
        }
        
        // Update metrics
        model.updates += 1;
        
        // Simulate accuracy improvement (would be calculated from actual performance)
        let accuracy_improvement = signal.quality * T::from(0.1).unwrap();
        model.metrics.accuracy = (model.metrics.accuracy + accuracy_improvement).min(T::one());
        
        // Update correlation based on signal confidence
        let correlation_update = (signal.confidence - T::from(0.5).unwrap()) * T::from(0.1).unwrap();
        model.metrics.correlation = (model.metrics.correlation + correlation_update)
            .max(T::zero()).min(T::one());
        
        Ok(LearningOutcome {
            improvement: model.metrics.accuracy - old_accuracy,
            new_accuracy: model.metrics.accuracy,
            convergence: if model.updates > 100 { T::from(0.9).unwrap() } else { T::from(0.5).unwrap() },
            metrics: HashMap::from([
                ("correlation".to_string(), model.metrics.correlation),
                ("stability".to_string(), model.metrics.stability),
            ]),
        })
    }
    
    /// Get current model for a reward source
    pub fn get_model(&self, source: &RewardSource) -> Option<&RewardModel<T>> {
        self.reward_models.get(source)
    }
    
    /// Get learning statistics
    pub fn get_learning_stats(&self) -> RewardLearningStats<T> {
        let total_updates = self.reward_models.values().map(|m| m.updates).sum();
        let avg_accuracy = if self.reward_models.is_empty() {
            T::zero()
        } else {
            let sum: f64 = self.reward_models.values()
                .map(|m| m.metrics.accuracy.to_f64().unwrap_or(0.0))
                .sum();
            T::from(sum / self.reward_models.len() as f64).unwrap()
        };
        
        RewardLearningStats {
            total_signals_processed: self.learning_history.len(),
            total_model_updates: total_updates,
            average_model_accuracy: avg_accuracy,
            active_sources: self.reward_models.len(),
            convergence_rate: T::from(0.7).unwrap(), // Would be calculated properly
        }
    }
}

/// Statistics for reward learning
#[derive(Debug, Clone)]
pub struct RewardLearningStats<T: Float> {
    /// Total signals processed
    pub total_signals_processed: usize,
    /// Total model updates
    pub total_model_updates: usize,
    /// Average accuracy across models
    pub average_model_accuracy: T,
    /// Number of active reward sources
    pub active_sources: usize,
    /// Overall convergence rate
    pub convergence_rate: T,
}

impl<T: Float> Default for RewardConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.01).unwrap(),
            discount_factor: T::from(0.95).unwrap(),
            enabled_sources: vec![
                RewardSource::GroundTruth,
                RewardSource::Human,
                RewardSource::SelfEvaluation,
            ],
            signal_params: SignalParameters {
                noise_threshold: T::from(0.1).unwrap(),
                smoothing_factor: T::from(0.3).unwrap(),
                outlier_sensitivity: T::from(2.0).unwrap(),
                temporal_window: 10,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_learning_system() {
        let config = RewardConfig::default();
        let mut system: RewardLearningSystem<f32> = RewardLearningSystem::new(config);
        
        let signal = RewardSignal {
            value: 0.8,
            components: HashMap::new(),
            source: RewardSource::Human,
            confidence: 0.9,
        };
        
        let result = system.learn_from_signal(signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_noise_filter() {
        let params = NoiseFilterParams {
            window_size: 5,
            threshold: 0.1,
            filter_type: FilterType::MovingAverage,
        };
        let filter: NoiseFilter<f32> = NoiseFilter::new(params);
        
        let signal = RewardSignal {
            value: 0.7,
            components: HashMap::new(),
            source: RewardSource::GroundTruth,
            confidence: 0.8,
        };
        
        let result = filter.process_signal(&signal);
        assert!(result.is_ok());
    }
}