//! Action engine for ReAct agent action selection and validation
//!
//! This module implements the action selection component of the ReAct framework,
//! responsible for choosing appropriate actions based on observations and thoughts,
//! validating action feasibility, and maintaining action history.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::Utc;
use uuid::Uuid;
use num_traits::Float;

use crate::error::{Result, VeritasError};
use crate::types::*;

/// Action engine for selecting and validating agent actions
pub struct ActionEngine<T: Float> {
    config: ActionConfig<T>,
    /// Action selection strategies
    strategies: HashMap<String, Box<dyn ActionStrategy<T>>>,
    /// Action validation rules
    validators: Vec<Box<dyn ActionValidator<T>>>,
    /// Action execution history
    history: ActionHistory<T>,
    /// Engine statistics
    stats: ActionStats,
}

/// Configuration for action engine
#[derive(Debug, Clone)]
pub struct ActionConfig<T: Float> {
    /// Default action selection strategy
    pub default_strategy: String,
    /// Maximum number of actions to consider
    pub max_actions: usize,
    /// Minimum confidence threshold for action execution
    pub min_confidence: T,
    /// Action timeout in milliseconds
    pub action_timeout_ms: u64,
    /// Enable action validation
    pub enable_validation: bool,
    /// Maximum action history size
    pub max_history_size: usize,
    /// Strategy-specific parameters
    pub strategy_params: HashMap<String, HashMap<String, String>>,
}

impl<T: Float> Default for ActionConfig<T> {
    fn default() -> Self {
        Self {
            default_strategy: "multimodal_weighted".to_string(),
            max_actions: 10,
            min_confidence: T::from(0.3).unwrap(),
            action_timeout_ms: 5000,
            enable_validation: true,
            max_history_size: 1000,
            strategy_params: HashMap::new(),
        }
    }
}

/// Action selection strategy trait
pub trait ActionStrategy<T: Float>: Send + Sync {
    /// Generate candidate actions based on observations and thoughts
    fn generate_actions(
        &self,
        observations: &Observations<T>,
        thoughts: &Thoughts,
    ) -> Result<Vec<ActionCandidate<T>>>;

    /// Select the best action from candidates
    fn select_action(
        &self,
        candidates: &[ActionCandidate<T>],
        context: &ActionContext<T>,
    ) -> Result<Action<T>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy parameters
    fn parameters(&self) -> HashMap<String, String>;
}

/// Action validation trait
pub trait ActionValidator<T: Float>: Send + Sync {
    /// Validate if an action is feasible and safe
    fn validate(
        &self,
        action: &Action<T>,
        context: &ActionContext<T>,
    ) -> Result<ValidationResult>;

    /// Get validator name
    fn name(&self) -> &str;
}

/// Action candidate with scoring
#[derive(Debug, Clone)]
pub struct ActionCandidate<T: Float> {
    /// The proposed action
    pub action: Action<T>,
    /// Selection score (higher is better)
    pub score: T,
    /// Reasoning for this action
    pub reasoning: String,
    /// Expected utility
    pub expected_utility: T,
    /// Risk assessment
    pub risk_score: T,
}

/// Context for action selection and validation
#[derive(Debug, Clone)]
pub struct ActionContext<T: Float> {
    /// Current system state
    pub system_state: SystemState,
    /// Available resources
    pub resources: ResourceState,
    /// Previous action outcomes
    pub recent_outcomes: Vec<ActionOutcome<T>>,
    /// Time constraints
    pub time_budget: Duration,
    /// Safety constraints
    pub safety_level: SafetyLevel,
}

/// Current system state
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Processing load (0.0 to 1.0)
    pub load: f64,
    /// Available memory
    pub available_memory_mb: usize,
    /// Active modalities
    pub active_modalities: Vec<ModalityType>,
    /// System health status
    pub health: HealthStatus,
}

/// Available resource state
#[derive(Debug, Clone)]
pub struct ResourceState {
    /// CPU resources available
    pub cpu_available: f64,
    /// Memory available in MB
    pub memory_available_mb: usize,
    /// GPU resources if available
    pub gpu_available: Option<f64>,
    /// Network bandwidth available
    pub bandwidth_mbps: f64,
}

/// Safety level for action execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    /// Allow all safe actions
    Normal,
    /// Require additional validation
    Cautious,
    /// Only allow read-only actions
    Conservative,
    /// Emergency mode - minimal actions only
    Emergency,
}

/// Result of action validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the action is valid
    pub is_valid: bool,
    /// Validation confidence
    pub confidence: f64,
    /// Reasons if invalid
    pub issues: Vec<String>,
    /// Suggested modifications
    pub suggestions: Vec<String>,
}

/// Outcome of an executed action
#[derive(Debug, Clone)]
pub struct ActionOutcome<T: Float> {
    /// The action that was executed
    pub action: Action<T>,
    /// Whether execution was successful
    pub success: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Outcome description
    pub description: String,
    /// Measured utility
    pub actual_utility: Option<T>,
    /// Timestamp
    pub timestamp: chrono::DateTime<Utc>,
}

/// Action execution history
#[derive(Debug, Clone)]
pub struct ActionHistory<T: Float> {
    /// Historical action outcomes
    pub outcomes: Vec<ActionOutcome<T>>,
    /// Success rate by action type
    pub success_rates: HashMap<ActionType, f64>,
    /// Average execution times
    pub avg_execution_times: HashMap<ActionType, Duration>,
    /// Total actions executed
    pub total_actions: usize,
}

impl<T: Float> ActionHistory<T> {
    /// Create new empty history
    pub fn new() -> Self {
        Self {
            outcomes: Vec::new(),
            success_rates: HashMap::new(),
            avg_execution_times: HashMap::new(),
            total_actions: 0,
        }
    }

    /// Add an action outcome to history
    pub fn add_outcome(&mut self, outcome: ActionOutcome<T>) {
        self.outcomes.push(outcome.clone());
        self.total_actions += 1;

        // Update success rates
        let action_type = outcome.action.action_type.clone();
        let current_rate = self.success_rates.get(&action_type).unwrap_or(&0.0);
        let count = self.outcomes.iter()
            .filter(|o| o.action.action_type == action_type)
            .count();
        
        let new_rate = if outcome.success {
            (current_rate * (count - 1) as f64 + 1.0) / count as f64
        } else {
            (current_rate * (count - 1) as f64) / count as f64
        };
        
        self.success_rates.insert(action_type.clone(), new_rate);

        // Update average execution times
        let current_avg = self.avg_execution_times.get(&action_type)
            .unwrap_or(&Duration::from_millis(0));
        let new_avg = Duration::from_nanos(
            (current_avg.as_nanos() as u64 * (count - 1) as u64 + 
             outcome.execution_time.as_nanos() as u64) / count as u64
        );
        self.avg_execution_times.insert(action_type, new_avg);

        // Maintain history size limit
        if self.outcomes.len() > 1000 { // Default limit
            self.outcomes.remove(0);
        }
    }

    /// Get success rate for an action type
    pub fn get_success_rate(&self, action_type: &ActionType) -> f64 {
        self.success_rates.get(action_type).unwrap_or(&0.0).clone()
    }

    /// Get recent outcomes for analysis
    pub fn get_recent_outcomes(&self, limit: usize) -> &[ActionOutcome<T>] {
        let start = if self.outcomes.len() > limit {
            self.outcomes.len() - limit
        } else {
            0
        };
        &self.outcomes[start..]
    }
}

/// Action engine statistics
#[derive(Debug, Clone, Default)]
pub struct ActionStats {
    /// Total actions selected
    pub actions_selected: usize,
    /// Average selection time
    pub avg_selection_time_ms: f64,
    /// Strategy usage counts
    pub strategy_usage: HashMap<String, usize>,
    /// Validation success rate
    pub validation_success_rate: f64,
    /// Action type distribution
    pub action_type_distribution: HashMap<ActionType, usize>,
}

impl<T: Float> ActionEngine<T> {
    /// Create a new action engine
    pub fn new(config: ActionConfig<T>) -> Result<Self> {
        let mut engine = Self {
            config,
            strategies: HashMap::new(),
            validators: Vec::new(),
            history: ActionHistory::new(),
            stats: ActionStats::default(),
        };

        // Initialize default strategies
        engine.add_strategy(Box::new(MultimodalWeightedStrategy::new()))?;
        engine.add_strategy(Box::new(ConfidenceBasedStrategy::new()))?;
        engine.add_strategy(Box::new(UtilityMaximizingStrategy::new()))?;

        // Initialize default validators
        engine.add_validator(Box::new(SafetyValidator::new()))?;
        engine.add_validator(Box::new(ResourceValidator::new()))?;
        engine.add_validator(Box::new(ConstraintValidator::new()))?;

        Ok(engine)
    }

    /// Add an action selection strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn ActionStrategy<T>>) -> Result<()> {
        let name = strategy.name().to_string();
        self.strategies.insert(name, strategy);
        Ok(())
    }

    /// Add an action validator
    pub fn add_validator(&mut self, validator: Box<dyn ActionValidator<T>>) -> Result<()> {
        self.validators.push(validator);
        Ok(())
    }

    /// Select action based on observations and thoughts
    pub fn select_action(
        &mut self,
        observations: &Observations<T>,
        thoughts: &Thoughts,
    ) -> Result<Action<T>> {
        let start_time = Instant::now();

        // Build action context
        let context = self.build_action_context(observations)?;

        // Get the configured strategy
        let strategy_name = &self.config.default_strategy;
        let strategy = self.strategies.get(strategy_name)
            .ok_or_else(|| VeritasError::action(
                format!("Strategy '{}' not found", strategy_name)
            ))?;

        // Generate action candidates
        let candidates = strategy.generate_actions(observations, thoughts)?;
        
        if candidates.is_empty() {
            return Err(VeritasError::action("No valid action candidates generated"));
        }

        // Select best action using strategy
        let mut selected_action = strategy.select_action(&candidates, &context)?;

        // Validate the selected action if enabled
        if self.config.enable_validation {
            self.validate_action(&mut selected_action, &context)?;
        }

        // Update statistics
        let selection_time = start_time.elapsed();
        self.update_stats(strategy_name, &selected_action, selection_time);

        Ok(selected_action)
    }

    /// Validate an action using all configured validators
    fn validate_action(
        &self,
        action: &mut Action<T>,
        context: &ActionContext<T>,
    ) -> Result<()> {
        for validator in &self.validators {
            let result = validator.validate(action, context)?;
            
            if !result.is_valid {
                return Err(VeritasError::action(
                    format!(
                        "Action validation failed ({}): {}",
                        validator.name(),
                        result.issues.join("; ")
                    )
                ));
            }

            // Apply any suggested modifications
            if !result.suggestions.is_empty() {
                // In a real implementation, we would apply suggestions here
                // For now, we'll just log them in the explanation
                action.explanation = format!(
                    "{}\nValidator suggestions from {}: {}",
                    action.explanation,
                    validator.name(),
                    result.suggestions.join("; ")
                );
            }
        }

        Ok(())
    }

    /// Build action context from current state
    fn build_action_context(&self, observations: &Observations<T>) -> Result<ActionContext<T>> {
        // In a real implementation, this would gather actual system state
        // For now, we'll provide reasonable defaults
        
        let system_state = SystemState {
            load: 0.3, // Assume moderate load
            available_memory_mb: 2048, // Assume 2GB available
            active_modalities: vec![ModalityType::Text, ModalityType::Vision], // Based on observations
            health: HealthStatus::Healthy,
        };

        let resources = ResourceState {
            cpu_available: 0.7,
            memory_available_mb: 2048,
            gpu_available: Some(0.8),
            bandwidth_mbps: 100.0,
        };

        let recent_outcomes = self.history.get_recent_outcomes(10).to_vec();

        Ok(ActionContext {
            system_state,
            resources,
            recent_outcomes,
            time_budget: Duration::from_millis(self.config.action_timeout_ms),
            safety_level: SafetyLevel::Normal,
        })
    }

    /// Update engine statistics
    fn update_stats(
        &mut self,
        strategy_name: &str,
        action: &Action<T>,
        selection_time: Duration,
    ) {
        self.stats.actions_selected += 1;

        // Update average selection time
        let total_time = self.stats.avg_selection_time_ms * (self.stats.actions_selected - 1) as f64;
        let new_time = selection_time.as_millis() as f64;
        self.stats.avg_selection_time_ms = (total_time + new_time) / self.stats.actions_selected as f64;

        // Update strategy usage
        *self.stats.strategy_usage.entry(strategy_name.to_string()).or_insert(0) += 1;

        // Update action type distribution
        *self.stats.action_type_distribution.entry(action.action_type.clone()).or_insert(0) += 1;
    }

    /// Record action outcome for learning
    pub fn record_outcome(&mut self, outcome: ActionOutcome<T>) {
        self.history.add_outcome(outcome);
    }

    /// Update engine configuration
    pub fn update_config(&mut self, config: ActionConfig<T>) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> &ActionStats {
        &self.stats
    }

    /// Get action history
    pub fn get_history(&self) -> &ActionHistory<T> {
        &self.history
    }
}

// ================================================================================================
// DEFAULT STRATEGY IMPLEMENTATIONS
// ================================================================================================

/// Multimodal weighted action selection strategy
pub struct MultimodalWeightedStrategy<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> MultimodalWeightedStrategy<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> ActionStrategy<T> for MultimodalWeightedStrategy<T> {
    fn generate_actions(
        &self,
        observations: &Observations<T>,
        thoughts: &Thoughts,
    ) -> Result<Vec<ActionCandidate<T>>> {
        let mut candidates = Vec::new();

        // Generate decision action if we have enough information
        if !thoughts.hypotheses.is_empty() {
            let decision_action = Action {
                id: Uuid::new_v4(),
                action_type: ActionType::MakeDecision,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("hypothesis".to_string(), 
                        serde_json::Value::String(thoughts.hypotheses[0].clone()));
                    params
                },
                expected_outcome: "Generate deception detection decision".to_string(),
                confidence: T::from(thoughts.confidence).unwrap(),
                explanation: format!("Decision based on {} hypotheses", thoughts.hypotheses.len()),
                timestamp: Utc::now(),
                decision: Some(Decision::Uncertain), // Will be determined later
            };

            candidates.push(ActionCandidate {
                action: decision_action,
                score: T::from(thoughts.confidence * 0.9).unwrap(),
                reasoning: "Primary decision action based on available evidence".to_string(),
                expected_utility: T::from(0.8).unwrap(),
                risk_score: T::from(0.2).unwrap(),
            });
        }

        // Generate analysis actions for each available modality
        for modality in &[ModalityType::Text, ModalityType::Vision, ModalityType::Audio, ModalityType::Physiological] {
            if observations.has_modality(*modality) {
                let analysis_action = Action {
                    id: Uuid::new_v4(),
                    action_type: ActionType::AnalyzeModality,
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("modality".to_string(), 
                            serde_json::Value::String(format!("{:?}", modality)));
                        params
                    },
                    expected_outcome: format!("Enhanced analysis of {:?} modality", modality),
                    confidence: T::from(0.7).unwrap(),
                    explanation: format!("Deep analysis of {:?} modality data", modality),
                    timestamp: Utc::now(),
                    decision: None,
                };

                candidates.push(ActionCandidate {
                    action: analysis_action,
                    score: T::from(0.6).unwrap(),
                    reasoning: format!("Additional {:?} analysis could provide more insights", modality),
                    expected_utility: T::from(0.5).unwrap(),
                    risk_score: T::from(0.1).unwrap(),
                });
            }
        }

        // Generate request for more data if confidence is low
        if thoughts.confidence < 0.7 {
            let request_action = Action {
                id: Uuid::new_v4(),
                action_type: ActionType::RequestMoreData,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("reason".to_string(), 
                        serde_json::Value::String("Low confidence in current analysis".to_string()));
                    params
                },
                expected_outcome: "Additional data to improve analysis confidence".to_string(),
                confidence: T::from(0.8).unwrap(),
                explanation: "Request additional data due to low confidence".to_string(),
                timestamp: Utc::now(),
                decision: None,
            };

            candidates.push(ActionCandidate {
                action: request_action,
                score: T::from(1.0 - thoughts.confidence).unwrap(),
                reasoning: "Low confidence suggests need for more data".to_string(),
                expected_utility: T::from(0.7).unwrap(),
                risk_score: T::from(0.05).unwrap(),
            });
        }

        Ok(candidates)
    }

    fn select_action(
        &self,
        candidates: &[ActionCandidate<T>],
        _context: &ActionContext<T>,
    ) -> Result<Action<T>> {
        // Select candidate with highest score
        let best_candidate = candidates.iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| VeritasError::action("No candidates available for selection"))?;

        Ok(best_candidate.action.clone())
    }

    fn name(&self) -> &str {
        "multimodal_weighted"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), "weighted_selection".to_string());
        params.insert("modality_weights".to_string(), "dynamic".to_string());
        params
    }
}

/// Confidence-based action selection strategy
pub struct ConfidenceBasedStrategy<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ConfidenceBasedStrategy<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> ActionStrategy<T> for ConfidenceBasedStrategy<T> {
    fn generate_actions(
        &self,
        observations: &Observations<T>,
        thoughts: &Thoughts,
    ) -> Result<Vec<ActionCandidate<T>>> {
        // Simplified implementation - delegate to multimodal strategy
        let multimodal = MultimodalWeightedStrategy::new();
        multimodal.generate_actions(observations, thoughts)
    }

    fn select_action(
        &self,
        candidates: &[ActionCandidate<T>],
        _context: &ActionContext<T>,
    ) -> Result<Action<T>> {
        // Select based on action confidence rather than score
        let best_candidate = candidates.iter()
            .max_by(|a, b| a.action.confidence.partial_cmp(&b.action.confidence)
                .unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| VeritasError::action("No candidates available for selection"))?;

        Ok(best_candidate.action.clone())
    }

    fn name(&self) -> &str {
        "confidence_based"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), "confidence_selection".to_string());
        params
    }
}

/// Utility maximizing action selection strategy
pub struct UtilityMaximizingStrategy<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> UtilityMaximizingStrategy<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> ActionStrategy<T> for UtilityMaximizingStrategy<T> {
    fn generate_actions(
        &self,
        observations: &Observations<T>,
        thoughts: &Thoughts,
    ) -> Result<Vec<ActionCandidate<T>>> {
        // Simplified implementation - delegate to multimodal strategy
        let multimodal = MultimodalWeightedStrategy::new();
        multimodal.generate_actions(observations, thoughts)
    }

    fn select_action(
        &self,
        candidates: &[ActionCandidate<T>],
        _context: &ActionContext<T>,
    ) -> Result<Action<T>> {
        // Select based on expected utility adjusted for risk
        let best_candidate = candidates.iter()
            .max_by(|a, b| {
                let utility_a = a.expected_utility.to_f64().unwrap_or(0.0) - 
                               a.risk_score.to_f64().unwrap_or(0.0) * 0.5;
                let utility_b = b.expected_utility.to_f64().unwrap_or(0.0) - 
                               b.risk_score.to_f64().unwrap_or(0.0) * 0.5;
                utility_a.partial_cmp(&utility_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| VeritasError::action("No candidates available for selection"))?;

        Ok(best_candidate.action.clone())
    }

    fn name(&self) -> &str {
        "utility_maximizing"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), "utility_optimization".to_string());
        params.insert("risk_weight".to_string(), "0.5".to_string());
        params
    }
}

// ================================================================================================
// DEFAULT VALIDATOR IMPLEMENTATIONS  
// ================================================================================================

/// Safety validator for actions
pub struct SafetyValidator<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> SafetyValidator<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> ActionValidator<T> for SafetyValidator<T> {
    fn validate(
        &self,
        action: &Action<T>,
        context: &ActionContext<T>,
    ) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check safety level constraints
        match context.safety_level {
            SafetyLevel::Emergency => {
                if !matches!(action.action_type, ActionType::MakeDecision) {
                    issues.push("Only decision actions allowed in emergency mode".to_string());
                }
            },
            SafetyLevel::Conservative => {
                if matches!(action.action_type, ActionType::UpdateModel | ActionType::RequestMoreData) {
                    issues.push("Modifying actions not allowed in conservative mode".to_string());
                }
            },
            SafetyLevel::Cautious => {
                if action.confidence.to_f64().unwrap_or(0.0) < 0.5 {
                    suggestions.push("Consider increasing confidence threshold in cautious mode".to_string());
                }
            },
            SafetyLevel::Normal => {
                // No additional constraints
            }
        }

        // Check system health
        if context.system_state.health != HealthStatus::Healthy {
            if matches!(action.action_type, ActionType::UpdateModel) {
                issues.push("Model updates not allowed when system health is degraded".to_string());
            }
        }

        let is_valid = issues.is_empty();
        let confidence = if is_valid { 0.9 } else { 0.1 };

        Ok(ValidationResult {
            is_valid,
            confidence,
            issues,
            suggestions,
        })
    }

    fn name(&self) -> &str {
        "safety_validator"
    }
}

/// Resource validator for actions
pub struct ResourceValidator<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ResourceValidator<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> ActionValidator<T> for ResourceValidator<T> {
    fn validate(
        &self,
        action: &Action<T>,
        context: &ActionContext<T>,
    ) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check memory availability
        if context.resources.memory_available_mb < 512 {
            if matches!(action.action_type, ActionType::AnalyzeModality) {
                issues.push("Insufficient memory for modality analysis".to_string());
            }
        }

        // Check CPU load
        if context.system_state.load > 0.9 {
            suggestions.push("Consider deferring non-critical actions due to high CPU load".to_string());
        }

        let is_valid = issues.is_empty();
        let confidence = if is_valid { 0.85 } else { 0.2 };

        Ok(ValidationResult {
            is_valid,
            confidence,
            issues,
            suggestions,
        })
    }

    fn name(&self) -> &str {
        "resource_validator"
    }
}

/// Constraint validator for actions
pub struct ConstraintValidator<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ConstraintValidator<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> ActionValidator<T> for ConstraintValidator<T> {
    fn validate(
        &self,
        action: &Action<T>,
        context: &ActionContext<T>,
    ) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check time constraints
        if context.time_budget < Duration::from_millis(1000) {
            if matches!(action.action_type, ActionType::AnalyzeModality | ActionType::UpdateModel) {
                issues.push("Action may exceed available time budget".to_string());
            }
        }

        // Check action confidence against minimum threshold
        if action.confidence.to_f64().unwrap_or(0.0) < 0.3 {
            suggestions.push("Action confidence is below recommended threshold".to_string());
        }

        let is_valid = issues.is_empty();
        let confidence = if is_valid { 0.8 } else { 0.3 };

        Ok(ValidationResult {
            is_valid,
            confidence,
            issues,
            suggestions,
        })
    }

    fn name(&self) -> &str {
        "constraint_validator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_engine_creation() {
        let config: ActionConfig<f32> = ActionConfig::default();
        let engine = ActionEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_action_history() {
        let mut history: ActionHistory<f32> = ActionHistory::new();
        
        let outcome = ActionOutcome {
            action: Action {
                id: Uuid::new_v4(),
                action_type: ActionType::MakeDecision,
                parameters: HashMap::new(),
                expected_outcome: "Test outcome".to_string(),
                confidence: 0.8,
                explanation: "Test action".to_string(),
                timestamp: Utc::now(),
                decision: Some(Decision::Truth),
            },
            success: true,
            execution_time: Duration::from_millis(100),
            description: "Successful test".to_string(),
            actual_utility: Some(0.9),
            timestamp: Utc::now(),
        };

        history.add_outcome(outcome);
        assert_eq!(history.total_actions, 1);
        assert_eq!(history.get_success_rate(&ActionType::MakeDecision), 1.0);
    }

    #[test]
    fn test_multimodal_strategy() {
        let strategy: MultimodalWeightedStrategy<f32> = MultimodalWeightedStrategy::new();
        assert_eq!(strategy.name(), "multimodal_weighted");
        
        let params = strategy.parameters();
        assert!(params.contains_key("type"));
    }

    #[test]
    fn test_safety_validator() {
        let validator: SafetyValidator<f32> = SafetyValidator::new();
        assert_eq!(validator.name(), "safety_validator");

        let action = Action {
            id: Uuid::new_v4(),
            action_type: ActionType::MakeDecision,
            parameters: HashMap::new(),
            expected_outcome: "Test".to_string(),
            confidence: 0.8,
            explanation: "Test action".to_string(),
            timestamp: Utc::now(),
            decision: Some(Decision::Truth),
        };

        let context = ActionContext {
            system_state: SystemState {
                load: 0.3,
                available_memory_mb: 2048,
                active_modalities: vec![ModalityType::Text],
                health: HealthStatus::Healthy,
            },
            resources: ResourceState {
                cpu_available: 0.7,
                memory_available_mb: 2048,
                gpu_available: Some(0.8),
                bandwidth_mbps: 100.0,
            },
            recent_outcomes: vec![],
            time_budget: Duration::from_secs(10),
            safety_level: SafetyLevel::Normal,
        };

        let result = validator.validate(&action, &context);
        assert!(result.is_ok());
        assert!(result.unwrap().is_valid);
    }
}