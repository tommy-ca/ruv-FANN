//! Learning module for GSPO (Generative Self-Play Optimization) framework
//!
//! This module implements a sophisticated self-play optimization system that enables
//! ReAct agents to improve their reasoning and decision-making through:
//! - Self-play scenarios with multiple agent interactions
//! - Generative reasoning pattern discovery
//! - Adaptive strategy optimization
//! - Multi-objective reward learning

pub mod gspo;
pub mod self_play;
pub mod pattern_discovery;
pub mod reward_learning;
pub mod strategy_optimization;
pub mod reinforcement;

pub use gspo::*;
pub use self_play::*;
pub use pattern_discovery::*;
pub use reward_learning::*;
pub use strategy_optimization::*;
pub use reinforcement::*;

use crate::error::Result;
use crate::types::*;
use num_traits::Float;

/// Core trait for learning systems
pub trait LearningSystem<T: Float>: Send + Sync {
    /// Type of experience data this system learns from
    type Experience;
    /// Type of learned knowledge/model
    type Model;
    /// Configuration type
    type Config;

    /// Learn from a single experience
    fn learn_from_experience(&mut self, experience: Self::Experience) -> Result<()>;

    /// Learn from a batch of experiences
    fn learn_from_batch(&mut self, experiences: &[Self::Experience]) -> Result<()>;

    /// Get the current learned model
    fn get_model(&self) -> &Self::Model;

    /// Update learning configuration
    fn update_config(&mut self, config: Self::Config) -> Result<()>;

    /// Get learning statistics
    fn get_learning_stats(&self) -> LearningStats;
}

/// Statistics for learning systems
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    /// Total experiences processed
    pub experiences_processed: usize,
    /// Learning iterations completed
    pub iterations_completed: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Model performance metrics
    pub performance_metrics: std::collections::HashMap<String, f64>,
    /// Convergence indicators
    pub convergence_score: f64,
    /// Time spent learning
    pub total_learning_time_ms: u64,
}

/// Experience for reinforcement learning
#[derive(Debug, Clone)]
pub struct Experience<T: Float> {
    /// State representation
    pub state: State<T>,
    /// Action taken
    pub action: Action<T>,
    /// Reward received
    pub reward: T,
    /// Next state
    pub next_state: State<T>,
    /// Whether episode terminated
    pub done: bool,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// State representation for learning
#[derive(Debug, Clone)]
pub struct State<T: Float> {
    /// Numeric feature vector
    pub features: Vec<T>,
    /// Symbolic features
    pub symbolic_features: std::collections::HashMap<String, String>,
    /// Temporal context
    pub temporal_context: Option<TemporalContext>,
    /// Confidence in state representation
    pub confidence: T,
}

/// Temporal context for sequential learning
#[derive(Debug, Clone)]
pub struct TemporalContext {
    /// Time step in sequence
    pub time_step: usize,
    /// Sequence length
    pub sequence_length: usize,
    /// Previous states in window
    pub history_window: Vec<String>, // Simplified representation
}

/// Reward signal for learning
#[derive(Debug, Clone)]
pub struct RewardSignal<T: Float> {
    /// Primary reward value
    pub value: T,
    /// Reward components (for multi-objective learning)
    pub components: std::collections::HashMap<String, T>,
    /// Reward source/type
    pub source: RewardSource,
    /// Confidence in reward signal
    pub confidence: T,
}

/// Sources of reward signals
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RewardSource {
    /// Ground truth feedback
    GroundTruth,
    /// Human feedback
    Human,
    /// Self-evaluation
    SelfEvaluation,
    /// Peer agent feedback
    PeerFeedback,
    /// Environmental feedback
    Environment,
}

/// Learning objectives for multi-objective optimization
#[derive(Debug, Clone)]
pub struct LearningObjectives<T: Float> {
    /// Primary objectives with weights
    pub primary: std::collections::HashMap<String, T>,
    /// Secondary objectives with weights  
    pub secondary: std::collections::HashMap<String, T>,
    /// Constraints that must be satisfied
    pub constraints: Vec<Constraint<T>>,
}

/// Learning constraint
#[derive(Debug, Clone)]
pub struct Constraint<T: Float> {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Target value or threshold
    pub target: T,
    /// Current value
    pub current: T,
    /// Whether constraint is satisfied
    pub satisfied: bool,
}

/// Types of learning constraints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintType {
    /// Minimum value constraint
    Minimum,
    /// Maximum value constraint
    Maximum,
    /// Equality constraint
    Equality,
    /// Range constraint
    Range,
}