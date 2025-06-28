//! Reinforcement learning for ReAct agents
//!
//! This module implements reinforcement learning algorithms specifically designed
//! for ReAct agents, enabling them to improve their reasoning and action selection
//! through experience and reward feedback.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;
use chrono::Utc;

use crate::error::{Result, VeritasError};
use crate::types::*;
use super::*;

/// Reinforcement learning coordinator for ReAct agents
pub struct ReinforcementLearner<T: Float> {
    config: RLConfig<T>,
    /// Policy network for action selection
    policy: PolicyNetwork<T>,
    /// Value network for state evaluation
    value_network: ValueNetwork<T>,
    /// Experience replay buffer
    replay_buffer: ReplayBuffer<T>,
    /// Learning statistics
    stats: RLStats<T>,
    /// Current training episode
    current_episode: usize,
}

/// Configuration for reinforcement learning
#[derive(Debug, Clone)]
pub struct RLConfig<T: Float> {
    /// Learning rate for policy updates
    pub policy_learning_rate: T,
    /// Learning rate for value updates
    pub value_learning_rate: T,
    /// Discount factor for future rewards
    pub discount_factor: T,
    /// Exploration rate (epsilon for epsilon-greedy)
    pub exploration_rate: T,
    /// Exploration decay rate
    pub exploration_decay: T,
    /// Minimum exploration rate
    pub min_exploration_rate: T,
    /// Target network update frequency
    pub target_update_frequency: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Memory buffer size
    pub buffer_size: usize,
    /// Training frequency (steps between updates)
    pub training_frequency: usize,
    /// Maximum episode length
    pub max_episode_length: usize,
    /// Reward shaping parameters
    pub reward_shaping: RewardShaping<T>,
}

impl<T: Float> Default for RLConfig<T> {
    fn default() -> Self {
        Self {
            policy_learning_rate: T::from(0.001).unwrap(),
            value_learning_rate: T::from(0.001).unwrap(),
            discount_factor: T::from(0.99).unwrap(),
            exploration_rate: T::from(0.3).unwrap(),
            exploration_decay: T::from(0.995).unwrap(),
            min_exploration_rate: T::from(0.05).unwrap(),
            target_update_frequency: 100,
            batch_size: 32,
            buffer_size: 10000,
            training_frequency: 4,
            max_episode_length: 200,
            reward_shaping: RewardShaping::default(),
        }
    }
}

/// Reward shaping configuration
#[derive(Debug, Clone)]
pub struct RewardShaping<T: Float> {
    /// Reward for correct decisions
    pub correct_decision_reward: T,
    /// Penalty for incorrect decisions
    pub incorrect_decision_penalty: T,
    /// Reward for high confidence correct decisions
    pub high_confidence_bonus: T,
    /// Penalty for high confidence incorrect decisions
    pub overconfidence_penalty: T,
    /// Reward for uncertainty when appropriate
    pub appropriate_uncertainty_reward: T,
    /// Reward for efficient reasoning
    pub efficiency_reward: T,
    /// Penalty for excessive reasoning time
    pub time_penalty_factor: T,
    /// Reward for explainable decisions
    pub explainability_reward: T,
}

impl<T: Float> Default for RewardShaping<T> {
    fn default() -> Self {
        Self {
            correct_decision_reward: T::from(1.0).unwrap(),
            incorrect_decision_penalty: T::from(-1.0).unwrap(),
            high_confidence_bonus: T::from(0.5).unwrap(),
            overconfidence_penalty: T::from(-0.5).unwrap(),
            appropriate_uncertainty_reward: T::from(0.3).unwrap(),
            efficiency_reward: T::from(0.2).unwrap(),
            time_penalty_factor: T::from(-0.001).unwrap(),
            explainability_reward: T::from(0.1).unwrap(),
        }
    }
}

/// Policy network for action selection
pub struct PolicyNetwork<T: Float> {
    /// Network weights
    weights: HashMap<String, Vec<T>>,
    /// Network architecture
    architecture: NetworkArchitecture,
    /// Optimizer state
    optimizer_state: OptimizerState<T>,
}

/// Value network for state evaluation
pub struct ValueNetwork<T: Float> {
    /// Network weights
    weights: HashMap<String, Vec<T>>,
    /// Network architecture
    architecture: NetworkArchitecture,
    /// Optimizer state
    optimizer_state: OptimizerState<T>,
}

/// Network architecture specification
#[derive(Debug, Clone)]
pub struct NetworkArchitecture {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationFunction,
    /// Dropout rate
    pub dropout_rate: f64,
}

/// Activation function types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    LeakyReLU,
}

/// Optimizer state for network training
#[derive(Debug, Clone)]
pub struct OptimizerState<T: Float> {
    /// Momentum terms
    pub momentum: HashMap<String, Vec<T>>,
    /// Velocity terms (for Adam optimizer)
    pub velocity: HashMap<String, Vec<T>>,
    /// Time step for adaptive learning rates
    pub time_step: usize,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule<T>,
}

/// Learning rate schedule
#[derive(Debug, Clone)]
pub struct LearningRateSchedule<T: Float> {
    /// Initial learning rate
    pub initial_rate: T,
    /// Decay type
    pub decay_type: DecayType,
    /// Decay parameters
    pub decay_params: HashMap<String, T>,
}

/// Learning rate decay types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecayType {
    Exponential,
    Linear,
    StepWise,
    CosineAnnealing,
    Polynomial,
}

/// Experience replay buffer
pub struct ReplayBuffer<T: Float> {
    /// Buffer for storing experiences
    buffer: VecDeque<Experience<T>>,
    /// Maximum buffer size
    capacity: usize,
    /// Priority weights for prioritized replay
    priorities: VecDeque<T>,
    /// Priority scaling factor
    alpha: T,
    /// Importance sampling factor
    beta: T,
}

impl<T: Float> ReplayBuffer<T> {
    /// Create new replay buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            priorities: VecDeque::with_capacity(capacity),
            alpha: T::from(0.6).unwrap(),
            beta: T::from(0.4).unwrap(),
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience<T>) -> Result<()> {
        // Calculate priority based on temporal difference error
        let priority = T::from(1.0).unwrap(); // Simplified - would normally use TD error
        
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }
        
        self.buffer.push_back(experience);
        self.priorities.push_back(priority);
        
        Ok(())
    }

    /// Sample batch of experiences
    pub fn sample(&self, batch_size: usize) -> Result<Vec<Experience<T>>> {
        if self.buffer.len() < batch_size {
            return Err(VeritasError::invalid_input(
                "Buffer doesn't contain enough experiences",
                "batch_size",
            ));
        }

        let mut batch = Vec::with_capacity(batch_size);
        
        // Simplified random sampling (would normally use priority-based sampling)
        for _ in 0..batch_size {
            let idx = fastrand::usize(0..self.buffer.len());
            batch.push(self.buffer[idx].clone());
        }
        
        Ok(batch)
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}

/// Reinforcement learning statistics
#[derive(Debug, Clone, Default)]
pub struct RLStats<T: Float> {
    /// Episodes completed
    pub episodes_completed: usize,
    /// Total steps taken
    pub total_steps: usize,
    /// Average episode reward
    pub avg_episode_reward: T,
    /// Average episode length
    pub avg_episode_length: f64,
    /// Policy loss over time
    pub policy_loss_history: Vec<T>,
    /// Value loss over time
    pub value_loss_history: Vec<T>,
    /// Exploration rate over time
    pub exploration_rate_history: Vec<T>,
    /// Learning rate over time
    pub learning_rate_history: Vec<T>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, T>,
}

impl<T: Float> ReinforcementLearner<T> {
    /// Create new reinforcement learner
    pub fn new(config: RLConfig<T>) -> Result<Self> {
        let policy = PolicyNetwork::new(config.policy_learning_rate)?;
        let value_network = ValueNetwork::new(config.value_learning_rate)?;
        let replay_buffer = ReplayBuffer::new(config.buffer_size);
        
        Ok(Self {
            config,
            policy,
            value_network,
            replay_buffer,
            stats: RLStats::default(),
            current_episode: 0,
        })
    }

    /// Train the agent for one episode
    pub fn train_episode(
        &mut self,
        initial_state: &State<T>,
        environment: &dyn Environment<T>,
    ) -> Result<EpisodeResult<T>> {
        let start_time = Instant::now();
        let mut state = initial_state.clone();
        let mut episode_reward = T::zero();
        let mut episode_steps = 0;
        let mut episode_experiences = Vec::new();

        while episode_steps < self.config.max_episode_length {
            // Select action using current policy
            let action = self.select_action(&state)?;
            
            // Execute action in environment
            let (next_state, reward, done) = environment.step(&state, &action)?;
            
            // Create experience
            let experience = Experience {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
                metadata: HashMap::new(),
            };
            
            // Store experience
            self.replay_buffer.add(experience.clone())?;
            episode_experiences.push(experience);
            
            episode_reward = episode_reward + reward;
            episode_steps += 1;
            state = next_state;
            
            // Train networks if enough experiences collected
            if episode_steps % self.config.training_frequency == 0 &&
               self.replay_buffer.size() >= self.config.batch_size {
                self.train_networks()?;
            }
            
            if done {
                break;
            }
        }
        
        // Update episode statistics
        self.update_episode_stats(episode_reward, episode_steps, start_time.elapsed());
        self.current_episode += 1;
        
        Ok(EpisodeResult {
            episode_number: self.current_episode,
            total_reward: episode_reward,
            steps_taken: episode_steps,
            duration: start_time.elapsed(),
            experiences: episode_experiences,
            final_state: state,
        })
    }

    /// Select action using current policy
    fn select_action(&self, state: &State<T>) -> Result<Action<T>> {
        // Use epsilon-greedy exploration
        if fastrand::f64() < self.config.exploration_rate.to_f64().unwrap_or(0.1) {
            // Explore: random action
            self.sample_random_action(state)
        } else {
            // Exploit: use policy
            self.policy.predict_action(state)
        }
    }

    /// Sample random action for exploration
    fn sample_random_action(&self, _state: &State<T>) -> Result<Action<T>> {
        // Generate random action (simplified implementation)
        let action_types = vec![
            ActionType::MakeDecision,
            ActionType::AnalyzeModality,
            ActionType::RequestMoreData,
        ];
        
        let action_type = action_types[fastrand::usize(0..action_types.len())].clone();
        
        Ok(Action {
            id: Uuid::new_v4(),
            action_type,
            parameters: HashMap::new(),
            expected_outcome: "Exploration action".to_string(),
            confidence: T::from(fastrand::f64()).unwrap(),
            explanation: "Random exploration action".to_string(),
            timestamp: Utc::now(),
            decision: None,
        })
    }

    /// Train policy and value networks
    fn train_networks(&mut self) -> Result<()> {
        // Sample batch from replay buffer
        let batch = self.replay_buffer.sample(self.config.batch_size)?;
        
        // Compute policy loss and update
        let policy_loss = self.compute_policy_loss(&batch)?;
        self.policy.update_weights(policy_loss)?;
        
        // Compute value loss and update
        let value_loss = self.compute_value_loss(&batch)?;
        self.value_network.update_weights(value_loss)?;
        
        // Update statistics
        self.stats.policy_loss_history.push(policy_loss);
        self.stats.value_loss_history.push(value_loss);
        
        Ok(())
    }

    /// Compute policy gradient loss
    fn compute_policy_loss(&self, batch: &[Experience<T>]) -> Result<T> {
        let mut total_loss = T::zero();
        
        for experience in batch {
            // Compute advantage (simplified)
            let value = self.value_network.predict(&experience.state)?;
            let next_value = if experience.done {
                T::zero()
            } else {
                self.value_network.predict(&experience.next_state)?
            };
            
            let target = experience.reward + self.config.discount_factor * next_value;
            let advantage = target - value;
            
            // Compute policy loss (simplified REINFORCE)
            let action_prob = self.policy.action_probability(&experience.state, &experience.action)?;
            let log_prob = action_prob.ln();
            let policy_loss = -log_prob * advantage;
            
            total_loss = total_loss + policy_loss;
        }
        
        Ok(total_loss / T::from(batch.len()).unwrap())
    }

    /// Compute value function loss
    fn compute_value_loss(&self, batch: &[Experience<T>]) -> Result<T> {
        let mut total_loss = T::zero();
        
        for experience in batch {
            let predicted_value = self.value_network.predict(&experience.state)?;
            let next_value = if experience.done {
                T::zero()
            } else {
                self.value_network.predict(&experience.next_state)?
            };
            
            let target_value = experience.reward + self.config.discount_factor * next_value;
            let td_error = target_value - predicted_value;
            let loss = td_error * td_error; // MSE loss
            
            total_loss = total_loss + loss;
        }
        
        Ok(total_loss / T::from(batch.len()).unwrap())
    }

    /// Update episode statistics
    fn update_episode_stats(&mut self, reward: T, steps: usize, duration: Duration) {
        self.stats.episodes_completed += 1;
        self.stats.total_steps += steps;
        
        // Update running averages
        let episode_count = T::from(self.stats.episodes_completed).unwrap();
        let prev_avg_reward = self.stats.avg_episode_reward;
        self.stats.avg_episode_reward = prev_avg_reward + (reward - prev_avg_reward) / episode_count;
        
        let prev_avg_length = self.stats.avg_episode_length;
        self.stats.avg_episode_length = prev_avg_length + 
            (steps as f64 - prev_avg_length) / self.stats.episodes_completed as f64;
        
        // Update exploration rate (decay)
        self.config.exploration_rate = (self.config.exploration_rate * self.config.exploration_decay)
            .max(self.config.min_exploration_rate);
        
        self.stats.exploration_rate_history.push(self.config.exploration_rate);
    }

    /// Calculate shaped reward based on agent performance
    pub fn calculate_shaped_reward(&self, experience: &Experience<T>) -> Result<T> {
        let mut shaped_reward = experience.reward;
        
        // Extract decision from action if available
        if let Some(decision) = &experience.action.decision {
            // Add confidence-based shaping
            let confidence = experience.action.confidence;
            
            // Assume we have ground truth available in metadata
            if let Some(ground_truth_str) = experience.metadata.get("ground_truth") {
                let ground_truth = ground_truth_str == "true";
                let correct = match (decision, ground_truth) {
                    (Decision::Truth, true) | (Decision::Deception, false) => true,
                    (Decision::Uncertain, _) => {
                        // Reward appropriate uncertainty
                        confidence.to_f64().unwrap_or(0.0) < 0.7
                    },
                    _ => false,
                };
                
                if correct {
                    shaped_reward = shaped_reward + self.config.reward_shaping.correct_decision_reward;
                    
                    // High confidence bonus for correct decisions
                    if confidence.to_f64().unwrap_or(0.0) > 0.8 {
                        shaped_reward = shaped_reward + self.config.reward_shaping.high_confidence_bonus;
                    }
                } else {
                    shaped_reward = shaped_reward + self.config.reward_shaping.incorrect_decision_penalty;
                    
                    // Overconfidence penalty
                    if confidence.to_f64().unwrap_or(0.0) > 0.8 {
                        shaped_reward = shaped_reward + self.config.reward_shaping.overconfidence_penalty;
                    }
                }
            }
        }
        
        // Add efficiency reward based on reasoning time
        if let Some(reasoning_time_str) = experience.metadata.get("reasoning_time_ms") {
            if let Ok(reasoning_time_ms) = reasoning_time_str.parse::<f64>() {
                let time_penalty = self.config.reward_shaping.time_penalty_factor * 
                    T::from(reasoning_time_ms).unwrap();
                shaped_reward = shaped_reward + time_penalty;
            }
        }
        
        // Add explainability reward
        if experience.metadata.contains_key("explanation_quality") {
            shaped_reward = shaped_reward + self.config.reward_shaping.explainability_reward;
        }
        
        Ok(shaped_reward)
    }

    /// Get current learning statistics
    pub fn get_stats(&self) -> &RLStats<T> {
        &self.stats
    }

    /// Get current configuration
    pub fn get_config(&self) -> &RLConfig<T> {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RLConfig<T>) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Save learned policy
    pub fn save_policy(&self, path: &str) -> Result<()> {
        // Implementation would save network weights to file
        // For now, just return success
        Ok(())
    }

    /// Load learned policy
    pub fn load_policy(&mut self, path: &str) -> Result<()> {
        // Implementation would load network weights from file
        // For now, just return success
        Ok(())
    }
}

/// Result of training episode
#[derive(Debug, Clone)]
pub struct EpisodeResult<T: Float> {
    /// Episode number
    pub episode_number: usize,
    /// Total reward accumulated
    pub total_reward: T,
    /// Number of steps taken
    pub steps_taken: usize,
    /// Episode duration
    pub duration: Duration,
    /// All experiences from episode
    pub experiences: Vec<Experience<T>>,
    /// Final state reached
    pub final_state: State<T>,
}

/// Environment interface for training
pub trait Environment<T: Float>: Send + Sync {
    /// Execute action and return next state, reward, and done flag
    fn step(&self, state: &State<T>, action: &Action<T>) -> Result<(State<T>, T, bool)>;
    
    /// Reset environment to initial state
    fn reset(&mut self) -> Result<State<T>>;
    
    /// Get current state
    fn current_state(&self) -> &State<T>;
    
    /// Check if episode is complete
    fn is_done(&self) -> bool;
}

impl<T: Float> PolicyNetwork<T> {
    /// Create new policy network
    pub fn new(learning_rate: T) -> Result<Self> {
        Ok(Self {
            weights: HashMap::new(),
            architecture: NetworkArchitecture {
                input_dim: 128, // Based on state representation
                hidden_dims: vec![256, 128, 64],
                output_dim: 32, // Action space size
                activation: ActivationFunction::ReLU,
                dropout_rate: 0.1,
            },
            optimizer_state: OptimizerState {
                momentum: HashMap::new(),
                velocity: HashMap::new(),
                time_step: 0,
                learning_rate_schedule: LearningRateSchedule {
                    initial_rate: learning_rate,
                    decay_type: DecayType::Exponential,
                    decay_params: HashMap::new(),
                },
            },
        })
    }

    /// Predict action from state
    pub fn predict_action(&self, state: &State<T>) -> Result<Action<T>> {
        // Forward pass through network (simplified)
        let action_scores = self.forward(state)?;
        
        // Select action with highest score
        let (best_action_idx, _) = action_scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &T::zero()));
        
        // Convert index to action (simplified)
        let action_type = match best_action_idx % 3 {
            0 => ActionType::MakeDecision,
            1 => ActionType::AnalyzeModality,
            _ => ActionType::RequestMoreData,
        };
        
        Ok(Action {
            id: Uuid::new_v4(),
            action_type,
            parameters: HashMap::new(),
            expected_outcome: "Policy predicted action".to_string(),
            confidence: action_scores[best_action_idx],
            explanation: "Action selected by policy network".to_string(),
            timestamp: Utc::now(),
            decision: None,
        })
    }

    /// Calculate action probability
    pub fn action_probability(&self, state: &State<T>, action: &Action<T>) -> Result<T> {
        let action_scores = self.forward(state)?;
        
        // Apply softmax to get probabilities
        let max_score = action_scores.iter().cloned().fold(T::neg_infinity(), T::max);
        let exp_scores: Vec<T> = action_scores.iter()
            .map(|&score| (score - max_score).exp())
            .collect();
        let sum_exp: T = exp_scores.iter().cloned().sum();
        
        // Return probability for the given action (simplified)
        let action_idx = match action.action_type {
            ActionType::MakeDecision => 0,
            ActionType::AnalyzeModality => 1,
            ActionType::RequestMoreData => 2,
            _ => 0,
        };
        
        Ok(exp_scores[action_idx % exp_scores.len()] / sum_exp)
    }

    /// Forward pass through network
    fn forward(&self, state: &State<T>) -> Result<Vec<T>> {
        // Simplified forward pass
        let input_features = &state.features;
        let output_size = self.architecture.output_dim;
        
        // Return random scores for now (would be actual network computation)
        Ok((0..output_size)
            .map(|_| T::from(fastrand::f64()).unwrap())
            .collect())
    }

    /// Update network weights
    pub fn update_weights(&mut self, loss: T) -> Result<()> {
        // Simplified weight update
        self.optimizer_state.time_step += 1;
        Ok(())
    }
}

impl<T: Float> ValueNetwork<T> {
    /// Create new value network
    pub fn new(learning_rate: T) -> Result<Self> {
        Ok(Self {
            weights: HashMap::new(),
            architecture: NetworkArchitecture {
                input_dim: 128,
                hidden_dims: vec![256, 128],
                output_dim: 1, // Single value output
                activation: ActivationFunction::ReLU,
                dropout_rate: 0.1,
            },
            optimizer_state: OptimizerState {
                momentum: HashMap::new(),
                velocity: HashMap::new(),
                time_step: 0,
                learning_rate_schedule: LearningRateSchedule {
                    initial_rate: learning_rate,
                    decay_type: DecayType::Exponential,
                    decay_params: HashMap::new(),
                },
            },
        })
    }

    /// Predict state value
    pub fn predict(&self, state: &State<T>) -> Result<T> {
        // Simplified value prediction
        let avg_feature = if state.features.is_empty() {
            T::zero()
        } else {
            state.features.iter().cloned().sum::<T>() / T::from(state.features.len()).unwrap()
        };
        
        Ok(avg_feature)
    }

    /// Update network weights
    pub fn update_weights(&mut self, loss: T) -> Result<()> {
        // Simplified weight update
        self.optimizer_state.time_step += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reinforcement_learner_creation() {
        let config: RLConfig<f32> = RLConfig::default();
        let learner = ReinforcementLearner::new(config);
        assert!(learner.is_ok());
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer: ReplayBuffer<f32> = ReplayBuffer::new(100);
        
        let experience = Experience {
            state: State {
                features: vec![1.0, 2.0, 3.0],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.8,
            },
            action: Action {
                id: Uuid::new_v4(),
                action_type: ActionType::MakeDecision,
                parameters: HashMap::new(),
                expected_outcome: "Test".to_string(),
                confidence: 0.7,
                explanation: "Test action".to_string(),
                timestamp: Utc::now(),
                decision: Some(Decision::Truth),
            },
            reward: 1.0,
            next_state: State {
                features: vec![2.0, 3.0, 4.0],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.9,
            },
            done: false,
            metadata: HashMap::new(),
        };
        
        buffer.add(experience).unwrap();
        assert_eq!(buffer.size(), 1);
    }

    #[test]
    fn test_policy_network() {
        let network: PolicyNetwork<f32> = PolicyNetwork::new(0.001).unwrap();
        let state = State {
            features: vec![1.0, 2.0, 3.0],
            symbolic_features: HashMap::new(),
            temporal_context: None,
            confidence: 0.8,
        };
        
        let action = network.predict_action(&state);
        assert!(action.is_ok());
    }

    #[test]
    fn test_reward_shaping() {
        let config: RLConfig<f32> = RLConfig::default();
        let learner = ReinforcementLearner::new(config).unwrap();
        
        let mut metadata = HashMap::new();
        metadata.insert("ground_truth".to_string(), "true".to_string());
        
        let experience = Experience {
            state: State {
                features: vec![1.0, 2.0, 3.0],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.8,
            },
            action: Action {
                id: Uuid::new_v4(),
                action_type: ActionType::MakeDecision,
                parameters: HashMap::new(),
                expected_outcome: "Test".to_string(),
                confidence: 0.9,
                explanation: "Test action".to_string(),
                timestamp: Utc::now(),
                decision: Some(Decision::Truth),
            },
            reward: 0.5,
            next_state: State {
                features: vec![2.0, 3.0, 4.0],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.9,
            },
            done: false,
            metadata,
        };
        
        let shaped_reward = learner.calculate_shaped_reward(&experience);
        assert!(shaped_reward.is_ok());
        assert!(shaped_reward.unwrap() > 0.5); // Should be higher due to correct decision
    }
}