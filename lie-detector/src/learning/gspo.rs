//! Generative Self-Play Optimization (GSPO) framework
//!
//! This module implements the core GSPO algorithm that enables ReAct agents to improve
//! through self-play interactions, generative scenario creation, and optimization.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;
use chrono::Utc;

use crate::error::{Result, VeritasError};
use crate::types::*;
use crate::agents::{ReactAgent, Memory, ReasoningEngine, ActionEngine};
use super::*;

/// GSPO framework coordinator
pub struct GSPOFramework<T: Float> {
    config: GSPOConfig<T>,
    /// Pool of agents for self-play
    agent_pool: Vec<Box<dyn ReactAgent<T>>>,
    /// Scenario generator for creating training situations
    scenario_generator: ScenarioGenerator<T>,
    /// Optimization engine for improving strategies
    optimizer: StrategyOptimizer<T>,
    /// Experience replay buffer
    experience_buffer: ExperienceBuffer<T>,
    /// Learning statistics
    stats: LearningStats,
    /// Current generation number
    generation: usize,
}

/// Configuration for GSPO framework
#[derive(Debug, Clone)]
pub struct GSPOConfig<T: Float> {
    /// Number of agents in the pool
    pub agent_pool_size: usize,
    /// Episodes per generation
    pub episodes_per_generation: usize,
    /// Maximum episode length
    pub max_episode_length: usize,
    /// Learning rate for optimization
    pub learning_rate: T,
    /// Exploration rate for agent actions
    pub exploration_rate: T,
    /// Experience buffer capacity
    pub buffer_capacity: usize,
    /// Batch size for learning updates
    pub batch_size: usize,
    /// Number of optimization steps per generation
    pub optimization_steps: usize,
    /// Scenario diversity factor
    pub diversity_factor: T,
    /// Elite selection ratio
    pub elite_ratio: T,
    /// Mutation rate for strategy evolution
    pub mutation_rate: T,
}

impl<T: Float> Default for GSPOConfig<T> {
    fn default() -> Self {
        Self {
            agent_pool_size: 8,
            episodes_per_generation: 50,
            max_episode_length: 100,
            learning_rate: T::from(0.001).unwrap(),
            exploration_rate: T::from(0.1).unwrap(),
            buffer_capacity: 10000,
            batch_size: 32,
            optimization_steps: 100,
            diversity_factor: T::from(0.3).unwrap(),
            elite_ratio: T::from(0.2).unwrap(),
            mutation_rate: T::from(0.05).unwrap(),
        }
    }
}

/// Scenario generator for creating diverse training situations
pub struct ScenarioGenerator<T: Float> {
    /// Template scenarios
    templates: Vec<ScenarioTemplate<T>>,
    /// Generation parameters
    generation_params: GenerationParams<T>,
    /// Scenario history for diversity tracking
    scenario_history: VecDeque<GeneratedScenario<T>>,
}

/// Template for generating scenarios
#[derive(Debug, Clone)]
pub struct ScenarioTemplate<T: Float> {
    /// Template name
    pub name: String,
    /// Scenario type
    pub scenario_type: ScenarioType,
    /// Parameter ranges for generation
    pub parameter_ranges: HashMap<String, (T, T)>,
    /// Required modalities
    pub required_modalities: Vec<ModalityType>,
    /// Difficulty level (0.0 to 1.0)
    pub difficulty: T,
    /// Priority weight for selection
    pub weight: T,
}

/// Types of scenarios for self-play
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScenarioType {
    /// Standard deception detection scenario
    Standard,
    /// High-stress interrogation scenario
    HighStress,
    /// Multi-person interaction scenario
    MultiPerson,
    /// Cross-cultural scenario
    CrossCultural,
    /// Low-confidence ambiguous scenario
    Ambiguous,
    /// Time-pressured quick decision scenario
    TimePressured,
    /// Mixed truthful/deceptive content
    Mixed,
}

/// Parameters for scenario generation
#[derive(Debug, Clone)]
pub struct GenerationParams<T: Float> {
    /// Diversity weight
    pub diversity_weight: T,
    /// Difficulty progression rate
    pub difficulty_progression: T,
    /// Noise level for parameter variation
    pub noise_level: T,
    /// Minimum scenario quality threshold
    pub quality_threshold: T,
}

/// Generated scenario for self-play
#[derive(Debug, Clone)]
pub struct GeneratedScenario<T: Float> {
    /// Unique scenario ID
    pub id: Uuid,
    /// Scenario type
    pub scenario_type: ScenarioType,
    /// Scenario parameters
    pub parameters: HashMap<String, T>,
    /// Generated observations
    pub observations: Observations<T>,
    /// Expected difficulty
    pub expected_difficulty: T,
    /// Ground truth label
    pub ground_truth: Option<Decision>,
    /// Quality score
    pub quality_score: T,
    /// Generation timestamp
    pub created_at: chrono::DateTime<Utc>,
}

/// Experience buffer for storing and sampling training data
pub struct ExperienceBuffer<T: Float> {
    /// Buffer storage
    buffer: VecDeque<Experience<T>>,
    /// Buffer capacity
    capacity: usize,
    /// Priority weights for sampling
    priorities: VecDeque<T>,
    /// Buffer statistics
    stats: BufferStats,
}

/// Statistics for experience buffer
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Total experiences added
    pub total_added: usize,
    /// Total experiences sampled
    pub total_sampled: usize,
    /// Average experience quality
    pub avg_quality: f64,
    /// Buffer turnover rate
    pub turnover_rate: f64,
}

/// Strategy optimizer for evolving agent behaviors
pub struct StrategyOptimizer<T: Float> {
    /// Current strategies being optimized
    strategies: HashMap<Uuid, AgentStrategy<T>>,
    /// Optimization algorithm
    algorithm: OptimizationAlgorithm<T>,
    /// Performance tracking
    performance_tracker: PerformanceTracker<T>,
}

/// Agent strategy representation
#[derive(Debug, Clone)]
pub struct AgentStrategy<T: Float> {
    /// Strategy ID
    pub id: Uuid,
    /// Strategy parameters
    pub parameters: HashMap<String, T>,
    /// Performance metrics
    pub performance: PerformanceMetrics<T>,
    /// Strategy version/generation
    pub generation: usize,
    /// Parent strategies (for genealogy tracking)
    pub parents: Vec<Uuid>,
}

/// Performance metrics for strategies
#[derive(Debug, Clone)]
pub struct PerformanceMetrics<T: Float> {
    /// Overall accuracy
    pub accuracy: T,
    /// Confidence calibration
    pub calibration: T,
    /// Reasoning quality
    pub reasoning_quality: T,
    /// Adaptation speed
    pub adaptation_speed: T,
    /// Robustness to noise
    pub robustness: T,
    /// Efficiency metrics
    pub efficiency: T,
}

/// Optimization algorithm for strategy evolution
pub struct OptimizationAlgorithm<T: Float> {
    /// Algorithm type
    pub algorithm_type: AlgorithmType,
    /// Algorithm parameters
    pub parameters: HashMap<String, T>,
    /// Optimization history
    pub history: Vec<OptimizationStep<T>>,
}

/// Types of optimization algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgorithmType {
    /// Genetic algorithm
    Genetic,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Gradient-based optimization
    GradientBased,
    /// Hybrid approach
    Hybrid,
}

/// Single optimization step
#[derive(Debug, Clone)]
pub struct OptimizationStep<T: Float> {
    /// Step number
    pub step: usize,
    /// Best performance achieved
    pub best_performance: T,
    /// Average performance
    pub avg_performance: T,
    /// Population diversity
    pub diversity: T,
    /// Convergence indicator
    pub convergence: T,
}

/// Performance tracking for strategies
pub struct PerformanceTracker<T: Float> {
    /// Performance history by strategy
    performance_history: HashMap<Uuid, Vec<PerformanceSnapshot<T>>>,
    /// Leaderboard of top strategies
    leaderboard: Vec<(Uuid, T)>,
    /// Performance trends
    trends: HashMap<String, TrendAnalysis<T>>,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot<T: Float> {
    /// Timestamp
    pub timestamp: chrono::DateTime<Utc>,
    /// Performance metrics
    pub metrics: PerformanceMetrics<T>,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Trend analysis for performance metrics
#[derive(Debug, Clone)]
pub struct TrendAnalysis<T: Float> {
    /// Trend direction (positive, negative, stable)
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: T,
    /// Statistical significance
    pub significance: T,
    /// Prediction for next period
    pub prediction: T,
}

/// Direction of performance trends
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Declining,
    Stable,
    Volatile,
}

impl<T: Float> GSPOFramework<T> {
    /// Create new GSPO framework
    pub fn new(config: GSPOConfig<T>) -> Result<Self> {
        let scenario_generator = ScenarioGenerator::new()?;
        let optimizer = StrategyOptimizer::new()?;
        let experience_buffer = ExperienceBuffer::new(config.buffer_capacity);

        Ok(Self {
            config,
            agent_pool: Vec::new(),
            scenario_generator,
            optimizer,
            experience_buffer,
            stats: LearningStats::default(),
            generation: 0,
        })
    }

    /// Initialize agent pool for self-play
    pub fn initialize_agent_pool(&mut self, base_config: DetectorConfig<T>) -> Result<()> {
        for i in 0..self.config.agent_pool_size {
            // Create slightly varied configurations for diversity
            let mut agent_config = base_config.clone();
            
            // Add some parameter variation
            let variation = T::from(0.1 * (i as f64) / self.config.agent_pool_size as f64).unwrap();
            agent_config.reasoning_config.temperature = 
                agent_config.reasoning_config.temperature + variation;

            // Create agent components
            let memory = Arc::new(Memory::new(agent_config.memory_config.clone())?);
            let reasoning_engine = Arc::new(ReasoningEngine::new(agent_config.reasoning_config.clone())?);
            let action_engine = Arc::new(ActionEngine::new(agent_config.action_config.clone())?);

            // Create agent
            let agent = crate::agents::create_custom_react_agent(
                agent_config,
                memory,
                reasoning_engine,
                action_engine,
            )?;

            self.agent_pool.push(agent);
        }

        Ok(())
    }

    /// Run single GSPO generation
    pub fn run_generation(&mut self) -> Result<GenerationResult<T>> {
        let start_time = Instant::now();
        
        // Generate scenarios for this generation
        let scenarios = self.scenario_generator.generate_batch(
            self.config.episodes_per_generation
        )?;

        // Run self-play episodes
        let mut episodes_data = Vec::new();
        for scenario in scenarios {
            let episode_data = self.run_self_play_episode(scenario)?;
            episodes_data.push(episode_data);
        }

        // Collect experiences for learning
        for episode_data in &episodes_data {
            for experience in &episode_data.experiences {
                self.experience_buffer.add_experience(experience.clone());
            }
        }

        // Optimize strategies based on experiences
        let optimization_result = self.optimizer.optimize_strategies(
            &mut self.experience_buffer,
            self.config.optimization_steps
        )?;

        // Update agent pool with optimized strategies
        self.update_agent_pool(&optimization_result)?;

        // Calculate generation metrics
        let generation_time = start_time.elapsed();
        let avg_performance = episodes_data.iter()
            .map(|ep| ep.performance_score.to_f64().unwrap_or(0.0))
            .sum::<f64>() / episodes_data.len() as f64;

        self.generation += 1;
        
        // Update statistics
        self.stats.iterations_completed += 1;
        self.stats.experiences_processed += episodes_data.iter()
            .map(|ep| ep.experiences.len())
            .sum::<usize>();
        self.stats.total_learning_time_ms += generation_time.as_millis() as u64;

        Ok(GenerationResult {
            generation: self.generation,
            episodes_run: episodes_data.len(),
            avg_performance: T::from(avg_performance).unwrap(),
            best_performance: optimization_result.best_performance,
            diversity_score: optimization_result.diversity_score,
            convergence_score: optimization_result.convergence_score,
            generation_time,
        })
    }

    /// Run self-play episode with generated scenario
    fn run_self_play_episode(&mut self, scenario: GeneratedScenario<T>) -> Result<EpisodeData<T>> {
        // Select two agents for self-play
        let agent1_idx = fastrand::usize(0..self.agent_pool.size());
        let agent2_idx = loop {
            let idx = fastrand::usize(0..self.agent_pool.size());
            if idx != agent1_idx { break idx; }
        };

        let mut experiences = Vec::new();
        let mut step = 0;
        let mut episode_reward = T::zero();

        // Run episode steps
        while step < self.config.max_episode_length {
            // Agent 1 processes scenario
            let agent1_result = self.run_agent_step(agent1_idx, &scenario, step)?;
            
            // Agent 2 evaluates agent 1's reasoning (peer feedback)
            let agent2_evaluation = self.evaluate_reasoning(agent2_idx, &agent1_result)?;

            // Create experience from this interaction
            let experience = Experience {
                state: agent1_result.state,
                action: agent1_result.action,
                reward: agent2_evaluation.reward,
                next_state: agent2_evaluation.next_state,
                done: agent1_result.done || step >= self.config.max_episode_length - 1,
                metadata: HashMap::from([
                    ("episode_id".to_string(), scenario.id.to_string()),
                    ("step".to_string(), step.to_string()),
                    ("evaluator".to_string(), format!("agent_{}", agent2_idx)),
                ]),
            };

            episode_reward = episode_reward + experience.reward;
            experiences.push(experience);

            if agent1_result.done {
                break;
            }

            step += 1;
        }

        // Calculate performance score
        let performance_score = self.calculate_episode_performance(&experiences, &scenario)?;

        Ok(EpisodeData {
            scenario_id: scenario.id,
            agent_ids: vec![agent1_idx, agent2_idx],
            experiences,
            total_reward: episode_reward,
            performance_score,
            episode_length: step,
        })
    }

    /// Run single step for an agent
    fn run_agent_step(
        &mut self,
        agent_idx: usize,
        scenario: &GeneratedScenario<T>,
        step: usize,
    ) -> Result<AgentStepResult<T>> {
        let agent = &mut self.agent_pool[agent_idx];
        
        // Agent observes scenario
        agent.observe(scenario.observations.clone())?;
        
        // Agent thinks about observations
        let thoughts = agent.think()?;
        
        // Agent selects action
        let action = agent.act()?;
        
        // Create state representation
        let state = State {
            features: vec![
                T::from(thoughts.confidence()).unwrap(),
                T::from(step as f64 / self.config.max_episode_length as f64).unwrap(),
                T::from(scenario.expected_difficulty.to_f64().unwrap()).unwrap(),
            ],
            symbolic_features: HashMap::from([
                ("scenario_type".to_string(), format!("{:?}", scenario.scenario_type)),
                ("action_type".to_string(), format!("{:?}", action.action_type)),
            ]),
            temporal_context: Some(TemporalContext {
                time_step: step,
                sequence_length: self.config.max_episode_length,
                history_window: vec![format!("step_{}", step)],
            }),
            confidence: action.confidence,
        };

        // Determine if episode should end
        let done = matches!(action.action_type, ActionType::MakeDecision) || 
                  step >= self.config.max_episode_length - 1;

        Ok(AgentStepResult {
            state,
            action,
            thoughts,
            done,
        })
    }

    /// Evaluate reasoning quality by peer agent
    fn evaluate_reasoning(
        &mut self,
        evaluator_idx: usize,
        step_result: &AgentStepResult<T>,
    ) -> Result<EvaluationResult<T>> {
        // Simplified evaluation - in practice this would be more sophisticated
        let reasoning_quality = step_result.thoughts.confidence();
        let action_appropriateness = step_result.action.confidence.to_f64().unwrap_or(0.0);
        
        // Calculate reward based on reasoning quality and action appropriateness
        let reward = T::from((reasoning_quality + action_appropriateness) / 2.0).unwrap();
        
        // Create next state (simplified)
        let next_state = State {
            features: step_result.state.features.clone(),
            symbolic_features: step_result.state.symbolic_features.clone(),
            temporal_context: step_result.state.temporal_context.clone(),
            confidence: reward,
        };

        Ok(EvaluationResult {
            reward,
            next_state,
            evaluation_confidence: T::from(0.8).unwrap(),
        })
    }

    /// Calculate performance score for an episode
    fn calculate_episode_performance(
        &self,
        experiences: &[Experience<T>],
        scenario: &GeneratedScenario<T>,
    ) -> Result<T> {
        if experiences.is_empty() {
            return Ok(T::zero());
        }

        // Calculate average reward
        let avg_reward = experiences.iter()
            .map(|exp| exp.reward.to_f64().unwrap_or(0.0))
            .sum::<f64>() / experiences.len() as f64;

        // Apply difficulty scaling
        let difficulty_factor = scenario.expected_difficulty.to_f64().unwrap_or(0.5);
        let scaled_performance = avg_reward * (1.0 + difficulty_factor);

        Ok(T::from(scaled_performance).unwrap())
    }

    /// Update agent pool with optimized strategies
    fn update_agent_pool(&mut self, optimization_result: &OptimizationResult<T>) -> Result<()> {
        // Apply elite selection and mutation
        let elite_count = (self.config.elite_ratio.to_f64().unwrap() * 
                          self.agent_pool.len() as f64) as usize;
        
        // Keep top performing agents unchanged
        // Mutate remaining agents based on optimization results
        for i in elite_count..self.agent_pool.len() {
            // Apply mutations to agent configurations
            // This is a simplified approach - in practice would be more sophisticated
            self.mutate_agent_strategy(i, &optimization_result.best_strategies)?;
        }

        Ok(())
    }

    /// Apply strategy mutations to agent
    fn mutate_agent_strategy(
        &mut self,
        agent_idx: usize,
        best_strategies: &[AgentStrategy<T>],
    ) -> Result<()> {
        // Simplified mutation - in practice would modify actual agent parameters
        // This would involve updating reasoning templates, action selection strategies, etc.
        Ok(())
    }

    /// Get current learning statistics
    pub fn get_learning_stats(&self) -> &LearningStats {
        &self.stats
    }

    /// Get current generation number
    pub fn get_generation(&self) -> usize {
        self.generation
    }
}

/// Result of running a single generation
#[derive(Debug, Clone)]
pub struct GenerationResult<T: Float> {
    pub generation: usize,
    pub episodes_run: usize,
    pub avg_performance: T,
    pub best_performance: T,
    pub diversity_score: T,
    pub convergence_score: T,
    pub generation_time: Duration,
}

/// Data from a single self-play episode
#[derive(Debug, Clone)]
pub struct EpisodeData<T: Float> {
    pub scenario_id: Uuid,
    pub agent_ids: Vec<usize>,
    pub experiences: Vec<Experience<T>>,
    pub total_reward: T,
    pub performance_score: T,
    pub episode_length: usize,
}

/// Result of a single agent step
#[derive(Debug, Clone)]
pub struct AgentStepResult<T: Float> {
    pub state: State<T>,
    pub action: Action<T>,
    pub thoughts: Thoughts,
    pub done: bool,
}

/// Result of peer evaluation
#[derive(Debug, Clone)]
pub struct EvaluationResult<T: Float> {
    pub reward: T,
    pub next_state: State<T>,
    pub evaluation_confidence: T,
}

/// Result of strategy optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult<T: Float> {
    pub best_performance: T,
    pub diversity_score: T,
    pub convergence_score: T,
    pub best_strategies: Vec<AgentStrategy<T>>,
}

// Implementations for supporting structures would continue here...
// This is a comprehensive foundation for the GSPO framework

impl<T: Float> ScenarioGenerator<T> {
    pub fn new() -> Result<Self> {
        let templates = Self::create_default_templates();
        Ok(Self {
            templates,
            generation_params: GenerationParams {
                diversity_weight: T::from(0.3).unwrap(),
                difficulty_progression: T::from(0.05).unwrap(),
                noise_level: T::from(0.1).unwrap(),
                quality_threshold: T::from(0.6).unwrap(),
            },
            scenario_history: VecDeque::new(),
        })
    }

    fn create_default_templates() -> Vec<ScenarioTemplate<T>> {
        vec![
            ScenarioTemplate {
                name: "Standard Interview".to_string(),
                scenario_type: ScenarioType::Standard,
                parameter_ranges: HashMap::from([
                    ("stress_level".to_string(), (T::from(0.2).unwrap(), T::from(0.8).unwrap())),
                    ("question_intensity".to_string(), (T::from(0.3).unwrap(), T::from(0.9).unwrap())),
                ]),
                required_modalities: vec![ModalityType::Text, ModalityType::Vision, ModalityType::Audio],
                difficulty: T::from(0.5).unwrap(),
                weight: T::from(1.0).unwrap(),
            },
            // Add more templates...
        ]
    }

    pub fn generate_batch(&mut self, count: usize) -> Result<Vec<GeneratedScenario<T>>> {
        let mut scenarios = Vec::new();
        for _ in 0..count {
            scenarios.push(self.generate_scenario()?);
        }
        Ok(scenarios)
    }

    fn generate_scenario(&mut self) -> Result<GeneratedScenario<T>> {
        // Select template based on weights and diversity
        let template = self.select_template()?;
        
        // Generate parameters within template ranges
        let mut parameters = HashMap::new();
        for (param_name, (min_val, max_val)) in &template.parameter_ranges {
            let range = max_val.to_f64().unwrap() - min_val.to_f64().unwrap();
            let value = min_val.to_f64().unwrap() + fastrand::f64() * range;
            parameters.insert(param_name.clone(), T::from(value).unwrap());
        }

        // Generate observations based on parameters
        let observations = self.generate_observations(&template, &parameters)?;

        let scenario = GeneratedScenario {
            id: Uuid::new_v4(),
            scenario_type: template.scenario_type.clone(),
            parameters,
            observations,
            expected_difficulty: template.difficulty,
            ground_truth: Some(if fastrand::bool() { Decision::Truth } else { Decision::Deception }),
            quality_score: T::from(0.8).unwrap(), // Would be calculated based on generation quality
            created_at: Utc::now(),
        };

        // Add to history for diversity tracking
        if self.scenario_history.len() >= 1000 {
            self.scenario_history.pop_front();
        }
        self.scenario_history.push_back(scenario.clone());

        Ok(scenario)
    }

    fn select_template(&self) -> Result<&ScenarioTemplate<T>> {
        // Weighted random selection with diversity consideration
        let total_weight: f64 = self.templates.iter()
            .map(|t| t.weight.to_f64().unwrap_or(0.0))
            .sum();
        
        let mut rng = fastrand::f64() * total_weight;
        for template in &self.templates {
            rng -= template.weight.to_f64().unwrap_or(0.0);
            if rng <= 0.0 {
                return Ok(template);
            }
        }
        
        // Fallback to first template
        self.templates.first()
            .ok_or_else(|| VeritasError::learning("No scenario templates available"))
    }

    fn generate_observations(
        &self,
        template: &ScenarioTemplate<T>,
        parameters: &HashMap<String, T>,
    ) -> Result<Observations<T>> {
        // Generate synthetic observations based on template and parameters
        // This is simplified - in practice would use more sophisticated generation
        
        let stress_level = parameters.get("stress_level")
            .unwrap_or(&T::from(0.5).unwrap())
            .to_f64().unwrap_or(0.5);

        Ok(Observations {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            vision: Some(VisionObservation {
                face_detected: true,
                micro_expressions: if stress_level > 0.6 {
                    vec!["tension".to_string(), "eye_movement".to_string()]
                } else {
                    vec![]
                },
                gaze_patterns: vec!["avoidant".to_string()],
                facial_landmarks: vec![(0.5, 0.5)], // Simplified
            }),
            audio: Some(AudioObservation {
                pitch_variations: vec![T::from(stress_level).unwrap()],
                stress_indicators: if stress_level > 0.7 {
                    vec!["voice_tremor".to_string()]
                } else {
                    vec![]
                },
                voice_quality: T::from(1.0 - stress_level * 0.3).unwrap(),
                speaking_rate: T::from(150.0 + stress_level * 50.0).unwrap(),
            }),
            text: Some(TextObservation {
                content: "I was not involved in the incident".to_string(),
                linguistic_features: vec!["denial".to_string(), "distancing".to_string()],
                sentiment_score: -0.2,
                deception_indicators: if stress_level > 0.6 {
                    vec!["hedging".to_string(), "qualifiers".to_string()]
                } else {
                    vec![]
                },
            }),
            physiological: Some(PhysiologicalObservation {
                stress_level: T::from(stress_level).unwrap(),
                arousal_level: T::from(stress_level * 0.8).unwrap(),
                heart_rate_variability: T::from(0.5 - stress_level * 0.3).unwrap(),
            }),
            context: ObservationContext {
                environment: format!("{:?}", template.scenario_type),
                subject_id: Some("test_subject".to_string()),
                session_id: Some(Uuid::new_v4().to_string()),
                interviewer_id: Some("interviewer_1".to_string()),
            },
        })
    }
}

impl<T: Float> ExperienceBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            priorities: VecDeque::with_capacity(capacity),
            stats: BufferStats::default(),
        }
    }

    pub fn add_experience(&mut self, experience: Experience<T>) {
        // Calculate priority based on reward magnitude and uncertainty
        let priority = experience.reward.abs() + T::from(0.1).unwrap();
        
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }
        
        self.buffer.push_back(experience);
        self.priorities.push_back(priority);
        self.stats.total_added += 1;
    }

    pub fn sample_batch(&mut self, batch_size: usize) -> Vec<Experience<T>> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let mut batch = Vec::with_capacity(batch_size);
        let actual_batch_size = batch_size.min(self.buffer.len());
        
        // Priority-based sampling
        for _ in 0..actual_batch_size {
            let idx = self.sample_priority_index();
            if let Some(experience) = self.buffer.get(idx) {
                batch.push(experience.clone());
            }
        }
        
        self.stats.total_sampled += batch.len();
        batch
    }

    fn sample_priority_index(&self) -> usize {
        let total_priority: f64 = self.priorities.iter()
            .map(|p| p.to_f64().unwrap_or(0.0))
            .sum();
        
        let mut rng = fastrand::f64() * total_priority;
        for (i, priority) in self.priorities.iter().enumerate() {
            rng -= priority.to_f64().unwrap_or(0.0);
            if rng <= 0.0 {
                return i;
            }
        }
        
        self.priorities.len().saturating_sub(1)
    }
}

impl<T: Float> StrategyOptimizer<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            strategies: HashMap::new(),
            algorithm: OptimizationAlgorithm {
                algorithm_type: AlgorithmType::Genetic,
                parameters: HashMap::new(),
                history: Vec::new(),
            },
            performance_tracker: PerformanceTracker {
                performance_history: HashMap::new(),
                leaderboard: Vec::new(),
                trends: HashMap::new(),
            },
        })
    }

    pub fn optimize_strategies(
        &mut self,
        experience_buffer: &mut ExperienceBuffer<T>,
        steps: usize,
    ) -> Result<OptimizationResult<T>> {
        // Sample experiences for optimization
        let experiences = experience_buffer.sample_batch(32);
        
        // Perform optimization steps
        for step in 0..steps {
            self.optimization_step(&experiences, step)?;
        }

        // Calculate results
        let best_performance = T::from(0.85).unwrap(); // Placeholder
        let diversity_score = T::from(0.6).unwrap();
        let convergence_score = T::from(0.7).unwrap();
        let best_strategies = self.get_top_strategies(5);

        Ok(OptimizationResult {
            best_performance,
            diversity_score,
            convergence_score,
            best_strategies,
        })
    }

    fn optimization_step(&mut self, experiences: &[Experience<T>], step: usize) -> Result<()> {
        // Simplified optimization step
        // In practice, this would implement sophisticated optimization algorithms
        Ok(())
    }

    fn get_top_strategies(&self, count: usize) -> Vec<AgentStrategy<T>> {
        // Return top performing strategies
        self.strategies.values()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .take(count)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gspo_framework_creation() {
        let config: GSPOConfig<f32> = GSPOConfig::default();
        let framework = GSPOFramework::new(config);
        assert!(framework.is_ok());
    }

    #[test]
    fn test_scenario_generator() {
        let mut generator: ScenarioGenerator<f32> = ScenarioGenerator::new().unwrap();
        let scenarios = generator.generate_batch(5);
        assert!(scenarios.is_ok());
        assert_eq!(scenarios.unwrap().len(), 5);
    }

    #[test]
    fn test_experience_buffer() {
        let mut buffer: ExperienceBuffer<f32> = ExperienceBuffer::new(100);
        
        let experience = Experience {
            state: State {
                features: vec![0.5, 0.3, 0.8],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.7,
            },
            action: Action {
                id: Uuid::new_v4(),
                action_type: ActionType::MakeDecision,
                parameters: HashMap::new(),
                expected_outcome: "Test".to_string(),
                confidence: 0.8,
                explanation: "Test action".to_string(),
                timestamp: Utc::now(),
                decision: Some(Decision::Truth),
            },
            reward: 0.9,
            next_state: State {
                features: vec![0.6, 0.4, 0.9],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.8,
            },
            done: true,
            metadata: HashMap::new(),
        };

        buffer.add_experience(experience);
        assert_eq!(buffer.stats.total_added, 1);

        let batch = buffer.sample_batch(1);
        assert_eq!(batch.len(), 1);
    }
}