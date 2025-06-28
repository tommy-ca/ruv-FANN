//! Agent framework for the Veritas Nexus system
//!
//! This module implements the ReAct (Reasoning and Acting) agent framework
//! following the pattern: Observe -> Think -> Act -> Explain
//!
//! The framework consists of several key components:
//! - ReactAgent: Main agent trait implementing the ReAct loop
//! - ReasoningEngine: Handles thought generation and reasoning
//! - ActionEngine: Manages action selection and execution
//! - Memory: Provides multi-layered memory system (short-term, long-term, episodic)

use async_trait::async_trait;
use std::sync::Arc;
use num_traits::Float;

use crate::error::Result;
use crate::types::*;

pub mod react_agent;
pub mod reasoning_engine;
pub mod action_engine;
pub mod memory;

pub use react_agent::*;
pub use reasoning_engine::*;
pub use action_engine::*;
pub use memory::*;

/// Core trait for ReAct agents implementing the Observe -> Think -> Act -> Explain pattern
#[async_trait]
pub trait ReactAgent<T: Float>: Send + Sync {
    /// Observe and process new information
    /// 
    /// This is the first step in the ReAct loop where the agent receives
    /// and processes multi-modal observations.
    fn observe(&mut self, observations: Observations<T>) -> Result<()>;

    /// Think and generate reasoning thoughts
    /// 
    /// This step involves analyzing observations, retrieving relevant memories,
    /// and generating structured thoughts about the current situation.
    fn think(&mut self) -> Result<Thoughts>;

    /// Act based on observations and thoughts
    /// 
    /// This step selects and executes actions based on the reasoning process,
    /// resulting in decisions about deception detection.
    fn act(&mut self) -> Result<Action<T>>;

    /// Explain the reasoning process
    /// 
    /// This step generates human-readable explanations of the entire
    /// reasoning process for transparency and explainability.
    fn explain(&self) -> ReasoningTrace;

    /// Get the agent's current configuration
    fn config(&self) -> &DetectorConfig<T>;

    /// Update the agent's configuration
    fn update_config(&mut self, config: DetectorConfig<T>) -> Result<()>;

    /// Reset the agent's state
    fn reset(&mut self) -> Result<()>;

    /// Get agent statistics and performance metrics
    fn get_stats(&self) -> AgentStats;
}

/// Agent performance statistics
#[derive(Debug, Clone, Default)]
pub struct AgentStats {
    /// Total number of observations processed
    pub observations_processed: usize,
    /// Total number of decisions made
    pub decisions_made: usize,
    /// Average reasoning time per decision
    pub avg_reasoning_time_ms: f64,
    /// Average confidence in decisions
    pub avg_confidence: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
    /// Error count
    pub error_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Short-term memory utilization (0.0 to 1.0)
    pub short_term_utilization: f64,
    /// Long-term memory utilization (0.0 to 1.0)
    pub long_term_utilization: f64,
    /// Episodic memory utilization (0.0 to 1.0)
    pub episodic_utilization: f64,
    /// Total memory entries
    pub total_entries: usize,
}

/// Factory function to create a ReAct agent with the given configuration
pub fn create_react_agent<T: Float>(config: DetectorConfig<T>) -> Result<Box<dyn ReactAgent<T>>> {
    let memory = Arc::new(Memory::new(config.memory_config.clone())?);
    let reasoning_engine = Arc::new(ReasoningEngine::new(config.reasoning_config.clone())?);
    let action_engine = Arc::new(ActionEngine::new(config.action_config.clone())?);
    
    let agent = DefaultReactAgent::new(config, memory, reasoning_engine, action_engine)?;
    Ok(Box::new(agent))
}

/// Create a ReAct agent with custom components
pub fn create_custom_react_agent<T: Float>(
    config: DetectorConfig<T>,
    memory: Arc<Memory<T>>,
    reasoning_engine: Arc<ReasoningEngine<T>>,
    action_engine: Arc<ActionEngine<T>>,
) -> Result<Box<dyn ReactAgent<T>>> {
    let agent = DefaultReactAgent::new(config, memory, reasoning_engine, action_engine)?;
    Ok(Box::new(agent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_react_agent() {
        let config: DetectorConfig<f32> = DetectorConfig::default();
        let agent = create_react_agent(config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_agent_stats_default() {
        let stats = AgentStats::default();
        assert_eq!(stats.observations_processed, 0);
        assert_eq!(stats.decisions_made, 0);
        assert_eq!(stats.avg_reasoning_time_ms, 0.0);
    }

    #[test]
    fn test_memory_usage_default() {
        let usage = MemoryUsage::default();
        assert_eq!(usage.short_term_utilization, 0.0);
        assert_eq!(usage.long_term_utilization, 0.0);
        assert_eq!(usage.episodic_utilization, 0.0);
        assert_eq!(usage.total_entries, 0);
    }
}