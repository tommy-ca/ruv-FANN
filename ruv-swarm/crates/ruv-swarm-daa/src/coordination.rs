//! Coordination modules for DAA agents

use crate::*;
use std::collections::HashMap;

/// Coordination memory for managing agent interactions
pub struct CoordinationMemory {
    pub shared_state: HashMap<String, serde_json::Value>,
    pub agent_locations: HashMap<String, AgentLocation>,
    pub coordination_history: Vec<CoordinationEvent>,
}

/// Agent location in coordination space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLocation {
    pub agent_id: String,
    pub position: [f64; 3], // 3D coordination space
    pub capabilities: Vec<String>,
    pub current_task: Option<String>,
    pub availability: f64,
}

/// Coordination event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub agent_id: String,
    pub task_id: String,
    pub event_type: String,
    pub metadata: serde_json::Value,
}

/// Types of coordination events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEventType {
    TaskAssignment,
    KnowledgeSharing,
    ConflictResolution,
    ResourceAllocation,
    PerformanceEvaluation,
}

impl CoordinationMemory {
    pub fn new() -> Self {
        Self {
            shared_state: HashMap::new(),
            agent_locations: HashMap::new(),
            coordination_history: Vec::new(),
        }
    }

    /// Store a coordination event
    pub async fn store_event(&mut self, event: CoordinationEvent) -> Result<(), DAAError> {
        self.coordination_history.push(event);
        Ok(())
    }

    /// Get the count of events
    pub async fn get_event_count(&self) -> Result<usize, DAAError> {
        Ok(self.coordination_history.len())
    }

    /// Get recent events
    pub async fn get_recent_events(
        &self,
        limit: usize,
    ) -> Result<Vec<CoordinationEvent>, DAAError> {
        let start = self.coordination_history.len().saturating_sub(limit);
        Ok(self.coordination_history[start..].to_vec())
    }
}
