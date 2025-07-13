//! Updated MCP Tool definitions with modern naming

use std::sync::Arc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Tool parameter schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub param_type: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Tool definition
#[derive(Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    #[serde(skip)]
    pub handler: Option<Arc<dyn ToolHandler>>,
}

/// Tool handler trait
pub trait ToolHandler: Send + Sync {
    fn handle(&self, params: Value) -> anyhow::Result<Value>;
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("handler", &self.handler.as_ref().map(|_| "<handler>"))
            .finish()
    }
}

/// Tool registry
pub struct ToolRegistry {
    tools: DashMap<String, Tool>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
        }
    }

    pub fn register(&self, tool: Tool) {
        let name = tool.name.clone();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Tool> {
        self.tools.get(name).map(|t| t.clone())
    }

    pub fn list_tools(&self) -> Vec<Tool> {
        self.tools
            .iter()
            .map(|entry| {
                let mut tool = entry.value().clone();
                tool.handler = None;
                tool
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.tools.len()
    }
}

/// Register all tools with updated names
pub fn register_tools(registry: &ToolRegistry) {
    // 1. Swarm initialization
    registry.register(Tool {
        name: "swarm_init".to_string(),
        description: "Initialize a new swarm with specified topology and configuration".to_string(),
        parameters: vec![
            ToolParameter {
                name: "topology".to_string(),
                description: "Swarm topology: mesh, hierarchical, ring, or star".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: Some(vec!["mesh".to_string(), "hierarchical".to_string(), "ring".to_string(), "star".to_string()]),
            },
            ToolParameter {
                name: "maxAgents".to_string(),
                description: "Maximum number of agents (default: 5)".to_string(),
                param_type: "number".to_string(),
                required: false,
                default: Some(json!(5)),
                enum_values: None,
            },
            ToolParameter {
                name: "strategy".to_string(),
                description: "Coordination strategy: balanced, specialized, or adaptive".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("balanced")),
                enum_values: Some(vec!["balanced".to_string(), "specialized".to_string(), "adaptive".to_string()]),
            },
        ],
        handler: None,
    });

    // 2. Agent spawn
    registry.register(Tool {
        name: "agent_spawn".to_string(),
        description: "Spawn a new agent in the swarm with specific capabilities".to_string(),
        parameters: vec![
            ToolParameter {
                name: "type".to_string(),
                description: "Type of agent to spawn".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: Some(vec![
                    "researcher".to_string(),
                    "coder".to_string(),
                    "analyst".to_string(),
                    "tester".to_string(),
                    "coordinator".to_string(),
                    "architect".to_string(),
                ]),
            },
            ToolParameter {
                name: "name".to_string(),
                description: "Optional custom name for the agent".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "capabilities".to_string(),
                description: "Array of capabilities for the agent".to_string(),
                param_type: "array".to_string(),
                required: false,
                default: Some(json!([])),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Continue with other tools...
    register_additional_tools(registry);
}

fn register_additional_tools(registry: &ToolRegistry) {
    // 3. Task orchestration
    registry.register(Tool {
        name: "task_orchestrate".to_string(),
        description: "Orchestrate complex tasks across the swarm using various strategies".to_string(),
        parameters: vec![
            ToolParameter {
                name: "task".to_string(),
                description: "Task description or objective".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "priority".to_string(),
                description: "Task priority level".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("medium")),
                enum_values: Some(vec!["low".to_string(), "medium".to_string(), "high".to_string(), "critical".to_string()]),
            },
            ToolParameter {
                name: "strategy".to_string(),
                description: "Execution strategy".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("adaptive")),
                enum_values: Some(vec!["parallel".to_string(), "sequential".to_string(), "adaptive".to_string()]),
            },
        ],
        handler: None,
    });

    // 4. Memory management
    registry.register(Tool {
        name: "memory_usage".to_string(),
        description: "Manage session memory storage for persistent coordination".to_string(),
        parameters: vec![
            ToolParameter {
                name: "action".to_string(),
                description: "Memory operation: store, retrieve, list, delete".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: Some(vec!["store".to_string(), "retrieve".to_string(), "list".to_string(), "delete".to_string()]),
            },
            ToolParameter {
                name: "key".to_string(),
                description: "Memory key for operations".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "value".to_string(),
                description: "Value to store (JSON)".to_string(),
                param_type: "object".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Add remaining tools...
    register_monitoring_tools(registry);
    register_neural_tools(registry);
}

fn register_monitoring_tools(registry: &ToolRegistry) {
    // Swarm status
    registry.register(Tool {
        name: "swarm_status".to_string(),
        description: "Get comprehensive swarm status and health information".to_string(),
        parameters: vec![
            ToolParameter {
                name: "verbose".to_string(),
                description: "Include detailed agent information".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                default: Some(json!(false)),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Swarm monitor
    registry.register(Tool {
        name: "swarm_monitor".to_string(),
        description: "Monitor swarm activity in real-time".to_string(),
        parameters: vec![
            ToolParameter {
                name: "duration".to_string(),
                description: "Monitoring duration in seconds".to_string(),
                param_type: "number".to_string(),
                required: false,
                default: Some(json!(10)),
                enum_values: None,
            },
        ],
        handler: None,
    });
}

fn register_neural_tools(registry: &ToolRegistry) {
    // Neural status
    registry.register(Tool {
        name: "neural_status".to_string(),
        description: "Get neural agent status and performance metrics".to_string(),
        parameters: vec![
            ToolParameter {
                name: "agentId".to_string(),
                description: "Specific neural agent ID (optional)".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Add more neural tools as needed...
}