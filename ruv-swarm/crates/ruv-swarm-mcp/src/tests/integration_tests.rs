//! Integration tests for RUV-Swarm MCP server functionality
//!
//! These tests verify that different components work together correctly
//! and that the MCP protocol implementation functions as expected.

use std::sync::Arc;
use std::time::Duration;

use ruv_swarm_core::SwarmConfig;
use uuid::Uuid;
use crate::{
    orchestrator::SwarmOrchestrator, McpConfig, McpRequest, McpResponse, McpServer,
};
use serde_json::json;
use tokio::time::timeout;

/// Test server creation
#[tokio::test]
async fn test_server_creation() {
    let orchestrator = Arc::new(SwarmOrchestrator::new(SwarmConfig::default()).await);
    let mcp_config = McpConfig::default();

    let server = McpServer::new(orchestrator, mcp_config);

    // Server should be created successfully - verify it has tools registered
    assert!(server.state.tools.count() > 0);
}

/// Test MCP request/response structures
#[test]
fn test_mcp_request_serialization() {
    let request = McpRequest {
        jsonrpc: "2.0".to_string(),
        method: "initialize".to_string(),
        params: Some(json!({})),
        id: Some(json!(1)),
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("\"jsonrpc\":\"2.0\""));
    assert!(json.contains("\"method\":\"initialize\""));

    let deserialized: McpRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.method, "initialize");
}

/// Test MCP response serialization
#[test]
fn test_mcp_response_serialization() {
    let response = McpResponse::success(Some(json!(1)), json!({"status": "ok"}));

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"jsonrpc\":\"2.0\""));
    assert!(json.contains("\"result\""));
    assert!(!json.contains("\"error\""));

    let error_response = McpResponse::error(Some(json!(2)), -32600, "Invalid request".to_string());

    let error_json = serde_json::to_string(&error_response).unwrap();
    assert!(error_json.contains("\"error\""));
    assert!(!error_json.contains("\"result\""));
}

/// Test tool registry
#[test]
fn test_tool_registry() {
    use crate::tools::{register_tools, ToolRegistry};

    let registry = ToolRegistry::new();
    register_tools(&registry);

    assert!(registry.count() > 0);

    // Check specific tools exist
    assert!(registry.get("ruv-swarm.spawn").is_some());
    assert!(registry.get("ruv-swarm.orchestrate").is_some());
    assert!(registry.get("ruv-swarm.query").is_some());

    let tools = registry.list_tools();
    assert!(tools.len() > 5);
}

/// Test orchestrator agent spawning
#[tokio::test]
async fn test_orchestrator_spawn_agent() {
    use crate::types::{AgentCapabilities, AgentType};

    // Use unique database for this test
    std::env::set_var("RUV_SWARM_DB_PATH", format!("test_spawn_agent_{}.db", Uuid::new_v4()));
    let swarm_config = SwarmConfig::default();
    let orchestrator = SwarmOrchestrator::new(swarm_config).await;

    let agent_id = orchestrator
        .spawn_agent(
            AgentType::Researcher,
            "Test Agent".to_string(),
            AgentCapabilities::default(),
        )
        .await
        .unwrap();

    assert!(!agent_id.is_nil());

    // List agents
    let agents = orchestrator.list_agents().await.unwrap();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].id, agent_id);
}

/// Test orchestrator task creation
#[tokio::test]
async fn test_orchestrator_task_creation() {

    let swarm_config = SwarmConfig::default();
    let orchestrator = SwarmOrchestrator::new(swarm_config).await;

    let task_id = orchestrator
        .create_task(
            "research".to_string(),
            "Test research task".to_string(),
            vec![],
            "high_priority".to_string(),
        )
        .await
        .unwrap();

    assert!(!task_id.is_nil());
}

/// Test swarm state query
#[tokio::test]
async fn test_swarm_state_query() {
    use crate::types::{AgentCapabilities, AgentType};

    // Use unique database for this test
    std::env::set_var("RUV_SWARM_DB_PATH", format!("test_swarm_state_{}.db", Uuid::new_v4()));
    let swarm_config = SwarmConfig::default();
    let orchestrator = SwarmOrchestrator::new(swarm_config).await;

    // Spawn some agents
    for i in 0..3 {
        orchestrator
            .spawn_agent(
                AgentType::Coder,
                format!("Agent {i}"),
                AgentCapabilities::default(),
            )
            .await
            .unwrap();
    }

    let state = orchestrator.get_swarm_state().await.unwrap();
    assert_eq!(state.total_agents, 3);
    assert_eq!(state.agents.len(), 3);
}

/// Test metrics retrieval
#[tokio::test]
async fn test_metrics() {
    // Use unique database for this test
    std::env::set_var("RUV_SWARM_DB_PATH", format!("test_metrics_{}.db", Uuid::new_v4()));
    let swarm_config = SwarmConfig::default();
    let orchestrator = SwarmOrchestrator::new(swarm_config).await;

    let metrics = orchestrator.get_performance_metrics().await.unwrap();
    assert_eq!(metrics.success_rate, 1.0);
    assert_eq!(metrics.total_tasks, 0);
}

/// Test optimization recommendations
#[tokio::test]
async fn test_optimization_recommendations() {
    // Use unique database for this test
    std::env::set_var("RUV_SWARM_DB_PATH", format!("test_optimization_{}.db", Uuid::new_v4()));
    let swarm_config = SwarmConfig::default();
    let orchestrator = SwarmOrchestrator::new(swarm_config).await;

    let recommendations = orchestrator.optimize_performance("throughput".to_string(), 0.8).await.unwrap();

    // Should have at least one recommendation for low utilization
    assert!(!recommendations.is_empty());
    assert!(recommendations
        .iter()
        .any(|r| r.recommendation_type == "scaling"));
}