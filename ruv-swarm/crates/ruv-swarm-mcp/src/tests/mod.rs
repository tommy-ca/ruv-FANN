//! Unit tests for ruv-swarm-mcp
//! 
//! This module contains comprehensive tests for the MCP server functionality,
//! organized by test type and focus area.

use crate::*;
use std::sync::Arc;
use ruv_swarm_core::SwarmConfig;
use uuid::Uuid;

#[test]
fn test_version_info() {
    assert_eq!(env!("CARGO_PKG_VERSION"), "1.0.5");
}

#[tokio::test]
async fn test_basic_mcp_server_creation() {
    // Use unique database for this test
    std::env::set_var("RUV_SWARM_DB_PATH", format!("test_basic_server_{}.db", Uuid::new_v4()));
    let orchestrator = Arc::new(SwarmOrchestrator::new(SwarmConfig::default()).await);
    let mcp_config = McpConfig::default();

    let server = McpServer::new(orchestrator, mcp_config);
    
    // Verify server creation succeeded
    assert_eq!(server.state.config.bind_addr.port(), 3000);
    assert_eq!(server.state.config.max_connections, 100);
}

// Core unit tests for each module
mod integration_tests;
mod security_tests;