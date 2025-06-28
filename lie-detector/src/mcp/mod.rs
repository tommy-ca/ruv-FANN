//! MCP (Model Context Protocol) Server Implementation for Veritas-Nexus
//!
//! This module provides a comprehensive MCP server for the Veritas-Nexus lie detection system.
//! It exposes tools, resources, and real-time capabilities for neural network-based deception detection.
//!
//! # Features
//!
//! - **Model Management**: Create, load, and manage multi-modal lie detection models
//! - **Training Tools**: Start, monitor, and manage training sessions with GSPO support
//! - **Inference Tools**: Perform real-time and batch lie detection analysis
//! - **Monitoring Tools**: System metrics, performance monitoring, and alerting
//! - **Resource Access**: Model weights, datasets, configurations, and results
//! - **Event Streaming**: Real-time updates for training and inference progress
//! - **Security**: API key authentication, mTLS, and authorization controls
//!
//! # Architecture
//!
//! The MCP server is built using the official Rust MCP SDK and follows a modular design:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    MCP Server (veritas-mcp)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Tools Layer           Resources Layer      Events Layer     │
//! │  ┌─────────────┐      ┌─────────────┐     ┌──────────────┐ │
//! │  │Model Tools  │      │Model Weights│     │Training Events│ │
//! │  │Train Tools  │      │Datasets     │     │Inference Logs │ │
//! │  │Infer Tools  │      │Configs      │     │Monitor Stream │ │
//! │  │Monitor Tools│      │Results      │     │Alert Events   │ │
//! │  └─────────────┘      └─────────────┘     └──────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust
//! use veritas_nexus::mcp::{VeritasServer, ServerConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = ServerConfig {
//!         host: "localhost".to_string(),
//!         port: 3000,
//!         enable_auth: true,
//!         ..Default::default()
//!     };
//!     
//!     let server = VeritasServer::new(config).await?;
//!     server.run().await?;
//!     
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub mod auth;
pub mod events;
pub mod resources;
pub mod server;
pub mod tools;

// Re-export commonly used types
pub use server::{VeritasServer, ServerConfig, ServerState};
pub use events::{EventManager, EventType, VeritasEvent};
pub use auth::{AuthManager, AuthLevel, AuthConfig};

/// Common error types for the MCP server
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },
    
    #[error("Model load failed: {reason}")]
    ModelLoadFailed { reason: String },
    
    #[error("Training failed: {reason}")]
    TrainingFailed { reason: String },
    
    #[error("Invalid dataset: {reason}")]
    InvalidDataset { reason: String },
    
    #[error("Inference timeout: operation exceeded {timeout_ms}ms")]
    InferenceTimeout { timeout_ms: u64 },
    
    #[error("Insufficient input: {reason}")]
    InsufficientInput { reason: String },
    
    #[error("Authentication failed: {reason}")]
    AuthFailed { reason: String },
    
    #[error("Permission denied: {reason}")]
    PermissionDenied { reason: String },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Rate limit exceeded: {limit} requests per {window}")]
    RateLimitExceeded { limit: u32, window: String },
    
    #[error("Internal server error: {message}")]
    Internal { message: String },
}

/// Result type alias for MCP operations
pub type McpResult<T> = Result<T, McpError>;

/// Common request metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    pub request_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub client_id: Option<String>,
    pub auth_level: AuthLevel,
    pub session_id: Option<String>,
}

impl RequestMetadata {
    pub fn new(auth_level: AuthLevel) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            client_id: None,
            auth_level,
            session_id: None,
        }
    }
}

/// Common response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub request_id: Uuid,
    pub processing_time_ms: u64,
    pub server_version: String,
    pub model_version: Option<String>,
}

/// Base response structure for all MCP operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub metadata: ResponseMetadata,
}

impl<T> McpResponse<T> {
    pub fn success(data: T, metadata: ResponseMetadata) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            metadata,
        }
    }
    
    pub fn error(error: String, metadata: ResponseMetadata) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            metadata,
        }
    }
}

/// Server health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub active_models: u32,
    pub active_sessions: u32,
    pub total_requests: u64,
    pub error_rate_percent: f64,
}

/// Server capabilities exposed via MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: Vec<String>,
    pub resources: Vec<String>,
    pub events: Vec<String>,
    pub supported_modalities: Vec<String>,
    pub max_model_size_mb: u64,
    pub max_concurrent_sessions: u32,
    pub supports_streaming: bool,
    pub supports_gspo: bool,
    pub supports_explainability: bool,
}

/// Server version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerVersion {
    pub server_version: String,
    pub mcp_version: String,
    pub sdk_version: String,
    pub build_date: String,
    pub commit_hash: String,
}

/// Initialize the MCP server with default configuration
pub async fn init_server() -> Result<VeritasServer> {
    let config = ServerConfig::default();
    VeritasServer::new(config).await
}

/// Initialize the MCP server with custom configuration
pub async fn init_server_with_config(config: ServerConfig) -> Result<VeritasServer> {
    VeritasServer::new(config).await
}

/// Utility function to get server capabilities
pub fn get_server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        tools: vec![
            "model/list".to_string(),
            "model/load".to_string(),
            "model/create".to_string(),
            "training/start".to_string(),
            "training/status".to_string(),
            "training/stop".to_string(),
            "inference/analyze".to_string(),
            "inference/stream".to_string(),
            "monitor/metrics".to_string(),
            "monitor/alerts".to_string(),
            "config/get".to_string(),
            "config/set".to_string(),
        ],
        resources: vec![
            "models/{model_id}/weights".to_string(),
            "models/{model_id}/config".to_string(),
            "datasets/{dataset_id}/manifest".to_string(),
            "datasets/{dataset_id}/samples".to_string(),
            "results/{analysis_id}/report".to_string(),
            "results/{analysis_id}/media".to_string(),
        ],
        events: vec![
            "training.started".to_string(),
            "training.epoch_completed".to_string(),
            "training.metric_update".to_string(),
            "training.completed".to_string(),
            "training.failed".to_string(),
            "inference.started".to_string(),
            "inference.result".to_string(),
            "inference.stream_update".to_string(),
            "inference.completed".to_string(),
            "system.resource_alert".to_string(),
            "system.model_loaded".to_string(),
            "system.error".to_string(),
        ],
        supported_modalities: vec![
            "vision".to_string(),
            "audio".to_string(),
            "text".to_string(),
            "physiological".to_string(),
        ],
        max_model_size_mb: 2048,
        max_concurrent_sessions: 100,
        supports_streaming: true,
        supports_gspo: true,
        supports_explainability: true,
    }
}

/// Utility function to get server version
pub fn get_server_version() -> ServerVersion {
    ServerVersion {
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        mcp_version: "1.0".to_string(),
        sdk_version: "0.1.0".to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("unknown").to_string(),
        commit_hash: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_capabilities() {
        let capabilities = get_server_capabilities();
        assert!(!capabilities.tools.is_empty());
        assert!(!capabilities.resources.is_empty());
        assert!(!capabilities.events.is_empty());
        assert!(capabilities.supports_streaming);
        assert!(capabilities.supports_gspo);
        assert!(capabilities.supports_explainability);
    }

    #[test]
    fn test_server_version() {
        let version = get_server_version();
        assert!(!version.server_version.is_empty());
        assert!(!version.mcp_version.is_empty());
        assert!(!version.sdk_version.is_empty());
    }

    #[test]
    fn test_request_metadata() {
        let metadata = RequestMetadata::new(AuthLevel::Admin);
        assert_eq!(metadata.auth_level, AuthLevel::Admin);
        assert!(metadata.timestamp <= chrono::Utc::now());
    }

    #[test]
    fn test_mcp_response() {
        let metadata = ResponseMetadata {
            request_id: Uuid::new_v4(),
            processing_time_ms: 100,
            server_version: "0.1.0".to_string(),
            model_version: Some("detector-v1".to_string()),
        };

        let response = McpResponse::success("test data", metadata.clone());
        assert!(response.success);
        assert_eq!(response.data.unwrap(), "test data");
        assert!(response.error.is_none());

        let error_response = McpResponse::<String>::error("test error".to_string(), metadata);
        assert!(!error_response.success);
        assert!(error_response.data.is_none());
        assert_eq!(error_response.error.unwrap(), "test error");
    }
}