//! MCP Tools Module for Veritas-Nexus
//!
//! This module contains all MCP tool implementations for the Veritas lie detection system.
//! Tools provide the core functionality exposed through the MCP interface including model
//! management, training, inference, and monitoring capabilities.

pub mod model_tools;
pub mod training_tools;
pub mod inference_tools;
pub mod monitoring_tools;

// Re-export tool handlers for easier access
pub use model_tools::ModelToolsHandler;
pub use training_tools::TrainingToolsHandler;
pub use inference_tools::InferenceToolsHandler;
pub use monitoring_tools::MonitoringToolsHandler;

use anyhow::Result;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::mcp::{McpResult, RequestMetadata, ResponseMetadata, McpResponse};

/// Base trait for all MCP tool handlers
pub trait ToolHandler: Send + Sync {
    /// Get the name of this tool handler
    fn name(&self) -> &'static str;
    
    /// Get the version of this tool handler
    fn version(&self) -> &'static str;
    
    /// Get a list of tools provided by this handler
    fn get_tools(&self) -> Vec<ToolDefinition>;
    
    /// Initialize the tool handler
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the tool handler
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
    
    /// Get handler statistics
    async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }
}

/// MCP tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub examples: Vec<ToolExample>,
}

/// Tool usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExample {
    pub description: String,
    pub input: serde_json::Value,
    pub expected_output: Option<serde_json::Value>,
}

/// Common tool response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub warnings: Vec<String>,
    pub metadata: ResponseMetadata,
}

impl<T> ToolResponse<T> {
    pub fn success(data: T, metadata: ResponseMetadata) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            warnings: Vec::new(),
            metadata,
        }
    }
    
    pub fn error(error: String, metadata: ResponseMetadata) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            warnings: Vec::new(),
            metadata,
        }
    }
    
    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}

impl<T> IntoResponse for ToolResponse<T>
where
    T: Serialize,
{
    fn into_response(self) -> axum::response::Response {
        axum::Json(self).into_response()
    }
}

/// Utility functions for tool implementations
pub mod utils {
    use super::*;
    use uuid::Uuid;
    use std::time::Instant;

    /// Create response metadata for a tool operation
    pub fn create_response_metadata(
        request_id: Uuid,
        start_time: Instant,
        model_version: Option<String>,
    ) -> ResponseMetadata {
        ResponseMetadata {
            request_id,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            server_version: env!("CARGO_PKG_VERSION").to_string(),
            model_version,
        }
    }

    /// Validate required fields in input data
    pub fn validate_required_fields(
        data: &serde_json::Value,
        required_fields: &[&str],
    ) -> McpResult<()> {
        if let Some(obj) = data.as_object() {
            for field in required_fields {
                if !obj.contains_key(*field) {
                    return Err(crate::mcp::McpError::InsufficientInput {
                        reason: format!("Missing required field: {}", field),
                    });
                }
            }
            Ok(())
        } else {
            Err(crate::mcp::McpError::InsufficientInput {
                reason: "Input must be a JSON object".to_string(),
            })
        }
    }

    /// Extract string field from JSON value
    pub fn extract_string_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> McpResult<String> {
        data.get(field_name)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| crate::mcp::McpError::InsufficientInput {
                reason: format!("Missing or invalid string field: {}", field_name),
            })
    }

    /// Extract optional string field from JSON value
    pub fn extract_optional_string_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> Option<String> {
        data.get(field_name)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Extract integer field from JSON value
    pub fn extract_int_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> McpResult<i64> {
        data.get(field_name)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| crate::mcp::McpError::InsufficientInput {
                reason: format!("Missing or invalid integer field: {}", field_name),
            })
    }

    /// Extract optional integer field from JSON value
    pub fn extract_optional_int_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> Option<i64> {
        data.get(field_name).and_then(|v| v.as_i64())
    }

    /// Extract float field from JSON value
    pub fn extract_float_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> McpResult<f64> {
        data.get(field_name)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| crate::mcp::McpError::InsufficientInput {
                reason: format!("Missing or invalid float field: {}", field_name),
            })
    }

    /// Extract optional float field from JSON value
    pub fn extract_optional_float_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> Option<f64> {
        data.get(field_name).and_then(|v| v.as_f64())
    }

    /// Extract boolean field from JSON value
    pub fn extract_bool_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> McpResult<bool> {
        data.get(field_name)
            .and_then(|v| v.as_bool())
            .ok_or_else(|| crate::mcp::McpError::InsufficientInput {
                reason: format!("Missing or invalid boolean field: {}", field_name),
            })
    }

    /// Extract optional boolean field from JSON value
    pub fn extract_optional_bool_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> Option<bool> {
        data.get(field_name).and_then(|v| v.as_bool())
    }

    /// Extract array field from JSON value
    pub fn extract_array_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> McpResult<Vec<serde_json::Value>> {
        data.get(field_name)
            .and_then(|v| v.as_array())
            .map(|arr| arr.clone())
            .ok_or_else(|| crate::mcp::McpError::InsufficientInput {
                reason: format!("Missing or invalid array field: {}", field_name),
            })
    }

    /// Extract optional array field from JSON value
    pub fn extract_optional_array_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> Option<Vec<serde_json::Value>> {
        data.get(field_name)
            .and_then(|v| v.as_array())
            .map(|arr| arr.clone())
    }

    /// Extract object field from JSON value
    pub fn extract_object_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> McpResult<serde_json::Map<String, serde_json::Value>> {
        data.get(field_name)
            .and_then(|v| v.as_object())
            .map(|obj| obj.clone())
            .ok_or_else(|| crate::mcp::McpError::InsufficientInput {
                reason: format!("Missing or invalid object field: {}", field_name),
            })
    }

    /// Extract optional object field from JSON value
    pub fn extract_optional_object_field(
        data: &serde_json::Value,
        field_name: &str,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        data.get(field_name)
            .and_then(|v| v.as_object())
            .map(|obj| obj.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_response_creation() {
        let metadata = ResponseMetadata {
            request_id: uuid::Uuid::new_v4(),
            processing_time_ms: 100,
            server_version: "0.1.0".to_string(),
            model_version: Some("detector-v1".to_string()),
        };

        let response = ToolResponse::success("test data", metadata.clone());
        assert!(response.success);
        assert_eq!(response.data.unwrap(), "test data");
        assert!(response.error.is_none());

        let error_response = ToolResponse::<String>::error("test error".to_string(), metadata);
        assert!(!error_response.success);
        assert!(error_response.data.is_none());
        assert_eq!(error_response.error.unwrap(), "test error");
    }

    #[test]
    fn test_validate_required_fields() {
        let data = json!({
            "field1": "value1",
            "field2": 42
        });

        let result = utils::validate_required_fields(&data, &["field1", "field2"]);
        assert!(result.is_ok());

        let result = utils::validate_required_fields(&data, &["field1", "field3"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_string_field() {
        let data = json!({
            "name": "test",
            "number": 42
        });

        assert_eq!(utils::extract_string_field(&data, "name").unwrap(), "test");
        assert!(utils::extract_string_field(&data, "number").is_err());
        assert!(utils::extract_string_field(&data, "missing").is_err());
    }

    #[test]
    fn test_extract_optional_fields() {
        let data = json!({
            "string_field": "test",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": true
        });

        assert_eq!(utils::extract_optional_string_field(&data, "string_field"), Some("test".to_string()));
        assert_eq!(utils::extract_optional_int_field(&data, "int_field"), Some(42));
        assert_eq!(utils::extract_optional_float_field(&data, "float_field"), Some(3.14));
        assert_eq!(utils::extract_optional_bool_field(&data, "bool_field"), Some(true));
        
        assert_eq!(utils::extract_optional_string_field(&data, "missing"), None);
    }
}