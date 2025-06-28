//! MCP Resources Module for Veritas-Nexus
//!
//! This module contains all MCP resource implementations for accessing and managing
//! model weights, datasets, configurations, and analysis results through the MCP interface.

pub mod model_resources;
pub mod data_resources;

// Re-export resource handlers for easier access
pub use model_resources::ModelResourcesHandler;
pub use data_resources::DataResourcesHandler;

use anyhow::Result;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::mcp::{McpResult, RequestMetadata, ResponseMetadata};

/// Base trait for all MCP resource handlers
pub trait ResourceHandler: Send + Sync {
    /// Get the name of this resource handler
    fn name(&self) -> &'static str;
    
    /// Get the version of this resource handler
    fn version(&self) -> &'static str;
    
    /// Get a list of resources provided by this handler
    fn get_resources(&self) -> Vec<ResourceDefinition>;
    
    /// Initialize the resource handler
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the resource handler
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
    
    /// Get handler statistics
    async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }
}

/// MCP resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDefinition {
    pub uri: String,
    pub name: String,
    pub mime_type: String,
    pub description: String,
    pub size_bytes: Option<u64>,
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
}

/// Resource access response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceResponse {
    pub uri: String,
    pub mime_type: String,
    pub content: ResourceContent,
    pub metadata: ResourceMetadata,
}

/// Resource content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResourceContent {
    Text { data: String },
    Binary { data: Vec<u8> },
    Json { data: serde_json::Value },
    Stream { chunk_size: usize, total_size: Option<u64> },
}

/// Resource metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    pub size_bytes: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub etag: Option<String>,
    pub cache_control: Option<String>,
    pub content_encoding: Option<String>,
}

/// Resource access request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub uri: String,
    pub range: Option<ByteRange>,
    pub if_none_match: Option<String>,
    pub if_modified_since: Option<chrono::DateTime<chrono::Utc>>,
}

/// Byte range for partial content requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: Option<u64>,
}

/// Resource listing response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceListResponse {
    pub resources: Vec<ResourceDefinition>,
    pub total_count: usize,
    pub has_more: bool,
    pub next_token: Option<String>,
}

/// Utility functions for resource implementations
pub mod utils {
    use super::*;
    use std::path::Path;
    use tokio::fs;

    /// Generate ETag for content
    pub fn generate_etag(content: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("\"{:x}\"", hasher.finish())
    }

    /// Get MIME type from file extension
    pub fn get_mime_type(path: &Path) -> String {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => "application/json".to_string(),
            Some("txt") | Some("log") => "text/plain".to_string(),
            Some("png") => "image/png".to_string(),
            Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
            Some("mp4") => "video/mp4".to_string(),
            Some("wav") => "audio/wav".to_string(),
            Some("mp3") => "audio/mpeg".to_string(),
            Some("bin") | Some("weights") => "application/octet-stream".to_string(),
            Some("csv") => "text/csv".to_string(),
            Some("xml") => "application/xml".to_string(),
            Some("zip") => "application/zip".to_string(),
            _ => "application/octet-stream".to_string(),
        }
    }

    /// Check if content matches ETag
    pub fn matches_etag(content: &[u8], etag: &str) -> bool {
        let generated_etag = generate_etag(content);
        generated_etag == etag
    }

    /// Check if file is modified since given timestamp
    pub async fn is_modified_since(
        file_path: &Path,
        since: chrono::DateTime<chrono::Utc>,
    ) -> Result<bool> {
        if let Ok(metadata) = fs::metadata(file_path).await {
            if let Ok(modified) = metadata.modified() {
                let modified_utc: chrono::DateTime<chrono::Utc> = modified.into();
                return Ok(modified_utc > since);
            }
        }
        Ok(true) // Assume modified if we can't determine
    }

    /// Get file metadata
    pub async fn get_file_metadata(file_path: &Path) -> Result<ResourceMetadata> {
        let metadata = fs::metadata(file_path).await?;
        let modified: chrono::DateTime<chrono::Utc> = metadata.modified()?.into();
        
        let content = fs::read(file_path).await?;
        let etag = generate_etag(&content);
        
        Ok(ResourceMetadata {
            size_bytes: metadata.len(),
            last_modified: modified,
            etag: Some(etag),
            cache_control: Some("public, max-age=3600".to_string()),
            content_encoding: None,
        })
    }

    /// Validate URI format
    pub fn validate_uri(uri: &str) -> McpResult<()> {
        if uri.is_empty() {
            return Err(crate::mcp::McpError::InsufficientInput {
                reason: "URI cannot be empty".to_string(),
            });
        }
        
        if uri.contains("..") {
            return Err(crate::mcp::McpError::InsufficientInput {
                reason: "URI cannot contain path traversal sequences".to_string(),
            });
        }
        
        Ok(())
    }

    /// Parse URI parameters
    pub fn parse_uri_params(uri: &str) -> HashMap<String, String> {
        let mut params = HashMap::new();
        
        if let Some(query_start) = uri.find('?') {
            let query = &uri[query_start + 1..];
            
            for pair in query.split('&') {
                if let Some(eq_pos) = pair.find('=') {
                    let key = &pair[..eq_pos];
                    let value = &pair[eq_pos + 1..];
                    params.insert(
                        urlencoding::decode(key).unwrap_or_default().to_string(),
                        urlencoding::decode(value).unwrap_or_default().to_string(),
                    );
                }
            }
        }
        
        params
    }

    /// Extract ID from URI path
    pub fn extract_id_from_uri(uri: &str, pattern: &str) -> Option<String> {
        // Simple pattern matching - in production would use a proper router
        let parts: Vec<&str> = uri.split('/').collect();
        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        
        if parts.len() != pattern_parts.len() {
            return None;
        }
        
        for (i, pattern_part) in pattern_parts.iter().enumerate() {
            if pattern_part.starts_with('{') && pattern_part.ends_with('}') {
                return Some(parts[i].to_string());
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_etag() {
        let content = b"test content";
        let etag = utils::generate_etag(content);
        assert!(etag.starts_with('"') && etag.ends_with('"'));
        
        // Same content should generate same ETag
        let etag2 = utils::generate_etag(content);
        assert_eq!(etag, etag2);
        
        // Different content should generate different ETag
        let etag3 = utils::generate_etag(b"different content");
        assert_ne!(etag, etag3);
    }

    #[test]
    fn test_get_mime_type() {
        assert_eq!(utils::get_mime_type(Path::new("test.json")), "application/json");
        assert_eq!(utils::get_mime_type(Path::new("test.txt")), "text/plain");
        assert_eq!(utils::get_mime_type(Path::new("test.png")), "image/png");
        assert_eq!(utils::get_mime_type(Path::new("test.unknown")), "application/octet-stream");
    }

    #[test]
    fn test_validate_uri() {
        assert!(utils::validate_uri("valid/path").is_ok());
        assert!(utils::validate_uri("").is_err());
        assert!(utils::validate_uri("../invalid").is_err());
        assert!(utils::validate_uri("valid/../path").is_err());
    }

    #[test]
    fn test_parse_uri_params() {
        let params = utils::parse_uri_params("path?key1=value1&key2=value2");
        assert_eq!(params.get("key1"), Some(&"value1".to_string()));
        assert_eq!(params.get("key2"), Some(&"value2".to_string()));
        
        let params = utils::parse_uri_params("path");
        assert!(params.is_empty());
    }

    #[test]
    fn test_extract_id_from_uri() {
        let id = utils::extract_id_from_uri("models/123/weights", "models/{model_id}/weights");
        assert_eq!(id, Some("123".to_string()));
        
        let id = utils::extract_id_from_uri("invalid/path", "models/{model_id}/weights");
        assert_eq!(id, None);
    }
}