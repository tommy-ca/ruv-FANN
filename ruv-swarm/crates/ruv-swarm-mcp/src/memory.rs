//! Session memory management for MCP operations

use std::collections::HashMap;
use serde_json::Value;
use anyhow::Result;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Session memory manager for persistent coordination
pub struct SessionMemory {
    storage: Arc<RwLock<HashMap<String, Value>>>,
}

impl Default for SessionMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionMemory {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store a value in session memory
    pub async fn store(&self, key: String, value: Value) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.insert(key, value);
        Ok(())
    }

    /// Retrieve a value from session memory
    pub async fn retrieve(&self, key: &str) -> Result<Option<Value>> {
        let storage = self.storage.read().await;
        Ok(storage.get(key).cloned())
    }

    /// List all keys matching a pattern
    pub async fn list(&self, pattern: Option<&str>) -> Result<Vec<String>> {
        let storage = self.storage.read().await;
        let keys: Vec<String> = if let Some(pattern) = pattern {
            storage
                .keys()
                .filter(|key| key.contains(pattern.trim_end_matches('*')))
                .cloned()
                .collect()
        } else {
            storage.keys().cloned().collect()
        };
        Ok(keys)
    }

    /// Delete a key from session memory
    pub async fn delete(&self, key: &str) -> Result<bool> {
        let mut storage = self.storage.write().await;
        Ok(storage.remove(key).is_some())
    }

    /// Clear all session memory
    pub async fn clear(&self) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.clear();
        Ok(())
    }

    /// Get memory usage statistics
    pub async fn usage_stats(&self) -> Result<MemoryUsageStats> {
        let storage = self.storage.read().await;
        Ok(MemoryUsageStats {
            total_keys: storage.len(),
            total_memory_bytes: storage
                .values()
                .map(|v| serde_json::to_string(v).unwrap_or_default().len())
                .sum(),
        })
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryUsageStats {
    pub total_keys: usize,
    pub total_memory_bytes: usize,
}