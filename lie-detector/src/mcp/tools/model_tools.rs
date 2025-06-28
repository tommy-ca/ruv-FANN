//! Model Management Tools for Veritas MCP Server
//!
//! This module provides comprehensive model management capabilities including
//! model creation, loading, unloading, and lifecycle management for the
//! Veritas lie detection system.

use anyhow::{Context, Result};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tokio::{
    fs,
    sync::RwLock,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::mcp::{
    auth::AuthLevel,
    events::{EventManager, EventType, VeritasEvent},
    server::ServerConfig,
    McpError, McpResult, RequestMetadata,
};

use super::{
    ToolDefinition, ToolExample, ToolHandler, ToolResponse,
    utils::{self, create_response_metadata},
};

/// Model status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    Created,
    Loading,
    Loaded,
    Unloading,
    Failed,
    Archived,
}

/// Supported modalities for lie detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Vision,
    Audio,
    Text,
    Physiological,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub vision_model: Option<String>,
    pub audio_model: Option<String>,
    pub text_model: Option<String>,
    pub fusion_strategy: String,
    pub hidden_layers: Vec<usize>,
    pub dropout_rate: f32,
    pub activation_function: String,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self {
            vision_model: Some("resnet".to_string()),
            audio_model: Some("lstm".to_string()),
            text_model: Some("bert".to_string()),
            fusion_strategy: "attention".to_string(),
            hidden_layers: vec![512, 256, 128],
            dropout_rate: 0.1,
            activation_function: "relu".to_string(),
        }
    }
}

/// Model metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub status: ModelStatus,
    pub modalities: Vec<Modality>,
    pub architecture: ModelArchitecture,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub created_by: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub file_path: Option<PathBuf>,
    pub file_size_bytes: Option<u64>,
    pub performance_metrics: Option<ModelMetrics>,
    pub is_active: bool,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub training_loss: f64,
    pub validation_loss: f64,
    pub inference_time_ms: f64,
    pub memory_usage_mb: f64,
}

/// Request structure for creating a new model
#[derive(Debug, Deserialize)]
pub struct CreateModelRequest {
    pub name: String,
    pub modalities: Vec<Modality>,
    pub architecture: Option<ModelArchitecture>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
}

/// Request structure for loading a model
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    pub device: Option<String>,
    pub precision: Option<String>,
    pub optimization_level: Option<String>,
}

/// Request structure for listing models
#[derive(Debug, Deserialize)]
pub struct ListModelsRequest {
    pub filter: Option<String>,
    pub status_filter: Option<Vec<ModelStatus>>,
    pub modality_filter: Option<Vec<Modality>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Response structure for model operations
#[derive(Debug, Serialize)]
pub struct ModelOperationResponse {
    pub model_id: String,
    pub operation: String,
    pub success: bool,
    pub message: String,
    pub model_info: Option<ModelInfo>,
}

/// Response structure for listing models
#[derive(Debug, Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
    pub total_count: usize,
    pub filtered_count: usize,
}

/// Loaded model instance
#[derive(Debug)]
struct LoadedModel {
    info: ModelInfo,
    loaded_at: Instant,
    last_used: Instant,
    usage_count: u64,
    memory_usage_mb: f64,
}

/// Model management handler
#[derive(Debug)]
pub struct ModelToolsHandler {
    config: ServerConfig,
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    loaded_models: Arc<RwLock<HashMap<String, LoadedModel>>>,
    model_storage_path: PathBuf,
    event_manager: Option<Arc<EventManager>>,
    max_loaded_models: usize,
    max_cache_size_mb: u64,
}

impl ModelToolsHandler {
    /// Create a new model tools handler
    pub async fn new(config: &ServerConfig) -> Result<Self> {
        let model_storage_path = PathBuf::from(&config.model_storage_path);
        
        // Ensure model storage directory exists
        if !model_storage_path.exists() {
            fs::create_dir_all(&model_storage_path).await
                .context("Failed to create model storage directory")?;
        }

        let handler = Self {
            config: config.clone(),
            models: Arc::new(RwLock::new(HashMap::new())),
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            model_storage_path,
            event_manager: None,
            max_loaded_models: 10,
            max_cache_size_mb: config.max_model_cache_mb,
        };

        // Load existing models from storage
        handler.load_models_from_storage().await?;

        info!("Model tools handler initialized with storage path: {:?}", 
              handler.model_storage_path);
        
        Ok(handler)
    }

    /// Set the event manager for publishing events
    pub fn set_event_manager(&mut self, event_manager: Arc<EventManager>) {
        self.event_manager = Some(event_manager);
    }

    /// List models with optional filtering
    pub async fn list_models(&self, metadata: RequestMetadata) -> impl IntoResponse {
        let start_time = Instant::now();
        debug!("Listing models for request: {}", metadata.request_id);

        // Check authorization
        if !metadata.auth_level.permits(AuthLevel::ReadOnly) {
            return ToolResponse::<ListModelsResponse>::error(
                "Insufficient permissions to list models".to_string(),
                create_response_metadata(metadata.request_id, start_time, None),
            );
        }

        let models = self.models.read().await;
        let models_vec: Vec<ModelInfo> = models.values().cloned().collect();
        
        let response = ListModelsResponse {
            total_count: models_vec.len(),
            filtered_count: models_vec.len(),
            models: models_vec,
        };

        ToolResponse::success(
            response,
            create_response_metadata(metadata.request_id, start_time, None),
        )
    }

    /// Create a new model
    pub async fn create_model(
        &self,
        request: CreateModelRequest,
        metadata: RequestMetadata,
    ) -> impl IntoResponse {
        let start_time = Instant::now();
        debug!("Creating model '{}' for request: {}", request.name, metadata.request_id);

        // Check authorization
        if !metadata.auth_level.permits(AuthLevel::Trainer) {
            return ToolResponse::<ModelOperationResponse>::error(
                "Insufficient permissions to create models".to_string(),
                create_response_metadata(metadata.request_id, start_time, None),
            );
        }

        let model_id = Uuid::new_v4().to_string();
        let model_info = ModelInfo {
            id: model_id.clone(),
            name: request.name.clone(),
            version: "1.0.0".to_string(),
            status: ModelStatus::Created,
            modalities: request.modalities,
            architecture: request.architecture.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            created_by: metadata.client_id.unwrap_or_else(|| "unknown".to_string()),
            description: request.description,
            tags: request.tags.unwrap_or_default(),
            file_path: None,
            file_size_bytes: None,
            performance_metrics: None,
            is_active: true,
        };

        // Store model info
        {
            let mut models = self.models.write().await;
            models.insert(model_id.clone(), model_info.clone());
        }

        // Save to storage
        if let Err(e) = self.save_model_to_storage(&model_info).await {
            error!("Failed to save model to storage: {}", e);
            return ToolResponse::<ModelOperationResponse>::error(
                format!("Failed to save model: {}", e),
                create_response_metadata(metadata.request_id, start_time, None),
            );
        }

        // Publish event
        if let Some(ref event_manager) = self.event_manager {
            let event = VeritasEvent::new(
                EventType::ModelCreated,
                serde_json::json!({
                    "model_id": model_id,
                    "model_name": request.name,
                    "modalities": model_info.modalities,
                }),
                serde_json::json!({
                    "created_by": model_info.created_by,
                    "request_id": metadata.request_id,
                }),
            );
            
            if let Err(e) = event_manager.publish_event(event).await {
                warn!("Failed to publish model creation event: {}", e);
            }
        }

        let response = ModelOperationResponse {
            model_id: model_id.clone(),
            operation: "create".to_string(),
            success: true,
            message: format!("Model '{}' created successfully", request.name),
            model_info: Some(model_info),
        };

        info!("Created model: {} ({})", request.name, model_id);

        ToolResponse::success(
            response,
            create_response_metadata(metadata.request_id, start_time, Some("1.0.0".to_string())),
        )
    }

    /// Load a model into memory
    pub async fn load_model(
        &self,
        request: LoadModelRequest,
        metadata: RequestMetadata,
    ) -> impl IntoResponse {
        let start_time = Instant::now();
        debug!("Loading model '{}' for request: {}", request.model_id, metadata.request_id);

        // Check authorization
        if !metadata.auth_level.permits(AuthLevel::Analyst) {
            return ToolResponse::<ModelOperationResponse>::error(
                "Insufficient permissions to load models".to_string(),
                create_response_metadata(metadata.request_id, start_time, None),
            );
        }

        // Get model info
        let model_info = {
            let models = self.models.read().await;
            match models.get(&request.model_id) {
                Some(info) => info.clone(),
                None => {
                    return ToolResponse::<ModelOperationResponse>::error(
                        format!("Model not found: {}", request.model_id),
                        create_response_metadata(metadata.request_id, start_time, None),
                    );
                }
            }
        };

        // Check if model is already loaded
        {
            let loaded_models = self.loaded_models.read().await;
            if loaded_models.contains_key(&request.model_id) {
                return ToolResponse::<ModelOperationResponse>::error(
                    format!("Model {} is already loaded", request.model_id),
                    create_response_metadata(metadata.request_id, start_time, Some(model_info.version.clone())),
                );
            }
        }

        // Check cache limits
        if let Err(e) = self.ensure_cache_capacity().await {
            warn!("Failed to ensure cache capacity: {}", e);
        }

        // Update model status to loading
        {
            let mut models = self.models.write().await;
            if let Some(model) = models.get_mut(&request.model_id) {
                model.status = ModelStatus::Loading;
                model.updated_at = chrono::Utc::now();
            }
        }

        // Simulate model loading (in a real implementation, this would load the actual model)
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Create loaded model instance
        let loaded_model = LoadedModel {
            info: model_info.clone(),
            loaded_at: Instant::now(),
            last_used: Instant::now(),
            usage_count: 0,
            memory_usage_mb: 256.0, // Placeholder value
        };

        // Store loaded model
        {
            let mut loaded_models = self.loaded_models.write().await;
            loaded_models.insert(request.model_id.clone(), loaded_model);
        }

        // Update model status to loaded
        {
            let mut models = self.models.write().await;
            if let Some(model) = models.get_mut(&request.model_id) {
                model.status = ModelStatus::Loaded;
                model.updated_at = chrono::Utc::now();
            }
        }

        // Publish event
        if let Some(ref event_manager) = self.event_manager {
            let event = VeritasEvent::new(
                EventType::SystemModelLoaded,
                serde_json::json!({
                    "model_id": request.model_id,
                    "model_name": model_info.name,
                    "device": request.device.unwrap_or_else(|| "cpu".to_string()),
                    "memory_usage_mb": 256.0,
                }),
                serde_json::json!({
                    "request_id": metadata.request_id,
                }),
            );
            
            if let Err(e) = event_manager.publish_event(event).await {
                warn!("Failed to publish model load event: {}", e);
            }
        }

        let response = ModelOperationResponse {
            model_id: request.model_id.clone(),
            operation: "load".to_string(),
            success: true,
            message: format!("Model '{}' loaded successfully", model_info.name),
            model_info: Some(model_info.clone()),
        };

        info!("Loaded model: {} ({})", model_info.name, request.model_id);

        ToolResponse::success(
            response,
            create_response_metadata(metadata.request_id, start_time, Some(model_info.version)),
        )
    }

    /// Unload a model from memory
    pub async fn unload_model(
        &self,
        model_id: &str,
        metadata: RequestMetadata,
    ) -> impl IntoResponse {
        let start_time = Instant::now();
        debug!("Unloading model '{}' for request: {}", model_id, metadata.request_id);

        // Check authorization
        if !metadata.auth_level.permits(AuthLevel::Analyst) {
            return ToolResponse::<ModelOperationResponse>::error(
                "Insufficient permissions to unload models".to_string(),
                create_response_metadata(metadata.request_id, start_time, None),
            );
        }

        // Remove from loaded models
        let loaded_model = {
            let mut loaded_models = self.loaded_models.write().await;
            loaded_models.remove(model_id)
        };

        match loaded_model {
            Some(loaded_model) => {
                // Update model status
                {
                    let mut models = self.models.write().await;
                    if let Some(model) = models.get_mut(model_id) {
                        model.status = ModelStatus::Created;
                        model.updated_at = chrono::Utc::now();
                    }
                }

                let response = ModelOperationResponse {
                    model_id: model_id.to_string(),
                    operation: "unload".to_string(),
                    success: true,
                    message: format!("Model '{}' unloaded successfully", loaded_model.info.name),
                    model_info: Some(loaded_model.info.clone()),
                };

                info!("Unloaded model: {} ({})", loaded_model.info.name, model_id);

                ToolResponse::success(
                    response,
                    create_response_metadata(metadata.request_id, start_time, Some(loaded_model.info.version)),
                )
            }
            None => {
                ToolResponse::<ModelOperationResponse>::error(
                    format!("Model {} is not loaded", model_id),
                    create_response_metadata(metadata.request_id, start_time, None),
                )
            }
        }
    }

    /// Get the number of active models
    pub async fn get_active_model_count(&self) -> u32 {
        let loaded_models = self.loaded_models.read().await;
        loaded_models.len() as u32
    }

    /// Load models from storage on startup
    async fn load_models_from_storage(&self) -> Result<()> {
        let manifest_path = self.model_storage_path.join("models.json");
        
        if !manifest_path.exists() {
            info!("No existing model manifest found, starting with empty model registry");
            return Ok(());
        }

        let manifest_content = fs::read_to_string(&manifest_path).await
            .context("Failed to read model manifest")?;
        
        let stored_models: HashMap<String, ModelInfo> = serde_json::from_str(&manifest_content)
            .context("Failed to parse model manifest")?;

        let mut models = self.models.write().await;
        *models = stored_models;

        info!("Loaded {} models from storage", models.len());
        Ok(())
    }

    /// Save model to storage
    async fn save_model_to_storage(&self, model_info: &ModelInfo) -> Result<()> {
        // Save individual model file
        let model_file_path = self.model_storage_path.join(format!("{}.json", model_info.id));
        let model_json = serde_json::to_string_pretty(model_info)
            .context("Failed to serialize model info")?;
        
        fs::write(&model_file_path, model_json).await
            .context("Failed to write model file")?;

        // Update manifest
        let models = self.models.read().await;
        let manifest_path = self.model_storage_path.join("models.json");
        let manifest_json = serde_json::to_string_pretty(&*models)
            .context("Failed to serialize model manifest")?;
        
        fs::write(&manifest_path, manifest_json).await
            .context("Failed to write model manifest")?;

        Ok(())
    }

    /// Ensure cache capacity by unloading least recently used models
    async fn ensure_cache_capacity(&self) -> Result<()> {
        let mut loaded_models = self.loaded_models.write().await;
        
        // Check model count limit
        while loaded_models.len() >= self.max_loaded_models {
            // Find least recently used model
            let lru_model_id = loaded_models
                .iter()
                .min_by_key(|(_, model)| model.last_used)
                .map(|(id, _)| id.clone());

            if let Some(model_id) = lru_model_id {
                info!("Unloading LRU model to make cache space: {}", model_id);
                loaded_models.remove(&model_id);
            } else {
                break;
            }
        }

        // Check memory usage limit
        let total_memory: f64 = loaded_models.values().map(|m| m.memory_usage_mb).sum();
        
        while total_memory > self.max_cache_size_mb as f64 {
            let lru_model_id = loaded_models
                .iter()
                .min_by_key(|(_, model)| model.last_used)
                .map(|(id, _)| id.clone());

            if let Some(model_id) = lru_model_id {
                info!("Unloading LRU model to reduce memory usage: {}", model_id);
                loaded_models.remove(&model_id);
            } else {
                break;
            }
        }

        Ok(())
    }
}

impl ToolHandler for ModelToolsHandler {
    fn name(&self) -> &'static str {
        "model_tools"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn get_tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "model/list".to_string(),
                description: "List all available lie detection models".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "enum": ["all", "trained", "untrained", "active"],
                            "default": "all"
                        }
                    }
                }),
                examples: vec![
                    ToolExample {
                        description: "List all models".to_string(),
                        input: serde_json::json!({"filter": "all"}),
                        expected_output: None,
                    }
                ],
            },
            ToolDefinition {
                name: "model/create".to_string(),
                description: "Create a new lie detection model".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "required": ["name", "modalities"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Model name"
                        },
                        "modalities": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["vision", "audio", "text", "physiological"]
                            }
                        },
                        "architecture": {
                            "type": "object",
                            "description": "Model architecture configuration"
                        }
                    }
                }),
                examples: vec![
                    ToolExample {
                        description: "Create a multi-modal model".to_string(),
                        input: serde_json::json!({
                            "name": "detector-v1",
                            "modalities": ["vision", "audio", "text"]
                        }),
                        expected_output: None,
                    }
                ],
            },
            ToolDefinition {
                name: "model/load".to_string(),
                description: "Load a model for inference".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "required": ["model_id"],
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Unique model identifier"
                        },
                        "device": {
                            "type": "string",
                            "enum": ["cpu", "cuda", "mps"],
                            "default": "cpu"
                        }
                    }
                }),
                examples: vec![
                    ToolExample {
                        description: "Load model on GPU".to_string(),
                        input: serde_json::json!({
                            "model_id": "detector-v1",
                            "device": "cuda"
                        }),
                        expected_output: None,
                    }
                ],
            },
        ]
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down model tools handler");
        
        // Unload all models
        {
            let mut loaded_models = self.loaded_models.write().await;
            loaded_models.clear();
        }

        // Save final state to storage
        let models = self.models.read().await;
        let manifest_path = self.model_storage_path.join("models.json");
        let manifest_json = serde_json::to_string_pretty(&*models)
            .context("Failed to serialize final model manifest")?;
        
        fs::write(&manifest_path, manifest_json).await
            .context("Failed to write final model manifest")?;

        info!("Model tools handler shutdown complete");
        Ok(())
    }

    async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let models = self.models.read().await;
        let loaded_models = self.loaded_models.read().await;
        
        stats.insert("total_models".to_string(), serde_json::json!(models.len()));
        stats.insert("loaded_models".to_string(), serde_json::json!(loaded_models.len()));
        stats.insert("max_loaded_models".to_string(), serde_json::json!(self.max_loaded_models));
        stats.insert("max_cache_size_mb".to_string(), serde_json::json!(self.max_cache_size_mb));
        
        let total_memory: f64 = loaded_models.values().map(|m| m.memory_usage_mb).sum();
        stats.insert("total_memory_usage_mb".to_string(), serde_json::json!(total_memory));
        
        let models_by_status: HashMap<String, usize> = models
            .values()
            .fold(HashMap::new(), |mut acc, model| {
                let status_str = format!("{:?}", model.status);
                *acc.entry(status_str).or_insert(0) += 1;
                acc
            });
        stats.insert("models_by_status".to_string(), serde_json::json!(models_by_status));

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_handler() -> ModelToolsHandler {
        let temp_dir = tempdir().unwrap();
        let config = ServerConfig {
            model_storage_path: temp_dir.path().to_string_lossy().to_string(),
            max_model_cache_mb: 1024,
            ..Default::default()
        };
        
        ModelToolsHandler::new(&config).await.unwrap()
    }

    #[tokio::test]
    async fn test_model_handler_creation() {
        let handler = create_test_handler().await;
        assert_eq!(handler.name(), "model_tools");
        assert_eq!(handler.version(), "1.0.0");
    }

    #[tokio::test]
    async fn test_model_creation() {
        let handler = create_test_handler().await;
        
        let request = CreateModelRequest {
            name: "test-model".to_string(),
            modalities: vec![Modality::Vision, Modality::Text],
            architecture: None,
            description: Some("Test model".to_string()),
            tags: Some(vec!["test".to_string()]),
        };

        let metadata = RequestMetadata::new(AuthLevel::Trainer);
        let _response = handler.create_model(request, metadata).await;
        
        // Verify model was created
        let models = handler.models.read().await;
        assert_eq!(models.len(), 1);
        
        let model = models.values().next().unwrap();
        assert_eq!(model.name, "test-model");
        assert_eq!(model.modalities.len(), 2);
    }

    #[tokio::test]
    async fn test_model_tools_definitions() {
        let handler = create_test_handler().await;
        let tools = handler.get_tools();
        
        assert_eq!(tools.len(), 3);
        assert!(tools.iter().any(|t| t.name == "model/list"));
        assert!(tools.iter().any(|t| t.name == "model/create"));
        assert!(tools.iter().any(|t| t.name == "model/load"));
    }

    #[tokio::test]
    async fn test_model_stats() {
        let handler = create_test_handler().await;
        let stats = handler.get_stats().await;
        
        assert!(stats.contains_key("total_models"));
        assert!(stats.contains_key("loaded_models"));
        assert!(stats.contains_key("max_loaded_models"));
    }
}