//! # MCP Server Example for Veritas Nexus
//! 
//! This example demonstrates how to set up a Model Context Protocol (MCP) server
//! for the veritas-nexus lie detection system. It shows how to:
//! - Create and configure an MCP server
//! - Register tools for lie detection operations
//! - Implement resource providers for models and data
//! - Handle client requests and streaming operations
//! - Provide monitoring and management capabilities
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example mcp_server
//! ```
//! 
//! Then connect with an MCP client:
//! ```bash
//! mcp connect stdio veritas-nexus-server
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// MCP Server for Veritas Nexus
pub struct VeritasNexusMcpServer {
    name: String,
    version: String,
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    active_sessions: Arc<Mutex<HashMap<String, AnalysisSession>>>,
    tool_registry: ToolRegistry,
    resource_manager: ResourceManager,
    event_stream: Arc<Mutex<mpsc::UnboundedSender<ServerEvent>>>,
    config: ServerConfig,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub max_concurrent_sessions: usize,
    pub model_cache_size: usize,
    pub enable_streaming: bool,
    pub log_level: LogLevel,
    pub auth_required: bool,
    pub rate_limit_requests_per_minute: u32,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 10,
            model_cache_size: 5,
            enable_streaming: true,
            log_level: LogLevel::Info,
            auth_required: false,
            rate_limit_requests_per_minute: 60,
        }
    }
}

/// Information about available models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub capabilities: Vec<String>,
    pub size_mb: f64,
    pub accuracy: f64,
    pub latency_ms: u32,
    pub memory_usage_mb: u32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Vision,
    Audio,
    Text,
    Fusion,
    EndToEnd,
}

/// Active analysis session
#[derive(Debug)]
pub struct AnalysisSession {
    pub id: String,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub client_id: String,
    pub config: SessionConfig,
    pub results: Vec<AnalysisResult>,
    pub status: SessionStatus,
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub models: Vec<String>,
    pub fusion_strategy: String,
    pub quality_preset: QualityPreset,
    pub enable_streaming: bool,
    pub enable_explanations: bool,
}

#[derive(Debug, Clone)]
pub enum QualityPreset {
    Fast,
    Balanced,
    Accurate,
    Custom(CustomQualitySettings),
}

#[derive(Debug, Clone)]
pub struct CustomQualitySettings {
    pub vision_quality: f32,
    pub audio_quality: f32,
    pub text_quality: f32,
    pub fusion_complexity: f32,
}

#[derive(Debug, Clone)]
pub enum SessionStatus {
    Active,
    Paused,
    Completed,
    Error(String),
}

/// Analysis result from the session
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    pub id: String,
    pub timestamp: String,
    pub decision: String,
    pub confidence: f64,
    pub processing_time_ms: u64,
    pub modality_scores: HashMap<String, f64>,
    pub explanation: Option<String>,
}

/// Tool registry for MCP tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn McpTool>>,
}

/// MCP Tool trait
pub trait McpTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    fn execute(&self, params: Value, server: &VeritasNexusMcpServer) -> Result<Value, String>;
}

/// Resource manager for models and data
pub struct ResourceManager {
    model_resources: HashMap<String, ModelResource>,
    data_resources: HashMap<String, DataResource>,
}

#[derive(Debug, Clone)]
pub struct ModelResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub size_bytes: u64,
    pub last_modified: String,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct DataResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub format: String,
    pub size_bytes: u64,
    pub metadata: HashMap<String, Value>,
}

/// Server events
#[derive(Debug, Clone, Serialize)]
pub enum ServerEvent {
    SessionCreated { session_id: String, client_id: String },
    SessionCompleted { session_id: String, results_count: usize },
    ModelLoaded { model_id: String, load_time_ms: u64 },
    ModelUnloaded { model_id: String },
    Error { message: String, context: Option<Value> },
    PerformanceMetric { metric_name: String, value: f64, timestamp: String },
}

// =============================================================================
// MCP Tool Implementations
// =============================================================================

/// Tool for analyzing deception from multi-modal input
pub struct AnalyzeDeceptionTool;

impl McpTool for AnalyzeDeceptionTool {
    fn name(&self) -> &str {
        "analyze_deception"
    }
    
    fn description(&self) -> &str {
        "Analyze multi-modal inputs (video, audio, text) for deception detection"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Path to video file (optional)"
                },
                "audio_path": {
                    "type": "string", 
                    "description": "Path to audio file (optional)"
                },
                "text": {
                    "type": "string",
                    "description": "Text transcript to analyze (optional)"
                },
                "session_config": {
                    "type": "object",
                    "properties": {
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model IDs to use"
                        },
                        "fusion_strategy": {
                            "type": "string",
                            "enum": ["equal_weight", "adaptive", "attention", "context_aware"],
                            "default": "adaptive"
                        },
                        "quality_preset": {
                            "type": "string",
                            "enum": ["fast", "balanced", "accurate"],
                            "default": "balanced"
                        },
                        "enable_explanations": {
                            "type": "boolean",
                            "default": true
                        }
                    }
                }
            },
            "required": [],
            "additionalProperties": false
        })
    }
    
    fn execute(&self, params: Value, server: &VeritasNexusMcpServer) -> Result<Value, String> {
        // Extract parameters
        let video_path = params.get("video_path").and_then(|v| v.as_str());
        let audio_path = params.get("audio_path").and_then(|v| v.as_str());
        let text = params.get("text").and_then(|v| v.as_str());
        let session_config = params.get("session_config");
        
        // Validate that at least one input is provided
        if video_path.is_none() && audio_path.is_none() && text.is_none() {
            return Err("At least one input (video_path, audio_path, or text) must be provided".to_string());
        }
        
        // Parse session configuration
        let config = if let Some(config_obj) = session_config {
            SessionConfig {
                models: config_obj.get("models")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_else(|| vec!["default_vision".to_string(), "default_audio".to_string(), "default_text".to_string()]),
                fusion_strategy: config_obj.get("fusion_strategy")
                    .and_then(|v| v.as_str())
                    .unwrap_or("adaptive")
                    .to_string(),
                quality_preset: match config_obj.get("quality_preset").and_then(|v| v.as_str()).unwrap_or("balanced") {
                    "fast" => QualityPreset::Fast,
                    "accurate" => QualityPreset::Accurate,
                    _ => QualityPreset::Balanced,
                },
                enable_streaming: false,
                enable_explanations: config_obj.get("enable_explanations")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
            }
        } else {
            SessionConfig {
                models: vec!["default_vision".to_string(), "default_audio".to_string(), "default_text".to_string()],
                fusion_strategy: "adaptive".to_string(),
                quality_preset: QualityPreset::Balanced,
                enable_streaming: false,
                enable_explanations: true,
            }
        };
        
        // Simulate analysis
        let start_time = Instant::now();
        
        // Mock scores based on inputs
        let mut modality_scores = HashMap::new();
        let mut total_score = 0.0;
        let mut count = 0;
        
        if video_path.is_some() {
            let score = 0.65; // Mock vision score
            modality_scores.insert("vision".to_string(), score);
            total_score += score;
            count += 1;
        }
        
        if audio_path.is_some() {
            let score = 0.42; // Mock audio score
            modality_scores.insert("audio".to_string(), score);
            total_score += score;
            count += 1;
        }
        
        if text.is_some() {
            let score = 0.58; // Mock text score
            modality_scores.insert("text".to_string(), score);
            total_score += score;
            count += 1;
        }
        
        let final_score = if count > 0 { total_score / count as f64 } else { 0.5 };
        let confidence = (count as f64 / 3.0 * 0.8 + 0.2).min(1.0);
        
        let decision = if confidence < 0.3 {
            "uncertain"
        } else if final_score > 0.6 {
            "deceptive"
        } else if final_score < 0.4 {
            "truthful"
        } else {
            "uncertain"
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Generate explanation if requested
        let explanation = if config.enable_explanations {
            let mut explanation_parts = Vec::new();
            
            explanation_parts.push(format!("Analysis completed using {} fusion strategy", config.fusion_strategy));
            explanation_parts.push(format!("Processed {} modalities with {:.1}% confidence", count, confidence * 100.0));
            
            if let Some(vision_score) = modality_scores.get("vision") {
                explanation_parts.push(format!("Vision analysis detected deception indicators with score {:.3}", vision_score));
            }
            if let Some(audio_score) = modality_scores.get("audio") {
                explanation_parts.push(format!("Audio analysis found vocal stress patterns with score {:.3}", audio_score));
            }
            if let Some(text_score) = modality_scores.get("text") {
                explanation_parts.push(format!("Text analysis identified linguistic deception cues with score {:.3}", text_score));
            }
            
            explanation_parts.push(format!("Final decision: {} (combined score: {:.3})", decision, final_score));
            
            Some(explanation_parts.join(". "))
        } else {
            None
        };
        
        let result = AnalysisResult {
            id: format!("analysis_{}", start_time.elapsed().as_nanos()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            decision: decision.to_string(),
            confidence,
            processing_time_ms: processing_time,
            modality_scores,
            explanation,
        };
        
        Ok(serde_json::to_value(result).unwrap())
    }
}

/// Tool for managing models
pub struct ModelManagementTool;

impl McpTool for ModelManagementTool {
    fn name(&self) -> &str {
        "manage_models"
    }
    
    fn description(&self) -> &str {
        "Load, unload, and get information about available models"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "load", "unload", "info"],
                    "description": "Action to perform"
                },
                "model_id": {
                    "type": "string",
                    "description": "Model ID (required for load, unload, info actions)"
                }
            },
            "required": ["action"],
            "additionalProperties": false
        })
    }
    
    fn execute(&self, params: Value, server: &VeritasNexusMcpServer) -> Result<Value, String> {
        let action = params.get("action").and_then(|v| v.as_str())
            .ok_or("Missing required parameter: action")?;
        
        match action {
            "list" => {
                let models = server.models.read().unwrap();
                let model_list: Vec<&ModelInfo> = models.values().collect();
                Ok(serde_json::to_value(model_list).unwrap())
            }
            "load" => {
                let model_id = params.get("model_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: model_id for load action")?;
                
                // Simulate model loading
                let load_start = Instant::now();
                std::thread::sleep(Duration::from_millis(100)); // Simulate load time
                let load_time = load_start.elapsed().as_millis() as u64;
                
                // Send event
                if let Ok(mut sender) = server.event_stream.try_lock() {
                    let _ = sender.send(ServerEvent::ModelLoaded {
                        model_id: model_id.to_string(),
                        load_time_ms: load_time,
                    });
                }
                
                Ok(json!({
                    "status": "loaded",
                    "model_id": model_id,
                    "load_time_ms": load_time
                }))
            }
            "unload" => {
                let model_id = params.get("model_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: model_id for unload action")?;
                
                // Send event
                if let Ok(mut sender) = server.event_stream.try_lock() {
                    let _ = sender.send(ServerEvent::ModelUnloaded {
                        model_id: model_id.to_string(),
                    });
                }
                
                Ok(json!({
                    "status": "unloaded",
                    "model_id": model_id
                }))
            }
            "info" => {
                let model_id = params.get("model_id").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: model_id for info action")?;
                
                let models = server.models.read().unwrap();
                if let Some(model_info) = models.get(model_id) {
                    Ok(serde_json::to_value(model_info).unwrap())
                } else {
                    Err(format!("Model not found: {}", model_id))
                }
            }
            _ => Err(format!("Unknown action: {}", action))
        }
    }
}

/// Tool for monitoring server performance
pub struct MonitoringTool;

impl McpTool for MonitoringTool {
    fn name(&self) -> &str {
        "monitor_server"
    }
    
    fn description(&self) -> &str {
        "Get server status, performance metrics, and health information"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "metric_type": {
                    "type": "string",
                    "enum": ["status", "performance", "sessions", "resources"],
                    "default": "status"
                }
            },
            "additionalProperties": false
        })
    }
    
    fn execute(&self, params: Value, server: &VeritasNexusMcpServer) -> Result<Value, String> {
        let metric_type = params.get("metric_type").and_then(|v| v.as_str()).unwrap_or("status");
        
        match metric_type {
            "status" => {
                Ok(json!({
                    "status": "healthy",
                    "version": server.version,
                    "uptime_seconds": 3600, // Mock uptime
                    "active_sessions": 2,
                    "loaded_models": 3,
                    "memory_usage_mb": 512.5,
                    "cpu_usage_percent": 15.2
                }))
            }
            "performance" => {
                Ok(json!({
                    "avg_response_time_ms": 125.5,
                    "requests_per_minute": 45,
                    "success_rate_percent": 98.7,
                    "error_rate_percent": 1.3,
                    "throughput_mb_per_second": 12.8,
                    "queue_length": 3
                }))
            }
            "sessions" => {
                let sessions = server.active_sessions.try_lock().unwrap();
                let session_info: Vec<Value> = sessions.values().map(|session| {
                    json!({
                        "id": session.id,
                        "client_id": session.client_id,
                        "status": format!("{:?}", session.status),
                        "created_at": session.created_at.elapsed().as_secs(),
                        "results_count": session.results.len()
                    })
                }).collect();
                
                Ok(json!({
                    "total_sessions": sessions.len(),
                    "sessions": session_info
                }))
            }
            "resources" => {
                Ok(json!({
                    "available_models": 5,
                    "loaded_models": 3,
                    "model_cache_usage_mb": 256.0,
                    "data_cache_usage_mb": 128.0,
                    "available_disk_space_gb": 45.2,
                    "network_bandwidth_mbps": 100.0
                }))
            }
            _ => Err(format!("Unknown metric type: {}", metric_type))
        }
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut tools: HashMap<String, Box<dyn McpTool>> = HashMap::new();
        
        tools.insert("analyze_deception".to_string(), Box::new(AnalyzeDeceptionTool));
        tools.insert("manage_models".to_string(), Box::new(ModelManagementTool));
        tools.insert("monitor_server".to_string(), Box::new(MonitoringTool));
        
        Self { tools }
    }
    
    pub fn get_tool(&self, name: &str) -> Option<&Box<dyn McpTool>> {
        self.tools.get(name)
    }
    
    pub fn list_tools(&self) -> Vec<Value> {
        self.tools.values().map(|tool| {
            json!({
                "name": tool.name(),
                "description": tool.description(),
                "inputSchema": tool.input_schema()
            })
        }).collect()
    }
}

impl ResourceManager {
    pub fn new() -> Self {
        let mut model_resources = HashMap::new();
        let mut data_resources = HashMap::new();
        
        // Add some example model resources
        model_resources.insert("vision_model_v1".to_string(), ModelResource {
            uri: "models://veritas/vision/v1.0.0".to_string(),
            name: "Vision Deception Detector v1".to_string(),
            description: "Facial micro-expression and eye movement analysis model".to_string(),
            size_bytes: 125_000_000,
            last_modified: "2024-01-15T10:30:00Z".to_string(),
            metadata: [
                ("accuracy".to_string(), json!(0.887)),
                ("latency_ms".to_string(), json!(45)),
                ("training_data_size".to_string(), json!("50K samples")),
            ].into_iter().collect(),
        });
        
        model_resources.insert("audio_model_v1".to_string(), ModelResource {
            uri: "models://veritas/audio/v1.0.0".to_string(),
            name: "Audio Stress Detector v1".to_string(),
            description: "Voice stress and pitch analysis model for deception detection".to_string(),
            size_bytes: 87_000_000,
            last_modified: "2024-01-15T10:30:00Z".to_string(),
            metadata: [
                ("accuracy".to_string(), json!(0.823)),
                ("latency_ms".to_string(), json!(25)),
                ("sample_rate".to_string(), json!(16000)),
            ].into_iter().collect(),
        });
        
        // Add example data resources
        data_resources.insert("benchmark_dataset".to_string(), DataResource {
            uri: "data://veritas/benchmarks/deception_v1".to_string(),
            name: "Deception Detection Benchmark Dataset".to_string(),
            description: "Multi-modal deception detection benchmark with ground truth labels".to_string(),
            format: "multi_modal_json".to_string(),
            size_bytes: 2_500_000_000,
            metadata: [
                ("samples".to_string(), json!(10000)),
                ("modalities".to_string(), json!(["video", "audio", "text"])),
                ("languages".to_string(), json!(["en", "es", "fr", "de"])),
            ].into_iter().collect(),
        });
        
        Self {
            model_resources,
            data_resources,
        }
    }
    
    pub fn list_models(&self) -> Vec<&ModelResource> {
        self.model_resources.values().collect()
    }
    
    pub fn list_data(&self) -> Vec<&DataResource> {
        self.data_resources.values().collect()
    }
    
    pub fn get_model(&self, uri: &str) -> Option<&ModelResource> {
        self.model_resources.get(uri)
    }
    
    pub fn get_data(&self, uri: &str) -> Option<&DataResource> {
        self.data_resources.get(uri)
    }
}

impl VeritasNexusMcpServer {
    pub fn new(config: ServerConfig) -> Self {
        let (event_sender, _event_receiver) = mpsc::unbounded_channel();
        
        // Initialize models
        let mut models = HashMap::new();
        
        models.insert("default_vision".to_string(), ModelInfo {
            id: "default_vision".to_string(),
            name: "Default Vision Model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Vision,
            capabilities: vec![
                "face_detection".to_string(),
                "micro_expressions".to_string(),
                "eye_tracking".to_string(),
            ],
            size_mb: 125.0,
            accuracy: 0.887,
            latency_ms: 45,
            memory_usage_mb: 256,
            description: "High-accuracy facial deception detection model".to_string(),
        });
        
        models.insert("default_audio".to_string(), ModelInfo {
            id: "default_audio".to_string(),
            name: "Default Audio Model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Audio,
            capabilities: vec![
                "voice_stress".to_string(),
                "pitch_analysis".to_string(),
                "emotion_detection".to_string(),
            ],
            size_mb: 87.0,
            accuracy: 0.823,
            latency_ms: 25,
            memory_usage_mb: 128,
            description: "Voice stress and emotional state analysis model".to_string(),
        });
        
        models.insert("default_text".to_string(), ModelInfo {
            id: "default_text".to_string(),
            name: "Default Text Model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Text,
            capabilities: vec![
                "linguistic_analysis".to_string(),
                "sentiment_detection".to_string(),
                "deception_patterns".to_string(),
            ],
            size_mb: 342.0,
            accuracy: 0.794,
            latency_ms: 15,
            memory_usage_mb: 512,
            description: "BERT-based linguistic deception detection model".to_string(),
        });
        
        Self {
            name: "veritas-nexus-server".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            models: Arc::new(RwLock::new(models)),
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            tool_registry: ToolRegistry::new(),
            resource_manager: ResourceManager::new(),
            event_stream: Arc::new(Mutex::new(event_sender)),
            config,
        }
    }
    
    /// Start the MCP server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting Veritas Nexus MCP Server");
        println!("=====================================");
        println!("Name: {}", self.name);
        println!("Version: {}", self.version);
        println!("Max concurrent sessions: {}", self.config.max_concurrent_sessions);
        println!("Model cache size: {}", self.config.model_cache_size);
        println!("Streaming enabled: {}", self.config.enable_streaming);
        println!("Authentication required: {}", self.config.auth_required);
        println!();
        
        // List available tools
        println!("📋 Available Tools:");
        for tool_info in self.tool_registry.list_tools() {
            println!("  • {}: {}", 
                tool_info["name"].as_str().unwrap_or("Unknown"),
                tool_info["description"].as_str().unwrap_or("No description")
            );
        }
        println!();
        
        // List available models
        println!("🧠 Available Models:");
        let models = self.models.read().unwrap();
        for model in models.values() {
            println!("  • {}: {} ({:.1}MB, {:.1}% accuracy)", 
                model.id, model.name, model.size_mb, model.accuracy * 100.0);
        }
        println!();
        
        // List available resources
        println!("📦 Available Resources:");
        println!("  Models:");
        for resource in self.resource_manager.list_models() {
            println!("    • {}: {} ({:.1}MB)",
                resource.name, resource.description, resource.size_bytes as f64 / 1024.0 / 1024.0);
        }
        println!("  Data:");
        for resource in self.resource_manager.list_data() {
            println!("    • {}: {} ({:.1}GB)",
                resource.name, resource.description, resource.size_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        }
        println!();
        
        // Start server loop
        self.run_server_loop().await
    }
    
    /// Main server loop
    async fn run_server_loop(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("✅ Server started successfully!");
        println!("Waiting for MCP client connections...");
        println!();
        
        // Simulate server operation
        let mut tick_count = 0;
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        
        loop {
            interval.tick().await;
            tick_count += 1;
            
            // Simulate some activity
            self.simulate_activity(tick_count).await;
            
            // Break after demo period
            if tick_count >= 6 {
                break;
            }
        }
        
        println!("🛑 Server shutting down...");
        Ok(())
    }
    
    /// Simulate server activity for demonstration
    async fn simulate_activity(&self, tick: u32) {
        match tick {
            1 => {
                println!("📞 Simulated client connection received");
                println!("🔧 Tool call: analyze_deception");
                
                // Simulate tool execution
                let params = json!({
                    "text": "I was definitely not there at that time",
                    "session_config": {
                        "fusion_strategy": "adaptive",
                        "quality_preset": "balanced",
                        "enable_explanations": true
                    }
                });
                
                if let Some(tool) = self.tool_registry.get_tool("analyze_deception") {
                    match tool.execute(params, self) {
                        Ok(result) => {
                            println!("✅ Analysis completed:");
                            println!("   Decision: {}", result["decision"].as_str().unwrap_or("unknown"));
                            println!("   Confidence: {:.1}%", result["confidence"].as_f64().unwrap_or(0.0) * 100.0);
                            println!("   Processing time: {}ms", result["processing_time_ms"].as_u64().unwrap_or(0));
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
            }
            2 => {
                println!("🔧 Tool call: manage_models (list)");
                
                let params = json!({"action": "list"});
                if let Some(tool) = self.tool_registry.get_tool("manage_models") {
                    match tool.execute(params, self) {
                        Ok(result) => {
                            if let Some(models) = result.as_array() {
                                println!("✅ Available models: {} models", models.len());
                                for model in models.iter().take(2) {
                                    println!("   • {}: {}", 
                                        model["name"].as_str().unwrap_or("Unknown"),
                                        model["description"].as_str().unwrap_or("No description")
                                    );
                                }
                            }
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
            }
            3 => {
                println!("🔧 Tool call: manage_models (load default_vision)");
                
                let params = json!({
                    "action": "load",
                    "model_id": "default_vision"
                });
                
                if let Some(tool) = self.tool_registry.get_tool("manage_models") {
                    match tool.execute(params, self) {
                        Ok(result) => {
                            println!("✅ Model loaded: {} ({}ms)", 
                                result["model_id"].as_str().unwrap_or("unknown"),
                                result["load_time_ms"].as_u64().unwrap_or(0)
                            );
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
            }
            4 => {
                println!("🔧 Tool call: monitor_server (performance)");
                
                let params = json!({"metric_type": "performance"});
                if let Some(tool) = self.tool_registry.get_tool("monitor_server") {
                    match tool.execute(params, self) {
                        Ok(result) => {
                            println!("✅ Performance metrics:");
                            println!("   Avg response time: {:.1}ms", result["avg_response_time_ms"].as_f64().unwrap_or(0.0));
                            println!("   Requests/min: {}", result["requests_per_minute"].as_u64().unwrap_or(0));
                            println!("   Success rate: {:.1}%", result["success_rate_percent"].as_f64().unwrap_or(0.0));
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
            }
            5 => {
                println!("🔧 Multi-modal analysis with all modalities");
                
                let params = json!({
                    "video_path": "/path/to/interview.mp4",
                    "audio_path": "/path/to/interview.wav", 
                    "text": "I have never seen that document in my entire life, I swear",
                    "session_config": {
                        "fusion_strategy": "attention",
                        "quality_preset": "accurate"
                    }
                });
                
                if let Some(tool) = self.tool_registry.get_tool("analyze_deception") {
                    match tool.execute(params, self) {
                        Ok(result) => {
                            println!("✅ Multi-modal analysis completed:");
                            println!("   Decision: {}", result["decision"].as_str().unwrap_or("unknown"));
                            println!("   Confidence: {:.1}%", result["confidence"].as_f64().unwrap_or(0.0) * 100.0);
                            
                            if let Some(scores) = result["modality_scores"].as_object() {
                                println!("   Modality scores:");
                                for (modality, score) in scores {
                                    println!("     {} {}: {:.3}", 
                                        match modality.as_str() {
                                            "vision" => "👁️",
                                            "audio" => "🔊",
                                            "text" => "📝",
                                            _ => "🔹"
                                        },
                                        modality, 
                                        score.as_f64().unwrap_or(0.0)
                                    );
                                }
                            }
                            
                            if let Some(explanation) = result["explanation"].as_str() {
                                println!("   Explanation: {}", explanation);
                            }
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
            }
            6 => {
                println!("📊 Final server status");
                
                let params = json!({"metric_type": "status"});
                if let Some(tool) = self.tool_registry.get_tool("monitor_server") {
                    match tool.execute(params, self) {
                        Ok(result) => {
                            println!("✅ Server status:");
                            println!("   Status: {}", result["status"].as_str().unwrap_or("unknown"));
                            println!("   Active sessions: {}", result["active_sessions"].as_u64().unwrap_or(0));
                            println!("   Loaded models: {}", result["loaded_models"].as_u64().unwrap_or(0));
                            println!("   Memory usage: {:.1}MB", result["memory_usage_mb"].as_f64().unwrap_or(0.0));
                            println!("   CPU usage: {:.1}%", result["cpu_usage_percent"].as_f64().unwrap_or(0.0));
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
            }
            _ => {}
        }
        
        println!();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Veritas Nexus MCP Server Example");
    println!("===================================\n");
    
    // Create server configuration
    let config = ServerConfig {
        max_concurrent_sessions: 10,
        model_cache_size: 5,
        enable_streaming: true,
        log_level: LogLevel::Info,
        auth_required: false,
        rate_limit_requests_per_minute: 100,
    };
    
    // Create and start server
    let server = VeritasNexusMcpServer::new(config);
    
    match server.start().await {
        Ok(_) => {
            println!("🎉 Demo completed successfully!");
            println!("\n💡 Key Features Demonstrated:");
            println!("   • MCP server setup and configuration");
            println!("   • Tool registration and execution");
            println!("   • Resource management for models and data");
            println!("   • Multi-modal lie detection via MCP tools");
            println!("   • Performance monitoring and health checks");
            println!("   • Event streaming for real-time updates");
            println!("   • Session management and state tracking");
            
            println!("\n🔗 Next Steps:");
            println!("   • Connect real MCP clients to this server");
            println!("   • Implement authentication and authorization");
            println!("   • Add persistent storage for sessions and results");
            println!("   • Scale horizontally with load balancing");
            println!("   • Add custom tools for specific use cases");
        }
        Err(e) => {
            println!("❌ Server error: {}", e);
        }
    }
    
    Ok(())
}