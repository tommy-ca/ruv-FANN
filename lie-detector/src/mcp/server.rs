//! Core MCP Server Implementation for Veritas-Nexus
//!
//! This module implements the main MCP server using the official Rust MCP SDK.
//! It provides the core server functionality, request handling, and integration
//! with the Veritas-Nexus lie detection system.

use anyhow::{Context, Result};
use axum::{
    extract::{State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post, put},
    Router,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    sync::{broadcast, RwLock},
    time::sleep,
};
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::mcp::{
    auth::{AuthManager, AuthLevel, AuthConfig},
    events::{EventManager, EventType, VeritasEvent},
    tools::{
        ToolHandler,
        model_tools::ModelToolsHandler,
        training_tools::TrainingToolsHandler,
        inference_tools::InferenceToolsHandler,
        monitoring_tools::MonitoringToolsHandler,
    },
    resources::{
        model_resources::ModelResourcesHandler,
        data_resources::DataResourcesHandler,
    },
    McpError, McpResult, RequestMetadata, ResponseMetadata, McpResponse,
    HealthStatus, ServerCapabilities, ServerVersion,
};

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Enable authentication
    pub enable_auth: bool,
    /// Authentication configuration
    pub auth_config: AuthConfig,
    /// Maximum request size in bytes
    pub max_request_size: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Enable WebSocket support
    pub enable_websockets: bool,
    /// Enable CORS
    pub enable_cors: bool,
    /// Log level
    pub log_level: String,
    /// Model storage path
    pub model_storage_path: String,
    /// Dataset storage path
    pub dataset_storage_path: String,
    /// Results storage path
    pub results_storage_path: String,
    /// Maximum model cache size in MB
    pub max_model_cache_mb: u64,
    /// Event buffer size
    pub event_buffer_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 3000,
            enable_auth: true,
            auth_config: AuthConfig::default(),
            max_request_size: 50 * 1024 * 1024, // 50MB
            request_timeout_seconds: 300, // 5 minutes
            max_connections: 100,
            enable_websockets: true,
            enable_cors: true,
            log_level: "info".to_string(),
            model_storage_path: "./models".to_string(),
            dataset_storage_path: "./datasets".to_string(),
            results_storage_path: "./results".to_string(),
            max_model_cache_mb: 2048, // 2GB
            event_buffer_size: 10000,
        }
    }
}

/// Server state shared across all handlers
#[derive(Debug)]
pub struct ServerState {
    /// Server configuration
    pub config: ServerConfig,
    /// Authentication manager
    pub auth_manager: AuthManager,
    /// Event manager for real-time updates
    pub event_manager: EventManager,
    /// Model tools handler
    pub model_tools: ModelToolsHandler,
    /// Training tools handler
    pub training_tools: TrainingToolsHandler,
    /// Inference tools handler
    pub inference_tools: InferenceToolsHandler,
    /// Monitoring tools handler
    pub monitoring_tools: MonitoringToolsHandler,
    /// Model resources handler
    pub model_resources: ModelResourcesHandler,
    /// Data resources handler
    pub data_resources: DataResourcesHandler,
    /// Server startup time
    pub startup_time: Instant,
    /// Request counter
    pub request_count: Arc<RwLock<u64>>,
    /// Error counter
    pub error_count: Arc<RwLock<u64>>,
    /// Active connections
    pub active_connections: Arc<RwLock<u32>>,
    /// Active sessions
    pub active_sessions: Arc<RwLock<HashMap<String, SessionInfo>>>,
}

/// Session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub client_id: String,
    pub auth_level: AuthLevel,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub request_count: u64,
}

/// Main MCP server implementation
#[derive(Debug)]
pub struct VeritasServer {
    state: Arc<ServerState>,
    shutdown_tx: Option<broadcast::Sender<()>>,
}

impl VeritasServer {
    /// Create a new MCP server with the given configuration
    pub async fn new(config: ServerConfig) -> Result<Self> {
        info!("Initializing Veritas MCP Server...");

        // Initialize authentication manager
        let auth_manager = AuthManager::new(config.auth_config.clone()).await
            .context("Failed to initialize authentication manager")?;

        // Initialize event manager
        let event_manager = EventManager::new(config.event_buffer_size).await
            .context("Failed to initialize event manager")?;

        // Initialize tool handlers
        let model_tools = ModelToolsHandler::new(&config).await
            .context("Failed to initialize model tools handler")?;
        
        let training_tools = TrainingToolsHandler::new(&config).await
            .context("Failed to initialize training tools handler")?;
        
        let inference_tools = InferenceToolsHandler::new(&config).await
            .context("Failed to initialize inference tools handler")?;
        
        let monitoring_tools = MonitoringToolsHandler::new(&config).await
            .context("Failed to initialize monitoring tools handler")?;

        // Initialize resource handlers
        let model_resources = ModelResourcesHandler::new(&config).await
            .context("Failed to initialize model resources handler")?;
        
        let data_resources = DataResourcesHandler::new(&config).await
            .context("Failed to initialize data resources handler")?;

        let state = Arc::new(ServerState {
            config,
            auth_manager,
            event_manager,
            model_tools,
            training_tools,
            inference_tools,
            monitoring_tools,
            model_resources,
            data_resources,
            startup_time: Instant::now(),
            request_count: Arc::new(RwLock::new(0)),
            error_count: Arc::new(RwLock::new(0)),
            active_connections: Arc::new(RwLock::new(0)),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        });

        info!("Veritas MCP Server initialized successfully");

        Ok(Self {
            state,
            shutdown_tx: None,
        })
    }

    /// Start the MCP server
    pub async fn run(mut self) -> Result<()> {
        let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        let addr = SocketAddr::from(([127, 0, 0, 1], self.state.config.port));
        
        info!("Starting Veritas MCP Server on {}", addr);

        // Build the router
        let app = self.build_router().await?;

        // Create the listener
        let listener = TcpListener::bind(addr).await
            .with_context(|| format!("Failed to bind to address {}", addr))?;

        info!("Server listening on {}", addr);

        // Start background tasks
        self.start_background_tasks().await?;

        // Serve the application
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown_rx.recv().await.ok();
                info!("Graceful shutdown initiated");
            })
            .await
            .context("Server failed to start")?;

        info!("Server shutdown complete");
        Ok(())
    }

    /// Build the HTTP router with all endpoints
    async fn build_router(&self) -> Result<Router> {
        let mut router = Router::new()
            // Health and status endpoints
            .route("/health", get(health_handler))
            .route("/status", get(status_handler))
            .route("/capabilities", get(capabilities_handler))
            .route("/version", get(version_handler))
            
            // Model management endpoints
            .route("/api/v1/models", get(list_models_handler))
            .route("/api/v1/models", post(create_model_handler))
            .route("/api/v1/models/:model_id", get(get_model_handler))
            .route("/api/v1/models/:model_id", put(update_model_handler))
            .route("/api/v1/models/:model_id/load", post(load_model_handler))
            .route("/api/v1/models/:model_id/unload", post(unload_model_handler))
            
            // Training endpoints
            .route("/api/v1/training/start", post(start_training_handler))
            .route("/api/v1/training/:training_id/status", get(training_status_handler))
            .route("/api/v1/training/:training_id/stop", post(stop_training_handler))
            
            // Inference endpoints
            .route("/api/v1/inference/analyze", post(analyze_handler))
            .route("/api/v1/inference/stream/start", post(start_stream_handler))
            .route("/api/v1/inference/stream/:stream_id/stop", post(stop_stream_handler))
            
            // Monitoring endpoints
            .route("/api/v1/monitor/metrics", get(metrics_handler))
            .route("/api/v1/monitor/alerts", get(list_alerts_handler))
            .route("/api/v1/monitor/alerts", post(create_alert_handler))
            
            // Configuration endpoints
            .route("/api/v1/config", get(get_config_handler))
            .route("/api/v1/config", put(update_config_handler))
            
            // Resource endpoints
            .route("/api/v1/resources/models/:model_id/weights", get(model_weights_handler))
            .route("/api/v1/resources/models/:model_id/config", get(model_config_handler))
            .route("/api/v1/resources/datasets/:dataset_id/manifest", get(dataset_manifest_handler))
            .route("/api/v1/resources/datasets/:dataset_id/samples", get(dataset_samples_handler))
            .route("/api/v1/resources/results/:analysis_id/report", get(analysis_report_handler))
            .route("/api/v1/resources/results/:analysis_id/media", get(analysis_media_handler))
            
            // State management
            .with_state(self.state.clone());

        // Add WebSocket endpoints if enabled
        if self.state.config.enable_websockets {
            router = router
                .route("/ws/events", get(events_websocket_handler))
                .route("/ws/inference", get(inference_websocket_handler))
                .route("/ws/monitoring", get(monitoring_websocket_handler));
        }

        // Add middleware layers
        let middleware = ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware,
            ))
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                request_counter_middleware,
            ));

        if self.state.config.enable_cors {
            let cors = CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any);
            
            router = router.layer(cors);
        }

        Ok(router.layer(middleware))
    }

    /// Start background tasks for monitoring and maintenance
    async fn start_background_tasks(&self) -> Result<()> {
        let state = self.state.clone();
        
        // Start session cleanup task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            loop {
                interval.tick().await;
                if let Err(e) = cleanup_expired_sessions(&state).await {
                    warn!("Session cleanup failed: {}", e);
                }
            }
        });

        // Start metrics collection task
        let state = self.state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // 1 minute
            loop {
                interval.tick().await;
                if let Err(e) = collect_system_metrics(&state).await {
                    warn!("Metrics collection failed: {}", e);
                }
            }
        });

        // Start event cleanup task
        let state = self.state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour
            loop {
                interval.tick().await;
                if let Err(e) = state.event_manager.cleanup_old_events().await {
                    warn!("Event cleanup failed: {}", e);
                }
            }
        });

        info!("Background tasks started successfully");
        Ok(())
    }

    /// Gracefully shutdown the server
    pub async fn shutdown(&self) -> Result<()> {
        info!("Initiating server shutdown...");
        
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }

        // Cleanup resources
        self.state.model_tools.shutdown().await?;
        self.state.training_tools.shutdown().await?;
        self.state.inference_tools.shutdown().await?;
        
        info!("Server shutdown complete");
        Ok(())
    }

    /// Get server state for testing
    #[cfg(test)]
    pub fn state(&self) -> Arc<ServerState> {
        self.state.clone()
    }
}

// Middleware implementations
async fn auth_middleware(
    State(state): State<Arc<ServerState>>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    if !state.config.enable_auth {
        return next.run(request).await;
    }

    // Extract authorization header
    let auth_result = if let Some(auth_header) = request.headers().get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            state.auth_manager.validate_token(auth_str).await
        } else {
            Err(McpError::AuthFailed {
                reason: "Invalid authorization header format".to_string(),
            })
        }
    } else {
        Err(McpError::AuthFailed {
            reason: "Missing authorization header".to_string(),
        })
    };

    match auth_result {
        Ok(auth_info) => {
            // Add auth info to request extensions
            request.extensions_mut().insert(auth_info);
            next.run(request).await
        }
        Err(e) => {
            error!("Authentication failed: {}", e);
            (StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                "error": "Authentication required",
                "message": e.to_string()
            }))).into_response()
        }
    }
}

async fn request_counter_middleware(
    State(state): State<Arc<ServerState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    // Increment request counter
    {
        let mut count = state.request_count.write().await;
        *count += 1;
    }

    // Increment active connections
    {
        let mut connections = state.active_connections.write().await;
        *connections += 1;
    }

    let response = next.run(request).await;

    // Decrement active connections
    {
        let mut connections = state.active_connections.write().await;
        if *connections > 0 {
            *connections -= 1;
        }
    }

    response
}

// Handler implementations will be added in the tool-specific modules
// These are placeholder implementations

async fn health_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let uptime = state.startup_time.elapsed();
    let status = HealthStatus {
        status: "healthy".to_string(),
        uptime_seconds: uptime.as_secs(),
        memory_usage_mb: get_memory_usage_mb(),
        cpu_usage_percent: get_cpu_usage_percent(),
        active_models: state.model_tools.get_active_model_count().await,
        active_sessions: state.active_sessions.read().await.len() as u32,
        total_requests: *state.request_count.read().await,
        error_rate_percent: calculate_error_rate(&state).await,
    };
    
    Json(status)
}

async fn status_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "timestamp": chrono::Utc::now()
    }))
}

async fn capabilities_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    Json(crate::mcp::get_server_capabilities())
}

async fn version_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    Json(crate::mcp::get_server_version())
}

// Placeholder handler implementations - these will be implemented in tool modules
async fn list_models_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    state.model_tools.list_models(RequestMetadata::new(AuthLevel::ReadOnly)).await.into_response()
}

async fn create_model_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn get_model_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn update_model_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn load_model_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn unload_model_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn start_training_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn training_status_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn stop_training_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn analyze_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn start_stream_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn stop_stream_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn metrics_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn list_alerts_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn create_alert_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn get_config_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn update_config_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn model_weights_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn model_config_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn dataset_manifest_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn dataset_samples_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn analysis_report_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn analysis_media_handler(State(_state): State<Arc<ServerState>>) -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}

async fn events_websocket_handler(
    ws: WebSocketUpgrade,
    State(_state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(handle_events_websocket)
}

async fn inference_websocket_handler(
    ws: WebSocketUpgrade,
    State(_state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(handle_inference_websocket)
}

async fn monitoring_websocket_handler(
    ws: WebSocketUpgrade,
    State(_state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(handle_monitoring_websocket)
}

// WebSocket handlers
async fn handle_events_websocket(mut socket: axum::extract::ws::WebSocket) {
    // WebSocket event streaming implementation
    while let Some(msg) = socket.recv().await {
        if msg.is_err() {
            break;
        }
        // Handle incoming messages and send events
    }
}

async fn handle_inference_websocket(mut socket: axum::extract::ws::WebSocket) {
    // WebSocket inference streaming implementation
    while let Some(msg) = socket.recv().await {
        if msg.is_err() {
            break;
        }
        // Handle real-time inference
    }
}

async fn handle_monitoring_websocket(mut socket: axum::extract::ws::WebSocket) {
    // WebSocket monitoring streaming implementation
    while let Some(msg) = socket.recv().await {
        if msg.is_err() {
            break;
        }
        // Handle monitoring updates
    }
}

// Utility functions
async fn cleanup_expired_sessions(state: &ServerState) -> Result<()> {
    let mut sessions = state.active_sessions.write().await;
    let now = Instant::now();
    let timeout = Duration::from_secs(3600); // 1 hour timeout
    
    sessions.retain(|_, session| {
        now.duration_since(session.last_activity) < timeout
    });
    
    Ok(())
}

async fn collect_system_metrics(state: &ServerState) -> Result<()> {
    // Collect and publish system metrics
    let event = VeritasEvent {
        event_type: EventType::SystemResourceAlert,
        timestamp: chrono::Utc::now(),
        data: serde_json::json!({
            "memory_usage_mb": get_memory_usage_mb(),
            "cpu_usage_percent": get_cpu_usage_percent(),
            "active_connections": *state.active_connections.read().await,
        }),
        metadata: serde_json::json!({
            "source": "metrics_collector"
        }),
    };
    
    state.event_manager.publish_event(event).await?;
    Ok(())
}

fn get_memory_usage_mb() -> f64 {
    // Placeholder implementation - would use actual system metrics
    128.0
}

fn get_cpu_usage_percent() -> f64 {
    // Placeholder implementation - would use actual system metrics
    15.0
}

async fn calculate_error_rate(state: &ServerState) -> f64 {
    let total_requests = *state.request_count.read().await;
    let error_count = *state.error_count.read().await;
    
    if total_requests == 0 {
        0.0
    } else {
        (error_count as f64 / total_requests as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let config = ServerConfig::default();
        let server = VeritasServer::new(config).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 3000);
        assert!(config.enable_auth);
        assert!(config.enable_websockets);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_session_info() {
        let session = SessionInfo {
            session_id: "test-session".to_string(),
            client_id: "test-client".to_string(),
            auth_level: AuthLevel::Admin,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            request_count: 0,
        };
        
        assert_eq!(session.session_id, "test-session");
        assert_eq!(session.auth_level, AuthLevel::Admin);
    }
}