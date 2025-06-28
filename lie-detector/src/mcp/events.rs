//! Event Management and Real-Time Streaming for Veritas MCP Server
//!
//! This module provides comprehensive event management capabilities for the MCP server,
//! including real-time event streaming, event persistence, and subscription management.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    pin::Pin,
    sync::Arc,
    task::{Context as TaskContext, Poll},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, RwLock},
    time::sleep,
};
use tokio_stream::wrappers::BroadcastStream;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::mcp::{McpError, McpResult};

/// Event types supported by the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // Training events
    TrainingStarted,
    TrainingEpochCompleted,
    TrainingMetricUpdate,
    TrainingCompleted,
    TrainingFailed,
    
    // Inference events
    InferenceStarted,
    InferenceResult,
    InferenceStreamUpdate,
    InferenceCompleted,
    
    // System events
    SystemResourceAlert,
    SystemModelLoaded,
    SystemError,
    
    // Model events
    ModelCreated,
    ModelUpdated,
    ModelDeleted,
    
    // User events
    UserLogin,
    UserLogout,
    UserAction,
    
    // Custom events
    Custom(String),
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::TrainingStarted => write!(f, "training.started"),
            EventType::TrainingEpochCompleted => write!(f, "training.epoch_completed"),
            EventType::TrainingMetricUpdate => write!(f, "training.metric_update"),
            EventType::TrainingCompleted => write!(f, "training.completed"),
            EventType::TrainingFailed => write!(f, "training.failed"),
            EventType::InferenceStarted => write!(f, "inference.started"),
            EventType::InferenceResult => write!(f, "inference.result"),
            EventType::InferenceStreamUpdate => write!(f, "inference.stream_update"),
            EventType::InferenceCompleted => write!(f, "inference.completed"),
            EventType::SystemResourceAlert => write!(f, "system.resource_alert"),
            EventType::SystemModelLoaded => write!(f, "system.model_loaded"),
            EventType::SystemError => write!(f, "system.error"),
            EventType::ModelCreated => write!(f, "model.created"),
            EventType::ModelUpdated => write!(f, "model.updated"),
            EventType::ModelDeleted => write!(f, "model.deleted"),
            EventType::UserLogin => write!(f, "user.login"),
            EventType::UserLogout => write!(f, "user.logout"),
            EventType::UserAction => write!(f, "user.action"),
            EventType::Custom(name) => write!(f, "custom.{}", name),
        }
    }
}

/// Main event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeritasEvent {
    /// Unique event identifier
    pub id: String,
    /// Event type
    pub event_type: EventType,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event data payload
    pub data: serde_json::Value,
    /// Additional metadata
    pub metadata: serde_json::Value,
    /// Event source identifier
    pub source: Option<String>,
    /// Related session ID
    pub session_id: Option<String>,
    /// Related user ID
    pub user_id: Option<String>,
    /// Event priority level
    pub priority: EventPriority,
}

impl VeritasEvent {
    /// Create a new event
    pub fn new(
        event_type: EventType,
        data: serde_json::Value,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: Utc::now(),
            data,
            metadata,
            source: None,
            session_id: None,
            user_id: None,
            priority: EventPriority::Normal,
        }
    }

    /// Create a new event with additional context
    pub fn with_context(
        event_type: EventType,
        data: serde_json::Value,
        metadata: serde_json::Value,
        source: Option<String>,
        session_id: Option<String>,
        user_id: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: Utc::now(),
            data,
            metadata,
            source,
            session_id,
            user_id,
            priority: EventPriority::Normal,
        }
    }

    /// Set event priority
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Event subscription configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSubscription {
    pub id: String,
    pub event_types: Vec<EventType>,
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub filter: Option<EventFilter>,
}

/// Event filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    pub priority_threshold: Option<EventPriority>,
    pub source_filter: Option<Vec<String>>,
    pub metadata_filter: Option<HashMap<String, serde_json::Value>>,
    pub time_window: Option<Duration>,
}

/// Event statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStats {
    pub total_events: u64,
    pub events_by_type: HashMap<String, u64>,
    pub events_by_priority: HashMap<String, u64>,
    pub active_subscriptions: u64,
    pub buffer_usage_percent: f64,
    pub last_event_timestamp: Option<DateTime<Utc>>,
}

/// Persisted event for long-term storage
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedEvent {
    event: VeritasEvent,
    persistence_timestamp: DateTime<Utc>,
}

/// Main event manager
#[derive(Debug)]
pub struct EventManager {
    /// Event broadcast channel
    event_tx: broadcast::Sender<VeritasEvent>,
    
    /// Event buffer for recent events
    event_buffer: Arc<RwLock<VecDeque<PersistedEvent>>>,
    
    /// Maximum buffer size
    max_buffer_size: usize,
    
    /// Active subscriptions
    subscriptions: Arc<RwLock<HashMap<String, EventSubscription>>>,
    
    /// Event statistics
    stats: Arc<RwLock<EventStats>>,
    
    /// Event persistence enabled
    persistence_enabled: bool,
    
    /// Event retention duration
    retention_duration: Duration,
}

impl EventManager {
    /// Create a new event manager
    pub async fn new(buffer_size: usize) -> Result<Self> {
        info!("Initializing event manager with buffer size: {}", buffer_size);

        let (event_tx, _) = broadcast::channel(buffer_size);
        
        let manager = Self {
            event_tx,
            event_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            max_buffer_size: buffer_size,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(EventStats {
                total_events: 0,
                events_by_type: HashMap::new(),
                events_by_priority: HashMap::new(),
                active_subscriptions: 0,
                buffer_usage_percent: 0.0,
                last_event_timestamp: None,
            })),
            persistence_enabled: true,
            retention_duration: Duration::from_secs(7 * 24 * 3600), // 7 days
        };

        info!("Event manager initialized successfully");
        Ok(manager)
    }

    /// Publish an event to all subscribers
    pub async fn publish_event(&self, event: VeritasEvent) -> McpResult<()> {
        debug!("Publishing event: {} - {}", event.event_type, event.id);

        // Store event in buffer
        if self.persistence_enabled {
            let persisted_event = PersistedEvent {
                event: event.clone(),
                persistence_timestamp: Utc::now(),
            };

            let mut buffer = self.event_buffer.write().await;
            
            // Remove old events if buffer is full
            while buffer.len() >= self.max_buffer_size {
                buffer.pop_front();
            }
            
            buffer.push_back(persisted_event);
        }

        // Update statistics
        self.update_stats(&event).await;

        // Broadcast event to subscribers
        if let Err(e) = self.event_tx.send(event.clone()) {
            warn!("Failed to broadcast event: {}", e);
            return Err(McpError::Internal {
                message: format!("Failed to broadcast event: {}", e),
            });
        }

        debug!("Event published successfully: {}", event.id);
        Ok(())
    }

    /// Subscribe to events with optional filtering
    pub async fn subscribe(
        &self,
        event_types: Vec<EventType>,
        session_id: Option<String>,
        user_id: Option<String>,
        filter: Option<EventFilter>,
    ) -> McpResult<EventSubscription> {
        let subscription = EventSubscription {
            id: Uuid::new_v4().to_string(),
            event_types,
            session_id,
            user_id,
            created_at: Utc::now(),
            last_activity: Utc::now(),
            filter,
        };

        let mut subscriptions = self.subscriptions.write().await;
        subscriptions.insert(subscription.id.clone(), subscription.clone());

        // Update subscription count in stats
        {
            let mut stats = self.stats.write().await;
            stats.active_subscriptions = subscriptions.len() as u64;
        }

        info!("Created event subscription: {}", subscription.id);
        Ok(subscription)
    }

    /// Unsubscribe from events
    pub async fn unsubscribe(&self, subscription_id: &str) -> McpResult<()> {
        let mut subscriptions = self.subscriptions.write().await;
        
        if subscriptions.remove(subscription_id).is_some() {
            // Update subscription count in stats
            {
                let mut stats = self.stats.write().await;
                stats.active_subscriptions = subscriptions.len() as u64;
            }
            
            info!("Removed event subscription: {}", subscription_id);
            Ok(())
        } else {
            Err(McpError::Internal {
                message: format!("Subscription not found: {}", subscription_id),
            })
        }
    }

    /// Get event stream for a subscription
    pub async fn get_event_stream(&self, subscription_id: &str) -> McpResult<EventStream> {
        let subscriptions = self.subscriptions.read().await;
        
        if let Some(subscription) = subscriptions.get(subscription_id) {
            let receiver = self.event_tx.subscribe();
            Ok(EventStream::new(receiver, subscription.clone()))
        } else {
            Err(McpError::Internal {
                message: format!("Subscription not found: {}", subscription_id),
            })
        }
    }

    /// Get historical events from buffer
    pub async fn get_historical_events(
        &self,
        event_types: Option<Vec<EventType>>,
        since: Option<DateTime<Utc>>,
        limit: Option<usize>,
    ) -> Vec<VeritasEvent> {
        let buffer = self.event_buffer.read().await;
        let mut events: Vec<VeritasEvent> = buffer
            .iter()
            .filter_map(|persisted| {
                let event = &persisted.event;
                
                // Filter by event types
                if let Some(ref types) = event_types {
                    if !types.contains(&event.event_type) {
                        return None;
                    }
                }
                
                // Filter by timestamp
                if let Some(since_time) = since {
                    if event.timestamp < since_time {
                        return None;
                    }
                }
                
                Some(event.clone())
            })
            .collect();

        // Sort by timestamp (newest first)
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Apply limit
        if let Some(limit) = limit {
            events.truncate(limit);
        }

        events
    }

    /// Get event statistics
    pub async fn get_stats(&self) -> EventStats {
        let stats = self.stats.read().await;
        let buffer = self.event_buffer.read().await;
        
        let mut stats_copy = stats.clone();
        stats_copy.buffer_usage_percent = (buffer.len() as f64 / self.max_buffer_size as f64) * 100.0;
        
        stats_copy
    }

    /// Cleanup old events from buffer
    pub async fn cleanup_old_events(&self) -> McpResult<()> {
        let cutoff_time = Utc::now() - chrono::Duration::from_std(self.retention_duration)
            .map_err(|e| McpError::Internal {
                message: format!("Invalid retention duration: {}", e),
            })?;

        let mut buffer = self.event_buffer.write().await;
        let initial_size = buffer.len();
        
        buffer.retain(|persisted| persisted.event.timestamp > cutoff_time);
        
        let removed_count = initial_size - buffer.len();
        if removed_count > 0 {
            info!("Cleaned up {} old events", removed_count);
        }

        Ok(())
    }

    /// Update event statistics
    async fn update_stats(&self, event: &VeritasEvent) {
        let mut stats = self.stats.write().await;
        
        stats.total_events += 1;
        stats.last_event_timestamp = Some(event.timestamp);
        
        // Update event type counts
        let type_key = event.event_type.to_string();
        *stats.events_by_type.entry(type_key).or_insert(0) += 1;
        
        // Update priority counts
        let priority_key = format!("{:?}", event.priority);
        *stats.events_by_priority.entry(priority_key).or_insert(0) += 1;
    }

    /// Shutdown the event manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down event manager");
        
        // Clear subscriptions
        {
            let mut subscriptions = self.subscriptions.write().await;
            subscriptions.clear();
        }
        
        // Optionally persist remaining events to storage
        if self.persistence_enabled {
            // Implementation would depend on storage backend
            debug!("Event buffer contains {} events", self.event_buffer.read().await.len());
        }
        
        info!("Event manager shutdown complete");
        Ok(())
    }
}

/// Event stream implementation
pub struct EventStream {
    receiver: BroadcastStream<VeritasEvent>,
    subscription: EventSubscription,
}

impl EventStream {
    fn new(receiver: broadcast::Receiver<VeritasEvent>, subscription: EventSubscription) -> Self {
        Self {
            receiver: BroadcastStream::new(receiver),
            subscription,
        }
    }

    /// Check if event matches subscription filter
    fn matches_filter(&self, event: &VeritasEvent) -> bool {
        // Check event type
        if !self.subscription.event_types.contains(&event.event_type) {
            return false;
        }

        // Apply additional filters if present
        if let Some(ref filter) = self.subscription.filter {
            // Priority filter
            if let Some(threshold) = filter.priority_threshold {
                if (event.priority as u8) < (threshold as u8) {
                    return false;
                }
            }
            
            // Source filter
            if let Some(ref sources) = filter.source_filter {
                if let Some(ref event_source) = event.source {
                    if !sources.contains(event_source) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            
            // Metadata filter
            if let Some(ref metadata_filter) = filter.metadata_filter {
                for (key, value) in metadata_filter {
                    if event.metadata.get(key) != Some(value) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl Stream for EventStream {
    type Item = Result<VeritasEvent, broadcast::error::RecvError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.receiver).poll_next(cx) {
                Poll::Ready(Some(Ok(event))) => {
                    if self.matches_filter(&event) {
                        return Poll::Ready(Some(Ok(event)));
                    }
                    // Continue polling if event doesn't match filter
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_event_manager_creation() {
        let manager = EventManager::new(100).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_event_publishing() {
        let manager = EventManager::new(100).await.unwrap();
        
        let event = VeritasEvent::new(
            EventType::SystemResourceAlert,
            serde_json::json!({"cpu_usage": 80.0}),
            serde_json::json!({"source": "monitor"}),
        );

        let result = manager.publish_event(event).await;
        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_events, 1);
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let manager = EventManager::new(100).await.unwrap();
        
        let subscription = manager.subscribe(
            vec![EventType::TrainingStarted, EventType::TrainingCompleted],
            Some("test-session".to_string()),
            Some("test-user".to_string()),
            None,
        ).await;

        assert!(subscription.is_ok());
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_subscriptions, 1);
    }

    #[tokio::test]
    async fn test_event_stream() {
        let manager = EventManager::new(100).await.unwrap();
        
        let subscription = manager.subscribe(
            vec![EventType::TrainingStarted],
            None,
            None,
            None,
        ).await.unwrap();

        let mut stream = manager.get_event_stream(&subscription.id).await.unwrap();

        // Publish an event
        tokio::spawn({
            let manager = Arc::new(manager);
            async move {
                sleep(Duration::from_millis(10)).await;
                let event = VeritasEvent::new(
                    EventType::TrainingStarted,
                    serde_json::json!({"model_id": "test-model"}),
                    serde_json::json!({}),
                );
                let _ = manager.publish_event(event).await;
            }
        });

        // Receive the event
        let received_event = tokio::time::timeout(
            Duration::from_millis(100),
            stream.next(),
        ).await;

        assert!(received_event.is_ok());
        let event = received_event.unwrap().unwrap().unwrap();
        assert_eq!(event.event_type, EventType::TrainingStarted);
    }

    #[tokio::test]
    async fn test_historical_events() {
        let manager = EventManager::new(100).await.unwrap();
        
        // Publish some events
        for i in 0..5 {
            let event = VeritasEvent::new(
                EventType::TrainingMetricUpdate,
                serde_json::json!({"epoch": i}),
                serde_json::json!({}),
            );
            let _ = manager.publish_event(event).await;
        }

        let events = manager.get_historical_events(
            Some(vec![EventType::TrainingMetricUpdate]),
            None,
            Some(3),
        ).await;

        assert_eq!(events.len(), 3);
        // Events should be sorted by timestamp (newest first)
        assert!(events[0].timestamp >= events[1].timestamp);
        assert!(events[1].timestamp >= events[2].timestamp);
    }

    #[test]
    fn test_event_priority() {
        let event = VeritasEvent::new(
            EventType::SystemError,
            serde_json::json!({"error": "test"}),
            serde_json::json!({}),
        ).with_priority(EventPriority::Critical);

        assert_eq!(event.priority, EventPriority::Critical);
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(EventType::TrainingStarted.to_string(), "training.started");
        assert_eq!(EventType::InferenceResult.to_string(), "inference.result");
        assert_eq!(EventType::SystemError.to_string(), "system.error");
        assert_eq!(EventType::Custom("test".to_string()).to_string(), "custom.test");
    }
}