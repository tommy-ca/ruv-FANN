//! CRUD operation tests for persistence layer

use crate::memory::MemoryStorage;
use crate::models::*;
use crate::*;
use chrono::Utc;
use uuid::Uuid;

#[tokio::test]
async fn test_agent_create_read() {
    let storage = MemoryStorage::new();

    let agent = AgentModel {
        id: "test-agent-1".to_string(),
        name: "Test Agent".to_string(),
        agent_type: "neural".to_string(),
        status: "idle".to_string(),
        capabilities: serde_json::json!(["compute", "analyze"]),
        config: serde_json::json!({"threads": 4}),
        created_at: Utc::now().timestamp(),
        updated_at: Utc::now().timestamp(),
    };

    // Create
    storage.store_agent(&agent).await.unwrap();

    // Read
    let retrieved = storage.get_agent(&agent.id).await.unwrap();
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, agent.id);
    assert_eq!(retrieved.name, agent.name);
    assert_eq!(retrieved.agent_type, agent.agent_type);
    assert_eq!(retrieved.status, agent.status);
}

#[tokio::test]
async fn test_agent_update() {
    let storage = MemoryStorage::new();

    let mut agent = AgentModel {
        id: "test-agent-2".to_string(),
        name: "Update Test Agent".to_string(),
        agent_type: "compute".to_string(),
        status: "idle".to_string(),
        capabilities: serde_json::json!([]),
        config: serde_json::json!({}),
        created_at: Utc::now().timestamp(),
        updated_at: Utc::now().timestamp(),
    };

    storage.store_agent(&agent).await.unwrap();

    // Update fields
    agent.status = "running".to_string();
    agent.capabilities = serde_json::json!(["neural", "quantum"]);
    agent.updated_at = Utc::now().timestamp();

    storage.update_agent(&agent).await.unwrap();

    // Verify update
    let retrieved = storage.get_agent(&agent.id).await.unwrap().unwrap();
    assert_eq!(retrieved.status, "running");
    assert_eq!(
        retrieved.capabilities,
        serde_json::json!(["neural", "quantum"])
    );
}

#[tokio::test]
async fn test_agent_delete() {
    let storage = MemoryStorage::new();

    let agent = AgentModel {
        id: "test-agent-3".to_string(),
        name: "Delete Test Agent".to_string(),
        agent_type: "temporary".to_string(),
        status: "idle".to_string(),
        capabilities: serde_json::json!([]),
        config: serde_json::json!({}),
        created_at: Utc::now().timestamp(),
        updated_at: Utc::now().timestamp(),
    };

    storage.store_agent(&agent).await.unwrap();

    // Verify exists
    assert!(storage.get_agent(&agent.id).await.unwrap().is_some());

    // Delete
    storage.delete_agent(&agent.id).await.unwrap();

    // Verify deleted
    assert!(storage.get_agent(&agent.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_task_crud_operations() {
    let storage = MemoryStorage::new();

    let task = TaskModel {
        id: Uuid::new_v4().to_string(),
        task_type: "training".to_string(),
        status: "pending".to_string(),
        priority: 5,
        payload: serde_json::json!({"model": "neural-net-v2", "epochs": 100}),
        assigned_to: None,
        result: None,
        error: None,
        retry_count: 0,
        created_at: Utc::now().timestamp(),
        updated_at: Utc::now().timestamp(),
        completed_at: None,
    };

    // Create
    storage.store_task(&task).await.unwrap();

    // Read
    let retrieved = storage.get_task(&task.id).await.unwrap();
    assert!(retrieved.is_some());

    let mut retrieved = retrieved.unwrap();
    assert_eq!(retrieved.task_type, "training");
    assert_eq!(retrieved.priority, 5);

    // Update
    retrieved.status = "running".to_string();
    retrieved.assigned_to = Some("agent-123".to_string());
    storage.update_task(&retrieved).await.unwrap();

    // Verify update
    let updated = storage.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "running");
    assert_eq!(updated.assigned_to, Some("agent-123".to_string()));
}

#[tokio::test]
async fn test_event_crud() {
    let storage = MemoryStorage::new();

    let event = EventModel {
        id: Uuid::new_v4().to_string(),
        event_type: "task_started".to_string(),
        agent_id: Some("agent-1".to_string()),
        task_id: Some("task-1".to_string()),
        data: serde_json::json!({
            "start_time": Utc::now().to_rfc3339(),
            "resources": {"cpu": 4, "memory": "8GB"}
        }),
        timestamp: Utc::now().timestamp(),
    };

    // Create
    storage.store_event(&event).await.unwrap();

    // Read by agent
    let events = storage.get_events_by_agent("agent-1", 10).await.unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, "task_started");

    // Read by type
    let typed_events = storage
        .get_events_by_type("task_started", 10)
        .await
        .unwrap();
    assert_eq!(typed_events.len(), 1);
}

#[tokio::test]
async fn test_message_crud() {
    let storage = MemoryStorage::new();

    let message = MessageModel {
        id: Uuid::new_v4().to_string(),
        sender_id: "agent-sender".to_string(),
        recipient_id: "agent-recipient".to_string(),
        message_type: "coordination".to_string(),
        payload: serde_json::json!({
            "action": "synchronize",
            "data": {"checkpoint": 42}
        }),
        read: false,
        timestamp: Utc::now().timestamp(),
    };

    // Create
    storage.store_message(&message).await.unwrap();

    // Read unread
    let unread = storage
        .get_unread_messages("agent-recipient")
        .await
        .unwrap();
    assert_eq!(unread.len(), 1);
    assert_eq!(unread[0].id, message.id);

    // Mark as read
    storage.mark_message_read(&message.id).await.unwrap();

    // Verify marked as read
    let unread_after = storage
        .get_unread_messages("agent-recipient")
        .await
        .unwrap();
    assert_eq!(unread_after.len(), 0);
}

#[tokio::test]
async fn test_metric_crud() {
    let storage = MemoryStorage::new();

    let agent_id = "metric-agent".to_string();

    // Create multiple metrics
    for i in 0..5 {
        let metric = MetricModel {
            id: Uuid::new_v4().to_string(),
            agent_id: agent_id.clone(),
            metric_type: "performance".to_string(),
            value: 80.0 + (i as f64),
            tags: serde_json::json!({
                "task": format!("task-{}", i),
                "node": "compute-1"
            }),
            timestamp: Utc::now().timestamp() + i,
        };

        storage.store_metric(&metric).await.unwrap();
    }

    // Read metrics
    let metrics = storage
        .get_metrics_by_agent(&agent_id, "performance")
        .await
        .unwrap();
    assert_eq!(metrics.len(), 5);

    // Verify ordering and values
    for (i, metric) in metrics.iter().enumerate() {
        assert_eq!(metric.value, 80.0 + (i as f64));
    }
}

#[tokio::test]
async fn test_bulk_operations() {
    let storage = MemoryStorage::new();

    // Bulk create agents
    let agent_ids: Vec<String> = (0..100).map(|i| format!("bulk-agent-{}", i)).collect();

    for id in &agent_ids {
        let agent = AgentModel {
            id: id.clone(),
            name: format!("Bulk Agent {}", id),
            agent_type: "worker".to_string(),
            status: "idle".to_string(),
            capabilities: serde_json::json!([]),
            config: serde_json::json!({}),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
        };

        storage.store_agent(&agent).await.unwrap();
    }

    // Verify all created
    let all_agents = storage.list_agents().await.unwrap();
    assert!(all_agents.len() >= 100);

    // Bulk read specific agents
    for id in agent_ids.iter().take(10) {
        let agent = storage.get_agent(id).await.unwrap();
        assert!(agent.is_some());
    }
}

#[tokio::test]
async fn test_complex_task_workflow() {
    let storage = MemoryStorage::new();

    let agent_id = "workflow-agent".to_string();
    let task_id = Uuid::new_v4().to_string();

    // 1. Create agent
    let agent = AgentModel {
        id: agent_id.clone(),
        name: "Workflow Agent".to_string(),
        agent_type: "orchestrator".to_string(),
        status: "idle".to_string(),
        capabilities: serde_json::json!(["coordinate", "monitor"]),
        config: serde_json::json!({}),
        created_at: Utc::now().timestamp(),
        updated_at: Utc::now().timestamp(),
    };
    storage.store_agent(&agent).await.unwrap();

    // 2. Create task
    let task = TaskModel {
        id: task_id.clone(),
        task_type: "complex_workflow".to_string(),
        status: "pending".to_string(),
        priority: 10,
        payload: serde_json::json!({"steps": 5}),
        assigned_to: None,
        result: None,
        error: None,
        retry_count: 0,
        created_at: Utc::now().timestamp(),
        updated_at: Utc::now().timestamp(),
        completed_at: None,
    };
    storage.store_task(&task).await.unwrap();

    // 3. Claim task
    let claimed = storage.claim_task(&task_id, &agent_id).await.unwrap();
    assert!(claimed);

    // 4. Log events
    for i in 0..5 {
        let event = EventModel {
            id: Uuid::new_v4().to_string(),
            event_type: "workflow_step".to_string(),
            agent_id: Some(agent_id.clone()),
            task_id: Some(task_id.clone()),
            data: serde_json::json!({"step": i + 1, "status": "completed"}),
            timestamp: Utc::now().timestamp() + i,
        };
        storage.store_event(&event).await.unwrap();
    }

    // 5. Complete task
    let mut task = storage.get_task(&task_id).await.unwrap().unwrap();
    task.status = "completed".to_string();
    task.result = Some(serde_json::json!({"steps_completed": 5}));
    task.completed_at = Some(Utc::now().timestamp());
    storage.update_task(&task).await.unwrap();

    // 6. Store final metric
    let metric = MetricModel {
        id: Uuid::new_v4().to_string(),
        agent_id: agent_id.clone(),
        metric_type: "workflow_duration".to_string(),
        value: 150.0, // seconds
        tags: serde_json::json!({"task_id": task_id}),
        timestamp: Utc::now().timestamp(),
    };
    storage.store_metric(&metric).await.unwrap();

    // Verify workflow completion
    let final_task = storage.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(final_task.status, "completed");
    assert!(final_task.result.is_some());

    let events = storage.get_events_by_agent(&agent_id, 10).await.unwrap();
    assert_eq!(events.len(), 5);
}
