use anyhow::{Context, Result};
use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

use crate::config::Config;
use crate::output::{OutputHandler, StatusLevel};
use crate::onboarding::run_onboarding_flow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmInit {
    pub swarm_id: String,
    pub topology: String,
    pub persistence: PersistenceConfig,
    pub initial_agents: Vec<AgentSpec>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub backend: String,
    pub connection: String,
    pub options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
}

/// Execute the init command
pub async fn execute(
    config: &Config,
    output: &OutputHandler,
    topology: String,
    persistence: Option<String>,
    config_file: Option<String>,
    non_interactive: bool,
    skip_onboarding: bool,
) -> Result<()> {
    // Handle onboarding flow unless skipped
    if !skip_onboarding && !non_interactive {
        if let Err(e) = run_onboarding_flow(config, output).await {
            output.warning(&format!("Onboarding flow failed: {}", e));
            output.info("Continuing with standard initialization...");
        }
    } else if !skip_onboarding {
        output.info("Skipping onboarding due to non-interactive mode");
    }

    output.section("Initializing RUV Swarm");

    // Validate topology
    let valid_topologies = vec!["mesh", "hierarchical", "ring", "star", "custom"];
    if !valid_topologies.contains(&topology.as_str()) {
        output.error(&format!(
            "Invalid topology '{}'. Valid options: {}",
            topology,
            valid_topologies.join(", ")
        ));
        return Err(anyhow::anyhow!("Invalid topology"));
    }

    // Determine persistence configuration
    let persistence_config = if let Some(backend) = persistence {
        configure_persistence(&backend, non_interactive)?
    } else if !non_interactive {
        // Ask user for persistence configuration
        let backends = vec!["memory", "sqlite", "postgres", "redis"];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select persistence backend")
            .items(&backends)
            .default(0)
            .interact()?;

        configure_persistence(backends[selection], non_interactive)?
    } else {
        // Default to memory backend
        PersistenceConfig {
            backend: "memory".to_string(),
            connection: ":memory:".to_string(),
            options: HashMap::new(),
        }
    };

    // Load or create initial configuration
    let mut swarm_init = if let Some(path) = config_file {
        output.info(&format!("Loading configuration from {}", path));
        load_config_file(path)?
    } else {
        // Create new configuration
        let swarm_id = Uuid::new_v4().to_string();

        let initial_agents = if !non_interactive {
            configure_initial_agents(output)?
        } else {
            // Default initial agents
            vec![AgentSpec {
                name: "orchestrator-1".to_string(),
                agent_type: "orchestrator".to_string(),
                capabilities: vec!["coordination".to_string(), "task_distribution".to_string()],
            }]
        };

        SwarmInit {
            swarm_id,
            topology: topology.clone(),
            persistence: persistence_config,
            initial_agents,
            created_at: chrono::Utc::now(),
        }
    };

    // Show configuration summary
    output.section("Configuration Summary");
    output.key_value(&[
        ("Swarm ID".to_string(), swarm_init.swarm_id.clone()),
        ("Topology".to_string(), swarm_init.topology.clone()),
        (
            "Persistence".to_string(),
            swarm_init.persistence.backend.clone(),
        ),
        (
            "Initial Agents".to_string(),
            swarm_init.initial_agents.len().to_string(),
        ),
    ]);

    // Confirm configuration
    if !non_interactive {
        if !output.confirm("Initialize swarm with this configuration?", true)? {
            output.warning("Swarm initialization cancelled");
            return Ok(());
        }
    }

    // Initialize the swarm
    let spinner = output.spinner("Initializing swarm components...");

    // Simulate initialization steps
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Create persistence backend
    initialize_persistence(&swarm_init.persistence, output).await?;

    // Set up swarm topology
    initialize_topology(&swarm_init.topology, output).await?;

    // Spawn initial agents
    for agent in &swarm_init.initial_agents {
        spawn_initial_agent(agent, output).await?;
    }

    if let Some(pb) = spinner {
        pb.finish_with_message("Swarm initialization complete");
    }

    // Save swarm configuration
    save_swarm_config(&swarm_init, config, output)?;

    output.success(&format!(
        "Swarm '{}' initialized successfully!",
        swarm_init.swarm_id
    ));

    // Show next steps
    output.section("Next Steps");
    output.list(
        &[
            format!("Spawn additional agents: ruv-swarm spawn <type>"),
            format!("Start orchestration: ruv-swarm orchestrate <strategy> <task>"),
            format!("Monitor swarm: ruv-swarm monitor"),
            format!("Check status: ruv-swarm status"),
        ],
        true,
    );

    Ok(())
}

fn configure_persistence(backend: &str, non_interactive: bool) -> Result<PersistenceConfig> {
    let mut options = HashMap::new();

    let connection = match backend {
        "memory" => ":memory:".to_string(),
        "sqlite" => {
            if non_interactive {
                "./swarm.db".to_string()
            } else {
                Input::<String>::with_theme(&ColorfulTheme::default())
                    .with_prompt("SQLite database path")
                    .default("./swarm.db".to_string())
                    .interact()?
            }
        }
        "postgres" => {
            if non_interactive {
                "postgresql://localhost/ruv_swarm".to_string()
            } else {
                Input::<String>::with_theme(&ColorfulTheme::default())
                    .with_prompt("PostgreSQL connection string")
                    .default("postgresql://localhost/ruv_swarm".to_string())
                    .interact()?
            }
        }
        "redis" => {
            if non_interactive {
                "redis://localhost:6379".to_string()
            } else {
                Input::<String>::with_theme(&ColorfulTheme::default())
                    .with_prompt("Redis connection string")
                    .default("redis://localhost:6379".to_string())
                    .interact()?
            }
        }
        _ => return Err(anyhow::anyhow!("Unsupported persistence backend")),
    };

    // Add backend-specific options
    if backend == "sqlite" && !non_interactive {
        if Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Enable WAL mode for better concurrency?")
            .default(true)
            .interact()?
        {
            options.insert("wal_mode".to_string(), "true".to_string());
        }
    }

    Ok(PersistenceConfig {
        backend: backend.to_string(),
        connection,
        options,
    })
}

fn configure_initial_agents(output: &OutputHandler) -> Result<Vec<AgentSpec>> {
    let mut agents = Vec::new();

    // Always add an orchestrator
    agents.push(AgentSpec {
        name: "orchestrator-1".to_string(),
        agent_type: "orchestrator".to_string(),
        capabilities: vec!["coordination".to_string(), "task_distribution".to_string()],
    });

    if Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Add additional initial agents?")
        .default(true)
        .interact()?
    {
        let agent_types = vec!["researcher", "coder", "analyst", "reviewer", "tester"];

        loop {
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select agent type")
                .items(&agent_types)
                .interact()?;

            let agent_type = agent_types[selection];
            let name = Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Agent name")
                .default(format!("{}-{}", agent_type, agents.len() + 1))
                .interact()?;

            let capabilities_str = Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Capabilities (comma-separated)")
                .default(default_capabilities(agent_type))
                .interact()?;

            let capabilities: Vec<String> = capabilities_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            agents.push(AgentSpec {
                name,
                agent_type: agent_type.to_string(),
                capabilities,
            });

            if !Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("Add another agent?")
                .default(false)
                .interact()?
            {
                break;
            }
        }
    }

    Ok(agents)
}

fn default_capabilities(agent_type: &str) -> String {
    match agent_type {
        "researcher" => "web_search,document_analysis,summarization",
        "coder" => "code_generation,refactoring,testing",
        "analyst" => "data_analysis,visualization,reporting",
        "reviewer" => "code_review,quality_assurance,best_practices",
        "tester" => "unit_testing,integration_testing,performance_testing",
        _ => "general",
    }
    .to_string()
}

fn load_config_file<P: AsRef<Path>>(path: P) -> Result<SwarmInit> {
    let content = std::fs::read_to_string(path)?;

    // Try to parse as YAML first, then JSON
    serde_yaml::from_str(&content)
        .or_else(|_| serde_json::from_str(&content))
        .context("Failed to parse configuration file")
}

async fn initialize_persistence(config: &PersistenceConfig, output: &OutputHandler) -> Result<()> {
    output.info(&format!(
        "Setting up {} persistence backend...",
        config.backend
    ));

    // Simulate persistence initialization
    match config.backend.as_str() {
        "memory" => {
            // Nothing to do for memory backend
        }
        "sqlite" => {
            // Create database file if needed
            if config.connection != ":memory:" {
                let path = Path::new(&config.connection);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
            }
        }
        "postgres" | "redis" => {
            // In a real implementation, we would test the connection here
        }
        _ => {}
    }

    Ok(())
}

async fn initialize_topology(topology: &str, output: &OutputHandler) -> Result<()> {
    output.info(&format!("Configuring {} topology...", topology));

    // Simulate topology setup
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    Ok(())
}

async fn spawn_initial_agent(agent: &AgentSpec, output: &OutputHandler) -> Result<()> {
    output.info(&format!(
        "Spawning {} agent '{}'...",
        agent.agent_type, agent.name
    ));

    // Simulate agent spawning
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Ok(())
}

fn save_swarm_config(
    swarm_init: &SwarmInit,
    config: &Config,
    output: &OutputHandler,
) -> Result<()> {
    let config_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    std::fs::create_dir_all(&config_dir)?;

    let swarm_file = config_dir.join(format!("swarm-{}.json", swarm_init.swarm_id));
    let content = serde_json::to_string_pretty(swarm_init)?;
    std::fs::write(&swarm_file, content)?;

    // Also save as "current" swarm
    let current_file = config_dir.join("current-swarm.json");
    std::fs::write(&current_file, serde_json::to_string_pretty(swarm_init)?)?;

    output.info(&format!("Configuration saved to {:?}", swarm_file));

    Ok(())
}
