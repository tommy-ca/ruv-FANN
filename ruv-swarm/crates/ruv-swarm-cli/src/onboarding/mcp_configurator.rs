//! MCP server configuration for seamless ruv-swarm integration

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// MCP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    pub servers: HashMap<String, MCPServerConfig>,
    pub auto_start: bool,
    pub stdio_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub enabled: bool,
}

/// Trait for MCP configuration
#[async_trait]
pub trait MCPConfigurator {
    async fn configure_for_ruv_swarm(&self) -> Result<MCPConfig>;
    async fn validate_configuration(&self, config: &MCPConfig) -> Result<bool>;
    async fn apply_configuration(&self, config: &MCPConfig) -> Result<()>;
}

/// Default implementation of MCP configurator
pub struct DefaultMCPConfigurator {
    claude_config_dir: Option<PathBuf>,
}

impl DefaultMCPConfigurator {
    pub fn new() -> Self {
        Self {
            claude_config_dir: Self::find_claude_config_dir(),
        }
    }

    /// Find Claude Code configuration directory
    fn find_claude_config_dir() -> Option<PathBuf> {
        // Try common Claude configuration locations
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;

        let possible_paths = vec![
            format!("{}/.claude", home),
            format!("{}/.config/claude", home),
            format!("{}/AppData/Roaming/Claude", home), // Windows
            format!("{}/Library/Application Support/Claude", home), // macOS
        ];

        for path in possible_paths {
            let config_path = PathBuf::from(path);
            if config_path.exists() {
                return Some(config_path);
            }
        }

        None
    }

    /// Generate default ruv-swarm MCP configuration
    fn generate_ruv_swarm_config(&self) -> MCPConfig {
        let mut servers = HashMap::new();

        // ruv-swarm MCP server configuration
        servers.insert(
            "ruv-swarm".to_string(),
            MCPServerConfig {
                command: "npx".to_string(),
                args: vec![
                    "ruv-swarm".to_string(),
                    "mcp".to_string(),
                    "start".to_string(),
                ],
                env: HashMap::new(),
                enabled: true,
            },
        );

        MCPConfig {
            servers,
            auto_start: true,
            stdio_enabled: true,
        }
    }

    /// Check if ruv-swarm is available via npx
    async fn check_ruv_swarm_availability(&self) -> Result<bool> {
        let output = Command::new("npx")
            .arg("ruv-swarm")
            .arg("--version")
            .output();

        match output {
            Ok(result) => Ok(result.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Write MCP configuration to Claude config
    async fn write_claude_mcp_config(&self, config: &MCPConfig) -> Result<()> {
        if let Some(config_dir) = &self.claude_config_dir {
            let mcp_config_path = config_dir.join("mcp.json");
            let config_json = serde_json::to_string_pretty(config)
                .context("Failed to serialize MCP configuration")?;

            std::fs::write(&mcp_config_path, config_json)
                .context("Failed to write MCP configuration file")?;

            Ok(())
        } else {
            Err(anyhow!("Claude configuration directory not found"))
        }
    }
}

#[async_trait]
impl MCPConfigurator for DefaultMCPConfigurator {
    async fn configure_for_ruv_swarm(&self) -> Result<MCPConfig> {
        // Check if ruv-swarm is available
        if !self.check_ruv_swarm_availability().await? {
            return Err(anyhow!(
                "ruv-swarm not available via npx. Please install with: npm install -g ruv-swarm"
            ));
        }

        // Generate configuration
        let config = self.generate_ruv_swarm_config();

        // Validate configuration
        self.validate_configuration(&config).await?;

        Ok(config)
    }

    async fn validate_configuration(&self, config: &MCPConfig) -> Result<bool> {
        // Validate that all server commands are available
        for (name, server_config) in &config.servers {
            let output = Command::new(&server_config.command)
                .args(&server_config.args)
                .arg("--help")
                .output();

            if output.is_err() || !output.unwrap().status.success() {
                return Err(anyhow!(
                    "MCP server '{}' command not available: {}",
                    name,
                    server_config.command
                ));
            }
        }

        Ok(true)
    }

    async fn apply_configuration(&self, config: &MCPConfig) -> Result<()> {
        // Write configuration to Claude's MCP config
        self.write_claude_mcp_config(config).await?;

        Ok(())
    }
}

impl Default for DefaultMCPConfigurator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_configurator_creation() {
        let configurator = DefaultMCPConfigurator::new();
        // Should not panic
        assert!(true);
    }

    #[tokio::test]
    async fn test_ruv_swarm_config_generation() {
        let configurator = DefaultMCPConfigurator::new();
        let config = configurator.generate_ruv_swarm_config();

        assert!(config.servers.contains_key("ruv-swarm"));
        assert!(config.auto_start);
        assert!(config.stdio_enabled);
    }

    #[test]
    fn test_config_serialization() {
        let configurator = DefaultMCPConfigurator::new();
        let config = configurator.generate_ruv_swarm_config();

        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<MCPConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }
}
