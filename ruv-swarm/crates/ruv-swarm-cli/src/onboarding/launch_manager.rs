//! Claude Code launch manager for seamless integration

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::timeout;

use super::mcp_configurator::MCPConfig;

/// Trait for managing Claude Code launches
#[async_trait]
pub trait LaunchManager {
    async fn launch_with_config(&self, claude_path: &str, mcp_config: &MCPConfig) -> Result<()>;
    async fn launch_simple(&self, claude_path: &str) -> Result<()>;
    async fn validate_launch(&self, claude_path: &str) -> Result<bool>;
}

/// Default implementation of launch manager
pub struct DefaultLaunchManager {
    launch_timeout: Duration,
}

impl DefaultLaunchManager {
    pub fn new() -> Self {
        Self {
            launch_timeout: Duration::from_secs(30),
        }
    }

    /// Create with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            launch_timeout: timeout,
        }
    }

    /// Prepare Claude Code environment for ruv-swarm
    async fn prepare_environment(&self, mcp_config: &MCPConfig) -> Result<()> {
        // Ensure all MCP servers are available
        for (name, server_config) in &mcp_config.servers {
            if !server_config.enabled {
                continue;
            }

            // Test that the server command is available
            let test_result = Command::new(&server_config.command)
                .args(&server_config.args)
                .arg("--help")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            match test_result {
                Ok(status) if status.success() => {
                    log::debug!("MCP server '{}' is available", name);
                }
                _ => {
                    return Err(anyhow!(
                        "MCP server '{}' is not available. Command: {} {}",
                        name,
                        server_config.command,
                        server_config.args.join(" ")
                    ));
                }
            }
        }

        Ok(())
    }

    /// Launch Claude Code with specific arguments
    async fn launch_claude(&self, claude_path: &str, args: Vec<String>) -> Result<()> {
        log::info!("Launching Claude Code: {} {}", claude_path, args.join(" "));

        let launch_future = async {
            let mut command = Command::new(claude_path);
            command.args(&args);

            // On Unix systems, detach the process
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                command.process_group(0);
            }

            // Launch Claude Code
            let mut child = command
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .context("Failed to launch Claude Code")?;

            // Wait briefly to ensure it started successfully
            tokio::time::sleep(Duration::from_millis(1000)).await;

            // Check if the process is still running
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        return Err(anyhow!("Claude Code exited with status: {}", status));
                    }
                }
                Ok(None) => {
                    // Process is still running, which is good
                    log::info!("Claude Code launched successfully");
                }
                Err(e) => {
                    return Err(anyhow!("Failed to check Claude Code process status: {}", e));
                }
            }

            Ok(())
        };

        timeout(self.launch_timeout, launch_future)
            .await
            .context("Claude Code launch timed out")?
    }

    /// Get appropriate launch arguments for the platform
    fn get_launch_args(&self, with_mcp: bool) -> Vec<String> {
        let mut args = Vec::new();

        if with_mcp {
            // Add MCP-related arguments if available
            args.push("--mcp-enable".to_string());
        }

        // Add any other common arguments
        args.push("--new-session".to_string());

        args
    }
}

#[async_trait]
impl LaunchManager for DefaultLaunchManager {
    async fn launch_with_config(&self, claude_path: &str, mcp_config: &MCPConfig) -> Result<()> {
        // First, prepare the environment
        self.prepare_environment(mcp_config).await?;

        // Validate that Claude Code can be launched
        self.validate_launch(claude_path).await?;

        // Get launch arguments
        let args = self.get_launch_args(true);

        // Launch Claude Code
        self.launch_claude(claude_path, args).await?;

        Ok(())
    }

    async fn launch_simple(&self, claude_path: &str) -> Result<()> {
        // Validate that Claude Code can be launched
        self.validate_launch(claude_path).await?;

        // Get basic launch arguments
        let args = self.get_launch_args(false);

        // Launch Claude Code
        self.launch_claude(claude_path, args).await?;

        Ok(())
    }

    async fn validate_launch(&self, claude_path: &str) -> Result<bool> {
        // Test that Claude Code responds to --version
        let output = Command::new(claude_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute Claude Code")?;

        if !output.status.success() {
            return Err(anyhow!(
                "Claude Code validation failed with status: {}",
                output.status
            ));
        }

        // Check that we got a reasonable version response
        let version_output = String::from_utf8_lossy(&output.stdout);
        if version_output.trim().is_empty() {
            return Err(anyhow!("Claude Code version command returned empty output"));
        }

        log::debug!("Claude Code version: {}", version_output.trim());
        Ok(true)
    }
}

impl Default for DefaultLaunchManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_launch_manager_creation() {
        let manager = DefaultLaunchManager::new();
        assert_eq!(manager.launch_timeout, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn test_launch_manager_with_timeout() {
        let timeout = Duration::from_secs(60);
        let manager = DefaultLaunchManager::with_timeout(timeout);
        assert_eq!(manager.launch_timeout, timeout);
    }

    #[test]
    fn test_launch_args_generation() {
        let manager = DefaultLaunchManager::new();

        let args_with_mcp = manager.get_launch_args(true);
        assert!(args_with_mcp.contains(&"--mcp-enable".to_string()));
        assert!(args_with_mcp.contains(&"--new-session".to_string()));

        let args_without_mcp = manager.get_launch_args(false);
        assert!(!args_without_mcp.contains(&"--mcp-enable".to_string()));
        assert!(args_without_mcp.contains(&"--new-session".to_string()));
    }

    #[tokio::test]
    async fn test_invalid_path_validation() {
        let manager = DefaultLaunchManager::new();
        let result = manager.validate_launch("/nonexistent/claude").await;
        assert!(result.is_err());
    }
}
