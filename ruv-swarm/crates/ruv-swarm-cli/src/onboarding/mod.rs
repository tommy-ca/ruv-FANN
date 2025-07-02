//! Comprehensive onboarding system for ruv-swarm
//! 
//! Provides seamless onboarding experience across platforms as specified
//! in GitHub PR #32 requirements.

use anyhow::{Context, Result};
use crate::config::Config;
use crate::output::OutputHandler;

pub mod claude_detector;
pub mod mcp_configurator;
pub mod interactive_prompt;
pub mod launch_manager;

pub use claude_detector::{ClaudeDetector, DefaultClaudeDetector, ClaudeInfo};
pub use mcp_configurator::{MCPConfigurator, DefaultMCPConfigurator, MCPConfig, MCPServerConfig};
pub use interactive_prompt::{InteractivePrompt, DefaultInteractivePrompt};
pub use launch_manager::{LaunchManager, DefaultLaunchManager};

/// Main onboarding flow orchestrator
pub async fn run_onboarding_flow(config: &Config, output: &OutputHandler) -> Result<()> {
    output.section("Welcome to RUV Swarm Onboarding");
    
    // Step 1: Detect Claude Code installation
    let detector = DefaultClaudeDetector::new();
    let claude_info = detector.detect().await
        .context("Failed to detect Claude Code installation")?;
    
    if claude_info.installed {
        output.success(&format!("✅ Claude Code detected: {}", claude_info.path));
    } else {
        output.warning("❌ Claude Code not found");
        output.info("Please install Claude Code from: https://claude.ai/download");
    }
    
    // Step 2: Configure MCP servers
    let configurator = DefaultMCPConfigurator::new();
    let mcp_config = configurator.configure_for_ruv_swarm()
        .await
        .context("Failed to configure MCP servers")?;
    
    output.info("✅ MCP configuration prepared");
    
    // Step 3: Interactive setup (if Claude Code is available)
    if claude_info.installed {
        let prompt = DefaultInteractivePrompt::new();
        if prompt.confirm("Would you like to launch Claude Code with ruv-swarm integration?", true)? {
            let launcher = DefaultLaunchManager::new();
            launcher.launch_with_config(&claude_info.path, &mcp_config)
                .await
                .context("Failed to launch Claude Code")?;
            
            output.success("✅ Claude Code launched with ruv-swarm integration");
        }
    }
    
    output.success("🎉 Onboarding completed successfully!");
    Ok(())
}

/// Onboarding configuration
#[derive(Debug, Clone)]
pub struct OnboardingConfig {
    pub skip_claude_detection: bool,
    pub skip_mcp_configuration: bool,
    pub skip_launch: bool,
    pub auto_launch: bool,
}

impl Default for OnboardingConfig {
    fn default() -> Self {
        Self {
            skip_claude_detection: false,
            skip_mcp_configuration: false,
            skip_launch: false,
            auto_launch: false,
        }
    }
}