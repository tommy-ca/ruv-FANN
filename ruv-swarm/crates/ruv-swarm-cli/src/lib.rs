// Library module for ruv-swarm-cli
// Re-exports all public functionality for use by external crates

pub mod commands;
pub mod config;
pub mod onboarding;
pub mod output;

// Re-export commonly used types
pub use config::Config;
pub use onboarding::{
    run_onboarding_flow, ClaudeDetector, ClaudeInfo, DefaultClaudeDetector,
    DefaultInteractivePrompt, DefaultLaunchManager, DefaultMCPConfigurator, InteractivePrompt,
    LaunchManager, MCPConfig, MCPConfigurator, MCPServerConfig, OnboardingConfig,
};
pub use output::OutputHandler;
