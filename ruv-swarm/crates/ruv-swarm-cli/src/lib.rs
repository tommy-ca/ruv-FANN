// Library module for ruv-swarm-cli
// Re-exports all public functionality for use by external crates

pub mod config;
pub mod output;
pub mod commands;
pub mod onboarding;

// Re-export commonly used types
pub use config::Config;
pub use output::OutputHandler;
pub use onboarding::{
    ClaudeDetector, DefaultClaudeDetector, ClaudeInfo,
    MCPConfigurator, DefaultMCPConfigurator, MCPConfig, MCPServerConfig,
    InteractivePrompt, DefaultInteractivePrompt,
    LaunchManager, DefaultLaunchManager,
    run_onboarding_flow, OnboardingConfig,
};