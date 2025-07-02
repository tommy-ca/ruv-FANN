//! Seamless onboarding module for ruv-swarm
//! 
//! This module provides a guided onboarding experience that automatically sets up
//! Claude Code with preconfigured MCP servers, eliminating manual configuration steps.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during onboarding
#[derive(Error, Debug)]
pub enum OnboardingError {
    #[error("Claude Code not found in system")]
    ClaudeCodeNotFound,
    
    #[error("Claude Code version {0} is incompatible (requires {1})")]
    IncompatibleVersion(String, String),
    
    #[error("Installation failed: {0}")]
    InstallationFailed(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Launch failed: {0}")]
    LaunchError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, OnboardingError>;

/// Platform-specific installation paths
#[derive(Debug, Clone)]
pub struct InstallationPaths {
    pub system_paths: Vec<PathBuf>,
    pub user_paths: Vec<PathBuf>,
    pub search_paths: Vec<PathBuf>,
}

/// Claude Code detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub found: bool,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    pub is_compatible: bool,
}

/// Installation options
#[derive(Debug, Clone)]
pub struct InstallOptions {
    pub target_dir: PathBuf,
    pub system_install: bool,
    pub force_reinstall: bool,
    pub version: Option<String>,
}

/// MCP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    pub command: String,
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
}

/// Complete MCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, MCPServerConfig>,
}

impl MCPConfig {
    pub fn new() -> Self {
        Self {
            mcp_servers: HashMap::new(),
        }
    }
    
    pub fn add_github_mcp(&mut self, token: Option<String>) {
        let mut env = HashMap::new();
        if let Some(token) = token {
            env.insert("GITHUB_TOKEN".to_string(), token);
        } else {
            env.insert("GITHUB_TOKEN".to_string(), "${GITHUB_TOKEN}".to_string());
        }
        
        self.mcp_servers.insert(
            "github".to_string(),
            MCPServerConfig {
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-github".to_string()],
                env,
            },
        );
    }
    
    pub fn add_ruv_swarm_mcp(&mut self, swarm_id: String, topology: String) {
        let mut env = HashMap::new();
        env.insert("SWARM_ID".to_string(), swarm_id);
        env.insert("SWARM_TOPOLOGY".to_string(), topology);
        
        self.mcp_servers.insert(
            "ruv-swarm".to_string(),
            MCPServerConfig {
                command: "npx".to_string(),
                args: vec!["ruv-swarm".to_string(), "mcp".to_string(), "start".to_string()],
                env,
            },
        );
    }
    
    pub fn has_servers(&self) -> bool {
        !self.mcp_servers.is_empty()
    }
}

/// User interaction choices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserChoice {
    Yes,
    No,
    Skip,
}

/// Launch options for Claude Code
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub mcp_config_path: PathBuf,
    pub skip_permissions: bool,
    pub wait_for_auth: bool,
}

/// Trait for detecting Claude Code installation
#[async_trait]
pub trait ClaudeDetector: Send + Sync {
    /// Check if Claude Code is installed and return detection result
    async fn detect(&self) -> Result<DetectionResult>;
    
    /// Get platform-specific installation paths
    fn get_installation_paths(&self) -> InstallationPaths;
    
    /// Validate Claude Code version compatibility
    async fn validate_version(&self, version: &str) -> Result<bool>;
}

/// Trait for installing Claude Code
#[async_trait]
pub trait Installer: Send + Sync {
    /// Download and install Claude Code
    async fn install(&self, options: InstallOptions) -> Result<PathBuf>;
    
    /// Check if installation requires elevated permissions
    async fn requires_elevation(&self, target_dir: &Path) -> Result<bool>;
    
    /// Verify installation was successful
    async fn verify_installation(&self, install_path: &Path) -> Result<bool>;
    
    /// Rollback failed installation
    async fn rollback(&self, install_path: &Path) -> Result<()>;
}

/// Trait for configuring MCP servers
#[async_trait]
pub trait MCPConfigurator: Send + Sync {
    /// Create MCP configuration file
    async fn create_config(&self, config: &MCPConfig, path: &Path) -> Result<()>;
    
    /// Load existing MCP configuration
    async fn load_config(&self, path: &Path) -> Result<MCPConfig>;
    
    /// Validate MCP configuration
    async fn validate_config(&self, config: &MCPConfig) -> Result<bool>;
    
    /// Check for GitHub token availability
    async fn check_github_token(&self) -> Result<Option<String>>;
    
    /// Generate swarm ID
    fn generate_swarm_id(&self) -> String;
}

/// Trait for user interaction
#[async_trait]
pub trait InteractivePrompt: Send + Sync {
    /// Show Y/N confirmation prompt
    async fn confirm(&self, message: &str, default: bool) -> Result<bool>;
    
    /// Show multiple choice prompt
    async fn choice(&self, message: &str, options: &[&str]) -> Result<usize>;
    
    /// Show text input prompt
    async fn input(&self, message: &str, default: Option<&str>) -> Result<String>;
    
    /// Show password input prompt (hidden)
    async fn password(&self, message: &str) -> Result<String>;
    
    /// Display info message
    async fn info(&self, message: &str);
    
    /// Display warning message
    async fn warning(&self, message: &str);
    
    /// Display error message
    async fn error(&self, message: &str);
    
    /// Display success message
    async fn success(&self, message: &str);
    
    /// Show progress bar
    async fn progress(&self, message: &str, current: u64, total: u64);
}

/// Trait for launching Claude Code
#[async_trait]
pub trait LaunchManager: Send + Sync {
    /// Launch Claude Code with MCP configuration
    async fn launch(&self, options: LaunchOptions) -> Result<()>;
    
    /// Check if Claude Code is already running
    async fn is_running(&self) -> Result<bool>;
    
    /// Wait for Claude Code to be ready
    async fn wait_for_ready(&self, timeout_secs: u64) -> Result<()>;
    
    /// Guide user through authentication
    async fn guide_auth(&self) -> Result<()>;
}

/// Checkpoint for rollback support
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub id: String,
    pub timestamp: std::time::SystemTime,
    pub description: String,
    pub data: HashMap<String, String>,
}

/// Trait for rollback support
#[async_trait]
pub trait Rollback: Send + Sync {
    /// Create a checkpoint
    async fn checkpoint(&mut self, description: &str) -> Result<String>;
    
    /// Rollback to a checkpoint
    async fn rollback(&mut self, checkpoint_id: &str) -> Result<()>;
    
    /// Commit changes (clear checkpoints)
    async fn commit(&mut self) -> Result<()>;
    
    /// List available checkpoints
    async fn list_checkpoints(&self) -> Result<Vec<Checkpoint>>;
}

/// Platform detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    MacOS,
    Linux,
    Unknown,
}

impl Platform {
    pub fn detect() -> Self {
        if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOS
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else {
            Platform::Unknown
        }
    }
}

/// Configuration for onboarding process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardingConfig {
    pub auto_accept: bool,
    pub claude_code_version: String,
    pub default_topology: String,
    pub default_max_agents: usize,
    pub retry_attempts: u32,
    pub retry_delay_ms: u64,
}

impl Default for OnboardingConfig {
    fn default() -> Self {
        Self {
            auto_accept: false,
            claude_code_version: ">=1.0.0".to_string(),
            default_topology: "mesh".to_string(),
            default_max_agents: 5,
            retry_attempts: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Main orchestrator for the onboarding process
pub struct OnboardingOrchestrator<D, I, M, P, L>
where
    D: ClaudeDetector,
    I: Installer,
    M: MCPConfigurator,
    P: InteractivePrompt,
    L: LaunchManager,
{
    pub detector: D,
    pub installer: I,
    pub configurator: M,
    pub prompt: P,
    pub launcher: L,
    pub config: OnboardingConfig,
}

impl<D, I, M, P, L> OnboardingOrchestrator<D, I, M, P, L>
where
    D: ClaudeDetector,
    I: Installer,
    M: MCPConfigurator,
    P: InteractivePrompt,
    L: LaunchManager,
{
    pub fn new(
        detector: D,
        installer: I,
        configurator: M,
        prompt: P,
        launcher: L,
        config: OnboardingConfig,
    ) -> Self {
        Self {
            detector,
            installer,
            configurator,
            prompt,
            launcher,
            config,
        }
    }
    
    /// Run the complete onboarding flow
    pub async fn run(&self) -> Result<()> {
        self.prompt.info("🚀 Welcome to ruv-swarm!").await;
        
        // Step 1: Detect Claude Code
        let detection = self.detector.detect().await?;
        
        if !detection.found {
            self.prompt.error("Claude Code not found").await;
            
            if self.config.auto_accept || self.prompt.confirm("Would you like to install Claude Code?", true).await? {
                self.install_claude_code().await?;
            } else {
                return Err(OnboardingError::ClaudeCodeNotFound);
            }
        } else if !detection.is_compatible {
            self.prompt.warning(&format!(
                "Claude Code version {} is incompatible (requires {})",
                detection.version.as_deref().unwrap_or("unknown"),
                self.config.claude_code_version
            )).await;
        }
        
        // Step 2: Configure MCP servers
        self.configure_mcp_servers().await?;
        
        // Step 3: Offer to launch
        if self.config.auto_accept || self.prompt.confirm("Ready to launch Claude Code?", true).await? {
            self.launch_claude_code().await?;
        }
        
        self.prompt.success("✨ Initialization complete!").await;
        Ok(())
    }
    
    async fn install_claude_code(&self) -> Result<()> {
        let paths = self.detector.get_installation_paths();
        let default_path = paths.user_paths.first()
            .or(paths.system_paths.first())
            .ok_or_else(|| OnboardingError::InstallationFailed("No installation paths available".to_string()))?;
        
        let options = InstallOptions {
            target_dir: default_path.clone(),
            system_install: false,
            force_reinstall: false,
            version: None,
        };
        
        self.prompt.info("📦 Downloading Claude Code...").await;
        let install_path = self.installer.install(options).await?;
        
        if self.installer.verify_installation(&install_path).await? {
            self.prompt.success("✅ Claude Code installed successfully!").await;
            Ok(())
        } else {
            Err(OnboardingError::InstallationFailed("Verification failed".to_string()))
        }
    }
    
    async fn configure_mcp_servers(&self) -> Result<()> {
        self.prompt.info("Setting up MCP servers...").await;
        
        let mut config = MCPConfig::new();
        
        // GitHub MCP
        if self.config.auto_accept || self.prompt.confirm("Would you like to install the GitHub MCP server?", true).await? {
            self.prompt.info("📝 Configuring GitHub MCP server...").await;
            
            if let Ok(Some(token)) = self.configurator.check_github_token().await {
                config.add_github_mcp(Some(token));
                self.prompt.success("✅ GitHub MCP server configured").await;
            } else {
                match self.prompt.choice(
                    "No GitHub token found",
                    &["Enter token now", "Continue without auth", "Skip GitHub MCP"]
                ).await? {
                    0 => {
                        let token = self.prompt.password("GitHub token: ").await?;
                        config.add_github_mcp(Some(token));
                    }
                    1 => {
                        config.add_github_mcp(None);
                        self.prompt.warning("⚠️ GitHub MCP configured with limited access").await;
                    }
                    _ => {}
                }
            }
        }
        
        // ruv-swarm MCP
        if self.config.auto_accept || self.prompt.confirm("Would you like to install the ruv-swarm MCP server?", true).await? {
            self.prompt.info("📝 Configuring ruv-swarm MCP server...").await;
            let swarm_id = self.configurator.generate_swarm_id();
            config.add_ruv_swarm_mcp(swarm_id, self.config.default_topology.clone());
            self.prompt.success("✅ ruv-swarm MCP server configured").await;
        }
        
        if config.has_servers() {
            let config_path = PathBuf::from(".claude/mcp.json");
            self.configurator.create_config(&config, &config_path).await?;
        }
        
        Ok(())
    }
    
    async fn launch_claude_code(&self) -> Result<()> {
        self.prompt.info("🚀 Launching Claude Code with MCP servers...").await;
        
        let options = LaunchOptions {
            mcp_config_path: PathBuf::from(".claude/mcp.json"),
            skip_permissions: true,
            wait_for_auth: true,
        };
        
        self.launcher.launch(options).await?;
        
        if self.launcher.wait_for_ready(30).await.is_ok() {
            self.prompt.info("📋 Please log in to your Anthropic account when prompted").await;
        }
        
        Ok(())
    }
}

// Re-export commonly used types
pub use self::{
    ClaudeDetector, Installer, MCPConfigurator, InteractivePrompt, LaunchManager,
    OnboardingError, Result, Platform, MCPConfig, OnboardingConfig,
    OnboardingOrchestrator, DetectionResult, InstallOptions, LaunchOptions,
};

// Platform-specific implementations will be in separate modules
#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

// Common implementations
pub mod common;