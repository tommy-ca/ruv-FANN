//! Common implementations for onboarding traits

use super::*;
use anyhow::Context;
use async_trait::async_trait;
use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select, Password};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

/// Default Claude detector implementation
pub struct DefaultClaudeDetector {
    search_paths: Vec<PathBuf>,
}

impl DefaultClaudeDetector {
    pub fn new() -> Self {
        Self {
            search_paths: Self::default_search_paths(),
        }
    }

    fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        // Add PATH directories
        if let Ok(path_var) = std::env::var("PATH") {
            let separator = if cfg!(windows) { ';' } else { ':' };
            for path in path_var.split(separator) {
                paths.push(PathBuf::from(path));
            }
        }
        
        // Add common installation directories
        if let Ok(home) = std::env::var("HOME") {
            paths.push(PathBuf::from(&home).join(".local").join("bin"));
            paths.push(PathBuf::from(&home).join("bin"));
        }
        
        #[cfg(target_os = "macos")]
        {
            paths.push(PathBuf::from("/usr/local/bin"));
            paths.push(PathBuf::from("/opt/homebrew/bin"));
            paths.push(PathBuf::from("/Applications/Claude.app/Contents/MacOS"));
            paths.push(PathBuf::from("/Applications/Claude Code.app/Contents/MacOS"));
        }
        
        #[cfg(target_os = "linux")]
        {
            paths.push(PathBuf::from("/usr/local/bin"));
            paths.push(PathBuf::from("/usr/bin"));
            paths.push(PathBuf::from("/opt/claude"));
            paths.push(PathBuf::from("/snap/bin"));
        }
        
        #[cfg(target_os = "windows")]
        {
            if let Ok(prog) = std::env::var("ProgramFiles") {
                paths.push(PathBuf::from(&prog).join("Claude"));
                paths.push(PathBuf::from(&prog).join("Anthropic").join("Claude"));
            }
        }
        
        paths
    }
    
    fn check_binary(&self, path: &Path) -> Option<String> {
        if !path.exists() {
            return None;
        }
        
        // Try to get version
        if let Ok(output) = Command::new(path).arg("--version").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Extract version using regex
                if let Ok(re) = regex::Regex::new(r"(\d+\.\d+\.\d+)") {
                    if let Some(caps) = re.captures(&stdout) {
                        return Some(caps[1].to_string());
                    }
                }
            }
        }
        
        // Binary exists but version unknown
        Some("unknown".to_string())
    }
}

#[async_trait]
impl ClaudeDetector for DefaultClaudeDetector {
    async fn detect(&self) -> Result<DetectionResult> {
        let binary_names = vec!["claude-code", "claude", "claude-cli"];
        
        for name in &binary_names {
            // Try which command first
            if let Ok(output) = Command::new("which").arg(name).output() {
                if output.status.success() {
                    let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !path_str.is_empty() {
                        let path = PathBuf::from(path_str);
                        if let Some(version) = self.check_binary(&path) {
                            return Ok(DetectionResult {
                                found: true,
                                path: Some(path),
                                version: Some(version.clone()),
                                is_compatible: version != "unknown",
                            });
                        }
                    }
                }
            }
            
            // Search in known paths
            for search_path in &self.search_paths {
                let candidate = search_path.join(name);
                if let Some(version) = self.check_binary(&candidate) {
                    return Ok(DetectionResult {
                        found: true,
                        path: Some(candidate),
                        version: Some(version.clone()),
                        is_compatible: version != "unknown",
                    });
                }
                
                // Windows .exe extension
                #[cfg(windows)]
                {
                    let candidate = search_path.join(format!("{}.exe", name));
                    if let Some(version) = self.check_binary(&candidate) {
                        return Ok(DetectionResult {
                            found: true,
                            path: Some(candidate),
                            version: Some(version.clone()),
                            is_compatible: version != "unknown",
                        });
                    }
                }
            }
        }
        
        Ok(DetectionResult {
            found: false,
            path: None,
            version: None,
            is_compatible: false,
        })
    }
    
    fn get_installation_paths(&self) -> InstallationPaths {
        let mut system_paths = vec![];
        let mut user_paths = vec![];
        
        #[cfg(unix)]
        {
            system_paths.push(PathBuf::from("/usr/local/bin"));
            if let Ok(home) = std::env::var("HOME") {
                user_paths.push(PathBuf::from(&home).join(".local").join("bin"));
            }
        }
        
        #[cfg(windows)]
        {
            if let Ok(prog) = std::env::var("ProgramFiles") {
                system_paths.push(PathBuf::from(&prog).join("Claude"));
            }
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                user_paths.push(PathBuf::from(&local).join("Claude"));
            }
        }
        
        InstallationPaths {
            system_paths,
            user_paths,
            search_paths: self.search_paths.clone(),
        }
    }
    
    async fn validate_version(&self, version: &str) -> Result<bool> {
        // Simple version check - could be enhanced with proper semver
        Ok(!version.starts_with("0."))
    }
}

/// Default MCP configurator implementation
pub struct DefaultMCPConfigurator;

#[async_trait]
impl MCPConfigurator for DefaultMCPConfigurator {
    async fn create_config(&self, config: &MCPConfig, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| OnboardingError::ConfigurationError(format!("Failed to create directory: {}", e)))?;
        }
        
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| OnboardingError::ConfigurationError(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(path, json)
            .map_err(|e| OnboardingError::ConfigurationError(format!("Failed to write config: {}", e)))?;
        
        Ok(())
    }
    
    async fn load_config(&self, path: &Path) -> Result<MCPConfig> {
        let content = fs::read_to_string(path)
            .map_err(|e| OnboardingError::ConfigurationError(format!("Failed to read config: {}", e)))?;
        
        serde_json::from_str(&content)
            .map_err(|e| OnboardingError::ConfigurationError(format!("Failed to parse config: {}", e)))
    }
    
    async fn validate_config(&self, config: &MCPConfig) -> Result<bool> {
        Ok(config.has_servers())
    }
    
    async fn check_github_token(&self) -> Result<Option<String>> {
        Ok(std::env::var("GITHUB_TOKEN")
            .or_else(|_| std::env::var("GH_TOKEN"))
            .ok())
    }
    
    fn generate_swarm_id(&self) -> String {
        Uuid::new_v4().to_string()
    }
}

/// Default interactive prompt implementation
pub struct DefaultInteractivePrompt {
    theme: ColorfulTheme,
}

impl DefaultInteractivePrompt {
    pub fn new() -> Self {
        Self {
            theme: ColorfulTheme::default(),
        }
    }
}

#[async_trait]
impl InteractivePrompt for DefaultInteractivePrompt {
    async fn confirm(&self, message: &str, default: bool) -> Result<bool> {
        Confirm::with_theme(&self.theme)
            .with_prompt(message)
            .default(default)
            .interact()
            .map_err(|e| OnboardingError::ConfigurationError(format!("Prompt error: {}", e)))
    }
    
    async fn choice(&self, message: &str, options: &[&str]) -> Result<usize> {
        Select::with_theme(&self.theme)
            .with_prompt(message)
            .items(options)
            .interact()
            .map_err(|e| OnboardingError::ConfigurationError(format!("Choice error: {}", e)))
    }
    
    async fn input(&self, message: &str, default: Option<&str>) -> Result<String> {
        let mut input = Input::<String>::with_theme(&self.theme)
            .with_prompt(message);
        
        if let Some(def) = default {
            input = input.default(def.to_string());
        }
        
        input.interact_text()
            .map_err(|e| OnboardingError::ConfigurationError(format!("Input error: {}", e)))
    }
    
    async fn password(&self, message: &str) -> Result<String> {
        Password::with_theme(&self.theme)
            .with_prompt(message)
            .interact()
            .map_err(|e| OnboardingError::ConfigurationError(format!("Password error: {}", e)))
    }
    
    async fn info(&self, message: &str) {
        println!("ℹ️  {}", message.blue());
    }
    
    async fn warning(&self, message: &str) {
        println!("⚠️  {}", message.yellow());
    }
    
    async fn error(&self, message: &str) {
        println!("❌ {}", message.red());
    }
    
    async fn success(&self, message: &str) {
        println!("✅ {}", message.green());
    }
    
    async fn progress(&self, message: &str, current: u64, total: u64) {
        let percentage = if total > 0 {
            (current as f32 / total as f32 * 100.0) as u32
        } else {
            0
        };
        
        let bar_width = 20;
        let filled = (bar_width * percentage) / 100;
        let empty = bar_width - filled;
        
        print!("\r{} [{}{}] {}%", 
            message,
            "█".repeat(filled as usize),
            "░".repeat(empty as usize),
            percentage
        );
        
        if current >= total {
            println!();
        }
    }
}

/// Default launch manager implementation
pub struct DefaultLaunchManager;

#[async_trait]
impl LaunchManager for DefaultLaunchManager {
    async fn launch(&self, options: LaunchOptions) -> Result<()> {
        let mut cmd = Command::new("claude-code");
        
        cmd.arg("--mcp-config")
           .arg(&options.mcp_config_path);
        
        if options.skip_permissions {
            cmd.arg("--dangerously-skip-permissions");
        }
        
        cmd.spawn()
            .map_err(|e| OnboardingError::LaunchError(format!("Failed to launch Claude Code: {}", e)))?;
        
        Ok(())
    }
    
    async fn is_running(&self) -> Result<bool> {
        // Simple check - could be enhanced with proper process detection
        #[cfg(unix)]
        {
            let output = Command::new("pgrep")
                .arg("-f")
                .arg("claude-code")
                .output()
                .map_err(|e| OnboardingError::LaunchError(format!("Failed to check process: {}", e)))?;
            
            Ok(output.status.success())
        }
        
        #[cfg(windows)]
        {
            // Windows implementation would use tasklist
            Ok(false)
        }
    }
    
    async fn wait_for_ready(&self, timeout_secs: u64) -> Result<()> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(timeout_secs);
        
        while start.elapsed() < timeout {
            if self.is_running().await? {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        
        Err(OnboardingError::LaunchError("Timeout waiting for Claude Code".to_string()))
    }
    
    async fn guide_auth(&self) -> Result<()> {
        // This would provide step-by-step auth guidance
        Ok(())
    }
}

/// Stub installer - actual implementation would download and install
pub struct StubInstaller;

#[async_trait]
impl Installer for StubInstaller {
    async fn install(&self, _options: InstallOptions) -> Result<PathBuf> {
        Err(OnboardingError::InstallationFailed(
            "Installation not implemented in stub".to_string()
        ))
    }
    
    async fn requires_elevation(&self, _target_dir: &Path) -> Result<bool> {
        Ok(false)
    }
    
    async fn verify_installation(&self, path: &Path) -> Result<bool> {
        Ok(path.exists())
    }
    
    async fn rollback(&self, _install_path: &Path) -> Result<()> {
        Ok(())
    }
}