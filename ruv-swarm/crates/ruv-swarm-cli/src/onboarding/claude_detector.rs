//! Claude Code detection for cross-platform onboarding

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Information about Claude Code installation
#[derive(Debug, Clone)]
pub struct ClaudeInfo {
    pub installed: bool,
    pub path: String,
    pub version: Option<String>,
}

/// Trait for Claude Code detection
#[async_trait]
pub trait ClaudeDetector {
    async fn detect(&self) -> Result<ClaudeInfo>;
    async fn validate_installation(&self, path: &str) -> Result<bool>;
}

/// Default implementation of Claude Code detector
pub struct DefaultClaudeDetector {
    search_paths: Vec<PathBuf>,
}

impl DefaultClaudeDetector {
    pub fn new() -> Self {
        Self {
            search_paths: Self::get_default_search_paths(),
        }
    }
    
    /// Get platform-specific search paths for Claude Code
    fn get_default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        // Common paths where Claude Code might be installed
        if cfg!(target_os = "windows") {
            // Windows paths
            paths.extend(vec![
                PathBuf::from("C:\\Program Files\\Claude\\claude.exe"),
                PathBuf::from("C:\\Program Files (x86)\\Claude\\claude.exe"),
                PathBuf::from(format!("{}\\AppData\\Local\\Claude\\claude.exe", 
                    std::env::var("USERPROFILE").unwrap_or_default())),
            ]);
        } else if cfg!(target_os = "macos") {
            // macOS paths
            paths.extend(vec![
                PathBuf::from("/Applications/Claude.app/Contents/MacOS/claude"),
                PathBuf::from(format!("{}/Applications/Claude.app/Contents/MacOS/claude",
                    std::env::var("HOME").unwrap_or_default())),
            ]);
        } else {
            // Linux paths
            paths.extend(vec![
                PathBuf::from("/usr/local/bin/claude"),
                PathBuf::from("/usr/bin/claude"),
                PathBuf::from(format!("{}/.local/bin/claude",
                    std::env::var("HOME").unwrap_or_default())),
                PathBuf::from("/opt/claude/claude"),
            ]);
        }
        
        // Also check PATH
        if let Ok(path_env) = std::env::var("PATH") {
            for path_dir in std::env::split_paths(&path_env) {
                let claude_path = if cfg!(target_os = "windows") {
                    path_dir.join("claude.exe")
                } else {
                    path_dir.join("claude")
                };
                paths.push(claude_path);
            }
        }
        
        paths
    }
    
    /// Detect Claude Code version
    async fn get_version(&self, path: &str) -> Option<String> {
        let output = Command::new(path)
            .arg("--version")
            .output()
            .ok()?;
            
        if output.status.success() {
            String::from_utf8(output.stdout)
                .ok()
                .map(|s| s.trim().to_string())
        } else {
            None
        }
    }
}

#[async_trait]
impl ClaudeDetector for DefaultClaudeDetector {
    async fn detect(&self) -> Result<ClaudeInfo> {
        // First try to find Claude in the search paths
        for path in &self.search_paths {
            if path.exists() {
                let path_str = path.to_string_lossy().to_string();
                if self.validate_installation(&path_str).await? {
                    let version = self.get_version(&path_str).await;
                    return Ok(ClaudeInfo {
                        installed: true,
                        path: path_str,
                        version,
                    });
                }
            }
        }
        
        // Try to find Claude using `which` or `where` command
        let find_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
        let claude_name = if cfg!(target_os = "windows") { "claude.exe" } else { "claude" };
        
        if let Ok(output) = Command::new(find_cmd).arg(claude_name).output() {
            if output.status.success() {
                let path_str = String::from_utf8(output.stdout)
                    .map_err(|_| anyhow!("Invalid UTF-8 in command output"))?
                    .trim()
                    .to_string();
                    
                if self.validate_installation(&path_str).await? {
                    let version = self.get_version(&path_str).await;
                    return Ok(ClaudeInfo {
                        installed: true,
                        path: path_str,
                        version,
                    });
                }
            }
        }
        
        // Claude Code not found
        Ok(ClaudeInfo {
            installed: false,
            path: String::new(),
            version: None,
        })
    }
    
    async fn validate_installation(&self, path: &str) -> Result<bool> {
        if !Path::new(path).exists() {
            return Ok(false);
        }
        
        // Try to run Claude with --help to validate it's working
        let output = Command::new(path)
            .arg("--help")
            .output()
            .map_err(|_| anyhow!("Failed to execute Claude command"))?;
            
        Ok(output.status.success())
    }
}

impl Default for DefaultClaudeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_claude_detector_creation() {
        let detector = DefaultClaudeDetector::new();
        assert!(!detector.search_paths.is_empty());
    }
    
    #[tokio::test]
    async fn test_invalid_path_validation() {
        let detector = DefaultClaudeDetector::new();
        let result = detector.validate_installation("/nonexistent/path").await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}