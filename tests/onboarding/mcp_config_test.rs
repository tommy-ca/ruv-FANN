//! Tests for MCP (Model Context Protocol) configuration generation
//! 
//! This module tests the generation and validation of MCP configuration
//! files for Claude Code integration.

use ruv_swarm::onboarding::{McpConfigGenerator, McpConfig, McpServerConfig, McpError};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSetup {
        temp_dir: TempDir,
        config_path: PathBuf,
    }

    impl TestSetup {
        fn new() -> Self {
            let temp_dir = TempDir::new().unwrap();
            let config_path = temp_dir.path().join("mcp.json");
            
            TestSetup {
                temp_dir,
                config_path,
            }
        }

        fn generator(&self) -> McpConfigGenerator {
            McpConfigGenerator::new(self.config_path.clone())
        }
    }

    #[test]
    fn test_generate_basic_mcp_config() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm".to_string(),
                    command: "npx".to_string(),
                    args: vec!["ruv-swarm".to_string(), "mcp".to_string(), "start".to_string()],
                    env: None,
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        let result = generator.generate(config);
        
        assert!(result.is_ok());
        assert!(setup.config_path.exists());
        
        // Verify JSON structure
        let content = fs::read_to_string(&setup.config_path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();
        
        assert!(json["servers"]["ruv-swarm"].is_object());
        assert_eq!(json["servers"]["ruv-swarm"]["command"], "npx");
        assert!(json["servers"]["ruv-swarm"]["args"].is_array());
    }

    #[test]
    fn test_generate_mcp_config_with_tcp() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm-tcp".to_string(),
                    command: "ruv-swarm".to_string(),
                    args: vec!["mcp".to_string(), "server".to_string()],
                    env: None,
                    stdio: false,
                    tcp: Some(("localhost".to_string(), 8080)),
                },
            ],
        };
        
        let result = generator.generate(config);
        
        assert!(result.is_ok());
        
        let content = fs::read_to_string(&setup.config_path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();
        
        assert_eq!(json["servers"]["ruv-swarm-tcp"]["tcp"]["host"], "localhost");
        assert_eq!(json["servers"]["ruv-swarm-tcp"]["tcp"]["port"], 8080);
    }

    #[test]
    fn test_generate_mcp_config_with_environment() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        let mut env_vars = std::collections::HashMap::new();
        env_vars.insert("RUST_LOG".to_string(), "debug".to_string());
        env_vars.insert("RUV_SWARM_MODE".to_string(), "production".to_string());
        
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm".to_string(),
                    command: "npx".to_string(),
                    args: vec!["ruv-swarm".to_string(), "mcp".to_string()],
                    env: Some(env_vars),
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        let result = generator.generate(config);
        
        assert!(result.is_ok());
        
        let content = fs::read_to_string(&setup.config_path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();
        
        assert_eq!(json["servers"]["ruv-swarm"]["env"]["RUST_LOG"], "debug");
        assert_eq!(json["servers"]["ruv-swarm"]["env"]["RUV_SWARM_MODE"], "production");
    }

    #[test]
    fn test_merge_with_existing_config() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        // Create existing config
        let existing_config = json!({
            "servers": {
                "other-server": {
                    "command": "other",
                    "args": ["start"]
                }
            }
        });
        
        fs::write(&setup.config_path, existing_config.to_string()).unwrap();
        
        // Generate new config with merge
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm".to_string(),
                    command: "npx".to_string(),
                    args: vec!["ruv-swarm".to_string(), "mcp".to_string()],
                    env: None,
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        let result = generator.merge(config);
        
        assert!(result.is_ok());
        
        let content = fs::read_to_string(&setup.config_path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();
        
        // Both servers should exist
        assert!(json["servers"]["other-server"].is_object());
        assert!(json["servers"]["ruv-swarm"].is_object());
    }

    #[test]
    fn test_validate_mcp_config() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        // Valid config
        let valid_config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm".to_string(),
                    command: "npx".to_string(),
                    args: vec!["ruv-swarm".to_string()],
                    env: None,
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        assert!(generator.validate(&valid_config).is_ok());
        
        // Invalid config - empty name
        let invalid_config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "".to_string(),
                    command: "npx".to_string(),
                    args: vec![],
                    env: None,
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        assert!(matches!(
            generator.validate(&invalid_config),
            Err(McpError::ValidationError(_))
        ));
    }

    #[test]
    fn test_detect_claude_mcp_location() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        // Create mock Claude config directory
        let claude_dir = setup.temp_dir.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        
        let detected_path = generator.detect_claude_config_path(Some(setup.temp_dir.path()));
        
        assert!(detected_path.is_some());
        assert_eq!(detected_path.unwrap(), claude_dir.join("mcp.json"));
    }

    #[test]
    fn test_backup_existing_config() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        // Create existing config
        let original_content = json!({
            "servers": {
                "existing": {
                    "command": "test"
                }
            }
        });
        
        fs::write(&setup.config_path, original_content.to_string()).unwrap();
        
        let result = generator.backup();
        
        assert!(result.is_ok());
        
        // Check backup exists
        let backup_path = setup.config_path.with_extension("json.backup");
        assert!(backup_path.exists());
        
        let backup_content = fs::read_to_string(backup_path).unwrap();
        assert_eq!(backup_content, original_content.to_string());
    }

    #[test]
    fn test_platform_specific_paths() {
        let generator = McpConfigGenerator::default();
        let paths = generator.get_platform_mcp_paths();
        
        #[cfg(target_os = "windows")]
        {
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("AppData")));
        }
        
        #[cfg(target_os = "macos")]
        {
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("Library/Application Support")));
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains(".config")));
        }
    }

    #[test]
    fn test_generate_launch_command() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        let command = generator.generate_launch_command();
        
        #[cfg(target_os = "windows")]
        assert!(command.contains("claude.exe") || command.contains("claude"));
        
        #[cfg(unix)]
        assert!(command.contains("claude"));
        
        // Should include MCP flag
        assert!(command.contains("--mcp") || command.contains("mcp"));
    }

    #[test]
    fn test_config_permissions() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm".to_string(),
                    command: "npx".to_string(),
                    args: vec!["ruv-swarm".to_string()],
                    env: None,
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        generator.generate(config).unwrap();
        
        // Check file permissions
        let metadata = fs::metadata(&setup.config_path).unwrap();
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata.permissions().mode();
            // Should be readable by user
            assert!(mode & 0o400 != 0);
        }
    }

    #[test]
    fn test_handle_malformed_existing_config() {
        let setup = TestSetup::new();
        let generator = setup.generator();
        
        // Write malformed JSON
        fs::write(&setup.config_path, "{ invalid json").unwrap();
        
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "ruv-swarm".to_string(),
                    command: "npx".to_string(),
                    args: vec!["ruv-swarm".to_string()],
                    env: None,
                    stdio: true,
                    tcp: None,
                },
            ],
        };
        
        // Should handle gracefully and create backup
        let result = generator.merge(config);
        
        assert!(result.is_ok());
        assert!(setup.config_path.with_extension("json.backup").exists());
    }
}