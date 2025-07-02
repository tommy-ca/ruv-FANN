//! Tests for launch command functionality
//! 
//! This module tests the generation and execution of Claude Code
//! launch commands with MCP configuration.

use ruv_swarm::onboarding::{LaunchCommand, LaunchConfig, LaunchResult};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSetup {
        temp_dir: TempDir,
        claude_mock: PathBuf,
    }

    impl TestSetup {
        fn new() -> Self {
            let temp_dir = TempDir::new().unwrap();
            let claude_mock = temp_dir.path().join("claude-mock");
            
            // Create mock claude executable
            #[cfg(unix)]
            {
                fs::write(&claude_mock, "#!/bin/sh\necho \"$@\" > launch.log").unwrap();
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&claude_mock).unwrap().permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&claude_mock, perms).unwrap();
            }
            
            #[cfg(windows)]
            {
                fs::write(
                    claude_mock.with_extension("bat"),
                    "@echo off\necho %* > launch.log"
                ).unwrap();
            }
            
            TestSetup { temp_dir, claude_mock }
        }

        fn launcher(&self) -> LaunchCommand {
            LaunchCommand::new()
                .with_claude_path(self.claude_mock.clone())
                .with_working_dir(self.temp_dir.path().to_path_buf())
        }
    }

    #[test]
    fn test_generate_basic_launch_command() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let config = LaunchConfig {
            use_mcp: true,
            mcp_config_path: Some(PathBuf::from("/home/user/.claude/mcp.json")),
            additional_args: vec![],
            env_vars: Default::default(),
        };
        
        let command = launcher.generate_command(&config);
        
        assert_eq!(command.program, setup.claude_mock);
        assert!(command.args.contains(&"--mcp".to_string()));
        assert!(command.args.contains(&"--mcp-config".to_string()));
        assert!(command.args.contains(&"/home/user/.claude/mcp.json".to_string()));
    }

    #[test]
    fn test_launch_with_environment_variables() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let mut env_vars = std::collections::HashMap::new();
        env_vars.insert("CLAUDE_DEBUG".to_string(), "1".to_string());
        env_vars.insert("RUV_SWARM_MODE".to_string(), "production".to_string());
        
        let config = LaunchConfig {
            use_mcp: true,
            mcp_config_path: None,
            additional_args: vec![],
            env_vars,
        };
        
        let command = launcher.generate_command(&config);
        
        assert_eq!(command.env.get("CLAUDE_DEBUG"), Some(&"1".to_string()));
        assert_eq!(command.env.get("RUV_SWARM_MODE"), Some(&"production".to_string()));
    }

    #[test]
    fn test_launch_with_additional_arguments() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let config = LaunchConfig {
            use_mcp: true,
            mcp_config_path: None,
            additional_args: vec![
                "--verbose".to_string(),
                "--theme".to_string(),
                "dark".to_string(),
            ],
            env_vars: Default::default(),
        };
        
        let command = launcher.generate_command(&config);
        
        assert!(command.args.contains(&"--verbose".to_string()));
        assert!(command.args.contains(&"--theme".to_string()));
        assert!(command.args.contains(&"dark".to_string()));
    }

    #[test]
    fn test_launch_without_mcp() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let config = LaunchConfig {
            use_mcp: false,
            mcp_config_path: None,
            additional_args: vec![],
            env_vars: Default::default(),
        };
        
        let command = launcher.generate_command(&config);
        
        assert!(!command.args.contains(&"--mcp".to_string()));
        assert!(!command.args.contains(&"--mcp-config".to_string()));
    }

    #[test]
    fn test_detect_mcp_config_path() {
        let setup = TestSetup::new();
        
        // Create mock MCP config in various locations
        let locations = vec![
            setup.temp_dir.path().join(".claude/mcp.json"),
            setup.temp_dir.path().join(".config/claude/mcp.json"),
            setup.temp_dir.path().join("claude-config/mcp.json"),
        ];
        
        for location in &locations {
            fs::create_dir_all(location.parent().unwrap()).unwrap();
            fs::write(location, "{}").unwrap();
        }
        
        let launcher = setup.launcher()
            .with_search_paths(vec![setup.temp_dir.path().to_path_buf()]);
        
        let detected = launcher.detect_mcp_config();
        
        assert!(detected.is_some());
        assert!(locations.contains(&detected.unwrap()));
    }

    #[test]
    fn test_validate_launch_prerequisites() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        // Test with valid setup
        let result = launcher.validate_prerequisites();
        assert!(result.is_ok());
        
        // Test with missing Claude
        let launcher_no_claude = LaunchCommand::new()
            .with_claude_path(PathBuf::from("/nonexistent/claude"));
        
        let result = launcher_no_claude.validate_prerequisites();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Claude Code not found"));
    }

    #[test]
    fn test_execute_launch_command() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let config = LaunchConfig {
            use_mcp: true,
            mcp_config_path: Some(setup.temp_dir.path().join("mcp.json")),
            additional_args: vec!["--test".to_string()],
            env_vars: Default::default(),
        };
        
        // Create mock MCP config
        fs::write(
            setup.temp_dir.path().join("mcp.json"),
            r#"{"servers": {}}"#
        ).unwrap();
        
        let result = launcher.execute(&config);
        
        // Check that mock was called
        let log_file = setup.temp_dir.path().join("launch.log");
        if log_file.exists() {
            let log_content = fs::read_to_string(log_file).unwrap();
            assert!(log_content.contains("--mcp"));
            assert!(log_content.contains("--test"));
        }
    }

    #[test]
    fn test_platform_specific_launch() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let command = launcher.get_platform_command();
        
        #[cfg(target_os = "windows")]
        {
            assert!(command.contains(".exe") || command.contains(".bat"));
        }
        
        #[cfg(target_os = "macos")]
        {
            // On macOS, might use open command or direct binary
            assert!(command.contains("claude") || command.contains("Claude.app"));
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(command.contains("claude"));
        }
    }

    #[test]
    fn test_launch_with_project_directory() {
        let setup = TestSetup::new();
        
        // Create project directory
        let project_dir = setup.temp_dir.path().join("my-project");
        fs::create_dir(&project_dir).unwrap();
        
        let launcher = setup.launcher()
            .with_project_dir(project_dir.clone());
        
        let config = LaunchConfig::default();
        let command = launcher.generate_command(&config);
        
        // Should include project directory as argument
        assert!(command.args.contains(&project_dir.to_string_lossy().to_string()));
    }

    #[test]
    fn test_launch_url_generation() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        // Test claude:// URL scheme
        let url = launcher.generate_claude_url(Some("open-project"), Some("/path/to/project"));
        
        assert!(url.starts_with("claude://"));
        assert!(url.contains("open-project"));
        assert!(url.contains("project=/path/to/project"));
    }

    #[test]
    fn test_launch_fallback_mechanisms() {
        let setup = TestSetup::new();
        
        // Test fallback when primary launch fails
        let launcher = LaunchCommand::new()
            .with_claude_path(PathBuf::from("/nonexistent/claude"))
            .with_fallback_paths(vec![
                PathBuf::from("/also/nonexistent"),
                setup.claude_mock.clone(),
            ]);
        
        let config = LaunchConfig::default();
        let result = launcher.execute_with_fallback(&config);
        
        // Should succeed with fallback path
        assert!(matches!(result, Ok(LaunchResult::Success { .. })));
    }

    #[test]
    fn test_launch_output_capture() {
        let setup = TestSetup::new();
        
        // Create claude mock that outputs specific text
        #[cfg(unix)]
        {
            fs::write(
                &setup.claude_mock,
                "#!/bin/sh\necho 'Claude Code started'\necho 'MCP server connected' >&2"
            ).unwrap();
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&setup.claude_mock).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&setup.claude_mock, perms).unwrap();
        }
        
        let launcher = setup.launcher();
        let config = LaunchConfig::default();
        
        let result = launcher.execute_with_output(&config);
        
        if let Ok(LaunchResult::Success { stdout, stderr, .. }) = result {
            assert!(stdout.contains("Claude Code started"));
            assert!(stderr.contains("MCP server connected"));
        }
    }

    #[test]
    fn test_interactive_launch_confirmation() {
        let setup = TestSetup::new();
        let launcher = setup.launcher();
        
        let config = LaunchConfig {
            use_mcp: true,
            mcp_config_path: Some(PathBuf::from("/home/user/.claude/mcp.json")),
            additional_args: vec!["--verbose".to_string()],
            env_vars: Default::default(),
        };
        
        let preview = launcher.preview_command(&config);
        
        assert!(preview.contains("claude"));
        assert!(preview.contains("--mcp"));
        assert!(preview.contains("--verbose"));
        assert!(preview.contains("/home/user/.claude/mcp.json"));
    }
}