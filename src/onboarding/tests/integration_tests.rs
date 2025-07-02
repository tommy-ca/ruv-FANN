use super::super::{ClaudeDetector, OnboardingPrompts, MCPConfigurator};
use tempfile::TempDir;
use std::fs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_onboarding_flow_no_claude() {
        let temp_dir = TempDir::new().unwrap();
        let detector = ClaudeDetector::with_search_paths(vec![temp_dir.path().to_path_buf()]);
        let prompts = OnboardingPrompts::new();
        let mut configurator = MCPConfigurator::new();
        
        // Simulate detection - Claude not found
        let claude_info = detector.detect().unwrap();
        assert!(claude_info.is_none());
        
        // Simulate user choosing to install
        let install_response = prompts.parse_yes_no("y", true);
        assert_eq!(install_response, super::super::prompts::PromptResponse::Yes);
        
        // Simulate MCP configuration
        configurator.add_github_mcp(Some("test_token".to_string())).unwrap();
        configurator.add_ruv_swarm_mcp("swarm123".to_string(), "mesh".to_string()).unwrap();
        
        // Verify configuration
        assert!(configurator.get_server("github").is_some());
        assert!(configurator.get_server("ruv-swarm").is_some());
    }

    #[test]
    fn test_partial_setup_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let mcp_path = temp_dir.path().join(".claude").join("mcp.json");
        
        // Create partial config (only GitHub)
        fs::create_dir_all(mcp_path.parent().unwrap()).unwrap();
        let partial_config = r#"{
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-github"],
                    "env": {}
                }
            }
        }"#;
        fs::write(&mcp_path, partial_config).unwrap();
        
        // Load and complete configuration
        let mut configurator = MCPConfigurator::load_from_file(&mcp_path).unwrap();
        assert!(configurator.get_server("github").is_some());
        assert!(configurator.get_server("ruv-swarm").is_none());
        
        // Add missing ruv-swarm
        configurator.add_ruv_swarm_mcp("swarm456".to_string(), "star".to_string()).unwrap();
        configurator.save_to_file(&mcp_path).unwrap();
        
        // Verify complete config
        let reloaded = MCPConfigurator::load_from_file(&mcp_path).unwrap();
        assert!(reloaded.get_server("github").is_some());
        assert!(reloaded.get_server("ruv-swarm").is_some());
    }

    #[test]
    fn test_non_interactive_mode() {
        let mut prompts = OnboardingPrompts::new();
        prompts.set_non_interactive(true);
        
        let temp_dir = TempDir::new().unwrap();
        let mut configurator = MCPConfigurator::new();
        
        // In non-interactive mode, should use defaults
        let response = prompts.auto_respond_yes_no(true);
        assert_eq!(response, super::super::prompts::PromptResponse::Yes);
        
        // Configure with defaults
        configurator.add_github_mcp(None).unwrap(); // No token in non-interactive
        configurator.add_ruv_swarm_mcp("auto-swarm".to_string(), "mesh".to_string()).unwrap();
        
        let config_path = temp_dir.path().join("mcp.json");
        configurator.save_to_file(&config_path).unwrap();
        assert!(config_path.exists());
    }

    #[test]
    fn test_error_handling_scenarios() {
        // Test permission denied scenario
        #[cfg(unix)]
        {
            let temp_dir = TempDir::new().unwrap();
            let readonly_dir = temp_dir.path().join("readonly");
            fs::create_dir(&readonly_dir).unwrap();
            
            // Make directory read-only
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&readonly_dir).unwrap().permissions();
            perms.set_mode(0o444);
            fs::set_permissions(&readonly_dir, perms).unwrap();
            
            let configurator = MCPConfigurator::new();
            let result = configurator.ensure_claude_directory(&readonly_dir.join(".claude"));
            assert!(result.is_err());
            
            // Restore permissions for cleanup
            perms.set_mode(0o755);
            fs::set_permissions(&readonly_dir, perms).unwrap();
        }
    }

    #[test]
    fn test_launch_command_generation() {
        let configurator = MCPConfigurator::new();
        let mcp_path = ".claude/mcp.json";
        
        let launch_cmd = configurator.generate_launch_command(mcp_path);
        assert!(launch_cmd.contains("claude-code"));
        assert!(launch_cmd.contains("--mcp-config"));
        assert!(launch_cmd.contains(mcp_path));
        assert!(launch_cmd.contains("--dangerously-skip-permissions"));
    }
}