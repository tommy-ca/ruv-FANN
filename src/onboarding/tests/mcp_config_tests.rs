use super::super::mcp_config::{MCPConfigurator, MCPServer, MCPConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_mcp_config_creation() {
        let configurator = MCPConfigurator::new();
        let config = configurator.create_empty_config();
        
        assert!(config.servers.is_empty());
        assert!(config.created_at > 0);
    }

    #[test]
    fn test_add_github_mcp_with_token() {
        let mut configurator = MCPConfigurator::new();
        let token = "test_github_token".to_string();
        
        configurator.add_github_mcp(Some(token.clone())).unwrap();
        
        let server = configurator.get_server("github").unwrap();
        assert_eq!(server.command, "npx");
        assert!(server.args.contains(&"-y".to_string()));
        assert!(server.args.contains(&"@modelcontextprotocol/server-github".to_string()));
        assert_eq!(server.env.get("GITHUB_TOKEN").unwrap(), &token);
    }

    #[test]
    fn test_add_github_mcp_without_token() {
        let mut configurator = MCPConfigurator::new();
        
        configurator.add_github_mcp(None).unwrap();
        
        let server = configurator.get_server("github").unwrap();
        assert!(server.env.is_empty());
    }

    #[test]
    fn test_add_ruv_swarm_mcp() {
        let mut configurator = MCPConfigurator::new();
        let swarm_id = "test-swarm-123".to_string();
        let topology = "mesh".to_string();
        
        configurator.add_ruv_swarm_mcp(swarm_id.clone(), topology.clone()).unwrap();
        
        let server = configurator.get_server("ruv-swarm").unwrap();
        assert_eq!(server.command, "npx");
        assert!(server.args.contains(&"ruv-swarm".to_string()));
        assert!(server.args.contains(&"mcp".to_string()));
        assert!(server.args.contains(&"start".to_string()));
        assert_eq!(server.env.get("SWARM_ID").unwrap(), &swarm_id);
        assert_eq!(server.env.get("SWARM_TOPOLOGY").unwrap(), &topology);
    }

    #[test]
    fn test_json_generation() {
        let mut configurator = MCPConfigurator::new();
        configurator.add_github_mcp(Some("token123".to_string())).unwrap();
        configurator.add_ruv_swarm_mcp("swarm123".to_string(), "star".to_string()).unwrap();
        
        let json = configurator.to_json().unwrap();
        
        // Parse back to verify structure
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["mcpServers"]["github"].is_object());
        assert!(parsed["mcpServers"]["ruv-swarm"].is_object());
        assert_eq!(parsed["mcpServers"]["github"]["command"], "npx");
        assert_eq!(parsed["mcpServers"]["ruv-swarm"]["env"]["SWARM_ID"], "swarm123");
    }

    #[test]
    fn test_save_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let mut configurator = MCPConfigurator::new();
        
        configurator.add_github_mcp(Some("token".to_string())).unwrap();
        
        let config_path = temp_dir.path().join(".claude").join("mcp.json");
        configurator.save_to_file(&config_path).unwrap();
        
        assert!(config_path.exists());
        let content = fs::read_to_string(&config_path).unwrap();
        assert!(content.contains("github"));
        assert!(content.contains("mcpServers"));
    }

    #[test]
    fn test_load_existing_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("mcp.json");
        
        // Create existing config
        let existing_config = r#"{
            "mcpServers": {
                "existing-server": {
                    "command": "test",
                    "args": ["arg1"],
                    "env": {}
                }
            }
        }"#;
        fs::write(&config_path, existing_config).unwrap();
        
        let mut configurator = MCPConfigurator::load_from_file(&config_path).unwrap();
        configurator.add_github_mcp(None).unwrap();
        
        // Should preserve existing server
        assert!(configurator.get_server("existing-server").is_some());
        assert!(configurator.get_server("github").is_some());
    }

    #[test]
    fn test_validate_config() {
        let mut configurator = MCPConfigurator::new();
        
        // Empty config should be valid
        assert!(configurator.validate().is_ok());
        
        // Add valid servers
        configurator.add_github_mcp(None).unwrap();
        configurator.add_ruv_swarm_mcp("id".to_string(), "mesh".to_string()).unwrap();
        assert!(configurator.validate().is_ok());
    }

    #[test]
    fn test_token_detection() {
        let configurator = MCPConfigurator::new();
        
        // Set test env vars
        std::env::set_var("TEST_GITHUB_TOKEN", "test_token");
        let token = configurator.detect_github_token_with_var("TEST_GITHUB_TOKEN");
        assert_eq!(token, Some("test_token".to_string()));
        
        // Clean up
        std::env::remove_var("TEST_GITHUB_TOKEN");
        
        // Test non-existent var
        let token = configurator.detect_github_token_with_var("NONEXISTENT_VAR");
        assert_eq!(token, None);
    }

    #[test]
    fn test_merge_configs() {
        let mut config1 = MCPConfigurator::new();
        config1.add_github_mcp(Some("token1".to_string())).unwrap();
        
        let mut config2 = MCPConfigurator::new();
        config2.add_ruv_swarm_mcp("swarm1".to_string(), "ring".to_string()).unwrap();
        
        config1.merge(config2).unwrap();
        
        assert!(config1.get_server("github").is_some());
        assert!(config1.get_server("ruv-swarm").is_some());
    }

    #[test]
    fn test_remove_server() {
        let mut configurator = MCPConfigurator::new();
        configurator.add_github_mcp(None).unwrap();
        configurator.add_ruv_swarm_mcp("id".to_string(), "mesh".to_string()).unwrap();
        
        assert!(configurator.get_server("github").is_some());
        configurator.remove_server("github").unwrap();
        assert!(configurator.get_server("github").is_none());
        assert!(configurator.get_server("ruv-swarm").is_some());
    }

    #[test]
    fn test_config_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let configurator = MCPConfigurator::new();
        
        let claude_dir = temp_dir.path().join(".claude");
        assert!(!claude_dir.exists());
        
        configurator.ensure_claude_directory(&claude_dir).unwrap();
        assert!(claude_dir.exists());
        assert!(claude_dir.is_dir());
    }

    #[test]
    fn test_server_equality() {
        let server1 = MCPServer {
            command: "npx".to_string(),
            args: vec!["arg1".to_string()],
            env: HashMap::new(),
        };
        
        let server2 = MCPServer {
            command: "npx".to_string(),
            args: vec!["arg1".to_string()],
            env: HashMap::new(),
        };
        
        assert_eq!(server1, server2);
    }
}