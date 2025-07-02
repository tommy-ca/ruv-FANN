//! Tests for ruv-swarm installation flow
//! 
//! This module tests the complete installation process including
//! dependency checking, file creation, and verification steps.

use ruv_swarm::onboarding::{Installer, InstallConfig, InstallResult, InstallError};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;
use mockall::predicate::*;
use mockall::mock;

#[cfg(test)]
mod tests {
    use super::*;

    // Mock for command execution
    mock! {
        CommandRunner {
            fn run(&self, cmd: &str, args: &[&str]) -> Result<String, String>;
            fn run_interactive(&self, cmd: &str, args: &[&str]) -> Result<(), String>;
        }
    }

    struct TestSetup {
        temp_dir: TempDir,
        home_dir: PathBuf,
        config_dir: PathBuf,
    }

    impl TestSetup {
        fn new() -> Self {
            let temp_dir = TempDir::new().unwrap();
            let home_dir = temp_dir.path().join("home");
            let config_dir = home_dir.join(".config/ruv-swarm");
            
            fs::create_dir_all(&config_dir).unwrap();
            
            TestSetup {
                temp_dir,
                home_dir,
                config_dir,
            }
        }

        fn installer(&self) -> Installer {
            Installer::new()
                .with_home_dir(self.home_dir.clone())
                .with_config_dir(self.config_dir.clone())
        }
    }

    #[test]
    fn test_install_fresh_system() {
        let setup = TestSetup::new();
        let mut installer = setup.installer();
        
        let config = InstallConfig {
            install_mcp: true,
            install_hooks: true,
            auto_accept: false,
            force_reinstall: false,
        };
        
        let result = installer.install(config);
        
        assert!(matches!(result, Ok(InstallResult::Success { .. })));
        
        // Verify files were created
        assert!(setup.config_dir.join("config.toml").exists());
        assert!(setup.config_dir.join("hooks").exists());
    }

    #[test]
    fn test_install_detect_existing_installation() {
        let setup = TestSetup::new();
        
        // Create existing installation
        fs::write(setup.config_dir.join("config.toml"), "# Existing config").unwrap();
        
        let mut installer = setup.installer();
        let config = InstallConfig {
            install_mcp: true,
            install_hooks: true,
            auto_accept: false,
            force_reinstall: false,
        };
        
        let result = installer.install(config);
        
        assert!(matches!(result, Ok(InstallResult::AlreadyInstalled { .. })));
    }

    #[test]
    fn test_install_force_reinstall() {
        let setup = TestSetup::new();
        
        // Create existing installation
        fs::write(setup.config_dir.join("config.toml"), "# Old config").unwrap();
        let old_content = fs::read_to_string(setup.config_dir.join("config.toml")).unwrap();
        
        let mut installer = setup.installer();
        let config = InstallConfig {
            install_mcp: true,
            install_hooks: true,
            auto_accept: true,
            force_reinstall: true,
        };
        
        let result = installer.install(config);
        
        assert!(matches!(result, Ok(InstallResult::Success { .. })));
        
        // Verify config was replaced
        let new_content = fs::read_to_string(setup.config_dir.join("config.toml")).unwrap();
        assert_ne!(old_content, new_content);
    }

    #[test]
    fn test_install_npm_package() {
        let setup = TestSetup::new();
        let mut cmd_runner = MockCommandRunner::new();
        
        // Mock npm install
        cmd_runner
            .expect_run()
            .with(eq("npm"), eq(&["install", "-g", "ruv-swarm"][..]))
            .times(1)
            .returning(|_, _| Ok("added 1 package".to_string()));
        
        // Mock npm list to verify installation
        cmd_runner
            .expect_run()
            .with(eq("npm"), eq(&["list", "-g", "ruv-swarm"][..]))
            .times(1)
            .returning(|_, _| Ok("ruv-swarm@0.2.0".to_string()));
        
        let mut installer = setup.installer()
            .with_command_runner(Box::new(cmd_runner));
        
        let result = installer.install_npm_package();
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_install_handle_npm_errors() {
        let setup = TestSetup::new();
        let mut cmd_runner = MockCommandRunner::new();
        
        // Mock npm install failure
        cmd_runner
            .expect_run()
            .with(eq("npm"), eq(&["install", "-g", "ruv-swarm"][..]))
            .times(1)
            .returning(|_, _| Err("npm ERR! permission denied".to_string()));
        
        let mut installer = setup.installer()
            .with_command_runner(Box::new(cmd_runner));
        
        let result = installer.install_npm_package();
        
        assert!(matches!(result, Err(InstallError::NpmError(_))));
    }

    #[test]
    fn test_install_create_config_files() {
        let setup = TestSetup::new();
        let installer = setup.installer();
        
        let result = installer.create_config_files();
        
        assert!(result.is_ok());
        
        // Verify all config files exist
        assert!(setup.config_dir.join("config.toml").exists());
        assert!(setup.config_dir.join("mcp.json").exists());
        assert!(setup.config_dir.join("hooks/pre-task.sh").exists());
        assert!(setup.config_dir.join("hooks/post-edit.sh").exists());
        
        // Verify config content
        let config_content = fs::read_to_string(setup.config_dir.join("config.toml")).unwrap();
        assert!(config_content.contains("[swarm]"));
        assert!(config_content.contains("default_topology"));
    }

    #[test]
    fn test_install_platform_specific_paths() {
        let setup = TestSetup::new();
        let installer = setup.installer();
        
        let paths = installer.get_platform_paths();
        
        #[cfg(target_os = "windows")]
        {
            assert!(paths.config_dir.to_str().unwrap().contains("AppData"));
            assert_eq!(paths.shell_rc, None);
        }
        
        #[cfg(target_os = "macos")]
        {
            assert!(paths.shell_rc.unwrap().to_str().unwrap().contains(".zshrc"));
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(paths.shell_rc.unwrap().to_str().unwrap().contains(".bashrc"));
        }
    }

    #[test]
    fn test_install_add_to_path() {
        let setup = TestSetup::new();
        let shell_rc = setup.home_dir.join(".bashrc");
        fs::write(&shell_rc, "# Existing bashrc\n").unwrap();
        
        let installer = setup.installer()
            .with_shell_rc(shell_rc.clone());
        
        let result = installer.add_to_path();
        
        assert!(result.is_ok());
        
        let content = fs::read_to_string(&shell_rc).unwrap();
        assert!(content.contains("ruv-swarm"));
        assert!(content.contains("PATH"));
    }

    #[test]
    fn test_install_verify_installation() {
        let setup = TestSetup::new();
        let mut cmd_runner = MockCommandRunner::new();
        
        // Mock ruv-swarm version check
        cmd_runner
            .expect_run()
            .with(eq("ruv-swarm"), eq(&["--version"][..]))
            .times(1)
            .returning(|_, _| Ok("ruv-swarm 0.2.0".to_string()));
        
        // Mock MCP server test
        cmd_runner
            .expect_run()
            .with(eq("ruv-swarm"), eq(&["mcp", "test"][..]))
            .times(1)
            .returning(|_, _| Ok("MCP server running".to_string()));
        
        let installer = setup.installer()
            .with_command_runner(Box::new(cmd_runner));
        
        let result = installer.verify_installation();
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_install_rollback_on_error() {
        let setup = TestSetup::new();
        let mut installer = setup.installer();
        
        // Start installation
        installer.begin_transaction();
        
        // Create some files
        fs::write(setup.config_dir.join("test.txt"), "test").unwrap();
        installer.track_file(setup.config_dir.join("test.txt"));
        
        // Simulate error and rollback
        let result = installer.rollback();
        
        assert!(result.is_ok());
        assert!(!setup.config_dir.join("test.txt").exists());
    }

    #[test]
    fn test_install_interactive_prompts() {
        let setup = TestSetup::new();
        let mut installer = setup.installer();
        
        // Mock user input
        installer.set_input_handler(|prompt| {
            match prompt {
                "Install MCP server? (y/n)" => "y".to_string(),
                "Install Claude hooks? (y/n)" => "n".to_string(),
                _ => "".to_string(),
            }
        });
        
        let config = installer.prompt_install_config();
        
        assert!(config.install_mcp);
        assert!(!config.install_hooks);
    }

    #[test]
    fn test_install_auto_accept_flag() {
        let setup = TestSetup::new();
        let mut installer = setup.installer();
        
        let config = InstallConfig {
            install_mcp: true,
            install_hooks: true,
            auto_accept: true,
            force_reinstall: false,
        };
        
        // Should not prompt when auto_accept is true
        let result = installer.install(config);
        
        assert!(matches!(result, Ok(_)));
        assert_eq!(installer.get_prompt_count(), 0);
    }

    #[test]
    fn test_install_permission_check() {
        #[cfg(unix)]
        {
            let setup = TestSetup::new();
            let installer = setup.installer();
            
            // Create directory with no write permission
            let readonly_dir = setup.temp_dir.path().join("readonly");
            fs::create_dir(&readonly_dir).unwrap();
            
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&readonly_dir).unwrap().permissions();
            perms.set_mode(0o555);
            fs::set_permissions(&readonly_dir, perms).unwrap();
            
            let result = installer
                .with_config_dir(readonly_dir.join(".config"))
                .check_permissions();
            
            assert!(matches!(result, Err(InstallError::PermissionDenied(_))));
        }
    }

    #[test]
    fn test_install_cleanup_on_success() {
        let setup = TestSetup::new();
        let installer = setup.installer();
        
        // Create temp files during installation
        let temp_file = setup.temp_dir.path().join("install.tmp");
        fs::write(&temp_file, "temporary").unwrap();
        
        installer.track_temp_file(temp_file.clone());
        let result = installer.cleanup();
        
        assert!(result.is_ok());
        assert!(!temp_file.exists());
    }
}