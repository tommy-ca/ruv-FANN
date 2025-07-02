//! Tests for error handling and recovery in onboarding
//! 
//! This module tests various error scenarios and recovery mechanisms
//! to ensure robust onboarding experience.

use ruv_swarm::onboarding::{
    OnboardingError, ErrorRecovery, RecoveryStrategy, OnboardingContext
};
use std::fs;
use std::io;
use std::path::PathBuf;
use tempfile::TempDir;
use mockall::predicate::*;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSetup {
        temp_dir: TempDir,
        context: OnboardingContext,
    }

    impl TestSetup {
        fn new() -> Self {
            let temp_dir = TempDir::new().unwrap();
            let context = OnboardingContext {
                home_dir: temp_dir.path().to_path_buf(),
                config_dir: temp_dir.path().join(".config/ruv-swarm"),
                state_file: temp_dir.path().join(".ruv-swarm-state.json"),
                log_file: temp_dir.path().join(".ruv-swarm-install.log"),
            };
            
            TestSetup { temp_dir, context }
        }

        fn recovery(&self) -> ErrorRecovery {
            ErrorRecovery::new(self.context.clone())
        }
    }

    #[test]
    fn test_handle_permission_denied() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        let error = OnboardingError::PermissionDenied {
            path: PathBuf::from("/usr/local/bin"),
            operation: "write".to_string(),
        };
        
        let strategy = recovery.suggest_recovery(&error);
        
        assert!(matches!(strategy, RecoveryStrategy::UseSudo));
        assert!(strategy.get_instructions().contains("sudo"));
    }

    #[test]
    fn test_handle_network_error() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        let error = OnboardingError::NetworkError {
            url: "https://registry.npmjs.org".to_string(),
            cause: io::Error::new(io::ErrorKind::TimedOut, "connection timeout"),
        };
        
        let strategy = recovery.suggest_recovery(&error);
        
        assert!(matches!(strategy, RecoveryStrategy::Retry { .. }));
        
        // Test retry with backoff
        let mut retry_count = 0;
        let result = recovery.retry_with_backoff(3, || {
            retry_count += 1;
            if retry_count < 3 {
                Err(error.clone())
            } else {
                Ok(())
            }
        });
        
        assert!(result.is_ok());
        assert_eq!(retry_count, 3);
    }

    #[test]
    fn test_handle_claude_not_found() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        let error = OnboardingError::ClaudeNotFound;
        
        let strategy = recovery.suggest_recovery(&error);
        
        assert!(matches!(strategy, RecoveryStrategy::InstallClaude { .. }));
        
        let instructions = strategy.get_instructions();
        assert!(instructions.contains("install Claude Code"));
        assert!(instructions.contains("https://"));
    }

    #[test]
    fn test_handle_corrupted_config() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        // Create corrupted config
        let config_path = setup.context.config_dir.join("config.toml");
        fs::create_dir_all(&setup.context.config_dir).unwrap();
        fs::write(&config_path, "invalid [toml content").unwrap();
        
        let error = OnboardingError::ConfigCorrupted {
            path: config_path.clone(),
            details: "invalid TOML".to_string(),
        };
        
        let strategy = recovery.suggest_recovery(&error);
        
        assert!(matches!(strategy, RecoveryStrategy::BackupAndRegenerate { .. }));
        
        // Execute recovery
        let result = recovery.execute_recovery(strategy);
        
        assert!(result.is_ok());
        assert!(config_path.with_extension("toml.backup").exists());
    }

    #[test]
    fn test_handle_disk_space_error() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        let error = OnboardingError::InsufficientDiskSpace {
            required: 100 * 1024 * 1024, // 100MB
            available: 50 * 1024 * 1024, // 50MB
        };
        
        let strategy = recovery.suggest_recovery(&error);
        
        assert!(matches!(strategy, RecoveryStrategy::CleanupSpace { .. }));
        
        let instructions = strategy.get_instructions();
        assert!(instructions.contains("50 MB"));
        assert!(instructions.contains("cleanup"));
    }

    #[test]
    fn test_handle_dependency_conflict() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        let error = OnboardingError::DependencyConflict {
            package: "ruv-swarm".to_string(),
            required_version: ">=0.2.0".to_string(),
            installed_version: Some("0.1.0".to_string()),
        };
        
        let strategy = recovery.suggest_recovery(&error);
        
        assert!(matches!(strategy, RecoveryStrategy::UpdateDependency { .. }));
        
        let instructions = strategy.get_instructions();
        assert!(instructions.contains("npm update"));
        assert!(instructions.contains("0.1.0"));
        assert!(instructions.contains("0.2.0"));
    }

    #[test]
    fn test_rollback_on_critical_error() {
        let setup = TestSetup::new();
        let mut recovery = setup.recovery();
        
        // Track changes
        let file1 = setup.context.config_dir.join("file1.txt");
        let file2 = setup.context.config_dir.join("file2.txt");
        
        fs::create_dir_all(&setup.context.config_dir).unwrap();
        fs::write(&file1, "content1").unwrap();
        fs::write(&file2, "content2").unwrap();
        
        recovery.track_change(file1.clone());
        recovery.track_change(file2.clone());
        
        // Simulate critical error
        let error = OnboardingError::Critical {
            message: "Unrecoverable error".to_string(),
        };
        
        let result = recovery.handle_critical_error(error);
        
        assert!(result.is_err());
        assert!(!file1.exists());
        assert!(!file2.exists());
    }

    #[test]
    fn test_error_logging() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        let errors = vec![
            OnboardingError::ClaudeNotFound,
            OnboardingError::NetworkError {
                url: "test.com".to_string(),
                cause: io::Error::new(io::ErrorKind::Other, "test"),
            },
            OnboardingError::PermissionDenied {
                path: PathBuf::from("/test"),
                operation: "read".to_string(),
            },
        ];
        
        for error in &errors {
            recovery.log_error(error);
        }
        
        assert!(setup.context.log_file.exists());
        
        let log_content = fs::read_to_string(&setup.context.log_file).unwrap();
        assert!(log_content.contains("ClaudeNotFound"));
        assert!(log_content.contains("NetworkError"));
        assert!(log_content.contains("PermissionDenied"));
    }

    #[test]
    fn test_recovery_state_persistence() {
        let setup = TestSetup::new();
        let mut recovery = setup.recovery();
        
        // Set recovery state
        recovery.set_recovery_point("npm_install");
        recovery.add_completed_step("claude_detection");
        recovery.add_completed_step("config_creation");
        
        // Save state
        recovery.save_state().unwrap();
        
        // Load in new instance
        let recovery2 = setup.recovery();
        let state = recovery2.load_state().unwrap();
        
        assert_eq!(state.recovery_point, Some("npm_install".to_string()));
        assert_eq!(state.completed_steps.len(), 2);
        assert!(state.completed_steps.contains(&"claude_detection".to_string()));
    }

    #[test]
    fn test_automatic_recovery_suggestions() {
        let setup = TestSetup::new();
        let recovery = setup.recovery();
        
        // Test various error patterns
        let test_cases = vec![
            (
                OnboardingError::CommandFailed {
                    command: "npm install".to_string(),
                    exit_code: Some(1),
                    stderr: "EACCES: permission denied".to_string(),
                },
                RecoveryStrategy::UseSudo,
            ),
            (
                OnboardingError::CommandFailed {
                    command: "npm install".to_string(),
                    exit_code: Some(1),
                    stderr: "ENOTFOUND registry.npmjs.org".to_string(),
                },
                RecoveryStrategy::CheckNetwork,
            ),
            (
                OnboardingError::CommandFailed {
                    command: "git clone".to_string(),
                    exit_code: Some(128),
                    stderr: "Permission denied (publickey)".to_string(),
                },
                RecoveryStrategy::ConfigureSSH,
            ),
        ];
        
        for (error, expected_strategy) in test_cases {
            let strategy = recovery.suggest_recovery(&error);
            assert_eq!(
                std::mem::discriminant(&strategy),
                std::mem::discriminant(&expected_strategy)
            );
        }
    }

    #[test]
    fn test_partial_installation_recovery() {
        let setup = TestSetup::new();
        let mut recovery = setup.recovery();
        
        // Simulate partial installation
        recovery.add_completed_step("claude_detection");
        recovery.add_completed_step("npm_install");
        recovery.set_recovery_point("mcp_config");
        
        // Simulate failure
        let error = OnboardingError::ConfigWriteFailed {
            path: setup.context.config_dir.join("mcp.json"),
            cause: io::Error::new(io::ErrorKind::PermissionDenied, "denied"),
        };
        
        let resume_point = recovery.get_resume_point(&error);
        
        assert_eq!(resume_point, "mcp_config");
        
        // Verify we can resume from this point
        let steps_to_skip = recovery.get_completed_steps();
        assert!(steps_to_skip.contains(&"claude_detection".to_string()));
        assert!(steps_to_skip.contains(&"npm_install".to_string()));
    }

    #[test]
    fn test_error_aggregation() {
        let setup = TestSetup::new();
        let mut recovery = setup.recovery();
        
        // Collect multiple errors
        recovery.add_error(OnboardingError::ClaudeNotFound);
        recovery.add_error(OnboardingError::NetworkError {
            url: "test1.com".to_string(),
            cause: io::Error::new(io::ErrorKind::Other, "error1"),
        });
        recovery.add_error(OnboardingError::NetworkError {
            url: "test2.com".to_string(),
            cause: io::Error::new(io::ErrorKind::Other, "error2"),
        });
        
        let error_summary = recovery.get_error_summary();
        
        assert!(error_summary.contains("3 errors"));
        assert!(error_summary.contains("ClaudeNotFound"));
        assert!(error_summary.contains("NetworkError (2 occurrences)"));
    }

    #[test]
    fn test_cleanup_on_abort() {
        let setup = TestSetup::new();
        let mut recovery = setup.recovery();
        
        // Create temporary files
        let temp_files = vec![
            setup.temp_dir.path().join("temp1.tmp"),
            setup.temp_dir.path().join("temp2.tmp"),
            setup.temp_dir.path().join(".download.partial"),
        ];
        
        for file in &temp_files {
            fs::write(file, "temp").unwrap();
            recovery.track_temp_file(file.clone());
        }
        
        // Abort and cleanup
        recovery.abort_and_cleanup();
        
        // Verify all temp files removed
        for file in &temp_files {
            assert!(!file.exists());
        }
    }
}