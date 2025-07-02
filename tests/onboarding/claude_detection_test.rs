//! Tests for Claude Code detection functionality
//! 
//! This module tests the ability to detect if Claude Code is installed
//! and available on the system, including various edge cases and platforms.

use ruv_swarm::onboarding::{ClaudeDetector, DetectionResult};
use std::env;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test fixture for mocking system environment
    struct TestEnvironment {
        temp_dir: TempDir,
        original_path: Option<String>,
    }

    impl TestEnvironment {
        fn new() -> Self {
            let temp_dir = TempDir::new().unwrap();
            let original_path = env::var("PATH").ok();
            TestEnvironment {
                temp_dir,
                original_path,
            }
        }

        fn set_path(&self, paths: Vec<&str>) {
            let path_string = paths.join(if cfg!(windows) { ";" } else { ":" });
            env::set_var("PATH", path_string);
        }

        fn create_mock_claude(&self, dir: &str, name: &str) -> PathBuf {
            let claude_dir = self.temp_dir.path().join(dir);
            fs::create_dir_all(&claude_dir).unwrap();
            let claude_path = claude_dir.join(name);
            
            #[cfg(unix)]
            {
                fs::write(&claude_path, "#!/bin/sh\necho 'Claude Code v1.0.0'").unwrap();
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&claude_path).unwrap().permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&claude_path, perms).unwrap();
            }
            
            #[cfg(windows)]
            {
                fs::write(&claude_path, "@echo off\necho Claude Code v1.0.0").unwrap();
            }
            
            claude_path
        }
    }

    impl Drop for TestEnvironment {
        fn drop(&mut self) {
            if let Some(ref path) = self.original_path {
                env::set_var("PATH", path);
            }
        }
    }

    #[test]
    fn test_detect_claude_in_path() {
        let env = TestEnvironment::new();
        let claude_path = env.create_mock_claude("bin", "claude");
        env.set_path(vec![claude_path.parent().unwrap().to_str().unwrap()]);

        let detector = ClaudeDetector::new();
        let result = detector.detect();

        assert!(matches!(result, DetectionResult::Found { .. }));
        if let DetectionResult::Found { path, version } = result {
            assert!(path.exists());
            assert!(version.is_some());
        }
    }

    #[test]
    fn test_detect_claude_not_found() {
        let env = TestEnvironment::new();
        env.set_path(vec!["/nonexistent/path"]);

        let detector = ClaudeDetector::new();
        let result = detector.detect();

        assert!(matches!(result, DetectionResult::NotFound));
    }

    #[test]
    fn test_detect_claude_common_locations() {
        let env = TestEnvironment::new();
        
        // Test common installation locations
        let common_paths = if cfg!(windows) {
            vec![
                "AppData/Local/Programs/claude",
                "Program Files/Claude",
                "claude/bin",
            ]
        } else if cfg!(target_os = "macos") {
            vec![
                "Applications/Claude.app/Contents/MacOS",
                ".local/bin",
                "opt/claude/bin",
            ]
        } else {
            vec![
                ".local/bin",
                "opt/claude/bin",
                "usr/local/bin",
            ]
        };

        for path in common_paths {
            let claude_path = env.create_mock_claude(path, "claude");
            let detector = ClaudeDetector::with_search_paths(vec![claude_path.parent().unwrap().to_path_buf()]);
            let result = detector.detect();
            
            assert!(matches!(result, DetectionResult::Found { .. }));
        }
    }

    #[test]
    fn test_detect_claude_version_parsing() {
        let env = TestEnvironment::new();
        
        // Create claude with specific version output
        let claude_dir = env.temp_dir.path().join("bin");
        fs::create_dir_all(&claude_dir).unwrap();
        let claude_path = claude_dir.join("claude");
        
        #[cfg(unix)]
        {
            fs::write(&claude_path, "#!/bin/sh\necho 'Claude Code v2.1.0-beta'").unwrap();
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&claude_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&claude_path, perms).unwrap();
        }
        
        env.set_path(vec![claude_dir.to_str().unwrap()]);
        
        let detector = ClaudeDetector::new();
        let result = detector.detect();
        
        if let DetectionResult::Found { version, .. } = result {
            assert_eq!(version, Some("2.1.0-beta".to_string()));
        } else {
            panic!("Expected Claude to be found");
        }
    }

    #[test]
    fn test_detect_claude_permission_denied() {
        #[cfg(unix)]
        {
            let env = TestEnvironment::new();
            let claude_path = env.create_mock_claude("bin", "claude");
            
            // Remove execute permission
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&claude_path).unwrap().permissions();
            perms.set_mode(0o644);
            fs::set_permissions(&claude_path, perms).unwrap();
            
            env.set_path(vec![claude_path.parent().unwrap().to_str().unwrap()]);
            
            let detector = ClaudeDetector::new();
            let result = detector.detect();
            
            assert!(matches!(result, DetectionResult::PermissionDenied { .. }));
        }
    }

    #[test]
    fn test_detect_claude_multiple_installations() {
        let env = TestEnvironment::new();
        
        // Create multiple claude installations
        let claude1 = env.create_mock_claude("bin1", "claude");
        let claude2 = env.create_mock_claude("bin2", "claude");
        
        env.set_path(vec![
            claude1.parent().unwrap().to_str().unwrap(),
            claude2.parent().unwrap().to_str().unwrap(),
        ]);
        
        let detector = ClaudeDetector::new();
        let results = detector.detect_all();
        
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| matches!(r, DetectionResult::Found { .. })));
    }

    #[test]
    fn test_detect_claude_with_alias() {
        let env = TestEnvironment::new();
        
        // Create claude with different names
        let names = vec!["claude", "claude-code", "claude.exe"];
        
        for name in names {
            let claude_path = env.create_mock_claude("bin", name);
            env.set_path(vec![claude_path.parent().unwrap().to_str().unwrap()]);
            
            let detector = ClaudeDetector::with_executable_names(vec![name.to_string()]);
            let result = detector.detect();
            
            assert!(matches!(result, DetectionResult::Found { .. }));
        }
    }

    #[test]
    fn test_detect_claude_development_mode() {
        let env = TestEnvironment::new();
        
        // Simulate development environment
        env::set_var("CLAUDE_DEV_MODE", "1");
        env::set_var("CLAUDE_DEV_PATH", env.temp_dir.path().join("dev/claude").to_str().unwrap());
        
        let detector = ClaudeDetector::new();
        let result = detector.detect();
        
        // Should detect development mode
        if let DetectionResult::Found { path, .. } = result {
            assert!(path.to_str().unwrap().contains("dev/claude"));
        } else {
            panic!("Expected Claude to be found in development mode");
        }
        
        env::remove_var("CLAUDE_DEV_MODE");
        env::remove_var("CLAUDE_DEV_PATH");
    }

    #[test]
    #[should_panic(expected = "Claude Code is required")]
    fn test_require_claude_panics_when_not_found() {
        let env = TestEnvironment::new();
        env.set_path(vec!["/nonexistent"]);
        
        let detector = ClaudeDetector::new();
        detector.require(); // Should panic
    }
}