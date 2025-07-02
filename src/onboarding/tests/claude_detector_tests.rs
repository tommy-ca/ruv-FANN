use super::super::claude_detector::{ClaudeDetector, ClaudeInfo};
use std::path::PathBuf;
use mockall::automock;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_claude_not_found() {
        let detector = ClaudeDetector::new();
        let result = detector.detect();
        
        match result {
            Ok(None) => {
                // Expected when Claude Code is not installed
            }
            Ok(Some(_)) => {
                // Claude Code is actually installed on this system
            }
            Err(_) => panic!("Detection should not fail"),
        }
    }

    #[test]
    fn test_claude_detection_common_paths() {
        let temp_dir = TempDir::new().unwrap();
        let detector = ClaudeDetector::with_search_paths(vec![
            temp_dir.path().to_path_buf(),
        ]);

        // Create mock Claude Code binary
        let claude_path = temp_dir.path().join("claude-code");
        fs::write(&claude_path, "#!/bin/bash\necho '1.0.0'").unwrap();
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&claude_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&claude_path, perms).unwrap();
        }

        let result = detector.detect().unwrap();
        assert!(result.is_some());
        
        if let Some(info) = result {
            assert_eq!(info.path, claude_path);
            assert!(info.is_executable);
        }
    }

    #[test]
    fn test_version_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let detector = ClaudeDetector::with_search_paths(vec![
            temp_dir.path().to_path_buf(),
        ]);

        // Create mock Claude Code binary that outputs version
        let claude_path = temp_dir.path().join("claude-code");
        fs::write(&claude_path, "#!/bin/bash\necho 'Claude Code 1.2.3'").unwrap();
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&claude_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&claude_path, perms).unwrap();
        }

        let info = detector.extract_version(&claude_path).unwrap();
        assert_eq!(info.version, Some("1.2.3".to_string()));
    }

    #[test] 
    fn test_platform_specific_paths() {
        let detector = ClaudeDetector::new();
        let paths = detector.get_platform_search_paths();
        
        #[cfg(target_os = "windows")]
        {
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("Program Files")));
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("AppData")));
        }
        
        #[cfg(target_os = "macos")]
        {
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("Applications")));
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("/usr/local/bin")));
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("/usr/local/bin")));
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains("/usr/bin")));
            assert!(paths.iter().any(|p| p.to_str().unwrap().contains(".local/bin")));
        }
    }

    #[test]
    fn test_permission_check() {
        let temp_dir = TempDir::new().unwrap();
        let detector = ClaudeDetector::new();

        // Create file without execute permission
        let claude_path = temp_dir.path().join("claude-code");
        fs::write(&claude_path, "binary content").unwrap();
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&claude_path).unwrap().permissions();
            perms.set_mode(0o644);
            fs::set_permissions(&claude_path, perms).unwrap();
        }

        let info = detector.check_binary(&claude_path).unwrap();
        
        #[cfg(unix)]
        assert!(!info.is_executable);
        
        #[cfg(windows)]
        assert!(info.is_executable); // Windows determines by extension
    }

    #[test]
    fn test_which_command_fallback() {
        let detector = ClaudeDetector::new();
        
        // This test will use actual `which` command if available
        let result = detector.try_which_command();
        
        // We can't guarantee claude-code exists, but the method shouldn't panic
        match result {
            Ok(Some(path)) => {
                assert!(path.exists());
            }
            Ok(None) => {
                // Claude Code not found via which - expected
            }
            Err(_) => {
                // which command not available - also acceptable
            }
        }
    }
}

// Mock trait for testing detector with injected behavior
#[automock]
trait SystemCommands {
    fn which(&self, binary: &str) -> Result<Option<PathBuf>, std::io::Error>;
    fn check_permissions(&self, path: &PathBuf) -> Result<bool, std::io::Error>;
}