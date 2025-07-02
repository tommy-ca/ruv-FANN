//! Tests for platform-specific onboarding adaptations
//! 
//! This module tests platform-specific behavior for Windows, macOS, and Linux
//! to ensure consistent experience across operating systems.

use ruv_swarm::onboarding::{PlatformAdapter, Platform, PlatformPaths, ShellType};
use std::env;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSetup {
        temp_dir: TempDir,
        home_dir: PathBuf,
    }

    impl TestSetup {
        fn new() -> Self {
            let temp_dir = TempDir::new().unwrap();
            let home_dir = temp_dir.path().join("home");
            fs::create_dir_all(&home_dir).unwrap();
            
            TestSetup { temp_dir, home_dir }
        }

        fn adapter(&self) -> PlatformAdapter {
            PlatformAdapter::new()
                .with_home_dir(self.home_dir.clone())
        }
    }

    #[test]
    fn test_detect_platform() {
        let adapter = PlatformAdapter::new();
        let platform = adapter.detect_platform();
        
        #[cfg(target_os = "windows")]
        assert!(matches!(platform, Platform::Windows));
        
        #[cfg(target_os = "macos")]
        assert!(matches!(platform, Platform::MacOS));
        
        #[cfg(target_os = "linux")]
        assert!(matches!(platform, Platform::Linux));
    }

    #[test]
    fn test_platform_paths() {
        let setup = TestSetup::new();
        let adapter = setup.adapter();
        
        let paths = adapter.get_platform_paths();
        
        // All platforms should have these
        assert!(paths.home_dir.exists());
        assert!(paths.config_dir.to_str().unwrap().contains("ruv-swarm"));
        
        #[cfg(target_os = "windows")]
        {
            assert!(paths.config_dir.to_str().unwrap().contains("AppData"));
            assert!(paths.data_dir.to_str().unwrap().contains("AppData"));
            assert_eq!(paths.shell_rc, None);
        }
        
        #[cfg(target_os = "macos")]
        {
            assert!(paths.config_dir.to_str().unwrap().contains(".config"));
            assert!(paths.data_dir.to_str().unwrap().contains("Library/Application Support"));
            assert!(paths.shell_rc.is_some());
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(paths.config_dir.to_str().unwrap().contains(".config"));
            assert!(paths.data_dir.to_str().unwrap().contains(".local/share"));
            assert!(paths.shell_rc.is_some());
        }
    }

    #[test]
    fn test_detect_shell() {
        let setup = TestSetup::new();
        let adapter = setup.adapter();
        
        // Test with environment variable
        env::set_var("SHELL", "/bin/zsh");
        let shell = adapter.detect_shell();
        assert!(matches!(shell, ShellType::Zsh));
        
        env::set_var("SHELL", "/bin/bash");
        let shell = adapter.detect_shell();
        assert!(matches!(shell, ShellType::Bash));
        
        env::set_var("SHELL", "/usr/bin/fish");
        let shell = adapter.detect_shell();
        assert!(matches!(shell, ShellType::Fish));
        
        #[cfg(target_os = "windows")]
        {
            env::remove_var("SHELL");
            let shell = adapter.detect_shell();
            assert!(matches!(shell, ShellType::PowerShell) || matches!(shell, ShellType::Cmd));
        }
    }

    #[test]
    fn test_shell_rc_files() {
        let setup = TestSetup::new();
        let adapter = setup.adapter();
        
        let test_cases = vec![
            (ShellType::Bash, ".bashrc"),
            (ShellType::Zsh, ".zshrc"),
            (ShellType::Fish, ".config/fish/config.fish"),
        ];
        
        for (shell_type, expected_file) in test_cases {
            let rc_file = adapter.get_shell_rc_file(shell_type);
            
            if let Some(rc_path) = rc_file {
                assert!(rc_path.to_str().unwrap().contains(expected_file));
            }
        }
        
        // Windows shells don't have RC files
        #[cfg(target_os = "windows")]
        {
            assert!(adapter.get_shell_rc_file(ShellType::PowerShell).is_none());
            assert!(adapter.get_shell_rc_file(ShellType::Cmd).is_none());
        }
    }

    #[test]
    fn test_path_modification() {
        let setup = TestSetup::new();
        let adapter = setup.adapter();
        
        let install_dir = setup.temp_dir.path().join("bin");
        fs::create_dir(&install_dir).unwrap();
        
        // Test PATH modification for different shells
        let shells = vec![
            ShellType::Bash,
            ShellType::Zsh,
            ShellType::Fish,
            ShellType::PowerShell,
        ];
        
        for shell in shells {
            let command = adapter.get_path_modification_command(&install_dir, shell);
            
            match shell {
                ShellType::Bash | ShellType::Zsh => {
                    assert!(command.contains("export PATH"));
                    assert!(command.contains(install_dir.to_str().unwrap()));
                }
                ShellType::Fish => {
                    assert!(command.contains("set -gx PATH"));
                    assert!(command.contains(install_dir.to_str().unwrap()));
                }
                ShellType::PowerShell => {
                    assert!(command.contains("$env:Path"));
                    assert!(command.contains(install_dir.to_str().unwrap()));
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_claude_installation_paths() {
        let adapter = PlatformAdapter::new();
        let search_paths = adapter.get_claude_search_paths();
        
        #[cfg(target_os = "windows")]
        {
            let expected_paths = vec![
                "C:\\Program Files\\Claude",
                "C:\\Program Files (x86)\\Claude",
                "%LOCALAPPDATA%\\Programs\\claude",
                "%APPDATA%\\claude",
            ];
            
            for path in expected_paths {
                let expanded = adapter.expand_path(path);
                assert!(search_paths.iter().any(|p| p.to_str().unwrap().contains("Claude")));
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            assert!(search_paths.iter().any(|p| p.to_str().unwrap().contains("Applications")));
            assert!(search_paths.iter().any(|p| p.to_str().unwrap().contains(".local/bin")));
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(search_paths.iter().any(|p| p.to_str().unwrap().contains("/usr/local/bin")));
            assert!(search_paths.iter().any(|p| p.to_str().unwrap().contains(".local/bin")));
        }
    }

    #[test]
    fn test_executable_extensions() {
        let adapter = PlatformAdapter::new();
        
        let executable_names = adapter.get_executable_names("claude");
        
        #[cfg(target_os = "windows")]
        {
            assert!(executable_names.contains(&"claude.exe".to_string()));
            assert!(executable_names.contains(&"claude.bat".to_string()));
            assert!(executable_names.contains(&"claude.cmd".to_string()));
        }
        
        #[cfg(unix)]
        {
            assert!(executable_names.contains(&"claude".to_string()));
            // No extensions on Unix
            assert!(!executable_names.iter().any(|n| n.contains('.')));
        }
    }

    #[test]
    fn test_permission_requirements() {
        let adapter = PlatformAdapter::new();
        
        let install_paths = vec![
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/opt/ruv-swarm"),
            PathBuf::from("~/.local/bin"),
        ];
        
        for path in install_paths {
            let needs_admin = adapter.requires_admin_permission(&path);
            
            #[cfg(unix)]
            {
                if path.starts_with("/usr") || path.starts_with("/opt") {
                    assert!(needs_admin);
                } else if path.to_str().unwrap().contains("~") {
                    assert!(!needs_admin);
                }
            }
            
            #[cfg(target_os = "windows")]
            {
                if path.to_str().unwrap().contains("Program Files") {
                    assert!(needs_admin);
                }
            }
        }
    }

    #[test]
    fn test_npm_global_directory() {
        let setup = TestSetup::new();
        let adapter = setup.adapter();
        
        let npm_dir = adapter.get_npm_global_directory();
        
        #[cfg(target_os = "windows")]
        {
            assert!(npm_dir.to_str().unwrap().contains("npm"));
        }
        
        #[cfg(unix)]
        {
            // Could be in various locations
            let possible = vec![
                "/usr/local/lib/node_modules",
                "~/.npm-global",
                "~/.local/lib/node_modules",
            ];
            
            let npm_str = npm_dir.to_str().unwrap();
            assert!(possible.iter().any(|p| npm_str.contains("npm") || npm_str.contains("node_modules")));
        }
    }

    #[test]
    fn test_create_desktop_shortcut() {
        let setup = TestSetup::new();
        let adapter = setup.adapter();
        
        let result = adapter.create_desktop_shortcut("Claude with ruv-swarm");
        
        #[cfg(target_os = "windows")]
        {
            if result.is_ok() {
                let desktop = adapter.get_desktop_path();
                let shortcut = desktop.join("Claude with ruv-swarm.lnk");
                // Note: Actual file creation would require Windows APIs
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            if result.is_ok() {
                // Would create an alias or .app bundle
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            if result.is_ok() {
                let desktop = adapter.get_desktop_path();
                let shortcut = desktop.join("claude-ruv-swarm.desktop");
                // Would create .desktop file
            }
        }
    }

    #[test]
    fn test_system_requirements_check() {
        let adapter = PlatformAdapter::new();
        
        let requirements = adapter.check_system_requirements();
        
        // Basic checks that should pass on test systems
        assert!(requirements.has_required_os_version);
        assert!(requirements.has_sufficient_memory);
        assert!(requirements.has_sufficient_disk_space);
        
        // Check specific requirements
        #[cfg(target_os = "windows")]
        {
            assert!(requirements.os_version.contains("Windows"));
        }
        
        #[cfg(target_os = "macos")]
        {
            assert!(requirements.os_version.contains("macOS") || requirements.os_version.contains("Darwin"));
        }
        
        #[cfg(target_os = "linux")]
        {
            assert!(requirements.os_version.contains("Linux"));
        }
    }

    #[test]
    fn test_terminal_emulator_detection() {
        let adapter = PlatformAdapter::new();
        
        let terminal = adapter.detect_terminal_emulator();
        
        // Check for common terminal emulators
        let known_terminals = vec![
            "Terminal", "iTerm", "konsole", "gnome-terminal",
            "xterm", "alacritty", "kitty", "cmd", "powershell",
            "Windows Terminal"
        ];
        
        if let Some(term) = terminal {
            assert!(known_terminals.iter().any(|&t| term.to_lowercase().contains(&t.to_lowercase())));
        }
    }

    #[test]
    fn test_platform_specific_instructions() {
        let adapter = PlatformAdapter::new();
        
        let instructions = adapter.get_post_install_instructions();
        
        #[cfg(target_os = "windows")]
        {
            assert!(instructions.contains("restart") || instructions.contains("new terminal"));
            assert!(instructions.contains("PATH"));
        }
        
        #[cfg(unix)]
        {
            assert!(instructions.contains("source") || instructions.contains("reload"));
            assert!(instructions.contains("shell") || instructions.contains("terminal"));
        }
    }
}