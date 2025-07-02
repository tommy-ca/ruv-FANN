//! Tests for interactive onboarding flow
//! 
//! This module tests the complete interactive experience including
//! prompts, user input handling, and flow navigation.

use ruv_swarm::onboarding::{InteractiveOnboarding, OnboardingState, UserChoice};
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock stdin/stdout for testing interactive prompts
    struct MockIO {
        input: Arc<Mutex<Vec<u8>>>,
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl MockIO {
        fn new(input_data: &str) -> Self {
            MockIO {
                input: Arc::new(Mutex::new(input_data.as_bytes().to_vec())),
                output: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_output(&self) -> String {
            let output = self.output.lock().unwrap();
            String::from_utf8_lossy(&output).to_string()
        }

        fn stdin(&self) -> MockStdin {
            MockStdin {
                data: Arc::clone(&self.input),
            }
        }

        fn stdout(&self) -> MockStdout {
            MockStdout {
                data: Arc::clone(&self.output),
            }
        }
    }

    struct MockStdin {
        data: Arc<Mutex<Vec<u8>>>,
    }

    impl Read for MockStdin {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let mut data = self.data.lock().unwrap();
            let len = std::cmp::min(buf.len(), data.len());
            buf[..len].copy_from_slice(&data[..len]);
            data.drain(..len);
            Ok(len)
        }
    }

    struct MockStdout {
        data: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for MockStdout {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut data = self.data.lock().unwrap();
            data.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_welcome_screen() {
        let io = MockIO::new("");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        onboarding.show_welcome();
        
        let output = io.get_output();
        assert!(output.contains("Welcome to ruv-swarm"));
        assert!(output.contains("seamless onboarding"));
        assert!(output.contains("Claude Code"));
    }

    #[test]
    fn test_claude_detection_prompt() {
        let io = MockIO::new("y\n");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        let result = onboarding.prompt_claude_detection();
        
        let output = io.get_output();
        assert!(output.contains("Detecting Claude Code"));
        assert!(matches!(result, Ok(UserChoice::Continue)));
    }

    #[test]
    fn test_installation_options_menu() {
        let io = MockIO::new("1\n");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        let result = onboarding.show_installation_menu();
        
        let output = io.get_output();
        assert!(output.contains("Installation Options"));
        assert!(output.contains("1) Full installation"));
        assert!(output.contains("2) MCP server only"));
        assert!(output.contains("3) Claude hooks only"));
        assert!(output.contains("4) Custom installation"));
        
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_progress_indicator() {
        let io = MockIO::new("");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        // Test progress updates
        onboarding.update_progress("Installing NPM package", 25);
        onboarding.update_progress("Configuring MCP server", 50);
        onboarding.update_progress("Setting up hooks", 75);
        onboarding.update_progress("Complete!", 100);
        
        let output = io.get_output();
        assert!(output.contains("Installing NPM package"));
        assert!(output.contains("[=="));
        assert!(output.contains("25%"));
        assert!(output.contains("100%"));
    }

    #[test]
    fn test_error_recovery_prompt() {
        let io = MockIO::new("r\n");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        let error = "Failed to install NPM package: permission denied";
        let result = onboarding.handle_error(error);
        
        let output = io.get_output();
        assert!(output.contains("Error occurred"));
        assert!(output.contains(error));
        assert!(output.contains("(r)etry"));
        assert!(output.contains("(s)kip"));
        assert!(output.contains("(a)bort"));
        
        assert!(matches!(result, Ok(UserChoice::Retry)));
    }

    #[test]
    fn test_configuration_review() {
        let io = MockIO::new("y\n");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        let config = OnboardingState {
            claude_detected: true,
            npm_installed: true,
            mcp_configured: true,
            hooks_installed: true,
            path_updated: false,
        };
        
        let result = onboarding.review_configuration(&config);
        
        let output = io.get_output();
        assert!(output.contains("Configuration Review"));
        assert!(output.contains("✓ Claude Code detected"));
        assert!(output.contains("✓ NPM package installed"));
        assert!(output.contains("✓ MCP server configured"));
        assert!(output.contains("✗ PATH not updated"));
        
        assert!(result.unwrap());
    }

    #[test]
    fn test_auto_accept_mode() {
        let io = MockIO::new("");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf())
            .with_auto_accept(true);
        
        // Should not prompt in auto-accept mode
        let result = onboarding.prompt_claude_detection();
        assert!(matches!(result, Ok(UserChoice::Continue)));
        
        let output = io.get_output();
        assert!(!output.contains("(y/n)"));
    }

    #[test]
    fn test_custom_installation_flow() {
        let io = MockIO::new("n\ny\ny\nn\ny\n");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        let result = onboarding.custom_installation_flow();
        
        let output = io.get_output();
        assert!(output.contains("Install NPM package?"));
        assert!(output.contains("Configure MCP server?"));
        assert!(output.contains("Install Claude hooks?"));
        assert!(output.contains("Add to PATH?"));
        
        assert!(result.is_ok());
        let config = result.unwrap();
        assert!(!config.install_npm);
        assert!(config.install_mcp);
        assert!(config.install_hooks);
        assert!(!config.add_to_path);
    }

    #[test]
    fn test_completion_screen() {
        let io = MockIO::new("");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        onboarding.show_completion(true);
        
        let output = io.get_output();
        assert!(output.contains("✨ Installation Complete!"));
        assert!(output.contains("Launch Claude Code"));
        assert!(output.contains("claude --mcp"));
    }

    #[test]
    fn test_state_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let state_file = temp_dir.path().join(".ruv-swarm-onboarding.json");
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_home_dir(temp_dir.path().to_path_buf());
        
        // Save state
        let state = OnboardingState {
            claude_detected: true,
            npm_installed: false,
            mcp_configured: true,
            hooks_installed: false,
            path_updated: true,
        };
        
        onboarding.save_state(&state).unwrap();
        assert!(state_file.exists());
        
        // Load state
        let loaded_state = onboarding.load_state().unwrap();
        assert_eq!(loaded_state.claude_detected, state.claude_detected);
        assert_eq!(loaded_state.npm_installed, state.npm_installed);
        assert_eq!(loaded_state.mcp_configured, state.mcp_configured);
    }

    #[test]
    fn test_interrupt_handling() {
        let io = MockIO::new("\x03"); // Ctrl+C
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        let result = onboarding.prompt_claude_detection();
        
        assert!(matches!(result, Ok(UserChoice::Abort)));
        
        let output = io.get_output();
        assert!(output.contains("Installation cancelled"));
    }

    #[test]
    fn test_color_output() {
        let io = MockIO::new("");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf())
            .with_color(true);
        
        onboarding.print_success("Installation successful!");
        onboarding.print_error("An error occurred");
        onboarding.print_warning("This might take a while");
        onboarding.print_info("Checking system...");
        
        let output = io.get_output();
        // Check for ANSI color codes
        assert!(output.contains("\x1b[32m")); // Green
        assert!(output.contains("\x1b[31m")); // Red
        assert!(output.contains("\x1b[33m")); // Yellow
        assert!(output.contains("\x1b[34m")); // Blue
    }

    #[test]
    fn test_spinner_animation() {
        let io = MockIO::new("");
        let temp_dir = TempDir::new().unwrap();
        
        let mut onboarding = InteractiveOnboarding::new()
            .with_stdin(Box::new(io.stdin()))
            .with_stdout(Box::new(io.stdout()))
            .with_home_dir(temp_dir.path().to_path_buf());
        
        // Start spinner
        onboarding.start_spinner("Installing...");
        std::thread::sleep(std::time::Duration::from_millis(100));
        onboarding.stop_spinner(true);
        
        let output = io.get_output();
        assert!(output.contains("Installing..."));
        assert!(output.contains("✓"));
    }
}