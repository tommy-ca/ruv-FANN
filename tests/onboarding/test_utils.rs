//! Shared test utilities and mock implementations for onboarding tests

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

/// Mock command runner for testing command execution
pub struct MockCommandRunner {
    responses: Arc<Mutex<HashMap<String, CommandResponse>>>,
    history: Arc<Mutex<Vec<ExecutedCommand>>>,
}

#[derive(Clone, Debug)]
pub struct CommandResponse {
    pub stdout: String,
    pub stderr: String,
    pub status: i32,
}

#[derive(Clone, Debug)]
pub struct ExecutedCommand {
    pub program: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

impl MockCommandRunner {
    pub fn new() -> Self {
        MockCommandRunner {
            responses: Arc::new(Mutex::new(HashMap::new())),
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_response(&self, command: &str, response: CommandResponse) {
        self.responses.lock().unwrap().insert(command.to_string(), response);
    }

    pub fn run(&self, program: &str, args: &[&str]) -> Result<Output, std::io::Error> {
        let command_key = format!("{} {}", program, args.join(" "));
        
        // Record execution
        self.history.lock().unwrap().push(ExecutedCommand {
            program: program.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            env: std::env::vars().collect(),
        });

        // Return mock response
        let responses = self.responses.lock().unwrap();
        if let Some(response) = responses.get(&command_key) {
            Ok(Output {
                status: std::process::ExitStatus::from_raw(response.status),
                stdout: response.stdout.as_bytes().to_vec(),
                stderr: response.stderr.as_bytes().to_vec(),
            })
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("No mock response for: {}", command_key),
            ))
        }
    }

    pub fn get_history(&self) -> Vec<ExecutedCommand> {
        self.history.lock().unwrap().clone()
    }

    pub fn was_called(&self, program: &str) -> bool {
        self.history.lock().unwrap()
            .iter()
            .any(|cmd| cmd.program == program)
    }
}

/// Mock file system for testing file operations
pub struct MockFileSystem {
    files: Arc<Mutex<HashMap<PathBuf, Vec<u8>>>>,
    temp_dir: TempDir,
}

impl MockFileSystem {
    pub fn new() -> std::io::Result<Self> {
        Ok(MockFileSystem {
            files: Arc::new(Mutex::new(HashMap::new())),
            temp_dir: TempDir::new()?,
        })
    }

    pub fn base_path(&self) -> &Path {
        self.temp_dir.path()
    }

    pub fn write_file(&self, path: &Path, content: &[u8]) -> std::io::Result<()> {
        let full_path = self.temp_dir.path().join(path);
        
        // Create parent directories
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Write to real file system for integration
        std::fs::write(&full_path, content)?;
        
        // Also track in memory
        self.files.lock().unwrap().insert(path.to_path_buf(), content.to_vec());
        
        Ok(())
    }

    pub fn read_file(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        let full_path = self.temp_dir.path().join(path);
        std::fs::read(full_path)
    }

    pub fn exists(&self, path: &Path) -> bool {
        let full_path = self.temp_dir.path().join(path);
        full_path.exists()
    }

    pub fn create_claude_mock(&self, name: &str) -> std::io::Result<PathBuf> {
        let claude_path = self.temp_dir.path().join("bin").join(name);
        std::fs::create_dir_all(claude_path.parent().unwrap())?;
        
        #[cfg(unix)]
        {
            std::fs::write(&claude_path, "#!/bin/sh\necho 'Claude Code v1.0.0'")?;
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&claude_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&claude_path, perms)?;
        }
        
        #[cfg(windows)]
        {
            std::fs::write(&claude_path, "@echo off\necho Claude Code v1.0.0")?;
        }
        
        Ok(claude_path)
    }
}

/// Mock user input for testing interactive prompts
pub struct MockUserInput {
    responses: Arc<Mutex<Vec<String>>>,
    index: Arc<Mutex<usize>>,
}

impl MockUserInput {
    pub fn new(responses: Vec<&str>) -> Self {
        MockUserInput {
            responses: Arc::new(Mutex::new(
                responses.iter().map(|s| s.to_string()).collect()
            )),
            index: Arc::new(Mutex::new(0)),
        }
    }

    pub fn read_line(&self) -> std::io::Result<String> {
        let mut index = self.index.lock().unwrap();
        let responses = self.responses.lock().unwrap();
        
        if *index < responses.len() {
            let response = responses[*index].clone();
            *index += 1;
            Ok(response)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "No more mock responses",
            ))
        }
    }
}

/// Test data fixtures
pub mod fixtures {
    use serde_json::json;
    
    pub fn valid_mcp_config() -> serde_json::Value {
        json!({
            "servers": {
                "ruv-swarm": {
                    "command": "npx",
                    "args": ["ruv-swarm", "mcp", "start"],
                    "stdio": true
                }
            }
        })
    }
    
    pub fn invalid_mcp_config() -> serde_json::Value {
        json!({
            "servers": {
                "invalid": {
                    // Missing required fields
                    "args": ["test"]
                }
            }
        })
    }
    
    pub fn claude_version_outputs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Claude Code v1.0.0\n", "1.0.0"),
            ("claude version 2.0.0-beta\n", "2.0.0-beta"),
            ("Version: 3.1.0\nBuild: 12345\n", "3.1.0"),
        ]
    }
}

/// Assertion helpers
#[macro_export]
macro_rules! assert_file_contains {
    ($path:expr, $content:expr) => {
        let file_content = std::fs::read_to_string($path)
            .expect(&format!("Failed to read file: {:?}", $path));
        assert!(
            file_content.contains($content),
            "File {:?} does not contain expected content: {}",
            $path,
            $content
        );
    };
}

#[macro_export]
macro_rules! assert_command_called {
    ($runner:expr, $program:expr) => {
        assert!(
            $runner.was_called($program),
            "Expected command '{}' to be called",
            $program
        );
    };
}

#[macro_export]
macro_rules! assert_command_not_called {
    ($runner:expr, $program:expr) => {
        assert!(
            !$runner.was_called($program),
            "Expected command '{}' NOT to be called",
            $program
        );
    };
}