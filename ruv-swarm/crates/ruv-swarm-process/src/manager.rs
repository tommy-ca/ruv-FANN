//! Process manager implementation

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use tokio::process::Command;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

use crate::{
    pid::{PidFile, PidInfo},
    platform,
    ProcessConfig, ProcessError, ProcessInfo, ProcessLifecycle, ProcessStatus,
};

/// Process manager for handling process lifecycle
pub struct ProcessManager {
    /// Directory for PID files
    pid_dir: PathBuf,
    
    /// Active processes (in-memory tracking)
    processes: Arc<RwLock<HashMap<String, ProcessInfo>>>,
}

impl ProcessManager {
    /// Create a new process manager
    pub fn new(pid_dir: Option<PathBuf>) -> Result<Self> {
        let pid_dir = pid_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .expect("Could not find home directory")
                .join(".ruv-swarm")
        });
        
        // Ensure PID directory exists
        std::fs::create_dir_all(&pid_dir)
            .context("Failed to create PID directory")?;
        
        Ok(Self {
            pid_dir,
            processes: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Get PID file path for a process
    fn pid_file_path(&self, name: &str) -> PathBuf {
        self.pid_dir.join(format!("{}.pid", name))
    }
    
    /// Get PID file for a process
    fn pid_file(&self, name: &str) -> PidFile {
        PidFile::new(self.pid_file_path(name))
    }
}

#[async_trait]
impl ProcessLifecycle for ProcessManager {
    async fn start(&self, name: &str, config: ProcessConfig) -> Result<ProcessInfo> {
        info!("Starting process '{}': {} {:?}", name, config.command, config.args);
        
        let pid_file = self.pid_file(name);
        
        // Check if already running
        if let Ok(existing) = self.status(name).await {
            if existing.running {
                return Err(ProcessError::AlreadyRunning(
                    name.to_string(),
                    existing.pid.unwrap(),
                ).into());
            }
        }
        
        // Clean up stale PID file
        pid_file.cleanup_if_stale().await?;
        
        // Start the process
        let child = if config.detached {
            // Platform-specific detached process spawning
            platform::spawn_detached(&config).await?
        } else {
            // Regular process spawning
            let mut cmd = Command::new(&config.command);
            cmd.args(&config.args);
            
            if let Some(ref dir) = config.working_dir {
                cmd.current_dir(dir);
            }
            
            for (key, value) in &config.env {
                cmd.env(key, value);
            }
            
            cmd.spawn()
                .context("Failed to spawn process")?
        };
        
        let pid = child.id().ok_or_else(|| {
            ProcessError::StartFailed("Could not get process ID".to_string())
        })?;
        
        let info = ProcessInfo {
            pid,
            name: name.to_string(),
            started: Utc::now(),
            command: config.command.clone(),
            args: config.args.clone(),
        };
        
        // Write PID file
        let pid_info = PidInfo {
            pid,
            name: name.to_string(),
            started: info.started,
            command: config.command,
            args: config.args,
        };
        
        pid_file.write(&pid_info).await?;
        
        // Track in memory
        self.processes.write().await.insert(name.to_string(), info.clone());
        
        info!("Process '{}' started with PID {}", name, pid);
        Ok(info)
    }
    
    async fn stop(&self, name: &str) -> Result<()> {
        info!("Stopping process '{}'", name);
        
        let status = self.status(name).await?;
        if !status.running {
            return Err(ProcessError::NotRunning(name.to_string()).into());
        }
        
        let pid = status.pid.unwrap();
        
        // Try graceful shutdown first
        platform::send_signal(pid, platform::Signal::Term)?;
        
        // Wait for graceful shutdown
        let timeout = Duration::from_secs(10);
        let start = tokio::time::Instant::now();
        
        while start.elapsed() < timeout {
            if !PidFile::is_process_running(pid) {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        
        // Force kill if still running
        if PidFile::is_process_running(pid) {
            warn!("Process {} did not stop gracefully, forcing kill", pid);
            platform::send_signal(pid, platform::Signal::Kill)?;
        }
        
        // Remove PID file
        let pid_file = self.pid_file(name);
        pid_file.remove().await?;
        
        // Remove from memory tracking
        self.processes.write().await.remove(name);
        
        info!("Process '{}' stopped", name);
        Ok(())
    }
    
    async fn status(&self, name: &str) -> Result<ProcessStatus> {
        let pid_file = self.pid_file(name);
        
        if !pid_file.exists() {
            return Ok(ProcessStatus {
                running: false,
                pid: None,
                started: None,
                uptime: None,
                command: None,
                args: None,
            });
        }
        
        match pid_file.read().await {
            Ok(info) => {
                let running = PidFile::is_process_running(info.pid);
                let uptime = if running {
                    Some((Utc::now() - info.started).num_seconds() as u64)
                } else {
                    None
                };
                
                Ok(ProcessStatus {
                    running,
                    pid: Some(info.pid),
                    started: Some(info.started),
                    uptime,
                    command: Some(info.command),
                    args: Some(info.args),
                })
            }
            Err(_) => Ok(ProcessStatus {
                running: false,
                pid: None,
                started: None,
                uptime: None,
                command: None,
                args: None,
            }),
        }
    }
    
    async fn restart(&self, name: &str, config: ProcessConfig) -> Result<ProcessInfo> {
        info!("Restarting process '{}'", name);
        
        // Stop if running
        if let Ok(status) = self.status(name).await {
            if status.running {
                self.stop(name).await?;
            }
        }
        
        // Start with new config
        self.start(name, config).await
    }
    
    async fn list(&self) -> Result<Vec<ProcessInfo>> {
        let mut processes = Vec::new();
        
        // Read all PID files
        let entries = std::fs::read_dir(&self.pid_dir)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("pid") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(status) = self.status(name).await {
                        if status.running {
                            processes.push(ProcessInfo {
                                pid: status.pid.unwrap(),
                                name: name.to_string(),
                                started: status.started.unwrap(),
                                command: status.command.unwrap_or_default(),
                                args: status.args.unwrap_or_default(),
                            });
                        }
                    }
                }
            }
        }
        
        Ok(processes)
    }
}