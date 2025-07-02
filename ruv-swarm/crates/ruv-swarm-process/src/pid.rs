//! PID file management for process tracking

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

#[derive(Debug, Error)]
pub enum PidFileError {
    #[error("PID file already exists for process {0}")]
    AlreadyExists(String),
    
    #[error("PID file not found for process {0}")]
    NotFound(String),
    
    #[error("PID file is corrupted")]
    Corrupted,
    
    #[error("Failed to acquire lock on PID file")]
    LockFailed,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Information stored in PID files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidInfo {
    pub pid: u32,
    pub name: String,
    pub started: DateTime<Utc>,
    pub command: String,
    pub args: Vec<String>,
}

/// PID file manager
pub struct PidFile {
    path: PathBuf,
}

impl PidFile {
    /// Create a new PID file manager for the given path
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
    
    /// Write PID information to file atomically
    pub async fn write(&self, info: &PidInfo) -> Result<(), PidFileError> {
        debug!("Writing PID file: {:?}", self.path);
        
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Serialize to JSON
        let json = serde_json::to_string_pretty(info)
            .context("Failed to serialize PID info")?;
        
        // Write atomically
        atomicwrites::AtomicFile::new(&self.path)
            .write(|f| f.write_all(json.as_bytes()))
            .map_err(|e| PidFileError::Io(e.into()))?;
        
        debug!("PID file written successfully");
        Ok(())
    }
    
    /// Read PID information from file
    pub async fn read(&self) -> Result<PidInfo, PidFileError> {
        debug!("Reading PID file: {:?}", self.path);
        
        let contents = tokio::fs::read_to_string(&self.path)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    PidFileError::NotFound(self.path.display().to_string())
                } else {
                    PidFileError::Io(e)
                }
            })?;
        
        let info: PidInfo = serde_json::from_str(&contents)
            .map_err(|_| PidFileError::Corrupted)?;
        
        Ok(info)
    }
    
    /// Remove PID file
    pub async fn remove(&self) -> Result<(), PidFileError> {
        debug!("Removing PID file: {:?}", self.path);
        
        tokio::fs::remove_file(&self.path)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    PidFileError::NotFound(self.path.display().to_string())
                } else {
                    PidFileError::Io(e)
                }
            })?;
        
        Ok(())
    }
    
    /// Check if PID file exists
    pub fn exists(&self) -> bool {
        self.path.exists()
    }
    
    /// Lock the PID file (for preventing race conditions)
    pub fn lock(&self) -> Result<fs::File, PidFileError> {
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&self.path)?;
        
        file.try_lock_exclusive()
            .map_err(|_| PidFileError::LockFailed)?;
        
        Ok(file)
    }
    
    /// Check if a process with the given PID is running
    pub fn is_process_running(pid: u32) -> bool {
        use sysinfo::{System, SystemExt, ProcessExt, Pid};
        
        let mut system = System::new();
        system.refresh_process(Pid::from(pid as usize));
        system.process(Pid::from(pid as usize)).is_some()
    }
    
    /// Clean up stale PID file if process is not running
    pub async fn cleanup_if_stale(&self) -> Result<bool, PidFileError> {
        if !self.exists() {
            return Ok(false);
        }
        
        match self.read().await {
            Ok(info) => {
                if !Self::is_process_running(info.pid) {
                    warn!("Removing stale PID file for process {} (PID {})", info.name, info.pid);
                    self.remove().await?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Err(PidFileError::Corrupted) => {
                warn!("Removing corrupted PID file");
                self.remove().await?;
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_pid_file_operations() {
        let dir = tempdir().unwrap();
        let pid_file = PidFile::new(dir.path().join("test.pid"));
        
        // Test write
        let info = PidInfo {
            pid: 12345,
            name: "test-process".to_string(),
            started: Utc::now(),
            command: "test".to_string(),
            args: vec!["arg1".to_string(), "arg2".to_string()],
        };
        
        pid_file.write(&info).await.unwrap();
        assert!(pid_file.exists());
        
        // Test read
        let read_info = pid_file.read().await.unwrap();
        assert_eq!(read_info.pid, info.pid);
        assert_eq!(read_info.name, info.name);
        
        // Test remove
        pid_file.remove().await.unwrap();
        assert!(!pid_file.exists());
    }
}