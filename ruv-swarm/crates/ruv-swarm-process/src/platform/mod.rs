//! Platform-specific process management

use anyhow::Result;
use tokio::process::Child;

use crate::{ProcessConfig, ProcessError};

#[cfg(unix)]
pub mod unix;

#[cfg(windows)]
pub mod windows;

/// Cross-platform signal types
#[derive(Debug, Clone, Copy)]
pub enum Signal {
    Term,
    Kill,
    Int,
}

/// Spawn a detached process (platform-specific)
pub async fn spawn_detached(config: &ProcessConfig) -> Result<Child> {
    #[cfg(unix)]
    {
        unix::spawn_detached(config).await
    }
    
    #[cfg(windows)]
    {
        windows::spawn_detached(config).await
    }
    
    #[cfg(not(any(unix, windows)))]
    {
        Err(ProcessError::Platform("Unsupported platform".to_string()).into())
    }
}

/// Send a signal to a process (platform-specific)
pub fn send_signal(pid: u32, signal: Signal) -> Result<()> {
    #[cfg(unix)]
    {
        unix::send_signal(pid, signal)
    }
    
    #[cfg(windows)]
    {
        windows::send_signal(pid, signal)
    }
    
    #[cfg(not(any(unix, windows)))]
    {
        Err(ProcessError::Platform("Unsupported platform".to_string()).into())
    }
}