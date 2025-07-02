//! Unix-specific process management

use std::os::unix::process::CommandExt;

use anyhow::{Context, Result};
use nix::sys::signal;
use nix::unistd::Pid;
use tokio::process::{Child, Command};
use tracing::debug;

use crate::{platform::Signal, ProcessConfig, ProcessError};

/// Spawn a detached process on Unix
pub async fn spawn_detached(config: &ProcessConfig) -> Result<Child> {
    debug!("Spawning detached process on Unix: {}", config.command);
    
    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args);
    
    if let Some(ref dir) = config.working_dir {
        cmd.current_dir(dir);
    }
    
    for (key, value) in &config.env {
        cmd.env(key, value);
    }
    
    // Unix-specific: create new process group
    unsafe {
        cmd.pre_exec(|| {
            // Detach from parent process group
            libc::setsid();
            Ok(())
        });
    }
    
    // Redirect stdout/stderr if specified
    if let Some(ref stdout_path) = config.stdout {
        let stdout_file = std::fs::File::create(stdout_path)
            .context("Failed to create stdout file")?;
        cmd.stdout(stdout_file);
    }
    
    if let Some(ref stderr_path) = config.stderr {
        let stderr_file = std::fs::File::create(stderr_path)
            .context("Failed to create stderr file")?;
        cmd.stderr(stderr_file);
    }
    
    let child = cmd.spawn()
        .context("Failed to spawn detached process")?;
    
    Ok(child)
}

/// Send a signal to a process on Unix
pub fn send_signal(pid: u32, signal: Signal) -> Result<()> {
    debug!("Sending signal {:?} to process {}", signal, pid);
    
    let pid = Pid::from_raw(pid as i32);
    
    let unix_signal = match signal {
        Signal::Term => signal::Signal::SIGTERM,
        Signal::Kill => signal::Signal::SIGKILL,
        Signal::Int => signal::Signal::SIGINT,
    };
    
    signal::kill(pid, unix_signal)
        .map_err(|e| ProcessError::Platform(format!("Failed to send signal: {}", e)))?;
    
    Ok(())
}