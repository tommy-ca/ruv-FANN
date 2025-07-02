# Rust MCP Detached Server Design

## Overview

This document outlines the design for implementing MCP detached server functionality in the Rust crates version of ruv-swarm, matching the capabilities of the npm package implementation.

## Current State Analysis

### Existing Components
- **ruv-swarm-mcp**: WebSocket-based MCP server on port 3000
- **ruv-swarm-cli**: CLI with init, spawn, orchestrate, status, monitor commands
- **No detached/daemon support**: Server runs in foreground only
- **No process management**: No PID tracking or lifecycle control

### Target State
- Full parity with npm package MCP detached functionality
- Cross-platform daemon support (Unix/Windows/macOS)
- Stdio protocol support for Claude Code integration
- Process management with PID files
- Health check endpoint

## Architecture Design

### New Components

#### 1. Process Management Module (`ruv-swarm-process`)
New crate for cross-platform process management:

```rust
// ruv-swarm-process/src/lib.rs
pub struct ProcessManager {
    pid_dir: PathBuf,
    processes: Arc<DashMap<String, ProcessInfo>>,
}

pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub started: DateTime<Utc>,
    pub command: String,
    pub args: Vec<String>,
}

impl ProcessManager {
    pub async fn start(&self, name: &str, config: ProcessConfig) -> Result<ProcessInfo>;
    pub async fn stop(&self, name: &str) -> Result<()>;
    pub async fn status(&self, name: &str) -> Result<ProcessStatus>;
    pub async fn restart(&self, name: &str, config: ProcessConfig) -> Result<ProcessInfo>;
}
```

#### 2. MCP Stdio Server Enhancement
Add stdio protocol support to existing MCP server:

```rust
// ruv-swarm-mcp/src/stdio.rs
pub struct StdioServer {
    orchestrator: Arc<SwarmOrchestrator>,
    reader: FramedRead<Stdin, JsonCodec>,
    writer: FramedWrite<Stdout, JsonCodec>,
}

impl StdioServer {
    pub async fn run(self) -> Result<()>;
}
```

#### 3. Daemon Wrapper (`ruv-swarm-daemon`)
Cross-platform daemon implementation:

```rust
// ruv-swarm-daemon/src/lib.rs
#[cfg(unix)]
pub use unix::daemonize;

#[cfg(windows)]
pub use windows::run_as_service;

pub struct DaemonConfig {
    pub name: String,
    pub pid_file: PathBuf,
    pub working_dir: PathBuf,
    pub stdout: Option<PathBuf>,
    pub stderr: Option<PathBuf>,
}

pub async fn run_daemon<F>(config: DaemonConfig, main: F) -> Result<()>
where
    F: Future<Output = Result<()>> + Send + 'static;
```

#### 4. CLI Enhancement
Add MCP subcommand to ruv-swarm-cli:

```rust
// ruv-swarm-cli/src/main.rs
#[derive(Subcommand, Debug)]
enum Commands {
    // ... existing commands ...
    
    /// Manage MCP (Model Context Protocol) server
    Mcp {
        #[command(subcommand)]
        command: McpCommands,
    },
}

#[derive(Subcommand, Debug)]
enum McpCommands {
    /// Start MCP server
    Start {
        /// Run in detached/background mode
        #[arg(long)]
        detached: bool,
        
        /// Protocol to use (stdio, websocket)
        #[arg(long, default_value = "stdio")]
        protocol: String,
        
        /// Port for health check (detached mode)
        #[arg(long, default_value = "9898")]
        health_port: u16,
    },
    
    /// Stop MCP server
    Stop,
    
    /// Check MCP server status
    Status,
    
    /// Restart MCP server
    Restart,
    
    /// List available MCP tools
    Tools,
}
```

### Implementation Plan

#### Phase 1: Process Management (Week 1)
1. Create `ruv-swarm-process` crate
2. Implement PID file handling with atomic operations
3. Add process lifecycle management
4. Add cross-platform process detection

#### Phase 2: Stdio Protocol (Week 1-2)
1. Add stdio module to `ruv-swarm-mcp`
2. Implement JSON-RPC codec for stdin/stdout
3. Add protocol switching in main.rs
4. Test with Claude Code integration

#### Phase 3: Daemon Support (Week 2)
1. Create `ruv-swarm-daemon` crate
2. Unix implementation using `daemonize` crate
3. Windows service implementation
4. macOS launchd support

#### Phase 4: CLI Integration (Week 2-3)
1. Add MCP subcommand structure
2. Implement start/stop/status/restart commands
3. Add health check client
4. Update help documentation

#### Phase 5: Testing & Documentation (Week 3)
1. Unit tests for each component
2. Integration tests matching npm version
3. Cross-platform testing
4. Documentation updates

## Technical Decisions

### Dependencies
- **Process Management**: `sysinfo`, `nix` (Unix), `windows-service` (Windows)
- **Daemon Support**: `daemonize` (Unix), custom Windows service wrapper
- **PID Files**: `fs2` for file locking, `atomicwrites` for atomic operations
- **Health Check**: Existing `axum` server with health endpoint

### File Locations
- **PID Directory**: `~/.ruv-swarm/` (same as npm version)
- **PID File**: `~/.ruv-swarm/mcp-server.pid`
- **Lock File**: `~/.ruv-swarm/mcp-server.lock`

### Compatibility
- Ensure PID files are compatible between npm and Rust versions
- Same health check port and endpoint
- Same CLI command structure
- Shared configuration approach

## Error Handling

### Process Errors
- `ProcessAlreadyRunning`: When attempting to start duplicate
- `ProcessNotFound`: When stopping non-existent process
- `PidFileCorrupted`: When PID file is invalid
- `InsufficientPermissions`: When lacking daemon privileges

### Recovery Strategies
- Automatic cleanup of stale PID files
- Force kill after graceful timeout
- Retry logic for transient failures
- Clear error messages for users

## Security Considerations

1. **PID File Permissions**: 0600 (user read/write only)
2. **Lock File Handling**: Prevent race conditions
3. **Signal Validation**: Only accept standard signals
4. **Port Binding**: Localhost only for health check
5. **Privilege Dropping**: After daemon fork on Unix

## Performance Targets

- Startup time: < 2 seconds
- Shutdown time: < 1 second (graceful)
- Memory usage: < 50MB for daemon process
- Health check latency: < 10ms

## Cross-Platform Considerations

### Unix/Linux
- Use standard daemon forking
- Handle SIGTERM/SIGINT for graceful shutdown
- PID file in `/var/run` or `~/.ruv-swarm`

### Windows
- Implement as Windows Service
- Use Service Control Manager
- PID file in `%APPDATA%\ruv-swarm`

### macOS
- Support both daemon and launchd
- Handle macOS-specific signals
- PID file in `~/Library/Application Support/ruv-swarm`

## Migration Path

1. Both versions use same PID file location
2. Detect if npm version is running before starting Rust version
3. Provide migration command to switch between versions
4. Document differences and compatibility

## Success Criteria

- [ ] Feature parity with npm implementation
- [ ] All integration tests pass
- [ ] Cross-platform support verified
- [ ] Performance targets met
- [ ] Documentation complete
- [ ] Claude Code integration working