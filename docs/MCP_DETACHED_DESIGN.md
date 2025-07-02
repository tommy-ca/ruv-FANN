# MCP Detached Server Design

## Overview
This document outlines the design for running the ruv-swarm MCP server in detached mode, allowing it to run as a background process while providing management capabilities.

## Requirements
1. Start MCP server in detached/background mode
2. Stop the detached MCP server gracefully
3. Ensure only one MCP instance runs per environment
4. Provide status checking for the running server
5. Automatic cleanup on unexpected termination

## Architecture

### Components

#### 1. Process Manager (`process-manager.js`)
- Handles spawning child processes in detached mode
- Manages PID files for tracking processes
- Implements graceful shutdown mechanisms
- Provides process status checking

#### 2. MCP Server Wrapper (`mcp-detached.js`)
- Wraps the existing MCP server implementation
- Adds process management hooks
- Handles signals for graceful shutdown
- Implements health checking endpoint

#### 3. CLI Commands Enhancement
- `ruv-swarm mcp start --detached` - Start in background
- `ruv-swarm mcp stop` - Stop the server
- `ruv-swarm mcp status` - Check server status
- `ruv-swarm mcp restart` - Restart the server

#### 4. PID File Management
- Location: `~/.ruv-swarm/mcp.pid`
- Contents: Process ID and start timestamp
- Lock file: `~/.ruv-swarm/mcp.lock` for singleton enforcement

### Flow Diagrams

#### Start Flow
```
1. User runs: ruv-swarm mcp start --detached
2. Check for existing PID file
   - If exists and process running: Error "Server already running"
   - If exists but process dead: Clean up and continue
3. Create lock file (atomic operation)
4. Spawn MCP server in detached mode
5. Write PID to file
6. Release lock
7. Return success with PID
```

#### Stop Flow
```
1. User runs: ruv-swarm mcp stop
2. Read PID file
   - If not exists: Error "No server running"
3. Send SIGTERM to process
4. Wait for graceful shutdown (max 10s)
5. If still running, send SIGKILL
6. Clean up PID file
7. Return success
```

### Error Handling
- Stale PID files: Detect and clean up
- Port conflicts: Check port availability before starting
- Permission issues: Proper error messages
- Unexpected termination: Clean up resources

### Testing Strategy
1. Unit tests for process manager functions
2. Integration tests for start/stop cycles
3. Edge case tests (multiple start attempts, missing PID files)
4. Signal handling tests
5. Resource cleanup tests

## Implementation Plan
1. Create test suite (TDD approach)
2. Implement process manager module
3. Create MCP server wrapper
4. Enhance CLI commands
5. Add documentation
6. Integration testing
7. Error handling refinement