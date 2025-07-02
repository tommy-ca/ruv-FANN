# Onboarding API Documentation

## Overview

The ruv-swarm onboarding system provides both command-line interfaces and programmatic APIs for seamless integration into development workflows. This document covers all public APIs available for the onboarding functionality.

## Table of Contents

- [Command Line Interface](#command-line-interface)
- [Node.js API](#nodejs-api)
- [Rust API](#rust-api)
- [Configuration Schema](#configuration-schema)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Command Line Interface

### `ruv-swarm init`

Initialize ruv-swarm with automatic Claude Code detection and MCP configuration.

#### Syntax
```bash
ruv-swarm init [options]
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-y, --yes` | Auto-accept all prompts | `false` |
| `-f, --force` | Force reconfiguration | `false` |
| `-v, --verbose` | Enable verbose output | `false` |
| `--launch` | Launch Claude Code after setup | `false` |
| `--config <file>` | Use custom configuration file | `null` |
| `--no-github` | Skip GitHub MCP setup | `false` |
| `--no-swarm` | Skip ruv-swarm MCP setup | `false` |
| `--user-install` | Install in user directory only | `false` |
| `--topology <type>` | Set swarm topology | `"mesh"` |
| `--max-agents <n>` | Set maximum agent count | `5` |

#### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Claude Code not found and installation declined |
| `4` | Configuration error |
| `5` | Permission denied |

#### Examples

```bash
# Basic initialization
ruv-swarm init

# Silent initialization with defaults
ruv-swarm init -y

# Initialize and launch immediately
ruv-swarm init --launch

# Force reconfiguration with custom topology
ruv-swarm init --force --topology hierarchical --max-agents 8

# Initialize without GitHub integration
ruv-swarm init --no-github
```

### `ruv-swarm launch`

Launch Claude Code with ruv-swarm MCP configuration.

#### Syntax
```bash
ruv-swarm launch [options]
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --verbose` | Enable verbose output | `false` |
| `--debug` | Launch with debug mode | `false` |
| `--config <file>` | Use specific MCP config | `.claude/mcp.json` |
| `--session-id <id>` | Resume specific session | `null` |
| `--no-mcp` | Launch without MCP servers | `false` |

#### Examples

```bash
# Standard launch
ruv-swarm launch

# Launch with debug output
ruv-swarm launch --debug

# Launch with custom config
ruv-swarm launch --config ./custom-mcp.json
```

### `ruv-swarm status`

Check status of Claude Code and MCP servers.

#### Syntax
```bash
ruv-swarm status [options]
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--json` | Output in JSON format | `false` |
| `--mcp` | Check MCP server status only | `false` |
| `--claude` | Check Claude Code status only | `false` |

#### Output Format

```
🐝 ruv-swarm Status

Claude Code:
  ✅ Installed: v1.2.0 at /usr/local/bin/claude-code
  ✅ Compatible: Yes
  ✅ Running: PID 12345

MCP Servers:
  ✅ ruv-swarm: Connected (127.0.0.1:8000)
  ✅ github: Connected (127.0.0.1:8001)

Configuration:
  ✅ Config file: .claude/mcp.json (valid)
  ✅ Environment: All variables set
  ✅ Permissions: Read/write access

Session:
  🎯 Active: swarm-abc123-xyz789
  📊 Agents: 3/5 active
  🧠 Memory: 12 items stored
```

### `ruv-swarm doctor`

Diagnose and fix common issues.

#### Syntax
```bash
ruv-swarm doctor [options]
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--fix` | Automatically fix detected issues | `false` |
| `--json` | Output results in JSON format | `false` |

#### Example Output

```
🔍 ruv-swarm Diagnostics

✅ Claude Code installation: OK
✅ MCP configuration: OK
❌ GitHub token: Invalid or missing
⚠️  Swarm memory: Size approaching limit

Recommendations:
  1. Update GitHub token: ruv-swarm configure --github-token <token>
  2. Clear old memory: ruv-swarm memory clean --older-than 7d

Run with --fix to automatically resolve fixable issues.
```

## Node.js API

### Installation

```bash
npm install ruv-swarm
```

### Core Functions

#### `runOnboarding(options)`

Main onboarding function that handles the complete setup process.

```javascript
import { runOnboarding } from 'ruv-swarm/onboarding';

const result = await runOnboarding({
  autoAccept: false,      // Skip prompts
  verbose: true,          // Detailed output
  forceReconfigure: false, // Force reconfiguration
  topology: 'mesh',       // Swarm topology
  maxAgents: 5,           // Maximum agents
  githubIntegration: true, // Enable GitHub MCP
  swarmIntegration: true  // Enable ruv-swarm MCP
});

// Result object
{
  success: boolean,
  claudeInfo: {
    installed: boolean,
    version: string,
    path: string
  },
  mcpConfig: {
    created: boolean,
    path: string,
    servers: string[]
  },
  launched: boolean,
  sessionId: string,
  error?: string
}
```

#### `detectClaudeCode(options)`

Detect Claude Code installation across platforms.

```javascript
import { detectClaudeCode } from 'ruv-swarm/onboarding';

const claudeInfo = await detectClaudeCode({
  searchPaths: ['/custom/path'], // Additional search paths
  timeout: 5000                  // Detection timeout in ms
});

// Returns
{
  installed: boolean,
  version: string,
  path: string,
  compatible: boolean,
  running: boolean,
  pid?: number
}
```

#### `generateMCPConfig(projectPath, config)`

Generate MCP configuration file.

```javascript
import { MCPConfig, generateMCPConfig } from 'ruv-swarm/onboarding';

const config = new MCPConfig();
config.addRuvSwarmMCP('swarm-id-123', 'hierarchical');
config.addGitHubMCP(process.env.GITHUB_TOKEN);

const result = await generateMCPConfig('/path/to/project', config);

// Returns
{
  success: boolean,
  path: string,
  config: object,
  backup?: string,
  error?: string
}
```

#### `launchClaudeCode(options)`

Launch Claude Code with MCP configuration.

```javascript
import { launchClaudeCode } from 'ruv-swarm/onboarding';

const result = await launchClaudeCode({
  mcpConfig: '.claude/mcp.json',
  args: ['--debug'],
  verbose: true,
  timeout: 10000
});

// Returns
{
  success: boolean,
  pid: number,
  sessionId: string,
  mcpServers: string[],
  error?: string
}
```

### Utility Functions

#### `isVersionCompatible(version, requirement)`

Check if Claude Code version meets requirements.

```javascript
import { isVersionCompatible } from 'ruv-swarm/onboarding';

const compatible = isVersionCompatible('1.2.0', '>=1.0.0');
// Returns: true
```

#### `detectGitHubToken()`

Auto-detect GitHub token from environment.

```javascript
import { detectGitHubToken } from 'ruv-swarm/onboarding';

const token = detectGitHubToken();
// Checks GITHUB_TOKEN, GH_TOKEN, etc.
```

#### `validateMCPConfig(configPath)`

Validate MCP configuration file.

```javascript
import { validateMCPConfig } from 'ruv-swarm/onboarding';

const validation = await validateMCPConfig('.claude/mcp.json');

// Returns
{
  valid: boolean,
  errors: string[],
  warnings: string[],
  schema: string
}
```

### Interactive CLI Class

#### `createCLI(options)`

Create interactive command-line interface.

```javascript
import { createCLI } from 'ruv-swarm/onboarding';

const cli = createCLI({
  autoAccept: false,
  verbose: true,
  colorize: true
});

// Welcome message
cli.welcome();

// Prompts
const proceed = await cli.confirm('Continue?', true);
const choice = await cli.choice('Select:', ['A', 'B', 'C']);
const input = await cli.prompt('Enter value:');
const password = await cli.promptPassword('Token:');

// Progress indicators
cli.startSpinner('Working...');
await someAsyncWork();
cli.succeedSpinner('Done!');

// Messages
cli.info('Information message');
cli.success('Success message');
cli.warning('Warning message');
cli.error('Error message');

// Error formatting
cli.formatError(error, [
  'Suggestion 1',
  'Suggestion 2'
]);
```

### Configuration Classes

#### `MCPConfig`

Builder for MCP server configurations.

```javascript
import { MCPConfig } from 'ruv-swarm/onboarding';

const config = new MCPConfig();

// Add servers
config.addRuvSwarmMCP(swarmId, topology, options);
config.addGitHubMCP(token, options);
config.addCustomMCP(name, command, args, env);

// Remove servers
config.removeServer(name);

// Generate configuration
const json = config.toJSON();
const yaml = config.toYAML();

// Validate
const validation = config.validate();
```

#### `OnboardingConfig`

Configuration for onboarding process.

```javascript
import { OnboardingConfig } from 'ruv-swarm/onboarding';

const config = new OnboardingConfig({
  autoAccept: false,
  verbose: true,
  claudeCode: {
    version: '>=1.0.0',
    installPath: 'auto'
  },
  mcpServers: {
    github: { enabled: true },
    ruvSwarm: { enabled: true, topology: 'mesh' }
  }
});

// Load from file
const config = OnboardingConfig.fromFile('config.json');

// Save to file
config.saveToFile('config.json');
```

## Rust API

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruv-swarm = "0.1.0"
```

### Core Structures

#### `OnboardingManager`

Main struct for managing the onboarding process.

```rust
use ruv_swarm::onboarding::{OnboardingManager, OnboardingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OnboardingConfig::default();
    let manager = OnboardingManager::new(config);
    
    let result = manager.run_onboarding().await?;
    
    if result.success {
        println!("Onboarding completed successfully!");
    }
    
    Ok(())
}
```

#### `ClaudeCodeDetector`

Detect and manage Claude Code installation.

```rust
use ruv_swarm::onboarding::ClaudeCodeDetector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let detector = ClaudeCodeDetector::new();
    
    let claude_info = detector.detect().await?;
    
    if claude_info.installed {
        println!("Found Claude Code v{} at {}", 
                 claude_info.version, claude_info.path);
    } else {
        // Install Claude Code
        detector.install(&claude_info).await?;
    }
    
    Ok(())
}
```

#### `MCPConfigurator`

Generate and manage MCP configurations.

```rust
use ruv_swarm::onboarding::{MCPConfigurator, MCPServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut configurator = MCPConfigurator::new();
    
    // Add ruv-swarm MCP server
    let swarm_config = MCPServerConfig::ruv_swarm(
        "swarm-123".to_string(),
        "mesh".to_string()
    );
    configurator.add_server("ruv-swarm", swarm_config);
    
    // Add GitHub MCP server
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        let github_config = MCPServerConfig::github(token);
        configurator.add_server("github", github_config);
    }
    
    // Generate configuration
    let config_path = std::path::Path::new(".claude/mcp.json");
    configurator.write_config(config_path).await?;
    
    Ok(())
}
```

#### `LaunchManager`

Handle Claude Code launching.

```rust
use ruv_swarm::onboarding::LaunchManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let launcher = LaunchManager::new();
    
    let result = launcher.launch_claude_code(
        Some(".claude/mcp.json"),
        vec!["--debug".to_string()],
        true  // verbose
    ).await?;
    
    println!("Claude Code launched with PID: {}", result.pid);
    
    Ok(())
}
```

### Error Types

```rust
use ruv_swarm::onboarding::OnboardingError;

match onboarding_result {
    Err(OnboardingError::ClaudeCodeNotFound) => {
        eprintln!("Claude Code not found. Please install it first.");
    }
    Err(OnboardingError::PermissionDenied(path)) => {
        eprintln!("Permission denied accessing: {}", path);
    }
    Err(OnboardingError::ConfigurationError(msg)) => {
        eprintln!("Configuration error: {}", msg);
    }
    Err(OnboardingError::NetworkError(e)) => {
        eprintln!("Network error during download: {}", e);
    }
    Ok(result) => {
        println!("Onboarding successful!");
    }
}
```

### Configuration Structures

#### `OnboardingConfig`

```rust
use ruv_swarm::onboarding::OnboardingConfig;

let config = OnboardingConfig {
    auto_accept: false,
    verbose: true,
    force_reconfigure: false,
    claude_code_version: ">=1.0.0".to_string(),
    default_topology: "mesh".to_string(),
    max_agents: 5,
    github_integration: true,
    swarm_integration: true,
    installation_path: None, // Auto-detect
    retry_attempts: 3,
    retry_delay: std::time::Duration::from_secs(1),
};

// Or use builder pattern
let config = OnboardingConfig::builder()
    .auto_accept(true)
    .verbose(true)
    .topology("hierarchical")
    .max_agents(8)
    .build();
```

#### `MCPServerConfig`

```rust
use ruv_swarm::onboarding::MCPServerConfig;

// ruv-swarm MCP server
let swarm_config = MCPServerConfig {
    command: "npx".to_string(),
    args: vec!["ruv-swarm".to_string(), "mcp".to_string(), "start".to_string()],
    env: {
        let mut env = std::collections::HashMap::new();
        env.insert("SWARM_ID".to_string(), "swarm-123".to_string());
        env.insert("SWARM_TOPOLOGY".to_string(), "mesh".to_string());
        env
    }
};

// GitHub MCP server
let github_config = MCPServerConfig {
    command: "npx".to_string(),
    args: vec!["-y".to_string(), "@modelcontextprotocol/server-github".to_string()],
    env: {
        let mut env = std::collections::HashMap::new();
        env.insert("GITHUB_TOKEN".to_string(), "${GITHUB_TOKEN}".to_string());
        env
    }
};
```

## Configuration Schema

### MCP Configuration JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "mcpServers": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "Command to execute"
          },
          "args": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Command arguments"
          },
          "env": {
            "type": "object",
            "additionalProperties": { "type": "string" },
            "description": "Environment variables"
          }
        },
        "required": ["command"],
        "additionalProperties": false
      }
    }
  },
  "required": ["mcpServers"],
  "additionalProperties": false
}
```

### Onboarding Configuration Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "onboarding": {
      "type": "object",
      "properties": {
        "autoAccept": { "type": "boolean", "default": false },
        "verbose": { "type": "boolean", "default": false },
        "defaultTopology": { 
          "type": "string", 
          "enum": ["mesh", "hierarchical", "ring", "star"],
          "default": "mesh"
        },
        "maxAgents": { 
          "type": "integer", 
          "minimum": 1, 
          "maximum": 50,
          "default": 5 
        }
      }
    },
    "claudeCode": {
      "type": "object",
      "properties": {
        "version": { "type": "string", "default": ">=1.0.0" },
        "installPath": { "type": "string", "default": "auto" }
      }
    },
    "mcpServers": {
      "type": "object",
      "properties": {
        "github": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": true },
            "autoToken": { "type": "boolean", "default": true }
          }
        },
        "ruvSwarm": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": true },
            "topology": { 
              "type": "string", 
              "enum": ["mesh", "hierarchical", "ring", "star"],
              "default": "mesh"
            }
          }
        }
      }
    }
  }
}
```

## Error Handling

### Error Categories

| Category | Description | Recovery |
|----------|-------------|----------|
| `DetectionError` | Claude Code detection failed | Retry, manual path |
| `InstallationError` | Installation process failed | Check permissions, retry |
| `ConfigurationError` | MCP config generation failed | Validate inputs, retry |
| `LaunchError` | Claude Code launch failed | Check config, restart |
| `NetworkError` | Download/connectivity issues | Retry with backoff |
| `PermissionError` | Insufficient permissions | Elevate, user install |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "CLAUDE_CODE_NOT_FOUND",
    "message": "Claude Code installation not detected",
    "category": "DetectionError",
    "suggestions": [
      "Install Claude Code from https://claude.ai/code",
      "Run 'ruv-swarm init' to auto-install",
      "Set CLAUDE_CODE_PATH environment variable"
    ],
    "details": {
      "searchPaths": ["/usr/local/bin", "/usr/bin", "~/.local/bin"],
      "platform": "linux",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
}
```

## Examples

### Complete Node.js Integration

```javascript
import { 
  runOnboarding, 
  detectClaudeCode, 
  launchClaudeCode 
} from 'ruv-swarm/onboarding';

async function setupDevelopmentEnvironment() {
  try {
    // Check if already set up
    const claudeInfo = await detectClaudeCode();
    
    if (!claudeInfo.installed) {
      console.log('Setting up ruv-swarm for the first time...');
      
      const result = await runOnboarding({
        autoAccept: false,
        verbose: true,
        topology: 'hierarchical',
        maxAgents: 8
      });
      
      if (!result.success) {
        throw new Error(`Onboarding failed: ${result.error}`);
      }
      
      console.log('✅ ruv-swarm setup complete!');
    } else {
      console.log('✅ ruv-swarm already configured, launching...');
      
      const launchResult = await launchClaudeCode({
        verbose: true
      });
      
      if (!launchResult.success) {
        throw new Error(`Launch failed: ${launchResult.error}`);
      }
    }
    
    console.log('🚀 Ready to develop with ruv-swarm!');
    
  } catch (error) {
    console.error('❌ Setup failed:', error.message);
    process.exit(1);
  }
}

setupDevelopmentEnvironment();
```

### Custom MCP Server Setup

```javascript
import { MCPConfig, generateMCPConfig } from 'ruv-swarm/onboarding';

// Create custom MCP configuration
const config = new MCPConfig();

// Add ruv-swarm with custom topology
config.addRuvSwarmMCP('my-project-swarm', 'hierarchical', {
  maxAgents: 10,
  enableNeuralPatterns: true
});

// Add GitHub integration
if (process.env.GITHUB_TOKEN) {
  config.addGitHubMCP(process.env.GITHUB_TOKEN);
}

// Add custom MCP server
config.addCustomMCP('my-tool', 'node', ['./my-mcp-server.js'], {
  'MY_API_KEY': process.env.MY_API_KEY
});

// Generate and save configuration
const result = await generateMCPConfig(process.cwd(), config);

if (result.success) {
  console.log(`✅ MCP configuration saved to ${result.path}`);
} else {
  console.error(`❌ Failed to save configuration: ${result.error}`);
}
```

### Rust Integration Example

```rust
use ruv_swarm::onboarding::{OnboardingManager, OnboardingConfig};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create configuration
    let config = OnboardingConfig::builder()
        .auto_accept(true)  // For CI/CD
        .verbose(true)
        .topology("mesh")
        .max_agents(5)
        .github_integration(true)
        .build();
    
    // Run onboarding
    let manager = OnboardingManager::new(config);
    let result = manager.run_onboarding().await?;
    
    if result.success {
        println!("✅ Onboarding completed successfully!");
        println!("Session ID: {}", result.session_id);
        
        // Launch Claude Code
        if let Some(launcher) = result.launcher {
            let launch_result = launcher.launch().await?;
            println!("🚀 Claude Code launched with PID: {}", launch_result.pid);
        }
    } else {
        eprintln!("❌ Onboarding failed: {}", 
                 result.error.unwrap_or_else(|| "Unknown error".to_string()));
        std::process::exit(1);
    }
    
    Ok(())
}
```

---

This API documentation covers all the public interfaces for the ruv-swarm onboarding system. For more examples and advanced usage, see the [User Guide](../guides/seamless-onboarding.md) and [Implementation Summary](../implementation/onboarding-summary.md).