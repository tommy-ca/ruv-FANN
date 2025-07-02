# Environment Variables Schema

## Overview

This document defines the environment variables used by the ruv-swarm onboarding system and MCP servers.

## Core Environment Variables

### GitHub MCP Server

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `GITHUB_TOKEN` | GitHub personal access token for API authentication | Yes* | - | `ghp_xxxxxxxxxxxx` |
| `GH_TOKEN` | Alternative to GITHUB_TOKEN (checked if GITHUB_TOKEN not found) | Yes* | - | `ghp_xxxxxxxxxxxx` |
| `GITHUB_API_URL` | GitHub API endpoint (for GitHub Enterprise) | No | `https://api.github.com` | `https://github.company.com/api/v3` |

*One of `GITHUB_TOKEN` or `GH_TOKEN` is required for full GitHub MCP functionality

### ruv-swarm MCP Server

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `SWARM_ID` | Unique identifier for the swarm instance | Yes | Auto-generated UUID | `550e8400-e29b-41d4-a716-446655440000` |
| `SWARM_TOPOLOGY` | Swarm topology configuration | Yes | `mesh` | `mesh`, `hierarchical`, `ring`, `star` |
| `SWARM_MAX_AGENTS` | Maximum number of agents in swarm | No | `5` | `10` |
| `SWARM_LOG_LEVEL` | Logging verbosity | No | `info` | `debug`, `info`, `warn`, `error` |
| `SWARM_PERSISTENCE_PATH` | Path for swarm state persistence | No | `.ruv-swarm/` | `/var/lib/ruv-swarm/` |

### Onboarding Process Variables

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `RUV_SWARM_AUTO_ACCEPT` | Auto-accept all prompts (-y flag equivalent) | No | `false` | `true` |
| `RUV_SWARM_CLAUDE_VERSION` | Required Claude Code version | No | `>=1.0.0` | `^1.2.0` |
| `RUV_SWARM_RETRY_ATTEMPTS` | Number of retry attempts for network operations | No | `3` | `5` |
| `RUV_SWARM_RETRY_DELAY` | Delay between retries in milliseconds | No | `1000` | `2000` |
| `RUV_SWARM_LOG_FILE` | Path to log file | No | - | `/var/log/ruv-swarm.log` |

### Platform-Specific Variables

#### Windows

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `LOCALAPPDATA` | Local application data directory | System | - | `C:\Users\Username\AppData\Local` |
| `PROGRAMFILES` | Program files directory | System | - | `C:\Program Files` |

#### macOS

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `HOME` | User home directory | System | - | `/Users/username` |

#### Linux

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `HOME` | User home directory | System | - | `/home/username` |
| `XDG_CONFIG_HOME` | User configuration directory | No | `$HOME/.config` | `/home/username/.config` |
| `XDG_DATA_HOME` | User data directory | No | `$HOME/.local/share` | `/home/username/.local/share` |

## Variable Resolution

### Token Resolution Order

For GitHub authentication, tokens are resolved in the following order:
1. `GITHUB_TOKEN` environment variable
2. `GH_TOKEN` environment variable  
3. Interactive prompt during onboarding
4. Fall back to limited functionality

### Path Resolution

Paths in environment variables support:
- Absolute paths: `/usr/local/bin/claude-code`
- Relative paths: `./bin/claude-code`
- Home directory expansion: `~/bin/claude-code`
- Environment variable expansion: `$HOME/bin/claude-code`

### Variable Substitution in MCP Config

The MCP configuration supports variable substitution using the `${VARIABLE}` syntax:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

Variables are resolved at runtime when Claude Code loads the configuration.

## Security Considerations

### Token Storage

- **Never** commit tokens to version control
- Use `.env` files with `.gitignore` for local development
- Use secure credential managers in production
- Tokens are never logged or displayed in output

### Permission Handling

- Onboarding process runs with user permissions by default
- System installation requires explicit elevation
- File operations check permissions before attempting writes

## Default Values Configuration

Default values can be overridden via a configuration file at:
- Linux/macOS: `~/.config/ruv-swarm/onboarding.toml`
- Windows: `%APPDATA%\ruv-swarm\onboarding.toml`

Example configuration:

```toml
[onboarding]
auto_accept = false
claude_code_version = ">=1.0.0"
default_topology = "mesh"
default_max_agents = 5
retry_attempts = 3
retry_delay_ms = 1000

[paths]
claude_search_paths = [
  "/usr/local/bin",
  "/usr/bin",
  "/opt/homebrew/bin",
  "~/.local/bin",
  "~/bin"
]

[mcp_servers]
github_enabled = true
ruv_swarm_enabled = true
```

## CI/CD Integration

For automated deployments, set these environment variables:

```bash
export RUV_SWARM_AUTO_ACCEPT=true
export GITHUB_TOKEN="your-token-here"
export SWARM_TOPOLOGY="hierarchical"
export SWARM_MAX_AGENTS=10
```

This enables non-interactive installation and configuration.