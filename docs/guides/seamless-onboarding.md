# Seamless Onboarding User Guide

## 🚀 Getting Started with ruv-swarm

The seamless onboarding system makes it incredibly easy to get started with ruv-swarm and Claude Code. In just a few commands, you'll have a fully configured development environment with intelligent multi-agent coordination.

### ⚡ 30-Second Quick Start

For the fastest setup experience:

```bash
# Install and setup in one command
npx ruv-swarm init --launch
```

This will:
- ✅ **Detect Claude Code** - Find existing installation or install automatically
- ✅ **Configure ruv-swarm MCP** - Set up seamless Claude Code integration
- ✅ **Initialize Swarm** - Create your first intelligent agent swarm
- ✅ **Launch Ready** - Start Claude Code with full ruv-swarm integration

### Option 1: Complete Setup
```bash
# Initialize ruv-swarm and launch Claude Code in one command
npx ruv-swarm init --launch

# Or step by step
npx ruv-swarm init
npx ruv-swarm launch
```

### Option 2: Auto-Accept Mode (CI/CD)
```bash
# Skip all prompts and use sensible defaults
npx ruv-swarm init -y --launch
```

That's it! Claude Code will open with ruv-swarm fully configured and ready to use.

## 📋 Step-by-Step Walkthrough

### Scenario 1: First-Time Setup

**You're new to both ruv-swarm and Claude Code**

```bash
npx ruv-swarm init
```

Expected flow:
```
🚀 Welcome to ruv-swarm!

🔍 Checking for Claude Code...
❌ Claude Code not found

? Install Claude Code? (Y/n) Y
⏳ Downloading Claude Code...
✅ Claude Code installed successfully

🔧 Setting up MCP servers...
📝 Configuring ruv-swarm MCP server...
✅ ruv-swarm MCP server configured

? Also install GitHub MCP server for enhanced features? (y/N) n
ℹ️  GitHub MCP can be added later with: ruv-swarm mcp add github

📋 Configuration Summary:
- Claude Code: Installed ✅
- ruv-swarm MCP: Configured ✅
- GitHub MCP: Not installed ⚪

? Initialize swarm with these settings? (Y/n) Y
✨ Swarm initialized successfully!

? Launch Claude Code now? (Y/n) Y
🚀 Launching Claude Code with ruv-swarm integration...
```

### Scenario 2: Existing Claude Code Installation

**You have Claude Code but no MCP configuration**

```bash
npx ruv-swarm init
```

Output:
```
🚀 Welcome to ruv-swarm!

🔍 Checking Claude Code installation...
✅ Found Claude Code v1.2.0

🔧 Setting up MCP servers...
📝 Configuring ruv-swarm MCP server...
✅ ruv-swarm MCP server configured

? Also install GitHub MCP server? (y/N) n
ℹ️  Skipping GitHub MCP server

📋 Configuration Summary:
- Claude Code: Already installed ✅
- ruv-swarm MCP: Configured ✅

✨ Setup complete!
```

### Scenario 3: Upgrade/Reconfigure

**You want to update your configuration**

```bash
npx ruv-swarm init --reconfigure
```

This will:
- Update MCP server configurations
- Verify Claude Code compatibility
- Refresh swarm settings
- Maintain existing preferences

## 🔧 What Gets Configured

### 1. Claude Code Detection & Installation
- **Detection**: Searches common installation paths across platforms
- **Version Check**: Ensures compatibility with ruv-swarm
- **Installation**: Downloads and installs if needed
- **Verification**: Tests installation and permissions

### 2. MCP Server Configuration

Creates `.claude/mcp.json` with ruv-swarm integration:

```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"],
      "env": {
        "SWARM_ID": "${SWARM_ID}",
        "SWARM_TOPOLOGY": "mesh"
      }
    }
  }
}
```

**Optional GitHub MCP** (only if explicitly requested):
```json
{
  "mcpServers": {
    "ruv-swarm": { "..." },
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### 3. Swarm Initialization
- Creates default swarm configuration
- Sets up persistence backend
- Configures optimal topology for your use case
- Initializes agent management system

## ⚙️ Configuration Options

### Environment Variables
```bash
# Customize the onboarding process
export CLAUDE_INSTALL_DIR="/opt/claude"           # Custom install location
export SWARM_TOPOLOGY="hierarchical"              # Default topology
export SWARM_PERSISTENCE="sqlite"                 # Persistence backend
export MCP_CONFIG_DIR="$HOME/.claude"            # MCP config location
```

### Command-Line Flags
```bash
# Skip specific steps
ruv-swarm init --skip-claude-install    # Don't install Claude Code
ruv-swarm init --skip-mcp-setup         # Don't configure MCP servers
ruv-swarm init --skip-swarm-init        # Don't initialize swarm

# Customize installation
ruv-swarm init --claude-version 1.2.0   # Specific Claude Code version
ruv-swarm init --topology mesh          # Set topology
ruv-swarm init --persistence memory     # Set persistence backend

# Non-interactive modes
ruv-swarm init -y                       # Accept all defaults
ruv-swarm init --silent                 # No output except errors
```

## 🎯 Advanced Scenarios

### Corporate Environment
```bash
# Install to user directory (no admin required)
ruv-swarm init --user-install

# Use corporate proxy
ruv-swarm init --proxy http://proxy.corp.com:8080

# Custom configuration
ruv-swarm init --config corporate-config.toml
```

### Development Team Setup
```bash
# Shared team configuration
ruv-swarm init --config team-swarm.toml --shared

# Enable all MCP servers
ruv-swarm init --enable-all-mcp

# Development optimizations
ruv-swarm init --dev-mode
```

### CI/CD Pipeline
```bash
# Completely automated setup
ruv-swarm init -y --silent --skip-launch

# Minimal installation
ruv-swarm init --minimal --headless
```

## 🔍 Verification & Troubleshooting

### Verify Installation
```bash
# Check ruv-swarm status
ruv-swarm status

# Verify Claude Code integration
ruv-swarm mcp status

# Test MCP connection
ruv-swarm mcp test
```

### Common Issues

**Claude Code not found after installation**
```bash
# Add to PATH manually
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Permission denied during installation**
```bash
# Install to user directory
ruv-swarm init --user-install

# Or fix permissions
sudo chown -R $USER:$USER ~/.npm
```

**MCP configuration not working**
```bash
# Regenerate MCP config
ruv-swarm mcp setup --force

# Validate configuration
ruv-swarm mcp validate
```

## 🚀 Next Steps

After successful onboarding:

1. **[First Swarm Guide](first-swarm.md)** - Create your first agent swarm
2. **[MCP Integration](mcp-integration.md)** - Advanced MCP features
3. **[API Reference](../api/core.md)** - Explore the full API
4. **[Examples](../examples/)** - Real-world usage examples

## 📞 Additional MCP Servers

### Adding GitHub MCP Later
```bash
# Add GitHub MCP server
ruv-swarm mcp add github

# Configure with token
ruv-swarm mcp configure github --token $GITHUB_TOKEN
```

### Other Available MCP Servers
```bash
# Filesystem operations
ruv-swarm mcp add filesystem

# Database access
ruv-swarm mcp add database --type postgresql

# Custom MCP server
ruv-swarm mcp add custom --command "node server.js" --port 3001
```

## 🎉 You're Ready!

With seamless onboarding complete, you now have:
- ✅ Claude Code installed and configured
- ✅ ruv-swarm MCP server running
- ✅ Intelligent agent swarm initialized
- ✅ Full integration between all components

Start building amazing things with your new multi-agent development environment! 🚀