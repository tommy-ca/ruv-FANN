# Seamless Onboarding User Guide

## 🚀 Getting Started with ruv-swarm

The seamless onboarding system makes it incredibly easy to get started with ruv-swarm and Claude Code. In just a few commands, you'll have a fully configured development environment with intelligent multi-agent coordination.

## ✨ What is Seamless Onboarding?

Seamless onboarding is ruv-swarm's intelligent setup system that:

- **Automatically detects** Claude Code on your system
- **Installs Claude Code** if not found (with your permission)
- **Configures MCP servers** for GitHub and ruv-swarm integration
- **Launches Claude Code** with optimal settings
- **Provides recovery guidance** if anything goes wrong

The entire process typically takes less than 40 seconds, even on a fresh system!

## 🎯 Quick Start (30 seconds)

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

### 1. Initialize Your Project

```bash
npx ruv-swarm init
```

This command will:
- ✅ Check for Claude Code installation
- ✅ Offer to install if missing
- ✅ Set up MCP server configuration
- ✅ Create `.claude/mcp.json` configuration file

**Example Output:**
```
🐝 ruv-swarm Onboarding

Checking Claude Code installation...
✅ Found Claude Code v1.2.0 at /usr/local/bin/claude-code

Setting up MCP servers...
? Configure GitHub integration? (Y/n) Y
? Enter GitHub token (or press Enter to skip): ghp_xxxxxxxxxxxx
✅ GitHub token configured

? Configure ruv-swarm MCP server? (Y/n) Y
✅ ruv-swarm MCP configured with ID: swarm-abc123-xyz789

Configuration saved to .claude/mcp.json
? Launch Claude Code now? (Y/n) Y
```

### 2. Launch Claude Code (if not done automatically)

```bash
npx ruv-swarm launch
```

Claude Code will open with:
- ✅ GitHub MCP server connected
- ✅ ruv-swarm MCP server active
- ✅ All tools available for multi-agent coordination

## 🛠️ Common Scenarios

### Scenario 1: Fresh Installation

**You don't have Claude Code installed**

```bash
npx ruv-swarm init
```

Output:
```
🐝 ruv-swarm Onboarding

Checking Claude Code installation...
❌ Claude Code not found

? Install Claude Code? (Y/n) Y
⏳ Downloading Claude Code...
✅ Claude Code installed successfully

Setting up MCP servers...
[... continues with configuration ...]
```

### Scenario 2: Existing Installation

**You have Claude Code but no MCP configuration**

```bash
npx ruv-swarm init
```

Output:
```
🐝 ruv-swarm Onboarding

Checking Claude Code installation...
✅ Found Claude Code v1.2.0

Setting up MCP servers...
[... skips installation, goes to configuration ...]
```

### Scenario 3: Upgrade/Reconfigure

**You want to update your configuration**

```bash
npx ruv-swarm init --force
```

This will:
- ✅ Check for Claude Code updates
- ✅ Regenerate MCP configuration
- ✅ Preserve existing settings where possible

### Scenario 4: Team Setup

**Setting up multiple developers**

```bash
# Each developer runs on their machine
npx ruv-swarm init -y  # Auto-accept defaults

# Or use a shared configuration
npx ruv-swarm init --config team-config.json
```

## 🔧 Configuration Options

### Command Line Flags

```bash
npx ruv-swarm init [options]

Options:
  -y, --yes          Auto-accept all prompts with defaults
  -f, --force        Force reconfiguration even if already set up
  -v, --verbose      Show detailed output
  --launch           Launch Claude Code after setup
  --config <file>    Use custom configuration file
  --no-github        Skip GitHub MCP setup
  --no-swarm         Skip ruv-swarm MCP setup
  --help             Show this help message
```

### Environment Variables

Set these before running to customize behavior:

```bash
# Skip all prompts
export RUV_SWARM_AUTO_ACCEPT=true

# Provide GitHub token automatically
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx

# Custom installation path for Claude Code
export CLAUDE_CODE_PATH=/custom/path

# Preferred swarm topology
export SWARM_TOPOLOGY=hierarchical
```

### Custom Configuration

Create a `ruv-swarm-config.json` file:

```json
{
  "onboarding": {
    "autoAccept": false,
    "verbose": true,
    "defaultTopology": "mesh",
    "maxAgents": 8
  },
  "claudeCode": {
    "version": ">=1.0.0",
    "installPath": "auto"
  },
  "mcpServers": {
    "github": {
      "enabled": true,
      "autoToken": true
    },
    "ruvSwarm": {
      "enabled": true,
      "topology": "mesh"
    }
  }
}
```

Use it with:
```bash
npx ruv-swarm init --config ruv-swarm-config.json
```

## 🔍 What Gets Created

The onboarding process creates the following files and configurations:

### 1. `.claude/mcp.json` - MCP Server Configuration

```json
{
  "$schema": "https://raw.githubusercontent.com/anthropics/claude-mcp/main/schemas/mcp.schema.json",
  "mcpServers": {
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"],
      "env": {
        "SWARM_ID": "swarm-abc123-xyz789",
        "SWARM_TOPOLOGY": "mesh",
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

Note: The GitHub MCP server is optional and can be added manually if needed.

### 2. Environment Variables

If you provided a GitHub token, it's stored securely in your environment.

### 3. Session Configuration

ruv-swarm tracks your sessions and provides:
- Persistent memory across Claude Code sessions
- Performance metrics and optimization
- Automatic error recovery

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Issue: "Claude Code not found"
**Problem**: Claude Code isn't installed or not in PATH

**Solutions**:
1. **Auto-install**: Let ruv-swarm install it for you
   ```bash
   npx ruv-swarm init
   # Choose Y when prompted to install
   ```

2. **Manual install**: Download from [Claude Code website](https://claude.ai/code)

3. **Custom path**: Specify where Claude Code is installed
   ```bash
   export CLAUDE_CODE_PATH=/path/to/claude-code
   npx ruv-swarm init
   ```

#### Issue: "Permission denied during installation"
**Problem**: Insufficient permissions to install in system directories

**Solutions**:
1. **Try user directory**: Run installation as current user
   ```bash
   npx ruv-swarm init --user-install
   ```

2. **Use sudo** (Linux/macOS):
   ```bash
   sudo npx ruv-swarm init
   ```

3. **Run as Administrator** (Windows): Right-click terminal and select "Run as Administrator"

#### Issue: "GitHub token invalid"
**Problem**: Token doesn't have required permissions or is expired

**Solutions**:
1. **Skip GitHub setup**: You can use ruv-swarm without GitHub integration
   ```bash
   npx ruv-swarm init --no-github
   ```

2. **Create new token**: Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
   - Required scopes: `repo`, `read:org`, `read:user`

3. **Update token later**:
   ```bash
   npx ruv-swarm configure --github-token ghp_newtoken
   ```

#### Issue: "Claude Code launch fails"
**Problem**: Claude Code won't start with MCP configuration

**Solutions**:
1. **Check configuration**:
   ```bash
   npx ruv-swarm validate-config
   ```

2. **Launch with debug**:
   ```bash
   npx ruv-swarm launch --debug
   ```

3. **Reset configuration**:
   ```bash
   npx ruv-swarm init --force
   ```

#### Issue: "MCP servers not connecting"
**Problem**: Claude Code can't connect to MCP servers

**Solutions**:
1. **Check server status**:
   ```bash
   npx ruv-swarm status
   ```

2. **Restart MCP servers**:
   ```bash
   npx ruv-swarm mcp restart
   ```

3. **Check logs**:
   ```bash
   npx ruv-swarm logs --mcp
   ```

### Getting Help

If you're still having issues:

1. **Run diagnostics**:
   ```bash
   npx ruv-swarm doctor
   ```

2. **Enable verbose logging**:
   ```bash
   npx ruv-swarm init --verbose
   ```

3. **Check the community**:
   - [GitHub Issues](https://github.com/ruvnet/ruv-swarm/issues)
   - [Discord Community](https://discord.gg/ruv-swarm)
   - [Documentation](https://ruv-swarm.dev/docs)

## 🎓 Next Steps

Once you have ruv-swarm set up:

### 1. Explore Multi-Agent Coordination
```bash
# In Claude Code, use the swarm tools
mcp__ruv-swarm__swarm_init {"topology": "mesh", "maxAgents": 5}
mcp__ruv-swarm__agent_spawn {"type": "researcher"}
mcp__ruv-swarm__task_orchestrate {"task": "analyze this codebase"}
```

### 2. Learn Key Concepts
- **[Swarm Coordination](./swarm-coordination.md)** - How agents work together
- **[Memory System](./memory-system.md)** - Persistent context across sessions
- **[Performance Optimization](./performance-optimization.md)** - Getting the best results

### 3. Advanced Usage
- **[Custom Agents](./custom-agents.md)** - Create specialized agent types
- **[Workflow Automation](./workflow-automation.md)** - Automate development tasks
- **[CI/CD Integration](./ci-cd-integration.md)** - Use in pipelines

## 🏆 Best Practices

### For Individual Developers
- Use default settings for most projects
- Enable GitHub integration for repository access
- Let swarms handle complex tasks while you focus on architecture

### For Teams
- Use shared configuration files for consistency
- Set up environment variables in CI/CD
- Enable session memory for knowledge sharing

### For Organizations
- Create custom MCP servers for internal tools
- Use hierarchical topologies for large projects
- Monitor performance metrics for optimization

## 📊 Performance Benefits

With seamless onboarding, you get immediate access to:

- **84.8% SWE-Bench solve rate** - Industry-leading problem-solving
- **32.3% token reduction** - More efficient Claude usage
- **2.8-4.4x speed improvement** - Faster development cycles
- **Persistent memory** - Context that survives across sessions

## 🔮 What's Next?

The onboarding system continuously improves:

- **Smart defaults** that learn from your preferences
- **Team templates** for consistent setup across organizations
- **IDE integration** for seamless development
- **Cloud sync** for settings across machines

---

**Ready to get started?** Run `npx ruv-swarm init` and experience the future of AI-assisted development!