# Getting Started with ruv-swarm

Welcome to ruv-swarm! This guide will get you up and running with the distributed swarm orchestration platform in just a few minutes.

## 🚀 Quick Start (30 seconds)

The fastest way to get started is with our seamless onboarding:

```bash
# Install ruv-swarm
npm install -g ruv-swarm

# Initialize with automatic setup
ruv-swarm init

# Launch Claude Code with MCP integration
ruv-swarm launch
```

That's it! The onboarding will:
- ✅ Detect or install Claude Code automatically
- ✅ Configure MCP servers for GitHub and ruv-swarm
- ✅ Set up your first swarm
- ✅ Launch Claude Code with full integration

## 📦 Installation Options

### NPM (Recommended)
```bash
npm install -g ruv-swarm
```

### Cargo (Rust)
```bash
cargo install ruv-swarm-cli
```

### Binary Download
Download the latest release from [GitHub Releases](https://github.com/ruvnet/ruv-FANN/releases).

## 🎯 Your First Swarm

### 1. Initialize a Swarm
```bash
ruv-swarm init --topology mesh
```

### 2. Spawn Agents
```bash
ruv-swarm spawn researcher --name "data-analyst"
ruv-swarm spawn coder --name "backend-dev"  
ruv-swarm spawn tester --name "qa-engineer"
```

### 3. Orchestrate Tasks
```bash
ruv-swarm orchestrate parallel "Build a REST API with authentication"
```

### 4. Monitor Progress
```bash
ruv-swarm status --detailed
ruv-swarm monitor --watch
```

## 🔧 Configuration

### Environment Variables
```bash
export GITHUB_TOKEN="your_github_token"      # For GitHub MCP integration
export SWARM_TOPOLOGY="mesh"                 # Default topology
export SWARM_PERSISTENCE="sqlite"            # Persistence backend
```

### Configuration File
Create `.ruv-swarm.toml`:
```toml
[swarm]
topology = "mesh"
max_agents = 10

[persistence]
backend = "sqlite"
database = "./swarm.db"

[mcp]
github_enabled = true
ruv_swarm_enabled = true
```

## 🧠 Core Concepts

### Agents
Specialized workers with different capabilities:
- **Researcher**: Data analysis, investigation, research
- **Coder**: Code generation, refactoring, implementation  
- **Analyst**: Code review, optimization, quality analysis
- **Tester**: Test generation, validation, QA
- **Coordinator**: Task orchestration, workflow management

### Topologies
Different coordination patterns:
- **Mesh**: Full peer-to-peer communication
- **Hierarchical**: Tree-like delegation structure
- **Ring**: Circular communication flow
- **Star**: Central coordinator with worker nodes

### Orchestration Strategies
- **Parallel**: Tasks executed simultaneously
- **Sequential**: Tasks executed in order
- **Adaptive**: Dynamic strategy based on task complexity

## 🔗 MCP Integration

ruv-swarm integrates with Claude Code via Model Context Protocol (MCP):

### GitHub MCP Server
```json
{
  "github": {
    "command": "npx",
    "args": ["@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_TOKEN": "${GITHUB_TOKEN}"
    }
  }
}
```

### ruv-swarm MCP Server  
```json
{
  "ruv-swarm": {
    "command": "npx", 
    "args": ["ruv-swarm", "mcp", "start"],
    "env": {
      "SWARM_ID": "${SWARM_ID}"
    }
  }
}
```

## 📊 Monitoring & Observability

### Real-time Status
```bash
ruv-swarm status
```

### Live Monitoring
```bash
ruv-swarm monitor --interval 5
```

### Performance Metrics
```bash
ruv-swarm metrics --export prometheus
```

## 🛠️ Common Workflows

### Development Workflow
```bash
# Initialize development swarm
ruv-swarm init --topology hierarchical --profile dev

# Spawn development team
ruv-swarm spawn coder --name "frontend"
ruv-swarm spawn coder --name "backend" 
ruv-swarm spawn tester --name "qa"

# Start development task
ruv-swarm orchestrate adaptive "Build user authentication system"
```

### Research Workflow
```bash
# Initialize research swarm
ruv-swarm init --topology mesh --profile research

# Spawn research team
ruv-swarm spawn researcher --name "literature-review"
ruv-swarm spawn researcher --name "data-analysis"
ruv-swarm spawn analyst --name "synthesis"

# Start research task
ruv-swarm orchestrate parallel "Research machine learning approaches for code generation"
```

## 🔧 Troubleshooting

### Common Issues

**Claude Code not found**
```bash
# Install Claude Code manually
curl -fsSL https://claude.ai/install.sh | sh

# Or use the guided installer
ruv-swarm init --install-claude
```

**MCP configuration issues**
```bash
# Regenerate MCP configuration
ruv-swarm mcp setup --force

# Validate configuration
ruv-swarm mcp validate
```

**Permission errors**
```bash
# Install to user directory
ruv-swarm init --user-install

# Or use sudo for system-wide install
sudo ruv-swarm init
```

## 📚 Next Steps

- **[Seamless Onboarding Guide](seamless-onboarding.md)** - Deep dive into the setup process
- **[MCP Integration Guide](mcp-integration.md)** - Advanced MCP configuration
- **[API Reference](../api/core.md)** - Complete API documentation
- **[Architecture Overview](../architecture/overview.md)** - System design and patterns

## 🆘 Getting Help

- **Documentation**: Browse the complete [docs](../README.md)
- **Examples**: Check out [example workflows](examples/)
- **Issues**: Report problems on [GitHub](https://github.com/ruvnet/ruv-FANN/issues)
- **Discussions**: Join the community [discussions](https://github.com/ruvnet/ruv-FANN/discussions)

---

*Ready to build amazing things with swarm intelligence!* 🚀