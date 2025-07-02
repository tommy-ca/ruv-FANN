# GitHub MCP Server Setup for ruv-swarm

The GitHub MCP (Model Context Protocol) server is now automatically configured when you run `ruv-swarm init`!

## 🚀 Automatic Setup

When you run `ruv-swarm init`, it will:

1. **Configure GitHub MCP Server** - Adds GitHub's official MCP server to `.mcp.json`
2. **Configure ruv-swarm MCP Server** - Adds ruv-swarm's MCP server
3. **Check for GitHub Token** - Uses GITHUB_TOKEN or GH_TOKEN if available
4. **Ready for Claude Code** - No restart needed, works on next Claude Code launch

## 📋 What Gets Configured

The `.mcp.json` file will contain:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token-here"
      }
    },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"],
      "env": {
        "SWARM_ID": "swarm-123456789",
        "SWARM_TOPOLOGY": "mesh"
      }
    }
  }
}
```

## 🔧 Manual Configuration

If you need to configure MCP servers manually:

```bash
# Run the configuration script
npm run configure-mcp

# Or directly
node configure-github-mcp.js
```

## 🔑 GitHub Authentication

For full GitHub access, you need to provide authentication:

### Option 1: Environment Variable
```bash
export GITHUB_TOKEN="your-github-token"
# or
export GH_TOKEN="your-github-token"
```

### Option 2: GitHub CLI
```bash
gh auth login
```

## 🛠️ Available GitHub MCP Tools

Once configured, you can use these tools in Claude Code:

- `mcp__github__issues_list` - List repository issues
- `mcp__github__issues_create` - Create new issues
- `mcp__github__issues_update` - Update existing issues
- `mcp__github__pr_create` - Create pull requests
- `mcp__github__pr_review` - Review pull requests
- `mcp__github__search_repositories` - Search GitHub repos
- `mcp__github__search_code` - Search code across GitHub
- And many more!

## 🐝 Available Swarm MCP Tools

The ruv-swarm MCP tools are also configured:

- `mcp__ruv-swarm__swarm_init` - Initialize swarm
- `mcp__ruv-swarm__agent_spawn` - Spawn agents
- `mcp__ruv-swarm__task_orchestrate` - Orchestrate tasks
- `mcp__ruv-swarm__swarm_status` - Check status
- And all other swarm coordination tools

## 🚨 Troubleshooting

### MCP servers not showing up in Claude Code?
1. Make sure `.mcp.json` exists in your project root
2. Restart Claude Code completely
3. Check Claude Code logs for MCP server errors

### GitHub authentication issues?
1. Verify your token: `gh auth status`
2. Check token permissions (needs repo access)
3. Try re-authenticating: `gh auth login`

### Manual MCP installation
If npx doesn't work, install globally:
```bash
npm install -g @modelcontextprotocol/server-github
```

## 📊 Integration Benefits

With both GitHub and ruv-swarm MCP servers:

1. **Seamless GitHub Integration** - Use GitHub directly from Claude Code
2. **Swarm Coordination** - Orchestrate multi-agent workflows
3. **Unified Workflow** - GitHub issues → Swarm tasks → Code → PRs
4. **No Context Switching** - Everything within Claude Code

## 🎯 Example Workflow

```javascript
// 1. Initialize swarm
mcp__ruv-swarm__swarm_init({ topology: "mesh", maxAgents: 5 })

// 2. List GitHub issues
mcp__github__issues_list({ owner: "ruvnet", repo: "ruv-FANN", state: "open" })

// 3. Spawn agents for each issue
mcp__ruv-swarm__agent_spawn({ type: "coder", name: "issue-fixer" })

// 4. Orchestrate work
mcp__ruv-swarm__task_orchestrate({ task: "Fix issue #123" })

// 5. Create PR when done
mcp__github__pr_create({ 
  title: "Fix: Issue #123",
  head: "fix/issue-123",
  base: "main"
})
```

## 🔄 Updates

The configuration is now part of `ruv-swarm init`, so every new project automatically gets GitHub MCP integration!