# GitHub Swarm Coordination Setup

Quick setup guide for GitHub-based swarm coordination in the ruv-swarm-cli project.

## Quick Start

### 1. Environment Setup

Copy and configure the environment file:
```bash
cp .env.example .env
# Edit .env with your settings
```

Key settings:
- `CLAUDE_SWARM_ID`: Your unique swarm identifier
- `GITHUB_OWNER`: Repository owner (default: ruvnet)
- `GITHUB_REPO`: Repository name (default: ruv-FANN)

### 2. Load Aliases

For quick access to swarm commands:
```bash
source scripts/setup-aliases.sh
```

### 3. Basic Workflow

```bash
# Check your status
swarm-status

# Find available work
swarm-tasks
tasks-high    # High priority only

# Claim a task
swarm-claim 123

# Post updates
swarm-update 123 "Implemented authentication logic"

# Complete task
swarm-complete 123 456  # 456 is PR number

# Or release if blocked
swarm-release 123 "Need more context"
```

## Available Scripts

### Main Helper Script
`scripts/swarm-helper.sh` - Full-featured swarm coordination tool

Commands:
- `status` - Show your swarm status and active tasks
- `tasks [priority]` - List available tasks
- `claim <issue>` - Claim an issue
- `update <issue> <msg>` - Post progress update
- `complete <issue> [pr]` - Mark as complete
- `release <issue>` - Release a claim
- `sync` - Sync with GitHub
- `conflicts` - Check for conflicts

### Aliases
After sourcing `scripts/setup-aliases.sh`:
- `swarm-status`, `swarm-tasks`, `swarm-claim`, etc.
- `tasks-high`, `tasks-medium`, `tasks-low`
- `gs-create-branch <issue>` - Create feature branch
- `gs-push` - Push current branch

## GitHub CLI Required

The swarm helper requires GitHub CLI. Install if needed:
```bash
# macOS
brew install gh

# Linux
sudo apt install gh

# Authenticate
gh auth login
```

## Working with Issues

### Label System
- `available` - Ready to claim
- `swarm-claimed` - Currently being worked on
- `priority: critical/high/medium/low` - Task priority
- `area: core/mcp/neural/wasm/docs` - Component area

### Best Practices
1. Check before claiming - avoid conflicts
2. Update regularly - every 30-60 minutes
3. Release if blocked - don't hold tasks
4. Complete or release within 24 hours

## Integration with ruv-swarm

The GitHub coordination works alongside ruv-swarm MCP tools:
- GitHub handles task assignment and tracking
- ruv-swarm provides AI agent coordination
- Use both for maximum efficiency

## Troubleshooting

### "gh: command not found"
Install GitHub CLI (see above)

### "Not authenticated"
Run `gh auth login`

### Can't claim issues
Check repository permissions with `gh auth status`

## Next Steps

1. Configure your `.env` file
2. Test with `scripts/swarm-helper.sh status`
3. Find and claim your first task
4. Use ruv-swarm MCP tools for implementation

For detailed collaboration guide, see:
`ruv-swarm/SWARM_COLLABORATION_GUIDE.md`