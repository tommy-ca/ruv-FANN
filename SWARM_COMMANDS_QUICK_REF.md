# 🚀 Swarm Commands Quick Reference

## Setup (One Time)

```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc):
export PATH="$PATH:/workspaces/ruv-swarm-cli/scripts"

# Or source the setup script each session:
source /workspaces/ruv-swarm-cli/setup-swarm-env.sh
```

## Daily Workflow Commands

### 🔍 Finding Work
```bash
swarm tasks              # List all available tasks
swarm tasks high         # High priority only
swarm tasks medium       # Medium priority
epic                     # Create new epic
```

### 📋 Managing Tasks
```bash
swarm claim 123          # Claim issue #123
swarm update 123 "Done with auth logic"  # Post update
swarm complete 123       # Mark as complete
swarm complete 123 456   # Complete with PR #456
swarm release 123        # Release if blocked
```

### 📊 Status & Monitoring
```bash
swarm status             # Your active work
swarm sync              # Sync with GitHub
swarm conflicts         # Check for conflicts
```

### 🎯 Creating Epics
```bash
epic                     # Interactive epic creation
epic quick "Title"       # Quick create with defaults
epic help               # Show epic help
```

## GitHub CLI Commands (Direct)

### Issues
```bash
gh issue list --label "available"        # Available work
gh issue view 123                        # View issue details
gh issue comment 123 --body "Update"     # Add comment
```

### Pull Requests
```bash
gh pr create --title "Fix: Issue #123"   # Create PR
gh pr list                               # List PRs
gh pr view 456                           # View PR
```

## Git Workflow

```bash
# Create feature branch
git checkout -b swarm/swarm-jed/issue-123

# Stage and commit
git add .
git commit -m "Fix: Authentication bug (#123)"

# Push to remote
git push -u origin $(git branch --show-current)

# Create PR
gh pr create --title "Fix: Authentication bug" --body "Closes #123"
```

## Environment Variables

Check your configuration:
```bash
echo $CLAUDE_SWARM_ID    # Your swarm ID
echo $GITHUB_OWNER       # Repository owner
echo $GITHUB_REPO        # Repository name
```

## Troubleshooting

```bash
# Check GitHub CLI auth
gh auth status

# Re-authenticate if needed
gh auth login

# Check if commands are in PATH
which swarm
which epic

# Reload environment
source setup-swarm-env.sh
```

## Tips

1. **Update regularly** - Post progress every 30-60 minutes
2. **Release if blocked** - Don't hold tasks you can't complete
3. **One task at a time** - Focus on single issues
4. **Clear communication** - Use issue comments for coordination

## Help

```bash
swarm help              # Show all swarm commands
epic help               # Show epic commands
gh issue --help         # GitHub issue help
gh pr --help           # GitHub PR help
```