# Add Swarm Commands to Your PATH

To make swarm commands available in all terminal sessions, add one of these lines to your shell profile:

## For Bash (~/.bashrc or ~/.bash_profile)

```bash
# Add swarm commands to PATH
export PATH="$PATH:/workspaces/ruv-swarm-cli/scripts"
source /workspaces/ruv-swarm-cli/setup-swarm-env.sh
```

## For Zsh (~/.zshrc)

```zsh
# Add swarm commands to PATH
export PATH="$PATH:/workspaces/ruv-swarm-cli/scripts"
source /workspaces/ruv-swarm-cli/setup-swarm-env.sh
```

## For Fish (~/.config/fish/config.fish)

```fish
# Add swarm commands to PATH
set -gx PATH $PATH /workspaces/ruv-swarm-cli/scripts
source /workspaces/ruv-swarm-cli/setup-swarm-env.sh
```

## Immediate Setup (Current Session Only)

```bash
source /workspaces/ruv-swarm-cli/setup-swarm-env.sh
```

## After Adding to Profile

1. Reload your shell:
   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

2. Test the commands:
   ```bash
   swarm help
   epic help
   ```

## Available Commands

Once configured, you'll have these commands available globally:

- `swarm status` - Check your status
- `swarm tasks` - List available tasks  
- `swarm claim <id>` - Claim an issue
- `swarm update <id> "message"` - Update progress
- `swarm complete <id>` - Mark complete
- `swarm release <id>` - Release claim
- `swarm sync` - Sync with GitHub
- `swarm conflicts` - Check conflicts
- `swarm help` - Show help

- `epic` - Create new epic interactively
- `epic quick "Title"` - Quick create epic
- `epic help` - Show epic help