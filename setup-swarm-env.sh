#!/bin/bash
# Setup script to configure swarm environment

# Get the directory where this script is located
SWARM_CLI_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export environment variables
export SWARM_CLI_ROOT
export PATH="$PATH:$SWARM_CLI_ROOT/scripts"

# Load .env file if it exists
if [ -f "$SWARM_CLI_ROOT/.env" ]; then
    # Read .env file line by line, skip comments and empty lines
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^[[:space:]]*# ]] && [[ -n "$key" ]]; then
            # Remove leading/trailing whitespace
            key=$(echo "$key" | xargs)
            # Skip PATH export line as we handle it separately
            if [[ "$key" != "export PATH"* ]]; then
                # Remove inline comments from value
                value=$(echo "$value" | sed 's/#.*//' | xargs)
                # Remove quotes if present
                value="${value%\"}"
                value="${value#\"}"
                # Export the variable
                export "$key=$value"
            fi
        fi
    done < "$SWARM_CLI_ROOT/.env"
fi

# Verify setup
echo "✅ Swarm environment configured!"
echo ""
echo "Configuration:"
echo "  SWARM_CLI_ROOT: $SWARM_CLI_ROOT"
echo "  CLAUDE_SWARM_ID: ${CLAUDE_SWARM_ID:-not set}"
echo "  GITHUB_OWNER: ${GITHUB_OWNER:-not set}"
echo "  GITHUB_REPO: ${GITHUB_REPO:-not set}"
echo ""

# Check if scripts are accessible
if command -v swarm >/dev/null 2>&1; then
    echo "✅ Swarm commands are available in PATH"
    echo ""
    echo "Available commands:"
    echo "  swarm status    - Check your swarm status"
    echo "  swarm tasks     - List available tasks"
    echo "  swarm claim     - Claim an issue"
    echo "  swarm update    - Post progress update"
    echo "  swarm complete  - Mark task complete"
    echo "  swarm help      - Show all commands"
    echo ""
    echo "  epic            - Create new epic"
    echo "  epic help       - Show epic commands"
else
    echo "⚠️  Scripts directory not in PATH"
    echo "Add this line to your shell profile:"
    echo "  export PATH=\"\$PATH:$SWARM_CLI_ROOT/scripts\""
fi

# Check GitHub CLI
if command -v gh >/dev/null 2>&1; then
    echo ""
    echo "✅ GitHub CLI is installed"
    if gh auth status >/dev/null 2>&1; then
        echo "✅ GitHub CLI is authenticated"
    else
        echo "⚠️  GitHub CLI needs authentication"
        echo "Run: gh auth login"
    fi
else
    echo ""
    echo "❌ GitHub CLI is not installed"
    echo "Install with: brew install gh (macOS) or sudo apt install gh (Linux)"
fi