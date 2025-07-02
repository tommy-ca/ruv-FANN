#!/bin/bash
# Setup aliases for GitHub swarm coordination
# Source this file in your shell: source scripts/setup-aliases.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Quick GitHub swarm commands using the helper script
alias swarm-status="$SCRIPT_DIR/swarm-helper.sh status"
alias swarm-tasks="$SCRIPT_DIR/swarm-helper.sh tasks"
alias swarm-claim="$SCRIPT_DIR/swarm-helper.sh claim"
alias swarm-update="$SCRIPT_DIR/swarm-helper.sh update"
alias swarm-complete="$SCRIPT_DIR/swarm-helper.sh complete"
alias swarm-release="$SCRIPT_DIR/swarm-helper.sh release"
alias swarm-sync="$SCRIPT_DIR/swarm-helper.sh sync"
alias swarm-conflicts="$SCRIPT_DIR/swarm-helper.sh conflicts"

# Quick task queries
alias tasks-high="$SCRIPT_DIR/swarm-helper.sh tasks high"
alias tasks-medium="$SCRIPT_DIR/swarm-helper.sh tasks medium"
alias tasks-low="$SCRIPT_DIR/swarm-helper.sh tasks low"

# Epic management
alias create-epic="$SCRIPT_DIR/create-epic.sh"
alias list-epics='gh issue list --label "epic" --repo "$GITHUB_OWNER/$GITHUB_REPO"'
alias available-epics='gh issue list --label "epic,available" --repo "$GITHUB_OWNER/$GITHUB_REPO"'

# Git shortcuts for swarm work
alias gs-create-branch='f() { git checkout -b "swarm/$CLAUDE_SWARM_ID/issue-$1"; }; f'
alias gs-push='git push -u origin $(git branch --show-current)'

echo "✓ Swarm aliases loaded!"
echo ""
echo "Available aliases:"
echo "  swarm-status     - Show your swarm status"
echo "  swarm-tasks      - List available tasks"
echo "  swarm-claim      - Claim an issue"
echo "  swarm-update     - Post progress update"
echo "  swarm-complete   - Mark issue complete"
echo "  swarm-release    - Release a claim"
echo "  swarm-sync       - Sync with GitHub"
echo "  swarm-conflicts  - Check for conflicts"
echo ""
echo "  tasks-high       - Show high priority tasks"
echo "  tasks-medium     - Show medium priority tasks"
echo "  tasks-low        - Show low priority tasks"
echo ""
echo "  create-epic      - Create a new epic issue"
echo "  list-epics       - List all epic issues"
echo "  available-epics  - List claimable epics"
echo ""
echo "  gs-create-branch <issue> - Create branch for issue"
echo "  gs-push          - Push current branch to origin"