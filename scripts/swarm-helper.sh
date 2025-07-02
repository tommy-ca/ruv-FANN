#!/bin/bash
# Swarm Collaboration Helper Script
# Simplifies GitHub-based swarm coordination

set -e

# Load environment variables
if [ -f .env ]; then
    # Read .env file line by line, skip comments and empty lines
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^[[:space:]]*# ]] && [[ -n "$key" ]]; then
            # Remove leading/trailing whitespace
            key=$(echo "$key" | xargs)
            # Remove inline comments from value
            value=$(echo "$value" | sed 's/#.*//' | xargs)
            # Remove quotes if present
            value="${value%\"}"
            value="${value#\"}"
            # Export the variable
            export "$key=$value"
        fi
    done < .env
fi

# Set defaults
GITHUB_OWNER=${GITHUB_OWNER:-"ruvnet"}
GITHUB_REPO=${GITHUB_REPO:-"ruv-FANN"}
CLAUDE_SWARM_ID=${CLAUDE_SWARM_ID:-"swarm-$(date +%s)"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
function print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if gh CLI is installed
function check_gh_cli() {
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) is not installed"
        echo "Install with: brew install gh (macOS) or sudo apt install gh (Linux)"
        echo "Then run: gh auth login"
        exit 1
    fi
    
    if ! gh auth status &> /dev/null; then
        print_error "GitHub CLI is not authenticated"
        echo "Run: gh auth login"
        exit 1
    fi
}

# Main commands
case "$1" in
    "status")
        check_gh_cli
        print_header "Swarm Status"
        echo "Swarm ID: $CLAUDE_SWARM_ID"
        echo "Repository: $GITHUB_OWNER/$GITHUB_REPO"
        echo ""
        print_header "My Active Tasks"
        gh issue list --assignee "@me" --state open --repo "$GITHUB_OWNER/$GITHUB_REPO"
        echo ""
        print_header "My Claimed Tasks"
        gh issue list --search "label:swarm-claimed \"$CLAUDE_SWARM_ID\" in:body" --state open --repo "$GITHUB_OWNER/$GITHUB_REPO"
        ;;
        
    "tasks")
        check_gh_cli
        print_header "Available Tasks"
        PRIORITY=${2:-"all"}
        if [ "$PRIORITY" = "all" ]; then
            gh issue list --label "available" --label "-swarm-claimed" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        else
            gh issue list --label "available" --label "-swarm-claimed" --label "priority: $PRIORITY" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        fi
        ;;
        
    "claim")
        ISSUE_NUMBER=$2
        if [ -z "$ISSUE_NUMBER" ]; then
            print_error "Usage: $0 claim <issue_number>"
            exit 1
        fi
        print_header "Claiming Issue #$ISSUE_NUMBER"
        gh issue edit "$ISSUE_NUMBER" --add-label "swarm-claimed" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        gh issue comment "$ISSUE_NUMBER" --body "🐝 Swarm ID: $CLAUDE_SWARM_ID claiming this task.

Plan:
1. Analyze requirements
2. Design solution
3. Implement changes
4. Add tests
5. Update documentation

Starting work now." --repo "$GITHUB_OWNER/$GITHUB_REPO"
        print_success "Issue #$ISSUE_NUMBER claimed"
        ;;
        
    "release")
        ISSUE_NUMBER=$2
        if [ -z "$ISSUE_NUMBER" ]; then
            print_error "Usage: $0 release <issue_number>"
            exit 1
        fi
        print_header "Releasing Issue #$ISSUE_NUMBER"
        gh issue edit "$ISSUE_NUMBER" --remove-label "swarm-claimed" --add-label "available" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        gh issue comment "$ISSUE_NUMBER" --body "🔄 Swarm ID: $CLAUDE_SWARM_ID releasing this task.
Reason: ${3:-"Unable to complete at this time"}" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        print_success "Issue #$ISSUE_NUMBER released"
        ;;
        
    "update")
        ISSUE_NUMBER=$2
        MESSAGE=$3
        if [ -z "$ISSUE_NUMBER" ] || [ -z "$MESSAGE" ]; then
            print_error "Usage: $0 update <issue_number> \"<message>\""
            exit 1
        fi
        print_header "Updating Issue #$ISSUE_NUMBER"
        gh issue comment "$ISSUE_NUMBER" --body "🔄 Progress Update from $CLAUDE_SWARM_ID:
$MESSAGE" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        print_success "Update posted"
        ;;
        
    "complete")
        ISSUE_NUMBER=$2
        PR_NUMBER=$3
        if [ -z "$ISSUE_NUMBER" ]; then
            print_error "Usage: $0 complete <issue_number> [pr_number]"
            exit 1
        fi
        print_header "Completing Issue #$ISSUE_NUMBER"
        
        COMMENT="✅ Task Complete by $CLAUDE_SWARM_ID:"
        if [ -n "$PR_NUMBER" ]; then
            COMMENT="$COMMENT
- Changes implemented in PR #$PR_NUMBER"
        fi
        COMMENT="$COMMENT
- All requirements met
- Tests added and passing
- Documentation updated
- Ready for review"
        
        gh issue comment "$ISSUE_NUMBER" --body "$COMMENT" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        gh issue edit "$ISSUE_NUMBER" --remove-label "swarm-claimed" --add-label "needs-review" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        print_success "Issue #$ISSUE_NUMBER marked complete"
        ;;
        
    "sync")
        print_header "Syncing with GitHub"
        git fetch origin
        echo ""
        print_header "Local Changes"
        git status -s
        echo ""
        print_header "Recent Remote Commits"
        git log --oneline origin/main -10
        ;;
        
    "conflicts")
        print_header "Potential Conflicts"
        gh issue list --search "label:swarm-claimed comments:>5" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        echo ""
        print_header "Stale Claims (>24h)"
        gh issue list --label "swarm-claimed" --search "updated:<1 day" --repo "$GITHUB_OWNER/$GITHUB_REPO"
        ;;
        
    "help"|*)
        echo "Swarm Collaboration Helper"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  status              Show your swarm status and active tasks"
        echo "  tasks [priority]    List available tasks (optional: high, medium, low)"
        echo "  claim <issue>       Claim an issue for your swarm"
        echo "  update <issue> <msg> Post progress update"
        echo "  complete <issue> [pr] Mark issue as complete"
        echo "  release <issue>     Release a claimed issue"
        echo "  sync               Sync with GitHub repository"
        echo "  conflicts          Show potential conflicts and stale claims"
        echo "  help               Show this help message"
        echo ""
        echo "Current Configuration:"
        echo "  Swarm ID: $CLAUDE_SWARM_ID"
        echo "  Repository: $GITHUB_OWNER/$GITHUB_REPO"
        ;;
esac