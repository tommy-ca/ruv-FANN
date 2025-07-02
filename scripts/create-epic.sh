#!/bin/bash
# Create an epic issue for swarm coordination

set -e

# Load environment variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/swarm-helper.sh" >/dev/null 2>&1 || true

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install with: brew install gh (macOS) or sudo apt install gh (Linux)"
    exit 1
fi

# Function to create epic
create_epic() {
    local title="$1"
    local description="$2"
    local priority="${3:-medium}"
    local area="${4:-core}"
    
    # Create the epic body
    local body="## 🎯 Epic: $title

### Description
$description

### Objectives
- [ ] Define requirements and scope
- [ ] Break down into subtasks
- [ ] Implement core functionality
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Review and refine

### Subtasks
<!-- Add subtask issues here as they are created -->
- [ ] #TBD - Subtask 1
- [ ] #TBD - Subtask 2
- [ ] #TBD - Subtask 3

### Acceptance Criteria
- All subtasks completed
- Tests passing
- Documentation updated
- Code reviewed

### Technical Considerations
<!-- Add any technical notes, dependencies, or considerations -->

### Progress Tracking
- **Status**: Planning
- **Estimated Effort**: TBD
- **Target Completion**: TBD

---
*This is an epic issue. Break it down into smaller tasks before implementation.*"

    # Create the issue with appropriate labels
    echo "Creating epic issue..."
    issue_url=$(gh issue create \
        --title "Epic: $title" \
        --body "$body" \
        --label "epic" \
        --label "available" \
        --label "priority: $priority" \
        --label "area: $area" \
        --repo "$GITHUB_OWNER/$GITHUB_REPO" 2>&1)
    
    # Check if creation failed
    if [ $? -ne 0 ]; then
        echo "❌ Error creating epic:"
        echo "$issue_url"
        exit 1
    fi
    
    # Extract issue number from URL
    issue_number=$(echo "$issue_url" | grep -oE '[0-9]+$')
    
    echo "✅ Epic created successfully!"
    echo "Issue #$issue_number: $issue_url"
    echo ""
    echo "Next steps:"
    echo "1. Review the epic at: $issue_url"
    echo "2. Break it down into subtasks"
    echo "3. Claim it with: swarm-claim $issue_number"
}

# Interactive mode
if [ $# -eq 0 ]; then
    echo "=== Create Epic Issue ==="
    echo ""
    
    # Get epic details
    read -p "Epic title: " title
    echo "Epic description (press Ctrl+D when done):"
    description=$(cat)
    
    echo ""
    echo "Priority levels: critical, high, medium, low"
    read -p "Priority [medium]: " priority
    priority=${priority:-medium}
    
    echo ""
    echo "Areas: core, mcp, neural, wasm, docs, tests"
    read -p "Area [core]: " area
    area=${area:-core}
    
    echo ""
    echo "Creating epic with:"
    echo "  Title: $title"
    echo "  Priority: $priority"
    echo "  Area: $area"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_epic "$title" "$description" "$priority" "$area"
    else
        echo "Cancelled"
    fi
    
# Command line mode
else
    case "$1" in
        "help"|"-h"|"--help")
            echo "Create Epic Issue for Swarm Coordination"
            echo ""
            echo "Usage:"
            echo "  $0                    - Interactive mode"
            echo "  $0 quick <title>      - Quick create with defaults"
            echo "  $0 help               - Show this help"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 quick \"Implement Neural Network Integration\""
            echo ""
            echo "Labels applied to epics:"
            echo "  - epic: Marks as an epic issue"
            echo "  - available: Ready to be claimed"
            echo "  - priority: [level]"
            echo "  - area: [component]"
            ;;
            
        "quick")
            if [ -z "$2" ]; then
                echo "Error: Title required for quick mode"
                echo "Usage: $0 quick \"Epic Title\""
                exit 1
            fi
            title="$2"
            description="This epic needs to be broken down into subtasks for implementation."
            create_epic "$title" "$description" "medium" "core"
            ;;
            
        *)
            echo "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
fi