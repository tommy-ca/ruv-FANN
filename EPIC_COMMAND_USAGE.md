# GitHub Epic Command Usage Guide

The `/github/epic` command is now fully implemented and ready to use in Claude Code!

## 🚀 Quick Start

### Create an Epic
```bash
# Basic usage
node src/github-epic-implementation.js create "My Epic Title"

# With options
node src/github-epic-implementation.js create "API Platform v2" \
  --components "Authentication,Database,API,Frontend,Testing" \
  --description "Complete API platform rewrite with modern architecture" \
  --weeks 12 \
  --priority high \
  --swarm
```

### Generate Template
```bash
# Create a template file for customization
node src/github-epic-implementation.js template
```

### Create Subtasks
```bash
# Create subtasks for existing epic #123
node src/github-epic-implementation.js subtasks 123
```

## 📋 Implementation Details

### Files Created:
1. **`/src/github-epic-implementation.js`** - Main implementation
2. **`/.claude/hooks/github-epic-hook.js`** - Claude Code integration hook
3. **`/.claude/commands/github/epic.md`** - Command documentation
4. **`/.claude/commands/github/epic-create.md`** - Detailed usage guide

### Features:
- ✅ Full epic structure matching issue #18 format
- ✅ Automatic subtask creation
- ✅ Customizable components and timeline
- ✅ Swarm coordination metadata support
- ✅ Priority levels and risk assessment
- ✅ Progress tracking sections
- ✅ Team assignment placeholders

## 🛠️ Prerequisites

1. **GitHub CLI** must be installed:
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   
   # Authenticate
   gh auth login
   ```

2. **Repository Access** - Ensure you have write access to the target repository

## 💡 Usage in Claude Code

In Claude Code, you can now use:

```
/github/epic create "Your Epic Title"
/github/epic template
/github/epic subtasks 123
```

The command will:
1. Generate a comprehensive epic body with all sections
2. Create the issue on GitHub using the `gh` CLI
3. Optionally create subtask issues for each component
4. Return the created issue URL

## 🎯 Epic Structure

Each epic includes:
- Status bar with progress percentages
- Key objectives and success metrics
- Component breakdown with subtasks
- System architecture diagram
- Performance metrics table
- Implementation timeline
- Related issues tracking
- Acceptance criteria
- Team assignments
- Risk assessment

## 🔧 Customization

Edit the `generateEpicBody()` method in `src/github-epic-implementation.js` to customize:
- Default components
- Timeline structure
- Metric definitions
- Architecture diagrams
- Team roles

## 📝 Example Output

The command creates epics that look exactly like: https://github.com/ruvnet/ruv-FANN/issues/18

With proper:
- 🚀 Emoji indicators
- Status tracking
- Component breakdowns
- Progress visualization
- Team coordination sections