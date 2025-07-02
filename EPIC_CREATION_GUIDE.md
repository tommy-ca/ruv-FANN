# 📋 Epic Creation Guide for Swarm Coordination

This guide explains how to create and manage epics for swarm coordination in the ruv-swarm project.

## What is an Epic?

An epic is a large body of work that can be broken down into smaller tasks (issues). It's perfect for coordinating complex features across multiple swarms.

## Creating an Epic

### Method 1: Using the Script (Recommended)

```bash
# Interactive mode
./scripts/create-epic.sh

# Quick mode
./scripts/create-epic.sh quick "Implement Advanced Neural Network Features"
```

### Method 2: Using GitHub CLI Directly

```bash
gh issue create \
  --title "Epic: Your Epic Title" \
  --label "epic,available,priority: high,area: neural" \
  --body "Epic description and breakdown..."
```

### Method 3: GitHub Web Interface

1. Go to Issues → New Issue
2. Choose "Epic" template (if available)
3. Fill in details and apply labels:
   - `epic` (required)
   - `available` (to make it claimable)
   - `priority: [critical/high/medium/low]`
   - `area: [core/mcp/neural/wasm/docs]`

## Epic Structure

### Required Components

```markdown
## 🎯 Epic: [Title]

### Description
Clear description of the epic's goal

### Objectives
- [ ] Major objective 1
- [ ] Major objective 2
- [ ] Major objective 3

### Subtasks
- [ ] #123 - Subtask 1 (update with real issue numbers)
- [ ] #124 - Subtask 2
- [ ] #125 - Subtask 3

### Success Criteria
- All subtasks completed
- Tests passing
- Documentation updated
```

## Working with Epics

### 1. Claiming an Epic

Epics are typically too large for one swarm. Instead:

```bash
# Don't claim the epic directly
# Instead, break it down first

# View the epic
gh issue view 100

# Create subtasks
gh issue create --title "Subtask: Implement Neural Layer" \
  --body "Part of Epic #100" \
  --label "available,priority: high"

# Then claim the subtask
swarm-claim 101
```

### 2. Breaking Down an Epic

When you claim an epic to break it down:

```bash
# Claim the epic temporarily
swarm-claim 100

# Add a comment with your breakdown plan
gh issue comment 100 --body "## Breakdown Plan

I'll create the following subtasks:
1. Research current implementation (#101)
2. Design new architecture (#102)
3. Implement core features (#103)
4. Add tests (#104)
5. Update documentation (#105)

Creating these now..."

# Create the subtasks
for task in "Research" "Design" "Implement" "Test" "Document"; do
  gh issue create --title "[$task] Neural Network Enhancement" \
    --body "Subtask of Epic #100" \
    --label "available"
done

# Update the epic with subtask numbers
gh issue edit 100 --body "Updated with subtasks"

# Release the epic
swarm-release 100 "Broken down into subtasks"
```

### 3. Tracking Epic Progress

```bash
# View epic and its subtasks
gh issue view 100

# Search for all subtasks
gh issue list --search "Epic #100 in:body"

# Check progress
gh issue list --search "Epic #100 in:body" --json state \
  | jq '[.[] | select(.state=="CLOSED")] | length'
```

## Best Practices

### 1. Epic Sizing
- Epics should represent 1-4 weeks of work
- Break down into 3-10 subtasks
- Each subtask should be 1-2 days of work

### 2. Clear Objectives
- Define specific, measurable goals
- List clear acceptance criteria
- Include technical requirements

### 3. Coordination
- One swarm breaks down the epic
- Multiple swarms can work on subtasks
- Use epic comments for coordination

### 4. Labels
Always include:
- `epic` - Identifies as an epic
- `available` - Makes it claimable
- `priority: [level]` - Sets urgency
- `area: [component]` - Identifies scope

## Example Epic Workflow

```bash
# 1. Create epic
./scripts/create-epic.sh quick "Implement WebAssembly Optimization"

# 2. Someone claims to break it down
swarm-claim 200

# 3. They create subtasks
gh issue create --title "[Research] WASM Performance Bottlenecks" \
  --body "Subtask of Epic #200: Research current bottlenecks" \
  --label "available,area: wasm"

# 4. They update the epic
gh issue comment 200 --body "Created subtasks: #201, #202, #203"

# 5. They release the epic
swarm-release 200 "Broken into subtasks"

# 6. Different swarms claim subtasks
swarm-claim 201  # Swarm A
swarm-claim 202  # Swarm B
swarm-claim 203  # Swarm C

# 7. Track overall progress
gh issue list --search "Epic #200 in:body"
```

## Epic Templates

### Feature Epic
```markdown
Epic: Implement [Feature Name]
- Research existing solutions
- Design architecture
- Implement core functionality
- Add test coverage
- Create documentation
- Performance optimization
```

### Refactoring Epic
```markdown
Epic: Refactor [Component Name]
- Analyze current implementation
- Design new structure
- Implement refactoring
- Ensure backward compatibility
- Update tests
- Update documentation
```

### Integration Epic
```markdown
Epic: Integrate [Service/Tool Name]
- Research integration requirements
- Design integration architecture
- Implement connection layer
- Add error handling
- Create tests
- Document usage
```

## Monitoring Epics

```bash
# List all epics
gh issue list --label "epic"

# List available epics
gh issue list --label "epic,available"

# List in-progress epics
gh issue list --label "epic" --search "Subtask in:body state:open"

# List completed epics
gh issue list --label "epic" --state closed
```

## Tips for Success

1. **Start with Research**: First subtask should be research/investigation
2. **Design Before Code**: Include a design subtask
3. **Test Everything**: Dedicated testing subtasks
4. **Document Always**: Documentation subtask at the end
5. **Review Regularly**: Check epic progress daily

Remember: Epics coordinate work across multiple swarms. Clear communication and good breakdown are key to success!