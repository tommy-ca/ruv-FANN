# /github/epic

Create comprehensive GitHub epics with the ruv-FANN issue #18 format.

## Usage

```
/github/epic create <title> [options]
/github/epic template
/github/epic subtasks <epic-number>
```

## Commands

### Create Epic

Creates a new GitHub epic with comprehensive structure:

```
/github/epic create "AI Agent Integration Platform"
```

Options:
- `--components` - Comma-separated list of components
- `--description` - Epic description
- `--weeks` - Timeline in weeks (default: 8)
- `--priority` - Priority level (low/medium/high/critical)
- `--swarm` - Enable swarm coordination metadata

### Generate Template

Create an epic template file:

```
/github/epic template
```

This generates `epic-template.md` that you can customize before creating the issue.

### Create Subtasks

Create subtask issues for an existing epic:

```
/github/epic subtasks 123
```

This creates subtask issues for each component in epic #123.

## Epic Structure

The epic follows this comprehensive format:

1. **Status Bar** - Progress tracking with percentages
2. **Overview** - Epic description and objectives
3. **Success Metrics** - Measurable targets
4. **Component Breakdown** - Detailed subtasks
5. **Architecture Diagram** - ASCII visualization
6. **Metrics Table** - Performance targets
7. **Timeline** - Week-by-week plan
8. **Related Issues** - Dependencies and subtasks
9. **Acceptance Criteria** - Definition of done
10. **Team Assignments** - Responsibility matrix

## Examples

### Basic Epic Creation

```bash
# Create a simple epic
/github/epic create "New Feature Development"

# With components
/github/epic create "API Platform" --components "Auth,Database,API,Tests"

# Full options
/github/epic create "ML Pipeline" \
  --components "Data Processing,Model Training,Inference,Monitoring" \
  --description "End-to-end ML pipeline implementation" \
  --weeks 12 \
  --priority high \
  --swarm
```

### Advanced Workflow

1. Generate template:
```
/github/epic template
```

2. Edit the template file to customize

3. Create epic from template:
```
gh issue create --body-file epic-template.md --label "epic"
```

4. Create subtasks:
```
/github/epic subtasks <issue-number>
```

## Template Variables

The epic template supports these variables:
- `{{title}}` - Epic title
- `{{description}}` - Epic description
- `{{components}}` - Component list
- `{{timeline}}` - Week count
- `{{date}}` - Current date
- `{{target_date}}` - Completion date
- `{{swarm_id}}` - Swarm coordination ID

## Integration with Swarm

When using `--swarm` flag, the epic includes:
- Swarm coordination section
- Agent assignments
- Memory storage keys
- Performance tracking

## Status Indicators

- ✅ Complete
- 🔄 In Progress
- 🆕 Planned
- ❌ Blocked
- 🟡 At Risk

## Priority Levels

- 🔴 HIGH/CRITICAL
- 🟡 MEDIUM
- 🟢 LOW

## Tips

1. **Use descriptive titles** - Include version numbers if applicable
2. **Break down components** - Aim for 4-8 components per epic
3. **Set realistic timelines** - Buffer for unknowns
4. **Define clear metrics** - Make success measurable
5. **Update regularly** - Keep status current

## See Also

- `/github/tasks` - Find and manage tasks
- `/github/coordinate` - Coordinate team work
- `/github/status` - Check epic status
- `/github/update` - Update epic progress