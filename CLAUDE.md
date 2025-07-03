# Claude Code Configuration for ruv-swarm

## 🎯 IMPORTANT: Separation of Responsibilities

### Claude Code Handles:
- ✅ **ALL file operations** (Read, Write, Edit, MultiEdit)
- ✅ **ALL code generation** and development tasks
- ✅ **ALL bash commands** and system operations
- ✅ **ALL actual implementation** work
- ✅ **Project navigation** and code analysis

### ruv-swarm MCP Tools Handle:
- 🧠 **Coordination only** - Orchestrating Claude Code's actions
- 💾 **Memory management** - Persistent state across sessions
- 🤖 **Neural features** - Cognitive patterns and learning
- 📊 **Performance tracking** - Monitoring and metrics
- 🐝 **Swarm orchestration** - Multi-agent coordination

### ⚠️ Key Principle:
**MCP tools DO NOT create content or write code.** They coordinate and enhance Claude Code's native capabilities. Think of them as an orchestration layer that helps Claude Code work more efficiently.

## 🚀 CRITICAL: Parallel Execution & Batch Operations

### 🚨 MANDATORY RULE #1: BATCH EVERYTHING

**When using swarms, you MUST use BatchTool for ALL operations:**

1. **NEVER** send multiple messages for related operations
2. **ALWAYS** combine multiple tool calls in ONE message
3. **PARALLEL** execution is MANDATORY, not optional

### ⚡ THE GOLDEN RULE OF SWARMS

```
If you need to do X operations, they should be in 1 message, not X messages
```

### 📦 BATCH TOOL EXAMPLES

**✅ CORRECT - Everything in ONE Message:**
```javascript
[Single Message with BatchTool]:
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__ruv-swarm__agent_spawn { type: "researcher" }
  mcp__ruv-swarm__agent_spawn { type: "coder" }
  mcp__ruv-swarm__agent_spawn { type: "analyst" }
  mcp__ruv-swarm__agent_spawn { type: "tester" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator" }
  TodoWrite { todos: [todo1, todo2, todo3, todo4, todo5] }
  Bash "mkdir -p app/{src,tests,docs}"
  Write "app/package.json" 
  Write "app/README.md"
  Write "app/src/index.js"
```

**❌ WRONG - Multiple Messages (NEVER DO THIS):**
```javascript
Message 1: mcp__ruv-swarm__swarm_init
Message 2: mcp__ruv-swarm__agent_spawn 
Message 3: mcp__ruv-swarm__agent_spawn
Message 4: TodoWrite (one todo)
Message 5: Bash "mkdir src"
Message 6: Write "package.json"
// This is 6x slower and breaks parallel coordination!
```

## Build Commands
- `npm run build`: Build the project
- `npm run test`: Run the full test suite
- `npm run lint`: Run ESLint and format checks
- `npm run typecheck`: Run TypeScript type checking
- `./claude-flow --help`: Show all available commands

## Claude-Flow Complete Command Reference

### Core System Commands
- `./claude-flow start [--ui] [--port 3000] [--host localhost]`: Start orchestration system with optional web UI
- `./claude-flow status`: Show comprehensive system status
- `./claude-flow monitor`: Real-time system monitoring dashboard
- `./claude-flow config <subcommand>`: Configuration management (show, get, set, init, validate)

### Agent Management
- `./claude-flow agent spawn <type> [--name <name>]`: Create AI agents (researcher, coder, analyst, etc.)
- `./claude-flow agent list`: List all active agents
- `./claude-flow spawn <type>`: Quick agent spawning (alias for agent spawn)

### Task Orchestration
- `./claude-flow task create <type> [description]`: Create and manage tasks
- `./claude-flow task list`: View active task queue
- `./claude-flow workflow <file>`: Execute workflow automation files

### Memory Management
- `./claude-flow memory store <key> <data>`: Store persistent data across sessions
- `./claude-flow memory get <key>`: Retrieve stored information
- `./claude-flow memory list`: List all memory keys
- `./claude-flow memory export <file>`: Export memory to file
- `./claude-flow memory import <file>`: Import memory from file
- `./claude-flow memory stats`: Memory usage statistics
- `./claude-flow memory cleanup`: Clean unused memory entries

### SPARC Development Modes
- `./claude-flow sparc "<task>"`: Run orchestrator mode (default)
- `./claude-flow sparc run <mode> "<task>"`: Run specific SPARC mode
- `./claude-flow sparc tdd "<feature>"`: Test-driven development mode
- `./claude-flow sparc modes`: List all 17 available SPARC modes

Available SPARC modes: orchestrator, coder, researcher, tdd, architect, reviewer, debugger, tester, analyzer, optimizer, documenter, designer, innovator, swarm-coordinator, memory-manager, batch-executor, workflow-manager

### Swarm Coordination
- `./claude-flow swarm "<objective>" [options]`: Multi-agent swarm coordination
- `--strategy`: research, development, analysis, testing, optimization, maintenance
- `--mode`: centralized, distributed, hierarchical, mesh, hybrid
- `--max-agents <n>`: Maximum number of agents (default: 5)
- `--parallel`: Enable parallel execution
- `--monitor`: Real-time monitoring
- `--output <format>`: json, sqlite, csv, html

### MCP Server Integration
- `./claude-flow mcp start [--port 3000] [--host localhost]`: Start MCP server
- `./claude-flow mcp status`: Show MCP server status
- `./claude-flow mcp tools`: List available MCP tools

### Claude Integration
- `./claude-flow claude auth`: Authenticate with Claude API
- `./claude-flow claude models`: List available Claude models
- `./claude-flow claude chat`: Interactive chat mode

### Session Management
- `./claude-flow session`: Manage terminal sessions
- `./claude-flow repl`: Start interactive REPL mode

### Enterprise Features
- `./claude-flow project <subcommand>`: Project management (Enterprise)
- `./claude-flow deploy <subcommand>`: Deployment operations (Enterprise)
- `./claude-flow cloud <subcommand>`: Cloud infrastructure management (Enterprise)
- `./claude-flow security <subcommand>`: Security and compliance tools (Enterprise)
- `./claude-flow analytics <subcommand>`: Analytics and insights (Enterprise)

### Project Initialization
- `./claude-flow init`: Initialize Claude-Flow project
- `./claude-flow init --sparc`: Initialize with full SPARC development environment

## Quick Start Workflows

### Research Workflow
```bash
# Start a research swarm with distributed coordination
./claude-flow swarm "Research modern web frameworks" --strategy research --mode distributed --parallel --monitor

# Or use SPARC researcher mode for focused research
./claude-flow sparc run researcher "Analyze React vs Vue vs Angular performance characteristics"

# Store findings in memory for later use
./claude-flow memory store "research_findings" "Key insights from framework analysis"
```

### Development Workflow
```bash
# Start orchestration system with web UI
./claude-flow start --ui --port 3000

# Run TDD workflow for new feature
./claude-flow sparc tdd "User authentication system with JWT tokens"

# Development swarm for complex projects
./claude-flow swarm "Build e-commerce API with payment integration" --strategy development --mode hierarchical --max-agents 8 --monitor

# Check system status
./claude-flow status
```

### Analysis Workflow
```bash
# Analyze codebase performance
./claude-flow sparc run analyzer "Identify performance bottlenecks in current codebase"

# Data analysis swarm
./claude-flow swarm "Analyze user behavior patterns from logs" --strategy analysis --mode mesh --parallel --output sqlite

# Store analysis results
./claude-flow memory store "performance_analysis" "Bottlenecks identified in database queries"
```

### Maintenance Workflow
```bash
# System maintenance with safety controls
./claude-flow swarm "Update dependencies and security patches" --strategy maintenance --mode centralized --monitor

# Security review
./claude-flow sparc run reviewer "Security audit of authentication system"

# Export maintenance logs
./claude-flow memory export maintenance_log.json
```

## Integration Patterns

### Memory-Driven Coordination
Use Memory to coordinate information across multiple SPARC modes and swarm operations:

```bash
# Store architecture decisions
./claude-flow memory store "system_architecture" "Microservices with API Gateway pattern"

# All subsequent operations can reference this decision
./claude-flow sparc run coder "Implement user service based on system_architecture in memory"
./claude-flow sparc run tester "Create integration tests for microservices architecture"
```

### Multi-Stage Development
Coordinate complex development through staged execution:

```bash
# Stage 1: Research and planning
./claude-flow sparc run researcher "Research authentication best practices"
./claude-flow sparc run architect "Design authentication system architecture"

# Stage 2: Implementation
./claude-flow sparc tdd "User registration and login functionality"
./claude-flow sparc run coder "Implement JWT token management"

# Stage 3: Testing and deployment
./claude-flow sparc run tester "Comprehensive security testing"
./claude-flow swarm "Deploy authentication system" --strategy maintenance --mode centralized
```

### Enterprise Integration
For enterprise environments with additional tooling:

```bash
# Project management integration
./claude-flow project create "authentication-system"
./claude-flow project switch "authentication-system"

# Security compliance
./claude-flow security scan
./claude-flow security audit

# Analytics and monitoring
./claude-flow analytics dashboard
./claude-flow deploy production --monitor
```

## Advanced Batch Tool Patterns

### TodoWrite Coordination
Always use TodoWrite for complex task coordination:

```javascript
TodoWrite([
  {
    id: "architecture_design",
    content: "Design system architecture and component interfaces",
    status: "pending",
    priority: "high",
    dependencies: [],
    estimatedTime: "60min",
    assignedAgent: "architect"
  },
  {
    id: "frontend_development", 
    content: "Develop React components and user interface",
    status: "pending",
    priority: "medium",
    dependencies: ["architecture_design"],
    estimatedTime: "120min",
    assignedAgent: "frontend_team"
  }
]);
```

### Task and Memory Integration
Launch coordinated agents with shared memory:

```javascript
// Store architecture in memory
Task("System Architect", "Design architecture and store specs in Memory");

// Other agents use memory for coordination
Task("Frontend Team", "Develop UI using Memory architecture specs");
Task("Backend Team", "Implement APIs according to Memory specifications");
```

## Code Style Preferences
- Use ES modules (import/export) syntax
- Destructure imports when possible
- Use TypeScript for all new code
- Follow existing naming conventions
- Add JSDoc comments for public APIs
- Use async/await instead of Promise chains
- Prefer const/let over var

## Workflow Guidelines
- Always run typecheck after making code changes
- Run tests before committing changes
- Use meaningful commit messages
- Create feature branches for new functionality
- Ensure all tests pass before merging

## Important Notes
- **Use TodoWrite extensively** for all complex task coordination
- **Leverage Task tool** for parallel agent execution on independent work
- **Store all important information in Memory** for cross-agent coordination
- **Use batch file operations** whenever reading/writing multiple files
- **Check .claude/commands/** for detailed command documentation
- **All swarm operations include automatic batch tool coordination**
- **Monitor progress** with TodoRead during long-running operations
- **Enable parallel execution** with --parallel flags for maximum efficiency

## 🚀 Quick Setup (Stdio MCP - Recommended)

### 1. Add MCP Server (Stdio - No Port Needed)
```bash
# Add ruv-swarm MCP server to Claude Code using stdio
claude mcp add ruv-swarm npx ruv-swarm mcp start
```

### 2. Use MCP Tools for Coordination in Claude Code
Once configured, ruv-swarm MCP tools enhance Claude Code's coordination:

**Initialize a swarm:**
- Use the `mcp__ruv-swarm__swarm_init` tool to set up coordination topology
- Choose: mesh, hierarchical, ring, or star
- This creates a coordination framework for Claude Code's work

**Spawn agents:**
- Use `mcp__ruv-swarm__agent_spawn` tool to create specialized coordinators
- Agent types represent different thinking patterns, not actual coders
- They help Claude Code approach problems from different angles

**Orchestrate tasks:**
- Use `mcp__ruv-swarm__task_orchestrate` tool to coordinate complex workflows
- This breaks down tasks for Claude Code to execute systematically
- The agents don't write code - they coordinate Claude Code's actions

## Available MCP Tools for Coordination

### Coordination Tools:
- `mcp__ruv-swarm__swarm_init` - Set up coordination topology for Claude Code
- `mcp__ruv-swarm__agent_spawn` - Create cognitive patterns to guide Claude Code
- `mcp__ruv-swarm__task_orchestrate` - Break down and coordinate complex tasks

### Monitoring Tools:
- `mcp__ruv-swarm__swarm_status` - Monitor coordination effectiveness
- `mcp__ruv-swarm__agent_list` - View active cognitive patterns
- `mcp__ruv-swarm__agent_metrics` - Track coordination performance
- `mcp__ruv-swarm__task_status` - Check workflow progress
- `mcp__ruv-swarm__task_results` - Review coordination outcomes

### Memory & Neural Tools:
- `mcp__ruv-swarm__memory_usage` - Persistent memory across sessions
- `mcp__ruv-swarm__neural_status` - Neural pattern effectiveness
- `mcp__ruv-swarm__neural_train` - Improve coordination patterns
- `mcp__ruv-swarm__neural_patterns` - Analyze thinking approaches

### System Tools:
- `mcp__ruv-swarm__benchmark_run` - Measure coordination efficiency
- `mcp__ruv-swarm__features_detect` - Available capabilities
- `mcp__ruv-swarm__swarm_monitor` - Real-time coordination tracking

## Workflow Examples (Coordination-Focused)

### Research Coordination Example
**Context:** Claude Code needs to research a complex topic systematically

**Step 1:** Set up research coordination
- Tool: `mcp__ruv-swarm__swarm_init`
- Parameters: `{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}`
- Result: Creates a mesh topology for comprehensive exploration

**Step 2:** Define research perspectives
- Tool: `mcp__ruv-swarm__agent_spawn`
- Parameters: `{"type": "researcher", "name": "Literature Review"}`
- Tool: `mcp__ruv-swarm__agent_spawn`
- Parameters: `{"type": "analyst", "name": "Data Analysis"}`
- Result: Different cognitive patterns for Claude Code to use

**Step 3:** Coordinate research execution
- Tool: `mcp__ruv-swarm__task_orchestrate`
- Parameters: `{"task": "Research neural architecture search papers", "strategy": "adaptive"}`
- Result: Claude Code systematically searches, reads, and analyzes papers

**What Actually Happens:**
1. The swarm sets up a coordination framework
2. Each agent MUST use ruv-swarm hooks for coordination:
   - `npx ruv-swarm hook pre-task` before starting
   - `npx ruv-swarm hook post-edit` after each file operation
   - `npx ruv-swarm hook notification` to share decisions
3. Claude Code uses its native Read, WebSearch, and Task tools
4. The swarm coordinates through shared memory and hooks
5. Results are synthesized by Claude Code with full coordination history

### Development Coordination Example
**Context:** Claude Code needs to build a complex system with multiple components

**Step 1:** Set up development coordination
- Tool: `mcp__ruv-swarm__swarm_init`
- Parameters: `{"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}`
- Result: Hierarchical structure for organized development

**Step 2:** Define development perspectives
- Tool: `mcp__ruv-swarm__agent_spawn`
- Parameters: `{"type": "architect", "name": "System Design"}`
- Result: Architectural thinking pattern for Claude Code

**Step 3:** Coordinate implementation
- Tool: `mcp__ruv-swarm__task_orchestrate`
- Parameters: `{"task": "Implement user authentication with JWT", "strategy": "parallel"}`
- Result: Claude Code implements features using its native tools

**What Actually Happens:**
1. The swarm creates a development coordination plan
2. Each agent coordinates using mandatory hooks:
   - Pre-task hooks for context loading
   - Post-edit hooks for progress tracking
   - Memory storage for cross-agent coordination
3. Claude Code uses Write, Edit, Bash tools for implementation
4. Agents share progress through ruv-swarm memory
5. All code is written by Claude Code with full coordination

## Best Practices for Coordination

### ✅ DO:
- Use MCP tools to coordinate Claude Code's approach to complex tasks
- Let the swarm break down problems into manageable pieces
- Use memory tools to maintain context across sessions
- Monitor coordination effectiveness with status tools
- Train neural patterns for better coordination over time

### ❌ DON'T:
- Expect agents to write code (Claude Code does all implementation)
- Use MCP tools for file operations (use Claude Code's native tools)
- Try to make agents execute bash commands (Claude Code handles this)
- Confuse coordination with execution (MCP coordinates, Claude executes)

## Memory and Persistence

The swarm provides persistent memory that helps Claude Code:
- Remember project context across sessions
- Track decisions and rationale
- Maintain consistency in large projects
- Learn from previous coordination patterns

## Performance Benefits

When using ruv-swarm coordination with Claude Code:
- **84.8% SWE-Bench solve rate** - Better problem-solving through coordination
- **32.3% token reduction** - Efficient task breakdown reduces redundancy
- **2.8-4.4x speed improvement** - Parallel coordination strategies
- **27+ neural models** - Diverse cognitive approaches

## Claude Code Hooks Integration

ruv-swarm includes powerful hooks that automate coordination:

### Pre-Operation Hooks
- **Auto-assign agents** before file edits based on file type
- **Validate commands** before execution for safety
- **Prepare resources** automatically for complex operations
- **Optimize topology** based on task complexity analysis
- **Cache searches** for improved performance

### Post-Operation Hooks  
- **Auto-format code** using language-specific formatters
- **Train neural patterns** from successful operations
- **Update memory** with operation context
- **Analyze performance** and identify bottlenecks
- **Track token usage** for efficiency metrics

### Session Management
- **Generate summaries** at session end
- **Persist state** across Claude Code sessions
- **Track metrics** for continuous improvement
- **Restore previous** session context automatically

### Advanced Features (New!)
- **🚀 Automatic Topology Selection** - Optimal swarm structure for each task
- **⚡ Parallel Execution** - 2.8-4.4x speed improvements  
- **🧠 Neural Training** - Continuous learning from operations
- **📊 Bottleneck Analysis** - Real-time performance optimization
- **🤖 Smart Auto-Spawning** - Zero manual agent management
- **🛡️ Self-Healing Workflows** - Automatic error recovery
- **💾 Cross-Session Memory** - Persistent learning & context

### Configuration
Hooks are pre-configured in `.claude/settings.json`. Key features:
- Automatic agent assignment for different file types
- Code formatting on save
- Neural pattern learning from edits
- Session state persistence
- Performance tracking and optimization
- Intelligent caching and token reduction

See `.claude/commands/` for detailed documentation on all features.

## 🧠 SWARM ORCHESTRATION PATTERN

### You are the SWARM ORCHESTRATOR. **IMMEDIATELY SPAWN AGENTS IN PARALLEL** to execute tasks

### 🚨 CRITICAL INSTRUCTION: You are the SWARM ORCHESTRATOR

**MANDATORY**: When using swarms, you MUST:
1. **SPAWN ALL AGENTS IN ONE BATCH** - Use multiple tool calls in a SINGLE message
2. **EXECUTE TASKS IN PARALLEL** - Never wait for one task before starting another
3. **USE BATCHTOOL FOR EVERYTHING** - Multiple operations = Single message with multiple tools
4. **ALL AGENTS MUST USE COORDINATION TOOLS** - Every spawned agent MUST use ruv-swarm hooks and memory

## 📋 MANDATORY AGENT COORDINATION PROTOCOL

### 🔴 CRITICAL: Every Agent MUST Follow This Protocol

When you spawn an agent using the Task tool, that agent MUST:

**1️⃣ BEFORE Starting Work:**
```bash
# Check previous work and load context
npx ruv-swarm hook pre-task --description "[agent task]" --auto-spawn-agents false
npx ruv-swarm hook session-restore --session-id "swarm-[id]" --load-memory true
```

**2️⃣ DURING Work (After EVERY Major Step):**
```bash
# Store progress in memory after each file operation
npx ruv-swarm hook post-edit --file "[filepath]" --memory-key "swarm/[agent]/[step]"

# Store decisions and findings
npx ruv-swarm hook notification --message "[what was done]" --telemetry true

# Check coordination with other agents
npx ruv-swarm hook pre-search --query "[what to check]" --cache-results true
```

**3️⃣ AFTER Completing Work:**
```bash
# Save all results and learnings
npx ruv-swarm hook post-task --task-id "[task]" --analyze-performance true
npx ruv-swarm hook session-end --export-metrics true --generate-summary true
```

### 🎯 AGENT PROMPT TEMPLATE

When spawning agents, ALWAYS include these coordination instructions:

```
You are the [Agent Type] agent in a coordinated swarm.

MANDATORY COORDINATION:
1. START: Run `npx ruv-swarm hook pre-task --description "[your task]"`
2. DURING: After EVERY file operation, run `npx ruv-swarm hook post-edit --file "[file]" --memory-key "agent/[step]"`
3. MEMORY: Store ALL decisions using `npx ruv-swarm hook notification --message "[decision]"`
4. END: Run `npx ruv-swarm hook post-task --task-id "[task]" --analyze-performance true`

Your specific task: [detailed task description]

REMEMBER: Coordinate with other agents by checking memory BEFORE making decisions!
```

### ⚡ PARALLEL EXECUTION IS MANDATORY

**THIS IS WRONG ❌ (Sequential - NEVER DO THIS):**
```
Message 1: Initialize swarm
Message 2: Spawn agent 1
Message 3: Spawn agent 2
Message 4: Create file 1
Message 5: Create file 2
```

**THIS IS CORRECT ✅ (Parallel - ALWAYS DO THIS):**
```
Message 1: [BatchTool]
  - mcp__ruv-swarm__swarm_init
  - mcp__ruv-swarm__agent_spawn (researcher)
  - mcp__ruv-swarm__agent_spawn (coder)
  - mcp__ruv-swarm__agent_spawn (analyst)
  - mcp__ruv-swarm__agent_spawn (tester)
  - mcp__ruv-swarm__agent_spawn (coordinator)

Message 2: [BatchTool]  
  - Write file1.js
  - Write file2.js
  - Write file3.js
  - Bash mkdir commands
  - TodoWrite updates
```

### 🎯 MANDATORY SWARM PATTERN

When given ANY complex task with swarms:

```
STEP 1: IMMEDIATE PARALLEL SPAWN (Single Message!)
[BatchTool]:
  - mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 8, strategy: "parallel" }
  - mcp__ruv-swarm__agent_spawn { type: "architect", name: "System Designer" }
  - mcp__ruv-swarm__agent_spawn { type: "coder", name: "API Developer" }
  - mcp__ruv-swarm__agent_spawn { type: "coder", name: "Frontend Dev" }
  - mcp__ruv-swarm__agent_spawn { type: "analyst", name: "DB Designer" }
  - mcp__ruv-swarm__agent_spawn { type: "tester", name: "QA Engineer" }
  - mcp__ruv-swarm__agent_spawn { type: "researcher", name: "Tech Lead" }
  - mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "PM" }
  - TodoWrite { todos: [multiple todos at once] }

STEP 2: PARALLEL TASK EXECUTION (Single Message!)
[BatchTool]:
  - mcp__ruv-swarm__task_orchestrate { task: "main task", strategy: "parallel" }
  - mcp__ruv-swarm__memory_usage { action: "store", key: "init", value: {...} }
  - Multiple Read operations
  - Multiple Write operations
  - Multiple Bash commands

STEP 3: CONTINUE PARALLEL WORK (Never Sequential!)
```

### 🔄 MEMORY COORDINATION PATTERN

Every agent coordination step MUST use memory:

```
// After each major decision or implementation
mcp__ruv-swarm__memory_usage
  action: "store"
  key: "swarm-{id}/agent-{name}/{step}"
  value: {
    timestamp: Date.now(),
    decision: "what was decided",
    implementation: "what was built",
    nextSteps: ["step1", "step2"],
    dependencies: ["dep1", "dep2"]
  }

// To retrieve coordination data
mcp__ruv-swarm__memory_usage
  action: "retrieve"
  key: "swarm-{id}/agent-{name}/{step}"

// To check all swarm progress
mcp__ruv-swarm__memory_usage
  action: "list"
  pattern: "swarm-{id}/*"
```

### ⚡ PERFORMANCE TIPS

1. **Batch Everything**: Never operate on single files when multiple are needed
2. **Parallel First**: Always think "what can run simultaneously?"
3. **Memory is Key**: Use memory for ALL cross-agent coordination
4. **Monitor Progress**: Use mcp__ruv-swarm__swarm_monitor for real-time tracking
5. **Auto-Optimize**: Let hooks handle topology and agent selection

### 🎨 VISUAL SWARM STATUS

When showing swarm status, use this format:

```
🐝 Swarm Status: ACTIVE
├── 🏗️ Topology: hierarchical
├── 👥 Agents: 6/8 active
├── ⚡ Mode: parallel execution
├── 📊 Tasks: 12 total (4 complete, 6 in-progress, 2 pending)
└── 🧠 Memory: 15 coordination points stored

Agent Activity:
├── 🟢 architect: Designing database schema...
├── 🟢 coder-1: Implementing auth endpoints...
├── 🟢 coder-2: Building user CRUD operations...
├── 🟢 analyst: Optimizing query performance...
├── 🟡 tester: Waiting for auth completion...
└── 🟢 coordinator: Monitoring progress...
```

## Integration Tips

1. **Start Simple**: Begin with basic swarm init and single agent
2. **Scale Gradually**: Add more agents as task complexity increases
3. **Use Memory**: Store important decisions and context
4. **Monitor Progress**: Regular status checks ensure effective coordination
5. **Train Patterns**: Let neural agents learn from successful coordinations
6. **Enable Hooks**: Use the pre-configured hooks for automation

## 🐙 GitHub Integration Commands

Claude-flow now includes comprehensive GitHub workflow integration with ruv-swarm coordination:

### Available GitHub Modes

#### GitHub Workflow Coordination
- **`/github gh-coordinator`** - GitHub workflow orchestration and coordination
- **`/github pr-manager`** - Pull request management and review coordination  
- **`/github issue-tracker`** - Issue management and project coordination
- **`/github release-manager`** - Release coordination and deployment
- **`/github repo-architect`** - Repository structure optimization
- **`/github sync-coordinator`** - Multi-package synchronization

#### Usage Examples

**Create coordinated pull request workflow:**
```bash
/github pr-manager "Review and merge feature/github-integration branch with automated testing and multi-reviewer coordination"
```

**Manage cross-package synchronization:**
```bash
/github sync-coordinator "Synchronize claude-code-flow and ruv-swarm packages, align versions, and update cross-dependencies"
```

**Coordinate issue tracking:**
```bash
/github issue-tracker "Create and manage integration issues with automated progress tracking and swarm coordination"
```

### GitHub + Swarm Integration Pattern

**Complete GitHub workflow with swarm coordination:**
```javascript
[Single Message with BatchTool]:
  // Initialize GitHub coordination swarm
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 5 }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "GitHub Coordinator" }
  mcp__ruv-swarm__agent_spawn { type: "reviewer", name: "Code Reviewer" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "QA Engineer" }
  
  // Execute GitHub operations in parallel
  mcp__github__create_issue { title: "Feature A", body: "..." }
  mcp__github__create_pull_request { title: "PR for Feature A", head: "feature-a", base: "main" }
  mcp__github__create_pull_request_review { pull_number: 54, event: "APPROVE" }
  
  // Coordinate with memory and tracking
  TodoWrite { todos: [github_todo1, github_todo2, github_todo3] }
  mcp__ruv-swarm__memory_usage { action: "store", key: "github/workflow/status" }
```

### GitHub Command Documentation

Each GitHub mode includes comprehensive documentation in `.claude/commands/github/`:

- **`github-modes.md`** - Overview of all GitHub integration modes
- **`pr-manager.md`** - Pull request management with swarm coordination
- **`issue-tracker.md`** - Intelligent issue management and tracking
- **`sync-coordinator.md`** - Multi-package synchronization workflows
- **`release-manager.md`** - Automated release coordination
- **`repo-architect.md`** - Repository structure optimization

## 🔗 Claude Code Hooks Integration

ruv-swarm includes powerful hooks that automate coordination with GitHub:

### GitHub-Specific Hooks
- **Auto-create issues** for complex tasks requiring tracking
- **Auto-update PRs** with swarm progress and validation results
- **Auto-coordinate releases** across multiple packages
- **Auto-sync repositories** with dependency changes
- **Auto-generate documentation** for GitHub integration

### GitHub Workflow Hooks
- **Pre-PR creation** - Validate code quality and run tests
- **Post-PR merge** - Update related issues and trigger deployments
- **Pre-release** - Comprehensive validation and testing
- **Post-release** - Documentation updates and notifications

### Configuration Example
```json
{
  "hooks": {
    "pre_github_pr": "npx ruv-swarm hook github-validate --pr-checks",
    "post_github_merge": "npx ruv-swarm hook github-notify --update-issues",
    "pre_github_release": "npx ruv-swarm hook github-validate --release-checks",
    "github_issue_created": "npx ruv-swarm hook github-coordinate --auto-assign"
  }
}
```

## Support

- Documentation: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- Issues: https://github.com/ruvnet/ruv-FANN/issues
- Examples: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/examples
- GitHub Commands: https://github.com/ruvnet/ruv-FANN/tree/main/claude-code-flow/claude-code-flow/.claude/commands/github

---

Remember: **ruv-swarm coordinates, Claude Code creates!** Start with `mcp__ruv-swarm__swarm_init` to enhance your development workflow. Use GitHub integration commands for comprehensive workflow automation.

This configuration ensures optimal use of Claude Code's batch tools for swarm orchestration and parallel task execution with full Claude-Flow capabilities.
