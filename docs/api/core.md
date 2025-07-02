# ruv-swarm Core API Reference

Complete API reference for the ruv-swarm core functionality.

## 🦀 Rust API

### Core Types

#### Swarm
```rust
pub struct Swarm {
    pub id: String,
    pub topology: Topology,
    pub agents: Vec<Agent>,
}

impl Swarm {
    pub fn new(id: String, topology: Topology) -> Self;
    pub async fn spawn_agent(&mut self, agent_type: AgentType) -> Result<AgentId>;
    pub async fn orchestrate(&self, strategy: Strategy, task: &str) -> Result<TaskResult>;
    pub fn status(&self) -> SwarmStatus;
}
```

#### Agent
```rust
pub struct Agent {
    pub id: AgentId,
    pub agent_type: AgentType,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
}

impl Agent {
    pub async fn execute(&self, task: &Task) -> Result<TaskResult>;
    pub fn get_capabilities(&self) -> &[String];
    pub fn is_available(&self) -> bool;
}
```

#### Task
```rust
pub struct Task {
    pub id: TaskId,
    pub description: String,
    pub priority: Priority,
    pub dependencies: Vec<TaskId>,
}

impl Task {
    pub fn new(description: String) -> Self;
    pub fn with_priority(mut self, priority: Priority) -> Self;
    pub fn with_dependencies(mut self, deps: Vec<TaskId>) -> Self;
}
```

### Enums

#### Topology
```rust
pub enum Topology {
    Mesh,
    Hierarchical,
    Ring,
    Star,
}
```

#### AgentType
```rust
pub enum AgentType {
    Researcher,
    Coder,
    Analyst,
    Tester,
    Coordinator,
}
```

#### Strategy
```rust
pub enum Strategy {
    Parallel,
    Sequential,
    Adaptive,
}
```

## 🟢 Node.js API

### Core Classes

#### RuvSwarm
```javascript
class RuvSwarm {
    constructor(options = {})
    
    // Swarm Management
    async initialize(topology = 'mesh')
    async spawn(agentType, options = {})
    async orchestrate(strategy, task, options = {})
    
    // Status & Monitoring
    getStatus()
    getAgents()
    getMetrics()
    
    // Utility
    static detectSIMDSupport()
    static getVersion()
}
```

#### SwarmAgent
```javascript
class SwarmAgent {
    constructor(id, type, capabilities)
    
    // Task Execution
    async execute(task)
    async collaborate(otherAgents, task)
    
    // State Management
    getStatus()
    getCapabilities()
    isAvailable()
}
```

### Usage Examples

#### Basic Swarm Creation
```javascript
import { RuvSwarm } from 'ruv-swarm';

// Initialize swarm
const swarm = new RuvSwarm({
    topology: 'mesh',
    maxAgents: 5
});

await swarm.initialize();

// Spawn agents
const researcher = await swarm.spawn('researcher', {
    name: 'data-analyst',
    capabilities: ['data_analysis', 'research', 'documentation']
});

const coder = await swarm.spawn('coder', {
    name: 'backend-dev', 
    capabilities: ['python', 'rust', 'api_development']
});

// Orchestrate task
const result = await swarm.orchestrate('parallel', 
    'Build a REST API with authentication', {
    timeout: 1800000, // 30 minutes
    priority: 'high'
});

console.log('Task completed:', result);
```

#### Advanced Agent Coordination
```javascript
// Create specialized swarm
const devSwarm = new RuvSwarm({ topology: 'hierarchical' });
await devSwarm.initialize();

// Spawn coordinated team
const architect = await devSwarm.spawn('coordinator', {
    name: 'system-architect',
    role: 'lead'
});

const frontend = await devSwarm.spawn('coder', {
    name: 'frontend-dev',
    specialization: 'react'
});

const backend = await devSwarm.spawn('coder', {
    name: 'backend-dev', 
    specialization: 'node'
});

const tester = await devSwarm.spawn('tester', {
    name: 'qa-engineer',
    focus: 'integration'
});

// Complex task orchestration
const project = await devSwarm.orchestrate('adaptive', 
    'Build a full-stack e-commerce application', {
    subtasks: [
        'Design system architecture',
        'Implement user authentication', 
        'Create product catalog',
        'Build shopping cart',
        'Add payment processing',
        'Write comprehensive tests'
    ],
    coordination: 'hierarchical',
    monitoring: true
});
```

## 🔧 Configuration API

### Environment Configuration
```javascript
// Configure via environment
process.env.SWARM_TOPOLOGY = 'mesh';
process.env.SWARM_MAX_AGENTS = '10';
process.env.SWARM_PERSISTENCE = 'sqlite';

// Or via configuration object
const config = {
    swarm: {
        topology: 'mesh',
        maxAgents: 10,
        persistence: {
            backend: 'sqlite',
            database: './swarm.db'
        }
    },
    mcp: {
        github: {
            enabled: true,
            token: process.env.GITHUB_TOKEN
        },
        ruvSwarm: {
            enabled: true,
            port: 3000
        }
    }
};

const swarm = new RuvSwarm(config);
```

### Runtime Configuration
```javascript
// Update configuration at runtime
await swarm.updateConfig({
    maxAgents: 15,
    topology: 'hierarchical'
});

// Scale agents dynamically
await swarm.scale(8); // Scale to 8 agents

// Change orchestration strategy
await swarm.setStrategy('adaptive');
```

## 📊 Monitoring & Metrics API

### Status Monitoring
```javascript
// Get current status
const status = swarm.getStatus();
console.log('Active agents:', status.activeAgents);
console.log('Queued tasks:', status.queuedTasks);
console.log('Completion rate:', status.completionRate);

// Real-time monitoring
swarm.on('agentStatusChange', (agent, status) => {
    console.log(`Agent ${agent.name} is now ${status}`);
});

swarm.on('taskComplete', (task, result) => {
    console.log(`Task "${task.description}" completed`);
});
```

### Performance Metrics
```javascript
// Get performance metrics
const metrics = await swarm.getMetrics();
console.log('Average task time:', metrics.averageTaskTime);
console.log('Success rate:', metrics.successRate);
console.log('Resource utilization:', metrics.resourceUtilization);

// Export metrics for monitoring systems
const prometheus = await swarm.exportMetrics('prometheus');
const datadog = await swarm.exportMetrics('datadog');
```

## 🔗 Integration APIs

### MCP Integration
```javascript
// Configure MCP servers
await swarm.configureMCP({
    github: {
        enabled: true,
        token: process.env.GITHUB_TOKEN
    },
    ruvSwarm: {
        enabled: true,
        endpoint: 'http://localhost:3000'
    }
});

// Launch Claude Code with MCP
await swarm.launchClaude({
    mcpConfig: './.claude/mcp.json',
    autoLogin: true
});
```

### Event System
```javascript
// Subscribe to events
swarm.on('swarmInitialized', () => {
    console.log('Swarm ready for tasks');
});

swarm.on('agentSpawned', (agent) => {
    console.log(`New agent: ${agent.name} (${agent.type})`);
});

swarm.on('taskOrchestrated', (task) => {
    console.log(`Task started: ${task.description}`);
});

swarm.on('error', (error) => {
    console.error('Swarm error:', error);
});
```

## 🛠️ Error Handling

### Error Types
```javascript
import { SwarmError, AgentError, TaskError } from 'ruv-swarm';

try {
    await swarm.orchestrate('invalid-strategy', 'task');
} catch (error) {
    if (error instanceof SwarmError) {
        console.log('Swarm configuration error:', error.message);
    } else if (error instanceof AgentError) {
        console.log('Agent execution error:', error.agent, error.message);
    } else if (error instanceof TaskError) {
        console.log('Task failed:', error.task, error.message);
    }
}
```

### Recovery Strategies
```javascript
// Automatic retry with backoff
const result = await swarm.orchestrate('parallel', task, {
    retry: {
        attempts: 3,
        backoff: 'exponential',
        initialDelay: 1000
    }
});

// Graceful degradation
const result = await swarm.orchestrate('adaptive', task, {
    fallback: 'sequential',
    partialSuccess: true
});
```

## 🔍 Debugging & Development

### Debug Mode
```javascript
// Enable debug logging
const swarm = new RuvSwarm({
    debug: true,
    logLevel: 'verbose'
});

// Debug specific components
swarm.enableDebug(['orchestration', 'agent-communication']);
```

### Testing Utilities
```javascript
// Create test swarm
const testSwarm = RuvSwarm.createTestSwarm({
    mockAgents: true,
    isolateNetworking: true
});

// Mock agent responses
testSwarm.mockAgent('researcher', {
    execute: async (task) => ({ 
        result: 'mocked research result',
        duration: 1000 
    })
});
```

---

For more examples and advanced usage, see the [guides](../guides/) and [examples](../../examples/) directories.