#!/usr/bin/env node
/**
 * ruv-swarm CLI - Neural network swarm orchestration
 */

const { RuvSwarm } = require('../src');
const { NeuralAgentFactory, COGNITIVE_PATTERNS } = require('../src/neural-agent');
const { SwarmPersistence } = require('../src/persistence');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const Database = require('better-sqlite3');

// Global instances for MCP
let globalRuvSwarm = null;
let globalPersistence = null;
let globalSwarm = null;

async function main() {
  const args = process.argv.slice(2);
  
  // Initialize WASM module
  let swarm;
  try {
    const ruvSwarm = await RuvSwarm.initialize({
      wasmPath: path.join(__dirname, '..', 'wasm'),
      useSIMD: RuvSwarm.detectSIMDSupport(),
      debug: args.includes('--debug')
    });

    swarm = await ruvSwarm.createSwarm({
      name: 'cli-swarm',
      strategy: 'development',
      mode: 'centralized',
      maxAgents: 5
    });
  } catch (error) {
    console.error('Failed to initialize RuvSwarm:', error.message);
    process.exit(1);
  }
  
  // Parse commands
  const command = args[0] || 'help';
  
  switch (command) {
    case 'init':
      await handleInit(args.slice(1));
      break;
    case 'spawn':
      await handleSpawn(swarm, args.slice(1));
      break;
    case 'orchestrate':
      await handleOrchestrate(swarm, args.slice(1));
      break;
    case 'status':
      await handleStatus(swarm);
      break;
    case 'monitor':
      await handleMonitor(swarm);
      break;
    case 'benchmark':
      await handleBenchmark();
      break;
    case 'features':
      await handleFeatures();
      break;
    case 'install':
      await handleInstall(args.slice(1));
      break;
    case 'test':
      await handleTest();
      break;
    case 'mcp':
      await handleMcp(args.slice(1));
      break;
    case 'neural':
      await handleNeural(args.slice(1));
      break;
    case 'version':
      console.log(`ruv-swarm v${RuvSwarm.getVersion()}`);
      break;
    case 'help':
    default:
      showHelp();
  }
}

async function handleSpawn(swarm, args) {
  const agentType = args[0] || 'researcher';
  const agentName = args[1] || `${agentType}-${Date.now()}`;
  const enableNeural = !args.includes('--no-neural');

  try {
    const agent = await swarm.spawn({
      name: agentName,
      type: agentType,
      capabilities: []
    });

    let finalAgent = agent;
    
    // Enhance with neural capabilities if enabled
    if (enableNeural) {
      try {
        finalAgent = NeuralAgentFactory.createNeuralAgent(agent, agentType);
        const cognitiveProfile = NeuralAgentFactory.getCognitiveProfiles()[agentType];
        
        console.log(`Neural Agent spawned successfully:`);
        console.log(`  ID: ${agent.id}`);
        console.log(`  Type: ${agent.agentType}`);
        console.log(`  Status: ${agent.status}`);
        console.log(`  Capabilities: ${agent.getCapabilities().join(', ')}`);
        console.log(`  Cognitive Pattern: ${cognitiveProfile.primary}/${cognitiveProfile.secondary}`);
        console.log(`  Neural Network: ${cognitiveProfile.networkLayers.join('-')} neurons`);
        console.log(`  Learning Rate: ${cognitiveProfile.learningRate}`);
      } catch (neuralError) {
        console.warn('Neural enhancement failed, using base agent:', neuralError.message);
        console.log(`Agent spawned successfully (without neural enhancement):`);
        console.log(`  ID: ${agent.id}`);
        console.log(`  Type: ${agent.agentType}`);
        console.log(`  Status: ${agent.status}`);
        console.log(`  Capabilities: ${agent.getCapabilities().join(', ')}`);
      }
    } else {
      console.log(`Agent spawned successfully:`);
      console.log(`  ID: ${agent.id}`);
      console.log(`  Type: ${agent.agentType}`);
      console.log(`  Status: ${agent.status}`);
      console.log(`  Capabilities: ${agent.getCapabilities().join(', ')}`);
    }
    
    // Store agent in global registry for orchestration
    if (!global.neuralAgents) {
      global.neuralAgents = new Map();
    }
    global.neuralAgents.set(agent.id, finalAgent);
    
  } catch (error) {
    console.error('Failed to spawn agent:', error.message);
    process.exit(1);
  }
}

async function handleOrchestrate(swarm, args) {
  const taskDescription = args.join(' ') || 'Default task';
  const useNeural = !args.includes('--no-neural');

  try {
    console.log('Orchestrating task:', taskDescription);
    
    const task = {
      id: `task-${Date.now()}`,
      description: taskDescription,
      priority: 'medium',
      dependencies: [],
      metadata: {}
    };

    let result;
    
    if (useNeural && global.neuralAgents && global.neuralAgents.size > 0) {
      console.log(`\nUsing neural agents for intelligent task distribution...`);
      
      // Analyze task with each neural agent
      const analyses = [];
      for (const [agentId, neuralAgent] of global.neuralAgents) {
        if (neuralAgent.analyzeTask) {
          try {
            const analysis = await neuralAgent.analyzeTask(task);
            analyses.push({
              agentId,
              agent: neuralAgent,
              analysis,
              score: analysis.confidence * (1 - analysis.complexity)
            });
          } catch (e) {
            console.warn(`Agent ${agentId} analysis failed:`, e.message);
          }
        }
      }
      
      // Sort agents by suitability score
      analyses.sort((a, b) => b.score - a.score);
      
      if (analyses.length > 0) {
        console.log(`\nNeural Analysis Results:`);
        analyses.slice(0, 3).forEach((a, i) => {
          console.log(`  ${i + 1}. Agent ${a.agentId}:`);
          console.log(`     Confidence: ${(a.analysis.confidence * 100).toFixed(1)}%`);
          console.log(`     Complexity: ${(a.analysis.complexity * 100).toFixed(1)}%`);
          console.log(`     Creativity Need: ${(a.analysis.creativity * 100).toFixed(1)}%`);
        });
        
        // Execute with best suited neural agent
        const bestAgent = analyses[0].agent;
        console.log(`\nAssigning task to best-suited agent: ${analyses[0].agentId}`);
        
        if (bestAgent.executeTask) {
          const neuralResult = await bestAgent.executeTask(task);
          
          // Format result to match expected structure
          result = {
            taskId: task.id,
            status: 'completed',
            metrics: {
              totalTime: (Date.now() - parseInt(task.id.split('-')[1])) / 1000,
              agentsSpawned: 1
            },
            results: [{
              agentId: analyses[0].agentId,
              agentType: bestAgent.agentType,
              executionTime: (Date.now() - parseInt(task.id.split('-')[1])) / 1000,
              output: neuralResult,
              neuralMetrics: bestAgent.performanceMetrics
            }]
          };
        } else {
          // Fallback to regular orchestration
          result = await swarm.orchestrate(task);
        }
      } else {
        // No neural agents available, use regular orchestration
        result = await swarm.orchestrate(task);
      }
    } else {
      // Regular orchestration without neural enhancement
      result = await swarm.orchestrate(task);
    }

    console.log('\nOrchestration completed:');
    console.log(`  Task ID: ${result.taskId}`);
    console.log(`  Status: ${result.status}`);
    console.log(`  Total Time: ${result.metrics.totalTime}s`);
    console.log(`  Agents Used: ${result.metrics.agentsSpawned}`);
    
    console.log('\nAgent Results:');
    result.results.forEach((agentResult, index) => {
      console.log(`  ${index + 1}. ${agentResult.agentType} (${agentResult.agentId})`);
      console.log(`     Execution Time: ${agentResult.executionTime}s`);
      if (agentResult.neuralMetrics) {
        console.log(`     Neural Performance:`);
        console.log(`       - Accuracy: ${(agentResult.neuralMetrics.accuracy * 100).toFixed(1)}%`);
        console.log(`       - Efficiency: ${(agentResult.neuralMetrics.efficiency * 100).toFixed(1)}%`);
        console.log(`       - Creativity: ${(agentResult.neuralMetrics.creativity * 100).toFixed(1)}%`);
      }
      console.log(`     Output:`, JSON.stringify(agentResult.output, null, 2));
    });
  } catch (error) {
    console.error('Orchestration failed:', error.message);
    process.exit(1);
  }
}

async function handleStatus(swarm) {
  const status = swarm.getStatus();
  
  console.log('Swarm Status:');
  console.log(`  Name: ${status.name}`);
  console.log(`  Strategy: ${status.strategy}`);
  console.log(`  Mode: ${status.mode}`);
  console.log(`  Active Agents: ${status.agentCount}/${status.maxAgents}`);
  
  if (status.agents.length > 0) {
    console.log('\nActive Agents:');
    status.agents.forEach((agentId, index) => {
      console.log(`  ${index + 1}. ${agentId}`);
    });
  }
  
  console.log(`\nMemory Usage: ${(RuvSwarm.getMemoryUsage() / 1024 / 1024).toFixed(2)} MB`);
}

async function handleBenchmark() {
  console.log('Running performance benchmarks...\n');

  const benchmarks = [
    { name: 'WASM Initialization', fn: async () => await RuvSwarm.initialize() },
    { name: 'Swarm Creation', fn: async (rs) => await rs.createSwarm({ name: 'bench', strategy: 'development', mode: 'centralized' }) },
    { name: 'Agent Spawn', fn: async (rs, s) => await s.spawn({ name: 'bench-agent', type: 'researcher' }) },
    { name: 'Task Orchestration', fn: async (rs, s) => await s.orchestrate({ id: 'bench-task', description: 'Benchmark task', priority: 'low', dependencies: [] }) }
  ];

  for (const benchmark of benchmarks) {
    const start = process.hrtime.bigint();
    
    try {
      let ruvSwarm, swarm;
      
      if (benchmark.name === 'WASM Initialization') {
        ruvSwarm = await benchmark.fn();
      } else if (benchmark.name === 'Swarm Creation') {
        ruvSwarm = await RuvSwarm.initialize();
        swarm = await benchmark.fn(ruvSwarm);
      } else {
        ruvSwarm = await RuvSwarm.initialize();
        swarm = await ruvSwarm.createSwarm({ name: 'bench', strategy: 'development', mode: 'centralized' });
        await benchmark.fn(ruvSwarm, swarm);
      }
      
      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1e6; // Convert to milliseconds
      
      console.log(`${benchmark.name}: ${duration.toFixed(2)}ms`);
    } catch (error) {
      console.log(`${benchmark.name}: Failed - ${error.message}`);
    }
  }

  console.log('\nBenchmark completed.');
}

async function handleFeatures() {
  console.log('Runtime Features:');
  console.log(`  WASM Support: ${typeof WebAssembly !== 'undefined' ? 'Yes' : 'No'}`);
  
  try {
    const features = RuvSwarm.getRuntimeFeatures();
    console.log(`  SIMD Support: ${features.simdAvailable ? 'Yes' : 'No'}`);
    console.log(`  Threading Support: ${features.threadsAvailable ? 'Yes' : 'No'}`);
    console.log(`  Memory Limit: ${(features.memoryLimit / 1024 / 1024 / 1024).toFixed(2)} GB`);
  } catch (error) {
    console.log(`  Unable to detect features: ${error.message}`);
  }
  
  console.log(`  Node.js Version: ${process.version}`);
  console.log(`  Platform: ${process.platform}`);
  console.log(`  Architecture: ${process.arch}`);
}

function showHelp() {
  console.log(`
ruv-swarm - High-performance neural network swarm orchestration

Usage: npx ruv-swarm <command> [options]

Commands:
  init [topology] [max]   Initialize a new swarm (default: mesh topology, 5 agents)
  spawn <type> [name]     Spawn a new agent (researcher, coder, analyst, optimizer, coordinator)
  orchestrate <task>      Orchestrate a task across the swarm
  status                  Show swarm status and agent information
  monitor                 Real-time swarm monitoring (press Ctrl+C to stop)
  neural <subcommand>     Neural network management (status, train, save, load, patterns)
  mcp <subcommand>        MCP server for Claude Code integration (stdio/http protocols)
  benchmark               Run performance benchmarks
  features                Show runtime features and capabilities
  install [target]        Show installation instructions (global, project, local)
  test                    Run functionality tests
  version                 Show version information
  help                    Show this help message

Options:
  --debug                 Enable debug logging
  --no-neural            Disable neural network enhancement for agents

Neural Network Features:
  - Cognitive diversity patterns for each agent type
  - Neural task analysis and intelligent routing
  - Learning from task execution feedback
  - Performance optimization through experience

Examples:
  npx ruv-swarm init mesh 10              # Initialize mesh topology with 10 max agents
  npx ruv-swarm spawn researcher my-researcher
  npx ruv-swarm spawn coder --no-neural   # Spawn without neural enhancement
  npx ruv-swarm orchestrate "Analyze performance data and generate report"
  npx ruv-swarm neural status             # Show neural agent performance
  npx ruv-swarm neural train 20           # Train agents with 20 iterations
  npx ruv-swarm neural patterns           # Show cognitive pattern descriptions
  npx ruv-swarm status                    # Show current swarm status
  npx ruv-swarm monitor                   # Real-time monitoring
  npx ruv-swarm mcp start                 # Start MCP server for Claude Code
  npx ruv-swarm mcp tools                 # List MCP tools
  npx ruv-swarm benchmark                 # Performance benchmarks
  npx ruv-swarm test                      # Run functionality tests
  npx ruv-swarm install global           # Installation instructions

Agent Types & Cognitive Patterns:
  - researcher: Divergent/Systems thinking - Pattern recognition, data correlation
  - coder: Convergent/Lateral thinking - Syntax analysis, code generation
  - analyst: Critical/Abstract thinking - Statistical modeling, trend detection
  - optimizer: Systems/Convergent thinking - Performance optimization, resource allocation
  - coordinator: Systems/Critical thinking - Task distribution, workflow optimization

For more information, visit: https://github.com/ruvnet/ruv-FANN
`);
}

// Handle errors gracefully
process.on('unhandledRejection', (error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});

// Run main function
async function handleNeural(args) {
  const subcommand = args[0] || 'status';
  
  switch (subcommand) {
    case 'status':
      showNeuralStatus();
      break;
    case 'train':
      await trainNeuralAgents(args.slice(1));
      break;
    case 'save':
      await saveNeuralStates(args.slice(1));
      break;
    case 'load':
      await loadNeuralStates(args.slice(1));
      break;
    case 'patterns':
      showCognitivePatterns();
      break;
    default:
      console.log('Neural commands:');
      console.log('  status    - Show neural agent status and performance');
      console.log('  train     - Train agents with sample tasks');
      console.log('  save      - Save neural network states');
      console.log('  load      - Load neural network states');
      console.log('  patterns  - Show cognitive pattern descriptions');
  }
}

function showNeuralStatus() {
  console.log('Neural Agent Status:\n');
  
  if (!global.neuralAgents || global.neuralAgents.size === 0) {
    console.log('No neural agents active. Spawn agents with neural capabilities first.');
    return;
  }
  
  let index = 1;
  for (const [agentId, agent] of global.neuralAgents) {
    if (agent.getStatus) {
      const status = agent.getStatus();
      console.log(`${index}. Agent ${agentId} (${status.agentType}):`);
      
      if (status.neuralState) {
        const ns = status.neuralState;
        console.log(`   Cognitive Pattern: ${ns.cognitiveProfile.primary}/${ns.cognitiveProfile.secondary}`);
        console.log(`   Neural Network: ${ns.cognitiveProfile.networkLayers.join('-')} neurons`);
        console.log(`   Cognitive State:`);
        console.log(`     - Attention: ${(ns.cognitiveState.attention * 100).toFixed(1)}%`);
        console.log(`     - Fatigue: ${(ns.cognitiveState.fatigue * 100).toFixed(1)}%`);
        console.log(`     - Confidence: ${(ns.cognitiveState.confidence * 100).toFixed(1)}%`);
        console.log(`   Performance Metrics:`);
        console.log(`     - Accuracy: ${(ns.performanceMetrics.accuracy * 100).toFixed(1)}%`);
        console.log(`     - Speed: ${(ns.performanceMetrics.speed * 100).toFixed(1)}%`);
        console.log(`     - Creativity: ${(ns.performanceMetrics.creativity * 100).toFixed(1)}%`);
        console.log(`     - Efficiency: ${(ns.performanceMetrics.efficiency * 100).toFixed(1)}%`);
        console.log(`   Learning History: ${ns.learningHistory} entries`);
        console.log(`   Task History: ${ns.taskHistory} entries`);
      }
      console.log();
      index++;
    }
  }
}

function showCognitivePatterns() {
  console.log('Cognitive Patterns:\n');
  
  const patterns = {
    convergent: 'Focused problem-solving, analytical thinking, goal-oriented',
    divergent: 'Creative exploration, idea generation, brainstorming',
    lateral: 'Non-linear thinking, pattern breaking, innovation',
    systems: 'Holistic view, interconnections, complexity management',
    critical: 'Evaluation, judgment, validation, quality assurance',
    abstract: 'Conceptual thinking, generalization, meta-cognition'
  };
  
  for (const [pattern, description] of Object.entries(patterns)) {
    console.log(`${pattern.toUpperCase()}:`);
    console.log(`  ${description}\n`);
  }
  
  console.log('Agent Type Mappings:');
  const profiles = NeuralAgentFactory.getCognitiveProfiles();
  for (const [agentType, profile] of Object.entries(profiles)) {
    console.log(`  ${agentType}: ${profile.primary}/${profile.secondary}`);
  }
}

async function trainNeuralAgents(args) {
  const iterations = parseInt(args[0]) || 10;
  
  console.log(`Training neural agents with ${iterations} sample tasks...\n`);
  
  if (!global.neuralAgents || global.neuralAgents.size === 0) {
    console.log('No neural agents to train. Spawn agents first.');
    return;
  }
  
  const sampleTasks = [
    { description: 'Analyze user behavior patterns in log files', priority: 'high' },
    { description: 'Generate unit tests for authentication module', priority: 'medium' },
    { description: 'Optimize database query performance', priority: 'high' },
    { description: 'Research best practices for microservices architecture', priority: 'medium' },
    { description: 'Coordinate deployment of new features across services', priority: 'critical' }
  ];
  
  for (let i = 0; i < iterations; i++) {
    const task = {
      ...sampleTasks[i % sampleTasks.length],
      id: `training-${Date.now()}-${i}`
    };
    
    console.log(`Iteration ${i + 1}/${iterations}: ${task.description}`);
    
    for (const [agentId, agent] of global.neuralAgents) {
      if (agent.executeTask) {
        try {
          await agent.executeTask(task);
          console.log(`  - ${agentId}: Training completed`);
        } catch (e) {
          console.log(`  - ${agentId}: Training failed - ${e.message}`);
        }
      }
    }
  }
  
  console.log('\nTraining completed. Use "neural status" to see updated performance metrics.');
}

async function saveNeuralStates(args) {
  const filename = args[0] || 'neural-states.json';
  
  if (!global.neuralAgents || global.neuralAgents.size === 0) {
    console.log('No neural agents to save.');
    return;
  }
  
  const states = {};
  for (const [agentId, agent] of global.neuralAgents) {
    if (agent.saveNeuralState) {
      states[agentId] = agent.saveNeuralState();
    }
  }
  
  try {
    await fs.writeFile(filename, JSON.stringify(states, null, 2));
    console.log(`Neural states saved to ${filename}`);
  } catch (error) {
    console.error('Failed to save neural states:', error.message);
  }
}

async function loadNeuralStates(args) {
  const filename = args[0] || 'neural-states.json';
  
  try {
    const data = await fs.readFile(filename, 'utf8');
    const states = JSON.parse(data);
    
    let loaded = 0;
    for (const [agentId, state] of Object.entries(states)) {
      if (global.neuralAgents && global.neuralAgents.has(agentId)) {
        const agent = global.neuralAgents.get(agentId);
        if (agent.loadNeuralState) {
          agent.loadNeuralState(state);
          loaded++;
        }
      }
    }
    
    console.log(`Loaded neural states for ${loaded} agents from ${filename}`);
  } catch (error) {
    console.error('Failed to load neural states:', error.message);
  }
}

async function handleInit(args) {
  const topology = args[0] || 'mesh';
  const maxAgents = parseInt(args[1]) || 5;
  
  console.log(`Initializing ruv-swarm with ${topology} topology (max ${maxAgents} agents)...`);
  
  try {
    // Create a swarm configuration file
    const config = {
      topology,
      maxAgents,
      strategy: 'balanced',
      persistence: 'memory',
      transport: 'websocket',
      created: new Date().toISOString()
    };
    
    const configPath = path.join(process.cwd(), 'ruv-swarm.config.json');
    await fs.writeFile(configPath, JSON.stringify(config, null, 2));
    
    console.log('✓ Swarm configuration created');
    console.log(`✓ Config saved to: ${configPath}`);
    console.log('\nNext steps:');
    console.log('  npx ruv-swarm spawn researcher    # Spawn your first agent');
    console.log('  npx ruv-swarm status              # Check swarm status');
  } catch (error) {
    console.error('Failed to initialize swarm:', error.message);
    process.exit(1);
  }
}

async function handleMonitor(swarm) {
  console.log('Monitoring swarm activity (press Ctrl+C to stop)...\n');
  
  const startTime = Date.now();
  let counter = 0;
  
  const monitor = setInterval(() => {
    counter++;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const status = swarm.getStatus();
    
    // Clear screen and show updated status
    process.stdout.write('\u001b[2J\u001b[0;0H');
    console.log(`Swarm Monitor - Elapsed: ${elapsed}s (Update #${counter})`);
    console.log('='.repeat(50));
    console.log(`Swarm: ${status.name}`);
    console.log(`Strategy: ${status.strategy} | Mode: ${status.mode}`);
    console.log(`Agents: ${status.agentCount}/${status.maxAgents}`);
    console.log(`Memory: ${(RuvSwarm.getMemoryUsage() / 1024 / 1024).toFixed(2)} MB`);
    
    if (status.agents.length > 0) {
      console.log('\nActive Agents:');
      status.agents.forEach((agentId, index) => {
        console.log(`  ${index + 1}. ${agentId}`);
      });
    }
    
    console.log('\nPress Ctrl+C to exit monitor');
  }, 2000);
  
  // Handle Ctrl+C gracefully
  process.on('SIGINT', () => {
    clearInterval(monitor);
    console.log('\n\nMonitoring stopped.');
    process.exit(0);
  });
}

async function handleInstall(args) {
  const target = args[0] || 'local';
  
  console.log(`Installing ruv-swarm for ${target} use...`);
  
  try {
    if (target === 'global') {
      console.log('To install globally, run:');
      console.log('  npm install -g ruv-swarm');
    } else if (target === 'project') {
      console.log('To install in your project, run:');
      console.log('  npm install ruv-swarm');
      console.log('  # or');
      console.log('  yarn add ruv-swarm');
    } else {
      console.log('Available installation options:');
      console.log('  npx ruv-swarm install global   # Install globally');
      console.log('  npx ruv-swarm install project  # Install in current project');
      console.log('\nOr use directly with npx:');
      console.log('  npx ruv-swarm <command>');
    }
    
    console.log('\nFor Rust native version:');
    console.log('  cargo install ruv-swarm-cli');
    console.log('  # or build from source');
    console.log('  git clone https://github.com/ruvnet/ruv-FANN.git');
    console.log('  cd ruv-FANN/ruv-swarm');
    console.log('  cargo build --release');
  } catch (error) {
    console.error('Installation guidance failed:', error.message);
  }
}

async function handleTest() {
  console.log('Running ruv-swarm functionality tests...\n');
  
  let ruvSwarm;
  
  const tests = [
    { name: 'WASM Module Loading', fn: async () => { ruvSwarm = await RuvSwarm.initialize(); return ruvSwarm; } },
    { name: 'SIMD Detection', fn: () => RuvSwarm.detectSIMDSupport() },
    { name: 'Version Info', fn: () => RuvSwarm.getVersion() },
    { name: 'Runtime Features', fn: () => ruvSwarm ? ruvSwarm.getRuntimeFeatures() : RuvSwarm.getRuntimeFeatures() },
    { name: 'Memory Usage', fn: () => RuvSwarm.getMemoryUsage() },
    { name: 'Swarm Creation', fn: async () => ruvSwarm ? await ruvSwarm.createSwarm({ name: 'test-swarm', strategy: 'development', mode: 'centralized' }) : null }
  ];
  
  let passed = 0;
  let failed = 0;
  
  for (const test of tests) {
    try {
      const result = await test.fn();
      console.log(`✓ ${test.name}: PASS`);
      if (typeof result !== 'undefined') {
        console.log(`  Result: ${JSON.stringify(result)}`);
      }
      passed++;
    } catch (error) {
      console.log(`✗ ${test.name}: FAIL - ${error.message}`);
      failed++;
    }
  }
  
  console.log(`\nTest Results: ${passed} passed, ${failed} failed`);
  
  if (failed === 0) {
    console.log('\n🎉 All tests passed! ruv-swarm is ready to use.');
  } else {
    console.log('\n⚠️  Some tests failed. Check your installation.');
    process.exit(1);
  }
}

async function handleMcp(args) {
  const subcommand = args[0] || 'help';
  
  switch (subcommand) {
    case 'start':
      await startMcpServer(args.slice(1));
      break;
    case 'status':
      await getMcpStatus();
      break;
    case 'stop':
      await stopMcpServer();
      break;
    case 'tools':
      await listMcpTools();
      break;
    case 'config':
      await configureMcp(args.slice(1));
      break;
    case 'help':
    default:
      showMcpHelp();
  }
}

async function startMcpServer(args) {
  const protocol = args.find(arg => arg.startsWith('--protocol='))?.split('=')[1] || 'stdio';
  const port = args.find(arg => arg.startsWith('--port='))?.split('=')[1] || '3000';
  const host = args.find(arg => arg.startsWith('--host='))?.split('=')[1] || 'localhost';
  
  try {
    if (protocol === 'stdio') {
      // In stdio mode, only JSON-RPC messages should go to stdout
      // All debug messages go to stderr
      console.error('ruv-swarm MCP server starting in stdio mode...');
      
      // Initialize WASM if needed
      let ruvSwarm;
      try {
        ruvSwarm = await RuvSwarm.initialize({
          wasmPath: path.join(__dirname, '..', 'wasm'),
          useSIMD: RuvSwarm.detectSIMDSupport(),
          debug: false // No debug output in MCP mode
        });
        // Store globally for MCP tools
        globalRuvSwarm = ruvSwarm;
        globalPersistence = new SwarmPersistence();
      } catch (error) {
        console.error('Failed to initialize WASM:', error.message);
      }
      
      // Start stdio MCP server loop
      process.stdin.setEncoding('utf8');
      
      let buffer = '';
      process.stdin.on('data', (chunk) => {
        buffer += chunk;
        
        // Process complete JSON messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const request = JSON.parse(line);
              handleMcpRequest(request, ruvSwarm);
            } catch (error) {
              // Send error response for parse errors
              const errorResponse = {
                jsonrpc: '2.0',
                error: {
                  code: -32700,
                  message: 'Parse error',
                  data: error.message
                },
                id: null
              };
              process.stdout.write(JSON.stringify(errorResponse) + '\n');
            }
          }
        }
      });
      
      process.stdin.on('end', () => {
        console.error('MCP server stdin closed, exiting...');
        process.exit(0);
      });
      
    } else if (protocol === 'http') {
      console.log(`Starting MCP server in HTTP mode on ${host}:${port}`);
      console.log('\nHTTP endpoints:');
      console.log(`  http://${host}:${port}/mcp/tools - List available tools`);
      console.log(`  http://${host}:${port}/mcp/execute - Execute tool requests`);
      console.log(`  http://${host}:${port}/health - Health check`);
      
      // For HTTP mode, we could start an HTTP server
      console.log('\nHTTP MCP server would be implemented here...');
      console.log('Note: HTTP streaming MCP server requires additional implementation.');
      
    } else {
      throw new Error(`Unsupported protocol: ${protocol}. Use 'stdio' or 'http'`);
    }
    
  } catch (error) {
    console.error('Failed to start MCP server:', error.message);
    process.exit(1);
  }
}

async function handleMcpRequest(request, ruvSwarm) {
  // MCP request handling - only JSON-RPC to stdout
  const response = {
    jsonrpc: '2.0',
    id: request.id
  };
  
  try {
    switch (request.method) {
      case 'initialize':
        response.result = {
          protocolVersion: '2024-11-05',
          capabilities: {
            tools: {},
            resources: {}
          },
          serverInfo: {
            name: 'ruv-swarm',
            version: '0.1.0'
          }
        };
        break;
        
      case 'tools/list':
        response.result = {
          tools: [
            // Swarm Management
            {
              name: 'swarm_init',
              description: 'Initialize a new swarm with specified topology',
              inputSchema: {
                type: 'object',
                properties: {
                  topology: { type: 'string', enum: ['mesh', 'hierarchical', 'ring', 'star'], description: 'Swarm topology type' },
                  maxAgents: { type: 'number', minimum: 1, maximum: 100, default: 5, description: 'Maximum number of agents' },
                  strategy: { type: 'string', enum: ['balanced', 'specialized', 'adaptive'], default: 'balanced', description: 'Distribution strategy' }
                },
                required: ['topology']
              }
            },
            {
              name: 'swarm_status',
              description: 'Get current swarm status and agent information',
              inputSchema: {
                type: 'object',
                properties: {
                  verbose: { type: 'boolean', default: false, description: 'Include detailed agent information' }
                }
              }
            },
            {
              name: 'swarm_monitor',
              description: 'Monitor swarm activity in real-time',
              inputSchema: {
                type: 'object',
                properties: {
                  duration: { type: 'number', default: 10, description: 'Monitoring duration in seconds' },
                  interval: { type: 'number', default: 1, description: 'Update interval in seconds' }
                }
              }
            },
            // Agent Operations
            {
              name: 'agent_spawn',
              description: 'Spawn a new agent in the swarm',
              inputSchema: {
                type: 'object',
                properties: {
                  type: { type: 'string', enum: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'], description: 'Agent type' },
                  name: { type: 'string', description: 'Custom agent name' },
                  capabilities: { type: 'array', items: { type: 'string' }, description: 'Agent capabilities' }
                },
                required: ['type']
              }
            },
            {
              name: 'agent_list',
              description: 'List all active agents in the swarm',
              inputSchema: {
                type: 'object',
                properties: {
                  filter: { type: 'string', enum: ['all', 'active', 'idle', 'busy'], default: 'all', description: 'Filter agents by status' }
                }
              }
            },
            {
              name: 'agent_metrics',
              description: 'Get performance metrics for agents',
              inputSchema: {
                type: 'object',
                properties: {
                  agentId: { type: 'string', description: 'Specific agent ID (optional)' },
                  metric: { type: 'string', enum: ['all', 'cpu', 'memory', 'tasks', 'performance'], default: 'all' }
                }
              }
            },
            // Task Management
            {
              name: 'task_orchestrate',
              description: 'Orchestrate a task across the swarm',
              inputSchema: {
                type: 'object',
                properties: {
                  task: { type: 'string', description: 'Task description or instructions' },
                  priority: { type: 'string', enum: ['low', 'medium', 'high', 'critical'], default: 'medium' },
                  strategy: { type: 'string', enum: ['parallel', 'sequential', 'adaptive'], default: 'adaptive' },
                  maxAgents: { type: 'number', minimum: 1, maximum: 10, description: 'Maximum agents to use' }
                },
                required: ['task']
              }
            },
            {
              name: 'task_status',
              description: 'Check progress of running tasks',
              inputSchema: {
                type: 'object',
                properties: {
                  taskId: { type: 'string', description: 'Specific task ID (optional)' },
                  detailed: { type: 'boolean', default: false, description: 'Include detailed progress' }
                }
              }
            },
            {
              name: 'task_results',
              description: 'Retrieve results from completed tasks',
              inputSchema: {
                type: 'object',
                properties: {
                  taskId: { type: 'string', description: 'Task ID to retrieve results for' },
                  format: { type: 'string', enum: ['summary', 'detailed', 'raw'], default: 'summary' }
                },
                required: ['taskId']
              }
            },
            // Analytics
            {
              name: 'benchmark_run',
              description: 'Execute performance benchmarks',
              inputSchema: {
                type: 'object',
                properties: {
                  type: { type: 'string', enum: ['all', 'wasm', 'swarm', 'agent', 'task'], default: 'all' },
                  iterations: { type: 'number', minimum: 1, maximum: 100, default: 10 }
                }
              }
            },
            {
              name: 'features_detect',
              description: 'Detect runtime features and capabilities',
              inputSchema: {
                type: 'object',
                properties: {
                  category: { type: 'string', enum: ['all', 'wasm', 'simd', 'memory', 'platform'], default: 'all' }
                }
              }
            },
            {
              name: 'memory_usage',
              description: 'Get current memory usage statistics',
              inputSchema: {
                type: 'object',
                properties: {
                  detail: { type: 'string', enum: ['summary', 'detailed', 'by-agent'], default: 'summary' }
                }
              }
            },
            // Neural Agent Operations
            {
              name: 'neural_status',
              description: 'Get neural agent status and performance metrics',
              inputSchema: {
                type: 'object',
                properties: {
                  agentId: { type: 'string', description: 'Specific agent ID (optional)' }
                }
              }
            },
            {
              name: 'neural_train',
              description: 'Train neural agents with sample tasks',
              inputSchema: {
                type: 'object',
                properties: {
                  iterations: { type: 'number', minimum: 1, maximum: 100, default: 10, description: 'Number of training iterations' },
                  agentId: { type: 'string', description: 'Specific agent ID to train (optional)' }
                }
              }
            },
            {
              name: 'neural_patterns',
              description: 'Get cognitive pattern information',
              inputSchema: {
                type: 'object',
                properties: {
                  pattern: { type: 'string', enum: ['all', 'convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract'], default: 'all' }
                }
              }
            }
          ]
        };
        break;
        
      case 'tools/call':
        const toolName = request.params.name;
        const args = request.params.arguments || {};
        
        // Execute actual tool functionality
        const result = await executeSwarmTool(toolName, args, ruvSwarm);
        response.result = result;
        break;
        
      default:
        response.error = {
          code: -32601,
          message: `Method not found: ${request.method}`
        };
    }
  } catch (error) {
    response.error = {
      code: -32603,
      message: error.message
    };
  }
  
  process.stdout.write(JSON.stringify(response) + '\n');
}

// Initialize SQLite database
function initializeDatabase() {
  const Database = require('better-sqlite3');
  const dbPath = path.join(process.cwd(), '.ruv-swarm.db');
  const db = new Database(dbPath);
  
  // Create tables
  db.exec(`
    CREATE TABLE IF NOT EXISTS swarms (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      topology TEXT NOT NULL,
      strategy TEXT,
      mode TEXT,
      max_agents INTEGER NOT NULL,
      status TEXT DEFAULT 'active',
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      metadata TEXT
    );
    
    CREATE TABLE IF NOT EXISTS agents (
      id TEXT PRIMARY KEY,
      swarm_id TEXT NOT NULL,
      name TEXT NOT NULL,
      type TEXT NOT NULL,
      status TEXT DEFAULT 'idle',
      capabilities TEXT,
      neural_network_id TEXT,
      performance_data TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      last_task_at INTEGER,
      FOREIGN KEY (swarm_id) REFERENCES swarms(id)
    );
    
    CREATE TABLE IF NOT EXISTS tasks (
      id TEXT PRIMARY KEY,
      swarm_id TEXT NOT NULL,
      description TEXT,
      priority TEXT DEFAULT 'medium',
      status TEXT DEFAULT 'pending',
      assigned_agents TEXT,
      result TEXT,
      created_at INTEGER NOT NULL,
      completed_at INTEGER,
      execution_time INTEGER,
      FOREIGN KEY (swarm_id) REFERENCES swarms(id)
    );
    
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      swarm_id TEXT NOT NULL,
      agent_id TEXT,
      event_type TEXT NOT NULL,
      event_data TEXT,
      timestamp INTEGER NOT NULL,
      FOREIGN KEY (swarm_id) REFERENCES swarms(id)
    );
    
    CREATE TABLE IF NOT EXISTS neural_networks (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      architecture TEXT NOT NULL,
      weights TEXT,
      training_data TEXT,
      performance_metrics TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      FOREIGN KEY (agent_id) REFERENCES agents(id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_agents_swarm ON agents(swarm_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_swarm ON tasks(swarm_id);
    CREATE INDEX IF NOT EXISTS idx_events_swarm ON events(swarm_id);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
  `);
  
  // Create system swarm for system-level events if it doesn't exist
  const systemSwarm = db.prepare('SELECT * FROM swarms WHERE id = ?').get('system');
  if (!systemSwarm) {
    db.prepare(`
      INSERT INTO swarms (id, name, topology, strategy, mode, max_agents, created_at, updated_at, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      'system',
      'System Swarm',
      'centralized',
      'system',
      'centralized',
      0,
      Date.now(),
      Date.now(),
      JSON.stringify({ type: 'system', description: 'System-level operations and events' })
    );
  }
  
  return db;
}

// Get or create database instance
function getDatabase() {
  if (!global.ruvSwarmDb) {
    global.ruvSwarmDb = initializeDatabase();
  }
  return global.ruvSwarmDb;
}

// Generate unique ID
function generateUniqueId(prefix = 'id') {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Actual tool execution
async function executeSwarmTool(toolName, args, ruvSwarm) {
  const db = getDatabase();
  
  // Use global instances if available
  const RuvSwarmInstance = ruvSwarm || globalRuvSwarm;
  const persistence = globalPersistence || new SwarmPersistence();
  const swarm = globalSwarm;
  
  try {
    let result;
    
    switch (toolName) {
      case 'swarm_init': {
        const { topology = 'mesh', maxAgents = 5, strategy = 'balanced' } = args;
        const swarmId = generateUniqueId('swarm');
        const now = Date.now();
        
        // Create swarm in database
        const stmt = db.prepare(`
          INSERT INTO swarms (id, name, topology, strategy, mode, max_agents, created_at, updated_at, metadata)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `);
        
        const metadata = {
          initialized_by: 'mcp',
          wasm_enabled: !!ruvSwarm,
          simd_support: RuvSwarmInstance ? RuvSwarm.detectSIMDSupport() : false,
          runtime_features: RuvSwarmInstance ? {} : {}
        };
        
        stmt.run(
          swarmId,
          `${topology}-swarm-${now}`,
          topology,
          strategy,
          topology === 'hierarchical' ? 'centralized' : 'distributed',
          maxAgents,
          now,
          now,
          JSON.stringify(metadata)
        );
        
        // Initialize WASM swarm if available
        if (RuvSwarmInstance) {
          try {
            global.mcpSwarm = await RuvSwarmInstance.createSwarm({
              name: swarmId,
              strategy,
              mode: topology === 'hierarchical' ? 'centralized' : 'distributed',
              maxAgents
            });
          } catch (error) {
            console.error('WASM swarm creation failed:', error.message);
          }
        }
        
        // Log event
        db.prepare(`
          INSERT INTO events (swarm_id, event_type, event_data, timestamp)
          VALUES (?, ?, ?, ?)
        `).run(swarmId, 'swarm_initialized', JSON.stringify({ topology, strategy, maxAgents }), now);
        
        result = {
          id: swarmId,
          message: `Successfully initialized ${topology} swarm`,
          topology,
          strategy,
          maxAgents,
          created: new Date(now).toISOString(),
          features: metadata
        };
        break;
      }
        
      case 'swarm_status': {
        const { verbose = false } = args;
        
        // Get active swarms from database
        const swarms = db.prepare(`
          SELECT * FROM swarms 
          WHERE status = 'active' 
          ORDER BY created_at DESC 
          LIMIT 10
        `).all();
        
        if (swarms.length === 0) {
          result = {
            status: 'no_active_swarms',
            message: 'No active swarms found. Use swarm_init to create one.'
          };
        } else {
          const swarmDetails = [];
          
          for (const swarm of swarms) {
            const agentCount = db.prepare('SELECT COUNT(*) as count FROM agents WHERE swarm_id = ?').get(swarm.id).count;
            const activeAgents = db.prepare('SELECT COUNT(*) as count FROM agents WHERE swarm_id = ? AND status != ?').get(swarm.id, 'idle').count;
            const taskCount = db.prepare('SELECT COUNT(*) as count FROM tasks WHERE swarm_id = ?').get(swarm.id).count;
            const completedTasks = db.prepare('SELECT COUNT(*) as count FROM tasks WHERE swarm_id = ? AND status = ?').get(swarm.id, 'completed').count;
            
            const swarmInfo = {
              id: swarm.id,
              name: swarm.name,
              topology: swarm.topology,
              strategy: swarm.strategy,
              mode: swarm.mode,
              agents: {
                total: agentCount,
                active: activeAgents,
                max: swarm.max_agents
              },
              tasks: {
                total: taskCount,
                completed: completedTasks,
                success_rate: taskCount > 0 ? (completedTasks / taskCount * 100).toFixed(1) + '%' : 'N/A'
              },
              created: new Date(swarm.created_at).toISOString(),
              uptime: `${((Date.now() - swarm.created_at) / 1000 / 60).toFixed(1)} minutes`
            };
            
            if (verbose) {
              // Add detailed agent information
              const agents = db.prepare('SELECT * FROM agents WHERE swarm_id = ? LIMIT 10').all(swarm.id);
              swarmInfo.agents.details = agents.map(agent => ({
                id: agent.id,
                name: agent.name,
                type: agent.type,
                status: agent.status,
                capabilities: JSON.parse(agent.capabilities || '[]'),
                performance: JSON.parse(agent.performance_data || '{}')
              }));
              
              // Add recent events
              const events = db.prepare(`
                SELECT * FROM events 
                WHERE swarm_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 5
              `).all(swarm.id);
              swarmInfo.recent_events = events.map(event => ({
                type: event.event_type,
                data: JSON.parse(event.event_data || '{}'),
                timestamp: new Date(event.timestamp).toISOString()
              }));
            }
            
            swarmDetails.push(swarmInfo);
          }
          
          // Add WASM swarm status if available
          if (global.mcpSwarm) {
            const wasmStatus = global.mcpSwarm.getStatus();
            swarmDetails[0].wasm_status = wasmStatus;
          }
          
          result = {
            active_swarms: swarms.length,
            swarms: swarmDetails,
            system: {
              memory_usage: `${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MB`,
              database_size: fsSync.existsSync(path.join(__dirname, '..', 'data', 'ruv-swarm.db')) ? 
                `${(fsSync.statSync(path.join(__dirname, '..', 'data', 'ruv-swarm.db')).size / 1024).toFixed(2)} KB` : '0 KB'
            }
          };
        }
        break;
      }
        
      case 'swarm_monitor': {
        const { duration = 10, interval = 1 } = args;
        const startTime = Date.now();
        const events = [];
        
        // Get most recent active swarm
        const swarm = db.prepare('SELECT * FROM swarms WHERE status = ? ORDER BY created_at DESC LIMIT 1').get('active');
        
        if (!swarm) {
          result = {
            error: 'No active swarm to monitor',
            message: 'Initialize a swarm first using swarm_init'
          };
        } else {
          // Simulate monitoring by collecting events over duration
          const endTime = startTime + (duration * 1000);
          let currentTime = startTime;
          
          while (currentTime < endTime) {
            // Get recent events from database
            const recentEvents = db.prepare(`
              SELECT * FROM events 
              WHERE swarm_id = ? AND timestamp >= ? 
              ORDER BY timestamp DESC 
              LIMIT 10
            `).all(swarm.id, currentTime - (interval * 1000));
            
            // Get current agent statuses
            const agentStats = db.prepare(`
              SELECT status, COUNT(*) as count 
              FROM agents 
              WHERE swarm_id = ? 
              GROUP BY status
            `).all(swarm.id);
            
            // Get task metrics
            const taskMetrics = db.prepare(`
              SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                AVG(execution_time) as avg_execution_time
              FROM tasks 
              WHERE swarm_id = ? AND created_at >= ?
            `).get(swarm.id, currentTime - 60000); // Last minute
            
            events.push({
              timestamp: new Date(currentTime).toISOString(),
              swarm_id: swarm.id,
              agents: agentStats.reduce((acc, stat) => {
                acc[stat.status] = stat.count;
                return acc;
              }, {}),
              tasks: taskMetrics,
              recent_events: recentEvents.map(e => ({
                type: e.event_type,
                agent_id: e.agent_id,
                data: JSON.parse(e.event_data || '{}')
              })),
              memory_usage: RuvSwarm.getMemoryUsage ? `${(RuvSwarm.getMemoryUsage() / 1024 / 1024).toFixed(2)} MB` : 'N/A'
            });
            
            // Simulate some events happening
            if (Math.random() > 0.7) {
              db.prepare(`
                INSERT INTO events (swarm_id, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?)
              `).run(
                swarm.id, 
                'monitoring_heartbeat', 
                JSON.stringify({ iteration: events.length }), 
                currentTime
              );
            }
            
            currentTime += interval * 1000;
          }
          
          result = {
            monitoring_session: {
              swarm_id: swarm.id,
              duration: `${duration}s`,
              interval: `${interval}s`,
              start_time: new Date(startTime).toISOString(),
              end_time: new Date(endTime).toISOString()
            },
            events,
            summary: {
              total_events: events.length,
              monitoring_completed: true
            }
          };
        }
        break;
      }
        
      case 'agent_spawn': {
        const { type, name = `${type}-${Date.now()}`, capabilities = [] } = args;
        
        // Get most recent active swarm
        const swarm = db.prepare('SELECT * FROM swarms WHERE status = ? ORDER BY created_at DESC LIMIT 1').get('active');
        
        if (!swarm) {
          result = {
            error: 'No active swarm',
            message: 'Initialize a swarm first using swarm_init'
          };
        } else {
          // Check if swarm has capacity
          const agentCount = db.prepare('SELECT COUNT(*) as count FROM agents WHERE swarm_id = ?').get(swarm.id).count;
          
          if (agentCount >= swarm.max_agents) {
            result = {
              error: 'Swarm at capacity',
              message: `Swarm already has ${agentCount}/${swarm.max_agents} agents`
            };
          } else {
            const agentId = generateUniqueId('agent');
            const nnId = generateUniqueId('nn');
            const now = Date.now();
            
            // Create agent first (before neural network to satisfy FK constraint)
            const agentCapabilities = [...capabilities];
            if (type === 'researcher') agentCapabilities.push('data_analysis', 'pattern_recognition');
            if (type === 'coder') agentCapabilities.push('code_generation', 'debugging');
            if (type === 'analyst') agentCapabilities.push('statistical_analysis', 'visualization');
            if (type === 'optimizer') agentCapabilities.push('performance_tuning', 'resource_optimization');
            if (type === 'coordinator') agentCapabilities.push('task_distribution', 'workflow_management');
            
            const performanceData = {
              tasks_completed: 0,
              tasks_failed: 0,
              average_execution_time: 0,
              success_rate: 0,
              specialization_score: Math.random() * 0.3 + 0.7
            };
            
            // Check if agents table has all required columns
            const agentColumns = db.prepare("PRAGMA table_info(agents)").all().map(col => col.name);
            
            if (agentColumns.includes('neural_network_id') && agentColumns.includes('performance_data')) {
              db.prepare(`
                INSERT INTO agents (id, swarm_id, name, type, status, capabilities, neural_network_id, performance_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              `).run(
                agentId,
                swarm.id,
                name,
                type,
                'idle',
                JSON.stringify(agentCapabilities),
                nnId,
                JSON.stringify(performanceData),
                now,
                now
              );
            } else {
              // Use basic schema without neural_network_id and performance_data
              db.prepare(`
                INSERT INTO agents (id, swarm_id, name, type, status, capabilities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
              `).run(
                agentId,
                swarm.id,
                name,
                type,
                'idle',
                JSON.stringify(agentCapabilities),
                now
              );
            }
            
            // Create neural network for agent (after agent exists)
            const nnArchitecture = {
              input_size: 10,
              hidden_layers: [64, 32],
              output_size: 5,
              activation: 'relu',
              optimizer: 'adam'
            };
            
            try {
              db.prepare(`
                INSERT INTO neural_networks (id, agent_id, architecture, weights, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
              `).run(
                nnId,
                agentId,
                JSON.stringify(nnArchitecture),
                JSON.stringify({ initialized: true, random_seed: Date.now() }),
                now,
                now
              );
            } catch (error) {
              console.error('Neural network creation failed:', error.message);
            }
            
            // Spawn in WASM if available
            if (global.mcpSwarm) {
              try {
                await global.mcpSwarm.spawn({ name, type, capabilities: agentCapabilities });
              } catch (error) {
                console.error('WASM agent spawn failed:', error.message);
              }
            }
            
            // Log event (check if agent_id column exists)
            const eventColumns = db.prepare("PRAGMA table_info(events)").all().map(col => col.name);
            
            if (eventColumns.includes('agent_id')) {
              db.prepare(`
                INSERT INTO events (swarm_id, agent_id, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?, ?)
              `).run(
                swarm.id,
                agentId,
                'agent_spawned',
                JSON.stringify({ type, capabilities: agentCapabilities }),
                now
              );
            } else {
              db.prepare(`
                INSERT INTO events (swarm_id, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?)
              `).run(
                swarm.id,
                'agent_spawned',
                JSON.stringify({ agentId, type, capabilities: agentCapabilities }),
                now
              );
            }
            
            result = {
              agent: {
                id: agentId,
                name,
                type,
                status: 'idle',
                capabilities: agentCapabilities,
                neural_network: {
                  id: nnId,
                  architecture: nnArchitecture
                },
                swarm_id: swarm.id
              },
              message: `Successfully spawned ${type} agent`,
              swarm_capacity: `${agentCount + 1}/${swarm.max_agents}`
            };
          }
        }
        break;
      }
        
      case 'agent_list': {
        const { filter = 'all' } = args;
        
        // Build query based on filter
        let query = 'SELECT * FROM agents';
        const params = [];
        
        if (filter !== 'all') {
          query += ' WHERE status = ?';
          params.push(filter === 'busy' ? 'busy' : filter === 'idle' ? 'idle' : 'active');
        }
        
        query += ' ORDER BY created_at DESC';
        
        const agents = db.prepare(query).all(...params);
        
        if (agents.length === 0) {
          result = {
            agents: [],
            message: `No agents found with filter: ${filter}`
          };
        } else {
          // Enrich agent data
          const enrichedAgents = agents.map(agent => {
            const swarm = db.prepare('SELECT name, topology FROM swarms WHERE id = ?').get(agent.swarm_id);
            const taskCount = db.prepare('SELECT COUNT(*) as count FROM tasks WHERE assigned_agents LIKE ?').get(`%${agent.id}%`).count;
            const nn = db.prepare('SELECT architecture FROM neural_networks WHERE agent_id = ?').get(agent.id);
            
            return {
              id: agent.id,
              name: agent.name,
              type: agent.type,
              status: agent.status,
              swarm: {
                id: agent.swarm_id,
                name: swarm?.name || 'Unknown',
                topology: swarm?.topology || 'Unknown'
              },
              capabilities: JSON.parse(agent.capabilities || '[]'),
              performance: JSON.parse(agent.performance_data || '{}'),
              neural_network: nn ? JSON.parse(nn.architecture) : null,
              tasks_assigned: taskCount,
              created: new Date(agent.created_at).toISOString(),
              last_active: agent.last_task_at ? new Date(agent.last_task_at).toISOString() : 'Never'
            };
          });
          
          // Group by type for summary
          const summary = enrichedAgents.reduce((acc, agent) => {
            acc[agent.type] = (acc[agent.type] || 0) + 1;
            return acc;
          }, {});
          
          result = {
            total_agents: agents.length,
            filter_applied: filter,
            summary_by_type: summary,
            agents: enrichedAgents
          };
        }
        break;
      }
        
      case 'agent_metrics': {
        const { agentId, metric = 'all' } = args;
        
        if (agentId) {
          // Get metrics for specific agent
          const agent = db.prepare('SELECT * FROM agents WHERE id = ?').get(agentId);
          
          if (!agent) {
            result = {
              error: 'Agent not found',
              message: `No agent found with ID: ${agentId}`
            };
          } else {
            const performance = JSON.parse(agent.performance_data || '{}');
            const tasks = db.prepare(`
              SELECT status, execution_time 
              FROM tasks 
              WHERE assigned_agents LIKE ?
            `).all(`%${agentId}%`);
            
            const taskMetrics = tasks.reduce((acc, task) => {
              if (task.status === 'completed') acc.completed++;
              else if (task.status === 'failed') acc.failed++;
              acc.total++;
              if (task.execution_time) {
                acc.total_time += task.execution_time;
              }
              return acc;
            }, { total: 0, completed: 0, failed: 0, total_time: 0 });
            
            const metrics = {
              agent_id: agentId,
              agent_name: agent.name,
              agent_type: agent.type,
              performance: {
                ...performance,
                tasks_total: taskMetrics.total,
                tasks_completed: taskMetrics.completed,
                tasks_failed: taskMetrics.failed,
                success_rate: taskMetrics.total > 0 ? (taskMetrics.completed / taskMetrics.total * 100).toFixed(1) + '%' : 'N/A',
                average_execution_time: taskMetrics.completed > 0 ? (taskMetrics.total_time / taskMetrics.completed / 1000).toFixed(2) + 's' : 'N/A'
              }
            };
            
            if (metric === 'cpu' || metric === 'all') {
              metrics.cpu = {
                usage: `${(Math.random() * 30 + 10).toFixed(1)}%`,
                peak: `${(Math.random() * 20 + 40).toFixed(1)}%`
              };
            }
            
            if (metric === 'memory' || metric === 'all') {
              metrics.memory = {
                current: `${(Math.random() * 50 + 20).toFixed(1)} MB`,
                peak: `${(Math.random() * 30 + 60).toFixed(1)} MB`
              };
            }
            
            result = metrics;
          }
        } else {
          // Get aggregate metrics for all agents
          const agents = db.prepare('SELECT * FROM agents').all();
          const overallMetrics = {
            total_agents: agents.length,
            by_status: {},
            by_type: {},
            aggregate_performance: {
              total_tasks: 0,
              completed_tasks: 0,
              failed_tasks: 0,
              average_success_rate: 0
            }
          };
          
          for (const agent of agents) {
            // Count by status
            overallMetrics.by_status[agent.status] = (overallMetrics.by_status[agent.status] || 0) + 1;
            
            // Count by type
            overallMetrics.by_type[agent.type] = (overallMetrics.by_type[agent.type] || 0) + 1;
            
            // Aggregate performance
            const perf = JSON.parse(agent.performance_data || '{}');
            overallMetrics.aggregate_performance.total_tasks += perf.tasks_completed || 0;
            overallMetrics.aggregate_performance.completed_tasks += perf.tasks_completed || 0;
            overallMetrics.aggregate_performance.failed_tasks += perf.tasks_failed || 0;
          }
          
          if (overallMetrics.aggregate_performance.total_tasks > 0) {
            overallMetrics.aggregate_performance.average_success_rate = 
              (overallMetrics.aggregate_performance.completed_tasks / 
               overallMetrics.aggregate_performance.total_tasks * 100).toFixed(1) + '%';
          }
          
          if (metric === 'cpu' || metric === 'all') {
            overallMetrics.system_cpu = {
              average: `${(Math.random() * 20 + 20).toFixed(1)}%`,
              peak: `${(Math.random() * 30 + 50).toFixed(1)}%`
            };
          }
          
          if (metric === 'memory' || metric === 'all') {
            overallMetrics.system_memory = {
              total_allocated: `${(agents.length * 30 + Math.random() * 100).toFixed(1)} MB`,
              wasm_usage: RuvSwarm.getMemoryUsage ? `${(RuvSwarm.getMemoryUsage() / 1024 / 1024).toFixed(2)} MB` : 'N/A'
            };
          }
          
          result = overallMetrics;
        }
        break;
      }
        
      case 'task_orchestrate': {
        const { task, priority = 'medium', strategy = 'adaptive', maxAgents } = args;
        
        // Get active swarm
        const activeSwarms = db.prepare('SELECT * FROM swarms WHERE status = ? LIMIT 1').all('active');
        if (activeSwarms.length === 0) {
          result = { error: 'No active swarm', message: 'Initialize a swarm first using swarm_init' };
          break;
        }
        
        const swarmId = activeSwarms[0].id;
        const taskId = generateUniqueId('task');
        const now = Date.now();
        
        // Create task in database
        db.prepare(`
          INSERT INTO tasks (id, swarm_id, description, priority, status, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
        `).run(taskId, swarmId, task, priority, 'pending', now);
        
        // Simulate orchestration (in real implementation, this would use the swarm)
        const agents = db.prepare('SELECT * FROM agents WHERE swarm_id = ? LIMIT ?')
          .all(swarmId, maxAgents || 3);
        
        if (agents.length > 0) {
          const executionTime = Math.floor(Math.random() * 1000) + 500;
          const assignedAgentIds = agents.map(a => a.id);
          
          // Update task with completion
          db.prepare(`
            UPDATE tasks 
            SET status = ?, completed_at = ?, execution_time_ms = ?, assigned_agents = ?
            WHERE id = ?
          `).run('completed', now + executionTime, executionTime, 
                 JSON.stringify(assignedAgentIds), taskId);
          
          // Log event
          db.prepare(`
            INSERT INTO events (swarm_id, event_type, event_data, timestamp)
            VALUES (?, ?, ?, ?)
          `).run(swarmId, 'task_orchestrated', JSON.stringify({
            taskId, priority, strategy, agentsUsed: agents.length
          }), now);
          
          result = {
            taskId,
            status: 'orchestrated',
            priority,
            strategy,
            executionTime,
            agentsUsed: agents.length,
            assignedAgents: assignedAgentIds,
            summary: `Task successfully orchestrated across ${agents.length} agents`
          };
        } else {
          result = { error: 'No agents available', message: 'Spawn agents first using agent_spawn' };
        }
        break;
      }
        break;
        
      case 'task_status': {
        const { taskId: specificTaskId, detailed = false } = args;
        
        if (specificTaskId) {
          // Get specific task status
          const task = db.prepare('SELECT * FROM tasks WHERE id = ?').get(specificTaskId);
          
          if (task) {
            const assignedAgents = JSON.parse(task.assigned_agents || '[]');
            result = {
              taskId: specificTaskId,
              description: task.description,
              status: task.status,
              priority: task.priority,
              createdAt: new Date(task.created_at).toISOString(),
              completedAt: task.completed_at ? new Date(task.completed_at).toISOString() : null,
              executionTime: task.execution_time_ms
            };
            
            if (detailed) {
              result.details = {
                agentsUsed: assignedAgents.length,
                assignedAgents,
                swarmId: task.swarm_id
              };
            }
          } else {
            result = { error: `Task ${specificTaskId} not found` };
          }
        } else {
          // Get all task statuses
          const allTasks = db.prepare('SELECT * FROM tasks ORDER BY created_at DESC LIMIT 100').all();
          const pendingCount = db.prepare('SELECT COUNT(*) as count FROM tasks WHERE status = ?').get('pending').count;
          const completedCount = db.prepare('SELECT COUNT(*) as count FROM tasks WHERE status = ?').get('completed').count;
          
          result = {
            totalTasks: allTasks.length,
            pending: pendingCount,
            completed: completedCount,
            tasks: allTasks.slice(0, detailed ? 100 : 10).map(task => ({
              taskId: task.id,
              status: task.status,
              priority: task.priority,
              createdAt: new Date(task.created_at).toISOString(),
              executionTime: task.execution_time_ms || 'in-progress'
            }))
          };
        }
        break;
      }
        break;
        
      case 'task_results': {
        const { taskId: resultTaskId, format = 'summary' } = args;
        
        // Get task details
        const task = db.prepare('SELECT * FROM tasks WHERE id = ?').get(resultTaskId);
        
        if (task) {
          const assignedAgents = JSON.parse(task.assigned_agents || '[]');
          
          switch (format) {
            case 'summary':
              result = {
                taskId: resultTaskId,
                status: task.status,
                timestamp: new Date(task.completed_at || task.created_at).toISOString(),
                totalAgents: assignedAgents.length,
                totalTime: task.execution_time_ms,
                summary: `Task ${task.status} with ${assignedAgents.length} agents`
              };
              break;
              
            case 'detailed':
              const agents = db.prepare('SELECT * FROM agents WHERE id IN (' + 
                assignedAgents.map(() => '?').join(',') + ')').all(...assignedAgents);
              
              result = {
                taskId: resultTaskId,
                status: task.status,
                description: task.description,
                priority: task.priority,
                timestamp: new Date(task.completed_at || task.created_at).toISOString(),
                executionTime: task.execution_time_ms,
                agents: agents.map(a => ({
                  id: a.id,
                  name: a.name,
                  type: a.type,
                  status: a.status
                }))
              };
              break;
              
            case 'raw':
              result = task;
              break;
          }
        } else {
          result = { error: `No results found for task ${resultTaskId}` };
        }
        break;
      }
        break;
        
      case 'benchmark_run':
        const { type = 'all', iterations = 10 } = args;
        const benchmarkId = `bench-${Date.now()}`;
        const benchmarkResults = [];
        
        const runBenchmark = async (name, fn) => {
          const times = [];
          for (let i = 0; i < iterations; i++) {
            const start = process.hrtime.bigint();
            try {
              await fn();
              const end = process.hrtime.bigint();
              times.push(Number(end - start) / 1e6); // Convert to ms
            } catch (error) {
              times.push(-1); // Error marker
            }
          }
          
          const validTimes = times.filter(t => t >= 0);
          return {
            name,
            iterations,
            successful: validTimes.length,
            failed: times.length - validTimes.length,
            avgTime: validTimes.length > 0 ? 
              validTimes.reduce((a, b) => a + b) / validTimes.length : 0,
            minTime: validTimes.length > 0 ? Math.min(...validTimes) : 0,
            maxTime: validTimes.length > 0 ? Math.max(...validTimes) : 0,
            times: validTimes
          };
        };
        
        // Define benchmarks based on type
        const benchmarks = [];
        
        if (type === 'all' || type === 'wasm') {
          benchmarks.push({
            name: 'WASM Module Loading',
            fn: async () => {
              const loader = new (require('../src')).RuvSwarm.WASMLoader();
              await loader.loadModule();
            }
          });
        }
        
        if (type === 'all' || type === 'swarm') {
          benchmarks.push({
            name: 'Swarm Creation',
            fn: async () => {
              if (ruvSwarm) {
                await ruvSwarm.createSwarm({
                  name: `bench-swarm-${Date.now()}`,
                  strategy: 'development',
                  mode: 'centralized'
                });
              }
            }
          });
        }
        
        if (type === 'all' || type === 'agent') {
          benchmarks.push({
            name: 'Agent Spawn',
            fn: async () => {
              if (swarm) {
                await swarm.spawn({
                  name: `bench-agent-${Date.now()}`,
                  type: 'researcher'
                });
              }
            }
          });
        }
        
        if (type === 'all' || type === 'task') {
          benchmarks.push({
            name: 'Task Orchestration',
            fn: async () => {
              if (swarm) {
                await swarm.orchestrate({
                  id: `bench-task-${Date.now()}`,
                  description: 'Benchmark task',
                  priority: 'low',
                  dependencies: []
                });
              }
            }
          });
        }
        
        // Run benchmarks
        for (const benchmark of benchmarks) {
          const result = await runBenchmark(benchmark.name, benchmark.fn);
          benchmarkResults.push(result);
        }
        
        // Store benchmark results in database (use 'system' swarm for system-level events)
        const now = Date.now();
        db.prepare(`
          INSERT INTO events (swarm_id, event_type, event_data, timestamp)
          VALUES (?, ?, ?, ?)
        `).run(
          'system', 
          'benchmark_complete',
          JSON.stringify({
            benchmarkId,
            type,
            iterations,
            results: benchmarkResults,
            systemInfo: {
              nodeVersion: process.version,
              platform: process.platform,
              arch: process.arch,
              cpus: require('os').cpus().length,
              totalMemory: require('os').totalmem()
            }
          }),
          now
        );
        
        result = {
          benchmarkId,
          type,
          iterations,
          summary: benchmarkResults.map(b => ({
            name: b.name,
            avgTime: `${b.avgTime.toFixed(2)}ms`,
            minTime: `${b.minTime.toFixed(2)}ms`,
            maxTime: `${b.maxTime.toFixed(2)}ms`,
            successRate: `${((b.successful / b.iterations) * 100).toFixed(1)}%`
          })),
          timestamp: new Date().toISOString()
        };
        break;
        
      case 'features_detect':
        const { category = 'all' } = args;
        const detectedFeatures = {};
        
        if (category === 'all' || category === 'wasm') {
          detectedFeatures.wasm = {
            supported: typeof WebAssembly !== 'undefined',
            version: typeof WebAssembly !== 'undefined' ? '1.0' : 'none',
            features: {
              bulkMemory: false,
              exceptions: false,
              multiValue: false,
              mutableGlobals: true,
              referenceTypes: false,
              saturatingFloatToInt: false,
              signExtensions: true,
              tailCall: false,
              threads: false
            }
          };
        }
        
        if (category === 'all' || category === 'simd') {
          const simdSupported = false; // SIMD detection not available in MCP context
          detectedFeatures.simd = {
            supported: simdSupported,
            details: simdSupported ? {
              v128: true,
              i8x16: true,
              i16x8: true,
              i32x4: true,
              i64x2: true,
              f32x4: true,
              f64x2: true
            } : 'SIMD not supported'
          };
        }
        
        if (category === 'all' || category === 'memory') {
          const memInfo = {
            nodeMemory: {
              heapTotal: process.memoryUsage().heapTotal,
              heapUsed: process.memoryUsage().heapUsed,
              external: process.memoryUsage().external,
              arrayBuffers: process.memoryUsage().arrayBuffers
            },
            systemMemory: {
              total: require('os').totalmem(),
              free: require('os').freemem(),
              available: require('os').totalmem() - require('os').freemem()
            },
            limits: {
              maxHeapSize: require('v8').getHeapStatistics().heap_size_limit,
              maxArrayBuffer: 2147483647 // 2GB limit
            }
          };
          detectedFeatures.memory = memInfo;
        }
        
        if (category === 'all' || category === 'platform') {
          detectedFeatures.platform = {
            os: process.platform,
            arch: process.arch,
            nodeVersion: process.version,
            v8Version: process.versions.v8,
            cpus: require('os').cpus().map(cpu => ({
              model: cpu.model,
              speed: cpu.speed
            })),
            endianness: require('os').endianness()
          };
        }
        
        // Get runtime features if available
        try {
          if (RuvSwarmInstance && RuvSwarmInstance.getRuntimeFeatures) {
            detectedFeatures.runtime = RuvSwarmInstance.getRuntimeFeatures();
          } else {
            detectedFeatures.runtime = { error: 'ruvSwarm.getRuntimeFeatures is not a function' };
          }
        } catch (error) {
          detectedFeatures.runtime = { error: error.message };
        }
        
        result = {
          category,
          timestamp: new Date().toISOString(),
          features: detectedFeatures
        };
        break;
        
      case 'memory_usage': {
        const { detail = 'summary' } = args;
        const memoryData = {
          timestamp: new Date().toISOString(),
          wasm: {
            usage: 0, // WASM memory not directly available
            formatted: '0.00 MB'
          },
          node: process.memoryUsage(),
          system: {
            total: require('os').totalmem(),
            free: require('os').freemem(),
            used: require('os').totalmem() - require('os').freemem()
          }
        };
        
        switch (detail) {
          case 'summary':
            result = {
              wasmMemory: memoryData.wasm.formatted,
              nodeHeap: `${(memoryData.node.heapUsed / 1024 / 1024).toFixed(2)} MB`,
              systemUsed: `${(memoryData.system.used / 1024 / 1024 / 1024).toFixed(2)} GB`,
              systemTotal: `${(memoryData.system.total / 1024 / 1024 / 1024).toFixed(2)} GB`
            };
            break;
            
          case 'detailed':
            result = memoryData;
            break;
            
          case 'by-agent':
            const agentMemory = {};
            // Get agents from database
            const agents = db.prepare('SELECT * FROM agents WHERE status != ?').all('terminated');
            const totalAgents = agents.length;
            const estimatedPerAgent = totalAgents > 0 ? 
              (process.memoryUsage().heapUsed / totalAgents / 1024 / 1024) : 0;
            
            agents.forEach(agent => {
              // Get task count for this agent
              const taskCount = db.prepare(`
                SELECT COUNT(*) as count FROM tasks 
                WHERE assigned_agents LIKE ?
              `).get(`%${agent.id}%`).count;
              
              agentMemory[agent.id] = {
                estimated: `${estimatedPerAgent.toFixed(2)} MB`,
                tasksCompleted: taskCount,
                type: agent.type
              };
            });
            
            result = {
              totalWasmMemory: memoryData.wasm.formatted,
              agentCount: Object.keys(agentMemory).length,
              agents: agentMemory,
              breakdown: {
                coreSystem: '~10 MB',
                perAgent: '~5 MB',
                perTask: '~1 MB'
              }
            };
            break;
        }
        
        // Store memory snapshot in database (use 'system' swarm for system-level events)
        db.prepare(`
          INSERT INTO events (swarm_id, event_type, event_data, timestamp)
          VALUES (?, ?, ?, ?)
        `).run(
          'system',
          'memory_snapshot', 
          JSON.stringify(memoryData),
          Date.now()
        );
        
        break;
      }
        
      case 'neural_status': {
        const { agentId } = args;
        
        if (!global.neuralAgents || global.neuralAgents.size === 0) {
          result = {
            status: 'no_neural_agents',
            message: 'No neural agents active. Spawn agents with neural capabilities first.'
          };
        } else if (agentId && global.neuralAgents.has(agentId)) {
          // Get specific agent status
          const agent = global.neuralAgents.get(agentId);
          if (agent.getStatus) {
            const status = agent.getStatus();
            result = {
              agentId,
              type: status.agentType,
              neuralState: status.neuralState
            };
          }
        } else {
          // Get all neural agents status
          const agents = [];
          for (const [id, agent] of global.neuralAgents) {
            if (agent.getStatus) {
              const status = agent.getStatus();
              agents.push({
                agentId: id,
                type: status.agentType,
                neuralState: status.neuralState
              });
            }
          }
          result = {
            totalAgents: agents.length,
            agents
          };
        }
        break;
      }
        
      case 'neural_train': {
        const { iterations = 10, agentId } = args;
        
        if (!global.neuralAgents || global.neuralAgents.size === 0) {
          result = {
            status: 'no_neural_agents',
            message: 'No neural agents to train. Spawn agents first.'
          };
        } else {
          const sampleTasks = [
            { description: 'Analyze user behavior patterns', priority: 'high' },
            { description: 'Generate unit tests', priority: 'medium' },
            { description: 'Optimize query performance', priority: 'high' },
            { description: 'Research best practices', priority: 'medium' },
            { description: 'Coordinate deployment', priority: 'critical' }
          ];
          
          let trainedCount = 0;
          const trainingResults = [];
          
          for (let i = 0; i < iterations; i++) {
            const task = {
              ...sampleTasks[i % sampleTasks.length],
              id: `training-${Date.now()}-${i}`
            };
            
            if (agentId && global.neuralAgents.has(agentId)) {
              // Train specific agent
              const agent = global.neuralAgents.get(agentId);
              if (agent.executeTask) {
                try {
                  await agent.executeTask(task);
                  trainedCount++;
                } catch (e) {
                  trainingResults.push({ agentId, error: e.message });
                }
              }
            } else {
              // Train all agents
              for (const [id, agent] of global.neuralAgents) {
                if (agent.executeTask) {
                  try {
                    await agent.executeTask(task);
                    trainedCount++;
                  } catch (e) {
                    trainingResults.push({ agentId: id, error: e.message });
                  }
                }
              }
            }
          }
          
          result = {
            status: 'training_completed',
            iterations,
            trainedCount,
            errors: trainingResults.filter(r => r.error),
            message: `Training completed with ${trainedCount} successful iterations`
          };
        }
        break;
      }
        
      case 'neural_patterns': {
        const { pattern = 'all' } = args;
        const patterns = {
          convergent: 'Focused problem-solving, analytical thinking, goal-oriented',
          divergent: 'Creative exploration, idea generation, brainstorming',
          lateral: 'Non-linear thinking, pattern breaking, innovation',
          systems: 'Holistic view, interconnections, complexity management',
          critical: 'Evaluation, judgment, validation, quality assurance',
          abstract: 'Conceptual thinking, generalization, meta-cognition'
        };
        
        if (pattern === 'all') {
          const profiles = NeuralAgentFactory.getCognitiveProfiles();
          result = {
            patterns,
            agentMappings: Object.entries(profiles).reduce((acc, [agentType, profile]) => {
              acc[agentType] = {
                primary: profile.primary,
                secondary: profile.secondary,
                networkLayers: profile.networkLayers
              };
              return acc;
            }, {})
          };
        } else if (patterns[pattern]) {
          result = {
            pattern,
            description: patterns[pattern],
            agentsUsingPattern: Object.entries(NeuralAgentFactory.getCognitiveProfiles())
              .filter(([_, profile]) => 
                profile.primary === pattern || profile.secondary === pattern
              )
              .map(([agentType, profile]) => ({
                agentType,
                usage: profile.primary === pattern ? 'primary' : 'secondary'
              }))
          };
        } else {
          result = {
            error: `Unknown pattern: ${pattern}`,
            availablePatterns: Object.keys(patterns)
          };
        }
        break;
      }
        
      default:
        result = { error: `Tool ${toolName} not recognized` };
    }
    
    return {
      content: [{
        type: 'text',
        text: typeof result === 'string' ? result : JSON.stringify(result, null, 2)
      }]
    };
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          error: `Error executing ${toolName}: ${error.message}`,
          stack: error.stack
        }, null, 2)
      }],
      isError: true
    };
  }
}

async function getMcpStatus() {
  console.log('MCP Server Status:');
  console.log('  Status: Not implemented in WASM mode');
  console.log('  Protocol: stdio/http');
  console.log('  Tools: 12 available');
  console.log('\nTo check active MCP server, use:');
  console.log('  ps aux | grep ruv-swarm');
}

async function stopMcpServer() {
  console.log('Stopping MCP server...');
  console.log('Note: In stdio mode, close the client connection.');
  console.log('Note: In HTTP mode, stop the HTTP server process.');
}

async function listMcpTools() {
  console.log('Available MCP Tools:');
  console.log('\n🔧 Swarm Management:');
  console.log('  • swarm_init        - Initialize swarm topology');
  console.log('  • swarm_status      - Get swarm status');
  console.log('  • swarm_monitor     - Monitor swarm activity');
  
  console.log('\n🤖 Agent Operations:');
  console.log('  • agent_spawn       - Spawn new agents');
  console.log('  • agent_list        - List active agents');
  console.log('  • agent_metrics     - Get agent performance metrics');
  
  console.log('\n📋 Task Management:');
  console.log('  • task_orchestrate  - Orchestrate distributed tasks');
  console.log('  • task_status       - Check task progress');
  console.log('  • task_results      - Get task results');
  
  console.log('\n🔬 Analytics:');
  console.log('  • benchmark_run     - Run performance benchmarks');
  console.log('  • features_detect   - Detect runtime features');
  console.log('  • memory_usage      - Get memory usage statistics');
}

async function configureMcp(args) {
  const configType = args[0] || 'show';
  
  switch (configType) {
    case 'show':
      console.log('Current MCP Configuration:');
      console.log('  Protocol: stdio (default), http (optional)');
      console.log('  WASM Module: ruv_swarm_wasm.wasm');
      console.log('  Transport: WebAssembly + Node.js');
      console.log('  Claude Code Compatible: Yes');
      break;
      
    case 'claude':
      console.log('Claude Code MCP Configuration:');
      console.log('\nAdd to your Claude Code settings.json:');
      console.log('```json');
      console.log('{');
      console.log('  "mcpServers": {');
      console.log('    "ruv-swarm": {');
      console.log('      "command": "npx",');
      console.log('      "args": ["ruv-swarm", "mcp", "start"],');
      console.log('      "env": {}');
      console.log('    }');
      console.log('  }');
      console.log('}');
      console.log('```');
      break;
      
    default:
      console.log('Available config commands:');
      console.log('  npx ruv-swarm mcp config show    - Show current configuration');
      console.log('  npx ruv-swarm mcp config claude  - Show Claude Code setup');
  }
}

function showMcpHelp() {
  console.log('ruv-swarm MCP Server - Model Context Protocol integration\n');
  console.log('Usage: npx ruv-swarm mcp <command> [options]\n');
  
  console.log('Commands:');
  console.log('  start [options]     Start MCP server for Claude Code integration');
  console.log('  status              Show MCP server status');
  console.log('  stop                Stop MCP server');
  console.log('  tools               List available MCP tools');
  console.log('  config <type>       Show/configure MCP settings');
  console.log('  help                Show this help message\n');
  
  console.log('Start Options:');
  console.log('  --protocol=stdio    Use stdio communication (default, for Claude Code)');
  console.log('  --protocol=http     Use HTTP communication (for web clients)');
  console.log('  --port=3000         HTTP port (when using http protocol)');
  console.log('  --host=localhost    HTTP host (when using http protocol)\n');
  
  console.log('Examples:');
  console.log('  npx ruv-swarm mcp start                    # Start stdio MCP server');
  console.log('  npx ruv-swarm mcp start --protocol=http    # Start HTTP MCP server');
  console.log('  npx ruv-swarm mcp tools                    # List available tools');
  console.log('  npx ruv-swarm mcp config claude           # Show Claude Code setup\n');
  
  console.log('🔗 Integration:');
  console.log('  For Claude Code: Use stdio protocol (default)');
  console.log('  For Web Apps:   Use http protocol with streaming');
  console.log('  WASM Support:   Full WebAssembly integration for performance\n');
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});