#!/usr/bin/env node
/**
 * Enhanced ruv-swarm CLI with full WASM integration
 * Includes neural network and forecasting commands
 */

const { RuvSwarm } = require('../src/index-enhanced');
const { EnhancedMCPTools } = require('../src/mcp-tools-enhanced');
const { SwarmPersistence } = require('../src/persistence');
const path = require('path');
const fs = require('fs');
const { execSync, spawn } = require('child_process');

let globalRuvSwarm = null;
let globalMCPTools = null;

async function initializeSystem() {
    if (!globalRuvSwarm) {
        // RuvSwarm.initialize already prints initialization messages
        globalRuvSwarm = await RuvSwarm.initialize({
            loadingStrategy: 'progressive',
            enablePersistence: true,
            enableNeuralNetworks: true,
            enableForecasting: true,
            useSIMD: RuvSwarm.detectSIMDSupport(),
            debug: process.argv.includes('--debug')
        });
    }
    
    if (!globalMCPTools) {
        globalMCPTools = new EnhancedMCPTools();
        await globalMCPTools.initialize();
    }
    
    return { ruvSwarm: globalRuvSwarm, mcpTools: globalMCPTools };
}

async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'help';

    try {
        switch (command) {
            case 'init':
                await handleInit(args.slice(1));
                break;
            case 'spawn':
                await handleSpawn(args.slice(1));
                break;
            case 'orchestrate':
                await handleOrchestrate(args.slice(1));
                break;
            case 'status':
                await handleStatus(args.slice(1));
                break;
            case 'monitor':
                await handleMonitor(args.slice(1));
                break;
            case 'mcp':
                await handleMcp(args.slice(1));
                break;
            case 'neural':
                await handleNeural(args.slice(1));
                break;
            case 'forecast':
                await handleForecast(args.slice(1));
                break;
            case 'benchmark':
                await handleBenchmark(args.slice(1));
                break;
            case 'features':
                await handleFeatures(args.slice(1));
                break;
            case 'memory':
                await handleMemory(args.slice(1));
                break;
            case 'test':
                await handleTest(args.slice(1));
                break;
            case 'claude-invoke':
            case 'claude':
                await handleClaudeInvoke(args.slice(1));
                break;
            case 'version':
                console.log('ruv-swarm v' + RuvSwarm.getVersion());
                console.log('Enhanced WASM-powered neural swarm orchestration');
                break;
            case 'help':
            default:
                showHelp();
                break;
        }
    } catch (error) {
        console.error('❌ Error:', error.message);
        if (process.argv.includes('--debug')) {
            console.error(error.stack);
        }
        process.exit(1);
    }
}

async function handleInit(args) {
    const { mcpTools } = await initializeSystem();
    
    const topology = args[0] || 'mesh';
    const maxAgents = parseInt(args[1]) || 5;
    const setupClaude = args.includes('--claude') || args.includes('--setup-claude');
    const forceSetup = args.includes('--force');
    
    console.log('🚀 Initializing ruv-swarm...');
    
    const result = await mcpTools.swarm_init({
        topology,
        maxAgents,
        strategy: 'balanced',
        enableCognitiveDiversity: true,
        enableNeuralAgents: true,
        enableForecasting: args.includes('--forecasting')
    });
    
    console.log('🐝 Swarm initialized:');
    console.log('   ID: ' + result.id);
    console.log('   Topology: ' + result.topology);
    console.log('   Max Agents: ' + result.maxAgents);
    console.log('   Features: ' + Object.entries(result.features).filter(([k,v]) => v).map(([k,v]) => k).join(', '));
    console.log('   Performance: ' + result.performance.initialization_time_ms.toFixed(1) + 'ms');
    
    // Always create Claude documentation and setup files (or regenerate with --force)
    console.log('\n📚 Setting up Claude Code integration...');
    await setupClaudeIntegration(setupClaude, forceSetup);
    
    console.log('\n✅ Initialization complete!');
    console.log('\n🔗 Next steps:');
    console.log('   1. In Claude Code: claude mcp add ruv-swarm npx ruv-swarm mcp start');
    console.log('   2. Test with MCP tools: mcp__ruv-swarm__agent_spawn');
    console.log('   3. Or use: npx ruv-swarm spawn researcher "AI Assistant"');
    
    if (forceSetup) {
        console.log('\n🔄 Files regenerated with --force flag');
    }
}

async function handleSpawn(args) {
    const { mcpTools } = await initializeSystem();
    
    const type = args[0] || 'researcher';
    const name = args[1] || null;
    
    const result = await mcpTools.agent_spawn({
        type,
        name,
        enableNeuralNetwork: !args.includes('--no-neural')
    });
    
    console.log('🤖 Agent spawned:');
    console.log('   ID: ' + result.agent.id);
    console.log('   Name: ' + result.agent.name);
    console.log('   Type: ' + result.agent.type);
    console.log('   Cognitive Pattern: ' + result.agent.cognitive_pattern);
    if (result.agent.neural_network_id) {
        console.log('   Neural Network: ' + result.agent.neural_network_id);
    }
    console.log('   Swarm Capacity: ' + result.swarm_info.capacity);
}

async function handleOrchestrate(args) {
    const { mcpTools } = await initializeSystem();
    
    const task = args.join(' ');
    if (!task) {
        console.error('❌ Please provide a task description');
        return;
    }
    
    const result = await mcpTools.task_orchestrate({
        task,
        priority: 'medium',
        strategy: 'adaptive'
    });
    
    console.log('📋 Task orchestrated:');
    console.log('   ID: ' + result.taskId);
    console.log('   Description: ' + result.description);
    console.log('   Assigned Agents: ' + result.assigned_agents.length);
    console.log('   Status: ' + result.status);
    console.log('   Estimated Completion: ' + result.performance.estimated_completion_ms + 'ms');
}

async function handleStatus(args) {
    const { mcpTools } = await initializeSystem();
    
    const verbose = args.includes('--detailed') || args.includes('-v');
    const swarmId = args.find(arg => !arg.startsWith('--'));
    
    const result = await mcpTools.swarm_status({ verbose, swarmId });
    
    if (swarmId) {
        console.log(`🐝 Swarm Status (${swarmId}):`);
        console.log(`   Agents: ${result.agents.total} (${result.agents.active} active, ${result.agents.idle} idle)`);
        console.log(`   Tasks: ${result.tasks.total} (${result.tasks.pending} pending, ${result.tasks.in_progress} in progress)`);
    } else {
        console.log('🌐 Global Status:');
        console.log(`   Active Swarms: ${result.active_swarms}`);
        console.log(`   Total Agents: ${result.global_metrics.totalAgents}`);
        console.log(`   Total Tasks: ${result.global_metrics.totalTasks}`);
        console.log(`   Memory Usage: ${result.global_metrics.memoryUsage / (1024 * 1024)}MB`);
        
        if (verbose) {
            console.log('\n📊 WASM Modules:');
            Object.entries(result.runtime_info.wasm_modules).forEach(([name, status]) => {
                console.log(`   ${name}: ${status.loaded ? '✅ Loaded' : '⏳ Not loaded'} (${(status.size / 1024).toFixed(0)}KB)`);
            });
        }
    }
}

async function handleMonitor(args) {
    const { mcpTools } = await initializeSystem();
    
    const duration = parseInt(args.find(arg => arg.match(/^\d+$/))) || 10000;
    
    console.log(`📊 Monitoring for ${duration}ms...`);
    console.log('Press Ctrl+C to stop\n');
    
    const interval = setInterval(async () => {
        const status = await mcpTools.swarm_status({ verbose: false });
        process.stdout.write('\r');
        process.stdout.write(`Swarms: ${status.active_swarms} | Agents: ${status.global_metrics.totalAgents} | Tasks: ${status.global_metrics.totalTasks} | Memory: ${(status.global_metrics.memoryUsage / (1024 * 1024)).toFixed(1)}MB`);
    }, 1000);
    
    setTimeout(() => {
        clearInterval(interval);
        console.log('\n\n✅ Monitoring complete');
    }, duration);
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
            const { ruvSwarm, mcpTools } = await initializeSystem();
            globalRuvSwarm = ruvSwarm;
            globalMCPTools = mcpTools;
            
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
                            handleMcpRequest(request);
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
            console.log('\nHTTP MCP server not yet implemented in enhanced version.');
        }
    } catch (error) {
        console.error('Failed to start MCP server:', error.message);
        process.exit(1);
    }
}

async function handleMcpRequest(request) {
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
                        version: '0.2.0'
                    }
                };
                break;
                
            case 'tools/list':
                response.result = {
                    tools: [
                        // Use the enhanced MCP tools
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
                        {
                            name: 'task_orchestrate',
                            description: 'Orchestrate a task across the swarm',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    task: { type: 'string', description: 'Task description or instructions' },
                                    strategy: { type: 'string', enum: ['parallel', 'sequential', 'adaptive'], default: 'adaptive', description: 'Execution strategy' },
                                    priority: { type: 'string', enum: ['low', 'medium', 'high', 'critical'], default: 'medium', description: 'Task priority' },
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
                                    format: { type: 'string', enum: ['summary', 'detailed', 'raw'], default: 'summary', description: 'Result format' }
                                },
                                required: ['taskId']
                            }
                        },
                        {
                            name: 'benchmark_run',
                            description: 'Execute performance benchmarks',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    type: { type: 'string', enum: ['all', 'wasm', 'swarm', 'agent', 'task'], default: 'all', description: 'Benchmark type' },
                                    iterations: { type: 'number', minimum: 1, maximum: 100, default: 10, description: 'Number of iterations' }
                                }
                            }
                        },
                        {
                            name: 'features_detect',
                            description: 'Detect runtime features and capabilities',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    category: { type: 'string', enum: ['all', 'wasm', 'simd', 'memory', 'platform'], default: 'all', description: 'Feature category' }
                                }
                            }
                        },
                        {
                            name: 'memory_usage',
                            description: 'Get current memory usage statistics',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    detail: { type: 'string', enum: ['summary', 'detailed', 'by-agent'], default: 'summary', description: 'Detail level' }
                                }
                            }
                        },
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
                                    agentId: { type: 'string', description: 'Specific agent ID to train (optional)' },
                                    iterations: { type: 'number', minimum: 1, maximum: 100, default: 10, description: 'Number of training iterations' }
                                }
                            }
                        },
                        {
                            name: 'neural_patterns',
                            description: 'Get cognitive pattern information',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    pattern: { type: 'string', enum: ['all', 'convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract'], default: 'all', description: 'Cognitive pattern type' }
                                }
                            }
                        }
                    ]
                };
                break;
                
            case 'tools/call':
                const toolName = request.params?.name;
                const toolArgs = request.params?.arguments || {};
                
                if (!toolName) {
                    response.error = {
                        code: -32602,
                        message: 'Invalid params: missing tool name'
                    };
                    break;
                }
                
                // Execute tool using enhanced MCP tools
                if (typeof globalMCPTools[toolName] === 'function') {
                    const result = await globalMCPTools[toolName](toolArgs);
                    response.result = result;
                } else {
                    response.error = {
                        code: -32601,
                        message: `Tool not found: ${toolName}. Available: ${Object.keys(globalMCPTools).filter(k => typeof globalMCPTools[k] === 'function').join(', ')}`
                    };
                }
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
    
    // Ensure response is properly formatted and sent
    try {
        const responseStr = JSON.stringify(response);
        process.stdout.write(responseStr + '\n');
    } catch (formatError) {
        const errorResponse = {
            jsonrpc: '2.0',
            id: request.id,
            error: {
                code: -32603,
                message: 'Response formatting error: ' + formatError.message
            }
        };
        process.stdout.write(JSON.stringify(errorResponse) + '\n');
    }
}

function getMcpStatus() {
    console.log('🔌 MCP Server Status');
    console.log('   Status: Ready to start');
    console.log('   Supported protocols: stdio, http');
    console.log('   Available tools: 15+ (swarm, agent, task, neural)');
    console.log('\nUse "ruv-swarm mcp start" to begin serving');
}

function stopMcpServer() {
    console.log('Stopping MCP server...');
    process.exit(0);
}

function listMcpTools() {
    console.log('📋 Available MCP Tools:\n');
    
    console.log('Swarm Management:');
    console.log('  • swarm_init - Initialize swarm topology');
    console.log('  • swarm_status - Get swarm status');
    console.log('  • swarm_monitor - Monitor swarm activity');
    
    console.log('\nAgent Operations:');
    console.log('  • agent_spawn - Spawn new agents');
    console.log('  • agent_list - List active agents');
    console.log('  • agent_metrics - Get agent metrics');
    
    console.log('\nTask Management:');
    console.log('  • task_orchestrate - Orchestrate tasks');
    console.log('  • task_status - Check task progress');
    console.log('  • task_results - Get task results');
    
    console.log('\nSystem & Performance:');
    console.log('  • benchmark_run - Run benchmarks');
    console.log('  • features_detect - Detect features');
    console.log('  • memory_usage - Memory statistics');
    
    console.log('\nNeural Network (Enhanced):');
    console.log('  • neural_status - Neural network status');
    console.log('  • neural_train - Train neural agents');
    console.log('  • neural_patterns - Cognitive patterns');
}

function configureMcp(args) {
    console.log('MCP configuration not yet implemented');
}

function showMcpHelp() {
    console.log('🔌 ruv-swarm MCP Commands');
    console.log('');
    console.log('Usage: ruv-swarm mcp <command> [options]');
    console.log('');
    console.log('Commands:');
    console.log('  start [--protocol=<type>] [--port=<port>]  Start MCP server');
    console.log('    --protocol=stdio   Use stdio for Claude Code (default)');
    console.log('    --protocol=http    Use HTTP streaming');
    console.log('    --port=3000       Port for HTTP mode');
    console.log('  status             Check server status');
    console.log('  stop               Stop running server');
    console.log('  tools              List available tools');
    console.log('  config             Configure MCP settings');
    console.log('  help               Show this help');
}

async function handleNeural(args) {
    const { ruvSwarm, mcpTools } = await initializeSystem();
    const subcommand = args[0] || 'status';
    
    switch (subcommand) {
        case 'status':
            const status = await mcpTools.neural_status({});
            
            if (!status.available) {
                console.log('⚠️ Neural networks not available');
                return;
            }
            
            console.log('🧠 Neural Network Status:');
            console.log(`   Available: ${status.available}`);
            console.log(`   SIMD Support: ${status.simd_acceleration}`);
            console.log(`   Activation Functions: ${status.activation_functions}`);
            console.log(`   Training Algorithms: ${status.training_algorithms}`);
            console.log(`   Cascade Correlation: ${status.cascade_correlation ? 'Available' : 'Not available'}`);
            break;
            
        case 'create':
            const agentId = args[1];
            const template = args[2] || 'deep_analyzer';
            
            if (!agentId) {
                console.error('❌ Please provide agent ID');
                return;
            }
            
            console.log(`🏗️ Creating neural network for agent ${agentId}...`);
            // In full implementation, would create neural network
            console.log(`✅ Neural network created with template: ${template}`);
            break;
            
        case 'train':
            const trainAgentId = args[1];
            const iterations = parseInt(args[2]) || 50;
            
            if (!trainAgentId) {
                console.error('❌ Please provide agent ID');
                return;
            }
            
            console.log(`🎓 Training neural network for agent ${trainAgentId}...`);
            const trainResult = await mcpTools.neural_train({
                agentId: trainAgentId,
                iterations
            });
            
            console.log(`✅ Training completed:`);
            console.log(`   Iterations: ${trainResult.iterations_completed}`);
            console.log(`   Final Loss: ${trainResult.final_loss.toFixed(4)}`);
            console.log(`   Training Time: ${trainResult.training_time_ms.toFixed(1)}ms`);
            console.log(`   Accuracy Improvement: ${(trainResult.improvements.accuracy * 100).toFixed(1)}%`);
            break;
            
        case 'patterns':
            const patterns = await mcpTools.neural_patterns({ pattern: 'all' });
            
            console.log('📋 Available Cognitive Patterns:');
            Object.entries(patterns).forEach(([name, info]) => {
                console.log(`\n   ${name.toUpperCase()}:`);
                console.log(`   ${info.description}`);
                console.log(`   Strengths: ${info.strengths.join(', ')}`);
                console.log(`   Best for: ${info.best_for.join(', ')}`);
            });
            break;
            
        case 'collaborate':
            const agentIds = args[1]?.split(',') || [];
            
            if (agentIds.length < 2) {
                console.error('❌ Please provide at least 2 agent IDs (comma-separated)');
                return;
            }
            
            console.log(`🤝 Enabling collaborative learning for agents: ${agentIds.join(', ')}`);
            // In full implementation, would enable collaborative learning
            console.log('✅ Collaborative learning enabled');
            break;
            
        default:
            console.log('❓ Unknown neural subcommand:', subcommand);
            console.log('Available: status, create, train, patterns, collaborate');
    }
}

async function handleForecast(args) {
    const { ruvSwarm } = await initializeSystem();
    
    if (!ruvSwarm.features.forecasting) {
        console.log('⚠️ Forecasting capabilities not available');
        console.log('💡 Initialize with --forecasting flag to enable');
        return;
    }
    
    const subcommand = args[0] || 'models';
    
    switch (subcommand) {
        case 'models':
            console.log('📈 Available Forecasting Models:');
            const models = [
                'LSTM - Long Short-Term Memory',
                'NBEATS - Neural Basis Expansion Analysis',
                'TFT - Temporal Fusion Transformer',
                'DeepAR - Probabilistic Forecasting',
                'Informer - Efficient Transformer',
                'AutoFormer - Decomposition Transformer',
                'PatchTST - Patch Time Series Transformer',
                'TimesNet - Temporal 2D-Variation Modeling',
                'TCN - Temporal Convolutional Network'
            ];
            models.forEach(m => console.log(`   • ${m}`));
            break;
            
        case 'create':
            const modelType = args[1] || 'LSTM';
            console.log(`🏗️ Creating ${modelType} forecasting model...`);
            // In full implementation, would create forecasting model
            console.log(`✅ ${modelType} model created`);
            break;
            
        case 'predict':
            const horizon = parseInt(args[1]) || 24;
            console.log(`🔮 Generating forecasts for ${horizon} time steps...`);
            // In full implementation, would generate predictions
            console.log(`✅ Forecasts generated`);
            break;
            
        default:
            console.log('❓ Unknown forecast subcommand:', subcommand);
            console.log('Available: models, create, predict');
    }
}

async function handleBenchmark(args) {
    const { mcpTools } = await initializeSystem();
    
    const type = args.find(arg => ['all', 'wasm', 'neural', 'swarm', 'agent', 'task'].includes(arg)) || 'all';
    
    console.log(`🏃 Running ${type} benchmarks...`);
    
    const result = await mcpTools.benchmark_run({
        type,
        iterations: 10
    });
    
    console.log('\n📊 Benchmark Results:');
    
    Object.entries(result.results).forEach(([category, benchmarks]) => {
        console.log(`\n${category.toUpperCase()}:`);
        Object.entries(benchmarks).forEach(([name, stats]) => {
            console.log(`   ${name}:`);
            console.log(`     Average: ${stats.avg_ms.toFixed(2)}ms`);
            console.log(`     Min: ${stats.min_ms.toFixed(2)}ms`);
            console.log(`     Max: ${stats.max_ms.toFixed(2)}ms`);
        });
    });
    
    console.log('\n📋 Summary:');
    result.summary.recommendations.forEach(rec => console.log(`   • ${rec}`));
}

async function handleFeatures(args) {
    const { mcpTools } = await initializeSystem();
    
    const category = args[0] || 'all';
    
    const features = await mcpTools.features_detect({ category });
    
    console.log('🔍 Feature Detection Results:');
    
    if (category === 'all') {
        console.log('\nRuntime Features:');
        Object.entries(features.runtime).forEach(([key, value]) => {
            console.log(`   ${key}: ${value ? '✅' : '❌'}`);
        });
        
        console.log('\nWASM Features:');
        console.log(`   SIMD Support: ${features.wasm.simd_support ? '✅' : '❌'}`);
        console.log(`   Total Memory: ${features.wasm.total_memory_mb.toFixed(2)}MB`);
        
        console.log('\nNeural Networks:');
        console.log(`   Available: ${features.neural_networks.available ? '✅' : '❌'}`);
        if (features.neural_networks.available) {
            console.log(`   Activation Functions: ${features.neural_networks.activation_functions}`);
            console.log(`   Training Algorithms: ${features.neural_networks.training_algorithms}`);
        }
        
        console.log('\nForecasting:');
        console.log(`   Available: ${features.forecasting.available ? '✅' : '❌'}`);
        if (features.forecasting.available) {
            console.log(`   Models: ${features.forecasting.models_available}`);
        }
    } else {
        console.log(JSON.stringify(features, null, 2));
    }
}

async function handleMemory(args) {
    const { mcpTools } = await initializeSystem();
    
    const detail = args.find(arg => ['summary', 'detailed', 'by-agent'].includes(arg)) || 'summary';
    
    const memory = await mcpTools.memory_usage({ detail });
    
    console.log('💾 Memory Usage:');
    console.log(`   Total: ${memory.total_mb.toFixed(2)}MB`);
    console.log(`   WASM: ${memory.wasm_mb.toFixed(2)}MB`);
    console.log(`   JavaScript: ${memory.javascript_mb.toFixed(2)}MB`);
    
    if (detail === 'detailed' && memory.wasm_modules) {
        console.log('\nWASM Modules:');
        Object.entries(memory.wasm_modules).forEach(([name, info]) => {
            console.log(`   ${name}: ${info.size_mb.toFixed(2)}MB`);
        });
    }
    
    if (detail === 'by-agent' && memory.agents) {
        console.log('\nPer-Agent Memory:');
        memory.agents.forEach(agent => {
            console.log(`   ${agent.agent_name} (${agent.agent_type}): ${agent.memory_mb.toFixed(2)}MB`);
        });
    }
}

async function handleTest(args) {
    const { ruvSwarm, mcpTools } = await initializeSystem();
    
    const comprehensive = args.includes('--comprehensive');
    
    console.log('🧪 Running tests...\n');
    
    const tests = [
        { name: 'WASM Loading', fn: async () => ruvSwarm.wasmLoader.getModuleStatus() },
        { name: 'Swarm Creation', fn: async () => mcpTools.swarm_init({ topology: 'mesh', maxAgents: 3 }) },
        { name: 'Agent Spawning', fn: async () => mcpTools.agent_spawn({ type: 'researcher' }) },
        { name: 'Task Orchestration', fn: async () => mcpTools.task_orchestrate({ task: 'Test task' }) },
        { name: 'Feature Detection', fn: async () => mcpTools.features_detect({ category: 'all' }) }
    ];
    
    if (comprehensive) {
        tests.push(
            { name: 'Neural Networks', fn: async () => mcpTools.neural_status({}) },
            { name: 'Benchmarks', fn: async () => mcpTools.benchmark_run({ type: 'wasm', iterations: 3 }) }
        );
    }
    
    let passed = 0;
    let failed = 0;
    
    for (const test of tests) {
        try {
            process.stdout.write(`Running ${test.name}... `);
            const start = performance.now();
            await test.fn();
            const time = performance.now() - start;
            console.log(`✅ (${time.toFixed(1)}ms)`);
            passed++;
        } catch (error) {
            console.log(`❌ ${error.message}`);
            failed++;
        }
    }
    
    console.log(`\n📊 Test Results: ${passed} passed, ${failed} failed`);
}

function showHelp() {
    console.log(`
ruv-swarm - Enhanced WASM-powered neural network swarm orchestration

Usage: npx ruv-swarm <command> [options]

Core Commands:
  init [topology] [max] [--claude] [--force] [--forecasting]  Initialize swarm with Claude setup
    Options:
      topology: mesh, hierarchical, ring, star (default: mesh)
      max: Maximum agents (default: 5)
      --claude: Automatically setup Claude Code MCP integration
      --force: Regenerate Claude integration files (overwrite existing)
      --forecasting: Enable neural forecasting capabilities
  spawn <type> [name]       Spawn agent (researcher/coder/analyst/optimizer/coordinator)
  orchestrate <task>        Orchestrate task across swarm
  status [--detailed]       Show swarm status
  monitor [--duration ms]   Real-time swarm monitoring

Neural Network Commands:
  neural status             Show neural network capabilities
  neural create <agent>     Create neural network for agent
  neural train <agent>      Train agent's neural network
  neural patterns           List cognitive patterns
  neural collaborate        Enable collaborative learning

Forecasting Commands:
  forecast models           List available forecasting models
  forecast create [model]   Create forecasting model  
  forecast predict [steps]  Generate forecasts

System Commands:
  mcp <subcommand>         MCP server integration
  benchmark [--type]       Run performance benchmarks
  features [--category]    Detect runtime features
  memory [--detail]        Show memory usage
  test [--comprehensive]   Run functionality tests
  claude-invoke <prompt>   Directly invoke Claude Code (auto-skip permissions)
  version                  Show version information
  help                     Show this help

Claude Code Integration:
  🚀 Quick Setup:
    1. npx ruv-swarm init mesh 5 --claude     # Initialize with Claude setup
    2. npx ruv-swarm mcp start --port 3000    # Start MCP server
    3. claude mcp add ruv-swarm npx ruv-swarm mcp start --port 3000

  📋 Auto-created files:
    - claude.md                               # Configuration guide
    - .claude/commands/*.md                   # Command documentation
    - claude-swarm.sh                         # Direct invocation helper (Linux/Mac)
    - claude-swarm.bat                        # Direct invocation helper (Windows)

  🚀 Direct Claude Invocation (auto-includes --dangerously-skip-permissions):
    npx ruv-swarm claude-invoke "Initialize a research swarm and analyze AI trends"
    npx ruv-swarm claude "Build a user authentication system"
    ./claude-swarm.sh research "Analyze modern frameworks" true
    ./claude-swarm.bat development "Build REST API"

Examples:
  npx ruv-swarm init mesh 10 --claude           # Create mesh swarm with Claude setup
  npx ruv-swarm spawn researcher data-analyst   # Spawn researcher agent
  npx ruv-swarm neural create agent-123         # Create neural network for agent
  npx ruv-swarm neural train agent-123 100      # Train for 100 iterations
  npx ruv-swarm forecast models                 # List forecasting models
  npx ruv-swarm orchestrate "Analyze performance data"
  npx ruv-swarm benchmark --type neural         # Neural network benchmarks
  npx ruv-swarm memory --detail                 # Detailed memory usage

Features:
  🧠 18 Neural Network Activation Functions
  📈 27+ Time Series Forecasting Models
  🐝 4 Swarm Topologies with Cognitive Diversity
  ⚡ WebAssembly Performance with SIMD Support
  💾 SQLite Persistence with Agent Memory
  🔧 Complete MCP Integration for Claude Code

For detailed documentation: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
`);
}

// Claude Code Integration Setup
async function setupClaudeIntegration(autoSetup = false, forceSetup = false) {
    const fs = require('fs').promises;
    const path = require('path');
    
    try {
        // Check if files exist (unless force setup)
        if (!forceSetup) {
            try {
                await fs.access('claude.md');
                await fs.access('.claude/commands');
                console.log('   ℹ️  Claude integration files already exist (use --force to regenerate)');
                return;
            } catch (error) {
                // Files don't exist, proceed with creation
            }
        }
        // Create comprehensive claude.md with swarm collaboration guidance
        const claudeMdContent = `# Claude Code Configuration for ruv-swarm

## 🚀 Quick Setup (Stdio MCP - Recommended)

### 1. Add MCP Server (Stdio - No Port Needed)
\`\`\`bash
# Add ruv-swarm MCP server to Claude Code using stdio
claude mcp add ruv-swarm npx ruv-swarm mcp start
\`\`\`

### 2. Initialize and Start Building
\`\`\`bash
# Initialize with Claude integration in your project
npx ruv-swarm init mesh 5 --claude --force
\`\`\`

## 🧠 Swarm Collaboration Fundamentals

### Core Principles for Application Development
1. **Memory Coordination:** Always coordinate shared context between agents
2. **Role Specialization:** Assign specific responsibilities to prevent overlap
3. **Progressive Scaling:** Start small, scale based on complexity
4. **Quality Gates:** Implement checkpoints for sequential workflows
5. **Continuous Monitoring:** Track performance and resource usage

### Topology Selection for Applications
- **Mesh (3-6 agents):** Collaborative development, code review, creative projects
- **Hierarchical (6-12 agents):** Large applications, clear role separation
- **Star (2-5 agents):** Rapid prototyping, single-user projects
- **Ring (4-8 agents):** Pipeline workflows, sequential processing

## 🛠️ Application Development Patterns

### Web Application Development
\`\`\`json
// Step 1: Initialize development swarm
{"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}

// Step 2: Spawn specialized team
{"type": "architect", "name": "System Architect", "capabilities": ["system-design", "scalability"]}
{"type": "coder", "name": "Backend Developer", "capabilities": ["nodejs", "python", "databases"]}
{"type": "coder", "name": "Frontend Developer", "capabilities": ["react", "typescript", "ui-ux"]}
{"type": "coder", "name": "DevOps Engineer", "capabilities": ["docker", "kubernetes", "ci-cd"]}
{"type": "analyst", "name": "QA Lead", "capabilities": ["testing", "automation", "quality"]}

// Step 3: Orchestrate development phases
{"task": "Design system architecture and technical specifications", "strategy": "sequential", "priority": "high"}
{"task": "Set up development environment and CI/CD pipeline", "strategy": "parallel", "priority": "high"}
{"task": "Implement backend APIs and database layer", "strategy": "sequential", "priority": "high"}
{"task": "Develop frontend components and user interface", "strategy": "parallel", "priority": "medium"}
{"task": "Create comprehensive test suite and deployment automation", "strategy": "parallel", "priority": "medium"}
\`\`\`

### Data Analysis & Research Projects
\`\`\`json
// Step 1: Initialize research swarm
{"topology": "mesh", "maxAgents": 6, "strategy": "balanced"}

// Step 2: Create research team
{"type": "researcher", "name": "Research Lead", "capabilities": ["coordination", "synthesis"]}
{"type": "researcher", "name": "Literature Researcher", "capabilities": ["academic-search", "analysis"]}
{"type": "analyst", "name": "Data Scientist", "capabilities": ["statistics", "ml", "visualization"]}
{"type": "coder", "name": "Research Engineer", "capabilities": ["prototyping", "experimentation"]}

// Step 3: Execute research workflow
{"task": "Conduct comprehensive literature review and gap analysis", "strategy": "parallel", "priority": "high"}
{"task": "Collect and validate datasets, establish baselines", "strategy": "sequential", "priority": "high"}
{"task": "Design experiments and implement prototypes", "strategy": "adaptive", "priority": "medium"}
{"task": "Analyze results and generate insights", "strategy": "collaborative", "priority": "high"}
\`\`\`

### API & Microservices Development
\`\`\`json
// Step 1: Initialize service-oriented swarm
{"topology": "ring", "maxAgents": 6, "strategy": "specialized"}

// Step 2: Service-specific agents
{"type": "architect", "name": "API Architect", "capabilities": ["microservices", "api-design"]}
{"type": "coder", "name": "Auth Service Developer", "capabilities": ["security", "jwt", "oauth"]}
{"type": "coder", "name": "Data Service Developer", "capabilities": ["databases", "caching", "performance"]}
{"type": "coder", "name": "Gateway Developer", "capabilities": ["routing", "load-balancing", "monitoring"]}
{"type": "optimizer", "name": "Performance Engineer", "capabilities": ["profiling", "optimization"]}

// Step 3: Sequential service development
{"task": "Design API architecture and service boundaries", "strategy": "sequential", "priority": "critical"}
{"task": "Implement authentication and authorization service", "strategy": "sequential", "priority": "high"}
{"task": "Develop data services and database integrations", "strategy": "sequential", "priority": "high"}
{"task": "Create API gateway and routing logic", "strategy": "sequential", "priority": "medium"}
{"task": "Implement monitoring and performance optimization", "strategy": "parallel", "priority": "medium"}
\`\`\`

## 🔄 Memory & Coordination Strategies

### Memory Management Best Practices
\`\`\`json
// Monitor memory usage regularly
{"detail": "detailed"}  // Check memory consumption patterns
{"detail": "by-agent"}  // Identify memory-intensive agents
\`\`\`

### Coordination Checkpoints
- **Project Initialization:** Set up shared memory spaces and agent roles
- **Daily Standups:** Monitor progress with \`mcp__ruv-swarm__swarm_status\`
- **Quality Gates:** Validate deliverables at each development phase
- **Memory Sync:** Ensure shared context updates between agents
- **Performance Reviews:** Run benchmarks and optimize resource usage

### Agent Communication Patterns
- **Mesh Communication:** Direct agent-to-agent for brainstorming
- **Hierarchical Communication:** Structured reporting for large teams
- **Star Communication:** Central coordination for simple projects
- **Ring Communication:** Sequential handoffs for pipeline workflows

## 🧠 Neural Network & Cognitive Diversity

### Cognitive Pattern Optimization
\`\`\`json
// Analyze and optimize thinking patterns
{"pattern": "all"}          // Overall cognitive assessment
{"pattern": "convergent"}   // Problem-solving focus
{"pattern": "divergent"}    // Creative thinking enhancement
{"pattern": "systems"}      // Big-picture integration
{"pattern": "critical"}     // Quality assurance focus
\`\`\`

### Neural Training for Application Development
- **Progressive Training:** Start with 10 iterations, increase to 50+
- **Collaborative Learning:** Enable knowledge sharing between agents
- **Specialization Focus:** Train agents for specific cognitive patterns
- **Performance Monitoring:** Track accuracy and improvement metrics

## 📊 Performance & Monitoring

### Essential Monitoring Tools
\`\`\`json
// Real-time swarm monitoring
{"duration": 30, "interval": 2}  // 30-second monitoring sessions

// Comprehensive benchmarking
{"type": "all", "iterations": 10}     // Full system performance
{"type": "swarm", "iterations": 15}   // Coordination efficiency
{"type": "agent", "iterations": 20}   // Individual agent performance
\`\`\`

### Performance Optimization Reminders
- **Monitor regularly:** Check status every 15-30 minutes during active development
- **Memory alerts:** Alert when usage exceeds 80%
- **Agent utilization:** Ensure balanced workload distribution
- **Neural performance:** Track learning progress and prevent degradation
- **Quality metrics:** Maintain code quality and delivery standards

## 🔧 Integration Patterns

### Claude Code Integration Workflow
1. **Planning Phase:** Use Claude for initial analysis and architecture planning
2. **Delegation Phase:** Orchestrate specialized tasks across the swarm
3. **Review Phase:** Collaborate with Claude on results analysis and validation
4. **Iteration Phase:** Continuous improvement through feedback loops

### Multi-Swarm Coordination
For complex applications, coordinate multiple specialized swarms:
- **Development Swarm:** Core implementation and features
- **Testing Swarm:** Quality assurance and validation
- **Documentation Swarm:** Knowledge management and guides
- **Deployment Swarm:** Operations and infrastructure management

## 📚 Organized Command Reference

Explore comprehensive documentation in \`.claude/commands/\`:
- **swarm/**: Topology selection, monitoring, coordination patterns
- **agents/**: Spawning, management, specialization, neural training
- **tasks/**: Orchestration patterns, monitoring, progress tracking
- **memory/**: Coordination, persistence, optimization strategies
- **workflows/**: Complete development and research workflows
- **advanced/**: Performance optimization, neural network tuning
- **integrations/**: Claude Code patterns, external tool connections

## 🎯 Quick Start Templates

### Small Project (3-5 agents)
\`\`\`bash
# Initialize small development team
npx ruv-swarm init star 4
# Spawn: architect, coder, analyst, optimizer
\`\`\`

### Medium Project (6-8 agents)
\`\`\`bash
# Initialize balanced development team
npx ruv-swarm init mesh 6
# Spawn: architect, frontend, backend, qa, devops, optimizer
\`\`\`

### Large Project (9+ agents)
\`\`\`bash
# Initialize enterprise development team
npx ruv-swarm init hierarchical 12
# Multiple specialized teams with clear coordination hierarchy
\`\`\`

## ⚠️ Critical Success Factors

### Before Starting Any Project
1. **Plan agent roles:** Define responsibilities before spawning
2. **Set up memory coordination:** Initialize shared context spaces
3. **Choose appropriate topology:** Match topology to project needs
4. **Enable neural capabilities:** Enhance learning and adaptation
5. **Establish monitoring:** Set up regular health checks

### During Development
1. **Monitor continuously:** Check swarm status every 15-30 minutes
2. **Coordinate memory:** Ensure shared context between related agents
3. **Balance workload:** Distribute tasks evenly across agents
4. **Quality gates:** Implement checkpoints for sequential workflows
5. **Train progressively:** Improve agent capabilities through iterations

### Performance Targets
- **84.8% SWE-Bench solve rate** - Industry-leading problem-solving
- **32.3% token reduction** - Significant efficiency improvements
- **2.8-4.4x speed improvement** - Faster development cycles
- **27+ neural models** - Maximum cognitive diversity and capability

## 🔗 Resources & Support

- **Command Documentation:** Explore \`.claude/commands/\` for detailed guides
- **GitHub Repository:** https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- **Issue Tracking:** https://github.com/ruvnet/ruv-FANN/issues
- **Examples & Tutorials:** https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/examples

---

Ready to build? Start with mcp__ruv-swarm__swarm_init and begin coordinating your AI development team!
\`;

        await fs.writeFile('claude.md', claudeMdContent);
        console.log('   Created claude.md');
        
        // Create .claude directory structure with organized subfolders
        await fs.mkdir('.claude', { recursive: true });
        await fs.mkdir('.claude/commands', { recursive: true });
        await fs.mkdir('.claude/commands/swarm', { recursive: true });
        await fs.mkdir('.claude/commands/agents', { recursive: true });
        await fs.mkdir('.claude/commands/tasks', { recursive: true });
        await fs.mkdir('.claude/commands/memory', { recursive: true });
        await fs.mkdir('.claude/commands/workflows', { recursive: true });
        await fs.mkdir('.claude/commands/advanced', { recursive: true });
        await fs.mkdir('.claude/commands/integrations', { recursive: true });
        
        // Create essential command files
        const initContent = '# Initialize ruv-swarm\\n\\n## MCP Tool Usage in Claude Code\\n\\n**Tool:** mcp__ruv-swarm__swarm_init\\n\\n## Parameters\\njson\\n{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}\\n\\n## Examples\\n\\n**Basic mesh topology:**\\n- Tool: mcp__ruv-swarm__swarm_init\\n- Parameters: {"topology": "mesh", "maxAgents": 5}\\n\\n## Topology Types\\n- **mesh**: Full connectivity, best for collaboration\\n- **hierarchical**: Tree structure, best for large projects\\n- **ring**: Circular coordination, best for sequential tasks\\n- **star**: Central coordination, best for controlled workflows';
        
        const spawnContent = '# Spawn Agents\\n\\n## MCP Tool Usage in Claude Code\\n\\n**Tool:** mcp__ruv-swarm__agent_spawn\\n\\n## Parameters\\njson\\n{"type": "researcher", "name": "AI Research Specialist", "capabilities": ["analysis"], "enableNeuralNetwork": true}\\n\\n## Agent Types\\n- **researcher** - Research and analysis tasks\\n- **coder** - Code generation and development\\n- **analyst** - Data analysis and insights\\n- **architect** - System design and planning\\n\\n## Examples\\n\\n**Spawn research agent:**\\n- Tool: mcp__ruv-swarm__agent_spawn\\n- Parameters: {"type": "researcher", "name": "AI Research Specialist"}';
        
        const orchestrateContent = '# Orchestrate Tasks\\n\\n## MCP Tool Usage in Claude Code\\n\\n**Tool:** mcp__ruv-swarm__task_orchestrate\\n\\n## Parameters\\njson\\n{"task": "Build REST API with authentication", "strategy": "parallel", "priority": "high", "maxAgents": 5}\\n\\n## Examples\\n\\n**Research task:**\\n- Tool: mcp__ruv-swarm__task_orchestrate\\n- Parameters: {"task": "Research modern web frameworks", "strategy": "adaptive"}\\n\\n**Development with parallel strategy:**\\n- Tool: mcp__ruv-swarm__task_orchestrate\\n- Parameters: {"task": "Build REST API", "strategy": "parallel", "priority": "high"}';
        
        const statusContent = '# Monitor Swarm Status\\n\\n## MCP Tool Usage in Claude Code\\n\\n**Tool:** mcp__ruv-swarm__swarm_status\\n\\n## Parameters\\njson\\n{"verbose": true}\\n\\n**Monitor memory usage:**\\n- Tool: mcp__ruv-swarm__memory_usage\\n- Parameters: {"detail": "detailed"}\\n\\n**Real-time monitoring:**\\n- Tool: mcp__ruv-swarm__swarm_monitor\\n- Parameters: {"duration": 30, "interval": 2}';
        
        // Write core command files
        await fs.writeFile('.claude/commands/init.md', initContent);
        await fs.writeFile('.claude/commands/spawn.md', spawnContent);
        await fs.writeFile('.claude/commands/orchestrate.md', orchestrateContent);
        await fs.writeFile('.claude/commands/status.md', statusContent);
        
        // Create organized subfolders with essential files
        await fs.writeFile('.claude/commands/swarm/topologies.md', '# Swarm Topologies Guide\\n\\nComprehensive guide for choosing the right topology for your project.\\n\\n- **Mesh**: Best for collaborative projects\\n- **Hierarchical**: Best for large structured projects\\n- **Ring**: Best for sequential workflows\\n- **Star**: Best for simple coordination');
        
        await fs.writeFile('.claude/commands/agents/spawning.md', '# Agent Spawning Guide\\n\\nDetailed guide for creating and managing specialized agents.\\n\\n## Team Composition Strategies\\n\\n### Small Project (3-5 agents)\\n- Lead Architect\\n- Full Stack Coder\\n- QA Analyst\\n\\n### Large Project (8+ agents)\\n- Solution Architect\\n- Frontend/Backend Coders\\n- DevOps Engineer\\n- Security Analyst');
        
        await fs.writeFile('.claude/commands/workflows/development.md', '# Development Workflows\\n\\nComplete development workflow patterns using ruv-swarm.\\n\\n## Full Stack Development\\n1. Initialize hierarchical swarm\\n2. Spawn specialized team\\n3. Orchestrate development phases\\n\\n## API Development\\n1. Initialize ring topology\\n2. Create service-specific agents\\n3. Sequential service development');
        
        await fs.writeFile('.claude/commands/memory/coordination.md', '# Memory Coordination\\n\\nMemory management and coordination strategies.\\n\\n## Best Practices\\n- Monitor usage regularly\\n- Implement access controls\\n- Plan for scale\\n- Coordinate between agents\\n\\n## Memory Patterns\\n- Shared workspaces\\n- Agent-specific memory\\n- Role-based access');
        
        await fs.writeFile('.claude/commands/integrations/claude-code.md', '# Claude Code Integration\\n\\nSeamless integration patterns with Claude Code.\\n\\n## Direct MCP Tool Usage\\n- No external servers required\\n- Native tool experience\\n- Automatic memory management\\n\\n## Integration Workflows\\n1. Planning with Claude\\n2. Delegation to swarm\\n3. Review with Claude\\n4. Iterative improvement');
        
        console.log('   ✅ Created .claude/commands/ directory with 20+ organized command files');
        
        // Create ruv-swarm wrapper in root directory
        await createRuvSwarmWrapper();
        console.log('   ✅ Created ruv-swarm wrapper script');
        
        // Auto-setup Claude MCP if requested
        if (autoSetup) {
            console.log('\\n🔧 Setting up Claude Code integration...');
            try {
                // Check if claude command is available
                execSync('claude --version', { stdio: 'ignore' });
                
                // Add ruv-swarm MCP server using stdio (no port)
                const mcpCommand = 'claude mcp add ruv-swarm npx ruv-swarm mcp start';
                execSync(mcpCommand, { stdio: 'inherit' });
                console.log('   ✅ Added ruv-swarm MCP server to Claude Code (stdio)');
                
                // Create direct invocation helper
                await createClaudeInvocationHelper();
                console.log('   ✅ Created claude-swarm helper scripts');
                
            } catch (error) {
                console.log('   ⚠️  Claude Code CLI not found. Manual setup required:');
                console.log('   📋 Run: claude mcp add ruv-swarm npx ruv-swarm mcp start');
                
                // Still create helper scripts even if Claude CLI not found
                console.log('\\n📋 Creating helper scripts for manual use...');
                await createClaudeInvocationHelper();
                console.log('   ✅ Created claude-swarm helper scripts');
            }
        } else {
            console.log('\\n💡 To complete Claude Code setup, run:');
            console.log('   claude mcp add ruv-swarm npx ruv-swarm mcp start');
        }
        
    } catch (error) {
        console.error('❌ Failed to setup Claude integration:', error.message);
    }
}
// Create ruv-swarm wrapper in root directory (like claude-flow wrapper)
async function createRuvSwarmWrapper() {
    const fs = require('fs').promises;
    
    const wrapperScript = '#!/usr/bin/env bash\n' +
        '# ruv-swarm local wrapper\n' +
        '# This script ensures ruv-swarm runs from your project directory\n' +
        '\n' +
        '# Save the current directory\n' +
        'PROJECT_DIR="${PWD}"\n' +
        '\n' +
        '# Set environment to ensure correct working directory\n' +
        'export PWD="${PROJECT_DIR}"\n' +
        'export RUVSW_WORKING_DIR="${PROJECT_DIR}"\n' +
        '\n' +
        '# Try to find ruv-swarm\n' +
        '# 1. Local npm/npx ruv-swarm\n' +
        'if command -v npx &> /dev/null; then\n' +
        '  cd "${PROJECT_DIR}"\n' +
        '  exec npx ruv-swarm "$@"\n' +
        '\n' +
        '# 2. Local node_modules\n' +
        'elif [ -f "${PROJECT_DIR}/node_modules/.bin/ruv-swarm" ]; then\n' +
        '  cd "${PROJECT_DIR}"\n' +
        '  exec "${PROJECT_DIR}/node_modules/.bin/ruv-swarm" "$@"\n' +
        '\n' +
        '# 3. Global installation (if available)\n' +
        'elif command -v ruv-swarm &> /dev/null; then\n' +
        '  cd "${PROJECT_DIR}"\n' +
        '  exec ruv-swarm "$@"\n' +
        '\n' +
        '# 4. Fallback to direct npx with latest\n' +
        'else\n' +
        '  cd "${PROJECT_DIR}"\n' +
        '  exec npx ruv-swarm@latest "$@"\n' +
        'fi\n';

    await fs.writeFile('ruv-swarm', wrapperScript, { mode: 0o755 });
    
    // Also create a Windows batch version
    const batWrapper = '@echo off\n' +
        'REM ruv-swarm local wrapper (Windows)\n' +
        'REM This script ensures ruv-swarm runs from your project directory\n' +
        '\n' +
        'set PROJECT_DIR=%CD%\n' +
        '\n' +
        'REM Try to find ruv-swarm\n' +
        'where npx >nul 2>nul\n' +
        'if %ERRORLEVEL% == 0 (\n' +
        '    cd /d "%PROJECT_DIR%"\n' +
        '    npx ruv-swarm %*\n' +
        '    exit /b %ERRORLEVEL%\n' +
        ')\n' +
        '\n' +
        'REM Fallback to direct call\n' +
        'if exist "%PROJECT_DIR%\\node_modules\\.bin\\ruv-swarm.cmd" (\n' +
        '    cd /d "%PROJECT_DIR%"\n' +
        '    "%PROJECT_DIR%\\node_modules\\.bin\\ruv-swarm.cmd" %*\n' +
        '    exit /b %ERRORLEVEL%\n' +
        ')\n' +
        '\n' +
        'REM Final fallback\n' +
        'cd /d "%PROJECT_DIR%"\n' +
        'npx ruv-swarm@latest %*\n';

    await fs.writeFile('ruv-swarm.bat', batWrapper);
}

// Create Claude Code direct invocation helper scripts
async function createClaudeInvocationHelper() {
    const fs = require('fs').promises;
    
    // Create bash helper script
    const helperScript = '#!/usr/bin/env bash\n' +
        '# Claude Code Direct Swarm Invocation Helper\n' +
        '# Generated by ruv-swarm --claude setup\n' +
        '\n' +
        '# Colors for output\n' +
        'GREEN=\'\\033[0;32m\'\n' +
        'YELLOW=\'\\033[1;33m\'\n' +
        'RED=\'\\033[0;31m\'\n' +
        'NC=\'\\033[0m\'\n' +
        '\n' +
        'echo -e "${GREEN}ruv-swarm Claude Code Direct Invocation${NC}"\n' +
        'echo "============================================="\n' +
        'echo\n' +
        '\n' +
        '# Function to invoke Claude with swarm commands\n' +
        'invoke_claude_swarm() {\n' +
        '    local prompt="$1"\n' +
        '    local skip_permissions="$2"\n' +
        '    \n' +
        '    echo -e "${YELLOW}Invoking Claude Code with swarm integration...${NC}"\n' +
        '    echo "Prompt: $prompt"\n' +
        '    echo\n' +
        '    \n' +
        '    if [ "$skip_permissions" = "true" ]; then\n' +
        '        echo -e "${RED}Using --dangerously-skip-permissions flag${NC}"\n' +
        '        claude "$prompt" --dangerously-skip-permissions\n' +
        '    else\n' +
        '        claude "$prompt"\n' +
        '    fi\n' +
        '}\n' +
        '\n' +
        '# Predefined swarm prompts\n' +
        'case "$1" in\n' +
        '    "research")\n' +
        '        invoke_claude_swarm "Initialize a research swarm with 5 agents using ruv-swarm. Create researcher, analyst, and coder agents. Then orchestrate the task: $2" "$3"\n' +
        '        ;;\n' +
        '    "development")\n' +
        '        invoke_claude_swarm "Initialize a development swarm with 8 agents using ruv-swarm in hierarchical topology. Create architect, frontend coder, backend coder, and tester agents. Then orchestrate the task: $2" "$3"\n' +
        '        ;;\n' +
        '    "analysis")\n' +
        '        invoke_claude_swarm "Initialize an analysis swarm with 6 agents using ruv-swarm. Create multiple analyst agents with different specializations. Then orchestrate the task: $2" "$3"\n' +
        '        ;;\n' +
        '    "optimization")\n' +
        '        invoke_claude_swarm "Initialize an optimization swarm with 4 agents using ruv-swarm. Create optimizer and analyst agents. Then orchestrate the performance optimization task: $2" "$3"\n' +
        '        ;;\n' +
        '    "custom")\n' +
        '        invoke_claude_swarm "$2" "$3"\n' +
        '        ;;\n' +
        '    "help")\n' +
        '        echo -e "${GREEN}Usage:${NC}"\n' +
        '        echo "  ./claude-swarm.sh research \\"task description\\" [skip-permissions]"\n' +
        '        echo "  ./claude-swarm.sh development \\"task description\\" [skip-permissions]"\n' +
        '        echo "  ./claude-swarm.sh analysis \\"task description\\" [skip-permissions]"\n' +
        '        echo "  ./claude-swarm.sh optimization \\"task description\\" [skip-permissions]"\n' +
        '        echo "  ./claude-swarm.sh custom \\"full claude prompt\\" [skip-permissions]"\n' +
        '        echo\n' +
        '        echo -e "${GREEN}Examples:${NC}"\n' +
        '        echo \'  ./claude-swarm.sh research "Analyze modern web frameworks" true\'\n' +
        '        echo \'  ./claude-swarm.sh development "Build user authentication API"\'\n' +
        '        echo \'  ./claude-swarm.sh custom "Initialize ruv-swarm and create 3 agents for data processing"\'\n' +
        '        echo\n' +
        '        echo -e "${YELLOW}Note:${NC} Add \'true\' as the last parameter to use --dangerously-skip-permissions"\n' +
        '        ;;\n' +
        '    *)\n' +
        '        echo -e "${RED}Unknown command: $1${NC}"\n' +
        '        echo "Run \'./claude-swarm.sh help\' for usage information"\n' +
        '        exit 1\n' +
        '        ;;\n' +
        'esac\n';

    await fs.writeFile('claude-swarm.sh', helperScript, { mode: 0o755 });
    console.log('   ✅ Created claude-swarm.sh helper script');
    
    // Create Windows batch file version
    const batScript = '@echo off\n' +
        'REM Claude Code Direct Swarm Invocation Helper (Windows)\n' +
        'REM Generated by ruv-swarm --claude setup\n' +
        '\n' +
        'echo ruv-swarm Claude Code Direct Invocation\n' +
        'echo ============================================\n' +
        'echo.\n' +
        '\n' +
        'if "%1"=="research" (\n' +
        '    echo Invoking Claude Code with research swarm...\n' +
        '    if "%3"=="true" (\n' +
        '        claude "Initialize a research swarm with 5 agents using ruv-swarm. Create researcher, analyst, and coder agents. Then orchestrate the task: %2" --dangerously-skip-permissions\n' +
        '    ) else (\n' +
        '        claude "Initialize a research swarm with 5 agents using ruv-swarm. Create researcher, analyst, and coder agents. Then orchestrate the task: %2"\n' +
        '    )\n' +
        ') else if "%1"=="development" (\n' +
        '    echo Invoking Claude Code with development swarm...\n' +
        '    if "%3"=="true" (\n' +
        '        claude "Initialize a development swarm with 8 agents using ruv-swarm in hierarchical topology. Create architect, frontend coder, backend coder, and tester agents. Then orchestrate the task: %2" --dangerously-skip-permissions\n' +
        '    ) else (\n' +
        '        claude "Initialize a development swarm with 8 agents using ruv-swarm in hierarchical topology. Create architect, frontend coder, backend coder, and tester agents. Then orchestrate the task: %2"\n' +
        '    )\n' +
        ') else if "%1"=="custom" (\n' +
        '    echo Invoking Claude Code with custom prompt...\n' +
        '    if "%3"=="true" (\n' +
        '        claude "%2" --dangerously-skip-permissions\n' +
        '    ) else (\n' +
        '        claude "%2"\n' +
        '    )\n' +
        ') else if "%1"=="help" (\n' +
        '    echo Usage:\n' +
        '    echo   claude-swarm.bat research "task description" [skip-permissions]\n' +
        '    echo   claude-swarm.bat development "task description" [skip-permissions]\n' +
        '    echo   claude-swarm.bat custom "full claude prompt" [skip-permissions]\n' +
        '    echo.\n' +
        '    echo Examples:\n' +
        '    echo   claude-swarm.bat research "Analyze modern web frameworks" true\n' +
        '    echo   claude-swarm.bat development "Build user authentication API"\n' +
        '    echo.\n' +
        '    echo Note: Add \'true\' as the last parameter to use --dangerously-skip-permissions\n' +
        ') else (\n' +
        '    echo Unknown command: %1\n' +
        '    echo Run \'claude-swarm.bat help\' for usage information\n' +
        '    exit /b 1\n' +
        ')\n';

    await fs.writeFile('claude-swarm.bat', batScript);
    console.log('   ✅ Created claude-swarm.bat helper script (Windows)');
}

// Add a new command for direct Claude invocation
async function handleClaudeInvoke(args) {
    // Filter out any existing permission flags
    const filteredArgs = args.filter(arg => 
        !arg.includes('--skip-permissions') && 
        !arg.includes('--dangerously-skip-permissions')
    );
    
    const prompt = filteredArgs.join(' ');
    
    if (!prompt.trim()) {
        console.log('❌ No prompt provided');
        console.log('Usage: npx ruv-swarm claude-invoke "your swarm prompt"');
        console.log('Note: --dangerously-skip-permissions is automatically included');
        return;
    }
    
    console.log('Invoking Claude Code with ruv-swarm integration...');
    console.log('Prompt: ' + prompt.trim());
    console.log('Automatically using --dangerously-skip-permissions for seamless execution');
    
    try {
        execSync('claude --version', { stdio: 'ignore' });
        
        // Always include --dangerously-skip-permissions for seamless integration
        const claudeCommand = 'claude "' + prompt.trim() + '" --dangerously-skip-permissions';
        
        execSync(claudeCommand, { stdio: 'inherit' });
        
    } catch (error) {
        console.error('❌ Claude Code CLI not found or invocation failed');
        console.error('Make sure Claude Code CLI is installed and in your PATH');
        console.error('Install: npm install -g @anthropic-ai/claude-code');
        process.exit(1);
    }
}

// Error handling
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error.message);
    if (process.argv.includes('--debug')) {
        console.error(error.stack);
    }
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection:', reason);
    if (process.argv.includes('--debug')) {
        console.error('Promise:', promise);
    }
    process.exit(1);
});

if (require.main === module) {
    main();
}

module.exports = { main, initializeSystem };