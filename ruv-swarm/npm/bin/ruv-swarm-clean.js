#!/usr/bin/env node

/**
 * ruv-swarm CLI Entry Point
 * Provides command-line interface for swarm coordination
 */

import { Command } from 'commander';
import { RuvSwarm, VERSION } from '../src/index.js';
import { MCPServer } from '../src/mcp-server.js';

const program = new Command();

program
    .name('ruv-swarm')
    .description('High-performance neural network swarm orchestration')
    .version(VERSION);

program
    .command('init')
    .description('Initialize a new swarm')
    .option('-t, --topology <type>', 'Swarm topology', 'hierarchical')
    .option('-a, --agents <number>', 'Maximum agents', '8')
    .option('-s, --strategy <type>', 'Distribution strategy', 'adaptive')
    .action(async (options) => {
        try {
            const swarm = new RuvSwarm({
                topology: options.topology,
                maxAgents: parseInt(options.agents),
                strategy: options.strategy
            });
            
            const result = await swarm.init();
            console.log(JSON.stringify(result, null, 2));
            
            if (result.success) {
                process.exit(0);
            } else {
                process.exit(1);
            }
        } catch (error) {
            console.error('Error:', error.message);
            process.exit(1);
        }
    });

program
    .command('status')
    .description('Get swarm status')
    .action(async () => {
        try {
            const swarm = new RuvSwarm();
            await swarm.init();
            const status = await swarm.getStatus();
            console.log(JSON.stringify(status, null, 2));
            await swarm.shutdown();
        } catch (error) {
            console.error('Error:', error.message);
            process.exit(1);
        }
    });

program
    .command('mcp')
    .description('MCP server operations')
    .command('start')
    .description('Start MCP server')
    .option('-m, --mode <type>', 'Server mode (stdio|websocket)', 'stdio')
    .option('-p, --port <number>', 'WebSocket port (if websocket mode)', '3001')
    .action(async (options) => {
        try {
            // Signal server readiness for testing
            if (process.env.MCP_TEST_MODE === 'true') {
                console.error('MCP server ready'); // Use stderr so it doesn't interfere with JSON-RPC
            }

            const server = new MCPServer({
                mode: options.mode,
                port: options.port ? parseInt(options.port) : 3001
            });
            
            await server.start();
            
            // Keep process alive
            process.on('SIGINT', async () => {
                console.error('Shutting down MCP server...');
                await server.stop();
                process.exit(0);
            });
            
        } catch (error) {
            console.error('MCP Server Error:', error.message);
            process.exit(1);
        }
    });

// Hook commands for Claude-Flow integration
program
    .command('hook')
    .description('Execute integration hooks')
    .argument('<hook-name>', 'Hook name to execute')
    .option('--description <desc>', 'Task description')
    .option('--file <path>', 'File path for post-edit hooks')
    .option('--message <msg>', 'Notification message')
    .option('--memory-key <key>', 'Memory storage key')
    .option('--auto-spawn-agents <bool>', 'Auto-spawn agents', 'true')
    .option('--session-id <id>', 'Session ID')
    .option('--load-memory <bool>', 'Load memory', 'false')
    .option('--task-id <id>', 'Task ID')
    .option('--analyze-performance <bool>', 'Analyze performance', 'false')
    .option('--export-metrics <bool>', 'Export metrics', 'false')
    .option('--generate-summary <bool>', 'Generate summary', 'false')
    .option('--query <query>', 'Search query')
    .option('--cache-results <bool>', 'Cache results', 'false')
    .option('--telemetry <bool>', 'Enable telemetry', 'false')
    .action(async (hookName, options) => {
        try {
            const swarm = new RuvSwarm();
            await swarm.init();
            
            const result = await swarm.hooks.triggerHook(hookName, {
                ...options,
                timestamp: Date.now()
            });
            
            console.log(JSON.stringify(result, null, 2));
            await swarm.shutdown();
        } catch (error) {
            console.error('Hook Error:', error.message);
            process.exit(1);
        }
    });

// Version command
program
    .command('version')
    .description('Show version information')
    .action(() => {
        console.log(`ruv-swarm v${VERSION}`);
        console.log('High-performance neural network swarm orchestration');
        console.log('Built for Claude-Flow integration');
    });

// Parse command line arguments
program.parse();