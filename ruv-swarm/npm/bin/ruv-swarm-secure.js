#!/usr/bin/env node

/**
 * ruv-swarm CLI with secure Claude integration
 * Version 1.0.16 - Security Enhanced
 */

import { Command } from 'commander';
import { RuvSwarm, VERSION } from '../src/index.js';
import { MCPServer } from '../src/mcp-server.js';
import { secureClaudeInvoke, secureNpxCommand } from '../src/secure-invoke.js';
import { execSync } from 'child_process';
import readline from 'readline';

const program = new Command();

program
    .name('ruv-swarm')
    .description('High-performance neural network swarm orchestration (Secure Edition)')
    .version('1.0.16-secure');

// All existing commands remain the same
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

// Secure Claude integration
program
    .command('claude')
    .description('Secure Claude Code integration')
    .command('invoke')
    .description('Invoke Claude Code with explicit permission control')
    .argument('<prompt>', 'The prompt for Claude')
    .option('--allow-permissions', 'Allow permission flags (requires confirmation)')
    .option('--permissions <perms...>', 'Specific permissions to grant')
    .action(async (prompt, options) => {
        try {
            console.log('🔒 Secure Claude Invocation');
            console.log('━'.repeat(50));
            
            // Check if permissions are requested
            if (options.allowPermissions && options.permissions) {
                console.log('\n⚠️  PERMISSION REQUEST:');
                console.log('The following permissions are being requested:');
                options.permissions.forEach(perm => {
                    console.log(`   • ${perm}`);
                });
                
                // Get user confirmation
                const rl = readline.createInterface({
                    input: process.stdin,
                    output: process.stdout
                });
                
                const answer = await new Promise((resolve) => {
                    rl.question('\nDo you want to grant these permissions? (yes/no): ', resolve);
                });
                rl.close();
                
                if (answer.toLowerCase() !== 'yes') {
                    console.log('❌ Permission denied by user');
                    process.exit(1);
                }
            }
            
            const command = await secureClaudeInvoke(prompt, {
                allowPermissions: options.allowPermissions,
                permissions: options.permissions || []
            });
            
            console.log('\n🚀 Executing:', command);
            execSync(command, { stdio: 'inherit' });
            
        } catch (error) {
            console.error('❌ Error:', error.message);
            process.exit(1);
        }
    });

// Secure npx command
program
    .command('secure-npx')
    .description('Execute npm packages with version pinning')
    .argument('<package>', 'Package name')
    .argument('<version>', 'Package version (required for security)')
    .argument('<command>', 'Command to run')
    .argument('[args...]', 'Additional arguments')
    .action(async (packageName, version, command, args) => {
        try {
            const npxCommand = await secureNpxCommand(packageName, version, command, args);
            
            console.log('\n🔒 Security Notice:');
            console.log(`   • Using pinned version: ${packageName}@${version}`);
            console.log(`   • This prevents supply chain attacks`);
            console.log(`   • Always verify package integrity before execution\n`);
            
            console.log('Suggested command:');
            console.log(`   ${npxCommand}`);
            console.log('\nTo execute, run the command above manually after verification.');
            
        } catch (error) {
            console.error('❌ Error:', error.message);
            process.exit(1);
        }
    });

// MCP server and other commands remain unchanged
program
    .command('mcp')
    .description('MCP server operations')
    .command('start')
    .description('Start MCP server')
    .option('-m, --mode <type>', 'Server mode (stdio|websocket)', 'stdio')
    .option('-p, --port <number>', 'WebSocket port (if websocket mode)', '3001')
    .action(async (options) => {
        try {
            if (process.env.MCP_TEST_MODE === 'true') {
                console.error('MCP server ready');
            }

            const server = new MCPServer({
                mode: options.mode,
                port: options.port ? parseInt(options.port) : 3001
            });
            
            await server.start();
            
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

// Hook commands remain the same but with security notice
program
    .command('hook')
    .description('Execute integration hooks (local execution only)')
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
        console.log('🔒 Security Notice: Hooks execute locally only');
        console.log('   Use with local ruv-swarm installation\n');
        
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

// Status command
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

// Version command
program
    .command('version')
    .description('Show version information')
    .action(() => {
        console.log(`ruv-swarm v1.0.16-secure`);
        console.log('High-performance neural network swarm orchestration');
        console.log('Security Enhanced Edition');
        console.log('\n🔒 Security Features:');
        console.log('   • Explicit permission control for Claude invocation');
        console.log('   • Required version pinning for npx commands');
        console.log('   • User confirmation for elevated permissions');
        console.log('   • Local-only hook execution');
    });

// Parse command line arguments
program.parse();