#!/usr/bin/env node
/**
 * Ultra-stable ruv-swarm MCP wrapper - Never crashes, ever
 * Adds circuit breaker pattern and ultra-defensive error handling
 */

import { spawn } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const SECURE_SCRIPT = join(__dirname, 'ruv-swarm-secure.js');
const MAX_RESTARTS = 10;
const RESTART_DELAY = 1000; // 1 second

let restartCount = 0;
let lastRestartTime = 0;

function log(message) {
    const timestamp = new Date().toISOString();
    console.error(`[${timestamp}] [ULTRA-STABLE] ${message}`);
}

function startMcpServer() {
    const now = Date.now();
    
    // Reset restart count if it's been more than 5 minutes
    if (now - lastRestartTime > 300000) {
        restartCount = 0;
    }
    
    if (restartCount >= MAX_RESTARTS) {
        log(`Maximum restarts (${MAX_RESTARTS}) reached. Server may have persistent issues.`);
        log('Please check logs and restart manually if needed.');
        return;
    }
    
    restartCount++;
    lastRestartTime = now;
    
    log(`Starting MCP server (attempt ${restartCount}/${MAX_RESTARTS})`);
    
    const child = spawn('node', [SECURE_SCRIPT, ...process.argv.slice(2)], {
        stdio: ['inherit', 'inherit', 'inherit'],
        env: { ...process.env, MCP_ULTRA_STABLE: 'true' }
    });
    
    child.on('exit', (code, signal) => {
        if (code === 0) {
            log('MCP server exited normally');
            return;
        }
        
        log(`MCP server crashed with code ${code} and signal ${signal}`);
        log(`Restarting in ${RESTART_DELAY}ms...`);
        
        setTimeout(() => {
            startMcpServer();
        }, RESTART_DELAY);
    });
    
    child.on('error', (error) => {
        log(`Failed to start MCP server: ${error.message}`);
        log(`Restarting in ${RESTART_DELAY}ms...`);
        
        setTimeout(() => {
            startMcpServer();
        }, RESTART_DELAY);
    });
    
    // Handle process termination signals
    process.on('SIGTERM', () => {
        log('Received SIGTERM, shutting down...');
        child.kill('SIGTERM');
        process.exit(0);
    });
    
    process.on('SIGINT', () => {
        log('Received SIGINT, shutting down...');
        child.kill('SIGINT');
        process.exit(0);
    });
}

// Start the server
log('Ultra-stable MCP wrapper starting...');
startMcpServer();