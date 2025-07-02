#!/usr/bin/env node

/**
 * MCP Server Wrapper for Detached Mode
 * This wrapper keeps the MCP server running in the background
 */

import http from 'http';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Get health check port from environment
const healthPort = parseInt(process.env.MCP_HEALTH_PORT || '9898');

// Start health check server
const healthServer = http.createServer((req, res) => {
  if (req.url === '/health' && req.method === 'GET') {
    const status = {
      status: 'healthy',
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
      mode: 'detached',
      pid: process.pid
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(status));
  } else {
    res.writeHead(404);
    res.end('Not Found');
  }
});

healthServer.listen(healthPort, 'localhost', () => {
  console.error(`Health check server listening on port ${healthPort}`);
});

// Start the actual MCP server in stdio mode
const mcpPath = path.join(__dirname, '..', 'bin', 'ruv-swarm-clean.js');
const mcpProcess = spawn(process.execPath, [mcpPath, 'mcp', 'start', '--protocol=stdio'], {
  stdio: ['pipe', 'pipe', 'inherit'],
  env: {
    ...process.env,
    MCP_WRAPPER_MODE: 'true'
  }
});

// Forward stdin/stdout for MCP protocol
process.stdin.pipe(mcpProcess.stdin);
mcpProcess.stdout.pipe(process.stdout);

// Handle process termination
function gracefulShutdown(signal) {
  console.error(`Received ${signal}, shutting down gracefully...`);
  
  // Stop health server
  healthServer.close(() => {
    console.error('Health server closed');
  });
  
  // Stop MCP process
  mcpProcess.kill('SIGTERM');
  
  // Give it time to shutdown gracefully
  setTimeout(() => {
    process.exit(0);
  }, 5000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Keep the process alive
setInterval(() => {
  // This keeps the process running
}, 1000);

// Log startup
console.error('MCP server wrapper started in detached mode');