#!/usr/bin/env node

/**
 * Integration tests for MCP detached server functionality
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import path from 'path';
import axios from 'axios';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CLI_PATH = path.join(__dirname, '..', 'bin', 'ruv-swarm-clean.js');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function execCommand(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [CLI_PATH, ...args], {
      encoding: 'utf8'
    });
    
    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });
    
    child.on('error', reject);
  });
}

async function runTests() {
  console.log('🧪 Running MCP Detached Server Integration Tests\n');
  
  let allTestsPassed = true;
  
  // Clean up any existing server
  console.log('Cleanup: Stopping any existing server...');
  await execCommand(['mcp', 'stop']);
  await sleep(1000);
  
  // Test 1: Initial status should show not running
  console.log('Test 1: Initial status check...');
  let result = await execCommand(['mcp', 'status']);
  if (result.stdout.includes('Not running')) {
    console.log('✅ Initial status correct\n');
  } else {
    console.log('❌ Expected "Not running" status\n');
    allTestsPassed = false;
  }
  
  // Test 2: Start server in detached mode
  console.log('Test 2: Starting server in detached mode...');
  result = await execCommand(['mcp', 'start', '--detached']);
  if (result.code === 0 && result.stdout.includes('started successfully')) {
    console.log('✅ Server started successfully');
    // Extract PID
    const pidMatch = result.stdout.match(/PID: (\d+)/);
    const pid = pidMatch ? pidMatch[1] : null;
    console.log(`   PID: ${pid}\n`);
  } else {
    console.log('❌ Failed to start server\n');
    console.log(result.stdout);
    allTestsPassed = false;
  }
  
  await sleep(2000); // Give server time to fully start
  
  // Test 3: Status should show running
  console.log('Test 3: Status check after start...');
  result = await execCommand(['mcp', 'status']);
  if (result.stdout.includes('Running')) {
    console.log('✅ Server is running\n');
  } else {
    console.log('❌ Expected "Running" status\n');
    allTestsPassed = false;
  }
  
  // Test 4: Health check endpoint
  console.log('Test 4: Health check endpoint...');
  try {
    const response = await axios.get('http://localhost:9898/health');
    if (response.data.status === 'healthy') {
      console.log('✅ Health check passed');
      console.log(`   Uptime: ${Math.floor(response.data.uptime)}s\n`);
    } else {
      console.log('❌ Health check failed\n');
      allTestsPassed = false;
    }
  } catch (error) {
    console.log('❌ Could not reach health endpoint\n');
    allTestsPassed = false;
  }
  
  // Test 5: Duplicate prevention
  console.log('Test 5: Duplicate prevention...');
  result = await execCommand(['mcp', 'start', '--detached']);
  if (result.code !== 0 && result.stderr.includes('already running')) {
    console.log('✅ Duplicate prevention working\n');
  } else {
    console.log('❌ Expected duplicate prevention error\n');
    allTestsPassed = false;
  }
  
  // Test 6: Restart command
  console.log('Test 6: Restart command...');
  result = await execCommand(['mcp', 'restart']);
  if (result.code === 0 && result.stdout.includes('restarted successfully')) {
    console.log('✅ Server restarted successfully\n');
  } else {
    console.log('❌ Failed to restart server\n');
    allTestsPassed = false;
  }
  
  await sleep(2000); // Give server time to restart
  
  // Test 7: Status after restart
  console.log('Test 7: Status check after restart...');
  result = await execCommand(['mcp', 'status']);
  if (result.stdout.includes('Running')) {
    console.log('✅ Server is running after restart\n');
  } else {
    console.log('❌ Server not running after restart\n');
    allTestsPassed = false;
  }
  
  // Test 8: Stop command
  console.log('Test 8: Stop command...');
  result = await execCommand(['mcp', 'stop']);
  if (result.code === 0 && result.stdout.includes('stopped successfully')) {
    console.log('✅ Server stopped successfully\n');
  } else {
    console.log('❌ Failed to stop server\n');
    allTestsPassed = false;
  }
  
  await sleep(1000);
  
  // Test 9: Status after stop
  console.log('Test 9: Status check after stop...');
  result = await execCommand(['mcp', 'status']);
  if (result.stdout.includes('Not running')) {
    console.log('✅ Server is stopped\n');
  } else {
    console.log('❌ Expected "Not running" status\n');
    allTestsPassed = false;
  }
  
  // Test 10: Health check should fail
  console.log('Test 10: Health check after stop...');
  try {
    await axios.get('http://localhost:9898/health', { timeout: 2000 });
    console.log('❌ Health check should have failed\n');
    allTestsPassed = false;
  } catch (error) {
    console.log('✅ Health check correctly fails when stopped\n');
  }
  
  // Summary
  console.log('─'.repeat(50));
  if (allTestsPassed) {
    console.log('✅ All tests passed!');
    process.exit(0);
  } else {
    console.log('❌ Some tests failed');
    process.exit(1);
  }
}

// Run tests
runTests().catch(error => {
  console.error('Test suite error:', error);
  process.exit(1);
});