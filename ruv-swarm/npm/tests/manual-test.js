#!/usr/bin/env node

// Manual test script for MCP detached functionality
import ProcessManager from '../src/process-manager.js';
import MCPDetached from '../src/mcp-detached.js';
import path from 'path';
import fs from 'fs-extra';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function testProcessManager() {
    console.log('\n🧪 Testing ProcessManager...\n');
    
    const testDir = path.join(__dirname, '.test-pm');
    await fs.ensureDir(testDir);
    
    const pm = new ProcessManager({ pidDir: testDir });
    
    try {
        // Test 1: Start a process
        console.log('Test 1: Starting a test process...');
        const result = await pm.start('test-process', {
            command: 'node',
            args: ['-e', 'setInterval(() => console.log("alive"), 1000); setTimeout(() => {}, 30000)'],
            detached: true
        });
        console.log(`✅ Process started with PID: ${result.pid}`);
        
        // Test 2: Check status
        console.log('\nTest 2: Checking process status...');
        const status = await pm.status('test-process');
        console.log(`✅ Process is ${status.running ? 'running' : 'not running'}`);
        
        // Test 3: Try to start duplicate
        console.log('\nTest 3: Testing duplicate prevention...');
        try {
            await pm.start('test-process', {
                command: 'node',
                args: ['-e', 'console.log("should not run")'],
                detached: true
            });
            console.log('❌ Duplicate process was allowed (should not happen)');
        } catch (error) {
            console.log(`✅ Duplicate prevented: ${error.message}`);
        }
        
        // Test 4: List processes
        console.log('\nTest 4: Listing processes...');
        const processes = await pm.listProcesses();
        console.log(`✅ Found ${processes.length} process(es)`);
        
        // Test 5: Stop process
        console.log('\nTest 5: Stopping process...');
        const stopResult = await pm.stop('test-process');
        console.log(`✅ Process stopped (forced: ${stopResult.forced})`);
        
        // Test 6: Verify stopped
        console.log('\nTest 6: Verifying process is stopped...');
        const finalStatus = await pm.status('test-process');
        console.log(`✅ Process is ${finalStatus.running ? 'still running!' : 'stopped'}`);
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    } finally {
        await fs.remove(testDir);
    }
}

async function testMCPDetached() {
    console.log('\n🧪 Testing MCPDetached...\n');
    
    const testDir = path.join(__dirname, '.test-mcp');
    await fs.ensureDir(testDir);
    
    const mcp = new MCPDetached({ 
        dataDir: testDir,
        healthCheckPort: 9899
    });
    
    try {
        // Test 1: Check initial status
        console.log('Test 1: Checking initial status...');
        const initialStatus = await mcp.status();
        console.log(`✅ Initial status: ${initialStatus.running ? 'running' : 'not running'}`);
        
        // Test 2: Start in detached mode
        console.log('\nTest 2: Starting MCP server...');
        const startResult = await mcp.start({ detached: true });
        console.log(`✅ MCP started with PID: ${startResult.pid}`);
        
        await sleep(2000); // Give it time to start
        
        // Test 3: Check running status
        console.log('\nTest 3: Checking running status...');
        const runningStatus = await mcp.status();
        console.log(`✅ Status: ${runningStatus.running ? 'running' : 'not running'}`);
        console.log(`   PID: ${runningStatus.pid}`);
        console.log(`   Mode: ${runningStatus.mode}`);
        
        // Test 4: Try health check
        console.log('\nTest 4: Testing health check...');
        try {
            const health = await mcp.checkHealth();
            console.log(`✅ Health check passed: ${health.status}`);
        } catch (error) {
            console.log(`⚠️  Health check failed: ${error.message}`);
        }
        
        // Test 5: Stop server
        console.log('\nTest 5: Stopping MCP server...');
        const stopResult = await mcp.stop();
        console.log(`✅ MCP stopped: ${stopResult.stopped ? 'success' : 'failed'}`);
        
        // Test 6: Verify stopped
        console.log('\nTest 6: Verifying server is stopped...');
        const finalStatus = await mcp.status();
        console.log(`✅ Final status: ${finalStatus.running ? 'still running!' : 'stopped'}`);
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    } finally {
        await fs.remove(testDir);
    }
}

async function runTests() {
    console.log('🚀 Starting manual tests for MCP Detached functionality\n');
    
    await testProcessManager();
    await testMCPDetached();
    
    console.log('\n✅ All tests completed!');
}

// Run tests
runTests().catch(console.error);