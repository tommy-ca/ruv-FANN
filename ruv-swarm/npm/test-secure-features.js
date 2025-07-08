#!/usr/bin/env node

/**
 * Test suite for ruv-swarm v1.0.16 secure features
 */

import { execSync } from 'child_process';
import { spawn } from 'child_process';

console.log('🧪 Testing ruv-swarm v1.0.16 secure features...\n');

const tests = [];
let passed = 0;
let failed = 0;

async function test(name, fn) {
    process.stdout.write(`Testing ${name}... `);
    try {
        await fn();
        console.log('✅');
        passed++;
    } catch (error) {
        console.log(`❌ ${error.message}`);
        failed++;
    }
}

// Test 1: Version command
await test('Version shows 1.0.16-secure', async () => {
    const output = execSync('node bin/ruv-swarm-secure.js version', { encoding: 'utf8' });
    if (!output.includes('1.0.16-secure')) {
        throw new Error('Wrong version');
    }
});

// Test 2: Secure npx command
await test('Secure npx command validation', async () => {
    const output = execSync('node bin/ruv-swarm-secure.js secure-npx ruv-swarm 1.0.16 mcp start', { encoding: 'utf8' });
    if (!output.includes('Using pinned version: ruv-swarm@1.0.16')) {
        throw new Error('Version pinning not working');
    }
});

// Test 3: Legacy binary still works
await test('Legacy binary functionality', async () => {
    const output = execSync('node bin/ruv-swarm-clean.js version', { encoding: 'utf8' });
    if (!output.includes('1.0.16')) {
        throw new Error('Legacy binary broken');
    }
});

// Test 4: Hook security notice
await test('Hook shows security notice', async () => {
    const output = execSync('node bin/ruv-swarm-secure.js hook pre-task --description "test" 2>&1', { encoding: 'utf8' });
    if (!output.includes('Security Notice: Hooks execute locally only')) {
        throw new Error('Security notice missing');
    }
});

// Test 5: MCP server works
await test('MCP server functionality', async () => {
    const child = spawn('node', ['bin/ruv-swarm-secure.js', 'mcp', 'start'], {
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    child.stdout.on('data', (data) => {
        output += data.toString();
    });
    
    // Wait for server to start
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Check if server responded with correct version
    if (!output.includes('1.0.16')) {
        child.kill();
        throw new Error('MCP server version incorrect');
    }
    
    child.kill();
});

// Test 6: Invalid package name rejection
await test('Invalid package name rejection', async () => {
    try {
        execSync('node bin/ruv-swarm-secure.js secure-npx "rm -rf /" 1.0.0 test', { encoding: 'utf8' });
        throw new Error('Should have rejected invalid package name');
    } catch (error) {
        if (!error.message.includes('Invalid package name')) {
            throw error;
        }
    }
});

// Test 7: All core commands work
await test('Core commands functional', async () => {
    const commands = [
        'node bin/ruv-swarm-secure.js init mesh 3',
        'node bin/ruv-swarm-secure.js status',
        'node bin/ruv-swarm-secure.js hook notification --message "test"'
    ];
    
    for (const cmd of commands) {
        const output = execSync(cmd, { encoding: 'utf8' });
        if (!output.includes('success') && !output.includes('initialized')) {
            throw new Error(`Command failed: ${cmd}`);
        }
    }
});

// Summary
console.log('\n📊 Security Test Summary:');
console.log(`   ✅ Passed: ${passed}`);
console.log(`   ❌ Failed: ${failed}`);
console.log(`   📈 Total: ${passed + failed}`);
console.log(`   🎯 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

if (failed === 0) {
    console.log('\n🎉 All security features tested successfully!');
    console.log('✅ ruv-swarm v1.0.16 is ready for production.');
    process.exit(0);
} else {
    console.log('\n⚠️  Some tests failed. Please review before publishing.');
    process.exit(1);
}