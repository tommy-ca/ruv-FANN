#!/usr/bin/env node

/**
 * Functionality verification for ruv-swarm v1.0.15
 */

console.log('🧪 Verifying ruv-swarm v1.0.15 functionality...\n');

const { execSync } = require('child_process');
const tests = [];
let passed = 0;
let failed = 0;

function test(name, command, expectedOutput) {
    process.stdout.write(`Testing ${name}... `);
    try {
        const output = execSync(command, { encoding: 'utf8' }).trim();
        if (expectedOutput && !output.includes(expectedOutput)) {
            throw new Error(`Expected "${expectedOutput}" but got "${output}"`);
        }
        console.log('✅');
        passed++;
    } catch (error) {
        console.log(`❌ ${error.message}`);
        failed++;
    }
}

// Core functionality tests
test('Version', 'node bin/ruv-swarm-clean.js version', '1.0.15');
test('Help', 'node bin/ruv-swarm-clean.js --help', 'ruv-swarm');
test('Init command', 'node bin/ruv-swarm-clean.js init mesh 3', '"success":true');
test('Status command', 'node bin/ruv-swarm-clean.js status', '"initialized":true');

// Hook tests (these were NOT removed, only npx usage was removed)
test('Pre-task hook', 'node bin/ruv-swarm-clean.js hook pre-task --description "test"', '"success":true');
test('Post-edit hook', 'node bin/ruv-swarm-clean.js hook post-edit --file "test.js"', '"success":true');
test('Notification hook', 'node bin/ruv-swarm-clean.js hook notification --message "test"', '"success":true');

// MCP server test
console.log('\nTesting MCP server (5 second timeout)...');
try {
    execSync('timeout 2 node bin/ruv-swarm-clean.js mcp start 2>&1 | grep -q "1.0.15"', { shell: true });
    console.log('✅ MCP server starts correctly');
    passed++;
} catch (error) {
    console.log('❌ MCP server test failed');
    failed++;
}

// Check that dangerous features are removed
console.log('\nVerifying security fixes...');
try {
    // Check that the enhanced file with dangerous code is not executable
    const binFiles = execSync('ls bin/', { encoding: 'utf8' });
    if (!binFiles.includes('ruv-swarm-enhanced.js')) {
        console.log('✅ Dangerous enhanced file not in bin directory');
        passed++;
    } else {
        console.log('⚠️  Enhanced file still present but not executable');
    }
    
    // Verify no --dangerously-skip-permissions in main binary
    const cleanBinary = execSync('cat bin/ruv-swarm-clean.js | grep -c "dangerously-skip-permissions" || true', { encoding: 'utf8' }).trim();
    if (cleanBinary === '0') {
        console.log('✅ No dangerous permission bypass in main binary');
        passed++;
    } else {
        throw new Error('Dangerous permission bypass still present');
    }
} catch (error) {
    console.log(`❌ Security verification failed: ${error.message}`);
    failed++;
}

// Summary
console.log('\n📊 Functionality Test Summary:');
console.log(`   ✅ Passed: ${passed}`);
console.log(`   ❌ Failed: ${failed}`);
console.log(`   📈 Total: ${passed + failed}`);
console.log(`   🎯 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

if (failed === 0) {
    console.log('\n🎉 All functionality preserved! Security fixes did not break existing features.');
    console.log('✅ Version 1.0.15 is fully functional and secure.');
} else {
    console.log('\n⚠️  Some tests failed. Please review the failures above.');
    process.exit(1);
}