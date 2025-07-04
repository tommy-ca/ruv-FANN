#!/usr/bin/env node

/**
 * Epic #66: Scope Management Validation Tests (Fixed API Version)
 * Tests all acceptance criteria for Global and Local Scopes
 */

import assert from 'assert';
import { ScopeManager } from '../src/scope-manager.js';

// Simple test runner for Node.js
class TestRunner {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
        this.errors = [];
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log('🧪 Epic #66: Scope Management Validation Tests');
        console.log('=' .repeat(60));
        
        for (const test of this.tests) {
            try {
                await test.fn();
                console.log(`✅ ${test.name}`);
                this.passed++;
            } catch (error) {
                console.error(`❌ ${test.name}`);
                console.error(`   ${error.message}`);
                this.failed++;
                this.errors.push({ test: test.name, error: error.message });
            }
        }

        this.printSummary();
        return this.failed === 0;
    }

    printSummary() {
        console.log('\n📊 Scope Management Test Results');
        console.log('─'.repeat(60));
        console.log(`Total Tests: ${this.passed + this.failed}`);
        console.log(`✅ Passed: ${this.passed}`);
        console.log(`❌ Failed: ${this.failed}`);
        
        if (this.errors.length > 0) {
            console.log('\n❌ Failed Tests:');
            this.errors.forEach(e => {
                console.log(`  - ${e.test}: ${e.error}`);
            });
        }
    }
}

const runner = new TestRunner();

// Test 1: Session Authority System
runner.test('Session Authority System - Unique Identifiers', async () => {
    const manager1 = new ScopeManager();
    const manager2 = new ScopeManager();
    await manager1.initialize();
    await manager2.initialize();
    
    const authority1 = manager1.sessionAuthority.authority;
    const authority2 = manager2.sessionAuthority.authority;
    
    assert(authority1.sessionId !== authority2.sessionId, 'Session IDs should be unique');
    assert(authority1.fingerprint !== authority2.fingerprint, 'Fingerprints should be unique');
    assert.match(authority1.sessionId, /^session-.+-\d+-\d+-[a-f0-9]+$/, 'Session ID format validation');
});

// Test 2: Memory Namespace Management
runner.test('Memory Namespace Management - Scoped Keys', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    const localScope = manager.createScope({ type: 'local', id: 'test-session' });
    await manager.memoryManager.store(localScope, 'key1', 'value1');
    
    const value = await manager.memoryManager.retrieve(localScope, 'key1');
    assert.strictEqual(value, 'value1', 'Memory should be stored with scoped key');
    
    // Test namespace isolation
    const globalScope = manager.createScope({ type: 'global', id: 'test-session' });
    const globalValue = await manager.memoryManager.retrieve(globalScope, 'key1');
    assert.strictEqual(globalValue, null, 'Different scopes should be isolated');
});

// Test 3: Scope Creation and Management
runner.test('Scope Creation and Management', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    // Test scope creation
    const localScope = manager.createScope({ type: 'local', id: 'test-local' });
    assert.strictEqual(localScope.type, 'local', 'Should create local scope');
    
    const globalScope = manager.createScope({ type: 'global', id: 'test-global' });
    assert.strictEqual(globalScope.type, 'global', 'Should create global scope');
    
    // Test scope retrieval
    const retrieved = manager.getScope(localScope.id);
    assert(retrieved, 'Should retrieve created scope');
    assert.strictEqual(retrieved.type, localScope.type, 'Retrieved scope should match');
});

// Test 4: Session Fingerprint Validation
runner.test('Session Fingerprint Validation', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    const authority = manager.sessionAuthority.authority;
    
    // Verify fingerprint format and content
    assert(authority.fingerprint.length >= 32, 'Fingerprint should be substantial');
    assert.match(authority.fingerprint, /^[a-f0-9]+$/, 'Fingerprint should be hex');
    
    // Test validation - same session should validate correctly
    const isValid = manager.sessionAuthority.validateSession(authority.sessionId, authority);
    assert(isValid, 'Session authority should validate correctly');
});

// Test 5: Error Handling and Edge Cases  
runner.test('Error Handling and Edge Cases', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    // Test non-existent scope access
    const value = manager.getScope('nonexistent-scope');
    assert.strictEqual(value, undefined, 'Should return undefined for non-existent scope');
    
    // Test valid scope creation (no validation error expected)
    const scope = manager.createScope({ type: 'local', id: 'test' });
    assert(scope, 'Should create valid scope');
    assert.strictEqual(scope.type, 'local', 'Should have correct type');
});

// Test 6: System Status and Statistics
runner.test('System Status and Statistics', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    // Create some scopes
    manager.createScope({ type: 'local', id: 'test-1' });
    manager.createScope({ type: 'global', id: 'test-2' });
    
    const status = manager.getStatus();
    assert(status, 'Should return status object');
    assert(typeof status.memory === 'object', 'Should include memory statistics');
    assert(typeof status.neural === 'object', 'Should include neural statistics');
    assert(typeof status.communication === 'object', 'Should include communication statistics');
});

// Test 7: Scope Configuration Export/Import
runner.test('Scope Configuration Export/Import', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    // Create some scopes
    manager.createScope({ type: 'local', id: 'export-test-1' });
    manager.createScope({ type: 'global', id: 'export-test-2' });
    
    const config = manager.exportConfig();
    assert(config, 'Should export configuration');
    assert(config.sessionId, 'Should include session ID');
    assert(Array.isArray(config.scopes), 'Should include scopes array');
    
    // Test import (same session)
    const imported = await manager.importConfig(config);
    assert(imported, 'Should import configuration successfully');
});

// Test 8: Memory Storage and Retrieval
runner.test('Memory Storage and Retrieval', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    const scope = manager.createScope({ type: 'global', id: 'memory-test' });
    
    // Store different data types
    const testData = {
        string: 'test-string',
        number: 42,
        object: { nested: true, array: [1, 2, 3] },
        boolean: true
    };
    
    for (const [key, value] of Object.entries(testData)) {
        await manager.memoryManager.store(scope, key, value);
        const retrieved = await manager.memoryManager.retrieve(scope, key);
        assert.deepStrictEqual(retrieved, value, `Should store and retrieve ${key} correctly`);
    }
});

// Test 9: Cross-Session Isolation (Local Scopes)
runner.test('Cross-Session Isolation - Local Scopes', async () => {
    const manager1 = new ScopeManager();
    const manager2 = new ScopeManager();
    await manager1.initialize();
    await manager2.initialize();
    
    // Create local scopes in different sessions
    const scope1 = manager1.createScope({ type: 'local', id: 'isolation-test' });
    const scope2 = manager2.createScope({ type: 'local', id: 'isolation-test' });
    
    // Store data in first session
    await manager1.memoryManager.store(scope1, 'secret', 'session1-data');
    
    // Try to access from second session (should be isolated)
    try {
        await manager2.memoryManager.retrieve(scope2, 'secret');
        // If we get here without error, the data should be null (isolated)
        const retrieved = await manager2.memoryManager.retrieve(scope2, 'secret');
        assert.strictEqual(retrieved, null, 'Cross-session local access should be isolated');
    } catch (error) {
        // This is also acceptable - explicit denial of access
        assert.match(error.message, /Unauthorized|access/i, 'Should prevent unauthorized access');
    }
});

// Test 10: Performance and Scalability
runner.test('Performance and Scalability', async () => {
    const manager = new ScopeManager();
    await manager.initialize();
    
    const scope = manager.createScope({ type: 'global', id: 'performance-test' });
    
    const startTime = Date.now();
    const operations = 50; // Reduced for faster testing
    
    // Perform bulk operations
    for (let i = 0; i < operations; i++) {
        await manager.memoryManager.store(scope, `key${i}`, `value${i}`);
    }
    
    const writeTime = Date.now() - startTime;
    
    // Read performance
    const readStart = Date.now();
    for (let i = 0; i < operations; i++) {
        await manager.memoryManager.retrieve(scope, `key${i}`);
    }
    const readTime = Date.now() - readStart;
    
    console.log(`   Write performance: ${operations} operations in ${writeTime}ms`);
    console.log(`   Read performance: ${operations} operations in ${readTime}ms`);
    
    // Performance assertions (generous for different environments)
    assert(writeTime < 10000, 'Write operations should complete within 10 seconds');
    assert(readTime < 5000, 'Read operations should complete within 5 seconds');
});

// Run all tests
async function runTests() {
    console.log('🚀 Starting Epic #66 Scope Management Tests\n');
    
    const success = await runner.run();
    
    console.log('\n' + '='.repeat(60));
    if (success) {
        console.log('🎉 All scope management tests passed! Epic #66 validated.');
        process.exit(0);
    } else {
        console.log('❌ Some scope management tests failed.');
        process.exit(1);
    }
}

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runTests().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

export { runTests };