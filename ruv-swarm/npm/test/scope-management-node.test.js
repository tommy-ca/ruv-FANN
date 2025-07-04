#!/usr/bin/env node

/**
 * Epic #66: Scope Management Validation Tests (Node.js Version)
 * Tests all acceptance criteria for Global and Local Scopes
 */

import assert from 'assert';
import { ScopeManager } from '../src/scope-manager.js';
import { MCPScopeTools } from '../src/mcp-scope-tools.js';

// Test runner for Node.js
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
runner.test('Session Authority System - Unique Identifiers', async() => {
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
runner.test('Memory Namespace Management - Scoped Keys', async() => {
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

// Test 3: Communication Boundary Management
runner.test('Communication Boundary Management - Local Scope Isolation', async() => {
  const manager1 = new ScopeManager();
  const manager2 = new ScopeManager();
  await manager1.initialize();
  await manager2.initialize();

  // Create local scopes for different sessions
  await manager1.createScope('local', manager1.sessionAuthority.sessionId);
  await manager2.createScope('local', manager2.sessionAuthority.sessionId);

  // Store data in manager1's local scope
  await manager1.setMemory(`local:${manager1.sessionAuthority.sessionId}:secret`, 'confidential');

  // Try to access from manager2 (should fail)
  try {
    await manager2.getMemory(`local:${manager1.sessionAuthority.sessionId}:secret`);
    assert.fail('Should not access other session local data');
  } catch (error) {
    assert.match(error.message, /Unauthorized access/, 'Should prevent cross-session access');
  }
});

// Test 4: Global Scope Communication
runner.test('Global Scope Communication - Cross-Session Access', async() => {
  const manager1 = new ScopeManager();
  const manager2 = new ScopeManager();
  await manager1.initialize();
  await manager2.initialize();

  // Create global scope and store data
  await manager1.createScope('global', 'shared');
  await manager1.setMemory('global:shared:public-data', 'accessible');

  // Access from different session (should work)
  const value = await manager2.getMemory('global:shared:public-data');
  assert.strictEqual(value, 'accessible', 'Global scope should allow cross-session access');
});

// Test 5: Scope Creation and Management
runner.test('Scope Creation and Management', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  // Test all scope types
  const scopes = [
    { type: 'global', id: 'public' },
    { type: 'local', id: manager.sessionAuthority.sessionId },
    { type: 'project', id: 'test-project' },
    { type: 'team', id: 'test-team' },
  ];

  for (const scope of scopes) {
    await manager.createScope(scope.type, scope.id);
    const created = await manager.getScope(scope.type, scope.id);
    assert(created, `Should create ${scope.type} scope`);
    assert.strictEqual(created.type, scope.type, 'Scope type should match');
    assert.strictEqual(created.id, scope.id, 'Scope ID should match');
  }
});

// Test 6: MCP Tools Integration
runner.test('MCP Tools Integration', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  const tools = new MCPScopeTools(manager);
  await tools.initialize();

  const toolMethods = tools.getTools();
  const expectedTools = [
    'scope_create', 'scope_list', 'scope_delete',
    'scope_switch', 'scope_memory_set', 'scope_memory_get',
  ];

  for (const tool of expectedTools) {
    assert(typeof toolMethods[tool] === 'function', `${tool} should be available`);
  }
});

// Test 7: Encryption and Security
runner.test('Encryption and Security', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  // Create sensitive scope
  await manager.createScope('local', manager.sessionAuthority.sessionId, { encrypted: true });

  // Store encrypted data
  const sensitiveData = { secret: 'top-secret', userId: 12345 };
  await manager.setMemory(`local:${manager.sessionAuthority.sessionId}:encrypted-data`, sensitiveData);

  // Retrieve and verify
  const retrieved = await manager.getMemory(`local:${manager.sessionAuthority.sessionId}:encrypted-data`);
  assert.deepStrictEqual(retrieved, sensitiveData, 'Encrypted data should be correctly stored and retrieved');
});

// Test 8: Session Fingerprint Validation
runner.test('Session Fingerprint Validation', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  const authority = manager.sessionAuthority;

  // Verify fingerprint format and content
  assert(authority.fingerprint.length >= 32, 'Fingerprint should be substantial');
  assert.match(authority.fingerprint, /^[a-f0-9]+$/, 'Fingerprint should be hex');

  // Test validation
  const isValid = manager.validateSessionAuthority(authority);
  assert(isValid, 'Session authority should validate correctly');
});

// Test 9: Scope Listing and Discovery
runner.test('Scope Listing and Discovery', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  // Create multiple scopes
  await manager.createScope('global', 'scope1');
  await manager.createScope('local', 'scope2');
  await manager.createScope('project', 'scope3');

  const scopes = await manager.listScopes();
  assert(scopes.length >= 3, 'Should list created scopes');

  const globalScopes = scopes.filter(s => s.type === 'global');
  assert(globalScopes.length >= 1, 'Should find global scopes');
});

// Test 10: Memory TTL and Cleanup
runner.test('Memory TTL and Cleanup', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  await manager.createScope('local', manager.sessionAuthority.sessionId);

  // Store with TTL
  await manager.setMemory(
    `local:${manager.sessionAuthority.sessionId}:temp-data`,
    'temporary',
    1, // 1 second TTL
  );

  // Should exist immediately
  let value = await manager.getMemory(`local:${manager.sessionAuthority.sessionId}:temp-data`);
  assert.strictEqual(value, 'temporary', 'Data should exist initially');

  // Wait for expiration
  await new Promise(resolve => setTimeout(resolve, 1100));

  // Should be expired
  value = await manager.getMemory(`local:${manager.sessionAuthority.sessionId}:temp-data`);
  assert.strictEqual(value, null, 'Data should expire after TTL');
});

// Test 11: Error Handling and Edge Cases
runner.test('Error Handling and Edge Cases', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  // Test invalid scope type
  try {
    await manager.createScope('invalid-type', 'test');
    assert.fail('Should reject invalid scope type');
  } catch (error) {
    assert.match(error.message, /Invalid scope type/, 'Should validate scope type');
  }

  // Test non-existent scope access
  const value = await manager.getMemory('nonexistent:scope:key');
  assert.strictEqual(value, null, 'Should return null for non-existent data');

  // Test empty key
  try {
    await manager.setMemory('', 'value');
    assert.fail('Should reject empty key');
  } catch (error) {
    assert.match(error.message, /Invalid.*key/, 'Should validate key format');
  }
});

// Test 12: Concurrent Operations
runner.test('Concurrent Operations', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  await manager.createScope('global', 'concurrent-test');

  // Perform concurrent writes
  const promises = [];
  for (let i = 0; i < 10; i++) {
    promises.push(
      manager.setMemory(`global:concurrent-test:key${i}`, `value${i}`),
    );
  }

  await Promise.all(promises);

  // Verify all writes succeeded
  for (let i = 0; i < 10; i++) {
    const value = await manager.getMemory(`global:concurrent-test:key${i}`);
    assert.strictEqual(value, `value${i}`, `Concurrent write ${i} should succeed`);
  }
});

// Test 13: Neural Network Integration
runner.test('Neural Network Integration', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  // Create neural-specific scope
  await manager.createScope('local', manager.sessionAuthority.sessionId, {
    neuralEnabled: true,
    isolationLevel: 'strict',
  });

  // Store neural patterns
  const neuralData = {
    weights: new Array(100).fill(0).map(() => Math.random()),
    biases: new Array(10).fill(0).map(() => Math.random()),
    learningRate: 0.001,
  };

  await manager.setMemory(
    `local:${manager.sessionAuthority.sessionId}:neural-patterns`,
    neuralData,
  );

  const retrieved = await manager.getMemory(
    `local:${manager.sessionAuthority.sessionId}:neural-patterns`,
  );

  assert.deepStrictEqual(retrieved, neuralData, 'Neural data should be preserved');
});

// Test 14: Performance and Scalability
runner.test('Performance and Scalability', async() => {
  const manager = new ScopeManager();
  await manager.initialize();

  await manager.createScope('global', 'performance-test');

  const startTime = Date.now();
  const operations = 100;

  // Perform bulk operations
  for (let i = 0; i < operations; i++) {
    await manager.setMemory(`global:performance-test:key${i}`, `value${i}`);
  }

  const writeTime = Date.now() - startTime;

  // Read performance
  const readStart = Date.now();
  for (let i = 0; i < operations; i++) {
    await manager.getMemory(`global:performance-test:key${i}`);
  }
  const readTime = Date.now() - readStart;

  console.log(`   Write performance: ${operations} operations in ${writeTime}ms`);
  console.log(`   Read performance: ${operations} operations in ${readTime}ms`);

  // Performance assertions
  assert(writeTime < 5000, 'Write operations should complete within 5 seconds');
  assert(readTime < 2000, 'Read operations should complete within 2 seconds');
});

// Run all tests
async function runTests() {
  console.log('🚀 Starting Epic #66 Scope Management Tests\n');

  const success = await runner.run();

  console.log(`\n${ '='.repeat(60)}`);
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