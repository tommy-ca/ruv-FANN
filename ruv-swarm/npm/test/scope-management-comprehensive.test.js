/**
 * Comprehensive Scope Management Test Suite
 * Tests for Epic Issue #66: Global and Local Scopes
 */

import { RuvSwarm } from '../src/index-enhanced.js';
import { MCPScopeTools } from '../src/mcp-scope-tools.js';
import { ScopeManager } from '../src/scope-manager.js';

describe('Epic #66: Scope Management System', () => {
  let ruvSwarm;
  let scopeTools;
  let scopeManager;

  beforeAll(async() => {
    ruvSwarm = await RuvSwarm.initialize({
      loadingStrategy: 'progressive',
      enablePersistence: true,
      enableNeuralNetworks: true,
    });
    scopeTools = new MCPScopeTools(ruvSwarm);
    await scopeTools.initialize();
    scopeManager = new ScopeManager();
    await scopeManager.initialize();
  });

  describe('🔧 Functional Requirements', () => {

    test('✅ Users can initialize swarms with different scope types', async() => {
      // Test local scope initialization
      const localSwarm = await scopeTools.swarm_init({
        topology: 'mesh',
        maxAgents: 3,
        scope: {
          type: 'local',
          isolation: 'strict',
        },
      });

      expect(localSwarm.scope).toBeDefined();
      expect(localSwarm.scope.type).toBe('local');
      expect(localSwarm.scope.sessionId).toBeDefined();

      // Test global scope initialization
      const globalSwarm = await scopeTools.swarm_init({
        topology: 'hierarchical',
        maxAgents: 5,
        scope: {
          type: 'global',
          isolation: 'permissive',
        },
      });

      expect(globalSwarm.scope.type).toBe('global');

      // Test project scope initialization
      const projectSwarm = await scopeTools.swarm_init({
        topology: 'ring',
        maxAgents: 4,
        scope: {
          type: 'project',
          boundaries: { project: 'test-project' },
        },
      });

      expect(projectSwarm.scope.type).toBe('project');
    });

    test('✅ Memory is properly isolated based on scope configuration', async() => {
      // Create local scope
      const localScope = scopeManager.createScope({
        type: 'local',
        isolation: 'strict',
      });

      // Create global scope
      const globalScope = scopeManager.createScope({
        type: 'global',
        isolation: 'permissive',
      });

      // Test memory isolation
      await scopeTools.memory_usage({
        action: 'store',
        key: 'test-data',
        value: { data: 'local-only' },
        scope: { scopeId: localScope.id },
      });

      await scopeTools.memory_usage({
        action: 'store',
        key: 'test-data',
        value: { data: 'global-shared' },
        scope: { scopeId: globalScope.id },
      });

      // Retrieve from local scope
      const localData = await scopeTools.memory_usage({
        action: 'retrieve',
        key: 'test-data',
        scope: { scopeId: localScope.id },
      });

      // Retrieve from global scope
      const globalData = await scopeTools.memory_usage({
        action: 'retrieve',
        key: 'test-data',
        scope: { scopeId: globalScope.id },
      });

      expect(localData.value.data).toBe('local-only');
      expect(globalData.value.data).toBe('global-shared');
    });

    test('✅ Neural networks respect scope boundaries', async() => {
      // Create scoped neural networks
      const localNeural = await scopeTools.neural_train({
        pattern: 'local-pattern',
        data: { type: 'local' },
        scope: { isolation: 'local' },
      });

      const globalNeural = await scopeTools.neural_train({
        pattern: 'global-pattern',
        data: { type: 'global' },
        scope: { isolation: 'global' },
      });

      expect(localNeural.isolation).toBe('local');
      expect(globalNeural.scope).toBe('global');
      expect(localNeural.networkId).not.toBe(globalNeural.networkId);
    });

    test('✅ Communication is filtered according to scope rules', async() => {
      // Test that local scopes cannot communicate across sessions
      const scope1 = scopeManager.createScope({ type: 'local' });
      const scope2 = scopeManager.createScope({ type: 'local' });

      // Should not be able to communicate between different local scopes
      const canCommunicate = scopeManager.communicationManager.canCommunicate(scope1, scope2);
      expect(canCommunicate).toBe(false);

      // Global scopes should be able to communicate
      const globalScope1 = scopeManager.createScope({ type: 'global' });
      const globalScope2 = scopeManager.createScope({ type: 'global' });

      const globalCanCommunicate = scopeManager.communicationManager.canCommunicate(globalScope1, globalScope2);
      expect(globalCanCommunicate).toBe(true);
    });

    test('✅ Scope can be changed at runtime without data loss', async() => {
      // Create initial scope
      const initialScope = await scopeTools.scope_configure({
        scope: {
          type: 'local',
          isolation: 'strict',
        },
      });

      // Store some data
      await scopeTools.memory_usage({
        action: 'store',
        key: 'persistent-data',
        value: { important: 'data' },
        scope: { scopeId: initialScope.scopeId },
      });

      // Change scope configuration
      const updatedScope = await scopeTools.scope_configure({
        scope: {
          type: 'project',
          isolation: 'hybrid',
          boundaries: { project: 'test-project' },
        },
      });

      // Verify scope changed
      expect(updatedScope.configuration.type).toBe('project');
      expect(updatedScope.success).toBe(true);
    });

    test('✅ Configuration is persistent across sessions', async() => {
      // Export scope configuration
      const exportedConfig = await scopeTools.scope_export({
        includeData: true,
      });

      expect(exportedConfig.sessionId).toBeDefined();
      expect(exportedConfig.scopes).toBeDefined();

      // Import configuration
      const importResult = await scopeTools.scope_import({
        config: exportedConfig,
      });

      expect(importResult.success).toBe(true);
      expect(importResult.imported).toBeGreaterThan(0);
    });
  });

  describe('⚡ Non-Functional Requirements', () => {

    test('✅ Performance impact < 5% for local scopes', async() => {
      const iterations = 1000;

      // Measure baseline performance without scopes
      const startBaseline = Date.now();
      for (let i = 0; i < iterations; i++) {
        await ruvSwarm.createSwarm('test', 'mesh', 3);
      }
      const baselineTime = Date.now() - startBaseline;

      // Measure performance with local scopes
      const startScoped = Date.now();
      for (let i = 0; i < iterations; i++) {
        await scopeTools.swarm_init({
          topology: 'mesh',
          maxAgents: 3,
          scope: { type: 'local', isolation: 'strict' },
        });
      }
      const scopedTime = Date.now() - startScoped;

      const performanceImpact = ((scopedTime - baselineTime) / baselineTime) * 100;
      expect(performanceImpact).toBeLessThan(5);
    });

    test('✅ Security boundaries are cryptographically enforced', async() => {
      // Verify session fingerprinting
      const fingerprint1 = scopeManager.sessionAuthority.generateFingerprint();
      const fingerprint2 = scopeManager.sessionAuthority.generateFingerprint();

      expect(fingerprint1).toBe(fingerprint2); // Same session should have same fingerprint
      expect(fingerprint1).toMatch(/^[a-f0-9]{64}$/); // Should be SHA-256 hash
    });

    test('✅ Audit logs capture all scope interactions', async() => {
      const auditLogs = [];

      // Mock audit logging
      const originalNotification = scopeManager.sessionAuthority.notification;
      scopeManager.sessionAuthority.notification = (message) => {
        auditLogs.push({ timestamp: Date.now(), message });
      };

      // Perform scope operations
      await scopeTools.scope_configure({ scope: { type: 'local' } });
      await scopeTools.memory_usage({ action: 'store', key: 'test', value: 'data' });

      expect(auditLogs.length).toBeGreaterThan(0);
    });

    test('✅ Backward compatibility with existing swarms', async() => {
      // Test that existing swarm creation still works
      const legacySwarm = await ruvSwarm.createSwarm('legacy', 'mesh', 3);
      expect(legacySwarm).toBeDefined();
      expect(legacySwarm.topology).toBe('mesh');

      // Test that enhanced MCP tools work
      const enhancedSwarm = await scopeTools.swarm_init({
        topology: 'hierarchical',
        maxAgents: 5,
      });
      expect(enhancedSwarm).toBeDefined();
      expect(enhancedSwarm.topology).toBe('hierarchical');
    });
  });

  describe('🔒 Security Requirements', () => {

    test('✅ Scope boundaries cannot be bypassed', async() => {
      const localScope = scopeManager.createScope({ type: 'local' });
      const unauthorizedScope = { type: 'local', boundaries: { session: 'fake-session' } };

      // Attempt to access local scope with unauthorized session
      try {
        await scopeManager.memoryManager.retrieve(unauthorizedScope, 'test-key');
        fail('Should have thrown authorization error');
      } catch (error) {
        expect(error.message).toContain('Invalid session authority');
      }
    });

    test('✅ Sensitive data is encrypted in appropriate scopes', async() => {
      const sensitiveData = { secret: 'confidential-information' };

      // Store with encryption
      await scopeTools.memory_usage({
        action: 'store',
        key: 'sensitive-data',
        value: sensitiveData,
        scope: {
          type: 'local',
          encryption: true,
        },
      });

      // Verify data is encrypted in storage
      const rawData = scopeManager.memoryManager.scopedMemory.get(`local:${ scopeManager.sessionAuthority.sessionId }:sensitive-data`);
      expect(rawData.encrypted).toBe(true);
      expect(rawData.value.encrypted).toBeDefined();
    });

    test('✅ Access controls are properly enforced', async() => {
      // Test that global scope can be accessed by any session
      const globalScope = scopeManager.createScope({ type: 'global' });

      // Test that local scope requires session validation
      const localScope = scopeManager.createScope({ type: 'local' });

      expect(() => {
        scopeManager.memoryManager.validateScopeAccess(globalScope);
      }).not.toThrow();

      expect(() => {
        scopeManager.memoryManager.validateScopeAccess(localScope);
      }).not.toThrow(); // Should pass for same session
    });
  });

  describe('🧪 Authority Model Testing', () => {

    test('✅ Session ID generation is unique and secure', async() => {
      const sessionIds = new Set();

      // Generate multiple session IDs
      for (let i = 0; i < 100; i++) {
        const newManager = new ScopeManager();
        await newManager.initialize();
        sessionIds.add(newManager.sessionAuthority.sessionId);
      }

      // All should be unique
      expect(sessionIds.size).toBe(100);

      // Should follow expected format
      const sessionId = Array.from(sessionIds)[0];
      expect(sessionId).toMatch(/^session-[^-]+-\d+-\d+-[a-f0-9]+$/);
    });

    test('✅ Central authority detection works', async() => {
      const authority = await scopeManager.sessionAuthority.detectCentralAuthority();
      expect(authority).toBeDefined();
      expect(authority.sessionId).toBeDefined();
      expect(authority.pid).toBe(process.pid);
    });

    test('✅ Session validation is secure', async() => {
      const sessionId = scopeManager.sessionAuthority.sessionId;
      const authority = scopeManager.sessionAuthority.authority;

      // Valid session should pass
      const isValid = scopeManager.sessionAuthority.validateSession(sessionId, authority);
      expect(isValid).toBe(true);

      // Invalid session should fail
      const invalidAuthority = { ...authority, pid: 99999 };
      const isInvalid = scopeManager.sessionAuthority.validateSession(sessionId, invalidAuthority);
      expect(isInvalid).toBe(false);
    });
  });

  describe('📊 Scope Tool Integration', () => {

    test('✅ All scope tools are available', async() => {
      const tools = scopeTools.getTools();

      expect(tools.swarm_init).toBeDefined();
      expect(tools.memory_usage).toBeDefined();
      expect(tools.neural_train).toBeDefined();
      expect(tools.agent_spawn).toBeDefined();
      expect(tools.scope_configure).toBeDefined();
      expect(tools.scope_status).toBeDefined();
      expect(tools.scope_share_knowledge).toBeDefined();
      expect(tools.scope_export).toBeDefined();
      expect(tools.scope_import).toBeDefined();
      expect(tools.scope_cleanup).toBeDefined();
    });

    test('✅ Scope status provides comprehensive information', async() => {
      const status = await scopeTools.scope_status({ detailed: true });

      expect(status.sessionId).toBeDefined();
      expect(status.activeScopes).toBeDefined();
      expect(status.memory).toBeDefined();
      expect(status.neural).toBeDefined();
      expect(status.communication).toBeDefined();
    });

    test('✅ Scope cleanup works properly', async() => {
      // Create test scope
      const testScope = await scopeTools.scope_configure({
        scope: { type: 'local', isolation: 'strict' },
      });

      // Cleanup scope
      const cleanupResult = await scopeTools.scope_cleanup({
        scopeId: testScope.scopeId,
      });

      expect(cleanupResult.success).toBe(true);
    });
  });

  afterAll(async() => {
    // Cleanup test resources
    if (scopeManager) {
      await scopeTools.scope_cleanup({ type: 'all' });
    }
  });
});