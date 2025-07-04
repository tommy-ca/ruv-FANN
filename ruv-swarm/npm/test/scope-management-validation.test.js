/**
 * Scope Management Validation Test
 * Validates Epic Issue #66: Global and Local Scopes core functionality
 */

import { ScopeManager } from '../src/scope-manager.js';
import { MCPScopeTools } from '../src/mcp-scope-tools.js';

describe('Epic #66: Scope Management Validation', () => {
    let scopeManager;

    beforeAll(async () => {
        scopeManager = new ScopeManager();
        await scopeManager.initialize();
    }, 60000);

    describe('🔧 Core Scope Management', () => {
        test('✅ ScopeManager initializes correctly', async () => {
            expect(scopeManager).toBeDefined();
            expect(scopeManager.sessionAuthority).toBeDefined();
            expect(scopeManager.memoryManager).toBeDefined();
            expect(scopeManager.communicationManager).toBeDefined();
            expect(scopeManager.neuralManager).toBeDefined();
        });

        test('✅ Session ID generation is unique and secure', async () => {
            const sessionId1 = scopeManager.sessionAuthority.sessionId;
            expect(sessionId1).toBeDefined();
            expect(sessionId1).toMatch(/^session-.+-\d+-\d+-[a-f0-9]+$/);

            // Create another manager to test uniqueness
            const scopeManager2 = new ScopeManager();
            await scopeManager2.initialize();
            const sessionId2 = scopeManager2.sessionAuthority.sessionId;
            
            expect(sessionId1).not.toBe(sessionId2);
        });

        test('✅ Can create different scope types', async () => {
            const localScope = scopeManager.createScope({
                type: 'local',
                isolation: 'strict'
            });
            expect(localScope.type).toBe('local');
            expect(localScope.isolation).toBe('strict');

            const globalScope = scopeManager.createScope({
                type: 'global',
                isolation: 'permissive'
            });
            expect(globalScope.type).toBe('global');
            expect(globalScope.isolation).toBe('permissive');

            const projectScope = scopeManager.createScope({
                type: 'project',
                boundaries: { project: 'test-project' }
            });
            expect(projectScope.type).toBe('project');
            expect(projectScope.boundaries.project).toBe('test-project');
        });

        test('✅ Memory isolation works correctly', async () => {
            const localScope = scopeManager.createScope({ type: 'local' });
            const globalScope = scopeManager.createScope({ type: 'global' });

            // Store data in local scope
            await scopeManager.memoryManager.store(localScope, 'test-key', { data: 'local-data' });
            
            // Store data in global scope with same key
            await scopeManager.memoryManager.store(globalScope, 'test-key', { data: 'global-data' });

            // Retrieve from each scope - should be isolated
            const localData = await scopeManager.memoryManager.retrieve(localScope, 'test-key');
            const globalData = await scopeManager.memoryManager.retrieve(globalScope, 'test-key');

            expect(localData.data).toBe('local-data');
            expect(globalData.data).toBe('global-data');
        });

        test('✅ Communication boundaries are enforced', async () => {
            const localScope1 = scopeManager.createScope({ type: 'local' });
            const localScope2 = scopeManager.createScope({ type: 'local' });
            const globalScope1 = scopeManager.createScope({ type: 'global' });
            const globalScope2 = scopeManager.createScope({ type: 'global' });

            // Local scopes from same session can communicate
            const localCanCommunicate = scopeManager.communicationManager.canCommunicate(localScope1, localScope2);
            expect(localCanCommunicate).toBe(true);

            // Global scopes can communicate
            const globalCanCommunicate = scopeManager.communicationManager.canCommunicate(globalScope1, globalScope2);
            expect(globalCanCommunicate).toBe(true);

            // Test cross-session isolation by creating scope with different session
            const crossSessionScope = {
                type: 'local',
                boundaries: { session: 'different-session-id' }
            };
            
            const crossSessionCanCommunicate = scopeManager.communicationManager.canCommunicate(localScope1, crossSessionScope);
            expect(crossSessionCanCommunicate).toBe(false);
        });

        test('✅ Session authority validation works', async () => {
            const sessionId = scopeManager.sessionAuthority.sessionId;
            const authority = scopeManager.sessionAuthority.authority;

            // Check that authority components are valid
            expect(authority.sessionId).toBe(sessionId);
            expect(authority.pid).toBe(process.pid);
            expect(authority.fingerprint).toBeDefined();

            // Invalid authority should fail
            const invalidAuthority = { ...authority, pid: 99999 };
            const isInvalid = scopeManager.sessionAuthority.validateSession(sessionId, invalidAuthority);
            expect(isInvalid).toBe(false);
        });

        test('✅ Scope fingerprinting is consistent', async () => {
            const fingerprint1 = scopeManager.sessionAuthority.generateFingerprint();
            const fingerprint2 = scopeManager.sessionAuthority.generateFingerprint();
            
            // Same session should have same fingerprint
            expect(fingerprint1).toBe(fingerprint2);
            expect(fingerprint1).toMatch(/^[a-f0-9]{64}$/); // SHA-256 hash
        });
    });

    describe('🔌 MCP Scope Tools Integration', () => {
        let scopeTools;
        let mockRuvSwarm;

        beforeAll(async () => {
            // Create a minimal mock RuvSwarm for testing
            mockRuvSwarm = {
                createSwarm: async () => ({
                    id: 'test-swarm',
                    topology: 'mesh',
                    agents: []
                }),
                memoryUsage: async () => ({ success: true }),
                neuralTrain: async () => ({ success: true }),
                agentSpawn: async () => ({ id: 'test-agent' })
            };

            scopeTools = new MCPScopeTools(mockRuvSwarm);
            await scopeTools.initialize();
        });

        test('✅ MCPScopeTools initializes with scope manager', async () => {
            expect(scopeTools).toBeDefined();
            expect(scopeTools.scopeManager).toBeDefined();
        });

        test('✅ All scope tools are available', async () => {
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

        test('✅ Scope status provides comprehensive information', async () => {
            const status = await scopeTools.scope_status({ detailed: true });
            
            expect(status.sessionId).toBeDefined();
            expect(status.activeScopes).toBeDefined();
            expect(status.memory).toBeDefined();
            expect(status.neural).toBeDefined();
            expect(status.communication).toBeDefined();
        });

        test('✅ Scope configuration works', async () => {
            const result = await scopeTools.scope_configure({
                scope: {
                    type: 'local',
                    isolation: 'strict'
                }
            });

            expect(result.success).toBe(true);
            expect(result.configuration).toBeDefined();
            expect(result.configuration.type).toBe('local');
        });

        test('✅ Memory operations are scope-aware', async () => {
            const storeResult = await scopeTools.memory_usage({
                action: 'store',
                key: 'test-scoped-data',
                value: { message: 'hello scope' },
                scope: { type: 'local' }
            });

            expect(storeResult.success).toBe(true);

            const retrieveResult = await scopeTools.memory_usage({
                action: 'retrieve',
                key: 'test-scoped-data',
                scope: { type: 'local' }
            });

            expect(retrieveResult.success).toBe(true);
            expect(retrieveResult.value.message).toBe('hello scope');
        });
    });

    describe('🔒 Security Validation', () => {
        test('✅ Encrypted data storage works', async () => {
            const localScope = scopeManager.createScope({ 
                type: 'local',
                encryption: true 
            });

            const sensitiveData = { secret: 'confidential-information' };
            
            await scopeManager.memoryManager.store(localScope, 'sensitive-key', sensitiveData, { encrypt: true });
            
            // Check that data is encrypted in storage
            const sessionId = scopeManager.sessionAuthority.sessionId;
            const storageKey = `local:${sessionId}:sensitive-key`;
            const rawData = scopeManager.memoryManager.scopedMemory.get(storageKey);
            
            expect(rawData.encrypted).toBe(true);
            expect(rawData.value.encrypted).toBeDefined();
            expect(rawData.value.data).not.toBe(sensitiveData); // Should be encrypted
        });

        test('✅ Scope access validation prevents unauthorized access', async () => {
            const localScope = scopeManager.createScope({ type: 'local' });
            
            // This should not throw for same session
            expect(() => {
                scopeManager.memoryManager.validateScopeAccess(localScope);
            }).not.toThrow();

            // Test with invalid scope (simulated different session)
            const invalidScope = {
                type: 'local',
                boundaries: { session: 'fake-session-id' }
            };

            try {
                await scopeManager.memoryManager.retrieve(invalidScope, 'test-key');
                fail('Should have thrown authorization error');
            } catch (error) {
                expect(error.message).toContain('Invalid session authority');
            }
        });
    });

    afterAll(async () => {
        // Cleanup test resources
        if (scopeManager) {
            // Clear active scopes
            scopeManager.activeScopes.clear();
        }
    });
});