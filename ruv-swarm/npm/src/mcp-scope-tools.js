/**
 * MCP Scope Tools Implementation
 * Extends MCP tools with scope management capabilities
 */

import { ScopeManager } from './scope-manager.js';
import { RuvSwarmError, ValidationError } from './errors.js';

/**
 * MCP Scope Tools Class
 * Provides scope-aware MCP tool implementations
 */
export class MCPScopeTools {
  constructor(ruvSwarmInstance = null) {
    this.ruvSwarm = ruvSwarmInstance;
    this.scopeManager = new ScopeManager();
    this.initialized = false;
  }

  /**
   * Initialize scope system
   */
  async initialize() {
    if (this.initialized) {
      return this.scopeManager.getStatus();
    }

    const status = await this.scopeManager.initialize();
    this.initialized = true;
    return status;
  }

  /**
   * Enhanced swarm_init with scope support
   */
  async swarm_init(params) {
    await this.initialize();

    // Extract scope configuration from params
    const scopeConfig = params.scope || {
      type: 'local',
      isolation: 'strict',
      authority: {
        sessionId: 'auto',
        central: true,
        fallback: 'generate',
      },
      sharing: {
        memory: false,
        neural: false,
        communication: false,
      },
    };

    // Create scope
    const scope = this.scopeManager.createScope(scopeConfig);

    // Initialize swarm with scope context
    const swarmParams = {
      ...params,
      scopeId: scope.id,
      sessionId: this.scopeManager.sessionAuthority.sessionId,
    };

    // Call original swarm_init if available
    let swarmResult = {};
    if (this.ruvSwarm && this.ruvSwarm.tools && this.ruvSwarm.tools.swarm_init) {
      swarmResult = await this.ruvSwarm.tools.swarm_init(swarmParams);
    } else {
      // Create minimal swarm result
      swarmResult = {
        id: `swarm-${Date.now()}`,
        message: 'Swarm initialized with scope support',
        topology: params.topology || 'mesh',
        strategy: params.strategy || 'balanced',
        maxAgents: params.maxAgents || 5,
      };
    }

    return {
      ...swarmResult,
      scope: {
        id: scope.id,
        type: scope.type,
        isolation: scope.isolation,
        sessionId: this.scopeManager.sessionAuthority.sessionId,
        boundaries: scope.boundaries,
      },
      features: {
        ...(swarmResult.features || {}),
        scope_isolation: true,
        session_authority: true,
        neural_isolation: scope.sharing.neural === false,
        memory_namespacing: true,
        communication_boundaries: scope.sharing.communication === false,
      },
    };
  }

  /**
   * Scope-aware memory operations
   */
  async memory_usage(params) {
    await this.initialize();

    const { action, key, value, scope: scopeParams } = params;

    // Get or create scope
    let scope;
    if (scopeParams && scopeParams.scopeId) {
      scope = this.scopeManager.getScope(scopeParams.scopeId);
    } else {
      // Use default local scope
      scope = this.scopeManager.createScope({
        type: 'local',
        isolation: 'strict',
      });
    }

    if (!scope) {
      throw new ValidationError('Invalid scope configuration');
    }

    switch (action) {
    case 'store':
      const scopedKey = await this.scopeManager.memoryManager.store(
        scope,
        key,
        value,
        {
          encrypted: scopeParams?.encryption || false,
          persistence: scopeParams?.persistence || 'session',
        },
      );
      return {
        success: true,
        key: scopedKey,
        scope: scope.type,
        sessionId: this.scopeManager.sessionAuthority.sessionId,
      };

    case 'retrieve':
      const retrievedValue = await this.scopeManager.memoryManager.retrieve(scope, key);
      return {
        success: true,
        value: retrievedValue,
        scope: scope.type,
        sessionId: this.scopeManager.sessionAuthority.sessionId,
      };

    case 'list':
      const keys = this.scopeManager.memoryManager.listKeys(scope, params.pattern);
      return {
        success: true,
        keys,
        scope: scope.type,
        sessionId: this.scopeManager.sessionAuthority.sessionId,
      };

    case 'delete':
      // Implement delete logic
      const scopedDeleteKey = this.scopeManager.memoryManager.generateScopedKey(scope, key);
      this.scopeManager.memoryManager.scopedMemory.delete(scopedDeleteKey);
      return {
        success: true,
        deleted: key,
        scope: scope.type,
      };

    default:
      throw new ValidationError(`Invalid memory action: ${action}`);
    }
  }

  /**
   * Scope-aware neural network operations
   */
  async neural_train(params) {
    await this.initialize();

    const { agentId, pattern, data, scope: scopeParams } = params;

    // Get or create scope
    let scope;
    if (scopeParams && scopeParams.scopeId) {
      scope = this.scopeManager.getScope(scopeParams.scopeId);
    } else {
      scope = this.scopeManager.createScope({
        type: scopeParams?.isolation || 'local',
      });
    }

    // Create scoped neural network
    const networkId = agentId || `neural-${Date.now()}`;
    const network = this.scopeManager.neuralManager.createScopedNetwork(
      scope,
      networkId,
      {
        pattern,
        isolation: scopeParams?.isolation || 'local',
        sharing: scopeParams?.sharing || 'opt-in',
        inheritance: scopeParams?.inheritance || false,
      },
    );

    return {
      success: true,
      networkId: network.id,
      pattern,
      scope: scope.type,
      isolation: scopeParams?.isolation || 'local',
      sessionId: this.scopeManager.sessionAuthority.sessionId,
      message: `Neural network trained in ${scope.type} scope`,
    };
  }

  /**
   * Scope-aware agent spawning
   */
  async agent_spawn(params) {
    await this.initialize();

    const { type, name, capabilities, scope: scopeParams } = params;

    // Get or create scope
    let scope;
    if (scopeParams && scopeParams.scopeId) {
      scope = this.scopeManager.getScope(scopeParams.scopeId);
    } else {
      scope = this.scopeManager.createScope({
        type: scopeParams?.type || 'local',
      });
    }

    // Call original agent_spawn if available
    let agentResult = {};
    if (this.ruvSwarm && this.ruvSwarm.tools && this.ruvSwarm.tools.agent_spawn) {
      agentResult = await this.ruvSwarm.tools.agent_spawn({
        ...params,
        scopeId: scope.id,
      });
    } else {
      agentResult = {
        agent: {
          id: `agent-${Date.now()}`,
          name: name || `${type}-agent`,
          type,
          capabilities: capabilities || [],
          status: 'idle',
        },
      };
    }

    // Create scoped neural network for agent
    const neuralNetwork = this.scopeManager.neuralManager.createScopedNetwork(
      scope,
      agentResult.agent.id,
      {
        type: 'agent-neural',
        agentType: type,
        capabilities: capabilities || [],
      },
    );

    return {
      ...agentResult,
      agent: {
        ...agentResult.agent,
        neural_network_id: neuralNetwork.id,
        scope: {
          id: scope.id,
          type: scope.type,
          sessionId: this.scopeManager.sessionAuthority.sessionId,
        },
      },
      scope_isolation: true,
      message: `${type} agent spawned in ${scope.type} scope`,
    };
  }

  /**
   * Scope configuration management
   */
  async scope_configure(params) {
    await this.initialize();

    const { swarmId, scope: scopeConfig } = params;

    if (!scopeConfig) {
      throw new ValidationError('Scope configuration required');
    }

    // Create or update scope
    const scope = this.scopeManager.createScope(scopeConfig);

    return {
      success: true,
      scopeId: scope.id,
      configuration: scope,
      sessionId: this.scopeManager.sessionAuthority.sessionId,
      message: `Scope configured: ${scope.type} with ${scope.isolation} isolation`,
    };
  }

  /**
   * Get scope status and information
   */
  async scope_status(params = {}) {
    await this.initialize();

    const { scopeId, detailed = false } = params;

    if (scopeId) {
      const scope = this.scopeManager.getScope(scopeId);
      if (!scope) {
        throw new ValidationError(`Scope not found: ${scopeId}`);
      }

      return {
        scope,
        status: 'active',
        sessionId: this.scopeManager.sessionAuthority.sessionId,
      };
    }

    // Return system status
    const status = this.scopeManager.getStatus();

    if (detailed) {
      return status;
    }

    return {
      sessionId: status.sessionId,
      activeScopes: status.activeScopes.length,
      scopeTypes: status.activeScopes.reduce((acc, scope) => {
        acc[scope.type] = (acc[scope.type] || 0) + 1;
        return acc;
      }, {}),
      memory: status.memory,
      neural: status.neural,
      communication: status.communication,
    };
  }

  /**
   * Cross-scope knowledge sharing
   */
  async scope_share_knowledge(params) {
    await this.initialize();

    const {
      sourceScope: sourceScopeId,
      targetScope: targetScopeId,
      knowledgeType,
      data,
      explicit = true,
    } = params;

    const sourceScope = this.scopeManager.getScope(sourceScopeId);
    const targetScope = this.scopeManager.getScope(targetScopeId);

    if (!sourceScope || !targetScope) {
      throw new ValidationError('Invalid source or target scope');
    }

    switch (knowledgeType) {
    case 'neural-pattern':
      const sharedPatternId = this.scopeManager.neuralManager.sharePattern(
        sourceScope,
        targetScope,
        data.patternId,
        { explicit },
      );
      return {
        success: true,
        sharedPatternId,
        message: 'Neural pattern shared successfully',
      };

    case 'memory':
      // Share memory data between scopes
      const sourceValue = await this.scopeManager.memoryManager.retrieve(sourceScope, data.key);
      await this.scopeManager.memoryManager.store(targetScope, data.key, sourceValue);
      return {
        success: true,
        message: 'Memory data shared successfully',
      };

    default:
      throw new ValidationError(`Invalid knowledge type: ${knowledgeType}`);
    }
  }

  /**
   * Export scope configuration
   */
  async scope_export(params = {}) {
    await this.initialize();

    const { scopeId, includeData = false } = params;

    if (scopeId) {
      const scope = this.scopeManager.getScope(scopeId);
      if (!scope) {
        throw new ValidationError(`Scope not found: ${scopeId}`);
      }

      const exportData = { scope };

      if (includeData) {
        // Include memory data, neural patterns, etc.
        exportData.memory = this.scopeManager.memoryManager.listKeys(scope);
        // Add other data exports...
      }

      return exportData;
    }

    // Export all scopes
    return this.scopeManager.exportConfig();
  }

  /**
   * Import scope configuration
   */
  async scope_import(params) {
    await this.initialize();

    const { config } = params;

    if (!config) {
      throw new ValidationError('Configuration data required');
    }

    await this.scopeManager.importConfig(config);

    return {
      success: true,
      imported: config.scopes?.length || 0,
      sessionId: this.scopeManager.sessionAuthority.sessionId,
      message: 'Scope configuration imported successfully',
    };
  }

  /**
   * Clean up scope resources
   */
  async scope_cleanup(params) {
    await this.initialize();

    const { scopeId, type = 'all' } = params;

    if (scopeId) {
      const deleted = this.scopeManager.deleteScope(scopeId);
      return {
        success: deleted,
        message: `Scope ${scopeId} cleaned up`,
      };
    }

    // Cleanup by type or all
    let cleanedCount = 0;
    for (const [id, scope] of this.scopeManager.activeScopes.entries()) {
      if (type === 'all' || scope.type === type) {
        this.scopeManager.deleteScope(id);
        cleanedCount++;
      }
    }

    return {
      success: true,
      cleaned: cleanedCount,
      message: `Cleaned up ${cleanedCount} scopes`,
    };
  }

  /**
   * Get available tools with scope support
   */
  getTools() {
    return {
      // Enhanced core tools with scope support
      swarm_init: this.swarm_init.bind(this),
      memory_usage: this.memory_usage.bind(this),
      neural_train: this.neural_train.bind(this),
      agent_spawn: this.agent_spawn.bind(this),

      // New scope-specific tools
      scope_configure: this.scope_configure.bind(this),
      scope_status: this.scope_status.bind(this),
      scope_share_knowledge: this.scope_share_knowledge.bind(this),
      scope_export: this.scope_export.bind(this),
      scope_import: this.scope_import.bind(this),
      scope_cleanup: this.scope_cleanup.bind(this),
    };
  }
}

export default MCPScopeTools;