/**
 * Scope Management System for ruv-swarm
 * Implements global and local scopes for swarm communication and data isolation
 */

import crypto from 'crypto';
import os from 'os';
import path from 'path';
import { promises as fs } from 'fs';
import { RuvSwarmError, ValidationError } from './errors.js';

/**
 * Session Authority System
 * Manages unique session identifiers and authority validation
 */
class SessionAuthority {
  constructor() {
    this.sessionId = null;
    this.authority = null;
    this.sessionMetadata = null;
    this.centralAuthority = null;
  }

  /**
   * Generate unique session identifier with authority validation
   */
  generateSessionId() {
    const hostname = os.hostname();
    const { pid } = process;
    const timestamp = Date.now();
    const salt = crypto.randomBytes(16).toString('hex');

    this.sessionId = `session-${hostname}-${pid}-${timestamp}-${salt}`;
    this.authority = {
      sessionId: this.sessionId,
      hostname,
      pid,
      timestamp,
      salt,
      fingerprint: this.generateFingerprint(),
    };

    return this.sessionId;
  }

  /**
   * Generate cryptographic fingerprint for session validation
   */
  generateFingerprint() {
    const metadata = {
      hostname: os.hostname(),
      platform: os.platform(),
      arch: os.arch(),
      pid: process.pid,
      cwd: process.cwd(),
      nodeVersion: process.version,
      sessionSalt: this.sessionId || 'default',
    };

    const hash = crypto.createHash('sha256');
    hash.update(JSON.stringify(metadata, Object.keys(metadata).sort()));
    return hash.digest('hex');
  }

  /**
   * Validate session authority
   */
  validateSession(sessionId, authority) {
    if (!sessionId || !authority) {
      return false;
    }

    // For same session, check if the authority matches current session
    if (authority.sessionId === this.sessionId) {
      return authority.sessionId === sessionId &&
             authority.pid === process.pid &&
             authority.hostname === os.hostname();
    }

    // For different sessions, return false (isolated)
    return false;
  }

  /**
   * Detect and register with central authority
   */
  async detectCentralAuthority() {
    const lockFile = path.join(os.tmpdir(), 'ruv-swarm-authority.lock');

    try {
      // Try to read existing authority
      const authorityData = await fs.readFile(lockFile, 'utf8');
      this.centralAuthority = JSON.parse(authorityData);

      // Validate if central authority is still active
      if (await this.validateCentralAuthority(this.centralAuthority)) {
        return this.centralAuthority;
      }
    } catch (error) {
      // No existing authority or invalid
    }

    // Become central authority
    return this.becomeCentralAuthority(lockFile);
  }

  /**
   * Become the central authority
   */
  async becomeCentralAuthority(lockFile) {
    const authority = {
      sessionId: this.sessionId || this.generateSessionId(),
      pid: process.pid,
      hostname: os.hostname(),
      timestamp: Date.now(),
      sessions: new Map(),
    };

    try {
      await fs.writeFile(lockFile, JSON.stringify(authority), 'utf8');
      this.centralAuthority = authority;
      return authority;
    } catch (error) {
      throw new RuvSwarmError(`Failed to establish central authority: ${error.message}`);
    }
  }

  /**
   * Validate if central authority is still active
   */
  async validateCentralAuthority(authority) {
    try {
      // Check if process still exists
      process.kill(authority.pid, 0);
      return true;
    } catch (error) {
      return false;
    }
  }
}

// Global memory store for cross-session sharing
const globalMemoryStore = new Map();

/**
 * Memory Namespace Manager
 * Handles memory isolation based on scope configuration
 */
class MemoryNamespaceManager {
  constructor(sessionAuthority) {
    this.sessionAuthority = sessionAuthority;
    this.namespaces = new Map();
    this.scopedMemory = new Map();
  }

  /**
   * Generate scoped memory key
   */
  generateScopedKey(scope, key) {
    switch (scope.type) {
    case 'local':
      return `local:${scope.boundaries.session}:${key}`;
    case 'project':
      return `project:${scope.boundaries.project}:${key}`;
    case 'team':
      return `team:${scope.boundaries.team}:${key}`;
    case 'global':
      return `global:${key}`;
    default:
      throw new ValidationError(`Invalid scope type: ${scope.type}`);
    }
  }

  /**
   * Store data with scope validation
   */
  async store(scope, key, value, options = {}) {
    // Validate scope boundaries
    this.validateScopeAccess(scope);

    const scopedKey = this.generateScopedKey(scope, key);
    const data = {
      value,
      timestamp: Date.now(),
      scope,
      authority: this.sessionAuthority.authority,
      encrypted: options.encrypted || false,
    };

    // Encrypt if required
    if (options.encrypted || scope.security?.encryption) {
      data.value = this.encryptData(data.value);
      data.encrypted = true;
    }

    // Store in appropriate memory store based on scope type
    if (scope.type === 'global') {
      globalMemoryStore.set(scopedKey, data);
    } else {
      this.scopedMemory.set(scopedKey, data);
    }
    return scopedKey;
  }

  /**
   * Retrieve data with scope validation
   */
  async retrieve(scope, key) {
    this.validateScopeAccess(scope);

    const scopedKey = this.generateScopedKey(scope, key);
    // Retrieve from appropriate memory store based on scope type
    const data = scope.type === 'global' ?
      globalMemoryStore.get(scopedKey) :
      this.scopedMemory.get(scopedKey);

    if (!data) {
      return null;
    }

    // Validate authority for local scopes
    if (scope.type === 'local' &&
        data.authority.sessionId !== this.sessionAuthority.sessionId) {
      throw new RuvSwarmError('Unauthorized access to local scope memory');
    }

    // Decrypt if needed
    if (data.encrypted) {
      data.value = this.decryptData(data.value);
    }

    return data.value;
  }

  /**
   * List keys by scope pattern
   */
  listKeys(scope, pattern = '*') {
    const scopePrefix = this.generateScopedKey(scope, '').slice(0, -1); // Remove trailing ':'
    const keys = [];

    for (const [key, data] of this.scopedMemory.entries()) {
      if (key.startsWith(scopePrefix)) {
        // Validate access
        try {
          this.validateScopeAccess(scope);
          keys.push(key.substring(scopePrefix.length + 1));
        } catch (error) {
          // Skip inaccessible keys
        }
      }
    }

    return keys;
  }

  /**
   * Validate scope access based on authority
   */
  validateScopeAccess(scope) {
    switch (scope.type) {
    case 'local':
      // For local scopes, check if this session has access
      if (scope.boundaries.session &&
            scope.boundaries.session !== this.sessionAuthority.sessionId) {
        throw new RuvSwarmError('Invalid session authority for local scope');
      }
      break;
    case 'project':
    case 'team':
    case 'global':
      // Additional validation logic can be added here
      break;
    default:
      throw new ValidationError(`Invalid scope type: ${scope.type}`);
    }
  }

  /**
   * Simple encryption for sensitive data
   */
  encryptData(data) {
    const algorithm = 'aes-256-cbc';
    const key = crypto.scryptSync('ruv-swarm-secret', 'salt', 32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv(algorithm, key, iv);

    let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
    encrypted += cipher.final('hex');

    return {
      encrypted,
      iv: iv.toString('hex'),
      algorithm,
    };
  }

  /**
   * Simple decryption for sensitive data
   */
  decryptData(encryptedData) {
    const key = crypto.scryptSync('ruv-swarm-secret', 'salt', 32);
    const iv = Buffer.from(encryptedData.iv, 'hex');
    const decipher = crypto.createDecipheriv(encryptedData.algorithm, key, iv);

    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return JSON.parse(decrypted);
  }
}

/**
 * Neural Network Isolation Manager
 * Manages neural network scoping and pattern sharing
 */
class NeuralIsolationManager {
  constructor(sessionAuthority) {
    this.sessionAuthority = sessionAuthority;
    this.neuralModels = new Map();
    this.patternRegistry = new Map();
  }

  /**
   * Create scoped neural network
   */
  createScopedNetwork(scope, networkId, config) {
    this.validateScopeAccess(scope);

    const scopedId = this.generateScopedNetworkId(scope, networkId);
    const network = {
      id: scopedId,
      config,
      scope,
      patterns: new Map(),
      authority: this.sessionAuthority.authority,
      created: Date.now(),
    };

    this.neuralModels.set(scopedId, network);
    return network;
  }

  /**
   * Get scoped neural network
   */
  getScopedNetwork(scope, networkId) {
    this.validateScopeAccess(scope);

    const scopedId = this.generateScopedNetworkId(scope, networkId);
    return this.neuralModels.get(scopedId);
  }

  /**
   * Share neural pattern across scopes
   */
  sharePattern(sourceScope, targetScope, patternId, options = {}) {
    if (!options.explicit) {
      throw new RuvSwarmError('Pattern sharing requires explicit permission');
    }

    const sourcePattern = this.getPattern(sourceScope, patternId);
    if (!sourcePattern) {
      throw new RuvSwarmError(`Pattern not found: ${patternId}`);
    }

    // Validate sharing permissions
    this.validatePatternSharing(sourceScope, targetScope);

    // Copy pattern to target scope
    const targetPatternId = this.generateScopedPatternId(targetScope, patternId);
    this.patternRegistry.set(targetPatternId, {
      ...sourcePattern,
      sharedFrom: sourceScope,
      sharedAt: Date.now(),
    });

    return targetPatternId;
  }

  /**
   * Generate scoped network ID
   */
  generateScopedNetworkId(scope, networkId) {
    switch (scope.type) {
    case 'local':
      return `local:${scope.boundaries.session}:neural:${networkId}`;
    case 'project':
      return `project:${scope.boundaries.project}:neural:${networkId}`;
    case 'team':
      return `team:${scope.boundaries.team}:neural:${networkId}`;
    case 'global':
      return `global:neural:${networkId}`;
    default:
      throw new ValidationError(`Invalid scope type: ${scope.type}`);
    }
  }

  /**
   * Generate scoped pattern ID
   */
  generateScopedPatternId(scope, patternId) {
    return `${this.generateScopedNetworkId(scope, 'patterns')}:${patternId}`;
  }

  /**
   * Validate scope access
   */
  validateScopeAccess(scope) {
    // Reuse logic from MemoryNamespaceManager
    switch (scope.type) {
    case 'local':
      if (scope.boundaries.session &&
            scope.boundaries.session !== this.sessionAuthority.sessionId) {
        throw new RuvSwarmError('Invalid session authority for local scope');
      }
      break;
    case 'project':
    case 'team':
    case 'global':
      // Global, project and team scopes allow broader access
      break;
    }
  }

  /**
   * Validate pattern sharing permissions
   */
  validatePatternSharing(sourceScope, targetScope) {
    // Implement sharing validation logic
    if (sourceScope.type === 'local' && targetScope.type !== 'local') {
      throw new RuvSwarmError('Cannot share local patterns to non-local scopes without explicit permission');
    }
  }

  /**
   * Get pattern by scope and ID
   */
  getPattern(scope, patternId) {
    const scopedId = this.generateScopedPatternId(scope, patternId);
    return this.patternRegistry.get(scopedId);
  }
}

/**
 * Communication Boundary Manager
 * Controls inter-swarm communication based on scope rules
 */
class CommunicationBoundaryManager {
  constructor(sessionAuthority) {
    this.sessionAuthority = sessionAuthority;
    this.communicationChannels = new Map();
    this.messageFilters = new Map();
  }

  /**
   * Create scoped communication channel
   */
  createChannel(scope, channelId) {
    this.validateScopeAccess(scope);

    const scopedChannelId = this.generateScopedChannelId(scope, channelId);
    const channel = {
      id: scopedChannelId,
      scope,
      participants: new Set(),
      messageHistory: [],
      authority: this.sessionAuthority.authority,
      created: Date.now(),
    };

    this.communicationChannels.set(scopedChannelId, channel);
    return channel;
  }

  /**
   * Send message with scope validation
   */
  sendMessage(scope, channelId, message, targetScope = null) {
    this.validateScopeAccess(scope);

    const channel = this.getChannel(scope, channelId);
    if (!channel) {
      throw new RuvSwarmError(`Channel not found: ${channelId}`);
    }

    // Validate cross-scope communication
    if (targetScope && !this.canCommunicate(scope, targetScope)) {
      throw new RuvSwarmError('Cross-scope communication not allowed');
    }

    const messageData = {
      id: crypto.randomUUID(),
      content: message,
      sender: scope,
      target: targetScope,
      timestamp: Date.now(),
      authority: this.sessionAuthority.authority,
    };

    channel.messageHistory.push(messageData);
    return messageData;
  }

  /**
   * Check if communication is allowed between scopes
   */
  canCommunicate(sourceScope, targetScope) {
    // Local scopes can only communicate within same session
    if (sourceScope.type === 'local' && targetScope.type === 'local') {
      return sourceScope.boundaries.session === targetScope.boundaries.session;
    }

    // Project scopes can communicate within same project
    if (sourceScope.type === 'project' && targetScope.type === 'project') {
      return sourceScope.boundaries.project === targetScope.boundaries.project;
    }

    // Team scopes can communicate within same team
    if (sourceScope.type === 'team' && targetScope.type === 'team') {
      return sourceScope.boundaries.team === targetScope.boundaries.team;
    }

    // Global scope can communicate with any scope
    if (sourceScope.type === 'global' || targetScope.type === 'global') {
      return true;
    }

    return false;
  }

  /**
   * Generate scoped channel ID
   */
  generateScopedChannelId(scope, channelId) {
    switch (scope.type) {
    case 'local':
      return `local:${scope.boundaries.session}:comm:${channelId}`;
    case 'project':
      return `project:${scope.boundaries.project}:comm:${channelId}`;
    case 'team':
      return `team:${scope.boundaries.team}:comm:${channelId}`;
    case 'global':
      return `global:comm:${channelId}`;
    default:
      throw new ValidationError(`Invalid scope type: ${scope.type}`);
    }
  }

  /**
   * Get channel by scope and ID
   */
  getChannel(scope, channelId) {
    const scopedId = this.generateScopedChannelId(scope, channelId);
    return this.communicationChannels.get(scopedId);
  }

  /**
   * Validate scope access
   */
  validateScopeAccess(scope) {
    switch (scope.type) {
    case 'local':
      if (scope.boundaries.session &&
            scope.boundaries.session !== this.sessionAuthority.sessionId) {
        throw new RuvSwarmError('Invalid session authority for local scope');
      }
      break;
    case 'project':
    case 'team':
    case 'global':
      // Global, project and team scopes allow broader access
      break;
    }
  }
}

/**
 * Main Scope Manager
 * Orchestrates all scope-related functionality
 */
export class ScopeManager {
  constructor(options = {}) {
    this.sessionAuthority = new SessionAuthority();
    this.memoryManager = new MemoryNamespaceManager(this.sessionAuthority);
    this.neuralManager = new NeuralIsolationManager(this.sessionAuthority);
    this.communicationManager = new CommunicationBoundaryManager(this.sessionAuthority);

    this.activeScopes = new Map();
    this.defaultScopeConfig = {
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
      boundaries: {},
      security: {
        encryption: true,
        auditLog: true,
      },
    };
  }

  /**
   * Initialize scope system
   */
  async initialize() {
    // Generate session ID
    this.sessionAuthority.generateSessionId();

    // Detect or become central authority
    await this.sessionAuthority.detectCentralAuthority();

    return {
      sessionId: this.sessionAuthority.sessionId,
      authority: this.sessionAuthority.authority,
      centralAuthority: this.sessionAuthority.centralAuthority,
    };
  }

  /**
   * Create new scope configuration
   */
  createScope(config = {}) {
    const scope = {
      ...this.defaultScopeConfig,
      ...config,
      id: crypto.randomUUID(),
      created: Date.now(),
    };

    // Set session boundary for local scopes
    if (scope.type === 'local') {
      scope.boundaries.session = this.sessionAuthority.sessionId;
    }

    this.activeScopes.set(scope.id, scope);
    return scope;
  }

  /**
   * Get scope by ID
   */
  getScope(scopeId) {
    return this.activeScopes.get(scopeId);
  }

  /**
   * Update scope configuration
   */
  updateScope(scopeId, updates) {
    const scope = this.activeScopes.get(scopeId);
    if (!scope) {
      throw new RuvSwarmError(`Scope not found: ${scopeId}`);
    }

    const updatedScope = { ...scope, ...updates, updated: Date.now() };
    this.activeScopes.set(scopeId, updatedScope);
    return updatedScope;
  }

  /**
   * Delete scope
   */
  deleteScope(scopeId) {
    const scope = this.activeScopes.get(scopeId);
    if (!scope) {
      throw new RuvSwarmError(`Scope not found: ${scopeId}`);
    }

    // Clean up associated resources
    this.cleanupScopeResources(scope);
    this.activeScopes.delete(scopeId);
    return true;
  }

  /**
   * Clean up resources associated with a scope
   */
  cleanupScopeResources(scope) {
    // Clean up memory
    const memoryKeys = this.memoryManager.listKeys(scope);
    memoryKeys.forEach(key => {
      const scopedKey = this.memoryManager.generateScopedKey(scope, key);
      this.memoryManager.scopedMemory.delete(scopedKey);
    });

    // Clean up neural networks
    // Clean up communication channels
    // Implementation details...
  }

  /**
   * Get system status
   */
  getStatus() {
    return {
      sessionId: this.sessionAuthority.sessionId,
      authority: this.sessionAuthority.authority,
      centralAuthority: this.sessionAuthority.centralAuthority,
      activeScopes: Array.from(this.activeScopes.values()),
      memory: {
        totalKeys: this.memoryManager.scopedMemory.size,
        byScope: this.getMemoryStatsByScope(),
      },
      neural: {
        totalNetworks: this.neuralManager.neuralModels.size,
        totalPatterns: this.neuralManager.patternRegistry.size,
      },
      communication: {
        totalChannels: this.communicationManager.communicationChannels.size,
      },
    };
  }

  /**
   * Get memory statistics by scope
   */
  getMemoryStatsByScope() {
    const stats = {};
    for (const [key, data] of this.memoryManager.scopedMemory.entries()) {
      const scopeType = key.split(':')[0];
      if (!stats[scopeType]) {
        stats[scopeType] = 0;
      }
      stats[scopeType]++;
    }
    return stats;
  }

  /**
   * Export configuration
   */
  exportConfig() {
    return {
      sessionId: this.sessionAuthority.sessionId,
      scopes: Array.from(this.activeScopes.values()),
      timestamp: Date.now(),
    };
  }

  /**
   * Import configuration
   */
  async importConfig(config) {
    if (config.sessionId !== this.sessionAuthority.sessionId) {
      throw new RuvSwarmError('Cannot import config from different session');
    }

    for (const scope of config.scopes) {
      this.activeScopes.set(scope.id, scope);
    }

    return true;
  }
}

export default ScopeManager;