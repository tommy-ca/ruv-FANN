/**
 * ruv-swarm: High-performance neural network swarm orchestration
 * Version: 1.0.16
 * 
 * Main entry point for ruv-swarm npm package
 * Provides coordination between Claude-Flow and WASM swarm engine
 */

// Import core modules
import { SwarmEngine } from './swarm-engine.js';
import { MCPTools } from './mcp-tools-enhanced.js';
import { SwarmPersistence } from './persistence.js';
import { HookSystem } from './hooks/index.js';

// Version info
export const VERSION = '1.0.16';
export const BUILD_DATE = new Date().toISOString();

/**
 * Main Swarm class that coordinates all components
 */
export class RuvSwarm {
    constructor(options = {}) {
        this.options = {
            maxAgents: 8,
            topology: 'hierarchical', 
            strategy: 'adaptive',
            enablePersistence: true,
            enableHooks: true,
            mcpMode: 'stdio',
            ...options
        };
        
        this.engine = null;
        this.mcpTools = null;
        this.persistence = null;
        this.hooks = null;
        this.initialized = false;
    }

    /**
     * Initialize the swarm system
     */
    async init() {
        try {
            // Initialize persistence layer
            if (this.options.enablePersistence) {
                this.persistence = new SwarmPersistence();
                await this.persistence.init();
            }

            // Initialize hook system
            if (this.options.enableHooks) {
                this.hooks = new HookSystem({
                    persistence: this.persistence,
                    autoConfig: true
                });
                await this.hooks.init();
            }

            // Initialize swarm engine
            this.engine = new SwarmEngine({
                ...this.options,
                persistence: this.persistence,
                hooks: this.hooks
            });
            await this.engine.init();

            // Initialize MCP tools
            this.mcpTools = new MCPTools({
                engine: this.engine,
                persistence: this.persistence,
                hooks: this.hooks,
                mode: this.options.mcpMode
            });
            await this.mcpTools.init();

            this.initialized = true;
            
            if (this.hooks) {
                await this.hooks.triggerHook('system-init', {
                    version: VERSION,
                    timestamp: Date.now(),
                    options: this.options
                });
            }

            return {
                success: true,
                version: VERSION,
                agents: 0,
                topology: this.options.topology
            };
        } catch (error) {
            console.error('Failed to initialize ruv-swarm:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get current swarm status
     */
    async getStatus() {
        if (!this.initialized) {
            return { initialized: false };
        }

        const engineStatus = await this.engine.getStatus();
        const persistence = this.persistence ? await this.persistence.getStats() : null;
        
        return {
            initialized: true,
            version: VERSION,
            engine: engineStatus,
            persistence,
            hooks: this.hooks ? this.hooks.getStatus() : null,
            mcp: this.mcpTools ? this.mcpTools.getStatus() : null
        };
    }

    /**
     * Spawn a new agent
     */
    async spawnAgent(type, options = {}) {
        if (!this.initialized) {
            throw new Error('Swarm not initialized. Call init() first.');
        }

        return await this.engine.spawnAgent(type, options);
    }

    /**
     * Orchestrate a task across the swarm
     */
    async orchestrateTask(task, options = {}) {
        if (!this.initialized) {
            throw new Error('Swarm not initialized. Call init() first.');
        }

        return await this.engine.orchestrateTask(task, options);
    }

    /**
     * Shutdown the swarm
     */
    async shutdown() {
        if (this.hooks) {
            await this.hooks.triggerHook('system-shutdown', {
                timestamp: Date.now()
            });
        }

        if (this.mcpTools) {
            await this.mcpTools.shutdown();
        }

        if (this.engine) {
            await this.engine.shutdown();
        }

        if (this.persistence) {
            await this.persistence.close();
        }

        this.initialized = false;
    }
}

// Export additional classes for advanced usage
export { SwarmEngine } from './swarm-engine.js';
export { MCPTools } from './mcp-tools-enhanced.js';
export { SwarmPersistence } from './persistence.js';
export { HookSystem } from './hooks/index.js';

// Default export
export default RuvSwarm;

// Legacy CommonJS support
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RuvSwarm;
    module.exports.RuvSwarm = RuvSwarm;
    module.exports.SwarmEngine = SwarmEngine;
    module.exports.MCPTools = MCPTools;
    module.exports.SwarmPersistence = SwarmPersistence;
    module.exports.HookSystem = HookSystem;
    module.exports.VERSION = VERSION;
}