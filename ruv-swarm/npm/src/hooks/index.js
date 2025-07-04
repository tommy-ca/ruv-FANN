/**
 * Hook System for Claude-Flow Integration
 * Provides automated coordination hooks for ruv-swarm
 */

export class HookSystem {
    constructor(options = {}) {
        this.options = options;
        this.persistence = options.persistence;
        this.sessionData = {
            notifications: [],
            tasks: [],
            memory: new Map()
        };
        this.hooks = new Map();
        this.autoConfig = options.autoConfig || true;
    }

    async init() {
        if (this.autoConfig) {
            await this.setupDefaultHooks();
        }
        console.log('HookSystem initialized');
        return { success: true };
    }

    async setupDefaultHooks() {
        // Pre-task hooks
        this.registerHook('pre-task', async (data) => {
            const notification = {
                type: 'pre-task',
                description: data.description || 'Task starting',
                timestamp: Date.now(),
                agentId: data.agentId || 'system'
            };
            
            this.sessionData.notifications.push(notification);
            
            if (this.persistence) {
                await this.storeNotificationInDatabase(notification);
            }
            
            return { success: true, notification };
        });

        // Post-edit hooks  
        this.registerHook('post-edit', async (data) => {
            const notification = {
                type: 'post-edit',
                file: data.file,
                memoryKey: data.memoryKey || `edit/${Date.now()}`,
                timestamp: Date.now(),
                agentId: data.agentId || 'system'
            };
            
            this.sessionData.notifications.push(notification);
            
            if (this.persistence) {
                await this.storeNotificationInDatabase(notification);
                await this.persistence.storeAgentMemory(
                    notification.agentId, 
                    notification.memoryKey, 
                    notification
                );
            }
            
            return { success: true, notification };
        });

        // Notification hooks
        this.registerHook('notification', async (data) => {
            const notification = {
                type: 'notification',
                message: data.message,
                telemetry: data.telemetry || false,
                timestamp: Date.now(),
                agentId: data.agentId || 'system'
            };
            
            this.sessionData.notifications.push(notification);
            
            if (this.persistence) {
                await this.storeNotificationInDatabase(notification);
            }
            
            return { success: true, notification };
        });

        // Session management hooks
        this.registerHook('session-restore', async (data) => {
            if (this.persistence && data.loadMemory) {
                const result = await this.persistence.getAgentMemory('system');
                if (result.success) {
                    console.log(`Restored ${result.memories.length} memories`);
                }
            }
            return { success: true, sessionId: data.sessionId };
        });

        this.registerHook('session-end', async (data) => {
            if (data.generateSummary) {
                const summary = {
                    notifications: this.sessionData.notifications.length,
                    tasks: this.sessionData.tasks.length,
                    timestamp: Date.now()
                };
                
                if (this.persistence) {
                    await this.persistence.storeAgentMemory('system', 'session-summary', summary);
                }
                
                return { success: true, summary };
            }
            return { success: true };
        });

        // Task management hooks
        this.registerHook('post-task', async (data) => {
            const task = {
                taskId: data.taskId,
                analyzePerformance: data.analyzePerformance || false,
                timestamp: Date.now(),
                status: 'completed'
            };
            
            this.sessionData.tasks.push(task);
            
            if (this.persistence) {
                await this.persistence.storeAgentMemory('system', `task/${task.taskId}`, task);
            }
            
            return { success: true, task };
        });

        // Search and caching hooks
        this.registerHook('pre-search', async (data) => {
            const search = {
                query: data.query,
                cacheResults: data.cacheResults || false,
                timestamp: Date.now()
            };
            
            return { success: true, search };
        });

        // System hooks
        this.registerHook('system-init', async (data) => {
            console.log(`System initialized v${data.version}`);
            return { success: true, version: data.version };
        });

        this.registerHook('system-shutdown', async (data) => {
            console.log('System shutting down');
            return { success: true };
        });
    }

    registerHook(name, handler) {
        this.hooks.set(name, handler);
    }

    async triggerHook(name, data = {}) {
        const handler = this.hooks.get(name);
        if (handler) {
            try {
                return await handler(data);
            } catch (error) {
                console.error(`Hook ${name} failed:`, error);
                return { success: false, error: error.message };
            }
        }
        
        console.warn(`Hook ${name} not found`);
        return { success: false, error: 'Hook not found' };
    }

    async storeNotificationInDatabase(notification) {
        if (this.persistence) {
            const memoryKey = `notifications/${notification.type}/${notification.timestamp}`;
            await this.persistence.storeAgentMemory(
                notification.agentId, 
                memoryKey, 
                {
                    ...notification,
                    source: 'hook-system',
                    storedAt: Date.now()
                }
            );
        }
    }

    getStatus() {
        return {
            hooksRegistered: this.hooks.size,
            notifications: this.sessionData.notifications.length,
            tasks: this.sessionData.tasks.length,
            memory: this.sessionData.memory.size
        };
    }

    async shutdown() {
        this.hooks.clear();
        this.sessionData = { notifications: [], tasks: [], memory: new Map() };
        console.log('HookSystem shutdown');
    }
}