/**
 * Swarm Persistence Layer
 * Provides database storage for swarm state and coordination
 */

import Database from 'better-sqlite3';
import { join } from 'path';

export class SwarmPersistence {
    constructor(options = {}) {
        this.dbPath = options.dbPath || join(process.cwd(), 'data', 'ruv-swarm.db');
        this.db = null;
    }

    async init() {
        try {
            this.db = new Database(this.dbPath);
            await this.createTables();
            console.log('SwarmPersistence initialized');
            return { success: true };
        } catch (error) {
            console.error('Failed to initialize persistence:', error);
            return { success: false, error: error.message };
        }
    }

    async createTables() {
        // Agent memory table
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                UNIQUE(agent_id, memory_key)
            )
        `);

        // Swarm state table
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS swarm_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_key TEXT UNIQUE NOT NULL,
                state_value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        `);

        // Task history table
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                task_data TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
        `);
    }

    async storeAgentMemory(agentId, memoryKey, memoryValue) {
        try {
            const stmt = this.db.prepare(`
                INSERT OR REPLACE INTO agent_memory 
                (agent_id, memory_key, memory_value, timestamp) 
                VALUES (?, ?, ?, ?)
            `);
            
            stmt.run(agentId, memoryKey, JSON.stringify(memoryValue), Date.now());
            return { success: true };
        } catch (error) {
            console.error('Failed to store agent memory:', error);
            return { success: false, error: error.message };
        }
    }

    async getAgentMemory(agentId, memoryKey = null) {
        try {
            let stmt, rows;
            
            if (memoryKey) {
                stmt = this.db.prepare(`
                    SELECT memory_value, timestamp 
                    FROM agent_memory 
                    WHERE agent_id = ? AND memory_key = ?
                `);
                rows = stmt.all(agentId, memoryKey);
            } else {
                stmt = this.db.prepare(`
                    SELECT memory_key, memory_value, timestamp 
                    FROM agent_memory 
                    WHERE agent_id = ?
                    ORDER BY timestamp DESC
                `);
                rows = stmt.all(agentId);
            }

            const memories = rows.map(row => ({
                ...row,
                memory_value: JSON.parse(row.memory_value)
            }));

            return { success: true, memories };
        } catch (error) {
            console.error('Failed to get agent memory:', error);
            return { success: false, error: error.message };
        }
    }

    async getStats() {
        try {
            const agentCount = this.db.prepare('SELECT COUNT(DISTINCT agent_id) as count FROM agent_memory').get().count;
            const memoryCount = this.db.prepare('SELECT COUNT(*) as count FROM agent_memory').get().count;
            const taskCount = this.db.prepare('SELECT COUNT(*) as count FROM task_history').get().count;

            return {
                agents: agentCount,
                memories: memoryCount,
                tasks: taskCount,
                dbPath: this.dbPath
            };
        } catch (error) {
            console.error('Failed to get stats:', error);
            return { error: error.message };
        }
    }

    async close() {
        if (this.db) {
            this.db.close();
            console.log('SwarmPersistence closed');
        }
    }
}