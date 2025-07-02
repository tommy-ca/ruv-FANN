/**
 * MCP Setup Module
 * Generates and manages MCP configuration for Claude Code
 */

import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'fs';
import path from 'path';
import os from 'os';

/**
 * MCP Configuration class
 */
export class MCPConfig {
  constructor() {
    this.servers = {};
  }

  /**
   * Add GitHub MCP server configuration
   * @param {string|null} token - GitHub token (optional)
   */
  addGitHubMCP(token) {
    this.servers.github = {
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-github'],
      env: token ? { GITHUB_TOKEN: token } : {}
    };
  }

  /**
   * Add ruv-swarm MCP server configuration
   * @param {string} swarmId - Swarm ID
   * @param {string} topology - Swarm topology (mesh, hierarchical, ring, star)
   */
  addRuvSwarmMCP(swarmId, topology = 'mesh') {
    this.servers['ruv-swarm'] = {
      command: 'npx',
      args: ['ruv-swarm', 'mcp', 'start'],
      env: {
        SWARM_ID: swarmId,
        SWARM_TOPOLOGY: topology
      }
    };
  }

  /**
   * Check if any servers are configured
   * @returns {boolean}
   */
  hasServers() {
    return Object.keys(this.servers).length > 0;
  }

  /**
   * Convert to MCP JSON format
   * @returns {string}
   */
  toJSON() {
    return JSON.stringify({
      mcpServers: this.servers
    }, null, 2);
  }
}

/**
 * Generate MCP configuration file
 * @param {string} projectPath - Path to project directory
 * @param {MCPConfig} config - MCP configuration
 * @returns {Promise<{success: boolean, path?: string, error?: string}>}
 */
export async function generateMCPConfig(projectPath, config) {
  try {
    const claudeDir = path.join(projectPath, '.claude');
    const mcpPath = path.join(claudeDir, 'mcp.json');

    // Create .claude directory if it doesn't exist
    if (!existsSync(claudeDir)) {
      mkdirSync(claudeDir, { recursive: true });
    }

    // Write MCP configuration
    writeFileSync(mcpPath, config.toJSON());

    return {
      success: true,
      path: mcpPath
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Detect GitHub token from environment
 * @returns {string|null}
 */
export function detectGitHubToken() {
  // Check common GitHub token environment variables
  const tokenVars = ['GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_PAT'];
  
  for (const varName of tokenVars) {
    const token = process.env[varName];
    if (token) {
      return token;
    }
  }

  // Check gh CLI config
  try {
    const ghConfigPath = path.join(os.homedir(), '.config', 'gh', 'hosts.yml');
    if (existsSync(ghConfigPath)) {
      const config = readFileSync(ghConfigPath, 'utf8');
      // Simple regex to find oauth_token
      const match = config.match(/oauth_token:\s*([^\s]+)/);
      if (match) {
        return match[1];
      }
    }
  } catch (error) {
    // Ignore errors reading gh config
  }

  return null;
}

/**
 * Validate MCP JSON structure
 * @param {string} jsonPath - Path to MCP JSON file
 * @returns {Promise<{valid: boolean, error?: string}>}
 */
export async function validateMCPConfig(jsonPath) {
  try {
    if (!existsSync(jsonPath)) {
      return {
        valid: false,
        error: 'MCP configuration file not found'
      };
    }

    const content = readFileSync(jsonPath, 'utf8');
    const config = JSON.parse(content);

    // Check required structure
    if (!config.mcpServers || typeof config.mcpServers !== 'object') {
      return {
        valid: false,
        error: 'Invalid MCP configuration: missing mcpServers object'
      };
    }

    // Validate each server configuration
    for (const [serverName, serverConfig] of Object.entries(config.mcpServers)) {
      if (!serverConfig.command) {
        return {
          valid: false,
          error: `Invalid MCP configuration: server '${serverName}' missing command`
        };
      }
      if (!Array.isArray(serverConfig.args)) {
        return {
          valid: false,
          error: `Invalid MCP configuration: server '${serverName}' args must be an array`
        };
      }
    }

    return { valid: true };
  } catch (error) {
    return {
      valid: false,
      error: `Failed to parse MCP configuration: ${error.message}`
    };
  }
}

/**
 * Generate a unique swarm ID
 * @returns {string}
 */
export function generateSwarmId() {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 7);
  return `swarm-${timestamp}-${random}`;
}

export default {
  MCPConfig,
  generateMCPConfig,
  detectGitHubToken,
  validateMCPConfig,
  generateSwarmId
};