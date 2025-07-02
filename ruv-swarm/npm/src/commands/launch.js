/**
 * Launch Command Module
 * Launches Claude Code with MCP configuration
 */

import { spawn } from 'child_process';
import { existsSync } from 'fs';
import path from 'path';
import { detectClaudeCode } from '../onboarding/claude-detector.js';
import { validateMCPConfig } from '../onboarding/mcp-setup.js';
import { createCLI } from '../onboarding/interactive-cli.js';

/**
 * Launch Claude Code with MCP configuration
 * @param {Object} options - Launch options
 * @returns {Promise<{success: boolean, error?: string}>}
 */
export async function launchClaudeCode(options = {}) {
  const cli = createCLI(options);
  
  try {
    // Detect Claude Code
    cli.startSpinner('Checking for Claude Code...');
    const claudeInfo = await detectClaudeCode();
    
    if (!claudeInfo.installed) {
      cli.failSpinner('Claude Code not found');
      return {
        success: false,
        error: 'Claude Code is not installed. Please run "ruv-swarm init" first.'
      };
    }
    
    cli.succeedSpinner('Claude Code found');

    // Check for MCP configuration
    const mcpPath = path.join(process.cwd(), '.claude', 'mcp.json');
    
    if (!existsSync(mcpPath)) {
      cli.warning('MCP configuration not found');
      const shouldInit = await cli.confirm('Would you like to run initialization now?', true);
      
      if (shouldInit) {
        return {
          success: false,
          error: 'Please run "ruv-swarm init" to set up MCP servers'
        };
      }
      
      // Continue without MCP
      cli.info('Launching Claude Code without MCP servers...');
    } else {
      // Validate MCP configuration
      cli.startSpinner('Validating MCP configuration...');
      const validation = await validateMCPConfig(mcpPath);
      
      if (!validation.valid) {
        cli.failSpinner('Invalid MCP configuration');
        return {
          success: false,
          error: validation.error
        };
      }
      
      cli.succeedSpinner('MCP servers configured');
    }

    // Launch Claude Code
    cli.startSpinner('Launching Claude Code...');
    
    const args = ['--dangerously-skip-permissions'];
    
    // Add MCP config if it exists
    if (existsSync(mcpPath)) {
      args.push('--mcp-config', mcpPath);
    }

    // Add any additional arguments
    if (options.args && Array.isArray(options.args)) {
      args.push(...options.args);
    }

    const claudeProcess = spawn(claudeInfo.path, args, {
      stdio: 'inherit',
      shell: false,
      detached: false
    });

    // Handle launch success
    cli.succeedSpinner('Claude Code launched successfully');
    cli.info('Please log in to your Anthropic account when prompted');

    // Set up process handlers
    claudeProcess.on('error', (error) => {
      cli.error(`Failed to launch Claude Code: ${error.message}`);
    });

    claudeProcess.on('exit', (code, signal) => {
      if (code !== null && code !== 0) {
        cli.warning(`Claude Code exited with code ${code}`);
      } else if (signal) {
        cli.info(`Claude Code terminated by signal ${signal}`);
      }
    });

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      cli.info('\nShutting down Claude Code...');
      claudeProcess.kill('SIGINT');
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      claudeProcess.kill('SIGTERM');
      process.exit(0);
    });

    return { success: true };

  } catch (error) {
    cli.failSpinner('Failed to launch Claude Code');
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Session management for Claude Code
 */
export class SessionManager {
  constructor() {
    this.sessions = new Map();
  }

  /**
   * Create a new session
   * @param {string} sessionId - Session ID
   * @param {Object} config - Session configuration
   */
  createSession(sessionId, config) {
    this.sessions.set(sessionId, {
      id: sessionId,
      config,
      startTime: Date.now(),
      status: 'active'
    });
  }

  /**
   * Get session by ID
   * @param {string} sessionId - Session ID
   * @returns {Object|null}
   */
  getSession(sessionId) {
    return this.sessions.get(sessionId) || null;
  }

  /**
   * Update session status
   * @param {string} sessionId - Session ID
   * @param {string} status - New status
   */
  updateStatus(sessionId, status) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.status = status;
      session.lastUpdate = Date.now();
    }
  }

  /**
   * End session
   * @param {string} sessionId - Session ID
   */
  endSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.status = 'ended';
      session.endTime = Date.now();
    }
  }

  /**
   * Get all active sessions
   * @returns {Array}
   */
  getActiveSessions() {
    return Array.from(this.sessions.values())
      .filter(session => session.status === 'active');
  }
}

/**
 * Launch with session management
 * @param {Object} options - Launch options
 * @returns {Promise<{success: boolean, sessionId?: string, error?: string}>}
 */
export async function launchWithSession(options = {}) {
  const sessionManager = new SessionManager();
  const sessionId = `claude-${Date.now()}`;
  
  // Create session
  sessionManager.createSession(sessionId, options);
  
  try {
    const result = await launchClaudeCode(options);
    
    if (result.success) {
      return {
        success: true,
        sessionId
      };
    } else {
      sessionManager.updateStatus(sessionId, 'failed');
      return result;
    }
  } catch (error) {
    sessionManager.updateStatus(sessionId, 'error');
    return {
      success: false,
      error: error.message
    };
  }
}

export default {
  launchClaudeCode,
  SessionManager,
  launchWithSession
};