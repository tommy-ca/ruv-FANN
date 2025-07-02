/**
 * Tests for Launch Command Module
 */

import { jest } from '@jest/globals';
import { spawn } from 'child_process';
import { existsSync } from 'fs';
import {
  launchClaudeCode,
  SessionManager,
  launchWithSession
} from '../../src/commands/launch.js';
import { detectClaudeCode } from '../../src/onboarding/claude-detector.js';
import { validateMCPConfig } from '../../src/onboarding/mcp-setup.js';

// Mock dependencies
jest.mock('child_process');
jest.mock('fs');
jest.mock('../../src/onboarding/claude-detector.js');
jest.mock('../../src/onboarding/mcp-setup.js');
jest.mock('../../src/onboarding/interactive-cli.js', () => ({
  createCLI: jest.fn(() => ({
    startSpinner: jest.fn(),
    succeedSpinner: jest.fn(),
    failSpinner: jest.fn(),
    warnSpinner: jest.fn(),
    info: jest.fn(),
    warning: jest.fn(),
    error: jest.fn(),
    confirm: jest.fn()
  }))
}));

describe('Launch Command', () => {
  let mockSpawn;
  let mockProcess;
  let mockCLI;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock spawn
    mockProcess = {
      on: jest.fn(),
      kill: jest.fn()
    };
    mockSpawn = spawn;
    mockSpawn.mockReturnValue(mockProcess);

    // Get mocked CLI
    const { createCLI } = require('../../src/onboarding/interactive-cli.js');
    mockCLI = createCLI();
  });

  describe('launchClaudeCode', () => {
    it('should launch Claude Code successfully', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code',
        version: '1.2.3'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({ valid: true });

      const result = await launchClaudeCode();

      expect(result.success).toBe(true);
      expect(mockCLI.succeedSpinner).toHaveBeenCalledWith('Claude Code found');
      expect(mockCLI.succeedSpinner).toHaveBeenCalledWith('MCP servers configured');
      expect(mockCLI.succeedSpinner).toHaveBeenCalledWith('Claude Code launched successfully');
      
      expect(mockSpawn).toHaveBeenCalledWith(
        '/usr/local/bin/claude-code',
        ['--dangerously-skip-permissions', '--mcp-config', expect.stringContaining('mcp.json')],
        {
          stdio: 'inherit',
          shell: false,
          detached: false
        }
      );
    });

    it('should fail when Claude Code not installed', async () => {
      detectClaudeCode.mockResolvedValue({ installed: false });

      const result = await launchClaudeCode();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Claude Code is not installed');
      expect(mockCLI.failSpinner).toHaveBeenCalledWith('Claude Code not found');
    });

    it('should prompt for init when MCP config missing', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(false);
      mockCLI.confirm.mockResolvedValue(true);

      const result = await launchClaudeCode();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Please run "ruv-swarm init"');
      expect(mockCLI.warning).toHaveBeenCalledWith('MCP configuration not found');
    });

    it('should launch without MCP when user declines init', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(false);
      mockCLI.confirm.mockResolvedValue(false);

      const result = await launchClaudeCode();

      expect(result.success).toBe(true);
      expect(mockCLI.info).toHaveBeenCalledWith('Launching Claude Code without MCP servers...');
      
      expect(mockSpawn).toHaveBeenCalledWith(
        '/usr/local/bin/claude-code',
        ['--dangerously-skip-permissions'],
        expect.any(Object)
      );
    });

    it('should fail on invalid MCP configuration', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({
        valid: false,
        error: 'Invalid JSON'
      });

      const result = await launchClaudeCode();

      expect(result.success).toBe(false);
      expect(result.error).toBe('Invalid JSON');
      expect(mockCLI.failSpinner).toHaveBeenCalledWith('Invalid MCP configuration');
    });

    it('should handle launch errors', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({ valid: true });
      
      const error = new Error('Spawn error');
      mockSpawn.mockImplementation(() => {
        throw error;
      });

      const result = await launchClaudeCode();

      expect(result.success).toBe(false);
      expect(result.error).toBe('Spawn error');
      expect(mockCLI.failSpinner).toHaveBeenCalledWith('Failed to launch Claude Code');
    });

    it('should add custom arguments', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({ valid: true });

      await launchClaudeCode({ args: ['--debug', '--verbose'] });

      expect(mockSpawn).toHaveBeenCalledWith(
        '/usr/local/bin/claude-code',
        [
          '--dangerously-skip-permissions',
          '--mcp-config',
          expect.any(String),
          '--debug',
          '--verbose'
        ],
        expect.any(Object)
      );
    });
  });

  describe('SessionManager', () => {
    let sessionManager;

    beforeEach(() => {
      sessionManager = new SessionManager();
    });

    it('should create a new session', () => {
      const config = { autoAccept: true };
      sessionManager.createSession('test-123', config);

      const session = sessionManager.getSession('test-123');
      expect(session).toBeTruthy();
      expect(session.id).toBe('test-123');
      expect(session.config).toEqual(config);
      expect(session.status).toBe('active');
      expect(session.startTime).toBeGreaterThan(0);
    });

    it('should update session status', () => {
      sessionManager.createSession('test-456', {});
      sessionManager.updateStatus('test-456', 'failed');

      const session = sessionManager.getSession('test-456');
      expect(session.status).toBe('failed');
      expect(session.lastUpdate).toBeGreaterThan(0);
    });

    it('should end session', () => {
      sessionManager.createSession('test-789', {});
      sessionManager.endSession('test-789');

      const session = sessionManager.getSession('test-789');
      expect(session.status).toBe('ended');
      expect(session.endTime).toBeGreaterThan(0);
    });

    it('should get active sessions', () => {
      sessionManager.createSession('active-1', {});
      sessionManager.createSession('active-2', {});
      sessionManager.createSession('ended-1', {});
      sessionManager.endSession('ended-1');

      const activeSessions = sessionManager.getActiveSessions();
      expect(activeSessions).toHaveLength(2);
      expect(activeSessions[0].id).toBe('active-1');
      expect(activeSessions[1].id).toBe('active-2');
    });

    it('should return null for non-existent session', () => {
      expect(sessionManager.getSession('non-existent')).toBeNull();
    });
  });

  describe('launchWithSession', () => {
    it('should launch with session management', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({ valid: true });

      const result = await launchWithSession({ verbose: true });

      expect(result.success).toBe(true);
      expect(result.sessionId).toMatch(/^claude-\d+$/);
    });

    it('should handle launch failure in session', async () => {
      detectClaudeCode.mockResolvedValue({ installed: false });

      const result = await launchWithSession();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Claude Code is not installed');
      expect(result.sessionId).toBeUndefined();
    });
  });

  describe('Process event handlers', () => {
    it('should handle process errors', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({ valid: true });

      await launchClaudeCode();

      // Get the error handler
      const errorHandler = mockProcess.on.mock.calls.find(
        call => call[0] === 'error'
      )[1];

      // Simulate error
      errorHandler(new Error('Process error'));

      expect(mockCLI.error).toHaveBeenCalledWith('Failed to launch Claude Code: Process error');
    });

    it('should handle process exit', async () => {
      detectClaudeCode.mockResolvedValue({
        installed: true,
        path: '/usr/local/bin/claude-code'
      });

      existsSync.mockReturnValue(true);
      validateMCPConfig.mockResolvedValue({ valid: true });

      await launchClaudeCode();

      // Get the exit handler
      const exitHandler = mockProcess.on.mock.calls.find(
        call => call[0] === 'exit'
      )[1];

      // Simulate exit with code
      exitHandler(1, null);
      expect(mockCLI.warning).toHaveBeenCalledWith('Claude Code exited with code 1');

      // Simulate exit with signal
      exitHandler(null, 'SIGTERM');
      expect(mockCLI.info).toHaveBeenCalledWith('Claude Code terminated by signal SIGTERM');
    });
  });
});