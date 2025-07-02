/**
 * Tests for Claude Code Detection Module
 */

import { jest } from '@jest/globals';
import os from 'os';
import { detectClaudeCode, isVersionCompatible } from '../../src/onboarding/claude-detector.js';

// Mock child_process and fs
jest.mock('child_process');
jest.mock('fs');

describe('Claude Code Detector', () => {
  let mockExecSync;
  let mockExistsSync;
  let mockPlatform;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Import mocked modules
    const childProcess = await import('child_process');
    const fs = await import('fs');
    
    mockExecSync = childProcess.execSync;
    mockExistsSync = fs.existsSync;
    
    // Mock os.platform
    mockPlatform = jest.spyOn(os, 'platform');
  });

  afterEach(() => {
    mockPlatform.mockRestore();
  });

  describe('detectClaudeCode', () => {
    it('should detect Claude Code via which command on Unix', async () => {
      mockPlatform.mockReturnValue('linux');
      mockExecSync.mockImplementation((cmd) => {
        if (cmd === 'which claude-code') {
          return '/usr/local/bin/claude-code\n';
        }
        if (cmd === '"/usr/local/bin/claude-code" --version') {
          return 'Claude Code v1.2.3\n';
        }
      });

      const result = await detectClaudeCode();

      expect(result).toEqual({
        installed: true,
        path: '/usr/local/bin/claude-code',
        version: '1.2.3'
      });
    });

    it('should detect Claude Code via where command on Windows', async () => {
      mockPlatform.mockReturnValue('win32');
      mockExecSync.mockImplementation((cmd) => {
        if (cmd === 'where claude-code') {
          return 'C:\\Program Files\\Claude Code\\claude-code.exe\n';
        }
        if (cmd === '"C:\\Program Files\\Claude Code\\claude-code.exe" --version') {
          return 'Claude Code v2.0.1\n';
        }
      });

      const result = await detectClaudeCode();

      expect(result).toEqual({
        installed: true,
        path: 'C:\\Program Files\\Claude Code\\claude-code.exe',
        version: '2.0.1'
      });
    });

    it('should check common paths when command not in PATH', async () => {
      mockPlatform.mockReturnValue('darwin');
      mockExecSync.mockImplementation(() => {
        throw new Error('Command not found');
      });
      
      mockExistsSync.mockImplementation((path) => {
        return path === '/Applications/Claude Code.app/Contents/MacOS/claude-code';
      });

      const result = await detectClaudeCode();

      expect(result.installed).toBe(true);
      expect(result.path).toBe('/Applications/Claude Code.app/Contents/MacOS/claude-code');
    });

    it('should return not installed when Claude Code not found', async () => {
      mockPlatform.mockReturnValue('linux');
      mockExecSync.mockImplementation(() => {
        throw new Error('Command not found');
      });
      mockExistsSync.mockReturnValue(false);

      const result = await detectClaudeCode();

      expect(result).toEqual({ installed: false });
    });

    it('should handle version command failures gracefully', async () => {
      mockPlatform.mockReturnValue('linux');
      mockExecSync.mockImplementation((cmd) => {
        if (cmd === 'which claude-code') {
          return '/usr/local/bin/claude-code\n';
        }
        // Version command fails
        throw new Error('Version command failed');
      });

      const result = await detectClaudeCode();

      expect(result).toEqual({
        installed: true,
        path: '/usr/local/bin/claude-code',
        version: null
      });
    });
  });

  describe('isVersionCompatible', () => {
    it('should accept version 1.0.0 and above', () => {
      expect(isVersionCompatible('1.0.0')).toBe(true);
      expect(isVersionCompatible('1.5.2')).toBe(true);
      expect(isVersionCompatible('2.0.0')).toBe(true);
    });

    it('should reject versions below 1.0.0', () => {
      expect(isVersionCompatible('0.9.9')).toBe(false);
      expect(isVersionCompatible('0.1.0')).toBe(false);
    });

    it('should handle invalid versions', () => {
      expect(isVersionCompatible(null)).toBe(false);
      expect(isVersionCompatible('')).toBe(false);
      expect(isVersionCompatible('invalid')).toBe(false);
    });
  });
});