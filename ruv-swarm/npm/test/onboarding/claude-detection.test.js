/**
 * Tests for Claude Code detection from NPM package
 * 
 * This test suite verifies the ability to detect Claude Code installation
 * from the JavaScript/Node.js environment.
 */

const { ClaudeDetector } = require('../../src/onboarding/claude-detector');
const { exec } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// Mock child_process
jest.mock('child_process');

describe('ClaudeDetector', () => {
  let detector;
  let tempDir;

  beforeEach(async () => {
    detector = new ClaudeDetector();
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'claude-test-'));
    
    // Reset mocks
    jest.clearAllMocks();
  });

  afterEach(async () => {
    // Cleanup temp directory
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  describe('detect()', () => {
    test('should detect Claude in PATH', async () => {
      // Mock successful claude command
      exec.mockImplementation((cmd, callback) => {
        if (cmd === 'claude --version') {
          callback(null, 'Claude Code v1.0.0\n', '');
        } else if (cmd === 'which claude' || cmd === 'where claude') {
          callback(null, '/usr/local/bin/claude\n', '');
        }
      });

      const result = await detector.detect();

      expect(result.found).toBe(true);
      expect(result.version).toBe('1.0.0');
      expect(result.path).toBe('/usr/local/bin/claude');
    });

    test('should handle Claude not found', async () => {
      // Mock command not found
      exec.mockImplementation((cmd, callback) => {
        callback(new Error('command not found'), '', 'claude: command not found');
      });

      const result = await detector.detect();

      expect(result.found).toBe(false);
      expect(result.version).toBeNull();
      expect(result.path).toBeNull();
    });

    test('should detect Claude in common locations', async () => {
      const commonPaths = ClaudeDetector.getCommonPaths();
      
      // Mock Claude not in PATH but in common location
      exec.mockImplementation((cmd, callback) => {
        if (cmd === 'claude --version') {
          callback(new Error('command not found'), '', '');
        } else if (cmd.includes('/Applications/Claude.app')) {
          callback(null, 'Claude Code v2.0.0\n', '');
        }
      });

      // Mock file existence check
      const originalAccess = fs.access;
      fs.access = jest.fn().mockImplementation(async (filePath) => {
        if (filePath.includes('/Applications/Claude.app')) {
          return Promise.resolve();
        }
        throw new Error('ENOENT');
      });

      const result = await detector.detect();

      expect(result.found).toBe(true);
      expect(result.version).toBe('2.0.0');
      
      fs.access = originalAccess;
    });

    test('should parse various version formats', async () => {
      const versionFormats = [
        { output: 'Claude Code v1.2.3', expected: '1.2.3' },
        { output: 'claude version 2.0.0-beta', expected: '2.0.0-beta' },
        { output: 'Version: 3.1.0\nBuild: 12345', expected: '3.1.0' },
        { output: 'claude-code 0.5.0', expected: '0.5.0' },
      ];

      for (const { output, expected } of versionFormats) {
        exec.mockImplementation((cmd, callback) => {
          if (cmd === 'claude --version') {
            callback(null, output, '');
          }
        });

        const result = await detector.detect();
        expect(result.version).toBe(expected);
      }
    });

    test('should handle permission errors', async () => {
      exec.mockImplementation((cmd, callback) => {
        const error = new Error('EACCES: permission denied');
        error.code = 'EACCES';
        callback(error, '', 'Permission denied');
      });

      const result = await detector.detect();

      expect(result.found).toBe(false);
      expect(result.error).toContain('permission');
    });
  });

  describe('detectAll()', () => {
    test('should find multiple Claude installations', async () => {
      const installations = [
        { path: '/usr/local/bin/claude', version: '1.0.0' },
        { path: '/opt/claude/bin/claude', version: '1.1.0' },
        { path: `${os.homedir()}/.local/bin/claude`, version: '2.0.0' },
      ];

      let callCount = 0;
      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('--version')) {
          const installation = installations[callCount % installations.length];
          callback(null, `Claude Code v${installation.version}`, '');
          callCount++;
        }
      });

      // Mock file checks
      const originalAccess = fs.access;
      fs.access = jest.fn().mockResolvedValue();

      const results = await detector.detectAll();

      expect(results.length).toBeGreaterThan(0);
      expect(results.every(r => r.found)).toBe(true);
      
      fs.access = originalAccess;
    });
  });

  describe('platform-specific detection', () => {
    test('should check Windows-specific paths', async () => {
      const originalPlatform = process.platform;
      Object.defineProperty(process, 'platform', {
        value: 'win32',
        configurable: true,
      });

      const paths = ClaudeDetector.getCommonPaths();
      
      expect(paths.some(p => p.includes('Program Files'))).toBe(true);
      expect(paths.some(p => p.includes('AppData'))).toBe(true);

      Object.defineProperty(process, 'platform', {
        value: originalPlatform,
        configurable: true,
      });
    });

    test('should check macOS-specific paths', async () => {
      const originalPlatform = process.platform;
      Object.defineProperty(process, 'platform', {
        value: 'darwin',
        configurable: true,
      });

      const paths = ClaudeDetector.getCommonPaths();
      
      expect(paths.some(p => p.includes('/Applications'))).toBe(true);
      expect(paths.some(p => p.includes('Claude.app'))).toBe(true);

      Object.defineProperty(process, 'platform', {
        value: originalPlatform,
        configurable: true,
      });
    });

    test('should check Linux-specific paths', async () => {
      const originalPlatform = process.platform;
      Object.defineProperty(process, 'platform', {
        value: 'linux',
        configurable: true,
      });

      const paths = ClaudeDetector.getCommonPaths();
      
      expect(paths.some(p => p.includes('/usr/local/bin'))).toBe(true);
      expect(paths.some(p => p.includes('.local/bin'))).toBe(true);
      expect(paths.some(p => p.includes('/opt'))).toBe(true);

      Object.defineProperty(process, 'platform', {
        value: originalPlatform,
        configurable: true,
      });
    });
  });

  describe('development mode detection', () => {
    test('should detect Claude in development mode', async () => {
      process.env.CLAUDE_DEV_MODE = '1';
      process.env.CLAUDE_DEV_PATH = path.join(tempDir, 'claude-dev');

      // Create mock dev claude
      await fs.mkdir(path.join(tempDir, 'claude-dev'), { recursive: true });
      await fs.writeFile(
        path.join(tempDir, 'claude-dev', 'claude'),
        '#!/bin/sh\necho "Claude Code dev"'
      );

      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('claude-dev')) {
          callback(null, 'Claude Code v0.0.1-dev', '');
        }
      });

      const result = await detector.detect();

      expect(result.found).toBe(true);
      expect(result.isDevelopment).toBe(true);
      expect(result.version).toContain('dev');

      delete process.env.CLAUDE_DEV_MODE;
      delete process.env.CLAUDE_DEV_PATH;
    });
  });

  describe('caching', () => {
    test('should cache detection results', async () => {
      exec.mockImplementation((cmd, callback) => {
        callback(null, 'Claude Code v1.0.0', '');
      });

      // First call
      const result1 = await detector.detect();
      expect(exec).toHaveBeenCalledTimes(2); // version + which

      // Second call should use cache
      const result2 = await detector.detect();
      expect(exec).toHaveBeenCalledTimes(2); // No additional calls

      expect(result1).toEqual(result2);
    });

    test('should invalidate cache after timeout', async () => {
      exec.mockImplementation((cmd, callback) => {
        callback(null, 'Claude Code v1.0.0', '');
      });

      detector.cacheTimeout = 100; // 100ms timeout

      await detector.detect();
      const initialCallCount = exec.mock.calls.length;

      // Wait for cache to expire
      await new Promise(resolve => setTimeout(resolve, 150));

      await detector.detect();
      expect(exec.mock.calls.length).toBeGreaterThan(initialCallCount);
    });
  });

  describe('error handling', () => {
    test('should handle ENOENT errors gracefully', async () => {
      const error = new Error('ENOENT');
      error.code = 'ENOENT';
      exec.mockImplementation((cmd, callback) => {
        callback(error, '', '');
      });

      const result = await detector.detect();

      expect(result.found).toBe(false);
      expect(result.error).toBeNull(); // ENOENT is expected, not an error
    });

    test('should handle unexpected errors', async () => {
      exec.mockImplementation((cmd, callback) => {
        callback(new Error('Unexpected error'), '', '');
      });

      const result = await detector.detect();

      expect(result.found).toBe(false);
      expect(result.error).toContain('Unexpected error');
    });
  });

  describe('validation', () => {
    test('should validate Claude executable', async () => {
      exec.mockImplementation((cmd, callback) => {
        if (cmd === 'claude --version') {
          callback(null, 'Not Claude', '');
        } else {
          callback(null, '/usr/bin/claude', '');
        }
      });

      const result = await detector.detect();

      expect(result.found).toBe(false);
      expect(result.error).toContain('Invalid Claude installation');
    });
  });
});