/**
 * Tests for error handling and recovery mechanisms
 * 
 * This test suite verifies robust error handling and recovery
 * strategies during the onboarding process.
 */

const { ErrorRecovery, OnboardingError } = require('../../src/onboarding/error-recovery');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

describe('ErrorRecovery', () => {
  let recovery;
  let tempDir;
  let logFile;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'error-recovery-test-'));
    logFile = path.join(tempDir, 'install.log');
    recovery = new ErrorRecovery({ logFile, tempDir });
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  describe('error classification', () => {
    test('should classify network errors', () => {
      const errors = [
        new Error('ENOTFOUND registry.npmjs.org'),
        new Error('ETIMEDOUT'),
        new Error('ECONNREFUSED'),
        new Error('getaddrinfo ENOTFOUND'),
      ];

      errors.forEach(error => {
        const classified = recovery.classifyError(error);
        expect(classified.type).toBe('network');
        expect(classified.recoverable).toBe(true);
      });
    });

    test('should classify permission errors', () => {
      const errors = [
        new Error('EACCES: permission denied'),
        new Error('EPERM: operation not permitted'),
        new Error('npm ERR! Error: EACCES'),
      ];

      errors.forEach(error => {
        const classified = recovery.classifyError(error);
        expect(classified.type).toBe('permission');
        expect(classified.recoverable).toBe(true);
      });
    });

    test('should classify disk space errors', () => {
      const errors = [
        new Error('ENOSPC: no space left on device'),
        new Error('npm ERR! code ENOSPC'),
      ];

      errors.forEach(error => {
        const classified = recovery.classifyError(error);
        expect(classified.type).toBe('disk_space');
        expect(classified.recoverable).toBe(false);
      });
    });

    test('should classify dependency errors', () => {
      const errors = [
        new Error('peer dep missing'),
        new Error('npm ERR! code ERESOLVE'),
        new Error('Cannot resolve dependency'),
      ];

      errors.forEach(error => {
        const classified = recovery.classifyError(error);
        expect(classified.type).toBe('dependency');
        expect(classified.recoverable).toBe(true);
      });
    });
  });

  describe('recovery strategies', () => {
    test('should suggest network error recovery', () => {
      const error = new OnboardingError('network', 'Connection timeout');
      const strategy = recovery.getRecoveryStrategy(error);

      expect(strategy.actions).toContain('retry');
      expect(strategy.actions).toContain('check_network');
      expect(strategy.actions).toContain('use_proxy');
      expect(strategy.message).toContain('network connection');
    });

    test('should suggest permission error recovery', () => {
      const error = new OnboardingError('permission', 'EACCES');
      const strategy = recovery.getRecoveryStrategy(error);

      expect(strategy.actions).toContain('use_sudo');
      expect(strategy.actions).toContain('change_npm_prefix');
      expect(strategy.message).toContain('permission');
      
      if (process.platform !== 'win32') {
        expect(strategy.commands).toContain('sudo npm install -g ruv-swarm');
      }
    });

    test('should suggest Claude not found recovery', () => {
      const error = new OnboardingError('claude_not_found', 'Claude Code not detected');
      const strategy = recovery.getRecoveryStrategy(error);

      expect(strategy.actions).toContain('install_claude');
      expect(strategy.actions).toContain('manual_path');
      expect(strategy.message).toContain('Claude Code');
      expect(strategy.urls).toContain('https://claude.ai/download');
    });
  });

  describe('retry mechanism', () => {
    test('should retry with exponential backoff', async () => {
      let attempts = 0;
      const operation = jest.fn().mockImplementation(() => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Network error');
        }
        return 'success';
      });

      const result = await recovery.retryWithBackoff(operation, {
        maxRetries: 3,
        initialDelay: 10,
      });

      expect(result).toBe('success');
      expect(operation).toHaveBeenCalledTimes(3);
    });

    test('should respect max retries', async () => {
      const operation = jest.fn().mockRejectedValue(new Error('Persistent error'));

      await expect(
        recovery.retryWithBackoff(operation, { maxRetries: 2 })
      ).rejects.toThrow('Persistent error');

      expect(operation).toHaveBeenCalledTimes(3); // Initial + 2 retries
    });

    test('should log retry attempts', async () => {
      const operation = jest.fn()
        .mockRejectedValueOnce(new Error('Attempt 1'))
        .mockRejectedValueOnce(new Error('Attempt 2'))
        .mockResolvedValue('success');

      await recovery.retryWithBackoff(operation);

      const log = await fs.readFile(logFile, 'utf-8');
      expect(log).toContain('Retry attempt 1');
      expect(log).toContain('Retry attempt 2');
    });
  });

  describe('rollback functionality', () => {
    test('should track and rollback file changes', async () => {
      // Create files
      const file1 = path.join(tempDir, 'file1.txt');
      const file2 = path.join(tempDir, 'file2.txt');
      
      await fs.writeFile(file1, 'content1');
      await fs.writeFile(file2, 'content2');
      
      // Track changes
      recovery.trackChange('create', file1);
      recovery.trackChange('create', file2);
      
      // Rollback
      await recovery.rollback();
      
      // Files should be deleted
      await expect(fs.access(file1)).rejects.toThrow();
      await expect(fs.access(file2)).rejects.toThrow();
    });

    test('should restore modified files', async () => {
      const file = path.join(tempDir, 'config.json');
      const originalContent = '{"original": true}';
      const modifiedContent = '{"modified": true}';
      
      // Create original file
      await fs.writeFile(file, originalContent);
      
      // Track modification
      recovery.trackChange('modify', file, originalContent);
      
      // Modify file
      await fs.writeFile(file, modifiedContent);
      
      // Rollback
      await recovery.rollback();
      
      // Should restore original content
      const content = await fs.readFile(file, 'utf-8');
      expect(content).toBe(originalContent);
    });

    test('should handle rollback errors gracefully', async () => {
      const nonExistentFile = path.join(tempDir, 'does-not-exist.txt');
      
      recovery.trackChange('create', nonExistentFile);
      
      // Should not throw
      await expect(recovery.rollback()).resolves.not.toThrow();
    });
  });

  describe('error logging', () => {
    test('should log errors with context', async () => {
      const error = new Error('Test error');
      const context = {
        step: 'npm_install',
        command: 'npm install -g ruv-swarm',
        timestamp: Date.now(),
      };

      await recovery.logError(error, context);

      const log = await fs.readFile(logFile, 'utf-8');
      expect(log).toContain('Test error');
      expect(log).toContain('npm_install');
      expect(log).toContain('npm install -g ruv-swarm');
    });

    test('should create error report', async () => {
      const errors = [
        { error: new Error('Error 1'), step: 'detect_claude' },
        { error: new Error('Error 2'), step: 'install_npm' },
        { error: new Error('Error 3'), step: 'configure_mcp' },
      ];

      for (const { error, step } of errors) {
        await recovery.logError(error, { step });
      }

      const report = await recovery.generateErrorReport();

      expect(report.totalErrors).toBe(3);
      expect(report.errorsByStep).toHaveProperty('detect_claude');
      expect(report.errorsByStep).toHaveProperty('install_npm');
      expect(report.recommendations).toBeInstanceOf(Array);
    });
  });

  describe('automatic fixes', () => {
    test('should fix npm prefix for permission errors', async () => {
      const homeDir = os.homedir();
      const npmPrefix = path.join(homeDir, '.npm-global');
      
      const fixed = await recovery.fixNpmPrefix();

      expect(fixed).toBe(true);
      
      // Check environment or config was updated
      const rcFile = path.join(homeDir, '.npmrc');
      if (await fs.access(rcFile).then(() => true).catch(() => false)) {
        const content = await fs.readFile(rcFile, 'utf-8');
        expect(content).toContain('prefix=');
      }
    });

    test('should clean npm cache for corruption errors', async () => {
      const execSpy = jest.spyOn(require('child_process'), 'exec')
        .mockImplementation((cmd, callback) => {
          callback(null, 'Cache cleaned', '');
        });

      const cleaned = await recovery.cleanNpmCache();

      expect(cleaned).toBe(true);
      expect(execSpy).toHaveBeenCalledWith(
        expect.stringContaining('npm cache clean --force'),
        expect.any(Function)
      );

      execSpy.mockRestore();
    });
  });

  describe('state persistence', () => {
    test('should save recovery state', async () => {
      recovery.setState({
        lastError: 'Network timeout',
        retryCount: 2,
        completedSteps: ['detect_claude', 'install_npm'],
        failedStep: 'configure_mcp',
      });

      await recovery.saveState();

      const stateFile = path.join(tempDir, '.recovery-state.json');
      const state = JSON.parse(await fs.readFile(stateFile, 'utf-8'));

      expect(state.lastError).toBe('Network timeout');
      expect(state.retryCount).toBe(2);
      expect(state.completedSteps).toEqual(['detect_claude', 'install_npm']);
    });

    test('should restore recovery state', async () => {
      const state = {
        lastError: 'Permission denied',
        retryCount: 1,
        completedSteps: ['detect_claude'],
        failedStep: 'install_npm',
      };

      const stateFile = path.join(tempDir, '.recovery-state.json');
      await fs.writeFile(stateFile, JSON.stringify(state));

      const restored = await recovery.loadState();

      expect(restored).toEqual(state);
    });
  });

  describe('user-friendly messages', () => {
    test('should provide clear error messages', () => {
      const testCases = [
        {
          error: new Error('EACCES: permission denied, access \'/usr/local/lib\''),
          expected: 'Permission denied when accessing /usr/local/lib',
        },
        {
          error: new Error('npm ERR! code E404\\nnpm ERR! 404 Not Found'),
          expected: 'Package not found in npm registry',
        },
        {
          error: new Error('getaddrinfo ENOTFOUND registry.npmjs.org'),
          expected: 'Unable to connect to registry.npmjs.org',
        },
      ];

      testCases.forEach(({ error, expected }) => {
        const message = recovery.getUserFriendlyMessage(error);
        expect(message).toContain(expected);
      });
    });
  });

  describe('recovery confirmation', () => {
    test('should track successful recoveries', async () => {
      const error = new OnboardingError('network', 'Connection failed');
      
      // Simulate recovery
      await recovery.attemptRecovery(error, async () => {
        // Recovery action succeeds
        return true;
      });

      const stats = recovery.getRecoveryStats();
      expect(stats.successfulRecoveries).toBe(1);
      expect(stats.recoveryMethods.network).toBe(1);
    });

    test('should track failed recoveries', async () => {
      const error = new OnboardingError('permission', 'Access denied');
      
      // Simulate failed recovery
      await recovery.attemptRecovery(error, async () => {
        throw new Error('Recovery failed');
      });

      const stats = recovery.getRecoveryStats();
      expect(stats.failedRecoveries).toBe(1);
    });
  });
});