/**
 * Tests for auto-accept (-y) flag functionality
 * 
 * This test suite verifies that the -y flag bypasses all prompts
 * and uses sensible defaults for automated installation.
 */

const { AutoAcceptOnboarding } = require('../../src/onboarding/auto-accept');
const { ClaudeDetector } = require('../../src/onboarding/claude-detector');
const { McpSetup } = require('../../src/onboarding/mcp-setup');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// Mock dependencies
jest.mock('../../src/onboarding/claude-detector');
jest.mock('../../src/onboarding/mcp-setup');

describe('AutoAcceptOnboarding', () => {
  let onboarding;
  let tempDir;
  let consoleSpy;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'auto-accept-test-'));
    onboarding = new AutoAcceptOnboarding({ homeDir: tempDir });
    consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    
    // Setup default mocks
    ClaudeDetector.mockImplementation(() => ({
      detect: jest.fn().mockResolvedValue({ found: true, version: '1.0.0', path: '/usr/bin/claude' }),
    }));
    
    McpSetup.mockImplementation(() => ({
      generateConfig: jest.fn().mockResolvedValue({ servers: { 'ruv-swarm': {} } }),
      writeConfig: jest.fn().mockResolvedValue(),
      installToClaudeConfig: jest.fn().mockResolvedValue({ success: true }),
      generateLaunchCommands: jest.fn().mockReturnValue({ default: 'claude --mcp' }),
    }));
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
    consoleSpy.mockRestore();
    jest.clearAllMocks();
  });

  describe('run()', () => {
    test('should complete full installation without prompts', async () => {
      const result = await onboarding.run();

      expect(result.success).toBe(true);
      expect(result.steps).toEqual({
        claudeDetection: true,
        npmInstallation: true,
        mcpConfiguration: true,
        hooksInstallation: true,
        pathConfiguration: true,
      });
      
      // Verify no prompts were shown
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('?'));
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('(y/n)'));
    });

    test('should skip interactive elements', async () => {
      await onboarding.run();

      // Should not show welcome screen
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('Welcome to'));
      
      // Should show progress messages instead
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Auto-accept mode'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Detecting Claude Code'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Installing NPM package'));
    });

    test('should use default values for all options', async () => {
      const config = onboarding.getDefaultConfig();

      expect(config).toEqual({
        installNpm: true,
        installMcp: true,
        installHooks: true,
        updatePath: true,
        mcpPort: 0, // stdio by default
        mcpProtocol: 'stdio',
      });
    });
  });

  describe('error handling', () => {
    test('should continue on non-critical errors', async () => {
      // Mock hook installation failure
      onboarding.installHooks = jest.fn().mockRejectedValue(new Error('Hook install failed'));

      const result = await onboarding.run();

      expect(result.success).toBe(true);
      expect(result.steps.hooksInstallation).toBe(false);
      expect(result.warnings).toContain('Hook installation failed but continuing');
    });

    test('should fail on critical errors', async () => {
      ClaudeDetector.mockImplementation(() => ({
        detect: jest.fn().mockResolvedValue({ found: false }),
      }));

      const result = await onboarding.run();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Claude Code not found');
    });
  });

  describe('NPM installation', () => {
    test('should install globally without prompting', async () => {
      const execSpy = jest.spyOn(require('child_process'), 'exec')
        .mockImplementation((cmd, callback) => {
          callback(null, 'added 1 package', '');
        });

      await onboarding.installNpmPackage();

      expect(execSpy).toHaveBeenCalledWith(
        expect.stringContaining('npm install -g ruv-swarm'),
        expect.any(Function)
      );
      
      execSpy.mockRestore();
    });

    test('should handle npm permission errors with sudo', async () => {
      const execSpy = jest.spyOn(require('child_process'), 'exec')
        .mockImplementationOnce((cmd, callback) => {
          callback(new Error('EACCES'), '', 'permission denied');
        })
        .mockImplementationOnce((cmd, callback) => {
          callback(null, 'added 1 package', '');
        });

      await onboarding.installNpmPackage();

      expect(execSpy).toHaveBeenCalledTimes(2);
      expect(execSpy).toHaveBeenLastCalledWith(
        expect.stringContaining('sudo npm install -g ruv-swarm'),
        expect.any(Function)
      );
      
      execSpy.mockRestore();
    });
  });

  describe('MCP configuration', () => {
    test('should use stdio protocol by default', async () => {
      const mockSetup = new McpSetup();
      await onboarding.configureMcp();

      expect(mockSetup.generateConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          protocol: 'stdio',
        })
      );
    });

    test('should handle existing MCP config', async () => {
      McpSetup.mockImplementation(() => ({
        detectClaudeConfigPath: jest.fn().mockResolvedValue('/home/user/.claude/mcp.json'),
        mergeWithExisting: jest.fn().mockResolvedValue({ servers: {} }),
        writeConfig: jest.fn().mockResolvedValue(),
        generateLaunchCommands: jest.fn().mockReturnValue({ default: 'claude --mcp' }),
      }));

      await onboarding.configureMcp();

      const mockSetup = McpSetup.mock.instances[0];
      expect(mockSetup.mergeWithExisting).toHaveBeenCalled();
    });
  });

  describe('hooks installation', () => {
    test('should install all hooks without prompting', async () => {
      const hooksDir = path.join(tempDir, '.config', 'ruv-swarm', 'hooks');
      
      await onboarding.installHooks();

      const hooks = await fs.readdir(hooksDir);
      expect(hooks).toContain('pre-task.sh');
      expect(hooks).toContain('post-edit.sh');
      expect(hooks).toContain('session-end.sh');
    });

    test('should make hooks executable on Unix', async () => {
      if (process.platform === 'win32') {
        return; // Skip on Windows
      }

      await onboarding.installHooks();

      const hookPath = path.join(tempDir, '.config', 'ruv-swarm', 'hooks', 'pre-task.sh');
      const stats = await fs.stat(hookPath);
      
      // Check execute permission
      expect(stats.mode & 0o111).toBeTruthy();
    });
  });

  describe('PATH configuration', () => {
    test('should update shell RC file automatically', async () => {
      const rcFile = path.join(tempDir, '.bashrc');
      await fs.writeFile(rcFile, '# Existing bashrc\n');

      await onboarding.updatePath(rcFile);

      const content = await fs.readFile(rcFile, 'utf-8');
      expect(content).toContain('ruv-swarm');
      expect(content).toContain('PATH');
      expect(content).toContain('# Added by ruv-swarm auto-install');
    });

    test('should detect appropriate shell RC file', async () => {
      const rcFile = await onboarding.detectShellRcFile();

      if (process.platform === 'win32') {
        expect(rcFile).toBeNull(); // No RC files on Windows
      } else {
        expect(rcFile).toMatch(/\.(bashrc|zshrc|config\/fish\/config\.fish)$/);
      }
    });
  });

  describe('completion', () => {
    test('should show brief completion message', async () => {
      const result = await onboarding.run();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('✅ Installation complete'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('claude --mcp'));
    });

    test('should save installation summary', async () => {
      await onboarding.run();

      const summaryPath = path.join(tempDir, '.ruv-swarm-install.json');
      const exists = await fs.access(summaryPath).then(() => true).catch(() => false);
      
      expect(exists).toBe(true);
      
      const summary = JSON.parse(await fs.readFile(summaryPath, 'utf-8'));
      expect(summary).toHaveProperty('timestamp');
      expect(summary).toHaveProperty('version');
      expect(summary).toHaveProperty('steps');
    });
  });

  describe('CI environment detection', () => {
    test('should detect CI environment', () => {
      const originalCI = process.env.CI;
      
      process.env.CI = 'true';
      expect(onboarding.isCI()).toBe(true);
      
      delete process.env.CI;
      process.env.GITHUB_ACTIONS = 'true';
      expect(onboarding.isCI()).toBe(true);
      
      delete process.env.GITHUB_ACTIONS;
      process.env.JENKINS_URL = 'http://jenkins';
      expect(onboarding.isCI()).toBe(true);
      
      // Restore
      delete process.env.JENKINS_URL;
      if (originalCI) process.env.CI = originalCI;
    });

    test('should use minimal output in CI', async () => {
      process.env.CI = 'true';
      
      await onboarding.run();
      
      // Should use minimal output
      expect(consoleSpy).toHaveBeenCalledTimes(expect.any(Number));
      const calls = consoleSpy.mock.calls.map(c => c[0]);
      expect(calls.every(msg => !msg.includes('━') && !msg.includes('│'))).toBe(true);
      
      delete process.env.CI;
    });
  });

  describe('dry run mode', () => {
    test('should simulate installation without making changes', async () => {
      onboarding = new AutoAcceptOnboarding({ homeDir: tempDir, dryRun: true });
      
      const result = await onboarding.run();

      expect(result.success).toBe(true);
      expect(result.dryRun).toBe(true);
      
      // Verify no actual changes were made
      const configDir = path.join(tempDir, '.config', 'ruv-swarm');
      const exists = await fs.access(configDir).then(() => true).catch(() => false);
      expect(exists).toBe(false);
    });
  });
});