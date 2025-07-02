/**
 * End-to-end integration tests for the onboarding process
 * 
 * This test suite verifies the complete onboarding flow
 * from start to finish in various scenarios.
 */

const { OnboardingOrchestrator } = require('../../src/onboarding');
const { exec } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// Mock child_process for controlled testing
jest.mock('child_process');

describe('Onboarding Integration Tests', () => {
  let orchestrator;
  let tempDir;
  let originalEnv;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'onboarding-integration-'));
    originalEnv = { ...process.env };
    
    // Set test environment
    process.env.HOME = tempDir;
    process.env.RUVSW_TEST_MODE = 'true';
    
    orchestrator = new OnboardingOrchestrator({
      homeDir: tempDir,
      testMode: true,
    });

    // Default mock implementations
    exec.mockImplementation((cmd, callback) => {
      if (cmd.includes('claude --version')) {
        callback(null, 'Claude Code v1.0.0', '');
      } else if (cmd.includes('npm install')) {
        callback(null, 'added 1 package', '');
      } else if (cmd.includes('which claude') || cmd.includes('where claude')) {
        callback(null, '/usr/local/bin/claude', '');
      } else {
        callback(null, '', '');
      }
    });
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
    process.env = originalEnv;
    jest.clearAllMocks();
  });

  describe('Happy Path Scenarios', () => {
    test('should complete full installation successfully', async () => {
      const result = await orchestrator.run({
        mode: 'interactive',
        installType: 'full',
      });

      expect(result.success).toBe(true);
      expect(result.installedComponents).toEqual({
        claude: true,
        npm: true,
        mcp: true,
        hooks: true,
        path: true,
      });

      // Verify files were created
      const configDir = path.join(tempDir, '.config', 'ruv-swarm');
      expect(await fs.access(configDir).then(() => true).catch(() => false)).toBe(true);
      
      const mcpConfig = path.join(tempDir, '.claude', 'mcp.json');
      expect(await fs.access(mcpConfig).then(() => true).catch(() => false)).toBe(true);
    });

    test('should handle auto-accept mode', async () => {
      const result = await orchestrator.run({
        mode: 'auto',
        autoAccept: true,
      });

      expect(result.success).toBe(true);
      expect(result.mode).toBe('auto');
      expect(result.userInteraction).toBe(false);
    });

    test('should support custom installation', async () => {
      const result = await orchestrator.run({
        mode: 'interactive',
        installType: 'custom',
        components: {
          npm: false,
          mcp: true,
          hooks: true,
          path: false,
        },
      });

      expect(result.success).toBe(true);
      expect(result.installedComponents.npm).toBe(false);
      expect(result.installedComponents.mcp).toBe(true);
      expect(result.installedComponents.hooks).toBe(true);
      expect(result.installedComponents.path).toBe(false);
    });
  });

  describe('Error Scenarios', () => {
    test('should handle Claude not found', async () => {
      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('claude')) {
          callback(new Error('command not found'), '', 'claude: command not found');
        } else {
          callback(null, '', '');
        }
      });

      const result = await orchestrator.run({ mode: 'auto' });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Claude Code not found');
      expect(result.suggestions).toContain('https://claude.ai/download');
    });

    test('should recover from npm installation failure', async () => {
      let npmAttempts = 0;
      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('npm install')) {
          npmAttempts++;
          if (npmAttempts === 1) {
            callback(new Error('EACCES'), '', 'permission denied');
          } else {
            callback(null, 'added 1 package', '');
          }
        } else if (cmd.includes('claude')) {
          callback(null, 'Claude Code v1.0.0', '');
        } else {
          callback(null, '', '');
        }
      });

      const result = await orchestrator.run({
        mode: 'auto',
        enableRecovery: true,
      });

      expect(result.success).toBe(true);
      expect(result.recoveryAttempts).toBeGreaterThan(0);
      expect(npmAttempts).toBe(2);
    });

    test('should handle network errors gracefully', async () => {
      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('npm install')) {
          callback(new Error('ETIMEDOUT'), '', 'network timeout');
        } else if (cmd.includes('claude')) {
          callback(null, 'Claude Code v1.0.0', '');
        } else {
          callback(null, '', '');
        }
      });

      const result = await orchestrator.run({
        mode: 'auto',
        skipOnError: ['npm'],
      });

      expect(result.success).toBe(true);
      expect(result.warnings).toContain('NPM installation failed but continuing');
      expect(result.installedComponents.npm).toBe(false);
      expect(result.installedComponents.mcp).toBe(true);
    });
  });

  describe('Platform-Specific Tests', () => {
    test('should handle Windows paths correctly', async () => {
      Object.defineProperty(process, 'platform', {
        value: 'win32',
        configurable: true,
      });

      const result = await orchestrator.run({ mode: 'auto' });

      if (result.success) {
        const mcpConfig = path.join(tempDir, '.claude', 'mcp.json');
        const config = JSON.parse(await fs.readFile(mcpConfig, 'utf-8'));
        
        // Windows should use npx or full paths
        expect(config.servers['ruv-swarm'].command).toMatch(/npx|\.exe$/);
      }

      Object.defineProperty(process, 'platform', {
        value: originalEnv.platform || 'linux',
        configurable: true,
      });
    });

    test('should handle macOS specific locations', async () => {
      Object.defineProperty(process, 'platform', {
        value: 'darwin',
        configurable: true,
      });

      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('claude')) {
          callback(null, 'Claude Code v1.0.0', '');
        } else if (cmd.includes('/Applications/Claude.app')) {
          callback(null, 'found', '');
        } else {
          callback(null, '', '');
        }
      });

      const result = await orchestrator.run({ mode: 'auto' });

      expect(result.success).toBe(true);
      expect(result.platform).toBe('darwin');

      Object.defineProperty(process, 'platform', {
        value: originalEnv.platform || 'linux',
        configurable: true,
      });
    });
  });

  describe('State Management', () => {
    test('should resume from previous installation', async () => {
      // Create partial installation state
      const stateFile = path.join(tempDir, '.ruv-swarm-install-state.json');
      await fs.writeFile(stateFile, JSON.stringify({
        version: '0.2.0',
        timestamp: Date.now() - 3600000, // 1 hour ago
        completed: ['claude_detection', 'npm_installation'],
        remaining: ['mcp_configuration', 'hooks_installation', 'path_configuration'],
        lastError: null,
      }));

      const result = await orchestrator.run({
        mode: 'auto',
        resume: true,
      });

      expect(result.success).toBe(true);
      expect(result.resumed).toBe(true);
      expect(result.skippedSteps).toEqual(['claude_detection', 'npm_installation']);
    });

    test('should not resume stale installations', async () => {
      const stateFile = path.join(tempDir, '.ruv-swarm-install-state.json');
      await fs.writeFile(stateFile, JSON.stringify({
        version: '0.1.0', // Old version
        timestamp: Date.now() - 86400000 * 7, // 7 days ago
        completed: ['claude_detection'],
      }));

      const result = await orchestrator.run({
        mode: 'auto',
        resume: true,
      });

      expect(result.resumed).toBe(false);
      expect(result.reason).toContain('State too old or version mismatch');
    });
  });

  describe('Verification Tests', () => {
    test('should verify installation completeness', async () => {
      await orchestrator.run({ mode: 'auto' });
      
      const verification = await orchestrator.verify();

      expect(verification.claude.installed).toBe(true);
      expect(verification.npm.installed).toBe(true);
      expect(verification.mcp.configured).toBe(true);
      expect(verification.hooks.installed).toBe(true);
      expect(verification.overall).toBe('complete');
    });

    test('should detect partial installations', async () => {
      // Simulate partial installation
      const configDir = path.join(tempDir, '.config', 'ruv-swarm');
      await fs.mkdir(configDir, { recursive: true });
      await fs.writeFile(path.join(configDir, 'config.toml'), '[swarm]');
      
      const verification = await orchestrator.verify();

      expect(verification.overall).toBe('partial');
      expect(verification.missing).toContain('mcp');
    });
  });

  describe('Cleanup and Rollback', () => {
    test('should clean up on failure', async () => {
      exec.mockImplementation((cmd, callback) => {
        if (cmd.includes('npm install')) {
          // Fail after some files are created
          setTimeout(() => {
            callback(new Error('Critical error'), '', '');
          }, 10);
        } else {
          callback(null, '', '');
        }
      });

      // Create some files that should be cleaned up
      const configDir = path.join(tempDir, '.config', 'ruv-swarm');
      await fs.mkdir(configDir, { recursive: true });
      await fs.writeFile(path.join(configDir, 'temp.txt'), 'temporary');

      const result = await orchestrator.run({
        mode: 'auto',
        cleanupOnError: true,
      });

      expect(result.success).toBe(false);
      
      // Temporary files should be cleaned up
      const tempExists = await fs.access(path.join(configDir, 'temp.txt'))
        .then(() => true)
        .catch(() => false);
      expect(tempExists).toBe(false);
    });
  });

  describe('Launch Command Generation', () => {
    test('should generate correct launch command', async () => {
      const result = await orchestrator.run({ mode: 'auto' });

      expect(result.launchCommand).toBeDefined();
      expect(result.launchCommand).toContain('claude');
      expect(result.launchCommand).toContain('--mcp');
      
      if (result.mcpConfigPath) {
        expect(result.launchCommand).toContain(result.mcpConfigPath);
      }
    });

    test('should provide platform-specific launch instructions', async () => {
      const result = await orchestrator.run({ mode: 'auto' });

      expect(result.launchInstructions).toBeDefined();
      
      if (process.platform === 'win32') {
        expect(result.launchInstructions).toContain('Command Prompt');
      } else {
        expect(result.launchInstructions).toContain('terminal');
      }
    });
  });

  describe('Telemetry and Analytics', () => {
    test('should collect installation metrics', async () => {
      const result = await orchestrator.run({
        mode: 'auto',
        collectMetrics: true,
      });

      expect(result.metrics).toBeDefined();
      expect(result.metrics.duration).toBeGreaterThan(0);
      expect(result.metrics.steps).toBeDefined();
      expect(result.metrics.errors).toBeDefined();
      expect(result.metrics.platform).toBe(process.platform);
    });
  });
});