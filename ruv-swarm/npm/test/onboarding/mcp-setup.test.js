/**
 * Tests for MCP server configuration setup
 * 
 * This test suite verifies the MCP (Model Context Protocol) configuration
 * generation and integration with Claude Code.
 */

const { McpSetup } = require('../../src/onboarding/mcp-setup');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

describe('McpSetup', () => {
  let setup;
  let tempDir;
  let configPath;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mcp-test-'));
    configPath = path.join(tempDir, 'mcp.json');
    setup = new McpSetup(configPath);
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  describe('generateConfig()', () => {
    test('should generate basic MCP config', async () => {
      const config = await setup.generateConfig();

      expect(config).toHaveProperty('servers');
      expect(config.servers).toHaveProperty('ruv-swarm');
      expect(config.servers['ruv-swarm']).toEqual({
        command: 'npx',
        args: ['ruv-swarm', 'mcp', 'start'],
        stdio: true,
      });
    });

    test('should generate config with custom options', async () => {
      const options = {
        serverName: 'my-swarm',
        command: 'ruv-swarm',
        args: ['mcp', 'server', '--port', '8080'],
        env: {
          RUST_LOG: 'debug',
          NODE_ENV: 'production',
        },
        tcp: {
          host: 'localhost',
          port: 8080,
        },
      };

      const config = await setup.generateConfig(options);

      expect(config.servers['my-swarm']).toMatchObject({
        command: 'ruv-swarm',
        args: ['mcp', 'server', '--port', '8080'],
        env: {
          RUST_LOG: 'debug',
          NODE_ENV: 'production',
        },
        tcp: {
          host: 'localhost',
          port: 8080,
        },
      });
    });

    test('should include workspace-specific configuration', async () => {
      const workspaceDir = path.join(tempDir, 'my-project');
      await fs.mkdir(workspaceDir, { recursive: true });

      setup.setWorkspaceDir(workspaceDir);
      const config = await setup.generateConfig();

      expect(config.servers['ruv-swarm'].env).toHaveProperty('RUV_SWARM_WORKSPACE');
      expect(config.servers['ruv-swarm'].env.RUV_SWARM_WORKSPACE).toBe(workspaceDir);
    });
  });

  describe('writeConfig()', () => {
    test('should write config to file', async () => {
      const config = await setup.generateConfig();
      await setup.writeConfig(config);

      const exists = await fs.access(configPath).then(() => true).catch(() => false);
      expect(exists).toBe(true);

      const content = await fs.readFile(configPath, 'utf-8');
      const parsed = JSON.parse(content);
      
      expect(parsed).toEqual(config);
    });

    test('should create parent directories if needed', async () => {
      const nestedPath = path.join(tempDir, 'deep', 'nested', 'dir', 'mcp.json');
      setup = new McpSetup(nestedPath);

      const config = await setup.generateConfig();
      await setup.writeConfig(config);

      const exists = await fs.access(nestedPath).then(() => true).catch(() => false);
      expect(exists).toBe(true);
    });

    test('should backup existing config', async () => {
      // Write initial config
      const initialConfig = { servers: { old: { command: 'old' } } };
      await fs.writeFile(configPath, JSON.stringify(initialConfig));

      // Write new config
      const newConfig = await setup.generateConfig();
      await setup.writeConfig(newConfig);

      // Check backup exists
      const backupPath = configPath + '.backup';
      const backupExists = await fs.access(backupPath).then(() => true).catch(() => false);
      expect(backupExists).toBe(true);

      const backupContent = await fs.readFile(backupPath, 'utf-8');
      expect(JSON.parse(backupContent)).toEqual(initialConfig);
    });
  });

  describe('mergeWithExisting()', () => {
    test('should merge with existing config', async () => {
      const existingConfig = {
        servers: {
          'other-server': {
            command: 'other',
            args: ['start'],
          },
        },
      };

      await fs.writeFile(configPath, JSON.stringify(existingConfig));

      const newConfig = await setup.generateConfig();
      const merged = await setup.mergeWithExisting(newConfig);

      expect(merged.servers).toHaveProperty('other-server');
      expect(merged.servers).toHaveProperty('ruv-swarm');
    });

    test('should handle malformed existing config', async () => {
      await fs.writeFile(configPath, '{ invalid json');

      const newConfig = await setup.generateConfig();
      const merged = await setup.mergeWithExisting(newConfig);

      // Should return new config when existing is invalid
      expect(merged).toEqual(newConfig);
    });

    test('should not overwrite existing ruv-swarm config by default', async () => {
      const existingConfig = {
        servers: {
          'ruv-swarm': {
            command: 'custom-command',
            args: ['custom', 'args'],
          },
        },
      };

      await fs.writeFile(configPath, JSON.stringify(existingConfig));

      const newConfig = await setup.generateConfig();
      const merged = await setup.mergeWithExisting(newConfig, { overwrite: false });

      expect(merged.servers['ruv-swarm'].command).toBe('custom-command');
    });
  });

  describe('detectClaudeConfigPath()', () => {
    test('should detect Claude config in common locations', async () => {
      const locations = [
        path.join(os.homedir(), '.claude', 'mcp.json'),
        path.join(os.homedir(), '.config', 'claude', 'mcp.json'),
        path.join(os.homedir(), 'Library', 'Application Support', 'Claude', 'mcp.json'),
      ];

      // Create one of the locations
      const testLocation = path.join(tempDir, '.claude', 'mcp.json');
      await fs.mkdir(path.dirname(testLocation), { recursive: true });
      await fs.writeFile(testLocation, '{}');

      // Mock home directory
      const originalHomedir = os.homedir;
      os.homedir = () => tempDir;

      const detected = await setup.detectClaudeConfigPath();
      expect(detected).toBe(testLocation);

      os.homedir = originalHomedir;
    });

    test('should return null if no config found', async () => {
      const detected = await setup.detectClaudeConfigPath();
      
      if (detected === null) {
        expect(detected).toBeNull();
      } else {
        // If found on system, verify it exists
        const exists = await fs.access(detected).then(() => true).catch(() => false);
        expect(exists).toBe(true);
      }
    });
  });

  describe('validateConfig()', () => {
    test('should validate correct config', async () => {
      const config = await setup.generateConfig();
      const validation = setup.validateConfig(config);

      expect(validation.valid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should catch missing servers object', () => {
      const validation = setup.validateConfig({});

      expect(validation.valid).toBe(false);
      expect(validation.errors).toContain('Missing servers object');
    });

    test('should catch invalid server config', () => {
      const config = {
        servers: {
          'invalid-server': {
            // Missing command
            args: ['test'],
          },
        },
      };

      const validation = setup.validateConfig(config);

      expect(validation.valid).toBe(false);
      expect(validation.errors.some(e => e.includes('command'))).toBe(true);
    });

    test('should validate TCP configuration', () => {
      const config = {
        servers: {
          'tcp-server': {
            command: 'test',
            tcp: {
              host: 'localhost',
              // Missing port
            },
          },
        },
      };

      const validation = setup.validateConfig(config);

      expect(validation.valid).toBe(false);
      expect(validation.errors.some(e => e.includes('port'))).toBe(true);
    });
  });

  describe('installToClaudeConfig()', () => {
    test('should install to detected Claude config', async () => {
      const claudeConfigDir = path.join(tempDir, '.claude');
      await fs.mkdir(claudeConfigDir, { recursive: true });
      
      const claudeConfigPath = path.join(claudeConfigDir, 'mcp.json');
      await fs.writeFile(claudeConfigPath, JSON.stringify({ servers: {} }));

      // Mock detection
      setup.detectClaudeConfigPath = async () => claudeConfigPath;

      const result = await setup.installToClaudeConfig();

      expect(result.success).toBe(true);
      expect(result.path).toBe(claudeConfigPath);

      const content = await fs.readFile(claudeConfigPath, 'utf-8');
      const config = JSON.parse(content);
      expect(config.servers).toHaveProperty('ruv-swarm');
    });

    test('should handle missing Claude config', async () => {
      setup.detectClaudeConfigPath = async () => null;

      const result = await setup.installToClaudeConfig();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Claude configuration not found');
    });
  });

  describe('generateLaunchCommand()', () => {
    test('should generate launch command with MCP', () => {
      const commands = setup.generateLaunchCommands();

      expect(commands.default).toContain('claude');
      expect(commands.default).toContain('--mcp');
    });

    test('should generate platform-specific commands', () => {
      const commands = setup.generateLaunchCommands();

      expect(commands).toHaveProperty('windows');
      expect(commands).toHaveProperty('macos');
      expect(commands).toHaveProperty('linux');

      if (process.platform === 'win32') {
        expect(commands.windows).toMatch(/claude\.exe|claude/);
      }
    });

    test('should include config path if provided', () => {
      const commands = setup.generateLaunchCommands('/custom/path/mcp.json');

      expect(commands.default).toContain('--mcp-config');
      expect(commands.default).toContain('/custom/path/mcp.json');
    });
  });

  describe('checkMcpServerStatus()', () => {
    test('should check if MCP server is running', async () => {
      // Mock server check
      const originalFetch = global.fetch;
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'running' }),
      });

      const status = await setup.checkMcpServerStatus();

      expect(status.running).toBe(true);
      expect(status.error).toBeUndefined();

      global.fetch = originalFetch;
    });

    test('should handle server not running', async () => {
      const originalFetch = global.fetch;
      global.fetch = jest.fn().mockRejectedValue(new Error('Connection refused'));

      const status = await setup.checkMcpServerStatus();

      expect(status.running).toBe(false);
      expect(status.error).toContain('Connection refused');

      global.fetch = originalFetch;
    });
  });

  describe('platform-specific behavior', () => {
    test('should use correct paths on Windows', async () => {
      const originalPlatform = process.platform;
      Object.defineProperty(process, 'platform', {
        value: 'win32',
        configurable: true,
      });

      const paths = setup.getDefaultPaths();
      
      expect(paths.some(p => p.includes('AppData'))).toBe(true);
      expect(paths.some(p => p.includes('Claude'))).toBe(true);

      Object.defineProperty(process, 'platform', {
        value: originalPlatform,
        configurable: true,
      });
    });

    test('should use correct paths on macOS', async () => {
      const originalPlatform = process.platform;
      Object.defineProperty(process, 'platform', {
        value: 'darwin',
        configurable: true,
      });

      const paths = setup.getDefaultPaths();
      
      expect(paths.some(p => p.includes('Library/Application Support'))).toBe(true);

      Object.defineProperty(process, 'platform', {
        value: originalPlatform,
        configurable: true,
      });
    });
  });
});