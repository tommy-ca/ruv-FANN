/**
 * Node.js onboarding system for ruv-swarm
 * 
 * Provides seamless onboarding experience across platforms as specified
 * in GitHub PR #32 requirements. This module exports all onboarding
 * functionality for use in Node.js environments.
 */

import { spawn, execSync } from 'child_process';
import { promises as fs, existsSync } from 'fs';
import { homedir } from 'os';
import path from 'path';
import readline from 'readline';

/**
 * Information about Claude Code installation
 */
export class ClaudeInfo {
  constructor(installed = false, path = '', version = null) {
    this.installed = installed;
    this.path = path;
    this.version = version;
  }
}

/**
 * MCP server configuration
 */
export class MCPServerConfig {
  constructor(command = '', args = [], env = {}, enabled = true) {
    this.command = command;
    this.args = args;
    this.env = env;
    this.enabled = enabled;
  }
}

/**
 * MCP configuration container
 */
export class MCPConfig {
  constructor(servers = {}, autoStart = true, stdioEnabled = true) {
    this.servers = servers;
    this.autoStart = autoStart;
    this.stdioEnabled = stdioEnabled;
  }
}

/**
 * Onboarding configuration
 */
export class OnboardingConfig {
  constructor(options = {}) {
    this.skipClaudeDetection = options.skipClaudeDetection || false;
    this.skipMcpConfiguration = options.skipMcpConfiguration || false;
    this.skipLaunch = options.skipLaunch || false;
    this.autoLaunch = options.autoLaunch || false;
  }
}

/**
 * Default Claude Code detector for Node.js
 */
export class DefaultClaudeDetector {
  constructor() {
    this.searchPaths = this._getDefaultSearchPaths();
  }

  /**
   * Get platform-specific search paths for Claude Code
   */
  _getDefaultSearchPaths() {
    const paths = [];
    const home = homedir();

    if (process.platform === 'win32') {
      // Windows paths
      paths.push(
        'C:\\Program Files\\Claude\\claude.exe',
        'C:\\Program Files (x86)\\Claude\\claude.exe',
        path.join(home, 'AppData', 'Local', 'Claude', 'claude.exe')
      );
    } else if (process.platform === 'darwin') {
      // macOS paths
      paths.push(
        '/Applications/Claude.app/Contents/MacOS/claude',
        path.join(home, 'Applications', 'Claude.app', 'Contents', 'MacOS', 'claude')
      );
    } else {
      // Linux paths
      paths.push(
        '/usr/local/bin/claude',
        '/usr/bin/claude',
        path.join(home, '.local', 'bin', 'claude'),
        '/opt/claude/claude'
      );
    }

    // Also check PATH
    if (process.env.PATH) {
      const pathDirs = process.env.PATH.split(path.delimiter);
      for (const pathDir of pathDirs) {
        const claudePath = process.platform === 'win32' 
          ? path.join(pathDir, 'claude.exe')
          : path.join(pathDir, 'claude');
        paths.push(claudePath);
      }
    }

    return paths;
  }

  /**
   * Detect Claude Code installation
   */
  async detect() {
    // First try to find Claude in the search paths
    for (const searchPath of this.searchPaths) {
      if (existsSync(searchPath)) {
        if (await this.validateInstallation(searchPath)) {
          const version = await this._getVersion(searchPath);
          return new ClaudeInfo(true, searchPath, version);
        }
      }
    }

    // Try to find Claude using which/where command
    const findCmd = process.platform === 'win32' ? 'where' : 'which';
    const claudeName = process.platform === 'win32' ? 'claude.exe' : 'claude';

    try {
      const result = execSync(`${findCmd} ${claudeName}`, { encoding: 'utf8' }).trim();
      if (result && await this.validateInstallation(result)) {
        const version = await this._getVersion(result);
        return new ClaudeInfo(true, result, version);
      }
    } catch (error) {
      // Command failed, Claude not found in PATH
    }

    // Claude Code not found
    return new ClaudeInfo(false, '', null);
  }

  /**
   * Validate Claude Code installation
   */
  async validateInstallation(claudePath) {
    if (!existsSync(claudePath)) {
      return false;
    }

    try {
      const result = execSync(`"${claudePath}" --help`, { encoding: 'utf8', timeout: 5000 });
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get Claude Code version
   */
  async _getVersion(claudePath) {
    try {
      const result = execSync(`"${claudePath}" --version`, { encoding: 'utf8', timeout: 5000 });
      return result.trim();
    } catch (error) {
      return null;
    }
  }
}

/**
 * Default MCP configurator for Node.js
 */
export class DefaultMCPConfigurator {
  constructor() {
    this.claudeConfigDir = this._findClaudeConfigDir();
  }

  /**
   * Find Claude Code configuration directory
   */
  _findClaudeConfigDir() {
    const home = homedir();
    const possiblePaths = [
      path.join(home, '.claude'),
      path.join(home, '.config', 'claude'),
      path.join(home, 'AppData', 'Roaming', 'Claude'), // Windows
      path.join(home, 'Library', 'Application Support', 'Claude'), // macOS
    ];

    for (const configPath of possiblePaths) {
      if (existsSync(configPath)) {
        return configPath;
      }
    }

    return null;
  }

  /**
   * Configure MCP servers for ruv-swarm
   */
  async configureForRuvSwarm() {
    // Check if ruv-swarm is available
    if (!await this._checkRuvSwarmAvailability()) {
      throw new Error('ruv-swarm not available via npx. Please install with: npm install -g ruv-swarm');
    }

    // Generate configuration
    const config = this._generateRuvSwarmConfig();

    // Validate configuration
    await this.validateConfiguration(config);

    return config;
  }

  /**
   * Generate default ruv-swarm MCP configuration
   */
  _generateRuvSwarmConfig() {
    const servers = {
      'ruv-swarm': new MCPServerConfig(
        'npx',
        ['ruv-swarm', 'mcp', 'start'],
        {},
        true
      ),
    };

    return new MCPConfig(servers, true, true);
  }

  /**
   * Check if ruv-swarm is available via npx
   */
  async _checkRuvSwarmAvailability() {
    try {
      execSync('npx ruv-swarm --version', { encoding: 'utf8', timeout: 10000 });
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate MCP configuration
   */
  async validateConfiguration(config) {
    // Validate that all server commands are available
    for (const [name, serverConfig] of Object.entries(config.servers)) {
      try {
        const testCmd = `${serverConfig.command} ${serverConfig.args.join(' ')} --help`;
        execSync(testCmd, { encoding: 'utf8', timeout: 5000 });
      } catch (error) {
        throw new Error(`MCP server '${name}' command not available: ${serverConfig.command}`);
      }
    }

    return true;
  }

  /**
   * Apply MCP configuration
   */
  async applyConfiguration(config) {
    // Write configuration to Claude's MCP config
    await this._writeClaudeMcpConfig(config);
  }

  /**
   * Write MCP configuration to Claude config
   */
  async _writeClaudeMcpConfig(config) {
    if (!this.claudeConfigDir) {
      throw new Error('Claude configuration directory not found');
    }

    const mcpConfigPath = path.join(this.claudeConfigDir, 'mcp.json');
    const configJson = JSON.stringify(config, null, 2);

    await fs.writeFile(mcpConfigPath, configJson);
  }
}

/**
 * Default interactive prompt for Node.js
 */
export class DefaultInteractivePrompt {
  constructor() {
    this.rl = null;
  }

  /**
   * Initialize readline interface
   */
  _initReadline() {
    if (!this.rl) {
      this.rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });
    }
  }

  /**
   * Check if we're in a non-interactive environment
   */
  _isNonInteractive() {
    return process.env.CI || 
           process.env.TERM === 'dumb' || 
           !process.stdin.isTTY;
  }

  /**
   * Confirmation prompt
   */
  async confirm(message, defaultValue = true) {
    if (this._isNonInteractive()) {
      this.info(`Non-interactive mode: using default '${defaultValue}' for: ${message}`);
      return defaultValue;
    }

    this._initReadline();
    const defaultText = defaultValue ? '[Y/n]' : '[y/N]';
    
    return new Promise((resolve) => {
      this.rl.question(`${message} ${defaultText}: `, (answer) => {
        const normalizedAnswer = answer.toLowerCase().trim();
        if (normalizedAnswer === '') {
          resolve(defaultValue);
        } else {
          resolve(normalizedAnswer === 'y' || normalizedAnswer === 'yes');
        }
      });
    });
  }

  /**
   * Input prompt
   */
  async input(message, defaultValue = null) {
    if (this._isNonInteractive()) {
      const defaultText = defaultValue || '';
      this.info(`Non-interactive mode: using default '${defaultText}' for: ${message}`);
      return defaultText;
    }

    this._initReadline();
    const defaultText = defaultValue ? ` [${defaultValue}]` : '';
    
    return new Promise((resolve) => {
      this.rl.question(`${message}${defaultText}: `, (answer) => {
        resolve(answer.trim() || defaultValue || '');
      });
    });
  }

  /**
   * Selection prompt
   */
  async select(message, options, defaultIndex = 0) {
    if (this._isNonInteractive()) {
      const defaultValue = options[defaultIndex] || '<invalid>';
      this.info(`Non-interactive mode: using default '${defaultValue}' for: ${message}`);
      return defaultIndex;
    }

    this._initReadline();
    console.log(message);
    options.forEach((option, index) => {
      const marker = index === defaultIndex ? '>' : ' ';
      console.log(`${marker} ${index + 1}. ${option}`);
    });

    return new Promise((resolve) => {
      this.rl.question(`Enter choice [1-${options.length}] (default: ${defaultIndex + 1}): `, (answer) => {
        const choice = parseInt(answer.trim()) || (defaultIndex + 1);
        const index = Math.max(0, Math.min(options.length - 1, choice - 1));
        resolve(index);
      });
    });
  }

  /**
   * Information message
   */
  info(message) {
    console.log(`\x1b[34mℹ️  ${message}\x1b[0m`);
  }

  /**
   * Warning message
   */
  warning(message) {
    console.log(`\x1b[33m⚠️  ${message}\x1b[0m`);
  }

  /**
   * Success message
   */
  success(message) {
    console.log(`\x1b[32m✅ ${message}\x1b[0m`);
  }

  /**
   * Close readline interface
   */
  close() {
    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }
  }
}

/**
 * Default launch manager for Node.js
 */
export class DefaultLaunchManager {
  constructor(options = {}) {
    this.launchTimeout = options.timeout || 30000;
  }

  /**
   * Launch Claude Code with MCP configuration
   */
  async launchWithConfig(claudePath, mcpConfig) {
    // First, prepare the environment
    await this._prepareEnvironment(mcpConfig);

    // Validate that Claude Code can be launched
    await this.validateLaunch(claudePath);

    // Get launch arguments
    const args = this._getLaunchArgs(true);

    // Launch Claude Code
    await this._launchClaude(claudePath, args);
  }

  /**
   * Launch Claude Code without MCP
   */
  async launchSimple(claudePath) {
    // Validate that Claude Code can be launched
    await this.validateLaunch(claudePath);

    // Get basic launch arguments
    const args = this._getLaunchArgs(false);

    // Launch Claude Code
    await this._launchClaude(claudePath, args);
  }

  /**
   * Validate Claude Code launch capability
   */
  async validateLaunch(claudePath) {
    if (!existsSync(claudePath)) {
      throw new Error(`Claude Code not found at: ${claudePath}`);
    }

    try {
      const result = execSync(`"${claudePath}" --version`, { encoding: 'utf8', timeout: 5000 });
      if (!result.trim()) {
        throw new Error('Claude Code version command returned empty output');
      }
      return true;
    } catch (error) {
      throw new Error(`Claude Code validation failed: ${error.message}`);
    }
  }

  /**
   * Prepare environment for MCP servers
   */
  async _prepareEnvironment(mcpConfig) {
    // Ensure all MCP servers are available
    for (const [name, serverConfig] of Object.entries(mcpConfig.servers)) {
      if (!serverConfig.enabled) {
        continue;
      }

      try {
        const testCmd = `${serverConfig.command} ${serverConfig.args.join(' ')} --help`;
        execSync(testCmd, { encoding: 'utf8', timeout: 5000 });
      } catch (error) {
        throw new Error(`MCP server '${name}' is not available. Command: ${serverConfig.command} ${serverConfig.args.join(' ')}`);
      }
    }
  }

  /**
   * Get appropriate launch arguments for the platform
   */
  _getLaunchArgs(withMcp) {
    const args = [];

    if (withMcp) {
      // Add MCP-related arguments if available
      args.push('--mcp-enable');
    }

    // Add any other common arguments
    args.push('--new-session');

    return args;
  }

  /**
   * Launch Claude Code with specific arguments
   */
  async _launchClaude(claudePath, args) {
    console.log(`Launching Claude Code: ${claudePath} ${args.join(' ')}`);

    return new Promise((resolve, reject) => {
      const child = spawn(claudePath, args, {
        detached: true,
        stdio: 'ignore'
      });

      // Timeout handling
      const timeout = setTimeout(() => {
        child.kill();
        reject(new Error('Claude Code launch timed out'));
      }, this.launchTimeout);

      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to launch Claude Code: ${error.message}`));
      });

      child.on('spawn', () => {
        clearTimeout(timeout);
        // Detach the process so it continues running
        child.unref();
        console.log('Claude Code launched successfully');
        resolve();
      });
    });
  }
}

/**
 * Main onboarding flow orchestrator
 */
export async function runOnboardingFlow(config = {}, outputHandler = null) {
  const onboardingConfig = new OnboardingConfig(config);
  const prompt = new DefaultInteractivePrompt();

  try {
    console.log('🎉 Welcome to RUV Swarm Onboarding');

    // Step 1: Detect Claude Code installation
    if (!onboardingConfig.skipClaudeDetection) {
      const detector = new DefaultClaudeDetector();
      const claudeInfo = await detector.detect();

      if (claudeInfo.installed) {
        prompt.success(`Claude Code detected: ${claudeInfo.path}`);
      } else {
        prompt.warning('Claude Code not found');
        prompt.info('Please install Claude Code from: https://claude.ai/download');
      }

      // Step 2: Configure MCP servers
      if (!onboardingConfig.skipMcpConfiguration) {
        const configurator = new DefaultMCPConfigurator();
        const mcpConfig = await configurator.configureForRuvSwarm();
        prompt.info('MCP configuration prepared');

        // Step 3: Interactive setup (if Claude Code is available)
        if (claudeInfo.installed && !onboardingConfig.skipLaunch) {
          const shouldLaunch = onboardingConfig.autoLaunch || 
            await prompt.confirm('Would you like to launch Claude Code with ruv-swarm integration?', true);

          if (shouldLaunch) {
            const launcher = new DefaultLaunchManager();
            await launcher.launchWithConfig(claudeInfo.path, mcpConfig);
            prompt.success('Claude Code launched with ruv-swarm integration');
          }
        }
      }
    }

    prompt.success('🎉 Onboarding completed successfully!');
  } catch (error) {
    prompt.warning(`Onboarding failed: ${error.message}`);
    throw error;
  } finally {
    prompt.close();
  }
}