/**
 * TypeScript interfaces for the seamless onboarding system
 * 
 * This module provides type definitions for the guided onboarding experience
 * that automatically sets up Claude Code with preconfigured MCP servers.
 */

/**
 * Errors that can occur during onboarding
 */
export class OnboardingError extends Error {
  constructor(
    message: string,
    public code: OnboardingErrorCode,
    public details?: any
  ) {
    super(message);
    this.name = 'OnboardingError';
  }
}

export enum OnboardingErrorCode {
  CLAUDE_CODE_NOT_FOUND = 'CLAUDE_CODE_NOT_FOUND',
  INCOMPATIBLE_VERSION = 'INCOMPATIBLE_VERSION',
  INSTALLATION_FAILED = 'INSTALLATION_FAILED',
  CONFIGURATION_ERROR = 'CONFIGURATION_ERROR',
  PERMISSION_DENIED = 'PERMISSION_DENIED',
  NETWORK_ERROR = 'NETWORK_ERROR',
  LAUNCH_ERROR = 'LAUNCH_ERROR',
}

/**
 * Platform-specific installation paths
 */
export interface InstallationPaths {
  systemPaths: string[];
  userPaths: string[];
  searchPaths: string[];
}

/**
 * Claude Code detection result
 */
export interface DetectionResult {
  found: boolean;
  path?: string;
  version?: string;
  isCompatible: boolean;
}

/**
 * Installation options
 */
export interface InstallOptions {
  targetDir: string;
  systemInstall: boolean;
  forceReinstall: boolean;
  version?: string;
}

/**
 * MCP server configuration
 */
export interface MCPServerConfig {
  command: string;
  args: string[];
  env?: Record<string, string>;
}

/**
 * Complete MCP configuration
 */
export interface MCPConfig {
  mcpServers: Record<string, MCPServerConfig>;
}

/**
 * User interaction choices
 */
export enum UserChoice {
  Yes = 'yes',
  No = 'no',
  Skip = 'skip',
}

/**
 * Launch options for Claude Code
 */
export interface LaunchOptions {
  mcpConfigPath: string;
  skipPermissions: boolean;
  waitForAuth: boolean;
}

/**
 * Interface for detecting Claude Code installation
 */
export interface ClaudeDetector {
  /**
   * Check if Claude Code is installed and return detection result
   */
  detect(): Promise<DetectionResult>;
  
  /**
   * Get platform-specific installation paths
   */
  getInstallationPaths(): InstallationPaths;
  
  /**
   * Validate Claude Code version compatibility
   */
  validateVersion(version: string): Promise<boolean>;
}

/**
 * Interface for installing Claude Code
 */
export interface Installer {
  /**
   * Download and install Claude Code
   */
  install(options: InstallOptions): Promise<string>;
  
  /**
   * Check if installation requires elevated permissions
   */
  requiresElevation(targetDir: string): Promise<boolean>;
  
  /**
   * Verify installation was successful
   */
  verifyInstallation(installPath: string): Promise<boolean>;
  
  /**
   * Rollback failed installation
   */
  rollback(installPath: string): Promise<void>;
}

/**
 * Interface for configuring MCP servers
 */
export interface MCPConfigurator {
  /**
   * Create MCP configuration file
   */
  createConfig(config: MCPConfig, path: string): Promise<void>;
  
  /**
   * Load existing MCP configuration
   */
  loadConfig(path: string): Promise<MCPConfig>;
  
  /**
   * Validate MCP configuration
   */
  validateConfig(config: MCPConfig): Promise<boolean>;
  
  /**
   * Check for GitHub token availability
   */
  checkGithubToken(): Promise<string | null>;
  
  /**
   * Generate swarm ID
   */
  generateSwarmId(): string;
}

/**
 * Interface for user interaction
 */
export interface InteractivePrompt {
  /**
   * Show Y/N confirmation prompt
   */
  confirm(message: string, defaultValue?: boolean): Promise<boolean>;
  
  /**
   * Show multiple choice prompt
   */
  choice(message: string, options: string[]): Promise<number>;
  
  /**
   * Show text input prompt
   */
  input(message: string, defaultValue?: string): Promise<string>;
  
  /**
   * Show password input prompt (hidden)
   */
  password(message: string): Promise<string>;
  
  /**
   * Display info message
   */
  info(message: string): void;
  
  /**
   * Display warning message
   */
  warning(message: string): void;
  
  /**
   * Display error message
   */
  error(message: string): void;
  
  /**
   * Display success message
   */
  success(message: string): void;
  
  /**
   * Show progress bar
   */
  progress(message: string, current: number, total: number): void;
}

/**
 * Interface for launching Claude Code
 */
export interface LaunchManager {
  /**
   * Launch Claude Code with MCP configuration
   */
  launch(options: LaunchOptions): Promise<void>;
  
  /**
   * Check if Claude Code is already running
   */
  isRunning(): Promise<boolean>;
  
  /**
   * Wait for Claude Code to be ready
   */
  waitForReady(timeoutSecs: number): Promise<void>;
  
  /**
   * Guide user through authentication
   */
  guideAuth(): Promise<void>;
}

/**
 * Checkpoint for rollback support
 */
export interface Checkpoint {
  id: string;
  timestamp: Date;
  description: string;
  data: Record<string, string>;
}

/**
 * Interface for rollback support
 */
export interface Rollback {
  /**
   * Create a checkpoint
   */
  checkpoint(description: string): Promise<string>;
  
  /**
   * Rollback to a checkpoint
   */
  rollback(checkpointId: string): Promise<void>;
  
  /**
   * Commit changes (clear checkpoints)
   */
  commit(): Promise<void>;
  
  /**
   * List available checkpoints
   */
  listCheckpoints(): Promise<Checkpoint[]>;
}

/**
 * Platform detection
 */
export enum Platform {
  Windows = 'windows',
  MacOS = 'macos',
  Linux = 'linux',
  Unknown = 'unknown',
}

/**
 * Configuration for onboarding process
 */
export interface OnboardingConfig {
  autoAccept: boolean;
  claudeCodeVersion: string;
  defaultTopology: string;
  defaultMaxAgents: number;
  retryAttempts: number;
  retryDelayMs: number;
}

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG: OnboardingConfig = {
  autoAccept: false,
  claudeCodeVersion: '>=1.0.0',
  defaultTopology: 'mesh',
  defaultMaxAgents: 5,
  retryAttempts: 3,
  retryDelayMs: 1000,
};

/**
 * Main orchestrator for the onboarding process
 */
export class OnboardingOrchestrator {
  constructor(
    private detector: ClaudeDetector,
    private installer: Installer,
    private configurator: MCPConfigurator,
    private prompt: InteractivePrompt,
    private launcher: LaunchManager,
    private config: OnboardingConfig = DEFAULT_CONFIG
  ) {}
  
  /**
   * Run the complete onboarding flow
   */
  async run(): Promise<void> {
    await this.prompt.info('🚀 Welcome to ruv-swarm!');
    
    // Step 1: Detect Claude Code
    const detection = await this.detector.detect();
    
    if (!detection.found) {
      await this.prompt.error('Claude Code not found');
      
      const shouldInstall = this.config.autoAccept || 
        await this.prompt.confirm('Would you like to install Claude Code?', true);
      
      if (shouldInstall) {
        await this.installClaudeCode();
      } else {
        throw new OnboardingError(
          'Claude Code not found',
          OnboardingErrorCode.CLAUDE_CODE_NOT_FOUND
        );
      }
    } else if (!detection.isCompatible) {
      await this.prompt.warning(
        `Claude Code version ${detection.version || 'unknown'} is incompatible ` +
        `(requires ${this.config.claudeCodeVersion})`
      );
    }
    
    // Step 2: Configure MCP servers
    await this.configureMCPServers();
    
    // Step 3: Offer to launch
    const shouldLaunch = this.config.autoAccept || 
      await this.prompt.confirm('Ready to launch Claude Code?', true);
    
    if (shouldLaunch) {
      await this.launchClaudeCode();
    }
    
    await this.prompt.success('✨ Initialization complete!');
  }
  
  private async installClaudeCode(): Promise<void> {
    const paths = this.detector.getInstallationPaths();
    const defaultPath = paths.userPaths[0] || paths.systemPaths[0];
    
    if (!defaultPath) {
      throw new OnboardingError(
        'No installation paths available',
        OnboardingErrorCode.INSTALLATION_FAILED
      );
    }
    
    const options: InstallOptions = {
      targetDir: defaultPath,
      systemInstall: false,
      forceReinstall: false,
    };
    
    await this.prompt.info('📦 Downloading Claude Code...');
    const installPath = await this.installer.install(options);
    
    if (await this.installer.verifyInstallation(installPath)) {
      await this.prompt.success('✅ Claude Code installed successfully!');
    } else {
      throw new OnboardingError(
        'Verification failed',
        OnboardingErrorCode.INSTALLATION_FAILED
      );
    }
  }
  
  private async configureMCPServers(): Promise<void> {
    await this.prompt.info('Setting up MCP servers...');
    
    const config: MCPConfig = { mcpServers: {} };
    let githubToken: string | undefined;
    
    // GitHub Authentication (for ruv-swarm features, not for GitHub MCP)
    const authenticateGithub = this.config.autoAccept || 
      await this.prompt.confirm('Would you like to authenticate to GitHub for enhanced ruv-swarm features?', true);
    
    if (authenticateGithub) {
      await this.prompt.info('🔑 Checking GitHub authentication...');
      
      const token = await this.configurator.checkGithubToken();
      
      if (token) {
        githubToken = token;
        await this.prompt.success('✅ Found GitHub token');
      } else {
        const choice = await this.prompt.choice(
          'No GitHub token found. GitHub integration enhances ruv-swarm with issue tracking, PR management, and more.',
          ['Enter token now', 'Skip authentication', 'Learn more']
        );
        
        switch (choice) {
          case 0:
            githubToken = await this.prompt.password('GitHub token: ');
            await this.prompt.success('✅ GitHub authentication configured');
            break;
          case 1:
            await this.prompt.info('ℹ️  Skipping GitHub authentication. Some ruv-swarm features will be limited.');
            break;
          default:
            await this.prompt.info('ℹ️  Visit https://github.com/settings/tokens to create a token with \'repo\' scope');
            await this.prompt.info('ℹ️  Skipping authentication for now. You can set GITHUB_TOKEN later.');
            break;
        }
      }
    }
    
    // ruv-swarm MCP
    const installRuvSwarm = this.config.autoAccept || 
      await this.prompt.confirm('Would you like to install the ruv-swarm MCP server?', true);
    
    if (installRuvSwarm) {
      await this.prompt.info('📝 Configuring ruv-swarm MCP server...');
      const swarmId = this.configurator.generateSwarmId();
      
      addRuvSwarmMCP(config, swarmId, this.config.defaultTopology, githubToken);
      await this.prompt.success('✅ ruv-swarm MCP server configured');
    }
    
    // Optional: Prompt for GitHub MCP if user explicitly wants it
    if (!this.config.autoAccept) {
      const installGithubMCP = await this.prompt.confirm('Would you like to also install the GitHub MCP server? (optional)', false);
      
      if (installGithubMCP) {
        await this.prompt.info('📝 Configuring GitHub MCP server...');
        addGithubMCP(config);
        await this.prompt.info('ℹ️  GitHub MCP server added. Set GITHUB_TOKEN environment variable for authentication.');
      }
    }
    
    if (Object.keys(config.mcpServers).length > 0) {
      await this.configurator.createConfig(config, '.claude/mcp.json');
    }
  }
  
  private async launchClaudeCode(): Promise<void> {
    await this.prompt.info('🚀 Launching Claude Code with MCP servers...');
    
    const options: LaunchOptions = {
      mcpConfigPath: '.claude/mcp.json',
      skipPermissions: true,
      waitForAuth: true,
    };
    
    await this.launcher.launch(options);
    
    try {
      await this.launcher.waitForReady(30);
      await this.prompt.info('📋 Please log in to your Anthropic account when prompted');
    } catch (error) {
      // Continue even if wait times out
    }
  }
}

/**
 * Helper to detect current platform
 */
export function detectPlatform(): Platform {
  switch (process.platform) {
    case 'win32':
      return Platform.Windows;
    case 'darwin':
      return Platform.MacOS;
    case 'linux':
      return Platform.Linux;
    default:
      return Platform.Unknown;
  }
}

/**
 * Helper to create default MCP configuration
 */
export function createDefaultMCPConfig(): MCPConfig {
  return {
    mcpServers: {},
  };
}

/**
 * Helper to add GitHub MCP server to config
 */
export function addGithubMCP(config: MCPConfig, token?: string): void {
  config.mcpServers.github = {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-github'],
    env: token ? { GITHUB_TOKEN: token } : { GITHUB_TOKEN: '${GITHUB_TOKEN}' },
  };
}

/**
 * Helper to add ruv-swarm MCP server to config
 */
export function addRuvSwarmMCP(
  config: MCPConfig,
  swarmId: string,
  topology: string = 'mesh',
  githubToken?: string
): void {
  config.mcpServers['ruv-swarm'] = {
    command: 'npx',
    args: ['ruv-swarm', 'mcp', 'start'],
    env: {
      SWARM_ID: swarmId,
      SWARM_TOPOLOGY: topology,
      // Include GitHub token for ruv-swarm's GitHub integration features
      GITHUB_TOKEN: githubToken || '${GITHUB_TOKEN}',
    },
  };
}