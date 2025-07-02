/**
 * Onboarding Module Exports
 * Central export point for all onboarding functionality
 */

export { detectClaudeCode, isVersionCompatible } from './claude-detector.js';
export {
  MCPConfig,
  generateMCPConfig,
  detectGitHubToken,
  validateMCPConfig,
  generateSwarmId
} from './mcp-setup.js';
export { InteractiveCLI, createCLI } from './interactive-cli.js';

// Re-export launch command for convenience
export { launchClaudeCode, SessionManager, launchWithSession } from '../commands/launch.js';

/**
 * Complete onboarding flow
 * This is a high-level function that orchestrates the entire onboarding process
 */
export async function runOnboarding(options = {}) {
  const { createCLI } = await import('./interactive-cli.js');
  const { detectClaudeCode } = await import('./claude-detector.js');
  const {
    MCPConfig,
    generateMCPConfig,
    detectGitHubToken,
    generateSwarmId
  } = await import('./mcp-setup.js');

  const cli = createCLI(options);
  
  try {
    cli.welcome();

    // Check for Claude Code
    cli.startSpinner('Checking for Claude Code...');
    const claudeInfo = await detectClaudeCode();
    
    if (!claudeInfo.installed) {
      cli.failSpinner('Claude Code not found');
      
      const shouldInstall = await cli.confirm('Would you like to install Claude Code?', true);
      if (!shouldInstall) {
        cli.warning('Claude Code installation skipped');
        cli.info('You can install it manually later');
        return { success: false, reason: 'Claude Code not installed' };
      }
      
      // TODO: Implement Claude Code installation
      cli.error('Claude Code installation not yet implemented');
      cli.info('Please install Claude Code manually from: https://claude.ai/download');
      return { success: false, reason: 'Installation not implemented' };
    }
    
    cli.succeedSpinner(`Claude Code found (v${claudeInfo.version})`);

    // Configure MCP servers
    cli.info('Setting up MCP servers...');
    const mcpConfig = new MCPConfig();
    let hasServers = false;

    // GitHub MCP
    const installGitHub = await cli.confirm('Would you like to install the GitHub MCP server?', true);
    if (installGitHub) {
      cli.startSpinner('Configuring GitHub MCP server...');
      
      const token = detectGitHubToken();
      if (!token) {
        cli.warnSpinner('No GitHub token found');
        const tokenChoice = await cli.choice(
          'No GitHub token found',
          [
            'Enter GitHub token now',
            'Continue without authentication (limited access)',
            'Skip GitHub MCP server'
          ]
        );
        
        switch (tokenChoice) {
          case 0:
            const enteredToken = await cli.promptPassword('GitHub token:');
            mcpConfig.addGitHubMCP(enteredToken);
            cli.success('GitHub MCP server configured');
            hasServers = true;
            break;
          case 1:
            cli.warning('GitHub MCP configured with limited access');
            cli.info('Run "gh auth login" later for full access');
            mcpConfig.addGitHubMCP(null);
            hasServers = true;
            break;
          case 2:
            cli.info('Skipping GitHub MCP server');
            break;
        }
      } else {
        mcpConfig.addGitHubMCP(token);
        cli.succeedSpinner('GitHub MCP server configured');
        hasServers = true;
      }
    }

    // ruv-swarm MCP
    const installRuvSwarm = await cli.confirm('Would you like to install the ruv-swarm MCP server?', true);
    if (installRuvSwarm) {
      cli.startSpinner('Configuring ruv-swarm MCP server...');
      const swarmId = generateSwarmId();
      mcpConfig.addRuvSwarmMCP(swarmId);
      cli.succeedSpinner('ruv-swarm MCP server configured');
      hasServers = true;
    }

    // Generate MCP config if servers were selected
    if (hasServers) {
      const result = await generateMCPConfig(process.cwd(), mcpConfig);
      if (!result.success) {
        cli.error(`Failed to create MCP configuration: ${result.error}`);
        return { success: false, reason: result.error };
      }
    }

    // Display summary
    const summary = {
      claudeCode: claudeInfo,
      githubMCP: mcpConfig.servers.github ? true : false,
      ruvSwarmMCP: mcpConfig.servers['ruv-swarm'] ? true : false,
      authentication: mcpConfig.servers.github?.env?.GITHUB_TOKEN ? 'ready' : 'limited'
    };
    
    cli.displaySummary(summary);

    // Confirm initialization
    const shouldInit = await cli.confirm('Initialize swarm with these settings?', true);
    if (!shouldInit) {
      cli.warning('Initialization cancelled');
      return { success: false, reason: 'User cancelled' };
    }

    cli.complete();

    // Offer to launch
    const shouldLaunch = await cli.confirm('Ready to launch Claude Code?', true);
    if (shouldLaunch) {
      const { launchClaudeCode } = await import('../commands/launch.js');
      await launchClaudeCode(options);
    }

    return { success: true };

  } catch (error) {
    cli.formatError(error, [
      'Check your internet connection',
      'Ensure you have proper permissions',
      'Try running with --verbose for more details'
    ]);
    return { success: false, error: error.message };
  }
}

export default {
  detectClaudeCode,
  isVersionCompatible,
  MCPConfig,
  generateMCPConfig,
  detectGitHubToken,
  validateMCPConfig,
  generateSwarmId,
  InteractiveCLI,
  createCLI,
  launchClaudeCode,
  SessionManager,
  launchWithSession,
  runOnboarding
};