#!/usr/bin/env node

/**
 * Example: Using ruv-swarm onboarding modules
 * 
 * This example demonstrates how to use the individual onboarding
 * components for custom integration scenarios.
 */

import {
  detectClaudeCode,
  MCPConfig,
  generateMCPConfig,
  detectGitHubToken,
  createCLI,
  launchClaudeCode
} from '../src/onboarding/index.js';

async function customOnboardingExample() {
  // Create CLI instance
  const cli = createCLI({ verbose: true });
  
  try {
    cli.welcome();

    // Step 1: Detect Claude Code
    console.log('\n--- Step 1: Detecting Claude Code ---');
    const claudeInfo = await detectClaudeCode();
    
    if (claudeInfo.installed) {
      console.log(`✅ Claude Code found at: ${claudeInfo.path}`);
      console.log(`   Version: ${claudeInfo.version}`);
    } else {
      console.log('❌ Claude Code not found');
      console.log('   Please install from: https://claude.ai/download');
    }

    // Step 2: Check for GitHub token
    console.log('\n--- Step 2: Checking GitHub Authentication ---');
    const githubToken = detectGitHubToken();
    
    if (githubToken) {
      console.log('✅ GitHub token detected');
      console.log(`   Token starts with: ${githubToken.substring(0, 8)}...`);
    } else {
      console.log('⚠️  No GitHub token found');
      console.log('   Run "gh auth login" or set GITHUB_TOKEN env var');
    }

    // Step 3: Create MCP configuration
    console.log('\n--- Step 3: Creating MCP Configuration ---');
    const mcpConfig = new MCPConfig();
    
    // Add GitHub MCP
    if (githubToken) {
      mcpConfig.addGitHubMCP(githubToken);
      console.log('✅ Added GitHub MCP server');
    }
    
    // Add ruv-swarm MCP
    mcpConfig.addRuvSwarmMCP('example-swarm-001', 'mesh');
    console.log('✅ Added ruv-swarm MCP server');

    // Display configuration
    console.log('\nGenerated MCP configuration:');
    console.log(mcpConfig.toJSON());

    // Step 4: Save configuration (optional)
    if (await cli.confirm('\nSave this configuration to .claude/mcp.json?', false)) {
      const result = await generateMCPConfig(process.cwd(), mcpConfig);
      
      if (result.success) {
        console.log(`✅ Configuration saved to: ${result.path}`);
      } else {
        console.log(`❌ Failed to save: ${result.error}`);
      }
    }

    // Step 5: Launch Claude Code (optional)
    if (claudeInfo.installed && await cli.confirm('\nLaunch Claude Code now?', false)) {
      console.log('\nLaunching Claude Code...');
      await launchClaudeCode({ verbose: true });
    }

  } catch (error) {
    cli.formatError(error, [
      'Check your permissions',
      'Ensure Claude Code is installed',
      'Try running with sudo if needed'
    ]);
  }
}

// Run the example
customOnboardingExample();