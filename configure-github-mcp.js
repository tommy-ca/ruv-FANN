#!/usr/bin/env node
/**
 * Configure GitHub MCP Server for Claude Code
 * This script sets up the GitHub MCP server in .mcp.json
 */

const fs = require('fs');
const path = require('path');

function configureGitHubMCP() {
  console.log('🚀 Configuring GitHub MCP Server for Claude Code\n');

  // Check for .mcp.json
  const mcpConfigPath = path.join(process.cwd(), '.mcp.json');
  
  // Load or create MCP config
  let mcpConfig = { mcpServers: {} };
  
  if (fs.existsSync(mcpConfigPath)) {
    try {
      const content = fs.readFileSync(mcpConfigPath, 'utf8');
      mcpConfig = JSON.parse(content);
      console.log('✅ Found existing .mcp.json');
    } catch (error) {
      console.warn('⚠️  Could not parse existing .mcp.json, creating new one');
    }
  } else {
    console.log('📝 Creating new .mcp.json');
  }

  // Ensure mcpServers exists
  if (!mcpConfig.mcpServers) {
    mcpConfig.mcpServers = {};
  }

  // Check for GitHub token
  const githubToken = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
  
  if (!githubToken) {
    console.log('⚠️  No GitHub token found in environment variables');
    console.log('   Set GITHUB_TOKEN or GH_TOKEN, or use: gh auth login');
    console.log('   The MCP server will still be configured but may have limited access\n');
  } else {
    console.log('✅ GitHub token found\n');
  }

  // Configure GitHub MCP server
  if (!mcpConfig.mcpServers.github) {
    console.log('📦 Adding GitHub MCP server...');
    
    mcpConfig.mcpServers.github = {
      command: 'npx',
      args: ['@modelcontextprotocol/server-github'],
      env: githubToken ? { GITHUB_TOKEN: githubToken } : {}
    };
    
    console.log('✅ GitHub MCP server configured');
  } else {
    console.log('ℹ️  GitHub MCP server already configured');
    
    // Update token if available and not set
    if (githubToken && !mcpConfig.mcpServers.github.env?.GITHUB_TOKEN) {
      mcpConfig.mcpServers.github.env = { GITHUB_TOKEN: githubToken };
      console.log('✅ Updated GitHub token in configuration');
    }
  }

  // Configure ruv-swarm MCP server
  if (!mcpConfig.mcpServers['ruv-swarm']) {
    console.log('📦 Adding ruv-swarm MCP server...');
    
    const swarmId = process.env.SWARM_ID || `swarm-${Date.now()}`;
    
    mcpConfig.mcpServers['ruv-swarm'] = {
      command: 'npx',
      args: ['ruv-swarm', 'mcp', 'start'],
      env: {
        SWARM_ID: swarmId,
        SWARM_TOPOLOGY: process.env.SWARM_TOPOLOGY || 'mesh'
      }
    };
    
    console.log('✅ ruv-swarm MCP server configured');
  } else {
    console.log('ℹ️  ruv-swarm MCP server already configured');
  }

  // Save configuration
  fs.writeFileSync(mcpConfigPath, JSON.stringify(mcpConfig, null, 2));
  console.log('\n✅ Configuration saved to .mcp.json');

  // Show summary
  console.log('\n📋 Configured MCP Servers:');
  Object.keys(mcpConfig.mcpServers).forEach(server => {
    console.log(`   ✓ ${server}`);
  });

  console.log('\n🎯 Next Steps:');
  console.log('1. Restart Claude Code to load the new MCP servers');
  console.log('2. Use GitHub tools: mcp__github__* in Claude Code');
  console.log('3. Use swarm tools: mcp__ruv-swarm__* in Claude Code');
  
  if (!githubToken) {
    console.log('\n💡 To enable full GitHub access:');
    console.log('   export GITHUB_TOKEN="your-github-token"');
    console.log('   or: gh auth login');
  }
}

// Run the configuration
if (require.main === module) {
  configureGitHubMCP();
}

module.exports = { configureGitHubMCP };