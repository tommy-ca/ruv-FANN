#!/usr/bin/env node
/**
 * GitHub Epic Command Hook for Claude Code
 * This makes the /github/epic command functional
 */

const GitHubEpicCommand = require('../../src/github-epic-implementation');

// Hook implementation for Claude Code
async function handleGitHubEpicCommand(args) {
  const command = new GitHubEpicCommand();
  
  try {
    // Parse the command from Claude Code format
    // Expected: /github/epic create "Title" --options
    const cleanArgs = args.filter(arg => arg && arg !== '/github/epic');
    
    return await command.execute(cleanArgs);
  } catch (error) {
    console.error('❌ Epic command failed:', error.message);
    
    // Provide helpful error messages
    if (error.message.includes('gh: command not found')) {
      console.error('\n📦 GitHub CLI not installed. Install with:');
      console.error('  macOS: brew install gh');
      console.error('  Linux: sudo apt install gh');
      console.error('  Then: gh auth login');
    }
    
    throw error;
  }
}

// Export for Claude Code integration
module.exports = { handleGitHubEpicCommand };

// If called directly, execute the command
if (require.main === module) {
  const args = process.argv.slice(2);
  handleGitHubEpicCommand(args)
    .then(() => process.exit(0))
    .catch(() => process.exit(1));
}