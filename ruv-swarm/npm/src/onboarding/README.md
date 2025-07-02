# ruv-swarm Onboarding Module

Seamless onboarding experience for ruv-swarm with automatic Claude Code detection and MCP configuration.

## 🚀 Features

- **Automatic Claude Code Detection** - Cross-platform detection of Claude Code installation
- **Interactive CLI** - Beautiful prompts with progress indicators
- **MCP Configuration** - Automatic setup of GitHub and ruv-swarm MCP servers
- **Token Detection** - Smart detection of GitHub authentication tokens
- **Session Management** - Track and manage Claude Code sessions
- **Error Recovery** - Graceful handling of failures with helpful suggestions

## 📦 Components

### 1. Claude Detector (`claude-detector.js`)

Detects Claude Code installation across different platforms:

```javascript
import { detectClaudeCode, isVersionCompatible } from 'ruv-swarm/onboarding';

const claudeInfo = await detectClaudeCode();
if (claudeInfo.installed) {
  console.log(`Found Claude Code v${claudeInfo.version} at ${claudeInfo.path}`);
  
  if (isVersionCompatible(claudeInfo.version)) {
    console.log('Version is compatible!');
  }
}
```

### 2. MCP Setup (`mcp-setup.js`)

Manages MCP server configuration:

```javascript
import { MCPConfig, generateMCPConfig, detectGitHubToken } from 'ruv-swarm/onboarding';

// Create configuration
const config = new MCPConfig();

// Add GitHub MCP with auto-detected token
const token = detectGitHubToken();
config.addGitHubMCP(token);

// Add ruv-swarm MCP
config.addRuvSwarmMCP('my-swarm-id', 'hierarchical');

// Generate config file
const result = await generateMCPConfig('/path/to/project', config);
```

### 3. Interactive CLI (`interactive-cli.js`)

Beautiful command-line interface:

```javascript
import { createCLI } from 'ruv-swarm/onboarding';

const cli = createCLI({ autoAccept: false, verbose: true });

// Welcome message
cli.welcome();

// Prompts
const proceed = await cli.confirm('Continue with setup?', true);
const choice = await cli.choice('Select option:', ['Option 1', 'Option 2']);
const token = await cli.promptPassword('Enter GitHub token:');

// Progress indicators
cli.startSpinner('Installing...');
// ... do work ...
cli.succeedSpinner('Installation complete!');

// Messages
cli.info('This is informational');
cli.success('Operation successful');
cli.warning('This needs attention');
cli.error('Something went wrong');
```

### 4. Launch Command (`launch.js`)

Launch Claude Code with MCP configuration:

```javascript
import { launchClaudeCode, launchWithSession } from 'ruv-swarm/onboarding';

// Simple launch
const result = await launchClaudeCode({
  args: ['--debug'],
  verbose: true
});

// Launch with session tracking
const sessionResult = await launchWithSession({
  autoAccept: true
});
console.log(`Session ID: ${sessionResult.sessionId}`);
```

## 🎯 Complete Onboarding Flow

The module provides a complete onboarding function:

```javascript
import { runOnboarding } from 'ruv-swarm/onboarding';

// Run full onboarding
const result = await runOnboarding({
  autoAccept: false,  // Interactive mode
  verbose: true       // Detailed output
});

if (result.success) {
  console.log('Onboarding completed successfully!');
}
```

## 🔧 Platform Support

The module supports detection and installation on:

- **Windows** - Checks Program Files, AppData, and user directories
- **macOS** - Checks Applications folder and common Unix paths
- **Linux** - Checks standard Unix paths and user directories

## 🛡️ Error Handling

All functions include comprehensive error handling:

```javascript
try {
  const result = await generateMCPConfig(projectPath, config);
  if (!result.success) {
    console.error(`Configuration failed: ${result.error}`);
  }
} catch (error) {
  cli.formatError(error, [
    'Check file permissions',
    'Ensure directory exists',
    'Try running with elevated permissions'
  ]);
}
```

## 🚦 Auto-Accept Mode

For CI/CD and automated setups:

```javascript
// All prompts will be auto-accepted with defaults
const cli = createCLI({ autoAccept: true, verbose: true });

// Or use environment variable
process.env.RUV_SWARM_AUTO_ACCEPT = 'true';
```

## 📊 Configuration Format

The generated `.claude/mcp.json` follows this format:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"],
      "env": {
        "SWARM_ID": "swarm-abc123-xyz789",
        "SWARM_TOPOLOGY": "mesh"
      }
    }
  }
}
```

## 🧪 Testing

Run the onboarding tests:

```bash
# Run all onboarding tests
npm test test/onboarding/*.test.js

# Run interactive test
node test/onboarding/run-tests.js

# Try the example
node examples/onboarding-example.js
```

## 📝 License

MIT OR Apache-2.0