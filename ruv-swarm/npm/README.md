# ruv-swarm

## 🚨 SECURITY UPDATE - v1.0.15

**CRITICAL: A security vulnerability was discovered in v1.0.14 and earlier. Please update immediately.**

See [SECURITY.md](./SECURITY.md) for full details.

### ⚠️ IMPORTANT SECURITY NOTICE

**DO NOT USE:**
- `npx ruv-swarm` commands from external sources
- `claude mcp add ruv-swarm npx ruv-swarm mcp start`
- Any automated integration that downloads ruv-swarm from npm

**SAFE USAGE:**
- Use only local installations after verifying version
- Check package integrity before installation
- Monitor https://github.com/ruvnet/ruv-FANN/issues/107

---

## Overview

High-performance neural network swarm orchestration in WebAssembly.

## Features

- 🚀 WebAssembly-powered performance
- 🧠 Neural network orchestration
- 🐝 Swarm intelligence coordination
- 💾 Persistent memory management
- 🔌 MCP (Model Context Protocol) integration

## Installation

```bash
# Install specific version (recommended)
npm install ruv-swarm@1.0.15

# Verify installation
npm list ruv-swarm
```

## Security Best Practices

1. **Always pin specific versions** in package.json
2. **Never use `npx` with ruv-swarm** from untrusted sources
3. **Verify package integrity** before installation
4. **Use local installations** when possible
5. **Monitor security advisories** regularly

## Usage

### Local Installation Only

```javascript
// Use local installation
import { RuvSwarm } from 'ruv-swarm';

const swarm = new RuvSwarm({
  topology: 'hierarchical',
  maxAgents: 8,
  strategy: 'adaptive'
});

await swarm.init();
```

### MCP Server (Local Only)

```bash
# Run from local installation only
./node_modules/.bin/ruv-swarm mcp start
```

## Documentation

- [Security Policy](./SECURITY.md)
- [API Documentation](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm)
- [Examples](https://github.com/ruvnet/ruv-FANN/tree/main/examples)

## License

MIT

## Reporting Security Issues

Please report security vulnerabilities to:
- GitHub Security Advisory: https://github.com/ruvnet/ruv-FANN/security/advisories
- Issue Tracker: https://github.com/ruvnet/ruv-FANN/issues