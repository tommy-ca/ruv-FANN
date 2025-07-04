# Claude Code Integration Guide for ruv-FANN

## Project Overview

**ruv-FANN** is an advanced AI agent orchestration platform combining multiple cutting-edge technologies:

- **Claude-Flow**: Advanced AI agent orchestration system v1.0.71
- **DAA (Decentralized Autonomous Architecture)**: Rust-based distributed system
- **QUDAG (Quantum DAG)**: Blockchain/P2P networking protocol
- **RUV-Swarm**: Multi-agent coordination and swarm intelligence
- **MCP Integration**: Full Model Context Protocol implementation
- **WASM Support**: WebAssembly modules for performance optimization
- **Distributed Training**: Federated learning and gradient sharing

## Architecture

### Core Components

1. **AI Agent Orchestration**
   - Multi-agent spawning and coordination
   - SPARC methodology implementation (17 modes)
   - Memory management and persistence
   - Task orchestration and workflow execution

2. **Distributed Systems**
   - Rust-based DAA orchestrator
   - P2P networking with QUDAG protocol
   - Distributed training coordination
   - Blockchain consensus mechanisms

3. **Runtime Environment**
   - **Primary**: Deno (TypeScript/JavaScript)
   - **Secondary**: Node.js compatibility
   - **Performance**: Rust binaries
   - **Web**: WebAssembly modules

4. **Integration Layer**
   - MCP server/client implementation
   - GitHub API integration
   - Claude API integration
   - Terminal virtualization

## Code Conventions

### Language Standards
- **TypeScript**: Strict mode, comprehensive typing
- **Rust**: 2021 edition, clippy compliance
- **JavaScript**: ES modules, async/await patterns
- **WASM**: Rust-compiled with wasm-pack

### Naming Conventions
- **Files**: kebab-case (e.g., `agent-manager.ts`)
- **Functions**: camelCase (e.g., `spawnAgent`)
- **Variables**: camelCase (e.g., `agentConfig`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_AGENTS`)
- **Classes**: PascalCase (e.g., `AgentManager`)
- **Interfaces**: PascalCase with 'I' prefix (e.g., `IAgentConfig`)

### Code Style
- **Indentation**: 2 spaces
- **Line Length**: 100 characters max
- **Semicolons**: Always use in TypeScript/JavaScript
- **Quotes**: Single quotes for strings, double quotes for JSX
- **Trailing Commas**: Always in multi-line objects/arrays

### Design Patterns
- **Dependency Injection**: Constructor injection preferred
- **Observer Pattern**: Event-driven architecture
- **Factory Pattern**: Agent creation and configuration
- **Singleton Pattern**: Global state management (sparingly)
- **Strategy Pattern**: Multiple AI model support

## Directory Structure

```
ruv-FANN/
├── src/                          # TypeScript source code
│   ├── cli/                      # Command-line interface
│   ├── agents/                   # Agent management
│   ├── memory/                   # Memory systems
│   └── workflows/                # Workflow execution
├── claude-code-flow/             # Main orchestration system
│   ├── src/                      # Core implementation
│   ├── mcp_config/               # MCP server configurations
│   ├── .claude/                  # Claude integration
│   └── bin/                      # Executable binaries
├── daa-repository/               # Rust DAA implementation
│   ├── src/                      # Rust source code
│   ├── docker/                   # Containerization
│   └── qudag/                    # Blockchain protocol
├── ruv-swarm/                    # Swarm intelligence
│   ├── src/                      # Rust implementation
│   ├── target/                   # Compiled binaries
│   └── wasm/                     # WebAssembly modules
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── memory/                       # Persistent memory
├── docs/                         # Documentation
└── scripts/                      # Build and utility scripts
```

## Development Workflow

### Prerequisites
1. **Deno**: Primary runtime environment
   ```bash
   curl -fsSL https://deno.land/x/install/install.sh | sh
   ```

2. **Rust**: System programming language
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Node.js**: Compatibility layer
   ```bash
   # Via nvm (recommended)
   nvm install 20
   ```

4. **Docker**: Containerization
   ```bash
   # Install Docker Engine
   curl -fsSL https://get.docker.com | sh
   ```

### Setup Process
1. **Environment Setup**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd ruv-FANN
   
   # Install dependencies
   npm install
   
   # Build Rust components
   cd daa-repository && cargo build --release
   
   # Build WASM modules
   cd ruv-swarm && wasm-pack build --target web
   ```

2. **Configuration**
   ```bash
   # Initialize Claude integration
   ./claude-code-flow/claude-code-flow/bin/claude-flow init --sparc
   
   # Configure MCP server
   cp claude-code-flow/claude-code-flow/mcp_config/mcp.json .
   
   # Set environment variables
   export CLAUDE_API_KEY="your-key"
   export GITHUB_TOKEN="your-token"
   ```

3. **Testing**
   ```bash
   # Run comprehensive tests
   ./scripts/test-all-capabilities.sh
   
   # Run specific test suites
   deno test --allow-all tests/
   cargo test --workspace
   ```

4. **Development Server**
   ```bash
   # Start MCP server
   ./claude-code-flow/claude-code-flow/bin/claude-flow mcp start --port 3000
   
   # Start orchestration
   ./claude-code-flow/claude-code-flow/bin/claude-flow start --ui
   ```

## Testing Strategy

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Multi-component interaction
3. **E2E Tests**: Complete workflow validation
4. **Performance Tests**: Load and stress testing
5. **Security Tests**: Vulnerability assessment

### Testing Commands
```bash
# All tests
npm test

# Unit tests only
deno test --allow-all tests/unit/

# Integration tests
deno test --allow-all tests/integration/

# Rust tests
cargo test --workspace

# Docker-based testing
docker-compose -f docker-compose.test.yml up --build
```

## Important Considerations

### Security
- **API Keys**: Never commit sensitive credentials
- **Authentication**: Use token-based auth for MCP
- **Network**: Secure communication channels
- **Data**: Encrypt sensitive memory storage
- **Containers**: Run with minimal privileges

### Performance
- **Memory Management**: Efficient agent lifecycle
- **Concurrency**: Async/await patterns
- **Caching**: Redis for session management
- **Load Balancing**: Distribute agent workloads
- **Monitoring**: Real-time performance metrics

### Compatibility
- **Multi-Platform**: Linux, macOS, Windows
- **Multi-Runtime**: Deno, Node.js, Docker
- **Multi-Language**: TypeScript, Rust, WASM
- **Multi-Transport**: stdio, HTTP, WebSocket

## Command Categories

### Core Commands
- `init`: Initialize project with Claude integration
- `start`: Start orchestration system
- `status`: Show system status
- `config`: Configuration management

### Agent Management
- `agent spawn <type>`: Create new AI agents
- `agent list`: List active agents
- `agent terminate <id>`: Stop specific agent
- `agent info <id>`: Show agent details

### Memory Operations
- `memory query <pattern>`: Search memory
- `memory store <key> <value>`: Store information
- `memory export <file>`: Export memory data
- `memory import <file>`: Import memory data

### SPARC Development
- `sparc modes`: List available modes
- `sparc run <mode> <task>`: Execute SPARC mode
- `sparc tdd <feature>`: Test-driven development
- `sparc architect <system>`: System architecture

### MCP Integration
- `mcp start`: Start MCP server
- `mcp status`: Check server status
- `mcp tools`: List available tools
- `mcp config`: Show configuration

### Swarm Coordination
- `swarm create <config>`: Create swarm
- `swarm join <id>`: Join existing swarm
- `swarm coordinate`: Multi-agent coordination
- `swarm status`: Show swarm health

## Dependencies

### Core Dependencies
- **@anthropic/claude-cli**: Claude API integration
- **commander**: CLI framework
- **express**: HTTP server
- **socket.io**: WebSocket communication
- **redis**: Session management
- **postgresql**: Data persistence

### Rust Dependencies
- **tokio**: Async runtime
- **serde**: Serialization
- **reqwest**: HTTP client
- **sqlx**: Database access
- **wasm-bindgen**: WebAssembly bindings

### Development Dependencies
- **typescript**: Type checking
- **deno**: Runtime environment
- **cargo**: Rust build system
- **wasm-pack**: WebAssembly toolchain
- **docker**: Containerization

## Troubleshooting

### Common Issues

1. **Deno Not Found**
   ```bash
   # Install Deno
   curl -fsSL https://deno.land/x/install/install.sh | sh
   export PATH="$HOME/.deno/bin:$PATH"
   ```

2. **MCP Server Won't Start**
   ```bash
   # Check port availability
   lsof -i :3000
   
   # Try different port
   ./claude-code-flow/claude-code-flow/bin/claude-flow mcp start --port 3001
   ```

3. **Rust Build Failures**
   ```bash
   # Update Rust toolchain
   rustup update
   
   # Clean and rebuild
   cargo clean && cargo build --release
   ```

4. **WASM Compilation Issues**
   ```bash
   # Install wasm-pack
   cargo install wasm-pack
   
   # Build with target
   wasm-pack build --target web
   ```

5. **Memory Issues**
   ```bash
   # Clear memory cache
   ./claude-code-flow/claude-code-flow/bin/claude-flow memory cleanup
   
   # Restart services
   ./claude-code-flow/claude-code-flow/bin/claude-flow restart
   ```

## Integration Points

### Claude Code Integration
- **Commands**: Available in `.claude/commands/`
- **Memory**: Persistent across sessions
- **Configuration**: `.claude/settings.local.json`
- **Templates**: Pre-configured workflows

### GitHub Integration
- **Actions**: Automated CI/CD pipelines
- **Issues**: Task management
- **Pull Requests**: Code review
- **Releases**: Version management

### Docker Integration
- **Multi-stage builds**: Optimized images
- **Development**: Hot reload support
- **Production**: Minimal footprint
- **Testing**: Isolated environments

## Advanced Features

### Distributed Training
- **Federated Learning**: Privacy-preserving ML
- **Gradient Sharing**: P2P coordination
- **Model Aggregation**: Consensus mechanisms
- **Performance Monitoring**: Training metrics

### Blockchain Integration
- **QUDAG Protocol**: Quantum-resistant consensus
- **Smart Contracts**: Automated execution
- **P2P Networking**: Decentralized communication
- **Cryptographic Security**: Advanced encryption

### AI Agent Orchestration
- **Multi-Model Support**: Various AI providers
- **Context Management**: Conversation history
- **Task Scheduling**: Intelligent workload distribution
- **Performance Optimization**: Resource allocation

## Version Information

- **ruv-FANN**: v1.0.0
- **Claude-Flow**: v1.0.71
- **DAA**: v0.1.3
- **RUV-Swarm**: v1.0.5
- **QUDAG**: v0.1.0

## Support and Resources

- **Documentation**: `/docs/` directory
- **Examples**: `/examples/` directory
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Wiki**: Comprehensive guides

---

*This integration guide is automatically updated. Last updated: 2025-07-04*