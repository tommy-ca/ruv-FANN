# Contributing to ruv-swarm

We welcome contributions to ruv-swarm! This guide will help you get started with contributing to the project.

## 🚀 Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ruv-FANN.git
   cd ruv-FANN
   ```

2. **Set Up Development Environment**
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install Node.js (for NPM package)
   nvm install 18
   nvm use 18
   
   # Install dependencies
   cargo build
   cd ruv-swarm/npm && npm install
   ```

3. **Run Tests**
   ```bash
   # Rust tests
   cargo test --workspace --all-features
   
   # Node.js tests
   cd ruv-swarm/npm && npm test
   ```

## 🏗️ Project Structure

```
ruv-swarm/
├── crates/                    # Rust crates
│   ├── ruv-swarm-core/       # Core swarm functionality
│   ├── ruv-swarm-cli/        # Command-line interface
│   ├── ruv-swarm-mcp/        # MCP server implementation
│   └── ...
├── npm/                       # Node.js package
│   ├── src/                  # TypeScript/JavaScript source
│   ├── test/                 # Test files
│   └── package.json
├── docs/                      # Documentation
├── examples/                  # Example projects
└── tests/                     # Integration tests
```

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/my-awesome-feature
```

### 2. Make Changes
- Follow the [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
cargo test --all-features
cd ruv-swarm/npm && npm test

# Run specific tests
cargo test onboarding
npm test -- test/onboarding
```

### 4. Submit a Pull Request
- Write a clear description of your changes
- Reference any related issues
- Ensure all CI checks pass

## 📝 Coding Standards

### Rust Code
- Follow `rustfmt` formatting: `cargo fmt`
- Pass `clippy` lints: `cargo clippy`
- Write comprehensive tests
- Document public APIs with doc comments

### JavaScript/TypeScript
- Use Prettier for formatting: `npm run format`
- Follow ESLint rules: `npm run lint`
- Write unit and integration tests
- Use TypeScript for type safety

### Documentation
- Write clear, concise documentation
- Include code examples
- Update relevant guides when adding features
- Use proper markdown formatting

## 🧪 Testing Guidelines

### Test Types
1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

### Test Structure
```rust
// Rust test example
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swarm_initialization() {
        let swarm = Swarm::new("test".to_string(), Topology::Mesh);
        assert_eq!(swarm.id, "test");
    }
}
```

```javascript
// JavaScript test example
describe('RuvSwarm', () => {
    test('should initialize with correct config', async () => {
        const swarm = new RuvSwarm({ topology: 'mesh' });
        await swarm.initialize();
        expect(swarm.getStatus().topology).toBe('mesh');
    });
});
```

## 🔍 Code Review Process

### Before Submitting
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No breaking changes (unless intentional)

### Review Criteria
- **Functionality**: Does the code work as intended?
- **Performance**: Is the code efficient?
- **Security**: Are there any security concerns?
- **Maintainability**: Is the code easy to understand and modify?

## 📋 Issue Guidelines

### Reporting Bugs
Use the bug report template and include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Rust version, etc.)

### Feature Requests
Use the feature request template and include:
- Problem description
- Proposed solution
- Alternative solutions considered
- Additional context

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to docs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## 🚀 Release Process

### Version Bumping
We follow [Semantic Versioning](https://semver.org/):
- **Patch** (0.0.x): Bug fixes
- **Minor** (0.x.0): New features, backward compatible
- **Major** (x.0.0): Breaking changes

### Release Checklist
- [ ] Update version in `Cargo.toml` and `package.json`
- [ ] Update `CHANGELOG.md`
- [ ] Create release PR
- [ ] Tag release after merge
- [ ] Publish to crates.io and npm

## 🎯 Areas for Contribution

### High Priority
- [ ] Performance optimizations
- [ ] Cross-platform compatibility
- [ ] Error handling improvements
- [ ] Documentation enhancements

### Medium Priority
- [ ] Additional agent types
- [ ] New orchestration strategies
- [ ] Monitoring improvements
- [ ] Integration examples

### Good First Issues
- [ ] Fix typos in documentation
- [ ] Add unit tests
- [ ] Improve error messages
- [ ] Add configuration validation

## 💬 Communication

### Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and discussion

### Code of Conduct
Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We expect all contributors to:
- Be respectful and inclusive
- Provide constructive feedback
- Help create a welcoming environment
- Focus on what's best for the community

## 🛠️ Development Tools

### Recommended VS Code Extensions
```json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "tamasfe.even-better-toml",
    "ms-vscode.vscode-typescript-next",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-eslint"
  ]
}
```

### Git Hooks (Optional)
```bash
# Install pre-commit hooks
echo '#!/bin/sh\ncargo fmt && cargo clippy' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## 📚 Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Testing in Rust](https://doc.rust-lang.org/book/ch11-00-testing.html)

## 🙏 Recognition

Contributors will be recognized in:
- Release notes for significant contributions
- Contributors section in README
- Special thanks for first-time contributors

Thank you for helping make ruv-swarm better! 🚀