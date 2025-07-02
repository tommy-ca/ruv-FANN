# Installation Guide

Complete installation guide for ruv-swarm across all supported platforms.

## 🚀 Quick Install (Recommended)

### NPM Global Install
```bash
npm install -g ruv-swarm
ruv-swarm init  # Automatic setup with onboarding
```

### Cargo Install
```bash
cargo install ruv-swarm-cli
ruv-swarm init
```

## 📦 Installation Methods

### 1. NPM Package Manager

#### Global Installation
```bash
# Install globally
npm install -g ruv-swarm

# Verify installation
ruv-swarm --version

# Initialize with onboarding
ruv-swarm init
```

#### Local Project Installation
```bash
# Install in project
npm install ruv-swarm

# Use via npx
npx ruv-swarm init

# Or add to package.json scripts
{
  "scripts": {
    "swarm:init": "ruv-swarm init",
    "swarm:start": "ruv-swarm orchestrate"
  }
}
```

### 2. Rust Cargo

#### From crates.io
```bash
# Install CLI
cargo install ruv-swarm-cli

# Install core library (for development)
cargo add ruv-swarm-core
```

#### From Source
```bash
# Clone repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN

# Build and install
cargo install --path ruv-swarm/crates/ruv-swarm-cli
```

### 3. Binary Downloads

#### GitHub Releases
Download pre-built binaries from [GitHub Releases](https://github.com/ruvnet/ruv-FANN/releases):

**Linux (x86_64)**
```bash
curl -L https://github.com/ruvnet/ruv-FANN/releases/latest/download/ruv-swarm-linux-x86_64.tar.gz | tar xz
sudo mv ruv-swarm /usr/local/bin/
```

**macOS (x86_64)**
```bash
curl -L https://github.com/ruvnet/ruv-FANN/releases/latest/download/ruv-swarm-macos-x86_64.tar.gz | tar xz
sudo mv ruv-swarm /usr/local/bin/
```

**macOS (ARM64)**
```bash
curl -L https://github.com/ruvnet/ruv-FANN/releases/latest/download/ruv-swarm-macos-arm64.tar.gz | tar xz
sudo mv ruv-swarm /usr/local/bin/
```

**Windows**
```powershell
# Download and extract ruv-swarm-windows-x86_64.zip
# Add to PATH or place in a directory already in PATH
```

## 🐳 Docker Installation

### Official Docker Image
```bash
# Pull image
docker pull ruvnet/ruv-swarm:latest

# Run with volume mount
docker run -v $(pwd):/workspace ruvnet/ruv-swarm:latest init

# Interactive mode
docker run -it ruvnet/ruv-swarm:latest bash
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  ruv-swarm:
    image: ruvnet/ruv-swarm:latest
    volumes:
      - .:/workspace
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
    command: ["orchestrate", "parallel", "Build application"]
```

## 🔧 Platform-Specific Instructions

### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Node.js (if using NPM method)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Rust (if using Cargo method)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install ruv-swarm
npm install -g ruv-swarm
# OR
cargo install ruv-swarm-cli
```

### CentOS/RHEL/Fedora
```bash
# Install Node.js
sudo dnf install nodejs npm
# OR for older versions
sudo yum install nodejs npm

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install ruv-swarm
npm install -g ruv-swarm
```

### macOS
```bash
# Using Homebrew (recommended)
brew install node
brew install rust

# Install ruv-swarm
npm install -g ruv-swarm

# Using MacPorts
sudo port install nodejs18
sudo port install rust

# Install ruv-swarm
npm install -g ruv-swarm
```

### Windows

#### Using winget
```powershell
# Install Node.js
winget install OpenJS.NodeJS

# Install Rust
winget install Rustlang.Rust.MSVC

# Install ruv-swarm
npm install -g ruv-swarm
```

#### Using Chocolatey
```powershell
# Install Node.js and Rust
choco install nodejs rust

# Install ruv-swarm
npm install -g ruv-swarm
```

#### Manual Installation
1. Download Node.js from [nodejs.org](https://nodejs.org/)
2. Download Rust from [rustup.rs](https://rustup.rs/)
3. Install both following their setup wizards
4. Open PowerShell/Command Prompt and run:
   ```powershell
   npm install -g ruv-swarm
   ```

## 🔐 Prerequisites & Dependencies

### System Requirements
- **Memory**: 512MB minimum, 2GB recommended
- **Disk**: 100MB for binaries, 1GB for workspace
- **Network**: Internet connection for package downloads

### Required Dependencies
- **Node.js**: 16.x or higher (for NPM install)
- **Rust**: 1.70.0 or higher (for Cargo install)
- **Git**: For repository operations

### Optional Dependencies
- **Docker**: For containerized deployment
- **Claude Code**: Auto-installed during onboarding
- **GitHub CLI**: For enhanced GitHub integration

## ⚙️ Post-Installation Setup

### 1. Verify Installation
```bash
# Check version
ruv-swarm --version

# Check available commands
ruv-swarm --help

# Run health check
ruv-swarm status
```

### 2. Initial Configuration
```bash
# Run guided onboarding (recommended)
ruv-swarm init

# Manual configuration
ruv-swarm init --skip-onboarding
```

### 3. Environment Setup
```bash
# Set GitHub token (optional but recommended)
export GITHUB_TOKEN="your_token_here"

# Set default configuration
export SWARM_TOPOLOGY="mesh"
export SWARM_PERSISTENCE="sqlite"
```

## 🔍 Troubleshooting Installation

### Common Issues

#### Permission Denied (Linux/macOS)
```bash
# Fix NPM permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Then reinstall
npm install -g ruv-swarm
```

#### Cargo Build Fails
```bash
# Update Rust
rustup update

# Clear cargo cache
cargo clean

# Retry installation
cargo install ruv-swarm-cli
```

#### Windows PowerShell Execution Policy
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then retry NPM install
npm install -g ruv-swarm
```

### Verification Commands
```bash
# Test CLI functionality
ruv-swarm --version
ruv-swarm status

# Test Node.js integration
node -e "const swarm = require('ruv-swarm'); console.log(swarm.getVersion());"

# Test Rust integration
cargo test --package ruv-swarm-core
```

## 🔄 Updating ruv-swarm

### NPM Update
```bash
npm update -g ruv-swarm
```

### Cargo Update
```bash
cargo install ruv-swarm-cli --force
```

### Binary Update
Download the latest release and replace the existing binary.

## 🗑️ Uninstallation

### NPM Uninstall
```bash
npm uninstall -g ruv-swarm
```

### Cargo Uninstall
```bash
cargo uninstall ruv-swarm-cli
```

### Manual Cleanup
```bash
# Remove configuration
rm -rf ~/.ruv-swarm

# Remove binary (if manually installed)
sudo rm /usr/local/bin/ruv-swarm
```

## 📚 Next Steps

After installation:
1. **[Getting Started Guide](../guides/getting-started.md)** - Your first swarm
2. **[Seamless Onboarding](../guides/seamless-onboarding.md)** - Automated setup
3. **[Configuration Guide](../guides/configuration.md)** - Customize your setup
4. **[API Reference](../api/core.md)** - Explore the APIs

## 🆘 Need Help?

- **Installation Issues**: [Troubleshooting Guide](../troubleshooting/installation.md)
- **Platform Issues**: [Platform-Specific Help](../troubleshooting/platform-issues.md)
- **General Help**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)