#!/bin/bash

# Start MCP Server Script - Environment Agnostic
# This script ensures the MCP server starts correctly across different environments

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPM_DIR="$(dirname "$SCRIPT_DIR")"
SWARM_ROOT="$(dirname "$NPM_DIR")"
MCP_DIR="$SWARM_ROOT/crates/ruv-swarm-mcp"

echo "🚀 Starting MCP Server..."
echo "NPM Dir: $NPM_DIR"
echo "MCP Dir: $MCP_DIR"

# Check if MCP directory exists
if [ ! -d "$MCP_DIR" ]; then
    echo "❌ Error: MCP server directory not found at $MCP_DIR"
    exit 1
fi

# Check if Cargo is available
if ! command -v cargo &> /dev/null; then
    echo "🔧 Setting up Rust environment..."
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo "❌ Error: Rust/Cargo not found. Please install Rust first:"
        echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
fi

# Verify cargo is now available
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Cargo still not available after sourcing environment"
    exit 1
fi

echo "✅ Cargo found: $(cargo --version)"

# Change to MCP directory and start server
cd "$MCP_DIR"

echo "🔧 Building and starting MCP server..."
cargo run --release

echo "✅ MCP server started successfully"