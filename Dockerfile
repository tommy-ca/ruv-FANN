# ruv-FANN: Multi-Stage Docker Build for Complete AI Agent Orchestration Platform
# This Dockerfile creates a comprehensive testing environment for all capabilities

# Stage 1: Base Environment with Multiple Runtimes
FROM ubuntu:24.04 as base
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    nodejs \
    npm \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Rust Installation
FROM base as rust-builder
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default stable
RUN rustup component add rustfmt clippy
RUN cargo install wasm-pack

# Stage 3: Deno Installation
FROM rust-builder as deno-builder
RUN curl -fsSL https://deno.land/x/install/install.sh | sh
ENV PATH="/root/.deno/bin:${PATH}"

# Stage 4: Node.js Tools
FROM deno-builder as node-builder
RUN npm install -g tsx typescript @types/node
RUN npm install -g @anthropic/claude-cli

# Stage 5: Development Environment
FROM node-builder as development
WORKDIR /workspace
COPY . .

# Install Node.js dependencies
RUN npm install || true

# Build Rust components
RUN cd daa-repository && cargo build --release || true

# Build WASM components
RUN cd ruv-swarm && wasm-pack build --target web || true

# Install Deno dependencies
RUN deno cache --reload src/cli/main.ts || true

# Stage 6: Testing Environment
FROM development as testing
RUN mkdir -p /test-results
ENV TEST_RESULTS_DIR="/test-results"
ENV DOCKER_TEST_MODE="true"

# Copy test configuration
COPY tests/ /workspace/tests/
COPY scripts/ /workspace/scripts/

# Make scripts executable
RUN chmod +x /workspace/scripts/* || true
RUN chmod +x /workspace/bin/* || true

# Stage 7: Production Environment
FROM node-builder as production
WORKDIR /app
COPY --from=development /workspace/dist/ ./dist/
COPY --from=development /workspace/bin/ ./bin/
COPY --from=development /workspace/claude-code-flow/claude-code-flow/bin/ ./claude-code-flow/bin/
COPY --from=development /workspace/target/release/ ./target/release/
COPY package.json ./
COPY deno.json ./
RUN npm ci --only=production || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["./bin/claude-flow", "--help"]