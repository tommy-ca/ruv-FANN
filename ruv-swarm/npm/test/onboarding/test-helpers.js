/**
 * Shared test helpers and utilities for onboarding tests
 */

const fs = require('fs').promises;
const path = require('path');
const os = require('os');

/**
 * Creates a temporary test directory
 */
async function createTestDir(prefix = 'onboarding-test-') {
  return await fs.mkdtemp(path.join(os.tmpdir(), prefix));
}

/**
 * Cleans up a test directory
 */
async function cleanupTestDir(dir) {
  try {
    await fs.rm(dir, { recursive: true, force: true });
  } catch (error) {
    console.warn(`Failed to cleanup test dir ${dir}:`, error);
  }
}

/**
 * Creates a mock Claude executable
 */
async function createMockClaude(dir, options = {}) {
  const {
    name = 'claude',
    version = '1.0.0',
    exitCode = 0,
  } = options;

  const binDir = path.join(dir, 'bin');
  await fs.mkdir(binDir, { recursive: true });

  const claudePath = path.join(binDir, name);
  
  if (process.platform === 'win32') {
    const batContent = `@echo off
if "%1"=="--version" (
  echo Claude Code v${version}
  exit /b ${exitCode}
)
echo Mock Claude`;
    await fs.writeFile(claudePath + '.bat', batContent);
    return claudePath + '.bat';
  } else {
    const shContent = `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Claude Code v${version}"
  exit ${exitCode}
fi
echo "Mock Claude"`;
    await fs.writeFile(claudePath, shContent);
    await fs.chmod(claudePath, 0o755);
    return claudePath;
  }
}

/**
 * Mock implementation of child_process.exec
 */
class MockExec {
  constructor() {
    this.responses = new Map();
    this.history = [];
    this.defaultResponse = { stdout: '', stderr: '', error: null };
  }

  addResponse(pattern, response) {
    this.responses.set(pattern, response);
    return this;
  }

  setDefault(response) {
    this.defaultResponse = response;
    return this;
  }

  exec(command, callback) {
    this.history.push({ command, timestamp: Date.now() });

    // Find matching response
    for (const [pattern, response] of this.responses) {
      if (typeof pattern === 'string' && command.includes(pattern)) {
        return this._respond(callback, response);
      } else if (pattern instanceof RegExp && pattern.test(command)) {
        return this._respond(callback, response);
      }
    }

    // Use default response
    return this._respond(callback, this.defaultResponse);
  }

  _respond(callback, response) {
    process.nextTick(() => {
      if (response.error) {
        callback(response.error, response.stdout || '', response.stderr || '');
      } else {
        callback(null, response.stdout || '', response.stderr || '');
      }
    });
  }

  getHistory() {
    return this.history;
  }

  wasCalledWith(pattern) {
    return this.history.some(call => 
      typeof pattern === 'string' 
        ? call.command.includes(pattern)
        : pattern.test(call.command)
    );
  }

  reset() {
    this.history = [];
    this.responses.clear();
  }
}

/**
 * Mock file system watcher
 */
class MockFSWatcher {
  constructor() {
    this.listeners = new Map();
  }

  on(event, listener) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(listener);
    return this;
  }

  emit(event, ...args) {
    const listeners = this.listeners.get(event) || [];
    listeners.forEach(listener => listener(...args));
  }

  close() {
    this.listeners.clear();
  }
}

/**
 * Test data fixtures
 */
const fixtures = {
  validMcpConfig: {
    servers: {
      'ruv-swarm': {
        command: 'npx',
        args: ['ruv-swarm', 'mcp', 'start'],
        stdio: true,
      },
    },
  },

  invalidMcpConfig: {
    servers: {
      'invalid': {
        // Missing command field
        args: ['test'],
      },
    },
  },

  claudeVersionOutputs: [
    { output: 'Claude Code v1.0.0\n', version: '1.0.0' },
    { output: 'claude version 2.0.0-beta\n', version: '2.0.0-beta' },
    { output: 'Version: 3.1.0\nBuild: 12345\n', version: '3.1.0' },
  ],

  errorResponses: {
    eacces: {
      error: new Error('EACCES: permission denied'),
      stderr: 'npm ERR! Error: EACCES: permission denied',
    },
    enotfound: {
      error: new Error('ENOTFOUND'),
      stderr: 'getaddrinfo ENOTFOUND registry.npmjs.org',
    },
    etimedout: {
      error: new Error('ETIMEDOUT'),
      stderr: 'network timeout',
    },
  },
};

/**
 * Assertion helpers
 */
async function assertFileContains(filePath, content) {
  const fileContent = await fs.readFile(filePath, 'utf-8');
  expect(fileContent).toContain(content);
}

async function assertFileExists(filePath) {
  const exists = await fs.access(filePath).then(() => true).catch(() => false);
  expect(exists).toBe(true);
}

async function assertFileNotExists(filePath) {
  const exists = await fs.access(filePath).then(() => true).catch(() => false);
  expect(exists).toBe(false);
}

function assertCommandCalled(mockExec, pattern) {
  expect(mockExec.wasCalledWith(pattern)).toBe(true);
}

function assertCommandNotCalled(mockExec, pattern) {
  expect(mockExec.wasCalledWith(pattern)).toBe(false);
}

/**
 * Wait helpers
 */
function waitFor(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitUntil(condition, timeout = 5000, interval = 100) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await condition()) {
      return true;
    }
    await waitFor(interval);
  }
  throw new Error('Timeout waiting for condition');
}

/**
 * Platform helpers
 */
function mockPlatform(platform) {
  const original = process.platform;
  Object.defineProperty(process, 'platform', {
    value: platform,
    configurable: true,
  });
  return () => {
    Object.defineProperty(process, 'platform', {
      value: original,
      configurable: true,
    });
  };
}

/**
 * Environment helpers
 */
function mockEnv(vars) {
  const original = { ...process.env };
  Object.assign(process.env, vars);
  return () => {
    process.env = original;
  };
}

module.exports = {
  createTestDir,
  cleanupTestDir,
  createMockClaude,
  MockExec,
  MockFSWatcher,
  fixtures,
  assertFileContains,
  assertFileExists,
  assertFileNotExists,
  assertCommandCalled,
  assertCommandNotCalled,
  waitFor,
  waitUntil,
  mockPlatform,
  mockEnv,
};