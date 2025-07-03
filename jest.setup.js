// Jest setup file for Node.js environment with ES modules support

// Set test environment flag BEFORE any imports
process.env.CLAUDE_FLOW_ENV = 'test';
process.env.NODE_ENV = 'test';

// Configure global test environment
global.console = {
  ...console,
  // Suppress verbose console output during tests unless in verbose mode
  log: process.env.JEST_VERBOSE ? console.log : jest.fn(),
  debug: process.env.JEST_VERBOSE ? console.debug : jest.fn(),
  info: process.env.JEST_VERBOSE ? console.info : jest.fn(),
  warn: console.warn, // Keep warnings visible
  error: console.error, // Keep errors visible
};

// Handle unhandled rejections in tests
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Mock logger configuration for tests
jest.mock('./src/core/logger.js', () => {
  const mockLogger = {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    configure: jest.fn(),
    child: jest.fn(() => mockLogger),
    close: jest.fn()
  };
  
  return {
    Logger: jest.fn(() => mockLogger),
    logger: mockLogger,
    LogLevel: {
      DEBUG: 0,
      INFO: 1,
      WARN: 2,
      ERROR: 3
    }
  };
});

// Mock ruv-swarm for tests that don't need real swarm functionality
jest.mock('ruv-swarm', () => ({
  SwarmManager: jest.fn().mockImplementation(() => ({
    initialize: jest.fn(),
    terminate: jest.fn(),
    getStatus: jest.fn().mockReturnValue({ active: false })
  })),
  Agent: jest.fn(),
  Memory: jest.fn(),
  default: jest.fn()
}), { virtual: true });

// Mock HTTP transport to avoid import.meta issues
jest.mock('./src/mcp/transports/http.js', () => ({
  HttpTransport: jest.fn().mockImplementation(() => ({
    start: jest.fn(),
    stop: jest.fn(),
    onRequest: jest.fn(),
    onNotification: jest.fn(),
    getHealthStatus: jest.fn().mockResolvedValue({
      healthy: true,
      metrics: { messagesReceived: 0, notificationsSent: 0 }
    })
  }))
}));

// Mock ClaudeAPI service will be done in individual test files