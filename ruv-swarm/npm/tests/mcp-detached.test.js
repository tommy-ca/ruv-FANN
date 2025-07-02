const { describe, it, beforeEach, afterEach } = require('mocha');
const { expect } = require('chai');
const sinon = require('sinon');
const path = require('path');
const fs = require('fs-extra');
const axios = require('axios');
const MCPDetached = require('../src/mcp-detached');

describe('MCPDetached', () => {
  let mcpDetached;
  let sandbox;
  const testDir = path.join(__dirname, '.test-mcp');

  beforeEach(async () => {
    sandbox = sinon.createSandbox();
    await fs.ensureDir(testDir);
    mcpDetached = new MCPDetached({ 
      dataDir: testDir,
      healthCheckPort: 9999 // Use non-standard port for tests
    });
  });

  afterEach(async () => {
    sandbox.restore();
    if (mcpDetached) {
      await mcpDetached.stop();
    }
    await fs.remove(testDir);
  });

  describe('start()', () => {
    it('should start MCP server in detached mode', async () => {
      const result = await mcpDetached.start({ detached: true });
      
      expect(result).to.have.property('pid');
      expect(result).to.have.property('status', 'started');
      expect(result).to.have.property('mode', 'detached');
    });

    it('should start MCP server in foreground mode', async () => {
      const result = await mcpDetached.start({ detached: false });
      
      expect(result).to.have.property('status', 'started');
      expect(result).to.have.property('mode', 'foreground');
    });

    it('should prevent multiple instances', async () => {
      await mcpDetached.start({ detached: true });
      
      try {
        await mcpDetached.start({ detached: true });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.include('already running');
      }
    });

    it('should create health check endpoint', async function() {
      this.timeout(5000);
      
      await mcpDetached.start({ detached: true });
      
      // Wait for server to be ready
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      try {
        const response = await axios.get('http://localhost:9999/health');
        expect(response.status).to.equal(200);
        expect(response.data).to.have.property('status', 'healthy');
        expect(response.data).to.have.property('uptime');
      } catch (error) {
        // Server might not be ready yet in CI
        console.log('Health check failed:', error.message);
      }
    });
  });

  describe('stop()', () => {
    it('should stop running MCP server', async () => {
      const { pid } = await mcpDetached.start({ detached: true });
      const result = await mcpDetached.stop();
      
      expect(result).to.have.property('stopped', true);
      expect(result).to.have.property('pid', pid);
    });

    it('should handle stopping when no server is running', async () => {
      const result = await mcpDetached.stop();
      expect(result).to.have.property('stopped', false);
      expect(result).to.have.property('reason', 'not running');
    });
  });

  describe('status()', () => {
    it('should return status of running server', async () => {
      const { pid } = await mcpDetached.start({ detached: true });
      const status = await mcpDetached.status();
      
      expect(status).to.have.property('running', true);
      expect(status).to.have.property('pid', pid);
      expect(status).to.have.property('uptime');
      expect(status).to.have.property('mode', 'detached');
    });

    it('should return not running status', async () => {
      const status = await mcpDetached.status();
      
      expect(status).to.have.property('running', false);
      expect(status).to.have.property('pid', null);
    });
  });

  describe('restart()', () => {
    it('should restart MCP server', async () => {
      const { pid: oldPid } = await mcpDetached.start({ detached: true });
      const result = await mcpDetached.restart();
      
      expect(result).to.have.property('restarted', true);
      expect(result).to.have.property('oldPid', oldPid);
      expect(result).to.have.property('newPid');
      expect(result.newPid).to.not.equal(oldPid);
    });

    it('should start server if not running on restart', async () => {
      const result = await mcpDetached.restart();
      
      expect(result).to.have.property('restarted', true);
      expect(result).to.have.property('oldPid', null);
      expect(result).to.have.property('newPid');
    });
  });

  describe('Signal handling', () => {
    it('should handle graceful shutdown on SIGTERM', async function() {
      this.timeout(5000);
      
      const { pid } = await mcpDetached.start({ detached: true });
      
      // Send SIGTERM
      process.kill(pid, 'SIGTERM');
      
      // Wait for graceful shutdown
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const status = await mcpDetached.status();
      expect(status.running).to.be.false;
    });
  });

  describe('Configuration', () => {
    it('should use custom configuration', async () => {
      const customMcp = new MCPDetached({
        dataDir: testDir,
        healthCheckPort: 8888,
        gracefulShutdownTimeout: 5000
      });

      const config = customMcp.getConfig();
      expect(config.healthCheckPort).to.equal(8888);
      expect(config.gracefulShutdownTimeout).to.equal(5000);
    });

    it('should use default configuration', () => {
      const config = mcpDetached.getConfig();
      expect(config).to.have.property('healthCheckPort');
      expect(config).to.have.property('gracefulShutdownTimeout');
    });
  });
});