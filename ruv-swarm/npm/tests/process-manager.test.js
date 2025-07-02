const { describe, it, beforeEach, afterEach } = require('mocha');
const { expect } = require('chai');
const sinon = require('sinon');
const fs = require('fs-extra');
const path = require('path');
const { spawn } = require('child_process');
const ProcessManager = require('../src/process-manager');

describe('ProcessManager', () => {
  let processManager;
  let sandbox;
  const testPidDir = path.join(__dirname, '.test-pids');

  beforeEach(async () => {
    sandbox = sinon.createSandbox();
    await fs.ensureDir(testPidDir);
    processManager = new ProcessManager({ pidDir: testPidDir });
  });

  afterEach(async () => {
    sandbox.restore();
    await fs.remove(testPidDir);
    // Clean up any spawned processes
    if (processManager) {
      await processManager.stopAll();
    }
  });

  describe('start()', () => {
    it('should start a process in detached mode', async () => {
      const result = await processManager.start('mcp-server', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'], // Sleep for 30s
        detached: true
      });

      expect(result).to.have.property('pid');
      expect(result).to.have.property('started');
      expect(result.pid).to.be.a('number');
      expect(result.pid).to.be.greaterThan(0);
    });

    it('should write PID file when starting detached process', async () => {
      const result = await processManager.start('test-process', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      const pidFile = path.join(testPidDir, 'test-process.pid');
      expect(await fs.pathExists(pidFile)).to.be.true;

      const pidData = await fs.readJson(pidFile);
      expect(pidData.pid).to.equal(result.pid);
      expect(pidData.started).to.exist;
    });

    it('should prevent starting duplicate processes', async () => {
      await processManager.start('duplicate-test', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      try {
        await processManager.start('duplicate-test', {
          command: 'node',
          args: ['-e', 'console.log("should not run")'],
          detached: true
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.include('already running');
      }
    });

    it('should clean up stale PID files', async () => {
      // Write a fake PID file with non-existent process
      const stalePidFile = path.join(testPidDir, 'stale-process.pid');
      await fs.writeJson(stalePidFile, {
        pid: 999999, // Unlikely to exist
        started: new Date().toISOString()
      });

      // Should succeed as it cleans up the stale PID
      const result = await processManager.start('stale-process', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      expect(result.pid).to.not.equal(999999);
    });
  });

  describe('stop()', () => {
    it('should stop a running process gracefully', async () => {
      const { pid } = await processManager.start('stop-test', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      const result = await processManager.stop('stop-test');
      expect(result.stopped).to.be.true;
      expect(result.pid).to.equal(pid);

      // Verify PID file is removed
      const pidFile = path.join(testPidDir, 'stop-test.pid');
      expect(await fs.pathExists(pidFile)).to.be.false;
    });

    it('should handle stopping non-existent process', async () => {
      try {
        await processManager.stop('non-existent');
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.include('not running');
      }
    });

    it('should force kill if graceful shutdown fails', async function() {
      this.timeout(15000); // Extend timeout for this test

      // Start a process that ignores SIGTERM
      const { pid } = await processManager.start('stubborn-process', {
        command: 'node',
        args: ['-e', `
          process.on('SIGTERM', () => console.log('Ignoring SIGTERM'));
          setTimeout(() => {}, 60000);
        `],
        detached: true
      });

      const result = await processManager.stop('stubborn-process', {
        gracefulTimeout: 2000 // 2 seconds before force kill
      });

      expect(result.stopped).to.be.true;
      expect(result.forced).to.be.true;
    });
  });

  describe('status()', () => {
    it('should return status of running process', async () => {
      const { pid } = await processManager.start('status-test', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      const status = await processManager.status('status-test');
      expect(status.running).to.be.true;
      expect(status.pid).to.equal(pid);
      expect(status.uptime).to.be.a('number');
    });

    it('should return not running for non-existent process', async () => {
      const status = await processManager.status('non-existent');
      expect(status.running).to.be.false;
      expect(status.pid).to.be.null;
    });
  });

  describe('restart()', () => {
    it('should restart a running process', async () => {
      const { pid: originalPid } = await processManager.start('restart-test', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      const result = await processManager.restart('restart-test', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      expect(result.oldPid).to.equal(originalPid);
      expect(result.newPid).to.not.equal(originalPid);
      expect(result.restarted).to.be.true;
    });
  });

  describe('isProcessRunning()', () => {
    it('should correctly identify running processes', async () => {
      const { pid } = await processManager.start('running-check', {
        command: 'node',
        args: ['-e', 'setTimeout(() => {}, 30000)'],
        detached: true
      });

      expect(processManager.isProcessRunning(pid)).to.be.true;
      expect(processManager.isProcessRunning(999999)).to.be.false;
    });
  });
});