import { spawn } from 'child_process';
import fs from 'fs-extra';
import path from 'path';
import os from 'os';

class ProcessManager {
  constructor(options = {}) {
    this.pidDir = options.pidDir || path.join(os.homedir(), '.ruv-swarm');
    this.processes = new Map();
    this.gracefulShutdownTimeout = options.gracefulShutdownTimeout || 10000;
    
    // Ensure PID directory exists
    fs.ensureDirSync(this.pidDir);
  }

  async start(name, options) {
    // Check if process is already running
    const status = await this.status(name);
    if (status.running) {
      throw new Error(`Process '${name}' is already running with PID ${status.pid}`);
    }

    // Clean up stale PID file if exists
    await this.cleanupStalePid(name);

    // Spawn the process
    const { command, args = [], detached = false, ...spawnOptions } = options;
    
    const childProcess = spawn(command, args, {
      detached,
      stdio: detached ? 'ignore' : 'inherit',
      ...spawnOptions
    });

    if (detached) {
      childProcess.unref();
    }

    const pid = childProcess.pid;
    const started = new Date().toISOString();

    // Store process reference
    this.processes.set(name, {
      process: childProcess,
      pid,
      started,
      detached
    });

    // Write PID file
    const pidFile = this.getPidFilePath(name);
    await fs.writeJson(pidFile, {
      pid,
      started,
      command,
      args
    });

    // Set up process exit handler
    childProcess.on('exit', async (code, signal) => {
      this.processes.delete(name);
      await fs.remove(pidFile);
    });

    return { pid, started };
  }

  async stop(name, options = {}) {
    const status = await this.status(name);
    if (!status.running) {
      throw new Error(`Process '${name}' is not running`);
    }

    const { gracefulTimeout = this.gracefulShutdownTimeout } = options;
    const processInfo = this.processes.get(name);
    
    let stopped = false;
    let forced = false;

    try {
      // Try graceful shutdown first
      process.kill(status.pid, 'SIGTERM');
      
      // Wait for graceful shutdown
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          // Force kill if graceful shutdown fails
          try {
            process.kill(status.pid, 'SIGKILL');
            forced = true;
            resolve();
          } catch (err) {
            reject(err);
          }
        }, gracefulTimeout);

        // Check if process exits gracefully
        const checkInterval = setInterval(() => {
          if (!this.isProcessRunning(status.pid)) {
            clearTimeout(timeout);
            clearInterval(checkInterval);
            resolve();
          }
        }, 100);
      });

      stopped = true;
    } catch (error) {
      // Process might have already exited
      if (error.code === 'ESRCH') {
        stopped = true;
      } else {
        throw error;
      }
    }

    // Clean up
    this.processes.delete(name);
    await fs.remove(this.getPidFilePath(name));

    return { stopped, pid: status.pid, forced };
  }

  async status(name) {
    const pidFile = this.getPidFilePath(name);
    
    try {
      const pidData = await fs.readJson(pidFile);
      const running = this.isProcessRunning(pidData.pid);
      
      if (!running) {
        // Clean up stale PID file
        await fs.remove(pidFile);
        return { running: false, pid: null };
      }

      const started = new Date(pidData.started);
      const uptime = Date.now() - started.getTime();

      return {
        running: true,
        pid: pidData.pid,
        started: pidData.started,
        uptime,
        command: pidData.command,
        args: pidData.args
      };
    } catch (error) {
      return { running: false, pid: null };
    }
  }

  async restart(name, options) {
    const status = await this.status(name);
    let oldPid = null;

    if (status.running) {
      oldPid = status.pid;
      await this.stop(name);
    }

    const result = await this.start(name, options);
    
    return {
      restarted: true,
      oldPid,
      newPid: result.pid
    };
  }

  async stopAll() {
    const names = Array.from(this.processes.keys());
    const results = [];

    for (const name of names) {
      try {
        const result = await this.stop(name);
        results.push({ name, ...result });
      } catch (error) {
        results.push({ name, error: error.message });
      }
    }

    return results;
  }

  isProcessRunning(pid) {
    try {
      // Send signal 0 to check if process exists
      process.kill(pid, 0);
      return true;
    } catch (error) {
      return false;
    }
  }

  getPidFilePath(name) {
    return path.join(this.pidDir, `${name}.pid`);
  }

  async cleanupStalePid(name) {
    const pidFile = this.getPidFilePath(name);
    
    try {
      const pidData = await fs.readJson(pidFile);
      if (!this.isProcessRunning(pidData.pid)) {
        await fs.remove(pidFile);
      }
    } catch (error) {
      // File doesn't exist, nothing to clean up
    }
  }

  async listProcesses() {
    const files = await fs.readdir(this.pidDir);
    const pidFiles = files.filter(f => f.endsWith('.pid'));
    const processes = [];

    for (const file of pidFiles) {
      const name = path.basename(file, '.pid');
      const status = await this.status(name);
      if (status.running) {
        processes.push({ name, ...status });
      }
    }

    return processes;
  }
}

export default ProcessManager;