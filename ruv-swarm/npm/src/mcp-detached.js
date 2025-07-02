import ProcessManager from './process-manager.js';
import path from 'path';
import fs from 'fs-extra';
import http from 'http';
import os from 'os';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class MCPDetached {
  constructor(options = {}) {
    this.dataDir = options.dataDir || path.join(os.homedir(), '.ruv-swarm');
    this.healthCheckPort = options.healthCheckPort || 9898;
    this.gracefulShutdownTimeout = options.gracefulShutdownTimeout || 10000;
    
    this.processManager = new ProcessManager({
      pidDir: this.dataDir,
      gracefulShutdownTimeout: this.gracefulShutdownTimeout
    });

    this.processName = 'mcp-server';
    this.healthServer = null;
    
    // Ensure data directory exists
    fs.ensureDirSync(this.dataDir);
  }

  async start(options = {}) {
    const { detached = true } = options;
    
    // Check if already running
    const status = await this.status();
    if (status.running) {
      throw new Error(`MCP server is already running with PID ${status.pid}`);
    }

    // Get the path to the MCP binary/script
    const mcpPath = this.getMCPExecutablePath();
    
    if (detached) {
      // Start MCP server in detached mode using the wrapper
      const result = await this.processManager.start(this.processName, {
        command: process.execPath,
        args: [mcpPath],
        detached: true,
        env: {
          ...process.env,
          MCP_HEALTH_PORT: this.healthCheckPort.toString(),
          MCP_MODE: 'detached',
          NODE_ENV: 'production'
        }
      });

      return {
        ...result,
        status: 'started',
        mode: 'detached',
        healthCheckPort: this.healthCheckPort
      };
    } else {
      // Run in foreground (for testing or direct execution)
      const { spawn } = await import('child_process');
      const child = spawn(process.execPath, [mcpPath, 'mcp', 'start'], {
        stdio: 'inherit',
        env: {
          ...process.env,
          MCP_MODE: 'foreground'
        }
      });

      return {
        status: 'started',
        mode: 'foreground',
        process: child
      };
    }
  }

  async stop() {
    const status = await this.status();
    
    if (!status.running) {
      return {
        stopped: false,
        reason: 'not running'
      };
    }

    // Stop health server if running
    if (this.healthServer) {
      await new Promise((resolve) => {
        this.healthServer.close(resolve);
      });
      this.healthServer = null;
    }

    // Stop MCP process
    const result = await this.processManager.stop(this.processName);
    
    return result;
  }

  async status() {
    const processStatus = await this.processManager.status(this.processName);
    
    if (!processStatus.running) {
      return {
        running: false,
        pid: null,
        mode: null
      };
    }

    // Try to get health status
    let health = null;
    try {
      health = await this.checkHealth();
    } catch (error) {
      // Health check failed, but process is running
    }

    return {
      ...processStatus,
      mode: 'detached',
      health
    };
  }

  async restart() {
    const status = await this.status();
    let oldPid = null;

    if (status.running) {
      oldPid = status.pid;
      await this.stop();
    }

    const result = await this.start({ detached: true });
    
    return {
      restarted: true,
      oldPid,
      newPid: result.pid
    };
  }

  async checkHealth() {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: 'localhost',
        port: this.healthCheckPort,
        path: '/health',
        method: 'GET',
        timeout: 5000
      };

      const req = http.request(options, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          try {
            const health = JSON.parse(data);
            resolve(health);
          } catch (error) {
            reject(new Error('Invalid health response'));
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Health check timeout'));
      });

      req.end();
    });
  }

  async startHealthServer() {
    if (this.healthServer) {
      return; // Already running
    }

    this.healthServer = http.createServer((req, res) => {
      if (req.url === '/health' && req.method === 'GET') {
        const status = {
          status: 'healthy',
          uptime: process.uptime(),
          timestamp: new Date().toISOString(),
          mode: 'detached'
        };

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(status));
      } else {
        res.writeHead(404);
        res.end('Not Found');
      }
    });

    await new Promise((resolve, reject) => {
      this.healthServer.listen(this.healthCheckPort, 'localhost', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  getMCPExecutablePath() {
    // For detached mode, use the wrapper script
    const wrapperPath = path.join(__dirname, 'mcp-server-wrapper.js');
    
    if (!fs.existsSync(wrapperPath)) {
      // Fallback to direct CLI script
      const binPath = path.join(__dirname, '..', 'bin', 'ruv-swarm-clean.js');
      if (!fs.existsSync(binPath)) {
        throw new Error(`MCP executable not found`);
      }
      return binPath;
    }

    return wrapperPath;
  }

  getConfig() {
    return {
      dataDir: this.dataDir,
      healthCheckPort: this.healthCheckPort,
      gracefulShutdownTimeout: this.gracefulShutdownTimeout,
      processName: this.processName
    };
  }

  // Handle process signals for graceful shutdown
  setupSignalHandlers() {
    const signals = ['SIGTERM', 'SIGINT'];
    
    signals.forEach(signal => {
      process.on(signal, async () => {
        console.log(`Received ${signal}, shutting down gracefully...`);
        
        try {
          await this.stop();
          process.exit(0);
        } catch (error) {
          console.error('Error during shutdown:', error);
          process.exit(1);
        }
      });
    });
  }
}

export default MCPDetached;