#!/usr/bin/env node

/**
 * MCP Performance Benchmarks
 * Tests MCP operation performance and throughput
 */

import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import chalk from 'chalk';

const PERFORMANCE_SAMPLES = parseInt(process.env.PERFORMANCE_SAMPLES || '100');
const BENCHMARK_ITERATIONS = parseInt(process.env.BENCHMARK_ITERATIONS || '10');

class MCPPerformanceBenchmarks {
  constructor() {
    this.results = [];
    this.errors = [];
    this.startTime = Date.now();
    this.benchmarks = [];
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const colorMap = {
      info: chalk.blue,
      success: chalk.green,
      error: chalk.red,
      warning: chalk.yellow
    };
    
    console.log(`[${timestamp}] ${colorMap[type](message)}`);
  }

  async measureLatency(operation, iterations = 10) {
    const latencies = [];
    
    for (let i = 0; i < iterations; i++) {
      const start = process.hrtime.bigint();
      
      try {
        await operation();
        const end = process.hrtime.bigint();
        const latency = Number(end - start) / 1000000; // Convert to milliseconds
        latencies.push(latency);
      } catch (error) {
        this.log(`Latency measurement failed: ${error.message}`, 'error');
        latencies.push(null);
      }
    }

    // Filter out null values
    const validLatencies = latencies.filter(l => l !== null);
    
    if (validLatencies.length === 0) {
      return {
        min: null,
        max: null,
        avg: null,
        median: null,
        p95: null,
        p99: null,
        success_rate: 0
      };
    }

    validLatencies.sort((a, b) => a - b);
    
    return {
      min: validLatencies[0],
      max: validLatencies[validLatencies.length - 1],
      avg: validLatencies.reduce((a, b) => a + b, 0) / validLatencies.length,
      median: validLatencies[Math.floor(validLatencies.length / 2)],
      p95: validLatencies[Math.floor(validLatencies.length * 0.95)],
      p99: validLatencies[Math.floor(validLatencies.length * 0.99)],
      success_rate: validLatencies.length / iterations
    };
  }

  async benchmarkSwarmInit() {
    this.log('Benchmarking Swarm Initialization...', 'info');
    
    const operation = async () => {
      return new Promise((resolve, reject) => {
        const initCmd = spawn('npx', ['ruv-swarm', 'init', 'mesh', '5'], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: '/app/ruv-swarm'
        });

        const timeout = setTimeout(() => {
          initCmd.kill();
          reject(new Error('Swarm init timeout'));
        }, 10000);

        initCmd.on('close', (code) => {
          clearTimeout(timeout);
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Swarm init failed with code ${code}`));
          }
        });

        initCmd.on('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });
    };

    try {
      const metrics = await this.measureLatency(operation, BENCHMARK_ITERATIONS);
      
      this.benchmarks.push({
        operation: 'Swarm Initialization',
        metrics: metrics,
        status: metrics.success_rate > 0.8 ? 'PASSED' : 'FAILED'
      });
      
      this.log(`Swarm Init Benchmark: ${metrics.success_rate > 0.8 ? 'PASSED' : 'FAILED'}`, 
               metrics.success_rate > 0.8 ? 'success' : 'error');
    } catch (error) {
      this.errors.push({
        benchmark: 'Swarm Initialization',
        error: error.message
      });
      this.log(`Swarm Init Benchmark: ERROR - ${error.message}`, 'error');
    }
  }

  async benchmarkAgentSpawn() {
    this.log('Benchmarking Agent Spawning...', 'info');
    
    const operation = async () => {
      return new Promise((resolve, reject) => {
        const spawnCmd = spawn('npx', ['ruv-swarm', 'spawn', 'researcher'], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: '/app/ruv-swarm'
        });

        const timeout = setTimeout(() => {
          spawnCmd.kill();
          reject(new Error('Agent spawn timeout'));
        }, 5000);

        spawnCmd.on('close', (code) => {
          clearTimeout(timeout);
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Agent spawn failed with code ${code}`));
          }
        });

        spawnCmd.on('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });
    };

    try {
      const metrics = await this.measureLatency(operation, BENCHMARK_ITERATIONS);
      
      this.benchmarks.push({
        operation: 'Agent Spawning',
        metrics: metrics,
        status: metrics.success_rate > 0.8 ? 'PASSED' : 'FAILED'
      });
      
      this.log(`Agent Spawn Benchmark: ${metrics.success_rate > 0.8 ? 'PASSED' : 'FAILED'}`, 
               metrics.success_rate > 0.8 ? 'success' : 'error');
    } catch (error) {
      this.errors.push({
        benchmark: 'Agent Spawning',
        error: error.message
      });
      this.log(`Agent Spawn Benchmark: ERROR - ${error.message}`, 'error');
    }
  }

  async benchmarkTaskOrchestration() {
    this.log('Benchmarking Task Orchestration...', 'info');
    
    const operation = async () => {
      return new Promise((resolve, reject) => {
        const orchestrateCmd = spawn('npx', ['ruv-swarm', 'orchestrate', 'test task'], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: '/app/ruv-swarm'
        });

        const timeout = setTimeout(() => {
          orchestrateCmd.kill();
          reject(new Error('Task orchestration timeout'));
        }, 15000);

        orchestrateCmd.on('close', (code) => {
          clearTimeout(timeout);
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Task orchestration failed with code ${code}`));
          }
        });

        orchestrateCmd.on('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });
    };

    try {
      const metrics = await this.measureLatency(operation, Math.min(BENCHMARK_ITERATIONS, 5));
      
      this.benchmarks.push({
        operation: 'Task Orchestration',
        metrics: metrics,
        status: metrics.success_rate > 0.6 ? 'PASSED' : 'FAILED'
      });
      
      this.log(`Task Orchestration Benchmark: ${metrics.success_rate > 0.6 ? 'PASSED' : 'FAILED'}`, 
               metrics.success_rate > 0.6 ? 'success' : 'error');
    } catch (error) {
      this.errors.push({
        benchmark: 'Task Orchestration',
        error: error.message
      });
      this.log(`Task Orchestration Benchmark: ERROR - ${error.message}`, 'error');
    }
  }

  async benchmarkMemoryOperations() {
    this.log('Benchmarking Memory Operations...', 'info');
    
    const operation = async () => {
      return new Promise((resolve, reject) => {
        const memCmd = spawn('npx', ['ruv-swarm', 'memory', 'usage'], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: '/app/ruv-swarm'
        });

        const timeout = setTimeout(() => {
          memCmd.kill();
          reject(new Error('Memory operation timeout'));
        }, 3000);

        memCmd.on('close', (code) => {
          clearTimeout(timeout);
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Memory operation failed with code ${code}`));
          }
        });

        memCmd.on('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });
    };

    try {
      const metrics = await this.measureLatency(operation, BENCHMARK_ITERATIONS);
      
      this.benchmarks.push({
        operation: 'Memory Operations',
        metrics: metrics,
        status: metrics.success_rate > 0.9 ? 'PASSED' : 'FAILED'
      });
      
      this.log(`Memory Operations Benchmark: ${metrics.success_rate > 0.9 ? 'PASSED' : 'FAILED'}`, 
               metrics.success_rate > 0.9 ? 'success' : 'error');
    } catch (error) {
      this.errors.push({
        benchmark: 'Memory Operations',
        error: error.message
      });
      this.log(`Memory Operations Benchmark: ERROR - ${error.message}`, 'error');
    }
  }

  async benchmarkThroughput() {
    this.log('Benchmarking MCP Throughput...', 'info');
    
    const startTime = Date.now();
    let operationsCompleted = 0;
    let operationsFailed = 0;
    
    const promises = [];
    
    for (let i = 0; i < PERFORMANCE_SAMPLES; i++) {
      const operation = async () => {
        try {
          const statusCmd = spawn('npx', ['ruv-swarm', 'status'], {
            stdio: ['pipe', 'pipe', 'pipe'],
            cwd: '/app/ruv-swarm'
          });

          return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
              statusCmd.kill();
              reject(new Error('Status timeout'));
            }, 5000);

            statusCmd.on('close', (code) => {
              clearTimeout(timeout);
              if (code === 0) {
                operationsCompleted++;
                resolve();
              } else {
                operationsFailed++;
                reject(new Error(`Status failed with code ${code}`));
              }
            });

            statusCmd.on('error', (error) => {
              clearTimeout(timeout);
              operationsFailed++;
              reject(error);
            });
          });
        } catch (error) {
          operationsFailed++;
          throw error;
        }
      };

      promises.push(operation());
    }

    try {
      await Promise.allSettled(promises);
      
      const endTime = Date.now();
      const duration = (endTime - startTime) / 1000; // Convert to seconds
      const throughput = operationsCompleted / duration;
      
      this.benchmarks.push({
        operation: 'MCP Throughput',
        metrics: {
          operations_completed: operationsCompleted,
          operations_failed: operationsFailed,
          total_operations: PERFORMANCE_SAMPLES,
          duration_seconds: duration,
          throughput_ops_per_second: throughput,
          success_rate: operationsCompleted / PERFORMANCE_SAMPLES
        },
        status: throughput > 5 ? 'PASSED' : 'FAILED'
      });
      
      this.log(`MCP Throughput Benchmark: ${throughput > 5 ? 'PASSED' : 'FAILED'} (${throughput.toFixed(2)} ops/sec)`, 
               throughput > 5 ? 'success' : 'error');
    } catch (error) {
      this.errors.push({
        benchmark: 'MCP Throughput',
        error: error.message
      });
      this.log(`MCP Throughput Benchmark: ERROR - ${error.message}`, 'error');
    }
  }

  async generateReport() {
    const endTime = Date.now();
    const duration = endTime - this.startTime;
    
    const passedBenchmarks = this.benchmarks.filter(b => b.status === 'PASSED').length;
    const failedBenchmarks = this.benchmarks.filter(b => b.status === 'FAILED').length;
    
    const report = {
      timestamp: new Date().toISOString(),
      duration: `${duration}ms`,
      summary: {
        total_benchmarks: this.benchmarks.length,
        passed: passedBenchmarks,
        failed: failedBenchmarks,
        errors: this.errors.length,
        success_rate: ((passedBenchmarks / this.benchmarks.length) * 100).toFixed(2) + '%'
      },
      benchmarks: this.benchmarks,
      errors: this.errors
    };

    this.log('\n=== MCP Performance Benchmark Report ===', 'info');
    this.log(`Total Benchmarks: ${report.summary.total_benchmarks}`, 'info');
    this.log(`Passed: ${report.summary.passed}`, 'success');
    this.log(`Failed: ${report.summary.failed}`, 'error');
    this.log(`Errors: ${report.summary.errors}`, 'error');
    this.log(`Success Rate: ${report.summary.success_rate}`, 'info');
    this.log(`Duration: ${report.duration}`, 'info');

    return report;
  }

  async benchmark() {
    this.log('Starting MCP Performance Benchmarks...', 'info');
    
    try {
      await this.benchmarkSwarmInit();
      await this.benchmarkAgentSpawn();
      await this.benchmarkTaskOrchestration();
      await this.benchmarkMemoryOperations();
      await this.benchmarkThroughput();
      
      const report = await this.generateReport();
      
      // Write report to file
      await import('fs').then(fs => {
        fs.writeFileSync('/tmp/mcp-performance-benchmarks.json', JSON.stringify(report, null, 2));
      });
      
      this.log('MCP Performance Benchmarks completed successfully!', 'success');
      return report;
    } catch (error) {
      this.log(`MCP Performance Benchmarks failed: ${error.message}`, 'error');
      throw error;
    }
  }
}

// Run benchmarks if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const benchmarker = new MCPPerformanceBenchmarks();
  benchmarker.benchmark()
    .then(report => {
      console.log('\n' + JSON.stringify(report, null, 2));
      process.exit(0);
    })
    .catch(error => {
      console.error('Benchmarks failed:', error);
      process.exit(1);
    });
}

export default MCPPerformanceBenchmarks;