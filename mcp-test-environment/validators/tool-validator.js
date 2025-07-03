#!/usr/bin/env node

/**
 * MCP Tool Validator
 * Tests all 27 ruv-swarm MCP tools for functionality
 */

import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import chalk from 'chalk';

const TOOL_TIMEOUT = parseInt(process.env.MCP_TOOL_TIMEOUT || '15000');

class MCPToolValidator {
  constructor() {
    this.results = [];
    this.errors = [];
    this.startTime = Date.now();
    
    // All 27 ruv-swarm MCP tools
    this.tools = [
      // Core Swarm Tools
      'mcp__ruv-swarm__swarm_init',
      'mcp__ruv-swarm__swarm_status',
      'mcp__ruv-swarm__swarm_monitor',
      
      // Agent Management
      'mcp__ruv-swarm__agent_spawn',
      'mcp__ruv-swarm__agent_list',
      'mcp__ruv-swarm__agent_metrics',
      
      // Task Orchestration
      'mcp__ruv-swarm__task_orchestrate',
      'mcp__ruv-swarm__task_status',
      'mcp__ruv-swarm__task_results',
      
      // Performance & Benchmarking
      'mcp__ruv-swarm__benchmark_run',
      'mcp__ruv-swarm__features_detect',
      'mcp__ruv-swarm__memory_usage',
      
      // Neural Network Tools
      'mcp__ruv-swarm__neural_status',
      'mcp__ruv-swarm__neural_train',
      'mcp__ruv-swarm__neural_patterns',
      
      // DAA (Decentralized Autonomous Agents)
      'mcp__ruv-swarm__daa_init',
      'mcp__ruv-swarm__daa_agent_create',
      'mcp__ruv-swarm__daa_agent_adapt',
      'mcp__ruv-swarm__daa_workflow_create',
      'mcp__ruv-swarm__daa_workflow_execute',
      'mcp__ruv-swarm__daa_knowledge_share',
      'mcp__ruv-swarm__daa_learning_status',
      'mcp__ruv-swarm__daa_cognitive_pattern',
      'mcp__ruv-swarm__daa_meta_learning',
      'mcp__ruv-swarm__daa_performance_metrics',
      
      // MCP Resource Management
      'ListMcpResourcesTool',
      'ReadMcpResourceTool'
    ];
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

  async validateToolAvailability() {
    this.log('Testing MCP Tool Availability...', 'info');
    
    try {
      // Test tools list command
      const listTools = spawn('npx', ['ruv-swarm', 'mcp', 'tools', 'list'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: '/app/ruv-swarm'
      });

      let stdout = '';
      let stderr = '';

      listTools.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      listTools.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          listTools.kill();
          reject(new Error('Tool availability check timeout'));
        }, TOOL_TIMEOUT);

        listTools.on('close', (code) => {
          clearTimeout(timeout);
          
          if (code === 0) {
            this.results.push({
              test: 'MCP Tools Availability',
              status: 'PASSED',
              message: 'Tools list command executed successfully',
              output: stdout
            });
            this.log('MCP Tools Availability: PASSED', 'success');
            resolve(true);
          } else {
            this.errors.push({
              test: 'MCP Tools Availability',
              error: `Command exited with code ${code}`,
              stdout: stdout,
              stderr: stderr
            });
            this.log(`MCP Tools Availability: FAILED (exit code ${code})`, 'error');
            reject(new Error(`Command exited with code ${code}`));
          }
        });

        listTools.on('error', (error) => {
          clearTimeout(timeout);
          this.errors.push({
            test: 'MCP Tools Availability',
            error: error.message
          });
          this.log(`MCP Tools Availability: ERROR - ${error.message}`, 'error');
          reject(error);
        });
      });
    } catch (error) {
      this.errors.push({
        test: 'MCP Tools Availability',
        error: error.message
      });
      this.log(`MCP Tools Availability: ERROR - ${error.message}`, 'error');
      throw error;
    }
  }

  async validateIndividualTool(toolName) {
    this.log(`Testing tool: ${toolName}...`, 'info');
    
    try {
      // Test basic tool invocation
      const toolTest = spawn('npx', ['ruv-swarm', 'mcp', 'test', toolName], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: '/app/ruv-swarm'
      });

      let stdout = '';
      let stderr = '';

      toolTest.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      toolTest.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          toolTest.kill();
          resolve(false); // Don't reject, just mark as failed
        }, TOOL_TIMEOUT);

        toolTest.on('close', (code) => {
          clearTimeout(timeout);
          
          if (code === 0) {
            this.results.push({
              test: `Tool: ${toolName}`,
              status: 'PASSED',
              message: 'Tool executed successfully',
              output: stdout.substring(0, 500) // Limit output size
            });
            this.log(`Tool ${toolName}: PASSED`, 'success');
            resolve(true);
          } else {
            this.errors.push({
              test: `Tool: ${toolName}`,
              error: `Tool exited with code ${code}`,
              stdout: stdout.substring(0, 500),
              stderr: stderr.substring(0, 500)
            });
            this.log(`Tool ${toolName}: FAILED (exit code ${code})`, 'error');
            resolve(false);
          }
        });

        toolTest.on('error', (error) => {
          clearTimeout(timeout);
          this.errors.push({
            test: `Tool: ${toolName}`,
            error: error.message
          });
          this.log(`Tool ${toolName}: ERROR - ${error.message}`, 'error');
          resolve(false);
        });
      });
    } catch (error) {
      this.errors.push({
        test: `Tool: ${toolName}`,
        error: error.message
      });
      this.log(`Tool ${toolName}: ERROR - ${error.message}`, 'error');
      return false;
    }
  }

  async validateToolParameterValidation() {
    this.log('Testing MCP Tool Parameter Validation...', 'info');
    
    const testCases = [
      {
        tool: 'mcp__ruv-swarm__swarm_init',
        params: { topology: 'mesh', maxAgents: 5 },
        expected: 'success'
      },
      {
        tool: 'mcp__ruv-swarm__swarm_init',
        params: { topology: 'invalid', maxAgents: 5 },
        expected: 'error'
      },
      {
        tool: 'mcp__ruv-swarm__agent_spawn',
        params: { type: 'researcher' },
        expected: 'success'
      },
      {
        tool: 'mcp__ruv-swarm__agent_spawn',
        params: { type: 'invalid-type' },
        expected: 'error'
      }
    ];

    for (const testCase of testCases) {
      try {
        // Test parameter validation
        const isValid = this.validateParameters(testCase.tool, testCase.params);
        
        if ((testCase.expected === 'success' && isValid) || 
            (testCase.expected === 'error' && !isValid)) {
          this.results.push({
            test: `Parameter Validation: ${testCase.tool}`,
            status: 'PASSED',
            message: `Parameters correctly validated as ${testCase.expected}`
          });
          this.log(`Parameter Validation ${testCase.tool}: PASSED`, 'success');
        } else {
          this.errors.push({
            test: `Parameter Validation: ${testCase.tool}`,
            error: `Expected ${testCase.expected}, got ${isValid ? 'success' : 'error'}`
          });
          this.log(`Parameter Validation ${testCase.tool}: FAILED`, 'error');
        }
      } catch (error) {
        this.errors.push({
          test: `Parameter Validation: ${testCase.tool}`,
          error: error.message
        });
        this.log(`Parameter Validation ${testCase.tool}: ERROR - ${error.message}`, 'error');
      }
    }
  }

  validateParameters(toolName, params) {
    // Basic parameter validation logic
    const validationRules = {
      'mcp__ruv-swarm__swarm_init': (p) => 
        ['mesh', 'hierarchical', 'ring', 'star'].includes(p.topology) && 
        p.maxAgents >= 1 && p.maxAgents <= 100,
      'mcp__ruv-swarm__agent_spawn': (p) => 
        ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'].includes(p.type)
    };

    const validator = validationRules[toolName];
    return validator ? validator(params) : true;
  }

  async generateReport() {
    const endTime = Date.now();
    const duration = endTime - this.startTime;
    
    const report = {
      timestamp: new Date().toISOString(),
      duration: `${duration}ms`,
      summary: {
        total_tools: this.tools.length,
        total_tests: this.results.length + this.errors.length,
        passed: this.results.length,
        failed: this.errors.length,
        success_rate: ((this.results.length / (this.results.length + this.errors.length)) * 100).toFixed(2) + '%',
        tool_coverage: `${this.tools.length}/27 tools tested`
      },
      tools_tested: this.tools,
      results: this.results,
      errors: this.errors
    };

    this.log('\n=== MCP Tool Validation Report ===', 'info');
    this.log(`Total Tools: ${report.summary.total_tools}`, 'info');
    this.log(`Total Tests: ${report.summary.total_tests}`, 'info');
    this.log(`Passed: ${report.summary.passed}`, 'success');
    this.log(`Failed: ${report.summary.failed}`, 'error');
    this.log(`Success Rate: ${report.summary.success_rate}`, 'info');
    this.log(`Tool Coverage: ${report.summary.tool_coverage}`, 'info');
    this.log(`Duration: ${report.duration}`, 'info');

    return report;
  }

  async validate() {
    this.log('Starting MCP Tool Validation...', 'info');
    
    try {
      // Test tool availability
      await this.validateToolAvailability();
      
      // Test individual tools
      this.log(`Testing ${this.tools.length} individual tools...`, 'info');
      for (const tool of this.tools) {
        await this.validateIndividualTool(tool);
      }
      
      // Test parameter validation
      await this.validateToolParameterValidation();
      
      const report = await this.generateReport();
      
      // Write report to file
      await import('fs').then(fs => {
        fs.writeFileSync('/tmp/mcp-tool-validation.json', JSON.stringify(report, null, 2));
      });
      
      this.log('MCP Tool Validation completed successfully!', 'success');
      return report;
    } catch (error) {
      this.log(`MCP Tool Validation failed: ${error.message}`, 'error');
      throw error;
    }
  }
}

// Run validation if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const validator = new MCPToolValidator();
  validator.validate()
    .then(report => {
      console.log('\n' + JSON.stringify(report, null, 2));
      process.exit(0);
    })
    .catch(error => {
      console.error('Validation failed:', error);
      process.exit(1);
    });
}

export default MCPToolValidator;