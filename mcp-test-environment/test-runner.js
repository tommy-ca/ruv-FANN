#!/usr/bin/env node

/**
 * MCP Test Runner
 * Orchestrates all MCP validation tests and generates comprehensive report
 */

import MCPProtocolValidator from './validators/protocol-validator.js';
import MCPToolValidator from './validators/tool-validator.js';
import MCPPerformanceBenchmarks from './performance/mcp-benchmarks.js';
import MCPIntegrationTests from './integration/mcp-integration-tests.js';
import chalk from 'chalk';
import fs from 'fs';
import Table from 'cli-table3';

class MCPTestRunner {
  constructor() {
    this.results = {};
    this.startTime = Date.now();
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const colorMap = {
      info: chalk.blue,
      success: chalk.green,
      error: chalk.red,
      warning: chalk.yellow,
      header: chalk.cyan.bold
    };
    
    console.log(`[${timestamp}] ${colorMap[type](message)}`);
  }

  async runProtocolValidation() {
    this.log('Running Protocol Validation...', 'header');
    
    try {
      const validator = new MCPProtocolValidator();
      const result = await validator.validate();
      this.results.protocol = { status: 'PASSED', result };
      this.log('Protocol Validation: COMPLETED', 'success');
    } catch (error) {
      this.results.protocol = { status: 'FAILED', error: error.message };
      this.log(`Protocol Validation: FAILED - ${error.message}`, 'error');
    }
  }

  async runToolValidation() {
    this.log('Running Tool Validation...', 'header');
    
    try {
      const validator = new MCPToolValidator();
      const result = await validator.validate();
      this.results.tools = { status: 'PASSED', result };
      this.log('Tool Validation: COMPLETED', 'success');
    } catch (error) {
      this.results.tools = { status: 'FAILED', error: error.message };
      this.log(`Tool Validation: FAILED - ${error.message}`, 'error');
    }
  }

  async runPerformanceBenchmarks() {
    this.log('Running Performance Benchmarks...', 'header');
    
    try {
      const benchmarker = new MCPPerformanceBenchmarks();
      const result = await benchmarker.benchmark();
      this.results.performance = { status: 'PASSED', result };
      this.log('Performance Benchmarks: COMPLETED', 'success');
    } catch (error) {
      this.results.performance = { status: 'FAILED', error: error.message };
      this.log(`Performance Benchmarks: FAILED - ${error.message}`, 'error');
    }
  }

  async runIntegrationTests() {
    this.log('Running Integration Tests...', 'header');
    
    try {
      const tester = new MCPIntegrationTests();
      const result = await tester.test();
      this.results.integration = { status: 'PASSED', result };
      this.log('Integration Tests: COMPLETED', 'success');
    } catch (error) {
      this.results.integration = { status: 'FAILED', error: error.message };
      this.log(`Integration Tests: FAILED - ${error.message}`, 'error');
    }
  }

  generateSummaryTable() {
    const table = new Table({
      head: ['Test Suite', 'Status', 'Tests', 'Passed', 'Failed', 'Success Rate'],
      colWidths: [20, 12, 8, 8, 8, 12]
    });

    Object.entries(this.results).forEach(([suite, data]) => {
      if (data.status === 'PASSED' && data.result) {
        const summary = data.result.summary;
        table.push([
          suite.charAt(0).toUpperCase() + suite.slice(1),
          chalk.green('PASSED'),
          summary.total_tests || summary.total || summary.total_benchmarks || 0,
          summary.passed || 0,
          summary.failed || summary.errors || 0,
          summary.success_rate || '0%'
        ]);
      } else {
        table.push([
          suite.charAt(0).toUpperCase() + suite.slice(1),
          chalk.red('FAILED'),
          '0',
          '0',
          '1',
          '0%'
        ]);
      }
    });

    return table.toString();
  }

  generateDetailedReport() {
    const endTime = Date.now();
    const duration = endTime - this.startTime;
    
    // Calculate overall statistics
    let totalTests = 0;
    let totalPassed = 0;
    let totalFailed = 0;
    let allSuitesPassed = true;

    Object.values(this.results).forEach(suite => {
      if (suite.status === 'FAILED') {
        allSuitesPassed = false;
        totalFailed += 1;
      } else if (suite.result) {
        const summary = suite.result.summary;
        totalTests += summary.total_tests || summary.total || summary.total_benchmarks || 0;
        totalPassed += summary.passed || 0;
        totalFailed += summary.failed || summary.errors || 0;
      }
    });

    const overallSuccessRate = totalTests > 0 ? ((totalPassed / totalTests) * 100).toFixed(2) : '0';

    const report = {
      timestamp: new Date().toISOString(),
      duration: `${duration}ms`,
      environment: {
        node_version: process.version,
        platform: process.platform,
        arch: process.arch,
        cwd: process.cwd()
      },
      overall_summary: {
        all_suites_passed: allSuitesPassed,
        total_test_suites: Object.keys(this.results).length,
        total_tests: totalTests,
        total_passed: totalPassed,
        total_failed: totalFailed,
        overall_success_rate: `${overallSuccessRate}%`,
        duration: `${duration}ms`
      },
      test_suites: this.results,
      recommendations: this.generateRecommendations()
    };

    return report;
  }

  generateRecommendations() {
    const recommendations = [];

    // Check protocol validation
    if (this.results.protocol?.status === 'FAILED') {
      recommendations.push({
        category: 'Protocol',
        priority: 'HIGH',
        issue: 'MCP protocol validation failed',
        solution: 'Check MCP server startup and protocol compliance'
      });
    }

    // Check tool validation
    if (this.results.tools?.status === 'FAILED') {
      recommendations.push({
        category: 'Tools',
        priority: 'HIGH',
        issue: 'MCP tools validation failed',
        solution: 'Verify all 27 ruv-swarm MCP tools are properly registered and functional'
      });
    }

    // Check performance
    if (this.results.performance?.status === 'FAILED') {
      recommendations.push({
        category: 'Performance',
        priority: 'MEDIUM',
        issue: 'Performance benchmarks failed',
        solution: 'Investigate performance bottlenecks and optimize MCP operations'
      });
    }

    // Check integration
    if (this.results.integration?.status === 'FAILED') {
      recommendations.push({
        category: 'Integration',
        priority: 'HIGH',
        issue: 'Integration tests failed',
        solution: 'Verify claude-code-flow and ruv-swarm integration configuration'
      });
    }

    // Success recommendations
    if (recommendations.length === 0) {
      recommendations.push({
        category: 'Success',
        priority: 'INFO',
        issue: 'All tests passed',
        solution: 'MCP server validation completed successfully - ready for production use'
      });
    }

    return recommendations;
  }

  async generateGitHubIssueReport() {
    const report = this.generateDetailedReport();
    const allPassed = report.overall_summary.all_suites_passed;
    
    const issueTitle = allPassed 
      ? '✅ MCP Validation: All Tests Passed' 
      : '❌ MCP Validation: Issues Found';

    const issueBody = `
# MCP Server and Protocol Validation Report

**Validation Date:** ${report.timestamp}
**Duration:** ${report.duration}
**Overall Status:** ${allPassed ? '✅ PASSED' : '❌ FAILED'}

## Summary

- **Total Test Suites:** ${report.overall_summary.total_test_suites}
- **Total Tests:** ${report.overall_summary.total_tests}
- **Tests Passed:** ${report.overall_summary.total_passed}
- **Tests Failed:** ${report.overall_summary.total_failed}
- **Success Rate:** ${report.overall_summary.overall_success_rate}

## Test Results

### Protocol Validation
${this.results.protocol?.status === 'PASSED' ? '✅' : '❌'} **${this.results.protocol?.status}**
${this.results.protocol?.status === 'FAILED' ? `Error: ${this.results.protocol.error}` : ''}

### Tool Validation (27 MCP Tools)
${this.results.tools?.status === 'PASSED' ? '✅' : '❌'} **${this.results.tools?.status}**
${this.results.tools?.status === 'FAILED' ? `Error: ${this.results.tools.error}` : ''}

### Performance Benchmarks
${this.results.performance?.status === 'PASSED' ? '✅' : '❌'} **${this.results.performance?.status}**
${this.results.performance?.status === 'FAILED' ? `Error: ${this.results.performance.error}` : ''}

### Integration Tests
${this.results.integration?.status === 'PASSED' ? '✅' : '❌'} **${this.results.integration?.status}**
${this.results.integration?.status === 'FAILED' ? `Error: ${this.results.integration.error}` : ''}

## Environment

- **Node.js:** ${report.environment.node_version}
- **Platform:** ${report.environment.platform}
- **Architecture:** ${report.environment.arch}

## Recommendations

${report.recommendations.map(rec => 
  `### ${rec.category} (${rec.priority})\n**Issue:** ${rec.issue}\n**Solution:** ${rec.solution}\n`
).join('\n')}

## Detailed Results

\`\`\`json
${JSON.stringify(report, null, 2)}
\`\`\`

---
*Generated by MCP Validation Agent - Task Force*
`;

    return { title: issueTitle, body: issueBody, report };
  }

  async run() {
    this.log('Starting Comprehensive MCP Validation...', 'header');
    this.log('==========================================', 'info');
    
    try {
      // Run all test suites
      await this.runProtocolValidation();
      await this.runToolValidation();
      await this.runPerformanceBenchmarks();
      await this.runIntegrationTests();
      
      // Generate reports
      const detailedReport = this.generateDetailedReport();
      const githubIssue = await this.generateGitHubIssueReport();
      
      // Save reports
      fs.writeFileSync('/tmp/mcp-validation-report.json', JSON.stringify(detailedReport, null, 2));
      fs.writeFileSync('/tmp/github-issue-report.md', githubIssue.body);
      
      // Display summary
      this.log('\n=== MCP VALIDATION SUMMARY ===', 'header');
      console.log(this.generateSummaryTable());
      
      this.log(`\nOverall Status: ${detailedReport.overall_summary.all_suites_passed ? 'PASSED' : 'FAILED'}`, 
               detailedReport.overall_summary.all_suites_passed ? 'success' : 'error');
      this.log(`Success Rate: ${detailedReport.overall_summary.overall_success_rate}`, 'info');
      this.log(`Duration: ${detailedReport.duration}`, 'info');
      
      // Output paths
      this.log('\n=== REPORT FILES ===', 'header');
      this.log('Detailed Report: /tmp/mcp-validation-report.json', 'info');
      this.log('GitHub Issue: /tmp/github-issue-report.md', 'info');
      
      return { 
        success: detailedReport.overall_summary.all_suites_passed,
        report: detailedReport,
        githubIssue: githubIssue
      };
    } catch (error) {
      this.log(`MCP Validation failed: ${error.message}`, 'error');
      throw error;
    }
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const runner = new MCPTestRunner();
  runner.run()
    .then(result => {
      console.log('\n=== VALIDATION COMPLETED ===');
      process.exit(result.success ? 0 : 1);
    })
    .catch(error => {
      console.error('Validation failed:', error);
      process.exit(1);
    });
}

export default MCPTestRunner;