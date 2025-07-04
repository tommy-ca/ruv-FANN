#!/usr/bin/env node

/**
 * Prepare test environment for CI/CD
 * Ensures required directories and baseline files exist
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const npmDir = path.dirname(__dirname);

async function prepareTestEnvironment() {
  console.log('🔧 Preparing test environment...');
  
  // Create required directories
  const directories = [
    path.join(npmDir, 'test'),
    path.join(npmDir, 'test', 'reports'),
    path.join(npmDir, 'test-results'),
    path.join(npmDir, 'coverage'),
  ];
  
  for (const dir of directories) {
    await fs.mkdir(dir, { recursive: true });
    console.log(`✅ Ensured directory: ${path.relative(npmDir, dir)}`);
  }
  
  // Create baseline performance file if it doesn't exist
  const baselineFile = path.join(npmDir, 'test', 'baseline-performance.json');
  try {
    await fs.access(baselineFile);
    console.log('✅ Baseline performance file exists');
  } catch {
    const baseline = {
      timestamp: new Date().toISOString(),
      commit: process.env.GITHUB_SHA || 'baseline',
      performance: {
        simdPerformance: '7.0x',
        speedOptimization: '3.5x',
        memoryUsage: '50MB',
        throughput: 1000
      },
      loadTesting: {
        maxConcurrentAgents: 60,
        avgResponseTime: 50,
        memoryPeak: '100MB',
        errorRate: 0.5
      },
      unitTests: {
        coverage: {
          lines: 95,
          branches: 90,
          functions: 92,
          statements: 94
        }
      },
      systemInfo: {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version
      }
    };
    
    await fs.writeFile(baselineFile, JSON.stringify(baseline, null, 2));
    console.log('✅ Created baseline performance file');
  }
  
  console.log('\n✅ Test environment prepared successfully');
}

prepareTestEnvironment().catch(error => {
  console.error('❌ Failed to prepare test environment:', error);
  process.exit(1);
});