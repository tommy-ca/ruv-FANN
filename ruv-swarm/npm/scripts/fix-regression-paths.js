#!/usr/bin/env node

/**
 * Fix broken paths in regression testing pipeline
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function fixRegressionPaths() {
  const filePath = path.join(__dirname, '..', 'test', 'regression-testing-pipeline.test.js');
  let content = await fs.readFile(filePath, 'utf8');
  
  // Fix all the broken syntax patterns
  const replacements = [
    // Fix package.json path
    ["await fs.readFile\\('__dirname \\+ '/\\.\\.''/package\\.json', 'utf8'\\)", 
     "await fs.readFile(path.join(__dirname, '..', 'package.json'), 'utf8')"],
    
    // Fix test database path  
    ["const testDbPath = '__dirname \\+ '/\\.\\.''/test/regression-test\\.db';",
     "const testDbPath = path.join(__dirname, 'regression-test.db');"],
     
    // Fix cwd patterns
    ["cwd: '__dirname \\+ '/\\.\\.'',", "cwd: path.join(__dirname, '..'),"],
    
    // Fix TypeScript files path
    ["const tsFiles = await this\\.findFiles\\('__dirname \\+ '/\\.\\.''/src', '\\.d\\.ts'\\);",
     "const tsFiles = await this.findFiles(path.join(__dirname, '..', 'src'), '.d.ts');"],
     
    // Fix various results file paths
    ["const resultsFile = '__dirname \\+ '/\\.\\.''/test-results/integration-results\\.json';",
     "const resultsFile = path.join(__dirname, '..', 'test-results', 'integration-results.json');"],
     
    ["const resultsFile = '__dirname \\+ '/\\.\\.''/test/validation-report\\.json';",
     "const resultsFile = path.join(__dirname, 'reports', 'validation-report.json');"],
     
    ["const resultsFile = '__dirname \\+ '/\\.\\.''/test/load-test-report\\.json';",
     "const resultsFile = path.join(__dirname, 'reports', 'load-test-report.json');"],
     
    ["const resultsFile = '__dirname \\+ '/\\.\\.''/test/security-audit-report\\.json';",
     "const resultsFile = path.join(__dirname, 'reports', 'security-audit-report.json');"],
     
    // Fix coverage path
    ["const coveragePath = '__dirname \\+ '/\\.\\.''/coverage/coverage-summary\\.json';",
     "const coveragePath = path.join(__dirname, '..', 'coverage', 'coverage-summary.json');"],
     
    // Fix persistence import
    ["const \\{ PersistenceManager \\} = require\\('__dirname \\+ '/\\.\\.''/src/persistence'\\);",
     "const { SwarmPersistence: PersistenceManager } = await import(path.join(__dirname, '..', 'src', 'persistence.js'));"],
     
    // Fix report path
    ["const reportPath = '__dirname \\+ '/\\.\\.''/test/regression-pipeline-report\\.json';",
     "const reportPath = path.join(__dirname, 'reports', 'regression-pipeline-report.json');"],
     
    // Fix github outputs path
    ["await fs\\.writeFile\\('__dirname \\+ '/\\.\\.''/test/github-outputs\\.txt', githubOutput\\);",
     "await fs.writeFile(path.join(__dirname, 'github-outputs.txt'), githubOutput);"],
     
    // Fix regression results XML path
    ["await fs\\.writeFile\\('__dirname \\+ '/\\.\\.''/test/regression-results\\.xml', junitXml\\);",
     "await fs.writeFile(path.join(__dirname, 'regression-results.xml'), junitXml);"],
  ];
  
  for (const [pattern, replacement] of replacements) {
    const regex = new RegExp(pattern, 'g');
    content = content.replace(regex, replacement);
  }
  
  await fs.writeFile(filePath, content);
  console.log('✅ Fixed regression testing pipeline paths');
}

fixRegressionPaths().catch(console.error);