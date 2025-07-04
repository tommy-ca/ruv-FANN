#!/usr/bin/env node

/**
 * Fix hardcoded paths in test files to use relative paths
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const npmDir = path.dirname(__dirname);

async function fixTestPaths() {
  const testDir = path.join(npmDir, 'test');
  const files = await fs.readdir(testDir);
  
  for (const file of files) {
    if (!file.endsWith('.js')) continue;
    
    const filePath = path.join(testDir, file);
    let content = await fs.readFile(filePath, 'utf8');
    let modified = false;
    
    // Replace hardcoded paths with relative ones
    const replacements = [
      [/\/workspaces\/ruv-FANN\/ruv-swarm\/npm/g, "__dirname + '/..'"],
      [/\/workspaces\/ruv-FANN\/daa-repository/g, "path.join(__dirname, '..', '..', '..', 'daa-repository')"],
      [/\/workspaces\/ruv-FANN/g, "path.join(__dirname, '..', '..', '..')"],
    ];
    
    for (const [pattern, replacement] of replacements) {
      const newContent = content.replace(pattern, replacement);
      if (newContent !== content) {
        content = newContent;
        modified = true;
      }
    }
    
    if (modified) {
      await fs.writeFile(filePath, content);
      console.log(`✅ Fixed paths in ${file}`);
    }
  }
}

fixTestPaths().catch(console.error);