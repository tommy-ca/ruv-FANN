/**
 * Claude Code Detection Module
 * Detects Claude Code installation across different platforms
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';
import path from 'path';
import os from 'os';

/**
 * Detects if Claude Code is installed and returns installation info
 * @returns {Promise<{installed: boolean, path?: string, version?: string}>}
 */
export async function detectClaudeCode() {
  const platform = os.platform();
  
  try {
    // Try to find Claude Code using 'which' command (Unix) or 'where' (Windows)
    const command = platform === 'win32' ? 'where' : 'which';
    const claudePath = execSync(`${command} claude-code`, { encoding: 'utf8' }).trim();
    
    if (claudePath) {
      // Get version
      const version = await getClaudeVersion(claudePath);
      return {
        installed: true,
        path: claudePath,
        version
      };
    }
  } catch (error) {
    // Command not found in PATH, check common installation locations
  }

  // Check platform-specific common locations
  const commonPaths = getCommonPaths(platform);
  
  for (const checkPath of commonPaths) {
    if (existsSync(checkPath)) {
      const version = await getClaudeVersion(checkPath);
      return {
        installed: true,
        path: checkPath,
        version
      };
    }
  }

  return { installed: false };
}

/**
 * Get Claude Code version from binary
 * @param {string} claudePath - Path to Claude Code binary
 * @returns {Promise<string|null>}
 */
async function getClaudeVersion(claudePath) {
  try {
    const versionOutput = execSync(`"${claudePath}" --version`, { encoding: 'utf8' });
    // Extract version from output (format: "Claude Code vX.Y.Z")
    const match = versionOutput.match(/v?(\d+\.\d+\.\d+)/);
    return match ? match[1] : null;
  } catch (error) {
    return null;
  }
}

/**
 * Get common installation paths by platform
 * @param {string} platform - Node.js platform string
 * @returns {string[]}
 */
function getCommonPaths(platform) {
  const homeDir = os.homedir();
  
  switch (platform) {
    case 'win32':
      return [
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'claude-code', 'claude-code.exe'),
        path.join(process.env.PROGRAMFILES || '', 'Claude Code', 'claude-code.exe'),
        path.join(homeDir, 'AppData', 'Local', 'Programs', 'claude-code', 'claude-code.exe'),
        path.join(homeDir, '.local', 'bin', 'claude-code.exe')
      ];
    
    case 'darwin':
      return [
        '/Applications/Claude Code.app/Contents/MacOS/claude-code',
        path.join(homeDir, 'Applications', 'Claude Code.app', 'Contents', 'MacOS', 'claude-code'),
        '/usr/local/bin/claude-code',
        path.join(homeDir, '.local', 'bin', 'claude-code')
      ];
    
    default: // Linux and other Unix-like
      return [
        '/usr/local/bin/claude-code',
        '/usr/bin/claude-code',
        '/opt/claude-code/claude-code',
        path.join(homeDir, '.local', 'bin', 'claude-code'),
        path.join(homeDir, 'bin', 'claude-code')
      ];
  }
}

/**
 * Check if Claude Code version is compatible
 * @param {string} version - Version string
 * @returns {boolean}
 */
export function isVersionCompatible(version) {
  if (!version) return false;
  
  const [major, minor] = version.split('.').map(Number);
  // Assuming we need at least version 1.0.0
  return major >= 1;
}

export default {
  detectClaudeCode,
  isVersionCompatible
};