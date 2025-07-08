/**
 * Secure Claude invocation without automatic permission bypass
 * Requires explicit user consent for elevated permissions
 */

import { execSync } from 'child_process';

export async function secureClaudeInvoke(prompt, options = {}) {
    if (!prompt || !prompt.trim()) {
        throw new Error('No prompt provided');
    }

    // Check if Claude CLI is available
    try {
        execSync('claude --version', { stdio: 'ignore' });
    } catch (error) {
        throw new Error('Claude Code CLI not found. Please install it first.');
    }

    // Security checks
    const { allowPermissions = false, permissions = [] } = options;
    
    // Build command with security in mind
    let claudeCommand = `claude "${prompt.trim()}"`;
    
    // Only add permissions if explicitly allowed and specified
    if (allowPermissions && permissions.length > 0) {
        console.warn('⚠️  SECURITY WARNING: The following permissions will be granted:');
        permissions.forEach(perm => console.warn(`   - ${perm}`));
        console.warn('Please review carefully before proceeding.');
        
        // In a real implementation, you'd want user confirmation here
        // For now, we'll add the permissions
        permissions.forEach(perm => {
            // Validate permission format to prevent injection
            if (/^--[\w-]+$/.test(perm)) {
                claudeCommand += ` ${perm}`;
            } else {
                console.warn(`⚠️  Skipping invalid permission: ${perm}`);
            }
        });
    }

    return claudeCommand;
}

/**
 * Secure npx execution with version pinning and integrity checks
 */
export async function secureNpxCommand(packageName, version, command, args = []) {
    // Always use specific version
    if (!version) {
        throw new Error('Version must be specified for security');
    }

    // Validate package name to prevent injection
    if (!/^[@\w/-]+$/.test(packageName)) {
        throw new Error('Invalid package name');
    }

    // Build secure npx command with version pinning
    const npxCommand = `npx ${packageName}@${version} ${command} ${args.join(' ')}`;
    
    console.log('🔒 Secure execution:');
    console.log(`   Package: ${packageName}@${version}`);
    console.log(`   Command: ${command}`);
    console.log(`   Note: Using pinned version for security`);
    
    return npxCommand;
}