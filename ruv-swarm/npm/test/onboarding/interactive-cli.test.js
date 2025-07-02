/**
 * Tests for Interactive CLI Module
 */

import { jest } from '@jest/globals';
import { InteractiveCLI, createCLI } from '../../src/onboarding/interactive-cli.js';
import inquirer from 'inquirer';
import ora from 'ora';
import chalk from 'chalk';

// Mock dependencies
jest.mock('inquirer');
jest.mock('ora');

describe('Interactive CLI', () => {
  let cli;
  let mockPrompt;
  let mockSpinner;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock inquirer prompt
    mockPrompt = jest.fn();
    inquirer.prompt = mockPrompt;

    // Mock ora spinner
    mockSpinner = {
      start: jest.fn().mockReturnThis(),
      succeed: jest.fn().mockReturnThis(),
      fail: jest.fn().mockReturnThis(),
      warn: jest.fn().mockReturnThis(),
      text: ''
    };
    ora.mockReturnValue(mockSpinner);

    // Spy on console methods
    jest.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    console.log.mockRestore();
  });

  describe('Basic functionality', () => {
    it('should create CLI instance with default options', () => {
      cli = new InteractiveCLI();
      expect(cli.autoAccept).toBe(false);
      expect(cli.verbose).toBe(false);
    });

    it('should create CLI instance with custom options', () => {
      cli = new InteractiveCLI({ autoAccept: true, verbose: true });
      expect(cli.autoAccept).toBe(true);
      expect(cli.verbose).toBe(true);
    });

    it('should display welcome message', () => {
      cli = new InteractiveCLI();
      cli.welcome();
      
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Welcome to ruv-swarm!')
      );
    });
  });

  describe('confirm method', () => {
    it('should prompt for confirmation', async () => {
      cli = new InteractiveCLI();
      mockPrompt.mockResolvedValue({ answer: true });

      const result = await cli.confirm('Continue?', true);

      expect(mockPrompt).toHaveBeenCalledWith([{
        type: 'confirm',
        name: 'answer',
        message: 'Continue?',
        default: true
      }]);
      expect(result).toBe(true);
    });

    it('should auto-accept when autoAccept is true', async () => {
      cli = new InteractiveCLI({ autoAccept: true });

      const result = await cli.confirm('Continue?', true);

      expect(mockPrompt).not.toHaveBeenCalled();
      expect(result).toBe(true);
    });

    it('should log auto-accept in verbose mode', async () => {
      cli = new InteractiveCLI({ autoAccept: true, verbose: true });

      await cli.confirm('Continue?', false);

      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Auto-accepting: Continue? (N)')
      );
    });
  });

  describe('choice method', () => {
    it('should prompt for choice selection', async () => {
      cli = new InteractiveCLI();
      mockPrompt.mockResolvedValue({ answer: 1 });

      const result = await cli.choice('Select option:', ['Option 1', 'Option 2', 'Option 3']);

      expect(mockPrompt).toHaveBeenCalledWith([{
        type: 'list',
        name: 'answer',
        message: 'Select option:',
        choices: [
          { name: 'Option 1', value: 0 },
          { name: 'Option 2', value: 1 },
          { name: 'Option 3', value: 2 }
        ]
      }]);
      expect(result).toBe(1);
    });

    it('should auto-select first option when autoAccept is true', async () => {
      cli = new InteractiveCLI({ autoAccept: true });

      const result = await cli.choice('Select option:', ['Option 1', 'Option 2']);

      expect(mockPrompt).not.toHaveBeenCalled();
      expect(result).toBe(0);
    });
  });

  describe('promptPassword method', () => {
    it('should prompt for password with masking', async () => {
      cli = new InteractiveCLI();
      mockPrompt.mockResolvedValue({ answer: 'secret123' });

      const result = await cli.promptPassword('Enter token:');

      expect(mockPrompt).toHaveBeenCalledWith([{
        type: 'password',
        name: 'answer',
        message: 'Enter token:',
        mask: '*'
      }]);
      expect(result).toBe('secret123');
    });
  });

  describe('Spinner methods', () => {
    beforeEach(() => {
      cli = new InteractiveCLI();
    });

    it('should start spinner', () => {
      cli.startSpinner('Loading...');
      
      expect(ora).toHaveBeenCalledWith('Loading...');
      expect(mockSpinner.start).toHaveBeenCalled();
      expect(cli.spinner).toBe(mockSpinner);
    });

    it('should update spinner text', () => {
      cli.startSpinner('Loading...');
      cli.updateSpinner('Still loading...');
      
      expect(cli.spinner.text).toBe('Still loading...');
    });

    it('should succeed spinner', () => {
      cli.startSpinner('Loading...');
      cli.succeedSpinner('Done!');
      
      expect(mockSpinner.succeed).toHaveBeenCalledWith('Done!');
      expect(cli.spinner).toBeNull();
    });

    it('should fail spinner', () => {
      cli.startSpinner('Loading...');
      cli.failSpinner('Failed!');
      
      expect(mockSpinner.fail).toHaveBeenCalledWith('Failed!');
      expect(cli.spinner).toBeNull();
    });

    it('should warn spinner', () => {
      cli.startSpinner('Loading...');
      cli.warnSpinner('Warning!');
      
      expect(mockSpinner.warn).toHaveBeenCalledWith('Warning!');
      expect(cli.spinner).toBeNull();
    });
  });

  describe('Message methods', () => {
    beforeEach(() => {
      cli = new InteractiveCLI();
    });

    it('should display info message', () => {
      cli.info('Information');
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('ℹ️  Information')
      );
    });

    it('should display success message', () => {
      cli.success('Success');
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('✅ Success')
      );
    });

    it('should display warning message', () => {
      cli.warning('Warning');
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('⚠️  Warning')
      );
    });

    it('should display error message', () => {
      cli.error('Error');
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('❌ Error')
      );
    });
  });

  describe('displaySummary method', () => {
    it('should display configuration summary', () => {
      cli = new InteractiveCLI();
      
      const config = {
        claudeCode: { installed: true, version: '1.2.3' },
        githubMCP: true,
        ruvSwarmMCP: true,
        authentication: 'ready'
      };

      cli.displaySummary(config);

      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Configuration Summary')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Claude Code: Installed ✅')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Version: 1.2.3')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('GitHub MCP: Configured ✅')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('ruv-swarm MCP: Configured ✅')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Authentication: Ready ✅')
      );
    });

    it('should display partial configuration', () => {
      cli = new InteractiveCLI();
      
      const config = {
        claudeCode: { installed: false },
        githubMCP: false,
        ruvSwarmMCP: true,
        authentication: 'limited'
      };

      cli.displaySummary(config);

      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Claude Code: Not installed ❌')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('GitHub MCP: Skipped ⚠️')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Authentication: Limited ⚠️')
      );
    });
  });

  describe('formatError method', () => {
    it('should format error with suggestions', () => {
      cli = new InteractiveCLI();
      
      const error = new Error('Permission denied');
      const suggestions = [
        'Try running with sudo',
        'Check file permissions'
      ];

      cli.formatError(error, suggestions);

      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Error: Permission denied')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('Suggestions:')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('1. Try running with sudo')
      );
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('2. Check file permissions')
      );
    });
  });

  describe('createCLI function', () => {
    it('should create CLI instance', () => {
      const cli = createCLI({ autoAccept: true });
      expect(cli).toBeInstanceOf(InteractiveCLI);
      expect(cli.autoAccept).toBe(true);
    });
  });
});