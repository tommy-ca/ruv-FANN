/**
 * Tests for interactive CLI prompts during onboarding
 * 
 * This test suite verifies the interactive prompts and user input handling
 * during the onboarding flow.
 */

const { InteractivePrompts } = require('../../src/onboarding/interactive-prompts');
const inquirer = require('inquirer');
const chalk = require('chalk');
const ora = require('ora');

// Mock external dependencies
jest.mock('inquirer');
jest.mock('chalk', () => ({
  green: jest.fn(text => text),
  red: jest.fn(text => text),
  yellow: jest.fn(text => text),
  blue: jest.fn(text => text),
  bold: jest.fn(text => text),
  dim: jest.fn(text => text),
}));
jest.mock('ora', () => {
  const spinner = {
    start: jest.fn().mockReturnThis(),
    stop: jest.fn().mockReturnThis(),
    succeed: jest.fn().mockReturnThis(),
    fail: jest.fn().mockReturnThis(),
    text: '',
  };
  return jest.fn(() => spinner);
});

describe('InteractivePrompts', () => {
  let prompts;
  let mockSpinner;

  beforeEach(() => {
    prompts = new InteractivePrompts();
    mockSpinner = ora();
    jest.clearAllMocks();
  });

  describe('welcome()', () => {
    test('should display welcome message', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      await prompts.welcome();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Welcome to ruv-swarm'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('seamless onboarding'));
      
      consoleSpy.mockRestore();
    });

    test('should show version and features', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      await prompts.welcome();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Features:'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Claude Code integration'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('MCP server'));
      
      consoleSpy.mockRestore();
    });
  });

  describe('confirmContinue()', () => {
    test('should prompt for continuation', async () => {
      inquirer.prompt.mockResolvedValue({ continue: true });

      const result = await prompts.confirmContinue('Ready to proceed?');

      expect(inquirer.prompt).toHaveBeenCalledWith([{
        type: 'confirm',
        name: 'continue',
        message: 'Ready to proceed?',
        default: true,
      }]);
      expect(result).toBe(true);
    });

    test('should handle user cancellation', async () => {
      inquirer.prompt.mockResolvedValue({ continue: false });

      const result = await prompts.confirmContinue();

      expect(result).toBe(false);
    });
  });

  describe('selectInstallationType()', () => {
    test('should show installation options', async () => {
      inquirer.prompt.mockResolvedValue({ type: 'full' });

      const result = await prompts.selectInstallationType();

      expect(inquirer.prompt).toHaveBeenCalledWith([{
        type: 'list',
        name: 'type',
        message: 'Select installation type:',
        choices: [
          { name: 'Full installation (recommended)', value: 'full' },
          { name: 'MCP server only', value: 'mcp' },
          { name: 'Claude hooks only', value: 'hooks' },
          { name: 'Custom installation', value: 'custom' },
        ],
        default: 'full',
      }]);
      expect(result).toBe('full');
    });
  });

  describe('customInstallation()', () => {
    test('should prompt for each component', async () => {
      inquirer.prompt
        .mockResolvedValueOnce({ install: true }) // NPM
        .mockResolvedValueOnce({ install: true }) // MCP
        .mockResolvedValueOnce({ install: false }) // Hooks
        .mockResolvedValueOnce({ install: true }); // PATH

      const result = await prompts.customInstallation();

      expect(inquirer.prompt).toHaveBeenCalledTimes(4);
      expect(result).toEqual({
        npm: true,
        mcp: true,
        hooks: false,
        path: true,
      });
    });
  });

  describe('showProgress()', () => {
    test('should show spinner with message', async () => {
      prompts.showProgress('Installing components...');

      expect(mockSpinner.start).toHaveBeenCalled();
      expect(mockSpinner.text).toBe('Installing components...');
    });

    test('should update progress message', async () => {
      prompts.showProgress('Starting...');
      prompts.updateProgress('50% complete');

      expect(mockSpinner.text).toBe('50% complete');
    });

    test('should stop progress on success', async () => {
      prompts.showProgress('Installing...');
      prompts.stopProgress(true, 'Installation complete!');

      expect(mockSpinner.succeed).toHaveBeenCalledWith('Installation complete!');
    });

    test('should stop progress on failure', async () => {
      prompts.showProgress('Installing...');
      prompts.stopProgress(false, 'Installation failed');

      expect(mockSpinner.fail).toHaveBeenCalledWith('Installation failed');
    });
  });

  describe('handleError()', () => {
    test('should prompt for error recovery', async () => {
      inquirer.prompt.mockResolvedValue({ action: 'retry' });

      const result = await prompts.handleError('Network timeout');

      expect(inquirer.prompt).toHaveBeenCalledWith([{
        type: 'list',
        name: 'action',
        message: expect.stringContaining('Network timeout'),
        choices: [
          { name: 'Retry', value: 'retry' },
          { name: 'Skip this step', value: 'skip' },
          { name: 'Abort installation', value: 'abort' },
        ],
      }]);
      expect(result).toBe('retry');
    });

    test('should suggest fixes for common errors', async () => {
      inquirer.prompt.mockResolvedValue({ action: 'retry' });

      await prompts.handleError('EACCES: permission denied');

      expect(inquirer.prompt).toHaveBeenCalledWith([{
        type: 'list',
        name: 'action',
        message: expect.stringContaining('permission'),
        choices: expect.arrayContaining([
          expect.objectContaining({ name: expect.stringContaining('sudo'), value: 'sudo' }),
        ]),
      }]);
    });
  });

  describe('reviewConfiguration()', () => {
    test('should display configuration summary', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      inquirer.prompt.mockResolvedValue({ confirm: true });

      const config = {
        claudeDetected: true,
        npmInstalled: true,
        mcpConfigured: true,
        hooksInstalled: false,
        pathUpdated: true,
      };

      const result = await prompts.reviewConfiguration(config);

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Configuration Summary'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('✓ Claude Code: Detected'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('✓ NPM Package: Installed'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('✗ Claude Hooks: Not installed'));
      
      expect(result).toBe(true);
      
      consoleSpy.mockRestore();
    });
  });

  describe('showCompletion()', () => {
    test('should show success message and next steps', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      await prompts.showCompletion();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('✨'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Installation Complete'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Next steps:'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('claude --mcp'));
      
      consoleSpy.mockRestore();
    });

    test('should show custom launch command if provided', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      await prompts.showCompletion('claude --mcp --config ~/.claude/mcp.json');

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('claude --mcp --config ~/.claude/mcp.json')
      );
      
      consoleSpy.mockRestore();
    });
  });

  describe('input validation', () => {
    test('should validate port number input', async () => {
      const validator = prompts.validators.port;

      expect(validator('8080')).toBe(true);
      expect(validator('65535')).toBe(true);
      expect(validator('0')).toContain('between 1 and 65535');
      expect(validator('70000')).toContain('between 1 and 65535');
      expect(validator('abc')).toContain('valid port number');
    });

    test('should validate file paths', async () => {
      const validator = prompts.validators.path;

      expect(validator('/home/user/.claude')).toBe(true);
      expect(validator('C:\\Users\\claude')).toBe(true);
      expect(validator('')).toContain('path cannot be empty');
    });
  });

  describe('interrupt handling', () => {
    test('should handle SIGINT gracefully', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      const exitSpy = jest.spyOn(process, 'exit').mockImplementation();

      prompts.setupInterruptHandler();
      process.emit('SIGINT');

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Installation cancelled'));
      expect(exitSpy).toHaveBeenCalledWith(0);
      
      consoleSpy.mockRestore();
      exitSpy.mockRestore();
    });
  });
});