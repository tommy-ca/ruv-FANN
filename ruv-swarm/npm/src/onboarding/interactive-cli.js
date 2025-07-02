/**
 * Interactive CLI Module
 * Provides interactive prompts and progress indicators for onboarding
 */

import inquirer from 'inquirer';
import ora from 'ora';
import chalk from 'chalk';
import cliProgress from 'cli-progress';

/**
 * Interactive CLI class for onboarding
 */
export class InteractiveCLI {
  constructor(options = {}) {
    this.autoAccept = options.autoAccept || false;
    this.verbose = options.verbose || false;
    this.spinner = null;
  }

  /**
   * Display welcome message
   */
  welcome() {
    console.log(chalk.cyan.bold('\n🚀 Welcome to ruv-swarm!\n'));
  }

  /**
   * Confirm prompt with Y/N
   * @param {string} message - Prompt message
   * @param {boolean} defaultValue - Default value
   * @returns {Promise<boolean>}
   */
  async confirm(message, defaultValue = true) {
    if (this.autoAccept) {
      if (this.verbose) {
        console.log(chalk.dim(`Auto-accepting: ${message} (${defaultValue ? 'Y' : 'N'})`));
      }
      return defaultValue;
    }

    const response = await inquirer.prompt([{
      type: 'confirm',
      name: 'answer',
      message,
      default: defaultValue
    }]);

    return response.answer;
  }

  /**
   * Multiple choice prompt
   * @param {string} message - Prompt message
   * @param {string[]} choices - Array of choices
   * @returns {Promise<number>} - Index of selected choice
   */
  async choice(message, choices) {
    if (this.autoAccept) {
      if (this.verbose) {
        console.log(chalk.dim(`Auto-selecting: ${message} (${choices[0]})`));
      }
      return 0;
    }

    const response = await inquirer.prompt([{
      type: 'list',
      name: 'answer',
      message,
      choices: choices.map((choice, index) => ({
        name: choice,
        value: index
      }))
    }]);

    return response.answer;
  }

  /**
   * Password/token prompt
   * @param {string} message - Prompt message
   * @returns {Promise<string>}
   */
  async promptPassword(message) {
    const response = await inquirer.prompt([{
      type: 'password',
      name: 'answer',
      message,
      mask: '*'
    }]);

    return response.answer;
  }

  /**
   * Start spinner
   * @param {string} text - Spinner text
   */
  startSpinner(text) {
    this.spinner = ora(text).start();
  }

  /**
   * Update spinner text
   * @param {string} text - New spinner text
   */
  updateSpinner(text) {
    if (this.spinner) {
      this.spinner.text = text;
    }
  }

  /**
   * Stop spinner with success
   * @param {string} text - Success message
   */
  succeedSpinner(text) {
    if (this.spinner) {
      this.spinner.succeed(text);
      this.spinner = null;
    }
  }

  /**
   * Stop spinner with failure
   * @param {string} text - Failure message
   */
  failSpinner(text) {
    if (this.spinner) {
      this.spinner.fail(text);
      this.spinner = null;
    }
  }

  /**
   * Stop spinner with warning
   * @param {string} text - Warning message
   */
  warnSpinner(text) {
    if (this.spinner) {
      this.spinner.warn(text);
      this.spinner = null;
    }
  }

  /**
   * Create and return a progress bar
   * @param {string} title - Progress bar title
   * @param {number} total - Total steps
   * @returns {cliProgress.SingleBar}
   */
  createProgressBar(title, total) {
    const bar = new cliProgress.SingleBar({
      format: `${title} |` + chalk.cyan('{bar}') + '| {percentage}% | {value}/{total}',
      barCompleteChar: '\u2588',
      barIncompleteChar: '\u2591',
      hideCursor: true
    });

    bar.start(total, 0);
    return bar;
  }

  /**
   * Display info message
   * @param {string} message - Info message
   */
  info(message) {
    console.log(chalk.blue('ℹ️  ' + message));
  }

  /**
   * Display success message
   * @param {string} message - Success message
   */
  success(message) {
    console.log(chalk.green('✅ ' + message));
  }

  /**
   * Display warning message
   * @param {string} message - Warning message
   */
  warning(message) {
    console.log(chalk.yellow('⚠️  ' + message));
  }

  /**
   * Display error message
   * @param {string} message - Error message
   */
  error(message) {
    console.log(chalk.red('❌ ' + message));
  }

  /**
   * Display configuration summary
   * @param {Object} config - Configuration object
   */
  displaySummary(config) {
    console.log(chalk.cyan('\n📋 Configuration Summary:'));
    
    if (config.claudeCode) {
      console.log(chalk.green(`- Claude Code: ${config.claudeCode.installed ? 'Installed ✅' : 'Not installed ❌'}`));
      if (config.claudeCode.version) {
        console.log(chalk.dim(`  Version: ${config.claudeCode.version}`));
      }
    }

    if (config.githubMCP !== undefined) {
      console.log(chalk[config.githubMCP ? 'green' : 'yellow'](`- GitHub MCP: ${config.githubMCP ? 'Configured ✅' : 'Skipped ⚠️'}`));
    }

    if (config.ruvSwarmMCP !== undefined) {
      console.log(chalk[config.ruvSwarmMCP ? 'green' : 'yellow'](`- ruv-swarm MCP: ${config.ruvSwarmMCP ? 'Configured ✅' : 'Skipped ⚠️'}`));
    }

    if (config.authentication !== undefined) {
      const authStatus = config.authentication === 'ready' ? 'Ready ✅' : 
                        config.authentication === 'limited' ? 'Limited ⚠️' : 
                        'Not configured ❌';
      console.log(chalk[config.authentication === 'ready' ? 'green' : 'yellow'](`- Authentication: ${authStatus}`));
    }

    console.log('');
  }

  /**
   * Display completion message
   */
  complete() {
    console.log(chalk.green.bold('\n✨ Initialization complete!\n'));
  }

  /**
   * Format error with suggestions
   * @param {Error} error - Error object
   * @param {string[]} suggestions - Array of suggestions
   */
  formatError(error, suggestions = []) {
    this.error(`Error: ${error.message}`);
    
    if (suggestions.length > 0) {
      console.log(chalk.yellow('\nSuggestions:'));
      suggestions.forEach((suggestion, index) => {
        console.log(chalk.yellow(`  ${index + 1}. ${suggestion}`));
      });
    }
  }
}

/**
 * Create CLI instance with options
 * @param {Object} options - CLI options
 * @returns {InteractiveCLI}
 */
export function createCLI(options = {}) {
  return new InteractiveCLI(options);
}

export default {
  InteractiveCLI,
  createCLI
};