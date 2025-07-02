//! Interactive prompts for user onboarding experience

use anyhow::{Context, Result};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use std::io::{self, Write};

/// Trait for interactive user prompts
pub trait InteractivePrompt {
    fn confirm(&self, message: &str, default: bool) -> Result<bool>;
    fn input(&self, message: &str, default: Option<&str>) -> Result<String>;
    fn select(&self, message: &str, options: &[&str], default: Option<usize>) -> Result<usize>;
    fn info(&self, message: &str);
    fn warning(&self, message: &str);
    fn success(&self, message: &str);
}

/// Default implementation of interactive prompt
pub struct DefaultInteractivePrompt {
    theme: ColorfulTheme,
}

impl DefaultInteractivePrompt {
    pub fn new() -> Self {
        Self {
            theme: ColorfulTheme::default(),
        }
    }

    /// Check if we're in a non-interactive environment
    fn is_non_interactive(&self) -> bool {
        // Check for common CI/non-interactive indicators
        std::env::var("CI").is_ok()
            || std::env::var("TERM").map(|t| t == "dumb").unwrap_or(false)
            || !atty::is(atty::Stream::Stdin)
    }

    /// Print colored message to stderr
    fn print_colored(&self, message: &str, color: &str) {
        let colored_msg = match color {
            "green" => format!("\x1b[32m{}\x1b[0m", message),
            "yellow" => format!("\x1b[33m{}\x1b[0m", message),
            "red" => format!("\x1b[31m{}\x1b[0m", message),
            "blue" => format!("\x1b[34m{}\x1b[0m", message),
            _ => message.to_string(),
        };

        eprintln!("{}", colored_msg);
    }
}

impl InteractivePrompt for DefaultInteractivePrompt {
    fn confirm(&self, message: &str, default: bool) -> Result<bool> {
        if self.is_non_interactive() {
            self.info(&format!(
                "Non-interactive mode: using default '{}' for: {}",
                default, message
            ));
            return Ok(default);
        }

        Confirm::with_theme(&self.theme)
            .with_prompt(message)
            .default(default)
            .interact()
            .context("Failed to get confirmation from user")
    }

    fn input(&self, message: &str, default: Option<&str>) -> Result<String> {
        if self.is_non_interactive() {
            let default_value = default.unwrap_or("");
            self.info(&format!(
                "Non-interactive mode: using default '{}' for: {}",
                default_value, message
            ));
            return Ok(default_value.to_string());
        }

        let mut input = Input::<String>::with_theme(&self.theme).with_prompt(message);

        if let Some(default_val) = default {
            input = input.default(default_val.to_string());
        }

        input.interact().context("Failed to get input from user")
    }

    fn select(&self, message: &str, options: &[&str], default: Option<usize>) -> Result<usize> {
        if self.is_non_interactive() {
            let default_idx = default.unwrap_or(0);
            let default_value = options.get(default_idx).unwrap_or(&"<invalid>");
            self.info(&format!(
                "Non-interactive mode: using default '{}' for: {}",
                default_value, message
            ));
            return Ok(default_idx);
        }

        let mut select = Select::with_theme(&self.theme)
            .with_prompt(message)
            .items(options);

        if let Some(default_idx) = default {
            select = select.default(default_idx);
        }

        select
            .interact()
            .context("Failed to get selection from user")
    }

    fn info(&self, message: &str) {
        self.print_colored(&format!("ℹ️  {}", message), "blue");
    }

    fn warning(&self, message: &str) {
        self.print_colored(&format!("⚠️  {}", message), "yellow");
    }

    fn success(&self, message: &str) {
        self.print_colored(&format!("✅ {}", message), "green");
    }
}

impl Default for DefaultInteractivePrompt {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait to check if we're in a terminal
mod atty {
    use std::os::unix::io::AsRawFd;

    pub enum Stream {
        Stdin,
        Stdout,
        Stderr,
    }

    pub fn is(stream: Stream) -> bool {
        let fd = match stream {
            Stream::Stdin => std::io::stdin().as_raw_fd(),
            Stream::Stdout => std::io::stdout().as_raw_fd(),
            Stream::Stderr => std::io::stderr().as_raw_fd(),
        };

        unsafe { libc::isatty(fd) != 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interactive_prompt_creation() {
        let prompt = DefaultInteractivePrompt::new();
        // Should not panic
        assert!(true);
    }

    #[test]
    fn test_non_interactive_detection() {
        let prompt = DefaultInteractivePrompt::new();

        // In test environment, this might be non-interactive
        let result = prompt.is_non_interactive();
        // Just test that it doesn't panic
        assert!(result == true || result == false);
    }

    #[test]
    fn test_message_formatting() {
        let prompt = DefaultInteractivePrompt::new();

        // These should not panic
        prompt.info("Test info message");
        prompt.warning("Test warning message");
        prompt.success("Test success message");
    }
}
