use super::super::prompts::{OnboardingPrompts, PromptResponse, PromptTheme};
use mockall::automock;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yes_no_prompt_defaults() {
        let prompts = OnboardingPrompts::new();
        
        // Test default true
        let response = prompts.parse_yes_no("", true);
        assert_eq!(response, PromptResponse::Yes);
        
        // Test default false
        let response = prompts.parse_yes_no("", false);
        assert_eq!(response, PromptResponse::No);
    }

    #[test]
    fn test_yes_no_parsing() {
        let prompts = OnboardingPrompts::new();
        
        // Test various yes inputs
        assert_eq!(prompts.parse_yes_no("y", false), PromptResponse::Yes);
        assert_eq!(prompts.parse_yes_no("Y", false), PromptResponse::Yes);
        assert_eq!(prompts.parse_yes_no("yes", false), PromptResponse::Yes);
        assert_eq!(prompts.parse_yes_no("YES", false), PromptResponse::Yes);
        
        // Test various no inputs
        assert_eq!(prompts.parse_yes_no("n", true), PromptResponse::No);
        assert_eq!(prompts.parse_yes_no("N", true), PromptResponse::No);
        assert_eq!(prompts.parse_yes_no("no", true), PromptResponse::No);
        assert_eq!(prompts.parse_yes_no("NO", true), PromptResponse::No);
    }

    #[test]
    fn test_progress_formatting() {
        let prompts = OnboardingPrompts::new();
        
        let progress = prompts.format_progress(50, 100);
        assert!(progress.contains("50%"));
        assert!(progress.contains("█"));
        
        let progress = prompts.format_progress(100, 100);
        assert!(progress.contains("100%"));
    }

    #[test]
    fn test_error_message_formatting() {
        let prompts = OnboardingPrompts::new();
        
        let error = prompts.format_error("File not found", Some("Check the file path"));
        assert!(error.contains("File not found"));
        assert!(error.contains("Check the file path"));
        
        let error = prompts.format_error("Permission denied", None);
        assert!(error.contains("Permission denied"));
    }

    #[test]
    fn test_help_text_formatting() {
        let prompts = OnboardingPrompts::new();
        
        let help = prompts.format_help(
            "Install Claude Code",
            "This will download and install Claude Code to your system"
        );
        assert!(help.contains("Install Claude Code"));
        assert!(help.contains("download and install"));
    }

    #[test]
    fn test_theme_colors() {
        let theme = PromptTheme::default();
        
        // Test that theme has appropriate color settings
        assert!(!theme.success_prefix.is_empty());
        assert!(!theme.error_prefix.is_empty());
        assert!(!theme.warning_prefix.is_empty());
        assert!(!theme.info_prefix.is_empty());
    }

    #[test]
    fn test_choice_prompt() {
        let prompts = OnboardingPrompts::new();
        
        let choices = vec![
            "Retry with sudo",
            "Install to user directory",
            "Skip installation"
        ];
        
        // Test choice parsing
        assert_eq!(prompts.parse_choice("1", &choices), Some(0));
        assert_eq!(prompts.parse_choice("2", &choices), Some(1));
        assert_eq!(prompts.parse_choice("3", &choices), Some(2));
        assert_eq!(prompts.parse_choice("4", &choices), None);
        assert_eq!(prompts.parse_choice("invalid", &choices), None);
    }

    #[test]
    fn test_non_interactive_mode() {
        let mut prompts = OnboardingPrompts::new();
        prompts.set_non_interactive(true);
        
        assert!(prompts.is_non_interactive());
        
        // In non-interactive mode, should use defaults
        let response = prompts.auto_respond_yes_no(true);
        assert_eq!(response, PromptResponse::Yes);
        
        let response = prompts.auto_respond_yes_no(false);
        assert_eq!(response, PromptResponse::No);
    }

    #[test]
    fn test_prompt_history() {
        let mut prompts = OnboardingPrompts::new();
        
        prompts.record_response("install_claude", PromptResponse::Yes);
        prompts.record_response("install_github_mcp", PromptResponse::No);
        
        assert_eq!(prompts.get_response("install_claude"), Some(&PromptResponse::Yes));
        assert_eq!(prompts.get_response("install_github_mcp"), Some(&PromptResponse::No));
        assert_eq!(prompts.get_response("unknown"), None);
    }
}

// Mock for testing interactive prompts
#[automock]
trait InteractivePrompt {
    fn prompt_yes_no(&self, message: &str, default: bool) -> PromptResponse;
    fn prompt_choice(&self, message: &str, choices: &[&str]) -> Option<usize>;
    fn prompt_text(&self, message: &str, default: &str) -> String;
}