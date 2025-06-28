//! Reasoning engine for thought generation and structured reasoning
//!
//! This module implements the reasoning component of the ReAct framework,
//! responsible for generating coherent thoughts and reasoning chains
//! based on observations and prior knowledge.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::Utc;
use uuid::Uuid;
use num_traits::Float;

use crate::error::{Result, VeritasError};
use crate::types::*;

/// Reasoning engine for generating structured thoughts
pub struct ReasoningEngine<T: Float> {
    config: ReasoningConfig<T>,
    /// Reasoning templates for different scenarios
    templates: HashMap<String, ReasoningTemplate>,
    /// Thought generation statistics
    stats: ReasoningStats,
}

/// Template for structured reasoning
#[derive(Debug, Clone)]
pub struct ReasoningTemplate {
    /// Template name
    pub name: String,
    /// Reasoning steps to follow
    pub steps: Vec<ReasoningTemplateStep>,
    /// Priority of this template
    pub priority: f64,
}

/// Individual step in a reasoning template
#[derive(Debug, Clone)]
pub struct ReasoningTemplateStep {
    /// Step description
    pub description: String,
    /// Type of reasoning
    pub reasoning_type: ReasoningType,
    /// Prompts for generating thoughts
    pub prompts: Vec<String>,
    /// Expected confidence range
    pub confidence_range: (f64, f64),
}

/// Statistics for reasoning engine performance
#[derive(Debug, Clone, Default)]
pub struct ReasoningStats {
    /// Total thoughts generated
    pub thoughts_generated: usize,
    /// Average generation time per thought
    pub avg_generation_time_ms: f64,
    /// Template usage frequency
    pub template_usage: HashMap<String, usize>,
    /// Average confidence in generated thoughts
    pub avg_confidence: f64,
}

impl<T: Float> ReasoningEngine<T> {
    /// Create a new reasoning engine
    pub fn new(config: ReasoningConfig<T>) -> Result<Self> {
        let templates = Self::create_default_templates();
        
        Ok(Self {
            config,
            templates,
            stats: ReasoningStats::default(),
        })
    }

    /// Create default reasoning templates
    fn create_default_templates() -> HashMap<String, ReasoningTemplate> {
        let mut templates = HashMap::new();

        // Multi-modal deception analysis template
        templates.insert("multimodal_analysis".to_string(), ReasoningTemplate {
            name: "Multimodal Deception Analysis".to_string(),
            steps: vec![
                ReasoningTemplateStep {
                    description: "Observe and catalog available modalities".to_string(),
                    reasoning_type: ReasoningType::Observation,
                    prompts: vec![
                        "What modalities are present in this analysis?".to_string(),
                        "What are the key observations from each modality?".to_string(),
                    ],
                    confidence_range: (0.8, 1.0),
                },
                ReasoningTemplateStep {
                    description: "Identify patterns within each modality".to_string(),
                    reasoning_type: ReasoningType::Pattern,
                    prompts: vec![
                        "What behavioral patterns emerge from the visual data?".to_string(),
                        "Are there vocal stress indicators or prosodic anomalies?".to_string(),
                        "What linguistic patterns suggest deception or truth?".to_string(),
                        "Do physiological responses align with stress or deception?".to_string(),
                    ],
                    confidence_range: (0.6, 0.9),
                },
                ReasoningTemplateStep {
                    description: "Analyze cross-modal consistency".to_string(),
                    reasoning_type: ReasoningType::Comparative,
                    prompts: vec![
                        "Are verbal and non-verbal cues consistent?".to_string(),
                        "Do facial expressions align with vocal patterns?".to_string(),
                        "Are physiological responses consistent with observed behavior?".to_string(),
                    ],
                    confidence_range: (0.7, 0.95),
                },
                ReasoningTemplateStep {
                    description: "Consider causal relationships".to_string(),
                    reasoning_type: ReasoningType::Causal,
                    prompts: vec![
                        "What might be causing observed behavioral changes?".to_string(),
                        "Are there environmental factors influencing the subject?".to_string(),
                        "Could cognitive load explain certain patterns?".to_string(),
                    ],
                    confidence_range: (0.5, 0.8),
                },
                ReasoningTemplateStep {
                    description: "Formulate hypotheses about deception".to_string(),
                    reasoning_type: ReasoningType::Hypothesis,
                    prompts: vec![
                        "Based on the evidence, is deception likely?".to_string(),
                        "What alternative explanations exist for the observed patterns?".to_string(),
                        "How confident can we be in this assessment?".to_string(),
                    ],
                    confidence_range: (0.4, 0.9),
                },
                ReasoningTemplateStep {
                    description: "Evaluate evidence strength".to_string(),
                    reasoning_type: ReasoningType::Evidence,
                    prompts: vec![
                        "Which pieces of evidence are strongest?".to_string(),
                        "Are there conflicting indicators?".to_string(),
                        "What additional evidence would strengthen this assessment?".to_string(),
                    ],
                    confidence_range: (0.6, 0.9),
                },
                ReasoningTemplateStep {
                    description: "Synthesize final assessment".to_string(),
                    reasoning_type: ReasoningType::Synthesis,
                    prompts: vec![
                        "Integrating all evidence, what is the most likely conclusion?".to_string(),
                        "What is the confidence level for this assessment?".to_string(),
                        "What are the key supporting factors?".to_string(),
                    ],
                    confidence_range: (0.5, 0.95),
                },
            ],
            priority: 1.0,
        });

        // Baseline comparison template
        templates.insert("baseline_analysis".to_string(), ReasoningTemplate {
            name: "Baseline Comparison Analysis".to_string(),
            steps: vec![
                ReasoningTemplateStep {
                    description: "Establish behavioral baseline".to_string(),
                    reasoning_type: ReasoningType::Observation,
                    prompts: vec![
                        "What appears to be the subject's normal behavioral baseline?".to_string(),
                        "Are there obvious deviations from typical behavior?".to_string(),
                    ],
                    confidence_range: (0.7, 0.9),
                },
                ReasoningTemplateStep {
                    description: "Compare current behavior to baseline".to_string(),
                    reasoning_type: ReasoningType::Comparative,
                    prompts: vec![
                        "How does current behavior differ from the baseline?".to_string(),
                        "Are these differences significant?".to_string(),
                        "Could these differences indicate deception?".to_string(),
                    ],
                    confidence_range: (0.6, 0.8),
                },
            ],
            priority: 0.8,
        });

        // Context-aware analysis template
        templates.insert("contextual_analysis".to_string(), ReasoningTemplate {
            name: "Contextual Analysis".to_string(),
            steps: vec![
                ReasoningTemplateStep {
                    description: "Analyze situational context".to_string(),
                    reasoning_type: ReasoningType::Observation,
                    prompts: vec![
                        "What is the context of this interaction?".to_string(),
                        "Are there external pressures or motivations?".to_string(),
                        "How might the environment affect behavior?".to_string(),
                    ],
                    confidence_range: (0.8, 1.0),
                },
                ReasoningTemplateStep {
                    description: "Consider cultural and individual factors".to_string(),
                    reasoning_type: ReasoningType::Causal,
                    prompts: vec![
                        "Could cultural background influence these behaviors?".to_string(),
                        "Are there individual differences to consider?".to_string(),
                        "What personal factors might affect responses?".to_string(),
                    ],
                    confidence_range: (0.5, 0.7),
                },
            ],
            priority: 0.6,
        });

        templates
    }

    /// Generate thoughts based on input prompt and context
    pub fn generate_thoughts(&self, prompt: &str) -> Result<Thoughts> {
        let start_time = Instant::now();
        let mut thoughts = Thoughts::new();

        // Select appropriate reasoning template
        let template = self.select_template(prompt)?;
        
        // Generate thoughts for each step in the template
        for step in &template.steps {
            let step_thoughts = self.generate_step_thoughts(step, prompt)?;
            for thought in step_thoughts {
                thoughts.thoughts.push(thought);
            }
        }

        // Add meta-reasoning thoughts
        thoughts.add_thought(
            format!(
                "Applied reasoning template: {} with {} steps",
                template.name, template.steps.len()
            ),
            ReasoningType::Synthesis
        );

        // Calculate generation time
        let generation_time = start_time.elapsed();
        thoughts.generation_time = generation_time;

        // Update statistics
        self.update_stats(&template.name, thoughts.thoughts.len(), generation_time);

        Ok(thoughts)
    }

    /// Select the most appropriate reasoning template for the given prompt
    fn select_template(&self, prompt: &str) -> Result<&ReasoningTemplate> {
        let prompt_lower = prompt.to_lowercase();
        
        // Simple keyword-based template selection
        if prompt_lower.contains("multi") || prompt_lower.contains("modal") 
           || (prompt_lower.contains("vision") && prompt_lower.contains("audio")) {
            return Ok(self.templates.get("multimodal_analysis").unwrap());
        }
        
        if prompt_lower.contains("baseline") || prompt_lower.contains("comparison") {
            return Ok(self.templates.get("baseline_analysis").unwrap());
        }
        
        if prompt_lower.contains("context") || prompt_lower.contains("environment") {
            return Ok(self.templates.get("contextual_analysis").unwrap());
        }
        
        // Default to multimodal analysis
        Ok(self.templates.get("multimodal_analysis").unwrap())
    }

    /// Generate thoughts for a specific reasoning step
    fn generate_step_thoughts(&self, step: &ReasoningTemplateStep, context: &str) -> Result<Vec<Thought>> {
        let mut thoughts = Vec::new();
        
        for prompt in &step.prompts {
            // Generate thought based on prompt and context
            let thought_content = self.generate_thought_content(prompt, context, &step.reasoning_type)?;
            
            // Calculate confidence within the step's range
            let confidence = self.calculate_thought_confidence(&thought_content, step);
            
            let thought = Thought {
                id: Uuid::new_v4(),
                content: thought_content,
                reasoning_type: step.reasoning_type.clone(),
                timestamp: Utc::now(),
                confidence,
            };
            
            thoughts.push(thought);
        }
        
        Ok(thoughts)
    }

    /// Generate thought content based on prompt, context, and reasoning type
    fn generate_thought_content(
        &self,
        prompt: &str,
        context: &str,
        reasoning_type: &ReasoningType,
    ) -> Result<String> {
        // In a real implementation, this would use a language model or
        // more sophisticated reasoning algorithms. For now, we provide
        // structured responses based on reasoning type.
        
        match reasoning_type {
            ReasoningType::Observation => {
                if context.contains("Face detected") {
                    Ok("Visual modality available with face detection. Need to analyze facial expressions, micro-expressions, and gaze patterns for deception indicators.".to_string())
                } else if context.contains("voice quality") {
                    Ok("Audio modality shows voice analysis data. Should examine vocal stress, pitch variations, and prosodic features.".to_string())
                } else if context.contains("Text analysis") {
                    Ok("Textual content available for linguistic analysis. Will examine word choice, sentiment, and structural patterns.".to_string())
                } else {
                    Ok("Multiple data modalities detected. Each should be analyzed for deception-relevant patterns.".to_string())
                }
            },
            
            ReasoningType::Pattern => {
                if context.contains("stress") {
                    Ok("Stress indicators present across modalities may suggest cognitive load consistent with deception attempts.".to_string())
                } else if context.contains("micro-expressions") {
                    Ok("Micro-expression patterns could reveal involuntary emotional leakage indicating deception.".to_string())
                } else {
                    Ok("Behavioral patterns should be analyzed for consistency with known deception signatures.".to_string())
                }
            },
            
            ReasoningType::Comparative => {
                Ok("Cross-modal comparison reveals potential inconsistencies between verbal and non-verbal channels that may indicate deception.".to_string())
            },
            
            ReasoningType::Causal => {
                Ok("Elevated stress responses may be caused by cognitive load from maintaining false narratives.".to_string())
            },
            
            ReasoningType::Hypothesis => {
                if context.contains("stress") && context.contains("inconsisten") {
                    Ok("Hypothesis: Subject showing signs of deception based on stress indicators and cross-modal inconsistencies.".to_string())
                } else {
                    Ok("Multiple competing hypotheses exist regarding the subject's truthfulness based on available evidence.".to_string())
                }
            },
            
            ReasoningType::Evidence => {
                Ok("Strongest evidence comes from physiological responses and vocal stress patterns. Visual cues provide supporting evidence.".to_string())
            },
            
            ReasoningType::Synthesis => {
                Ok("Integrating all modalities and evidence sources to form coherent assessment of deception likelihood.".to_string())
            },
        }
    }

    /// Calculate confidence for a generated thought
    fn calculate_thought_confidence(&self, content: &str, step: &ReasoningTemplateStep) -> f64 {
        // Base confidence from template step
        let base_confidence = (step.confidence_range.0 + step.confidence_range.1) / 2.0;
        
        // Adjust based on content specificity
        let specificity_bonus = if content.len() > 100 { 0.1 } else { 0.0 };
        
        // Apply temperature-based variation
        let temperature = self.config.temperature.to_f64().unwrap_or(0.7);
        let variation = (fastrand::f64() - 0.5) * temperature * 0.2;
        
        // Clamp to valid range
        (base_confidence + specificity_bonus + variation).clamp(0.0, 1.0)
    }

    /// Update reasoning statistics
    fn update_stats(&self, template_name: &str, thought_count: usize, generation_time: Duration) {
        // Note: In a real implementation, this would use interior mutability
        // or return updated stats rather than trying to mutate through &self
        // For now, we'll leave this as a placeholder
    }

    /// Update engine configuration
    pub fn update_config(&self, config: ReasoningConfig<T>) -> Result<()> {
        // Note: In a real implementation with interior mutability,
        // we would update the configuration here
        Ok(())
    }

    /// Get reasoning statistics
    pub fn get_stats(&self) -> &ReasoningStats {
        &self.stats
    }

    /// Add a custom reasoning template
    pub fn add_template(&mut self, template: ReasoningTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Remove a reasoning template
    pub fn remove_template(&mut self, name: &str) -> Option<ReasoningTemplate> {
        self.templates.remove(name)
    }

    /// List available templates
    pub fn list_templates(&self) -> Vec<&str> {
        self.templates.keys().map(|k| k.as_str()).collect()
    }
}

/// Utility functions for reasoning
impl<T: Float> ReasoningEngine<T> {
    /// Analyze prompt complexity to determine reasoning approach
    pub fn analyze_prompt_complexity(prompt: &str) -> f64 {
        let word_count = prompt.split_whitespace().count();
        let sentence_count = prompt.matches(['.', '!', '?']).count().max(1);
        let avg_sentence_length = word_count as f64 / sentence_count as f64;
        
        // Complexity score based on length and structure
        let complexity = (avg_sentence_length / 20.0).min(1.0) + 
                        (word_count as f64 / 200.0).min(1.0);
        
        complexity.clamp(0.1, 1.0)
    }

    /// Extract key entities and concepts from prompt
    pub fn extract_key_concepts(prompt: &str) -> Vec<String> {
        let keywords = [
            "deception", "truth", "lie", "honest", "dishonest",
            "stress", "anxiety", "nervous", "calm",
            "voice", "speech", "audio", "vocal",
            "face", "facial", "expression", "micro-expression",
            "text", "linguistic", "language", "words",
            "physiological", "heart", "blood", "pressure",
            "baseline", "comparison", "pattern", "consistency"
        ];
        
        let prompt_lower = prompt.to_lowercase();
        keywords.iter()
            .filter(|&keyword| prompt_lower.contains(keyword))
            .map(|&keyword| keyword.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_engine_creation() {
        let config: ReasoningConfig<f32> = ReasoningConfig::default();
        let engine = ReasoningEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_template_creation() {
        let templates = ReasoningEngine::<f32>::create_default_templates();
        assert!(templates.contains_key("multimodal_analysis"));
        assert!(templates.contains_key("baseline_analysis"));
        assert!(templates.contains_key("contextual_analysis"));
    }

    #[test]
    fn test_thought_generation() {
        let config: ReasoningConfig<f32> = ReasoningConfig::default();
        let engine = ReasoningEngine::new(config).unwrap();
        
        let prompt = "Analyze multi-modal observations for deception detection";
        let thoughts = engine.generate_thoughts(prompt);
        
        assert!(thoughts.is_ok());
        let thoughts = thoughts.unwrap();
        assert!(!thoughts.thoughts.is_empty());
        assert!(thoughts.thoughts.iter().any(|t| matches!(t.reasoning_type, ReasoningType::Observation)));
    }

    #[test]
    fn test_template_selection() {
        let config: ReasoningConfig<f32> = ReasoningConfig::default();
        let engine = ReasoningEngine::new(config).unwrap();
        
        let multimodal_prompt = "Analyze vision and audio data together";
        let template = engine.select_template(multimodal_prompt).unwrap();
        assert_eq!(template.name, "Multimodal Deception Analysis");
        
        let baseline_prompt = "Compare current behavior to baseline";
        let template = engine.select_template(baseline_prompt).unwrap();
        assert_eq!(template.name, "Baseline Comparison Analysis");
    }

    #[test]
    fn test_prompt_complexity_analysis() {
        let simple_prompt = "Test.";
        let complex_prompt = "This is a very long and complex prompt with multiple sentences. It contains various concepts and ideas that need to be processed. The analysis should reflect this complexity.";
        
        let simple_complexity = ReasoningEngine::<f32>::analyze_prompt_complexity(simple_prompt);
        let complex_complexity = ReasoningEngine::<f32>::analyze_prompt_complexity(complex_prompt);
        
        assert!(complex_complexity > simple_complexity);
    }

    #[test]
    fn test_key_concept_extraction() {
        let prompt = "Analyze facial expressions and voice patterns for deception indicators";
        let concepts = ReasoningEngine::<f32>::extract_key_concepts(prompt);
        
        assert!(concepts.contains(&"facial".to_string()));
        assert!(concepts.contains(&"voice".to_string()));
        assert!(concepts.contains(&"deception".to_string()));
    }

    #[test]
    fn test_thought_confidence_calculation() {
        let config: ReasoningConfig<f32> = ReasoningConfig::default();
        let engine = ReasoningEngine::new(config).unwrap();
        
        let step = ReasoningTemplateStep {
            description: "Test step".to_string(),
            reasoning_type: ReasoningType::Observation,
            prompts: vec!["Test prompt".to_string()],
            confidence_range: (0.7, 0.9),
        };
        
        let content = "This is a detailed analysis of the available evidence and patterns.";
        let confidence = engine.calculate_thought_confidence(content, &step);
        
        assert!(confidence >= 0.0 && confidence <= 1.0);
        assert!(confidence >= 0.5); // Should be reasonably high for detailed content
    }
}