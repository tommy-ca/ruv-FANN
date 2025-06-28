//! Default implementation of the ReAct agent framework
//!
//! This module provides the main implementation of the ReactAgent trait,
//! orchestrating the Observe -> Think -> Act -> Explain reasoning loop.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::Utc;
use uuid::Uuid;
use num_traits::Float;

use crate::error::{Result, VeritasError};
use crate::types::*;
use super::{ReactAgent, AgentStats, MemoryUsage, Memory, MemoryType, MemoryEntry, ReasoningEngine, ActionEngine};

/// Default implementation of the ReAct agent
pub struct DefaultReactAgent<T: Float> {
    /// Agent configuration
    config: DetectorConfig<T>,
    /// Memory system
    memory: Arc<Memory<T>>,
    /// Reasoning engine
    reasoning_engine: Arc<ReasoningEngine<T>>,
    /// Action engine
    action_engine: Arc<ActionEngine<T>>,
    /// Current observations
    current_observations: Option<Observations<T>>,
    /// Current thoughts
    current_thoughts: Option<Thoughts>,
    /// Current reasoning trace
    current_trace: ReasoningTrace,
    /// Agent statistics
    stats: AgentStats,
    /// Current iteration in reasoning loop
    current_iteration: usize,
}

impl<T: Float> DefaultReactAgent<T> {
    /// Create a new ReAct agent
    pub fn new(
        config: DetectorConfig<T>,
        memory: Arc<Memory<T>>,
        reasoning_engine: Arc<ReasoningEngine<T>>,
        action_engine: Arc<ActionEngine<T>>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            memory,
            reasoning_engine,
            action_engine,
            current_observations: None,
            current_thoughts: None,
            current_trace: ReasoningTrace::new(),
            stats: AgentStats::default(),
            current_iteration: 0,
        })
    }

    /// Process observations and extract key information
    fn process_observations(&self, observations: &Observations<T>) -> Result<String> {
        let mut summary = Vec::new();
        
        // Process vision observations
        if let Some(vision) = &observations.vision {
            if vision.face_detected {
                summary.push("Face detected in vision input".to_string());
                
                if !vision.micro_expressions.is_empty() {
                    summary.push(format!(
                        "Detected {} micro-expressions",
                        vision.micro_expressions.len()
                    ));
                }
                
                if !vision.gaze_patterns.is_empty() {
                    summary.push(format!(
                        "Tracked {} gaze patterns",
                        vision.gaze_patterns.len()
                    ));
                }
            } else {
                summary.push("No face detected in vision input".to_string());
            }
        }
        
        // Process audio observations
        if let Some(audio) = &observations.audio {
            summary.push(format!(
                "Audio analysis: voice quality {:.2}, speaking rate {:.1} WPM",
                audio.voice_quality.to_f64().unwrap_or(0.0),
                audio.speaking_rate.to_f64().unwrap_or(0.0)
            ));
            
            if !audio.stress_indicators.is_empty() {
                summary.push(format!(
                    "Detected {} stress indicators in voice",
                    audio.stress_indicators.len()
                ));
            }
        }
        
        // Process text observations
        if let Some(text) = &observations.text {
            summary.push(format!(
                "Text analysis: '{}' (sentiment: {:.2})",
                text.content.chars().take(50).collect::<String>(),
                text.sentiment_score
            ));
            
            if !text.deception_indicators.is_empty() {
                summary.push(format!(
                    "Found {} textual deception indicators",
                    text.deception_indicators.len()
                ));
            }
        }
        
        // Process physiological observations
        if let Some(physio) = &observations.physiological {
            summary.push(format!(
                "Physiological: stress {:.2}, arousal {:.2}, HRV {:.2}",
                physio.stress_level.to_f64().unwrap_or(0.0),
                physio.arousal_level.to_f64().unwrap_or(0.0),
                physio.heart_rate_variability.to_f64().unwrap_or(0.0)
            ));
        }
        
        Ok(summary.join("; "))
    }

    /// Generate contextual prompt for reasoning
    fn generate_reasoning_prompt(&self, observations: &Observations<T>) -> Result<String> {
        let observation_summary = self.process_observations(observations)?;
        
        // Retrieve relevant memories
        let relevant_memories = self.memory.retrieve_relevant(&observation_summary)?;
        let memory_context = if relevant_memories.is_empty() {
            "No relevant prior experience found.".to_string()
        } else {
            format!(
                "Relevant prior experience: {}",
                relevant_memories.iter()
                    .take(3) // Limit to top 3 memories
                    .map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        };
        
        let prompt = format!(
            "Analyze the following multi-modal observations for deception detection:\n\n\
            Observations: {}\n\n\
            Context: Subject ID: {}, Session: {}, Environment: {}\n\n\
            Prior Experience: {}\n\n\
            Task: Reason step-by-step about potential deception indicators. \
            Consider cross-modal consistency, behavioral baselines, and known deception patterns. \
            Generate specific thoughts about each modality and their integration.",
            observation_summary,
            observations.context.subject_id.as_ref().unwrap_or(&"unknown".to_string()),
            observations.context.session_id.as_ref().unwrap_or(&"unknown".to_string()),
            observations.context.environment,
            memory_context
        );
        
        Ok(prompt)
    }

    /// Add a reasoning step to the current trace
    fn add_reasoning_step(
        &mut self,
        step_type: ReasoningStepType,
        input: String,
        output: String,
        confidence: f64,
        execution_time: Duration,
    ) {
        let step = ReasoningStep {
            id: Uuid::new_v4(),
            step_type,
            input,
            output,
            confidence,
            timestamp: Utc::now(),
            execution_time,
        };
        
        self.current_trace.add_step(step);
    }

    /// Update agent statistics
    fn update_stats(&mut self, reasoning_time: Duration, confidence: f64) {
        self.stats.observations_processed += 1;
        self.stats.decisions_made += 1;
        
        // Update average reasoning time
        let total_time = self.stats.avg_reasoning_time_ms * (self.stats.decisions_made - 1) as f64;
        let new_time = reasoning_time.as_millis() as f64;
        self.stats.avg_reasoning_time_ms = (total_time + new_time) / self.stats.decisions_made as f64;
        
        // Update average confidence
        let total_confidence = self.stats.avg_confidence * (self.stats.decisions_made - 1) as f64;
        self.stats.avg_confidence = (total_confidence + confidence) / self.stats.decisions_made as f64;
        
        // Update memory usage
        self.stats.memory_usage = self.memory.get_usage_stats();
    }
}

#[async_trait]
impl<T: Float> ReactAgent<T> for DefaultReactAgent<T> {
    fn observe(&mut self, observations: Observations<T>) -> Result<()> {
        let start_time = Instant::now();
        
        // Store observations
        self.current_observations = Some(observations.clone());
        
        // Reset reasoning state for new observations
        self.current_thoughts = None;
        self.current_trace = ReasoningTrace::new();
        self.current_iteration = 0;
        
        // Process and store observations in memory
        let observation_summary = self.process_observations(&observations)?;
        
        // Store in episodic memory
        self.memory.store_episodic_memory(MemoryEntry {
            id: Uuid::new_v4(),
            content: observation_summary.clone(),
            memory_type: MemoryType::Episodic,
            confidence: 1.0,
            timestamp: Utc::now(),
            access_count: 0,
            relevance_score: 1.0,
            metadata: std::collections::HashMap::new(),
        })?;
        
        let execution_time = start_time.elapsed();
        self.add_reasoning_step(
            ReasoningStepType::Observe,
            format!("Observations ID: {}", observations.id),
            observation_summary,
            1.0,
            execution_time,
        );
        
        Ok(())
    }

    fn think(&mut self) -> Result<Thoughts> {
        let start_time = Instant::now();
        
        let observations = self.current_observations.as_ref()
            .ok_or_else(|| VeritasError::reasoning("No observations available for thinking"))?;
        
        // Generate reasoning prompt
        let prompt = self.generate_reasoning_prompt(observations)?;
        
        // Generate thoughts using reasoning engine
        let mut thoughts = self.reasoning_engine.generate_thoughts(&prompt)?;
        
        // Retrieve and integrate relevant memories
        let relevant_memories = self.memory.retrieve_relevant(&prompt)?;
        
        if !relevant_memories.is_empty() {
            thoughts.add_thought(
                format!(
                    "Retrieved {} relevant memories from past experiences: {}",
                    relevant_memories.len(),
                    relevant_memories.iter()
                        .take(2)
                        .map(|m| m.content.clone())
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
                ReasoningType::Evidence
            );
        }
        
        // Add cross-modal analysis thoughts
        if observations.vision.is_some() && observations.audio.is_some() {
            thoughts.add_thought(
                "Cross-modal analysis: Comparing visual and auditory cues for consistency".to_string(),
                ReasoningType::Comparative
            );
        }
        
        if observations.text.is_some() {
            thoughts.add_thought(
                "Linguistic analysis: Examining word choice, sentence structure, and semantic patterns".to_string(),
                ReasoningType::Pattern
            );
        }
        
        if observations.physiological.is_some() {
            thoughts.add_thought(
                "Physiological baseline analysis: Assessing autonomic nervous system responses".to_string(),
                ReasoningType::Causal
            );
        }
        
        // Store thoughts
        self.current_thoughts = Some(thoughts.clone());
        
        let execution_time = start_time.elapsed();
        self.add_reasoning_step(
            ReasoningStepType::Think,
            prompt,
            format!("Generated {} thoughts", thoughts.thoughts.len()),
            0.8,
            execution_time,
        );
        
        Ok(thoughts)
    }

    fn act(&mut self) -> Result<Action<T>> {
        let start_time = Instant::now();
        
        let observations = self.current_observations.as_ref()
            .ok_or_else(|| VeritasError::action("No observations available for action"))?;
        
        let thoughts = self.current_thoughts.as_ref()
            .ok_or_else(|| VeritasError::action("No thoughts available for action"))?;
        
        // Generate action using action engine
        let action = self.action_engine.select_action(observations, thoughts)?;
        
        // Store decision in memory for future reference
        if action.decision != Decision::Uncertain {
            let decision_memory = MemoryEntry {
                id: Uuid::new_v4(),
                content: format!(
                    "Decision: {} (confidence: {:.2}) based on: {}",
                    action.decision,
                    action.confidence.to_f64().unwrap_or(0.0),
                    action.explanation
                ),
                memory_type: MemoryType::LongTerm,
                confidence: action.confidence.to_f64().unwrap_or(0.0),
                timestamp: Utc::now(),
                access_count: 0,
                relevance_score: action.confidence.to_f64().unwrap_or(0.0),
                metadata: std::collections::HashMap::new(),
            };
            
            self.memory.store_long_term_memory(decision_memory)?;
        }
        
        let execution_time = start_time.elapsed();
        self.add_reasoning_step(
            ReasoningStepType::Act,
            format!("Thoughts: {} items", thoughts.thoughts.len()),
            format!("Action: {} (confidence: {:.2})", action.decision, action.confidence.to_f64().unwrap_or(0.0)),
            action.confidence.to_f64().unwrap_or(0.0),
            execution_time,
        );
        
        // Update statistics
        self.update_stats(
            self.current_trace.total_time,
            action.confidence.to_f64().unwrap_or(0.0)
        );
        
        Ok(action)
    }

    fn explain(&self) -> ReasoningTrace {
        let mut trace = self.current_trace.clone();
        
        // Add explanation step
        let explanation_step = ReasoningStep {
            id: Uuid::new_v4(),
            step_type: ReasoningStepType::Explain,
            input: "Complete reasoning process".to_string(),
            output: format!(
                "ReAct cycle completed with {} steps over {:.2}ms",
                trace.steps.len(),
                trace.total_time.as_millis()
            ),
            confidence: 1.0,
            timestamp: Utc::now(),
            execution_time: Duration::from_millis(1),
        };
        
        trace.add_step(explanation_step);
        trace
    }

    fn config(&self) -> &DetectorConfig<T> {
        &self.config
    }

    fn update_config(&mut self, config: DetectorConfig<T>) -> Result<()> {
        self.config = config;
        // Update component configurations
        self.reasoning_engine.update_config(self.config.reasoning_config.clone())?;
        self.action_engine.update_config(self.config.action_config.clone())?;
        self.memory.update_config(self.config.memory_config.clone())?;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.current_observations = None;
        self.current_thoughts = None;
        self.current_trace = ReasoningTrace::new();
        self.current_iteration = 0;
        self.stats = AgentStats::default();
        
        // Reset components
        self.memory.clear_short_term_memory()?;
        
        Ok(())
    }

    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_observations<T: Float>() -> Observations<T> {
        Observations {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            vision: Some(VisionObservation {
                face_detected: true,
                micro_expressions: vec![],
                gaze_patterns: vec![],
                facial_landmarks: vec![],
            }),
            audio: Some(AudioObservation {
                pitch_variations: vec![],
                stress_indicators: vec![],
                voice_quality: T::from(0.7).unwrap(),
                speaking_rate: T::from(160.0).unwrap(),
            }),
            text: Some(TextObservation {
                content: "I did not take the money".to_string(),
                linguistic_features: vec![],
                sentiment_score: -0.2,
                deception_indicators: vec![],
            }),
            physiological: None,
            context: ObservationContext {
                environment: "controlled".to_string(),
                subject_id: Some("test_subject".to_string()),
                session_id: Some("test_session".to_string()),
                interviewer_id: Some("test_interviewer".to_string()),
            },
        }
    }

    #[tokio::test]
    async fn test_react_agent_creation() {
        let config: DetectorConfig<f32> = DetectorConfig::default();
        let memory = Arc::new(Memory::new(config.memory_config.clone()).unwrap());
        let reasoning_engine = Arc::new(ReasoningEngine::new(config.reasoning_config.clone()).unwrap());
        let action_engine = Arc::new(ActionEngine::new(config.action_config.clone()).unwrap());
        
        let agent = DefaultReactAgent::new(config, memory, reasoning_engine, action_engine);
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_react_loop() {
        let config: DetectorConfig<f32> = DetectorConfig::default();
        let memory = Arc::new(Memory::new(config.memory_config.clone()).unwrap());
        let reasoning_engine = Arc::new(ReasoningEngine::new(config.reasoning_config.clone()).unwrap());
        let action_engine = Arc::new(ActionEngine::new(config.action_config.clone()).unwrap());
        
        let mut agent = DefaultReactAgent::new(config, memory, reasoning_engine, action_engine).unwrap();
        
        // Test the full ReAct loop
        let observations = create_test_observations();
        
        // Observe
        let observe_result = agent.observe(observations);
        assert!(observe_result.is_ok());
        
        // Think
        let think_result = agent.think();
        assert!(think_result.is_ok());
        let thoughts = think_result.unwrap();
        assert!(!thoughts.thoughts.is_empty());
        
        // Act
        let act_result = agent.act();
        assert!(act_result.is_ok());
        let action = act_result.unwrap();
        assert!(matches!(action.decision, Decision::Truth | Decision::Deception | Decision::Uncertain));
        
        // Explain
        let trace = agent.explain();
        assert!(!trace.steps.is_empty());
        assert!(trace.steps.len() >= 3); // At least Observe, Think, Act steps
    }

    #[test]
    fn test_process_observations() {
        let config: DetectorConfig<f32> = DetectorConfig::default();
        let memory = Arc::new(Memory::new(config.memory_config.clone()).unwrap());
        let reasoning_engine = Arc::new(ReasoningEngine::new(config.reasoning_config.clone()).unwrap());
        let action_engine = Arc::new(ActionEngine::new(config.action_config.clone()).unwrap());
        
        let agent = DefaultReactAgent::new(config, memory, reasoning_engine, action_engine).unwrap();
        let observations = create_test_observations();
        
        let summary = agent.process_observations(&observations);
        assert!(summary.is_ok());
        let summary_text = summary.unwrap();
        assert!(summary_text.contains("Face detected"));
        assert!(summary_text.contains("voice quality"));
        assert!(summary_text.contains("I did not take the money"));
    }

    #[test]
    fn test_agent_reset() {
        let config: DetectorConfig<f32> = DetectorConfig::default();
        let memory = Arc::new(Memory::new(config.memory_config.clone()).unwrap());
        let reasoning_engine = Arc::new(ReasoningEngine::new(config.reasoning_config.clone()).unwrap());
        let action_engine = Arc::new(ActionEngine::new(config.action_config.clone()).unwrap());
        
        let mut agent = DefaultReactAgent::new(config, memory, reasoning_engine, action_engine).unwrap();
        
        // Set some state
        let observations = create_test_observations();
        agent.observe(observations).unwrap();
        
        // Reset
        let reset_result = agent.reset();
        assert!(reset_result.is_ok());
        
        // Check state is cleared
        assert!(agent.current_observations.is_none());
        assert!(agent.current_thoughts.is_none());
        assert_eq!(agent.current_iteration, 0);
    }
}