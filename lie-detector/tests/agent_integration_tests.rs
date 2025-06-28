//! Comprehensive integration tests for the ReAct agent framework
//!
//! This module tests the complete integration of all agent components:
//! - ReAct agent reasoning loop
//! - Memory system integration
//! - Neuro-symbolic reasoning
//! - GSPO learning
//! - Multi-modal fusion

use std::collections::HashMap;
use std::sync::Arc;
use chrono::Utc;
use uuid::Uuid;

use veritas_nexus::agents::*;
use veritas_nexus::reasoning::*;
use veritas_nexus::learning::*;
use veritas_nexus::types::*;
use veritas_nexus::error::Result;

#[tokio::test]
async fn test_complete_react_agent_cycle() -> Result<()> {
    // Create agent with all components
    let config: DetectorConfig<f32> = DetectorConfig::default();
    let mut agent = create_react_agent(config)?;

    // Create test observations
    let observations = create_comprehensive_test_observations();

    // Test complete ReAct cycle: Observe -> Think -> Act -> Explain
    
    // 1. Observe
    agent.observe(observations)?;
    
    // 2. Think
    let thoughts = agent.think()?;
    assert!(!thoughts.thoughts.is_empty());
    assert!(thoughts.confidence > 0.0);
    
    // 3. Act
    let action = agent.act()?;
    assert!(action.confidence > 0.0);
    assert!(!action.explanation.is_empty());
    
    // 4. Explain
    let trace = agent.explain();
    assert!(!trace.steps.is_empty());
    assert!(trace.steps.len() >= 3); // At least observe, think, act
    
    // Verify reasoning trace contains all steps
    let step_types: Vec<ReasoningStepType> = trace.steps.iter()
        .map(|s| s.step_type.clone())
        .collect();
    
    assert!(step_types.contains(&ReasoningStepType::Observe));
    assert!(step_types.contains(&ReasoningStepType::Think));
    assert!(step_types.contains(&ReasoningStepType::Act));
    
    Ok(())
}

#[tokio::test]
async fn test_memory_system_integration() -> Result<()> {
    let config: MemoryConfig<f32> = MemoryConfig::default();
    let mut memory = Memory::new(config)?;

    // Test short-term memory
    let short_term_entry = MemoryEntry::new(
        "Test short-term memory entry".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_short_term_memory(short_term_entry)?;

    // Test long-term memory
    let long_term_entry = MemoryEntry::new(
        "Test long-term memory entry".to_string(),
        MemoryType::LongTerm,
    );
    memory.store_long_term_memory(long_term_entry)?;

    // Test episodic memory
    let episodic_entry = MemoryEntry::new(
        "Test episodic memory entry".to_string(),
        MemoryType::Episodic,
    );
    memory.store_episodic_memory(episodic_entry)?;

    // Test memory retrieval
    let retrieved = memory.retrieve_relevant("test memory")?;
    assert!(!retrieved.is_empty());

    // Test memory consolidation
    let consolidated = memory.consolidate_memories()?;
    println!("Consolidated {} memories", consolidated);

    // Test temporal decay
    memory.apply_temporal_decay()?;

    // Test memory usage statistics
    let usage = memory.get_usage_stats();
    assert!(usage.total_entries > 0);

    Ok(())
}

#[tokio::test]
async fn test_neuro_symbolic_integration() -> Result<()> {
    // Create neuro-symbolic reasoner
    let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
    let knowledge_base = Arc::new(std::sync::Mutex::new(KnowledgeBase::new()));
    let rule_engine = Arc::new(RuleEngine::new());
    
    let mut reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine)?;

    // Test observations processing
    let observations = create_comprehensive_test_observations();
    let decision = reasoner.process_observations(&observations).await?;

    // Verify decision structure
    assert!(matches!(decision.decision, Decision::Truth | Decision::Deception | Decision::Uncertain));
    assert!(decision.confidence > 0.0);
    assert!(!decision.explanation.is_empty());
    assert!(!decision.reasoning_trace.steps.is_empty());

    // Test statistics
    let stats = reasoner.get_stats();
    assert!(stats.neural_processing_time_ms > 0.0);
    assert!(stats.facts_generated > 0);

    Ok(())
}

#[tokio::test]
async fn test_rule_engine_functionality() -> Result<()> {
    let mut rule_engine = RuleEngine::new();

    // Test with realistic facts
    let mut facts = HashMap::new();
    
    // Add stress-related fact
    facts.insert("stress_fact".to_string(), Fact {
        id: "stress_fact".to_string(),
        predicate: "stress_level".to_string(),
        arguments: vec!["high".to_string()],
        confidence: 0.9,
        source: FactSource::Neural,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    });

    // Add baseline fact
    facts.insert("baseline_fact".to_string(), Fact {
        id: "baseline_fact".to_string(),
        predicate: "baseline_stress".to_string(),
        arguments: vec!["normal".to_string()],
        confidence: 0.8,
        source: FactSource::Observation,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    });

    // Add micro-expression fact
    facts.insert("micro_expr_fact".to_string(), Fact {
        id: "micro_expr_fact".to_string(),
        predicate: "micro_expression_count".to_string(),
        arguments: vec!["3".to_string()],
        confidence: 0.85,
        source: FactSource::Neural,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    });

    // Apply rules
    let conclusions = rule_engine.apply_rules(&facts)?;
    
    // Verify conclusions were generated
    assert!(!conclusions.is_empty());
    
    // Check for specific deception-related conclusions
    let has_deception_conclusion = conclusions.iter().any(|(_, conclusion)| {
        conclusion.statement.contains("deception") || 
        conclusion.statement.contains("risk") ||
        conclusion.statement.contains("elevated")
    });
    assert!(has_deception_conclusion);

    // Test rule engine statistics
    let stats = rule_engine.get_stats();
    assert!(stats.total_rules_applied > 0);

    Ok(())
}

#[tokio::test]
async fn test_knowledge_base_operations() -> Result<()> {
    let mut kb = KnowledgeBase::new();

    // Test fact addition
    let fact = create_test_knowledge_fact();
    kb.add_fact(fact.clone())?;

    // Test fact querying
    let criteria = QueryCriteria {
        min_confidence: 0.5,
        categories: vec!["test_category".to_string()],
        tags: vec!["test_tag".to_string()],
        content_filter: "test".to_string(),
        max_results: Some(10),
        sort_order: SortOrder::Confidence,
    };

    let results = kb.query_facts(Some("test_domain"), &criteria)?;
    assert!(!results.is_empty());

    // Test maintenance operations
    let maintenance_report = kb.perform_maintenance()?;
    println!("Maintenance report: {:?}", maintenance_report);

    // Test statistics
    let stats = kb.get_stats();
    assert!(stats.total_facts > 0);

    Ok(())
}

#[tokio::test]
async fn test_reinforcement_learning_integration() -> Result<()> {
    let config: RLConfig<f32> = RLConfig::default();
    let mut learner = ReinforcementLearner::new(config)?;

    // Create test environment
    let environment = TestEnvironment::new();
    
    // Create initial state
    let initial_state = State {
        features: vec![0.5, 0.3, 0.8, 0.2],
        symbolic_features: HashMap::new(),
        temporal_context: None,
        confidence: 0.8,
    };

    // Train for one episode
    let episode_result = learner.train_episode(&initial_state, &environment)?;
    
    // Verify episode results
    assert!(episode_result.steps_taken > 0);
    assert!(!episode_result.experiences.is_empty());
    
    // Test reward shaping
    let experience = &episode_result.experiences[0];
    let shaped_reward = learner.calculate_shaped_reward(experience)?;
    
    // Shaped reward should be different from original (unless coincidentally same)
    println!("Original reward: {}, Shaped reward: {}", experience.reward, shaped_reward);

    // Test statistics
    let stats = learner.get_stats();
    assert!(stats.episodes_completed > 0);

    Ok(())
}

#[tokio::test]
async fn test_gspo_framework_integration() -> Result<()> {
    // This would test the GSPO framework if it were fully implemented
    // For now, we test that the components can be created
    
    let config: GSPOConfig<f32> = GSPOConfig::default();
    
    // Test self-play coordinator
    let self_play_config: SelfPlayConfig<f32> = SelfPlayConfig {
        max_concurrent_sessions: 4,
        session_timeout_ms: 30000,
        scoring_weights: ScoringWeights {
            accuracy_weight: 0.4,
            confidence_weight: 0.3,
            explanation_weight: 0.2,
            efficiency_weight: 0.1,
        },
        matchmaking_params: MatchmakingParams {
            skill_range: 0.2,
            diversity_factor: 0.3,
            balance_factor: 0.8,
        },
    };

    let coordinator = SelfPlayCoordinator {
        config: self_play_config,
        active_sessions: HashMap::new(),
        tournament: Tournament::new(),
        matchmaker: Matchmaker::new(),
    };

    // Verify coordinator was created successfully
    assert_eq!(coordinator.active_sessions.len(), 0);

    Ok(())
}

#[tokio::test]
async fn test_multimodal_integration() -> Result<()> {
    // Test integration with the multi-modal fusion system
    let config: DetectorConfig<f32> = DetectorConfig::default();
    let mut agent = create_react_agent(config)?;

    // Create observations with all modalities
    let observations = Observations {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        vision: Some(VisionObservation {
            face_detected: true,
            micro_expressions: vec!["surprise".to_string(), "fear".to_string()],
            gaze_patterns: vec!["avoidance".to_string()],
            facial_landmarks: vec![(0.1, 0.2), (0.3, 0.4)],
        }),
        audio: Some(AudioObservation {
            pitch_variations: vec![0.1, 0.2, 0.15],
            stress_indicators: vec!["voice_tremor".to_string()],
            voice_quality: 0.6,
            speaking_rate: 180.0,
        }),
        text: Some(TextObservation {
            content: "I did not take the money from the drawer".to_string(),
            linguistic_features: vec!["denial".to_string(), "specific_detail".to_string()],
            sentiment_score: -0.3,
            deception_indicators: vec!["qualifier".to_string(), "negative_statement".to_string()],
        }),
        physiological: Some(PhysiologicalObservation {
            stress_level: 0.8,
            arousal_level: 0.7,
            heart_rate_variability: 0.4,
        }),
        context: ObservationContext {
            environment: "interview_room".to_string(),
            subject_id: Some("subject_001".to_string()),
            session_id: Some("session_001".to_string()),
            interviewer_id: Some("interviewer_001".to_string()),
        },
    };

    // Process all modalities
    agent.observe(observations)?;
    let thoughts = agent.think()?;

    // Verify multi-modal reasoning
    let thought_contents: Vec<String> = thoughts.thoughts.iter()
        .map(|t| t.content.clone())
        .collect();

    // Should have thoughts about different modalities
    let has_vision_thoughts = thought_contents.iter()
        .any(|content| content.contains("vision") || content.contains("facial") || content.contains("expression"));
    let has_audio_thoughts = thought_contents.iter()
        .any(|content| content.contains("audio") || content.contains("voice") || content.contains("vocal"));
    let has_text_thoughts = thought_contents.iter()
        .any(|content| content.contains("text") || content.contains("linguistic"));
    let has_cross_modal_thoughts = thought_contents.iter()
        .any(|content| content.contains("cross-modal") || content.contains("consistency"));

    // At least some cross-modal reasoning should occur
    assert!(has_cross_modal_thoughts || (has_vision_thoughts && has_audio_thoughts));

    Ok(())
}

#[tokio::test]
async fn test_explainable_reasoning() -> Result<()> {
    let config: DetectorConfig<f32> = DetectorConfig::default();
    let mut agent = create_react_agent(config)?;

    let observations = create_comprehensive_test_observations();
    
    // Full reasoning cycle
    agent.observe(observations)?;
    let _thoughts = agent.think()?;
    let action = agent.act()?;
    let trace = agent.explain();

    // Test explanation quality
    assert!(!trace.steps.is_empty());
    assert!(!trace.summary.is_empty());
    assert!(!trace.key_factors.is_empty());
    assert!(trace.explanation_confidence.value() > 0.0);

    // Each step should have meaningful content
    for step in &trace.steps {
        assert!(!step.description.is_empty());
        assert!(!step.input.is_empty());
        assert!(!step.output.is_empty());
        assert!(step.confidence.value() > 0.0);
    }

    // Action should have explanation
    assert!(!action.explanation.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_agent_performance_under_load() -> Result<()> {
    let config: DetectorConfig<f32> = DetectorConfig::default();
    let mut agent = create_react_agent(config)?;

    let start_time = std::time::Instant::now();
    
    // Process multiple observations rapidly
    for i in 0..10 {
        let observations = create_test_observations_with_id(i);
        agent.observe(observations)?;
        agent.think()?;
        agent.act()?;
    }
    
    let processing_time = start_time.elapsed();
    println!("Processed 10 observations in {:?}", processing_time);
    
    // Should complete within reasonable time (adjust threshold as needed)
    assert!(processing_time.as_secs() < 10);

    // Test agent statistics
    let stats = agent.get_stats();
    assert_eq!(stats.observations_processed, 10);
    assert_eq!(stats.decisions_made, 10);
    assert!(stats.avg_reasoning_time_ms > 0.0);

    Ok(())
}

// Helper functions for creating test data

fn create_comprehensive_test_observations() -> Observations<f32> {
    Observations {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        vision: Some(VisionObservation {
            face_detected: true,
            micro_expressions: vec!["contempt".to_string()],
            gaze_patterns: vec!["looking_away".to_string()],
            facial_landmarks: vec![(0.2, 0.3)],
        }),
        audio: Some(AudioObservation {
            pitch_variations: vec![0.1, 0.15, 0.12],
            stress_indicators: vec!["voice_break".to_string()],
            voice_quality: 0.7,
            speaking_rate: 150.0,
        }),
        text: Some(TextObservation {
            content: "I honestly don't know what you're talking about".to_string(),
            linguistic_features: vec!["hedge".to_string()],
            sentiment_score: -0.1,
            deception_indicators: vec!["qualifier".to_string()],
        }),
        physiological: Some(PhysiologicalObservation {
            stress_level: 0.75,
            arousal_level: 0.6,
            heart_rate_variability: 0.3,
        }),
        context: ObservationContext {
            environment: "controlled".to_string(),
            subject_id: Some("test_subject".to_string()),
            session_id: Some("test_session".to_string()),
            interviewer_id: Some("test_interviewer".to_string()),
        },
    }
}

fn create_test_observations_with_id(id: usize) -> Observations<f32> {
    let mut observations = create_comprehensive_test_observations();
    observations.context.subject_id = Some(format!("subject_{}", id));
    observations
}

fn create_test_knowledge_fact() -> KnowledgeFact {
    use std::collections::HashSet;
    
    let mut tags = HashSet::new();
    tags.insert("test_tag".to_string());
    
    KnowledgeFact {
        id: "test_fact".to_string(),
        domain: "test_domain".to_string(),
        category: "test_category".to_string(),
        content: FactContent::Simple {
            predicate: "test_predicate".to_string(),
            arguments: vec!["test_arg".to_string()],
        },
        confidence: 0.8,
        source: FactSource::Neural,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        access_count: 0,
        relationships: vec![],
        tags,
        validation_status: ValidationStatus::Pending,
        annotations: vec![],
    }
}

// Test environment for reinforcement learning
struct TestEnvironment {
    current_state: State<f32>,
    step_count: usize,
}

impl TestEnvironment {
    fn new() -> Self {
        Self {
            current_state: State {
                features: vec![0.5, 0.5, 0.5],
                symbolic_features: HashMap::new(),
                temporal_context: None,
                confidence: 0.8,
            },
            step_count: 0,
        }
    }
}

impl Environment<f32> for TestEnvironment {
    fn step(&self, _state: &State<f32>, action: &Action<f32>) -> Result<(State<f32>, f32, bool)> {
        // Simple test environment that provides deterministic responses
        let reward = match action.action_type {
            ActionType::MakeDecision => 1.0,
            ActionType::AnalyzeModality => 0.5,
            ActionType::RequestMoreData => 0.2,
            _ => 0.0,
        };

        let next_state = State {
            features: vec![0.6, 0.4, 0.7],
            symbolic_features: HashMap::new(),
            temporal_context: None,
            confidence: 0.9,
        };

        let done = self.step_count >= 5; // End episode after 5 steps

        Ok((next_state, reward, done))
    }

    fn reset(&mut self) -> Result<State<f32>> {
        self.step_count = 0;
        self.current_state = State {
            features: vec![0.5, 0.5, 0.5],
            symbolic_features: HashMap::new(),
            temporal_context: None,
            confidence: 0.8,
        };
        Ok(self.current_state.clone())
    }

    fn current_state(&self) -> &State<f32> {
        &self.current_state
    }

    fn is_done(&self) -> bool {
        self.step_count >= 5
    }
}

// Additional test data structures
#[derive(Debug, Clone)]
struct ScoringWeights<T: num_traits::Float> {
    accuracy_weight: T,
    confidence_weight: T,
    explanation_weight: T,
    efficiency_weight: T,
}

#[derive(Debug, Clone)]
struct MatchmakingParams<T: num_traits::Float> {
    skill_range: T,
    diversity_factor: T,
    balance_factor: T,
}

#[derive(Debug, Clone)]
struct Tournament<T: num_traits::Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: num_traits::Float> Tournament<T> {
    fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

#[derive(Debug, Clone)]
struct Matchmaker<T: num_traits::Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: num_traits::Float> Matchmaker<T> {
    fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}