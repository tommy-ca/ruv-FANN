//! Comprehensive tests for reasoning components
//!
//! This module tests the neuro-symbolic reasoning system components
//! in isolation and integration scenarios.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use chrono::Utc;
use uuid::Uuid;

use veritas_nexus::reasoning::*;
use veritas_nexus::types::*;
use veritas_nexus::error::Result;

#[tokio::test]
async fn test_rule_engine_comprehensive() -> Result<()> {
    let mut rule_engine = RuleEngine::new();

    // Test rule engine has default rules
    let deception_rules = rule_engine.get_rules_by_category("deception_detection");
    assert!(deception_rules.is_some());
    assert!(!deception_rules.unwrap().is_empty());

    // Test adding custom rule
    let custom_rule = Rule {
        id: "custom_test_rule".to_string(),
        name: "Custom Test Rule".to_string(),
        premises: vec![
            Premise {
                predicate: "test_condition".to_string(),
                arguments: vec!["true".to_string()],
                negated: false,
                weight: 1.0,
            }
        ],
        conclusion: Conclusion {
            statement: "test_conclusion".to_string(),
            conclusion_type: ConclusionType::Direct,
            confidence: 0.9,
            evidence: vec!["test_condition(true)".to_string()],
            counter_evidence: vec![],
        },
        confidence: 0.9,
        priority: 50,
        metadata: HashMap::new(),
    };

    rule_engine.add_rule("test_category", custom_rule)?;

    // Test rule retrieval
    let retrieved_rule = rule_engine.get_rule("custom_test_rule");
    assert!(retrieved_rule.is_some());
    assert_eq!(retrieved_rule.unwrap().name, "Custom Test Rule");

    // Test rule application with matching facts
    let mut facts = HashMap::new();
    facts.insert("test_fact".to_string(), Fact {
        id: "test_fact".to_string(),
        predicate: "test_condition".to_string(),
        arguments: vec!["true".to_string()],
        confidence: 0.8,
        source: FactSource::Neural,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    });

    let conclusions = rule_engine.apply_rules(&facts)?;
    assert!(!conclusions.is_empty());

    // Should find our custom rule conclusion
    let has_test_conclusion = conclusions.iter()
        .any(|(rule_id, conclusion)| {
            rule_id == "custom_test_rule" && conclusion.statement == "test_conclusion"
        });
    assert!(has_test_conclusion);

    // Test rule removal
    let removed = rule_engine.remove_rule("custom_test_rule")?;
    assert!(removed);

    let retrieved_after_removal = rule_engine.get_rule("custom_test_rule");
    assert!(retrieved_after_removal.is_none());

    Ok(())
}

#[tokio::test]
async fn test_rule_engine_conflict_resolution() -> Result<()> {
    let mut rule_engine = RuleEngine::new();

    // Add conflicting rules
    let rule1 = Rule {
        id: "high_confidence_rule".to_string(),
        name: "High Confidence Rule".to_string(),
        premises: vec![
            Premise {
                predicate: "stress_level".to_string(),
                arguments: vec!["high".to_string()],
                negated: false,
                weight: 1.0,
            }
        ],
        conclusion: Conclusion {
            statement: "definitely_deceptive".to_string(),
            conclusion_type: ConclusionType::Direct,
            confidence: 0.95,
            evidence: vec!["stress_level(high)".to_string()],
            counter_evidence: vec![],
        },
        confidence: 0.95,
        priority: 100,
        metadata: HashMap::new(),
    };

    let rule2 = Rule {
        id: "low_confidence_rule".to_string(),
        name: "Low Confidence Rule".to_string(),
        premises: vec![
            Premise {
                predicate: "stress_level".to_string(),
                arguments: vec!["high".to_string()],
                negated: false,
                weight: 1.0,
            }
        ],
        conclusion: Conclusion {
            statement: "possibly_truthful".to_string(),
            conclusion_type: ConclusionType::Probabilistic,
            confidence: 0.3,
            evidence: vec!["stress_level(high)".to_string()],
            counter_evidence: vec![],
        },
        confidence: 0.3,
        priority: 50,
        metadata: HashMap::new(),
    };

    rule_engine.add_rule("conflict_test", rule1)?;
    rule_engine.add_rule("conflict_test", rule2)?;

    // Create facts that match both rules
    let mut facts = HashMap::new();
    facts.insert("stress_fact".to_string(), Fact {
        id: "stress_fact".to_string(),
        predicate: "stress_level".to_string(),
        arguments: vec!["high".to_string()],
        confidence: 0.9,
        source: FactSource::Neural,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    });

    let conclusions = rule_engine.apply_rules(&facts)?;
    
    // Should prioritize higher confidence rule
    assert!(!conclusions.is_empty());
    let high_conf_conclusion = conclusions.iter()
        .find(|(rule_id, _)| rule_id == "high_confidence_rule");
    assert!(high_conf_conclusion.is_some());

    Ok(())
}

#[tokio::test]
async fn test_knowledge_base_comprehensive() -> Result<()> {
    let mut kb = KnowledgeBase::new();

    // Test default initialization
    let stats = kb.get_stats();
    println!("Initial KB stats: {:?}", stats);

    // Test structured fact addition
    let structured_fact = KnowledgeFact {
        id: "structured_test_fact".to_string(),
        domain: "behavioral_patterns".to_string(),
        category: "micro_expressions".to_string(),
        content: FactContent::Structured {
            entity: "subject_001".to_string(),
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("expression_type".to_string(), AttributeValue::String("contempt".to_string()));
                attrs.insert("intensity".to_string(), AttributeValue::Number(0.8));
                attrs.insert("duration_ms".to_string(), AttributeValue::Number(150.0));
                attrs
            },
        },
        confidence: 0.85,
        source: FactSource::Neural,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        access_count: 0,
        relationships: vec![],
        tags: {
            let mut tags = std::collections::HashSet::new();
            tags.insert("facial_expression".to_string());
            tags.insert("deception_indicator".to_string());
            tags
        },
        validation_status: ValidationStatus::Pending,
        annotations: vec![],
    };

    kb.add_fact(structured_fact)?;

    // Test statistical fact
    let statistical_fact = KnowledgeFact {
        id: "statistical_fact".to_string(),
        domain: "performance_metrics".to_string(),
        category: "accuracy".to_string(),
        content: FactContent::Statistical {
            statistic: "detection_accuracy".to_string(),
            value: 0.87,
            confidence_interval: Some((0.83, 0.91)),
            sample_size: Some(1000),
        },
        confidence: 0.9,
        source: FactSource::Expert,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        access_count: 0,
        relationships: vec![],
        tags: std::collections::HashSet::new(),
        validation_status: ValidationStatus::ExpertValidated,
        annotations: vec![
            ExpertAnnotation {
                expert_id: "expert_001".to_string(),
                annotation_type: AnnotationType::Validation,
                content: "Validated through controlled study".to_string(),
                confidence: 0.95,
                timestamp: Utc::now(),
            }
        ],
    };

    kb.add_fact(statistical_fact)?;

    // Test rule-based fact
    let rule_fact = KnowledgeFact {
        id: "rule_fact".to_string(),
        domain: "deception_detection".to_string(),
        category: "inference_rules".to_string(),
        content: FactContent::Rule {
            premises: vec![
                "stress_level(high)".to_string(),
                "baseline_stress(normal)".to_string(),
            ],
            conclusion: "elevated_deception_risk".to_string(),
            rule_type: RuleType::Probabilistic,
        },
        confidence: 0.75,
        source: FactSource::Expert,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        access_count: 0,
        relationships: vec![
            FactRelationship {
                relationship_type: RelationshipType::Supports,
                target_fact_id: "structured_test_fact".to_string(),
                strength: 0.6,
                metadata: HashMap::new(),
            }
        ],
        tags: {
            let mut tags = std::collections::HashSet::new();
            tags.insert("inference_rule".to_string());
            tags
        },
        validation_status: ValidationStatus::UnderReview,
        annotations: vec![],
    };

    kb.add_fact(rule_fact)?;

    // Test comprehensive querying
    let comprehensive_criteria = QueryCriteria {
        min_confidence: 0.7,
        categories: vec!["micro_expressions".to_string(), "accuracy".to_string()],
        tags: vec!["deception_indicator".to_string()],
        content_filter: "".to_string(),
        max_results: Some(20),
        sort_order: SortOrder::Confidence,
    };

    let results = kb.query_facts(None, &comprehensive_criteria)?;
    assert!(!results.is_empty());

    // Test domain-specific querying
    let domain_results = kb.query_facts(Some("behavioral_patterns"), &QueryCriteria::default())?;
    assert!(!domain_results.is_empty());

    // Test validation status filtering
    let validated_criteria = QueryCriteria {
        min_confidence: 0.0,
        categories: vec![],
        tags: vec![],
        content_filter: "".to_string(),
        max_results: None,
        sort_order: SortOrder::Confidence,
    };

    let validated_results = kb.query_facts(None, &validated_criteria)?;
    let expert_validated_count = validated_results.iter()
        .filter(|fact| fact.validation_status == ValidationStatus::ExpertValidated)
        .count();
    assert!(expert_validated_count > 0);

    // Test maintenance operations
    let maintenance_report = kb.perform_maintenance()?;
    println!("Maintenance report: {:?}", maintenance_report);

    // Verify final statistics
    let final_stats = kb.get_stats();
    assert!(final_stats.total_facts >= 3);
    assert!(final_stats.facts_by_domain.contains_key("behavioral_patterns"));
    assert!(final_stats.facts_by_domain.contains_key("performance_metrics"));

    Ok(())
}

#[tokio::test]
async fn test_neuro_symbolic_reasoning_pipeline() -> Result<()> {
    // Create complete neuro-symbolic reasoning system
    let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
    let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
    let rule_engine = Arc::new(RuleEngine::new());
    
    let mut reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine)?;

    // Create comprehensive test observations
    let observations = create_test_observations_all_modalities();

    // Test complete reasoning pipeline
    let decision = reasoner.process_observations(&observations).await?;

    // Verify decision structure and quality
    assert!(matches!(decision.decision, Decision::Truth | Decision::Deception | Decision::Uncertain));
    assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);
    assert!(!decision.explanation.is_empty());
    
    // Verify reasoning trace quality
    assert!(!decision.reasoning_trace.steps.is_empty());
    assert!(decision.reasoning_trace.steps.len() >= 3); // Neural, symbolic, integration
    assert!(!decision.reasoning_trace.summary.is_empty());
    assert!(!decision.reasoning_trace.key_factors.is_empty());

    // Test statistics
    let stats = reasoner.get_stats();
    assert!(stats.neural_processing_time_ms > 0.0);
    assert!(stats.symbolic_processing_time_ms >= 0.0);
    assert!(stats.integration_time_ms > 0.0);
    assert!(stats.facts_generated > 0);
    assert!(stats.neural_symbolic_agreement >= 0.0 && stats.neural_symbolic_agreement <= 1.0);

    // Test multiple processing cycles for consistency
    for _ in 0..3 {
        let observations = create_test_observations_with_variation();
        let decision = reasoner.process_observations(&observations).await?;
        assert!(decision.confidence > 0.0);
    }

    Ok(())
}

#[tokio::test]
async fn test_neural_symbolic_feature_extraction() -> Result<()> {
    let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
    let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
    let rule_engine = Arc::new(RuleEngine::new());
    
    let reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine)?;

    // Test vision feature extraction
    let vision_obs = VisionObservation {
        face_detected: true,
        micro_expressions: vec!["surprise".to_string(), "contempt".to_string()],
        gaze_patterns: vec!["avoidance".to_string(), "darting".to_string()],
        facial_landmarks: vec![(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
    };

    let vision_features = reasoner.process_vision_modality(&vision_obs)?;
    assert!(!vision_features.is_empty());
    
    // Should have face detection feature
    assert!(vision_features.iter().any(|f| f.name == "face_detected"));
    assert!(vision_features.iter().any(|f| f.name == "micro_expression_count"));
    assert!(vision_features.iter().any(|f| f.name == "gaze_pattern_count"));

    // Test audio feature extraction
    let audio_obs = AudioObservation {
        pitch_variations: vec![0.1, 0.2, 0.15, 0.25],
        stress_indicators: vec!["voice_tremor".to_string(), "pitch_break".to_string()],
        voice_quality: 0.65,
        speaking_rate: 170.0,
    };

    let audio_features = reasoner.process_audio_modality(&audio_obs)?;
    assert!(!audio_features.is_empty());
    assert!(audio_features.iter().any(|f| f.name == "voice_quality"));
    assert!(audio_features.iter().any(|f| f.name == "speaking_rate"));
    assert!(audio_features.iter().any(|f| f.name == "stress_indicator_count"));

    // Test text feature extraction
    let text_obs = TextObservation {
        content: "I absolutely did not do anything wrong, honestly".to_string(),
        linguistic_features: vec!["qualifier".to_string(), "intensifier".to_string()],
        sentiment_score: -0.2,
        deception_indicators: vec!["qualification".to_string(), "over_emphasis".to_string()],
    };

    let text_features = reasoner.process_text_modality(&text_obs)?;
    assert!(!text_features.is_empty());
    assert!(text_features.iter().any(|f| f.name == "sentiment_score"));
    assert!(text_features.iter().any(|f| f.name == "deception_indicator_count"));
    assert!(text_features.iter().any(|f| f.name == "text_length"));

    // Test physiological feature extraction
    let physio_obs = PhysiologicalObservation {
        stress_level: 0.8,
        arousal_level: 0.75,
        heart_rate_variability: 0.3,
    };

    let physio_features = reasoner.process_physiological_modality(&physio_obs)?;
    assert!(!physio_features.is_empty());
    assert!(physio_features.iter().any(|f| f.name == "stress_level"));
    assert!(physio_features.iter().any(|f| f.name == "arousal_level"));
    assert!(physio_features.iter().any(|f| f.name == "heart_rate_variability"));

    Ok(())
}

#[tokio::test]
async fn test_symbolic_fact_conversion() -> Result<()> {
    let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
    let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
    let rule_engine = Arc::new(RuleEngine::new());
    
    let reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine)?;

    // Create neural output with multiple features
    let neural_output = NeuralOutput {
        raw_scores: vec![0.7, 0.3],
        probabilities: vec![0.7, 0.3],
        features: vec![
            Feature {
                name: "stress_level".to_string(),
                value: 0.85,
                weight: 0.9,
                feature_type: "continuous".to_string(),
            },
            Feature {
                name: "micro_expression_count".to_string(),
                value: 3.0,
                weight: 0.8,
                feature_type: "count".to_string(),
            },
            Feature {
                name: "voice_quality".to_string(),
                value: 0.4,
                weight: 0.7,
                feature_type: "continuous".to_string(),
            },
        ],
        attention_weights: Some(vec![0.3, 0.5, 0.2]),
        layer_activations: HashMap::new(),
        confidence: 0.75,
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("model_version".to_string(), "v1.0".to_string());
            meta
        },
    };

    // Convert to symbolic facts
    let facts = reasoner.neural_to_symbolic(&neural_output)?;
    
    // Verify fact conversion
    assert!(!facts.is_empty());
    
    // Should have facts for high-weight features
    let stress_fact = facts.iter().find(|f| f.predicate == "stress_level");
    assert!(stress_fact.is_some());
    assert!(stress_fact.unwrap().confidence > 0.5);

    let expression_fact = facts.iter().find(|f| f.predicate == "micro_expression_count");
    assert!(expression_fact.is_some());

    // Should have neural prediction facts
    let neural_prediction_facts: Vec<_> = facts.iter()
        .filter(|f| f.predicate == "neural_prediction")
        .collect();
    assert!(!neural_prediction_facts.is_empty());

    // Verify fact metadata
    for fact in &facts {
        assert_eq!(fact.source, FactSource::Neural);
        assert!(fact.confidence >= 0.0 && fact.confidence <= 1.0);
        assert!(!fact.arguments.is_empty());
    }

    Ok(())
}

#[tokio::test]
async fn test_reasoning_performance_and_scalability() -> Result<()> {
    let config: NeuroSymbolicConfig<f32> = NeuroSymbolicConfig::default();
    let knowledge_base = Arc::new(Mutex::new(KnowledgeBase::new()));
    let rule_engine = Arc::new(RuleEngine::new());
    
    let mut reasoner = NeuroSymbolicReasoner::new(config, knowledge_base, rule_engine)?;

    let start_time = std::time::Instant::now();
    
    // Process multiple observations to test performance
    for i in 0..5 {
        let observations = create_test_observations_with_id(i);
        let decision = reasoner.process_observations(&observations).await?;
        assert!(decision.confidence > 0.0);
    }
    
    let total_time = start_time.elapsed();
    println!("Processed 5 observations in {:?}", total_time);
    
    // Should complete within reasonable time
    assert!(total_time.as_secs() < 10);

    // Test with large number of facts
    let mut large_facts = HashMap::new();
    for i in 0..100 {
        large_facts.insert(
            format!("fact_{}", i),
            Fact {
                id: format!("fact_{}", i),
                predicate: format!("test_predicate_{}", i % 10),
                arguments: vec![format!("value_{}", i)],
                confidence: 0.5 + (i as f64 / 200.0),
                source: FactSource::Neural,
                timestamp: Utc::now(),
                metadata: HashMap::new(),
            }
        );
    }

    // Test rule application performance with many facts
    let rule_start = std::time::Instant::now();
    let rule_engine = Arc::new(RuleEngine::new());
    let conclusions = rule_engine.apply_rules(&large_facts)?;
    let rule_time = rule_start.elapsed();
    
    println!("Applied rules to 100 facts in {:?}", rule_time);
    assert!(rule_time.as_secs() < 5);

    Ok(())
}

// Helper functions for test data creation

fn create_test_observations_all_modalities() -> Observations<f32> {
    Observations {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        vision: Some(VisionObservation {
            face_detected: true,
            micro_expressions: vec!["fear".to_string(), "surprise".to_string()],
            gaze_patterns: vec!["avoidance".to_string()],
            facial_landmarks: vec![(0.2, 0.3), (0.4, 0.5)],
        }),
        audio: Some(AudioObservation {
            pitch_variations: vec![0.1, 0.2, 0.15],
            stress_indicators: vec!["tremor".to_string()],
            voice_quality: 0.6,
            speaking_rate: 160.0,
        }),
        text: Some(TextObservation {
            content: "I think maybe I might have seen something".to_string(),
            linguistic_features: vec!["hedge".to_string(), "uncertainty".to_string()],
            sentiment_score: -0.1,
            deception_indicators: vec!["qualifier".to_string()],
        }),
        physiological: Some(PhysiologicalObservation {
            stress_level: 0.7,
            arousal_level: 0.65,
            heart_rate_variability: 0.35,
        }),
        context: ObservationContext {
            environment: "interview".to_string(),
            subject_id: Some("test_subject_comprehensive".to_string()),
            session_id: Some("test_session_comprehensive".to_string()),
            interviewer_id: Some("test_interviewer".to_string()),
        },
    }
}

fn create_test_observations_with_variation() -> Observations<f32> {
    let base_obs = create_test_observations_all_modalities();
    let mut varied_obs = base_obs;
    
    // Add some variation
    if let Some(ref mut audio) = varied_obs.audio {
        audio.voice_quality = 0.5 + fastrand::f32() * 0.3; // 0.5-0.8 range
        audio.speaking_rate = 140.0 + fastrand::f32() * 40.0; // 140-180 range
    }
    
    if let Some(ref mut physio) = varied_obs.physiological {
        physio.stress_level = 0.6 + fastrand::f32() * 0.3; // 0.6-0.9 range
    }
    
    varied_obs
}

fn create_test_observations_with_id(id: usize) -> Observations<f32> {
    let mut obs = create_test_observations_all_modalities();
    obs.context.subject_id = Some(format!("test_subject_{}", id));
    obs.id = Uuid::new_v4();
    obs
}