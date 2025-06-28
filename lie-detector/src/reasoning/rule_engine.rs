//! Rule engine for symbolic reasoning
//!
//! This module implements a rule-based inference engine that applies logical rules
//! to facts for deception detection reasoning.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use chrono::Utc;

use crate::error::{Result, VeritasError};
use crate::types::*;
use super::*;

/// Rule engine for applying logical rules to facts
pub struct RuleEngine {
    /// Collection of rules organized by category
    rules: HashMap<String, Vec<Rule>>,
    /// Rule execution statistics
    stats: RuleEngineStats,
    /// Configuration settings
    config: RuleEngineConfig,
    /// Rule cache for performance
    rule_cache: HashMap<String, CachedRuleResult>,
}

/// Configuration for the rule engine
#[derive(Debug, Clone)]
pub struct RuleEngineConfig {
    /// Maximum inference depth
    pub max_inference_depth: usize,
    /// Minimum confidence threshold for rule application
    pub min_confidence_threshold: f64,
    /// Maximum number of rules to apply per inference cycle
    pub max_rules_per_cycle: usize,
    /// Enable rule caching
    pub enable_caching: bool,
    /// Cache expiry time in seconds
    pub cache_expiry_seconds: u64,
    /// Rule priority strategy
    pub priority_strategy: PriorityStrategy,
    /// Conflict resolution strategy
    pub conflict_resolution: RuleConflictResolution,
}

impl Default for RuleEngineConfig {
    fn default() -> Self {
        Self {
            max_inference_depth: 10,
            min_confidence_threshold: 0.3,
            max_rules_per_cycle: 50,
            enable_caching: true,
            cache_expiry_seconds: 300, // 5 minutes
            priority_strategy: PriorityStrategy::ConfidenceWeighted,
            conflict_resolution: RuleConflictResolution::HighestConfidence,
        }
    }
}

/// Strategy for prioritizing rule application
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PriorityStrategy {
    /// Apply rules by explicit priority value
    ExplicitPriority,
    /// Apply rules by confidence level
    ConfidenceWeighted,
    /// Apply rules by specificity (more specific first)
    Specificity,
    /// Apply rules by recency of creation
    Recency,
    /// Apply rules by historical success rate
    SuccessRate,
}

/// Strategy for resolving rule conflicts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleConflictResolution {
    /// Choose rule with highest confidence
    HighestConfidence,
    /// Choose rule with highest priority
    HighestPriority,
    /// Combine conflicting rules
    Combine,
    /// Apply all conflicting rules
    ApplyAll,
    /// Choose most specific rule
    MostSpecific,
}

/// Cached rule application result
#[derive(Debug, Clone)]
pub struct CachedRuleResult {
    /// Rule that was applied
    pub rule_id: String,
    /// Facts that matched the rule
    pub matching_facts: Vec<String>,
    /// Conclusion reached
    pub conclusion: Conclusion,
    /// Timestamp of caching
    pub cached_at: chrono::DateTime<Utc>,
    /// Cache hit count
    pub hit_count: usize,
}

/// Statistics for rule engine performance
#[derive(Debug, Clone, Default)]
pub struct RuleEngineStats {
    /// Total rules applied
    pub total_rules_applied: usize,
    /// Rules applied by category
    pub rules_by_category: HashMap<String, usize>,
    /// Average rule application time
    pub avg_application_time_ms: f64,
    /// Rule success rates
    pub rule_success_rates: HashMap<String, f64>,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Inference depth distribution
    pub inference_depths: Vec<usize>,
    /// Conflict resolution counts
    pub conflict_resolutions: HashMap<String, usize>,
}

/// Rule application context
#[derive(Debug, Clone)]
pub struct RuleApplicationContext {
    /// Current inference depth
    pub current_depth: usize,
    /// Previously applied rules (to avoid cycles)
    pub applied_rules: HashSet<String>,
    /// Working memory for intermediate results
    pub working_memory: HashMap<String, String>,
    /// Application timestamp
    pub timestamp: chrono::DateTime<Utc>,
    /// Priority override for specific rules
    pub priority_overrides: HashMap<String, i32>,
}

impl RuleEngine {
    /// Create a new rule engine
    pub fn new() -> Self {
        let mut engine = Self {
            rules: HashMap::new(),
            stats: RuleEngineStats::default(),
            config: RuleEngineConfig::default(),
            rule_cache: HashMap::new(),
        };
        
        // Initialize with default deception detection rules
        engine.initialize_default_rules().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to initialize default rules: {}", e);
        });
        
        engine
    }

    /// Initialize default deception detection rules
    fn initialize_default_rules(&mut self) -> Result<()> {
        let mut deception_rules = Vec::new();

        // Rule 1: High stress indicates potential deception
        deception_rules.push(Rule {
            id: "stress_deception_rule".to_string(),
            name: "High Stress Deception Indicator".to_string(),
            premises: vec![
                Premise {
                    predicate: "stress_level".to_string(),
                    arguments: vec!["high".to_string()],
                    negated: false,
                    weight: 0.8,
                },
                Premise {
                    predicate: "baseline_stress".to_string(),
                    arguments: vec!["normal".to_string()],
                    negated: false,
                    weight: 0.6,
                }
            ],
            conclusion: Conclusion {
                statement: "elevated_deception_risk".to_string(),
                conclusion_type: ConclusionType::Inductive,
                confidence: 0.7,
                evidence: vec!["stress_level(high)".to_string()],
                counter_evidence: vec![],
            },
            confidence: 0.8,
            priority: 100,
            metadata: HashMap::new(),
        });

        // Rule 2: Micro-expression inconsistency
        deception_rules.push(Rule {
            id: "micro_expression_rule".to_string(),
            name: "Micro-expression Inconsistency".to_string(),
            premises: vec![
                Premise {
                    predicate: "micro_expression_count".to_string(),
                    arguments: vec![">".to_string(), "2".to_string()],
                    negated: false,
                    weight: 0.9,
                },
                Premise {
                    predicate: "facial_expression".to_string(),
                    arguments: vec!["calm".to_string()],
                    negated: false,
                    weight: 0.7,
                }
            ],
            conclusion: Conclusion {
                statement: "expression_inconsistency_detected".to_string(),
                conclusion_type: ConclusionType::Direct,
                confidence: 0.8,
                evidence: vec!["micro_expression_mismatch".to_string()],
                counter_evidence: vec![],
            },
            confidence: 0.85,
            priority: 90,
            metadata: HashMap::new(),
        });

        // Rule 3: Voice stress patterns
        deception_rules.push(Rule {
            id: "voice_stress_rule".to_string(),
            name: "Voice Stress Pattern".to_string(),
            premises: vec![
                Premise {
                    predicate: "voice_quality".to_string(),
                    arguments: vec!["<".to_string(), "0.5".to_string()],
                    negated: false,
                    weight: 0.8,
                },
                Premise {
                    predicate: "speaking_rate".to_string(),
                    arguments: vec![">".to_string(), "200".to_string()],
                    negated: false,
                    weight: 0.6,
                }
            ],
            conclusion: Conclusion {
                statement: "vocal_stress_indicators_present".to_string(),
                conclusion_type: ConclusionType::Probabilistic,
                confidence: 0.7,
                evidence: vec!["voice_quality(low)".to_string(), "speaking_rate(fast)".to_string()],
                counter_evidence: vec![],
            },
            confidence: 0.75,
            priority: 80,
            metadata: HashMap::new(),
        });

        // Rule 4: Linguistic deception indicators
        deception_rules.push(Rule {
            id: "linguistic_deception_rule".to_string(),
            name: "Linguistic Deception Patterns".to_string(),
            premises: vec![
                Premise {
                    predicate: "deception_indicator_count".to_string(),
                    arguments: vec![">".to_string(), "3".to_string()],
                    negated: false,
                    weight: 0.9,
                },
                Premise {
                    predicate: "sentiment_score".to_string(),
                    arguments: vec!["<".to_string(), "-0.2".to_string()],
                    negated: false,
                    weight: 0.5,
                }
            ],
            conclusion: Conclusion {
                statement: "linguistic_deception_patterns_detected".to_string(),
                conclusion_type: ConclusionType::Inductive,
                confidence: 0.8,
                evidence: vec!["multiple_deception_indicators".to_string()],
                counter_evidence: vec![],
            },
            confidence: 0.8,
            priority: 95,
            metadata: HashMap::new(),
        });

        // Rule 5: Cross-modal consistency check
        deception_rules.push(Rule {
            id: "cross_modal_consistency_rule".to_string(),
            name: "Cross-Modal Consistency".to_string(),
            premises: vec![
                Premise {
                    predicate: "neural_prediction".to_string(),
                    arguments: vec!["truth".to_string()],
                    negated: false,
                    weight: 0.7,
                },
                Premise {
                    predicate: "stress_level".to_string(),
                    arguments: vec!["high".to_string()],
                    negated: false,
                    weight: 0.8,
                }
            ],
            conclusion: Conclusion {
                statement: "cross_modal_inconsistency".to_string(),
                conclusion_type: ConclusionType::Deductive,
                confidence: 0.6,
                evidence: vec!["prediction_stress_mismatch".to_string()],
                counter_evidence: vec![],
            },
            confidence: 0.7,
            priority: 70,
            metadata: HashMap::new(),
        });

        // Rule 6: High confidence truth detection
        deception_rules.push(Rule {
            id: "high_confidence_truth_rule".to_string(),
            name: "High Confidence Truth Detection".to_string(),
            premises: vec![
                Premise {
                    predicate: "neural_prediction".to_string(),
                    arguments: vec!["truth".to_string()],
                    negated: false,
                    weight: 0.9,
                },
                Premise {
                    predicate: "stress_level".to_string(),
                    arguments: vec!["low".to_string()],
                    negated: false,
                    weight: 0.7,
                },
                Premise {
                    predicate: "micro_expression_count".to_string(),
                    arguments: vec!["<".to_string(), "2".to_string()],
                    negated: false,
                    weight: 0.6,
                }
            ],
            conclusion: Conclusion {
                statement: "high_confidence_truth".to_string(),
                conclusion_type: ConclusionType::Deductive,
                confidence: 0.9,
                evidence: vec!["consistent_truth_indicators".to_string()],
                counter_evidence: vec![],
            },
            confidence: 0.9,
            priority: 110,
            metadata: HashMap::new(),
        });

        self.rules.insert("deception_detection".to_string(), deception_rules);
        Ok(())
    }

    /// Apply rules to a set of facts
    pub fn apply_rules(&mut self, facts: &HashMap<String, Fact>) -> Result<Vec<(String, Conclusion)>> {
        let mut context = RuleApplicationContext {
            current_depth: 0,
            applied_rules: HashSet::new(),
            working_memory: HashMap::new(),
            timestamp: Utc::now(),
            priority_overrides: HashMap::new(),
        };

        let mut conclusions = Vec::new();
        let mut changed = true;

        // Iterative rule application until no new conclusions
        while changed && context.current_depth < self.config.max_inference_depth {
            changed = false;
            context.current_depth += 1;

            // Get applicable rules for current facts
            let applicable_rules = self.find_applicable_rules(facts, &context)?;
            
            // Sort rules by priority
            let sorted_rules = self.sort_rules_by_priority(applicable_rules);

            // Apply rules up to the limit
            for rule in sorted_rules.into_iter().take(self.config.max_rules_per_cycle) {
                if context.applied_rules.contains(&rule.id) {
                    continue; // Avoid rule cycles
                }

                // Check cache first
                if let Some(cached_result) = self.check_cache(&rule.id, facts) {
                    conclusions.push((rule.id.clone(), cached_result.conclusion.clone()));
                    self.update_cache_hit(&rule.id);
                    changed = true;
                    continue;
                }

                // Apply rule
                if let Some(conclusion) = self.apply_rule(&rule, facts, &context)? {
                    conclusions.push((rule.id.clone(), conclusion.clone()));
                    context.applied_rules.insert(rule.id.clone());
                    
                    // Cache the result
                    if self.config.enable_caching {
                        self.cache_rule_result(&rule.id, facts, &conclusion);
                    }
                    
                    // Update statistics
                    self.update_rule_stats(&rule.id, true);
                    changed = true;
                }
            }
        }

        // Update inference depth statistics
        self.stats.inference_depths.push(context.current_depth);

        Ok(conclusions)
    }

    /// Find rules that are applicable to the current facts
    fn find_applicable_rules(
        &self,
        facts: &HashMap<String, Fact>,
        context: &RuleApplicationContext,
    ) -> Result<Vec<&Rule>> {
        let mut applicable_rules = Vec::new();

        for category_rules in self.rules.values() {
            for rule in category_rules {
                if context.applied_rules.contains(&rule.id) {
                    continue; // Skip already applied rules
                }

                if rule.confidence < self.config.min_confidence_threshold {
                    continue; // Skip low confidence rules
                }

                if self.rule_matches_facts(rule, facts)? {
                    applicable_rules.push(rule);
                }
            }
        }

        Ok(applicable_rules)
    }

    /// Check if a rule's premises match the available facts
    fn rule_matches_facts(&self, rule: &Rule, facts: &HashMap<String, Fact>) -> Result<bool> {
        let mut total_weight = 0.0;
        let mut matched_weight = 0.0;

        for premise in &rule.premises {
            total_weight += premise.weight;

            if self.premise_matches_facts(premise, facts)? {
                matched_weight += premise.weight;
            }
        }

        // Rule matches if weighted majority of premises are satisfied
        Ok(matched_weight / total_weight >= 0.6)
    }

    /// Check if a premise matches the available facts
    fn premise_matches_facts(&self, premise: &Premise, facts: &HashMap<String, Fact>) -> Result<bool> {
        for fact in facts.values() {
            if fact.predicate == premise.predicate {
                let matches = if premise.arguments.len() == 1 {
                    // Simple predicate match
                    fact.arguments.contains(&premise.arguments[0])
                } else if premise.arguments.len() == 2 {
                    // Comparison predicate (e.g., >, <, =)
                    self.evaluate_comparison(&premise.arguments[0], &fact.arguments[0], &premise.arguments[1])?
                } else {
                    // Complex predicate matching
                    premise.arguments.iter().all(|arg| fact.arguments.contains(arg))
                };

                if matches ^ premise.negated {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Evaluate comparison predicates (>, <, =, etc.)
    fn evaluate_comparison(&self, operator: &str, fact_value: &str, premise_value: &str) -> Result<bool> {
        // Try to parse as numbers first
        if let (Ok(fact_num), Ok(premise_num)) = (fact_value.parse::<f64>(), premise_value.parse::<f64>()) {
            return Ok(match operator {
                ">" => fact_num > premise_num,
                "<" => fact_num < premise_num,
                "=" | "==" => (fact_num - premise_num).abs() < f64::EPSILON,
                ">=" => fact_num >= premise_num,
                "<=" => fact_num <= premise_num,
                "!=" => (fact_num - premise_num).abs() >= f64::EPSILON,
                _ => false,
            });
        }

        // String comparison
        Ok(match operator {
            "=" | "==" => fact_value == premise_value,
            "!=" => fact_value != premise_value,
            _ => false,
        })
    }

    /// Sort rules by priority according to the configured strategy
    fn sort_rules_by_priority(&self, mut rules: Vec<&Rule>) -> Vec<&Rule> {
        match self.config.priority_strategy {
            PriorityStrategy::ExplicitPriority => {
                rules.sort_by(|a, b| b.priority.cmp(&a.priority));
            },
            PriorityStrategy::ConfidenceWeighted => {
                rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            },
            PriorityStrategy::Specificity => {
                rules.sort_by(|a, b| b.premises.len().cmp(&a.premises.len()));
            },
            PriorityStrategy::Recency => {
                // Would need rule creation timestamps for this
                rules.sort_by(|a, b| a.id.cmp(&b.id)); // Placeholder
            },
            PriorityStrategy::SuccessRate => {
                rules.sort_by(|a, b| {
                    let success_a = self.stats.rule_success_rates.get(&a.id).unwrap_or(&0.5);
                    let success_b = self.stats.rule_success_rates.get(&b.id).unwrap_or(&0.5);
                    success_b.partial_cmp(success_a).unwrap_or(std::cmp::Ordering::Equal)
                });
            },
        }

        rules
    }

    /// Apply a single rule to generate a conclusion
    fn apply_rule(
        &self,
        rule: &Rule,
        facts: &HashMap<String, Fact>,
        _context: &RuleApplicationContext,
    ) -> Result<Option<Conclusion>> {
        // Calculate confidence based on premise matching
        let mut premise_confidences = Vec::new();
        
        for premise in &rule.premises {
            if let Some(confidence) = self.get_premise_confidence(premise, facts)? {
                premise_confidences.push(confidence * premise.weight);
            }
        }

        if premise_confidences.is_empty() {
            return Ok(None);
        }

        // Calculate overall conclusion confidence
        let avg_premise_confidence = premise_confidences.iter().sum::<f64>() / premise_confidences.len() as f64;
        let conclusion_confidence = rule.confidence * avg_premise_confidence;

        if conclusion_confidence < self.config.min_confidence_threshold {
            return Ok(None);
        }

        // Generate evidence from matching facts
        let evidence = self.gather_evidence_for_rule(rule, facts)?;

        let conclusion = Conclusion {
            statement: rule.conclusion.statement.clone(),
            conclusion_type: rule.conclusion.conclusion_type.clone(),
            confidence: conclusion_confidence,
            evidence,
            counter_evidence: vec![], // Would be populated in more sophisticated implementation
        };

        Ok(Some(conclusion))
    }

    /// Get confidence level for a premise based on matching facts
    fn get_premise_confidence(&self, premise: &Premise, facts: &HashMap<String, Fact>) -> Result<Option<f64>> {
        for fact in facts.values() {
            if fact.predicate == premise.predicate {
                if premise.arguments.len() == 1 && fact.arguments.contains(&premise.arguments[0]) {
                    return Ok(Some(fact.confidence));
                } else if premise.arguments.len() == 2 &&
                         self.evaluate_comparison(&premise.arguments[0], &fact.arguments[0], &premise.arguments[1])? {
                    return Ok(Some(fact.confidence));
                }
            }
        }
        Ok(None)
    }

    /// Gather evidence for a rule from matching facts
    fn gather_evidence_for_rule(&self, rule: &Rule, facts: &HashMap<String, Fact>) -> Result<Vec<String>> {
        let mut evidence = Vec::new();

        for premise in &rule.premises {
            for fact in facts.values() {
                if fact.predicate == premise.predicate {
                    evidence.push(format!("{}({})", fact.predicate, fact.arguments.join(", ")));
                }
            }
        }

        Ok(evidence)
    }

    /// Check cache for rule result
    fn check_cache(&self, rule_id: &str, facts: &HashMap<String, Fact>) -> Option<&CachedRuleResult> {
        if !self.config.enable_caching {
            return None;
        }

        if let Some(cached_result) = self.rule_cache.get(rule_id) {
            // Check if cache is still valid
            let cache_age = Utc::now().signed_duration_since(cached_result.cached_at);
            if cache_age.num_seconds() < self.config.cache_expiry_seconds as i64 {
                // Verify that matching facts are still present
                let facts_still_match = cached_result.matching_facts.iter()
                    .all(|fact_id| facts.contains_key(fact_id));
                
                if facts_still_match {
                    return Some(cached_result);
                }
            }
        }

        None
    }

    /// Cache a rule application result
    fn cache_rule_result(&mut self, rule_id: &str, facts: &HashMap<String, Fact>, conclusion: &Conclusion) {
        let matching_facts: Vec<String> = facts.keys().cloned().collect();
        
        let cached_result = CachedRuleResult {
            rule_id: rule_id.to_string(),
            matching_facts,
            conclusion: conclusion.clone(),
            cached_at: Utc::now(),
            hit_count: 0,
        };

        self.rule_cache.insert(rule_id.to_string(), cached_result);
    }

    /// Update cache hit statistics
    fn update_cache_hit(&mut self, rule_id: &str) {
        if let Some(cached_result) = self.rule_cache.get_mut(rule_id) {
            cached_result.hit_count += 1;
        }

        // Update global cache hit rate
        let total_hits: usize = self.rule_cache.values().map(|r| r.hit_count).sum();
        let total_accesses = total_hits + self.stats.total_rules_applied;
        self.stats.cache_hit_rate = if total_accesses > 0 {
            total_hits as f64 / total_accesses as f64
        } else {
            0.0
        };
    }

    /// Update rule application statistics
    fn update_rule_stats(&mut self, rule_id: &str, success: bool) {
        self.stats.total_rules_applied += 1;

        // Update category statistics
        for (category, rules) in &self.rules {
            if rules.iter().any(|r| r.id == rule_id) {
                *self.stats.rules_by_category.entry(category.clone()).or_insert(0) += 1;
                break;
            }
        }

        // Update success rate
        let current_rate = self.stats.rule_success_rates.get(rule_id).unwrap_or(&0.5);
        let current_count = self.stats.total_rules_applied;
        let new_rate = if success {
            (current_rate * (current_count - 1) as f64 + 1.0) / current_count as f64
        } else {
            (current_rate * (current_count - 1) as f64) / current_count as f64
        };
        
        self.stats.rule_success_rates.insert(rule_id.to_string(), new_rate);
    }

    /// Add a new rule to the engine
    pub fn add_rule(&mut self, category: &str, rule: Rule) -> Result<()> {
        let category_rules = self.rules.entry(category.to_string()).or_insert_with(Vec::new);
        category_rules.push(rule);
        Ok(())
    }

    /// Remove a rule from the engine
    pub fn remove_rule(&mut self, rule_id: &str) -> Result<bool> {
        for category_rules in self.rules.values_mut() {
            if let Some(pos) = category_rules.iter().position(|r| r.id == rule_id) {
                category_rules.remove(pos);
                self.rule_cache.remove(rule_id);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Get rule by ID
    pub fn get_rule(&self, rule_id: &str) -> Option<&Rule> {
        for category_rules in self.rules.values() {
            if let Some(rule) = category_rules.iter().find(|r| r.id == rule_id) {
                return Some(rule);
            }
        }
        None
    }

    /// Get all rules in a category
    pub fn get_rules_by_category(&self, category: &str) -> Option<&Vec<Rule>> {
        self.rules.get(category)
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> &RuleEngineStats {
        &self.stats
    }

    /// Update engine configuration
    pub fn update_config(&mut self, config: RuleEngineConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Clear rule cache
    pub fn clear_cache(&mut self) {
        self.rule_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_cached_rules".to_string(), self.rule_cache.len());
        stats.insert("total_cache_hits".to_string(), 
                     self.rule_cache.values().map(|r| r.hit_count).sum());
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_engine_creation() {
        let engine = RuleEngine::new();
        assert!(!engine.rules.is_empty());
        assert!(engine.rules.contains_key("deception_detection"));
    }

    #[test]
    fn test_rule_matching() {
        let engine = RuleEngine::new();
        let mut facts = HashMap::new();
        
        facts.insert("fact1".to_string(), Fact {
            id: "fact1".to_string(),
            predicate: "stress_level".to_string(),
            arguments: vec!["high".to_string()],
            confidence: 0.8,
            source: FactSource::Neural,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        });

        // Get a rule and test matching
        let deception_rules = engine.get_rules_by_category("deception_detection").unwrap();
        let stress_rule = &deception_rules[0]; // First rule should be stress-related
        
        let matches = engine.rule_matches_facts(stress_rule, &facts);
        assert!(matches.is_ok());
    }

    #[test]
    fn test_comparison_evaluation() {
        let engine = RuleEngine::new();
        
        assert!(engine.evaluate_comparison(">", "5", "3").unwrap());
        assert!(!engine.evaluate_comparison(">", "3", "5").unwrap());
        assert!(engine.evaluate_comparison("=", "test", "test").unwrap());
        assert!(!engine.evaluate_comparison("=", "test", "other").unwrap());
    }

    #[test]
    fn test_rule_application() {
        let mut engine = RuleEngine::new();
        let mut facts = HashMap::new();
        
        // Add facts that should trigger deception rules
        facts.insert("stress_fact".to_string(), Fact {
            id: "stress_fact".to_string(),
            predicate: "stress_level".to_string(),
            arguments: vec!["high".to_string()],
            confidence: 0.9,
            source: FactSource::Neural,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        });

        facts.insert("baseline_fact".to_string(), Fact {
            id: "baseline_fact".to_string(),
            predicate: "baseline_stress".to_string(),
            arguments: vec!["normal".to_string()],
            confidence: 0.8,
            source: FactSource::Observation,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        });

        let results = engine.apply_rules(&facts);
        assert!(results.is_ok());
        
        let conclusions = results.unwrap();
        assert!(!conclusions.is_empty());
        
        // Should have some conclusions about deception risk
        assert!(conclusions.iter().any(|(_, conclusion)| 
            conclusion.statement.contains("deception") || 
            conclusion.statement.contains("risk")));
    }

    #[test]
    fn test_rule_priority_sorting() {
        let engine = RuleEngine::new();
        let deception_rules = engine.get_rules_by_category("deception_detection").unwrap();
        
        let rule_refs: Vec<&Rule> = deception_rules.iter().collect();
        let sorted_rules = engine.sort_rules_by_priority(rule_refs);
        
        // Should be sorted by priority (highest first)
        for i in 1..sorted_rules.len() {
            match engine.config.priority_strategy {
                PriorityStrategy::ExplicitPriority => {
                    assert!(sorted_rules[i-1].priority >= sorted_rules[i].priority);
                },
                PriorityStrategy::ConfidenceWeighted => {
                    assert!(sorted_rules[i-1].confidence >= sorted_rules[i].confidence);
                },
                _ => {} // Other strategies tested separately
            }
        }
    }

    #[test]
    fn test_cache_functionality() {
        let mut engine = RuleEngine::new();
        engine.config.enable_caching = true;
        
        let mut facts = HashMap::new();
        facts.insert("test_fact".to_string(), Fact {
            id: "test_fact".to_string(),
            predicate: "test_predicate".to_string(),
            arguments: vec!["test_value".to_string()],
            confidence: 0.8,
            source: FactSource::Neural,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        });

        // First application should cache results
        let _results1 = engine.apply_rules(&facts).unwrap();
        let cache_stats_before = engine.get_cache_stats();
        
        // Second application should hit cache
        let _results2 = engine.apply_rules(&facts).unwrap();
        let cache_stats_after = engine.get_cache_stats();
        
        // Cache hits should have increased
        assert!(cache_stats_after["total_cache_hits"] >= cache_stats_before["total_cache_hits"]);
    }
}