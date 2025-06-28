//! Pattern discovery for learning from experience
//!
//! This module implements algorithms for discovering patterns in agent behavior,
//! reasoning processes, and decision outcomes.

use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Result;
use crate::types::*;
use super::*;

/// Pattern discovery system for extracting insights from agent experiences
pub struct PatternDiscoveryEngine<T: Float> {
    /// Configuration
    pub config: PatternConfig<T>,
    /// Discovered patterns
    pub patterns: HashMap<Uuid, Pattern<T>>,
    /// Pattern mining algorithms
    pub miners: Vec<Box<dyn PatternMiner<T>>>,
}

/// Configuration for pattern discovery
#[derive(Debug, Clone)]
pub struct PatternConfig<T: Float> {
    /// Minimum support threshold
    pub min_support: T,
    /// Minimum confidence threshold
    pub min_confidence: T,
    /// Pattern complexity limit
    pub max_complexity: usize,
    /// Discovery algorithms to use
    pub algorithms: Vec<DiscoveryAlgorithm>,
}

/// Types of pattern discovery algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryAlgorithm {
    FrequentSequences,
    DecisionTrees,
    AssociationRules,
    ClusterAnalysis,
    TemporalPatterns,
}

/// Discovered pattern
#[derive(Debug, Clone)]
pub struct Pattern<T: Float> {
    /// Pattern identifier
    pub id: Uuid,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Support value
    pub support: T,
    /// Confidence value
    pub confidence: T,
    /// Pattern elements
    pub elements: Vec<PatternElement<T>>,
    /// Usage statistics
    pub usage_stats: PatternUsage,
}

/// Types of patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternType {
    Sequential,
    Associative,
    Temporal,
    Causal,
    Behavioral,
}

/// Element within a pattern
#[derive(Debug, Clone)]
pub struct PatternElement<T: Float> {
    /// Element type
    pub element_type: String,
    /// Element value
    pub value: String,
    /// Confidence in this element
    pub confidence: T,
    /// Frequency of occurrence
    pub frequency: usize,
}

/// Pattern usage statistics
#[derive(Debug, Clone, Default)]
pub struct PatternUsage {
    /// Times pattern was applied
    pub applications: usize,
    /// Success rate when applied
    pub success_rate: f64,
    /// Average improvement
    pub avg_improvement: f64,
    /// Contexts where pattern is useful
    pub useful_contexts: Vec<String>,
}

/// Trait for pattern mining algorithms
pub trait PatternMiner<T: Float>: Send + Sync {
    /// Mine patterns from experiences
    fn mine_patterns(&self, experiences: &[Experience<T>]) -> Result<Vec<Pattern<T>>>;
    
    /// Get algorithm name
    fn name(&self) -> &str;
    
    /// Get algorithm parameters
    fn parameters(&self) -> HashMap<String, String>;
}

/// Frequent sequence pattern miner
pub struct FrequentSequenceMiner<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> FrequentSequenceMiner<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> PatternMiner<T> for FrequentSequenceMiner<T> {
    fn mine_patterns(&self, experiences: &[Experience<T>]) -> Result<Vec<Pattern<T>>> {
        let mut patterns = Vec::new();
        
        // Extract action sequences
        let sequences = self.extract_action_sequences(experiences);
        
        // Find frequent subsequences
        let frequent_sequences = self.find_frequent_subsequences(&sequences, 0.1);
        
        // Convert to patterns
        for (sequence, support) in frequent_sequences {
            let pattern = Pattern {
                id: Uuid::new_v4(),
                pattern_type: PatternType::Sequential,
                description: format!("Action sequence: {}", sequence.join(" -> ")),
                support: T::from(support).unwrap(),
                confidence: T::from(0.8).unwrap(), // Would be calculated properly
                elements: sequence.into_iter()
                    .map(|action| PatternElement {
                        element_type: "action".to_string(),
                        value: action,
                        confidence: T::from(0.8).unwrap(),
                        frequency: 1,
                    })
                    .collect(),
                usage_stats: PatternUsage::default(),
            };
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn name(&self) -> &str {
        "frequent_sequence_miner"
    }
    
    fn parameters(&self) -> HashMap<String, String> {
        HashMap::from([
            ("algorithm".to_string(), "apriori".to_string()),
            ("min_length".to_string(), "2".to_string()),
        ])
    }
}

impl<T: Float> FrequentSequenceMiner<T> {
    fn extract_action_sequences(&self, experiences: &[Experience<T>]) -> Vec<Vec<String>> {
        experiences.iter()
            .map(|exp| format!("{:?}", exp.action.action_type))
            .collect::<Vec<_>>()
            .windows(3)
            .map(|window| window.to_vec())
            .collect()
    }
    
    fn find_frequent_subsequences(&self, sequences: &[Vec<String>], min_support: f64) -> Vec<(Vec<String>, f64)> {
        let mut frequent = Vec::new();
        let total_count = sequences.len() as f64;
        
        // Simple frequency counting (would be more sophisticated in practice)
        let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
        for sequence in sequences {
            *counts.entry(sequence.clone()).or_insert(0) += 1;
        }
        
        for (sequence, count) in counts {
            let support = count as f64 / total_count;
            if support >= min_support {
                frequent.push((sequence, support));
            }
        }
        
        frequent
    }
}

/// Decision tree pattern miner
pub struct DecisionTreeMiner<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> DecisionTreeMiner<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float> PatternMiner<T> for DecisionTreeMiner<T> {
    fn mine_patterns(&self, experiences: &[Experience<T>]) -> Result<Vec<Pattern<T>>> {
        let mut patterns = Vec::new();
        
        // Build decision tree from experiences
        let tree = self.build_decision_tree(experiences)?;
        
        // Extract rules from tree
        let rules = self.extract_rules(&tree);
        
        // Convert rules to patterns
        for rule in rules {
            let pattern = Pattern {
                id: Uuid::new_v4(),
                pattern_type: PatternType::Causal,
                description: rule.description,
                support: rule.support,
                confidence: rule.confidence,
                elements: rule.conditions.into_iter()
                    .map(|cond| PatternElement {
                        element_type: "condition".to_string(),
                        value: cond,
                        confidence: T::from(0.9).unwrap(),
                        frequency: 1,
                    })
                    .collect(),
                usage_stats: PatternUsage::default(),
            };
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn name(&self) -> &str {
        "decision_tree_miner"
    }
    
    fn parameters(&self) -> HashMap<String, String> {
        HashMap::from([
            ("algorithm".to_string(), "c45".to_string()),
            ("min_samples_split".to_string(), "5".to_string()),
        ])
    }
}

/// Simple decision tree node
#[derive(Debug, Clone)]
pub struct DecisionNode<T: Float> {
    pub feature: Option<String>,
    pub threshold: Option<T>,
    pub left: Option<Box<DecisionNode<T>>>,
    pub right: Option<Box<DecisionNode<T>>>,
    pub prediction: Option<String>,
    pub confidence: T,
}

/// Decision rule extracted from tree
#[derive(Debug, Clone)]
pub struct DecisionRule<T: Float> {
    pub conditions: Vec<String>,
    pub prediction: String,
    pub support: T,
    pub confidence: T,
    pub description: String,
}

impl<T: Float> DecisionTreeMiner<T> {
    fn build_decision_tree(&self, experiences: &[Experience<T>]) -> Result<DecisionNode<T>> {
        // Simplified decision tree construction
        // In practice, this would use proper splitting criteria like information gain
        
        if experiences.is_empty() {
            return Ok(DecisionNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some("unknown".to_string()),
                confidence: T::zero(),
            });
        }
        
        // For simplicity, create a leaf node with majority prediction
        let prediction = "majority_action".to_string(); // Would be calculated from data
        
        Ok(DecisionNode {
            feature: None,
            threshold: None,
            left: None,
            right: None,
            prediction: Some(prediction),
            confidence: T::from(0.8).unwrap(),
        })
    }
    
    fn extract_rules(&self, tree: &DecisionNode<T>) -> Vec<DecisionRule<T>> {
        let mut rules = Vec::new();
        
        // Simplified rule extraction
        if let Some(ref prediction) = tree.prediction {
            let rule = DecisionRule {
                conditions: vec!["base_condition".to_string()],
                prediction: prediction.clone(),
                support: T::from(0.5).unwrap(),
                confidence: tree.confidence,
                description: format!("Rule: {} -> {}", "condition", prediction),
            };
            rules.push(rule);
        }
        
        rules
    }
}

impl<T: Float> PatternDiscoveryEngine<T> {
    /// Create new pattern discovery engine
    pub fn new(config: PatternConfig<T>) -> Self {
        let mut miners: Vec<Box<dyn PatternMiner<T>>> = Vec::new();
        
        // Add configured miners
        for algorithm in &config.algorithms {
            match algorithm {
                DiscoveryAlgorithm::FrequentSequences => {
                    miners.push(Box::new(FrequentSequenceMiner::new()));
                },
                DiscoveryAlgorithm::DecisionTrees => {
                    miners.push(Box::new(DecisionTreeMiner::new()));
                },
                _ => {
                    // Other algorithms would be implemented similarly
                }
            }
        }
        
        Self {
            config,
            patterns: HashMap::new(),
            miners,
        }
    }
    
    /// Discover patterns from experiences
    pub fn discover_patterns(&mut self, experiences: &[Experience<T>]) -> Result<Vec<Uuid>> {
        let mut discovered_ids = Vec::new();
        
        for miner in &self.miners {
            let patterns = miner.mine_patterns(experiences)?;
            
            for pattern in patterns {
                // Filter by support and confidence thresholds
                if pattern.support >= self.config.min_support && 
                   pattern.confidence >= self.config.min_confidence {
                    let id = pattern.id;
                    self.patterns.insert(id, pattern);
                    discovered_ids.push(id);
                }
            }
        }
        
        Ok(discovered_ids)
    }
    
    /// Get pattern by ID
    pub fn get_pattern(&self, id: &Uuid) -> Option<&Pattern<T>> {
        self.patterns.get(id)
    }
    
    /// Get all patterns of a specific type
    pub fn get_patterns_by_type(&self, pattern_type: PatternType) -> Vec<&Pattern<T>> {
        self.patterns.values()
            .filter(|p| p.pattern_type == pattern_type)
            .collect()
    }
    
    /// Update pattern usage statistics
    pub fn update_pattern_usage(&mut self, pattern_id: Uuid, success: bool, improvement: f64) {
        if let Some(pattern) = self.patterns.get_mut(&pattern_id) {
            pattern.usage_stats.applications += 1;
            
            let current_successes = pattern.usage_stats.success_rate * 
                                   (pattern.usage_stats.applications - 1) as f64;
            let new_successes = if success { current_successes + 1.0 } else { current_successes };
            pattern.usage_stats.success_rate = new_successes / pattern.usage_stats.applications as f64;
            
            let current_total = pattern.usage_stats.avg_improvement * 
                               (pattern.usage_stats.applications - 1) as f64;
            pattern.usage_stats.avg_improvement = (current_total + improvement) / 
                                                  pattern.usage_stats.applications as f64;
        }
    }
}

impl<T: Float> Default for PatternConfig<T> {
    fn default() -> Self {
        Self {
            min_support: T::from(0.1).unwrap(),
            min_confidence: T::from(0.7).unwrap(),
            max_complexity: 10,
            algorithms: vec![
                DiscoveryAlgorithm::FrequentSequences,
                DiscoveryAlgorithm::DecisionTrees,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_discovery_engine() {
        let config = PatternConfig::default();
        let mut engine: PatternDiscoveryEngine<f32> = PatternDiscoveryEngine::new(config);
        
        let experiences = vec![]; // Would create test experiences
        let result = engine.discover_patterns(&experiences);
        assert!(result.is_ok());
    }

    #[test]
    fn test_frequent_sequence_miner() {
        let miner: FrequentSequenceMiner<f32> = FrequentSequenceMiner::new();
        assert_eq!(miner.name(), "frequent_sequence_miner");
        
        let experiences = vec![]; // Would create test experiences
        let result = miner.mine_patterns(&experiences);
        assert!(result.is_ok());
    }
}