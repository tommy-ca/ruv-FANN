//! Memory system for ReAct agents
//!
//! This module implements a comprehensive memory system with three distinct layers:
//! - Short-term memory: Temporary working memory for current reasoning session
//! - Long-term memory: Persistent knowledge and learned patterns
//! - Episodic memory: Specific experiences and their outcomes
//!
//! The memory system supports similarity-based retrieval, temporal decay,
//! and automatic consolidation from short-term to long-term storage.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use num_traits::Float;

use crate::error::{Result, VeritasError};
use crate::types::*;

/// Multi-layered memory system for ReAct agents
pub struct Memory<T: Float> {
    config: MemoryConfig<T>,
    /// Short-term working memory
    short_term: ShortTermMemory<T>,
    /// Long-term persistent memory
    long_term: LongTermMemory<T>,
    /// Episodic experience memory
    episodic: EpisodicMemory<T>,
    /// Memory usage statistics
    stats: MemoryStats,
}

/// Types of memory storage
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryType {
    ShortTerm,
    LongTerm,
    Episodic,
}

/// Memory entry with metadata
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub content: String,
    pub memory_type: MemoryType,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub access_count: usize,
    pub relevance_score: f64,
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            memory_type,
            confidence: 1.0,
            timestamp: Utc::now(),
            access_count: 0,
            relevance_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Update access statistics
    pub fn accessed(&mut self) {
        self.access_count += 1;
        // Boost relevance with access, but with diminishing returns
        self.relevance_score = (self.relevance_score + 0.1).min(2.0);
    }

    /// Apply temporal decay to relevance
    pub fn apply_decay(&mut self, decay_rate: f64) {
        let hours_since = Utc::now().signed_duration_since(self.timestamp).num_hours() as f64;
        let decay_factor = (-decay_rate * hours_since).exp();
        self.relevance_score *= decay_factor;
        self.relevance_score = self.relevance_score.max(0.1); // Minimum relevance
    }

    /// Calculate similarity to a query string
    pub fn similarity(&self, query: &str) -> f64 {
        // Simple word-based similarity
        let content_words: Vec<&str> = self.content.to_lowercase().split_whitespace().collect();
        let query_words: Vec<&str> = query.to_lowercase().split_whitespace().collect();
        
        if content_words.is_empty() || query_words.is_empty() {
            return 0.0;
        }
        
        let mut common_words = 0;
        for query_word in &query_words {
            if content_words.contains(query_word) {
                common_words += 1;
            }
        }
        
        // Jaccard similarity with word overlap bonus
        let union_size = content_words.len() + query_words.len() - common_words;
        let base_similarity = common_words as f64 / union_size as f64;
        
        // Boost for exact phrase matches
        let phrase_bonus = if self.content.to_lowercase().contains(&query.to_lowercase()) {
            0.2
        } else {
            0.0
        };
        
        (base_similarity + phrase_bonus).min(1.0)
    }
}

/// Short-term working memory for current session
#[derive(Debug)]
pub struct ShortTermMemory<T: Float> {
    entries: VecDeque<MemoryEntry>,
    capacity: usize,
    current_size: usize,
}

impl<T: Float> ShortTermMemory<T> {
    /// Create new short-term memory with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            current_size: 0,
        }
    }

    /// Store entry in short-term memory
    pub fn store(&mut self, mut entry: MemoryEntry) -> Result<()> {
        entry.memory_type = MemoryType::ShortTerm;
        
        // Remove oldest entry if at capacity
        if self.current_size >= self.capacity {
            self.entries.pop_front();
        } else {
            self.current_size += 1;
        }
        
        self.entries.push_back(entry);
        Ok(())
    }

    /// Retrieve entries matching query
    pub fn retrieve(&mut self, query: &str, limit: usize) -> Vec<MemoryEntry> {
        let mut results: Vec<_> = self.entries
            .iter_mut()
            .map(|entry| {
                entry.accessed();
                (entry.similarity(query), entry.clone())
            })
            .filter(|(similarity, _)| *similarity > 0.1)
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter()
            .take(limit)
            .map(|(_, entry)| entry)
            .collect()
    }

    /// Get all entries for consolidation
    pub fn get_all(&self) -> Vec<MemoryEntry> {
        self.entries.iter().cloned().collect()
    }

    /// Clear all short-term memory
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_size = 0;
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f64 {
        self.current_size as f64 / self.capacity as f64
    }
}

/// Long-term persistent memory for learned patterns
#[derive(Debug)]
pub struct LongTermMemory<T: Float> {
    entries: HashMap<Uuid, MemoryEntry>,
    capacity: usize,
    indices: MemoryIndices,
}

/// Indices for efficient memory retrieval
#[derive(Debug)]
pub struct MemoryIndices {
    /// Word-based inverted index
    word_index: HashMap<String, Vec<Uuid>>,
    /// Confidence-based index
    confidence_index: Vec<(f64, Uuid)>,
    /// Temporal index
    temporal_index: Vec<(DateTime<Utc>, Uuid)>,
}

impl MemoryIndices {
    /// Create new empty indices
    pub fn new() -> Self {
        Self {
            word_index: HashMap::new(),
            confidence_index: Vec::new(),
            temporal_index: Vec::new(),
        }
    }

    /// Add entry to indices
    pub fn add_entry(&mut self, entry: &MemoryEntry) {
        // Add to word index
        for word in entry.content.to_lowercase().split_whitespace() {
            self.word_index.entry(word.to_string())
                .or_insert_with(Vec::new)
                .push(entry.id);
        }

        // Add to confidence index
        self.confidence_index.push((entry.confidence, entry.id));
        self.confidence_index.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Add to temporal index
        self.temporal_index.push((entry.timestamp, entry.id));
        self.temporal_index.sort_by(|a, b| b.0.cmp(&a.0));
    }

    /// Remove entry from indices
    pub fn remove_entry(&mut self, entry_id: Uuid) {
        // Remove from word index
        for entry_list in self.word_index.values_mut() {
            entry_list.retain(|&id| id != entry_id);
        }

        // Remove from confidence index
        self.confidence_index.retain(|(_, id)| *id != entry_id);

        // Remove from temporal index
        self.temporal_index.retain(|(_, id)| *id != entry_id);
    }

    /// Find entries by word overlap
    pub fn find_by_words(&self, words: &[&str]) -> Vec<Uuid> {
        let mut candidates: HashMap<Uuid, usize> = HashMap::new();
        
        for word in words {
            if let Some(entry_ids) = self.word_index.get(&word.to_lowercase()) {
                for &entry_id in entry_ids {
                    *candidates.entry(entry_id).or_insert(0) += 1;
                }
            }
        }

        // Sort by word overlap count
        let mut results: Vec<_> = candidates.into_iter().collect();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.into_iter().map(|(id, _)| id).collect()
    }
}

impl<T: Float> LongTermMemory<T> {
    /// Create new long-term memory with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
            indices: MemoryIndices::new(),
        }
    }

    /// Store entry in long-term memory
    pub fn store(&mut self, mut entry: MemoryEntry) -> Result<()> {
        entry.memory_type = MemoryType::LongTerm;

        // Remove lowest relevance entries if at capacity
        while self.entries.len() >= self.capacity {
            self.evict_lowest_relevance();
        }

        // Add to indices
        self.indices.add_entry(&entry);

        // Store entry
        self.entries.insert(entry.id, entry);
        Ok(())
    }

    /// Retrieve entries matching query
    pub fn retrieve(&mut self, query: &str, limit: usize) -> Vec<MemoryEntry> {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let candidate_ids = self.indices.find_by_words(&query_words);

        let mut results: Vec<_> = candidate_ids
            .into_iter()
            .filter_map(|id| self.entries.get_mut(&id))
            .map(|entry| {
                entry.accessed();
                (entry.similarity(query) * entry.relevance_score, entry.clone())
            })
            .filter(|(score, _)| *score > 0.1)
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter()
            .take(limit)
            .map(|(_, entry)| entry)
            .collect()
    }

    /// Apply temporal decay to all entries
    pub fn apply_decay(&mut self, decay_rate: f64) {
        for entry in self.entries.values_mut() {
            entry.apply_decay(decay_rate);
        }
    }

    /// Evict entry with lowest relevance score
    fn evict_lowest_relevance(&mut self) {
        if let Some((&worst_id, _)) = self.entries
            .iter()
            .min_by(|a, b| a.1.relevance_score.partial_cmp(&b.1.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)) {
            
            self.indices.remove_entry(worst_id);
            self.entries.remove(&worst_id);
        }
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f64 {
        self.entries.len() as f64 / self.capacity as f64
    }

    /// Get total number of entries
    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

/// Episodic memory for specific experiences
#[derive(Debug)]
pub struct EpisodicMemory<T: Float> {
    episodes: VecDeque<Episode<T>>,
    capacity: usize,
}

/// An episode containing context and outcome
#[derive(Debug, Clone)]
pub struct Episode<T: Float> {
    pub id: Uuid,
    pub context: String,
    pub actions_taken: Vec<String>,
    pub outcome: EpisodeOutcome<T>,
    pub timestamp: DateTime<Utc>,
    pub access_count: usize,
    pub relevance_score: f64,
}

/// Outcome of an episode
#[derive(Debug, Clone)]
pub struct EpisodeOutcome<T: Float> {
    pub success: bool,
    pub confidence: T,
    pub learned_patterns: Vec<String>,
    pub feedback: Option<String>,
}

impl<T: Float> EpisodicMemory<T> {
    /// Create new episodic memory with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Store new episode
    pub fn store_episode(&mut self, episode: Episode<T>) -> Result<()> {
        // Remove oldest episode if at capacity
        if self.episodes.len() >= self.capacity {
            self.episodes.pop_front();
        }

        self.episodes.push_back(episode);
        Ok(())
    }

    /// Retrieve similar episodes
    pub fn retrieve_similar(&mut self, context: &str, limit: usize) -> Vec<Episode<T>> {
        let mut results: Vec<_> = self.episodes
            .iter_mut()
            .map(|episode| {
                episode.access_count += 1;
                let similarity = self.episode_similarity(&episode.context, context);
                (similarity * episode.relevance_score, episode.clone())
            })
            .filter(|(score, _)| *score > 0.1)
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter()
            .take(limit)
            .map(|(_, episode)| episode)
            .collect()
    }

    /// Calculate similarity between episode contexts
    fn episode_similarity(&self, context1: &str, context2: &str) -> f64 {
        let words1: Vec<&str> = context1.to_lowercase().split_whitespace().collect();
        let words2: Vec<&str> = context2.to_lowercase().split_whitespace().collect();
        
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }
        
        let mut common = 0;
        for word1 in &words1 {
            if words2.contains(word1) {
                common += 1;
            }
        }
        
        common as f64 / (words1.len() + words2.len() - common) as f64
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f64 {
        self.episodes.len() as f64 / self.capacity as f64
    }

    /// Get total number of episodes
    pub fn size(&self) -> usize {
        self.episodes.len()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total retrievals performed
    pub total_retrievals: usize,
    /// Average retrieval time
    pub avg_retrieval_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Consolidation events
    pub consolidations: usize,
    /// Memory type usage
    pub type_usage: HashMap<MemoryType, usize>,
}

impl<T: Float> Memory<T> {
    /// Create new memory system with given configuration
    pub fn new(config: MemoryConfig<T>) -> Result<Self> {
        Ok(Self {
            short_term: ShortTermMemory::new(config.short_term_capacity),
            long_term: LongTermMemory::new(config.long_term_capacity),
            episodic: EpisodicMemory::new(config.episodic_capacity),
            config,
            stats: MemoryStats::default(),
        })
    }

    /// Store entry in short-term memory
    pub fn store_short_term_memory(&mut self, entry: MemoryEntry) -> Result<()> {
        self.short_term.store(entry)?;
        *self.stats.type_usage.entry(MemoryType::ShortTerm).or_insert(0) += 1;
        Ok(())
    }

    /// Store entry in long-term memory
    pub fn store_long_term_memory(&mut self, entry: MemoryEntry) -> Result<()> {
        self.long_term.store(entry)?;
        *self.stats.type_usage.entry(MemoryType::LongTerm).or_insert(0) += 1;
        Ok(())
    }

    /// Store episodic memory
    pub fn store_episodic_memory(&mut self, entry: MemoryEntry) -> Result<()> {
        // Convert memory entry to episode
        let episode = Episode {
            id: entry.id,
            context: entry.content,
            actions_taken: vec![], // Would be populated with actual actions
            outcome: EpisodeOutcome {
                success: entry.confidence > 0.7,
                confidence: T::from(entry.confidence).unwrap(),
                learned_patterns: vec![],
                feedback: None,
            },
            timestamp: entry.timestamp,
            access_count: entry.access_count,
            relevance_score: entry.relevance_score,
        };

        self.episodic.store_episode(episode)?;
        *self.stats.type_usage.entry(MemoryType::Episodic).or_insert(0) += 1;
        Ok(())
    }

    /// Retrieve relevant memories across all types
    pub fn retrieve_relevant(&mut self, query: &str) -> Result<Vec<MemoryEntry>> {
        let start_time = Instant::now();
        
        let limit_per_type = 5;
        let mut results = Vec::new();

        // Retrieve from short-term memory
        let mut short_term_results = self.short_term.retrieve(query, limit_per_type);
        results.append(&mut short_term_results);

        // Retrieve from long-term memory
        let mut long_term_results = self.long_term.retrieve(query, limit_per_type);
        results.append(&mut long_term_results);

        // Retrieve from episodic memory (convert episodes to memory entries)
        let episodes = self.episodic.retrieve_similar(query, limit_per_type);
        for episode in episodes {
            let entry = MemoryEntry {
                id: episode.id,
                content: episode.context,
                memory_type: MemoryType::Episodic,
                confidence: episode.outcome.confidence.to_f64().unwrap_or(0.0),
                timestamp: episode.timestamp,
                access_count: episode.access_count,
                relevance_score: episode.relevance_score,
                metadata: HashMap::new(),
            };
            results.push(entry);
        }

        // Sort by relevance and similarity
        results.sort_by(|a, b| {
            let score_a = a.similarity(query) * a.relevance_score;
            let score_b = b.similarity(query) * b.relevance_score;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update statistics
        let retrieval_time = start_time.elapsed();
        self.update_retrieval_stats(retrieval_time, !results.is_empty());

        Ok(results.into_iter().take(15).collect()) // Return top 15 results
    }

    /// Consolidate short-term memories to long-term storage
    pub fn consolidate_memories(&mut self) -> Result<usize> {
        let short_term_entries = self.short_term.get_all();
        let consolidation_threshold = self.config.similarity_threshold.to_f64().unwrap_or(0.7);
        
        let mut consolidated = 0;
        
        for entry in short_term_entries {
            // Only consolidate high-confidence, frequently accessed entries
            if entry.confidence >= consolidation_threshold && entry.access_count > 1 {
                self.long_term.store(entry)?;
                consolidated += 1;
            }
        }

        if consolidated > 0 {
            self.short_term.clear();
            self.stats.consolidations += 1;
        }

        Ok(consolidated)
    }

    /// Apply temporal decay to all memories
    pub fn apply_temporal_decay(&mut self) -> Result<()> {
        let decay_rate = self.config.decay_rate.to_f64().unwrap_or(0.01);
        self.long_term.apply_decay(decay_rate);
        Ok(())
    }

    /// Clear short-term memory
    pub fn clear_short_term_memory(&mut self) -> Result<()> {
        self.short_term.clear();
        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_usage_stats(&self) -> MemoryUsage {
        MemoryUsage {
            short_term_utilization: self.short_term.utilization(),
            long_term_utilization: self.long_term.utilization(),
            episodic_utilization: self.episodic.utilization(),
            total_entries: self.short_term.current_size + self.long_term.size() + self.episodic.size(),
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MemoryConfig<T>) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Update retrieval statistics
    fn update_retrieval_stats(&mut self, retrieval_time: Duration, cache_hit: bool) {
        self.stats.total_retrievals += 1;
        
        // Update average retrieval time
        let total_time = self.stats.avg_retrieval_time_ms * (self.stats.total_retrievals - 1) as f64;
        let new_time = retrieval_time.as_millis() as f64;
        self.stats.avg_retrieval_time_ms = (total_time + new_time) / self.stats.total_retrievals as f64;
        
        // Update cache hit rate
        let total_hits = self.stats.cache_hit_rate * (self.stats.total_retrievals - 1) as f64;
        let new_hits = if cache_hit { total_hits + 1.0 } else { total_hits };
        self.stats.cache_hit_rate = new_hits / self.stats.total_retrievals as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let config: MemoryConfig<f32> = MemoryConfig::default();
        let memory = Memory::new(config);
        assert!(memory.is_ok());
    }

    #[test]
    fn test_memory_entry_similarity() {
        let entry = MemoryEntry::new(
            "The subject showed signs of stress during questioning".to_string(),
            MemoryType::ShortTerm
        );
        
        let high_sim = entry.similarity("stress signs during interview");
        let low_sim = entry.similarity("happy celebration party");
        
        assert!(high_sim > low_sim);
        assert!(high_sim > 0.3);
        assert!(low_sim < 0.1);
    }

    #[test]
    fn test_short_term_memory() {
        let mut st_memory: ShortTermMemory<f32> = ShortTermMemory::new(3);
        
        for i in 0..5 {
            let entry = MemoryEntry::new(
                format!("Memory entry {}", i),
                MemoryType::ShortTerm
            );
            st_memory.store(entry).unwrap();
        }
        
        assert_eq!(st_memory.current_size, 3); // Should cap at capacity
        assert_eq!(st_memory.utilization(), 1.0);
    }

    #[test]
    fn test_long_term_memory() {
        let mut lt_memory: LongTermMemory<f32> = LongTermMemory::new(100);
        
        let entry = MemoryEntry::new(
            "Important pattern about deception indicators".to_string(),
            MemoryType::LongTerm
        );
        
        lt_memory.store(entry).unwrap();
        assert_eq!(lt_memory.size(), 1);
        
        let results = lt_memory.retrieve("deception patterns", 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_episodic_memory() {
        let mut ep_memory: EpisodicMemory<f32> = EpisodicMemory::new(100);
        
        let episode = Episode {
            id: Uuid::new_v4(),
            context: "Subject was nervous during questioning about finances".to_string(),
            actions_taken: vec!["analyze_voice".to_string(), "check_micro_expressions".to_string()],
            outcome: EpisodeOutcome {
                success: true,
                confidence: 0.85,
                learned_patterns: vec!["nervous_voice_pattern".to_string()],
                feedback: Some("Accurate detection".to_string()),
            },
            timestamp: Utc::now(),
            access_count: 0,
            relevance_score: 1.0,
        };
        
        ep_memory.store_episode(episode).unwrap();
        assert_eq!(ep_memory.size(), 1);
        
        let similar = ep_memory.retrieve_similar("nervous questioning", 5);
        assert!(!similar.is_empty());
    }

    #[test]
    fn test_memory_consolidation() {
        let config: MemoryConfig<f32> = MemoryConfig::default();
        let mut memory = Memory::new(config).unwrap();
        
        // Add high-confidence entries to short-term memory
        for i in 0..3 {
            let mut entry = MemoryEntry::new(
                format!("High confidence pattern {}", i),
                MemoryType::ShortTerm
            );
            entry.confidence = 0.9;
            entry.access_count = 5; // Frequently accessed
            memory.store_short_term_memory(entry).unwrap();
        }
        
        let consolidated = memory.consolidate_memories().unwrap();
        assert_eq!(consolidated, 3);
        
        // Short-term should be cleared after consolidation
        let usage = memory.get_usage_stats();
        assert_eq!(usage.short_term_utilization, 0.0);
    }

    #[test]
    fn test_temporal_decay() {
        let config: MemoryConfig<f32> = MemoryConfig::default();
        let mut memory = Memory::new(config).unwrap();
        
        let mut entry = MemoryEntry::new(
            "Test entry for decay".to_string(),
            MemoryType::LongTerm
        );
        entry.relevance_score = 1.0;
        memory.store_long_term_memory(entry).unwrap();
        
        memory.apply_temporal_decay().unwrap();
        
        // Relevance should have decayed (though minimally for recent entries)
        let usage = memory.get_usage_stats();
        assert!(usage.long_term_utilization > 0.0);
    }
}