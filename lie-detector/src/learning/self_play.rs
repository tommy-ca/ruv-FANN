//! Self-play coordination for agent training
//!
//! This module manages multi-agent interactions for training through competitive
//! and cooperative scenarios.

use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

use crate::error::{Result, VeritasError};
use crate::types::*;
use super::*;

/// Self-play coordinator for managing agent interactions
pub struct SelfPlayCoordinator<T: Float> {
    /// Configuration for self-play
    pub config: SelfPlayConfig<T>,
    /// Active game sessions
    pub active_sessions: HashMap<Uuid, GameSession<T>>,
    /// Tournament manager
    pub tournament: Tournament<T>,
    /// Matchmaking system
    pub matchmaker: Matchmaker<T>,
}

/// Configuration for self-play system
#[derive(Debug, Clone)]
pub struct SelfPlayConfig<T: Float> {
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Session timeout duration
    pub session_timeout_ms: u64,
    /// Scoring weights
    pub scoring_weights: ScoringWeights<T>,
    /// Matchmaking parameters
    pub matchmaking_params: MatchmakingParams<T>,
}

/// Game session between agents
#[derive(Debug, Clone)]
pub struct GameSession<T: Float> {
    /// Session identifier
    pub id: Uuid,
    /// Participating agents
    pub agents: Vec<usize>,
    /// Current scenario
    pub scenario: GeneratedScenario<T>,
    /// Game state
    pub state: GameState<T>,
    /// Session history
    pub history: Vec<GameEvent<T>>,
    /// Start time
    pub started_at: chrono::DateTime<Utc>,
}

/// Current state of a game session
#[derive(Debug, Clone)]
pub struct GameState<T: Float> {
    /// Current turn
    pub current_turn: usize,
    /// Active player
    pub active_player: usize,
    /// Game phase
    pub phase: GamePhase,
    /// Scores by agent
    pub scores: HashMap<usize, T>,
    /// Game parameters
    pub parameters: HashMap<String, T>,
}

/// Phases of a game session
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    /// Initialization phase
    Setup,
    /// Active play phase
    Playing,
    /// Evaluation phase
    Evaluation,
    /// Session completed
    Completed,
}

/// Events that occur during game sessions
#[derive(Debug, Clone)]
pub struct GameEvent<T: Float> {
    /// Event identifier
    pub id: Uuid,
    /// Agent that triggered the event
    pub agent_id: usize,
    /// Event type
    pub event_type: EventType,
    /// Event data
    pub data: EventData<T>,
    /// Timestamp
    pub timestamp: chrono::DateTime<Utc>,
}

/// Types of game events
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    /// Agent made an observation
    Observation,
    /// Agent generated thoughts
    Reasoning,
    /// Agent took an action
    Action,
    /// Agent provided evaluation
    Evaluation,
    /// Session state changed
    StateChange,
}

/// Data associated with game events
#[derive(Debug, Clone)]
pub struct EventData<T: Float> {
    /// Event-specific data
    pub data: HashMap<String, String>,
    /// Numeric parameters
    pub parameters: HashMap<String, T>,
    /// Event confidence/quality
    pub confidence: T,
}

/// Tournament system for organized competitions
pub struct Tournament<T: Float> {
    /// Tournament configuration
    pub config: TournamentConfig<T>,
    /// Tournament brackets
    pub brackets: Vec<TournamentBracket<T>>,
    /// Tournament statistics
    pub stats: TournamentStats<T>,
}

/// Configuration for tournaments
#[derive(Debug, Clone)]
pub struct TournamentConfig<T: Float> {
    /// Tournament type
    pub tournament_type: TournamentType,
    /// Number of rounds
    pub rounds: usize,
    /// Participants per match
    pub participants_per_match: usize,
    /// Scoring system
    pub scoring_system: ScoringSystem<T>,
}

/// Types of tournaments
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TournamentType {
    /// Single elimination
    SingleElimination,
    /// Double elimination
    DoubleElimination,
    /// Round robin
    RoundRobin,
    /// Swiss system
    Swiss,
}

/// Tournament bracket structure
#[derive(Debug, Clone)]
pub struct TournamentBracket<T: Float> {
    /// Bracket identifier
    pub id: Uuid,
    /// Round number
    pub round: usize,
    /// Matches in this bracket
    pub matches: Vec<TournamentMatch<T>>,
    /// Advancement rules
    pub advancement: AdvancementRules<T>,
}

/// Individual tournament match
#[derive(Debug, Clone)]
pub struct TournamentMatch<T: Float> {
    /// Match identifier
    pub id: Uuid,
    /// Participating agents
    pub participants: Vec<usize>,
    /// Match result
    pub result: Option<MatchResult<T>>,
    /// Match metadata
    pub metadata: HashMap<String, String>,
}

/// Result of a tournament match
#[derive(Debug, Clone)]
pub struct MatchResult<T: Float> {
    /// Winner (if any)
    pub winner: Option<usize>,
    /// Final scores
    pub scores: HashMap<usize, T>,
    /// Performance metrics
    pub metrics: HashMap<String, T>,
    /// Match duration
    pub duration: std::time::Duration,
}

/// Rules for advancing in tournaments
#[derive(Debug, Clone)]
pub struct AdvancementRules<T: Float> {
    /// Minimum score to advance
    pub min_score: T,
    /// Advancement ratio
    pub advancement_ratio: T,
    /// Tiebreaker rules
    pub tiebreaker: TiebreakerRule,
}

/// Tiebreaker rules for tournaments
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TiebreakerRule {
    /// Highest average score
    HighestAverage,
    /// Best head-to-head record
    HeadToHead,
    /// Most consistent performance
    LeastVariance,
    /// Random selection
    Random,
}

/// Tournament statistics
#[derive(Debug, Clone)]
pub struct TournamentStats<T: Float> {
    /// Total matches played
    pub total_matches: usize,
    /// Average match duration
    pub avg_match_duration: std::time::Duration,
    /// Performance distribution
    pub performance_distribution: HashMap<String, T>,
    /// Participation rates
    pub participation_rates: HashMap<usize, T>,
}

/// Matchmaking system for pairing agents
pub struct Matchmaker<T: Float> {
    /// Matchmaking configuration
    pub config: MatchmakingConfig<T>,
    /// Agent ratings
    pub ratings: HashMap<usize, AgentRating<T>>,
    /// Match history
    pub match_history: Vec<MatchRecord<T>>,
}

/// Configuration for matchmaking
#[derive(Debug, Clone)]
pub struct MatchmakingConfig<T: Float> {
    /// Rating system type
    pub rating_system: RatingSystem,
    /// Skill variance tolerance
    pub skill_tolerance: T,
    /// Diversity factor
    pub diversity_factor: T,
    /// History weighting
    pub history_weight: T,
}

/// Rating systems for agents
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RatingSystem {
    /// Elo rating system
    Elo,
    /// Glicko rating system
    Glicko,
    /// TrueSkill system
    TrueSkill,
    /// Custom system
    Custom,
}

/// Agent rating and metadata
#[derive(Debug, Clone)]
pub struct AgentRating<T: Float> {
    /// Current rating
    pub rating: T,
    /// Rating uncertainty
    pub uncertainty: T,
    /// Number of games played
    pub games_played: usize,
    /// Recent performance trend
    pub trend: T,
    /// Specialized scores
    pub specialized_scores: HashMap<String, T>,
}

/// Record of a completed match
#[derive(Debug, Clone)]
pub struct MatchRecord<T: Float> {
    /// Match identifier
    pub match_id: Uuid,
    /// Participants and their ratings at the time
    pub participants: Vec<(usize, T)>,
    /// Match outcome
    pub outcome: MatchOutcome<T>,
    /// Match quality rating
    pub quality: T,
    /// Match timestamp
    pub timestamp: chrono::DateTime<Utc>,
}

/// Outcome of a match
#[derive(Debug, Clone)]
pub struct MatchOutcome<T: Float> {
    /// Final rankings
    pub rankings: Vec<usize>,
    /// Performance scores
    pub scores: HashMap<usize, T>,
    /// Improvement indicators
    pub improvements: HashMap<usize, T>,
}

/// Scoring weights for different aspects
#[derive(Debug, Clone)]
pub struct ScoringWeights<T: Float> {
    /// Accuracy weight
    pub accuracy: T,
    /// Speed weight
    pub speed: T,
    /// Reasoning quality weight
    pub reasoning_quality: T,
    /// Adaptability weight
    pub adaptability: T,
    /// Consistency weight
    pub consistency: T,
}

/// Matchmaking parameters
#[derive(Debug, Clone)]
pub struct MatchmakingParams<T: Float> {
    /// Preferred skill difference
    pub skill_difference: T,
    /// Maximum wait time
    pub max_wait_time: std::time::Duration,
    /// Minimum match quality
    pub min_match_quality: T,
}

/// Scoring system for tournaments
#[derive(Debug, Clone)]
pub struct ScoringSystem<T: Float> {
    /// Points for winning
    pub win_points: T,
    /// Points for tie/draw
    pub tie_points: T,
    /// Points for losing
    pub loss_points: T,
    /// Bonus point categories
    pub bonus_categories: HashMap<String, T>,
}

impl<T: Float> SelfPlayCoordinator<T> {
    /// Create new self-play coordinator
    pub fn new(config: SelfPlayConfig<T>) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
            tournament: Tournament::new(TournamentConfig::default()),
            matchmaker: Matchmaker::new(MatchmakingConfig::default()),
        }
    }

    /// Start new game session between agents
    pub fn start_session(
        &mut self,
        agents: Vec<usize>,
        scenario: GeneratedScenario<T>,
    ) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        
        let session = GameSession {
            id: session_id,
            agents: agents.clone(),
            scenario,
            state: GameState {
                current_turn: 0,
                active_player: agents[0],
                phase: GamePhase::Setup,
                scores: agents.iter().map(|&id| (id, T::zero())).collect(),
                parameters: HashMap::new(),
            },
            history: Vec::new(),
            started_at: Utc::now(),
        };

        self.active_sessions.insert(session_id, session);
        Ok(session_id)
    }

    /// Process game event
    pub fn process_event(
        &mut self,
        session_id: Uuid,
        event: GameEvent<T>,
    ) -> Result<()> {
        if let Some(session) = self.active_sessions.get_mut(&session_id) {
            session.history.push(event.clone());
            
            // Update game state based on event
            match event.event_type {
                EventType::Action => {
                    self.process_action_event(session, &event)?;
                },
                EventType::Evaluation => {
                    self.process_evaluation_event(session, &event)?;
                },
                _ => {
                    // Handle other event types
                }
            }
        }
        
        Ok(())
    }

    /// Process action event
    fn process_action_event(
        &mut self,
        session: &mut GameSession<T>,
        event: &GameEvent<T>,
    ) -> Result<()> {
        // Update scores based on action quality
        if let Some(score) = session.state.scores.get_mut(&event.agent_id) {
            *score = *score + event.data.confidence * T::from(0.1).unwrap();
        }

        // Advance turn
        session.state.current_turn += 1;
        
        // Switch active player
        let current_idx = session.agents.iter()
            .position(|&id| id == session.state.active_player)
            .unwrap_or(0);
        let next_idx = (current_idx + 1) % session.agents.len();
        session.state.active_player = session.agents[next_idx];

        Ok(())
    }

    /// Process evaluation event
    fn process_evaluation_event(
        &mut self,
        session: &mut GameSession<T>,
        event: &GameEvent<T>,
    ) -> Result<()> {
        // Update scores based on evaluation quality
        if let Some(score) = session.state.scores.get_mut(&event.agent_id) {
            *score = *score + event.data.confidence * T::from(0.05).unwrap();
        }

        Ok(())
    }

    /// Complete session and calculate final results
    pub fn complete_session(&mut self, session_id: Uuid) -> Result<MatchResult<T>> {
        if let Some(session) = self.active_sessions.remove(&session_id) {
            // Calculate final scores
            let mut final_scores = session.state.scores.clone();
            
            // Apply scoring weights
            for (agent_id, score) in final_scores.iter_mut() {
                *score = *score * self.config.scoring_weights.accuracy;
            }

            // Determine winner
            let winner = final_scores.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&agent_id, _)| agent_id);

            // Calculate metrics
            let mut metrics = HashMap::new();
            metrics.insert("session_length".to_string(), 
                          T::from(session.history.len() as f64).unwrap());
            metrics.insert("turns_played".to_string(), 
                          T::from(session.state.current_turn as f64).unwrap());

            let result = MatchResult {
                winner,
                scores: final_scores,
                metrics,
                duration: Utc::now().signed_duration_since(session.started_at)
                    .to_std().unwrap_or(std::time::Duration::from_secs(0)),
            };

            // Update agent ratings
            self.update_ratings(&session.agents, &result)?;

            Ok(result)
        } else {
            Err(VeritasError::learning("Session not found"))
        }
    }

    /// Update agent ratings based on match results
    fn update_ratings(&mut self, agents: &[usize], result: &MatchResult<T>) -> Result<()> {
        for &agent_id in agents {
            let rating = self.matchmaker.ratings.entry(agent_id)
                .or_insert_with(|| AgentRating {
                    rating: T::from(1200.0).unwrap(), // Default Elo rating
                    uncertainty: T::from(350.0).unwrap(),
                    games_played: 0,
                    trend: T::zero(),
                    specialized_scores: HashMap::new(),
                });

            // Update based on performance
            if let Some(&score) = result.scores.get(&agent_id) {
                let performance_factor = score.to_f64().unwrap_or(0.0);
                let rating_change = T::from(32.0 * performance_factor).unwrap(); // Simplified Elo
                rating.rating = rating.rating + rating_change;
                rating.games_played += 1;
                
                // Update uncertainty (decreases with more games)
                rating.uncertainty = rating.uncertainty * T::from(0.95).unwrap();
            }
        }

        Ok(())
    }

    /// Find optimal match using matchmaker
    pub fn find_match(&mut self, agent_pool: &[usize]) -> Result<Vec<usize>> {
        self.matchmaker.find_optimal_match(agent_pool, 2)
    }
}

impl<T: Float> Tournament<T> {
    pub fn new(config: TournamentConfig<T>) -> Self {
        Self {
            config,
            brackets: Vec::new(),
            stats: TournamentStats {
                total_matches: 0,
                avg_match_duration: std::time::Duration::from_secs(0),
                performance_distribution: HashMap::new(),
                participation_rates: HashMap::new(),
            },
        }
    }
}

impl<T: Float> Matchmaker<T> {
    pub fn new(config: MatchmakingConfig<T>) -> Self {
        Self {
            config,
            ratings: HashMap::new(),
            match_history: Vec::new(),
        }
    }

    /// Find optimal match between agents
    pub fn find_optimal_match(&self, candidates: &[usize], team_size: usize) -> Result<Vec<usize>> {
        if candidates.len() < team_size {
            return Err(VeritasError::learning("Not enough candidates for match"));
        }

        // Simple skill-based matching
        let mut rated_candidates: Vec<_> = candidates.iter()
            .map(|&id| {
                let rating = self.ratings.get(&id)
                    .map(|r| r.rating.to_f64().unwrap_or(1200.0))
                    .unwrap_or(1200.0);
                (id, rating)
            })
            .collect();

        rated_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select agents with similar ratings
        Ok(rated_candidates.into_iter()
           .take(team_size)
           .map(|(id, _)| id)
           .collect())
    }
}

// Default implementations
impl<T: Float> Default for TournamentConfig<T> {
    fn default() -> Self {
        Self {
            tournament_type: TournamentType::RoundRobin,
            rounds: 5,
            participants_per_match: 2,
            scoring_system: ScoringSystem {
                win_points: T::from(3.0).unwrap(),
                tie_points: T::from(1.0).unwrap(),
                loss_points: T::zero(),
                bonus_categories: HashMap::new(),
            },
        }
    }
}

impl<T: Float> Default for MatchmakingConfig<T> {
    fn default() -> Self {
        Self {
            rating_system: RatingSystem::Elo,
            skill_tolerance: T::from(200.0).unwrap(),
            diversity_factor: T::from(0.3).unwrap(),
            history_weight: T::from(0.1).unwrap(),
        }
    }
}

impl<T: Float> Default for SelfPlayConfig<T> {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 10,
            session_timeout_ms: 300000, // 5 minutes
            scoring_weights: ScoringWeights {
                accuracy: T::from(0.4).unwrap(),
                speed: T::from(0.2).unwrap(),
                reasoning_quality: T::from(0.3).unwrap(),
                adaptability: T::from(0.05).unwrap(),
                consistency: T::from(0.05).unwrap(),
            },
            matchmaking_params: MatchmakingParams {
                skill_difference: T::from(100.0).unwrap(),
                max_wait_time: std::time::Duration::from_secs(30),
                min_match_quality: T::from(0.6).unwrap(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_play_coordinator() {
        let config = SelfPlayConfig::default();
        let mut coordinator: SelfPlayCoordinator<f32> = SelfPlayCoordinator::new(config);
        
        // Test would create a mock scenario and agents
        assert_eq!(coordinator.active_sessions.len(), 0);
    }

    #[test]
    fn test_matchmaker() {
        let config = MatchmakingConfig::default();
        let matchmaker: Matchmaker<f32> = Matchmaker::new(config);
        
        let candidates = vec![1, 2, 3, 4];
        let match_result = matchmaker.find_optimal_match(&candidates, 2);
        assert!(match_result.is_ok());
        assert_eq!(match_result.unwrap().len(), 2);
    }
}