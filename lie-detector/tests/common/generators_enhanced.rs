/// Enhanced data generators for comprehensive property-based testing
/// 
/// This module provides advanced generators for creating diverse test data that explores
/// edge cases, performance scenarios, and realistic patterns across all system components

use proptest::prelude::*;
use num_traits::Float;
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use chrono::{DateTime, Utc, Duration};

/// Seedable random number generator for deterministic testing
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Probability value generator with enhanced edge case coverage
pub fn enhanced_probability<T: Float + Arbitrary>() -> impl Strategy<Value = T> {
    prop::strategy::Union::new([
        9 => any::<f64>()
            .prop_map(|f| f.abs().fract())
            .prop_map(|f| T::from(f).unwrap_or_else(|| T::zero()))
            .boxed(),
        1 => prop::sample::select(vec![T::zero(), T::one()])
            .boxed(),
    ])
}

/// Generate realistic multi-modal test scenarios
pub mod realistic_scenarios {
    use super::*;
    
    /// Generate interview or interrogation scenarios
    pub fn interview_scenario() -> impl Strategy<Value = InterviewScenario> {
        (
            1..10usize, // number of questions
            prop::sample::select(vec!["formal", "casual", "adversarial", "supportive"]),
            0.0..1.0f64, // stress level
            prop::collection::vec("(yes|no|maybe|I don't know)".to_string(), 1..20),
        ).prop_map(|(question_count, interview_style, stress_level, responses)| {
            InterviewScenario {
                question_count,
                interview_style: interview_style.to_string(),
                stress_level,
                responses,
                duration_minutes: question_count * 2, // 2 min per question
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct InterviewScenario {
        pub question_count: usize,
        pub interview_style: String,
        pub stress_level: f64,
        pub responses: Vec<String>,
        pub duration_minutes: usize,
    }
    
    /// Generate court testimony scenarios
    pub fn court_testimony() -> impl Strategy<Value = CourtTestimony> {
        (
            prop::sample::select(vec!["witness", "defendant", "expert"]),
            1..30usize, // statements
            prop::sample::select(vec!["criminal", "civil", "family"]),
            0.0..1.0f64, // credibility baseline
        ).prop_map(|(role, statement_count, case_type, credibility)| {
            CourtTestimony {
                role: role.to_string(),
                statement_count,
                case_type: case_type.to_string(),
                baseline_credibility: credibility,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct CourtTestimony {
        pub role: String,
        pub statement_count: usize,
        pub case_type: String,
        pub baseline_credibility: f64,
    }
    
    /// Generate security screening scenarios
    pub fn security_screening() -> impl Strategy<Value = SecurityScreening> {
        (
            prop::sample::select(vec!["airport", "border", "government", "corporate"]),
            prop::collection::vec(prop::sample::select(vec!["ID_check", "bag_search", "interview", "background_verify"]), 1..5),
            0.0..1.0f64, // threat level
        ).prop_map(|(location, procedures, threat_level)| {
            SecurityScreening {
                location: location.to_string(),
                procedures: procedures.into_iter().map(|s| s.to_string()).collect(),
                threat_level,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct SecurityScreening {
        pub location: String,
        pub procedures: Vec<String>,
        pub threat_level: f64,
    }
}

/// Performance and stress testing data generators
pub mod performance {
    use super::*;
    
    /// Generate large-scale datasets for performance testing
    pub fn large_multimodal_dataset<T: Float + Arbitrary>(
        samples: usize,
        vision_features: usize,
        audio_features: usize,
        text_features: usize,
    ) -> impl Strategy<Value = LargeDataset<T>> {
        (
            prop::collection::vec(prop::collection::vec(any::<u8>(), vision_features), samples),
            prop::collection::vec(prop::collection::vec(-1.0f32..1.0f32, audio_features), samples),
            prop::collection::vec(prop::collection::vec(enhanced_probability::<T>(), text_features), samples),
            prop::collection::vec(any::<bool>(), samples), // labels
        ).prop_map(|(vision_data, audio_data, text_data, labels)| {
            LargeDataset {
                vision_data,
                audio_data,
                text_data,
                labels,
                sample_count: samples,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct LargeDataset<T: Float> {
        pub vision_data: Vec<Vec<u8>>,
        pub audio_data: Vec<Vec<f32>>,
        pub text_data: Vec<Vec<T>>,
        pub labels: Vec<bool>,
        pub sample_count: usize,
    }
    
    /// Generate streaming performance scenarios
    pub fn streaming_scenario() -> impl Strategy<Value = StreamingScenario> {
        (
            1..100usize, // frames per second
            1..3600usize, // duration in seconds
            1..10usize, // concurrent streams
            1..100u64, // processing latency ms
        ).prop_map(|(fps, duration, streams, latency)| {
            StreamingScenario {
                fps,
                duration_seconds: duration,
                concurrent_streams: streams,
                target_latency_ms: latency,
                total_frames: fps * duration * streams,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct StreamingScenario {
        pub fps: usize,
        pub duration_seconds: usize,
        pub concurrent_streams: usize,
        pub target_latency_ms: u64,
        pub total_frames: usize,
    }
    
    /// Generate memory pressure scenarios
    pub fn memory_pressure() -> impl Strategy<Value = MemoryPressureScenario> {
        (
            1_000..10_000_000usize, // allocation size
            1..1000usize, // allocation count
            0..5000u64, // delay between allocations (ms)
            prop::sample::select(vec!["linear", "exponential", "random"]),
        ).prop_map(|(size, count, delay, pattern)| {
            MemoryPressureScenario {
                allocation_size: size,
                allocation_count: count,
                delay_ms: delay,
                allocation_pattern: pattern.to_string(),
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct MemoryPressureScenario {
        pub allocation_size: usize,
        pub allocation_count: usize,
        pub delay_ms: u64,
        pub allocation_pattern: String,
    }
}

/// Adversarial and edge case generators
pub mod adversarial {
    use super::*;
    
    /// Generate adversarial inputs designed to fool the system
    pub fn adversarial_inputs<T: Float + Arbitrary>() -> impl Strategy<Value = AdversarialInputs<T>> {
        (
            prop::sample::select(vec!["gradient_ascent", "random_noise", "targeted_perturbation"]),
            0.001..0.1f64, // perturbation magnitude
            enhanced_probability::<T>(), // target confidence
        ).prop_map(|(attack_type, magnitude, target)| {
            AdversarialInputs {
                attack_type: attack_type.to_string(),
                perturbation_magnitude: magnitude,
                target_confidence: target,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct AdversarialInputs<T: Float> {
        pub attack_type: String,
        pub perturbation_magnitude: f64,
        pub target_confidence: T,
    }
    
    /// Generate data poisoning scenarios
    pub fn data_poisoning<T: Float + Arbitrary>() -> impl Strategy<Value = DataPoisoning<T>> {
        (
            0.01..0.3f64, // poisoning ratio
            prop::sample::select(vec!["label_flipping", "feature_corruption", "backdoor_injection"]),
            enhanced_probability::<T>(), // poison strength
        ).prop_map(|(ratio, poison_type, strength)| {
            DataPoisoning {
                poisoning_ratio: ratio,
                poison_type: poison_type.to_string(),
                poison_strength: strength,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct DataPoisoning<T: Float> {
        pub poisoning_ratio: f64,
        pub poison_type: String,
        pub poison_strength: T,
    }
    
    /// Generate privacy attack scenarios
    pub fn privacy_attack() -> impl Strategy<Value = PrivacyAttack> {
        (
            prop::sample::select(vec!["membership_inference", "model_inversion", "attribute_inference"]),
            0.1..0.9f64, // attack success rate
            1..1000usize, // query budget
        ).prop_map(|(attack_type, success_rate, queries)| {
            PrivacyAttack {
                attack_type: attack_type.to_string(),
                expected_success_rate: success_rate,
                query_budget: queries,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct PrivacyAttack {
        pub attack_type: String,
        pub expected_success_rate: f64,
        pub query_budget: usize,
    }
}

/// Cross-modal consistency and temporal generators
pub mod cross_modal {
    use super::*;
    
    /// Generate temporally aligned multi-modal data
    pub fn temporal_alignment<T: Float + Arbitrary>(duration_seconds: usize) -> impl Strategy<Value = TemporalAlignment<T>> {
        let base_time = Utc::now();
        (
            prop::collection::vec(enhanced_probability::<T>(), duration_seconds * 30), // 30 FPS video
            prop::collection::vec(-1.0f32..1.0f32, duration_seconds * 16000), // 16kHz audio
            prop::collection::vec(0..1000usize, duration_seconds), // text events per second
        ).prop_map(move |(vision_scores, audio_samples, text_events)| {
            let timestamps: Vec<DateTime<Utc>> = (0..duration_seconds)
                .map(|i| base_time + Duration::seconds(i as i64))
                .collect();
            
            TemporalAlignment {
                duration_seconds,
                vision_scores,
                audio_samples,
                text_events,
                timestamps,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct TemporalAlignment<T: Float> {
        pub duration_seconds: usize,
        pub vision_scores: Vec<T>,
        pub audio_samples: Vec<f32>,
        pub text_events: Vec<usize>,
        pub timestamps: Vec<DateTime<Utc>>,
    }
    
    /// Generate cross-modal consistency patterns
    pub fn consistency_patterns<T: Float + Arbitrary>() -> impl Strategy<Value = ConsistencyPattern<T>> {
        (
            prop::sample::select(vec!["consistent", "partially_consistent", "conflicting", "noisy"]),
            0.0..1.0f64, // consistency strength
            prop::collection::vec(enhanced_probability::<T>(), 4), // modality scores
        ).prop_map(|(pattern_type, strength, base_scores)| {
            let modality_scores = match pattern_type {
                "consistent" => {
                    let target = base_scores[0];
                    let noise = T::from(0.05 * strength).unwrap_or_else(|| T::zero());
                    vec![
                        target,
                        target + noise,
                        target - noise,
                        target + noise / T::from(2.0).unwrap(),
                    ]
                },
                "conflicting" => {
                    base_scores // Use original diverse scores
                },
                _ => {
                    // Add controlled noise based on pattern
                    base_scores.into_iter().enumerate().map(|(i, score)| {
                        let noise_factor = T::from(0.1 * strength * (i + 1) as f64).unwrap_or_else(|| T::zero());
                        score + noise_factor
                    }).collect()
                }
            };
            
            ConsistencyPattern {
                pattern_type: pattern_type.to_string(),
                consistency_strength: strength,
                vision_score: modality_scores[0],
                audio_score: modality_scores[1],
                text_score: modality_scores[2],
                physiological_score: modality_scores[3],
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct ConsistencyPattern<T: Float> {
        pub pattern_type: String,
        pub consistency_strength: f64,
        pub vision_score: T,
        pub audio_score: T,
        pub text_score: T,
        pub physiological_score: T,
    }
}

/// Biometric and physiological data generators
pub mod biometric {
    use super::*;
    
    /// Generate realistic ECG patterns
    pub fn ecg_pattern(duration_seconds: usize, heart_rate_bpm: f32) -> impl Strategy<Value = ECGPattern> {
        let sample_rate = 250; // 250 Hz typical for ECG
        let samples = duration_seconds * sample_rate;
        
        prop::collection::vec(-2.0f32..2.0f32, samples)
            .prop_map(move |noise| {
                let mut ecg_data = Vec::new();
                let beat_interval = (60.0 * sample_rate as f32 / heart_rate_bpm) as usize;
                
                for i in 0..samples {
                    let beat_phase = (i % beat_interval) as f32 / beat_interval as f32;
                    let ecg_value = generate_ecg_waveform(beat_phase) + noise[i] * 0.1;
                    ecg_data.push(ecg_value);
                }
                
                ECGPattern {
                    samples: ecg_data,
                    sample_rate,
                    heart_rate_bpm,
                    duration_seconds,
                }
            })
    }
    
    #[derive(Debug, Clone)]
    pub struct ECGPattern {
        pub samples: Vec<f32>,
        pub sample_rate: usize,
        pub heart_rate_bpm: f32,
        pub duration_seconds: usize,
    }
    
    /// Generate EEG-like patterns
    pub fn eeg_pattern(channels: usize, duration_seconds: usize) -> impl Strategy<Value = EEGPattern> {
        let sample_rate = 256; // 256 Hz typical for EEG
        let samples_per_channel = duration_seconds * sample_rate;
        
        prop::collection::vec(
            prop::collection::vec(-100.0f32..100.0f32, samples_per_channel),
            channels
        ).prop_map(move |channel_data| {
            EEGPattern {
                channels: channel_data,
                sample_rate,
                duration_seconds,
                channel_count: channels,
            }
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct EEGPattern {
        pub channels: Vec<Vec<f32>>,
        pub sample_rate: usize,
        pub duration_seconds: usize,
        pub channel_count: usize,
    }
    
    /// Generate GSR (Galvanic Skin Response) patterns
    pub fn gsr_pattern(duration_seconds: usize, stress_level: f32) -> impl Strategy<Value = GSRPattern> {
        let sample_rate = 32; // 32 Hz typical for GSR
        let samples = duration_seconds * sample_rate;
        
        prop::collection::vec(0.0f32..0.5f32, samples)
            .prop_map(move |noise| {
                let baseline = 2.0 + stress_level * 3.0; // 2-5 microsiemens range
                let samples: Vec<f32> = (0..samples)
                    .map(|i| {
                        let t = i as f32 / sample_rate as f32;
                        let stress_component = stress_level * 0.5 * (0.1 * t).sin();
                        baseline + stress_component + noise[i]
                    })
                    .collect();
                
                GSRPattern {
                    samples,
                    sample_rate,
                    duration_seconds,
                    baseline_stress: stress_level,
                }
            })
    }
    
    #[derive(Debug, Clone)]
    pub struct GSRPattern {
        pub samples: Vec<f32>,
        pub sample_rate: usize,
        pub duration_seconds: usize,
        pub baseline_stress: f32,
    }
    
    fn generate_ecg_waveform(phase: f32) -> f32 {
        // Simplified ECG waveform generation
        if phase < 0.2 {
            // P wave
            0.2 * (std::f32::consts::PI * phase / 0.2).sin()
        } else if phase < 0.4 {
            // QRS complex
            if phase < 0.35 {
                -0.3 * ((phase - 0.2) / 0.15).sin()
            } else {
                1.2 * ((phase - 0.35) / 0.05).sin()
            }
        } else if phase < 0.6 {
            // T wave
            0.4 * (std::f32::consts::PI * (phase - 0.4) / 0.2).sin()
        } else {
            // Baseline
            0.0
        }
    }
}

/// Utility functions for test data validation and transformation
pub mod utils {
    use super::*;
    
    /// Validate that generated probabilities are in valid range
    pub fn validate_probability<T: Float>(prob: T) -> bool {
        prob >= T::zero() && prob <= T::one() && prob.is_finite()
    }
    
    /// Validate feature vector for NaN/Inf values
    pub fn validate_feature_vector<T: Float>(features: &[T]) -> bool {
        features.iter().all(|&f| f.is_finite())
    }
    
    /// Transform probability to logit space for numerical stability testing
    pub fn to_logit<T: Float>(prob: T) -> T {
        let eps = T::from(1e-8).unwrap();
        let clamped = prob.max(eps).min(T::one() - eps);
        (clamped / (T::one() - clamped)).ln()
    }
    
    /// Transform logit back to probability
    pub fn from_logit<T: Float>(logit: T) -> T {
        T::one() / (T::one() + (-logit).exp())
    }
    
    /// Calculate statistical properties of generated data
    pub fn calculate_statistics<T: Float>(data: &[T]) -> DataStatistics<T> {
        if data.is_empty() {
            return DataStatistics::default();
        }
        
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / T::from(data.len()).unwrap();
        
        let variance = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(data.len()).unwrap();
        
        let std_dev = variance.sqrt();
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let median = if data.len() % 2 == 0 {
            let mid = data.len() / 2;
            (sorted_data[mid - 1] + sorted_data[mid]) / T::from(2.0).unwrap()
        } else {
            sorted_data[data.len() / 2]
        };
        
        DataStatistics {
            mean,
            median,
            std_dev,
            min: sorted_data[0],
            max: sorted_data[data.len() - 1],
            count: data.len(),
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct DataStatistics<T: Float> {
        pub mean: T,
        pub median: T,
        pub std_dev: T,
        pub min: T,
        pub max: T,
        pub count: usize,
    }
    
    impl<T: Float> Default for DataStatistics<T> {
        fn default() -> Self {
            Self {
                mean: T::zero(),
                median: T::zero(),
                std_dev: T::zero(),
                min: T::zero(),
                max: T::zero(),
                count: 0,
            }
        }
    }
}