/// Data generators for property-based testing using proptest
/// 
/// This module provides generators for creating diverse test data that explores
/// edge cases and property invariants across all system components

use proptest::prelude::*;
use num_traits::Float;
use std::collections::HashMap;

/// Probability value generator (0.0 to 1.0)
pub fn probability<T: Float + Arbitrary>() -> impl Strategy<Value = T> {
    any::<f64>()
        .prop_map(|f| f.abs().fract())
        .prop_map(|f| T::from(f).unwrap_or_else(|| T::zero()))
}

/// Confidence score generator (0.0 to 1.0)
pub fn confidence_score<T: Float + Arbitrary>() -> impl Strategy<Value = T> {
    probability::<T>()
}

/// Feature vector generator
pub fn feature_vector<T: Float + Arbitrary>(
    dim: usize,
    range: std::ops::Range<f64>,
) -> impl Strategy<Value = Vec<T>> {
    prop::collection::vec(
        range.prop_map(|f| T::from(f).unwrap_or_else(|| T::zero())),
        dim..=dim,
    )
}

/// Vision data generator
#[derive(Debug, Clone)]
pub struct VisionDataGen {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl VisionDataGen {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels }
    }
    
    /// Generate pixel data
    pub fn pixel_data(&self) -> impl Strategy<Value = Vec<u8>> {
        let size = self.width * self.height * self.channels;
        prop::collection::vec(any::<u8>(), size..=size)
    }
    
    /// Generate face landmarks
    pub fn face_landmarks(&self) -> impl Strategy<Value = Vec<(f32, f32)>> {
        let w = self.width as f32;
        let h = self.height as f32;
        prop::collection::vec(
            (0.0..w, 0.0..h),
            68..=68, // Standard 68-point model
        )
    }
    
    /// Generate facial regions
    pub fn facial_regions(&self) -> impl Strategy<Value = HashMap<String, Vec<(f32, f32)>>> {
        let regions = vec!["left_eye", "right_eye", "nose", "mouth", "jaw"];
        let w = self.width as f32;
        let h = self.height as f32;
        
        prop::collection::hash_map(
            prop::sample::select(regions),
            prop::collection::vec((0.0..w, 0.0..h), 2..=10),
            1..=5,
        )
        .prop_map(|map| {
            map.into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        })
    }
}

/// Audio data generator
#[derive(Debug, Clone)]
pub struct AudioDataGen {
    pub sample_rate: u32,
    pub max_duration_ms: u32,
}

impl AudioDataGen {
    pub fn new(sample_rate: u32, max_duration_ms: u32) -> Self {
        Self { sample_rate, max_duration_ms }
    }
    
    /// Generate audio samples
    pub fn audio_samples(&self) -> impl Strategy<Value = Vec<f32>> {
        let max_samples = (self.sample_rate * self.max_duration_ms / 1000) as usize;
        prop::collection::vec(-1.0f32..1.0f32, 100..=max_samples)
    }
    
    /// Generate pitch contour
    pub fn pitch_contour(&self, sample_count: usize) -> impl Strategy<Value = Vec<f32>> {
        let frame_count = sample_count / 256; // Typical frame size
        prop::collection::vec(80.0f32..800.0f32, frame_count..=frame_count)
    }
    
    /// Generate energy contour
    pub fn energy_contour(&self, sample_count: usize) -> impl Strategy<Value = Vec<f32>> {
        let frame_count = sample_count / 256;
        prop::collection::vec(0.0f32..1.0f32, frame_count..=frame_count)
    }
    
    /// Generate formants
    pub fn formants(&self, sample_count: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        let frame_count = sample_count / 256;
        prop::collection::vec(
            prop::collection::vec(200.0f32..3000.0f32, 3..=3), // F1, F2, F3
            frame_count..=frame_count,
        )
    }
}

/// Text data generator
pub struct TextDataGen;

impl TextDataGen {
    /// Generate text content with specified characteristics
    pub fn text_content(
        word_count: std::ops::Range<usize>,
        deceptive_probability: f64,
    ) -> impl Strategy<Value = String> {
        let truthful_words = vec![
            "definitely", "clearly", "exactly", "precisely", "certainly",
            "specifically", "absolutely", "completely", "totally", "really",
        ];
        
        let deceptive_words = vec![
            "maybe", "possibly", "might", "could", "perhaps", "sort of",
            "kind of", "I think", "I guess", "probably", "supposedly",
        ];
        
        word_count.prop_flat_map(move |count| {
            prop::collection::vec(
                prop::strategy::Union::new([
                    10 => prop::sample::select(truthful_words.clone()).boxed(),
                    (deceptive_probability * 10.0) as u32 => prop::sample::select(deceptive_words.clone()).boxed(),
                ]),
                count,
            )
        })
        .prop_map(|words| words.join(" "))
    }
    
    /// Generate linguistic features
    pub fn linguistic_features() -> impl Strategy<Value = HashMap<String, f32>> {
        prop::collection::hash_map(
            prop::sample::select(vec![
                "complexity", "certainty", "specificity", "first_person_ratio",
                "past_tense_ratio", "hedge_word_ratio", "emotion_intensity",
                "sentence_length_avg", "word_frequency_score",
            ]),
            0.0f32..1.0f32,
            5..=9,
        )
        .prop_map(|map| {
            map.into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        })
    }
}

/// Physiological data generator
#[derive(Debug, Clone)]
pub struct PhysiologicalDataGen {
    pub duration_seconds: u32,
    pub sampling_rate_hz: f32,
}

impl PhysiologicalDataGen {
    pub fn new(duration_seconds: u32, sampling_rate_hz: f32) -> Self {
        Self { duration_seconds, sampling_rate_hz }
    }
    
    /// Generate heart rate data
    pub fn heart_rate(&self) -> impl Strategy<Value = Vec<f32>> {
        let samples = (self.duration_seconds as f32 * self.sampling_rate_hz) as usize;
        prop::collection::vec(40.0f32..180.0f32, samples..=samples)
    }
    
    /// Generate skin conductance data
    pub fn skin_conductance(&self) -> impl Strategy<Value = Vec<f32>> {
        let samples = (self.duration_seconds as f32 * self.sampling_rate_hz) as usize;
        prop::collection::vec(0.1f32..20.0f32, samples..=samples)
    }
    
    /// Generate blood pressure data
    pub fn blood_pressure(&self) -> impl Strategy<Value = Vec<(f32, f32)>> {
        let samples = (self.duration_seconds as f32 * self.sampling_rate_hz) as usize;
        prop::collection::vec(
            (80.0f32..200.0f32, 40.0f32..120.0f32), // (systolic, diastolic)
            samples..=samples,
        )
    }
    
    /// Generate respiration rate data
    pub fn respiration_rate(&self) -> impl Strategy<Value = Vec<f32>> {
        let samples = (self.duration_seconds as f32 * self.sampling_rate_hz) as usize;
        prop::collection::vec(8.0f32..40.0f32, samples..=samples)
    }
}

/// Neural network configuration generator
pub fn network_config<T: Float + Arbitrary>() -> impl Strategy<Value = NetworkConfigGen<T>> {
    (1..=1000usize, 1..=10usize, 0.001f64..0.1f64, 10..=1000usize)
        .prop_map(|(input_size, layers, learning_rate, epochs)| {
            NetworkConfigGen {
                input_size,
                hidden_layers: (1..=layers).map(|i| input_size / (i + 1).max(1)).collect(),
                output_size: 1,
                learning_rate: T::from(learning_rate).unwrap_or_else(|| T::from(0.01).unwrap()),
                epochs,
            }
        })
}

#[derive(Debug, Clone)]
pub struct NetworkConfigGen<T: Float> {
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub learning_rate: T,
    pub epochs: usize,
}

/// Fusion strategy configuration generator
pub fn fusion_config<T: Float + Arbitrary>() -> impl Strategy<Value = FusionConfigGen<T>> {
    (
        prop::collection::vec(probability::<T>(), 2..=5), // modality weights
        0.1f64..0.9f64, // confidence threshold
        prop::sample::select(vec!["early", "late", "attention", "hybrid"]), // strategy type
    )
    .prop_map(|(weights, confidence_threshold, strategy)| {
        FusionConfigGen {
            modality_weights: weights,
            confidence_threshold: T::from(confidence_threshold).unwrap_or_else(|| T::from(0.5).unwrap()),
            strategy_type: strategy.to_string(),
        }
    })
}

#[derive(Debug, Clone)]
pub struct FusionConfigGen<T: Float> {
    pub modality_weights: Vec<T>,
    pub confidence_threshold: T,
    pub strategy_type: String,
}

/// Agent configuration generator
pub fn agent_config<T: Float + Arbitrary>() -> impl Strategy<Value = AgentConfigGen<T>> {
    (
        1..=20usize, // max reasoning steps
        0.1f64..0.9f64, // uncertainty threshold
        10..=1000usize, // memory capacity
        0.01f64..0.3f64, // learning rate
    )
    .prop_map(|(max_steps, uncertainty_threshold, memory_capacity, learning_rate)| {
        AgentConfigGen {
            max_reasoning_steps: max_steps,
            uncertainty_threshold: T::from(uncertainty_threshold).unwrap_or_else(|| T::from(0.3).unwrap()),
            memory_capacity,
            learning_rate: T::from(learning_rate).unwrap_or_else(|| T::from(0.1).unwrap()),
        }
    })
}

#[derive(Debug, Clone)]
pub struct AgentConfigGen<T: Float> {
    pub max_reasoning_steps: usize,
    pub uncertainty_threshold: T,
    pub memory_capacity: usize,
    pub learning_rate: T,
}

/// Complete system configuration generator
pub fn system_config<T: Float + Arbitrary>() -> impl Strategy<Value = SystemConfigGen<T>> {
    (
        network_config::<T>(),
        fusion_config::<T>(),
        agent_config::<T>(),
        any::<bool>(), // enable_gpu
        any::<bool>(), // enable_parallel
    )
    .prop_map(|(network, fusion, agent, enable_gpu, enable_parallel)| {
        SystemConfigGen {
            network,
            fusion,
            agent,
            enable_gpu,
            enable_parallel,
        }
    })
}

#[derive(Debug, Clone)]
pub struct SystemConfigGen<T: Float> {
    pub network: NetworkConfigGen<T>,
    pub fusion: FusionConfigGen<T>,
    pub agent: AgentConfigGen<T>,
    pub enable_gpu: bool,
    pub enable_parallel: bool,
}

/// Generate edge case inputs for robustness testing
pub mod edge_cases {
    use super::*;
    
    /// Generate empty or minimal inputs
    pub fn minimal_inputs<T: Float + Arbitrary>() -> impl Strategy<Value = MinimalInputs<T>> {
        (
            prop::option::of(prop::collection::vec(any::<u8>(), 0..=10)), // minimal image
            prop::option::of(prop::collection::vec(-1.0f32..1.0f32, 0..=10)), // minimal audio
            prop::option::of("".to_string()..="hi".to_string()), // minimal text
        )
        .prop_map(|(image, audio, text)| MinimalInputs { image, audio, text })
    }
    
    #[derive(Debug, Clone)]
    pub struct MinimalInputs<T: Float> {
        pub image: Option<Vec<u8>>,
        pub audio: Option<Vec<f32>>,
        pub text: Option<String>,
    }
    
    /// Generate extreme values
    pub fn extreme_values<T: Float + Arbitrary>() -> impl Strategy<Value = ExtremeValues<T>> {
        (
            prop::sample::select(vec![T::zero(), T::one(), T::infinity(), T::neg_infinity()]),
            prop::sample::select(vec![f32::MIN, f32::MAX, f32::INFINITY, f32::NEG_INFINITY]),
            0usize..=0usize, // empty collections
        )
        .prop_map(|(extreme_float, extreme_f32, _zero_size)| ExtremeValues {
            extreme_probability: extreme_float,
            extreme_audio_sample: extreme_f32,
            empty_feature_vector: vec![],
        })
    }
    
    #[derive(Debug, Clone)]
    pub struct ExtremeValues<T: Float> {
        pub extreme_probability: T,
        pub extreme_audio_sample: f32,
        pub empty_feature_vector: Vec<T>,
    }
}