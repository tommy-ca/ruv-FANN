/// Test fixtures for generating consistent test data
/// 
/// This module provides standardized test data for all modalities and components

use num_traits::Float;
use std::collections::HashMap;

/// Sample vision data for testing
#[derive(Debug, Clone)]
pub struct VisionTestData {
    pub image_width: usize,
    pub image_height: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
    pub face_landmarks: Vec<(f32, f32)>,
    pub facial_regions: HashMap<String, Vec<(f32, f32)>>,
}

impl VisionTestData {
    /// Create a simple test image with known patterns
    pub fn new_simple() -> Self {
        let width = 224;
        let height = 224;
        let channels = 3;
        let size = width * height * channels;
        
        // Create a gradient pattern for testing
        let mut pixels = Vec::with_capacity(size);
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let value = ((x + y + c) % 256) as u8;
                    pixels.push(value);
                }
            }
        }
        
        // Mock face landmarks (68-point model)
        let face_landmarks = Self::generate_face_landmarks();
        
        // Mock facial regions
        let mut facial_regions = HashMap::new();
        facial_regions.insert("left_eye".to_string(), vec![(50.0, 60.0), (70.0, 80.0)]);
        facial_regions.insert("right_eye".to_string(), vec![(150.0, 60.0), (170.0, 80.0)]);
        facial_regions.insert("mouth".to_string(), vec![(100.0, 140.0), (140.0, 160.0)]);
        
        Self {
            image_width: width,
            image_height: height,
            channels,
            pixels,
            face_landmarks,
            facial_regions,
        }
    }
    
    /// Generate realistic face landmark points
    fn generate_face_landmarks() -> Vec<(f32, f32)> {
        // 68-point facial landmark model coordinates (normalized to 224x224)
        vec![
            // Jaw line (0-16)
            (84.0, 186.0), (86.0, 196.0), (90.0, 206.0), (96.0, 214.0),
            (104.0, 220.0), (114.0, 224.0), (124.0, 226.0), (134.0, 226.0),
            (144.0, 224.0), (154.0, 220.0), (162.0, 214.0), (168.0, 206.0),
            (172.0, 196.0), (174.0, 186.0), (174.0, 176.0), (172.0, 166.0),
            (168.0, 156.0),
            // Right eyebrow (17-21)
            (96.0, 100.0), (106.0, 92.0), (118.0, 90.0), (130.0, 92.0), (140.0, 98.0),
            // Left eyebrow (22-26)
            (148.0, 98.0), (158.0, 92.0), (170.0, 90.0), (182.0, 92.0), (192.0, 100.0),
            // Nose (27-35)
            (144.0, 112.0), (144.0, 122.0), (144.0, 132.0), (144.0, 142.0),
            (136.0, 148.0), (140.0, 150.0), (144.0, 152.0), (148.0, 150.0), (152.0, 148.0),
            // Right eye (36-41)
            (106.0, 112.0), (114.0, 108.0), (122.0, 108.0), (130.0, 112.0),
            (122.0, 116.0), (114.0, 116.0),
            // Left eye (42-47)
            (150.0, 112.0), (158.0, 108.0), (166.0, 108.0), (174.0, 112.0),
            (166.0, 116.0), (158.0, 116.0),
            // Mouth (48-67)
            (122.0, 170.0), (130.0, 166.0), (138.0, 164.0), (144.0, 166.0),
            (150.0, 164.0), (158.0, 166.0), (166.0, 170.0), (158.0, 178.0),
            (150.0, 182.0), (144.0, 184.0), (138.0, 182.0), (130.0, 178.0),
            (126.0, 170.0), (138.0, 170.0), (144.0, 172.0), (150.0, 170.0),
            (162.0, 170.0), (150.0, 176.0), (144.0, 178.0), (138.0, 176.0),
        ]
    }
    
    /// Create test data with deceptive patterns
    pub fn new_deceptive() -> Self {
        let mut data = Self::new_simple();
        
        // Modify landmarks to simulate micro-expressions
        data.face_landmarks[48] = (120.0, 172.0); // Mouth corner tension
        data.face_landmarks[54] = (168.0, 172.0); // Asymmetric mouth
        
        // Add stress patterns in eye region
        data.face_landmarks[36] = (104.0, 114.0); // Eye squeeze
        data.face_landmarks[42] = (152.0, 114.0); // Asymmetric eyes
        
        data
    }
    
    /// Create test data with truthful patterns
    pub fn new_truthful() -> Self {
        let mut data = Self::new_simple();
        
        // Symmetric, relaxed facial features
        data.face_landmarks[48] = (122.0, 170.0); // Relaxed mouth
        data.face_landmarks[54] = (166.0, 170.0); // Symmetric mouth
        
        data
    }
}

/// Sample audio data for testing
#[derive(Debug, Clone)]
pub struct AudioTestData {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub duration_ms: u32,
    pub pitch_contour: Vec<f32>,
    pub energy_contour: Vec<f32>,
    pub formants: Vec<Vec<f32>>,
}

impl AudioTestData {
    /// Create simple test audio (sine wave)
    pub fn new_simple() -> Self {
        let sample_rate = 16000;
        let duration_ms = 1000;
        let sample_count = (sample_rate * duration_ms / 1000) as usize;
        
        // Generate sine wave at 440 Hz
        let frequency = 440.0;
        let samples: Vec<f32> = (0..sample_count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();
        
        // Mock pitch contour (Hz)
        let pitch_contour = vec![440.0; sample_count / 100];
        
        // Mock energy contour
        let energy_contour = vec![0.5; sample_count / 100];
        
        // Mock formants (F1, F2, F3)
        let formants = vec![
            vec![700.0, 1220.0, 2600.0]; sample_count / 100
        ];
        
        Self {
            sample_rate,
            samples,
            duration_ms,
            pitch_contour,
            energy_contour,
            formants,
        }
    }
    
    /// Create audio with stress indicators
    pub fn new_stressed() -> Self {
        let mut data = Self::new_simple();
        
        // Simulate voice stress patterns
        for (i, pitch) in data.pitch_contour.iter_mut().enumerate() {
            // Add pitch instability
            let variation = 0.1 * (i as f32 * 0.1).sin();
            *pitch += *pitch * variation;
        }
        
        // Reduce energy (quieter under stress)
        for energy in data.energy_contour.iter_mut() {
            *energy *= 0.7;
        }
        
        // Shift formants (vocal tract tension)
        for formant_set in data.formants.iter_mut() {
            formant_set[0] += 50.0; // F1 shift
            formant_set[1] += 100.0; // F2 shift
        }
        
        data
    }
    
    /// Create relaxed audio patterns
    pub fn new_relaxed() -> Self {
        let mut data = Self::new_simple();
        
        // Stable pitch
        for pitch in data.pitch_contour.iter_mut() {
            *pitch = 440.0; // Stable fundamental frequency
        }
        
        // Consistent energy
        for energy in data.energy_contour.iter_mut() {
            *energy = 0.6;
        }
        
        data
    }
}

/// Sample text data for testing
#[derive(Debug, Clone)]
pub struct TextTestData {
    pub content: String,
    pub word_count: usize,
    pub sentence_count: usize,
    pub linguistic_features: HashMap<String, f32>,
}

impl TextTestData {
    /// Create simple truthful statement
    pub fn new_truthful() -> Self {
        let content = "I went to the store yesterday to buy groceries. I purchased milk, bread, and eggs. The cashier was very friendly and helpful.".to_string();
        
        let mut linguistic_features = HashMap::new();
        linguistic_features.insert("complexity".to_string(), 0.6);
        linguistic_features.insert("certainty".to_string(), 0.8);
        linguistic_features.insert("specificity".to_string(), 0.7);
        linguistic_features.insert("first_person_ratio".to_string(), 0.3);
        linguistic_features.insert("past_tense_ratio".to_string(), 0.8);
        
        Self {
            content,
            word_count: 22,
            sentence_count: 3,
            linguistic_features,
        }
    }
    
    /// Create deceptive statement patterns
    pub fn new_deceptive() -> Self {
        let content = "I think I might have gone to some store or something. Maybe I bought some things, I'm not really sure. The person there was okay I guess.".to_string();
        
        let mut linguistic_features = HashMap::new();
        linguistic_features.insert("complexity".to_string(), 0.3);
        linguistic_features.insert("certainty".to_string(), 0.2);
        linguistic_features.insert("specificity".to_string(), 0.1);
        linguistic_features.insert("first_person_ratio".to_string(), 0.4);
        linguistic_features.insert("hedge_word_ratio".to_string(), 0.6);
        
        Self {
            content,
            word_count: 26,
            sentence_count: 3,
            linguistic_features,
        }
    }
    
    /// Create neutral statement
    pub fn new_neutral() -> Self {
        let content = "The weather today is partly cloudy with temperatures around 70 degrees. Traffic on the main roads appears to be moving normally.".to_string();
        
        let mut linguistic_features = HashMap::new();
        linguistic_features.insert("complexity".to_string(), 0.5);
        linguistic_features.insert("certainty".to_string(), 0.6);
        linguistic_features.insert("specificity".to_string(), 0.6);
        linguistic_features.insert("first_person_ratio".to_string(), 0.0);
        linguistic_features.insert("present_tense_ratio".to_string(), 0.8);
        
        Self {
            content,
            word_count: 21,
            sentence_count: 2,
            linguistic_features,
        }
    }
}

/// Sample physiological data for testing
#[derive(Debug, Clone)]
pub struct PhysiologicalTestData {
    pub heart_rate_bpm: Vec<f32>,
    pub skin_conductance: Vec<f32>,
    pub blood_pressure: Vec<(f32, f32)>, // (systolic, diastolic)
    pub respiration_rate: Vec<f32>,
    pub sampling_rate_hz: f32,
}

impl PhysiologicalTestData {
    /// Create baseline physiological data
    pub fn new_baseline() -> Self {
        let samples = 300; // 5 minutes at 1Hz
        let sampling_rate_hz = 1.0;
        
        // Normal resting values with natural variation
        let heart_rate_bpm: Vec<f32> = (0..samples)
            .map(|i| 72.0 + 5.0 * (i as f32 * 0.1).sin())
            .collect();
        
        let skin_conductance: Vec<f32> = (0..samples)
            .map(|_| 2.0 + rand::random::<f32>() * 0.5)
            .collect();
        
        let blood_pressure: Vec<(f32, f32)> = (0..samples)
            .map(|_| (120.0 + rand::random::<f32>() * 10.0, 80.0 + rand::random::<f32>() * 5.0))
            .collect();
        
        let respiration_rate: Vec<f32> = (0..samples)
            .map(|_| 16.0 + rand::random::<f32>() * 2.0)
            .collect();
        
        Self {
            heart_rate_bpm,
            skin_conductance,
            blood_pressure,
            respiration_rate,
            sampling_rate_hz,
        }
    }
    
    /// Create stressed physiological patterns
    pub fn new_stressed() -> Self {
        let mut data = Self::new_baseline();
        
        // Elevated heart rate
        for hr in data.heart_rate_bpm.iter_mut() {
            *hr += 20.0; // Stress increases HR
        }
        
        // Increased skin conductance
        for sc in data.skin_conductance.iter_mut() {
            *sc *= 1.5;
        }
        
        // Elevated blood pressure
        for (systolic, diastolic) in data.blood_pressure.iter_mut() {
            *systolic += 15.0;
            *diastolic += 10.0;
        }
        
        // Faster respiration
        for rr in data.respiration_rate.iter_mut() {
            *rr += 4.0;
        }
        
        data
    }
}

/// Combined multi-modal test fixture
#[derive(Debug, Clone)]
pub struct MultiModalTestData {
    pub vision: VisionTestData,
    pub audio: AudioTestData,
    pub text: TextTestData,
    pub physiological: PhysiologicalTestData,
    pub ground_truth_label: bool, // true = deceptive, false = truthful
    pub confidence: f32,
}

impl MultiModalTestData {
    /// Create consistent deceptive sample across all modalities
    pub fn new_deceptive() -> Self {
        Self {
            vision: VisionTestData::new_deceptive(),
            audio: AudioTestData::new_stressed(),
            text: TextTestData::new_deceptive(),
            physiological: PhysiologicalTestData::new_stressed(),
            ground_truth_label: true,
            confidence: 0.85,
        }
    }
    
    /// Create consistent truthful sample across all modalities
    pub fn new_truthful() -> Self {
        Self {
            vision: VisionTestData::new_truthful(),
            audio: AudioTestData::new_relaxed(),
            text: TextTestData::new_truthful(),
            physiological: PhysiologicalTestData::new_baseline(),
            ground_truth_label: false,
            confidence: 0.9,
        }
    }
    
    /// Create mixed signals (challenging case)
    pub fn new_mixed() -> Self {
        Self {
            vision: VisionTestData::new_truthful(),  // Visual calm
            audio: AudioTestData::new_stressed(),    // Audio stress
            text: TextTestData::new_neutral(),       // Neutral text
            physiological: PhysiologicalTestData::new_baseline(), // Normal physiology
            ground_truth_label: false, // Actually truthful despite mixed signals
            confidence: 0.6, // Lower confidence due to conflicting signals
        }
    }
}