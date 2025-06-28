/// Mock implementations for testing system components in isolation
/// 
/// This module provides mock implementations of core traits and components
/// to enable unit testing without dependencies on the full system

use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock deception score for testing
#[derive(Debug, Clone, PartialEq)]
pub struct MockDeceptionScore<T: Float> {
    pub probability: T,
    pub confidence: T,
    pub contributing_factors: Vec<(String, T)>,
    pub explanation: String,
}

impl<T: Float> MockDeceptionScore<T> {
    pub fn new(probability: T, confidence: T) -> Self {
        Self {
            probability,
            confidence,
            contributing_factors: Vec::new(),
            explanation: "Mock explanation".to_string(),
        }
    }
    
    pub fn with_factors(mut self, factors: Vec<(String, T)>) -> Self {
        self.contributing_factors = factors;
        self
    }
    
    pub fn with_explanation(mut self, explanation: &str) -> Self {
        self.explanation = explanation.to_string();
        self
    }
}

/// Mock modality analyzer for testing
pub struct MockModalityAnalyzer<T: Float> {
    pub expected_output: MockDeceptionScore<T>,
    pub call_count: Arc<Mutex<usize>>,
    pub last_input: Arc<Mutex<Option<Vec<u8>>>>,
    pub should_error: bool,
}

impl<T: Float> MockModalityAnalyzer<T> {
    pub fn new(expected_output: MockDeceptionScore<T>) -> Self {
        Self {
            expected_output,
            call_count: Arc::new(Mutex::new(0)),
            last_input: Arc::new(Mutex::new(None)),
            should_error: false,
        }
    }
    
    pub fn with_error(mut self) -> Self {
        self.should_error = true;
        self
    }
    
    pub fn analyze(&self, input: &[u8]) -> Result<MockDeceptionScore<T>, String> {
        *self.call_count.lock().unwrap() += 1;
        *self.last_input.lock().unwrap() = Some(input.to_vec());
        
        if self.should_error {
            Err("Mock error".to_string())
        } else {
            Ok(self.expected_output.clone())
        }
    }
    
    pub fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
    
    pub fn get_last_input(&self) -> Option<Vec<u8>> {
        self.last_input.lock().unwrap().clone()
    }
}

/// Mock fusion strategy for testing
pub struct MockFusionStrategy<T: Float> {
    pub expected_decision: MockFusedDecision<T>,
    pub call_count: Arc<Mutex<usize>>,
    pub last_scores: Arc<Mutex<Vec<MockDeceptionScore<T>>>>,
}

#[derive(Debug, Clone)]
pub struct MockFusedDecision<T: Float> {
    pub deception_probability: T,
    pub confidence: T,
    pub modality_contributions: HashMap<String, T>,
    pub explanation: String,
}

impl<T: Float> MockFusionStrategy<T> {
    pub fn new(expected_decision: MockFusedDecision<T>) -> Self {
        Self {
            expected_decision,
            call_count: Arc::new(Mutex::new(0)),
            last_scores: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn fuse(&self, scores: &[MockDeceptionScore<T>]) -> Result<MockFusedDecision<T>, String> {
        *self.call_count.lock().unwrap() += 1;
        *self.last_scores.lock().unwrap() = scores.to_vec();
        
        Ok(self.expected_decision.clone())
    }
    
    pub fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

/// Mock ReAct agent for testing
pub struct MockReactAgent<T: Float> {
    pub reasoning_steps: Vec<MockReasoningStep<T>>,
    pub actions: Vec<MockAction<T>>,
    pub current_step: Arc<Mutex<usize>>,
    pub should_error: bool,
}

#[derive(Debug, Clone)]
pub struct MockReasoningStep<T: Float> {
    pub observation: String,
    pub thought: String,
    pub action: MockAction<T>,
    pub confidence: T,
}

#[derive(Debug, Clone)]
pub struct MockAction<T: Float> {
    pub action_type: String,
    pub parameters: HashMap<String, T>,
    pub expected_outcome: String,
}

impl<T: Float> MockReactAgent<T> {
    pub fn new() -> Self {
        Self {
            reasoning_steps: Vec::new(),
            actions: Vec::new(),
            current_step: Arc::new(Mutex::new(0)),
            should_error: false,
        }
    }
    
    pub fn with_steps(mut self, steps: Vec<MockReasoningStep<T>>) -> Self {
        self.reasoning_steps = steps;
        self
    }
    
    pub fn with_error(mut self) -> Self {
        self.should_error = true;
        self
    }
    
    pub fn reason(&self, observation: &str) -> Result<MockReasoningStep<T>, String> {
        if self.should_error {
            return Err("Mock reasoning error".to_string());
        }
        
        let step_idx = *self.current_step.lock().unwrap();
        if step_idx < self.reasoning_steps.len() {
            *self.current_step.lock().unwrap() += 1;
            Ok(self.reasoning_steps[step_idx].clone())
        } else {
            Err("No more reasoning steps".to_string())
        }
    }
    
    pub fn act(&self, thought: &str) -> Result<MockAction<T>, String> {
        if self.should_error {
            return Err("Mock action error".to_string());
        }
        
        Ok(MockAction {
            action_type: "mock_action".to_string(),
            parameters: HashMap::new(),
            expected_outcome: format!("Response to: {}", thought),
        })
    }
}

/// Mock neural network for testing
pub struct MockNeuralNetwork<T: Float> {
    pub layers: Vec<usize>,
    pub weights: Vec<Vec<T>>,
    pub training_history: Arc<Mutex<Vec<T>>>,
    pub prediction_results: Arc<Mutex<Vec<T>>>,
}

impl<T: Float> MockNeuralNetwork<T> {
    pub fn new(layers: Vec<usize>) -> Self {
        let weights = layers.windows(2)
            .map(|layer_pair| vec![T::from(0.5).unwrap(); layer_pair[0] * layer_pair[1]])
            .collect();
        
        Self {
            layers,
            weights,
            training_history: Arc::new(Mutex::new(Vec::new())),
            prediction_results: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn forward(&self, input: &[T]) -> Result<Vec<T>, String> {
        if input.len() != self.layers[0] {
            return Err(format!("Input size mismatch: expected {}, got {}", self.layers[0], input.len()));
        }
        
        // Simple mock forward pass
        let output = vec![T::from(0.5).unwrap(); self.layers.last().unwrap_or(&1)];
        self.prediction_results.lock().unwrap().extend(output.iter().cloned());
        
        Ok(output)
    }
    
    pub fn train(&self, input: &[T], target: &[T]) -> Result<T, String> {
        let prediction = self.forward(input)?;
        
        // Mock MSE calculation
        let error = target.iter()
            .zip(prediction.iter())
            .map(|(t, p)| (*t - *p) * (*t - *p))
            .fold(T::zero(), |acc, x| acc + x) / T::from(target.len()).unwrap();
        
        self.training_history.lock().unwrap().push(error);
        Ok(error)
    }
    
    pub fn get_training_history(&self) -> Vec<T> {
        self.training_history.lock().unwrap().clone()
    }
}

/// Mock streaming pipeline for testing
pub struct MockStreamingPipeline<T: Float> {
    pub processed_frames: Arc<Mutex<usize>>,
    pub processing_times: Arc<Mutex<Vec<std::time::Duration>>>,
    pub should_error: bool,
    pub latency_ms: u64,
}

impl<T: Float> MockStreamingPipeline<T> {
    pub fn new() -> Self {
        Self {
            processed_frames: Arc::new(Mutex::new(0)),
            processing_times: Arc::new(Mutex::new(Vec::new())),
            should_error: false,
            latency_ms: 10,
        }
    }
    
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }
    
    pub fn with_error(mut self) -> Self {
        self.should_error = true;
        self
    }
    
    pub async fn process_frame(&self, _frame: &[u8]) -> Result<MockDeceptionScore<T>, String> {
        if self.should_error {
            return Err("Mock processing error".to_string());
        }
        
        let start = std::time::Instant::now();
        
        // Simulate processing time
        tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        
        let duration = start.elapsed();
        self.processing_times.lock().unwrap().push(duration);
        *self.processed_frames.lock().unwrap() += 1;
        
        Ok(MockDeceptionScore::new(
            T::from(0.6).unwrap(),
            T::from(0.8).unwrap(),
        ))
    }
    
    pub fn get_processed_count(&self) -> usize {
        *self.processed_frames.lock().unwrap()
    }
    
    pub fn get_average_processing_time(&self) -> Option<std::time::Duration> {
        let times = self.processing_times.lock().unwrap();
        if times.is_empty() {
            None
        } else {
            let total: std::time::Duration = times.iter().sum();
            Some(total / times.len() as u32)
        }
    }
}

/// Mock data loader for testing
pub struct MockDataLoader<T: Float> {
    pub data: Vec<(Vec<T>, T)>, // (features, label)
    pub batch_size: usize,
    pub current_batch: Arc<Mutex<usize>>,
}

impl<T: Float> MockDataLoader<T> {
    pub fn new(data: Vec<(Vec<T>, T)>, batch_size: usize) -> Self {
        Self {
            data,
            batch_size,
            current_batch: Arc::new(Mutex::new(0)),
        }
    }
    
    pub fn next_batch(&self) -> Option<Vec<(Vec<T>, T)>> {
        let mut current = self.current_batch.lock().unwrap();
        let start_idx = *current * self.batch_size;
        
        if start_idx >= self.data.len() {
            return None;
        }
        
        let end_idx = std::cmp::min(start_idx + self.batch_size, self.data.len());
        let batch = self.data[start_idx..end_idx].to_vec();
        
        *current += 1;
        Some(batch)
    }
    
    pub fn reset(&self) {
        *self.current_batch.lock().unwrap() = 0;
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Mock memory manager for testing
pub struct MockMemoryManager<T: Float> {
    pub allocations: Arc<Mutex<HashMap<String, Vec<T>>>>,
    pub peak_usage: Arc<Mutex<usize>>,
    pub current_usage: Arc<Mutex<usize>>,
}

impl<T: Float> MockMemoryManager<T> {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            peak_usage: Arc::new(Mutex::new(0)),
            current_usage: Arc::new(Mutex::new(0)),
        }
    }
    
    pub fn allocate(&self, name: &str, size: usize) -> Result<Vec<T>, String> {
        let buffer = vec![T::zero(); size];
        
        {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.insert(name.to_string(), buffer.clone());
            
            let mut current = self.current_usage.lock().unwrap();
            *current += size;
            
            let mut peak = self.peak_usage.lock().unwrap();
            if *current > *peak {
                *peak = *current;
            }
        }
        
        Ok(buffer)
    }
    
    pub fn deallocate(&self, name: &str) -> Result<(), String> {
        let mut allocations = self.allocations.lock().unwrap();
        if let Some(buffer) = allocations.remove(name) {
            let mut current = self.current_usage.lock().unwrap();
            *current = current.saturating_sub(buffer.len());
            Ok(())
        } else {
            Err(format!("Buffer '{}' not found", name))
        }
    }
    
    pub fn get_current_usage(&self) -> usize {
        *self.current_usage.lock().unwrap()
    }
    
    pub fn get_peak_usage(&self) -> usize {
        *self.peak_usage.lock().unwrap()
    }
}

/// Helper macros for creating mock objects
#[macro_export]
macro_rules! mock_score {
    ($prob:expr, $conf:expr) => {
        MockDeceptionScore::new($prob, $conf)
    };
    ($prob:expr, $conf:expr, $explanation:expr) => {
        MockDeceptionScore::new($prob, $conf).with_explanation($explanation)
    };
}

#[macro_export]
macro_rules! mock_analyzer {
    ($score:expr) => {
        MockModalityAnalyzer::new($score)
    };
    ($score:expr, error) => {
        MockModalityAnalyzer::new($score).with_error()
    };
}

#[macro_export]
macro_rules! mock_network {
    ($($layer:expr),+) => {
        MockNeuralNetwork::new(vec![$($layer),+])
    };
}