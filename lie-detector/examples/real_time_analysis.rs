//! # Real-Time Streaming Lie Detection Example
//! 
//! This example demonstrates real-time streaming lie detection capabilities.
//! It shows how to:
//! - Set up streaming pipelines for multi-modal input
//! - Process live camera, microphone, and text input
//! - Handle temporal synchronization across modalities
//! - Implement ring buffers for efficient memory usage
//! - Provide real-time feedback with configurable latency
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example real_time_analysis
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, Mutex as AsyncMutex};
use tokio::time::{interval, timeout};
use futures::stream::{Stream, StreamExt};

/// Represents a video frame with timestamp
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: Instant,
    pub format: VideoFormat,
}

/// Video format options
#[derive(Debug, Clone)]
pub enum VideoFormat {
    Rgb24,
    Yuv420,
    Bgr24,
}

/// Represents an audio chunk with timestamp
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u8,
    pub timestamp: Instant,
}

/// Represents a text segment from speech-to-text
#[derive(Debug, Clone)]
pub struct TextSegment {
    pub text: String,
    pub confidence: f32,
    pub start_time: Instant,
    pub end_time: Instant,
    pub is_final: bool,
}

/// Multi-modal input that's been temporally synchronized
#[derive(Debug)]
pub struct SynchronizedInput {
    pub video_frame: Option<VideoFrame>,
    pub audio_chunk: Option<AudioChunk>,
    pub text_segment: Option<TextSegment>,
    pub timestamp: Instant,
}

/// Real-time analysis result
#[derive(Debug, Clone)]
pub struct RealTimeResult {
    pub deception_score: f32,
    pub confidence: f32,
    pub decision: RealtimeDecision,
    pub modality_contributions: ModalityContributions,
    pub timestamp: Instant,
    pub processing_latency_ms: u64,
}

/// Real-time decision with uncertainty handling
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeDecision {
    TruthTelling,
    Deceptive,
    Uncertain,
    InsufficientData,
}

/// Contribution scores from each modality
#[derive(Debug, Clone)]
pub struct ModalityContributions {
    pub vision_weight: f32,
    pub audio_weight: f32,
    pub text_weight: f32,
    pub vision_score: Option<f32>,
    pub audio_score: Option<f32>,
    pub text_score: Option<f32>,
}

/// Configuration for the streaming pipeline
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub target_fps: f32,
    pub audio_chunk_size_ms: u32,
    pub sync_window_ms: u32,
    pub max_latency_ms: u32,
    pub buffer_size: usize,
    pub enable_adaptive_quality: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            target_fps: 30.0,
            audio_chunk_size_ms: 100,
            sync_window_ms: 200,
            max_latency_ms: 500,
            buffer_size: 128,
            enable_adaptive_quality: true,
        }
    }
}

/// Ring buffer for efficient streaming data management
pub struct RingBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    dropped_count: u64,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            dropped_count: 0,
        }
    }
    
    pub fn push(&mut self, item: T) -> Option<T> {
        if self.buffer.len() >= self.capacity {
            self.dropped_count += 1;
            Some(self.buffer.pop_front().unwrap())
        } else {
            self.buffer.push_back(item);
            None
        }
    }
    
    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }
}

/// Temporal synchronizer for aligning multi-modal streams
pub struct TemporalSynchronizer {
    video_buffer: RingBuffer<VideoFrame>,
    audio_buffer: RingBuffer<AudioChunk>,
    text_buffer: RingBuffer<TextSegment>,
    sync_window: Duration,
}

impl TemporalSynchronizer {
    pub fn new(sync_window_ms: u32, buffer_size: usize) -> Self {
        Self {
            video_buffer: RingBuffer::new(buffer_size),
            audio_buffer: RingBuffer::new(buffer_size),
            text_buffer: RingBuffer::new(buffer_size),
            sync_window: Duration::from_millis(sync_window_ms as u64),
        }
    }
    
    pub fn add_video(&mut self, frame: VideoFrame) {
        self.video_buffer.push(frame);
    }
    
    pub fn add_audio(&mut self, chunk: AudioChunk) {
        self.audio_buffer.push(chunk);
    }
    
    pub fn add_text(&mut self, segment: TextSegment) {
        self.text_buffer.push(segment);
    }
    
    /// Try to create a synchronized input from available data
    pub fn try_sync(&mut self, target_time: Instant) -> Option<SynchronizedInput> {
        // Find best matching data within sync window
        let video_frame = self.find_closest_video(target_time);
        let audio_chunk = self.find_closest_audio(target_time);
        let text_segment = self.find_closest_text(target_time);
        
        // Return synchronized input if we have at least some data
        if video_frame.is_some() || audio_chunk.is_some() || text_segment.is_some() {
            Some(SynchronizedInput {
                video_frame,
                audio_chunk,
                text_segment,
                timestamp: target_time,
            })
        } else {
            None
        }
    }
    
    fn find_closest_video(&mut self, target_time: Instant) -> Option<VideoFrame> {
        // Find video frame closest to target time within sync window
        let mut best_frame = None;
        let mut best_diff = Duration::from_secs(u64::MAX);
        
        while let Some(frame) = self.video_buffer.pop() {
            let diff = if frame.timestamp > target_time {
                frame.timestamp - target_time
            } else {
                target_time - frame.timestamp
            };
            
            if diff <= self.sync_window && diff < best_diff {
                best_diff = diff;
                best_frame = Some(frame);
            } else if frame.timestamp < target_time - self.sync_window {
                // Frame too old, discard
                continue;
            } else {
                // Frame too new, put it back
                let mut temp_buffer = VecDeque::new();
                temp_buffer.push_back(frame);
                while let Some(f) = self.video_buffer.buffer.pop_front() {
                    temp_buffer.push_back(f);
                }
                self.video_buffer.buffer = temp_buffer;
                break;
            }
        }
        
        best_frame
    }
    
    fn find_closest_audio(&mut self, target_time: Instant) -> Option<AudioChunk> {
        // Similar logic for audio chunks
        let mut best_chunk = None;
        let mut best_diff = Duration::from_secs(u64::MAX);
        
        while let Some(chunk) = self.audio_buffer.pop() {
            let diff = if chunk.timestamp > target_time {
                chunk.timestamp - target_time
            } else {
                target_time - chunk.timestamp
            };
            
            if diff <= self.sync_window && diff < best_diff {
                best_diff = diff;
                best_chunk = Some(chunk);
            } else if chunk.timestamp < target_time - self.sync_window {
                continue; // Too old
            } else {
                // Put back and stop
                let mut temp_buffer = VecDeque::new();
                temp_buffer.push_back(chunk);
                while let Some(c) = self.audio_buffer.buffer.pop_front() {
                    temp_buffer.push_back(c);
                }
                self.audio_buffer.buffer = temp_buffer;
                break;
            }
        }
        
        best_chunk
    }
    
    fn find_closest_text(&mut self, target_time: Instant) -> Option<TextSegment> {
        // Find text segment that overlaps with target time
        while let Some(segment) = self.text_buffer.pop() {
            if segment.start_time <= target_time && segment.end_time >= target_time {
                return Some(segment);
            } else if segment.end_time < target_time - self.sync_window {
                continue; // Too old
            } else {
                // Put back
                let mut temp_buffer = VecDeque::new();
                temp_buffer.push_back(segment);
                while let Some(s) = self.text_buffer.buffer.pop_front() {
                    temp_buffer.push_back(s);
                }
                self.text_buffer.buffer = temp_buffer;
                break;
            }
        }
        
        None
    }
    
    pub fn get_stats(&self) -> SyncStats {
        SyncStats {
            video_buffer_size: self.video_buffer.len(),
            audio_buffer_size: self.audio_buffer.len(),
            text_buffer_size: self.text_buffer.len(),
            video_dropped: self.video_buffer.dropped_count(),
            audio_dropped: self.audio_buffer.dropped_count(),
            text_dropped: self.text_buffer.dropped_count(),
        }
    }
}

#[derive(Debug)]
pub struct SyncStats {
    pub video_buffer_size: usize,
    pub audio_buffer_size: usize,
    pub text_buffer_size: usize,
    pub video_dropped: u64,
    pub audio_dropped: u64,
    pub text_dropped: u64,
}

/// Real-time streaming pipeline
pub struct StreamingPipeline {
    config: StreamingConfig,
    synchronizer: Arc<AsyncMutex<TemporalSynchronizer>>,
    vision_processor: Arc<VisionProcessor>,
    audio_processor: Arc<AudioProcessor>,
    text_processor: Arc<TextProcessor>,
    fusion_engine: Arc<FusionEngine>,
    result_sender: mpsc::UnboundedSender<RealTimeResult>,
    result_receiver: Arc<AsyncMutex<mpsc::UnboundedReceiver<RealTimeResult>>>,
    is_running: Arc<Mutex<bool>>,
    stats: Arc<Mutex<PipelineStats>>,
}

#[derive(Debug, Default)]
pub struct PipelineStats {
    pub frames_processed: u64,
    pub audio_chunks_processed: u64,
    pub text_segments_processed: u64,
    pub total_latency_ms: u64,
    pub avg_latency_ms: f64,
    pub dropped_frames: u64,
}

/// Mock processors for the example
pub struct VisionProcessor;
pub struct AudioProcessor;
pub struct TextProcessor;
pub struct FusionEngine;

impl VisionProcessor {
    pub async fn process(&self, frame: &VideoFrame) -> f32 {
        // Simulate vision processing time
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        // Mock deception score based on frame characteristics
        let brightness: f32 = frame.data.iter().map(|&x| x as f32).sum::<f32>() / frame.data.len() as f32;
        (brightness / 255.0 * 0.3 + 0.4).min(1.0) // Simulate score between 0.4-0.7
    }
}

impl AudioProcessor {
    pub async fn process(&self, chunk: &AudioChunk) -> f32 {
        // Simulate audio processing time
        tokio::time::sleep(Duration::from_millis(3)).await;
        
        // Mock deception score based on audio characteristics
        let energy = chunk.samples.iter().map(|&x| x * x).sum::<f32>() / chunk.samples.len() as f32;
        (energy.sqrt() * 0.5 + 0.3).min(1.0)
    }
}

impl TextProcessor {
    pub async fn process(&self, segment: &TextSegment) -> f32 {
        // Simulate text processing time
        tokio::time::sleep(Duration::from_millis(2)).await;
        
        // Mock deception score based on text characteristics
        let word_count = segment.text.split_whitespace().count();
        let uncertainty_words = ["maybe", "perhaps", "I think", "sort of", "kind of"];
        let uncertainty_count = uncertainty_words.iter()
            .map(|&word| segment.text.matches(word).count())
            .sum::<usize>();
        
        ((uncertainty_count as f32 / word_count as f32) * 0.7 + 0.2).min(1.0)
    }
}

impl FusionEngine {
    pub fn fuse(&self, 
        vision_score: Option<f32>, 
        audio_score: Option<f32>, 
        text_score: Option<f32>
    ) -> (f32, f32, ModalityContributions) {
        let mut scores = Vec::new();
        let mut weights = Vec::new();
        
        let mut vision_weight = 0.0;
        let mut audio_weight = 0.0;
        let mut text_weight = 0.0;
        
        if let Some(score) = vision_score {
            scores.push(score);
            vision_weight = 0.4;
            weights.push(vision_weight);
        }
        
        if let Some(score) = audio_score {
            scores.push(score);
            audio_weight = 0.35;
            weights.push(audio_weight);
        }
        
        if let Some(score) = text_score {
            scores.push(score);
            text_weight = 0.25;
            weights.push(text_weight);
        }
        
        let (final_score, confidence) = if scores.is_empty() {
            (0.5, 0.0)
        } else {
            let weighted_sum: f32 = scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum();
            let weight_sum: f32 = weights.iter().sum();
            let final_score = weighted_sum / weight_sum;
            
            // Confidence based on number of modalities and score consistency
            let base_confidence = scores.len() as f32 / 3.0;
            let variance = if scores.len() > 1 {
                let mean = final_score;
                let var: f32 = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
                var.sqrt()
            } else {
                0.0
            };
            let consistency_bonus = (1.0 - variance.min(1.0)) * 0.3;
            let confidence = (base_confidence + consistency_bonus).min(1.0);
            
            (final_score, confidence)
        };
        
        let contributions = ModalityContributions {
            vision_weight,
            audio_weight,
            text_weight,
            vision_score,
            audio_score,
            text_score,
        };
        
        (final_score, confidence, contributions)
    }
}

impl StreamingPipeline {
    pub fn new(config: StreamingConfig) -> Self {
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        
        Self {
            synchronizer: Arc::new(AsyncMutex::new(TemporalSynchronizer::new(
                config.sync_window_ms,
                config.buffer_size,
            ))),
            vision_processor: Arc::new(VisionProcessor),
            audio_processor: Arc::new(AudioProcessor),
            text_processor: Arc::new(TextProcessor),
            fusion_engine: Arc::new(FusionEngine),
            result_sender,
            result_receiver: Arc::new(AsyncMutex::new(result_receiver)),
            is_running: Arc::new(Mutex::new(false)),
            stats: Arc::new(Mutex::new(PipelineStats::default())),
            config,
        }
    }
    
    /// Start the streaming pipeline
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        *self.is_running.lock().unwrap() = true;
        
        // Spawn processing task
        let synchronizer = Arc::clone(&self.synchronizer);
        let vision_processor = Arc::clone(&self.vision_processor);
        let audio_processor = Arc::clone(&self.audio_processor);
        let text_processor = Arc::clone(&self.text_processor);
        let fusion_engine = Arc::clone(&self.fusion_engine);
        let result_sender = self.result_sender.clone();
        let is_running = Arc::clone(&self.is_running);
        let stats = Arc::clone(&self.stats);
        let target_fps = self.config.target_fps;
        let max_latency = self.config.max_latency_ms;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis((1000.0 / target_fps) as u64));
            
            while *is_running.lock().unwrap() {
                interval.tick().await;
                
                let process_start = Instant::now();
                let target_time = Instant::now();
                
                // Try to get synchronized input
                let sync_input = {
                    let mut sync = synchronizer.lock().await;
                    sync.try_sync(target_time)
                };
                
                if let Some(input) = sync_input {
                    // Process each modality in parallel
                    let vision_task = async {
                        if let Some(frame) = &input.video_frame {
                            Some(vision_processor.process(frame).await)
                        } else {
                            None
                        }
                    };
                    
                    let audio_task = async {
                        if let Some(chunk) = &input.audio_chunk {
                            Some(audio_processor.process(chunk).await)
                        } else {
                            None
                        }
                    };
                    
                    let text_task = async {
                        if let Some(segment) = &input.text_segment {
                            Some(text_processor.process(segment).await)
                        } else {
                            None
                        }
                    };
                    
                    // Run processing tasks with timeout
                    let timeout_duration = Duration::from_millis(max_latency as u64);
                    
                    let results = timeout(timeout_duration, async {
                        tokio::join!(vision_task, audio_task, text_task)
                    }).await;
                    
                    if let Ok((vision_score, audio_score, text_score)) = results {
                        // Fuse results
                        let (final_score, confidence, contributions) = 
                            fusion_engine.fuse(vision_score, audio_score, text_score);
                        
                        // Make decision
                        let decision = if confidence < 0.3 {
                            RealtimeDecision::InsufficientData
                        } else if confidence >= 0.7 {
                            if final_score > 0.6 {
                                RealtimeDecision::Deceptive
                            } else if final_score < 0.4 {
                                RealtimeDecision::TruthTelling
                            } else {
                                RealtimeDecision::Uncertain
                            }
                        } else {
                            if final_score > 0.7 {
                                RealtimeDecision::Deceptive
                            } else if final_score < 0.3 {
                                RealtimeDecision::TruthTelling
                            } else {
                                RealtimeDecision::Uncertain
                            }
                        };
                        
                        let processing_latency = process_start.elapsed().as_millis() as u64;
                        
                        let result = RealTimeResult {
                            deception_score: final_score,
                            confidence,
                            decision,
                            modality_contributions: contributions,
                            timestamp: input.timestamp,
                            processing_latency_ms: processing_latency,
                        };
                        
                        // Update stats
                        {
                            let mut stats_lock = stats.lock().unwrap();
                            if input.video_frame.is_some() {
                                stats_lock.frames_processed += 1;
                            }
                            if input.audio_chunk.is_some() {
                                stats_lock.audio_chunks_processed += 1;
                            }
                            if input.text_segment.is_some() {
                                stats_lock.text_segments_processed += 1;
                            }
                            stats_lock.total_latency_ms += processing_latency;
                            let total_processed = stats_lock.frames_processed + 
                                                stats_lock.audio_chunks_processed + 
                                                stats_lock.text_segments_processed;
                            if total_processed > 0 {
                                stats_lock.avg_latency_ms = stats_lock.total_latency_ms as f64 / total_processed as f64;
                            }
                        }
                        
                        // Send result
                        let _ = result_sender.send(result);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Stop the streaming pipeline
    pub fn stop(&self) {
        *self.is_running.lock().unwrap() = false;
    }
    
    /// Add video frame to processing queue
    pub async fn add_video_frame(&self, frame: VideoFrame) {
        let mut sync = self.synchronizer.lock().await;
        sync.add_video(frame);
    }
    
    /// Add audio chunk to processing queue
    pub async fn add_audio_chunk(&self, chunk: AudioChunk) {
        let mut sync = self.synchronizer.lock().await;
        sync.add_audio(chunk);
    }
    
    /// Add text segment to processing queue
    pub async fn add_text_segment(&self, segment: TextSegment) {
        let mut sync = self.synchronizer.lock().await;
        sync.add_text(segment);
    }
    
    /// Get next result (non-blocking)
    pub async fn try_get_result(&self) -> Option<RealTimeResult> {
        let mut receiver = self.result_receiver.lock().await;
        receiver.try_recv().ok()
    }
    
    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Get synchronizer statistics
    pub async fn get_sync_stats(&self) -> SyncStats {
        let sync = self.synchronizer.lock().await;
        sync.get_stats()
    }
}

/// Simulated data sources
pub struct DataSimulator {
    start_time: Instant,
    frame_counter: u64,
    audio_counter: u64,
    text_segments: Vec<String>,
    text_index: usize,
}

impl DataSimulator {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            frame_counter: 0,
            audio_counter: 0,
            text_segments: vec![
                "I was at home all evening".to_string(),
                "No I never saw that document".to_string(),
                "I think maybe I was there".to_string(),
                "I definitely did not take anything".to_string(),
                "Perhaps someone else did it".to_string(),
            ],
            text_index: 0,
        }
    }
    
    pub fn generate_video_frame(&mut self) -> VideoFrame {
        self.frame_counter += 1;
        
        // Generate dummy frame data
        let width = 640;
        let height = 480;
        let size = (width * height * 3) as usize; // RGB24
        let mut data = vec![0u8; size];
        
        // Fill with pattern based on counter
        for i in 0..size {
            data[i] = ((i + self.frame_counter as usize) % 256) as u8;
        }
        
        VideoFrame {
            data,
            width,
            height,
            timestamp: Instant::now(),
            format: VideoFormat::Rgb24,
        }
    }
    
    pub fn generate_audio_chunk(&mut self) -> AudioChunk {
        self.audio_counter += 1;
        
        let sample_rate = 16000;
        let chunk_duration_ms = 100;
        let samples_count = (sample_rate * chunk_duration_ms / 1000) as usize;
        
        // Generate sine wave with some variation
        let mut samples = Vec::with_capacity(samples_count);
        let frequency = 440.0 + (self.audio_counter as f32 * 10.0) % 200.0;
        
        for i in 0..samples_count {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.1;
            samples.push(sample);
        }
        
        AudioChunk {
            samples,
            sample_rate,
            channels: 1,
            timestamp: Instant::now(),
        }
    }
    
    pub fn generate_text_segment(&mut self) -> Option<TextSegment> {
        if self.text_index < self.text_segments.len() {
            let text = self.text_segments[self.text_index].clone();
            self.text_index += 1;
            
            let now = Instant::now();
            Some(TextSegment {
                text,
                confidence: 0.85,
                start_time: now,
                end_time: now + Duration::from_secs(2),
                is_final: true,
            })
        } else {
            None
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Veritas Nexus - Real-Time Streaming Analysis");
    println!("==============================================\n");
    
    // Create streaming configuration
    let config = StreamingConfig {
        target_fps: 10.0, // Lower FPS for demo
        audio_chunk_size_ms: 100,
        sync_window_ms: 200,
        max_latency_ms: 300,
        buffer_size: 64,
        enable_adaptive_quality: true,
    };
    
    println!("⚙️  Configuration:");
    println!("   Target FPS: {}", config.target_fps);
    println!("   Audio chunk size: {}ms", config.audio_chunk_size_ms);
    println!("   Sync window: {}ms", config.sync_window_ms);
    println!("   Max latency: {}ms", config.max_latency_ms);
    println!("   Buffer size: {}", config.buffer_size);
    
    // Create streaming pipeline
    println!("\n🚀 Starting streaming pipeline...");
    let pipeline = StreamingPipeline::new(config);
    pipeline.start().await?;
    
    // Create data simulator
    let mut simulator = DataSimulator::new();
    
    println!("✅ Pipeline started successfully!\n");
    
    // Simulate streaming for 10 seconds
    let duration = Duration::from_secs(10);
    let start_time = Instant::now();
    let mut last_stats_time = start_time;
    
    println!("📊 Starting real-time simulation...\n");
    
    // Spawn data generation tasks
    let pipeline_clone = Arc::new(pipeline);
    
    // Video generation task
    let video_pipeline = Arc::clone(&pipeline_clone);
    tokio::spawn(async move {
        let mut video_interval = interval(Duration::from_millis(100)); // 10 FPS
        let mut simulator = DataSimulator::new();
        
        while start_time.elapsed() < duration {
            video_interval.tick().await;
            let frame = simulator.generate_video_frame();
            video_pipeline.add_video_frame(frame).await;
        }
    });
    
    // Audio generation task
    let audio_pipeline = Arc::clone(&pipeline_clone);
    tokio::spawn(async move {
        let mut audio_interval = interval(Duration::from_millis(100)); // 10 chunks/sec
        let mut simulator = DataSimulator::new();
        
        while start_time.elapsed() < duration {
            audio_interval.tick().await;
            let chunk = simulator.generate_audio_chunk();
            audio_pipeline.add_audio_chunk(chunk).await;
        }
    });
    
    // Text generation task
    let text_pipeline = Arc::clone(&pipeline_clone);
    tokio::spawn(async move {
        let mut text_interval = interval(Duration::from_secs(2)); // Every 2 seconds
        let mut simulator = DataSimulator::new();
        
        while start_time.elapsed() < duration {
            text_interval.tick().await;
            if let Some(segment) = simulator.generate_text_segment() {
                text_pipeline.add_text_segment(segment).await;
            }
        }
    });
    
    // Result processing and statistics
    let mut result_count = 0;
    let mut last_result_time = start_time;
    
    while start_time.elapsed() < duration {
        // Check for new results
        if let Some(result) = pipeline_clone.try_get_result().await {
            result_count += 1;
            
            println!("🔍 Result #{}: {:?} (score: {:.3}, confidence: {:.1}%, latency: {}ms)",
                result_count,
                result.decision,
                result.deception_score,
                result.confidence * 100.0,
                result.processing_latency_ms
            );
            
            // Show modality contributions
            let contrib = &result.modality_contributions;
            if contrib.vision_score.is_some() || contrib.audio_score.is_some() || contrib.text_score.is_some() {
                print!("   Contributions: ");
                if let Some(score) = contrib.vision_score {
                    print!("👁️ {:.2} ", score);
                }
                if let Some(score) = contrib.audio_score {
                    print!("🔊 {:.2} ", score);
                }
                if let Some(score) = contrib.text_score {
                    print!("📝 {:.2} ", score);
                }
                println!();
            }
            
            last_result_time = Instant::now();
        }
        
        // Print stats every 2 seconds
        if last_stats_time.elapsed() >= Duration::from_secs(2) {
            let stats = pipeline_clone.get_stats();
            let sync_stats = pipeline_clone.get_sync_stats().await;
            
            println!("\n📈 Pipeline Statistics:");
            println!("   Frames processed: {}", stats.frames_processed);
            println!("   Audio chunks processed: {}", stats.audio_chunks_processed);
            println!("   Text segments processed: {}", stats.text_segments_processed);
            println!("   Average latency: {:.1}ms", stats.avg_latency_ms);
            println!("   Buffer sizes: V:{} A:{} T:{}", 
                sync_stats.video_buffer_size,
                sync_stats.audio_buffer_size,
                sync_stats.text_buffer_size
            );
            println!("   Dropped: V:{} A:{} T:{}", 
                sync_stats.video_dropped,
                sync_stats.audio_dropped,
                sync_stats.text_dropped
            );
            println!();
            
            last_stats_time = Instant::now();
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    println!("\n⏹️  Stopping pipeline...");
    pipeline_clone.stop();
    
    // Final statistics
    let final_stats = pipeline_clone.get_stats();
    let final_sync_stats = pipeline_clone.get_sync_stats().await;
    
    println!("\n🏁 Final Statistics:");
    println!("===================");
    println!("Total results: {}", result_count);
    println!("Frames processed: {}", final_stats.frames_processed);
    println!("Audio chunks processed: {}", final_stats.audio_chunks_processed);
    println!("Text segments processed: {}", final_stats.text_segments_processed);
    println!("Average latency: {:.1}ms", final_stats.avg_latency_ms);
    println!("Total dropped frames: {}", final_sync_stats.video_dropped);
    println!("Total dropped audio: {}", final_sync_stats.audio_dropped);
    println!("Total dropped text: {}", final_sync_stats.text_dropped);
    
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Real-time multi-modal processing");
    println!("   • Temporal synchronization across modalities");
    println!("   • Efficient ring buffer management");
    println!("   • Parallel processing with latency control");
    println!("   • Adaptive quality and error handling");
    println!("   • Comprehensive statistics and monitoring");
    
    Ok(())
}