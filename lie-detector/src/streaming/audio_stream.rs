//! Streaming audio processing for memory-efficient analysis
//! 
//! This module provides streaming interfaces for processing audio data
//! in chunks, reducing memory usage for long recordings.

use crate::{Result, VeritasError};
use crate::modalities::audio::{AudioFeatures, AudioConfig};
use crate::optimization::{ObjectPool, PooledObject, AudioChunk};
use super::{StreamProcessor, StreamConfig};
use std::sync::Arc;
use parking_lot::Mutex;

/// Streaming audio analyzer
pub struct StreamingAudioAnalyzer {
    config: AudioConfig,
    chunk_pool: Arc<ObjectPool<AudioChunk>>,
    overlap_buffer: Vec<f32>,
    overlap_size: usize,
    sample_rate: u32,
    processed_samples: usize,
    state: AnalyzerState,
}

/// Internal state for the analyzer
struct AnalyzerState {
    /// Running mean for normalization
    running_mean: f64,
    /// Running variance
    running_variance: f64,
    /// Sample count for statistics
    sample_count: usize,
    /// Previous frame for delta features
    previous_features: Option<Vec<f32>>,
    /// Energy threshold for voice activity
    energy_threshold: f32,
}

impl StreamingAudioAnalyzer {
    /// Create a new streaming audio analyzer
    pub fn new(config: AudioConfig, chunk_pool: Arc<ObjectPool<AudioChunk>>) -> Result<Self> {
        let overlap_size = config.chunk_size / 4; // 25% overlap
        
        Ok(Self {
            config: config.clone(),
            chunk_pool,
            overlap_buffer: Vec::with_capacity(overlap_size),
            overlap_size,
            sample_rate: config.sample_rate,
            processed_samples: 0,
            state: AnalyzerState {
                running_mean: 0.0,
                running_variance: 0.0,
                sample_count: 0,
                previous_features: None,
                energy_threshold: 0.01,
            },
        })
    }
    
    /// Process audio samples with minimal allocation
    pub fn process_samples(&mut self, samples: &[f32]) -> Result<Option<AudioFeatures>> {
        // Update running statistics
        self.update_statistics(samples);
        
        // Check if we have enough samples
        let total_samples = self.overlap_buffer.len() + samples.len();
        if total_samples < self.config.chunk_size {
            // Not enough samples yet, buffer them
            self.overlap_buffer.extend_from_slice(samples);
            return Ok(None);
        }
        
        // Get a chunk from the pool
        let mut chunk = self.chunk_pool.get();
        chunk.samples.clear();
        
        // Add overlap from previous chunk
        chunk.samples.extend_from_slice(&self.overlap_buffer);
        
        // Add new samples
        let samples_needed = self.config.chunk_size - self.overlap_buffer.len();
        chunk.samples.extend_from_slice(&samples[..samples_needed]);
        
        // Update overlap buffer for next chunk
        self.overlap_buffer.clear();
        if samples.len() > samples_needed {
            let overlap_start = samples.len().saturating_sub(self.overlap_size);
            self.overlap_buffer.extend_from_slice(&samples[overlap_start..]);
        }
        
        // Process the chunk
        let features = self.analyze_chunk(&chunk)?;
        
        self.processed_samples += chunk.samples.len();
        
        Ok(Some(features))
    }
    
    /// Analyze a single chunk
    fn analyze_chunk(&mut self, chunk: &AudioChunk) -> Result<AudioFeatures> {
        // Apply normalization using running statistics
        let normalized = self.normalize_chunk(&chunk.samples);
        
        // Extract basic features
        let energy = self.calculate_energy(&normalized);
        let zcr = self.calculate_zcr(&normalized);
        let spectral_centroid = self.calculate_spectral_centroid(&normalized)?;
        
        // Voice activity detection
        let is_speech = energy > self.state.energy_threshold && zcr < 0.3;
        
        // Create minimal features structure
        let features = AudioFeatures {
            mfcc: None, // Computed separately if needed
            pitch: None, // Computed separately if needed
            stress: None, // Computed separately if needed
            voice_activity: crate::modalities::audio::VoiceActivity {
                is_speech,
                confidence: if is_speech { 0.8 } else { 0.2 },
                energy_threshold: self.state.energy_threshold,
                zero_crossing_rate: zcr,
                spectral_centroid,
            },
            spectral: crate::modalities::audio::SpectralFeatures {
                centroid: spectral_centroid,
                rolloff: 0.0, // Computed on demand
                flux: 0.0, // Computed on demand
                flatness: 0.0, // Computed on demand
                entropy: 0.0, // Computed on demand
                contrast: vec![], // Computed on demand
            },
            energy: crate::modalities::audio::EnergyFeatures {
                rms: energy,
                peak: self.calculate_peak(&normalized),
                dynamic_range: 0.0, // Computed on demand
            },
            quality: crate::modalities::audio::VoiceQuality::default(),
            timestamp: std::time::Duration::from_secs_f64(
                self.processed_samples as f64 / self.sample_rate as f64
            ),
        };
        
        // Update adaptive threshold
        if !is_speech {
            self.state.energy_threshold = 
                0.9 * self.state.energy_threshold + 0.1 * energy;
        }
        
        Ok(features)
    }
    
    /// Update running statistics for normalization
    fn update_statistics(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.state.sample_count += 1;
            let delta = sample as f64 - self.state.running_mean;
            self.state.running_mean += delta / self.state.sample_count as f64;
            let delta2 = sample as f64 - self.state.running_mean;
            self.state.running_variance += delta * delta2;
        }
    }
    
    /// Normalize chunk using running statistics
    fn normalize_chunk(&self, samples: &[f32]) -> Vec<f32> {
        if self.state.sample_count < 2 {
            return samples.to_vec();
        }
        
        let std_dev = (self.state.running_variance / (self.state.sample_count - 1) as f64).sqrt();
        let mean = self.state.running_mean as f32;
        let std = std_dev as f32;
        
        samples.iter()
            .map(|&s| {
                if std > 0.0 {
                    (s - mean) / std
                } else {
                    s - mean
                }
            })
            .collect()
    }
    
    /// Calculate RMS energy
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }
    
    /// Calculate zero crossing rate
    fn calculate_zcr(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let crossings = samples.windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        
        crossings as f32 / (samples.len() - 1) as f32
    }
    
    /// Calculate spectral centroid (simplified)
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> Result<f32> {
        // Simplified spectral centroid calculation
        // In practice, you'd use FFT
        let weighted_sum: f32 = samples.iter()
            .enumerate()
            .map(|(i, &s)| i as f32 * s.abs())
            .sum();
        
        let magnitude_sum: f32 = samples.iter().map(|s| s.abs()).sum();
        
        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate peak amplitude
    fn calculate_peak(&self, samples: &[f32]) -> f32 {
        samples.iter()
            .map(|s| s.abs())
            .fold(0.0f32, |max, s| max.max(s))
    }
}

impl StreamProcessor for StreamingAudioAnalyzer {
    type Input = AudioChunk;
    type Output = AudioFeatures;
    type Config = AudioConfig;
    
    fn process_chunk(&mut self, chunk: Self::Input) -> Result<Self::Output> {
        self.analyze_chunk(&chunk)
    }
    
    fn min_chunk_size(&self) -> usize {
        self.config.chunk_size
    }
    
    fn supports_parallel(&self) -> bool {
        false // Audio processing needs sequential order
    }
}

/// Audio file streamer for reading audio files in chunks
pub struct AudioFileStreamer {
    reader: Box<dyn AudioReader>,
    chunk_size: usize,
    sample_rate: u32,
    channels: u16,
    chunk_pool: Arc<ObjectPool<AudioChunk>>,
}

/// Trait for audio readers
pub trait AudioReader: Send {
    /// Read next chunk of samples
    fn read_samples(&mut self, buffer: &mut [f32]) -> Result<usize>;
    
    /// Get sample rate
    fn sample_rate(&self) -> u32;
    
    /// Get number of channels
    fn channels(&self) -> u16;
    
    /// Get total number of samples (if known)
    fn total_samples(&self) -> Option<usize>;
}

impl AudioFileStreamer {
    /// Create a new audio file streamer
    pub fn new(
        reader: Box<dyn AudioReader>,
        chunk_size: usize,
        chunk_pool: Arc<ObjectPool<AudioChunk>>,
    ) -> Self {
        let sample_rate = reader.sample_rate();
        let channels = reader.channels();
        
        Self {
            reader,
            chunk_size,
            sample_rate,
            channels,
            chunk_pool,
        }
    }
    
    /// Read the next chunk
    pub fn read_chunk(&mut self) -> Result<Option<PooledObject<AudioChunk>>> {
        let mut chunk = self.chunk_pool.get();
        chunk.samples.clear();
        chunk.samples.resize(self.chunk_size, 0.0);
        chunk.sample_rate = self.sample_rate;
        
        let samples_read = self.reader.read_samples(&mut chunk.samples)?;
        
        if samples_read == 0 {
            return Ok(None);
        }
        
        chunk.samples.truncate(samples_read);
        Ok(Some(chunk))
    }
    
    /// Create an iterator over chunks
    pub fn chunks(self) -> AudioChunkIterator {
        AudioChunkIterator { streamer: self }
    }
}

/// Iterator over audio chunks
pub struct AudioChunkIterator {
    streamer: AudioFileStreamer,
}

impl Iterator for AudioChunkIterator {
    type Item = Result<PooledObject<AudioChunk>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self.streamer.read_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Memory-efficient audio buffer with ring buffer implementation
pub struct RingAudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    size: usize,
}

impl RingAudioBuffer {
    /// Create a new ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
            size: 0,
        }
    }
    
    /// Write samples to the buffer
    pub fn write(&mut self, samples: &[f32]) -> usize {
        let available = self.capacity - self.size;
        let to_write = samples.len().min(available);
        
        for i in 0..to_write {
            self.buffer[self.write_pos] = samples[i];
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
        
        self.size += to_write;
        to_write
    }
    
    /// Read samples from the buffer
    pub fn read(&mut self, output: &mut [f32]) -> usize {
        let to_read = output.len().min(self.size);
        
        for i in 0..to_read {
            output[i] = self.buffer[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
        
        self.size -= to_read;
        to_read
    }
    
    /// Get number of available samples
    pub fn available(&self) -> usize {
        self.size
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingAudioBuffer::new(10);
        
        // Write some samples
        let written = buffer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(written, 5);
        assert_eq!(buffer.available(), 5);
        
        // Read some samples
        let mut output = vec![0.0; 3];
        let read = buffer.read(&mut output);
        assert_eq!(read, 3);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.available(), 2);
        
        // Write more samples (wrapping)
        let written = buffer.write(&[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        assert_eq!(written, 6);
        assert_eq!(buffer.available(), 8);
        
        // Try to write when full
        let written = buffer.write(&[12.0, 13.0, 14.0]);
        assert_eq!(written, 2); // Only 2 slots available
        assert!(buffer.is_full());
    }
}