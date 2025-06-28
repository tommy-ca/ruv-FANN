//! Chunk-based processing utilities for streaming data
//! 
//! This module provides utilities for processing data in chunks
//! with minimal memory allocation and efficient batching.

use crate::{Result, VeritasError};
use std::sync::Arc;
use parking_lot::Mutex;
use crossbeam_channel::{bounded, Receiver, Sender};

/// Adaptive chunk processor that adjusts chunk size based on performance
pub struct AdaptiveChunkProcessor<T> {
    min_chunk_size: usize,
    max_chunk_size: usize,
    current_chunk_size: usize,
    performance_history: Vec<PerformanceMetric>,
    adaptation_enabled: bool,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
struct PerformanceMetric {
    chunk_size: usize,
    processing_time_ms: f64,
    throughput_mbps: f64,
    memory_used_mb: f64,
}

impl<T> AdaptiveChunkProcessor<T> {
    /// Create a new adaptive chunk processor
    pub fn new(min_chunk_size: usize, max_chunk_size: usize) -> Self {
        Self {
            min_chunk_size,
            max_chunk_size,
            current_chunk_size: min_chunk_size,
            performance_history: Vec::with_capacity(100),
            adaptation_enabled: true,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Process data with adaptive chunking
    pub fn process<F>(
        &mut self,
        data: &[T],
        mut processor: F,
    ) -> Result<Vec<ProcessingResult>>
    where
        F: FnMut(&[T]) -> Result<ProcessingResult>,
        T: Clone,
    {
        let mut results = Vec::new();
        let mut offset = 0;
        
        while offset < data.len() {
            let chunk_size = self.current_chunk_size.min(data.len() - offset);
            let chunk_end = offset + chunk_size;
            
            let start_time = std::time::Instant::now();
            let start_memory = get_current_memory_usage();
            
            let result = processor(&data[offset..chunk_end])?;
            
            let elapsed = start_time.elapsed();
            let memory_delta = get_current_memory_usage() - start_memory;
            
            // Record performance metrics
            if self.adaptation_enabled {
                self.record_metric(
                    chunk_size,
                    elapsed.as_secs_f64() * 1000.0,
                    calculate_throughput(chunk_size, elapsed),
                    memory_delta as f64 / (1024.0 * 1024.0),
                );
                
                self.adapt_chunk_size();
            }
            
            results.push(result);
            offset = chunk_end;
        }
        
        Ok(results)
    }
    
    /// Record performance metric
    fn record_metric(
        &mut self,
        chunk_size: usize,
        processing_time_ms: f64,
        throughput_mbps: f64,
        memory_used_mb: f64,
    ) {
        let metric = PerformanceMetric {
            chunk_size,
            processing_time_ms,
            throughput_mbps,
            memory_used_mb,
        };
        
        self.performance_history.push(metric);
        
        // Keep only recent history
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }
    
    /// Adapt chunk size based on performance history
    fn adapt_chunk_size(&mut self) {
        if self.performance_history.len() < 10 {
            return; // Not enough data
        }
        
        // Calculate average throughput for different chunk sizes
        let mut size_performance: std::collections::HashMap<usize, (f64, usize)> = 
            std::collections::HashMap::new();
        
        for metric in &self.performance_history {
            let entry = size_performance.entry(metric.chunk_size).or_insert((0.0, 0));
            entry.0 += metric.throughput_mbps;
            entry.1 += 1;
        }
        
        // Find best performing chunk size
        let best_size = size_performance.iter()
            .map(|(size, (total_throughput, count))| {
                (*size, total_throughput / *count as f64)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(size, _)| size)
            .unwrap_or(self.current_chunk_size);
        
        // Gradually adjust toward best size
        if best_size > self.current_chunk_size {
            self.current_chunk_size = (self.current_chunk_size * 5 / 4)
                .min(best_size)
                .min(self.max_chunk_size);
        } else if best_size < self.current_chunk_size {
            self.current_chunk_size = (self.current_chunk_size * 4 / 5)
                .max(best_size)
                .max(self.min_chunk_size);
        }
    }
    
    /// Get current chunk size
    pub fn current_chunk_size(&self) -> usize {
        self.current_chunk_size
    }
    
    /// Enable or disable adaptation
    pub fn set_adaptation_enabled(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
    }
}

/// Processing result placeholder
#[derive(Debug)]
pub struct ProcessingResult {
    pub items_processed: usize,
    pub features_extracted: usize,
    pub processing_time_ms: f64,
}

/// Parallel chunk processor using work stealing
pub struct ParallelChunkProcessor<T: Send + Sync> {
    num_workers: usize,
    chunk_queue: Arc<Mutex<Vec<Vec<T>>>>,
    result_sender: Sender<ProcessingResult>,
    result_receiver: Receiver<ProcessingResult>,
}

impl<T: Send + Sync + 'static> ParallelChunkProcessor<T> {
    /// Create a new parallel chunk processor
    pub fn new(num_workers: usize) -> Self {
        let (result_sender, result_receiver) = bounded(num_workers * 2);
        
        Self {
            num_workers,
            chunk_queue: Arc::new(Mutex::new(Vec::new())),
            result_sender,
            result_receiver,
        }
    }
    
    /// Process chunks in parallel
    pub fn process_parallel<F>(
        &mut self,
        chunks: Vec<Vec<T>>,
        processor: F,
    ) -> Result<Vec<ProcessingResult>>
    where
        F: Fn(&[T]) -> Result<ProcessingResult> + Send + Sync + 'static + Clone,
    {
        // Fill work queue
        {
            let mut queue = self.chunk_queue.lock();
            *queue = chunks;
        }
        
        // Spawn workers
        let mut handles = Vec::new();
        for _ in 0..self.num_workers {
            let queue = self.chunk_queue.clone();
            let sender = self.result_sender.clone();
            let proc = processor.clone();
            
            let handle = std::thread::spawn(move || {
                loop {
                    // Steal work from queue
                    let chunk = {
                        let mut queue = queue.lock();
                        queue.pop()
                    };
                    
                    let Some(chunk) = chunk else {
                        break; // No more work
                    };
                    
                    // Process chunk
                    match proc(&chunk) {
                        Ok(result) => {
                            if sender.send(result).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Chunk processing error: {}", e);
                        }
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all workers
        for handle in handles {
            handle.join().map_err(|_| {
                VeritasError::StreamError("Worker thread panicked".to_string())
            })?;
        }
        
        // Collect results
        let mut results = Vec::new();
        while let Ok(result) = self.result_receiver.try_recv() {
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Zero-copy chunk view for avoiding allocations
pub struct ChunkView<'a, T> {
    data: &'a [T],
    offset: usize,
    chunk_size: usize,
}

impl<'a, T> ChunkView<'a, T> {
    /// Create a new chunk view
    pub fn new(data: &'a [T], chunk_size: usize) -> Self {
        Self {
            data,
            offset: 0,
            chunk_size,
        }
    }
    
    /// Get the next chunk without allocation
    pub fn next_chunk(&mut self) -> Option<&'a [T]> {
        if self.offset >= self.data.len() {
            return None;
        }
        
        let end = (self.offset + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.offset..end];
        self.offset = end;
        
        Some(chunk)
    }
    
    /// Reset to beginning
    pub fn reset(&mut self) {
        self.offset = 0;
    }
    
    /// Get remaining data
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }
}

/// Chunked writer for efficient output
pub struct ChunkedWriter<W: std::io::Write> {
    writer: W,
    buffer: Vec<u8>,
    buffer_size: usize,
    bytes_written: usize,
}

impl<W: std::io::Write> ChunkedWriter<W> {
    /// Create a new chunked writer
    pub fn new(writer: W, buffer_size: usize) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            bytes_written: 0,
        }
    }
    
    /// Write data, buffering until chunk is full
    pub fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        let mut offset = 0;
        
        while offset < data.len() {
            let available = self.buffer_size - self.buffer.len();
            let to_write = available.min(data.len() - offset);
            
            self.buffer.extend_from_slice(&data[offset..offset + to_write]);
            offset += to_write;
            
            if self.buffer.len() >= self.buffer_size {
                self.flush_buffer()?;
            }
        }
        
        Ok(())
    }
    
    /// Flush any buffered data
    pub fn flush(&mut self) -> std::io::Result<()> {
        if !self.buffer.is_empty() {
            self.flush_buffer()?;
        }
        self.writer.flush()
    }
    
    /// Get total bytes written
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }
    
    fn flush_buffer(&mut self) -> std::io::Result<()> {
        self.writer.write_all(&self.buffer)?;
        self.bytes_written += self.buffer.len();
        self.buffer.clear();
        Ok(())
    }
}

// Helper functions

fn get_current_memory_usage() -> usize {
    // Simplified - in practice would use platform-specific APIs
    #[cfg(feature = "memory-profiling")]
    {
        use crate::optimization::memory_profiler;
        let (current, _, _) = memory_profiler::current_memory_stats();
        current
    }
    #[cfg(not(feature = "memory-profiling"))]
    {
        0
    }
}

fn calculate_throughput(bytes: usize, elapsed: std::time::Duration) -> f64 {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        mb / seconds
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_chunk_processor() {
        let mut processor = AdaptiveChunkProcessor::new(10, 100);
        let data: Vec<i32> = (0..1000).collect();
        
        let results = processor.process(&data, |chunk| {
            Ok(ProcessingResult {
                items_processed: chunk.len(),
                features_extracted: chunk.len() / 2,
                processing_time_ms: 1.0,
            })
        }).unwrap();
        
        assert!(!results.is_empty());
        let total_processed: usize = results.iter()
            .map(|r| r.items_processed)
            .sum();
        assert_eq!(total_processed, 1000);
    }
    
    #[test]
    fn test_chunk_view() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut view = ChunkView::new(&data, 3);
        
        let chunk1 = view.next_chunk().unwrap();
        assert_eq!(chunk1, &[1, 2, 3]);
        
        let chunk2 = view.next_chunk().unwrap();
        assert_eq!(chunk2, &[4, 5, 6]);
        
        assert_eq!(view.remaining(), 4);
        
        view.reset();
        let chunk = view.next_chunk().unwrap();
        assert_eq!(chunk, &[1, 2, 3]);
    }
    
    #[test]
    fn test_chunked_writer() {
        let output = Vec::new();
        let mut writer = ChunkedWriter::new(output, 10);
        
        writer.write(b"Hello").unwrap();
        writer.write(b"World").unwrap();
        writer.write(b"Test").unwrap();
        
        writer.flush().unwrap();
        
        assert_eq!(writer.bytes_written(), 14);
    }
}