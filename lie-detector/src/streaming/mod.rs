//! Streaming data processing for memory efficiency
//! 
//! This module provides streaming interfaces for processing large
//! audio and video data without loading everything into memory.

use crate::{Result, VeritasError};
use std::io::{Read, Write, BufReader};
use std::sync::Arc;
use crossbeam_channel::{bounded, Receiver, Sender};
use num_cpus;

pub mod audio_stream;
pub mod image_stream;
pub mod chunk_processor;
pub mod lazy_loader;

pub use audio_stream::*;
pub use image_stream::*;
pub use chunk_processor::*;
pub use lazy_loader::*;

/// Configuration for streaming processing
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Size of chunks to process
    pub chunk_size: usize,
    /// Number of chunks to buffer
    pub buffer_size: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Number of worker threads
    pub num_workers: usize,
    /// Memory limit per chunk (bytes)
    pub memory_limit_per_chunk: Option<usize>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 16384, // 16KB chunks
            buffer_size: 16,
            parallel: true,
            num_workers: num_cpus::get(),
            memory_limit_per_chunk: Some(10 * 1024 * 1024), // 10MB per chunk
        }
    }
}

/// Trait for streamable data processing
pub trait StreamProcessor: Send + Sync {
    /// Input chunk type
    type Input: Send + Sync;
    /// Output type
    type Output: Send + Sync;
    /// Configuration type
    type Config: Clone + Send + Sync;
    
    /// Process a single chunk
    fn process_chunk(&mut self, chunk: Self::Input) -> Result<Self::Output>;
    
    /// Get the minimum chunk size
    fn min_chunk_size(&self) -> usize;
    
    /// Check if processor supports parallel processing
    fn supports_parallel(&self) -> bool {
        true
    }
    
    /// Finalize processing (called after all chunks)
    fn finalize(&mut self) -> Result<Option<Self::Output>> {
        Ok(None)
    }
}

/// Generic streaming pipeline
pub struct StreamingPipeline<P: StreamProcessor> {
    processor: Arc<parking_lot::Mutex<P>>,
    config: StreamConfig,
    input_sender: Sender<P::Input>,
    output_receiver: Receiver<P::Output>,
    worker_handles: Vec<std::thread::JoinHandle<()>>,
}

impl<P: StreamProcessor + 'static> StreamingPipeline<P> {
    /// Create a new streaming pipeline
    pub fn new(processor: P, config: StreamConfig) -> Result<Self> {
        let (input_sender, input_receiver) = bounded(config.buffer_size);
        let (output_sender, output_receiver) = bounded(config.buffer_size);
        
        let processor = Arc::new(parking_lot::Mutex::new(processor));
        let num_workers = if config.parallel { config.num_workers } else { 1 };
        
        let mut worker_handles = Vec::new();
        
        for _ in 0..num_workers {
            let processor_clone = processor.clone();
            let input_rx = input_receiver.clone();
            let output_tx = output_sender.clone();
            
            let handle = std::thread::spawn(move || {
                while let Ok(chunk) = input_rx.recv() {
                    let mut proc = processor_clone.lock();
                    match proc.process_chunk(chunk) {
                        Ok(result) => {
                            if output_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Chunk processing error: {}", e);
                        }
                    }
                }
            });
            
            worker_handles.push(handle);
        }
        
        Ok(Self {
            processor,
            config,
            input_sender,
            output_receiver,
            worker_handles,
        })
    }
    
    /// Send a chunk for processing
    pub fn send_chunk(&self, chunk: P::Input) -> Result<()> {
        self.input_sender.send(chunk)
            .map_err(|_| VeritasError::io_error("Failed to send chunk to processing pipeline"))
    }
    
    /// Receive a processed result
    pub fn receive_result(&self) -> Result<P::Output> {
        self.output_receiver.recv()
            .map_err(|_| VeritasError::io_error("Failed to receive result from processing pipeline"))
    }
    
    /// Try to receive a result without blocking
    pub fn try_receive_result(&self) -> Option<P::Output> {
        self.output_receiver.try_recv().ok()
    }
    
    /// Finalize the pipeline and wait for completion
    pub fn finalize(self) -> Result<Vec<P::Output>> {
        // Close input channel
        drop(self.input_sender);
        
        // Wait for workers to finish
        for handle in self.worker_handles {
            handle.join().map_err(|_| {
                VeritasError::internal_error_with_location("Worker thread panicked", "StreamingPipeline::finalize")
            })?;
        }
        
        // Collect remaining results
        let mut results = Vec::new();
        while let Ok(result) = self.output_receiver.recv() {
            results.push(result);
        }
        
        // Call processor finalize
        let mut processor = self.processor.lock();
        if let Some(final_result) = processor.finalize()? {
            results.push(final_result);
        }
        
        Ok(results)
    }
}

/// Iterator-based streaming for memory efficiency
pub struct StreamIterator<T> {
    receiver: Receiver<T>,
    finished: bool,
}

impl<T> StreamIterator<T> {
    /// Create a new stream iterator
    pub fn new(receiver: Receiver<T>) -> Self {
        Self {
            receiver,
            finished: false,
        }
    }
}

impl<T> Iterator for StreamIterator<T> {
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        
        match self.receiver.recv() {
            Ok(item) => Some(item),
            Err(_) => {
                self.finished = true;
                None
            }
        }
    }
}

/// Buffered stream reader for efficient I/O
pub struct BufferedStreamReader<R: Read> {
    reader: BufReader<R>,
    buffer: Vec<u8>,
    chunk_size: usize,
}

impl<R: Read> BufferedStreamReader<R> {
    /// Create a new buffered stream reader
    pub fn new(reader: R, chunk_size: usize) -> Self {
        Self {
            reader: BufReader::with_capacity(chunk_size * 2, reader),
            buffer: vec![0; chunk_size],
            chunk_size,
        }
    }
    
    /// Read the next chunk
    pub fn read_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        match self.reader.read_exact(&mut self.buffer) {
            Ok(()) => Ok(Some(self.buffer.clone())),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Try to read remaining data
                let mut remaining = Vec::new();
                self.reader.read_to_end(&mut remaining)?;
                if remaining.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(remaining))
                }
            }
            Err(e) => Err(e.into()),
        }
    }
}

/// Memory-bounded queue for streaming
pub struct BoundedQueue<T> {
    items: crossbeam_queue::ArrayQueue<T>,
    memory_limit: usize,
    current_memory: Arc<AtomicUsize>,
}

impl<T> BoundedQueue<T> {
    /// Create a new bounded queue
    pub fn new(capacity: usize, memory_limit: usize) -> Self {
        Self {
            items: crossbeam_queue::ArrayQueue::new(capacity),
            memory_limit,
            current_memory: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Try to push an item, respecting memory limits
    pub fn try_push(&self, item: T, item_size: usize) -> Result<()> {
        let current = self.current_memory.load(Ordering::Relaxed);
        if current + item_size > self.memory_limit {
            return Err(VeritasError::memory_error(
                "Queue memory limit exceeded"
            ));
        }
        
        if self.items.push(item).is_err() {
            return Err(VeritasError::io_error(
                "Queue capacity exceeded"
            ));
        }
        
        self.current_memory.fetch_add(item_size, Ordering::Relaxed);
        Ok(())
    }
    
    /// Try to pop an item
    pub fn try_pop(&self, item_size: usize) -> Option<T> {
        self.items.pop().map(|item| {
            self.current_memory.fetch_sub(item_size, Ordering::Relaxed);
            item
        })
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed)
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestProcessor;
    
    impl StreamProcessor for TestProcessor {
        type Input = Vec<u8>;
        type Output = usize;
        type Config = ();
        
        fn process_chunk(&mut self, chunk: Vec<u8>) -> Result<usize> {
            Ok(chunk.len())
        }
        
        fn min_chunk_size(&self) -> usize {
            1024
        }
    }
    
    #[test]
    fn test_streaming_pipeline() {
        let processor = TestProcessor;
        let config = StreamConfig::default();
        let pipeline = StreamingPipeline::new(processor, config).unwrap();
        
        // Send some chunks
        pipeline.send_chunk(vec![0; 1024]).unwrap();
        pipeline.send_chunk(vec![0; 2048]).unwrap();
        
        // Receive results
        let result1 = pipeline.receive_result().unwrap();
        assert_eq!(result1, 1024);
        
        let result2 = pipeline.receive_result().unwrap();
        assert_eq!(result2, 2048);
        
        let results = pipeline.finalize().unwrap();
        assert!(results.is_empty()); // All results already received
    }
    
    #[test]
    fn test_bounded_queue() {
        let queue: BoundedQueue<Vec<u8>> = BoundedQueue::new(10, 1024);
        
        // Push items
        queue.try_push(vec![0; 100], 100).unwrap();
        queue.try_push(vec![0; 200], 200).unwrap();
        
        assert_eq!(queue.memory_usage(), 300);
        
        // Pop items
        let item1 = queue.try_pop(100).unwrap();
        assert_eq!(item1.len(), 100);
        assert_eq!(queue.memory_usage(), 200);
        
        // Try to exceed memory limit
        let result = queue.try_push(vec![0; 900], 900);
        assert!(result.is_err());
    }
}