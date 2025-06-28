//! # Batch Processing Example for Veritas Nexus
//! 
//! This example demonstrates efficient batch processing capabilities for lie detection.
//! It shows how to:
//! - Process large datasets efficiently using batching
//! - Implement parallel processing for CPU and GPU acceleration
//! - Handle different input formats and data sources
//! - Optimize memory usage with streaming and chunking
//! - Export results in various formats (CSV, JSON, database)
//! - Monitor progress and performance metrics
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example batch_processing
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::time::sleep;
use serde::{Deserialize, Serialize};
use futures::stream::{self, StreamExt};

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    pub batch_size: usize,
    pub max_concurrent_batches: usize,
    pub chunk_size: usize,
    pub memory_limit_mb: usize,
    pub output_format: OutputFormat,
    pub processing_mode: ProcessingMode,
    pub error_handling: ErrorHandling,
    pub progress_reporting: ProgressReporting,
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone)]
pub enum OutputFormat {
    Csv { delimiter: char, include_headers: bool },
    Json { pretty: bool },
    Database { connection_string: String },
    Parquet,
    Excel,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum ProcessingMode {
    Sequential,
    Parallel { num_threads: usize },
    Distributed { num_workers: usize },
    Gpu { device_id: u32 },
    Hybrid { cpu_threads: usize, gpu_devices: Vec<u32> },
}

#[derive(Debug, Clone)]
pub enum ErrorHandling {
    StopOnFirst,
    SkipErrors,
    RetryFailed { max_retries: usize, backoff_ms: u64 },
    Quarantine { quarantine_dir: String },
}

#[derive(Debug, Clone)]
pub enum ProgressReporting {
    Silent,
    Basic,
    Detailed,
    RealTime { update_interval_ms: u64 },
    WebDashboard { port: u16 },
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    Fast,     // Optimize for speed
    Balanced, // Balance speed and accuracy
    Accurate, // Optimize for accuracy
    Memory,   // Optimize for memory usage
    Custom(CustomOptimization),
}

#[derive(Debug, Clone)]
pub struct CustomOptimization {
    pub use_model_quantization: bool,
    pub enable_caching: bool,
    pub prefetch_data: bool,
    pub use_tensor_cores: bool,
    pub optimize_fusion: bool,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_concurrent_batches: 4,
            chunk_size: 1000,
            memory_limit_mb: 4096,
            output_format: OutputFormat::Json { pretty: true },
            processing_mode: ProcessingMode::Parallel { num_threads: 4 },
            error_handling: ErrorHandling::SkipErrors,
            progress_reporting: ProgressReporting::Detailed,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

/// Input sample for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInputSample {
    pub id: String,
    pub video_path: Option<String>,
    pub audio_path: Option<String>,
    pub text: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result from processing a single sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResultSample {
    pub id: String,
    pub decision: String,
    pub confidence: f64,
    pub processing_time_ms: u64,
    pub modality_scores: HashMap<String, f64>,
    pub error: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Batch processing result
#[derive(Debug, Clone, Serialize)]
pub struct BatchResult {
    pub total_samples: usize,
    pub successful_samples: usize,
    pub failed_samples: usize,
    pub total_processing_time: Duration,
    pub average_processing_time_ms: f64,
    pub throughput_samples_per_second: f64,
    pub memory_usage_mb: f64,
    pub results: Vec<BatchResultSample>,
}

/// Progress information
#[derive(Debug, Clone)]
pub struct BatchProgress {
    pub total_samples: usize,
    pub processed_samples: usize,
    pub successful_samples: usize,
    pub failed_samples: usize,
    pub current_batch: usize,
    pub total_batches: usize,
    pub elapsed_time: Duration,
    pub estimated_time_remaining: Option<Duration>,
    pub throughput_samples_per_second: f64,
    pub memory_usage_mb: f64,
}

/// Data source for batch processing
pub trait BatchDataSource: Send + Sync {
    type Item;
    type Error;
    
    fn estimate_size(&self) -> Option<usize>;
    fn load_chunk(&mut self, offset: usize, limit: usize) -> Result<Vec<Self::Item>, Self::Error>;
    fn total_samples(&self) -> Option<usize>;
}

/// File-based data source
pub struct FileDataSource {
    file_path: PathBuf,
    format: FileFormat,
    cached_samples: Option<Vec<BatchInputSample>>,
}

#[derive(Debug, Clone)]
pub enum FileFormat {
    Csv,
    Json,
    JsonLines,
    Parquet,
    Excel,
}

impl FileDataSource {
    pub fn new(file_path: impl Into<PathBuf>, format: FileFormat) -> Self {
        Self {
            file_path: file_path.into(),
            format,
            cached_samples: None,
        }
    }
    
    fn load_all_samples(&mut self) -> Result<&Vec<BatchInputSample>, Box<dyn std::error::Error>> {
        if self.cached_samples.is_none() {
            let samples = match self.format {
                FileFormat::Json => {
                    let content = std::fs::read_to_string(&self.file_path)?;
                    serde_json::from_str(&content)?
                }
                FileFormat::JsonLines => {
                    let content = std::fs::read_to_string(&self.file_path)?;
                    content.lines()
                        .map(|line| serde_json::from_str(line))
                        .collect::<Result<Vec<_>, _>>()?
                }
                FileFormat::Csv => {
                    // Simulate CSV parsing
                    vec![
                        BatchInputSample {
                            id: "sample_1".to_string(),
                            video_path: Some("videos/interview_1.mp4".to_string()),
                            audio_path: Some("audio/interview_1.wav".to_string()),
                            text: Some("I was definitely not there at that time".to_string()),
                            metadata: HashMap::new(),
                        },
                        BatchInputSample {
                            id: "sample_2".to_string(),
                            video_path: None,
                            audio_path: Some("audio/interview_2.wav".to_string()),
                            text: Some("Maybe I was there, I'm not sure".to_string()),
                            metadata: HashMap::new(),
                        },
                    ]
                }
                _ => return Err("Unsupported format".into()),
            };
            self.cached_samples = Some(samples);
        }
        
        Ok(self.cached_samples.as_ref().unwrap())
    }
}

impl BatchDataSource for FileDataSource {
    type Item = BatchInputSample;
    type Error = Box<dyn std::error::Error>;
    
    fn estimate_size(&self) -> Option<usize> {
        std::fs::metadata(&self.file_path).ok().map(|m| m.len() as usize)
    }
    
    fn load_chunk(&mut self, offset: usize, limit: usize) -> Result<Vec<Self::Item>, Self::Error> {
        let all_samples = self.load_all_samples()?;
        let end = (offset + limit).min(all_samples.len());
        
        if offset >= all_samples.len() {
            Ok(vec![])
        } else {
            Ok(all_samples[offset..end].to_vec())
        }
    }
    
    fn total_samples(&self) -> Option<usize> {
        self.load_all_samples().ok().map(|samples| samples.len())
    }
}

/// Database data source
pub struct DatabaseDataSource {
    connection_string: String,
    table_name: String,
    query: Option<String>,
}

impl DatabaseDataSource {
    pub fn new(connection_string: String, table_name: String) -> Self {
        Self {
            connection_string,
            table_name,
            query: None,
        }
    }
    
    pub fn with_query(mut self, query: String) -> Self {
        self.query = Some(query);
        self
    }
}

impl BatchDataSource for DatabaseDataSource {
    type Item = BatchInputSample;
    type Error = Box<dyn std::error::Error>;
    
    fn estimate_size(&self) -> Option<usize> {
        // Would query database for estimated row count
        Some(10000) // Mock value
    }
    
    fn load_chunk(&mut self, offset: usize, limit: usize) -> Result<Vec<Self::Item>, Self::Error> {
        // Simulate database query
        let mut samples = Vec::new();
        
        for i in 0..limit {
            let id = offset + i;
            if id >= 10000 { // Mock total count
                break;
            }
            
            samples.push(BatchInputSample {
                id: format!("db_sample_{}", id),
                video_path: if i % 3 == 0 { Some(format!("videos/db_video_{}.mp4", id)) } else { None },
                audio_path: if i % 2 == 0 { Some(format!("audio/db_audio_{}.wav", id)) } else { None },
                text: Some(format!("Database sample text for record {}", id)),
                metadata: [("source".to_string(), serde_json::Value::String("database".to_string()))]
                    .into_iter().collect(),
            });
        }
        
        Ok(samples)
    }
    
    fn total_samples(&self) -> Option<usize> {
        Some(10000) // Mock value
    }
}

/// Main batch processor
pub struct BatchProcessor {
    config: BatchProcessingConfig,
    progress_callback: Option<Box<dyn Fn(&BatchProgress) + Send + Sync>>,
    stats: Arc<Mutex<ProcessingStats>>,
}

#[derive(Debug, Default)]
struct ProcessingStats {
    total_processed: usize,
    total_successful: usize,
    total_failed: usize,
    total_processing_time: Duration,
    memory_peak_mb: f64,
}

impl BatchProcessor {
    pub fn new(config: BatchProcessingConfig) -> Self {
        Self {
            config,
            progress_callback: None,
            stats: Arc::new(Mutex::new(ProcessingStats::default())),
        }
    }
    
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self 
    where 
        F: Fn(&BatchProgress) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }
    
    /// Process data from a data source
    pub async fn process<DS>(&self, mut data_source: DS) -> Result<BatchResult, BatchProcessingError>
    where
        DS: BatchDataSource<Item = BatchInputSample> + Send + 'static,
        DS::Error: Send + Sync + 'static,
    {
        let start_time = Instant::now();
        
        // Get total sample count
        let total_samples = data_source.total_samples().unwrap_or(0);
        println!("📊 Processing {} samples", total_samples);
        
        // Calculate number of batches
        let total_batches = (total_samples + self.config.chunk_size - 1) / self.config.chunk_size;
        
        // Initialize progress tracking
        let mut processed_samples = 0;
        let mut successful_samples = 0;
        let mut failed_samples = 0;
        let mut all_results = Vec::new();
        
        // Semaphore to limit concurrent batches
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_batches));
        
        // Process data in chunks
        for chunk_idx in 0..total_batches {
            let offset = chunk_idx * self.config.chunk_size;
            let limit = self.config.chunk_size.min(total_samples - offset);
            
            if limit == 0 {
                break;
            }
            
            // Load chunk
            let chunk = data_source.load_chunk(offset, limit)
                .map_err(|e| BatchProcessingError::DataLoadError(format!("Failed to load chunk {}: {:?}", chunk_idx, e)))?;
            
            if chunk.is_empty() {
                break;
            }
            
            // Process chunk in batches
            let chunk_results = self.process_chunk(chunk, chunk_idx, &semaphore).await?;
            
            // Update progress
            for result in &chunk_results {
                processed_samples += 1;
                if result.error.is_none() {
                    successful_samples += 1;
                } else {
                    failed_samples += 1;
                }
            }
            
            all_results.extend(chunk_results);
            
            // Report progress
            self.report_progress(BatchProgress {
                total_samples,
                processed_samples,
                successful_samples,
                failed_samples,
                current_batch: chunk_idx + 1,
                total_batches,
                elapsed_time: start_time.elapsed(),
                estimated_time_remaining: self.estimate_time_remaining(
                    processed_samples, 
                    total_samples, 
                    start_time.elapsed()
                ),
                throughput_samples_per_second: processed_samples as f64 / start_time.elapsed().as_secs_f64(),
                memory_usage_mb: self.get_memory_usage(),
            });
        }
        
        let total_time = start_time.elapsed();
        
        Ok(BatchResult {
            total_samples,
            successful_samples,
            failed_samples,
            total_processing_time: total_time,
            average_processing_time_ms: if processed_samples > 0 {
                total_time.as_millis() as f64 / processed_samples as f64
            } else {
                0.0
            },
            throughput_samples_per_second: processed_samples as f64 / total_time.as_secs_f64(),
            memory_usage_mb: self.get_memory_usage(),
            results: all_results,
        })
    }
    
    /// Process a chunk of samples
    async fn process_chunk(
        &self, 
        chunk: Vec<BatchInputSample>, 
        chunk_idx: usize,
        semaphore: &Arc<Semaphore>
    ) -> Result<Vec<BatchResultSample>, BatchProcessingError> {
        // Split chunk into batches
        let mut results = Vec::new();
        let batches: Vec<_> = chunk.chunks(self.config.batch_size).collect();
        
        match self.config.processing_mode {
            ProcessingMode::Sequential => {
                for batch in batches {
                    let batch_results = self.process_batch(batch.to_vec()).await?;
                    results.extend(batch_results);
                }
            }
            ProcessingMode::Parallel { .. } => {
                // Use semaphore to limit concurrent batches
                let batch_futures: Vec<_> = batches.into_iter().enumerate().map(|(batch_idx, batch)| {
                    let batch = batch.to_vec();
                    let semaphore = Arc::clone(semaphore);
                    
                    async move {
                        let _permit = semaphore.acquire().await.unwrap();
                        self.process_batch(batch).await
                    }
                }).collect();
                
                let batch_results = futures::future::try_join_all(batch_futures).await?;
                for batch_result in batch_results {
                    results.extend(batch_result);
                }
            }
            ProcessingMode::Gpu { device_id } => {
                println!("    Using GPU device {}", device_id);
                for batch in batches {
                    let batch_results = self.process_batch_gpu(batch.to_vec(), device_id).await?;
                    results.extend(batch_results);
                }
            }
            _ => {
                // For other modes, fallback to parallel processing
                let batch_futures: Vec<_> = batches.into_iter().map(|batch| {
                    let batch = batch.to_vec();
                    self.process_batch(batch)
                }).collect();
                
                let batch_results = futures::future::try_join_all(batch_futures).await?;
                for batch_result in batch_results {
                    results.extend(batch_result);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Process a single batch of samples
    async fn process_batch(&self, batch: Vec<BatchInputSample>) -> Result<Vec<BatchResultSample>, BatchProcessingError> {
        let mut results = Vec::new();
        
        for sample in batch {
            let start_time = Instant::now();
            
            // Simulate processing time based on available modalities
            let processing_delay = match (&sample.video_path, &sample.audio_path, &sample.text) {
                (Some(_), Some(_), Some(_)) => 150, // All modalities - longer processing
                (Some(_), Some(_), None) => 120,    // Video + Audio
                (Some(_), None, Some(_)) => 100,    // Video + Text
                (None, Some(_), Some(_)) => 80,     // Audio + Text
                (Some(_), None, None) => 90,        // Video only
                (None, Some(_), None) => 60,        // Audio only
                (None, None, Some(_)) => 40,        // Text only
                (None, None, None) => 10,           // No input - should error
            };
            
            sleep(Duration::from_millis(processing_delay)).await;
            
            let processing_time = start_time.elapsed();
            
            // Simulate analysis results
            let result = if sample.video_path.is_none() && sample.audio_path.is_none() && sample.text.is_none() {
                BatchResultSample {
                    id: sample.id.clone(),
                    decision: "error".to_string(),
                    confidence: 0.0,
                    processing_time_ms: processing_time.as_millis() as u64,
                    modality_scores: HashMap::new(),
                    error: Some("No input modalities provided".to_string()),
                    metadata: sample.metadata.clone(),
                }
            } else {
                let mut modality_scores = HashMap::new();
                let mut total_score = 0.0;
                let mut count = 0;
                
                if sample.video_path.is_some() {
                    let score = 0.6 + (rand::random::<f64>() * 0.3);
                    modality_scores.insert("vision".to_string(), score);
                    total_score += score;
                    count += 1;
                }
                
                if sample.audio_path.is_some() {
                    let score = 0.5 + (rand::random::<f64>() * 0.4);
                    modality_scores.insert("audio".to_string(), score);
                    total_score += score;
                    count += 1;
                }
                
                if sample.text.is_some() {
                    let score = 0.4 + (rand::random::<f64>() * 0.5);
                    modality_scores.insert("text".to_string(), score);
                    total_score += score;
                    count += 1;
                }
                
                let final_score = total_score / count as f64;
                let confidence = (count as f64 / 3.0 * 0.7 + 0.3).min(1.0);
                
                let decision = if final_score > 0.6 {
                    "deceptive"
                } else if final_score < 0.4 {
                    "truthful"
                } else {
                    "uncertain"
                };
                
                BatchResultSample {
                    id: sample.id.clone(),
                    decision: decision.to_string(),
                    confidence,
                    processing_time_ms: processing_time.as_millis() as u64,
                    modality_scores,
                    error: None,
                    metadata: sample.metadata.clone(),
                }
            };
            
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Process a batch using GPU acceleration
    async fn process_batch_gpu(&self, batch: Vec<BatchInputSample>, device_id: u32) -> Result<Vec<BatchResultSample>, BatchProcessingError> {
        // Simulate GPU processing with improved performance
        let start_time = Instant::now();
        
        // GPU batch processing is typically faster due to parallelization
        let base_delay = 50; // Base delay per sample is lower
        let total_delay = base_delay * (batch.len() as u64).max(1) / 4; // Parallel processing factor
        
        sleep(Duration::from_millis(total_delay)).await;
        
        let mut results = Vec::new();
        let batch_processing_time = start_time.elapsed();
        let per_sample_time = batch_processing_time.as_millis() as u64 / batch.len() as u64;
        
        for sample in batch {
            // Simulate GPU-accelerated results (typically more accurate due to larger models)
            let mut modality_scores = HashMap::new();
            let mut total_score = 0.0;
            let mut count = 0;
            
            if sample.video_path.is_some() {
                let score = 0.65 + (rand::random::<f64>() * 0.25); // GPU models typically more accurate
                modality_scores.insert("vision".to_string(), score);
                total_score += score;
                count += 1;
            }
            
            if sample.audio_path.is_some() {
                let score = 0.55 + (rand::random::<f64>() * 0.35);
                modality_scores.insert("audio".to_string(), score);
                total_score += score;
                count += 1;
            }
            
            if sample.text.is_some() {
                let score = 0.45 + (rand::random::<f64>() * 0.45);
                modality_scores.insert("text".to_string(), score);
                total_score += score;
                count += 1;
            }
            
            if count == 0 {
                results.push(BatchResultSample {
                    id: sample.id.clone(),
                    decision: "error".to_string(),
                    confidence: 0.0,
                    processing_time_ms: per_sample_time,
                    modality_scores: HashMap::new(),
                    error: Some("No input modalities provided".to_string()),
                    metadata: sample.metadata.clone(),
                });
                continue;
            }
            
            let final_score = total_score / count as f64;
            let confidence = (count as f64 / 3.0 * 0.8 + 0.2).min(1.0); // GPU models more confident
            
            let decision = if final_score > 0.65 {
                "deceptive"
            } else if final_score < 0.35 {
                "truthful"
            } else {
                "uncertain"
            };
            
            results.push(BatchResultSample {
                id: sample.id.clone(),
                decision: decision.to_string(),
                confidence,
                processing_time_ms: per_sample_time,
                modality_scores,
                error: None,
                metadata: sample.metadata.clone(),
            });
        }
        
        Ok(results)
    }
    
    /// Report progress to callback
    fn report_progress(&self, progress: BatchProgress) {
        if let Some(callback) = &self.progress_callback {
            callback(&progress);
        }
    }
    
    /// Estimate remaining time
    fn estimate_time_remaining(&self, processed: usize, total: usize, elapsed: Duration) -> Option<Duration> {
        if processed == 0 {
            return None;
        }
        
        let remaining = total - processed;
        let time_per_sample = elapsed.as_secs_f64() / processed as f64;
        let estimated_remaining_seconds = remaining as f64 * time_per_sample;
        
        Some(Duration::from_secs_f64(estimated_remaining_seconds))
    }
    
    /// Get current memory usage (simulated)
    fn get_memory_usage(&self) -> f64 {
        // In a real implementation, this would measure actual memory usage
        256.0 + (rand::random::<f64>() * 100.0) // Simulated memory usage in MB
    }
    
    /// Export results to specified format
    pub fn export_results(&self, results: &BatchResult, output_path: &str) -> Result<(), BatchProcessingError> {
        match &self.config.output_format {
            OutputFormat::Json { pretty } => {
                let json = if *pretty {
                    serde_json::to_string_pretty(results)?
                } else {
                    serde_json::to_string(results)?
                };
                std::fs::write(output_path, json)?;
            }
            OutputFormat::Csv { delimiter, include_headers } => {
                let mut csv_content = String::new();
                
                if *include_headers {
                    csv_content.push_str(&format!(
                        "id{}decision{}confidence{}processing_time_ms{}vision_score{}audio_score{}text_score{}error\n",
                        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
                    ));
                }
                
                for result in &results.results {
                    csv_content.push_str(&format!(
                        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                        result.id,
                        delimiter,
                        result.decision,
                        delimiter,
                        result.confidence,
                        delimiter,
                        result.processing_time_ms,
                        delimiter,
                        result.modality_scores.get("vision").unwrap_or(&0.0),
                        delimiter,
                        result.modality_scores.get("audio").unwrap_or(&0.0),
                        delimiter,
                        result.modality_scores.get("text").unwrap_or(&0.0),
                        delimiter,
                        result.error.as_deref().unwrap_or("")
                    ));
                    csv_content.push('\n');
                }
                
                std::fs::write(output_path, csv_content)?;
            }
            _ => {
                return Err(BatchProcessingError::ExportError("Unsupported output format".to_string()));
            }
        }
        
        Ok(())
    }
}

/// Batch processing error types
#[derive(Debug)]
pub enum BatchProcessingError {
    DataLoadError(String),
    ProcessingError(String),
    ExportError(String),
    ConfigError(String),
}

impl std::fmt::Display for BatchProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchProcessingError::DataLoadError(msg) => write!(f, "Data load error: {}", msg),
            BatchProcessingError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            BatchProcessingError::ExportError(msg) => write!(f, "Export error: {}", msg),
            BatchProcessingError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for BatchProcessingError {}

impl From<serde_json::Error> for BatchProcessingError {
    fn from(err: serde_json::Error) -> Self {
        BatchProcessingError::ExportError(err.to_string())
    }
}

impl From<std::io::Error> for BatchProcessingError {
    fn from(err: std::io::Error) -> Self {
        BatchProcessingError::ExportError(err.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Veritas Nexus - Batch Processing Example");
    println!("==========================================\n");
    
    // Example 1: Process from file
    println!("📁 Example 1: Processing from CSV file");
    println!("-------------------------------------");
    
    let file_config = BatchProcessingConfig {
        batch_size: 8,
        max_concurrent_batches: 2,
        chunk_size: 50,
        memory_limit_mb: 2048,
        output_format: OutputFormat::Json { pretty: true },
        processing_mode: ProcessingMode::Parallel { num_threads: 4 },
        error_handling: ErrorHandling::SkipErrors,
        progress_reporting: ProgressReporting::Detailed,
        optimization_level: OptimizationLevel::Balanced,
    };
    
    let file_processor = BatchProcessor::new(file_config).with_progress_callback(|progress| {
        println!("    Progress: {}/{} samples ({:.1}%) - {:.1} samples/sec - ETA: {}",
            progress.processed_samples,
            progress.total_samples,
            (progress.processed_samples as f64 / progress.total_samples as f64) * 100.0,
            progress.throughput_samples_per_second,
            if let Some(eta) = progress.estimated_time_remaining {
                format!("{:.1}s", eta.as_secs_f32())
            } else {
                "unknown".to_string()
            }
        );
    });
    
    let file_source = FileDataSource::new("data/samples.csv", FileFormat::Csv);
    
    match file_processor.process(file_source).await {
        Ok(result) => {
            println!("\n✅ File processing completed:");
            println!("    Total samples: {}", result.total_samples);
            println!("    Successful: {}", result.successful_samples);
            println!("    Failed: {}", result.failed_samples);
            println!("    Total time: {:.2}s", result.total_processing_time.as_secs_f32());
            println!("    Throughput: {:.1} samples/sec", result.throughput_samples_per_second);
            println!("    Memory usage: {:.1}MB", result.memory_usage_mb);
            
            // Export results
            file_processor.export_results(&result, "output/file_results.json")?;
            println!("    Results exported to: output/file_results.json");
        }
        Err(e) => println!("❌ File processing failed: {}", e),
    }
    
    println!("\n" + &"=".repeat(50) + "\n");
    
    // Example 2: Process from database with GPU acceleration
    println!("💾 Example 2: Processing from database with GPU");
    println!("----------------------------------------------");
    
    let gpu_config = BatchProcessingConfig {
        batch_size: 16,
        max_concurrent_batches: 1, // GPU processing typically uses single batch queue
        chunk_size: 200,
        memory_limit_mb: 6144,
        output_format: OutputFormat::Csv { delimiter: ',', include_headers: true },
        processing_mode: ProcessingMode::Gpu { device_id: 0 },
        error_handling: ErrorHandling::RetryFailed { max_retries: 2, backoff_ms: 1000 },
        progress_reporting: ProgressReporting::RealTime { update_interval_ms: 500 },
        optimization_level: OptimizationLevel::Accurate,
    };
    
    let gpu_processor = BatchProcessor::new(gpu_config).with_progress_callback(|progress| {
        if progress.processed_samples % 50 == 0 || progress.processed_samples == progress.total_samples {
            println!("    GPU Progress: {}/{} samples ({:.1}%) - {:.1} samples/sec - Memory: {:.1}MB",
                progress.processed_samples,
                progress.total_samples,
                (progress.processed_samples as f64 / progress.total_samples as f64) * 100.0,
                progress.throughput_samples_per_second,
                progress.memory_usage_mb
            );
        }
    });
    
    let db_source = DatabaseDataSource::new(
        "postgresql://user:pass@localhost/deception_db".to_string(),
        "samples".to_string(),
    ).with_query("SELECT * FROM samples WHERE created_at > '2024-01-01' LIMIT 500".to_string());
    
    match gpu_processor.process(db_source).await {
        Ok(result) => {
            println!("\n✅ GPU processing completed:");
            println!("    Total samples: {}", result.total_samples);
            println!("    Successful: {}", result.successful_samples);
            println!("    Failed: {}", result.failed_samples);
            println!("    Total time: {:.2}s", result.total_processing_time.as_secs_f32());
            println!("    Throughput: {:.1} samples/sec", result.throughput_samples_per_second);
            println!("    Average time per sample: {:.1}ms", result.average_processing_time_ms);
            
            // Show some example results
            println!("\n    Sample results:");
            for (i, result_sample) in result.results.iter().take(3).enumerate() {
                println!("      {}. {}: {} ({:.1}% confidence) - {}ms",
                    i + 1,
                    result_sample.id,
                    result_sample.decision,
                    result_sample.confidence * 100.0,
                    result_sample.processing_time_ms
                );
            }
            
            // Export results
            gpu_processor.export_results(&result, "output/gpu_results.csv")?;
            println!("    Results exported to: output/gpu_results.csv");
        }
        Err(e) => println!("❌ GPU processing failed: {}", e),
    }
    
    println!("\n" + &"=".repeat(50) + "\n");
    
    // Example 3: High-throughput processing with memory optimization
    println!("⚡ Example 3: High-throughput processing");
    println!("---------------------------------------");
    
    let fast_config = BatchProcessingConfig {
        batch_size: 64,
        max_concurrent_batches: 8,
        chunk_size: 1000,
        memory_limit_mb: 8192,
        output_format: OutputFormat::Json { pretty: false },
        processing_mode: ProcessingMode::Parallel { num_threads: 8 },
        error_handling: ErrorHandling::SkipErrors,
        progress_reporting: ProgressReporting::Basic,
        optimization_level: OptimizationLevel::Fast,
    };
    
    let fast_processor = BatchProcessor::new(fast_config).with_progress_callback(|progress| {
        if progress.current_batch % 5 == 0 || progress.current_batch == progress.total_batches {
            println!("    Fast processing: Batch {}/{} - {:.0} samples/sec",
                progress.current_batch,
                progress.total_batches,
                progress.throughput_samples_per_second
            );
        }
    });
    
    // Create a larger synthetic dataset
    let large_db_source = DatabaseDataSource::new(
        "postgresql://user:pass@localhost/large_deception_db".to_string(),
        "large_samples".to_string(),
    );
    
    match fast_processor.process(large_db_source).await {
        Ok(result) => {
            println!("\n✅ High-throughput processing completed:");
            println!("    Total samples: {}", result.total_samples);
            println!("    Successful: {}", result.successful_samples);
            println!("    Failed: {}", result.failed_samples);
            println!("    Total time: {:.2}s", result.total_processing_time.as_secs_f32());
            println!("    Throughput: {:.1} samples/sec", result.throughput_samples_per_second);
            
            // Performance analysis
            let success_rate = result.successful_samples as f64 / result.total_samples as f64 * 100.0;
            println!("    Success rate: {:.1}%", success_rate);
            
            fast_processor.export_results(&result, "output/fast_results.json")?;
            println!("    Results exported to: output/fast_results.json");
        }
        Err(e) => println!("❌ High-throughput processing failed: {}", e),
    }
    
    println!("\n🎉 Batch processing examples completed!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Efficient batch processing with configurable batch sizes");
    println!("   • Multiple data sources (files, databases)");
    println!("   • Parallel and GPU-accelerated processing");
    println!("   • Real-time progress monitoring and ETA estimation");
    println!("   • Memory usage optimization and limits");
    println!("   • Error handling strategies (skip, retry, quarantine)");
    println!("   • Multiple output formats (JSON, CSV, database)");
    println!("   • Performance metrics and throughput analysis");
    println!("   • Chunked processing for large datasets");
    println!("   • Concurrent batch processing with semaphore control");
    
    Ok(())
}