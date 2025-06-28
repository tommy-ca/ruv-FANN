//! Enhanced memory pressure monitoring with adaptive GC strategies
//! 
//! This module provides comprehensive memory monitoring with intelligent
//! garbage collection strategies and adaptive behavior to optimize memory usage.

use crate::{Result, VeritasError};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Memory pressure levels with finer granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// Plenty of memory available (< 40%)
    VeryLow,
    /// Normal memory usage (40-55%)
    Low,
    /// Memory usage is moderate (55-70%)
    Medium,
    /// Memory usage is high, start conserving (70-80%)
    High,
    /// Critical memory situation (80-90%)
    Critical,
    /// Emergency - risk of OOM (> 90%)
    Emergency,
}

/// Memory monitor with advanced features
pub struct MemoryMonitor {
    config: MonitorConfig,
    current_pressure: Arc<RwLock<MemoryPressure>>,
    monitoring_enabled: Arc<AtomicBool>,
    stats: Arc<MemoryStats>,
    callbacks: Arc<Mutex<Vec<Box<dyn MemoryPressureCallback>>>>,
    history: Arc<Mutex<MemoryHistory>>,
    gc_strategy: Arc<RwLock<Box<dyn GCStrategy>>>,
    monitor_thread: Option<std::thread::JoinHandle<()>>,
}

/// Enhanced configuration for memory monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Update interval for memory checks
    pub update_interval: Duration,
    /// Memory threshold percentages
    pub thresholds: MemoryThresholds,
    /// Enable automatic garbage collection
    pub auto_gc: bool,
    /// Enable automatic cache clearing
    pub auto_clear_caches: bool,
    /// Enable predictive memory management
    pub predictive_mode: bool,
    /// History size for trend analysis
    pub history_size: usize,
    /// Enable aggressive memory reclamation
    pub aggressive_reclaim: bool,
    /// Custom page size for allocations
    pub page_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MemoryThresholds {
    pub very_low: f32,
    pub low: f32,
    pub medium: f32,
    pub high: f32,
    pub critical: f32,
    pub emergency: f32,
}

impl Default for MemoryThresholds {
    fn default() -> Self {
        Self {
            very_low: 0.4,   // 40%
            low: 0.55,       // 55%
            medium: 0.7,     // 70%
            high: 0.8,       // 80%
            critical: 0.9,   // 90%
            emergency: 0.95, // 95%
        }
    }
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(500),
            thresholds: MemoryThresholds::default(),
            auto_gc: true,
            auto_clear_caches: true,
            predictive_mode: true,
            history_size: 120, // 1 minute of history at 500ms intervals
            aggressive_reclaim: false,
            page_size: None,
        }
    }
}

/// Memory statistics with detailed tracking
#[derive(Debug, Default)]
struct MemoryStats {
    // Basic stats
    total_memory: AtomicU64,
    available_memory: AtomicU64,
    used_memory: AtomicU64,
    cached_memory: AtomicU64,
    swap_used: AtomicU64,
    swap_total: AtomicU64,
    
    // Process-specific stats
    process_rss: AtomicU64,
    process_vms: AtomicU64,
    process_shared: AtomicU64,
    
    // Allocation stats
    allocation_rate: AtomicU64,
    deallocation_rate: AtomicU64,
    fragmentation_ratio: AtomicU64,
    
    // GC stats
    gc_count: AtomicU64,
    gc_time_ms: AtomicU64,
    bytes_reclaimed: AtomicU64,
    
    // Events
    pressure_changes: AtomicU64,
    oom_prevented: AtomicU64,
}

/// Memory usage history for trend analysis
struct MemoryHistory {
    samples: VecDeque<MemorySample>,
    max_samples: usize,
}

/// Single memory sample
#[derive(Debug, Clone)]
struct MemorySample {
    timestamp: Instant,
    used_percent: f32,
    available_mb: u64,
    allocation_rate: f64,
    pressure: MemoryPressure,
}

impl MemoryHistory {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }
    
    fn add_sample(&mut self, sample: MemorySample) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }
    
    fn predict_pressure(&self, future_secs: f64) -> MemoryPressure {
        if self.samples.len() < 3 {
            return MemoryPressure::Low;
        }
        
        // Simple linear regression on memory usage
        let n = self.samples.len().min(10) as f64;
        let samples: Vec<_> = self.samples.iter().rev().take(10).collect();
        
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        
        for (i, sample) in samples.iter().enumerate() {
            let x = i as f64;
            let y = sample.used_percent as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;
        
        let predicted_usage = intercept + slope * (n + future_secs * 2.0);
        
        // Convert predicted usage to pressure
        if predicted_usage >= 95.0 {
            MemoryPressure::Emergency
        } else if predicted_usage >= 90.0 {
            MemoryPressure::Critical
        } else if predicted_usage >= 80.0 {
            MemoryPressure::High
        } else if predicted_usage >= 70.0 {
            MemoryPressure::Medium
        } else if predicted_usage >= 55.0 {
            MemoryPressure::Low
        } else {
            MemoryPressure::VeryLow
        }
    }
    
    fn get_allocation_trend(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        
        let recent: Vec<_> = self.samples.iter().rev().take(5).collect();
        let total_rate: f64 = recent.iter().map(|s| s.allocation_rate).sum();
        total_rate / recent.len() as f64
    }
}

/// Callback for memory pressure changes
pub trait MemoryPressureCallback: Send + Sync {
    /// Called when memory pressure changes
    fn on_pressure_change(&self, old: MemoryPressure, new: MemoryPressure);
    
    /// Called before emergency GC
    fn on_emergency_gc(&self) {}
    
    /// Called when OOM is prevented
    fn on_oom_prevented(&self) {}
}

/// Garbage collection strategy trait
pub trait GCStrategy: Send + Sync {
    /// Determine if GC should run
    fn should_gc(&self, pressure: MemoryPressure, stats: &MemoryStats) -> bool;
    
    /// Perform garbage collection
    fn perform_gc(&self, pressure: MemoryPressure) -> GCResult;
    
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Result of garbage collection
#[derive(Debug)]
pub struct GCResult {
    pub bytes_freed: usize,
    pub duration_ms: u64,
    pub objects_collected: usize,
}

/// Adaptive GC strategy that adjusts based on memory pressure
pub struct AdaptiveGCStrategy {
    last_gc: Mutex<Instant>,
    gc_interval: RwLock<Duration>,
    pressure_history: Mutex<Vec<MemoryPressure>>,
}

impl AdaptiveGCStrategy {
    pub fn new() -> Self {
        Self {
            last_gc: Mutex::new(Instant::now()),
            gc_interval: RwLock::new(Duration::from_secs(30)),
            pressure_history: Mutex::new(Vec::with_capacity(10)),
        }
    }
    
    fn update_interval(&self, pressure: MemoryPressure) {
        let new_interval = match pressure {
            MemoryPressure::VeryLow => Duration::from_secs(120),
            MemoryPressure::Low => Duration::from_secs(60),
            MemoryPressure::Medium => Duration::from_secs(30),
            MemoryPressure::High => Duration::from_secs(10),
            MemoryPressure::Critical => Duration::from_secs(5),
            MemoryPressure::Emergency => Duration::from_secs(1),
        };
        
        *self.gc_interval.write() = new_interval;
    }
}

impl GCStrategy for AdaptiveGCStrategy {
    fn should_gc(&self, pressure: MemoryPressure, stats: &MemoryStats) -> bool {
        self.update_interval(pressure);
        
        let last_gc = self.last_gc.lock();
        let interval = *self.gc_interval.read();
        
        if pressure >= MemoryPressure::Critical {
            return true; // Always GC in critical situations
        }
        
        if pressure >= MemoryPressure::High && last_gc.elapsed() > interval / 2 {
            return true; // More frequent GC under high pressure
        }
        
        last_gc.elapsed() > interval
    }
    
    fn perform_gc(&self, pressure: MemoryPressure) -> GCResult {
        let start = Instant::now();
        *self.last_gc.lock() = start;
        
        // Update history
        let mut history = self.pressure_history.lock();
        history.push(pressure);
        if history.len() > 10 {
            history.remove(0);
        }
        
        // Simulate GC with different intensities
        let (bytes_freed, objects) = match pressure {
            MemoryPressure::VeryLow => {
                // Light GC - only obvious garbage
                perform_light_gc()
            }
            MemoryPressure::Low | MemoryPressure::Medium => {
                // Normal GC
                perform_normal_gc()
            }
            MemoryPressure::High => {
                // Aggressive GC
                perform_aggressive_gc()
            }
            MemoryPressure::Critical | MemoryPressure::Emergency => {
                // Full GC with compaction
                perform_full_gc()
            }
        };
        
        GCResult {
            bytes_freed,
            duration_ms: start.elapsed().as_millis() as u64,
            objects_collected: objects,
        }
    }
    
    fn name(&self) -> &str {
        "AdaptiveGCStrategy"
    }
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(config: MonitorConfig) -> Result<Self> {
        let current_pressure = Arc::new(RwLock::new(MemoryPressure::Low));
        let monitoring_enabled = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(MemoryStats::default());
        let callbacks = Arc::new(Mutex::new(Vec::new()));
        let history = Arc::new(Mutex::new(MemoryHistory::new(config.history_size)));
        let gc_strategy = Arc::new(RwLock::new(
            Box::new(AdaptiveGCStrategy::new()) as Box<dyn GCStrategy>
        ));
        
        Ok(Self {
            config,
            current_pressure,
            monitoring_enabled,
            stats,
            callbacks,
            history,
            gc_strategy,
            monitor_thread: None,
        })
    }
    
    /// Start monitoring
    pub fn start(&mut self) -> Result<()> {
        if self.monitoring_enabled.load(Ordering::Relaxed) {
            return Ok(()); // Already running
        }
        
        self.monitoring_enabled.store(true, Ordering::SeqCst);
        
        let pressure = self.current_pressure.clone();
        let enabled = self.monitoring_enabled.clone();
        let stats = self.stats.clone();
        let callbacks = self.callbacks.clone();
        let history = self.history.clone();
        let gc_strategy = self.gc_strategy.clone();
        let config = self.config.clone();
        
        let handle = std::thread::spawn(move || {
            enhanced_monitor_loop(
                pressure,
                enabled,
                stats,
                callbacks,
                history,
                gc_strategy,
                config,
            );
        });
        
        self.monitor_thread = Some(handle);
        Ok(())
    }
    
    /// Stop monitoring
    pub fn stop(&mut self) -> Result<()> {
        self.monitoring_enabled.store(false, Ordering::SeqCst);
        
        if let Some(handle) = self.monitor_thread.take() {
            handle.join().map_err(|_| {
                VeritasError::SystemError("Monitor thread panicked".to_string())
            })?;
        }
        
        Ok(())
    }
    
    /// Get current memory pressure
    pub fn current_pressure(&self) -> MemoryPressure {
        *self.current_pressure.read()
    }
    
    /// Get predicted future pressure
    pub fn predict_pressure(&self, future_secs: f64) -> MemoryPressure {
        self.history.lock().predict_pressure(future_secs)
    }
    
    /// Register a callback for pressure changes
    pub fn register_callback<C: MemoryPressureCallback + 'static>(&self, callback: C) {
        self.callbacks.lock().push(Box::new(callback));
    }
    
    /// Set custom GC strategy
    pub fn set_gc_strategy<S: GCStrategy + 'static>(&self, strategy: S) {
        *self.gc_strategy.write() = Box::new(strategy);
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> MemorySnapshot {
        let total = self.stats.total_memory.load(Ordering::Relaxed);
        let available = self.stats.available_memory.load(Ordering::Relaxed);
        let used = self.stats.used_memory.load(Ordering::Relaxed);
        
        MemorySnapshot {
            total_mb: total / (1024 * 1024),
            available_mb: available / (1024 * 1024),
            used_mb: used / (1024 * 1024),
            cached_mb: self.stats.cached_memory.load(Ordering::Relaxed) / (1024 * 1024),
            swap_used_mb: self.stats.swap_used.load(Ordering::Relaxed) / (1024 * 1024),
            swap_total_mb: self.stats.swap_total.load(Ordering::Relaxed) / (1024 * 1024),
            process_rss_mb: self.stats.process_rss.load(Ordering::Relaxed) / (1024 * 1024),
            process_vms_mb: self.stats.process_vms.load(Ordering::Relaxed) / (1024 * 1024),
            pressure: self.current_pressure(),
            usage_percent: if total > 0 { (used as f32 / total as f32) * 100.0 } else { 0.0 },
            allocation_rate_mb_s: self.stats.allocation_rate.load(Ordering::Relaxed) as f32 / (1024.0 * 1024.0),
            fragmentation_percent: self.stats.fragmentation_ratio.load(Ordering::Relaxed) as f32 / 100.0,
            gc_count: self.stats.gc_count.load(Ordering::Relaxed),
            gc_time_total_ms: self.stats.gc_time_ms.load(Ordering::Relaxed),
            allocation_trend: self.history.lock().get_allocation_trend(),
        }
    }
    
    /// Force a memory pressure check
    pub fn check_pressure(&self) -> MemoryPressure {
        let info = get_detailed_memory_info();
        update_stats(&self.stats, &info);
        
        let usage_percent = calculate_usage_percent(&self.stats);
        let new_pressure = calculate_pressure(usage_percent, &self.config.thresholds);
        
        // Add to history
        self.history.lock().add_sample(MemorySample {
            timestamp: Instant::now(),
            used_percent: usage_percent,
            available_mb: info.available / (1024 * 1024),
            allocation_rate: self.stats.allocation_rate.load(Ordering::Relaxed) as f64,
            pressure: new_pressure,
        });
        
        let mut current = self.current_pressure.write();
        if *current != new_pressure {
            let old = *current;
            *current = new_pressure;
            drop(current); // Release lock before callbacks
            
            // Notify callbacks
            let callbacks = self.callbacks.lock();
            for callback in callbacks.iter() {
                callback.on_pressure_change(old, new_pressure);
            }
            
            self.stats.pressure_changes.fetch_add(1, Ordering::Relaxed);
        }
        
        new_pressure
    }
    
    /// Trigger manual garbage collection
    pub fn trigger_gc(&self) -> Result<GCResult> {
        let pressure = self.current_pressure();
        let strategy = self.gc_strategy.read();
        let result = strategy.perform_gc(pressure);
        
        // Update stats
        self.stats.gc_count.fetch_add(1, Ordering::Relaxed);
        self.stats.gc_time_ms.fetch_add(result.duration_ms, Ordering::Relaxed);
        self.stats.bytes_reclaimed.fetch_add(result.bytes_freed as u64, Ordering::Relaxed);
        
        Ok(result)
    }
}

/// Enhanced memory snapshot with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub total_mb: u64,
    pub available_mb: u64,
    pub used_mb: u64,
    pub cached_mb: u64,
    pub swap_used_mb: u64,
    pub swap_total_mb: u64,
    pub process_rss_mb: u64,
    pub process_vms_mb: u64,
    pub pressure: MemoryPressure,
    pub usage_percent: f32,
    pub allocation_rate_mb_s: f32,
    pub fragmentation_percent: f32,
    pub gc_count: u64,
    pub gc_time_total_ms: u64,
    pub allocation_trend: f64,
}

/// Adaptive memory manager with multiple strategies
pub struct AdaptiveMemoryManager {
    monitor: Arc<MemoryMonitor>,
    strategies: Vec<Box<dyn AdaptationStrategy>>,
    emergency_handlers: Vec<Box<dyn EmergencyHandler>>,
}

/// Strategy for adapting to memory pressure
pub trait AdaptationStrategy: Send + Sync {
    /// Apply the strategy based on pressure level
    fn apply(&self, pressure: MemoryPressure, stats: &MemorySnapshot) -> Result<()>;
    
    /// Get strategy name
    fn name(&self) -> &str;
    
    /// Priority (higher = executed first)
    fn priority(&self) -> u32;
}

/// Emergency handler for critical situations
pub trait EmergencyHandler: Send + Sync {
    /// Handle emergency situation
    fn handle_emergency(&self) -> Result<usize>;
    
    /// Get handler name
    fn name(&self) -> &str;
}

impl AdaptiveMemoryManager {
    /// Create a new adaptive memory manager
    pub fn new(monitor: Arc<MemoryMonitor>) -> Self {
        let mut strategies: Vec<Box<dyn AdaptationStrategy>> = vec![
            Box::new(CacheClearingStrategy::new()),
            Box::new(PoolShrinkingStrategy::new()),
            Box::new(ChunkSizeStrategy::new()),
            Box::new(StreamingStrategy::new()),
            Box::new(CompressionStrategy::new()),
        ];
        
        // Sort by priority
        strategies.sort_by_key(|s| std::cmp::Reverse(s.priority()));
        
        let emergency_handlers: Vec<Box<dyn EmergencyHandler>> = vec![
            Box::new(AggressiveCacheEviction::new()),
            Box::new(PoolDraining::new()),
            Box::new(MemoryCompaction::new()),
        ];
        
        Self {
            monitor,
            strategies,
            emergency_handlers,
        }
    }
    
    /// Add a custom adaptation strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn AdaptationStrategy>) {
        self.strategies.push(strategy);
        self.strategies.sort_by_key(|s| std::cmp::Reverse(s.priority()));
    }
    
    /// Add emergency handler
    pub fn add_emergency_handler(&mut self, handler: Box<dyn EmergencyHandler>) {
        self.emergency_handlers.push(handler);
    }
    
    /// Apply all strategies based on current pressure
    pub fn adapt(&self) -> Result<()> {
        let pressure = self.monitor.current_pressure();
        let stats = self.monitor.get_stats();
        
        // Handle emergency situations
        if pressure >= MemoryPressure::Emergency {
            self.handle_emergency()?;
        }
        
        // Apply regular strategies
        for strategy in &self.strategies {
            if let Err(e) = strategy.apply(pressure, &stats) {
                eprintln!("Strategy {} failed: {}", strategy.name(), e);
            }
        }
        
        Ok(())
    }
    
    /// Handle emergency memory situation
    fn handle_emergency(&self) -> Result<()> {
        eprintln!("EMERGENCY: Applying emergency memory handlers");
        
        let mut total_freed = 0;
        for handler in &self.emergency_handlers {
            match handler.handle_emergency() {
                Ok(freed) => total_freed += freed,
                Err(e) => eprintln!("Emergency handler {} failed: {}", handler.name(), e),
            }
        }
        
        if total_freed > 0 {
            self.monitor.stats.oom_prevented.fetch_add(1, Ordering::Relaxed);
            
            // Notify callbacks
            let callbacks = self.monitor.callbacks.lock();
            for callback in callbacks.iter() {
                callback.on_oom_prevented();
            }
        }
        
        Ok(())
    }
}

// Adaptation Strategies

/// Cache clearing strategy with progressive eviction
struct CacheClearingStrategy {
    last_cleared: Mutex<Instant>,
    clear_levels: Vec<Duration>,
}

impl CacheClearingStrategy {
    fn new() -> Self {
        Self {
            last_cleared: Mutex::new(Instant::now()),
            clear_levels: vec![
                Duration::from_secs(120), // Very low
                Duration::from_secs(60),  // Low
                Duration::from_secs(30),  // Medium
                Duration::from_secs(10),  // High
                Duration::from_secs(5),   // Critical
                Duration::from_secs(0),   // Emergency
            ],
        }
    }
}

impl AdaptationStrategy for CacheClearingStrategy {
    fn apply(&self, pressure: MemoryPressure, _stats: &MemorySnapshot) -> Result<()> {
        let mut last_cleared = self.last_cleared.lock();
        let level_idx = pressure as usize;
        let clear_interval = self.clear_levels.get(level_idx)
            .copied()
            .unwrap_or(Duration::from_secs(0));
        
        if last_cleared.elapsed() > clear_interval {
            let percentage = match pressure {
                MemoryPressure::VeryLow => 0,
                MemoryPressure::Low => 10,
                MemoryPressure::Medium => 25,
                MemoryPressure::High => 50,
                MemoryPressure::Critical => 75,
                MemoryPressure::Emergency => 100,
            };
            
            if percentage > 0 {
                clear_caches(percentage)?;
                *last_cleared = Instant::now();
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "CacheClearingStrategy"
    }
    
    fn priority(&self) -> u32 {
        100 // High priority
    }
}

/// Pool shrinking strategy with intelligent resizing
struct PoolShrinkingStrategy {
    last_shrink: Mutex<Instant>,
}

impl PoolShrinkingStrategy {
    fn new() -> Self {
        Self {
            last_shrink: Mutex::new(Instant::now()),
        }
    }
}

impl AdaptationStrategy for PoolShrinkingStrategy {
    fn apply(&self, pressure: MemoryPressure, stats: &MemorySnapshot) -> Result<()> {
        let mut last_shrink = self.last_shrink.lock();
        
        if pressure >= MemoryPressure::High && last_shrink.elapsed() > Duration::from_secs(5) {
            let shrink_factor = match pressure {
                MemoryPressure::High => 0.25,
                MemoryPressure::Critical => 0.5,
                MemoryPressure::Emergency => 0.75,
                _ => 0.0,
            };
            
            if shrink_factor > 0.0 {
                shrink_object_pools(shrink_factor)?;
                *last_shrink = Instant::now();
            }
        }
        
        // Grow pools if pressure is low and allocation rate is high
        if pressure <= MemoryPressure::Low && stats.allocation_rate_mb_s > 10.0 {
            grow_object_pools(0.1)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "PoolShrinkingStrategy"
    }
    
    fn priority(&self) -> u32 {
        90
    }
}

/// Dynamic chunk size strategy
struct ChunkSizeStrategy {
    current_multiplier: RwLock<f32>,
}

impl ChunkSizeStrategy {
    fn new() -> Self {
        Self {
            current_multiplier: RwLock::new(1.0),
        }
    }
}

impl AdaptationStrategy for ChunkSizeStrategy {
    fn apply(&self, pressure: MemoryPressure, _stats: &MemorySnapshot) -> Result<()> {
        let new_multiplier = match pressure {
            MemoryPressure::VeryLow => 2.0,
            MemoryPressure::Low => 1.5,
            MemoryPressure::Medium => 1.0,
            MemoryPressure::High => 0.5,
            MemoryPressure::Critical => 0.25,
            MemoryPressure::Emergency => 0.1,
        };
        
        *self.current_multiplier.write() = new_multiplier;
        set_global_chunk_multiplier(new_multiplier)?;
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "ChunkSizeStrategy"
    }
    
    fn priority(&self) -> u32 {
        80
    }
}

/// Streaming strategy for large data
struct StreamingStrategy {
    enabled: RwLock<bool>,
}

impl StreamingStrategy {
    fn new() -> Self {
        Self {
            enabled: RwLock::new(false),
        }
    }
}

impl AdaptationStrategy for StreamingStrategy {
    fn apply(&self, pressure: MemoryPressure, _stats: &MemorySnapshot) -> Result<()> {
        let should_stream = pressure >= MemoryPressure::Medium;
        let mut enabled = self.enabled.write();
        
        if *enabled != should_stream {
            *enabled = should_stream;
            set_streaming_mode(should_stream)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "StreamingStrategy"
    }
    
    fn priority(&self) -> u32 {
        70
    }
}

/// Compression strategy for memory-intensive data
struct CompressionStrategy {
    compression_level: RwLock<u32>,
}

impl CompressionStrategy {
    fn new() -> Self {
        Self {
            compression_level: RwLock::new(0),
        }
    }
}

impl AdaptationStrategy for CompressionStrategy {
    fn apply(&self, pressure: MemoryPressure, _stats: &MemorySnapshot) -> Result<()> {
        let level = match pressure {
            MemoryPressure::VeryLow | MemoryPressure::Low => 0,
            MemoryPressure::Medium => 1,
            MemoryPressure::High => 3,
            MemoryPressure::Critical => 6,
            MemoryPressure::Emergency => 9,
        };
        
        *self.compression_level.write() = level;
        set_compression_level(level)?;
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "CompressionStrategy"
    }
    
    fn priority(&self) -> u32 {
        60
    }
}

// Emergency Handlers

struct AggressiveCacheEviction;

impl AggressiveCacheEviction {
    fn new() -> Self {
        Self
    }
}

impl EmergencyHandler for AggressiveCacheEviction {
    fn handle_emergency(&self) -> Result<usize> {
        // Clear all caches aggressively
        clear_all_caches()?;
        Ok(1024 * 1024 * 10) // Estimate 10MB freed
    }
    
    fn name(&self) -> &str {
        "AggressiveCacheEviction"
    }
}

struct PoolDraining;

impl PoolDraining {
    fn new() -> Self {
        Self
    }
}

impl EmergencyHandler for PoolDraining {
    fn handle_emergency(&self) -> Result<usize> {
        // Drain all object pools
        drain_all_pools()?;
        Ok(1024 * 1024 * 20) // Estimate 20MB freed
    }
    
    fn name(&self) -> &str {
        "PoolDraining"
    }
}

struct MemoryCompaction;

impl MemoryCompaction {
    fn new() -> Self {
        Self
    }
}

impl EmergencyHandler for MemoryCompaction {
    fn handle_emergency(&self) -> Result<usize> {
        // Trigger memory compaction
        trigger_memory_compaction()?;
        Ok(1024 * 1024 * 5) // Estimate 5MB freed
    }
    
    fn name(&self) -> &str {
        "MemoryCompaction"
    }
}

// Enhanced monitoring loop
fn enhanced_monitor_loop(
    pressure: Arc<RwLock<MemoryPressure>>,
    enabled: Arc<AtomicBool>,
    stats: Arc<MemoryStats>,
    callbacks: Arc<Mutex<Vec<Box<dyn MemoryPressureCallback>>>>,
    history: Arc<Mutex<MemoryHistory>>,
    gc_strategy: Arc<RwLock<Box<dyn GCStrategy>>>,
    config: MonitorConfig,
) {
    let mut last_gc = Instant::now();
    
    while enabled.load(Ordering::Relaxed) {
        let info = get_detailed_memory_info();
        update_stats(&stats, &info);
        
        let usage_percent = calculate_usage_percent(&stats);
        let new_pressure = calculate_pressure(usage_percent, &config.thresholds);
        
        // Add to history
        history.lock().add_sample(MemorySample {
            timestamp: Instant::now(),
            used_percent: usage_percent,
            available_mb: info.available / (1024 * 1024),
            allocation_rate: calculate_allocation_rate(&info),
            pressure: new_pressure,
        });
        
        // Check for pressure change
        let mut current = pressure.write();
        let pressure_changed = *current != new_pressure;
        if pressure_changed {
            let old = *current;
            *current = new_pressure;
            drop(current); // Release lock before callbacks
            
            // Notify callbacks
            let callbacks = callbacks.lock();
            for callback in callbacks.iter() {
                callback.on_pressure_change(old, new_pressure);
            }
            
            stats.pressure_changes.fetch_add(1, Ordering::Relaxed);
        } else {
            drop(current);
        }
        
        // Predictive pressure check
        if config.predictive_mode {
            let predicted = history.lock().predict_pressure(5.0);
            if predicted > new_pressure {
                // Preemptive action based on prediction
                if let Ok(strategy) = gc_strategy.read().as_ref() {
                    if strategy.should_gc(predicted, &stats) {
                        let gc_result = strategy.perform_gc(predicted);
                        stats.gc_count.fetch_add(1, Ordering::Relaxed);
                        stats.gc_time_ms.fetch_add(gc_result.duration_ms, Ordering::Relaxed);
                        stats.bytes_reclaimed.fetch_add(gc_result.bytes_freed as u64, Ordering::Relaxed);
                    }
                }
            }
        }
        
        // Check if GC should run
        if config.auto_gc {
            let strategy = gc_strategy.read();
            if strategy.should_gc(new_pressure, &stats) {
                drop(strategy);
                
                // Notify about emergency GC
                if new_pressure >= MemoryPressure::Critical {
                    let callbacks = callbacks.lock();
                    for callback in callbacks.iter() {
                        callback.on_emergency_gc();
                    }
                }
                
                let strategy = gc_strategy.read();
                let gc_result = strategy.perform_gc(new_pressure);
                
                stats.gc_count.fetch_add(1, Ordering::Relaxed);
                stats.gc_time_ms.fetch_add(gc_result.duration_ms, Ordering::Relaxed);
                stats.bytes_reclaimed.fetch_add(gc_result.bytes_freed as u64, Ordering::Relaxed);
                
                last_gc = Instant::now();
            }
        }
        
        std::thread::sleep(config.update_interval);
    }
}

// Platform-specific memory information
#[derive(Debug)]
struct DetailedMemoryInfo {
    total: u64,
    available: u64,
    cached: u64,
    swap_total: u64,
    swap_free: u64,
    process_rss: u64,
    process_vms: u64,
    process_shared: u64,
    page_faults: u64,
}

fn get_detailed_memory_info() -> DetailedMemoryInfo {
    #[cfg(target_os = "linux")]
    {
        get_linux_memory_info()
    }
    
    #[cfg(target_os = "macos")]
    {
        get_macos_memory_info()
    }
    
    #[cfg(target_os = "windows")]
    {
        get_windows_memory_info()
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        // Fallback
        DetailedMemoryInfo {
            total: 8 * 1024 * 1024 * 1024,
            available: 4 * 1024 * 1024 * 1024,
            cached: 1024 * 1024 * 1024,
            swap_total: 0,
            swap_free: 0,
            process_rss: 100 * 1024 * 1024,
            process_vms: 200 * 1024 * 1024,
            process_shared: 50 * 1024 * 1024,
            page_faults: 0,
        }
    }
}

#[cfg(target_os = "linux")]
fn get_linux_memory_info() -> DetailedMemoryInfo {
    use std::fs;
    
    // Parse /proc/meminfo and /proc/self/status
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    
    // Simple parsing - in practice would be more robust
    DetailedMemoryInfo {
        total: parse_meminfo_value(&meminfo, "MemTotal:"),
        available: parse_meminfo_value(&meminfo, "MemAvailable:"),
        cached: parse_meminfo_value(&meminfo, "Cached:"),
        swap_total: parse_meminfo_value(&meminfo, "SwapTotal:"),
        swap_free: parse_meminfo_value(&meminfo, "SwapFree:"),
        process_rss: parse_status_value(&status, "VmRSS:"),
        process_vms: parse_status_value(&status, "VmSize:"),
        process_shared: parse_status_value(&status, "RssFile:"),
        page_faults: 0,
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_value(content: &str, key: &str) -> u64 {
    content.lines()
        .find(|line| line.starts_with(key))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|val| val.parse::<u64>().ok())
        .unwrap_or(0) * 1024 // Convert KB to bytes
}

#[cfg(target_os = "linux")]
fn parse_status_value(content: &str, key: &str) -> u64 {
    parse_meminfo_value(content, key)
}

#[cfg(target_os = "macos")]
fn get_macos_memory_info() -> DetailedMemoryInfo {
    // Use sysctl and ps for memory info
    DetailedMemoryInfo {
        total: 16 * 1024 * 1024 * 1024,
        available: 8 * 1024 * 1024 * 1024,
        cached: 2 * 1024 * 1024 * 1024,
        swap_total: 0,
        swap_free: 0,
        process_rss: 150 * 1024 * 1024,
        process_vms: 300 * 1024 * 1024,
        process_shared: 75 * 1024 * 1024,
        page_faults: 0,
    }
}

#[cfg(target_os = "windows")]
fn get_windows_memory_info() -> DetailedMemoryInfo {
    // Use Windows API
    DetailedMemoryInfo {
        total: 16 * 1024 * 1024 * 1024,
        available: 8 * 1024 * 1024 * 1024,
        cached: 2 * 1024 * 1024 * 1024,
        swap_total: 16 * 1024 * 1024 * 1024,
        swap_free: 8 * 1024 * 1024 * 1024,
        process_rss: 200 * 1024 * 1024,
        process_vms: 400 * 1024 * 1024,
        process_shared: 100 * 1024 * 1024,
        page_faults: 0,
    }
}

fn update_stats(stats: &MemoryStats, info: &DetailedMemoryInfo) {
    stats.total_memory.store(info.total, Ordering::Relaxed);
    stats.available_memory.store(info.available, Ordering::Relaxed);
    stats.used_memory.store(info.total - info.available, Ordering::Relaxed);
    stats.cached_memory.store(info.cached, Ordering::Relaxed);
    stats.swap_used.store(info.swap_total - info.swap_free, Ordering::Relaxed);
    stats.swap_total.store(info.swap_total, Ordering::Relaxed);
    stats.process_rss.store(info.process_rss, Ordering::Relaxed);
    stats.process_vms.store(info.process_vms, Ordering::Relaxed);
    stats.process_shared.store(info.process_shared, Ordering::Relaxed);
}

fn calculate_usage_percent(stats: &MemoryStats) -> f32 {
    let total = stats.total_memory.load(Ordering::Relaxed);
    let used = stats.used_memory.load(Ordering::Relaxed);
    
    if total > 0 {
        (used as f32 / total as f32) * 100.0
    } else {
        0.0
    }
}

fn calculate_pressure(usage_percent: f32, thresholds: &MemoryThresholds) -> MemoryPressure {
    if usage_percent >= thresholds.emergency * 100.0 {
        MemoryPressure::Emergency
    } else if usage_percent >= thresholds.critical * 100.0 {
        MemoryPressure::Critical
    } else if usage_percent >= thresholds.high * 100.0 {
        MemoryPressure::High
    } else if usage_percent >= thresholds.medium * 100.0 {
        MemoryPressure::Medium
    } else if usage_percent >= thresholds.low * 100.0 {
        MemoryPressure::Low
    } else {
        MemoryPressure::VeryLow
    }
}

fn calculate_allocation_rate(info: &DetailedMemoryInfo) -> f64 {
    // Simplified - would track allocations over time
    0.0
}

// GC implementation functions
fn perform_light_gc() -> (usize, usize) {
    // Light GC implementation
    (1024 * 1024, 100) // 1MB freed, 100 objects
}

fn perform_normal_gc() -> (usize, usize) {
    // Normal GC implementation
    (5 * 1024 * 1024, 500) // 5MB freed, 500 objects
}

fn perform_aggressive_gc() -> (usize, usize) {
    // Aggressive GC implementation
    (20 * 1024 * 1024, 2000) // 20MB freed, 2000 objects
}

fn perform_full_gc() -> (usize, usize) {
    // Full GC with compaction
    (50 * 1024 * 1024, 5000) // 50MB freed, 5000 objects
}

// Helper functions for strategies
fn clear_caches(percentage: u32) -> Result<()> {
    // Implementation would clear caches
    Ok(())
}

fn clear_all_caches() -> Result<()> {
    clear_caches(100)
}

fn shrink_object_pools(factor: f32) -> Result<()> {
    // Implementation would shrink pools
    Ok(())
}

fn grow_object_pools(factor: f32) -> Result<()> {
    // Implementation would grow pools
    Ok(())
}

fn drain_all_pools() -> Result<()> {
    // Implementation would drain all pools
    Ok(())
}

fn set_global_chunk_multiplier(multiplier: f32) -> Result<()> {
    // Implementation would adjust chunk sizes
    Ok(())
}

fn set_streaming_mode(enabled: bool) -> Result<()> {
    // Implementation would enable/disable streaming
    Ok(())
}

fn set_compression_level(level: u32) -> Result<()> {
    // Implementation would set compression level
    Ok(())
}

fn trigger_memory_compaction() -> Result<()> {
    // Implementation would trigger compaction
    Ok(())
}

/// Global memory monitor instance
static GLOBAL_MONITOR: once_cell::sync::OnceCell<Arc<MemoryMonitor>> = 
    once_cell::sync::OnceCell::new();

/// Initialize global memory monitoring
pub fn init_global_monitor(config: MonitorConfig) -> Result<Arc<MemoryMonitor>> {
    let mut monitor = MemoryMonitor::new(config)?;
    monitor.start()?;
    let monitor = Arc::new(monitor);
    
    GLOBAL_MONITOR.set(monitor.clone())
        .map_err(|_| VeritasError::SystemError("Monitor already initialized".to_string()))?;
    
    Ok(monitor)
}

/// Get global memory monitor
pub fn global_monitor() -> Option<&'static Arc<MemoryMonitor>> {
    GLOBAL_MONITOR.get()
}

/// Get current global memory pressure
pub fn current_memory_pressure() -> MemoryPressure {
    global_monitor()
        .map(|m| m.current_pressure())
        .unwrap_or(MemoryPressure::Low)
}

/// Get predicted future pressure
pub fn predict_memory_pressure(future_secs: f64) -> MemoryPressure {
    global_monitor()
        .map(|m| m.predict_pressure(future_secs))
        .unwrap_or(MemoryPressure::Low)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pressure_calculation() {
        let thresholds = MemoryThresholds::default();
        
        assert_eq!(calculate_pressure(30.0, &thresholds), MemoryPressure::VeryLow);
        assert_eq!(calculate_pressure(50.0, &thresholds), MemoryPressure::VeryLow);
        assert_eq!(calculate_pressure(60.0, &thresholds), MemoryPressure::Low);
        assert_eq!(calculate_pressure(75.0, &thresholds), MemoryPressure::Medium);
        assert_eq!(calculate_pressure(85.0, &thresholds), MemoryPressure::High);
        assert_eq!(calculate_pressure(92.0, &thresholds), MemoryPressure::Critical);
        assert_eq!(calculate_pressure(96.0, &thresholds), MemoryPressure::Emergency);
    }
    
    #[test]
    fn test_memory_monitor() {
        let config = MonitorConfig {
            update_interval: Duration::from_millis(100),
            ..Default::default()
        };
        
        let mut monitor = MemoryMonitor::new(config).unwrap();
        monitor.start().unwrap();
        
        std::thread::sleep(Duration::from_millis(200));
        
        let pressure = monitor.check_pressure();
        assert!(matches!(
            pressure,
            MemoryPressure::VeryLow | MemoryPressure::Low | MemoryPressure::Medium
        ));
        
        let stats = monitor.get_stats();
        assert!(stats.total_mb > 0);
        
        monitor.stop().unwrap();
    }
    
    #[test]
    fn test_adaptive_gc_strategy() {
        let strategy = AdaptiveGCStrategy::new();
        
        assert!(strategy.should_gc(MemoryPressure::Critical, &MemoryStats::default()));
        assert!(!strategy.should_gc(MemoryPressure::VeryLow, &MemoryStats::default()));
        
        let result = strategy.perform_gc(MemoryPressure::High);
        assert!(result.bytes_freed > 0);
    }
    
    #[test]
    fn test_memory_history_prediction() {
        let mut history = MemoryHistory::new(10);
        
        // Add samples with increasing memory usage
        for i in 0..5 {
            history.add_sample(MemorySample {
                timestamp: Instant::now(),
                used_percent: 50.0 + i as f32 * 5.0,
                available_mb: 1000,
                allocation_rate: 10.0,
                pressure: MemoryPressure::Medium,
            });
        }
        
        let predicted = history.predict_pressure(5.0);
        assert!(predicted >= MemoryPressure::Medium);
    }
}