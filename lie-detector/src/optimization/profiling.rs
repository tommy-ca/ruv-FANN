//! CPU profiling and performance monitoring module.
//!
//! This module provides tools for identifying CPU hotspots and performance bottlenecks
//! in the lie detection system, particularly for SIMD optimization targets.

use crate::{Result, VeritasError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::fmt;

/// Performance metrics collected during profiling.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Function call statistics
    pub function_stats: HashMap<String, FunctionStats>,
    /// SIMD usage statistics
    pub simd_stats: SimdUsageStats,
    /// Cache performance metrics
    pub cache_stats: CacheStats,
    /// Total profiling duration
    pub total_duration: Duration,
    /// Number of samples collected
    pub sample_count: usize,
}

/// Statistics for a single function.
#[derive(Debug, Clone)]
pub struct FunctionStats {
    /// Function name
    pub name: String,
    /// Total time spent in function
    pub total_time: Duration,
    /// Number of calls
    pub call_count: usize,
    /// Average time per call
    pub avg_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Percentage of total execution time
    pub percentage: f64,
}

/// SIMD usage statistics.
#[derive(Debug, Clone)]
pub struct SimdUsageStats {
    /// Number of vectorized operations
    pub vectorized_ops: usize,
    /// Number of scalar fallback operations
    pub scalar_fallback_ops: usize,
    /// Average vector width utilization (0.0 to 1.0)
    pub vector_utilization: f64,
    /// Most used SIMD instruction set
    pub preferred_instruction_set: String,
}

/// Cache performance statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Estimated L1 cache hit rate
    pub l1_hit_rate: f64,
    /// Estimated L2 cache hit rate
    pub l2_hit_rate: f64,
    /// Number of cache-aligned accesses
    pub aligned_accesses: usize,
    /// Number of unaligned accesses
    pub unaligned_accesses: usize,
    /// Memory bandwidth utilization (bytes/sec)
    pub memory_bandwidth: f64,
}

/// CPU profiler for identifying performance hotspots.
pub struct Profiler {
    /// Internal state for collecting metrics
    state: Arc<Mutex<ProfilerState>>,
    /// Whether profiling is currently active
    active: bool,
}

/// Internal profiler state.
struct ProfilerState {
    function_timings: HashMap<String, Vec<Duration>>,
    simd_operations: SimdOperationCounter,
    cache_operations: CacheOperationCounter,
    start_time: Instant,
}

/// Counter for SIMD operations.
#[derive(Default)]
struct SimdOperationCounter {
    vectorized: usize,
    scalar: usize,
    instruction_sets: HashMap<String, usize>,
}

/// Counter for cache operations.
#[derive(Default)]
struct CacheOperationCounter {
    aligned: usize,
    unaligned: usize,
    bytes_transferred: usize,
}

/// Timer guard for automatic timing of function calls.
pub struct TimerGuard<'a> {
    profiler: &'a mut Profiler,
    function_name: String,
    start_time: Instant,
}

impl<'a> Drop for TimerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        self.profiler.record_timing(&self.function_name, duration);
    }
}

impl Profiler {
    /// Create a new profiler.
    pub fn new() -> Result<Self> {
        Ok(Self {
            state: Arc::new(Mutex::new(ProfilerState {
                function_timings: HashMap::new(),
                simd_operations: SimdOperationCounter::default(),
                cache_operations: CacheOperationCounter::default(),
                start_time: Instant::now(),
            })),
            active: true,
        })
    }
    
    /// Start timing a function call.
    pub fn start_timer(&mut self, function_name: &str) -> TimerGuard {
        TimerGuard {
            profiler: self,
            function_name: function_name.to_string(),
            start_time: Instant::now(),
        }
    }
    
    /// Record a function timing manually.
    pub fn record_timing(&mut self, function_name: &str, duration: Duration) {
        if !self.active {
            return;
        }
        
        if let Ok(mut state) = self.state.lock() {
            state.function_timings
                .entry(function_name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }
    
    /// Record a SIMD operation.
    pub fn record_simd_operation(&mut self, instruction_set: &str, vectorized: bool) {
        if !self.active {
            return;
        }
        
        if let Ok(mut state) = self.state.lock() {
            if vectorized {
                state.simd_operations.vectorized += 1;
            } else {
                state.simd_operations.scalar += 1;
            }
            
            *state.simd_operations
                .instruction_sets
                .entry(instruction_set.to_string())
                .or_insert(0) += 1;
        }
    }
    
    /// Record a cache operation.
    pub fn record_cache_operation(&mut self, aligned: bool, bytes: usize) {
        if !self.active {
            return;
        }
        
        if let Ok(mut state) = self.state.lock() {
            if aligned {
                state.cache_operations.aligned += 1;
            } else {
                state.cache_operations.unaligned += 1;
            }
            state.cache_operations.bytes_transferred += bytes;
        }
    }
    
    /// Get collected performance metrics.
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let state = self.state.lock().unwrap();
        let total_duration = state.start_time.elapsed();
        
        // Calculate function statistics
        let mut function_stats = HashMap::new();
        let mut total_function_time = Duration::from_secs(0);
        
        for (name, timings) in &state.function_timings {
            if timings.is_empty() {
                continue;
            }
            
            let total_time: Duration = timings.iter().sum();
            let avg_time = total_time / timings.len() as u32;
            let min_time = *timings.iter().min().unwrap();
            let max_time = *timings.iter().max().unwrap();
            
            total_function_time += total_time;
            
            function_stats.insert(
                name.clone(),
                FunctionStats {
                    name: name.clone(),
                    total_time,
                    call_count: timings.len(),
                    avg_time,
                    min_time,
                    max_time,
                    percentage: 0.0, // Will be calculated later
                },
            );
        }
        
        // Calculate percentages
        let total_ns = total_function_time.as_nanos() as f64;
        for stats in function_stats.values_mut() {
            stats.percentage = if total_ns > 0.0 {
                (stats.total_time.as_nanos() as f64 / total_ns) * 100.0
            } else {
                0.0
            };
        }
        
        // Calculate SIMD statistics
        let total_simd_ops = state.simd_operations.vectorized + state.simd_operations.scalar;
        let vector_utilization = if total_simd_ops > 0 {
            state.simd_operations.vectorized as f64 / total_simd_ops as f64
        } else {
            0.0
        };
        
        let preferred_instruction_set = state.simd_operations
            .instruction_sets
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "none".to_string());
        
        let simd_stats = SimdUsageStats {
            vectorized_ops: state.simd_operations.vectorized,
            scalar_fallback_ops: state.simd_operations.scalar,
            vector_utilization,
            preferred_instruction_set,
        };
        
        // Calculate cache statistics
        let total_cache_ops = state.cache_operations.aligned + state.cache_operations.unaligned;
        let cache_stats = CacheStats {
            l1_hit_rate: if total_cache_ops > 0 {
                state.cache_operations.aligned as f64 / total_cache_ops as f64
            } else {
                0.0
            },
            l2_hit_rate: 0.85, // Estimate
            aligned_accesses: state.cache_operations.aligned,
            unaligned_accesses: state.cache_operations.unaligned,
            memory_bandwidth: if total_duration.as_secs() > 0 {
                state.cache_operations.bytes_transferred as f64 / total_duration.as_secs_f64()
            } else {
                0.0
            },
        };
        
        PerformanceMetrics {
            function_stats,
            simd_stats,
            cache_stats,
            total_duration,
            sample_count: state.function_timings.values().map(|v| v.len()).sum(),
        }
    }
    
    /// Reset profiler statistics.
    pub fn reset(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.function_timings.clear();
            state.simd_operations = SimdOperationCounter::default();
            state.cache_operations = CacheOperationCounter::default();
            state.start_time = Instant::now();
        }
    }
    
    /// Enable or disable profiling.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

impl fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n=== Performance Profiling Report ===")?;
        writeln!(f, "Total Duration: {:.3}s", self.total_duration.as_secs_f64())?;
        writeln!(f, "Total Samples: {}", self.sample_count)?;
        
        writeln!(f, "\n--- CPU Hotspots ---")?;
        let mut sorted_functions: Vec<_> = self.function_stats.values().collect();
        sorted_functions.sort_by(|a, b| b.percentage.partial_cmp(&a.percentage).unwrap());
        
        for (i, stats) in sorted_functions.iter().enumerate().take(10) {
            writeln!(
                f,
                "{}. {} ({:.1}%)",
                i + 1,
                stats.name,
                stats.percentage
            )?;
            writeln!(
                f,
                "   Calls: {}, Total: {:.3}ms, Avg: {:.3}μs, Min: {:.3}μs, Max: {:.3}μs",
                stats.call_count,
                stats.total_time.as_secs_f64() * 1000.0,
                stats.avg_time.as_micros(),
                stats.min_time.as_micros(),
                stats.max_time.as_micros()
            )?;
        }
        
        writeln!(f, "\n--- SIMD Utilization ---")?;
        writeln!(f, "Vectorized Operations: {}", self.simd_stats.vectorized_ops)?;
        writeln!(f, "Scalar Fallback Operations: {}", self.simd_stats.scalar_fallback_ops)?;
        writeln!(f, "Vector Utilization: {:.1}%", self.simd_stats.vector_utilization * 100.0)?;
        writeln!(f, "Preferred Instruction Set: {}", self.simd_stats.preferred_instruction_set)?;
        
        writeln!(f, "\n--- Cache Performance ---")?;
        writeln!(f, "L1 Cache Hit Rate: {:.1}%", self.cache_stats.l1_hit_rate * 100.0)?;
        writeln!(f, "L2 Cache Hit Rate: {:.1}%", self.cache_stats.l2_hit_rate * 100.0)?;
        writeln!(f, "Aligned Accesses: {}", self.cache_stats.aligned_accesses)?;
        writeln!(f, "Unaligned Accesses: {}", self.cache_stats.unaligned_accesses)?;
        writeln!(f, "Memory Bandwidth: {:.2} MB/s", self.cache_stats.memory_bandwidth / 1_000_000.0)?;
        
        Ok(())
    }
}

/// Metrics collector for continuous monitoring.
pub struct MetricsCollector {
    profiler: Profiler,
    collection_interval: Duration,
    history: Vec<PerformanceMetrics>,
    max_history_size: usize,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new(collection_interval: Duration) -> Result<Self> {
        Ok(Self {
            profiler: Profiler::new()?,
            collection_interval,
            history: Vec::new(),
            max_history_size: 100,
        })
    }
    
    /// Collect current metrics and add to history.
    pub fn collect(&mut self) {
        let metrics = self.profiler.get_metrics();
        self.history.push(metrics);
        
        // Maintain history size limit
        if self.history.len() > self.max_history_size {
            self.history.remove(0);
        }
        
        self.profiler.reset();
    }
    
    /// Get aggregated metrics over the collection history.
    pub fn get_aggregated_metrics(&self) -> Option<PerformanceMetrics> {
        if self.history.is_empty() {
            return None;
        }
        
        // This is a simplified aggregation
        // In a real implementation, you'd properly aggregate all metrics
        self.history.last().cloned()
    }
    
    /// Get performance trend analysis.
    pub fn get_trend_analysis(&self) -> TrendAnalysis {
        if self.history.len() < 2 {
            return TrendAnalysis::default();
        }
        
        // Analyze trends in key metrics
        let recent = &self.history[self.history.len() - 1];
        let previous = &self.history[self.history.len() - 2];
        
        TrendAnalysis {
            vector_utilization_trend: calculate_trend(
                previous.simd_stats.vector_utilization,
                recent.simd_stats.vector_utilization,
            ),
            cache_hit_rate_trend: calculate_trend(
                previous.cache_stats.l1_hit_rate,
                recent.cache_stats.l1_hit_rate,
            ),
            memory_bandwidth_trend: calculate_trend(
                previous.cache_stats.memory_bandwidth,
                recent.cache_stats.memory_bandwidth,
            ),
        }
    }
}

/// Performance trend analysis.
#[derive(Debug, Clone, Default)]
pub struct TrendAnalysis {
    /// Trend in vector utilization (-1.0 = decreasing, 0.0 = stable, 1.0 = increasing)
    pub vector_utilization_trend: f64,
    /// Trend in cache hit rate
    pub cache_hit_rate_trend: f64,
    /// Trend in memory bandwidth usage
    pub memory_bandwidth_trend: f64,
}

/// Calculate trend between two values.
fn calculate_trend(previous: f64, current: f64) -> f64 {
    if previous == 0.0 {
        return 0.0;
    }
    
    let change = (current - previous) / previous;
    change.max(-1.0).min(1.0)
}

/// CPU hotspot analyzer for identifying optimization targets.
pub struct HotspotAnalyzer {
    metrics: PerformanceMetrics,
}

impl HotspotAnalyzer {
    /// Create a new hotspot analyzer from performance metrics.
    pub fn new(metrics: PerformanceMetrics) -> Self {
        Self { metrics }
    }
    
    /// Identify functions that would benefit most from SIMD optimization.
    pub fn identify_simd_targets(&self) -> Vec<OptimizationTarget> {
        let mut targets = Vec::new();
        
        for (name, stats) in &self.metrics.function_stats {
            // High CPU usage functions with low vectorization
            if stats.percentage > 5.0 && self.metrics.simd_stats.vector_utilization < 0.5 {
                targets.push(OptimizationTarget {
                    function_name: name.clone(),
                    optimization_type: OptimizationType::Simd,
                    expected_speedup: estimate_simd_speedup(stats),
                    priority: calculate_priority(stats),
                });
            }
        }
        
        targets.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        targets
    }
    
    /// Identify functions that would benefit from cache optimization.
    pub fn identify_cache_targets(&self) -> Vec<OptimizationTarget> {
        let mut targets = Vec::new();
        
        for (name, stats) in &self.metrics.function_stats {
            // Functions with poor cache performance
            if stats.percentage > 3.0 && self.metrics.cache_stats.l1_hit_rate < 0.8 {
                targets.push(OptimizationTarget {
                    function_name: name.clone(),
                    optimization_type: OptimizationType::Cache,
                    expected_speedup: estimate_cache_speedup(&self.metrics.cache_stats),
                    priority: calculate_priority(stats),
                });
            }
        }
        
        targets.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        targets
    }
}

/// Optimization target identified by the analyzer.
#[derive(Debug, Clone)]
pub struct OptimizationTarget {
    /// Function name to optimize
    pub function_name: String,
    /// Type of optimization recommended
    pub optimization_type: OptimizationType,
    /// Expected speedup factor (1.0 = no speedup, 2.0 = 2x faster)
    pub expected_speedup: f64,
    /// Priority score (0.0 to 100.0)
    pub priority: f64,
}

/// Type of optimization recommended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    /// SIMD vectorization
    Simd,
    /// Cache optimization
    Cache,
    /// Both SIMD and cache
    Both,
}

/// Estimate potential speedup from SIMD optimization.
fn estimate_simd_speedup(stats: &FunctionStats) -> f64 {
    // Simplified estimation based on function characteristics
    // In reality, this would analyze the actual operations performed
    let base_speedup = 2.0; // Conservative estimate
    
    // Adjust based on function execution time (longer functions benefit more)
    let time_factor = (stats.avg_time.as_micros() as f64 / 1000.0).min(2.0);
    
    base_speedup * time_factor
}

/// Estimate potential speedup from cache optimization.
fn estimate_cache_speedup(cache_stats: &CacheStats) -> f64 {
    // Estimate based on cache miss rate
    let miss_rate = 1.0 - cache_stats.l1_hit_rate;
    1.0 + (miss_rate * 0.5) // Up to 50% improvement from better cache usage
}

/// Calculate optimization priority.
fn calculate_priority(stats: &FunctionStats) -> f64 {
    // Priority based on CPU time percentage and call frequency
    let time_weight = stats.percentage;
    let frequency_weight = (stats.call_count as f64).log10() * 10.0;
    
    (time_weight + frequency_weight).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(profiler.is_ok());
    }
    
    #[test]
    fn test_function_timing() {
        let mut profiler = Profiler::new().unwrap();
        
        // Simulate function timing
        profiler.record_timing("test_function", Duration::from_millis(10));
        profiler.record_timing("test_function", Duration::from_millis(15));
        profiler.record_timing("test_function", Duration::from_millis(12));
        
        let metrics = profiler.get_metrics();
        let stats = metrics.function_stats.get("test_function").unwrap();
        
        assert_eq!(stats.call_count, 3);
        assert_eq!(stats.min_time, Duration::from_millis(10));
        assert_eq!(stats.max_time, Duration::from_millis(15));
    }
    
    #[test]
    fn test_simd_statistics() {
        let mut profiler = Profiler::new().unwrap();
        
        profiler.record_simd_operation("AVX2", true);
        profiler.record_simd_operation("AVX2", true);
        profiler.record_simd_operation("SSE", false);
        
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.simd_stats.vectorized_ops, 2);
        assert_eq!(metrics.simd_stats.scalar_fallback_ops, 1);
        assert!(metrics.simd_stats.vector_utilization > 0.6);
    }
    
    #[test]
    fn test_hotspot_analysis() {
        let mut profiler = Profiler::new().unwrap();
        
        // Simulate a hot function
        for _ in 0..100 {
            profiler.record_timing("hot_function", Duration::from_micros(500));
        }
        
        // Simulate a cold function
        profiler.record_timing("cold_function", Duration::from_micros(10));
        
        let metrics = profiler.get_metrics();
        let analyzer = HotspotAnalyzer::new(metrics);
        let targets = analyzer.identify_simd_targets();
        
        // The hot function should be identified as a target
        assert!(!targets.is_empty());
        assert_eq!(targets[0].function_name, "hot_function");
    }
}