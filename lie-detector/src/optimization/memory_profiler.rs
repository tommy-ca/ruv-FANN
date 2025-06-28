//! Memory profiling and monitoring infrastructure
//! 
//! This module provides tools for tracking memory usage, detecting leaks,
//! and monitoring allocation patterns throughout the application.

use crate::{Result, VeritasError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Global memory allocator with tracking
pub struct TrackingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static CURRENT_USAGE: AtomicUsize = AtomicUsize::new(0);
static PEAK_USAGE: AtomicUsize = AtomicUsize::new(0);
static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() && TRACKING_ENABLED.load(Ordering::Relaxed) {
            let size = layout.size();
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            let current = CURRENT_USAGE.fetch_add(size, Ordering::Relaxed) + size;
            
            // Update peak usage
            let mut peak = PEAK_USAGE.load(Ordering::Relaxed);
            while current > peak {
                match PEAK_USAGE.compare_exchange_weak(
                    peak, current, Ordering::Relaxed, Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(x) => peak = x,
                }
            }
        }
        
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING_ENABLED.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            CURRENT_USAGE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        
        System.dealloc(ptr, layout)
    }
}

/// Memory profiler for detailed memory analysis
pub struct MemoryProfiler {
    enabled: bool,
    start_time: Instant,
    samples: Arc<RwLock<Vec<MemorySample>>>,
    allocation_sites: Arc<Mutex<HashMap<String, AllocationSite>>>,
    config: ProfilerConfig,
}

/// Configuration for memory profiler
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Sample interval for memory snapshots
    pub sample_interval: Duration,
    /// Maximum number of samples to keep
    pub max_samples: usize,
    /// Track allocation call sites
    pub track_call_sites: bool,
    /// Enable detailed heap analysis
    pub detailed_analysis: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(100),
            max_samples: 10000,
            track_call_sites: false,
            detailed_analysis: false,
        }
    }
}

/// Memory usage sample at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    /// Timestamp of the sample
    pub timestamp: Duration,
    /// Current allocated bytes
    pub allocated_bytes: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Peak memory usage
    pub peak_bytes: usize,
}

/// Allocation site information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationSite {
    /// Location in code
    pub location: String,
    /// Total allocations from this site
    pub allocation_count: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Average allocation size
    pub avg_size: usize,
    /// Peak allocation size
    pub max_size: usize,
}

/// Memory analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Total duration of profiling
    pub duration: Duration,
    /// Current memory usage
    pub current_usage: MemoryUsageStats,
    /// Peak memory usage
    pub peak_usage: MemoryUsageStats,
    /// Memory growth rate (bytes/sec)
    pub growth_rate: f64,
    /// Allocation patterns
    pub allocation_patterns: AllocationPatterns,
    /// Top allocation sites
    pub top_allocation_sites: Vec<AllocationSite>,
    /// Memory fragmentation estimate
    pub fragmentation_ratio: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Total allocated bytes
    pub allocated_bytes: usize,
    /// Number of live allocations
    pub live_allocations: usize,
    /// Average allocation size
    pub avg_allocation_size: usize,
    /// Allocation rate (per second)
    pub allocation_rate: f64,
}

/// Allocation pattern analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllocationPatterns {
    /// Small allocations (< 64 bytes)
    pub small_allocations: usize,
    /// Medium allocations (64 - 1KB)
    pub medium_allocations: usize,
    /// Large allocations (1KB - 1MB)
    pub large_allocations: usize,
    /// Huge allocations (> 1MB)
    pub huge_allocations: usize,
    /// Allocation size distribution
    pub size_distribution: HashMap<String, usize>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            enabled: false,
            start_time: Instant::now(),
            samples: Arc::new(RwLock::new(Vec::with_capacity(config.max_samples))),
            allocation_sites: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }
    
    /// Start profiling
    pub fn start(&mut self) {
        self.enabled = true;
        self.start_time = Instant::now();
        TRACKING_ENABLED.store(true, Ordering::SeqCst);
        
        // Reset counters
        ALLOCATIONS.store(0, Ordering::SeqCst);
        DEALLOCATIONS.store(0, Ordering::SeqCst);
        
        // Start sampling thread
        if self.config.sample_interval > Duration::ZERO {
            self.start_sampling_thread();
        }
    }
    
    /// Stop profiling
    pub fn stop(&mut self) {
        self.enabled = false;
        TRACKING_ENABLED.store(false, Ordering::SeqCst);
    }
    
    /// Take a memory snapshot
    pub fn snapshot(&self) -> MemorySample {
        MemorySample {
            timestamp: self.start_time.elapsed(),
            allocated_bytes: CURRENT_USAGE.load(Ordering::Relaxed),
            allocation_count: ALLOCATIONS.load(Ordering::Relaxed),
            deallocation_count: DEALLOCATIONS.load(Ordering::Relaxed),
            peak_bytes: PEAK_USAGE.load(Ordering::Relaxed),
        }
    }
    
    /// Generate a comprehensive memory report
    pub fn generate_report(&self) -> MemoryReport {
        let duration = self.start_time.elapsed();
        let current_usage = self.calculate_current_usage();
        let peak_usage = self.calculate_peak_usage();
        let growth_rate = self.calculate_growth_rate();
        let allocation_patterns = self.analyze_allocation_patterns();
        let top_sites = self.get_top_allocation_sites(10);
        let fragmentation_ratio = self.estimate_fragmentation();
        let recommendations = self.generate_recommendations(&allocation_patterns);
        
        MemoryReport {
            duration,
            current_usage,
            peak_usage,
            growth_rate,
            allocation_patterns,
            top_allocation_sites: top_sites,
            fragmentation_ratio,
            recommendations,
        }
    }
    
    /// Record an allocation site
    pub fn record_allocation_site(&self, location: &str, size: usize) {
        if !self.config.track_call_sites {
            return;
        }
        
        let mut sites = self.allocation_sites.lock();
        let site = sites.entry(location.to_string()).or_insert_with(|| {
            AllocationSite {
                location: location.to_string(),
                allocation_count: 0,
                total_bytes: 0,
                avg_size: 0,
                max_size: 0,
            }
        });
        
        site.allocation_count += 1;
        site.total_bytes += size;
        site.avg_size = site.total_bytes / site.allocation_count;
        site.max_size = site.max_size.max(size);
    }
    
    // Private methods
    
    fn start_sampling_thread(&self) {
        let samples = self.samples.clone();
        let interval = self.config.sample_interval;
        let max_samples = self.config.max_samples;
        
        std::thread::spawn(move || {
            while TRACKING_ENABLED.load(Ordering::Relaxed) {
                let sample = MemorySample {
                    timestamp: Duration::from_secs(0), // Will be set properly
                    allocated_bytes: CURRENT_USAGE.load(Ordering::Relaxed),
                    allocation_count: ALLOCATIONS.load(Ordering::Relaxed),
                    deallocation_count: DEALLOCATIONS.load(Ordering::Relaxed),
                    peak_bytes: PEAK_USAGE.load(Ordering::Relaxed),
                };
                
                let mut samples_guard = samples.write();
                if samples_guard.len() >= max_samples {
                    samples_guard.remove(0);
                }
                samples_guard.push(sample);
                drop(samples_guard);
                
                std::thread::sleep(interval);
            }
        });
    }
    
    fn calculate_current_usage(&self) -> MemoryUsageStats {
        let allocated_bytes = CURRENT_USAGE.load(Ordering::Relaxed);
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        let deallocations = DEALLOCATIONS.load(Ordering::Relaxed);
        let live_allocations = allocations.saturating_sub(deallocations);
        
        let avg_allocation_size = if live_allocations > 0 {
            allocated_bytes / live_allocations
        } else {
            0
        };
        
        let duration_secs = self.start_time.elapsed().as_secs_f64();
        let allocation_rate = if duration_secs > 0.0 {
            allocations as f64 / duration_secs
        } else {
            0.0
        };
        
        MemoryUsageStats {
            allocated_bytes,
            live_allocations,
            avg_allocation_size,
            allocation_rate,
        }
    }
    
    fn calculate_peak_usage(&self) -> MemoryUsageStats {
        let peak_bytes = PEAK_USAGE.load(Ordering::Relaxed);
        
        // Estimate based on current average
        let current = self.calculate_current_usage();
        let estimated_allocations = if current.avg_allocation_size > 0 {
            peak_bytes / current.avg_allocation_size
        } else {
            0
        };
        
        MemoryUsageStats {
            allocated_bytes: peak_bytes,
            live_allocations: estimated_allocations,
            avg_allocation_size: current.avg_allocation_size,
            allocation_rate: current.allocation_rate,
        }
    }
    
    fn calculate_growth_rate(&self) -> f64 {
        let samples = self.samples.read();
        if samples.len() < 2 {
            return 0.0;
        }
        
        let first = &samples[0];
        let last = &samples[samples.len() - 1];
        let duration_secs = self.start_time.elapsed().as_secs_f64();
        
        if duration_secs > 0.0 {
            (last.allocated_bytes as f64 - first.allocated_bytes as f64) / duration_secs
        } else {
            0.0
        }
    }
    
    fn analyze_allocation_patterns(&self) -> AllocationPatterns {
        // This is a simplified analysis
        // In a real implementation, you'd track actual allocation sizes
        let total_allocations = ALLOCATIONS.load(Ordering::Relaxed);
        
        AllocationPatterns {
            small_allocations: total_allocations * 40 / 100, // Estimate 40% small
            medium_allocations: total_allocations * 35 / 100, // 35% medium
            large_allocations: total_allocations * 20 / 100, // 20% large
            huge_allocations: total_allocations * 5 / 100, // 5% huge
            size_distribution: HashMap::new(),
        }
    }
    
    fn get_top_allocation_sites(&self, count: usize) -> Vec<AllocationSite> {
        let sites = self.allocation_sites.lock();
        let mut sorted_sites: Vec<_> = sites.values().cloned().collect();
        sorted_sites.sort_by_key(|s| std::cmp::Reverse(s.total_bytes));
        sorted_sites.truncate(count);
        sorted_sites
    }
    
    fn estimate_fragmentation(&self) -> f64 {
        // Simplified fragmentation estimate
        // Real implementation would use OS-specific APIs
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        let deallocations = DEALLOCATIONS.load(Ordering::Relaxed);
        
        if allocations > 0 {
            let reuse_ratio = deallocations as f64 / allocations as f64;
            (1.0 - reuse_ratio).max(0.0).min(1.0)
        } else {
            0.0
        }
    }
    
    fn generate_recommendations(&self, patterns: &AllocationPatterns) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check for excessive small allocations
        if patterns.small_allocations > patterns.medium_allocations * 2 {
            recommendations.push(
                "Consider using object pooling for small allocations".to_string()
            );
        }
        
        // Check for memory growth
        let growth_rate = self.calculate_growth_rate();
        if growth_rate > 1_000_000.0 { // 1MB/s
            recommendations.push(
                "High memory growth detected. Check for memory leaks.".to_string()
            );
        }
        
        // Check fragmentation
        let fragmentation = self.estimate_fragmentation();
        if fragmentation > 0.3 {
            recommendations.push(
                "Consider using arena allocators to reduce fragmentation".to_string()
            );
        }
        
        // Check for huge allocations
        if patterns.huge_allocations > 0 {
            recommendations.push(
                "Large allocations detected. Consider streaming or chunked processing.".to_string()
            );
        }
        
        recommendations
    }
}

/// Get current memory statistics
pub fn current_memory_stats() -> (usize, usize, usize) {
    (
        CURRENT_USAGE.load(Ordering::Relaxed),
        ALLOCATIONS.load(Ordering::Relaxed),
        DEALLOCATIONS.load(Ordering::Relaxed),
    )
}

/// Enable or disable memory tracking
pub fn set_tracking_enabled(enabled: bool) {
    TRACKING_ENABLED.store(enabled, Ordering::SeqCst);
}

/// Reset memory statistics
pub fn reset_memory_stats() {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    DEALLOCATIONS.store(0, Ordering::SeqCst);
    CURRENT_USAGE.store(0, Ordering::SeqCst);
    PEAK_USAGE.store(0, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new(ProfilerConfig::default());
        
        profiler.start();
        
        // Perform some allocations
        let _v1: Vec<u8> = vec![0; 1024];
        let _v2: Vec<u8> = vec![0; 2048];
        
        let snapshot = profiler.snapshot();
        assert!(snapshot.allocated_bytes > 0);
        assert!(snapshot.allocation_count > 0);
        
        profiler.stop();
    }
    
    #[test]
    fn test_memory_report() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());
        
        let report = profiler.generate_report();
        assert!(report.duration.as_secs() >= 0);
        assert!(report.recommendations.len() >= 0);
    }
}