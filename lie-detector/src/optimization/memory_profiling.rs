//! Enhanced memory profiling and monitoring infrastructure
//! 
//! This module provides comprehensive tools for tracking memory usage, detecting leaks,
//! monitoring allocation patterns, and optimizing memory usage throughout the application.

use crate::{Result, VeritasError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, BTreeMap};
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::backtrace::Backtrace;

/// Global memory allocator with comprehensive tracking
pub struct TrackingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static CURRENT_USAGE: AtomicUsize = AtomicUsize::new(0);
static PEAK_USAGE: AtomicUsize = AtomicUsize::new(0);
static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);

// Size class tracking
static SMALL_ALLOCS: AtomicUsize = AtomicUsize::new(0);      // < 64 bytes
static MEDIUM_ALLOCS: AtomicUsize = AtomicUsize::new(0);     // 64 - 1KB
static LARGE_ALLOCS: AtomicUsize = AtomicUsize::new(0);      // 1KB - 1MB
static HUGE_ALLOCS: AtomicUsize = AtomicUsize::new(0);       // > 1MB

// Allocation tracking map
lazy_static::lazy_static! {
    static ref ALLOCATION_MAP: RwLock<HashMap<usize, AllocationInfo>> = RwLock::new(HashMap::new());
    static ref CALL_SITE_STATS: RwLock<HashMap<String, CallSiteStats>> = RwLock::new(HashMap::new());
}

/// Detailed allocation information
#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    layout: Layout,
    timestamp: Instant,
    #[cfg(feature = "backtrace")]
    backtrace: Backtrace,
    thread_id: std::thread::ThreadId,
}

/// Call site statistics
#[derive(Debug, Clone, Default)]
struct CallSiteStats {
    allocation_count: usize,
    total_bytes: usize,
    peak_bytes: usize,
    avg_lifetime: Duration,
    allocations_per_second: f64,
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() && TRACKING_ENABLED.load(Ordering::Relaxed) {
            let size = layout.size();
            
            // Update size class counters
            match size {
                0..=63 => SMALL_ALLOCS.fetch_add(1, Ordering::Relaxed),
                64..=1023 => MEDIUM_ALLOCS.fetch_add(1, Ordering::Relaxed),
                1024..=1048575 => LARGE_ALLOCS.fetch_add(1, Ordering::Relaxed),
                _ => HUGE_ALLOCS.fetch_add(1, Ordering::Relaxed),
            };
            
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
            
            // Track allocation details
            if let Ok(mut map) = ALLOCATION_MAP.try_write() {
                map.insert(ptr as usize, AllocationInfo {
                    size,
                    layout,
                    timestamp: Instant::now(),
                    #[cfg(feature = "backtrace")]
                    backtrace: Backtrace::capture(),
                    thread_id: std::thread::current().id(),
                });
            }
        }
        
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING_ENABLED.load(Ordering::Relaxed) {
            let size = layout.size();
            
            // Update size class counters
            match size {
                0..=63 => SMALL_ALLOCS.fetch_sub(1, Ordering::Relaxed),
                64..=1023 => MEDIUM_ALLOCS.fetch_sub(1, Ordering::Relaxed),
                1024..=1048575 => LARGE_ALLOCS.fetch_sub(1, Ordering::Relaxed),
                _ => HUGE_ALLOCS.fetch_sub(1, Ordering::Relaxed),
            };
            
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            CURRENT_USAGE.fetch_sub(size, Ordering::Relaxed);
            
            // Remove from tracking map
            if let Ok(mut map) = ALLOCATION_MAP.try_write() {
                if let Some(info) = map.remove(&(ptr as usize)) {
                    let lifetime = info.timestamp.elapsed();
                    
                    // Update call site statistics
                    #[cfg(feature = "backtrace")]
                    if let Ok(mut stats) = CALL_SITE_STATS.try_write() {
                        // Extract call site from backtrace
                        let call_site = extract_call_site(&info.backtrace);
                        let site_stats = stats.entry(call_site).or_default();
                        site_stats.avg_lifetime = Duration::from_secs_f64(
                            (site_stats.avg_lifetime.as_secs_f64() * site_stats.allocation_count as f64
                                + lifetime.as_secs_f64()) / (site_stats.allocation_count + 1) as f64
                        );
                    }
                }
            }
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
    heap_snapshots: Arc<RwLock<Vec<HeapSnapshot>>>,
    config: ProfilerConfig,
    sampling_thread: Option<std::thread::JoinHandle<()>>,
}

/// Enhanced configuration for memory profiler
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
    /// Enable heap snapshots
    pub heap_snapshots: bool,
    /// Heap snapshot interval
    pub snapshot_interval: Duration,
    /// Track object lifetimes
    pub track_lifetimes: bool,
    /// Memory leak detection threshold
    pub leak_threshold: Duration,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(100),
            max_samples: 10000,
            track_call_sites: true,
            detailed_analysis: false,
            heap_snapshots: false,
            snapshot_interval: Duration::from_secs(60),
            track_lifetimes: true,
            leak_threshold: Duration::from_secs(300), // 5 minutes
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
    /// Size class distribution
    pub size_distribution: SizeDistribution,
    /// Thread-specific stats
    pub thread_stats: HashMap<String, ThreadMemoryStats>,
}

/// Size class distribution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SizeDistribution {
    pub small_count: usize,    // < 64 bytes
    pub medium_count: usize,   // 64 - 1KB
    pub large_count: usize,    // 1KB - 1MB
    pub huge_count: usize,     // > 1MB
}

/// Thread-specific memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadMemoryStats {
    pub allocated_bytes: usize,
    pub allocation_count: usize,
    pub peak_bytes: usize,
}

/// Heap snapshot for memory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeapSnapshot {
    pub timestamp: Instant,
    pub total_allocated: usize,
    pub object_count: usize,
    pub type_distribution: HashMap<String, TypeStats>,
    pub largest_allocations: Vec<LargeAllocation>,
}

/// Type-specific memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeStats {
    pub count: usize,
    pub total_bytes: usize,
    pub avg_size: usize,
}

/// Large allocation info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeAllocation {
    pub size: usize,
    pub type_name: String,
    pub location: String,
    pub age: Duration,
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
    /// Current live allocations
    pub live_count: usize,
    /// Current live bytes
    pub live_bytes: usize,
    /// Average lifetime
    pub avg_lifetime: Duration,
}

/// Enhanced memory analysis report
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
    /// Detected memory leaks
    pub potential_leaks: Vec<MemoryLeak>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
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
    /// Size class distribution
    pub size_distribution: SizeDistribution,
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
    /// Temporal patterns
    pub temporal_patterns: Vec<TemporalPattern>,
    /// Hot allocation paths
    pub hot_paths: Vec<HotPath>,
}

/// Temporal allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_type: String,
    pub frequency: f64,
    pub impact: f64,
}

/// Hot allocation path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPath {
    pub location: String,
    pub allocation_rate: f64,
    pub total_bytes: usize,
}

/// Potential memory leak
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    pub location: String,
    pub size: usize,
    pub age: Duration,
    pub growth_rate: f64,
    pub confidence: f64,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OpportunityType,
    pub location: String,
    pub potential_savings: usize,
    pub impact: f64,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunityType {
    ObjectPooling,
    ArenaAllocation,
    StringInterning,
    CompactDataStructure,
    StreamingProcessing,
    CacheSizing,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            enabled: false,
            start_time: Instant::now(),
            samples: Arc::new(RwLock::new(Vec::with_capacity(config.max_samples))),
            allocation_sites: Arc::new(Mutex::new(HashMap::new())),
            heap_snapshots: Arc::new(RwLock::new(Vec::new())),
            config,
            sampling_thread: None,
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
            let samples = Arc::clone(&self.samples);
            let interval = self.config.sample_interval;
            let max_samples = self.config.max_samples;
            let heap_snapshots = Arc::clone(&self.heap_snapshots);
            let snapshot_interval = self.config.snapshot_interval;
            let track_heap = self.config.heap_snapshots;
            
            self.sampling_thread = Some(std::thread::spawn(move || {
                let mut last_snapshot = Instant::now();
                
                while TRACKING_ENABLED.load(Ordering::Relaxed) {
                    // Take memory sample
                    let sample = MemorySample {
                        timestamp: Duration::from_secs(0), // Will be set properly
                        allocated_bytes: CURRENT_USAGE.load(Ordering::Relaxed),
                        allocation_count: ALLOCATIONS.load(Ordering::Relaxed),
                        deallocation_count: DEALLOCATIONS.load(Ordering::Relaxed),
                        peak_bytes: PEAK_USAGE.load(Ordering::Relaxed),
                        size_distribution: SizeDistribution {
                            small_count: SMALL_ALLOCS.load(Ordering::Relaxed),
                            medium_count: MEDIUM_ALLOCS.load(Ordering::Relaxed),
                            large_count: LARGE_ALLOCS.load(Ordering::Relaxed),
                            huge_count: HUGE_ALLOCS.load(Ordering::Relaxed),
                        },
                        thread_stats: collect_thread_stats(),
                    };
                    
                    // Store sample
                    if let Ok(mut samples_guard) = samples.write() {
                        if samples_guard.len() >= max_samples {
                            samples_guard.remove(0);
                        }
                        samples_guard.push(sample);
                    }
                    
                    // Take heap snapshot if enabled
                    if track_heap && last_snapshot.elapsed() >= snapshot_interval {
                        if let Some(snapshot) = take_heap_snapshot() {
                            if let Ok(mut snapshots) = heap_snapshots.write() {
                                snapshots.push(snapshot);
                            }
                        }
                        last_snapshot = Instant::now();
                    }
                    
                    std::thread::sleep(interval);
                }
            }));
        }
    }
    
    /// Stop profiling
    pub fn stop(&mut self) {
        self.enabled = false;
        TRACKING_ENABLED.store(false, Ordering::SeqCst);
        
        // Wait for sampling thread
        if let Some(thread) = self.sampling_thread.take() {
            let _ = thread.join();
        }
    }
    
    /// Take a memory snapshot
    pub fn snapshot(&self) -> MemorySample {
        MemorySample {
            timestamp: self.start_time.elapsed(),
            allocated_bytes: CURRENT_USAGE.load(Ordering::Relaxed),
            allocation_count: ALLOCATIONS.load(Ordering::Relaxed),
            deallocation_count: DEALLOCATIONS.load(Ordering::Relaxed),
            peak_bytes: PEAK_USAGE.load(Ordering::Relaxed),
            size_distribution: SizeDistribution {
                small_count: SMALL_ALLOCS.load(Ordering::Relaxed),
                medium_count: MEDIUM_ALLOCS.load(Ordering::Relaxed),
                large_count: LARGE_ALLOCS.load(Ordering::Relaxed),
                huge_count: HUGE_ALLOCS.load(Ordering::Relaxed),
            },
            thread_stats: collect_thread_stats(),
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
        let potential_leaks = self.detect_memory_leaks();
        let optimization_opportunities = self.find_optimization_opportunities();
        let recommendations = self.generate_recommendations(
            &allocation_patterns,
            &potential_leaks,
            &optimization_opportunities
        );
        
        MemoryReport {
            duration,
            current_usage,
            peak_usage,
            growth_rate,
            allocation_patterns,
            top_allocation_sites: top_sites,
            fragmentation_ratio,
            potential_leaks,
            optimization_opportunities,
            recommendations,
        }
    }
    
    /// Record an allocation site
    pub fn record_allocation_site(&self, location: &str, size: usize) {
        if !self.config.track_call_sites {
            return;
        }
        
        let mut sites = self.allocation_sites.lock().unwrap();
        let site = sites.entry(location.to_string()).or_insert_with(|| {
            AllocationSite {
                location: location.to_string(),
                allocation_count: 0,
                total_bytes: 0,
                avg_size: 0,
                max_size: 0,
                live_count: 0,
                live_bytes: 0,
                avg_lifetime: Duration::ZERO,
            }
        });
        
        site.allocation_count += 1;
        site.total_bytes += size;
        site.avg_size = site.total_bytes / site.allocation_count;
        site.max_size = site.max_size.max(size);
        site.live_count += 1;
        site.live_bytes += size;
    }
    
    // Private methods
    
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
            size_distribution: SizeDistribution {
                small_count: SMALL_ALLOCS.load(Ordering::Relaxed),
                medium_count: MEDIUM_ALLOCS.load(Ordering::Relaxed),
                large_count: LARGE_ALLOCS.load(Ordering::Relaxed),
                huge_count: HUGE_ALLOCS.load(Ordering::Relaxed),
            },
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
            size_distribution: current.size_distribution,
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
        let small = SMALL_ALLOCS.load(Ordering::Relaxed);
        let medium = MEDIUM_ALLOCS.load(Ordering::Relaxed);
        let large = LARGE_ALLOCS.load(Ordering::Relaxed);
        let huge = HUGE_ALLOCS.load(Ordering::Relaxed);
        
        // Detect temporal patterns
        let temporal_patterns = self.detect_temporal_patterns();
        
        // Find hot allocation paths
        let hot_paths = self.find_hot_paths();
        
        AllocationPatterns {
            small_allocations: small,
            medium_allocations: medium,
            large_allocations: large,
            huge_allocations: huge,
            size_distribution: HashMap::new(),
            temporal_patterns,
            hot_paths,
        }
    }
    
    fn detect_temporal_patterns(&self) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        let samples = self.samples.read();
        if samples.len() < 10 {
            return patterns;
        }
        
        // Detect periodic allocation spikes
        let mut allocation_rates = Vec::new();
        for window in samples.windows(2) {
            let rate = (window[1].allocation_count - window[0].allocation_count) as f64;
            allocation_rates.push(rate);
        }
        
        // Simple spike detection
        let avg_rate = allocation_rates.iter().sum::<f64>() / allocation_rates.len() as f64;
        let spike_threshold = avg_rate * 2.0;
        
        let spike_count = allocation_rates.iter().filter(|&&r| r > spike_threshold).count();
        if spike_count > samples.len() / 10 {
            patterns.push(TemporalPattern {
                pattern_type: "Periodic Allocation Spikes".to_string(),
                frequency: spike_count as f64 / samples.len() as f64,
                impact: 0.7,
            });
        }
        
        patterns
    }
    
    fn find_hot_paths(&self) -> Vec<HotPath> {
        let mut paths = Vec::new();
        
        if let Ok(sites) = self.allocation_sites.lock() {
            let mut sorted_sites: Vec<_> = sites.values()
                .map(|site| HotPath {
                    location: site.location.clone(),
                    allocation_rate: site.allocation_count as f64 / self.start_time.elapsed().as_secs_f64(),
                    total_bytes: site.total_bytes,
                })
                .collect();
            
            sorted_sites.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes));
            paths = sorted_sites.into_iter().take(5).collect();
        }
        
        paths
    }
    
    fn get_top_allocation_sites(&self, count: usize) -> Vec<AllocationSite> {
        let sites = self.allocation_sites.lock().unwrap();
        let mut sorted_sites: Vec<_> = sites.values().cloned().collect();
        sorted_sites.sort_by_key(|s| std::cmp::Reverse(s.total_bytes));
        sorted_sites.truncate(count);
        sorted_sites
    }
    
    fn estimate_fragmentation(&self) -> f64 {
        // Enhanced fragmentation estimate
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        let deallocations = DEALLOCATIONS.load(Ordering::Relaxed);
        
        if allocations == 0 {
            return 0.0;
        }
        
        // Factor in size class distribution
        let small = SMALL_ALLOCS.load(Ordering::Relaxed) as f64;
        let total = allocations as f64;
        let small_ratio = small / total;
        
        // High ratio of small allocations increases fragmentation
        let fragmentation_factor = small_ratio * 0.5;
        
        // Factor in allocation/deallocation ratio
        let reuse_ratio = deallocations as f64 / allocations as f64;
        let fragmentation = (fragmentation_factor + (1.0 - reuse_ratio) * 0.5).min(1.0);
        
        fragmentation
    }
    
    fn detect_memory_leaks(&self) -> Vec<MemoryLeak> {
        let mut leaks = Vec::new();
        
        if let Ok(allocations) = ALLOCATION_MAP.read() {
            let now = Instant::now();
            let threshold = self.config.leak_threshold;
            
            // Group by approximate location (if backtraces are available)
            let mut location_groups: HashMap<String, Vec<&AllocationInfo>> = HashMap::new();
            
            for (_, info) in allocations.iter() {
                if now.duration_since(info.timestamp) > threshold {
                    #[cfg(feature = "backtrace")]
                    let location = extract_call_site(&info.backtrace);
                    #[cfg(not(feature = "backtrace"))]
                    let location = "Unknown".to_string();
                    
                    location_groups.entry(location).or_default().push(info);
                }
            }
            
            // Analyze each location group
            for (location, allocs) in location_groups {
                let total_size: usize = allocs.iter().map(|a| a.size).sum();
                let avg_age = allocs.iter()
                    .map(|a| now.duration_since(a.timestamp))
                    .sum::<Duration>() / allocs.len() as u32;
                
                // Calculate growth rate (simplified)
                let growth_rate = total_size as f64 / avg_age.as_secs_f64();
                
                leaks.push(MemoryLeak {
                    location,
                    size: total_size,
                    age: avg_age,
                    growth_rate,
                    confidence: 0.8, // High confidence for old allocations
                });
            }
        }
        
        leaks.sort_by(|a, b| b.size.cmp(&a.size));
        leaks.truncate(10);
        leaks
    }
    
    fn find_optimization_opportunities(&self) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();
        
        // Check for object pooling opportunities
        if let Ok(sites) = self.allocation_sites.lock() {
            for site in sites.values() {
                // High frequency, small allocations are good pooling candidates
                if site.allocation_count > 1000 && site.avg_size < 1024 {
                    opportunities.push(OptimizationOpportunity {
                        opportunity_type: OpportunityType::ObjectPooling,
                        location: site.location.clone(),
                        potential_savings: site.total_bytes / 2, // Conservative estimate
                        impact: 0.7,
                        recommendation: format!(
                            "Consider object pooling for {} byte allocations",
                            site.avg_size
                        ),
                    });
                }
                
                // Large allocations might benefit from streaming
                if site.max_size > 1_048_576 {
                    opportunities.push(OptimizationOpportunity {
                        opportunity_type: OpportunityType::StreamingProcessing,
                        location: site.location.clone(),
                        potential_savings: site.max_size / 4,
                        impact: 0.8,
                        recommendation: "Consider streaming or chunked processing for large data".to_string(),
                    });
                }
            }
        }
        
        // Check for string interning opportunities
        let small_count = SMALL_ALLOCS.load(Ordering::Relaxed);
        if small_count > ALLOCATIONS.load(Ordering::Relaxed) / 2 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OpportunityType::StringInterning,
                location: "Global".to_string(),
                potential_savings: small_count * 32, // Estimate
                impact: 0.6,
                recommendation: "High number of small allocations suggests string interning could help".to_string(),
            });
        }
        
        opportunities.sort_by(|a, b| b.impact.partial_cmp(&a.impact).unwrap());
        opportunities.truncate(10);
        opportunities
    }
    
    fn generate_recommendations(&self, 
                               patterns: &AllocationPatterns,
                               leaks: &[MemoryLeak],
                               opportunities: &[OptimizationOpportunity]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check for excessive small allocations
        if patterns.small_allocations > patterns.medium_allocations * 2 {
            recommendations.push(
                "✓ Implement object pooling for frequently allocated small objects".to_string()
            );
        }
        
        // Check for memory growth
        let growth_rate = self.calculate_growth_rate();
        if growth_rate > 1_000_000.0 { // 1MB/s
            recommendations.push(
                "⚠️ High memory growth detected. Review allocation patterns and check for leaks.".to_string()
            );
        }
        
        // Check fragmentation
        let fragmentation = self.estimate_fragmentation();
        if fragmentation > 0.3 {
            recommendations.push(
                "✓ Use arena allocators for temporary allocations to reduce fragmentation".to_string()
            );
        }
        
        // Check for huge allocations
        if patterns.huge_allocations > 0 {
            recommendations.push(
                "✓ Implement streaming or lazy loading for large data processing".to_string()
            );
        }
        
        // Memory leak recommendations
        if !leaks.is_empty() {
            recommendations.push(format!(
                "⚠️ {} potential memory leaks detected. Review long-lived allocations.",
                leaks.len()
            ));
        }
        
        // Optimization recommendations
        for opp in opportunities.iter().take(3) {
            recommendations.push(format!("✓ {}", opp.recommendation));
        }
        
        // GPU memory recommendations
        if patterns.large_allocations > 100 {
            recommendations.push(
                "✓ Consider using pinned memory pools for GPU transfers".to_string()
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
    SMALL_ALLOCS.store(0, Ordering::SeqCst);
    MEDIUM_ALLOCS.store(0, Ordering::SeqCst);
    LARGE_ALLOCS.store(0, Ordering::SeqCst);
    HUGE_ALLOCS.store(0, Ordering::SeqCst);
}

// Helper functions

fn collect_thread_stats() -> HashMap<String, ThreadMemoryStats> {
    // In a real implementation, this would collect per-thread statistics
    HashMap::new()
}

fn take_heap_snapshot() -> Option<HeapSnapshot> {
    // In a real implementation, this would capture a detailed heap snapshot
    None
}

#[cfg(feature = "backtrace")]
fn extract_call_site(backtrace: &Backtrace) -> String {
    // Extract the most relevant call site from backtrace
    backtrace.to_string().lines()
        .find(|line| line.contains("src/") && !line.contains("memory_profiling"))
        .unwrap_or("Unknown")
        .to_string()
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
        assert!(!report.recommendations.is_empty());
        assert!(report.fragmentation_ratio >= 0.0 && report.fragmentation_ratio <= 1.0);
    }
    
    #[test]
    fn test_size_distribution() {
        reset_memory_stats();
        set_tracking_enabled(true);
        
        // Small allocation
        let _small: Vec<u8> = vec![0; 32];
        
        // Medium allocation
        let _medium: Vec<u8> = vec![0; 512];
        
        // Large allocation
        let _large: Vec<u8> = vec![0; 8192];
        
        let small = SMALL_ALLOCS.load(Ordering::Relaxed);
        let medium = MEDIUM_ALLOCS.load(Ordering::Relaxed);
        let large = LARGE_ALLOCS.load(Ordering::Relaxed);
        
        assert!(small > 0 || medium > 0 || large > 0);
        
        set_tracking_enabled(false);
    }
}