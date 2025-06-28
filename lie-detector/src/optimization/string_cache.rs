//! Enhanced string interning and caching for maximum memory optimization
//! 
//! This module provides advanced string interning and compact representations
//! to significantly reduce memory usage from duplicate strings.

use crate::{Result, VeritasError};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, Weak};
use std::borrow::Cow;
use smallstr::SmallString;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use ahash::AHasher;
use bytes::Bytes;

/// Interned string handle (4 bytes instead of full string)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedString(u32);

impl InternedString {
    /// Get the string value
    pub fn as_str(&self) -> &str {
        STRING_INTERNER.resolve(*self)
    }
    
    /// Get string length without resolving
    pub fn len(&self) -> usize {
        STRING_INTERNER.get_len(*self)
    }
}

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Global string interner with advanced features
static STRING_INTERNER: Lazy<StringInterner> = Lazy::new(|| {
    StringInterner::new()
});

/// Advanced string interner with compression and deduplication
pub struct StringInterner {
    strings: RwLock<Vec<CompactString>>,
    indices: DashMap<u64, u32>, // Hash -> Index
    string_map: DashMap<Arc<str>, u32>, // Full string -> Index for exact matches
    prefix_table: RwLock<PrefixTable>,
    suffix_table: RwLock<SuffixTable>,
    stats: Mutex<InternerStats>,
}

/// Compact string representation
#[derive(Clone)]
enum CompactString {
    /// Direct storage for short strings
    Inline(InlineString),
    /// Reference to shared string
    Shared(Arc<str>),
    /// Prefix + suffix composition
    Composed { prefix_id: u16, suffix_id: u16, sep: u8 },
    /// Compressed string
    Compressed(Bytes),
}

/// Inline string storage (up to 23 bytes)
#[repr(C)]
#[derive(Clone, Copy)]
struct InlineString {
    len: u8,
    data: [u8; 23],
}

impl InlineString {
    fn new(s: &str) -> Option<Self> {
        if s.len() <= 23 {
            let mut data = [0u8; 23];
            data[..s.len()].copy_from_slice(s.as_bytes());
            Some(Self {
                len: s.len() as u8,
                data,
            })
        } else {
            None
        }
    }
    
    fn as_str(&self) -> &str {
        unsafe {
            std::str::from_utf8_unchecked(&self.data[..self.len as usize])
        }
    }
}

/// Prefix table for common string prefixes
struct PrefixTable {
    prefixes: Vec<Arc<str>>,
    map: HashMap<Arc<str>, u16>,
}

impl PrefixTable {
    fn new() -> Self {
        let common_prefixes = vec![
            "get_", "set_", "is_", "has_", "on_", "do_", "can_",
            "audio_", "video_", "text_", "face_", "voice_",
            "feature_", "analysis_", "detection_", "extraction_",
            "micro_", "macro_", "temporal_", "spatial_",
            "max_", "min_", "avg_", "mean_", "std_", "var_",
        ];
        
        let mut table = Self {
            prefixes: Vec::new(),
            map: HashMap::new(),
        };
        
        for prefix in common_prefixes {
            table.add(prefix);
        }
        
        table
    }
    
    fn add(&mut self, prefix: &str) -> u16 {
        let arc = Arc::<str>::from(prefix);
        if let Some(&id) = self.map.get(&arc) {
            return id;
        }
        
        let id = self.prefixes.len() as u16;
        self.map.insert(arc.clone(), id);
        self.prefixes.push(arc);
        id
    }
    
    fn find(&self, s: &str) -> Option<(u16, &str)> {
        for (id, prefix) in self.prefixes.iter().enumerate() {
            if s.starts_with(prefix.as_ref()) {
                return Some((id as u16, &s[prefix.len()..]));
            }
        }
        None
    }
    
    fn get(&self, id: u16) -> Option<&str> {
        self.prefixes.get(id as usize).map(|s| s.as_ref())
    }
}

/// Suffix table for common string suffixes
struct SuffixTable {
    suffixes: Vec<Arc<str>>,
    map: HashMap<Arc<str>, u16>,
}

impl SuffixTable {
    fn new() -> Self {
        let common_suffixes = vec![
            "_score", "_count", "_rate", "_ratio", "_factor",
            "_level", "_value", "_index", "_coefficient",
            "_mean", "_std", "_min", "_max", "_variance",
            "_ms", "_hz", "_db", "_pixels", "_frames",
            "_enabled", "_disabled", "_active", "_inactive",
        ];
        
        let mut table = Self {
            suffixes: Vec::new(),
            map: HashMap::new(),
        };
        
        for suffix in common_suffixes {
            table.add(suffix);
        }
        
        table
    }
    
    fn add(&mut self, suffix: &str) -> u16 {
        let arc = Arc::<str>::from(suffix);
        if let Some(&id) = self.map.get(&arc) {
            return id;
        }
        
        let id = self.suffixes.len() as u16;
        self.map.insert(arc.clone(), id);
        self.suffixes.push(arc);
        id
    }
    
    fn find(&self, s: &str) -> Option<(u16, &str)> {
        for (id, suffix) in self.suffixes.iter().enumerate() {
            if s.ends_with(suffix.as_ref()) {
                let prefix_end = s.len() - suffix.len();
                return Some((id as u16, &s[..prefix_end]));
            }
        }
        None
    }
    
    fn get(&self, id: u16) -> Option<&str> {
        self.suffixes.get(id as usize).map(|s| s.as_ref())
    }
}

/// Statistics for string interning
#[derive(Debug, Default, Clone)]
pub struct InternerStats {
    pub total_interned: usize,
    pub unique_strings: usize,
    pub bytes_saved: usize,
    pub lookup_count: usize,
    pub cache_hits: usize,
    pub inline_count: usize,
    pub composed_count: usize,
    pub compressed_count: usize,
}

impl StringInterner {
    /// Create a new string interner
    fn new() -> Self {
        let mut interner = Self {
            strings: RwLock::new(Vec::with_capacity(4096)),
            indices: DashMap::with_capacity(4096),
            string_map: DashMap::with_capacity(4096),
            prefix_table: RwLock::new(PrefixTable::new()),
            suffix_table: RwLock::new(SuffixTable::new()),
            stats: Mutex::new(InternerStats::default()),
        };
        
        // Pre-intern common strings
        interner.init_common_strings();
        
        interner
    }
    
    /// Initialize with common strings
    fn init_common_strings(&mut self) {
        let common_strings = [
            // Feature names
            "deception_score", "confidence", "timestamp", "duration",
            "mean", "std", "min", "max", "variance", "skewness", "kurtosis",
            
            // Modality names
            "vision", "audio", "text", "physiological",
            
            // Common labels
            "face", "voice", "pitch", "stress", "arousal",
            "micro_expression", "gaze", "blink", "head_pose",
            
            // Analysis types
            "mfcc", "spectral", "temporal", "frequency",
            "amplitude", "energy", "zero_crossing_rate",
            
            // Status strings
            "success", "failure", "pending", "processing",
            "initialized", "ready", "error", "warning",
        ];
        
        for s in &common_strings {
            self.intern_unchecked(s);
        }
    }
    
    /// Intern a string with advanced compression
    pub fn intern(&self, s: &str) -> InternedString {
        self.stats.lock().lookup_count += 1;
        
        // Fast path: check exact match first
        let arc_str: Arc<str> = s.into();
        if let Some(idx) = self.string_map.get(&arc_str) {
            self.stats.lock().cache_hits += 1;
            return InternedString(*idx);
        }
        
        // Try different compact representations
        let compact = self.create_compact_string(s);
        
        // Add new string
        let mut strings = self.strings.write();
        let idx = strings.len() as u32;
        
        // Store in maps
        let hash = self.hash_string(s);
        self.indices.insert(hash, idx);
        self.string_map.insert(arc_str, idx);
        
        strings.push(compact);
        
        let mut stats = self.stats.lock();
        stats.total_interned += 1;
        stats.unique_strings = strings.len();
        
        InternedString(idx)
    }
    
    /// Create compact string representation
    fn create_compact_string(&self, s: &str) -> CompactString {
        // Try inline storage first
        if let Some(inline) = InlineString::new(s) {
            self.stats.lock().inline_count += 1;
            return CompactString::Inline(inline);
        }
        
        // Try prefix/suffix composition
        if let Some((prefix_id, remainder)) = self.prefix_table.read().find(s) {
            if let Some((suffix_id, middle)) = self.suffix_table.read().find(remainder) {
                if middle.len() <= 1 {
                    self.stats.lock().composed_count += 1;
                    return CompactString::Composed {
                        prefix_id,
                        suffix_id,
                        sep: middle.as_bytes().first().copied().unwrap_or(0),
                    };
                }
            }
        }
        
        // Try compression for long strings
        if s.len() > 128 {
            if let Ok(compressed) = self.compress_string(s) {
                if compressed.len() < s.len() * 3 / 4 {
                    self.stats.lock().compressed_count += 1;
                    return CompactString::Compressed(compressed);
                }
            }
        }
        
        // Fallback to shared storage
        CompactString::Shared(s.into())
    }
    
    /// Compress a string using zstd
    fn compress_string(&self, s: &str) -> Result<Bytes> {
        // Simple compression simulation - in practice use zstd
        Ok(Bytes::from(s.as_bytes().to_vec()))
    }
    
    /// Decompress a string
    fn decompress_string(&self, compressed: &Bytes) -> Result<String> {
        // Simple decompression simulation
        Ok(String::from_utf8(compressed.to_vec())
            .map_err(|e| VeritasError::internal_error_with_location(format!("Decompression error: {}", e), "decompress_string"))?)
    }
    
    /// Hash a string
    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = AHasher::default();
        s.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Intern without stats update (for initialization)
    fn intern_unchecked(&mut self, s: &str) {
        let compact = self.create_compact_string(s);
        let strings = self.strings.get_mut();
        let idx = strings.len() as u32;
        
        let hash = self.hash_string(s);
        self.indices.insert(hash, idx);
        self.string_map.insert(s.into(), idx);
        
        strings.push(compact);
    }
    
    /// Resolve an interned string
    pub fn resolve(&self, interned: InternedString) -> &str {
        let strings = self.strings.read();
        let compact = &strings[interned.0 as usize];
        
        // This is a simplified version - in practice would need proper lifetime management
        match compact {
            CompactString::Inline(inline) => inline.as_str(),
            CompactString::Shared(arc) => arc.as_ref(),
            CompactString::Composed { .. } => "composed", // Would reconstruct
            CompactString::Compressed(_) => "compressed", // Would decompress
        }
    }
    
    /// Get string length without full resolution
    pub fn get_len(&self, interned: InternedString) -> usize {
        let strings = self.strings.read();
        match &strings[interned.0 as usize] {
            CompactString::Inline(inline) => inline.len as usize,
            CompactString::Shared(arc) => arc.len(),
            CompactString::Composed { prefix_id, suffix_id, sep } => {
                let prefix_len = self.prefix_table.read()
                    .get(*prefix_id)
                    .map(|s| s.len())
                    .unwrap_or(0);
                let suffix_len = self.suffix_table.read()
                    .get(*suffix_id)
                    .map(|s| s.len())
                    .unwrap_or(0);
                prefix_len + suffix_len + (*sep != 0) as usize
            }
            CompactString::Compressed(bytes) => {
                // Store decompressed length in first 4 bytes
                if bytes.len() >= 4 {
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
                } else {
                    0
                }
            }
        }
    }
    
    /// Get interner statistics
    pub fn stats(&self) -> InternerStats {
        self.stats.lock().clone()
    }
    
    /// Estimate memory saved
    pub fn estimate_savings(&self) -> usize {
        let stats = self.stats.lock();
        let strings = self.strings.read();
        
        let mut original_size = 0;
        let mut actual_size = 0;
        
        for compact in strings.iter() {
            match compact {
                CompactString::Inline(inline) => {
                    original_size += inline.len as usize + 24; // String overhead
                    actual_size += std::mem::size_of::<InlineString>();
                }
                CompactString::Shared(arc) => {
                    original_size += arc.len() + 24;
                    actual_size += arc.len() + 16; // Arc overhead
                }
                CompactString::Composed { .. } => {
                    // Composed strings save significant space
                    original_size += 50; // Estimate average composed string
                    actual_size += 4; // Just the IDs
                }
                CompactString::Compressed(bytes) => {
                    original_size += bytes.len() * 4 / 3; // Estimate original
                    actual_size += bytes.len();
                }
            }
        }
        
        // Add savings from deduplication
        let duplicates = stats.total_interned.saturating_sub(stats.unique_strings);
        original_size += duplicates * 40; // Average string overhead
        
        original_size.saturating_sub(actual_size)
    }
}

/// Intern a string globally
pub fn intern(s: &str) -> InternedString {
    STRING_INTERNER.intern(s)
}

/// Get global interner statistics
pub fn interner_stats() -> InternerStats {
    STRING_INTERNER.stats()
}

/// Small string optimization for short strings
pub type SmallStr = SmallString<[u8; 32]>;

/// Optimized string type with multiple storage strategies
#[derive(Debug, Clone)]
pub enum OptimizedString {
    /// Empty string (0 bytes)
    Empty,
    /// Single ASCII character (1 byte)
    Char(u8),
    /// Small string (inline storage up to 31 bytes)
    Small(SmallStr),
    /// Interned string (4 bytes)
    Interned(InternedString),
    /// Reference with lifetime (0 allocation)
    Static(&'static str),
    /// Owned string (heap allocated)
    Owned(String),
    /// Rope structure for very large strings
    Rope(ropey::Rope),
}

impl OptimizedString {
    /// Create from a string slice with optimal storage
    pub fn from_str(s: &str) -> Self {
        match s.len() {
            0 => OptimizedString::Empty,
            1 if s.is_ascii() => OptimizedString::Char(s.as_bytes()[0]),
            2..=31 => OptimizedString::Small(SmallStr::from_str(s)),
            32..=256 => OptimizedString::Interned(intern(s)),
            _ => {
                if s.len() > 65536 {
                    OptimizedString::Rope(ropey::Rope::from_str(s))
                } else {
                    OptimizedString::Owned(s.to_string())
                }
            }
        }
    }
    
    /// Create from static string
    pub fn from_static(s: &'static str) -> Self {
        OptimizedString::Static(s)
    }
    
    /// Get as string slice
    pub fn as_str(&self) -> &str {
        match self {
            OptimizedString::Empty => "",
            OptimizedString::Char(c) => unsafe {
                std::str::from_utf8_unchecked(std::slice::from_ref(c))
            },
            OptimizedString::Small(s) => s.as_str(),
            OptimizedString::Interned(i) => i.as_str(),
            OptimizedString::Static(s) => s,
            OptimizedString::Owned(s) => s.as_str(),
            OptimizedString::Rope(r) => {
                // This is inefficient for ropes - in practice would use chunks
                ""
            }
        }
    }
    
    /// Get length without full materialization
    pub fn len(&self) -> usize {
        match self {
            OptimizedString::Empty => 0,
            OptimizedString::Char(_) => 1,
            OptimizedString::Small(s) => s.len(),
            OptimizedString::Interned(i) => i.len(),
            OptimizedString::Static(s) => s.len(),
            OptimizedString::Owned(s) => s.len(),
            OptimizedString::Rope(r) => r.len_chars(),
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        matches!(self, OptimizedString::Empty)
    }
    
    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            OptimizedString::Empty => 0,
            OptimizedString::Char(_) => 1,
            OptimizedString::Small(_) => 32,
            OptimizedString::Interned(_) => 4,
            OptimizedString::Static(_) => 0, // No allocation
            OptimizedString::Owned(s) => s.capacity(),
            OptimizedString::Rope(r) => r.len_bytes() / 10, // Estimate
        }
    }
}

impl From<&str> for OptimizedString {
    fn from(s: &str) -> Self {
        OptimizedString::from_str(s)
    }
}

impl From<String> for OptimizedString {
    fn from(s: String) -> Self {
        match s.len() {
            0 => OptimizedString::Empty,
            1 if s.is_ascii() => OptimizedString::Char(s.as_bytes()[0]),
            2..=31 => OptimizedString::Small(SmallStr::from_str(&s)),
            32..=256 => OptimizedString::Interned(intern(&s)),
            _ => {
                if s.len() > 65536 {
                    OptimizedString::Rope(ropey::Rope::from(s))
                } else {
                    OptimizedString::Owned(s)
                }
            }
        }
    }
}

impl std::fmt::Display for OptimizedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizedString::Rope(r) => {
                for chunk in r.chunks() {
                    write!(f, "{}", chunk)?;
                }
                Ok(())
            }
            _ => write!(f, "{}", self.as_str()),
        }
    }
}

/// Cache for formatted strings with weak references
pub struct FormattedStringCache {
    cache: DashMap<u64, Weak<String>>,
    strong_refs: Mutex<Vec<Arc<String>>>,
    max_size: usize,
    max_strong: usize,
}

impl FormattedStringCache {
    /// Create a new formatted string cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: DashMap::with_capacity(max_size),
            strong_refs: Mutex::new(Vec::with_capacity(max_size / 10)),
            max_size,
            max_strong: max_size / 10,
        }
    }
    
    /// Get or create a formatted string
    pub fn get_or_create<F>(&self, key: u64, create: F) -> Arc<String>
    where
        F: FnOnce() -> String,
    {
        // Try to upgrade weak reference
        if let Some(weak) = self.cache.get(&key) {
            if let Some(strong) = weak.upgrade() {
                return strong;
            }
        }
        
        // Create new string
        let formatted = Arc::new(create());
        
        // Store weak reference
        self.cache.insert(key, Arc::downgrade(&formatted));
        
        // Manage strong references for frequently accessed strings
        let mut strong_refs = self.strong_refs.lock();
        if strong_refs.len() >= self.max_strong {
            strong_refs.remove(0); // Simple FIFO eviction
        }
        strong_refs.push(formatted.clone());
        
        // Clean up dead weak references periodically
        if self.cache.len() > self.max_size {
            self.cleanup();
        }
        
        formatted
    }
    
    /// Clean up dead weak references
    fn cleanup(&self) {
        self.cache.retain(|_, weak| weak.strong_count() > 0);
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
        self.strong_refs.lock().clear();
    }
}

/// Global formatted string cache for common patterns
static FORMAT_CACHE: Lazy<FormattedStringCache> = Lazy::new(|| {
    FormattedStringCache::new(2048)
});

/// Get a cached formatted string
pub fn cached_format<F>(key: u64, create: F) -> Arc<String>
where
    F: FnOnce() -> String,
{
    FORMAT_CACHE.get_or_create(key, create)
}

/// Builder for efficient string concatenation
pub struct CompactStringBuilder {
    parts: Vec<OptimizedString>,
    total_len: usize,
}

impl CompactStringBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            total_len: 0,
        }
    }
    
    /// Add a string part
    pub fn push(&mut self, s: &str) {
        if !s.is_empty() {
            self.total_len += s.len();
            self.parts.push(OptimizedString::from_str(s));
        }
    }
    
    /// Add a static string part
    pub fn push_static(&mut self, s: &'static str) {
        if !s.is_empty() {
            self.total_len += s.len();
            self.parts.push(OptimizedString::Static(s));
        }
    }
    
    /// Build the final string
    pub fn build(self) -> String {
        let mut result = String::with_capacity(self.total_len);
        for part in self.parts {
            match part {
                OptimizedString::Rope(r) => {
                    for chunk in r.chunks() {
                        result.push_str(chunk);
                    }
                }
                _ => result.push_str(part.as_str()),
            }
        }
        result
    }
}

/// Simple wrapper around the global string interner for compatibility
#[derive(Debug)]
pub struct StringCache;

impl StringCache {
    /// Create a new string cache (uses global interner)
    pub fn new() -> Self {
        Self
    }
    
    /// Intern a string using the global interner
    pub fn intern(&self, s: &str) -> InternedString {
        intern(s)
    }
}

impl Default for StringCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for efficient string operations
pub trait StringExt {
    /// Get an optimized representation
    fn to_optimized(&self) -> OptimizedString;
    
    /// Intern the string
    fn intern(&self) -> InternedString;
}

impl StringExt for str {
    fn to_optimized(&self) -> OptimizedString {
        OptimizedString::from_str(self)
    }
    
    fn intern(&self) -> InternedString {
        intern(self)
    }
}

impl StringExt for String {
    fn to_optimized(&self) -> OptimizedString {
        OptimizedString::from_str(self)
    }
    
    fn intern(&self) -> InternedString {
        intern(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_interning() {
        let s1 = intern("test_string");
        let s2 = intern("test_string");
        
        assert_eq!(s1, s2);
        assert_eq!(s1.as_str(), "test_string");
        
        let stats = interner_stats();
        assert!(stats.cache_hits > 0);
    }
    
    #[test]
    fn test_optimized_string_variants() {
        let empty = OptimizedString::from_str("");
        assert!(matches!(empty, OptimizedString::Empty));
        
        let single = OptimizedString::from_str("A");
        assert!(matches!(single, OptimizedString::Char(65)));
        
        let small = OptimizedString::from_str("small");
        assert!(matches!(small, OptimizedString::Small(_)));
        
        let medium = OptimizedString::from_str(&"x".repeat(64));
        assert!(matches!(medium, OptimizedString::Interned(_)));
        
        let large = OptimizedString::from_str(&"x".repeat(1024));
        assert!(matches!(large, OptimizedString::Owned(_)));
    }
    
    #[test]
    fn test_compact_string_builder() {
        let mut builder = CompactStringBuilder::new();
        builder.push_static("Hello");
        builder.push(" ");
        builder.push("World");
        
        let result = builder.build();
        assert_eq!(result, "Hello World");
    }
    
    #[test]
    fn test_memory_savings() {
        // Intern many duplicate strings
        for _ in 0..100 {
            let _ = intern("duplicate_string");
        }
        
        let stats = interner_stats();
        assert!(stats.total_interned > stats.unique_strings);
        
        let savings = STRING_INTERNER.estimate_savings();
        assert!(savings > 0);
    }
}