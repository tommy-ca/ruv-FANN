//! Lazy loading infrastructure for memory-efficient data access
//!
//! This module provides lazy loading capabilities for large datasets,
//! enabling processing of massive files without loading them entirely into memory.

use crate::{Result, VeritasError};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use lru::LruCache;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::num::NonZeroUsize;

/// Lazy loader for large files with caching and memory mapping
pub struct LazyFileLoader {
    file_path: PathBuf,
    file_size: u64,
    mmap: Option<Mmap>,
    cache: Arc<Mutex<LruCache<u64, Vec<u8>>>>,
    config: LazyLoaderConfig,
    stats: Arc<LoaderStats>,
}

/// Configuration for lazy loader
#[derive(Debug, Clone)]
pub struct LazyLoaderConfig {
    /// Enable memory mapping for files
    pub use_mmap: bool,
    /// Minimum file size for mmap (bytes)
    pub mmap_threshold: u64,
    /// Cache size in number of chunks
    pub cache_size: usize,
    /// Chunk size for reading (bytes)
    pub chunk_size: usize,
    /// Prefetch adjacent chunks
    pub prefetch_enabled: bool,
    /// Number of chunks to prefetch
    pub prefetch_count: usize,
    /// Enable compression for cached chunks
    pub compress_cache: bool,
}

impl Default for LazyLoaderConfig {
    fn default() -> Self {
        Self {
            use_mmap: true,
            mmap_threshold: 10 * 1024 * 1024, // 10MB
            cache_size: 100,
            chunk_size: 64 * 1024, // 64KB
            prefetch_enabled: true,
            prefetch_count: 2,
            compress_cache: false,
        }
    }
}

/// Loader statistics
#[derive(Debug, Default)]
struct LoaderStats {
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    bytes_read: AtomicU64,
    prefetch_hits: AtomicUsize,
}

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

impl LazyFileLoader {
    /// Create a new lazy file loader
    pub fn new<P: AsRef<Path>>(path: P, config: LazyLoaderConfig) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();
        
        // Create memory map if appropriate
        let mmap = if config.use_mmap && file_size >= config.mmap_threshold {
            unsafe {
                MmapOptions::new()
                    .map(&file)
                    .ok()
            }
        } else {
            None
        };
        
        let cache = Arc::new(Mutex::new(LruCache::new(
            NonZeroUsize::new(config.cache_size).unwrap()
        )));
        
        Ok(Self {
            file_path,
            file_size,
            mmap,
            cache,
            config,
            stats: Arc::new(LoaderStats::default()),
        })
    }
    
    /// Read data at specific offset without loading entire file
    pub fn read_at(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        if offset + length as u64 > self.file_size {
            return Err(VeritasError::io_error_with_path(
                "Read beyond file end",
                self.file_path.to_string_lossy(),
                std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Read beyond file end")
            ));
        }
        
        // Try memory map first
        if let Some(mmap) = &self.mmap {
            self.stats.bytes_read.fetch_add(length as u64, Ordering::Relaxed);
            return Ok(mmap[offset as usize..(offset + length as u64) as usize].to_vec());
        }
        
        // Try cache
        let chunk_id = offset / self.config.chunk_size as u64;
        let chunk_offset = (offset % self.config.chunk_size as u64) as usize;
        
        let mut cache = self.cache.lock();
        if let Some(chunk_data) = cache.get(&chunk_id) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            let end = (chunk_offset + length).min(chunk_data.len());
            return Ok(chunk_data[chunk_offset..end].to_vec());
        }
        
        drop(cache); // Release lock before reading
        
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // Read from file
        let chunk_data = self.read_chunk(chunk_id)?;
        
        // Prefetch if enabled
        if self.config.prefetch_enabled {
            self.prefetch_chunks(chunk_id);
        }
        
        // Extract requested data
        let end = (chunk_offset + length).min(chunk_data.len());
        let result = chunk_data[chunk_offset..end].to_vec();
        
        // Cache the chunk
        self.cache.lock().put(chunk_id, chunk_data);
        
        Ok(result)
    }
    
    /// Read a chunk from file
    fn read_chunk(&self, chunk_id: u64) -> Result<Vec<u8>> {
        let mut file = File::open(&self.file_path)?;
        let offset = chunk_id * self.config.chunk_size as u64;
        file.seek(SeekFrom::Start(offset))?;
        
        let remaining = self.file_size - offset;
        let chunk_size = (self.config.chunk_size as u64).min(remaining) as usize;
        
        let mut buffer = vec![0u8; chunk_size];
        file.read_exact(&mut buffer)?;
        
        self.stats.bytes_read.fetch_add(chunk_size as u64, Ordering::Relaxed);
        
        // Compress if configured
        if self.config.compress_cache {
            // In practice, would use zstd or similar
            Ok(buffer)
        } else {
            Ok(buffer)
        }
    }
    
    /// Prefetch adjacent chunks
    fn prefetch_chunks(&self, current_chunk: u64) {
        let prefetch_count = self.config.prefetch_count as u64;
        
        std::thread::spawn({
            let cache = self.cache.clone();
            let config = self.config.clone();
            let file_path = self.file_path.clone();
            let file_size = self.file_size;
            let stats = self.stats.clone();
            
            move || {
                for i in 1..=prefetch_count {
                    let chunk_id = current_chunk + i;
                    let offset = chunk_id * config.chunk_size as u64;
                    
                    if offset >= file_size {
                        break;
                    }
                    
                    // Check if already cached
                    if cache.lock().contains(&chunk_id) {
                        stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    
                    // Read chunk
                    if let Ok(mut file) = File::open(&file_path) {
                        if file.seek(SeekFrom::Start(offset)).is_ok() {
                            let remaining = file_size - offset;
                            let chunk_size = (config.chunk_size as u64).min(remaining) as usize;
                            
                            let mut buffer = vec![0u8; chunk_size];
                            if file.read_exact(&mut buffer).is_ok() {
                                cache.lock().put(chunk_id, buffer);
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Get file size
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize, u64) {
        (
            self.stats.cache_hits.load(Ordering::Relaxed),
            self.stats.cache_misses.load(Ordering::Relaxed),
            self.stats.bytes_read.load(Ordering::Relaxed),
        )
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.lock().clear();
    }
}

/// Lazy dataset loader for machine learning
pub struct LazyDataset<T> {
    /// Index mapping: dataset index -> (file_id, offset, length)
    index: Vec<(usize, u64, usize)>,
    /// File loaders
    loaders: Vec<LazyFileLoader>,
    /// Deserializer function
    deserializer: Arc<dyn Fn(&[u8]) -> Result<T> + Send + Sync>,
    /// Optional sample cache
    sample_cache: Option<Arc<Mutex<LruCache<usize, T>>>>,
}

impl<T: Clone + Send + Sync + 'static> LazyDataset<T> {
    /// Create a new lazy dataset
    pub fn new<F>(deserializer: F, cache_samples: bool) -> Self
    where
        F: Fn(&[u8]) -> Result<T> + Send + Sync + 'static,
    {
        let sample_cache = if cache_samples {
            Some(Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(1000).unwrap()
            ))))
        } else {
            None
        };
        
        Self {
            index: Vec::new(),
            loaders: Vec::new(),
            deserializer: Arc::new(deserializer),
            sample_cache,
        }
    }
    
    /// Add a data file to the dataset
    pub fn add_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        entries: Vec<(u64, usize)>, // (offset, length) pairs
        config: LazyLoaderConfig,
    ) -> Result<()> {
        let loader = LazyFileLoader::new(path, config)?;
        let file_id = self.loaders.len();
        
        // Add entries to index
        for (offset, length) in entries {
            self.index.push((file_id, offset, length));
        }
        
        self.loaders.push(loader);
        Ok(())
    }
    
    /// Get dataset length
    pub fn len(&self) -> usize {
        self.index.len()
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
    
    /// Get a sample by index
    pub fn get(&self, index: usize) -> Result<T> {
        if index >= self.index.len() {
            return Err(VeritasError::IndexError(format!(
                "Index {} out of bounds for dataset of size {}",
                index,
                self.index.len()
            )));
        }
        
        // Check cache first
        if let Some(cache) = &self.sample_cache {
            if let Some(sample) = cache.lock().get(&index) {
                return Ok(sample.clone());
            }
        }
        
        // Load from file
        let (file_id, offset, length) = self.index[index];
        let data = self.loaders[file_id].read_at(offset, length)?;
        let sample = (self.deserializer)(&data)?;
        
        // Cache if enabled
        if let Some(cache) = &self.sample_cache {
            cache.lock().put(index, sample.clone());
        }
        
        Ok(sample)
    }
    
    /// Get multiple samples efficiently
    pub fn get_batch(&self, indices: &[usize]) -> Result<Vec<T>> {
        let mut samples = Vec::with_capacity(indices.len());
        
        // Group by file for efficient loading
        let mut file_groups: HashMap<usize, Vec<(usize, u64, usize)>> = HashMap::new();
        
        for &idx in indices {
            if idx >= self.index.len() {
                return Err(VeritasError::IndexError(format!(
                    "Index {} out of bounds", idx
                )));
            }
            
            let (file_id, offset, length) = self.index[idx];
            file_groups.entry(file_id)
                .or_insert_with(Vec::new)
                .push((idx, offset, length));
        }
        
        // Load samples grouped by file
        let mut results: HashMap<usize, T> = HashMap::new();
        
        for (file_id, requests) in file_groups {
            for (idx, offset, length) in requests {
                if let Some(cache) = &self.sample_cache {
                    if let Some(sample) = cache.lock().get(&idx) {
                        results.insert(idx, sample.clone());
                        continue;
                    }
                }
                
                let data = self.loaders[file_id].read_at(offset, length)?;
                let sample = (self.deserializer)(&data)?;
                
                if let Some(cache) = &self.sample_cache {
                    cache.lock().put(idx, sample.clone());
                }
                
                results.insert(idx, sample);
            }
        }
        
        // Collect in order
        for &idx in indices {
            if let Some(sample) = results.remove(&idx) {
                samples.push(sample);
            }
        }
        
        Ok(samples)
    }
    
    /// Create an iterator over the dataset
    pub fn iter(&self) -> LazyDatasetIterator<T> {
        LazyDatasetIterator {
            dataset: self,
            current: 0,
        }
    }
}

/// Iterator for lazy dataset
pub struct LazyDatasetIterator<'a, T> {
    dataset: &'a LazyDataset<T>,
    current: usize,
}

impl<'a, T: Clone + Send + Sync> Iterator for LazyDatasetIterator<'a, T> {
    type Item = Result<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;
        }
        
        let result = self.dataset.get(self.current);
        self.current += 1;
        Some(result)
    }
}

/// Streaming data loader with backpressure
pub struct StreamingDataLoader<T> {
    dataset: Arc<LazyDataset<T>>,
    batch_size: usize,
    buffer_size: usize,
    prefetch_factor: usize,
    shuffle: bool,
    epoch_size: Option<usize>,
}

impl<T: Clone + Send + Sync + 'static> StreamingDataLoader<T> {
    /// Create a new streaming data loader
    pub fn new(
        dataset: LazyDataset<T>,
        batch_size: usize,
        buffer_size: usize,
    ) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            buffer_size,
            prefetch_factor: 2,
            shuffle: false,
            epoch_size: None,
        }
    }
    
    /// Enable shuffling
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    
    /// Set prefetch factor
    pub fn with_prefetch(mut self, factor: usize) -> Self {
        self.prefetch_factor = factor;
        self
    }
    
    /// Set epoch size (for infinite datasets)
    pub fn with_epoch_size(mut self, size: usize) -> Self {
        self.epoch_size = Some(size);
        self
    }
    
    /// Start streaming batches
    pub fn stream(&self) -> Result<BatchStream<T>> {
        let (sender, receiver) = crossbeam_channel::bounded(self.buffer_size);
        let dataset = self.dataset.clone();
        let batch_size = self.batch_size;
        let shuffle = self.shuffle;
        let epoch_size = self.epoch_size.unwrap_or_else(|| dataset.len());
        let prefetch_factor = self.prefetch_factor;
        
        // Start background loading thread
        std::thread::spawn(move || {
            let mut indices: Vec<usize> = (0..dataset.len()).collect();
            let mut epoch = 0;
            
            loop {
                if shuffle {
                    use rand::seq::SliceRandom;
                    indices.shuffle(&mut rand::thread_rng());
                }
                
                // Process in batches
                for chunk in indices.chunks(batch_size * prefetch_factor) {
                    let mut batch_indices = Vec::new();
                    
                    for &idx in chunk {
                        batch_indices.push(idx);
                        
                        if batch_indices.len() == batch_size {
                            match dataset.get_batch(&batch_indices) {
                                Ok(batch) => {
                                    if sender.send(Ok(batch)).is_err() {
                                        return; // Receiver dropped
                                    }
                                }
                                Err(e) => {
                                    let _ = sender.send(Err(e));
                                    return;
                                }
                            }
                            batch_indices.clear();
                        }
                    }
                    
                    // Send remaining samples
                    if !batch_indices.is_empty() {
                        match dataset.get_batch(&batch_indices) {
                            Ok(batch) => {
                                if sender.send(Ok(batch)).is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = sender.send(Err(e));
                                return;
                            }
                        }
                    }
                }
                
                epoch += 1;
                if epoch_size > 0 && epoch * dataset.len() >= epoch_size {
                    break;
                }
            }
        });
        
        Ok(BatchStream { receiver })
    }
}

/// Stream of batches
pub struct BatchStream<T> {
    receiver: crossbeam_channel::Receiver<Result<Vec<T>>>,
}

impl<T> BatchStream<T> {
    /// Get next batch
    pub fn next_batch(&self) -> Option<Result<Vec<T>>> {
        self.receiver.recv().ok()
    }
    
    /// Try to get next batch without blocking
    pub fn try_next_batch(&self) -> Option<Result<Vec<T>>> {
        self.receiver.try_recv().ok()
    }
}

impl<T> Iterator for BatchStream<T> {
    type Item = Result<Vec<T>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

/// Memory-mapped array for zero-copy access
pub struct MmapArray<T> {
    mmap: Mmap,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy> MmapArray<T> {
    /// Create from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        
        if file_size % std::mem::size_of::<T>() != 0 {
            return Err(VeritasError::io_error_with_path(
                "File size not aligned with type size",
                path.to_string_lossy(),
                std::io::Error::new(std::io::ErrorKind::InvalidData, "File size not aligned with type size")
            ));
        }
        
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let len = file_size / std::mem::size_of::<T>();
        
        Ok(Self {
            mmap,
            len,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        
        unsafe {
            let ptr = self.mmap.as_ptr() as *const T;
            Some(*ptr.add(index))
        }
    }
    
    /// Get slice
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        if start > end || end > self.len {
            return None;
        }
        
        unsafe {
            let ptr = self.mmap.as_ptr() as *const T;
            Some(std::slice::from_raw_parts(ptr.add(start), end - start))
        }
    }
}

// Make it Send + Sync for parallel processing
unsafe impl<T: Send> Send for MmapArray<T> {}
unsafe impl<T: Sync> Sync for MmapArray<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_lazy_file_loader() {
        // Create test file
        let mut file = NamedTempFile::new().unwrap();
        let data = b"Hello, World! This is a test file for lazy loading.";
        file.write_all(data).unwrap();
        file.flush().unwrap();
        
        let config = LazyLoaderConfig {
            chunk_size: 10,
            ..Default::default()
        };
        
        let loader = LazyFileLoader::new(file.path(), config).unwrap();
        
        // Test reading at different offsets
        let chunk1 = loader.read_at(0, 5).unwrap();
        assert_eq!(&chunk1, b"Hello");
        
        let chunk2 = loader.read_at(7, 5).unwrap();
        assert_eq!(&chunk2, b"World");
        
        // Check cache hit
        let (hits, misses, _) = loader.stats();
        let _ = loader.read_at(0, 5).unwrap(); // Should hit cache
        let (new_hits, _, _) = loader.stats();
        assert!(new_hits > hits);
    }
    
    #[test]
    fn test_lazy_dataset() {
        // Create test files
        let mut file1 = NamedTempFile::new().unwrap();
        file1.write_all(b"sample1sample2").unwrap();
        file1.flush().unwrap();
        
        let mut file2 = NamedTempFile::new().unwrap();
        file2.write_all(b"sample3sample4").unwrap();
        file2.flush().unwrap();
        
        // Create dataset
        let mut dataset = LazyDataset::new(
            |data: &[u8]| Ok(String::from_utf8(data.to_vec()).unwrap()),
            true
        );
        
        dataset.add_file(
            file1.path(),
            vec![(0, 7), (7, 7)], // Two samples of 7 bytes each
            LazyLoaderConfig::default()
        ).unwrap();
        
        dataset.add_file(
            file2.path(),
            vec![(0, 7), (7, 7)],
            LazyLoaderConfig::default()
        ).unwrap();
        
        assert_eq!(dataset.len(), 4);
        
        // Test getting samples
        let sample1 = dataset.get(0).unwrap();
        assert_eq!(sample1, "sample1");
        
        let sample3 = dataset.get(2).unwrap();
        assert_eq!(sample3, "sample3");
        
        // Test batch loading
        let batch = dataset.get_batch(&[0, 2, 3]).unwrap();
        assert_eq!(batch, vec!["sample1", "sample3", "sample4"]);
    }
    
    #[test]
    fn test_mmap_array() {
        // Create test file with integers
        let mut file = NamedTempFile::new().unwrap();
        let data: Vec<u32> = vec![1, 2, 3, 4, 5];
        let bytes: Vec<u8> = data.iter()
            .flat_map(|&x| x.to_ne_bytes())
            .collect();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        
        let array = MmapArray::<u32>::from_file(file.path()).unwrap();
        assert_eq!(array.len(), 5);
        
        assert_eq!(array.get(0), Some(1));
        assert_eq!(array.get(4), Some(5));
        assert_eq!(array.get(5), None);
        
        let slice = array.slice(1, 4).unwrap();
        assert_eq!(slice, &[2, 3, 4]);
    }
}