/// Unit tests for all veritas-nexus modules
/// 
/// This module organizes unit tests by component, ensuring comprehensive
/// coverage of individual functions and methods in isolation

pub mod modalities;
pub mod fusion;
pub mod agents;
pub mod neural;
pub mod streaming;
pub mod utils;
pub mod optimization;
pub mod reasoning;

// Re-export common test utilities
pub use crate::common::*;