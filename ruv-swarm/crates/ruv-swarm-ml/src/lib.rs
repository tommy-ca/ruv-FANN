//! Machine learning integration for RUV Swarm
//!
//! This crate provides neural forecasting capabilities for agent-specific time series prediction,
//! ensemble methods, and swarm-level forecasting coordination.

#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(unused_imports)] // TODO: Remove when implementation is complete

extern crate alloc;

#[cfg(target_arch = "wasm32")]
use alloc::{
    boxed::Box,
    collections::HashMap,
    format,
    string::{String, ToString},
    vec::Vec,
};

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;

pub mod agent_forecasting;
pub mod ensemble;
pub mod models;
pub mod time_series;

#[cfg(target_arch = "wasm32")]
pub mod wasm_bindings;

// Re-export main types
pub use agent_forecasting::{
    AgentForecastContext, AgentForecastingManager, ForecastDomain, ForecastRequirements,
    ModelSpecialization,
};

pub use ensemble::{EnsembleConfig, EnsembleForecaster, EnsembleStrategy, OptimizationMetric};

pub use models::{ForecastModel, ModelFactory, ModelType};

pub use time_series::{TimeSeriesData, TimeSeriesProcessor, TransformationType};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindings::*;
