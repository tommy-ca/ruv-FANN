//! Common utility functions and helpers.

use std::time::{Duration, Instant};

/// Timer utility for performance measurement.
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Create a new timer and start timing.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    /// Get elapsed time since creation.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }
    
    /// Get elapsed time in microseconds.
    pub fn elapsed_us(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Common mathematical utilities.
pub mod math {
    /// Fast inverse square root approximation.
    pub fn fast_inv_sqrt(x: f32) -> f32 {
        let i = x.to_bits();
        let i = 0x5f3759df - (i >> 1);
        let y = f32::from_bits(i);
        y * (1.5 - 0.5 * x * y * y)
    }
    
    /// Clamp value between min and max.
    pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
        value.max(min).min(max)
    }
    
    /// Linear interpolation.
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timer() {
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
    
    #[test]
    fn test_math_utils() {
        assert!((math::fast_inv_sqrt(4.0) - 0.5).abs() < 0.1);
        assert_eq!(math::clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(math::clamp(-1.0, 0.0, 10.0), 0.0);
        assert_eq!(math::clamp(15.0, 0.0, 10.0), 10.0);
        assert_eq!(math::lerp(0.0, 10.0, 0.5), 5.0);
    }
}