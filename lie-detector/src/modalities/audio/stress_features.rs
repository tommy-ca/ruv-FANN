//! Stress Feature Detection Module
//!
//! This module implements vocal stress detection algorithms including jitter, shimmer,
//! and other prosodic features that are indicators of deception and emotional stress.

use std::collections::VecDeque;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::*;
use super::utils::*;

/// Vocal stress features for lie detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressFeatures {
    /// Pitch perturbation (jitter) measures
    pub jitter: JitterFeatures,
    
    /// Amplitude perturbation (shimmer) measures
    pub shimmer: ShimmerFeatures,
    
    /// Voice breaks and irregularities
    pub voice_breaks: VoiceBreakFeatures,
    
    /// Speaking rate and timing features
    pub temporal: TemporalFeatures,
    
    /// Spectral stress indicators
    pub spectral_stress: SpectralStressFeatures,
    
    /// Overall stress level (0.0 to 1.0)
    pub stress_level: f32,
}

/// Jitter (pitch perturbation) measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterFeatures {
    /// Absolute jitter in seconds
    pub absolute_jitter: f32,
    
    /// Relative jitter as percentage
    pub relative_jitter: f32,
    
    /// RAP (Relative Average Perturbation)
    pub rap: f32,
    
    /// PPQ5 (five-point Period Perturbation Quotient)
    pub ppq5: f32,
    
    /// DDP (Difference of Differences of Periods)
    pub ddp: f32,
}

/// Shimmer (amplitude perturbation) measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShimmerFeatures {
    /// Absolute shimmer in dB
    pub absolute_shimmer: f32,
    
    /// Relative shimmer as percentage
    pub relative_shimmer: f32,
    
    /// APQ3 (three-point Amplitude Perturbation Quotient)
    pub apq3: f32,
    
    /// APQ5 (five-point Amplitude Perturbation Quotient)
    pub apq5: f32,
    
    /// APQ11 (eleven-point Amplitude Perturbation Quotient)
    pub apq11: f32,
    
    /// DDA (Difference of Differences of Amplitudes)
    pub dda: f32,
}

/// Voice break and irregularity features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceBreakFeatures {
    /// Number of voice breaks per second
    pub break_rate: f32,
    
    /// Total duration of voice breaks
    pub break_duration: f32,
    
    /// Degree of voice irregularity
    pub irregularity_degree: f32,
    
    /// Subharmonic segments
    pub subharmonic_ratio: f32,
}

/// Temporal and timing features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Speaking rate in syllables per second
    pub speaking_rate: f32,
    
    /// Articulation rate (excluding pauses)
    pub articulation_rate: f32,
    
    /// Pause frequency
    pub pause_frequency: f32,
    
    /// Mean pause duration
    pub mean_pause_duration: f32,
    
    /// Speech-to-pause ratio
    pub speech_pause_ratio: f32,
    
    /// Rhythm variability
    pub rhythm_variability: f32,
}

/// Spectral stress indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralStressFeatures {
    /// Spectral tilt (energy distribution)
    pub spectral_tilt: f32,
    
    /// High-frequency emphasis
    pub high_freq_emphasis: f32,
    
    /// Formant frequency variations
    pub formant_variation: f32,
    
    /// Spectral entropy
    pub spectral_entropy: f32,
    
    /// Cepstral peak prominence
    pub cepstral_peak_prominence: f32,
}

/// Vocal stress detector with temporal analysis
pub struct StressDetector {
    sample_rate: u32,
    frame_size: usize,
    hop_length: usize,
    
    // Period tracking for jitter
    period_history: VecDeque<f32>,
    
    // Amplitude tracking for shimmer
    amplitude_history: VecDeque<f32>,
    
    // Voice activity tracking
    speech_segments: VecDeque<SpeechSegment>,
    
    // Configuration
    max_history: usize,
    min_period: f32,
    max_period: f32,
}

#[derive(Debug, Clone)]
struct SpeechSegment {
    start_time: f32,
    duration: f32,
    is_voiced: bool,
    fundamental_frequency: f32,
    amplitude: f32,
}

impl StressDetector {
    /// Create new stress detector
    pub fn new(config: &AudioConfig) -> Result<Self> {
        let sample_rate = config.sample_rate;
        let min_freq = config.min_frequency;
        let max_freq = config.max_frequency;
        
        Ok(Self {
            sample_rate,
            frame_size: config.chunk_size,
            hop_length: config.hop_length,
            period_history: VecDeque::with_capacity(200),
            amplitude_history: VecDeque::with_capacity(200),
            speech_segments: VecDeque::with_capacity(100),
            max_history: 200,
            min_period: sample_rate as f32 / max_freq,
            max_period: sample_rate as f32 / min_freq,
        })
    }
    
    /// Analyze vocal stress in audio chunk
    pub fn analyze_stress(
        &mut self,
        data: &[f32],
        fundamental_frequency: f32,
        voice_activity: &VoiceActivity,
    ) -> Result<StressFeatures> {
        
        // Update speech segment tracking
        self.update_speech_segments(data, fundamental_frequency, voice_activity)?;
        
        // Extract periods and amplitudes if voiced
        if voice_activity.is_speech && fundamental_frequency > 0.0 {
            self.extract_periods_and_amplitudes(data, fundamental_frequency)?;
        }
        
        // Compute jitter features
        let jitter = self.compute_jitter_features()?;
        
        // Compute shimmer features
        let shimmer = self.compute_shimmer_features()?;
        
        // Analyze voice breaks
        let voice_breaks = self.analyze_voice_breaks()?;
        
        // Analyze temporal features
        let temporal = self.analyze_temporal_features()?;
        
        // Compute spectral stress features
        let spectral_stress = self.compute_spectral_stress(data)?;
        
        // Compute overall stress level
        let stress_level = self.compute_overall_stress_level(
            &jitter, &shimmer, &voice_breaks, &temporal, &spectral_stress
        );
        
        Ok(StressFeatures {
            jitter,
            shimmer,
            voice_breaks,
            temporal,
            spectral_stress,
            stress_level,
        })
    }
    
    /// Extract period and amplitude information
    fn extract_periods_and_amplitudes(
        &mut self,
        data: &[f32],
        fundamental_frequency: f32,
    ) -> Result<()> {
        let period_samples = self.sample_rate as f32 / fundamental_frequency;
        
        if period_samples >= self.min_period && period_samples <= self.max_period {
            // Add to period history
            if self.period_history.len() >= self.max_history {
                self.period_history.pop_front();
            }
            self.period_history.push_back(period_samples);
            
            // Compute RMS amplitude for this frame
            let amplitude = rms_energy(data);
            
            // Add to amplitude history
            if self.amplitude_history.len() >= self.max_history {
                self.amplitude_history.pop_front();
            }
            self.amplitude_history.push_back(amplitude);
        }
        
        Ok(())
    }
    
    /// Update speech segment tracking
    fn update_speech_segments(
        &mut self,
        data: &[f32],
        fundamental_frequency: f32,
        voice_activity: &VoiceActivity,
    ) -> Result<()> {
        let current_time = self.speech_segments.len() as f32 * self.hop_length as f32 / self.sample_rate as f32;
        let duration = data.len() as f32 / self.sample_rate as f32;
        let amplitude = rms_energy(data);
        
        let segment = SpeechSegment {
            start_time: current_time,
            duration,
            is_voiced: voice_activity.is_speech && fundamental_frequency > 0.0,
            fundamental_frequency,
            amplitude,
        };
        
        if self.speech_segments.len() >= 1000 { // Keep last 1000 segments
            self.speech_segments.pop_front();
        }
        self.speech_segments.push_back(segment);
        
        Ok(())
    }
    
    /// Compute jitter features
    fn compute_jitter_features(&self) -> Result<JitterFeatures> {
        if self.period_history.len() < 3 {
            return Ok(JitterFeatures {
                absolute_jitter: 0.0,
                relative_jitter: 0.0,
                rap: 0.0,
                ppq5: 0.0,
                ddp: 0.0,
            });
        }
        
        let periods: Vec<f32> = self.period_history.iter().copied().collect();
        
        // Absolute jitter (average absolute difference between consecutive periods)
        let absolute_jitter = {
            let mut sum_diff = 0.0;
            for i in 1..periods.len() {
                sum_diff += (periods[i] - periods[i-1]).abs();
            }
            sum_diff / (periods.len() - 1) as f32 / self.sample_rate as f32
        };
        
        // Relative jitter (percentage)
        let mean_period = periods.iter().sum::<f32>() / periods.len() as f32;
        let relative_jitter = if mean_period > 0.0 {
            (absolute_jitter * self.sample_rate as f32 / mean_period) * 100.0
        } else {
            0.0
        };
        
        // RAP (Relative Average Perturbation)
        let rap = if periods.len() >= 3 {
            let mut sum_rap = 0.0;
            for i in 1..periods.len()-1 {
                let avg_neighbor = (periods[i-1] + periods[i] + periods[i+1]) / 3.0;
                if avg_neighbor > 0.0 {
                    sum_rap += (periods[i] - avg_neighbor).abs() / avg_neighbor;
                }
            }
            (sum_rap / (periods.len() - 2) as f32) * 100.0
        } else {
            0.0
        };
        
        // PPQ5 (five-point Period Perturbation Quotient)
        let ppq5 = if periods.len() >= 5 {
            let mut sum_ppq = 0.0;
            for i in 2..periods.len()-2 {
                let avg_5point = (periods[i-2] + periods[i-1] + periods[i] + periods[i+1] + periods[i+2]) / 5.0;
                if avg_5point > 0.0 {
                    sum_ppq += (periods[i] - avg_5point).abs() / avg_5point;
                }
            }
            (sum_ppq / (periods.len() - 4) as f32) * 100.0
        } else {
            0.0
        };
        
        // DDP (Difference of Differences of Periods)
        let ddp = if periods.len() >= 3 {
            let mut sum_ddp = 0.0;
            for i in 1..periods.len()-1 {
                let diff1 = periods[i] - periods[i-1];
                let diff2 = periods[i+1] - periods[i];
                sum_ddp += (diff2 - diff1).abs();
            }
            sum_ddp / mean_period / (periods.len() - 2) as f32 * 100.0
        } else {
            0.0
        };
        
        Ok(JitterFeatures {
            absolute_jitter,
            relative_jitter,
            rap,
            ppq5,
            ddp,
        })
    }
    
    /// Compute shimmer features
    fn compute_shimmer_features(&self) -> Result<ShimmerFeatures> {
        if self.amplitude_history.len() < 3 {
            return Ok(ShimmerFeatures {
                absolute_shimmer: 0.0,
                relative_shimmer: 0.0,
                apq3: 0.0,
                apq5: 0.0,
                apq11: 0.0,
                dda: 0.0,
            });
        }
        
        let amplitudes: Vec<f32> = self.amplitude_history.iter().copied().collect();
        
        // Convert to dB
        let db_amplitudes: Vec<f32> = amplitudes.iter()
            .map(|&a| if a > 1e-10 { 20.0 * a.log10() } else { -200.0 })
            .collect();
        
        // Absolute shimmer (average absolute difference in dB)
        let absolute_shimmer = {
            let mut sum_diff = 0.0;
            for i in 1..db_amplitudes.len() {
                sum_diff += (db_amplitudes[i] - db_amplitudes[i-1]).abs();
            }
            sum_diff / (db_amplitudes.len() - 1) as f32
        };
        
        // Relative shimmer (percentage)
        let mean_amplitude = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
        let relative_shimmer = if mean_amplitude > 0.0 {
            let mut sum_rel = 0.0;
            for i in 1..amplitudes.len() {
                sum_rel += (amplitudes[i] - amplitudes[i-1]).abs() / mean_amplitude;
            }
            (sum_rel / (amplitudes.len() - 1) as f32) * 100.0
        } else {
            0.0
        };
        
        // APQ3 (three-point Amplitude Perturbation Quotient)
        let apq3 = if db_amplitudes.len() >= 3 {
            let mut sum_apq = 0.0;
            for i in 1..db_amplitudes.len()-1 {
                let avg_3point = (db_amplitudes[i-1] + db_amplitudes[i] + db_amplitudes[i+1]) / 3.0;
                sum_apq += (db_amplitudes[i] - avg_3point).abs();
            }
            sum_apq / (db_amplitudes.len() - 2) as f32
        } else {
            0.0
        };
        
        // APQ5 (five-point Amplitude Perturbation Quotient)
        let apq5 = if db_amplitudes.len() >= 5 {
            let mut sum_apq = 0.0;
            for i in 2..db_amplitudes.len()-2 {
                let avg_5point = (db_amplitudes[i-2] + db_amplitudes[i-1] + db_amplitudes[i] 
                                + db_amplitudes[i+1] + db_amplitudes[i+2]) / 5.0;
                sum_apq += (db_amplitudes[i] - avg_5point).abs();
            }
            sum_apq / (db_amplitudes.len() - 4) as f32
        } else {
            0.0
        };
        
        // APQ11 (eleven-point Amplitude Perturbation Quotient)
        let apq11 = if db_amplitudes.len() >= 11 {
            let mut sum_apq = 0.0;
            for i in 5..db_amplitudes.len()-5 {
                let mut avg_11point = 0.0;
                for j in i-5..=i+5 {
                    avg_11point += db_amplitudes[j];
                }
                avg_11point /= 11.0;
                sum_apq += (db_amplitudes[i] - avg_11point).abs();
            }
            sum_apq / (db_amplitudes.len() - 10) as f32
        } else {
            0.0
        };
        
        // DDA (Difference of Differences of Amplitudes)
        let dda = if db_amplitudes.len() >= 3 {
            let mut sum_dda = 0.0;
            for i in 1..db_amplitudes.len()-1 {
                let diff1 = db_amplitudes[i] - db_amplitudes[i-1];
                let diff2 = db_amplitudes[i+1] - db_amplitudes[i];
                sum_dda += (diff2 - diff1).abs();
            }
            sum_dda / (db_amplitudes.len() - 2) as f32
        } else {
            0.0
        };
        
        Ok(ShimmerFeatures {
            absolute_shimmer,
            relative_shimmer,
            apq3,
            apq5,
            apq11,
            dda,
        })
    }
    
    /// Analyze voice breaks and irregularities
    fn analyze_voice_breaks(&self) -> Result<VoiceBreakFeatures> {
        if self.speech_segments.is_empty() {
            return Ok(VoiceBreakFeatures {
                break_rate: 0.0,
                break_duration: 0.0,
                irregularity_degree: 0.0,
                subharmonic_ratio: 0.0,
            });
        }
        
        let segments: Vec<&SpeechSegment> = self.speech_segments.iter().collect();
        let total_duration: f32 = segments.iter().map(|s| s.duration).sum();
        
        // Count voice breaks (transitions from voiced to unvoiced)
        let mut breaks = 0;
        let mut break_duration = 0.0;
        
        for i in 1..segments.len() {
            if segments[i-1].is_voiced && !segments[i].is_voiced {
                breaks += 1;
            }
            if !segments[i].is_voiced {
                break_duration += segments[i].duration;
            }
        }
        
        let break_rate = if total_duration > 0.0 {
            breaks as f32 / total_duration
        } else {
            0.0
        };
        
        // Compute irregularity degree based on F0 variations
        let irregularity_degree = self.compute_pitch_irregularity(&segments);
        
        // Compute subharmonic ratio (simplified)
        let subharmonic_ratio = self.compute_subharmonic_ratio(&segments);
        
        Ok(VoiceBreakFeatures {
            break_rate,
            break_duration,
            irregularity_degree,
            subharmonic_ratio,
        })
    }
    
    /// Compute pitch irregularity
    fn compute_pitch_irregularity(&self, segments: &[&SpeechSegment]) -> f32 {
        let voiced_segments: Vec<f32> = segments.iter()
            .filter(|s| s.is_voiced && s.fundamental_frequency > 0.0)
            .map(|s| s.fundamental_frequency)
            .collect();
        
        if voiced_segments.len() < 2 {
            return 0.0;
        }
        
        // Compute coefficient of variation for F0
        let mean_f0 = voiced_segments.iter().sum::<f32>() / voiced_segments.len() as f32;
        let variance = voiced_segments.iter()
            .map(|&f0| (f0 - mean_f0).powi(2))
            .sum::<f32>() / voiced_segments.len() as f32;
        
        if mean_f0 > 0.0 {
            variance.sqrt() / mean_f0
        } else {
            0.0
        }
    }
    
    /// Compute subharmonic ratio (simplified)
    fn compute_subharmonic_ratio(&self, segments: &[&SpeechSegment]) -> f32 {
        let voiced_segments: Vec<&SpeechSegment> = segments.iter()
            .filter(|s| s.is_voiced && s.fundamental_frequency > 0.0)
            .copied()
            .collect();
        
        if voiced_segments.len() < 2 {
            return 0.0;
        }
        
        // Count segments with potential subharmonics (F0 jumps by factor of 2)
        let mut subharmonic_count = 0;
        
        for i in 1..voiced_segments.len() {
            let f0_ratio = voiced_segments[i].fundamental_frequency / voiced_segments[i-1].fundamental_frequency;
            if (f0_ratio - 0.5).abs() < 0.1 || (f0_ratio - 2.0).abs() < 0.2 {
                subharmonic_count += 1;
            }
        }
        
        subharmonic_count as f32 / voiced_segments.len() as f32
    }
    
    /// Analyze temporal features
    fn analyze_temporal_features(&self) -> Result<TemporalFeatures> {
        if self.speech_segments.is_empty() {
            return Ok(TemporalFeatures {
                speaking_rate: 0.0,
                articulation_rate: 0.0,
                pause_frequency: 0.0,
                mean_pause_duration: 0.0,
                speech_pause_ratio: 0.0,
                rhythm_variability: 0.0,
            });
        }
        
        let segments: Vec<&SpeechSegment> = self.speech_segments.iter().collect();
        let total_duration: f32 = segments.iter().map(|s| s.duration).sum();
        
        // Speech and pause durations
        let speech_duration: f32 = segments.iter()
            .filter(|s| s.is_voiced)
            .map(|s| s.duration)
            .sum();
        
        let pause_segments: Vec<f32> = segments.iter()
            .filter(|s| !s.is_voiced)
            .map(|s| s.duration)
            .collect();
        
        let pause_duration: f32 = pause_segments.iter().sum();
        
        // Estimate syllable count (simplified - based on voiced segments)
        let syllable_count = segments.iter()
            .filter(|s| s.is_voiced)
            .count() as f32 / 2.0; // Rough approximation
        
        // Speaking rate (including pauses)
        let speaking_rate = if total_duration > 0.0 {
            syllable_count / total_duration
        } else {
            0.0
        };
        
        // Articulation rate (excluding pauses)
        let articulation_rate = if speech_duration > 0.0 {
            syllable_count / speech_duration
        } else {
            0.0
        };
        
        // Pause frequency
        let pause_frequency = if total_duration > 0.0 {
            pause_segments.len() as f32 / total_duration
        } else {
            0.0
        };
        
        // Mean pause duration
        let mean_pause_duration = if !pause_segments.is_empty() {
            pause_duration / pause_segments.len() as f32
        } else {
            0.0
        };
        
        // Speech to pause ratio
        let speech_pause_ratio = if pause_duration > 0.0 {
            speech_duration / pause_duration
        } else {
            f32::INFINITY
        };
        
        // Rhythm variability (coefficient of variation of segment durations)
        let rhythm_variability = self.compute_rhythm_variability(&segments);
        
        Ok(TemporalFeatures {
            speaking_rate,
            articulation_rate,
            pause_frequency,
            mean_pause_duration,
            speech_pause_ratio,
            rhythm_variability,
        })
    }
    
    /// Compute rhythm variability
    fn compute_rhythm_variability(&self, segments: &[&SpeechSegment]) -> f32 {
        let durations: Vec<f32> = segments.iter().map(|s| s.duration).collect();
        
        if durations.len() < 2 {
            return 0.0;
        }
        
        let mean_duration = durations.iter().sum::<f32>() / durations.len() as f32;
        let variance = durations.iter()
            .map(|&d| (d - mean_duration).powi(2))
            .sum::<f32>() / durations.len() as f32;
        
        if mean_duration > 0.0 {
            variance.sqrt() / mean_duration
        } else {
            0.0
        }
    }
    
    /// Compute spectral stress features
    fn compute_spectral_stress(&self, data: &[f32]) -> Result<SpectralStressFeatures> {
        // Simplified spectral analysis - in real implementation would use FFT
        
        // Spectral tilt (energy in high vs low frequencies)
        let spectral_tilt = self.compute_spectral_tilt(data);
        
        // High-frequency emphasis
        let high_freq_emphasis = self.compute_high_freq_emphasis(data);
        
        // Formant variation (simplified)
        let formant_variation = 0.0; // Would require formant tracking
        
        // Spectral entropy
        let spectral_entropy = self.compute_spectral_entropy(data);
        
        // Cepstral peak prominence (simplified)
        let cepstral_peak_prominence = 0.0; // Would require cepstral analysis
        
        Ok(SpectralStressFeatures {
            spectral_tilt,
            high_freq_emphasis,
            formant_variation,
            spectral_entropy,
            cepstral_peak_prominence,
        })
    }
    
    /// Compute spectral tilt (simplified)
    fn compute_spectral_tilt(&self, data: &[f32]) -> f32 {
        // Split into low and high frequency bands (simplified)
        let mid_point = data.len() / 2;
        let low_energy: f32 = data[..mid_point].iter().map(|x| x * x).sum();
        let high_energy: f32 = data[mid_point..].iter().map(|x| x * x).sum();
        
        if low_energy > 0.0 {
            10.0 * (high_energy / low_energy).log10()
        } else {
            0.0
        }
    }
    
    /// Compute high-frequency emphasis
    fn compute_high_freq_emphasis(&self, data: &[f32]) -> f32 {
        // Apply high-pass filter (simplified difference filter)
        let mut filtered_energy = 0.0;
        for i in 1..data.len() {
            let diff = data[i] - data[i-1];
            filtered_energy += diff * diff;
        }
        
        let total_energy: f32 = data.iter().map(|x| x * x).sum();
        
        if total_energy > 0.0 {
            filtered_energy / total_energy
        } else {
            0.0
        }
    }
    
    /// Compute spectral entropy (simplified)
    fn compute_spectral_entropy(&self, data: &[f32]) -> f32 {
        // Simplified using time-domain energy distribution
        let frame_size = 64;
        let num_frames = data.len() / frame_size;
        
        if num_frames < 2 {
            return 0.0;
        }
        
        let mut frame_energies = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let start = i * frame_size;
            let end = (start + frame_size).min(data.len());
            let energy: f32 = data[start..end].iter().map(|x| x * x).sum();
            frame_energies.push(energy);
        }
        
        let total_energy: f32 = frame_energies.iter().sum();
        if total_energy == 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for energy in frame_energies {
            if energy > 0.0 {
                let prob = energy / total_energy;
                entropy -= prob * prob.log2();
            }
        }
        
        entropy
    }
    
    /// Compute overall stress level
    fn compute_overall_stress_level(
        &self,
        jitter: &JitterFeatures,
        shimmer: &ShimmerFeatures,
        voice_breaks: &VoiceBreakFeatures,
        temporal: &TemporalFeatures,
        spectral_stress: &SpectralStressFeatures,
    ) -> f32 {
        // Weighted combination of stress indicators
        let mut stress_score = 0.0;
        let mut weight_sum = 0.0;
        
        // Jitter contribution (high jitter indicates stress)
        let jitter_stress = (jitter.relative_jitter / 2.0).min(1.0); // Normalize to 0-1
        stress_score += jitter_stress * 0.25;
        weight_sum += 0.25;
        
        // Shimmer contribution
        let shimmer_stress = (shimmer.relative_shimmer / 10.0).min(1.0); // Normalize to 0-1
        stress_score += shimmer_stress * 0.25;
        weight_sum += 0.25;
        
        // Voice breaks contribution
        let break_stress = (voice_breaks.break_rate * 10.0).min(1.0) + 
                          voice_breaks.irregularity_degree.min(1.0);
        stress_score += (break_stress / 2.0) * 0.2;
        weight_sum += 0.2;
        
        // Temporal features contribution
        let temporal_stress = if temporal.speaking_rate > 0.0 {
            // Fast speaking or irregular rhythm indicates stress
            let rate_stress = ((temporal.speaking_rate - 3.0) / 3.0).abs().min(1.0);
            let rhythm_stress = temporal.rhythm_variability.min(1.0);
            (rate_stress + rhythm_stress) / 2.0
        } else {
            0.0
        };
        stress_score += temporal_stress * 0.15;
        weight_sum += 0.15;
        
        // Spectral stress contribution
        let spectral_stress_level = (spectral_stress.high_freq_emphasis * 2.0).min(1.0);
        stress_score += spectral_stress_level * 0.15;
        weight_sum += 0.15;
        
        if weight_sum > 0.0 {
            stress_score / weight_sum
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stress_detector_creation() {
        let config = AudioConfig::default();
        let detector = StressDetector::new(&config);
        assert!(detector.is_ok());
    }
    
    #[test]
    fn test_jitter_computation() {
        let config = AudioConfig::default();
        let mut detector = StressDetector::new(&config).unwrap();
        
        // Add some periods with jitter
        detector.period_history.extend([100.0, 102.0, 98.0, 101.0, 99.0]);
        
        let jitter = detector.compute_jitter_features().unwrap();
        
        assert!(jitter.absolute_jitter > 0.0);
        assert!(jitter.relative_jitter > 0.0);
    }
    
    #[test]
    fn test_shimmer_computation() {
        let config = AudioConfig::default();
        let mut detector = StressDetector::new(&config).unwrap();
        
        // Add some amplitudes with shimmer
        detector.amplitude_history.extend([1.0, 1.1, 0.9, 1.05, 0.95]);
        
        let shimmer = detector.compute_shimmer_features().unwrap();
        
        assert!(shimmer.relative_shimmer > 0.0);
    }
    
    #[test]
    fn test_stress_level_computation() {
        let config = AudioConfig::default();
        let detector = StressDetector::new(&config).unwrap();
        
        let jitter = JitterFeatures {
            absolute_jitter: 0.001,
            relative_jitter: 1.0,
            rap: 1.0,
            ppq5: 1.0,
            ddp: 1.0,
        };
        
        let shimmer = ShimmerFeatures {
            absolute_shimmer: 0.5,
            relative_shimmer: 5.0,
            apq3: 0.5,
            apq5: 0.5,
            apq11: 0.5,
            dda: 0.5,
        };
        
        let voice_breaks = VoiceBreakFeatures {
            break_rate: 0.1,
            break_duration: 0.05,
            irregularity_degree: 0.2,
            subharmonic_ratio: 0.1,
        };
        
        let temporal = TemporalFeatures {
            speaking_rate: 4.0,
            articulation_rate: 5.0,
            pause_frequency: 0.5,
            mean_pause_duration: 0.2,
            speech_pause_ratio: 4.0,
            rhythm_variability: 0.3,
        };
        
        let spectral_stress = SpectralStressFeatures {
            spectral_tilt: -5.0,
            high_freq_emphasis: 0.3,
            formant_variation: 0.1,
            spectral_entropy: 2.5,
            cepstral_peak_prominence: 10.0,
        };
        
        let stress_level = detector.compute_overall_stress_level(
            &jitter, &shimmer, &voice_breaks, &temporal, &spectral_stress
        );
        
        assert!(stress_level >= 0.0);
        assert!(stress_level <= 1.0);
    }
}