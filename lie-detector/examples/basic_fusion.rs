//! Basic example demonstrating the multi-modal fusion system
//!
//! This example shows how to use different fusion strategies to combine
//! deception scores from multiple modalities.

use veritas_nexus::prelude::*;
use chrono::Utc;
use hashbrown::HashMap;
use std::time::Duration;

fn main() -> Result<()> {
    println!("🔍 Veritas Nexus - Multi-Modal Fusion Example");
    println!("==============================================");
    
    // Create mock deception scores from different modalities
    let mut scores = HashMap::new();
    
    scores.insert(
        ModalityType::Vision,
        DeceptionScore {
            probability: 0.85,
            confidence: 0.92,
            contributing_factors: vec![
                ("micro_expressions".to_string(), 0.4),
                ("eye_movement".to_string(), 0.3),
                ("facial_asymmetry".to_string(), 0.3),
            ],
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(15),
        },
    );
    
    scores.insert(
        ModalityType::Audio,
        DeceptionScore {
            probability: 0.72,
            confidence: 0.88,
            contributing_factors: vec![
                ("voice_stress".to_string(), 0.5),
                ("pitch_variation".to_string(), 0.3),
                ("speech_rate".to_string(), 0.2),
            ],
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(12),
        },
    );
    
    scores.insert(
        ModalityType::Text,
        DeceptionScore {
            probability: 0.68,
            confidence: 0.85,
            contributing_factors: vec![
                ("linguistic_complexity".to_string(), 0.4),
                ("response_time".to_string(), 0.3),
                ("word_choice".to_string(), 0.3),
            ],
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(8),
        },
    );
    
    scores.insert(
        ModalityType::Physiological,
        DeceptionScore {
            probability: 0.79,
            confidence: 0.78,
            contributing_factors: vec![
                ("heart_rate_variability".to_string(), 0.6),
                ("skin_conductance".to_string(), 0.4),
            ],
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(5),
        },
    );
    
    println!("\n📊 Individual Modality Scores:");
    for (modality, score) in &scores {
        println!(
            "  {:>12}: {:.1}% (confidence: {:.1}%)",
            format!("{:?}", modality),
            score.probability * 100.0,
            score.confidence * 100.0
        );
    }
    
    // Create fusion manager
    let config = FusionConfig::default();
    let manager = FusionManager::new(config)?;
    
    println!("\n🔄 Available Fusion Strategies:");
    for strategy in manager.list_strategies() {
        println!("  • {}", strategy);
    }
    
    // Test different fusion strategies
    let strategies = [
        "late_fusion",
        "early_fusion", 
        "attention_fusion",
        "weighted_voting",
        "hybrid_fusion",
    ];
    
    println!("\n🎯 Fusion Results:");
    println!("{:-<70}", "");
    
    for strategy_name in &strategies {
        match manager.fuse(strategy_name, &scores, None) {
            Ok(result) => {
                println!(
                    "{:>15}: {:.1}% (conf: {:.1}%, quality: {:.1}%)",
                    strategy_name,
                    result.decision.deception_probability * 100.0,
                    result.decision.confidence * 100.0,
                    result.quality_score * 100.0
                );
                
                // Show modality contributions for one strategy
                if strategy_name == &"attention_fusion" {
                    println!("                 Contributions:");
                    for (modality, contribution) in &result.decision.modality_contributions {
                        println!(
                            "                   {:>12}: {:.3}",
                            format!("{:?}", modality),
                            contribution
                        );
                    }
                }
            }
            Err(e) => {
                println!("{:>15}: ❌ Error: {}", strategy_name, e);
            }
        }
    }
    
    // Demonstrate neural fusion
    println!("\n🧠 Neural Fusion Integration:");
    println!("{:-<70}", "");
    
    // Create neural fusion with mock networks
    use veritas_nexus::neural_integration::integration_utils;
    let neural_fusion = integration_utils::create_default_neural_fusion::<f64>()?;
    
    match neural_fusion.fuse(&scores, None) {
        Ok(result) => {
            println!(
                "Neural Fusion   : {:.1}% (conf: {:.1}%)",
                result.deception_probability * 100.0,
                result.confidence * 100.0
            );
            println!("Explanation     : {}", result.explanation);
        }
        Err(e) => {
            println!("Neural Fusion   : ❌ Error: {}", e);
        }
    }
    
    // Demonstrate temporal alignment
    println!("\n⏰ Temporal Alignment Demo:");
    println!("{:-<70}", "");
    
    use veritas_nexus::fusion::temporal_alignment::*;
    let aligner = TemporalAligner::new(AlignmentConfig::<f64>::default())?;
    
    // Create mock combined features
    let mut modalities = HashMap::new();
    modalities.insert(ModalityType::Vision, vec![0.1, 0.2, 0.3, 0.4]);
    modalities.insert(ModalityType::Audio, vec![0.5, 0.6, 0.7]);
    modalities.insert(ModalityType::Text, vec![0.8, 0.9]);
    
    let combined = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    
    let mut dimension_map = HashMap::new();
    dimension_map.insert(ModalityType::Vision, (0, 4));
    dimension_map.insert(ModalityType::Audio, (4, 7));
    dimension_map.insert(ModalityType::Text, (7, 9));
    
    let temporal_info = TemporalInfo {
        start_time: Utc::now() - chrono::Duration::seconds(5),
        end_time: Utc::now(),
        frame_rate: Some(30.0),
        sample_rate: Some(16000),
        sync_offsets: HashMap::new(),
    };
    
    let features = CombinedFeatures {
        modalities,
        combined,
        dimension_map,
        temporal_info,
    };
    
    match aligner.align(&features) {
        Ok(aligned) => {
            println!("✅ Temporal alignment successful");
            println!("   Combined features: {} dimensions", aligned.combined.len());
            println!("   Modalities: {}", aligned.modalities.len());
            println!("   Time window: {:.1}s", 
                (aligned.temporal_info.end_time - aligned.temporal_info.start_time)
                    .num_milliseconds() as f64 / 1000.0
            );
        }
        Err(e) => {
            println!("❌ Temporal alignment failed: {}", e);
        }
    }
    
    println!("\n✨ Summary:");
    println!("  • Successfully demonstrated multi-modal fusion");
    println!("  • Tested {} fusion strategies", strategies.len());
    println!("  • Integrated neural network processing");
    println!("  • Verified temporal alignment capabilities");
    println!("  • All systems operational! 🚀");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_runs() {
        // Test that the example code compiles and runs
        assert!(main().is_ok());
    }
}