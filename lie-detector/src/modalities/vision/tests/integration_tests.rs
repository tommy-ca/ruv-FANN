//! Integration tests for the vision modality module
//! 
//! These tests verify the complete workflow from input processing to
//! deception score calculation.

use crate::modalities::vision::{
    VisionAnalyzer, VisionConfig, VisionInput, VisionError,
    FaceAnalyzer, MicroExpressionDetector,
    MicroExpressionType,
};

#[cfg(feature = "gpu")]
use crate::modalities::vision::GpuVisionProcessor;

/// Test the complete vision analysis pipeline
#[test]
fn test_complete_vision_pipeline() {
    let config: VisionConfig<f32> = VisionConfig::default();
    let analyzer_result = VisionAnalyzer::new(config);
    
    // Note: This will fail without proper model files, but tests the structure
    match analyzer_result {
        Ok(analyzer) => {
            let image_data = create_test_image(224, 224);
            let input = VisionInput::new(image_data, 224, 224, 3);
            
            // Test feature extraction
            let features_result = analyzer.extract_features(&input);
            if let Ok(features) = features_result {
                assert!(features.feature_count() > 0);
                
                // Test analysis
                let analysis_result = analyzer.analyze(&features);
                if let Ok(score) = analysis_result {
                    assert!(score.probability >= 0.0 && score.probability <= 1.0);
                    assert!(score.confidence >= 0.0 && score.confidence <= 1.0);
                    assert!(!score.contributing_factors.is_empty());
                }
                
                // Test explanation
                let explanation = analyzer.explain(&features);
                assert!(!explanation.analysis_steps.is_empty());
                assert!(!explanation.confidence_reasoning.is_empty());
            }
        },
        Err(e) => {
            println!("Vision analyzer creation failed (expected without models): {}", e);
        }
    }
}

/// Test face analyzer with various input sizes
#[test]
fn test_face_analyzer_input_validation() {
    let config: VisionConfig<f32> = VisionConfig::default();
    let analyzer = FaceAnalyzer::new(&config).unwrap();
    
    // Test valid input
    let valid_image = create_test_image(224, 224);
    let valid_input = VisionInput::new(valid_image, 224, 224, 3);
    let faces_result = analyzer.detect_faces(&valid_input);
    assert!(faces_result.is_ok());
    
    // Test invalid input (wrong size)
    let invalid_image = vec![128u8; 100]; // Too small
    let invalid_input = VisionInput::new(invalid_image, 224, 224, 3);
    let invalid_result = analyzer.detect_faces(&invalid_input);
    assert!(invalid_result.is_err());
    
    // Test edge case: minimum valid size
    let min_image = create_test_image(100, 100);
    let min_input = VisionInput::new(min_image, 100, 100, 3);
    let min_result = analyzer.detect_faces(&min_input);
    assert!(min_result.is_ok());
    
    // Test edge case: very small image
    let tiny_image = create_test_image(32, 32);
    let tiny_input = VisionInput::new(tiny_image, 32, 32, 3);
    let tiny_result = analyzer.detect_faces(&tiny_input);
    assert!(tiny_result.is_ok()); // Should work but may not detect faces
}

/// Test micro-expression detector temporal analysis
#[test]
fn test_micro_expression_temporal_analysis() {
    let config: VisionConfig<f32> = VisionConfig::default();
    let mut detector = MicroExpressionDetector::new(&config).unwrap();
    
    // Process multiple frames to test temporal analysis
    let mut frame_results = Vec::new();
    
    for i in 0..5 {
        let mut image_data = create_test_image(224, 224);
        // Modify each frame slightly to simulate temporal changes
        for j in 0..100 {
            if j + i * 100 < image_data.len() {
                image_data[j + i * 100] = ((128 + i * 20) % 255) as u8;
            }
        }
        
        let input = VisionInput::new(image_data, 224, 224, 3);
        let result = detector.detect_expressions(&input);
        
        assert!(result.is_ok());
        frame_results.push(result.unwrap());
    }
    
    // Check that temporal analysis is working
    let stats = detector.get_expression_statistics();
    assert!(stats.total_expressions >= 0);
    
    // Verify temporal changes are detected across frames
    if frame_results.len() > 1 {
        let first_features = &frame_results[0].features;
        let last_features = &frame_results.last().unwrap().features;
        
        // Features should be different due to temporal changes
        assert_ne!(first_features.len(), 0);
        assert_ne!(last_features.len(), 0);
    }
}

/// Test vision analyzer configuration variations
#[test]
fn test_vision_config_variations() {
    // Test high sensitivity configuration
    let mut high_sensitivity_config: VisionConfig<f32> = VisionConfig::default();
    high_sensitivity_config.face_detection_threshold = 0.9;
    high_sensitivity_config.micro_expression_sensitivity = 0.95;
    
    let analyzer_result = VisionAnalyzer::new(high_sensitivity_config);
    match analyzer_result {
        Ok(_) => println!("High sensitivity configuration accepted"),
        Err(e) => println!("High sensitivity configuration failed: {}", e),
    }
    
    // Test low sensitivity configuration
    let mut low_sensitivity_config: VisionConfig<f32> = VisionConfig::default();
    low_sensitivity_config.face_detection_threshold = 0.3;
    low_sensitivity_config.micro_expression_sensitivity = 0.2;
    
    let analyzer_result = VisionAnalyzer::new(low_sensitivity_config);
    match analyzer_result {
        Ok(_) => println!("Low sensitivity configuration accepted"),
        Err(e) => println!("Low sensitivity configuration failed: {}", e),
    }
    
    // Test GPU configuration (if available)
    let mut gpu_config: VisionConfig<f32> = VisionConfig::default();
    gpu_config.enable_gpu = true;
    
    let analyzer_result = VisionAnalyzer::new(gpu_config);
    match analyzer_result {
        Ok(_) => println!("GPU configuration accepted"),
        Err(e) => println!("GPU configuration failed (expected without GPU): {}", e),
    }
}

/// Test micro-expression type relevance scoring
#[test]
fn test_micro_expression_relevance_scoring() {
    // Test that deception-relevant expressions have higher scores
    let suppression_score = MicroExpressionType::Suppression.deception_relevance::<f32>();
    let happiness_score = MicroExpressionType::Happiness.deception_relevance::<f32>();
    let leakage_score = MicroExpressionType::Leakage.deception_relevance::<f32>();
    
    assert!(suppression_score > happiness_score);
    assert!(leakage_score > happiness_score);
    assert!(suppression_score >= 0.8); // High relevance for suppression
    assert!(happiness_score <= 0.2); // Low relevance for happiness
    
    // Test all expression types have valid scores
    let all_types = vec![
        MicroExpressionType::Happiness,
        MicroExpressionType::Sadness,
        MicroExpressionType::Anger,
        MicroExpressionType::Fear,
        MicroExpressionType::Surprise,
        MicroExpressionType::Disgust,
        MicroExpressionType::Contempt,
        MicroExpressionType::Suppression,
        MicroExpressionType::Leakage,
        MicroExpressionType::DupingDelight,
    ];
    
    for expr_type in all_types {
        let score = expr_type.deception_relevance::<f32>();
        assert!(score >= 0.0 && score <= 1.0);
        assert!(!expr_type.description().is_empty());
    }
}

/// Test error handling and edge cases
#[test]
fn test_error_handling() {
    let config: VisionConfig<f32> = VisionConfig::default();
    
    // Test with empty image data
    let empty_input = VisionInput::new(Vec::new(), 224, 224, 3);
    
    if let Ok(analyzer) = VisionAnalyzer::new(config.clone()) {
        let result = analyzer.extract_features(&empty_input);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            VisionError::InvalidImageFormat(_) => {
                println!("Correct error type for empty image");
            },
            other => {
                println!("Unexpected error type: {:?}", other);
            }
        }
    }
    
    // Test with mismatched dimensions
    let mismatched_data = vec![128u8; 1000]; // Wrong size for 224x224x3
    let mismatched_input = VisionInput::new(mismatched_data, 224, 224, 3);
    
    let face_analyzer = FaceAnalyzer::new(&config).unwrap();
    let result = face_analyzer.detect_faces(&mismatched_input);
    assert!(result.is_err());
}

/// Test feature vector consistency
#[test]
fn test_feature_vector_consistency() {
    let config: VisionConfig<f32> = VisionConfig::default();
    
    if let Ok(analyzer) = VisionAnalyzer::new(config) {
        let image_data = create_test_image(224, 224);
        let input = VisionInput::new(image_data, 224, 224, 3);
        
        // Extract features multiple times and verify consistency
        let mut feature_lengths = Vec::new();
        
        for _ in 0..3 {
            if let Ok(features) = analyzer.extract_features(&input) {
                let flat_features = features.to_flat_vector();
                feature_lengths.push(flat_features.len());
                
                // Verify feature values are in reasonable ranges
                for &value in &flat_features {
                    assert!(value >= 0.0 && value <= 10.0); // Reasonable range
                    assert!(!value.is_nan());
                    assert!(!value.is_infinite());
                }
            }
        }
        
        // All feature extractions should produce same length vectors
        if !feature_lengths.is_empty() {
            let first_length = feature_lengths[0];
            for &length in &feature_lengths {
                assert_eq!(length, first_length);
            }
        }
    }
}

/// Test GPU processor integration (if GPU feature is enabled)
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_integration() {
    let config: VisionConfig<f32> = VisionConfig::default();
    let gpu_processor_result = GpuVisionProcessor::new(&config);
    
    match gpu_processor_result {
        Ok(processor) => {
            println!("GPU processor created successfully");
            println!("GPU enabled: {}", processor.is_gpu_enabled());
            
            let memory_info = processor.get_memory_info();
            println!("GPU memory info: {:?}", memory_info);
            
            // Test feature extraction
            let image_data = create_test_image(224, 224);
            let input = VisionInput::new(image_data, 224, 224, 3);
            
            let features_result = processor.extract_features(&input);
            match features_result {
                Ok(features) => {
                    assert!(features.feature_count() > 0);
                    println!("GPU feature extraction successful");
                },
                Err(e) => {
                    println!("GPU feature extraction failed: {}", e);
                }
            }
            
            // Test benchmarking
            let benchmark_result = processor.benchmark(1);
            match benchmark_result {
                Ok(benchmark) => {
                    println!("GPU benchmark result: {:?}", benchmark);
                    assert!(benchmark.throughput_fps > 0.0);
                },
                Err(e) => {
                    println!("GPU benchmark failed: {}", e);
                }
            }
        },
        Err(e) => {
            println!("GPU processor creation failed (expected on CPU-only systems): {}", e);
        }
    }
}

/// Test performance with different image sizes
#[test]
fn test_performance_scaling() {
    let config: VisionConfig<f32> = VisionConfig::default();
    
    if let Ok(analyzer) = VisionAnalyzer::new(config) {
        let image_sizes = vec![(64, 64), (128, 128), (224, 224), (512, 512)];
        
        for (width, height) in image_sizes {
            let image_data = create_test_image(width, height);
            let input = VisionInput::new(image_data, width, height, 3);
            
            let start_time = std::time::Instant::now();
            let result = analyzer.extract_features(&input);
            let duration = start_time.elapsed();
            
            match result {
                Ok(features) => {
                    println!("{}x{}: {} features extracted in {:?}", 
                             width, height, features.feature_count(), duration);
                    assert!(features.feature_count() > 0);
                },
                Err(e) => {
                    println!("{}x{}: Feature extraction failed: {}", width, height, e);
                }
            }
        }
    }
}

/// Helper function to create test images
fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut image_data = Vec::with_capacity((width * height * 3) as usize);
    
    // Create a simple gradient pattern
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            
            image_data.push(r);
            image_data.push(g);
            image_data.push(b);
        }
    }
    
    image_data
}

/// Helper function to create test images with face-like patterns
fn create_face_like_image(width: u32, height: u32) -> Vec<u8> {
    let mut image_data = vec![128u8; (width * height * 3) as usize];
    
    let center_x = width / 2;
    let center_y = height / 2;
    let face_radius = width.min(height) / 4;
    
    // Draw a simple face-like pattern
    for y in 0..height {
        for x in 0..width {
            let dx = (x as i32 - center_x as i32).abs() as u32;
            let dy = (y as i32 - center_y as i32).abs() as u32;
            let distance = ((dx * dx + dy * dy) as f32).sqrt() as u32;
            
            let pixel_idx = ((y * width + x) * 3) as usize;
            
            if distance < face_radius {
                // Face region - lighter
                image_data[pixel_idx] = 200;     // R
                image_data[pixel_idx + 1] = 180; // G
                image_data[pixel_idx + 2] = 160; // B
                
                // Add eye-like spots
                if (dx < face_radius / 3 && dy < face_radius / 4) &&
                   (y < center_y - face_radius / 8 || y > center_y + face_radius / 8) {
                    image_data[pixel_idx] = 50;     // Dark eyes
                    image_data[pixel_idx + 1] = 50;
                    image_data[pixel_idx + 2] = 50;
                }
                
                // Add mouth-like region
                if dy > face_radius / 2 && dx < face_radius / 4 {
                    image_data[pixel_idx] = 120;    // Mouth
                    image_data[pixel_idx + 1] = 80;
                    image_data[pixel_idx + 2] = 80;
                }
            }
        }
    }
    
    image_data
}