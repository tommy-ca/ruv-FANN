// Test basic functionality of Veritas Nexus
use std::collections::HashMap;

// Mock the basic types for testing
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub decision: String,
    pub confidence: f64,
    pub reasoning: Vec<String>,
}

#[derive(Debug, Clone)]  
pub struct TextInput {
    pub content: String,
}

// Mock analyzer
pub struct BasicTextAnalyzer;

impl BasicTextAnalyzer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn analyze(&self, input: &TextInput) -> AnalysisResult {
        // Simple rule-based analysis for testing
        let content = input.content.to_lowercase();
        
        let deception_indicators = vec![
            "definitely", "absolutely", "never", "always", "swear"
        ];
        
        let matches: usize = deception_indicators.iter()
            .map(|&indicator| content.matches(indicator).count())
            .sum();
            
        let confidence = (matches as f64) * 0.3;
        let is_deceptive = confidence > 0.5;
        
        AnalysisResult {
            decision: if is_deceptive { "Deceptive".to_string() } else { "Truthful".to_string() },
            confidence,
            reasoning: vec![
                format!("Found {} deception indicators", matches),
                format!("Confidence calculation: {} * 0.3 = {}", matches, confidence)
            ]
        }
    }
}

fn main() {
    println!("🧪 Testing Basic Veritas Nexus Functionality");
    
    let analyzer = BasicTextAnalyzer::new();
    
    // Test cases
    let test_cases = vec![
        ("I went to the store", "Normal statement"),
        ("I definitely never took the money", "High deception indicators"),
        ("I absolutely swear I always tell the truth", "Multiple indicators"),
        ("The weather is nice today", "Low deception indicators"),
    ];
    
    println!("\n📊 Analysis Results:");
    println!("{:-<80}", "");
    
    for (text, description) in test_cases {
        let input = TextInput { content: text.to_string() };
        let result = analyzer.analyze(&input);
        
        println!("Text: \"{}\"", text);
        println!("Description: {}", description);
        println!("Decision: {} (Confidence: {:.1}%)", result.decision, result.confidence * 100.0);
        println!("Reasoning:");
        for reason in &result.reasoning {
            println!("  - {}", reason);
        }
        println!("{:-<80}", "");
    }
    
    println!("✅ Basic functionality test completed!");
}