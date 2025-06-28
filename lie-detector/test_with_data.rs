// Test with actual sample data from examples/data/samples.csv
use std::fs;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SampleData {
    pub id: String,
    pub text: String,
    pub actual_label: String,
    pub confidence: f64,
    pub environment: String,
    pub age: u32,
}

#[derive(Debug)]
pub struct AnalysisResult {
    pub predicted_label: String,
    pub confidence: f64,
    pub reasoning: Vec<String>,
    pub accuracy: bool,
}

pub struct LieDetector {
    deception_keywords: Vec<&'static str>,
    hedging_words: Vec<&'static str>,
}

impl LieDetector {
    pub fn new() -> Self {
        Self {
            deception_keywords: vec![
                "definitely", "absolutely", "never", "always", "completely",
                "entirely", "totally", "perfectly", "exactly", "precisely"
            ],
            hedging_words: vec![
                "maybe", "possibly", "might", "think", "sort of", "kind of",
                "you know", "well", "like", "um", "probably", "perhaps"
            ]
        }
    }
    
    pub fn analyze(&self, sample: &SampleData) -> AnalysisResult {
        let text = sample.text.to_lowercase();
        let mut reasoning = Vec::new();
        let mut deception_score = 0.0;
        
        // Check for deception keywords (strong indicators)
        let deception_matches: usize = self.deception_keywords.iter()
            .map(|&keyword| text.matches(keyword).count())
            .sum();
        
        if deception_matches > 0 {
            deception_score += deception_matches as f64 * 0.3;
            reasoning.push(format!("Found {} strong deception indicators", deception_matches));
        }
        
        // Check for hedging words (uncertainty indicators)
        let hedging_matches: usize = self.hedging_words.iter()
            .map(|&keyword| text.matches(keyword).count())
            .sum();
            
        if hedging_matches > 0 {
            deception_score += hedging_matches as f64 * 0.15;
            reasoning.push(format!("Found {} hedging/uncertainty words", hedging_matches));
        }
        
        // Text length analysis (very short or very long statements)
        let word_count = text.split_whitespace().count();
        if word_count < 5 {
            deception_score += 0.2;
            reasoning.push("Very short statement may indicate evasiveness".to_string());
        } else if word_count > 30 {
            deception_score += 0.1;
            reasoning.push("Long statement may indicate over-explaining".to_string());
        }
        
        // Normalize confidence
        let confidence = deception_score.min(1.0);
        let predicted_label = if confidence > 0.5 { "deceptive" } else { "truthful" };
        
        reasoning.push(format!("Total deception score: {:.2}", deception_score));
        reasoning.push(format!("Normalized confidence: {:.2}", confidence));
        
        let accuracy = predicted_label == sample.actual_label;
        
        AnalysisResult {
            predicted_label: predicted_label.to_string(),
            confidence,
            reasoning,
            accuracy,
        }
    }
}

fn parse_csv_line(line: &str) -> Option<SampleData> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 8 {
        return None;
    }
    
    // Extract text (handling quoted text with commas)
    let text_start = line.find('"')?;
    let text_end = line.rfind('"')?;
    let text = &line[text_start+1..text_end];
    
    // Parse other fields
    let remaining = &line[text_end+1..];
    let remaining_parts: Vec<&str> = remaining.split(',').collect();
    if remaining_parts.len() < 4 {
        return None;
    }
    
    Some(SampleData {
        id: parts[0].to_string(),
        text: text.to_string(),
        actual_label: remaining_parts[1].to_string(),
        confidence: remaining_parts[2].parse().unwrap_or(0.0),
        environment: remaining_parts[3].to_string(),
        age: remaining_parts[4].parse().unwrap_or(0),
    })
}

fn main() {
    println!("🧪 Testing Veritas Nexus with Real Sample Data");
    
    // Read sample data
    let csv_content = match fs::read_to_string("examples/data/samples.csv") {
        Ok(content) => content,
        Err(e) => {
            println!("❌ Failed to read samples.csv: {}", e);
            println!("Make sure you're running from the project root directory");
            return;
        }
    };
    
    let lines: Vec<&str> = csv_content.lines().collect();
    if lines.len() < 2 {
        println!("❌ CSV file appears to be empty or malformed");
        return;
    }
    
    // Parse samples (skip header)
    let mut samples = Vec::new();
    for line in &lines[1..] {
        if let Some(sample) = parse_csv_line(line) {
            samples.push(sample);
        }
    }
    
    println!("📊 Loaded {} samples from CSV", samples.len());
    
    // Initialize lie detector
    let detector = LieDetector::new();
    
    // Analyze each sample
    let mut correct_predictions = 0;
    let mut total_predictions = 0;
    
    println!("\n🔍 Analysis Results:");
    println!("{:=<100}", "");
    
    for sample in &samples {
        let result = detector.analyze(sample);
        total_predictions += 1;
        if result.accuracy {
            correct_predictions += 1;
        }
        
        let accuracy_symbol = if result.accuracy { "✅" } else { "❌" };
        
        println!("\nSample: {}", sample.id);
        println!("Text: \"{}\"", sample.text);
        println!("Actual: {} | Predicted: {} | Confidence: {:.1}% {}", 
                 sample.actual_label, 
                 result.predicted_label, 
                 result.confidence * 100.0,
                 accuracy_symbol);
        println!("Reasoning:");
        for reason in &result.reasoning {
            println!("  • {}", reason);
        }
        println!("{:-<100}", "");
    }
    
    // Calculate overall accuracy
    let accuracy = (correct_predictions as f64 / total_predictions as f64) * 100.0;
    
    println!("\n📈 Overall Performance:");
    println!("Correct Predictions: {}/{}", correct_predictions, total_predictions);
    println!("Accuracy: {:.1}%", accuracy);
    
    // Analyze by environment
    let mut controlled_correct = 0;
    let mut controlled_total = 0;
    let mut field_correct = 0;
    let mut field_total = 0;
    
    for sample in &samples {
        let result = detector.analyze(sample);
        match sample.environment.as_str() {
            "controlled" => {
                controlled_total += 1;
                if result.accuracy { controlled_correct += 1; }
            }
            "field" => {
                field_total += 1;
                if result.accuracy { field_correct += 1; }
            }
            _ => {}
        }
    }
    
    if controlled_total > 0 {
        println!("Controlled Environment: {:.1}% ({}/{})", 
                 (controlled_correct as f64 / controlled_total as f64) * 100.0,
                 controlled_correct, controlled_total);
    }
    
    if field_total > 0 {
        println!("Field Environment: {:.1}% ({}/{})", 
                 (field_correct as f64 / field_total as f64) * 100.0,
                 field_correct, field_total);
    }
    
    println!("\n✅ Real data analysis completed!");
    
    if accuracy >= 70.0 {
        println!("🎉 Good performance! Algorithm shows promise.");
    } else if accuracy >= 50.0 {
        println!("📊 Moderate performance. Algorithm needs refinement.");
    } else {
        println!("⚠️  Low performance. Algorithm requires significant improvement.");
    }
}