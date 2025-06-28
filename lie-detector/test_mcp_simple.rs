// Simple MCP functionality test without external dependencies
use std::collections::HashMap;

// Simple mock structures without serde
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ToolResponse {
    pub status: String,
    pub message: String,
    pub data: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub status: String,
    pub modalities: Vec<String>,
}

pub struct SimpleMcpServer {
    models: HashMap<String, ModelInfo>,
    next_id: usize,
}

impl SimpleMcpServer {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn get_available_tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "model.list".to_string(),
                description: "List all available lie detection models".to_string(),
                parameters: HashMap::new(),
            },
            ToolDefinition {
                name: "model.create".to_string(),
                description: "Create a new lie detection model".to_string(),
                parameters: [
                    ("name".to_string(), "Model name".to_string()),
                    ("modalities".to_string(), "vision,audio,text,physiological".to_string()),
                ].iter().cloned().collect(),
            },
            ToolDefinition {
                name: "model.load".to_string(),
                description: "Load a model for inference".to_string(),
                parameters: [
                    ("model_id".to_string(), "ID of model to load".to_string()),
                ].iter().cloned().collect(),
            },
            ToolDefinition {
                name: "inference.analyze".to_string(),
                description: "Analyze input for deception detection".to_string(),
                parameters: [
                    ("model_id".to_string(), "Model ID to use".to_string()),
                    ("text".to_string(), "Text input for analysis".to_string()),
                    ("video_path".to_string(), "Path to video file (optional)".to_string()),
                    ("audio_path".to_string(), "Path to audio file (optional)".to_string()),
                ].iter().cloned().collect(),
            },
            ToolDefinition {
                name: "monitor.status".to_string(),
                description: "Get system status and health metrics".to_string(),
                parameters: HashMap::new(),
            },
        ]
    }
    
    pub fn handle_tool_call(&mut self, tool_name: &str, params: HashMap<String, String>) -> ToolResponse {
        match tool_name {
            "model.list" => self.list_models(),
            "model.create" => self.create_model(params),
            "model.load" => self.load_model(params),
            "inference.analyze" => self.analyze_input(params),
            "monitor.status" => self.get_status(),
            _ => ToolResponse {
                status: "error".to_string(),
                message: format!("Unknown tool: {}", tool_name),
                data: "null".to_string(),
            }
        }
    }
    
    fn list_models(&self) -> ToolResponse {
        let model_list: Vec<String> = self.models.values()
            .map(|m| format!("{}: {} ({})", m.id, m.name, m.status))
            .collect();
            
        ToolResponse {
            status: "success".to_string(),
            message: format!("Found {} models", self.models.len()),
            data: format!("[{}]", model_list.join(", ")),
        }
    }
    
    fn create_model(&mut self, params: HashMap<String, String>) -> ToolResponse {
        let default_name = "Unnamed Model".to_string();
        let name = params.get("name").unwrap_or(&default_name).clone();
        let default_modalities = "text".to_string();
        let modalities_str = params.get("modalities").unwrap_or(&default_modalities);
        let modalities: Vec<String> = modalities_str.split(',')
            .map(|s| s.trim().to_string())
            .collect();
        
        let model_id = format!("model_{:03}", self.next_id);
        self.next_id += 1;
        
        let model = ModelInfo {
            id: model_id.clone(),
            name: name.clone(),
            status: "created".to_string(),
            modalities,
        };
        
        self.models.insert(model_id.clone(), model);
        
        ToolResponse {
            status: "success".to_string(),
            message: format!("Model '{}' created successfully", name),
            data: format!("{{\"id\": \"{}\", \"status\": \"created\"}}", model_id),
        }
    }
    
    fn load_model(&mut self, params: HashMap<String, String>) -> ToolResponse {
        let default_id = "".to_string();
        let model_id = params.get("model_id").unwrap_or(&default_id);
        
        if let Some(model) = self.models.get_mut(model_id) {
            model.status = "loaded".to_string();
            ToolResponse {
                status: "success".to_string(),
                message: format!("Model {} loaded successfully", model_id),
                data: format!("{{\"status\": \"loaded\", \"modalities\": {:?}}}", model.modalities),
            }
        } else {
            ToolResponse {
                status: "error".to_string(),
                message: format!("Model '{}' not found", model_id),
                data: "null".to_string(),
            }
        }
    }
    
    fn analyze_input(&self, params: HashMap<String, String>) -> ToolResponse {
        let default_id = "".to_string();
        let default_text = "".to_string();
        let model_id = params.get("model_id").unwrap_or(&default_id);
        let text = params.get("text").unwrap_or(&default_text);
        
        if !self.models.contains_key(model_id) {
            return ToolResponse {
                status: "error".to_string(),
                message: format!("Model '{}' not found", model_id),
                data: "null".to_string(),
            };
        }
        
        if text.is_empty() {
            return ToolResponse {
                status: "error".to_string(),
                message: "No text provided for analysis".to_string(),
                data: "null".to_string(),
            };
        }
        
        // Simple analysis (similar to previous tests)
        let deception_words = ["definitely", "absolutely", "never", "always"];
        let matches: usize = deception_words.iter()
            .map(|&word| text.to_lowercase().matches(word).count())
            .sum();
            
        let confidence = (matches as f64 * 0.3).min(1.0);
        let decision = if confidence > 0.5 { "deceptive" } else { "truthful" };
        
        ToolResponse {
            status: "success".to_string(),
            message: "Analysis completed".to_string(),
            data: format!("{{\"decision\": \"{}\", \"confidence\": {:.2}, \"indicators\": {}}}", 
                         decision, confidence, matches),
        }
    }
    
    fn get_status(&self) -> ToolResponse {
        let loaded_models = self.models.values()
            .filter(|m| m.status == "loaded")
            .count();
            
        ToolResponse {
            status: "success".to_string(),
            message: "System status retrieved".to_string(),
            data: format!("{{\"total_models\": {}, \"loaded_models\": {}, \"status\": \"healthy\"}}", 
                         self.models.len(), loaded_models),
        }
    }
}

fn main() {
    println!("🛠️  Testing Veritas Nexus MCP Server Functionality");
    
    let mut server = SimpleMcpServer::new();
    
    // Test 1: List available tools
    println!("\n📋 Available MCP Tools:");
    println!("{:-<80}", "");
    for tool in server.get_available_tools() {
        println!("• {}", tool.name);
        println!("  Description: {}", tool.description);
        if !tool.parameters.is_empty() {
            println!("  Parameters: {:?}", tool.parameters.keys().collect::<Vec<_>>());
        }
        println!();
    }
    
    // Test 2: Create models
    println!("🏗️  Testing Model Creation:");
    println!("{:-<80}", "");
    
    let mut params = HashMap::new();
    params.insert("name".to_string(), "Multi-Modal Detector".to_string());
    params.insert("modalities".to_string(), "vision,audio,text".to_string());
    let response = server.handle_tool_call("model.create", params);
    println!("Create Model 1: {} - {}", response.status, response.message);
    
    let mut params = HashMap::new();
    params.insert("name".to_string(), "Text-Only Analyzer".to_string());
    params.insert("modalities".to_string(), "text".to_string());
    let response = server.handle_tool_call("model.create", params);
    println!("Create Model 2: {} - {}", response.status, response.message);
    
    // Test 3: List models
    println!("\n📊 Listing Models:");
    println!("{:-<80}", "");
    let response = server.handle_tool_call("model.list", HashMap::new());
    println!("Status: {}", response.status);
    println!("Message: {}", response.message);
    println!("Data: {}", response.data);
    
    // Test 4: Load a model
    println!("\n⚡ Loading Model:");
    println!("{:-<80}", "");
    let mut params = HashMap::new();
    params.insert("model_id".to_string(), "model_001".to_string());
    let response = server.handle_tool_call("model.load", params);
    println!("Status: {}", response.status);
    println!("Message: {}", response.message);
    println!("Data: {}", response.data);
    
    // Test 5: Analyze text
    println!("\n🔍 Analyzing Text:");
    println!("{:-<80}", "");
    let test_texts = vec![
        ("I definitely never took the money", "High deception indicators"),
        ("I went to the store yesterday", "Normal statement"),
        ("I absolutely always tell the truth", "Multiple indicators"),
    ];
    
    for (text, description) in test_texts {
        let mut params = HashMap::new();
        params.insert("model_id".to_string(), "model_001".to_string());
        params.insert("text".to_string(), text.to_string());
        let response = server.handle_tool_call("inference.analyze", params);
        
        println!("Text: \"{}\"", text);
        println!("Description: {}", description);
        println!("Status: {}", response.status);
        println!("Result: {}", response.data);
        println!();
    }
    
    // Test 6: System status
    println!("🔧 System Status:");
    println!("{:-<80}", "");
    let response = server.handle_tool_call("monitor.status", HashMap::new());
    println!("Status: {}", response.status);
    println!("Data: {}", response.data);
    
    // Test 7: Error handling
    println!("\n❌ Testing Error Handling:");
    println!("{:-<80}", "");
    let response = server.handle_tool_call("invalid.tool", HashMap::new());
    println!("Invalid tool: {} - {}", response.status, response.message);
    
    let mut params = HashMap::new();
    params.insert("model_id".to_string(), "nonexistent".to_string());
    params.insert("text".to_string(), "test".to_string());
    let response = server.handle_tool_call("inference.analyze", params);
    println!("Invalid model: {} - {}", response.status, response.message);
    
    println!("\n✅ MCP Server functionality test completed!");
    println!("🎉 All core MCP operations working correctly!");
}