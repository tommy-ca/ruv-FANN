//! # Cascade Training Example for Veritas Nexus
//! 
//! This example demonstrates how to train models in a cascaded approach for lie detection.
//! It shows how to:
//! - Set up training pipelines for each modality (vision, audio, text)
//! - Implement progressive training with increasing complexity
//! - Train fusion models to combine modality outputs
//! - Use transfer learning and fine-tuning strategies
//! - Monitor training progress and validation metrics
//! - Save and load trained models
//! 
//! ## Usage
//! 
//! ```bash
//! cargo run --example cascade_training
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use serde::{Deserialize, Serialize};

/// Training configuration for the cascade approach
#[derive(Debug, Clone)]
pub struct CascadeTrainingConfig {
    pub stages: Vec<TrainingStage>,
    pub global_config: GlobalTrainingConfig,
    pub validation_config: ValidationConfig,
    pub save_config: SaveConfig,
}

/// Global training configuration
#[derive(Debug, Clone)]
pub struct GlobalTrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub early_stopping_patience: usize,
    pub validation_split: f32,
    pub random_seed: u64,
    pub device: Device,
    pub mixed_precision: bool,
    pub gradient_clipping: Option<f32>,
}

/// Training device options
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Metal,
    Auto,
}

/// Individual training stage in the cascade
#[derive(Debug, Clone)]
pub struct TrainingStage {
    pub name: String,
    pub stage_type: StageType,
    pub model_config: ModelConfig,
    pub data_config: DataConfig,
    pub training_config: StageTrainingConfig,
    pub dependencies: Vec<String>, // Previous stages this depends on
}

/// Type of training stage
#[derive(Debug, Clone)]
pub enum StageType {
    VisionPretraining,
    AudioPretraining,
    TextPretraining,
    VisionFinetuning,
    AudioFinetuning,
    TextFinetuning,
    EarlyFusion,
    LateFusion,
    AttentionFusion,
    MetaLearning,
    Distillation,
}

/// Model configuration for each stage
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub input_shape: Vec<usize>,
    pub output_size: usize,
    pub hidden_layers: Vec<usize>,
    pub activation: Activation,
    pub dropout_rate: f32,
    pub batch_norm: bool,
    pub pretrained_weights: Option<String>,
}

/// Model architecture options
#[derive(Debug, Clone)]
pub enum Architecture {
    ConvNet,
    ResNet { layers: usize },
    Transformer { heads: usize, layers: usize },
    RNN { cell_type: RnnCell, layers: usize },
    MLP,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum RnnCell {
    LSTM,
    GRU,
    Vanilla,
}

#[derive(Debug, Clone)]
pub enum Activation {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
}

/// Data configuration for training
#[derive(Debug, Clone)]
pub struct DataConfig {
    pub train_data_path: String,
    pub val_data_path: Option<String>,
    pub test_data_path: Option<String>,
    pub data_format: DataFormat,
    pub preprocessing: PreprocessingConfig,
    pub augmentation: AugmentationConfig,
}

#[derive(Debug, Clone)]
pub enum DataFormat {
    Video,
    Audio,
    Text,
    MultiModal,
    Csv,
    Json,
    Custom(String),
}

/// Preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub normalize: bool,
    pub standardize: bool,
    pub resize: Option<(usize, usize)>,
    pub crop: Option<(usize, usize)>,
    pub frame_rate: Option<f32>,
    pub sample_rate: Option<u32>,
    pub max_length: Option<usize>,
}

/// Data augmentation configuration
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub enabled: bool,
    pub horizontal_flip: bool,
    pub rotation_degrees: Option<f32>,
    pub brightness_range: Option<(f32, f32)>,
    pub noise_level: Option<f32>,
    pub time_stretch: Option<(f32, f32)>,
    pub pitch_shift: Option<(f32, f32)>,
    pub synonym_replacement: Option<f32>,
}

/// Stage-specific training configuration
#[derive(Debug, Clone)]
pub struct StageTrainingConfig {
    pub learning_rate: Option<f32>, // Override global if specified
    pub batch_size: Option<usize>,
    pub max_epochs: Option<usize>,
    pub loss_function: LossFunction,
    pub optimizer: Optimizer,
    pub scheduler: Option<LearningRateScheduler>,
    pub metrics: Vec<Metric>,
    pub freeze_layers: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum LossFunction {
    CrossEntropy,
    BinaryCrossEntropy,
    FocalLoss { alpha: f32, gamma: f32 },
    MSE,
    MAE,
    Huber { delta: f32 },
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum Optimizer {
    Adam { beta1: f32, beta2: f32, eps: f32 },
    SGD { momentum: f32, nesterov: bool },
    AdamW { weight_decay: f32 },
    RMSprop { alpha: f32, eps: f32 },
}

#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    StepLR { step_size: usize, gamma: f32 },
    ExponentialLR { gamma: f32 },
    CosineAnnealing { t_max: usize },
    ReduceOnPlateau { patience: usize, factor: f32 },
}

#[derive(Debug, Clone)]
pub enum Metric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    Loss,
    Custom(String),
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub frequency: ValidationFrequency,
    pub metrics: Vec<Metric>,
    pub early_stopping_metric: String,
    pub early_stopping_mode: EarlyStoppingMode,
    pub save_best_model: bool,
    pub validation_batch_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum ValidationFrequency {
    EveryEpoch,
    EveryNEpochs(usize),
    EveryNSteps(usize),
}

#[derive(Debug, Clone)]
pub enum EarlyStoppingMode {
    Min, // Stop when metric stops decreasing
    Max, // Stop when metric stops increasing
}

/// Model saving configuration
#[derive(Debug, Clone)]
pub struct SaveConfig {
    pub save_dir: String,
    pub save_frequency: SaveFrequency,
    pub save_format: SaveFormat,
    pub keep_n_best: usize,
    pub save_optimizer_state: bool,
}

#[derive(Debug, Clone)]
pub enum SaveFrequency {
    EveryEpoch,
    BestOnly,
    Final,
    EveryNEpochs(usize),
}

#[derive(Debug, Clone)]
pub enum SaveFormat {
    PyTorch,
    ONNX,
    TensorFlow,
    Custom(String),
}

/// Training progress and metrics
#[derive(Debug, Clone, Serialize)]
pub struct TrainingProgress {
    pub stage_name: String,
    pub epoch: usize,
    pub step: usize,
    pub train_loss: f32,
    pub train_metrics: HashMap<String, f32>,
    pub val_loss: Option<f32>,
    pub val_metrics: HashMap<String, f32>,
    pub learning_rate: f32,
    pub elapsed_time: Duration,
    pub estimated_time_remaining: Option<Duration>,
}

/// Training result for a stage
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage_name: String,
    pub final_train_loss: f32,
    pub final_val_loss: Option<f32>,
    pub best_val_metric: Option<f32>,
    pub training_time: Duration,
    pub final_metrics: HashMap<String, f32>,
    pub model_path: String,
    pub converged: bool,
    pub early_stopped: bool,
}

/// Complete cascade training result
#[derive(Debug)]
pub struct CascadeTrainingResult {
    pub stage_results: Vec<StageResult>,
    pub total_training_time: Duration,
    pub final_ensemble_performance: HashMap<String, f32>,
    pub model_paths: HashMap<String, String>,
}

/// Mock trainer for demonstration
pub struct CascadeTrainer {
    config: CascadeTrainingConfig,
    progress_callback: Option<Box<dyn Fn(&TrainingProgress) + Send + Sync>>,
    trained_models: Arc<Mutex<HashMap<String, String>>>, // stage_name -> model_path
}

impl CascadeTrainer {
    pub fn new(config: CascadeTrainingConfig) -> Self {
        Self {
            config,
            progress_callback: None,
            trained_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self 
    where 
        F: Fn(&TrainingProgress) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }
    
    /// Train all stages in the cascade
    pub async fn train(&self) -> Result<CascadeTrainingResult, TrainingError> {
        let start_time = Instant::now();
        let mut stage_results = Vec::new();
        
        println!("🚀 Starting cascade training with {} stages", self.config.stages.len());
        
        // Sort stages by dependencies (topological sort)
        let sorted_stages = self.sort_stages_by_dependencies()?;
        
        for stage in sorted_stages {
            println!("\n📚 Training stage: {} ({:?})", stage.name, stage.stage_type);
            
            // Check dependencies
            self.check_dependencies(&stage)?;
            
            // Train the stage
            let result = self.train_stage(&stage).await?;
            
            // Save model path for future stages
            {
                let mut models = self.trained_models.lock().unwrap();
                models.insert(stage.name.clone(), result.model_path.clone());
            }
            
            stage_results.push(result);
        }
        
        // Evaluate final ensemble
        println!("\n🎯 Evaluating final ensemble performance...");
        let ensemble_performance = self.evaluate_ensemble().await?;
        
        let total_time = start_time.elapsed();
        
        // Get final model paths
        let model_paths = self.trained_models.lock().unwrap().clone();
        
        println!("✅ Cascade training completed in {:.2}s", total_time.as_secs_f32());
        
        Ok(CascadeTrainingResult {
            stage_results,
            total_training_time: total_time,
            final_ensemble_performance: ensemble_performance,
            model_paths,
        })
    }
    
    /// Sort stages by dependencies using topological sort
    fn sort_stages_by_dependencies(&self) -> Result<Vec<TrainingStage>, TrainingError> {
        let mut sorted = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();
        
        fn visit(
            stage_name: &str,
            stages: &[TrainingStage],
            visited: &mut std::collections::HashSet<String>,
            temp_visited: &mut std::collections::HashSet<String>,
            sorted: &mut Vec<TrainingStage>,
        ) -> Result<(), TrainingError> {
            if temp_visited.contains(stage_name) {
                return Err(TrainingError::CircularDependency(stage_name.to_string()));
            }
            
            if visited.contains(stage_name) {
                return Ok(());
            }
            
            temp_visited.insert(stage_name.to_string());
            
            let stage = stages.iter()
                .find(|s| s.name == stage_name)
                .ok_or_else(|| TrainingError::StageNotFound(stage_name.to_string()))?;
            
            for dep in &stage.dependencies {
                visit(dep, stages, visited, temp_visited, sorted)?;
            }
            
            temp_visited.remove(stage_name);
            visited.insert(stage_name.to_string());
            sorted.push(stage.clone());
            
            Ok(())
        }
        
        for stage in &self.config.stages {
            if !visited.contains(&stage.name) {
                visit(&stage.name, &self.config.stages, &mut visited, &mut temp_visited, &mut sorted)?;
            }
        }
        
        Ok(sorted)
    }
    
    /// Check that all dependencies are satisfied
    fn check_dependencies(&self, stage: &TrainingStage) -> Result<(), TrainingError> {
        let models = self.trained_models.lock().unwrap();
        
        for dep in &stage.dependencies {
            if !models.contains_key(dep) {
                return Err(TrainingError::MissingDependency {
                    stage: stage.name.clone(),
                    dependency: dep.clone(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Train a single stage
    async fn train_stage(&self, stage: &TrainingStage) -> Result<StageResult, TrainingError> {
        let start_time = Instant::now();
        
        // Get training parameters
        let learning_rate = stage.training_config.learning_rate
            .unwrap_or(self.config.global_config.learning_rate);
        let batch_size = stage.training_config.batch_size
            .unwrap_or(self.config.global_config.batch_size);
        let max_epochs = stage.training_config.max_epochs
            .unwrap_or(self.config.global_config.max_epochs);
        
        println!("  Configuration:");
        println!("    Architecture: {:?}", stage.model_config.architecture);
        println!("    Learning rate: {}", learning_rate);
        println!("    Batch size: {}", batch_size);
        println!("    Max epochs: {}", max_epochs);
        println!("    Loss function: {:?}", stage.training_config.loss_function);
        println!("    Optimizer: {:?}", stage.training_config.optimizer);
        
        // Simulate model initialization
        println!("  Initializing model...");
        sleep(Duration::from_millis(500)).await;
        
        // Simulate data loading
        println!("  Loading training data from: {}", stage.data_config.train_data_path);
        sleep(Duration::from_millis(300)).await;
        
        let samples_per_epoch = match stage.stage_type {
            StageType::VisionPretraining | StageType::VisionFinetuning => 10000,
            StageType::AudioPretraining | StageType::AudioFinetuning => 8000,
            StageType::TextPretraining | StageType::TextFinetuning => 15000,
            _ => 5000,
        };
        
        let steps_per_epoch = (samples_per_epoch + batch_size - 1) / batch_size;
        
        println!("  Training data: {} samples, {} steps per epoch", samples_per_epoch, steps_per_epoch);
        
        let mut best_val_metric = None;
        let mut patience_counter = 0;
        let mut converged = false;
        let mut early_stopped = false;
        
        // Training loop
        for epoch in 1..=max_epochs {
            let epoch_start = Instant::now();
            
            // Simulate training steps
            let mut epoch_train_loss = 0.0;
            let mut train_metrics = HashMap::new();
            
            for step in 1..=steps_per_epoch {
                // Simulate forward pass and loss computation
                let step_loss = self.simulate_training_step(epoch, step, &stage.stage_type);
                epoch_train_loss += step_loss;
                
                // Progress callback every 10 steps
                if step % 10 == 0 || step == steps_per_epoch {
                    let progress = TrainingProgress {
                        stage_name: stage.name.clone(),
                        epoch,
                        step,
                        train_loss: epoch_train_loss / step as f32,
                        train_metrics: train_metrics.clone(),
                        val_loss: None,
                        val_metrics: HashMap::new(),
                        learning_rate,
                        elapsed_time: start_time.elapsed(),
                        estimated_time_remaining: Some(Duration::from_secs(
                            ((max_epochs - epoch) * 30 + ((steps_per_epoch - step) * 30 / steps_per_epoch)) as u64
                        )),
                    };
                    
                    if let Some(callback) = &self.progress_callback {
                        callback(&progress);
                    }
                }
                
                // Small delay to simulate computation
                sleep(Duration::from_millis(5)).await;
            }
            
            epoch_train_loss /= steps_per_epoch as f32;
            
            // Simulate metrics computation
            train_metrics.insert("accuracy".to_string(), 0.6 + epoch as f32 * 0.02);
            train_metrics.insert("f1_score".to_string(), 0.55 + epoch as f32 * 0.025);
            
            // Simulate validation
            let (val_loss, val_metrics) = if epoch % 1 == 0 { // Validate every epoch
                sleep(Duration::from_millis(100)).await; // Simulate validation time
                
                let val_loss = epoch_train_loss * 1.1; // Validation usually slightly higher
                let mut val_metrics = HashMap::new();
                val_metrics.insert("accuracy".to_string(), train_metrics["accuracy"] - 0.05);
                val_metrics.insert("f1_score".to_string(), train_metrics["f1_score"] - 0.03);
                
                (Some(val_loss), val_metrics)
            } else {
                (None, HashMap::new())
            };
            
            let epoch_time = epoch_start.elapsed();
            
            // Print epoch summary
            print!("    Epoch {}/{}: loss={:.4}, acc={:.3}", 
                epoch, max_epochs, epoch_train_loss, train_metrics["accuracy"]);
            
            if let Some(val_acc) = val_metrics.get("accuracy") {
                print!(", val_loss={:.4}, val_acc={:.3}", val_loss.unwrap(), val_acc);
            }
            
            println!(" ({:.1}s)", epoch_time.as_secs_f32());
            
            // Early stopping check
            if let Some(val_acc) = val_metrics.get("accuracy") {
                if best_val_metric.is_none() || *val_acc > best_val_metric.unwrap() {
                    best_val_metric = Some(*val_acc);
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    
                    if patience_counter >= self.config.global_config.early_stopping_patience {
                        println!("    Early stopping triggered (patience: {})", patience_counter);
                        early_stopped = true;
                        break;
                    }
                }
            }
            
            // Check convergence
            if epoch_train_loss < 0.01 {
                println!("    Training converged (loss < 0.01)");
                converged = true;
                break;
            }
        }
        
        // Save model
        let model_path = format!("{}/{}_model.pt", self.config.save_config.save_dir, stage.name);
        println!("  Saving model to: {}", model_path);
        sleep(Duration::from_millis(200)).await;
        
        let training_time = start_time.elapsed();
        
        // Compute final metrics
        let mut final_metrics = HashMap::new();
        final_metrics.insert("final_train_accuracy".to_string(), 0.85 + (rand::random::<f32>() * 0.1));
        final_metrics.insert("final_val_accuracy".to_string(), 0.82 + (rand::random::<f32>() * 0.08));
        
        Ok(StageResult {
            stage_name: stage.name.clone(),
            final_train_loss: 0.1 + (rand::random::<f32>() * 0.05),
            final_val_loss: Some(0.12 + (rand::random::<f32>() * 0.05)),
            best_val_metric,
            training_time,
            final_metrics,
            model_path,
            converged,
            early_stopped,
        })
    }
    
    /// Simulate a training step
    fn simulate_training_step(&self, epoch: usize, step: usize, stage_type: &StageType) -> f32 {
        // Base loss that decreases over time
        let base_loss = 1.0 / (1.0 + 0.1 * epoch as f32 + 0.001 * step as f32);
        
        // Add stage-specific characteristics
        let stage_modifier = match stage_type {
            StageType::VisionPretraining => 1.2, // Vision harder to train initially
            StageType::AudioPretraining => 1.0,
            StageType::TextPretraining => 0.9,   // Text often easier
            StageType::VisionFinetuning => 0.7,  // Fine-tuning starts lower
            StageType::AudioFinetuning => 0.7,
            StageType::TextFinetuning => 0.6,
            StageType::EarlyFusion => 0.8,
            StageType::LateFusion => 0.75,
            StageType::AttentionFusion => 0.65,  // More sophisticated fusion
            StageType::MetaLearning => 0.5,
            StageType::Distillation => 0.4,     // Knowledge distillation very effective
        };
        
        // Add some noise
        let noise = (rand::random::<f32>() - 0.5) * 0.1;
        
        (base_loss * stage_modifier + noise).max(0.01)
    }
    
    /// Evaluate the final ensemble performance
    async fn evaluate_ensemble(&self) -> Result<HashMap<String, f32>, TrainingError> {
        println!("  Loading all trained models...");
        sleep(Duration::from_millis(500)).await;
        
        println!("  Running ensemble evaluation...");
        sleep(Duration::from_millis(800)).await;
        
        let mut performance = HashMap::new();
        
        // Simulate ensemble performance (usually better than individual models)
        performance.insert("ensemble_accuracy".to_string(), 0.91 + (rand::random::<f32>() * 0.05));
        performance.insert("ensemble_precision".to_string(), 0.89 + (rand::random::<f32>() * 0.06));
        performance.insert("ensemble_recall".to_string(), 0.87 + (rand::random::<f32>() * 0.07));
        performance.insert("ensemble_f1_score".to_string(), 0.88 + (rand::random::<f32>() * 0.06));
        performance.insert("ensemble_auc".to_string(), 0.94 + (rand::random::<f32>() * 0.04));
        
        Ok(performance)
    }
}

/// Training error types
#[derive(Debug)]
pub enum TrainingError {
    StageNotFound(String),
    CircularDependency(String),
    MissingDependency { stage: String, dependency: String },
    DataLoadError(String),
    ModelInitError(String),
    TrainingFailed(String),
    ValidationFailed(String),
    SaveError(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::StageNotFound(stage) => write!(f, "Stage not found: {}", stage),
            TrainingError::CircularDependency(stage) => write!(f, "Circular dependency detected in stage: {}", stage),
            TrainingError::MissingDependency { stage, dependency } => {
                write!(f, "Stage '{}' missing dependency: '{}'", stage, dependency)
            }
            TrainingError::DataLoadError(msg) => write!(f, "Data loading error: {}", msg),
            TrainingError::ModelInitError(msg) => write!(f, "Model initialization error: {}", msg),
            TrainingError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            TrainingError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            TrainingError::SaveError(msg) => write!(f, "Model save error: {}", msg),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Helper function to create a comprehensive cascade training configuration
fn create_cascade_config() -> CascadeTrainingConfig {
    CascadeTrainingConfig {
        stages: vec![
            // Stage 1: Vision Pretraining
            TrainingStage {
                name: "vision_pretraining".to_string(),
                stage_type: StageType::VisionPretraining,
                model_config: ModelConfig {
                    architecture: Architecture::ResNet { layers: 50 },
                    input_shape: vec![3, 224, 224],
                    output_size: 512,
                    hidden_layers: vec![2048, 1024],
                    activation: Activation::ReLU,
                    dropout_rate: 0.3,
                    batch_norm: true,
                    pretrained_weights: Some("imagenet".to_string()),
                },
                data_config: DataConfig {
                    train_data_path: "data/vision/pretrain".to_string(),
                    val_data_path: Some("data/vision/val".to_string()),
                    test_data_path: None,
                    data_format: DataFormat::Video,
                    preprocessing: PreprocessingConfig {
                        normalize: true,
                        standardize: true,
                        resize: Some((224, 224)),
                        crop: Some((224, 224)),
                        frame_rate: Some(30.0),
                        sample_rate: None,
                        max_length: None,
                    },
                    augmentation: AugmentationConfig {
                        enabled: true,
                        horizontal_flip: true,
                        rotation_degrees: Some(15.0),
                        brightness_range: Some((0.8, 1.2)),
                        noise_level: Some(0.02),
                        time_stretch: None,
                        pitch_shift: None,
                        synonym_replacement: None,
                    },
                },
                training_config: StageTrainingConfig {
                    learning_rate: Some(0.001),
                    batch_size: Some(32),
                    max_epochs: Some(20),
                    loss_function: LossFunction::CrossEntropy,
                    optimizer: Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
                    scheduler: Some(LearningRateScheduler::StepLR { step_size: 7, gamma: 0.1 }),
                    metrics: vec![Metric::Accuracy, Metric::F1Score],
                    freeze_layers: vec![],
                },
                dependencies: vec![],
            },
            
            // Stage 2: Audio Pretraining
            TrainingStage {
                name: "audio_pretraining".to_string(),
                stage_type: StageType::AudioPretraining,
                model_config: ModelConfig {
                    architecture: Architecture::ConvNet,
                    input_shape: vec![1, 128, 512], // Spectrogram
                    output_size: 256,
                    hidden_layers: vec![512, 256],
                    activation: Activation::ReLU,
                    dropout_rate: 0.25,
                    batch_norm: true,
                    pretrained_weights: None,
                },
                data_config: DataConfig {
                    train_data_path: "data/audio/pretrain".to_string(),
                    val_data_path: Some("data/audio/val".to_string()),
                    test_data_path: None,
                    data_format: DataFormat::Audio,
                    preprocessing: PreprocessingConfig {
                        normalize: true,
                        standardize: true,
                        resize: None,
                        crop: None,
                        frame_rate: None,
                        sample_rate: Some(16000),
                        max_length: Some(10), // 10 seconds
                    },
                    augmentation: AugmentationConfig {
                        enabled: true,
                        horizontal_flip: false,
                        rotation_degrees: None,
                        brightness_range: None,
                        noise_level: Some(0.01),
                        time_stretch: Some((0.8, 1.2)),
                        pitch_shift: Some((-2.0, 2.0)),
                        synonym_replacement: None,
                    },
                },
                training_config: StageTrainingConfig {
                    learning_rate: Some(0.001),
                    batch_size: Some(64),
                    max_epochs: Some(15),
                    loss_function: LossFunction::CrossEntropy,
                    optimizer: Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
                    scheduler: Some(LearningRateScheduler::ExponentialLR { gamma: 0.95 }),
                    metrics: vec![Metric::Accuracy, Metric::AUC],
                    freeze_layers: vec![],
                },
                dependencies: vec![],
            },
            
            // Stage 3: Text Pretraining
            TrainingStage {
                name: "text_pretraining".to_string(),
                stage_type: StageType::TextPretraining,
                model_config: ModelConfig {
                    architecture: Architecture::Transformer { heads: 8, layers: 6 },
                    input_shape: vec![512], // Max sequence length
                    output_size: 768,
                    hidden_layers: vec![768, 256],
                    activation: Activation::GELU,
                    dropout_rate: 0.1,
                    batch_norm: false,
                    pretrained_weights: Some("bert-base-uncased".to_string()),
                },
                data_config: DataConfig {
                    train_data_path: "data/text/pretrain".to_string(),
                    val_data_path: Some("data/text/val".to_string()),
                    test_data_path: None,
                    data_format: DataFormat::Text,
                    preprocessing: PreprocessingConfig {
                        normalize: false,
                        standardize: false,
                        resize: None,
                        crop: None,
                        frame_rate: None,
                        sample_rate: None,
                        max_length: Some(512),
                    },
                    augmentation: AugmentationConfig {
                        enabled: true,
                        horizontal_flip: false,
                        rotation_degrees: None,
                        brightness_range: None,
                        noise_level: None,
                        time_stretch: None,
                        pitch_shift: None,
                        synonym_replacement: Some(0.1),
                    },
                },
                training_config: StageTrainingConfig {
                    learning_rate: Some(2e-5),
                    batch_size: Some(16),
                    max_epochs: Some(10),
                    loss_function: LossFunction::CrossEntropy,
                    optimizer: Optimizer::AdamW { weight_decay: 0.01 },
                    scheduler: Some(LearningRateScheduler::CosineAnnealing { t_max: 10 }),
                    metrics: vec![Metric::Accuracy, Metric::F1Score],
                    freeze_layers: vec!["embeddings".to_string()],
                },
                dependencies: vec![],
            },
            
            // Stage 4: Vision Fine-tuning
            TrainingStage {
                name: "vision_finetuning".to_string(),
                stage_type: StageType::VisionFinetuning,
                model_config: ModelConfig {
                    architecture: Architecture::ResNet { layers: 50 },
                    input_shape: vec![3, 224, 224],
                    output_size: 2, // Binary classification
                    hidden_layers: vec![512, 128],
                    activation: Activation::ReLU,
                    dropout_rate: 0.4,
                    batch_norm: true,
                    pretrained_weights: None, // Will load from vision_pretraining
                },
                data_config: DataConfig {
                    train_data_path: "data/deception/vision/train".to_string(),
                    val_data_path: Some("data/deception/vision/val".to_string()),
                    test_data_path: Some("data/deception/vision/test".to_string()),
                    data_format: DataFormat::Video,
                    preprocessing: PreprocessingConfig {
                        normalize: true,
                        standardize: true,
                        resize: Some((224, 224)),
                        crop: Some((224, 224)),
                        frame_rate: Some(30.0),
                        sample_rate: None,
                        max_length: None,
                    },
                    augmentation: AugmentationConfig {
                        enabled: true,
                        horizontal_flip: false, // Don't flip faces
                        rotation_degrees: Some(5.0),
                        brightness_range: Some((0.9, 1.1)),
                        noise_level: Some(0.01),
                        time_stretch: None,
                        pitch_shift: None,
                        synonym_replacement: None,
                    },
                },
                training_config: StageTrainingConfig {
                    learning_rate: Some(1e-4),
                    batch_size: Some(16),
                    max_epochs: Some(25),
                    loss_function: LossFunction::FocalLoss { alpha: 0.25, gamma: 2.0 },
                    optimizer: Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
                    scheduler: Some(LearningRateScheduler::ReduceOnPlateau { patience: 3, factor: 0.5 }),
                    metrics: vec![Metric::Accuracy, Metric::Precision, Metric::Recall, Metric::F1Score, Metric::AUC],
                    freeze_layers: vec!["layer1".to_string(), "layer2".to_string()],
                },
                dependencies: vec!["vision_pretraining".to_string()],
            },
            
            // Stage 5: Late Fusion
            TrainingStage {
                name: "late_fusion".to_string(),
                stage_type: StageType::LateFusion,
                model_config: ModelConfig {
                    architecture: Architecture::MLP,
                    input_shape: vec![1036], // 512 + 256 + 768 from three modalities
                    output_size: 2,
                    hidden_layers: vec![512, 256, 128],
                    activation: Activation::ReLU,
                    dropout_rate: 0.3,
                    batch_norm: true,
                    pretrained_weights: None,
                },
                data_config: DataConfig {
                    train_data_path: "data/deception/multimodal/train".to_string(),
                    val_data_path: Some("data/deception/multimodal/val".to_string()),
                    test_data_path: Some("data/deception/multimodal/test".to_string()),
                    data_format: DataFormat::MultiModal,
                    preprocessing: PreprocessingConfig {
                        normalize: true,
                        standardize: true,
                        resize: None,
                        crop: None,
                        frame_rate: None,
                        sample_rate: None,
                        max_length: None,
                    },
                    augmentation: AugmentationConfig {
                        enabled: false, // Features already extracted
                        horizontal_flip: false,
                        rotation_degrees: None,
                        brightness_range: None,
                        noise_level: None,
                        time_stretch: None,
                        pitch_shift: None,
                        synonym_replacement: None,
                    },
                },
                training_config: StageTrainingConfig {
                    learning_rate: Some(5e-4),
                    batch_size: Some(64),
                    max_epochs: Some(30),
                    loss_function: LossFunction::CrossEntropy,
                    optimizer: Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
                    scheduler: Some(LearningRateScheduler::StepLR { step_size: 10, gamma: 0.5 }),
                    metrics: vec![Metric::Accuracy, Metric::Precision, Metric::Recall, Metric::F1Score, Metric::AUC],
                    freeze_layers: vec![],
                },
                dependencies: vec![
                    "vision_finetuning".to_string(),
                    "audio_pretraining".to_string(),
                    "text_pretraining".to_string(),
                ],
            },
        ],
        global_config: GlobalTrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 20,
            early_stopping_patience: 5,
            validation_split: 0.2,
            random_seed: 42,
            device: Device::Auto,
            mixed_precision: true,
            gradient_clipping: Some(1.0),
        },
        validation_config: ValidationConfig {
            frequency: ValidationFrequency::EveryEpoch,
            metrics: vec![Metric::Accuracy, Metric::F1Score, Metric::AUC],
            early_stopping_metric: "f1_score".to_string(),
            early_stopping_mode: EarlyStoppingMode::Max,
            save_best_model: true,
            validation_batch_size: Some(64),
        },
        save_config: SaveConfig {
            save_dir: "models/cascade_training".to_string(),
            save_frequency: SaveFrequency::BestOnly,
            save_format: SaveFormat::PyTorch,
            keep_n_best: 3,
            save_optimizer_state: true,
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Veritas Nexus - Cascade Training Example");
    println!("===========================================\n");
    
    // Create training configuration
    let config = create_cascade_config();
    
    println!("📋 Training Configuration:");
    println!("  Total stages: {}", config.stages.len());
    println!("  Global learning rate: {}", config.global_config.learning_rate);
    println!("  Global batch size: {}", config.global_config.batch_size);
    println!("  Max epochs per stage: {}", config.global_config.max_epochs);
    println!("  Early stopping patience: {}", config.global_config.early_stopping_patience);
    println!("  Device: {:?}", config.global_config.device);
    println!("  Mixed precision: {}", config.global_config.mixed_precision);
    
    println!("\n📚 Training Stages:");
    for (i, stage) in config.stages.iter().enumerate() {
        println!("  {}. {} ({:?})", i + 1, stage.name, stage.stage_type);
        if !stage.dependencies.is_empty() {
            println!("     Dependencies: {:?}", stage.dependencies);
        }
    }
    
    // Create trainer with progress callback
    let trainer = CascadeTrainer::new(config).with_progress_callback(|progress| {
        if progress.step % 50 == 0 || progress.step == 1 {
            print!("      Step {}: loss={:.4}, lr={:.2e}", 
                progress.step, progress.train_loss, progress.learning_rate);
            
            if let Some(eta) = progress.estimated_time_remaining {
                print!(", ETA={:.0}s", eta.as_secs_f32());
            }
            
            println!();
        }
    });
    
    println!("\n🚀 Starting cascade training...");
    
    // Run training
    match trainer.train().await {
        Ok(result) => {
            println!("\n🎉 Training completed successfully!");
            println!("=====================================");
            
            println!("\n📊 Stage Results:");
            for stage_result in &result.stage_results {
                println!("  {}:", stage_result.stage_name);
                println!("    Training time: {:.1}s", stage_result.training_time.as_secs_f32());
                println!("    Final train loss: {:.4}", stage_result.final_train_loss);
                if let Some(val_loss) = stage_result.final_val_loss {
                    println!("    Final val loss: {:.4}", val_loss);
                }
                if let Some(best_metric) = stage_result.best_val_metric {
                    println!("    Best val metric: {:.4}", best_metric);
                }
                println!("    Converged: {}", stage_result.converged);
                println!("    Early stopped: {}", stage_result.early_stopped);
                println!("    Model saved to: {}", stage_result.model_path);
                
                println!("    Final metrics:");
                for (metric, value) in &stage_result.final_metrics {
                    println!("      {}: {:.4}", metric, value);
                }
                println!();
            }
            
            println!("🏆 Final Ensemble Performance:");
            for (metric, value) in &result.final_ensemble_performance {
                println!("  {}: {:.4}", metric, value);
            }
            
            println!("\n⏱️  Total training time: {:.1}s", result.total_training_time.as_secs_f32());
            
            println!("\n💾 Trained Models:");
            for (stage, path) in &result.model_paths {
                println!("  {}: {}", stage, path);
            }
            
            println!("\n💡 Key Features Demonstrated:");
            println!("   • Multi-stage cascade training pipeline");
            println!("   • Dependency resolution and topological sorting");
            println!("   • Progressive training from pretraining to fusion");
            println!("   • Transfer learning and fine-tuning strategies");
            println!("   • Early stopping and learning rate scheduling");
            println!("   • Comprehensive validation and metrics tracking");
            println!("   • Model saving and checkpoint management");
            println!("   • Multi-modal fusion training");
        }
        Err(e) => {
            println!("❌ Training failed: {}", e);
        }
    }
    
    Ok(())
}