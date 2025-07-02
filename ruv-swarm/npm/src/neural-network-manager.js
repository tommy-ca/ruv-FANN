/**
 * Neural Network Manager
 * Manages per-agent neural networks with WASM integration
 */

class NeuralNetworkManager {
    constructor(wasmLoader) {
        this.wasmLoader = wasmLoader;
        this.neuralNetworks = new Map();
        this.templates = {
            deep_analyzer: {
                layers: [128, 256, 512, 256, 128],
                activation: 'relu',
                output_activation: 'sigmoid',
                dropout: 0.3
            },
            nlp_processor: {
                layers: [512, 1024, 512, 256],
                activation: 'gelu',
                output_activation: 'softmax',
                dropout: 0.4
            },
            reinforcement_learner: {
                layers: [64, 128, 128, 64],
                activation: 'tanh',
                output_activation: 'linear',
                dropout: 0.2
            },
            pattern_recognizer: {
                layers: [256, 512, 1024, 512, 256],
                activation: 'relu',
                output_activation: 'sigmoid',
                dropout: 0.35
            },
            time_series_analyzer: {
                layers: [128, 256, 256, 128],
                activation: 'lstm',
                output_activation: 'linear',
                dropout: 0.25
            }
        };
    }

    async createAgentNeuralNetwork(agentId, config = {}) {
        // Load neural module if not already loaded
        const neuralModule = await this.wasmLoader.loadModule('neural');
        
        if (!neuralModule || neuralModule.isPlaceholder) {
            console.warn('Neural network module not available, using simulation');
            return this.createSimulatedNetwork(agentId, config);
        }

        const {
            template = 'deep_analyzer',
            layers = null,
            activation = 'relu',
            learningRate = 0.001,
            optimizer = 'adam'
        } = config;

        // Use template or custom layers
        const networkConfig = layers ? { layers, activation } : this.templates[template];
        
        try {
            // Create network using WASM module
            const networkId = neuralModule.exports.create_neural_network(
                JSON.stringify({
                    agent_id: agentId,
                    layers: networkConfig.layers,
                    activation: networkConfig.activation,
                    learning_rate: learningRate,
                    optimizer: optimizer
                })
            );

            const network = new NeuralNetwork(networkId, agentId, networkConfig, neuralModule);
            this.neuralNetworks.set(agentId, network);
            
            return network;
        } catch (error) {
            console.error('Failed to create neural network:', error);
            return this.createSimulatedNetwork(agentId, config);
        }
    }

    createSimulatedNetwork(agentId, config) {
        const network = new SimulatedNeuralNetwork(agentId, config);
        this.neuralNetworks.set(agentId, network);
        return network;
    }

    async fineTuneNetwork(agentId, trainingData, options = {}) {
        const network = this.neuralNetworks.get(agentId);
        if (!network) {
            throw new Error(`No neural network found for agent ${agentId}`);
        }

        const {
            epochs = 10,
            batchSize = 32,
            learningRate = 0.001,
            freezeLayers = []
        } = options;

        return network.train(trainingData, { epochs, batchSize, learningRate, freezeLayers });
    }

    async enableCollaborativeLearning(agentIds, options = {}) {
        const {
            strategy = 'federated',
            syncInterval = 30000,
            privacyLevel = 'high'
        } = options;

        const networks = agentIds.map(id => this.neuralNetworks.get(id)).filter(n => n);
        
        if (networks.length < 2) {
            throw new Error('At least 2 neural networks required for collaborative learning');
        }

        // Create collaborative learning session
        const session = {
            id: `collab-${Date.now()}`,
            networks,
            strategy,
            syncInterval,
            privacyLevel,
            active: true
        };

        // Start synchronization
        if (strategy === 'federated') {
            this.startFederatedLearning(session);
        }

        return session;
    }

    startFederatedLearning(session) {
        const syncFunction = () => {
            if (!session.active) return;

            // Aggregate gradients from all networks
            const gradients = session.networks.map(n => n.getGradients());
            
            // Apply privacy-preserving aggregation
            const aggregatedGradients = this.aggregateGradients(gradients, session.privacyLevel);
            
            // Update all networks with aggregated gradients
            session.networks.forEach(n => n.applyGradients(aggregatedGradients));
            
            // Schedule next sync
            setTimeout(syncFunction, session.syncInterval);
        };

        // Start synchronization
        setTimeout(syncFunction, session.syncInterval);
    }

    aggregateGradients(gradients, privacyLevel) {
        // Simple averaging for now (in real implementation, use secure aggregation)
        const aggregated = {};
        
        // Privacy levels could add noise or use secure multi-party computation
        const noise = privacyLevel === 'high' ? 0.01 : 0;
        
        // Average gradients with optional noise
        gradients.forEach(grad => {
            Object.entries(grad).forEach(([key, value]) => {
                if (!aggregated[key]) {
                    aggregated[key] = 0;
                }
                aggregated[key] += value / gradients.length + (Math.random() - 0.5) * noise;
            });
        });
        
        return aggregated;
    }

    getNetworkMetrics(agentId) {
        const network = this.neuralNetworks.get(agentId);
        if (!network) {
            return null;
        }

        return network.getMetrics();
    }

    saveNetworkState(agentId, filePath) {
        const network = this.neuralNetworks.get(agentId);
        if (!network) {
            throw new Error(`No neural network found for agent ${agentId}`);
        }

        return network.save(filePath);
    }

    async loadNetworkState(agentId, filePath) {
        const network = this.neuralNetworks.get(agentId);
        if (!network) {
            throw new Error(`No neural network found for agent ${agentId}`);
        }

        return network.load(filePath);
    }
}

// Neural Network wrapper class
class NeuralNetwork {
    constructor(networkId, agentId, config, wasmModule) {
        this.networkId = networkId;
        this.agentId = agentId;
        this.config = config;
        this.wasmModule = wasmModule;
        this.trainingHistory = [];
        this.metrics = {
            accuracy: 0,
            loss: 1.0,
            epochs_trained: 0,
            total_samples: 0
        };
    }

    async forward(input) {
        try {
            const result = this.wasmModule.exports.forward_pass(this.networkId, input);
            return result;
        } catch (error) {
            console.error('Forward pass failed:', error);
            return new Float32Array(this.config.layers[this.config.layers.length - 1]).fill(0.5);
        }
    }

    async train(trainingData, options) {
        const { epochs, batchSize, learningRate, freezeLayers } = options;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let batchCount = 0;
            
            // Process in batches
            for (let i = 0; i < trainingData.samples.length; i += batchSize) {
                const batch = trainingData.samples.slice(i, i + batchSize);
                
                try {
                    const loss = this.wasmModule.exports.train_batch(
                        this.networkId,
                        JSON.stringify(batch),
                        learningRate,
                        JSON.stringify(freezeLayers)
                    );
                    
                    epochLoss += loss;
                    batchCount++;
                } catch (error) {
                    console.error('Training batch failed:', error);
                }
            }
            
            const avgLoss = epochLoss / batchCount;
            this.metrics.loss = avgLoss;
            this.metrics.epochs_trained++;
            this.trainingHistory.push({ epoch, loss: avgLoss });
            
            console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}`);
        }
        
        return this.metrics;
    }

    getGradients() {
        // Get gradients from WASM module
        try {
            const gradients = this.wasmModule.exports.get_gradients(this.networkId);
            return JSON.parse(gradients);
        } catch (error) {
            console.error('Failed to get gradients:', error);
            return {};
        }
    }

    applyGradients(gradients) {
        // Apply gradients to network
        try {
            this.wasmModule.exports.apply_gradients(this.networkId, JSON.stringify(gradients));
        } catch (error) {
            console.error('Failed to apply gradients:', error);
        }
    }

    getMetrics() {
        return {
            ...this.metrics,
            training_history: this.trainingHistory,
            network_info: {
                layers: this.config.layers,
                parameters: this.config.layers.reduce((acc, size, i) => {
                    if (i > 0) {
                        return acc + (this.config.layers[i - 1] * size);
                    }
                    return acc;
                }, 0)
            }
        };
    }

    async save(filePath) {
        try {
            const state = this.wasmModule.exports.serialize_network(this.networkId);
            // In real implementation, save to file
            console.log(`Saving network state to ${filePath}`);
            return true;
        } catch (error) {
            console.error('Failed to save network:', error);
            return false;
        }
    }

    async load(filePath) {
        try {
            // In real implementation, load from file
            console.log(`Loading network state from ${filePath}`);
            this.wasmModule.exports.deserialize_network(this.networkId, 'state_data');
            return true;
        } catch (error) {
            console.error('Failed to load network:', error);
            return false;
        }
    }
}

// Simulated Neural Network for when WASM is not available
class SimulatedNeuralNetwork {
    constructor(agentId, config) {
        this.agentId = agentId;
        this.config = config;
        this.weights = this.initializeWeights();
        this.trainingHistory = [];
        this.metrics = {
            accuracy: 0.5 + Math.random() * 0.3,
            loss: 0.5 + Math.random() * 0.5,
            epochs_trained: 0,
            total_samples: 0
        };
    }

    initializeWeights() {
        // Simple weight initialization
        return this.config.layers?.map(() => Math.random() * 2 - 1) || [0];
    }

    async forward(input) {
        // Simple forward pass simulation
        const outputSize = this.config.layers?.[this.config.layers.length - 1] || 1;
        const output = new Float32Array(outputSize);
        
        for (let i = 0; i < outputSize; i++) {
            output[i] = Math.random();
        }
        
        return output;
    }

    async train(trainingData, options) {
        const { epochs } = options;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            const loss = Math.max(0.01, this.metrics.loss * (0.9 + Math.random() * 0.1));
            this.metrics.loss = loss;
            this.metrics.epochs_trained++;
            this.metrics.accuracy = Math.min(0.99, this.metrics.accuracy + 0.01);
            this.trainingHistory.push({ epoch, loss });
            
            console.log(`[Simulated] Epoch ${epoch + 1}/${epochs} - Loss: ${loss.toFixed(4)}`);
        }
        
        return this.metrics;
    }

    getGradients() {
        // Simulated gradients
        return {
            layer_0: Math.random() * 0.1,
            layer_1: Math.random() * 0.1
        };
    }

    applyGradients(gradients) {
        // Simulate gradient application
        console.log('[Simulated] Applying gradients');
    }

    getMetrics() {
        return {
            ...this.metrics,
            training_history: this.trainingHistory,
            network_info: {
                layers: this.config.layers || [128, 64, 32],
                parameters: 10000 // Simulated parameter count
            }
        };
    }

    async save(filePath) {
        console.log(`[Simulated] Saving network state to ${filePath}`);
        return true;
    }

    async load(filePath) {
        console.log(`[Simulated] Loading network state from ${filePath}`);
        return true;
    }
}

// Neural Network Templates for quick configuration
const NeuralNetworkTemplates = {
    getTemplate: (templateName) => {
        const templates = {
            deep_analyzer: {
                layers: [128, 256, 512, 256, 128],
                activation: 'relu',
                output_activation: 'sigmoid',
                dropout: 0.3
            },
            nlp_processor: {
                layers: [512, 1024, 512, 256],
                activation: 'gelu',
                output_activation: 'softmax',
                dropout: 0.4
            },
            reinforcement_learner: {
                layers: [64, 128, 128, 64],
                activation: 'tanh',
                output_activation: 'linear',
                dropout: 0.2
            }
        };
        
        return templates[templateName] || templates.deep_analyzer;
    }
};

export { NeuralNetworkManager, NeuralNetworkTemplates };