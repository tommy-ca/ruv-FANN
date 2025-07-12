/**
 * TypeScript definitions for CUDA-WASM Neural Integration
 * 
 * Seamless integration between CUDA-WASM transpiler and ruv-FANN neural networks
 * for GPU-accelerated neural computation in web browsers and Node.js.
 * 
 * @version 1.0.0
 * @author CUDA-WASM Integration Team
 */

declare module "cuda-wasm-neural" {
  
  // ==== Core Configuration Types ====
  
  export interface BridgeConfig {
    enable_gpu: boolean;
    gpu_device: "auto" | "high_performance" | "low_power" | "discrete" | "integrated";
    memory_pool_size: number; // Memory pool size in MB
    enable_monitoring: boolean;
    auto_fallback: boolean;
    batch_size: number;
    precision: "float16" | "float32" | "float64";
  }
  
  export interface DeviceInfo {
    name: string;
    vendor: string;
    device_type: string;
    memory_size: number; // Memory size in bytes
    compute_units: number;
    max_workgroup_size: number;
    supports_f16: boolean;
    supports_f64: boolean;
  }
  
  export interface SystemCapabilities {
    cuda_transpilation: boolean;
    gpu_acceleration: boolean;
    wasm_support: boolean;
    performance_monitoring: boolean;
    memory_pooling: boolean;
    auto_fallback: boolean;
    batch_processing: boolean;
    precision_f16: boolean;
    precision_f32: boolean;
    precision_f64: boolean;
  }
  
  // ==== Operation Types ====
  
  export type ActivationFunction = 
    | "sigmoid"
    | "relu" 
    | "tanh"
    | "leaky_relu"
    | "swish"
    | "gelu";
  
  export interface MatrixMultiplyParams {
    a_rows: number;
    a_cols: number;
    b_cols: number;
  }
  
  export interface VectorAddParams {
    size: number;
  }
  
  export interface ActivationFunctionParams {
    function: ActivationFunction;
    size: number;
  }
  
  export interface ConvolutionParams {
    channels: number;
    kernel_size: number;
    stride: number;
  }
  
  export interface NeuralNetworkParams {
    layer_sizes: number[];
  }
  
  export interface CustomKernelParams {
    kernel_source: string;
    name: string;
  }
  
  // ==== Result Types ====
  
  export interface OperationResult {
    readonly data: Float32Array;
    readonly execution_time: number; // Time in milliseconds
    readonly gpu_used: boolean;
    readonly throughput: number; // Operations per second
  }
  
  export interface MemoryStats {
    readonly total_allocated: number;
    readonly gpu_allocated: number;
    readonly cpu_allocated: number;
    readonly peak_usage: number;
    readonly allocations: number;
    readonly deallocations: number;
  }
  
  export interface PerformanceStats {
    readonly total_operations: number;
    readonly average_execution_time: number; // Time in seconds
    readonly gpu_utilization: number; // 0.0 to 1.0
    readonly memory_bandwidth: number; // Bytes per second
    readonly throughput: number; // Operations per second
  }
  
  export interface PerformanceDegradation {
    operation: string;
    expected_time: number;
    actual_time: number;
    degradation_factor: number;
    suggested_action: string;
  }
  
  // ==== Main Classes ====
  
  /**
   * Main neural bridge for GPU-accelerated neural operations
   */
  export class NeuralBridge {
    /**
     * Create a new neural bridge with default configuration
     */
    constructor();
    
    /**
     * Create a neural bridge with custom configuration
     */
    static withConfig(config: Partial<BridgeConfig>): NeuralBridge;
    
    /**
     * Check if GPU acceleration is available
     */
    isGpuAvailable(): boolean;
    
    /**
     * Get GPU device information
     */
    getDeviceInfo(): DeviceInfo | null;
    
    /**
     * Execute matrix multiplication on GPU
     */
    matrixMultiply(
      a: Float32Array,
      b: Float32Array,
      rows_a: number,
      cols_a: number,
      cols_b: number
    ): Promise<OperationResult>;
    
    /**
     * Execute vector addition on GPU
     */
    vectorAdd(a: Float32Array, b: Float32Array): Promise<OperationResult>;
    
    /**
     * Apply activation function on GPU
     */
    activationFunction(
      input: Float32Array,
      function_name: ActivationFunction
    ): Promise<OperationResult>;
    
    /**
     * Execute forward propagation through neural network
     */
    forwardPropagation(
      input: Float32Array,
      layer_sizes: number[]
    ): Promise<OperationResult>;
    
    /**
     * Execute backward propagation for training
     */
    backwardPropagation(
      input: Float32Array,
      layer_sizes: number[]
    ): Promise<OperationResult>;
    
    /**
     * Execute custom CUDA kernel
     */
    executeCustomKernel(
      input: Float32Array,
      kernel_source: string,
      kernel_name: string
    ): Promise<OperationResult>;
    
    /**
     * Execute convolution operation
     */
    convolution(
      input: Float32Array,
      channels: number,
      kernel_size: number,
      stride: number
    ): Promise<OperationResult>;
    
    /**
     * Get current memory statistics
     */
    getMemoryStats(): MemoryStats;
    
    /**
     * Get performance statistics
     */
    getPerformanceStats(): PerformanceStats;
    
    /**
     * Detect performance degradation
     */
    detectPerformanceDegradation(): PerformanceDegradation | null;
    
    /**
     * Create a batch processor for efficient bulk operations
     */
    createBatchProcessor(): BatchProcessor;
    
    /**
     * Transpile CUDA code to WGSL
     */
    transpileCudaKernel(cuda_source: string): Promise<string>;
  }
  
  /**
   * Batch processor for efficient bulk operations
   */
  export class BatchProcessor {
    /**
     * Process multiple matrix multiplications in batch
     */
    batchMatrixMultiply(
      matrices: Array<{
        a: Float32Array;
        b: Float32Array;
        config: MatrixMultiplyParams;
      }>
    ): Promise<OperationResult[]>;
    
    /**
     * Process multiple activation functions in batch
     */
    batchActivationFunctions(
      inputs: Float32Array[],
      function_name: ActivationFunction
    ): Promise<OperationResult[]>;
    
    /**
     * Process multiple neural networks in batch
     */
    batchNeuralNetworks(
      inputs: Float32Array[],
      layer_sizes: number[][]
    ): Promise<OperationResult[]>;
    
    /**
     * Process mixed operations in batch
     */
    processBatch(
      operations: Array<{
        type: "matrix_multiply" | "vector_add" | "activation" | "neural_network" | "custom";
        input: Float32Array;
        params: any;
      }>
    ): Promise<OperationResult[]>;
  }
  
  // ==== Utility Functions ====
  
  /**
   * Initialize the neural integration system
   */
  export function initialize(): Promise<void>;
  
  /**
   * Get system capabilities
   */
  export function getSystemCapabilities(): SystemCapabilities;
  
  /**
   * Create default bridge configuration
   */
  export function createDefaultConfig(): BridgeConfig;
  
  /**
   * Create optimized configuration for specific use case
   */
  export function createOptimizedConfig(
    useCase: "inference" | "training" | "batch_processing" | "memory_constrained"
  ): BridgeConfig;
  
  /**
   * Benchmark neural operations
   */
  export function runBenchmark(config?: Partial<BridgeConfig>): Promise<BenchmarkResults>;
  
  export interface BenchmarkResults {
    vector_add: {
      sizes: number[];
      gpu_times: number[];
      cpu_times: number[];
      speedups: number[];
    };
    matrix_multiply: {
      sizes: number[];
      gpu_gflops: number[];
      cpu_gflops: number[];
      speedups: number[];
    };
    activation_functions: {
      [key in ActivationFunction]: {
        gpu_time: number;
        cpu_time: number;
        speedup: number;
        throughput: number;
      };
    };
    neural_networks: {
      architectures: number[][];
      gpu_times: number[];
      cpu_times: number[];
      speedups: number[];
    };
    overall_summary: {
      average_speedup: number;
      best_operations: string[];
      recommended_use_cases: string[];
    };
  }
  
  // ==== Advanced Features ====
  
  /**
   * Custom kernel compiler
   */
  export class KernelCompiler {
    /**
     * Compile CUDA source to optimized GPU kernel
     */
    static compileCuda(source: string, options?: CompilerOptions): Promise<CompiledKernel>;
    
    /**
     * Optimize kernel for specific GPU architecture
     */
    static optimizeForDevice(kernel: CompiledKernel, device: DeviceInfo): Promise<CompiledKernel>;
  }
  
  export interface CompilerOptions {
    optimization_level: "O0" | "O1" | "O2" | "O3";
    target_architecture?: string;
    enable_fast_math?: boolean;
    use_shared_memory?: boolean;
    workgroup_size?: [number, number, number];
  }
  
  export interface CompiledKernel {
    name: string;
    wgsl_source: string;
    entry_point: string;
    workgroup_size: [number, number, number];
    shared_memory_size: number;
    register_count: number;
    optimization_report: string;
  }
  
  /**
   * Performance profiler for detailed analysis
   */
  export class PerformanceProfiler {
    constructor(bridge: NeuralBridge);
    
    /**
     * Start profiling session
     */
    startProfiling(): void;
    
    /**
     * Stop profiling and get results
     */
    stopProfiling(): ProfilingResults;
    
    /**
     * Get real-time performance metrics
     */
    getMetrics(): PerformanceMetrics;
  }
  
  export interface ProfilingResults {
    total_time: number;
    gpu_time: number;
    memory_transfer_time: number;
    kernel_breakdown: {
      [kernel_name: string]: {
        call_count: number;
        total_time: number;
        average_time: number;
        occupancy: number;
      };
    };
    memory_usage: {
      peak_gpu: number;
      peak_cpu: number;
      transfer_count: number;
      transfer_volume: number;
    };
    bottlenecks: Array<{
      type: "memory_bandwidth" | "compute" | "memory_latency" | "cpu_gpu_sync";
      severity: "low" | "medium" | "high";
      description: string;
      suggestion: string;
    }>;
  }
  
  export interface PerformanceMetrics {
    current_gpu_utilization: number;
    current_memory_usage: number;
    throughput: number;
    latency: number;
    operations_per_second: number;
  }
  
  // ==== Integration with Popular ML Libraries ====
  
  /**
   * TensorFlow.js integration helpers
   */
  export namespace TensorFlowJS {
    /**
     * Convert TensorFlow.js tensor to CUDA-WASM format
     */
    function tensorToFloat32Array(tensor: any): Float32Array;
    
    /**
     * Convert CUDA-WASM result back to TensorFlow.js tensor
     */
    function float32ArrayToTensor(data: Float32Array, shape: number[]): any;
    
    /**
     * Create GPU-accelerated custom ops for TensorFlow.js
     */
    function createCustomOp(kernel_source: string, name: string): any;
  }
  
  /**
   * ONNX.js integration helpers
   */
  export namespace ONNXJS {
    /**
     * Execute ONNX model with CUDA-WASM acceleration
     */
    function runModel(model: any, inputs: Float32Array[]): Promise<Float32Array[]>;
    
    /**
     * Optimize ONNX model for GPU execution
     */
    function optimizeModel(model: any): Promise<any>;
  }
  
  // ==== Error Types ====
  
  export class NeuralIntegrationError extends Error {
    constructor(message: string, public code: string, public details?: any);
  }
  
  export class GpuInitializationError extends NeuralIntegrationError {
    constructor(message: string, details?: any);
  }
  
  export class TranspilationError extends NeuralIntegrationError {
    constructor(message: string, details?: any);
  }
  
  export class MemoryError extends NeuralIntegrationError {
    constructor(message: string, details?: any);
  }
  
  export class PerformanceError extends NeuralIntegrationError {
    constructor(message: string, details?: any);
  }
  
  // ==== Event System ====
  
  export interface EventMap {
    "operation_start": { operation: string; timestamp: number };
    "operation_complete": { operation: string; duration: number; success: boolean };
    "memory_pressure": { type: "cpu" | "gpu"; usage: number; threshold: number };
    "performance_degradation": PerformanceDegradation;
    "gpu_error": { error: string; recovery_action?: string };
    "fallback_triggered": { from: "gpu" | "cpu"; to: "gpu" | "cpu"; reason: string };
  }
  
  export class EventEmitter {
    on<K extends keyof EventMap>(event: K, listener: (data: EventMap[K]) => void): void;
    off<K extends keyof EventMap>(event: K, listener: (data: EventMap[K]) => void): void;
    emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void;
  }
  
  // ==== Worker Support ====
  
  /**
   * Web Worker wrapper for background neural processing
   */
  export class NeuralWorker {
    constructor(config?: Partial<BridgeConfig>);
    
    /**
     * Execute operation in background worker
     */
    execute(operation: any, input: Float32Array): Promise<OperationResult>;
    
    /**
     * Terminate worker
     */
    terminate(): void;
  }
  
  // ==== Version and Build Info ====
  
  export const VERSION: string;
  export const BUILD_DATE: string;
  export const CUDA_VERSION: string;
  export const WEBGPU_VERSION: string;
  export const FEATURES: string[];
}

// ==== Module Augmentation for Browser Globals ====

declare global {
  interface Window {
    CudaWasmNeural?: typeof import("cuda-wasm-neural");
  }
  
  interface Navigator {
    gpu?: {
      requestAdapter(options?: any): Promise<any>;
    };
  }
}

// ==== Usage Examples in Documentation ====

/**
 * @example Basic Usage
 * ```typescript
 * import { NeuralBridge, initialize } from 'cuda-wasm-neural';
 * 
 * // Initialize the system
 * await initialize();
 * 
 * // Create neural bridge
 * const bridge = new NeuralBridge();
 * 
 * // Check GPU availability
 * if (bridge.isGpuAvailable()) {
 *   console.log('GPU acceleration available!');
 *   console.log('Device:', bridge.getDeviceInfo());
 * }
 * 
 * // Perform matrix multiplication
 * const a = new Float32Array([1, 2, 3, 4]);
 * const b = new Float32Array([5, 6, 7, 8]);
 * const result = await bridge.matrixMultiply(a, b, 2, 2, 2);
 * 
 * console.log('Result:', result.data);
 * console.log('Execution time:', result.execution_time, 'ms');
 * console.log('GPU used:', result.gpu_used);
 * ```
 * 
 * @example Neural Network Forward Propagation
 * ```typescript
 * import { NeuralBridge } from 'cuda-wasm-neural';
 * 
 * const bridge = new NeuralBridge();
 * 
 * // Define network architecture: 4 inputs -> 8 hidden -> 2 outputs
 * const layerSizes = [4, 8, 2];
 * const input = new Float32Array([0.5, 0.8, 0.2, 0.9]);
 * 
 * const result = await bridge.forwardPropagation(input, layerSizes);
 * console.log('Network output:', result.data);
 * ```
 * 
 * @example Batch Processing
 * ```typescript
 * import { NeuralBridge } from 'cuda-wasm-neural';
 * 
 * const bridge = new NeuralBridge();
 * const batchProcessor = bridge.createBatchProcessor();
 * 
 * // Process multiple activation functions in batch
 * const inputs = [
 *   new Float32Array([-2, -1, 0, 1, 2]),
 *   new Float32Array([-1, 0, 1, 2, 3]),
 *   new Float32Array([0, 1, 2, 3, 4])
 * ];
 * 
 * const results = await batchProcessor.batchActivationFunctions(inputs, 'relu');
 * console.log('Batch results:', results.map(r => r.data));
 * ```
 * 
 * @example Custom CUDA Kernel
 * ```typescript
 * import { NeuralBridge } from 'cuda-wasm-neural';
 * 
 * const bridge = new NeuralBridge();
 * 
 * // Custom kernel for element-wise square
 * const kernelSource = `
 *   __global__ void element_square(float* input, float* output, int size) {
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (idx < size) {
 *       output[idx] = input[idx] * input[idx];
 *     }
 *   }
 * `;
 * 
 * const input = new Float32Array([1, 2, 3, 4, 5]);
 * const result = await bridge.executeCustomKernel(input, kernelSource, 'element_square');
 * console.log('Squared:', result.data); // [1, 4, 9, 16, 25]
 * ```
 * 
 * @example Performance Monitoring
 * ```typescript
 * import { NeuralBridge, PerformanceProfiler } from 'cuda-wasm-neural';
 * 
 * const bridge = new NeuralBridge();
 * const profiler = new PerformanceProfiler(bridge);
 * 
 * profiler.startProfiling();
 * 
 * // Perform operations...
 * await bridge.matrixMultiply(a, b, 100, 100, 100);
 * 
 * const results = profiler.stopProfiling();
 * console.log('Profiling results:', results);
 * console.log('Bottlenecks:', results.bottlenecks);
 * ```
 */