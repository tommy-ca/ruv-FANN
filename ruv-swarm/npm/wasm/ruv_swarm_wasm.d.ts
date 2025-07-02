/* tslint:disable */
/* eslint-disable */
export function js_array_to_vec_f32(array: any): Float32Array;
export function vec_f32_to_js_array(vec: Float32Array): Float32Array;
export function get_wasm_memory_usage(): bigint;
export function console_log(message: string): void;
export function console_error(message: string): void;
export function console_warn(message: string): void;
export function format_js_error(error: any): string;
/**
 * SIMD feature detection and runtime capabilities
 */
export function detect_simd_capabilities(): string;
export function run_simd_verification_suite(): string;
export function simd_performance_report(size: number, iterations: number): string;
/**
 * Comprehensive SIMD feature validation
 */
export function validate_simd_implementation(): boolean;
export function init(): void;
export function create_neural_network(layers: Uint32Array, activation: ActivationFunction): WasmNeuralNetwork;
export function create_swarm_orchestrator(topology: string): WasmSwarmOrchestrator;
export function create_forecasting_model(model_type: string): WasmForecastingModel;
export function get_version(): string;
export function get_features(): string;
export enum ActivationFunction {
  Linear = 0,
  Sigmoid = 1,
  SymmetricSigmoid = 2,
  Tanh = 3,
  ReLU = 4,
  LeakyReLU = 5,
  Swish = 6,
  Gaussian = 7,
  Elliot = 8,
  SymmetricElliot = 9,
  Sine = 10,
  Cosine = 11,
  SinSymmetric = 12,
  CosSymmetric = 13,
  ThresholdSymmetric = 14,
  Threshold = 15,
  StepSymmetric = 16,
  Step = 17,
}
export class PerformanceTimer {
  free(): void;
  constructor(name: string);
  elapsed(): number;
  log(): void;
}
export class RuntimeFeatures {
  free(): void;
  constructor();
  get_features_json(): string;
  readonly simd_available: boolean;
  readonly threads_available: boolean;
  readonly memory_limit: bigint;
}
/**
 * Performance benchmarking utilities
 */
export class SimdBenchmark {
  free(): void;
  constructor();
  /**
   * Benchmark SIMD vs scalar dot product
   */
  benchmark_dot_product(size: number, iterations: number): string;
  /**
   * Benchmark SIMD vs scalar activation functions
   */
  benchmark_activation(size: number, iterations: number, activation: string): string;
}
/**
 * SIMD-accelerated matrix operations
 */
export class SimdMatrixOps {
  free(): void;
  constructor();
  /**
   * SIMD-optimized matrix-vector multiplication
   */
  matrix_vector_multiply(matrix: Float32Array, vector: Float32Array, rows: number, cols: number): Float32Array;
  /**
   * SIMD-optimized matrix-matrix multiplication (small matrices)
   */
  matrix_multiply(a: Float32Array, b: Float32Array, a_rows: number, a_cols: number, b_cols: number): Float32Array;
}
/**
 * SIMD-accelerated vector operations
 */
export class SimdVectorOps {
  free(): void;
  constructor();
  /**
   * SIMD-optimized vector dot product
   */
  dot_product(a: Float32Array, b: Float32Array): number;
  /**
   * SIMD-optimized vector addition
   */
  vector_add(a: Float32Array, b: Float32Array): Float32Array;
  /**
   * SIMD-optimized vector scaling
   */
  vector_scale(vec: Float32Array, scalar: number): Float32Array;
  /**
   * SIMD-optimized activation function application
   */
  apply_activation(vec: Float32Array, activation: string): Float32Array;
}
export class WasmAgent {
  free(): void;
  constructor(id: string, agent_type: string);
  set_status(status: string): void;
  add_capability(capability: string): void;
  has_capability(capability: string): boolean;
  readonly id: string;
  readonly agent_type: string;
  readonly status: string;
}
export class WasmForecastingModel {
  free(): void;
  constructor(model_type: string);
  predict(input: Float64Array): Float64Array;
  get_model_type(): string;
}
export class WasmNeuralNetwork {
  free(): void;
  constructor(layers: Uint32Array, activation: ActivationFunction);
  randomize_weights(min: number, max: number): void;
  set_weights(weights: Float64Array): void;
  get_weights(): Float64Array;
  run(inputs: Float64Array): Float64Array;
}
export class WasmSwarmOrchestrator {
  free(): void;
  constructor(topology: string);
  spawn(config: string): string;
  orchestrate(config: string): WasmTaskResult;
  add_agent(agent_id: string): void;
  get_agent_count(): number;
  get_topology(): string;
  get_status(detailed: boolean): string;
}
export class WasmTaskResult {
  private constructor();
  free(): void;
  readonly task_id: string;
  readonly description: string;
  readonly status: string;
  readonly assigned_agents: string[];
  readonly priority: string;
}
