// Temporary stub for testing
class TranspilationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'TranspilationError';
    }
}

class WebGPUError extends Error {
    constructor(message) {
        super(message);
        this.name = 'WebGPUError';
    }
}

class KernelExecutionError extends Error {
    constructor(message) {
        super(message);
        this.name = 'KernelExecutionError';
    }
}

module.exports = {
    // Main functions
    transpileCuda: (code) => ({ success: true, output: code }),
    analyzeKernel: (kernel) => ({ complexity: 'low', memory: 0, threads: 1 }),
    benchmark: async () => ({ performance: 100, status: 'complete' }),
    getVersion: () => ({ version: '1.0.0', features: ['webgpu', 'wasm', 'simd'] }),
    validateCudaCode: (code) => ({ isValid: true, errors: [], warnings: [] }),
    parseCudaKernels: (code) => [{ name: 'kernel', parameters: [] }],
    isWebGPUAvailable: () => false, // Node.js doesn't have WebGPU
    configure: (options) => { /* no-op */ },
    
    // CUDA API compatibility
    cudaDeviceGetCount: () => 0,
    cudaSetDevice: () => {},
    cudaMalloc: () => null,
    cudaFree: () => {},
    cudaMemcpy: () => {},
    cudaMemcpyAsync: () => {},
    setKernelSource: () => {},
    launchKernel: () => {},
    transpileCudaToRust: () => "",
    createWebGPUContext: async () => ({}),
    createCudaContext: () => ({}),
    
    // Error classes
    TranspilationError,
    WebGPUError,
    KernelExecutionError,
    
    // Metadata
    version: "1.0.0",
    test: true
};