/* wasm-module-loader.js
 * Universal (ESM + CJS) progressive WASM loader.
 * Author: Bron refactor 2025‑07‑01
 */

import path from 'node:path';
import { promises as fs } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { webcrypto as nodeCrypto } from 'node:crypto';

// ────────────────────────────────────────────────────────────────────────────────
// helpers ────────────────────────────────────────────────────────────────────────
const crypto = globalThis.crypto ?? nodeCrypto;          // browser | Node
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ────────────────────────────────────────────────────────────────────────────────
class WasmModuleLoader {
  constructor () {
    this.modules         = new Map();
    this.loadingPromises = new Map();
    this.loadingStrategy = 'on-demand';                  // eager | on-demand | progressive
    this.baseDir         = __dirname;                    // one-liner dirname
    this.moduleManifest  = {
      /* The only compiled artefact today. Others are historical → optional */
      core: {
        path        : '../wasm/ruv_swarm_wasm_bg.wasm',
        jsBindings  : '../wasm/ruv_swarm_wasm.js',
        size        : 512 * 1024,
        priority    : 'high',
        dependencies: [],
        exists      : true,
        type        : 'wasm-bindgen'
      },

      /* legacy / optional stubs */
      neural      : { path: '../wasm/ruv-fann.wasm',              size: 1024 * 1024, priority: 'medium', dependencies: ['core'], exists: false, optional: true },
      forecasting : { path: '../wasm/neuro-divergent.wasm',       size: 1536 * 1024, priority: 'medium', dependencies: ['core'], exists: false, optional: true },
      swarm       : { path: '../wasm/ruv-swarm-orchestration.wasm',size: 768  * 1024, priority: 'high'  , dependencies: ['core'], exists: false, optional: true },
      persistence : { path: '../wasm/ruv-swarm-persistence.wasm',  size: 256  * 1024, priority: 'high'  , dependencies: ['core'], exists: false, optional: true }
    };
  }

  /* ── public API ───────────────────────────────────────────────────────────── */
  async initialize (strategy = 'progressive') {
    this.loadingStrategy = strategy;
    if (strategy === 'eager')       return this.#loadAllModules();
    if (strategy === 'progressive') return this.#loadCoreOnly();
    if (strategy === 'on-demand')   return this.#setupLazyProxies();
    throw new Error(`Unknown loading strategy: ${strategy}`);
  }

  async loadModule (name) {
    if (this.modules.has(name))          return this.modules.get(name);
    if (this.loadingPromises.has(name))  return this.loadingPromises.get(name);

    const info = this.moduleManifest[name];
    if (!info) throw new Error(`Unknown module: ${name}`);

    /* optional module not built – silently alias to core when possible */
    if (!info.exists && info.optional) {
      await this.loadModule('core');
      const coreMod = this.modules.get('core');
      this.modules.set(name, coreMod);
      return coreMod;
    }

    /* ensure deps first */
    await Promise.all(info.dependencies.map(dep => this.loadModule(dep)));

    const p = (name === 'core')
      ? this.#loadCoreBindings()
      : this.#instantiateRaw(name, info);

    this.loadingPromises.set(name, p);
    try {
      const m = await p;
      this.modules.set(name, m);
      console.log(`✅  Loaded WASM module: ${name} (${this.#fmt(info.size)})`);
      return m;
    } finally {
      this.loadingPromises.delete(name);
    }
  }

  getModuleStatus () {
    const s = {};
    for (const [n, i] of Object.entries(this.moduleManifest)) {
      s[n] = {
        loaded      : this.modules.has(n),
        loading     : this.loadingPromises.has(n),
        placeholder : this.modules.get(n)?.isPlaceholder ?? false,
        size        : i.size,
        priority    : i.priority,
        deps        : i.dependencies
      };
    }
    return s;
  }

  /* ── internal ─────────────────────────────────────────────────────────────── */
  async #instantiateRaw (name, info) {
    const wasmPath = path.join(this.baseDir, info.path);

    let buffer;
    if (typeof window !== 'undefined') {                 // browser
      const resp = await fetch(wasmPath);
      if (!resp.ok) throw new Error(`fetch failed: ${resp.statusText}`);
      buffer = await resp.arrayBuffer();
    } else {                                             // Node
      buffer = await fs.readFile(wasmPath).catch(() => null);
      if (!buffer) return this.#placeholder(name);
    }

    const imports  = this.#importsFor(name);
    const { instance, module } = await WebAssembly.instantiate(buffer, imports);
    return { instance, module, exports: instance.exports, memory: instance.exports.memory };
  }

  async #loadCoreBindings () {
    /* Use our enhanced WASM bindings loader */
    try {
      // Use dynamic import with URL for ES module compatibility
      const loaderURL = pathToFileURL(
        path.join(this.baseDir, '..', 'wasm', 'wasm-bindings-loader.mjs')
      ).href;
      
      // Import the loader module
      const loaderModule = await import(loaderURL);
      const bindingsLoader = loaderModule.default;
      
      // Initialize the loader
      await bindingsLoader.initialize();
      
      return {
        instance: { exports: bindingsLoader },
        module  : null,
        exports : bindingsLoader,
        memory  : bindingsLoader.memory,
        getTotalMemoryUsage: () => bindingsLoader.getTotalMemoryUsage()
      };
    } catch (error) {
      console.error('Failed to load core module via bindings loader:', error);
      return this.#placeholder('core');
    }
  }

  #importsFor (name) {
    const base = {
      env: { memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }) },
      wasi_snapshot_preview1: {
        proc_exit             : c => { throw new Error(`WASI exit ${c}`); },
        fd_write              : () => 0,
        /* …minimal stubs… */
        random_get            : (ptr, len) => {
          const view = new Uint8Array(base.env.memory.buffer, ptr, len);
          crypto.getRandomValues(view);
          return 0;
        }
      }
    };

    if (name === 'neural') {
      base.neural = { log_training_progress: (e, l) => console.log(`Epoch ${e} → loss ${l}`) };
    } else if (name === 'forecasting') {
      base.forecasting = { log_forecast: (m, h) => console.log(`Forecast ${m}, horizon ${h}`) };
    }
    return base;
  }

  #placeholder (name) {
    console.warn(`⚠️  Using placeholder for missing module '${name}'`);
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 10 });
    return {
      instance: { exports: { memory: mem } },
      module  : null,
      exports : { memory: mem },
      memory  : mem,
      isPlaceholder: true
    };
  }

  async #loadCoreOnly ()            { await this.loadModule('core'); }
  async #loadAllModules ()          { return Promise.all(Object.keys(this.moduleManifest).map(n => this.loadModule(n))); }
  #setupLazyProxies () {
    const proxyBag = {};
    for (const n of Object.keys(this.moduleManifest)) {
      proxyBag[n] = new Proxy({}, {
        get : (_, p) => {
          if (!this.modules.has(n)) throw new Error(
            `Module '${n}' not yet loaded; await loader.loadModule('${n}') first`
          );
          return this.modules.get(n).exports[p];
        }
      });
    }
    return proxyBag;
  }

  #fmt (b) {
    if (!b) return '0 B';
    const k = 1024, i = Math.floor(Math.log(b) / Math.log(k));
    return `${(b / k ** i).toFixed(1)} ${['B','KB','MB','GB'][i]}`;
  }
  
  getTotalMemoryUsage() {
    let totalBytes = 0;
    
    for (const [name, module] of this.modules.entries()) {
      if (module && module.memory && module.memory.buffer) {
        totalBytes += module.memory.buffer.byteLength;
      }
    }
    
    return totalBytes;
  }
}

// ────────────────────────────────────────────────────────────────────────────────
export { WasmModuleLoader };
export default WasmModuleLoader;           // ESM default

/* CJS interop --------------------------------------------------------------- */
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
  module.exports = WasmModuleLoader;       // require('./wasm-module-loader')
}
