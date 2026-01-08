/**
 * WebGPU Accelerated Vector Search
 *
 * GPU-accelerated similarity search using WebGPU compute shaders.
 * Provides 10-100x performance improvement for large datasets.
 *
 * Features:
 * - GPU-accelerated cosine similarity computation
 * - Parallel batch processing
 * - Automatic device detection and CPU fallback
 * - Memory-efficient GPU buffer management
 * - Performance comparison (GPU vs CPU)
 */
import { VectorSearchError } from './errors';
export interface WebGPUSearchOptions {
    useGPU?: boolean;
    batchSize?: number;
    enableTiming?: boolean;
}
export interface SearchResult {
    index: number;
    similarity: number;
}
export interface BatchSearchResult {
    results: SearchResult[][];
    gpuTime: number;
    cpuTime: number;
    usedGPU: boolean;
}
export interface PerformanceMetrics {
    gpuTime: number;
    cpuTime: number;
    speedup: number;
    vectorCount: number;
    vectorDimension: number;
    usedGPU: boolean;
}
export declare class WebGPUUnsupportedError extends VectorSearchError {
    constructor();
}
export declare class WebGPUInitializationError extends VectorSearchError {
    constructor(message: string, cause?: Error);
}
export declare class WebGPUVectorSearch {
    private device;
    private initialized;
    private useGPU;
    private batchSize;
    private enableTiming;
    private dimension;
    private metrics;
    constructor(dimension?: number, options?: WebGPUSearchOptions);
    /**
     * Initialize WebGPU device
     *
     * @throws {WebGPUUnsupportedError} If WebGPU is not supported
     * @throws {WebGPUInitializationError} If device initialization fails
     */
    initializeGPU(): Promise<void>;
    /**
     * Check if WebGPU is available and initialized
     */
    isGPUSupported(): boolean;
    /**
     * Check if WebGPU is supported in current browser
     */
    static isBrowserSupported(): boolean;
    /**
     * Search for similar vectors using GPU acceleration
     *
     * @param query - Query vector
     * @param vectors - Flat array of vectors to search
     * @param k - Number of top results to return
     * @returns Array of search results with indices and similarities
     */
    search(query: number[], vectors: number[], k: number): Promise<SearchResult[]>;
    /**
     * Batch search using GPU acceleration
     *
     * Processes multiple queries in parallel on GPU.
     *
     * @param queries - Array of query vectors
     * @param vectors - Flat array of vectors to search
     * @param k - Number of top results per query
     * @returns Batch search results with timing information
     */
    batchSearch(queries: number[][], vectors: number[], k: number): Promise<BatchSearchResult>;
    private searchGPU;
    private batchSearchGPU;
    private searchCPU;
    /**
     * Create a GPU buffer from data
     */
    private createBuffer;
    /**
     * Get WebGPU compute shader for cosine similarity
     */
    private getCosineSimilarityShader;
    /**
     * Calculate optimal batch size based on vector dimension
     */
    private calculateOptimalBatchSize;
    /**
     * Get performance metrics for all searches
     */
    getMetrics(): PerformanceMetrics[];
    /**
     * Get average speedup across all searches
     */
    getAverageSpeedup(): number;
    /**
     * Clear performance metrics
     */
    clearMetrics(): void;
    /**
     * Get performance comparison summary
     */
    getPerformanceSummary(): string;
    /**
     * Clean up GPU resources
     */
    destroy(): void;
}
//# sourceMappingURL=webgpu-vector-search.d.ts.map