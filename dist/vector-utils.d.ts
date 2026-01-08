/**
 * Vector Math Utilities
 *
 * Shared utility functions for vector operations.
 * Optimized for in-browser semantic search.
 */
/** Default embedding dimension for common models (e.g., sentence-transformers) */
export declare const DEFAULT_EMBEDDING_DIM = 384;
/** Minimum value considered non-zero for normalization safety */
export declare const EPSILON = 1e-10;
/** Character-to-token ratio for rough token estimation */
export declare const CHARS_PER_TOKEN = 4;
/** Maximum cache size for embedding cache (LRU eviction) */
export declare const MAX_EMBEDDING_CACHE_SIZE = 1000;
/** Minimum word length for keyword matching in hybrid search */
export declare const MIN_KEYWORD_LENGTH = 3;
/** Keyword match boost factor in hybrid search */
export declare const KEYWORD_MATCH_BOOST = 0.05;
/** Default similarity threshold for semantic search */
export declare const DEFAULT_SIMILARITY_THRESHOLD = 0.7;
/** Default result limit for semantic search */
export declare const DEFAULT_SEARCH_LIMIT = 10;
/** Hash seed multiplier for hash-based embeddings */
export declare const HASH_MULTIPLIER = 1103515245;
/** Hash seed additive constant for hash-based embeddings */
export declare const HASH_ADDITIVE = 12345;
/** Hash bitmask for 32-bit integer generation */
export declare const HASH_BITMASK = 2147483647;
/** Hash modulo for vector normalization */
export declare const HASH_MODULO = 1000;
/** Vector slot boost for hash-based embeddings */
export declare const VECTOR_SLOT_BOOST = 0.3;
/**
 * Calculates cosine similarity between two vectors.
 *
 * This is the standard similarity metric for vector search.
 * Returns a value between 0 (orthogonal) and 1 (identical direction).
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine similarity value between 0 and 1
 * @returns 0 if vectors have different lengths or zero magnitude
 *
 * @example
 * ```typescript
 * const vec1 = [1, 2, 3]
 * const vec2 = [2, 4, 6]
 * const similarity = cosineSimilarity(vec1, vec2)
 * console.log(similarity) // 1.0 (parallel vectors)
 * ```
 */
export declare function cosineSimilarity(a: number[], b: number[]): number;
/**
 * Normalizes a vector to unit length (L2 normalization).
 *
 * @param v - The vector to normalize
 * @returns A new unit-length vector pointing in the same direction
 * @returns The original vector if it has zero magnitude
 *
 * @example
 * ```typescript
 * const vec = [3, 4]
 * const normalized = normalizeVector(vec)
 * console.log(normalized) // [0.6, 0.8]
 * ```
 */
export declare function normalizeVector(v: number[]): number[];
/**
 * Calculates the dot product of two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The dot product (sum of element-wise products)
 * @throws {Error} If vectors have different lengths
 *
 * @example
 * ```typescript
 * const dotProduct = dotProduct([1, 2, 3], [4, 5, 6])
 * console.log(dotProduct) // 32 (1*4 + 2*5 + 3*6)
 * ```
 */
export declare function dotProduct(a: number[], b: number[]): number;
/**
 * Calculates the Euclidean (L2) distance between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The Euclidean distance
 *
 * @example
 * ```typescript
 * const distance = euclideanDistance([0, 0], [3, 4])
 * console.log(distance) // 5
 * ```
 */
export declare function euclideanDistance(a: number[], b: number[]): number;
/**
 * Generates a hash-based embedding for text (placeholder for real embeddings).
 *
 * This is a simple deterministic embedding based on text hashing.
 * In production, replace with real embeddings from WebLLM or an API.
 *
 * @param text - The input text
 * @param dimensions - The embedding dimension size
 * @returns A vector of the specified dimensions
 *
 * @example
 * ```typescript
 * const embedding = hashEmbedding('hello world', 384)
 * console.log(embedding.length) // 384
 * ```
 */
export declare function hashEmbedding(text: string, dimensions: number): number[];
/**
 * Calculates cosine similarities between a query vector and multiple vectors.
 *
 * @param query - The query vector
 * @param vectors - Flat array of multiple vectors
 * @param dimension - The dimension of each vector
 * @returns Array of similarity scores
 *
 * @example
 * ```typescript
 * const query = [1, 2]
 * const vectors = [1, 2, 3, 4, 5, 6] // 3 vectors of dimension 2
 * const scores = batchCosineSimilarity(query, vectors, 2)
 * console.log(scores) // [1.0, 0.93, ...]
 * ```
 */
export declare function batchCosineSimilarity(query: number[], vectors: number[], dimension: number): number[];
/**
 * Finds the top-k most similar vectors to a query.
 *
 * @param query - The query vector
 * @param vectors - Flat array of multiple vectors
 * @param dimension - The dimension of each vector
 * @param k - Number of top results to return
 * @returns Flat array alternating [index, score, index, score, ...]
 *
 * @example
 * ```typescript
 * const query = [1, 2]
 * const vectors = [1, 2, 3, 4, 5, 6]
 * const topK = topKSimilar(query, vectors, 2, 2)
 * console.log(topK) // [0, 1.0, 1, 0.93]
 * ```
 */
export declare function topKSimilar(query: number[], vectors: number[], dimension: number, k: number): number[];
/**
 * Calculates the mean of multiple vectors.
 *
 * @param vectors - Flat array of multiple vectors
 * @param dimension - The dimension of each vector
 * @returns The mean vector
 *
 * @example
 * ```typescript
 * const vectors = [1, 2, 3, 4, 5, 6] // 3 vectors of dimension 2
 * const mean = vectorMean(vectors, 2)
 * console.log(mean) // [3, 4]
 * ```
 */
export declare function vectorMean(vectors: number[], dimension: number): number[];
/**
 * Calculates the weighted sum of multiple vectors.
 *
 * @param vectors - Flat array of multiple vectors
 * @param weights - Weight for each vector
 * @param dimension - The dimension of each vector
 * @returns The weighted sum vector
 *
 * @example
 * ```typescript
 * const vectors = [1, 2, 3, 4] // 2 vectors of dimension 2
 * const weights = [0.3, 0.7]
 * const weighted = weightedSum(vectors, weights, 2)
 * console.log(weighted) // [2.4, 3.4]
 * ```
 */
export declare function weightedSum(vectors: number[], weights: number[], dimension: number): number[];
/**
 * Estimates the memory size required to store vectors.
 *
 * @param numVectors - Number of vectors
 * @param dimension - Dimension of each vector
 * @returns Estimated size in bytes (assuming float32)
 *
 * @example
 * ```typescript
 * const size = estimateMemorySize(1000, 384)
 * console.log(size) // 1536000 (~1.5MB)
 * ```
 */
export declare function estimateMemorySize(numVectors: number, dimension: number): number;
/**
 * Recommends a batch size based on vector dimension.
 *
 * @param vectorDimension - The dimension of vectors
 * @returns Recommended batch size for processing
 *
 * @example
 * ```typescript
 * const batchSize = recommendedBatchSize(384)
 * console.log(batchSize) // 128
 * ```
 */
export declare function recommendedBatchSize(vectorDimension: number): number;
/**
 * Estimates the number of tokens in a text string.
 *
 * Uses a simple approximation: ~4 characters per token.
 * For accurate counts, use a proper tokenizer.
 *
 * @param text - The text to estimate tokens for
 * @returns Estimated token count
 *
 * @example
 * ```typescript
 * const tokens = estimateTokens('Hello world')
 * console.log(tokens) // ~3
 * ```
 */
export declare function estimateTokens(text: string): number;
//# sourceMappingURL=vector-utils.d.ts.map