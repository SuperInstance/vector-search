/**
 * Comprehensive tests for Vector Math Utilities
 */

import { describe, it, expect } from 'vitest'
import {
  cosineSimilarity,
  normalizeVector,
  dotProduct,
  euclideanDistance,
  hashEmbedding,
  batchCosineSimilarity,
  topKSimilar,
  vectorMean,
  weightedSum,
  estimateMemorySize,
  recommendedBatchSize,
  estimateTokens,
  DEFAULT_EMBEDDING_DIM,
  DEFAULT_SIMILARITY_THRESHOLD,
  DEFAULT_SEARCH_LIMIT,
  EPSILON,
  CHARS_PER_TOKEN,
  MAX_EMBEDDING_CACHE_SIZE,
  MIN_KEYWORD_LENGTH,
  KEYWORD_MATCH_BOOST,
} from './vector-utils'

// ============================================================================
// CONSTANTS
// ============================================================================

describe('Constants', () => {
  it('should have correct DEFAULT_EMBEDDING_DIM', () => {
    expect(DEFAULT_EMBEDDING_DIM).toBe(384)
  })

  it('should have correct EPSILON', () => {
    expect(EPSILON).toBe(1e-10)
  })

  it('should have correct CHARS_PER_TOKEN', () => {
    expect(CHARS_PER_TOKEN).toBe(4)
  })

  it('should have correct MAX_EMBEDDING_CACHE_SIZE', () => {
    expect(MAX_EMBEDDING_CACHE_SIZE).toBe(1000)
  })

  it('should have correct MIN_KEYWORD_LENGTH', () => {
    expect(MIN_KEYWORD_LENGTH).toBe(3)
  })

  it('should have correct KEYWORD_MATCH_BOOST', () => {
    expect(KEYWORD_MATCH_BOOST).toBe(0.05)
  })

  it('should have correct DEFAULT_SIMILARITY_THRESHOLD', () => {
    expect(DEFAULT_SIMILARITY_THRESHOLD).toBe(0.7)
  })

  it('should have correct DEFAULT_SEARCH_LIMIT', () => {
    expect(DEFAULT_SEARCH_LIMIT).toBe(10)
  })
})

// ============================================================================
// COSINE SIMILARITY
// ============================================================================

describe('cosineSimilarity', () => {
  it('should return 1.0 for identical vectors', () => {
    const a = [1, 2, 3, 4, 5]
    expect(cosineSimilarity(a, a)).toBeCloseTo(1.0, 10)
  })

  it('should return 1.0 for parallel vectors', () => {
    const a = [1, 2, 3]
    const b = [2, 4, 6]
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 10)
  })

  it('should return 0.0 for orthogonal vectors', () => {
    const a = [1, 0]
    const b = [0, 1]
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 10)
  })

  it('should return -1.0 for opposite vectors', () => {
    const a = [1, 2, 3]
    const b = [-1, -2, -3]
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 10)
  })

  it('should return 0.0 for zero vectors', () => {
    const a = [0, 0, 0]
    const b = [1, 2, 3]
    expect(cosineSimilarity(a, b)).toBe(0)
  })

  it('should return 0.0 when both vectors are zero', () => {
    const a = [0, 0, 0]
    const b = [0, 0, 0]
    expect(cosineSimilarity(a, b)).toBe(0)
  })

  it('should return 0.0 for vectors with different lengths', () => {
    const a = [1, 2, 3]
    const b = [1, 2]
    expect(cosineSimilarity(a, b)).toBe(0)
  })

  it('should handle single dimension vectors', () => {
    expect(cosineSimilarity([5], [5])).toBeCloseTo(1.0, 10)
    expect(cosineSimilarity([5], [0])).toBe(0)
    expect(cosineSimilarity([5], [-5])).toBeCloseTo(-1.0, 10)
  })

  it('should handle 2D vectors correctly', () => {
    const a = [3, 4]
    const b = [4, 3]
    const expected = (3 * 4 + 4 * 3) / (5 * 5) // 24/25 = 0.96
    expect(cosineSimilarity(a, b)).toBeCloseTo(expected, 10)
  })

  it('should handle high dimensional vectors', () => {
    const a = Array.from({ length: 384 }, (_, i) => Math.sin(i))
    const b = Array.from({ length: 384 }, (_, i) => Math.cos(i))
    const result = cosineSimilarity(a, b)
    expect(result).toBeGreaterThanOrEqual(-1)
    expect(result).toBeLessThanOrEqual(1)
  })

  it('should handle vectors with negative values', () => {
    const a = [-1, -2, -3]
    const b = [1, 2, 3]
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 10)
  })

  it('should handle vectors with fractional values', () => {
    const a = [0.1, 0.2, 0.3]
    const b = [0.2, 0.4, 0.6]
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 10)
  })

  it('should handle near-identical vectors', () => {
    const a = [1, 2, 3, 4, 5]
    const b = [1.0001, 2.0001, 3.0001, 4.0001, 5.0001]
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 4)
  })
})

// ============================================================================
// NORMALIZE VECTOR
// ============================================================================

describe('normalizeVector', () => {
  it('should normalize a 2D vector correctly', () => {
    const v = [3, 4]
    const normalized = normalizeVector(v)
    expect(normalized[0]).toBeCloseTo(0.6, 10)
    expect(normalized[1]).toBeCloseTo(0.8, 10)
  })

  it('should produce a unit-length vector', () => {
    const v = [1, 2, 3, 4, 5]
    const normalized = normalizeVector(v)
    const magnitude = Math.sqrt(normalized.reduce((sum, val) => sum + val * val, 0))
    expect(magnitude).toBeCloseTo(1.0, 10)
  })

  it('should return original vector for zero magnitude', () => {
    const v = [0, 0, 0]
    const result = normalizeVector(v)
    expect(result).toEqual([0, 0, 0])
  })

  it('should preserve direction', () => {
    const v = [1, 2, 3]
    const normalized = normalizeVector(v)
    const ratio = normalized[0] / v[0]
    expect(normalized[1] / v[1]).toBeCloseTo(ratio, 10)
    expect(normalized[2] / v[2]).toBeCloseTo(ratio, 10)
  })

  it('should handle single element vectors', () => {
    const v = [5]
    const normalized = normalizeVector(v)
    expect(normalized[0]).toBeCloseTo(1.0, 10)
  })

  it('should handle negative values', () => {
    const v = [-3, -4]
    const normalized = normalizeVector(v)
    expect(normalized[0]).toBeCloseTo(-0.6, 10)
    expect(normalized[1]).toBeCloseTo(-0.8, 10)
  })

  it('should handle already normalized vectors', () => {
    const v = [0.6, 0.8] // already unit length
    const normalized = normalizeVector(v)
    expect(normalized[0]).toBeCloseTo(0.6, 10)
    expect(normalized[1]).toBeCloseTo(0.8, 10)
  })
})

// ============================================================================
// DOT PRODUCT
// ============================================================================

describe('dotProduct', () => {
  it('should calculate dot product correctly', () => {
    const a = [1, 2, 3]
    const b = [4, 5, 6]
    expect(dotProduct(a, b)).toBe(32) // 1*4 + 2*5 + 3*6
  })

  it('should return 0 for orthogonal vectors', () => {
    const a = [1, 0]
    const b = [0, 1]
    expect(dotProduct(a, b)).toBe(0)
  })

  it('should throw for mismatched dimensions', () => {
    const a = [1, 2, 3]
    const b = [1, 2]
    expect(() => dotProduct(a, b)).toThrow('Vector dimension mismatch')
  })

  it('should handle empty vectors', () => {
    expect(dotProduct([], [])).toBe(0)
  })

  it('should handle single dimension', () => {
    expect(dotProduct([5], [3])).toBe(15)
  })

  it('should handle negative values', () => {
    const a = [1, -2, 3]
    const b = [4, 5, -6]
    expect(dotProduct(a, b)).toBe(-24) // 4 - 10 - 18
  })

  it('should handle zero vectors', () => {
    const a = [0, 0, 0]
    const b = [1, 2, 3]
    expect(dotProduct(a, b)).toBe(0)
  })

  it('should match cosine similarity formula: dot(a,b) = |a| * |b| * cos(theta)', () => {
    const a = [1, 0]
    const b = [1, 1]
    const dp = dotProduct(a, b)
    const normA = Math.sqrt(dotProduct(a, a))
    const normB = Math.sqrt(dotProduct(b, b))
    const cosAngle = dp / (normA * normB)
    expect(cosAngle).toBeCloseTo(Math.SQRT1_2, 10) // cos(45 degrees)
  })
})

// ============================================================================
// EUCLIDEAN DISTANCE
// ============================================================================

describe('euclideanDistance', () => {
  it('should calculate distance for 2D vectors', () => {
    const a = [0, 0]
    const b = [3, 4]
    expect(euclideanDistance(a, b)).toBeCloseTo(5, 10)
  })

  it('should return 0 for identical vectors', () => {
    const a = [1, 2, 3, 4, 5]
    expect(euclideanDistance(a, a)).toBe(0)
  })

  it('should throw for mismatched dimensions', () => {
    const a = [1, 2, 3]
    const b = [1, 2]
    expect(() => euclideanDistance(a, b)).toThrow('Vector dimension mismatch')
  })

  it('should be symmetric', () => {
    const a = [1, 5, 3]
    const b = [4, 2, 8]
    expect(euclideanDistance(a, b)).toBeCloseTo(euclideanDistance(b, a), 10)
  })

  it('should satisfy triangle inequality', () => {
    const a = [0, 0]
    const b = [3, 4]
    const c = [6, 8]
    const dAB = euclideanDistance(a, b)
    const dBC = euclideanDistance(b, c)
    const dAC = euclideanDistance(a, c)
    expect(dAB + dBC).toBeGreaterThanOrEqual(dAC)
  })

  it('should handle empty vectors', () => {
    expect(euclideanDistance([], [])).toBe(0)
  })

  it('should handle negative coordinates', () => {
    const a = [-1, -2]
    const b = [2, 1]
    expect(euclideanDistance(a, b)).toBeCloseTo(Math.sqrt(9 + 9), 10) // sqrt(18)
  })

  it('should handle fractional values', () => {
    const a = [0.5, 0.5]
    const b = [1.5, 1.5]
    expect(euclideanDistance(a, b)).toBeCloseTo(Math.sqrt(2), 10)
  })
})

// ============================================================================
// HASH EMBEDDING
// ============================================================================

describe('hashEmbedding', () => {
  it('should return vector of correct dimensions', () => {
    const embedding = hashEmbedding('hello world', 384)
    expect(embedding).toHaveLength(384)
  })

  it('should return values between 0 and 1', () => {
    const embedding = hashEmbedding('test text', 100)
    for (const val of embedding) {
      expect(val).toBeGreaterThanOrEqual(0)
      expect(val).toBeLessThanOrEqual(1)
    }
  })

  it('should be deterministic (same input = same output)', () => {
    const a = hashEmbedding('test', 128)
    const b = hashEmbedding('test', 128)
    expect(a).toEqual(b)
  })

  it('should produce different embeddings for different texts', () => {
    const a = hashEmbedding('hello', 128)
    const b = hashEmbedding('world', 128)
    let different = false
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) {
        different = true
        break
      }
    }
    expect(different).toBe(true)
  })

  it('should handle empty string', () => {
    const embedding = hashEmbedding('', 50)
    expect(embedding).toHaveLength(50)
  })

  it('should handle very long text', () => {
    const longText = 'a'.repeat(10000)
    const embedding = hashEmbedding(longText, 384)
    expect(embedding).toHaveLength(384)
  })

  it('should handle special characters', () => {
    const embedding = hashEmbedding('hello 🌍 world! @#$%^&*()', 64)
    expect(embedding).toHaveLength(64)
  })

  it('should handle custom dimensions', () => {
    expect(hashEmbedding('test', 64)).toHaveLength(64)
    expect(hashEmbedding('test', 128)).toHaveLength(128)
    expect(hashEmbedding('test', 768)).toHaveLength(768)
    expect(hashEmbedding('test', 1536)).toHaveLength(1536)
  })

  it('should be case sensitive', () => {
    const a = hashEmbedding('Hello', 64)
    const b = hashEmbedding('hello', 64)
    expect(a).not.toEqual(b)
  })

  it('should apply slot boost for word hash', () => {
    const embedding = hashEmbedding('hello world', 384)
    // At least one value should be boosted above initial range
    const hasBoosted = embedding.some(v => v > 0.3)
    // This is a probabilistic test, but very likely to pass
    expect(hasBoosted).toBe(true)
  })
})

// ============================================================================
// BATCH COSINE SIMILARITY
// ============================================================================

describe('batchCosineSimilarity', () => {
  it('should calculate similarities for multiple vectors', () => {
    const query = [1, 0]
    const vectors = [1, 0, 0, 1, 1, 1] // 3 vectors of dim 2
    const scores = batchCosineSimilarity(query, vectors, 2)
    expect(scores).toHaveLength(3)
    expect(scores[0]).toBeCloseTo(1.0, 10) // identical
    expect(scores[1]).toBeCloseTo(0.0, 10) // orthogonal
  })

  it('should handle single vector in batch', () => {
    const query = [1, 2, 3]
    const vectors = [1, 2, 3]
    const scores = batchCosineSimilarity(query, vectors, 3)
    expect(scores).toHaveLength(1)
    expect(scores[0]).toBeCloseTo(1.0, 10)
  })

  it('should handle empty vectors array', () => {
    const query = [1, 2, 3]
    const vectors: number[] = []
    const scores = batchCosineSimilarity(query, vectors, 3)
    expect(scores).toHaveLength(0)
  })

  it('should return correct similarity ordering', () => {
    const query = [1, 0, 0]
    const vectors = [
      0, 1, 0,     // orthogonal
      1, 0, 0,     // identical
      0.5, 0.5, 0, // 45 degrees
    ]
    const scores = batchCosineSimilarity(query, vectors, 3)
    expect(scores[1]).toBeGreaterThan(scores[2])
    expect(scores[2]).toBeGreaterThan(scores[0])
  })

  it('should handle large batches', () => {
    const query = [1, 2, 3, 4]
    const vectors = new Array(100 * 4).fill(0).map((_, i) => (i % 4) + 1)
    const scores = batchCosineSimilarity(query, vectors, 4)
    expect(scores).toHaveLength(100)
    for (const score of scores) {
      expect(score).toBeGreaterThanOrEqual(-1)
      expect(score).toBeLessThanOrEqual(1)
    }
  })
})

// ============================================================================
// TOP K SIMILAR
// ============================================================================

describe('topKSimilar', () => {
  it('should return top-k results sorted by similarity', () => {
    const query = [1, 0, 0]
    const vectors = [
      1, 0, 0,     // idx 0, sim=1.0
      0, 1, 0,     // idx 1, sim=0.0
      0.5, 0.5, 0, // idx 2, sim~0.707
      0, 0, 1,     // idx 3, sim=0.0
    ]
    const result = topKSimilar(query, vectors, 3, 3)

    expect(result).toHaveLength(6) // 3 pairs of [index, score]
    expect(result[0]).toBe(0)   // index of most similar
    expect(result[1]).toBeCloseTo(1.0, 5)
    expect(result[2]).toBe(2)   // index of second most similar
  })

  it('should return fewer results when k > number of vectors', () => {
    const query = [1, 0]
    const vectors = [1, 0, 0, 1]
    const result = topKSimilar(query, vectors, 2, 10)
    expect(result).toHaveLength(4) // 2 pairs
  })

  it('should handle k=1', () => {
    const query = [1, 0]
    const vectors = [0, 1, 1, 0]
    const result = topKSimilar(query, vectors, 2, 1)
    expect(result).toHaveLength(2) // 1 pair
    expect(result[0]).toBe(1) // index 1 is identical
  })

  it('should return alternating index and score', () => {
    const query = [1, 0]
    const vectors = [1, 0, 0, 1, 1, 0]
    const result = topKSimilar(query, vectors, 2, 3)

    // result should be [idx0, score0, idx1, score1, idx2, score2]
    expect(result.length % 2).toBe(0)
    for (let i = 0; i < result.length; i += 2) {
      expect(Number.isInteger(result[i])).toBe(true) // index
      expect(typeof result[i + 1]).toBe('number')     // score
    }
  })

  it('should return results in descending similarity order', () => {
    const query = [1, 2, 3]
    const vectors = [
      3, 2, 1,
      1, 2, 3,
      2, 4, 6,
      0, 0, 0,
    ]
    const result = topKSimilar(query, vectors, 3, 4)

    const scores: number[] = []
    for (let i = 1; i < result.length; i += 2) {
      scores.push(result[i])
    }
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i - 1]).toBeGreaterThanOrEqual(scores[i])
    }
  })

  it('should handle empty vectors', () => {
    const query = [1, 2]
    const vectors: number[] = []
    const result = topKSimilar(query, vectors, 2, 5)
    expect(result).toHaveLength(0)
  })
})

// ============================================================================
// VECTOR MEAN
// ============================================================================

describe('vectorMean', () => {
  it('should calculate mean of two vectors', () => {
    const vectors = [1, 2, 3, 4] // 2 vectors of dim 2
    const mean = vectorMean(vectors, 2)
    expect(mean).toEqual([2, 3])
  })

  it('should calculate mean of multiple vectors', () => {
    const vectors = [1, 2, 3, 4, 5, 6] // 3 vectors of dim 2
    const mean = vectorMean(vectors, 2)
    expect(mean[0]).toBeCloseTo(3, 10) // (1+3+5)/3
    expect(mean[1]).toBeCloseTo(4, 10) // (2+4+6)/3
  })

  it('should return same vector for single vector', () => {
    const vectors = [1, 2, 3, 4]
    const mean = vectorMean(vectors, 4)
    expect(mean).toEqual([1, 2, 3, 4])
  })

  it('should handle empty vectors array', () => {
    const vectors: number[] = []
    const mean = vectorMean(vectors, 3)
    // 0/0 = NaN in JS
    expect(mean).toHaveLength(3)
    expect(mean[0]).toBeNaN()
  })

  it('should handle high dimensional vectors', () => {
    const n = 10
    const dim = 384
    const vectors = new Array(n * dim).fill(0).map((_, i) => (i % dim) + 1)
    const mean = vectorMean(vectors, dim)
    expect(mean).toHaveLength(dim)
    for (let j = 0; j < dim; j++) {
      expect(mean[j]).toBeCloseTo((j + 1), 10) // each value appears exactly once
    }
  })

  it('should handle vectors with negative values', () => {
    const vectors = [1, -1, -1, 1] // 2 vectors of dim 2
    const mean = vectorMean(vectors, 2)
    expect(mean).toEqual([0, 0])
  })
})

// ============================================================================
// WEIGHTED SUM
// ============================================================================

describe('weightedSum', () => {
  it('should calculate weighted sum correctly', () => {
    const vectors = [1, 2, 3, 4] // 2 vectors of dim 2
    const weights = [0.3, 0.7]
    const result = weightedSum(vectors, weights, 2)
    expect(result[0]).toBeCloseTo(1 * 0.3 + 3 * 0.7, 10) // 2.4
    expect(result[1]).toBeCloseTo(2 * 0.3 + 4 * 0.7, 10) // 3.4
  })

  it('should return zero vector when all weights are zero', () => {
    const vectors = [1, 2, 3, 4]
    const weights = [0, 0]
    const result = weightedSum(vectors, weights, 2)
    expect(result).toEqual([0, 0])
  })

  it('should return first vector when its weight is 1', () => {
    const vectors = [1, 2, 3, 4]
    const weights = [1, 0]
    const result = weightedSum(vectors, weights, 2)
    expect(result).toEqual([1, 2])
  })

  it('should return second vector when its weight is 1', () => {
    const vectors = [1, 2, 3, 4]
    const weights = [0, 1]
    const result = weightedSum(vectors, weights, 2)
    expect(result).toEqual([3, 4])
  })

  it('should handle multiple weights', () => {
    const vectors = [1, 0, 0, 1, 0, 1] // 3 vectors of dim 2
    const weights = [0.5, 0.3, 0.2]
    const result = weightedSum(vectors, weights, 2)
    expect(result[0]).toBeCloseTo(1 * 0.5 + 0 * 0.3 + 0 * 0.2, 10)
    expect(result[1]).toBeCloseTo(0 * 0.5 + 1 * 0.3 + 1 * 0.2, 10)
  })

  it('should handle negative weights', () => {
    const vectors = [1, 2, 3, 4]
    const weights = [2, -1]
    const result = weightedSum(vectors, weights, 2)
    expect(result[0]).toBeCloseTo(1 * 2 + 3 * (-1), 10) // -1
    expect(result[1]).toBeCloseTo(2 * 2 + 4 * (-1), 10) // 0
  })
})

// ============================================================================
// MEMORY ESTIMATION
// ============================================================================

describe('estimateMemorySize', () => {
  it('should estimate correctly for 1000 vectors of 384 dims', () => {
    expect(estimateMemorySize(1000, 384)).toBe(1000 * 384 * 4)
  })

  it('should return 0 for 0 vectors', () => {
    expect(estimateMemorySize(0, 384)).toBe(0)
  })

  it('should return 0 for 0 dimensions', () => {
    expect(estimateMemorySize(1000, 0)).toBe(0)
  })

  it('should scale linearly with vector count', () => {
    const base = estimateMemorySize(100, 384)
    expect(estimateMemorySize(200, 384)).toBe(base * 2)
    expect(estimateMemorySize(500, 384)).toBe(base * 5)
  })

  it('should scale linearly with dimension', () => {
    const base = estimateMemorySize(100, 128)
    expect(estimateMemorySize(100, 256)).toBe(base * 2)
    expect(estimateMemorySize(100, 384)).toBe(base * 3)
  })
})

// ============================================================================
// RECOMMENDED BATCH SIZE
// ============================================================================

describe('recommendedBatchSize', () => {
  it('should return 256 for dimensions <= 128', () => {
    expect(recommendedBatchSize(64)).toBe(256)
    expect(recommendedBatchSize(128)).toBe(256)
  })

  it('should return 128 for dimensions <= 384', () => {
    expect(recommendedBatchSize(200)).toBe(128)
    expect(recommendedBatchSize(384)).toBe(128)
  })

  it('should return 64 for dimensions <= 768', () => {
    expect(recommendedBatchSize(500)).toBe(64)
    expect(recommendedBatchSize(768)).toBe(64)
  })

  it('should return 32 for dimensions > 768', () => {
    expect(recommendedBatchSize(1024)).toBe(32)
    expect(recommendedBatchSize(1536)).toBe(32)
  })
})

// ============================================================================
// TOKEN ESTIMATION
// ============================================================================

describe('estimateTokens', () => {
  it('should estimate tokens for short text', () => {
    expect(estimateTokens('hello')).toBe(2) // ceil(5/4) = 2
  })

  it('should estimate tokens for longer text', () => {
    expect(estimateTokens('hello world')).toBe(3) // ceil(11/4)
  })

  it('should return 0 for empty string', () => {
    expect(estimateTokens('')).toBe(0)
  })

  it('should handle single character', () => {
    expect(estimateTokens('a')).toBe(1) // ceil(1/4)
  })

  it('should scale roughly linearly', () => {
    const short = estimateTokens('hello world')
    const long = estimateTokens('hello world hello world hello world')
    expect(long).toBeGreaterThan(short)
  })

  it('should handle text with spaces', () => {
    const text = 'a b c d' // 7 chars
    expect(estimateTokens(text)).toBe(Math.ceil(7 / 4)) // 2
  })

  it('should handle exactly divisible text', () => {
    const text = 'abcd' // exactly 4 chars
    expect(estimateTokens(text)).toBe(1) // ceil(4/4)
  })
})
