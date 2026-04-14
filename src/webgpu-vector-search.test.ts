/**
 * Tests for WebGPU Vector Search
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import {
  WebGPUVectorSearch,
  WebGPUUnsupportedError,
  WebGPUInitializationError,
} from './webgpu-vector-search'
import { VectorSearchError } from './errors'
import { cosineSimilarity } from './vector-utils'

describe('WebGPUVectorSearch', () => {
  let search: WebGPUVectorSearch
  const dimension = 4

  beforeEach(() => {
    search = new WebGPUVectorSearch(dimension)
  })

  afterEach(() => {
    search.destroy()
  })

  describe('Initialization', () => {
    it('should create instance with default options', () => {
      expect(search).toBeInstanceOf(WebGPUVectorSearch)
    })

    it('should create instance with custom options', () => {
      const customSearch = new WebGPUVectorSearch(384, {
        useGPU: true,
        batchSize: 64,
        enableTiming: true,
      })
      expect(customSearch).toBeInstanceOf(WebGPUVectorSearch)
      customSearch.destroy()
    })

    it('should detect browser support', () => {
      const isSupported = WebGPUVectorSearch.isBrowserSupported()
      expect(typeof isSupported).toBe('boolean')
    })

    it('should check GPU support before initialization', () => {
      expect(search.isGPUSupported()).toBe(false)
    })

    it('should check GPU support after initialization attempt', async () => {
      try { await search.initializeGPU() } catch { /* expected */ }
      expect(typeof search.isGPUSupported()).toBe('boolean')
    })

    it('should create instance with GPU disabled', () => {
      const cpuSearch = new WebGPUVectorSearch(dimension, { useGPU: false })
      expect(cpuSearch.isGPUSupported()).toBe(false)
      cpuSearch.destroy()
    })
  })

  describe('CPU Fallback', () => {
    it('should perform CPU search correctly', async () => {
      const query = [1, 2, 3, 4]
      const vectors = [1, 2, 3, 4, 2, 4, 6, 8, 1, 0, 0, 0]
      const k = 2
      const results = await search.search(query, vectors, k)
      expect(results).toHaveLength(2)
      expect(results[0].similarity).toBeGreaterThanOrEqual(results[1].similarity)
    })

    it('should handle empty vectors array', async () => {
      const query = [1, 2, 3, 4]
      const results = await search.search(query, [], 2)
      expect(results).toHaveLength(0)
    })

    it('should handle k larger than vector count', async () => {
      const query = [1, 2, 3, 4]
      const vectors = [1, 2, 3, 4, 2, 4, 6, 8]
      const results = await search.search(query, vectors, 10)
      expect(results.length).toBeLessThanOrEqual(2)
    })

    it('should validate query dimension', async () => {
      const query = [1, 2, 3]
      const vectors = [1, 2, 3, 4]
      await expect(search.search(query, vectors, 1)).rejects.toThrow()
    })

    it('should validate vectors array length', async () => {
      const query = [1, 2, 3, 4]
      const vectors = [1, 2, 3]
      await expect(search.search(query, vectors, 1)).rejects.toThrow()
    })

    it('should compute cosine similarity correctly', async () => {
      const s3 = new WebGPUVectorSearch(3)
      const query = [1, 0, 0]
      const vectors = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const results = await s3.search(query, vectors, 3)
      expect(results[0].similarity).toBeCloseTo(1.0)
      expect(results[1].similarity).toBeCloseTo(0.0)
      expect(results[2].similarity).toBeCloseTo(0.0)
      s3.destroy()
    })
  })

  describe('Batch Search', () => {
    it('should perform batch search with CPU fallback', async () => {
      const queries = [[1, 2, 3, 4], [2, 4, 6, 8]]
      const vectors = [1, 2, 3, 4, 2, 4, 6, 8, 1, 0, 0, 0]
      const batchResult = await search.batchSearch(queries, vectors, 2)
      expect(batchResult.results).toHaveLength(2)
      expect(batchResult.results[0]).toHaveLength(2)
      expect(batchResult.results[1]).toHaveLength(2)
      expect(batchResult.usedGPU).toBe(false)
    })

    it('should handle empty queries array', async () => {
      const batchResult = await search.batchSearch([], [1, 2, 3, 4], 2)
      expect(batchResult.results).toHaveLength(0)
    })

    it('should handle single query in batch', async () => {
      const batchResult = await search.batchSearch([[1, 2, 3, 4]], [1, 2, 3, 4, 2, 4, 6, 8], 1)
      expect(batchResult.results).toHaveLength(1)
      expect(batchResult.results[0]).toHaveLength(1)
    })
  })

  describe('Performance Metrics', () => {
    it('should track metrics when timing enabled', async () => {
      const ts = new WebGPUVectorSearch(dimension, { enableTiming: true, useGPU: false })
      await ts.search([1, 2, 3, 4], [1, 2, 3, 4, 2, 4, 6, 8], 1)
      const metrics = ts.getMetrics()
      expect(metrics).toHaveLength(1)
      expect(metrics[0].usedGPU).toBe(false)
      expect(metrics[0].cpuTime).toBeGreaterThan(0)
      ts.destroy()
    })

    it('should not track metrics when timing disabled', async () => {
      const us = new WebGPUVectorSearch(dimension, { enableTiming: false, useGPU: false })
      await us.search([1, 2, 3, 4], [1, 2, 3, 4], 1)
      expect(us.getMetrics()).toHaveLength(0)
      us.destroy()
    })

    it('should calculate average speedup', () => {
      expect(search.getAverageSpeedup()).toBe(1)
    })

    it('should clear metrics', async () => {
      await search.search([1, 2, 3, 4], [1, 2, 3, 4], 1)
      expect(search.getMetrics().length).toBeGreaterThan(0)
      search.clearMetrics()
      expect(search.getMetrics()).toHaveLength(0)
    })

    it('should generate performance summary', async () => {
      await search.search([1, 2, 3, 4], [1, 2, 3, 4], 1)
      const summary = search.getPerformanceSummary()
      expect(summary).toContain('Performance Summary')
      expect(summary).toContain('CPU Searches')
    })
  })

  describe('Edge Cases', () => {
    it('should handle very small vectors', async () => {
      const ss = new WebGPUVectorSearch(2)
      const results = await ss.search([1, 0], [1, 0, 0, 1], 2)
      expect(results).toHaveLength(2)
      ss.destroy()
    })

    it('should handle large vector dimension', async () => {
      const dim = 1536
      const ls = new WebGPUVectorSearch(dim)
      const query = Array.from({ length: dim }, (_, i) => i % 10)
      const results = await ls.search(query, [...query, ...query.map(v => v * 2)], 1)
      expect(results).toHaveLength(1)
      expect(results[0].similarity).toBeCloseTo(1.0)
      ls.destroy()
    })

    it('should handle zero magnitude vectors', async () => {
      const results = await search.search([0, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], 2)
      const zeroResult = results.find(r => r.index === 1)
      expect(zeroResult?.similarity).toBe(0)
    })

    it('should handle very large k value', async () => {
      const results = await search.search([1, 2, 3, 4], [1, 2, 3, 4, 2, 4, 6, 8, 1, 0, 0, 0], 1000)
      expect(results.length).toBe(3)
    })

    it('should handle k=0', async () => {
      const results = await search.search([1, 2, 3, 4], [1, 2, 3, 4, 2, 4, 6, 8], 0)
      expect(results).toHaveLength(0)
    })

    it('should handle single vector', async () => {
      const results = await search.search([1, 2, 3, 4], [1, 2, 3, 4], 1)
      expect(results).toHaveLength(1)
      expect(results[0].similarity).toBeCloseTo(1.0)
    })
  })

  describe('Cleanup', () => {
    it('should destroy resources', () => {
      expect(() => new WebGPUVectorSearch(dimension).destroy()).not.toThrow()
    })

    it('should handle multiple destroy calls', () => {
      const s = new WebGPUVectorSearch(dimension)
      s.destroy()
      expect(() => s.destroy()).not.toThrow()
    })
  })

  describe('Similarity Accuracy', () => {
    it('should produce same results as CPU cosine similarity', async () => {
      const query = [1, 2, 3, 4]
      const vectors = [1, 2, 3, 4, 2, 4, 6, 8]
      const results = await search.search(query, vectors, 2)
      expect(results[0].similarity).toBeCloseTo(cosineSimilarity(query, vectors.slice(0, 4)), 5)
      expect(results[1].similarity).toBeCloseTo(cosineSimilarity(query, vectors.slice(4, 8)), 5)
    })

    it('should maintain ordering by similarity', async () => {
      const s3 = new WebGPUVectorSearch(3)
      const query = [1, 0, 0]
      const vectors = [1, 0, 0, 0.9, 0.1, 0, 0.5, 0.5, 0, 0, 1, 0]
      const results = await s3.search(query, vectors, 4)
      expect(results[0].similarity).toBeGreaterThan(results[1].similarity)
      expect(results[1].similarity).toBeGreaterThan(results[2].similarity)
      expect(results[2].similarity).toBeGreaterThan(results[3].similarity)
      s3.destroy()
    })

    it('should return all identical vectors for parallel input', async () => {
      const query = [1, 2, 3, 4]
      const vectors = [2, 4, 6, 8, 10, 20, 30, 40, 1, 2, 3, 4]
      const results = await search.search(query, vectors, 3)
      for (const r of results) expect(r.similarity).toBeCloseTo(1.0, 10)
    })
  })

  describe('Default dimension (384)', () => {
    it('should work with default 384 dimensions', async () => {
      const s384 = new WebGPUVectorSearch(384)
      const query = Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.1))
      const doc = Array.from({ length: 384 }, (_, i) => Math.cos(i * 0.1))
      const results = await s384.search(query, [...query, ...doc], 2)
      expect(results).toHaveLength(2)
      expect(results[0].similarity).toBeCloseTo(1.0, 5)
      s384.destroy()
    })
  })
})

describe('WebGPU Errors', () => {
  it('should throw WebGPUUnsupportedError when GPU not available', async () => {
    const origGPU = (global as any).navigator?.gpu
    Object.defineProperty(navigator, 'gpu', { get: () => undefined, configurable: true })
    const s = new WebGPUVectorSearch(384)
    await expect(s.initializeGPU()).rejects.toThrow(WebGPUUnsupportedError)
    Object.defineProperty(navigator, 'gpu', { get: () => origGPU, configurable: true })
    s.destroy()
  })

  it('WebGPUUnsupportedError should have correct properties', () => {
    const err = new WebGPUUnsupportedError()
    expect(err).toBeInstanceOf(VectorSearchError)
    expect(err.category).toBe('system')
    expect(err.recovery).toBe('recoverable')
  })

  it('WebGPUInitializationError should have correct properties', () => {
    const err = new WebGPUInitializationError('device failed')
    expect(err).toBeInstanceOf(VectorSearchError)
    expect(err.category).toBe('system')
    expect(err.severity).toBe('high')
    expect(err.message).toContain('device failed')
  })

  it('WebGPUInitializationError should support cause', () => {
    const cause = new Error('adapter error')
    const err = new WebGPUInitializationError('failed', cause)
    expect(err.cause).toBe(cause)
  })
})

describe('Integration with Vector Utils', () => {
  it('should work with vector utilities', async () => {
    const { normalizeVector } = await import('./vector-utils')
    const query = normalizeVector([3, 4])
    const s = new WebGPUVectorSearch(2)
    const results = await s.search(query, [3, 4, 6, 8], 2)
    expect(results).toHaveLength(2)
    expect(results[0].similarity).toBeCloseTo(1.0)
    s.destroy()
  })

  it('should handle hash-based embeddings', async () => {
    const { hashEmbedding } = await import('./vector-utils')
    const q = hashEmbedding('test query', 384)
    const d1 = hashEmbedding('similar query', 384)
    const d2 = hashEmbedding('different content', 384)
    const s = new WebGPUVectorSearch(384)
    const results = await s.search(q, [...d1, ...d2], 2)
    expect(results).toHaveLength(2)
    expect(results[0].similarity).toBeGreaterThanOrEqual(0)
    expect(results[0].similarity).toBeLessThanOrEqual(1)
    s.destroy()
  })
})
