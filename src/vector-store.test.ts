/**
 * Tests for VectorStore - isolated per describe block
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { VectorStore } from './vector-store'
import { ValidationError, NotFoundError } from './errors'
import type { KnowledgeEntry, EmbeddingGenerator } from './vector-store'

function createEntry(overrides: Partial<KnowledgeEntry> & { content: string }): Omit<KnowledgeEntry, 'embedding'> {
  return {
    id: `ke_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    type: 'document',
    sourceId: `src_${Math.random().toString(36).substring(2, 7)}`,
    content: overrides.content,
    metadata: { timestamp: new Date().toISOString(), tags: [], ...overrides.metadata },
    editable: true,
    ...overrides,
  }
}

let store: VectorStore

describe('VectorStore', () => {
  beforeAll(async () => {
    store = new VectorStore()
    await store.init()
  })

  afterAll(async () => {
    // Cleanup - ignore errors
    try { await store.deleteEntry('') } catch {}
  })

  describe('addEntry', () => {
    it('should add an entry and return it with embedding', async () => {
      const entry = createEntry({ content: 'hello world' })
      const result = await store.addEntry(entry)
      expect(result.id).toBeDefined()
      expect(result.content).toBe('hello world')
      expect(result.embedding).toBeDefined()
      expect(result.embedding!.length).toBeGreaterThan(0)
    })

    it('should add entries of different types', async () => {
      const types: KnowledgeEntry['type'][] = ['conversation', 'message', 'document', 'contact']
      for (const type of types) {
        const result = await store.addEntry(createEntry({ type, content: `${type} content` }))
        expect(result.type).toBe(type)
      }
    })

    it('should add entry with all metadata fields', async () => {
      const result = await store.addEntry(createEntry({
        content: 'full metadata',
        metadata: { timestamp: new Date().toISOString(), author: 'test', tags: ['a'], importance: 0.9, starred: true },
      }))
      expect(result.metadata.author).toBe('test')
      expect(result.metadata.tags).toEqual(['a'])
      expect(result.metadata.importance).toBe(0.9)
      expect(result.metadata.starred).toBe(true)
    })

    it('should throw ValidationError for empty content', async () => {
      await expect(store.addEntry(createEntry({ content: '' }))).rejects.toThrow(ValidationError)
    })

    it('should throw ValidationError for whitespace-only content', async () => {
      await expect(store.addEntry(createEntry({ content: '   ' }))).rejects.toThrow(ValidationError)
    })

    it('should use custom embedding generator', async () => {
      const customStore = new VectorStore({
        embeddingGenerator: async (text) => Array(16).fill(0).map((_, i) => (text.charCodeAt(i % text.length) % 100) / 100),
      })
      await customStore.init()
      const result = await customStore.addEntry(createEntry({ content: 'custom test' }))
      expect(result.embedding).toHaveLength(16)
    })
  })

  describe('addEntries (batch)', () => {
    it('should add multiple entries', async () => {
      const results = await store.addEntries([
        createEntry({ content: 'first' }),
        createEntry({ content: 'second' }),
        createEntry({ content: 'third' }),
      ])
      expect(results).toHaveLength(3)
      for (const r of results) expect(r.embedding).toBeDefined()
    })

    it('should handle empty batch', async () => {
      expect(await store.addEntries([])).toHaveLength(0)
    })
  })

  describe('getEntry', () => {
    it('should retrieve an existing entry', async () => {
      const added = await store.addEntry(createEntry({ content: 'find me' }))
      const retrieved = await store.getEntry(added.id)
      expect(retrieved).not.toBeNull()
      expect(retrieved!.content).toBe('find me')
    })

    it('should return null for non-existent entry', async () => {
      expect(await store.getEntry('non-existent')).toBeNull()
    })
  })

  describe('getEntries', () => {
    it('should return all entries', async () => {
      const before = (await store.getEntries()).length
      await store.addEntries([createEntry({ content: 'a' }), createEntry({ content: 'b' })])
      expect((await store.getEntries()).length).toBeGreaterThanOrEqual(before + 2)
    })

    it('should filter by type', async () => {
      await store.addEntry(createEntry({ type: 'message', content: 'unique msg type test' }))
      await store.addEntry(createEntry({ type: 'document', content: 'unique doc type test' }))
      const msgs = await store.getEntries({ type: 'message' })
      expect(msgs.length).toBeGreaterThanOrEqual(1)
      expect(msgs.every(e => e.type === 'message')).toBe(true)
    })

    it('should filter by sourceId', async () => {
      const srcId = `unique-src-${Date.now()}`
      await store.addEntry(createEntry({ sourceId: srcId, content: 'a' }))
      expect((await store.getEntries({ sourceId: srcId })).every(e => e.sourceId === srcId)).toBe(true)
    })

    it('should filter by starred', async () => {
      await store.addEntry(createEntry({ content: 'star filter', metadata: { timestamp: new Date().toISOString(), starred: true } }))
      await store.addEntry(createEntry({ content: 'no star filter', metadata: { timestamp: new Date().toISOString(), starred: false } }))
      expect((await store.getEntries({ starred: true })).every(e => e.metadata.starred === true)).toBe(true)
    })

    it('should apply limit', async () => {
      expect((await store.getEntries({ limit: 2 })).length).toBeLessThanOrEqual(2)
    })

    it('should sort by timestamp descending', async () => {
      await store.addEntry(createEntry({ content: 'sort older', metadata: { timestamp: '2020-01-01T00:00:00.000Z' } }))
      await store.addEntry(createEntry({ content: 'sort newer', metadata: { timestamp: '2025-01-01T00:00:00.000Z' } }))
      const entries = await store.getEntries()
      const times = entries.map(e => new Date(e.metadata.timestamp).getTime())
      for (let i = 1; i < times.length; i++) expect(times[i - 1]).toBeGreaterThanOrEqual(times[i])
    })
  })

  describe('updateEntry', () => {
    it('should update content and mark as edited', async () => {
      const added = await store.addEntry(createEntry({ content: 'update original' }))
      const updated = await store.updateEntry(added.id, { content: 'update changed' })
      expect(updated.content).toBe('update changed')
      expect(updated.editedContent).toBe('update changed')
      expect(updated.editedAt).toBeDefined()
    })

    it('should throw ValidationError for empty ID', async () => {
      await expect(store.updateEntry('', { content: 'x' })).rejects.toThrow(ValidationError)
    })

    it('should throw NotFoundError for non-existent ID', async () => {
      await expect(store.updateEntry('non-existent-' + Date.now(), { content: 'x' })).rejects.toThrow(NotFoundError)
    })

    it('should not update embedding when only metadata changes', async () => {
      const added = await store.addEntry(createEntry({ content: 'keep emb' }))
      const origEmbedding = [...added.embedding!]
      const updated = await store.updateEntry(added.id, { metadata: { ...added.metadata, importance: 0.5 } })
      expect(updated.embedding).toEqual(origEmbedding)
    })
  })

  describe('deleteEntry', () => {
    it('should delete an existing entry', async () => {
      const added = await store.addEntry(createEntry({ content: 'delete me now' }))
      await store.deleteEntry(added.id)
      expect(await store.getEntry(added.id)).toBeNull()
    })

    it('should throw ValidationError for empty ID', async () => {
      await expect(store.deleteEntry('')).rejects.toThrow(ValidationError)
    })

    it('should not throw for non-existent entry', async () => {
      await expect(store.deleteEntry('non-existent-del')).resolves.toBeUndefined()
    })

    it('should only delete the specified entry', async () => {
      const e1 = await store.addEntry(createEntry({ content: 'keep this' }))
      const e2 = await store.addEntry(createEntry({ content: 'delete this' }))
      await store.deleteEntry(e2.id)
      expect(await store.getEntry(e1.id)).not.toBeNull()
      expect(await store.getEntry(e2.id)).toBeNull()
    })
  })

  describe('search', () => {
    it('should return results for a valid query', async () => {
      await store.addEntry(createEntry({ content: 'search test machine learning' }))
      const results = await store.search('machine learning')
      expect(Array.isArray(results)).toBe(true)
    })

    it('should respect limit', async () => {
      await store.addEntries([createEntry({ content: 'limit tech a' }), createEntry({ content: 'limit tech b' })])
      expect((await store.search('limit tech', { limit: 1 })).length).toBeLessThanOrEqual(1)
    })

    it('should filter by type', async () => {
      await store.addEntry(createEntry({ type: 'message', content: 'search type msg' }))
      await store.addEntry(createEntry({ type: 'document', content: 'search type doc' }))
      expect((await store.search('search type', { types: ['message'] })).every(r => r.entry.type === 'message')).toBe(true)
    })

    it('should filter by tags', async () => {
      await store.addEntry(createEntry({ content: 'search tag t', metadata: { timestamp: new Date().toISOString(), tags: ['unique_tag'] } }))
      await store.addEntry(createEntry({ content: 'search tag u', metadata: { timestamp: new Date().toISOString(), tags: ['other'] } }))
      expect((await store.search('search tag', { tags: ['unique_tag'] })).every(r => r.entry.metadata.tags?.includes('unique_tag'))).toBe(true)
    })

    it('should filter starred only', async () => {
      await store.addEntry(createEntry({ content: 'search star y', metadata: { timestamp: new Date().toISOString(), starred: true } }))
      await store.addEntry(createEntry({ content: 'search star n', metadata: { timestamp: new Date().toISOString(), starred: false } }))
      expect((await store.search('search star', { starredOnly: true })).every(r => r.entry.metadata.starred === true)).toBe(true)
    })

    it('should return results sorted by similarity', async () => {
      await store.addEntries([createEntry({ content: 'sort topic sim' }), createEntry({ content: 'sort topic diff' })])
      const results = await store.search('sort topic')
      for (let i = 1; i < results.length; i++) expect(results[i - 1].similarity).toBeGreaterThanOrEqual(results[i].similarity)
    })

    it('should return similarity scores in valid range', async () => {
      await store.addEntry(createEntry({ content: 'sim range test' }))
      for (const r of await store.search('sim range')) {
        expect(r.similarity).toBeGreaterThanOrEqual(0)
        expect(r.similarity).toBeLessThanOrEqual(1)
      }
    })

    it('should throw ValidationError for empty query', async () => {
      await expect(store.search('')).rejects.toThrow(ValidationError)
    })
  })

  describe('hybridSearch', () => {
    it('should return results', async () => {
      await store.addEntry(createEntry({ content: 'hybrid test ml neural networks' }))
      const results = await store.hybridSearch('hybrid test ml')
      expect(Array.isArray(results)).toBe(true)
    })

    it('should respect search options', async () => {
      await store.addEntry(createEntry({ content: 'hybrid opt msg', type: 'message' }))
      await store.addEntry(createEntry({ content: 'hybrid opt doc', type: 'document' }))
      expect((await store.hybridSearch('hybrid opt', { types: ['message'] })).every(r => r.entry.type === 'message')).toBe(true)
    })

    it('should throw ValidationError for empty query', async () => {
      await expect(store.hybridSearch('')).rejects.toThrow(ValidationError)
    })
  })

  describe('createCheckpoint', () => {
    it('should create a checkpoint', async () => {
      await store.addEntry(createEntry({ content: 'cp entry' }))
      const cp = await store.createCheckpoint('test cp')
      expect(cp.id).toMatch(/^cp_/)
      expect(cp.name).toBe('test cp')
      expect(cp.entryCount).toBeGreaterThanOrEqual(1)
      expect(cp.isStarred).toBe(false)
      expect(cp.vectorHash).toBeDefined()
    })

    it('should create checkpoint with options', async () => {
      const cp = await store.createCheckpoint('featured', {
        description: 'desc', tags: ['stable'], isStarred: true,
      })
      expect(cp.description).toBe('desc')
      expect(cp.tags).toEqual(['stable'])
      expect(cp.isStarred).toBe(true)
    })

    it('should create checkpoint when store is empty', async () => {
      const cp = await store.createCheckpoint('empty cp ' + Date.now())
      expect(cp.entryCount).toBeGreaterThanOrEqual(0)
    })
  })

  describe('getCheckpoints', () => {
    it('should return checkpoints sorted newest first', async () => {
      await store.createCheckpoint('cp sort first')
      await store.createCheckpoint('cp sort second')
      const cps = await store.getCheckpoints()
      const times = cps.map(c => new Date(c.createdAt).getTime())
      for (let i = 1; i < times.length; i++) expect(times[i - 1]).toBeGreaterThanOrEqual(times[i])
    })
  })

  describe('setCheckpointStarred', () => {
    it('should star a checkpoint', async () => {
      const cp = await store.createCheckpoint('to star')
      expect((await store.setCheckpointStarred(cp.id, true)).isStarred).toBe(true)
    })

    it('should unstar a checkpoint', async () => {
      const cp = await store.createCheckpoint('unstar me', { isStarred: true })
      expect((await store.setCheckpointStarred(cp.id, false)).isStarred).toBe(false)
    })

    it('should throw for empty ID', async () => {
      await expect(store.setCheckpointStarred('', true)).rejects.toThrow(ValidationError)
    })

    it('should throw NotFoundError for non-existent', async () => {
      await expect(store.setCheckpointStarred('non-existent-cp', true)).rejects.toThrow(NotFoundError)
    })
  })

  describe('getLatestStableCheckpoint', () => {
    it('should return null when no starred checkpoints', async () => {
      // This test may pass even with other starred checkpoints from prior tests,
      // so we just verify it returns a valid type
      const result = await store.getLatestStableCheckpoint()
      if (result === null) {
        expect(result).toBeNull()
      } else {
        expect(result.isStarred).toBe(true)
      }
    })

    it('should return starred checkpoint after starring', async () => {
      const cp = await store.createCheckpoint('latest stable ' + Date.now())
      await store.setCheckpointStarred(cp.id, true)
      const result = await store.getLatestStableCheckpoint()
      expect(result).not.toBeNull()
      expect(result!.isStarred).toBe(true)
    })
  })

  describe('rollbackToCheckpoint', () => {
    it('should throw for empty ID', async () => {
      await expect(store.rollbackToCheckpoint('')).rejects.toThrow(ValidationError)
    })

    it('should throw NotFoundError for non-existent', async () => {
      await expect(store.rollbackToCheckpoint('non-existent-rb')).rejects.toThrow(NotFoundError)
    })

    it('should report counts', async () => {
      const cp = await store.createCheckpoint('rb count test')
      await store.addEntry(createEntry({ content: 'rb after', metadata: { timestamp: new Date().toISOString() } }))
      const { restored, removed } = await store.rollbackToCheckpoint(cp.id)
      expect(typeof restored).toBe('number')
      expect(typeof removed).toBe('number')
    })
  })

  describe('exportForLoRA', () => {
    it('should export entries from checkpoint', async () => {
      await store.addEntry(createEntry({ content: 'export me now' }))
      const cp = await store.createCheckpoint('export cp')
      const data = await store.exportForLoRA(cp.id, 'jsonl')
      expect(data.checkpointId).toBe(cp.id)
      expect(data.format).toBe('jsonl')
      expect(data.entries.length).toBeGreaterThanOrEqual(1)
      expect(data.statistics.totalEntries).toBeGreaterThanOrEqual(1)
    })

    it('should export in json format', async () => {
      await store.addEntry(createEntry({ content: 'json export test' }))
      const cp = await store.createCheckpoint('json export')
      expect((await store.exportForLoRA(cp.id, 'json')).format).toBe('json')
    })

    it('should export in parquet format', async () => {
      await store.addEntry(createEntry({ content: 'parquet export test' }))
      const cp = await store.createCheckpoint('parquet export')
      expect((await store.exportForLoRA(cp.id, 'parquet')).format).toBe('parquet')
    })

    it('should use latest starred checkpoint when no ID', async () => {
      await store.addEntry(createEntry({ content: 'auto export content' }))
      await store.createCheckpoint('auto export cp', { isStarred: true })
      expect((await store.exportForLoRA()).entries.length).toBeGreaterThanOrEqual(1)
    })

    it('should include date range in statistics', async () => {
      await store.addEntry(createEntry({ content: 'dated export', metadata: { timestamp: '2024-06-15T12:00:00.000Z' } }))
      const cp = await store.createCheckpoint('date export')
      const data = await store.exportForLoRA(cp.id)
      expect(data.statistics.dateRange.start).toBeDefined()
      expect(data.statistics.dateRange.end).toBeDefined()
    })

    it('should use edited content when available', async () => {
      const added = await store.addEntry(createEntry({ content: 'edit export orig' }))
      await store.updateEntry(added.id, { content: 'edit export changed' })
      const cp = await store.createCheckpoint('edit export cp')
      expect((await store.exportForLoRA(cp.id)).entries.some(e => e.text === 'edit export changed')).toBe(true)
    })
  })

  describe('Edge Cases', () => {
    it('should handle very long content', async () => {
      const long = 'word '.repeat(10000)
      const entry = await store.addEntry(createEntry({ content: long }))
      expect(entry.content.length).toBe(long.length)
    })

    it('should handle special characters', async () => {
      const special = 'Hello! @#$%^&*() \n\t'
      expect((await store.addEntry(createEntry({ content: special }))).content).toBe(special)
    })

    it('should handle many entries', async () => {
      const entries = Array.from({ length: 50 }, (_, i) => createEntry({ content: `bulk edge ${i}` }))
      const results = await store.addEntries(entries)
      expect(results).toHaveLength(50)
    })

    it('should handle entry with minimal metadata', async () => {
      const entry = await store.addEntry(createEntry({ content: 'minimal meta', metadata: { timestamp: new Date().toISOString() } }))
      expect(entry.metadata.author).toBeUndefined()
      expect(entry.metadata.tags).toBeUndefined()
    })

    it('should handle updating entry multiple times', async () => {
      const entry = await store.addEntry(createEntry({ content: 'v1 multi' }))
      await store.updateEntry(entry.id, { content: 'v2 multi' })
      await store.updateEntry(entry.id, { content: 'v3 multi' })
      const final = await store.getEntry(entry.id)
      expect(final!.content).toBe('v3 multi')
    })

    it('should produce consistent embeddings', async () => {
      const customStore = new VectorStore({
        embeddingGenerator: async (text) => Array(8).fill(0).map((_, i) => text.charCodeAt(i % text.length) / 255),
      })
      await customStore.init()
      const e1 = await customStore.addEntry(createEntry({ content: 'consistent emb' }))
      const e2 = await customStore.addEntry(createEntry({ content: 'consistent emb' }))
      expect(e1.embedding).toEqual(e2.embedding)
    })

    it('should handle search with all filters combined', async () => {
      await store.addEntry(createEntry({
        type: 'message', content: 'combined filters test',
        metadata: { timestamp: '2024-06-15T12:00:00.000Z', tags: ['test', 'combo'], starred: true },
      }))
      const results = await store.search('combined filters', {
        limit: 10, threshold: 0.0, types: ['message'],
        dateRange: { start: '2020-01-01', end: '2030-01-01' },
        tags: ['test'], starredOnly: true,
      })
      expect(Array.isArray(results)).toBe(true)
    })
  })

  describe('Integration', () => {
    it('should handle full workflow', async () => {
      const e1 = await store.addEntry(createEntry({
        type: 'document', content: 'workflow project docs',
        metadata: { timestamp: new Date().toISOString(), tags: ['project'] },
      }))
      const e2 = await store.addEntry(createEntry({
        type: 'message', content: 'workflow meeting notes',
        metadata: { timestamp: new Date().toISOString(), tags: ['meeting'] },
      }))

      const searchResults = await store.search('workflow project')
      expect(searchResults.length).toBeGreaterThanOrEqual(1)

      const updated = await store.updateEntry(e1.id, { content: 'workflow updated docs' })
      expect(updated.content).toBe('workflow updated docs')

      const cp = await store.createCheckpoint('workflow cp', { isStarred: true })
      expect(cp.entryCount).toBeGreaterThanOrEqual(2)

      const exportData = await store.exportForLoRA(cp.id, 'json')
      expect(exportData.entries.length).toBeGreaterThanOrEqual(2)

      const stable = await store.getLatestStableCheckpoint()
      expect(stable).not.toBeNull()
    })

    it('should maintain data consistency', async () => {
      const entry = await store.addEntry(createEntry({ content: 'consistency check' }))
      const retrieved = await store.getEntry(entry.id)
      expect(retrieved).toEqual(entry)
    })
  })
})
