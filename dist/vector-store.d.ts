/**
 * Vector Store - In-Browser Semantic Search
 *
 * Privacy-first vector search system using IndexedDB for persistence.
 * Supports semantic search, checkpointing, and hybrid search.
 *
 * Features:
 * - Local-only storage (no server required)
 * - Semantic similarity search
 * - Hybrid search (semantic + keyword)
 * - Checkpoint system for rollback
 * - LoRA training export
 * - Automatic embedding generation
 */
export interface KnowledgeEntry {
    id: string;
    type: 'conversation' | 'message' | 'document' | 'contact';
    sourceId: string;
    content: string;
    embedding?: number[];
    metadata: {
        timestamp: string;
        author?: string;
        contactId?: string;
        conversationId?: string;
        tags?: string[];
        importance?: number;
        starred?: boolean;
    };
    editable: boolean;
    editedContent?: string;
    editedAt?: string;
}
export interface Checkpoint {
    id: string;
    name: string;
    createdAt: string;
    entryCount: number;
    isStarred: boolean;
    description?: string;
    tags: string[];
    vectorHash: string;
}
export interface LoRAExport {
    checkpointId: string;
    format: 'jsonl' | 'json' | 'parquet';
    entries: Array<{
        text: string;
        metadata: Record<string, unknown>;
    }>;
    statistics: {
        totalEntries: number;
        totalTokens: number;
        avgQuality: number;
        dateRange: {
            start: string;
            end: string;
        };
    };
}
export interface KnowledgeSearchOptions {
    limit?: number;
    threshold?: number;
    types?: KnowledgeEntry['type'][];
    dateRange?: {
        start: string;
        end: string;
    };
    tags?: string[];
    starredOnly?: boolean;
}
export interface KnowledgeSearchResult {
    entry: KnowledgeEntry;
    similarity: number;
    highlights?: string[];
}
export interface EmbeddingGenerator {
    (text: string): Promise<number[]>;
}
export declare class VectorStore {
    private db;
    private embeddingCache;
    private cacheAccessOrder;
    private readonly maxCacheSize;
    private customEmbeddingGenerator?;
    constructor(options?: {
        embeddingGenerator?: EmbeddingGenerator;
    });
    /**
     * Initialize the database
     */
    init(): Promise<void>;
    /**
     * Adds a knowledge entry with automatically generated embedding.
     *
     * @param entry - The entry to add (embedding will be generated)
     * @returns Promise resolving to the complete entry with embedding
     * @throws {StorageError} If database operation fails
     * @throws {ValidationError} If entry content is empty
     *
     * @example
     * ```typescript
     * const entry = await vectorStore.addEntry({
     *   id: 'ke_123',
     *   type: 'message',
     *   sourceId: 'msg_456',
     *   content: 'Important information',
     *   metadata: { timestamp: new Date().toISOString() },
     *   editable: true
     * })
     * ```
     */
    addEntry(entry: Omit<KnowledgeEntry, 'embedding'>): Promise<KnowledgeEntry>;
    /**
     * Adds multiple knowledge entries efficiently.
     *
     * Processes entries sequentially to avoid blocking.
     *
     * @param entries - Array of entries to add
     * @returns Promise resolving to array of complete entries with embeddings
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const results = await vectorStore.addEntries([
     *   { type: 'message', sourceId: 'msg1', content: 'Text 1', ... },
     *   { type: 'message', sourceId: 'msg2', content: 'Text 2', ... }
     * ])
     * ```
     */
    addEntries(entries: Omit<KnowledgeEntry, 'embedding'>[]): Promise<KnowledgeEntry[]>;
    /**
     * Updates an existing knowledge entry.
     *
     * If content changes, generates a new embedding and marks entry as edited.
     *
     * @param id - The entry ID to update
     * @param updates - Partial updates to apply
     * @returns Promise resolving to the updated entry
     * @throws {ValidationError} If ID is empty
     * @throws {NotFoundError} If entry doesn't exist
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const updated = await vectorStore.updateEntry('ke_123', {
     *   content: 'Updated content'
     * })
     * console.log(updated.editedContent) // 'Updated content'
     * ```
     */
    updateEntry(id: string, updates: Partial<Pick<KnowledgeEntry, 'content' | 'editedContent' | 'editedAt' | 'metadata'>>): Promise<KnowledgeEntry>;
    /**
     * Retrieves a knowledge entry by its ID.
     *
     * @param id - The entry ID to retrieve
     * @returns Promise resolving to the entry or null if not found
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const entry = await vectorStore.getEntry('ke_123')
     * if (entry) {
     *   console.log(entry.content)
     * }
     * ```
     */
    getEntry(id: string): Promise<KnowledgeEntry | null>;
    /**
     * Retrieves knowledge entries with optional filtering and pagination.
     *
     * @param filter - Optional filters to apply
     * @returns Promise resolving to array of filtered entries
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * // Get starred message entries
     * const entries = await vectorStore.getEntries({
     *   type: 'message',
     *   starred: true,
     *   limit: 10
     * })
     * ```
     */
    getEntries(filter?: {
        type?: KnowledgeEntry['type'];
        sourceId?: string;
        starred?: boolean;
        limit?: number;
        offset?: number;
    }): Promise<KnowledgeEntry[]>;
    /**
     * Deletes a knowledge entry by its ID.
     *
     * @param id - The entry ID to delete
     * @returns Promise that resolves when deletion is complete
     * @throws {ValidationError} If ID is empty
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * await vectorStore.deleteEntry('ke_123')
     * ```
     */
    deleteEntry(id: string): Promise<void>;
    /**
     * Performs semantic search for similar entries using vector similarity.
     *
     * @param query - The search query text
     * @param options - Search configuration options
     * @returns Promise resolving to array of search results with similarity scores
     * @throws {ValidationError} If query is empty
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const results = await vectorStore.search('project deadline', {
     *   limit: 5,
     *   threshold: 0.8,
     *   types: ['message']
     * })
     * results.forEach(r => {
     *   console.log(`${r.similarity}: ${r.entry.content}`)
     * })
     * ```
     */
    search(query: string, options?: KnowledgeSearchOptions): Promise<KnowledgeSearchResult[]>;
    /**
     * Performs hybrid search combining semantic and keyword matching.
     *
     * Boosts semantic search results with keyword matching for better relevance.
     *
     * @param query - The search query text
     * @param options - Search configuration options (same as search())
     * @returns Promise resolving to array of search results with boosted scores
     *
     * @example
     * ```typescript
     * const results = await vectorStore.hybridSearch('important meeting', {
     *   limit: 10
     * })
     * ```
     */
    hybridSearch(query: string, options?: KnowledgeSearchOptions): Promise<KnowledgeSearchResult[]>;
    /**
     * Creates a checkpoint of the current knowledge state.
     *
     * Checkpoints allow you to save and restore knowledge states.
     *
     * @param name - Descriptive name for the checkpoint
     * @param options - Additional checkpoint options
     * @returns Promise resolving to the created checkpoint
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const checkpoint = await vectorStore.createCheckpoint('Before reindex', {
     *   description: 'State before rebuilding knowledge base',
     *   tags: ['stable', 'pre-migration'],
     *   isStarred: true
     * })
     * ```
     */
    createCheckpoint(name: string, options?: {
        description?: string;
        tags?: string[];
        isStarred?: boolean;
    }): Promise<Checkpoint>;
    /**
     * Retrieves all checkpoints, sorted by creation time (newest first).
     *
     * @returns Promise resolving to array of checkpoints
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const checkpoints = await vectorStore.getCheckpoints()
     * checkpoints.forEach(cp => {
     *   console.log(`${cp.name}: ${cp.entryCount} entries`)
     * })
     * ```
     */
    getCheckpoints(): Promise<Checkpoint[]>;
    /**
     * Stars or unstars a checkpoint.
     *
     * Starred checkpoints are considered stable/reference points.
     *
     * @param checkpointId - The checkpoint ID
     * @param starred - Whether to star the checkpoint
     * @returns Promise resolving to the updated checkpoint
     * @throws {ValidationError} If checkpointId is empty
     * @throws {NotFoundError} If checkpoint doesn't exist
     *
     * @example
     * ```typescript
     * await vectorStore.setCheckpointStarred('cp_123', true)
     * ```
     */
    setCheckpointStarred(checkpointId: string, starred: boolean): Promise<Checkpoint>;
    /**
     * Rolls back the knowledge base to a previous checkpoint.
     *
     * Removes entries created after the checkpoint and restores edited entries
     * to their original state at checkpoint time.
     *
     * @param checkpointId - The checkpoint to roll back to
     * @returns Promise resolving to counts of restored and removed entries
     * @throws {ValidationError} If checkpointId is empty
     * @throws {NotFoundError} If checkpoint doesn't exist
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const { restored, removed } = await vectorStore.rollbackToCheckpoint('cp_123')
     * console.log(`Restored ${restored} entries, removed ${removed}`)
     * ```
     */
    rollbackToCheckpoint(checkpointId: string): Promise<{
        restored: number;
        removed: number;
    }>;
    /**
     * Gets the latest starred (stable) checkpoint.
     *
     * @returns Promise resolving to the checkpoint or null if none starred
     *
     * @example
     * ```typescript
     * const stable = await vectorStore.getLatestStableCheckpoint()
     * if (stable) {
     *   console.log(`Latest stable: ${stable.name}`)
     * }
     * ```
     */
    getLatestStableCheckpoint(): Promise<Checkpoint | null>;
    /**
     * Exports knowledge entries for LoRA training.
     *
     * Exports entries from checkpoint time in specified format.
     *
     * @param checkpointId - Optional checkpoint to export from (defaults to latest starred)
     * @param format - Export format ('jsonl', 'json', or 'parquet')
     * @returns Promise resolving to export data with statistics
     * @throws {ValidationError} If no checkpoint found
     * @throws {StorageError} If database operation fails
     *
     * @example
     * ```typescript
     * const export = await vectorStore.exportForLoRA(undefined, 'jsonl')
     * console.log(`Exporting ${export.statistics.totalEntries} entries`)
     * ```
     */
    exportForLoRA(checkpointId?: string, format?: LoRAExport['format']): Promise<LoRAExport>;
    /**
     * Generate embedding for text
     * Uses custom generator if provided, otherwise hash-based fallback.
     */
    private generateEmbedding;
    /**
     * Cache an embedding with LRU eviction
     */
    private setCachedEmbedding;
    private ensureInitialized;
    private putEntry;
    private getCheckpoint;
    private putCheckpoint;
    private hashVectors;
    private estimateTokens;
}
//# sourceMappingURL=vector-store.d.ts.map