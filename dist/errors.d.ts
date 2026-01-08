/**
 * Error Types for Vector Search
 *
 * Simplified error handling system for the vector search package.
 */
export type ErrorCategory = 'system' | 'validation' | 'not-found' | 'quota';
export type ErrorSeverity = 'critical' | 'high' | 'medium' | 'low';
export type RecoveryPotential = 'recoverable' | 'fatal';
/**
 * Base error class for vector search errors
 */
export declare class VectorSearchError extends Error {
    readonly category: ErrorCategory;
    readonly severity: ErrorSeverity;
    readonly recovery: RecoveryPotential;
    readonly technicalDetails?: string;
    readonly context?: Record<string, unknown>;
    readonly timestamp: number;
    constructor(message: string, options: {
        category: ErrorCategory;
        severity: ErrorSeverity;
        recovery: RecoveryPotential;
        technicalDetails?: string;
        context?: Record<string, unknown>;
        cause?: Error;
    });
}
/**
 * Database storage error
 */
export declare class StorageError extends VectorSearchError {
    constructor(message: string, options?: {
        technicalDetails?: string;
        severity?: ErrorSeverity;
        context?: Record<string, unknown>;
        cause?: Error;
    });
}
/**
 * Validation error
 */
export declare class ValidationError extends VectorSearchError {
    readonly field?: string;
    readonly value?: unknown;
    constructor(message: string, options?: {
        field?: string;
        value?: unknown;
        technicalDetails?: string;
        context?: Record<string, unknown>;
        cause?: Error;
    });
}
/**
 * Not found error
 */
export declare class NotFoundError extends VectorSearchError {
    readonly resource: string;
    readonly id?: string;
    constructor(resource: string, id?: string, options?: {
        technicalDetails?: string;
        context?: Record<string, unknown>;
        cause?: Error;
    });
}
/**
 * Storage quota exceeded error
 */
export declare class QuotaError extends VectorSearchError {
    readonly usedBytes: number;
    readonly totalBytes: number;
    constructor(usedBytes: number, totalBytes: number, options?: {
        technicalDetails?: string;
        context?: Record<string, unknown>;
        cause?: Error;
    });
}
//# sourceMappingURL=errors.d.ts.map