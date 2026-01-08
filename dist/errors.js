/**
 * Error Types for Vector Search
 *
 * Simplified error handling system for the vector search package.
 */
/**
 * Base error class for vector search errors
 */
export class VectorSearchError extends Error {
    constructor(message, options) {
        super(message);
        this.name = this.constructor.name;
        this.category = options.category;
        this.severity = options.severity;
        this.recovery = options.recovery;
        this.technicalDetails = options.technicalDetails;
        this.context = options.context;
        this.timestamp = Date.now();
        if (options.cause) {
            Object.defineProperty(this, 'cause', {
                value: options.cause,
                writable: false,
                enumerable: false,
                configurable: true,
            });
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        if (typeof Error.captureStackTrace === 'function') {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            Error.captureStackTrace(this, this.constructor);
        }
    }
}
/**
 * Database storage error
 */
export class StorageError extends VectorSearchError {
    constructor(message, options = {}) {
        super(message, {
            category: 'system',
            severity: options.severity || 'critical',
            recovery: 'fatal',
            technicalDetails: options.technicalDetails,
            context: options.context,
            cause: options.cause,
        });
    }
}
/**
 * Validation error
 */
export class ValidationError extends VectorSearchError {
    constructor(message, options = {}) {
        super(message, {
            category: 'validation',
            severity: 'low',
            recovery: 'recoverable',
            technicalDetails: options.technicalDetails,
            context: { field: options.field, value: options.value, ...options.context },
            cause: options.cause,
        });
        this.field = options.field;
        this.value = options.value;
    }
}
/**
 * Not found error
 */
export class NotFoundError extends VectorSearchError {
    constructor(resource, id, options = {}) {
        super(`Resource not found: ${resource}${id ? ` (${id})` : ''}`, {
            category: 'not-found',
            severity: 'medium',
            recovery: 'recoverable',
            technicalDetails: options.technicalDetails,
            context: { resource, id, ...options.context },
            cause: options.cause,
        });
        this.resource = resource;
        this.id = id;
    }
}
/**
 * Storage quota exceeded error
 */
export class QuotaError extends VectorSearchError {
    constructor(usedBytes, totalBytes, options = {}) {
        const usedMB = Math.round(usedBytes / (1024 * 1024));
        const totalMB = Math.round(totalBytes / (1024 * 1024));
        super(`Storage quota exceeded: ${usedMB}MB used of ${totalMB}MB`, {
            category: 'quota',
            severity: 'high',
            recovery: 'recoverable',
            technicalDetails: options.technicalDetails || `Quota: ${totalBytes} bytes, Used: ${usedBytes} bytes`,
            context: { usedBytes, totalBytes, ...options.context },
            cause: options.cause,
        });
        this.usedBytes = usedBytes;
        this.totalBytes = totalBytes;
    }
}
