/**
 * Comprehensive tests for Error Types
 */

import { describe, it, expect } from 'vitest'
import {
  VectorSearchError,
  StorageError,
  ValidationError,
  NotFoundError,
  QuotaError,
} from './errors'

// ============================================================================
// VectorSearchError (base class)
// ============================================================================

describe('VectorSearchError', () => {
  it('should be an instance of Error', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
    })
    expect(err).toBeInstanceOf(Error)
    expect(err).toBeInstanceOf(VectorSearchError)
  })

  it('should store message correctly', () => {
    const err = new VectorSearchError('something went wrong', {
      category: 'validation',
      severity: 'low',
      recovery: 'recoverable',
    })
    expect(err.message).toBe('something went wrong')
  })

  it('should store category', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
    })
    expect(err.category).toBe('system')
  })

  it('should store severity', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'high',
      recovery: 'fatal',
    })
    expect(err.severity).toBe('high')
  })

  it('should store recovery potential', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'recoverable',
    })
    expect(err.recovery).toBe('recoverable')
  })

  it('should store technical details when provided', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
      technicalDetails: 'DB connection failed',
    })
    expect(err.technicalDetails).toBe('DB connection failed')
  })

  it('should have undefined technical details when not provided', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
    })
    expect(err.technicalDetails).toBeUndefined()
  })

  it('should store context when provided', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
      context: { key: 'value', count: 42 },
    })
    expect(err.context).toEqual({ key: 'value', count: 42 })
  })

  it('should have undefined context when not provided', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
    })
    expect(err.context).toBeUndefined()
  })

  it('should set name to constructor name', () => {
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
    })
    expect(err.name).toBe('VectorSearchError')
  })

  it('should set timestamp to a number', () => {
    const before = Date.now()
    const err = new VectorSearchError('test', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
    })
    const after = Date.now()
    expect(err.timestamp).toBeGreaterThanOrEqual(before)
    expect(err.timestamp).toBeLessThanOrEqual(after)
  })

  it('should support cause option', () => {
    const cause = new Error('original error')
    const err = new VectorSearchError('wrapper', {
      category: 'system',
      severity: 'critical',
      recovery: 'fatal',
      cause,
    })
    expect(err.cause).toBe(cause)
  })

  it('should accept all valid category values', () => {
    const categories = ['system', 'validation', 'not-found', 'quota'] as const
    for (const cat of categories) {
      const err = new VectorSearchError('test', {
        category: cat,
        severity: 'low',
        recovery: 'recoverable',
      })
      expect(err.category).toBe(cat)
    }
  })

  it('should accept all valid severity values', () => {
    const severities = ['critical', 'high', 'medium', 'low'] as const
    for (const sev of severities) {
      const err = new VectorSearchError('test', {
        category: 'system',
        severity: sev,
        recovery: 'fatal',
      })
      expect(err.severity).toBe(sev)
    }
  })

  it('should accept all valid recovery values', () => {
    const recoveries = ['recoverable', 'fatal'] as const
    for (const rec of recoveries) {
      const err = new VectorSearchError('test', {
        category: 'system',
        severity: 'low',
        recovery: rec,
      })
      expect(err.recovery).toBe(rec)
    }
  })
})

// ============================================================================
// StorageError
// ============================================================================

describe('StorageError', () => {
  it('should be an instance of VectorSearchError', () => {
    const err = new StorageError('db failed')
    expect(err).toBeInstanceOf(VectorSearchError)
    expect(err).toBeInstanceOf(StorageError)
  })

  it('should have category "system"', () => {
    const err = new StorageError('db failed')
    expect(err.category).toBe('system')
  })

  it('should have severity "critical" by default', () => {
    const err = new StorageError('db failed')
    expect(err.severity).toBe('critical')
  })

  it('should have recovery "fatal" by default', () => {
    const err = new StorageError('db failed')
    expect(err.recovery).toBe('fatal')
  })

  it('should accept custom severity', () => {
    const err = new StorageError('db failed', { severity: 'high' })
    expect(err.severity).toBe('high')
  })

  it('should accept technical details', () => {
    const err = new StorageError('db failed', {
      technicalDetails: 'Connection timeout after 5000ms',
    })
    expect(err.technicalDetails).toBe('Connection timeout after 5000ms')
  })

  it('should accept context', () => {
    const err = new StorageError('db failed', {
      context: { dbName: 'test', operation: 'read' },
    })
    expect(err.context).toEqual({ dbName: 'test', operation: 'read' })
  })

  it('should accept cause', () => {
    const cause = new Error('network error')
    const err = new StorageError('db failed', { cause })
    expect(err.cause).toBe(cause)
  })

  it('should set name to StorageError', () => {
    const err = new StorageError('db failed')
    expect(err.name).toBe('StorageError')
  })

  it('should work with no options', () => {
    const err = new StorageError('simple error')
    expect(err.message).toBe('simple error')
    expect(err.technicalDetails).toBeUndefined()
    expect(err.context).toBeUndefined()
  })
})

// ============================================================================
// ValidationError
// ============================================================================

describe('ValidationError', () => {
  it('should be an instance of VectorSearchError', () => {
    const err = new ValidationError('invalid input')
    expect(err).toBeInstanceOf(VectorSearchError)
    expect(err).toBeInstanceOf(ValidationError)
  })

  it('should have category "validation"', () => {
    const err = new ValidationError('invalid input')
    expect(err.category).toBe('validation')
  })

  it('should have severity "low"', () => {
    const err = new ValidationError('invalid input')
    expect(err.severity).toBe('low')
  })

  it('should have recovery "recoverable"', () => {
    const err = new ValidationError('invalid input')
    expect(err.recovery).toBe('recoverable')
  })

  it('should store field name', () => {
    const err = new ValidationError('invalid input', { field: 'email' })
    expect(err.field).toBe('email')
  })

  it('should store field value', () => {
    const err = new ValidationError('invalid input', {
      field: 'age',
      value: -5,
    })
    expect(err.field).toBe('age')
    expect(err.value).toBe(-5)
  })

  it('should include field and value in context', () => {
    const err = new ValidationError('invalid input', {
      field: 'name',
      value: '',
    })
    expect(err.context).toEqual({ field: 'name', value: '' })
  })

  it('should set name to ValidationError', () => {
    const err = new ValidationError('invalid input')
    expect(err.name).toBe('ValidationError')
  })

  it('should work with no options', () => {
    const err = new ValidationError('simple validation error')
    expect(err.message).toBe('simple validation error')
    expect(err.field).toBeUndefined()
    expect(err.value).toBeUndefined()
  })

  it('should accept technical details', () => {
    const err = new ValidationError('invalid input', {
      technicalDetails: 'Expected non-empty string',
    })
    expect(err.technicalDetails).toBe('Expected non-empty string')
  })
})

// ============================================================================
// NotFoundError
// ============================================================================

describe('NotFoundError', () => {
  it('should be an instance of VectorSearchError', () => {
    const err = new NotFoundError('entry', '123')
    expect(err).toBeInstanceOf(VectorSearchError)
    expect(err).toBeInstanceOf(NotFoundError)
  })

  it('should have category "not-found"', () => {
    const err = new NotFoundError('entry', '123')
    expect(err.category).toBe('not-found')
  })

  it('should have severity "medium"', () => {
    const err = new NotFoundError('entry', '123')
    expect(err.severity).toBe('medium')
  })

  it('should have recovery "recoverable"', () => {
    const err = new NotFoundError('entry', '123')
    expect(err.recovery).toBe('recoverable')
  })

  it('should include resource name in message', () => {
    const err = new NotFoundError('checkpoint', 'cp_abc')
    expect(err.message).toContain('checkpoint')
    expect(err.message).toContain('cp_abc')
  })

  it('should store resource type', () => {
    const err = new NotFoundError('knowledge entry', 'ke_123')
    expect(err.resource).toBe('knowledge entry')
  })

  it('should store resource id', () => {
    const err = new NotFoundError('entry', 'id_456')
    expect(err.id).toBe('id_456')
  })

  it('should handle missing id', () => {
    const err = new NotFoundError('entry')
    expect(err.id).toBeUndefined()
    expect(err.message).toContain('entry')
    expect(err.message).not.toContain('undefined')
  })

  it('should include resource and id in context', () => {
    const err = new NotFoundError('checkpoint', 'cp_test')
    expect(err.context).toEqual({ resource: 'checkpoint', id: 'cp_test' })
  })

  it('should set name to NotFoundError', () => {
    const err = new NotFoundError('entry')
    expect(err.name).toBe('NotFoundError')
  })

  it('should accept technical details', () => {
    const err = new NotFoundError('entry', '123', {
      technicalDetails: 'No record found in database',
    })
    expect(err.technicalDetails).toBe('No record found in database')
  })
})

// ============================================================================
// QuotaError
// ============================================================================

describe('QuotaError', () => {
  it('should be an instance of VectorSearchError', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024)
    expect(err).toBeInstanceOf(VectorSearchError)
    expect(err).toBeInstanceOf(QuotaError)
  })

  it('should have category "quota"', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024)
    expect(err.category).toBe('quota')
  })

  it('should have severity "high"', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024)
    expect(err.severity).toBe('high')
  })

  it('should have recovery "recoverable"', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024)
    expect(err.recovery).toBe('recoverable')
  })

  it('should store usedBytes and totalBytes', () => {
    const err = new QuotaError(50000000, 100000000)
    expect(err.usedBytes).toBe(50000000)
    expect(err.totalBytes).toBe(100000000)
  })

  it('should include MB values in message', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024)
    expect(err.message).toContain('50MB')
    expect(err.message).toContain('100MB')
  })

  it('should include quota info in technical details by default', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024)
    expect(err.technicalDetails).toContain('104857600')
    expect(err.technicalDetails).toContain('52428800')
  })

  it('should accept custom technical details', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024, {
      technicalDetails: 'Custom details',
    })
    expect(err.technicalDetails).toBe('Custom details')
  })

  it('should accept context', () => {
    const err = new QuotaError(50 * 1024 * 1024, 100 * 1024 * 1024, {
      context: { storeName: 'entries' },
    })
    expect(err.context).toEqual({ usedBytes: 50 * 1024 * 1024, totalBytes: 100 * 1024 * 1024, storeName: 'entries' })
  })

  it('should set name to QuotaError', () => {
    const err = new QuotaError(1, 1)
    expect(err.name).toBe('QuotaError')
  })

  it('should handle zero values', () => {
    const err = new QuotaError(0, 0)
    expect(err.message).toContain('0MB')
  })
})

// ============================================================================
// Error Hierarchy
// ============================================================================

describe('Error Hierarchy', () => {
  it('StorageError should be catchable as VectorSearchError', () => {
    const throwIt = () => {
      throw new StorageError('fail')
    }
    expect(throwIt).toThrow(VectorSearchError)
  })

  it('ValidationError should be catchable as VectorSearchError', () => {
    const throwIt = () => {
      throw new ValidationError('fail')
    }
    expect(throwIt).toThrow(VectorSearchError)
  })

  it('NotFoundError should be catchable as VectorSearchError', () => {
    const throwIt = () => {
      throw new NotFoundError('res')
    }
    expect(throwIt).toThrow(VectorSearchError)
  })

  it('QuotaError should be catchable as VectorSearchError', () => {
    const throwIt = () => {
      throw new QuotaError(1, 1)
    }
    expect(throwIt).toThrow(VectorSearchError)
  })

  it('should use instanceof to differentiate error types', () => {
    const errors = [
      new StorageError('a'),
      new ValidationError('b'),
      new NotFoundError('c'),
      new QuotaError(1, 1),
    ]

    expect(errors[0]).toBeInstanceOf(StorageError)
    expect(errors[0]).not.toBeInstanceOf(ValidationError)

    expect(errors[1]).toBeInstanceOf(ValidationError)
    expect(errors[1]).not.toBeInstanceOf(StorageError)

    expect(errors[2]).toBeInstanceOf(NotFoundError)
    expect(errors[2]).not.toBeInstanceOf(StorageError)

    expect(errors[3]).toBeInstanceOf(QuotaError)
    expect(errors[3]).not.toBeInstanceOf(StorageError)
  })
})
