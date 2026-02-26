import { useState, useCallback } from 'react'
import { queryRag } from '../lib/api'
import type { QueryResponse } from '../lib/types'

interface UseRagQueryReturn {
  submitQuery: (question: string) => Promise<QueryResponse | null>
  isQuerying: boolean
  error: string | null
}

export function useRagQuery(): UseRagQueryReturn {
  const [isQuerying, setIsQuerying] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submitQuery = useCallback(
    async (question: string): Promise<QueryResponse | null> => {
      setIsQuerying(true)
      setError(null)

      try {
        const response = await queryRag(question)
        return response
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Query failed')
        return null
      } finally {
        setIsQuerying(false)
      }
    },
    [],
  )

  return { submitQuery, isQuerying, error }
}
