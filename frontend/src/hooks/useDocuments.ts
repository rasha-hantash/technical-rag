import { useState, useEffect, useCallback, useRef } from 'react'
import { listDocuments } from '../lib/api'
import type { DocumentResponse } from '../lib/types'

interface UseDocumentsReturn {
  documents: DocumentResponse[]
  isLoading: boolean
  refresh: () => Promise<void>
}

export function useDocuments(): UseDocumentsReturn {
  const [documents, setDocuments] = useState<DocumentResponse[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const refresh = useCallback(async () => {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } catch {
      // Backend might not be running yet
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    const hasProcessing = documents.some((d) => d.status === 'processing')
    if (hasProcessing) {
      pollingRef.current = setInterval(refresh, 2000)
    } else if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [documents, refresh])

  return { documents, isLoading, refresh }
}
