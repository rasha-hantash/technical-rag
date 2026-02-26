import type { BatchIngestResponse, QueryResponse, DocumentResponse } from './types'

export async function ingestBatch(files: File[]): Promise<BatchIngestResponse> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }

  const res = await fetch('/api/v1/rag/ingest/batch', {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Batch upload failed (${res.status})`)
  }

  return res.json()
}

export async function queryRag(
  question: string,
  topK = 5,
): Promise<QueryResponse> {
  const res = await fetch('/api/v1/rag/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Query failed (${res.status})`)
  }

  return res.json()
}

export async function listDocuments(): Promise<DocumentResponse[]> {
  const res = await fetch('/api/v1/documents')
  if (!res.ok) {
    throw new Error(`Failed to list documents (${res.status})`)
  }
  return res.json()
}

export function getPdfFileUrl(documentId: string): string {
  return `/api/v1/documents/${documentId}/file`
}
