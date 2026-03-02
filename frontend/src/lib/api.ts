import type {
  BatchIngestResponse,
  QueryResponse,
  DocumentResponse,
  BookMetadata,
  BookSection,
} from "./types";

export async function ingestBatch(
  files: File[],
  metadata?: Partial<BookMetadata>,
  tags?: string[],
): Promise<BatchIngestResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  if (metadata?.title) formData.append("title", metadata.title);
  if (metadata?.author) formData.append("author", metadata.author);
  if (metadata?.edition) formData.append("edition", metadata.edition);
  if (metadata?.publication_year != null)
    formData.append("publication_year", String(metadata.publication_year));
  if (tags && tags.length > 0) {
    formData.append("tags", JSON.stringify(tags));
  }

  const res = await fetch("/api/v1/rag/ingest/batch", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Batch upload failed (${res.status})`);
  }

  return res.json();
}

export async function queryRag(
  question: string,
  topK = 5,
  tags?: string[] | null,
): Promise<QueryResponse> {
  const body: Record<string, unknown> = { question, top_k: topK };
  if (tags && tags.length > 0) {
    body.tags = tags;
  }
  const res = await fetch("/api/v1/rag/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Query failed (${res.status})`);
  }

  return res.json();
}

export async function listDocuments(): Promise<DocumentResponse[]> {
  const res = await fetch("/api/v1/documents");
  if (!res.ok) {
    throw new Error(`Failed to list documents (${res.status})`);
  }
  return res.json();
}

export function getPdfFileUrl(documentId: string): string {
  return `/api/v1/documents/${documentId}/file`;
}

export async function updateBookMetadata(
  documentId: string,
  metadata: Partial<BookMetadata>,
): Promise<DocumentResponse> {
  const res = await fetch(`/api/v1/documents/${documentId}/metadata`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata),
  });
  if (!res.ok) throw new Error(`Failed to update metadata (${res.status})`);
  return res.json();
}

export async function getBookSections(
  documentId: string,
): Promise<BookSection[]> {
  const res = await fetch(`/api/v1/documents/${documentId}/sections`);
  if (!res.ok) throw new Error(`Failed to get sections (${res.status})`);
  return res.json();
}
