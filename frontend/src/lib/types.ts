export interface SourceResponse {
  chunk_id: string | null;
  document_id: string | null;
  file_path: string;
  page_number: number | null;
  content: string;
  content_preview: string;
  bbox: [number, number, number, number] | null;
  section_hierarchy: string | null;
  book_title: string | null;
  book_author: string | null;
  publication_year: number | null;
  tags: string[];
}

export interface QueryResponse {
  answer: string;
  sources: SourceResponse[];
  chunks_used: number;
}

export interface BatchIngestItemResponse {
  file_name: string;
  document_id: string | null;
  chunks_count: number;
  was_duplicate: boolean;
  error: string | null;
}

export interface BatchIngestResponse {
  results: BatchIngestItemResponse[];
  successful: number;
  duplicates: number;
  failed: number;
}

export interface DocumentResponse {
  id: string;
  file_path: string;
  chunks_count: number;
  status: "processing" | "processed" | "error";
  file_size: number | null;
  created_at: string;
  title: string | null;
  author: string | null;
  edition: string | null;
  publication_year: number | null;
  tags: string[];
}

export interface BookMetadata {
  title: string;
  author: string;
  edition: string;
  publication_year: number | null;
}

export interface BookSection {
  section_hierarchy: string;
  chunk_count: number;
  start_page: number | null;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceResponse[];
}
