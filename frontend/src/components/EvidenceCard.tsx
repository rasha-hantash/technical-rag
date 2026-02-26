import type { SourceResponse } from '../lib/types'
import { PdfPageViewer } from './PdfPageViewer'

interface EvidenceCardProps {
  source: SourceResponse
  index: number
  isExpanded: boolean
  onToggle: () => void
}

function shortFileName(filePath: string): string {
  const parts = filePath.split('/')
  return parts[parts.length - 1] || filePath
}

export function EvidenceCard({
  source,
  index,
  isExpanded,
  onToggle,
}: EvidenceCardProps) {
  return (
    <div
      className={`border rounded-lg transition-colors ${
        isExpanded
          ? 'border-terracotta bg-terracotta-light'
          : 'border-border-warm bg-warm-white hover:border-border-warm-hover'
      }`}
    >
      <button
        onClick={onToggle}
        className="w-full text-left px-3 py-2.5 flex items-start gap-2"
      >
        <svg
          className={`h-4 w-4 mt-0.5 shrink-0 transition-transform ${
            isExpanded ? 'rotate-90 text-terracotta' : 'text-stone-400'
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5l7 7-7 7"
          />
        </svg>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-xs">
            <span className="inline-flex items-center rounded bg-cream-dark px-1.5 py-0.5 font-medium text-stone-600">
              [{index + 1}]
            </span>
            {source.page_number && (
              <span className="text-stone-500">
                Page {source.page_number}
              </span>
            )}
            <span className="text-stone-400 truncate">
              {shortFileName(source.file_path)}
            </span>
          </div>
          <p className="mt-1 text-xs text-stone-600 line-clamp-2 leading-relaxed">
            {source.content_preview}
          </p>
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-border-warm p-3 space-y-3">
          <div className="text-xs text-stone-700 leading-relaxed max-h-32 overflow-y-auto whitespace-pre-wrap">
            {source.content}
          </div>

          {source.page_number && source.document_id && (
            <div className="rounded-lg overflow-hidden border border-border-warm">
              <PdfPageViewer
                documentId={source.document_id}
                pageNumber={source.page_number}
                bbox={source.bbox}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
