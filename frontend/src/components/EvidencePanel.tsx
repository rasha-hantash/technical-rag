import { useState } from 'react'
import type { SourceResponse } from '../lib/types'
import { EvidenceCard } from './EvidenceCard'

interface EvidencePanelProps {
  sources: SourceResponse[]
  isLoading: boolean
}

export function EvidencePanel({ sources, isLoading }: EvidencePanelProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)

  const handleToggle = (index: number) => {
    setExpandedIndex((prev) => (prev === index ? null : index))
  }

  return (
    <div className="flex h-full flex-col border-l border-border-warm bg-cream">
      <div className="border-b border-border-warm bg-warm-white px-4 py-3">
        <h2 className="text-sm font-medium text-stone-700">
          Sources
          {sources.length > 0 && (
            <span className="ml-1.5 text-stone-400">({sources.length})</span>
          )}
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {isLoading && (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="animate-pulse rounded-lg border border-border-warm bg-warm-white p-3"
              >
                <div className="h-3 w-24 rounded bg-stone-200 mb-2" />
                <div className="h-3 w-full rounded bg-stone-100" />
                <div className="h-3 w-2/3 rounded bg-stone-100 mt-1" />
              </div>
            ))}
          </div>
        )}

        {!isLoading && sources.length === 0 && (
          <div className="flex items-center justify-center h-32 text-sm text-stone-400">
            No evidence for this query
          </div>
        )}

        {!isLoading &&
          sources.map((source, i) => (
            <EvidenceCard
              key={source.chunk_id ?? i}
              source={source}
              index={i}
              isExpanded={expandedIndex === i}
              onToggle={() => handleToggle(i)}
            />
          ))}
      </div>
    </div>
  )
}
