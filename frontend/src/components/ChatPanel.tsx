import { useRef, useEffect } from 'react'
import { DocumentList } from './DocumentList'
import { MessageBubble } from './MessageBubble'
import { QueryInput } from './QueryInput'
import type { Message, DocumentResponse } from '../lib/types'

interface ChatPanelProps {
  messages: Message[]
  documents: DocumentResponse[]
  isUploading: boolean
  uploadingFileName: string | null
  uploadError: string | null
  isQuerying: boolean
  onClearUploadError: () => void
  onFilesSelected: (files: FileList) => void
  onSubmitQuery: (question: string) => void
}

export function ChatPanel({
  messages,
  documents,
  isUploading,
  uploadingFileName,
  uploadError,
  isQuerying,
  onClearUploadError,
  onFilesSelected,
  onSubmitQuery,
}: ChatPanelProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex h-full flex-col">
      <DocumentList
        documents={documents}
        isUploading={isUploading}
        uploadingFileName={uploadingFileName}
        error={uploadError}
        onClearError={onClearUploadError}
        onFilesSelected={onFilesSelected}
      />

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && documents.length > 0 && (
          <div className="flex items-center justify-center h-full text-sm text-stone-400">
            Ask a question about your documents
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {isQuerying && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-warm-white border border-border-warm px-4 py-3">
              <div className="flex items-center gap-2 text-sm text-stone-400">
                <svg
                  className="h-4 w-4 animate-spin"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Thinking...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <QueryInput
        disabled={false}
        isQuerying={isQuerying}
        hasDocuments={documents.length > 0}
        onSubmit={onSubmitQuery}
      />
    </div>
  )
}
