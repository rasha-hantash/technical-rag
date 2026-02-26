import { useState, useRef } from 'react'
import type { DocumentResponse } from '../lib/types'

interface DocumentListProps {
  documents: DocumentResponse[]
  isUploading: boolean
  uploadingFileName: string | null
  error: string | null
  onClearError: () => void
  onFilesSelected: (files: FileList) => void
}

function formatFileSize(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function StatusBadge({ status }: { status: DocumentResponse['status'] }) {
  switch (status) {
    case 'processed':
      return (
        <span className="inline-flex items-center rounded-full bg-green-50 px-2 py-0.5 text-xs font-medium text-green-700">
          Processed
        </span>
      )
    case 'processing':
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700">
          <svg
            className="h-3 w-3 animate-spin"
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
          Processing
        </span>
      )
    case 'error':
      return (
        <span className="inline-flex items-center rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700">
          Error
        </span>
      )
  }
}

export function DocumentList({
  documents,
  isUploading,
  uploadingFileName,
  error,
  onClearError,
  onFilesSelected,
}: DocumentListProps) {
  const [collapsed, setCollapsed] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFolderClick = () => {
    folderInputRef.current?.click()
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files)
      e.target.value = ''
    }
  }

  const handleFolderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const pdfFiles = Array.from(e.target.files).filter((f) =>
        f.name.toLowerCase().endsWith('.pdf'),
      )
      if (pdfFiles.length > 0) {
        const dt = new DataTransfer()
        for (const file of pdfFiles) {
          dt.items.add(file)
        }
        onFilesSelected(dt.files)
      }
      e.target.value = ''
    }
  }

  const processedCount = documents.filter((d) => d.status === 'processed').length
  const totalChunks = documents.reduce((sum, d) => sum + d.chunks_count, 0)

  return (
    <div className="border-b border-border-warm bg-warm-white">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={handleChange}
      />
      <input
        ref={folderInputRef}
        type="file"
        className="hidden"
        onChange={handleFolderChange}
        {...({ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
      />

      {documents.length === 0 && !isUploading ? (
        <div className="text-center px-4 py-4">
          <p className="text-sm text-stone-500 mb-2">
            Upload PDF documents to get started
          </p>
          <div className="flex gap-2 justify-center">
            <button
              onClick={handleClick}
              className="inline-flex items-center gap-1.5 rounded-lg bg-terracotta px-3.5 py-2 text-sm font-medium text-white hover:bg-terracotta-hover transition-colors"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              Upload files
            </button>
            <button
              onClick={handleFolderClick}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border-warm px-3.5 py-2 text-sm font-medium text-stone-700 hover:bg-cream transition-colors"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                />
              </svg>
              Upload folder
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* Header bar */}
          <div className="flex items-center justify-between px-4 py-3">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="flex items-center gap-2 text-sm text-stone-700 hover:text-stone-900 transition-colors"
            >
              <svg
                className={`h-4 w-4 transition-transform ${collapsed ? '' : 'rotate-90'}`}
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
              <svg
                className="h-4 w-4 text-stone-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <span>
                {documents.length} {documents.length === 1 ? 'file' : 'files'}
                {' · '}
                {processedCount} processed
                {' · '}
                {totalChunks} {totalChunks === 1 ? 'chunk' : 'chunks'}
              </span>
            </button>

            <div className="flex items-center gap-2">
              {isUploading && uploadingFileName && (
                <span className="inline-flex items-center gap-1 text-xs text-terracotta">
                  <svg
                    className="h-3 w-3 animate-spin"
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
                  {uploadingFileName}
                </span>
              )}
              <button
                onClick={handleClick}
                disabled={isUploading}
                className="inline-flex items-center gap-1 rounded-lg border border-border-warm px-2.5 py-1.5 text-xs font-medium text-stone-600 hover:bg-cream transition-colors disabled:opacity-50"
              >
                <svg
                  className="h-3 w-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                Upload files
              </button>
              <button
                onClick={handleFolderClick}
                disabled={isUploading}
                className="inline-flex items-center gap-1 rounded-lg border border-border-warm px-2.5 py-1.5 text-xs font-medium text-stone-600 hover:bg-cream transition-colors disabled:opacity-50"
              >
                <svg
                  className="h-3 w-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                  />
                </svg>
                Upload folder
              </button>
            </div>
          </div>

          {/* Collapsible file list */}
          {!collapsed && (
            <div className="max-h-48 overflow-y-auto border-t border-border-warm">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center gap-3 px-4 py-2 text-sm border-b border-border-warm last:border-b-0 hover:bg-cream/50 transition-colors"
                >
                  <svg
                    className="h-4 w-4 flex-shrink-0 text-stone-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <span className="flex-1 truncate text-stone-700">
                    {doc.file_path.split('/').pop() || doc.file_path}
                  </span>
                  {doc.file_size !== null && (
                    <span className="text-xs text-stone-400 flex-shrink-0">
                      {formatFileSize(doc.file_size)}
                    </span>
                  )}
                  <span className="text-xs text-stone-400 flex-shrink-0">
                    {doc.chunks_count} {doc.chunks_count === 1 ? 'chunk' : 'chunks'}
                  </span>
                  <StatusBadge status={doc.status} />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {error && (
        <div className="px-4 py-2 flex items-center gap-2 text-xs text-red-600 border-t border-border-warm">
          <span>{error}</span>
          <button onClick={onClearError} className="underline hover:no-underline">
            dismiss
          </button>
        </div>
      )}
    </div>
  )
}
