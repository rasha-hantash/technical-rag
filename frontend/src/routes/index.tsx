import { useState, useCallback } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { ChatPanel, EvidencePanel } from '../components'
import { useDocuments, useIngest, useRagQuery } from '../hooks'
import type { Message, SourceResponse } from '../lib/types'

export const Route = createFileRoute('/')({
  component: Home,
})

function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [currentSources, setCurrentSources] = useState<SourceResponse[]>([])
  const [showEvidence, setShowEvidence] = useState(false)

  const { documents, refresh } = useDocuments()

  const { uploadFiles, isUploading, uploadingFileName, error: uploadError, clearError } =
    useIngest(refresh)

  const { submitQuery, isQuerying } = useRagQuery()

  const handleSubmitQuery = useCallback(
    async (question: string) => {
      setMessages((prev) => [...prev, { role: 'user', content: question }])
      setCurrentSources([])

      const response = await submitQuery(question)

      if (response) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: response.answer,
            sources: response.sources,
          },
        ])
        setCurrentSources(response.sources)
        if (response.sources.length > 0) {
          setShowEvidence(true)
        }
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: 'Sorry, something went wrong. Please try again.',
          },
        ])
      }
    },
    [submitQuery],
  )

  return (
    <div className="flex h-screen">
      <div
        className={`transition-all duration-300 ease-in-out ${
          showEvidence ? 'w-1/2' : 'w-full'
        }`}
      >
        <ChatPanel
          messages={messages}
          documents={documents}
          isUploading={isUploading}
          uploadingFileName={uploadingFileName}
          uploadError={uploadError}
          isQuerying={isQuerying}
          onClearUploadError={clearError}
          onFilesSelected={uploadFiles}
          onSubmitQuery={handleSubmitQuery}
        />
      </div>

      {showEvidence && (
        <div className="w-1/2 transition-all duration-300 ease-in-out">
          <EvidencePanel sources={currentSources} isLoading={isQuerying} />
        </div>
      )}
    </div>
  )
}
