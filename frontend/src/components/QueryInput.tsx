import { useState, useRef, useEffect } from 'react'

interface QueryInputProps {
  disabled: boolean
  isQuerying: boolean
  hasDocuments: boolean
  onSubmit: (question: string) => void
}

export function QueryInput({
  disabled,
  isQuerying,
  hasDocuments,
  onSubmit,
}: QueryInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const isDisabled = disabled || isQuerying || !hasDocuments

  const handleSubmit = () => {
    const trimmed = value.trim()
    if (trimmed && !isDisabled) {
      onSubmit(trimmed)
      setValue('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + 'px'
    }
  }, [value])

  return (
    <div className="border-t border-border-warm bg-warm-white p-4">
      <div
        className={`flex items-end gap-2 rounded-xl border px-3 py-2 transition-colors ${
          isDisabled
            ? 'border-border-warm bg-cream'
            : 'border-border-warm bg-warm-white focus-within:border-terracotta'
        }`}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isDisabled}
          placeholder={
            hasDocuments
              ? 'Ask a question about your documents...'
              : 'Upload a document first...'
          }
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm text-text-primary placeholder-stone-400 outline-none disabled:text-stone-400 disabled:cursor-not-allowed"
        />
        <button
          onClick={handleSubmit}
          disabled={isDisabled || !value.trim()}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-terracotta text-white transition-colors hover:bg-terracotta-hover disabled:bg-stone-300 disabled:cursor-not-allowed"
        >
          {isQuerying ? (
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
          ) : (
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
                d="M5 12h14M12 5l7 7-7 7"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
