import type { Message } from '../lib/types'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-user-bubble text-text-primary'
            : 'text-text-primary'
        }`}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>
        {message.sources && message.sources.length > 0 && (
          <div className={`mt-2 pt-2 border-t text-xs ${
            isUser
              ? 'border-stone-300/30 text-stone-500'
              : 'border-border-warm/40 text-stone-400'
          }`}>
            {message.sources.length} source{message.sources.length !== 1 ? 's' : ''} found
          </div>
        )}
      </div>
    </div>
  )
}
