'use client';

import { memo, useRef, useEffect } from 'react';
import { StreamMessage } from '../lib/useStreamChat';

interface ChatMessageProps {
  message: StreamMessage;
  index?: number;
}

export const ChatMessage = memo(({ message, index = 0 }: ChatMessageProps) => {
  const isAi = message.role === 'assistant';
  const bubbleRef = useRef<HTMLDivElement>(null);
  const hasContent = message.content.trim().length > 0;

  // 空内容时不显示
  if (!hasContent) {
    return null;
  }

  return (
    <div
      className={`
        flex w-full mb-8
        ${isAi ? 'justify-start' : 'justify-end'}
      `}
    >
      <div
        ref={bubbleRef}
        className={`
          max-w-[85%] px-5 py-3.5 shadow-bubble border border-black/[0.01]
          leading-relaxed tracking-wide text-[15.5px]
          ${isAi
            ? 'animate-bubble-pop-left bg-paper-ai text-paper-text rounded-[22px] rounded-tl-[4px]'
            : 'animate-bubble-pop-right bg-paper-user text-paper-text rounded-[22px] rounded-tr-[4px]'
          }
        `}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>
      </div>
    </div>
  );
});
