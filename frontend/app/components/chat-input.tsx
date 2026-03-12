'use client';

import { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  onInputStart?: () => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, onInputStart, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  return (
    <div className="flex-none pb-6 px-6 relative z-10">
      <form
        onSubmit={handleSubmit}
        className="relative flex items-center bg-white/70 rounded-[30px] p-1.5 border border-paper-border transition-all duration-500 focus-within:border-paper-accent shadow-[0_4px_12px_rgba(0,0,0,0.06),inset_0_2px_8px_rgba(0,0,0,0.08)]"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => {
            const newValue = e.target.value;
            setInput(newValue);
            // 用户开始输入时通知父组件
            if (newValue.length > 0 && onInputStart) {
              onInputStart();
            }
          }}
          onFocus={() => console.log('🎯 Input focus, disabled:', disabled)}
          placeholder="在这里，和榛子聊聊..."
          disabled={disabled}
          className="flex-1 bg-transparent px-5 py-3.5 outline-none text-paper-text placeholder:text-paper-muted/60 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || disabled}
          className="p-3 bg-paper-ai text-paper-text rounded-full disabled:opacity-30 disabled:grayscale transition-all duration-300 hover:scale-105 active:scale-95 shadow-sm"
        >
          <Send className="w-5 h-5 stroke-[1.5]" />
        </button>
      </form>
    </div>
  );
}
