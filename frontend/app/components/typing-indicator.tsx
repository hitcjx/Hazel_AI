'use client';

export function TypingIndicator() {
  return (
    <div className="flex w-full mb-6 justify-start">
      <div className="bg-morandi-ai rounded-2xl rounded-tl-sm px-5 py-3 shadow-morandi">
        <div className="flex gap-1">
          <div className="w-2 h-2 bg-morandi-muted rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 bg-morandi-muted rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 bg-morandi-muted rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}
