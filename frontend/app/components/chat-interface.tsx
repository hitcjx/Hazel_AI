'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { X } from 'lucide-react';
import { ChatMessage } from './chat-message';
import { ChatInput } from './chat-input';
import { TypingIndicator } from './typing-indicator';
import { useStreamChat, StreamMessage } from '../lib/useStreamChat';

interface ChatInterfaceProps {
  sessionId: string;
  studentName?: string;
}

export function ChatInterface({ sessionId, studentName }: ChatInterfaceProps) {
  const router = useRouter();
  const { messages, sendMessage, isLoading } = useStreamChat(sessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const prevMessageCountRef = useRef(0);
  const [isUserTyping, setIsUserTyping] = useState(false);

  // 监听 isLoading 变化
  useEffect(() => {
    console.log('🔍 ChatInterface: isLoading 变化:', isLoading, '时间:', Date.now());
  }, [isLoading]);

  // 页面关闭时自动保存数据
  useEffect(() => {
    const handleBeforeUnload = () => {
      const sessionId = localStorage.getItem('hazel_session_id');
      if (sessionId) {
        // 使用 sendBeacon 发送 GET 请求
        navigator.sendBeacon(
          `http://localhost:8000/api/session/end?session_id=${sessionId}`
        );
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  // 自动滚动到底部（只在消息数量增加时触发，避免频繁滚动）
  useEffect(() => {
    const currentCount = messages.length;
    if (currentCount > prevMessageCountRef.current) {
      // 消息数量增加了，滚动到底部
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
      prevMessageCountRef.current = currentCount;
    }
    // 如果只是消息内容更新（流式传输），不滚动
  }, [messages]);

  // 当AI消息完成流式输出时，向上滚动一点
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    // 只有当最后一条是AI消息且不再streaming时才滚动
    if (lastMessage && lastMessage.role === 'assistant' && !lastMessage.isStreaming && !isLoading) {
      // 等待一小段时间确保DOM已更新
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }, 100);
    }
  }, [messages, isLoading]);

  const handleSendMessage = (message: string) => {
    setIsUserTyping(false);
    sendMessage(message);
  };

  const handleUserInputStart = () => {
    setIsUserTyping(true);
  };

  const handleExit = () => {
    // 清除本地存储
    localStorage.removeItem('hazel_student_id');
    localStorage.removeItem('hazel_student_name');
    localStorage.removeItem('hazel_session_id');

    // 强制刷新页面回到首页
    window.location.href = '/';
  };

  // 检查是否有正在流式传输的消息
  const hasStreamingMessage = messages.some(msg => msg.isStreaming);

  // 计算气泡批次索引（仅对连续的AI多气泡应用延迟）
  const getBubbleIndex = (msgIndex: number): number => {
    const message = messages[msgIndex];
    if (message.role !== 'assistant') return 0; // 用户消息不延迟

    let bubbleCount = 0;
    // 从当前位置往前数连续的AI消息
    for (let i = msgIndex; i >= 0; i--) {
      if (messages[i].role === 'assistant') {
        if (i < msgIndex) {
          bubbleCount++;
        }
      } else {
        break; // 遇到用户消息，停止计数
      }
    }
    return bubbleCount;
  };

  // 动态状态文字
  const getStatusText = () => {
    // AI正在流式输出时显示"正在输入..."
    if (isLoading && hasStreamingMessage) return '正在输入...';
    // 其他时候都是"正在倾听中..."
    return '正在倾听中...';
  };

  return (
    <>
      {/* 左上角：用户信息 - 固定在屏幕左上角 */}
      <div className="fixed top-6 left-6 z-20">
        {studentName && (
          <span className="text-xs text-paper-muted tracking-widest">
            {studentName}
          </span>
        )}
      </div>

      {/* 右上角：退出按钮 - 固定在屏幕右上角 */}
      <div className="fixed top-6 right-6 z-20">
        <button
          onClick={handleExit}
          className="text-sm text-paper-muted hover:text-paper-text transition-colors flex items-center gap-1"
          aria-label="结束对话"
        >
          退出
          <X className="w-4 h-4 stroke-[1.5]" />
        </button>
      </div>

      {/* 物理信纸容器 - 固定高度的纸张，始终保持四个圆角 */}
      <div className="min-h-screen flex flex-col relative justify-center py-6 px-4">
        {/* 中央纸张 - 固定最小高度92vh，始终保持四个圆角 */}
        <div className="max-w-3xl mx-auto w-full min-h-[92vh] max-h-[96vh] bg-paper-surface rounded-[32px] border shadow-paper-3d flex flex-col relative overflow-hidden"
             style={{
               borderColor: 'rgba(140, 120, 90, 0.25)',
               borderWidth: '1px',
               boxShadow: '0 4px 12px rgba(140, 120, 90, 0.25), 0 12px 30px rgba(140, 120, 90, 0.2), 0 30px 60px rgba(140, 120, 90, 0.15), 0 60px 100px rgba(140, 120, 90, 0.1)'
             }}>

          {/* 纸张纹理层 - 真实水彩纸纹理 */}
          <div
            className="absolute inset-0 pointer-events-none rounded-[32px] z-0"
            style={{
              backgroundImage: `url('/paper.jpg')`,
              backgroundSize: '400px 400px',
              opacity: 0.6
            }}>
          </div>



          {/* 顶部区域 - Logo和标题，印在信笺上 */}
          <div className="flex-none pt-6 pb-4 px-6 flex flex-col items-center relative z-10">
            <div className="flex items-center gap-2">
              <Image
                src="/logo.png"
                alt="榛子"
                width={48}
                height={48}
                className="animate-breathe"
              />
              <h1 className="text-paper-text font-medium tracking-widest text-xl inline-flex">
                <span className="animate-float-left">榛</span>
                <span className="animate-float-right">子</span>
              </h1>
            </div>
            <span className="text-[10px] text-paper-status animate-status-breathe mt-0.5">
              {getStatusText()}
            </span>
          </div>

          {/* 消息区域 */}
          <div className="flex-1 overflow-y-auto hide-scrollbar px-6 py-6 relative z-10">
            {messages.length === 0 ? (
              <div className="text-center mt-20">
                <p className="text-paper-muted">
                  今天的对话开始了，有什么想聊的吗？
                </p>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <ChatMessage key={message.id} message={message} index={getBubbleIndex(index)} />
                ))}
                {!hasStreamingMessage && isLoading && <TypingIndicator />}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* 输入框 */}
          <ChatInput
            onSend={handleSendMessage}
            onInputStart={handleUserInputStart}
            disabled={isLoading}
          />
        </div>
      </div>
    </>
  );
}
