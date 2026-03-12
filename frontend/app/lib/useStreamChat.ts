'use client';

import { useState, useCallback, useRef } from 'react';

export interface StreamMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;  // 标记是否正在流式传输
}

export function useStreamChat(sessionId: string) {
  const [messages, setMessages] = useState<StreamMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  // 缓存等待新气泡期间到达的内容
  const pendingContentRef = useRef('');
  const isWaitingForNewBubbleRef = useRef(false);

  const sendMessage = useCallback(async (content: string) => {
    const flowStart = Date.now();
    console.log('🚀 sendMessage 调用, isLoading:', isLoading, 'content:', content);

    if (!content.trim() || isLoading) {
      console.log('❌ sendMessage 被阻止: content为空或isLoading=true');
      return;
    }

    // 添加用户消息
    const userMessageId = Date.now().toString();
    const userMessage: StreamMessage = {
      id: userMessageId,
      role: 'user',
      content: content.trim(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    console.log('✅ setIsLoading(true) 已调用，时间:', Date.now() - flowStart, 'ms');

    // 创建AI消息占位符（标记为正在流式传输）
    const aiMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      {
        id: aiMessageId,
        role: 'assistant',
        content: '',
        isStreaming: true,
      },
    ]);
    console.log('✅ AI占位符已创建，时间:', Date.now() - flowStart, 'ms');

    let aiContent = '';  // 局部变量，用于累积内容

    try {
      // 取消之前的请求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      // 设置60秒超时
      const timeoutId = setTimeout(() => {
        abortControllerRef.current?.abort();
      }, 60000);

      const fetchStart = Date.now();
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content }],
          session_id: sessionId,
        }),
        signal: abortControllerRef.current.signal,
      });

      clearTimeout(timeoutId);

      console.log('⏱️ Fetch返回，耗时:', Date.now() - fetchStart, 'ms');

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }

      const readerStart = Date.now();
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }
      console.log('⏱️ Reader创建，耗时:', Date.now() - readerStart, 'ms');

      const decoder = new TextDecoder();
      let buffer = '';
      let chunkCount = 0;
      const startTime = Date.now();
      let firstChunkReceived = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        if (!firstChunkReceived) {
          console.log('⏱️ 第一个chunk接收，耗时:', Date.now() - startTime, 'ms');
          firstChunkReceived = true;
        }

        const chunkText = decoder.decode(value, { stream: true });
        chunkCount++;
        if (chunkCount % 10 === 0) {  // 每10个chunk记录一次，避免日志太多
          console.log(`📦 接收chunk #${chunkCount}, 时间: ${Date.now() - startTime}ms`);
        }

        buffer += chunkText;
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            if (data === '[DONE]') {
              console.log('🏁 [DONE] 信号接收，时间:', Date.now());
              // 流结束，标记当前消息为已完成
              if (aiContent.trim()) {
                setMessages((prev) => {
                  const lastIndex = prev.length - 1;
                  const lastMessage = prev[lastIndex];
                  if (lastMessage && lastMessage.id === aiMessageId) {
                    console.log('✅ [DONE] 标记消息完成，时间:', Date.now());
                    const newMessages = [...prev];
                    newMessages[lastIndex] = {
                      ...lastMessage,
                      content: aiContent.trim(),
                      isStreaming: false,
                    };
                    return newMessages;
                  }
                  return prev;
                });
              }
              continue;
            }

            try {
              const parsed = JSON.parse(data);

              if (parsed.error) {
                console.error('Backend error:', parsed.error);
                throw new Error(parsed.error);
              }

              if (parsed.content) {
                // 如果正在等待新气泡，缓存内容
                if (isWaitingForNewBubbleRef.current) {
                  pendingContentRef.current += parsed.content;
                } else {
                  // 追加文本到当前消息
                  aiContent += parsed.content;

                  // 更新最后一条消息的内容（创建新对象）
                  setMessages((prev) => {
                    const lastIndex = prev.length - 1;
                    const lastMessage = prev[lastIndex];
                    if (lastMessage && lastMessage.isStreaming) {
                      const newMessages = [...prev];
                      newMessages[lastIndex] = {
                        ...lastMessage,
                        content: aiContent,
                      };
                      return newMessages;
                    }
                    return prev;
                  });
                }
              }

              if (parsed.split) {
                // 后端也发送了 split 信号，作为双重保险
                const currentContent = aiContent.trim();
                aiContent = '';

                // 立即完成当前消息
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastIndex = prev.length - 1;
                  const lastMessage = prev[lastIndex];

                  // 完成当前消息
                  if (lastMessage && lastMessage.isStreaming) {
                    newMessages[lastIndex] = {
                      ...lastMessage,
                      content: currentContent,
                      isStreaming: false,
                    };
                  }

                  return newMessages;
                });

                // 标记正在等待新气泡
                isWaitingForNewBubbleRef.current = true;

                // 延迟添加新气泡
                setTimeout(() => {
                  const cachedContent = pendingContentRef.current;
                  pendingContentRef.current = '';
                  isWaitingForNewBubbleRef.current = false;

                  setMessages((prev) => [
                    ...prev,
                    {
                      id: `${aiMessageId}-${Date.now()}`,
                      role: 'assistant',
                      content: cachedContent,
                      isStreaming: true,
                    },
                  ]);
                }, 400);
              }
            } catch (e) {
              // 忽略解析错误
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream chat error:', error);

      // 判断是否是超时错误
      const isTimeout = error instanceof Error && (
        error.name === 'AbortError' ||
        error.message.includes('timeout') ||
        error.message.includes('aborted')
      );

      // 添加错误消息
      setMessages((prev) => {
        // 移除正在streaming的空消息
        const filtered = prev.filter(msg => msg.id !== aiMessageId || msg.content);
        return [
          ...filtered,
          {
            id: `${aiMessageId}-error`,
            role: 'assistant',
            content: isTimeout
              ? '抱歉，服务器响应超时了。\n\n可能是因为网络不稳定或服务繁忙，请稍后再试。'
              : `抱歉，我遇到了一些问题。请稍后再试。\n\n错误: ${error instanceof Error ? error.message : String(error)}`,
            isStreaming: false,
          },
        ];
      });
    } finally {
      console.log('🔄 Finally 块开始执行，时间:', Date.now());
      setIsLoading(false);
      console.log('✅ setIsLoading(false) 已调用，时间:', Date.now());
      // 不再需要在这里标记消息为完成，因为 [DONE] 已经处理了
      // 移除重复的 setMessages 调用以避免额外的渲染和卡顿
    }
  }, [sessionId, isLoading]);

  return {
    messages,
    sendMessage,
    isLoading,
  };
}
