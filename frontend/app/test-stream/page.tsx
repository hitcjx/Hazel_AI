'use client';

import { useState } from 'react';

export default function TestStreamPage() {
  const [logs, setLogs] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const addLog = (log: string) => {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    setLogs(prev => [...prev, `[${timestamp}] ${log}`]);
    console.log(`[${timestamp}]`, log);
  };

  const testStream = async () => {
    setIsLoading(true);
    addLog('🚀 开始测试流式请求...');

    try {
      addLog('📡 发送POST请求到 http://localhost:8000/api/chat/stream');

      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: '你好' }],
          session_id: 'test-session',
        }),
      });

      addLog(`📊 响应状态: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      addLog('✅ 响应成功，开始读取流...');

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let chunkCount = 0;
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          addLog(`🏁 流结束，总共收到 ${chunkCount} 个数据块`);
          addLog(`📝 完整文本: "${fullText}"`);
          break;
        }

        chunkCount++;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            if (data === '[DONE]') {
              addLog('✅ 收到 [DONE] 信号');
              break;
            }

            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                fullText += parsed.content;
                if (chunkCount % 5 === 0) {
                  addLog(`📝 进度 (#${chunkCount}): "${fullText}"`);
                }
              }
              if (parsed.error) {
                addLog(`❌ 后端错误: ${parsed.error}`);
              }
            } catch (e) {
              // 忽略解析错误
            }
          }
        }
      }

      addLog('✅ 测试完成！');

    } catch (error) {
      addLog(`❌ 错误: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">流式API测试页面</h1>

        <button
          onClick={testStream}
          disabled={isLoading}
          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed mb-6"
        >
          {isLoading ? '测试中...' : '开始测试'}
        </button>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">日志输出</h2>
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-96 overflow-y-auto">
            {logs.length === 0 ? (
              <p className="text-gray-500">点击"开始测试"按钮开始测试...</p>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="mb-1">{log}</div>
              ))
            )}
          </div>
        </div>

        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold mb-2">💡 说明</h3>
          <p className="text-sm text-gray-700">
            这个页面会直接调用后端的流式API，并在上面显示详细的日志输出。
            如果这个测试页面能正常工作，说明后端API是正常的，问题可能出在主聊天页面的逻辑上。
          </p>
        </div>
      </div>
    </div>
  );
}
