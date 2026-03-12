'use client';

import { useState, useEffect } from 'react';

interface MethodPrompt {
  [key: string]: string;
}

export default function SFBTDebugPage() {
  const [persona, setPersona] = useState('');
  const [methods, setMethods] = useState<MethodPrompt>({});
  const [commonRules, setCommonRules] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('例外情境');
  const [userMessage, setUserMessage] = useState('');
  const [history, setHistory] = useState('');
  const [response, setResponse] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPrompts, setShowPrompts] = useState(false);

  const methodNames = ['例外情境', '奇迹问题', '应对问题', '关系问题', '量尺问题', '赞美'];

  // 加载 prompts
  useEffect(() => {
    fetch('/api/sfbt/prompts')
      .then(res => res.json())
      .then(data => {
        if (data.persona) setPersona(data.persona);
        if (data.methods) setMethods(data.methods);
        if (data.common_rules) setCommonRules(data.common_rules);
      });
  }, []);

  // 保存 prompts
  const handleSave = async () => {
    await fetch('/api/sfbt/prompts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persona, methods, common_rules: commonRules })
    });
    alert('保存成功');
  };

  // 测试调用
  const handleTest = async () => {
    setLoading(true);
    setResponse('');
    setSystemPrompt('');

    const historyLines = history.split('\n').filter(l => l.trim());
    const historyMessages = historyLines.map(line => {
      const [role, ...contentParts] = line.split(':');
      return { role: role.trim(), content: contentParts.join(':').trim() };
    });

    try {
      const res = await fetch('/api/sfbt/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method: selectedMethod,
          persona,
          user_message: userMessage,
          history: historyMessages
        })
      });
      const data = await res.json();
      if (data.error) {
        setResponse('Error: ' + data.error);
      } else {
        setSystemPrompt(data.system_prompt);
        setResponse(data.response);
      }
    } catch (e: any) {
      setResponse('Error: ' + e.message);
    }
    setLoading(false);
  };

  const handleMethodChange = (method: string, value: string) => {
    setMethods(prev => ({ ...prev, [method]: value }));
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">SFBT 干预方法调试工具</h1>
          <div className="space-x-2">
            <button
              onClick={() => setShowPrompts(!showPrompts)}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              {showPrompts ? '隐藏 Prompt' : '编辑 Prompt'}
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              保存修改
            </button>
          </div>
        </div>

        {/* Prompt 编辑区域 */}
        {showPrompts && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <div className="grid grid-cols-2 gap-4">
              {/* Persona */}
              <div>
                <label className="block font-semibold mb-2">Persona</label>
                <textarea
                  value={persona}
                  onChange={e => setPersona(e.target.value)}
                  className="w-full h-32 p-2 border rounded font-mono text-sm"
                  placeholder="输入 persona..."
                />
              </div>

              {/* Common Rules */}
              <div>
                <label className="block font-semibold mb-2">通用规则</label>
                <textarea
                  value={commonRules}
                  onChange={e => setCommonRules(e.target.value)}
                  className="w-full h-32 p-2 border rounded font-mono text-sm"
                  placeholder="输入通用规则..."
                />
              </div>
            </div>

            {/* 方法列表 */}
            <div className="mt-4">
              <label className="block font-semibold mb-2">干预方法</label>
              <div className="grid grid-cols-3 gap-4">
                {methodNames.map(method => (
                  <div key={method} className="border rounded p-2">
                    <div className="font-semibold mb-1 text-sm">{method}</div>
                    <textarea
                      value={methods[method] || ''}
                      onChange={e => handleMethodChange(method, e.target.value)}
                      className="w-full h-24 p-1 border rounded text-xs font-mono"
                      placeholder={`输入 ${method} 的 scaffold...`}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 测试区域 */}
        <div className="grid grid-cols-2 gap-6">
          {/* 左侧：输入 */}
          <div className="space-y-4">
            {/* 方法选择 */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block font-semibold mb-2">选择干预方法</label>
              <div className="flex flex-wrap gap-2">
                {methodNames.map(method => (
                  <button
                    key={method}
                    onClick={() => setSelectedMethod(method)}
                    className={`px-3 py-1 rounded ${
                      selectedMethod === method
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {method}
                  </button>
                ))}
              </div>
            </div>

            {/* 用户消息 */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block font-semibold mb-2">用户消息</label>
              <textarea
                value={userMessage}
                onChange={e => setUserMessage(e.target.value)}
                className="w-full h-24 p-2 border rounded"
                placeholder="输入测试消息..."
              />
            </div>

            {/* 历史 */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block font-semibold mb-2">对话历史 (每行: role: content)</label>
              <textarea
                value={history}
                onChange={e => setHistory(e.target.value)}
                className="w-full h-32 p-2 border rounded font-mono text-sm"
                placeholder="user: 你好&#10;assistant: 你好呀~"
              />
            </div>

            <button
              onClick={handleTest}
              disabled={loading || !userMessage}
              className="w-full py-3 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400"
            >
              {loading ? '调用中...' : '测试调用'}
            </button>
          </div>

          {/* 右侧：输出 */}
          <div className="space-y-4">
            {/* 组合后的 System Prompt */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block font-semibold mb-2">组合后的 System Prompt</label>
              <pre className="bg-gray-50 p-3 rounded text-xs overflow-auto max-h-48 whitespace-pre-wrap">
                {systemPrompt || '(点击测试后显示)'}
              </pre>
            </div>

            {/* Avatar 响应 */}
            <div className="bg-white rounded-lg shadow p-4 flex-1">
              <label className="block font-semibold mb-2">Avatar 响应</label>
              <div className="bg-gray-50 p-3 rounded min-h-[200px] whitespace-pre-wrap">
                {loading ? '调用中...' : response || '(点击测试后显示)'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
