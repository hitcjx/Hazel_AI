'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Image from 'next/image';

interface Message {
  role: string;
  content: string;
  timestamp: string;
}

interface StudentInfo {
  student_id: string;
  student_name: string;
  risk_level: string;
}

export default function SessionDetailPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('session_id');

  const [messages, setMessages] = useState<Message[]>([]);
  const [studentInfo, setStudentInfo] = useState<StudentInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const teacherRole = localStorage.getItem('teacher_role');
    if (!sessionId || teacherRole !== 'psychologist') {
      router.push('/teacher/dashboard');
      return;
    }

    fetchMessages(sessionId, teacherRole);
  }, [sessionId, router]);

  const fetchMessages = async (sessionId: string, teacherRole: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/teacher/session-messages?session_id=${sessionId}&teacher_role=${teacherRole}`
      );
      const data = await response.json();

      if (data.error || data.detail) {
        setError(data.error || data.detail);
        setTimeout(() => router.push('/teacher/dashboard'), 3000);
        return;
      }

      setStudentInfo({
        student_id: data.student_id,
        student_name: data.student_name,
        risk_level: data.risk_level
      });
      setMessages(data.messages || []);
    } catch (err) {
      setError('网络错误，无法加载对话记录');
      setTimeout(() => router.push('/teacher/dashboard'), 3000);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-[#8B7355]">加载中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 mb-4">{error}</div>
          <div className="text-[#8B7E6A]">正在返回...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="max-w-3xl mx-auto">
        {/* 顶部导航 */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.back()}
              className="text-[#8B7355] hover:text-[#5D4E37] transition-colors"
            >
              ← 返回
            </button>
            <Image src="/logo.png" alt="榛子" width={40} height={40} />
            <div>
              <h1 className="text-lg font-medium text-[#5D4E37]">对话记录</h1>
              <p className="text-xs text-[#8B7E6A]">心理老师后台</p>
            </div>
          </div>
        </div>

        {/* 学生信息 */}
        {studentInfo && (
          <div className="bg-white rounded-xl border border-[#E8DFD0] p-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div>
                  <div className="text-lg font-medium text-[#5D4E37]">{studentInfo.student_name}</div>
                  <div className="text-sm text-[#8B7E6A]">学号: {studentInfo.student_id}</div>
                </div>
                <span className={`px-3 py-1 rounded-lg text-sm font-medium ${
                  studentInfo.risk_level === 'HIGH'
                    ? 'bg-red-100 text-red-700'
                    : studentInfo.risk_level === 'MEDIUM'
                    ? 'bg-yellow-100 text-yellow-700'
                    : 'bg-green-100 text-green-700'
                }`}>
                  {studentInfo.risk_level === 'HIGH' ? '高风险' :
                   studentInfo.risk_level === 'MEDIUM' ? '需关注' : '正常'}
                </span>
              </div>
              <div className="text-xs text-[#8B7E6A]">
                会话ID: {sessionId?.slice(0, 8)}...
              </div>
            </div>
          </div>
        )}

        {/* 对话记录 */}
        <div className="bg-white rounded-xl border border-[#E8DFD0] p-6">
          <h2 className="text-base font-medium text-[#5D4E37] mb-4">对话内容</h2>

          {messages.length === 0 ? (
            <div className="text-center py-12 text-[#A99D8A]">
              暂无对话记录
            </div>
          ) : (
            <div className="space-y-4 max-h-[600px] overflow-y-auto">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      message.role === 'user'
                        ? 'bg-[#5D4E37] text-white'
                        : 'bg-[#FDF8F0] text-[#5D4E37] border border-[#E8DFD0]'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs opacity-70">
                        {message.role === 'user' ? '学生' : '榛子'}
                      </span>
                      <span className="text-xs opacity-50">
                        {formatTime(message.timestamp)}
                      </span>
                    </div>
                    <div className="text-sm whitespace-pre-wrap leading-relaxed">
                      {message.content}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
