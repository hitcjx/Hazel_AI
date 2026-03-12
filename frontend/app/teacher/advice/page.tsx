'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

interface AssessmentData {
  student_id: string;
  student_name: string;
  risk_level: string;
  turn_count: number;
  advice: {
    overall_assessment: string;
    risk_concerns?: string[];
    teacher_advice: string;
    need_referral?: boolean;
    referral_reason?: string;
    dimension_scores?: Record<string, number>;
  };
}

export default function StudentAdvice() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<AssessmentData | null>(null);
  const [error, setError] = useState('');
  const router = useRouter();
  const searchParams = useSearchParams();

  const sessionId = searchParams.get('session_id') || '';

  useEffect(() => {
    const teacherId = localStorage.getItem('teacher_id');
    const teacherRole = localStorage.getItem('teacher_role');

    if (!teacherId || !teacherRole) {
      router.push('/teacher/login');
      return;
    }

    if (!sessionId) {
      router.push('/teacher/dashboard');
      return;
    }

    fetchAdvice(teacherRole);
  }, [router, sessionId]);

  const fetchAdvice = async (teacherRole: string) => {
    const teacherId = localStorage.getItem('teacher_id');
    try {
      const response = await fetch(
        `http://localhost:8000/api/teacher/assessment-advice?session_id=${sessionId}&teacher_id=${teacherId}&teacher_role=${teacherRole}`
      );
      const result = await response.json();

      if (response.status === 403) {
        setError(result.detail || '无权查看此学生的评估');
        return;
      }

      if (result.error) {
        setError(result.error);
        return;
      }

      setData(result);
    } catch (error) {
      console.error('Failed to fetch advice:', error);
      setError('获取建议失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadge = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">正常</span>;
      case 'MEDIUM':
        return <span className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm">需关注</span>;
      case 'HIGH':
        return <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm">高风险</span>;
      default:
        return <span className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">未知</span>;
    }
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
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <button
            onClick={() => router.back()}
            className="px-4 py-2 bg-[#8B7355] text-white rounded-lg"
          >
            返回
          </button>
        </div>
      </div>
    );
  }

  const role = localStorage.getItem('teacher_role');

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="max-w-3xl mx-auto">
        {/* 顶部导航 */}
        <div className="flex items-center gap-4 mb-6">
          <button
            onClick={() => router.back()}
            className="p-2 hover:bg-[#F5EEE6] rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 text-[#8B7355]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <h1 className="text-xl font-medium text-[#5D4E37]">学生评估详情</h1>
        </div>

        {/* 学生基本信息 */}
        <div className="bg-[#FDF8F0] rounded-2xl p-6 border border-[#E8DFD0] mb-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[#8B7E6A] text-sm">学号</p>
              <p className="text-[#5D4E37] font-medium">{data?.student_id}</p>
            </div>
            <div>
              <p className="text-[#8B7E6A] text-sm">姓名</p>
              <p className="text-[#5D4E37] font-medium">{data?.student_name || '-'}</p>
            </div>
            <div>
              <p className="text-[#8B7E6A] text-sm">对话轮次</p>
              <p className="text-[#5D4E37] font-medium">{data?.turn_count || 0}</p>
            </div>
            <div>
              <p className="text-[#8B7E6A] text-sm">心理健康状态</p>
              {getRiskBadge(data?.risk_level || 'LOW')}
            </div>
          </div>
        </div>

        {/* 心理老师可见：评估分数 */}
        {role === 'psychologist' && data?.advice?.dimension_scores && (
          <div className="bg-[#FDF8F0] rounded-2xl p-6 border border-[#E8DFD0] mb-6">
            <h2 className="text-lg font-medium text-[#5D4E37] mb-4">评估分数</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(data.advice.dimension_scores).map(([dimension, score]) => (
                <div key={dimension} className="bg-white rounded-lg p-3 text-center">
                  <p className="text-xs text-[#8B7E6A] mb-1">{dimension}</p>
                  <p className="text-xl font-medium text-[#5D4E37]">{score?.toFixed(1) || '-'}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* AI 评估建议 */}
        <div className="bg-[#FDF8F0] rounded-2xl p-6 border border-[#E8DFD0]">
          <h2 className="text-lg font-medium text-[#5D4E37] mb-4">🤖 AI 评估建议</h2>

          {/* 总体评估 */}
          <div className="mb-6">
            <h3 className="text-[#8B7E6A] text-sm font-medium mb-2">总体评估</h3>
            <p className="text-[#5D4E37] leading-relaxed">
              {data?.advice?.overall_assessment || '暂无评估'}
            </p>
          </div>

          {/* 风险关注点 */}
          {data?.advice?.risk_concerns && data.advice.risk_concerns.length > 0 && (
            <div className="mb-6">
              <h3 className="text-[#8B7E6A] text-sm font-medium mb-2">⚠️ 需要关注的风险点</h3>
              <ul className="space-y-2">
                {data.advice.risk_concerns.map((concern, index) => (
                  <li key={index} className="flex items-start gap-2 text-[#5D4E37]">
                    <span className="text-yellow-500">•</span>
                    <span>{concern}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 对老师的建议 */}
          <div className="mb-6">
            <h3 className="text-[#8B7E6A] text-sm font-medium mb-2">💡 建议</h3>
            <p className="text-[#5D4E37] leading-relaxed">
              {data?.advice?.teacher_advice || '暂无建议'}
            </p>
          </div>

          {/* 是否需要转介 */}
          {data?.advice?.need_referral && (
            <div className="bg-red-50 rounded-lg p-4 border border-red-100">
              <h3 className="text-red-700 font-medium mb-2">🔄 需要转介心理老师</h3>
              <p className="text-red-600 text-sm">{data.advice.referral_reason}</p>
            </div>
          )}
        </div>

        {/* 提示信息 */}
        {role === 'normal' && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700">
            💡 此为AI建议，普通老师无权查看具体评分和聊天记录。如需更详细信息，请联系心理老师。
          </div>
        )}
      </div>
    </div>
  );
}
