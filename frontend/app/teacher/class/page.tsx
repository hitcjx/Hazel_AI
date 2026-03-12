'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Image from 'next/image';
import { PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface Student {
  student_id: string;
  name: string;
  session_id: string | null;
  risk_level: string;
  turn_count: number;
  has_session: boolean;
  final_score: number | null;
}

interface ClassStats {
  risk_distribution: { name: string; value: number; color: string }[];
  session_trend: { order: number; time: string; score: number }[];
  dimension_scores: { dimension: string; score: number }[];
}

interface PasswordResetRequest {
  id: number;
  user_id: string;
  user_type: string;
  user_name: string;
  status: string;
  created_at: string;
}

export default function ClassDetail() {
  const [loading, setLoading] = useState(true);
  const [role, setRole] = useState<string>('');
  const [students, setStudents] = useState<Student[]>([]);
  const [stats, setStats] = useState<ClassStats | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [resetRequests, setResetRequests] = useState<PasswordResetRequest[]>([]);
  const router = useRouter();
  const searchParams = useSearchParams();

  const grade = searchParams.get('grade') || '';
  const classNum = searchParams.get('class') || '';

  useEffect(() => {
    const teacherId = localStorage.getItem('teacher_id');
    const teacherRole = localStorage.getItem('teacher_role');

    if (!teacherId || !teacherRole) {
      router.push('/teacher/login');
      return;
    }

    setRole(teacherRole);
    fetchStudents(teacherId);
    fetchStats(teacherId);
    fetchResetRequests(teacherId);
  }, [router, grade, classNum]);

  const fetchStudents = async (teacherId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/teacher/students?teacher_id=${teacherId}&grade=${grade}&class_num=${classNum}`
      );
      const data = await response.json();

      if (data.error) {
        alert(data.error);
        router.push('/teacher/dashboard');
        return;
      }

      setStudents(data.students || []);
    } catch (error) {
      console.error('Failed to fetch students:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async (teacherId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/teacher/class-stats?teacher_id=${teacherId}&grade=${grade}&class_num=${classNum}`
      );
      const data = await response.json();
      if (!data.error) {
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const fetchResetRequests = async (teacherId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/teacher/password-reset-requests?teacher_id=${teacherId}`
      );
      const data = await response.json();
      if (!data.error) {
        setResetRequests(data.requests || []);
      }
    } catch (error) {
      console.error('Failed to fetch reset requests:', error);
    }
  };

  const handleApproveReset = async (requestId: number) => {
    const teacherId = localStorage.getItem('teacher_id');
    if (!teacherId) return;

    if (!confirm('确定要批准该密码重置申请吗？学生的密码将被重置为学号。')) return;

    try {
      const res = await fetch('http://localhost:8000/api/teacher/approve-password-reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teacher_id: teacherId, request_id: requestId })
      });
      const data = await res.json();
      if (data.success) {
        alert('密码已重置');
        fetchResetRequests(teacherId); // 刷新列表
      } else {
        alert(data.message || '操作失败');
      }
    } catch (err) {
      alert('网络错误');
    }
  };

  const handleStudentClick = (student: Student) => {
    // 只有黄/红风险等级才能点击查看建议
    if (student.risk_level === 'MEDIUM' || student.risk_level === 'HIGH') {
      router.push(`/teacher/advice?session_id=${student.session_id}`);
    }
  };

  // 过滤后的学生列表
  const filteredStudents = students.filter((student) => {
    // 搜索过滤（学号或姓名）
    const matchesSearch = searchQuery === '' ||
      student.student_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      student.name.toLowerCase().includes(searchQuery.toLowerCase());
    // 风险等级过滤
    const matchesRisk = riskFilter === 'all' || student.risk_level === riskFilter;
    return matchesSearch && matchesRisk;
  });

  const getRiskBadge = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs">正常</span>;
      case 'MEDIUM':
        return <span className="px-2 py-1 bg-yellow-100 text-yellow-700 rounded text-xs">需关注</span>;
      case 'HIGH':
        return <span className="px-2 py-1 bg-red-100 text-red-700 rounded text-xs">高风险</span>;
      default:
        return <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">无数据</span>;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-[#8B7355]">加载中...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* 顶部导航 */}
        <div className="flex items-center gap-4 mb-6">
          {role === 'psychologist' && (
            <button
              onClick={() => router.push('/teacher/dashboard')}
              className="p-2 hover:bg-[#F5EEE6] rounded-lg transition-colors"
            >
              <svg className="w-5 h-5 text-[#8B7355]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}
          <div>
            <h1 className="text-xl font-medium text-[#5D4E37]">
              {grade}级 {classNum}班
            </h1>
            <p className="text-sm text-[#8B7E6A]">
              {searchQuery || riskFilter !== 'all'
                ? `筛选结果: ${filteredStudents.length}/${students.length} 人`
                : `共 ${students.length} 名学生`}
            </p>
          </div>
        </div>

        {/* 密码重置申请列表 */}
        {resetRequests.length > 0 && (
          <div className="bg-orange-50 rounded-2xl p-4 border border-orange-200 mb-6">
            <h3 className="text-lg font-medium text-orange-800 mb-3">密码重置申请</h3>
            <div className="space-y-2">
              {resetRequests.map((req) => (
                <div key={req.id} className="flex items-center justify-between bg-white rounded-lg p-3">
                  <div>
                    <span className="text-[#5D4E37] font-medium">{req.user_name || req.user_id}</span>
                    <span className="text-[#8B7E6A] text-sm ml-2">({req.user_id})</span>
                    <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                      req.user_type === 'teacher' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
                    }`}>
                      {req.user_type === 'teacher' ? '老师' : '学生'}
                    </span>
                  </div>
                  <button
                    onClick={() => handleApproveReset(req.id)}
                    className="px-3 py-1 bg-orange-500 text-white rounded-lg text-sm hover:bg-orange-600"
                  >
                    批准重置
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 统计图表 */}
        {stats && stats.session_trend.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* 饼图 - 风险分布 */}
            <div className="bg-white rounded-2xl p-6 border border-[#E8DFD0] flex flex-col items-center">
              <h3 className="text-lg font-medium text-[#5D4E37] mb-4 w-full">风险等级分布</h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={stats.risk_distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                    onClick={(_, index) => {
                      const name = stats.risk_distribution[index].name;
                      const riskMap: Record<string, string> = {
                        '正常': 'LOW',
                        '需关注': 'MEDIUM',
                        '高风险': 'HIGH'
                      };
                      const riskLevel = riskMap[name];
                      if (riskLevel) {
                        setRiskFilter(riskLevel);
                        setSearchQuery('');
                      }
                    }}
                    style={{ cursor: 'pointer' }}
                    label={({ name, value }) => value > 0 ? `${name}: ${value}` : ''}
                  >
                    {stats.risk_distribution.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.color}
                        stroke={riskFilter === ['LOW', 'MEDIUM', 'HIGH'][index] ? '#5D4E37' : undefined}
                        strokeWidth={riskFilter !== 'all' ? 2 : 0}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* 折线图 - 会话趋势 */}
            <div className="bg-white rounded-2xl p-6 border border-[#E8DFD0] flex flex-col items-center">
              <h3 className="text-lg font-medium text-[#5D4E37] mb-4 w-full">会话趋势</h3>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={stats.session_trend} margin={{ top: 20, right: 10, bottom: 20, left: 0 }}>
                  <defs>
                    <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="order"
                    tick={{ fontSize: 12 }}
                    label={{ value: '会话次数', position: 'bottom', offset: 5, fontSize: 12 }}
                    interval={0}
                  />
                  <YAxis
                    domain={[0, 5]}
                    ticks={[0, 1, 2, 3, 4, 5]}
                    tick={{ fontSize: 12 }}
                    width={35}
                  />
                  <CartesianGrid strokeDasharray="3 3" stroke="#E8DFD0" horizontal vertical />
                  <Tooltip
                    formatter={(value: number, name: string, props: any) => [
                      `第${props.payload.order}次 - ${props.payload.time}`,
                      `评分: ${props.payload.score.toFixed(1)}`
                    ]}
                  />
                  <Area
                    type="monotone"
                    dataKey="score"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    fill="url(#scoreGradient)"
                    dot={{ fill: '#F59E0B', r: 5 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 学生列表 */}
        <div className="bg-[#FDF8F0] rounded-2xl p-6 border border-[#E8DFD0]">
          {/* 搜索和筛选 */}
          <div className="flex gap-3 mb-4">
            <div className="flex-1 relative">
              <input
                type="text"
                placeholder="搜索学号或姓名..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-4 py-2 pl-10 border border-[#E8DFD0] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B7355] text-sm"
              />
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8B7E6A]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
              className="px-4 py-2 border border-[#E8DFD0] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#8B7355] text-sm text-[#5D4E37] bg-white"
            >
              <option value="all">全部</option>
              <option value="LOW">正常</option>
              <option value="MEDIUM">需关注</option>
              <option value="HIGH">高风险</option>
            </select>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-[#8B7E6A] border-b border-[#E8DFD0]">
                  <th className="pb-3 font-medium">学号</th>
                  <th className="pb-3 font-medium">姓名</th>
                  <th className="pb-3 font-medium">状态</th>
                  {role === 'psychologist' && (
                    <th className="pb-3 font-medium">评估分数</th>
                  )}
                  <th className="pb-3 font-medium">对话轮次</th>
                  <th className="pb-3 font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredStudents.map((student) => (
                  <tr
                    key={student.student_id}
                    className={`border-b border-[#E8DFD0] last:border-0 ${
                      (student.risk_level === 'MEDIUM' || student.risk_level === 'HIGH')
                        ? 'bg-yellow-50 hover:bg-yellow-100 cursor-pointer'
                        : 'hover:bg-[#F5EEE6]'
                    } transition-colors`}
                    onClick={() => handleStudentClick(student)}
                  >
                    <td className="py-3 text-[#5D4E37]">{student.student_id}</td>
                    <td className="py-3 text-[#5D4E37]">{student.name}</td>
                    <td className="py-3">
                      {getRiskBadge(student.risk_level)}
                    </td>
                    {role === 'psychologist' && (
                      <td className="py-3 text-[#5D4E37]">
                        {student.final_score !== null ? student.final_score.toFixed(1) : '-'}
                      </td>
                    )}
                    <td className="py-3 text-[#8B7E6A]">
                      {student.has_session ? student.turn_count : '-'}
                    </td>
                    <td className="py-3">
                      <button
                        onClick={async (e) => {
                          e.stopPropagation();
                          if (!confirm(`确定要重置 ${student.name} 的密码吗？`)) return;
                          try {
                            const res = await fetch('http://localhost:8000/api/teacher/reset-student-password', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({
                                teacher_id: localStorage.getItem('teacher_id'),
                                student_id: student.student_id
                              })
                            });
                            const data = await res.json();
                            if (data.success) {
                              alert('密码已重置为学号');
                            } else {
                              alert(data.message || '重置失败');
                            }
                          } catch (err) {
                            alert('网络错误');
                          }
                        }}
                        className="text-xs text-[#8B7355] hover:underline"
                      >
                        重置密码
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {students.length === 0 ? (
            <div className="text-center py-12 text-[#A99D8A]">
              暂无学生数据
            </div>
          ) : filteredStudents.length === 0 ? (
            <div className="text-center py-12 text-[#A99D8A]">
              没有匹配的学生
            </div>
          ) : null}

          {/* 提示信息 */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700">
            💡 点击<span className="font-medium">黄色/红色</span>行的学生可查看AI建议
            {role === 'psychologist' ? '或详细报告' : ''}
          </div>
        </div>
      </div>
    </div>
  );
}
