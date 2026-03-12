'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ClassInfo {
  grade: string;
  class: string;
  student_count: number;
  session_count: number;
  risk_low: number;
  risk_medium: number;
  risk_high: number;
}

interface HighRiskStudent {
  student_id: string;
  name: string;
  grade: string;
  class: string;
  latest_session_id: string;
  risk_level: string;
  latest_time: string;
  avg_score: number | null;
}

interface TrendDataPoint {
  assessment_num: number;
  avg_score: number;
  count: number;
}

export default function TeacherDashboard() {
  const [loading, setLoading] = useState(true);
  const [role, setRole] = useState<string>('');
  const [teacherName, setTeacherName] = useState<string>('');
  const [classes, setClasses] = useState<ClassInfo[]>([]);
  const [highRiskStudents, setHighRiskStudents] = useState<HighRiskStudent[]>([]);
  const [trendData, setTrendData] = useState<TrendDataPoint[]>([]);
  const router = useRouter();

  useEffect(() => {
    const teacherId = localStorage.getItem('teacher_id');
    const teacherRole = localStorage.getItem('teacher_role');

    if (!teacherId || !teacherRole) {
      router.push('/teacher/login');
      return;
    }

    setRole(teacherRole);
    fetchClasses(teacherId);

    if (teacherRole === 'psychologist') {
      fetchHighRiskStudents(teacherId);
      fetchTrendData(teacherId);
    }
  }, [router]);

  const fetchClasses = async (teacherId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/teacher/classes?teacher_id=${teacherId}`);
      const data = await response.json();

      if (data.error) {
        alert(data.error);
        router.push('/teacher/login');
        return;
      }

      setTeacherName(data.teacher_name || '');
      setClasses(data.classes || []);

      // 班主任只有一个班级，直接跳转到班级页面
      if (data.classes && data.classes.length === 1 && data.role === 'normal') {
        const cls = data.classes[0];
        router.push(`/teacher/class?grade=${cls.grade}&class=${cls.class}`);
      }
    } catch (error) {
      console.error('Failed to fetch classes:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchHighRiskStudents = async (teacherId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/teacher/high-risk-students?teacher_id=${teacherId}`);
      const data = await response.json();
      if (!data.error) {
        setHighRiskStudents(data.students || []);
      }
    } catch (error) {
      console.error('Failed to fetch high risk students:', error);
    }
  };

  const fetchTrendData = async (teacherId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/teacher/assessment-trend?teacher_id=${teacherId}`);
      const data = await response.json();
      if (!data.error) {
        setTrendData(data.trend || []);
      }
    } catch (error) {
      console.error('Failed to fetch trend data:', error);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push('/teacher/login');
  };

  const handleClassClick = (grade: string, classNum: string) => {
    router.push(`/teacher/class?grade=${grade}&class=${classNum}`);
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
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Image src="/logo.png" alt="榛子" width={50} height={50} />
            <div>
              <h1 className="text-xl font-medium text-[#5D4E37]">教师后台</h1>
              <p className="text-sm text-[#8B7E6A]">
                {teacherName} ({role === 'psychologist' ? '心理老师' : '班主任'})
              </p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="px-4 py-2 text-sm text-[#8B7E6A] hover:text-[#5D4E37] transition-colors"
          >
            退出登录
          </button>
        </div>

        {/* 心理老师专属区域 */}
        {role === 'psychologist' && (
          <div className="space-y-4 mb-6">
            {/* 教师管理入口 */}
            <button
              onClick={() => router.push('/teacher/teachers')}
              className="w-full text-left p-4 bg-white rounded-xl border border-[#E8DFD0] hover:border-[#8B7355] hover:shadow-md transition-all"
            >
              <div className="flex items-center justify-between">
                <span className="text-lg font-medium text-[#5D4E37]">教师管理</span>
                <span className="text-sm text-[#8B7E6A]">重置密码 →</span>
              </div>
            </button>

            {/* 评估趋势图 */}
            <div className="bg-white rounded-xl border border-[#E8DFD0] p-4">
              <h3 className="text-base font-medium text-[#5D4E37] mb-4">评估趋势图（第n次评估平均分）</h3>
              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E8DFD0" />
                    <XAxis
                      dataKey="assessment_num"
                      label={{ value: '第n次评估', position: 'insideBottom', offset: -5 }}
                      stroke="#8B7E6A"
                    />
                    <YAxis
                      label={{ value: '综合分数', angle: -90, position: 'insideLeft' }}
                      stroke="#8B7E6A"
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FDF8F0', border: '1px solid #E8DFD0', borderRadius: '8px' }}
                      formatter={(value: number) => value.toFixed(2)}
                      labelFormatter={(label: number) => `第${label}次评估`}
                    />
                    <Line
                      type="monotone"
                      dataKey="avg_score"
                      stroke="#5D4E37"
                      strokeWidth={2}
                      dot={{ fill: '#8B7355', r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-12 text-[#A99D8A]">
                  暂无评估数据
                </div>
              )}
            </div>

            {/* 高危学生列表 */}
            <div className="bg-white rounded-xl border border-[#E8DFD0] p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-medium text-[#5D4E37]">高危学生列表</h3>
                <span className="text-sm text-[#8B7E6A]">{highRiskStudents.length}人</span>
              </div>
              {highRiskStudents.length > 0 ? (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {highRiskStudents.map((student) => (
                    <button
                      key={student.student_id}
                      onClick={() => router.push(`/teacher/session?session_id=${student.latest_session_id}`)}
                      className="w-full text-left p-3 bg-[#FDF8F0] rounded-lg hover:bg-[#F5EEE6] border border-[#E8DFD0] hover:border-[#8B7355] transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[#5D4E37] font-medium">{student.name}</span>
                            <span className={`px-2 py-0.5 rounded text-xs ${
                              student.risk_level === 'HIGH'
                                ? 'bg-red-100 text-red-700'
                                : 'bg-yellow-100 text-yellow-700'
                            }`}>
                              {student.risk_level === 'HIGH' ? '高风险' : '需关注'}
                            </span>
                          </div>
                          <div className="text-xs text-[#8B7E6A]">
                            {student.grade}级 {student.class}班
                            {student.avg_score && ` · 综合分数: ${student.avg_score}`}
                          </div>
                        </div>
                        <span className="text-xs text-[#8B7355]">查看对话 →</span>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-[#A99D8A]">
                  暂无高危学生
                </div>
              )}
            </div>
          </div>
        )}

        {/* 班级列表 */}
        <div className="bg-[#FDF8F0] rounded-2xl p-6 border border-[#E8DFD0]">
          <h2 className="text-lg font-medium text-[#5D4E37] mb-6">
            {role === 'psychologist' ? '所有班级' : '我的班级'}
          </h2>

          {classes.length === 0 ? (
            <div className="text-center py-12 text-[#A99D8A]">
              暂无班级数据
            </div>
          ) : (
            <div className="grid gap-4">
              {classes.map((cls, index) => (
                <button
                  key={index}
                  onClick={() => handleClassClick(cls.grade, cls.class)}
                  className="w-full text-left p-4 bg-white rounded-xl border border-[#E8DFD0] hover:border-[#8B7355] hover:shadow-md transition-all"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-lg font-medium text-[#5D4E37]">
                      {cls.grade}级 {cls.class}班
                    </span>
                    <span className="text-sm text-[#8B7E6A]">
                      {cls.student_count}人 · {cls.session_count}次会话
                    </span>
                  </div>

                  {/* 风险等级分布条 */}
                  <div className="flex gap-2">
                    <div
                      className="h-3 bg-green-400 rounded-l"
                      style={{ flex: cls.risk_low || 0.1 }}
                    />
                    <div
                      className="h-3 bg-yellow-400"
                      style={{ flex: cls.risk_medium || 0.1 }}
                    />
                    <div
                      className="h-3 bg-red-400 rounded-r"
                      style={{ flex: cls.risk_high || 0.1 }}
                    />
                  </div>

                  {/* 风险统计 */}
                  <div className="flex gap-4 mt-2 text-xs">
                    <span className="text-green-600">正常: {cls.risk_low}</span>
                    <span className="text-yellow-600">需关注: {cls.risk_medium}</span>
                    <span className="text-red-600">高风险: {cls.risk_high}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
