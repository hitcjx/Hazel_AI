'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';

interface Teacher {
  teacher_id: string;
  name: string;
  role: string;
  grade: string;
  class: string;
  is_active: number;
}

interface PasswordResetRequest {
  id: number;
  user_id: string;
  user_type: string;
  user_name: string;
  status: string;
  created_at: string;
}

export default function TeacherManagement() {
  const [loading, setLoading] = useState(true);
  const [role, setRole] = useState<string>('');
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [resetRequests, setResetRequests] = useState<PasswordResetRequest[]>([]);
  const router = useRouter();

  useEffect(() => {
    const teacherId = localStorage.getItem('teacher_id');
    const teacherRole = localStorage.getItem('teacher_role');

    if (!teacherId || !teacherRole) {
      router.push('/teacher/login');
      return;
    }

    // 只有心理老师可以访问
    if (teacherRole !== 'psychologist') {
      router.push('/teacher/dashboard');
      return;
    }

    setRole(teacherRole);
    fetchTeachers(teacherId);
    fetchResetRequests(teacherId);
  }, [router]);

  const fetchTeachers = async (teacherId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/teacher/teachers?teacher_id=${teacherId}`);
      const data = await response.json();

      if (data.error) {
        alert(data.error);
        router.push('/teacher/dashboard');
        return;
      }

      setTeachers(data.teachers || []);
    } catch (error) {
      console.error('Failed to fetch teachers:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (targetTeacherId: string) => {
    const teacherId = localStorage.getItem('teacher_id');
    if (!confirm(`确定要重置 ${targetTeacherId} 的密码吗？`)) return;

    try {
      const res = await fetch('http://localhost:8000/api/teacher/reset-teacher-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          admin_teacher_id: teacherId,
          target_teacher_id: targetTeacherId
        })
      });
      const data = await res.json();
      if (data.success) {
        alert('密码已重置为教师ID');
      } else {
        alert(data.message || '重置失败');
      }
    } catch (err) {
      alert('网络错误');
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

    if (!confirm('确定要批准该密码重置申请吗？')) return;

    try {
      const res = await fetch('http://localhost:8000/api/teacher/approve-password-reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teacher_id: teacherId, request_id: requestId })
      });
      const data = await res.json();
      if (data.success) {
        alert('密码已重置');
        fetchResetRequests(teacherId);
      } else {
        alert(data.message || '操作失败');
      }
    } catch (err) {
      alert('网络错误');
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push('/teacher/login');
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
              <h1 className="text-xl font-medium text-[#5D4E37]">教师管理</h1>
              <p className="text-sm text-[#8B7E6A]">心理老师后台</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => router.push('/teacher/dashboard')}
              className="px-4 py-2 text-sm text-[#8B7355] hover:text-[#5D4E37]"
            >
              返回班级
            </button>
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-sm text-[#8B7E6A] hover:text-[#5D4E37]"
            >
              退出登录
            </button>
          </div>
        </div>

        {/* 教师列表 */}
        <div className="bg-[#FDF8F0] rounded-2xl p-6 border border-[#E8DFD0]">
          <h2 className="text-lg font-medium text-[#5D4E37] mb-6">所有教师</h2>

          {teachers.length === 0 ? (
            <div className="text-center py-12 text-[#A99D8A]">
              暂无教师数据
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-[#8B7E6A] border-b border-[#E8DFD0]">
                    <th className="pb-3 font-medium">教师ID</th>
                    <th className="pb-3 font-medium">姓名</th>
                    <th className="pb-3 font-medium">角色</th>
                    <th className="pb-3 font-medium">年级</th>
                    <th className="pb-3 font-medium">班级</th>
                    <th className="pb-3 font-medium">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {teachers.map((teacher) => (
                    <tr key={teacher.teacher_id} className="border-b border-[#E8DFD0] last:border-0 hover:bg-[#F5EEE6]">
                      <td className="py-3 text-[#5D4E37]">{teacher.teacher_id}</td>
                      <td className="py-3 text-[#5D4E37]">{teacher.name || '-'}</td>
                      <td className="py-3">
                        <span className={`px-2 py-1 rounded text-xs ${
                          teacher.role === 'psychologist'
                            ? 'bg-purple-100 text-purple-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}>
                          {teacher.role === 'psychologist' ? '心理老师' : '班主任'}
                        </span>
                      </td>
                      <td className="py-3 text-[#5D4E37]">{teacher.grade || '-'}</td>
                      <td className="py-3 text-[#5D4E37]">{teacher.class || '-'}</td>
                      <td className="py-3">
                        {teacher.role !== 'psychologist' && (
                          <button
                            onClick={() => handleResetPassword(teacher.teacher_id)}
                            className="text-xs text-[#8B7355] hover:underline"
                          >
                            重置密码
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* 密码重置申请列表 */}
        {resetRequests.length > 0 && (
          <div className="bg-orange-50 rounded-2xl p-6 border border-orange-200 mt-6">
            <h3 className="text-lg font-medium text-orange-800 mb-4">密码重置申请</h3>
            <div className="space-y-3">
              {resetRequests.map((req) => (
                <div key={req.id} className="flex items-center justify-between bg-white rounded-lg p-4">
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
                    className="px-4 py-2 bg-orange-500 text-white rounded-lg text-sm hover:bg-orange-600"
                  >
                    批准重置
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
