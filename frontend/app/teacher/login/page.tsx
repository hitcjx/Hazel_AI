'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';

export default function TeacherLogin() {
  const [teacherId, setTeacherId] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [showForgotModal, setShowForgotModal] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!teacherId.trim()) {
      setErrorMessage('请输入教师ID');
      return;
    }

    if (!password) {
      setErrorMessage('请输入密码');
      return;
    }

    setIsLoading(true);
    setErrorMessage('');

    try {
      const response = await fetch('http://localhost:8000/api/teacher/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ teacher_id: teacherId.trim(), password: password }),
      });

      if (!response.ok) {
        throw new Error('验证失败');
      }

      const data = await response.json();

      if (data.valid) {
        // 保存登录信息
        localStorage.setItem('teacher_id', data.teacher_id);
        localStorage.setItem('teacher_role', data.role);
        localStorage.setItem('teacher_name', data.name || '');
        localStorage.setItem('teacher_grade', data.grade || '');
        localStorage.setItem('teacher_class', data.class || '');

        // 跳转到仪表盘
        router.push('/teacher/dashboard');
      } else {
        setErrorMessage(data.message || '教师ID无效');
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrorMessage('网络连接失败，请检查后端服务是否启动');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      {/* 登录容器 */}
      <div className="w-full max-w-md bg-[#FDF8F0] rounded-[32px] shadow-lg p-12 flex flex-col items-center border border-[#E8DFD0]">
        {/* Logo */}
        <div className="mb-6 flex justify-center">
          <Image src="/logo.png" alt="榛子" width={100} height={100} priority />
        </div>

        {/* 标题 */}
        <h1 className="text-2xl font-medium text-[#5D4E37] mb-2">
          教师后台管理
        </h1>

        {/* 欢迎语 */}
        <p className="text-base text-[#8B7E6A] text-center max-w-md px-4 mb-8">
          请输入您的教师ID登录管理后台
        </p>

        {/* 登录表单 */}
        <form onSubmit={handleSubmit} className="w-full">
          <div className="mb-4">
            <input
              type="text"
              value={teacherId}
              onChange={(e) => setTeacherId(e.target.value)}
              placeholder="请输入教师ID (如 T001 或 P001)"
              disabled={isLoading}
              className="w-full px-6 py-4 bg-white/70 focus:bg-white rounded-full border border-[#E8DFD0]
                       text-[#5D4E37] placeholder:text-[#A99D8A]
                       outline-none focus:border-[#8B7355] transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed"
              autoFocus
            />
          </div>

          <div className="mb-4">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="请输入密码"
              disabled={isLoading}
              className="w-full px-6 py-4 bg-white/70 focus:bg-white rounded-full border border-[#E8DFD0]
                       text-[#5D4E37] placeholder:text-[#A99D8A]
                       outline-none focus:border-[#8B7355] transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>

          {/* 提示信息 */}
          <p className="text-sm text-[#A99D8A] text-center mb-6">
            普通老师以 T 开头，心理老师以 P 开头
          </p>

          {/* 错误信息 */}
          {errorMessage && (
            <p className="text-red-400 text-center mb-4 text-sm">
              {errorMessage}
            </p>
          )}

          {/* 提交按钮 */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full px-8 py-4 bg-[#8B7355] hover:bg-[#7A6548]
                     text-white rounded-full font-medium
                     transition-all duration-200
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? '登录中...' : '登录'}
          </button>
        </form>

        {/* 忘记密码链接 */}
        <button
          onClick={() => setShowForgotModal(true)}
          className="mt-4 text-sm text-[#8B7355] hover:underline"
        >
          忘记密码？
        </button>

        {/* 忘记密码弹窗 */}
        {showForgotModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[200]" onClick={() => setShowForgotModal(false)}>
            <div className="bg-white rounded-2xl p-8 max-w-sm mx-4 text-center" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-lg font-medium text-[#5D4E37] mb-4">忘记密码？</h3>
              <p className="text-[#8B7E6A] mb-6">
                请联系心理老师帮您重置密码
              </p>
              <button
                onClick={() => setShowForgotModal(false)}
                className="px-6 py-2 bg-[#8B7355] text-white rounded-full"
              >
                知道了
              </button>
            </div>
          </div>
        )}

        {/* 返回学生端链接 */}
        <div className="mt-6 text-center">
          <a
            href="/"
            className="text-sm text-[#A99D8A] hover:text-[#8B7355] transition-colors"
          >
            ← 返回学生端
          </a>
        </div>
      </div>
    </div>
  );
}
