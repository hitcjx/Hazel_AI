'use client';

import { useState } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';

type LoginRole = 'student' | 'teacher';

interface LoginScreenProps {
  onLoginSuccess?: (studentId: string, name: string, sessionId: string) => void;
}

export function LoginScreen({ onLoginSuccess }: LoginScreenProps) {
  const router = useRouter();
  const [role, setRole] = useState<LoginRole>('student');
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [showForgotModal, setShowForgotModal] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!id.trim()) {
      setErrorMessage(role === 'student' ? '请输入学号' : '请输入教师ID');
      return;
    }

    if (!password) {
      setErrorMessage('请输入密码');
      return;
    }

    setIsLoading(true);
    setErrorMessage('');

    try {
      if (role === 'student') {
        // 学生登录
        const response = await fetch('http://localhost:8000/api/auth/login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ student_id: id.trim(), password: password }),
        });

        const data = await response.json();

        if (data.success) {
          localStorage.setItem('hazel_student_id', data.student_id);
          localStorage.setItem('hazel_student_name', data.name || '');
          localStorage.setItem('hazel_session_id', data.session_id);
          onLoginSuccess?.(data.student_id, data.name || '', data.session_id);
        } else {
          setErrorMessage(data.message || '登录失败');
        }
      } else {
        // 教师登录
        const response = await fetch('http://localhost:8000/api/teacher/verify', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ teacher_id: id.trim(), password: password }),
        });

        const data = await response.json();

        if (data.valid) {
          localStorage.setItem('teacher_id', data.teacher_id);
          localStorage.setItem('teacher_role', data.role);
          localStorage.setItem('teacher_name', data.name || '');
          localStorage.setItem('teacher_grade', data.grade || '');
          localStorage.setItem('teacher_class', data.class || '');
          router.push('/teacher/dashboard');
        } else {
          setErrorMessage(data.message || '登录失败');
        }
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrorMessage('网络连接失败，请检查后端服务是否启动');
    } finally {
      setIsLoading(false);
    }
  };

  // 根据角色选择样式
  const isStudent = role === 'student';

  const styles = {
    container: isStudent
      ? 'bg-[url("/background3.webp")] bg-cover bg-center'
      : 'bg-[url("/background4.webp")] bg-cover bg-center',
    card: isStudent
      ? 'bg-paper-surface shadow-paper'
      : 'bg-[#FDF8F0] shadow-lg',
    text: isStudent ? 'text-paper-text' : 'text-[#5D4E37]',
    muted: isStudent ? 'text-paper-muted' : 'text-[#8B7E6A]',
    border: 'border-[#E8DFD0]',
    focus: isStudent ? 'focus:border-paper-ai' : 'focus:border-[#8B7355]',
    button: isStudent
      ? 'bg-paper-ai hover:bg-opacity-80 shadow-morandi'
      : 'bg-[#8B7355] hover:bg-[#7A6548]',
    buttonText: isStudent ? 'text-paper-text' : 'text-white',
    input: isStudent
      ? 'bg-white/50 text-paper-text placeholder:text-paper-muted shadow-md hover:shadow-lg focus:shadow-lg'
      : 'bg-white/70 text-[#5D4E37] placeholder:text-[#A99D8A] shadow-md hover:shadow-lg focus:shadow-lg',
    toggleActive: isStudent
      ? 'bg-paper-ai text-paper-text'
      : 'bg-[#8B7355] text-white',
    toggleInactive: isStudent
      ? 'text-paper-muted hover:text-paper-text'
      : 'text-[#8B7E6A] hover:text-[#5D4E37]',
    error: isStudent ? 'text-red-300/80' : 'text-red-400',
  };

  return (
    <div className={`fixed inset-0 flex flex-col items-center justify-center z-[100] px-4 ${styles.container}`}>
      {/* 悬浮纸张容器 */}
      <div className={`w-full max-w-md rounded-[32px] p-12 flex flex-col items-center ${styles.card}`}>
        {/* Logo */}
        <div className="mb-6 animate-breathe flex justify-center">
          <Image src="/logo.png" alt="榛子" width={100} height={100} priority />
        </div>

        {/* 角色切换 */}
        <div className={`flex mb-6 ${isStudent ? 'bg-white/30' : 'bg-[#E8DFD0]'} rounded-full p-1`}>
          <button
            type="button"
            onClick={() => { setRole('student'); setId(''); setErrorMessage(''); }}
            className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${
              isStudent ? styles.toggleActive : styles.toggleInactive
            }`}
          >
            我是学生
          </button>
          <button
            type="button"
            onClick={() => { setRole('teacher'); setId(''); setErrorMessage(''); }}
            className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${
              !isStudent ? styles.toggleActive : styles.toggleInactive
            }`}
          >
            我是老师
          </button>
        </div>

        {/* 欢迎语 */}
        <p className={`text-lg font-medium text-center max-w-md px-4 leading-relaxed mb-8 ${styles.text}`}>
          {isStudent
            ? '你好呀！我是榛子。这里是你的专属小树洞，让我们开始今天的对话吧。'
            : "老师您好，我是您的心理助教'榛子'。\n让我们一起关注孩子们的成长点滴。"}
        </p>

        {/* 登录表单 */}
        <form onSubmit={handleSubmit} className="w-full">
          <div className="mb-4">
            <input
              type="text"
              value={id}
              onChange={(e) => setId(e.target.value)}
              placeholder={isStudent ? '请输入学号' : '请输入教师ID (如 T001)'}
              disabled={isLoading}
              className={`w-full px-6 py-4 rounded-full ${styles.input}
                       ${styles.border} ${styles.focus}
                       outline-none transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed`}
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
              className={`w-full px-6 py-4 rounded-full ${styles.input}
                       ${styles.border} ${styles.focus}
                       outline-none transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed`}
            />
          </div>

          {/* 错误信息 */}
          {errorMessage && (
            <p className={`text-center mb-4 text-sm ${styles.error}`}>
              {errorMessage}
            </p>
          )}

          {/* 提交按钮 */}
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full px-8 py-4 rounded-full font-medium
                     transition-all duration-200
                     disabled:opacity-50 disabled:cursor-not-allowed
                     ${styles.button} ${styles.buttonText}
                     ${isStudent ? 'hover:shadow-lg' : 'shadow-md hover:shadow-lg'}`}
          >
            {isLoading ? '登录中...' : (isStudent ? '开始对话' : '登录')}
          </button>
        </form>

        {/* 忘记密码链接 */}
        <button
          type="button"
          onClick={() => setShowForgotModal(true)}
          className={`mt-4 text-sm underline ${styles.muted} hover:${styles.text}`}
        >
          忘记密码？
        </button>

        {/* 忘记密码弹窗 */}
        {showForgotModal && (
          <ForgotPasswordModal onClose={() => setShowForgotModal(false)} styles={styles} isStudent={isStudent} />
        )}
      </div>
    </div>
  );
}

// 忘记密码弹窗组件
function ForgotPasswordModal({ onClose, styles, isStudent }: { onClose: () => void; styles: any; isStudent: boolean }) {
  const [userId, setUserId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!userId.trim()) {
      setMessage(isStudent ? '请输入学号' : '请输入教师ID');
      return;
    }

    setIsLoading(true);
    setMessage('');

    try {
      const res = await fetch('http://localhost:8000/api/auth/request-password-reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId.trim(),
          user_type: isStudent ? 'student' : 'teacher'
        })
      });
      const data = await res.json();

      if (data.success) {
        setSuccess(true);
        setMessage(isStudent ? '申请已提交，请等待老师审核' : '申请已提交，请等待心理老师审核');
      } else {
        setMessage(data.message || '提交失败');
      }
    } catch (err) {
      setMessage('网络错误，请稍后重试');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[200]" onClick={onClose}>
      <div className="bg-white rounded-2xl p-8 max-w-sm mx-4" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-medium text-[#5D4E37] mb-4 text-center">忘记密码</h3>

        {!success ? (
          <>
            <p className="text-[#8B7E6A] text-sm mb-4 text-center">
              {isStudent
                ? '请输入您的学号，老师审核后将为您重置密码'
                : '请输入您的教师ID，心理老师审核后将为您重置密码'}
            </p>
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder={isStudent ? '请输入学号' : '请输入教师ID'}
                disabled={isLoading}
                className={`w-full px-4 py-3 rounded-lg mb-4
                         border ${styles.border} ${styles.focus}
                         outline-none transition-all text-[#5D4E37]
                         disabled:opacity-50`}
                autoFocus
              />

              {message && (
                <p className={`text-sm text-center mb-4 ${success ? 'text-green-600' : 'text-red-500'}`}>
                  {message}
                </p>
              )}

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="flex-1 px-4 py-2 border border-[#E8DFD0] text-[#8B7E6A] rounded-lg hover:bg-[#F5EEE6]"
                  disabled={isLoading}
                >
                  取消
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className={`flex-1 px-4 py-2 rounded-lg ${styles.button} ${styles.buttonText}
                           disabled:opacity-50`}
                >
                  {isLoading ? '提交中...' : '提交申请'}
                </button>
              </div>
            </form>
          </>
        ) : (
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-green-600 mb-4">{message}</p>
            <button
              onClick={onClose}
              className="px-6 py-2 bg-[#8B7355] text-white rounded-full"
            >
              知道了
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
