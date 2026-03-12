'use client';

import { useState, useEffect } from 'react';
import { WelcomeScreen } from './components/welcome-screen';
import { LoginScreen } from './components/login-screen';
import { ChatInterface } from './components/chat-interface';

type AppScreen = 'welcome' | 'login' | 'chat';

export default function Home() {
  const [currentScreen, setCurrentScreen] = useState<AppScreen>('welcome');
  const [studentInfo, setStudentInfo] = useState<{ id: string; name: string; sessionId: string } | null>(null);

  // 检查是否已有登录信息
  useEffect(() => {
    const savedSessionId = localStorage.getItem('hazel_session_id');
    const savedStudentId = localStorage.getItem('hazel_student_id');
    const savedStudentName = localStorage.getItem('hazel_student_name');

    if (savedSessionId && savedStudentId) {
      setStudentInfo({
        id: savedStudentId,
        name: savedStudentName || '',
        sessionId: savedSessionId,
      });
      setCurrentScreen('chat');
    }
  }, []);

  const handleStart = () => {
    setCurrentScreen('login');
  };

  const handleLoginSuccess = (studentId: string, name: string, sessionId: string) => {
    setStudentInfo({ id: studentId, name, sessionId });
    setCurrentScreen('chat');
  };

  return (
    <>
      {currentScreen === 'welcome' && (
        <WelcomeScreen onStart={handleStart} />
      )}

      {currentScreen === 'login' && (
        <LoginScreen onLoginSuccess={handleLoginSuccess} />
      )}

      {currentScreen === 'chat' && studentInfo && (
        <ChatInterface
          sessionId={studentInfo.sessionId}
          studentName={studentInfo.name}
        />
      )}
    </>
  );
}
