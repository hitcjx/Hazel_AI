'use client';

export default function TeacherLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="teacher-page">
      {children}
      <style jsx global>{`
        .teacher-page::before {
          content: '';
          position: fixed;
          inset: -20px;
          background: url('/background4.webp') center/cover no-repeat;
          filter: blur(0.5px);
          z-index: -1;
        }
        .teacher-page::after {
          content: '';
          position: fixed;
          inset: 0;
          /* 移除遮罩，使用全局遮罩即可 */
          /* background: rgba(255, 243, 220, 0.3); */
          z-index: 0;
          pointer-events: none;
        }
      `}</style>
    </div>
  );
}
