import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: {
          bg: "#FDF6EC",      // 午后阳光背景 - 极暖杏米色
          surface: "#FFF8E8", // 纸张容器 - 明显米黄色
          ai: "#D4D9D2",      // AI消息气泡 - 豆蔻绿
          user: "#DDD0C0",    // 用户消息气泡 - 暖棕色（加深）
          text: "#5D4E37",    // 主要文字 - 暖深棕
          muted: "#8B7355",   // 次要文字 - 暖中棕
          status: "#A69685",  // 状态文字 - 暖褐色
          border: "#E8DFD0",  // 边框 - 暖米色
        edge: "#F2EFE8",    // 纸张边缘 - 极浅米白
          input: "#FFFFFF",   // 输入框背景 - 纯白
          accent: "#C9B8A0",  // 强调色 - 暖金
        }
      },
      boxShadow: {
        'paper': '0 10px 40px -10px rgba(139, 115, 85, 0.15), 0 20px 60px -20px rgba(139, 115, 85, 0.1)',
        'paper-3d': '0 4px 10px rgba(0,0,0,0.04), 0 20px 50px -10px rgba(120, 110, 100, 0.15)',
        'paper-edge': '0 10px 40px -10px rgba(139, 115, 85, 0.12), 0 20px 60px -20px rgba(139, 115, 85, 0.08)',
        'morandi': '0 2px 8px -2px rgba(0, 0, 0, 0.05)',
        'morandi-lg': '0 10px 25px -5px rgba(0, 0, 0, 0.03), 0 8px 10px -6px rgba(0, 0, 0, 0.03)',
        'morandi-focus': '0 0 0 3px rgba(194, 201, 191, 0.2)',
        'morandi-sm': '0 2px 4px rgba(0,0,0,0.02)',
        'morandi-md': '0 4px 12px -2px rgba(0,0,0,0.04)',
        'morandi-floating': '0 12px 30px -10px rgba(0,0,0,0.08)',
        'bubble': '0 2px 8px -2px rgba(93, 78, 55, 0.08), 0 4px 16px -4px rgba(93, 78, 55, 0.04)',
      },
      animation: {
        'breathe': 'breathe 4s ease-in-out infinite',
        'float-left': 'floatLeft 3s ease-in-out infinite',
        'float-right': 'floatRight 3s ease-in-out infinite 0.5s',
        'bubble-pop': 'bubblePop 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards',
        'bubble-pop-left': 'bubblePopLeft 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards',
        'bubble-pop-right': 'bubblePopRight 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards',
        'fade-in-up': 'fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'status-breathe': 'statusFade 3s ease-in-out infinite',
      },
      keyframes: {
        breathe: {
          '0%, 100%': { transform: 'rotate(-5deg)' },
          '50%': { transform: 'rotate(5deg)' },
        },
        floatLeft: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-1.5px)' },
        },
        floatRight: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-1.5px)' },
        },
        bubblePopLeft: {
          '0%': { opacity: '0', transform: 'translateX(-30px) scale(0.8)', filter: 'blur(4px)' },
          '70%': { transform: 'translateX(3px) scale(1.02)' },
          '100%': { opacity: '1', transform: 'translateX(0) scale(1)', filter: 'blur(0)' },
        },
        bubblePopRight: {
          '0%': { opacity: '0', transform: 'translateX(30px) scale(0.8)', filter: 'blur(4px)' },
          '70%': { transform: 'translateX(-3px) scale(1.02)' },
          '100%': { opacity: '1', transform: 'translateX(0) scale(1)', filter: 'blur(0)' },
        },
        bubblePop: {
          '0%': { opacity: '0', transform: 'translateY(40px) scale(0.6)', filter: 'blur(8px)' },
          '60%': { transform: 'translateY(-5px) scale(1.08)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)', filter: 'blur(0)' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(10px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        statusFade: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.8' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
