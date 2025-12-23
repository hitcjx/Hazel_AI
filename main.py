"""
main.py - 纯净交互版
功能：流式对话，无干扰 UI，会话结束后自动导出记录
"""
import sys
import json
import time
import os
from datetime import datetime
from the_avatar import TheAvatar
from state_manager import SessionState
import uuid
from state_manager import MessageRole

class CleanInterface:
    def __init__(self):
        print("⏳ 系统初始化中，请稍候... (加载模型可能需要 1-2 分钟)")
        self.avatar = TheAvatar()
        self.session = SessionState(session_id=str(uuid.uuid4()))
        self.start_time = datetime.now()
        
    def clear_screen(self):
        # 简单清屏，提升沉浸感
        os.system('cls' if os.name == 'nt' else 'clear')

    def save_log(self):
        """打包导出聊天记录"""
        duration = datetime.now() - self.start_time
        filename = f"chat_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = self.session.export_logs()
        export_data["duration_seconds"] = duration.total_seconds()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        return filename

    def run(self):
        self.clear_screen()
        print("\n" + "="*50)
        print("🌲 心理咨询室 (输入 'quit' 结束对话)")
        print("="*50 + "\n")
        
        # 开场白
        greeting = "你好呀。我是榛子。今天想聊点什么？"
        print(f"🤖 榛子: {greeting}\n")
        self.session.add_message(MessageRole.ASSISTANT, greeting)

        while True:
            try:
                # 1. 获取输入
                user_input = input("👤 你: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', '结束']:
                    print("\n正在整理本次咨询记录，请稍候...")
                    break
                
                # 2. 交互 (只显示回复)
                print("🤖 榛子: ", end="", flush=True)
                
                # 消费生成器
                for char in self.avatar.chat(self.session, user_input):
                    print(char, end="", flush=True)
                
                print("\n") # 只有这里才换行
                
                # 此时后台已经在跑 assessment 了，
                # 用户可以立刻输入，但如果输入太快，
                # 下一次的 print("🤖 榛子: ") 可能会因为等锁稍微卡顿一下
                
            except KeyboardInterrupt:
                print("\n\n(检测到中断，正在保存记录...)")
                break
            except Exception as e:
                print(f"\n[系统错误] {str(e)}")
                break
        
        # 结束处理
        file_path = self.save_log()
        print("="*50)
        print(f"✅ 对话已结束。")
        print(f"📂 完整记录（包含指令与评分）已保存至: {file_path}")
        print("="*50)

if __name__ == "__main__":
    app = CleanInterface()
    app.run()