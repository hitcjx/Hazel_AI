import config
from state_manager import SessionState, MessageRole, RiskLevel
from the_brain import TheBrain
from llm_engine import LLMEngine
import threading

class TheAvatar:
    """
    [cite: 1] The Avatar: 执行层，负责 Prompt 组装与最终对话生成。
    """
    def __init__(self):
        # 初始化 LLM 引擎 
        self.llm_engine = LLMEngine(use_mock=False)
        
        # 初始化大脑，注入 LLM 引擎
        self.brain = TheBrain(self.llm_engine)
        
        # 加载人设 [cite: 3]
        self.persona = config.prompts.get("persona")

    def chat(self, session: SessionState, user_msg: str): # 返回 generator
        # 1. 记录用户输入
        session.add_message(MessageRole.USER, user_msg)
        
        # 2. 【快速回路】
        instruction = self.brain.fast_reaction(session, user_msg)
        final_prompt = self._assemble_prompt(session, instruction)
        
        # 3. 【流式输出】
        streamer = self.llm_engine.generate_avatar_response(final_prompt)
        
        full_response_text = ""
        for new_text in streamer:
            full_response_text += new_text
            yield new_text # 实时吐字
        
        # 4. 记录完整回复
        session.add_message(MessageRole.ASSISTANT, full_response_text)
        
        # [新增] 记录详细日志 (用于导出)
        session.add_log(user_msg, instruction, full_response_text)
        
        # 5. 【慢速回路 - 真正异步化】
        # 使用 Daemon 线程，这样如果主程序退出，评估线程也会自动结束
        # 传递 user_msg 的副本，防止引用问题
        bg_thread = threading.Thread(
            target=self.brain.slow_assessment_update,
            args=(session, user_msg),
            daemon=True
        )
        bg_thread.start()

    def _assemble_prompt(self, session: SessionState, instruction: str) -> str:
        """
        动态 Prompt 组装工厂
        """
        # [cite: 3] 角色基础设定
        system_block = f"{self.persona}\n"
        if session.is_crisis_mode:
            system_block = "" # 危机模式下，Safety Protocol 本身就是完整的人设
        
        # [cite: 4] 来自 The Brain 的 Next Instruction
        instruction_block = (
            f"\n### 当前核心指令 (System Instruction) ###\n"
            f"{instruction}\n"
            f"请严格遵循上述指令生成回复，保持语气自然温暖。\n"
        )
        
        # [cite: 5] 最近 10 轮对话 (由 session.get_history_text 提供)
        history_block = (
            f"\n### 对话历史 (Recent History) ###\n"
            f"{session.get_history_text()}\n"
        )
        
        # 这里的格式是为了适配 generate_avatar_response 中的输入
        # 如果是 ChatModel，通常会拆分为 messages list，但在 prompt_designing 中暗示了组装过程
        full_prompt = f"{system_block}\n{instruction_block}\n{history_block}\nAssistant:"
        
        return full_prompt

# =============================================================================
# 简单测试入口 (如果直接运行此文件)
# =============================================================================
if __name__ == "__main__":
    import uuid
    
    # 初始化
    avatar = TheAvatar()
    session = SessionState(session_id=str(uuid.uuid4()))
    
    print("=== 心理咨询 AI 系统启动 (Type 'quit' to exit) ===")
    print(f"当前阶段: {session.current_stage.name}")
    
    while True:
        user_input = input("\nStudent: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        reply = avatar.chat(session, user_input)
        print(f"\nAI: {reply}")
        
        # 打印调试信息：查看当前 Brain 的状态
        if session.risk_level != RiskLevel.NORMAL:
            print(f"[Debug] Risk Level: {session.risk_level}")
        # print(f"[Debug] Stage: {session.current_stage.name}")