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

        # [新增] 记录详细日志，包含快速回路的 Guard 模型输出
        session.add_log(
            user_msg,
            instruction,
            full_response_text,
            guard_raw=self.brain.last_guard_raw_output
        )

        # 5. 【慢速回路 - 真正异步化】
        # 使用 Daemon 线程，这样如果主程序退出，评估线程也会自动结束
        # 传递 user_msg 的副本，防止引用问题
        bg_thread = threading.Thread(
            target=self._slow_assessment_wrapper,
            args=(session, user_msg),
            daemon=True
        )
        bg_thread.start()

    def _slow_assessment_wrapper(self, session: SessionState, user_msg: str):
        """包装慢速评估，完成后更新日志"""
        self.brain.slow_assessment_update(session, user_msg)
        # 更新最后一条日志的 Brain 模型输出（包括阻力判断）
        session.update_last_log_model_outputs(
            brain_risk_raw=self.brain.last_brain_risk_raw_output,
            brain_assessment_raw=self.brain.last_brain_assessment_raw_output,
            resistance_raw=self.brain.last_resistance_raw_output
        )

    def _assemble_prompt(self, session: SessionState, instruction: str) -> str:
        """
        动态 Prompt 组装工厂 - 自然对话版

        核心改进：
        1. 去掉所有 ### 分隔符
        2. 去掉"请严格遵循"等命令式语言
        3. 让整个 prompt 读起来像"角色设定 + 当前心理状态"
        """
        # 危机模式：只用 instruction（本身已经是完整的危机干预协议）
        if session.is_crisis_mode:
            history_text = session.get_history_text()
            full_prompt = f"{instruction}\n\n{history_text}\n\nAssistant:"
            return full_prompt

        # 正常模式：自然融合 persona + instruction
        # 把 instruction 当作"此刻的对话重点"，而非"任务指令"
        system_prompt = f"""{self.persona}

【此刻的对话重点】
{instruction}

以上是你的人设和当前对话的关注点。请自然地融入对话中，像真正的知心学姐一样聊天。
"""

        # 历史对话：直接拼接，不加任何标题
        history_text = session.get_history_text()

        # 最终组装：系统设定 + 历史对话 + 提示开始回复
        full_prompt = f"{system_prompt}\n{history_text}\n\nAssistant:"

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