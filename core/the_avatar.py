import core.config as config
from core.state_manager import SessionState, MessageRole, RiskLevel
from core.the_brain import TheBrain
from core.llm_engine import LLMEngine
import threading
import uuid

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

        # 初始化记忆管理器（可选）
        try:
            from memory_manager import MemoryManager
            self.memory_manager = MemoryManager()
        except ImportError:
            print("[Warning] mem0ai not installed. Memory features disabled.")
            self.memory_manager = None

    def start_session(self, user_id: str = None) -> SessionState:
        """
        开始新会话

        Args:
            user_id: 用户ID（如果为 None，则生成新ID）

        Returns:
            初始化的 SessionState
        """
        # 1. 生成或使用 user_id
        if not user_id:
            user_id = str(uuid.uuid4())

        # 2. 初始化会话状态
        session = SessionState(
            session_id=str(uuid.uuid4()),
            user_id=user_id
        )

        # 3. 打印欢迎信息
        if self.memory_manager:
            # 可以在这里预加载用户信息（可选）
            print(f"[会话开始] user_id: {user_id}")

        return session

    def end_session(self, session: SessionState):
        """
        结束会话

        Args:
            session: 会话状态
        """
        # 不需要额外保存，因为每轮对话已经自动添加到 mem0
        print(f"[会话结束] user_id: {session.user_id}, turns: {session.turn_count}")

    def chat(self, session: SessionState, user_msg: str):
        """
        与用户对话（流式输出）

        Args:
            session: 会话状态
            user_msg: 用户消息

        Yields:
            生成的文本片段
        """
        import time
        start_time = time.time()

        # 1. 记录用户输入
        session.add_message(MessageRole.USER, user_msg)
        print(f"  ⏱️ [chat] 用户消息已记录，耗时: {int((time.time()-start_time)*1000)}ms")

        # 2. 【快速回路】生成指令
        brain_start = time.time()
        instruction = self.brain.fast_reaction(session, user_msg)
        print(f"  ⏱️ [chat] fast_reaction 完成，耗时: {int((time.time()-brain_start)*1000)}ms")

        prompt_start = time.time()
        final_prompt = self._assemble_prompt(session, instruction, user_msg)
        print(f"  ⏱️ [chat] prompt组装完成，耗时: {int((time.time()-prompt_start)*1000)}ms")
        print(f"  ⏱️ [chat] Prompt总长度: {len(final_prompt)} 字符")

        # 3. 【流式输出】生成回复
        llm_start = time.time()
        streamer = self.llm_engine.generate_avatar_response(final_prompt)
        print(f"  🔍 [chat] LLM生成器已创建，准备流式输出...")

        full_response_text = ""
        chunk_count = 0
        for new_text in streamer:
            if chunk_count == 0:
                print(f"  ⏱️ [chat] 第一个chunk收到，总耗时: {int((time.time()-start_time)*1000)}ms (LLM推理: {int((time.time()-llm_start)*1000)}ms)")
            chunk_count += 1
            full_response_text += new_text
            yield new_text  # 实时吐字

        print(f"  ⏱️ [chat] 流式输出完成，共 {chunk_count} 个chunks")

        # 4. 空响应检测和兜底
        if not full_response_text or not full_response_text.strip():
            print(f"  ⚠️ [chat] 检测到空响应，使用兜底回复")
            full_response_text = "抱歉，我刚才好像没听清楚，能再跟我说说吗？"
            yield full_response_text  # 补发兜底回复

        # 5. 记录完整回复
        session.add_message(MessageRole.ASSISTANT, full_response_text)

        # 6. 记录详细日志
        session.add_log(
            user_msg,
            instruction,
            full_response_text,
            guard_raw=self.brain.last_guard_raw_output,
            sfbt_raw=self.brain.last_sfbt_raw_output
        )

        # 7. 【新增】保存对话到 mem0（异步后台执行，避免阻塞流式输出）
        if self.memory_manager:
            def save_memory_background():
                try:
                    self.memory_manager.add_conversation(
                        user_id=session.user_id,
                        user_msg=user_msg,
                        assistant_msg=full_response_text,
                        metadata={
                            "session_id": session.session_id,
                            "turn": session.turn_count,
                            "risk_level": session.risk_level.name,
                            "stage": session.current_stage.name
                        }
                    )
                    print(f"✅ 记忆保存完成: {session.user_id}")
                except Exception as e:
                    print(f"⚠️ 记忆保存失败: {e}")

            # 启动后台线程，不等待完成
            memory_thread = threading.Thread(
                target=save_memory_background,
                daemon=True,
                name=f"MemorySave-{session.user_id[:8]}"
            )
            memory_thread.start()
            print(f"  🔄 [chat] 记忆保存已启动后台线程")

        # 8. 【慢速回路 - 异步评估】
        # 使用 Daemon 线程，这样如果主程序退出，评估线程也会自动结束
        current_turn = session.turn_count  # 保存当前轮次ID
        bg_thread = threading.Thread(
            target=self._slow_assessment_wrapper,
            args=(session, user_msg, current_turn),
            daemon=True
        )
        bg_thread.start()

    def _slow_assessment_wrapper(self, session: SessionState, user_msg: str, turn_id: int):
        """包装慢速评估，完成后更新日志"""
        try:
            self.brain.slow_assessment_update(session, user_msg)
            # 更新指定轮次的日志（修复并发竞态）
            session.update_last_log_model_outputs(
                brain_risk_raw=self.brain.last_brain_risk_raw_output,
                brain_assessment_raw=self.brain.last_brain_assessment_raw_output,
                resistance_raw=self.brain.last_resistance_raw_output,
                turn_id=turn_id  # 明确指定要更新哪一轮的日志
            )
        except Exception as e:
            import traceback
            print(f"[WARNING] slow_assessment_update failed: {e}")
            traceback.print_exc()

    def _assemble_prompt(self, session: SessionState, instruction: str, user_msg: str) -> str:
        """
        统一Prompt组装架构

        完整组装形式：persona + [相关记忆] + instruction + history

        适用于所有模式：
        - Normal Mode (4个stage: warmup/topic_follow/sfbt/closing)
        - Resistance Mode (3个级别: passive/defensive/hostile)
        - Crisis Mode (危机干预)

        优先级：Crisis > Resistance > Normal Stage

        Args:
            session: 会话状态
            instruction: Brain 生成的指令
            user_msg: 用户当前输入（用于搜索相关记忆）

        Returns:
            完整的 prompt
        """
        parts = [self.persona]

        # 1. 从 mem0 搜索相关记忆
        if self.memory_manager:
            relevant_memories = self.memory_manager.search(
                query=user_msg,
                user_id=session.user_id,
                limit=5
            )
            if relevant_memories:
                memory_texts = []
                # 确保是列表类型
                memories_list = list(relevant_memories)[:5]  # 最多5条
                for m in memories_list:
                    if isinstance(m, dict):
                        memory_texts.append(m.get('memory', ''))
                    else:
                        memory_texts.append(str(m))
                parts.append(f"【相关记忆】\n" + "\n".join(f"- {m}" for m in memory_texts))

        # 2. instruction
        parts.append(instruction)

        # 3. history（优化格式：U: 用户 A: Avatar）
        history_text = self._get_optimized_history(session)
        parts.append(f"\n{history_text}")

        return "\n\n".join(parts)

    def _get_optimized_history(self, session: SessionState) -> str:
        """
        获取优化的历史记录格式

        格式：U: xxx A: xxx（节省 token）

        优化策略：最近5轮完整 + 早期对话截断

        Args:
            session: 会话状态

        Returns:
            优化后的历史文本
        """
        messages = list(session.history)

        # 分为两部分：最近5轮（完整）+ 更早的对话（截断）
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        older_messages = messages[:-5] if len(messages) > 5 else []

        # 优化格式：用户:/咨询师: (避免LLM学习缩写格式)
        lines = []

        # 早期对话：只取前30字
        for msg in older_messages:
            role = "用户" if msg.role == MessageRole.USER else "咨询师"
            content = msg.content[:30]
            if len(msg.content) > 30:
                content += "..."
            lines.append(f"{role}: {content}")

        # 最近5轮：完整保留（最多100字）
        for msg in recent_messages:
            role = "用户" if msg.role == MessageRole.USER else "咨询师"
            content = msg.content[:100]
            if len(msg.content) > 100:
                content += "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _detect_topic_switch(self, session: SessionState) -> int:
        """
        检测话题跳跃，动态调整 history window

        Args:
            session: 会话状态

        Returns:
            history window 大小（默认 10，检测到跳跃时扩展到 15）
        """
        if len(session.history) < 5:
            return 10  # 对话太少，使用默认值

        # 获取最近的用户输入（前2轮）
        recent_msgs = list(session.history)[-4:]  # 最近 2 轮（用户+Avatar）
        recent_user_inputs = [m.content for m in recent_msgs if m.role == MessageRole.USER]

        if len(recent_user_inputs) < 2:
            return 10

        # 简单判断：如果用户输入包含"对了"、"说起"、"换个话题"等词，可能是话题跳跃
        jump_keywords = ["对了", "说起", "换个话题", "另外", "说到", "我想起"]
        latest_input = recent_user_inputs[-1]

        for keyword in jump_keywords:
            if keyword in latest_input:
                print(f"[检测到话题跳跃] history window 扩展到 15")
                return 15

        # 默认 10 轮
        return 10

