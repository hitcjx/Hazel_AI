import re
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import config
from config import ASSESSMENT_RULES
from state_manager import (
    SessionState, 
    AssessmentState, 
    AssessmentDimension, 
    RiskLevel, 
    ConsultationStage,
    MessageRole
)
from llm_engine import LLMEngine 

# =============================================================================
# The Brain (核心逻辑控制器)
# =============================================================================
class TheBrain:
    def __init__(self, llm_engine: LLMEngine):
        self.llm = llm_engine

    def fast_reaction(self, session: SessionState, user_msg: str) -> str:
        """
        [修改] 快速反应回路 (Fast Loop)
        不调用 8B 模型，仅进行安全检查 + 返回当前阶段的预设指令
        """
        # 1. 快速安全扫描 (Regex + 1.5B)
        is_crisis = self.llm.fast_risk_check(user_msg)
        
        if is_crisis:
            # 如果小模型觉得有风险，直接切入危机模式，不再犹豫
            session.risk_level = RiskLevel.CRISIS
            return self._generate_crisis_instruction(session, user_msg)
        
        # 2. 如果无危机，直接生成当前阶段的指令 (不等待评估更新)
        # 这里的 instruction 也是基于 config 的预设文本，生成极快
        return self._generate_instruction(session, user_msg)

    def slow_assessment_update(self, session: SessionState, user_msg: str):
        """
        [新增] 慢速评估回路 (Slow Loop)
        在 Avatar 回复后调用，使用 8B 模型精细更新状态
        """
        # 1. 详细风险分析 (记录数据用，也可作为二次校验)
        # 虽然 Avatar 已经回复了，但我们依然记录一下精确的风险等级
        risk_level = self.llm.analyze_risk_level(user_msg, session.get_history_text())
        session.risk_level = risk_level

        # 2. 更新维度评分
        if session.current_stage != ConsultationStage.CLOSING_EMPOWERMENT:
            updates = self.llm.assess_dimensions_update(user_msg, session.assessment)
            for dim_name, data in updates.items():
                if dim_name in AssessmentDimension.__members__:
                    session.assessment.update_score(
                        dimension=AssessmentDimension[dim_name],
                        score=data.get('score'),
                        evidence=data.get('evidence'),
                        current_turn=session.turn_count
                    )
        
        # 3. 阶段流转判断 (为下一轮做准备)
        self._manage_stage_transition(session)

    def _generate_crisis_instruction(self, session: SessionState, user_msg: str) -> str:
        """生成红色危机干预指令 (CAMS/PFA)"""
        # 从评估中识别痛苦核心，用于模板填充
        top_risk = session.assessment.get_highest_risk_dimension()
        pain_point = "当前的痛苦"
        if top_risk:
            pain_point = f"【{config.ASSESSMENT_RULES[top_risk.name]['name_cn']}】方面的问题"
        
        # 用模板变量替换
        return config.prompts.get("safety", "crisis_instruction", pain_point=pain_point)

    def _manage_stage_transition(self, session: SessionState):
        """
         阶段流转逻辑表
        """
        current = session.current_stage
        turn = session.turn_count
        
        # 1. 破冰 -> 共情 (Warmup -> Deep Dive)
        # 进：对话开始；出：至少2-3轮且用户开始表露
        if current == ConsultationStage.WARM_UP_SCAN:
            has_risk_signal = session.risk_level != RiskLevel.NORMAL
            # 如果有风险信号，或者聊了超过3轮，进入深挖
            if turn >= 3 or has_risk_signal:
                session.current_stage = ConsultationStage.EMPATHY_DEEP_DIVE
        
        # 2. 共情 -> 重构 (Deep Dive -> Reframing)
        # 进：捕捉到情绪；出：维度槽填满
        elif current == ConsultationStage.EMPATHY_DEEP_DIVE:
            # 检查核心维度是否已评估 [cite: 41]
            # 生理功能(Physical)和认知偏差(Cognitive)是核心
            phy_assessed = session.assessment.dimensions[AssessmentDimension.PHYSICAL_FUNCTION].is_assessed
            cog_assessed = session.assessment.dimensions[AssessmentDimension.COGNITIVE_DISTORTION].is_assessed
            
            # 逻辑：核心维度已评估 OR 轮数过多(防止无限深挖)
            if (phy_assessed and cog_assessed) or turn > 20:
                # 只有在风险可控时才进入干intervention阶段
                if session.risk_level != RiskLevel.CRISIS:
                    session.current_stage = ConsultationStage.REFRAMING_SFBT

        # 3. 重构 -> 结束 (Reframing -> Closing)
        # 出：发现用户情绪好转 或 时间到
        elif current == ConsultationStage.REFRAMING_SFBT:
            if turn >= config.HISTORY_WINDOW_SIZE * 2: # 模拟时间限制
                session.current_stage = ConsultationStage.CLOSING_EMPOWERMENT

    def _generate_instruction(self, session: SessionState, user_msg: str) -> str:
        """
        核心策略生成器
        """
        stage = session.current_stage
        
        # === 阶段 1: 破冰与扫描 ===
        if stage == ConsultationStage.WARM_UP_SCAN:
            # [cite: 25-30] 开放式寒暄，扫描压力源
            return config.prompts.get("stages", "warmup")

        # === 阶段 2: 共情与深挖 ===
        elif stage == ConsultationStage.EMPATHY_DEEP_DIVE:
            # [cite: 40-41] 检查闲置维度
            idle_dim = session.assessment.get_idle_dimension(session.turn_count)
            
            # [cite: 45] 策略 B: 触发目标维度
            if idle_dim:
                return self._get_bridging_instruction(idle_dim)
            
            # [cite: 43] 策略 A: 自然对话 (Topic Follow)
            return config.prompts.get("stages", "topic_follow")

        # === 阶段 3: 重构与干预 (SFBT) ===
        elif stage == ConsultationStage.REFRAMING_SFBT:
            return self._handle_sfbt_logic(session, user_msg)

        # === 阶段 4: 结束与赋能 ===
        elif stage == ConsultationStage.CLOSING_EMPOWERMENT:
           return config.prompts.get("stages", "closing")

        return "保持共情，自然回应."

    def _handle_sfbt_logic(self, session: SessionState, user_msg: str) -> str:
        """
        [cite: 101-121] SFBT 核心逻辑：评量提问 -> 分支策略
        """
        # 1. 检测是否刚问完 Scaling Question
        user_score = self.llm.extract_scaling_score(user_msg)
        # 此时应默认进入“低分/痛苦”应对模式，而不是死循环
        if user_score is None and session.sfbt_scaling_done:
            user_score = 3.0 # 默认视为痛苦状态 [cite: 107]
        
        # 如果还没有完成打分提问，且用户也没有主动报分数
        if not session.sfbt_scaling_done and user_score is None:
            # 标记下一轮我们已经发出了提问 (或者在检测到用户回答了分数后标记为 True，这里简化逻辑，只要发出指令就视为流程推进)
            # 更严谨的逻辑是：发出指令 -> 下一轮检测到分数 -> 标记 True
            # 这里我们采用：只要 Brain 决定问，就假设 Avatar 会问。下一轮主要看 user_score。
            
            return config.prompts.get("sfbt", "scaling_question")

        # 如果用户回答了分数，或者我们之前已经问过了
        # 更新状态位：只要提取到了分数，就说明 Scaling 环节完成了
        if user_score is not None:
             session.sfbt_scaling_done = True

        X = user_score if user_score is not None else 3.0
        
        if X <= 3:
            return config.prompts.get(
                "sfbt", "coping", 
                score=X, 
                prev_stressors="这段时间的压力", 
                prev_score_minus_1=max(0, X-1)
            )
        elif 4 <= X <= 6:
            return config.prompts.get("sfbt", "exception", score=X, next_score=min(10, X+1))
        else:
            return config.prompts.get("sfbt", "smallest_step", score=X, target_score=min(10, X+0.5))
    def _get_bridging_instruction(self, dimension: AssessmentDimension) -> str:
        return config.prompts.get("bridging", dimension.name)