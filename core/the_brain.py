import re
import threading
import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import core.config as config
from core.config import ASSESSMENT_RULES
from core.state_manager import (
    SessionState,
    AssessmentState,
    AssessmentDimension,
    RiskLevel,
    ResistanceLevel,
    ConsultationStage,
    MessageRole,
    Message
)
from core.llm_engine import LLMEngine

# 导入 SFBT 系统
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sfbt.strategy_generator import SFBTSplitter, SFBTStrategyGenerator 

# =============================================================================
# The Brain (核心逻辑控制器)
# =============================================================================
class TheBrain:
    def __init__(self, llm_engine: LLMEngine):
        self.llm = llm_engine
        # 用于临时存储当前轮次的模型原始输出
        self.last_guard_raw_output = ""
        self.last_brain_risk_raw_output = ""
        self.last_brain_assessment_raw_output = ""
        self.last_resistance_raw_output = ""  # 新增：阻力判断原始输出
        self.last_sfbt_raw_output = ""  # 新增：SFBT策略器原始JSON输出

        # 初始化 SFBT 系统
        self.sfbt_splitter = SFBTSplitter()       # 快脑
        self.sfbt_generator = SFBTStrategyGenerator()  # 策略器

    def fast_reaction(self, session: SessionState, user_msg: str) -> str:
        """
        [修改] 快速反应回路 (Fast Loop)
        不调用 8B 模型，仅进行安全检查 + 阻力检测 + 返回当前阶段的预设指令
        """
        import time
        fast_start = time.time()

        # 重置当前轮次的输出记录
        self.last_guard_raw_output = ""
        self.last_brain_risk_raw_output = ""
        self.last_brain_assessment_raw_output = ""
        self.last_resistance_raw_output = ""
        self.last_sfbt_raw_output = ""

        # SFBT 和 Closing 阶段：跳过所有检测（Guard/阻力/风险/五维度），直接生成指令
        if session.current_stage in (ConsultationStage.REFRAMING_SFBT,
                                     ConsultationStage.CLOSING_EMPOWERMENT):
            return self._generate_instruction(session, user_msg)

        # 1. 快速安全扫描 (Regex + 1.5B) - 非 SFBT/Closing 阶段执行
        is_crisis, guard_raw = self.llm.fast_risk_check(user_msg)
        self.last_guard_raw_output = guard_raw

        if is_crisis:
            session.risk_level = RiskLevel.CRISIS
            return self._generate_crisis_instruction(session, user_msg)

        # 2. 快速阻力检测（正则）
        detected_resistance = self.llm.quick_resistance_check(user_msg)

        if detected_resistance:
            # 检测到阻力，重置积极回应计数
            session.resistance.positive_responses_after_resistance = 0

            # 如果检测到阻力
            if not session.resistance.llm_confirmed and session.resistance.consecutive_count == 0:
                # 首次检测到阻力，标记为待确认（慢速回路会确认）
                session.resistance.level = detected_resistance
                session.resistance.evidence = user_msg
                # 暂不增加consecutive_count，等LLM确认后再增加
            elif session.resistance.llm_confirmed and session.resistance.consecutive_count >= 1:
                # 二次触发（已被LLM确认过），触发应对策略
                session.resistance.consecutive_count += 1
                session.resistance.level = detected_resistance  # 修复：更新阻力等级
                session.resistance.evidence = user_msg
                return self._generate_resistance_instruction(session, detected_resistance)
        else:
            # 未检测到阻力，增加积极回应计数
            session.resistance.positive_responses_after_resistance += 1
            # 只有连续N次积极回应才完全重置（缓冲窗口机制）
            if session.resistance.positive_responses_after_resistance >= config.RESISTANCE_RESET_THRESHOLD:
                session.reset_resistance()

        # 3. 如果无危机、无二次阻力，生成正常指令
        import time
        print(f"⏱️ [Brain] fast_reaction 总耗时: {int((time.time()-fast_start)*1000)}ms")
        return self._generate_instruction(session, user_msg)

    def slow_assessment_update(self, session: SessionState, user_msg: str):
        """
        [修改] 慢速评估回路 (Slow Loop)
        在 Avatar 回复后调用，使用 8B 模型精细更新状态
        """
        # SFBT 和 Closing 阶段：只记录数据，不进行风险/阻力/维度检测
        if session.current_stage in (ConsultationStage.REFRAMING_SFBT,
                                     ConsultationStage.CLOSING_EMPOWERMENT):
            print(f"[Brain] 当前阶段 {session.current_stage.name}，跳过风险/阻力/维度检测")
            return

        # 1. 详细风险分析 (记录数据用，也可作为二次校验)
        risk_level, risk_raw = self.llm.analyze_risk_level(user_msg, session.get_history_text())
        self.last_brain_risk_raw_output = risk_raw
        print(f"[Brain] 风险分析原始输出: {risk_raw[:300]}...")
        print(f"[Brain] 解析后风险等级: {risk_level}")

        # 危机观察期机制：防止过快退出危机模式
        if risk_level == RiskLevel.CRISIS:
            # 检测到危机，立即进入危机模式
            session.risk_level = RiskLevel.CRISIS
            session.crisis_stabilization_turns = 0  # 重置观察期计数
            if not session.crisis_triggered:
                session.crisis_triggered = True
                session.crisis_trigger_time = datetime.now()
        elif session.crisis_triggered and risk_level < RiskLevel.CRISIS:
            # 曾经触发过危机，现在检测到风险降低，进入观察期
            session.crisis_stabilization_turns += 1
            # 连续N轮稳定才真正降级
            if session.crisis_stabilization_turns >= config.CRISIS_STABILIZATION_REQUIRED:
                session.risk_level = risk_level
                session.crisis_stabilization_turns = 0
            # 否则保持CRISIS状态，继续观察
        else:
            # 非危机状态的正常更新
            session.risk_level = risk_level

        # 2. 阻力LLM确认（仅在首次正则检测时）
        if (not session.resistance.llm_confirmed and
            session.resistance.consecutive_count == 0 and
            session.resistance.evidence):  # 有待确认的阻力

            has_resistance, confirmed_level, resistance_raw = self.llm.analyze_resistance(
                user_msg, session.get_history_text()
            )
            self.last_resistance_raw_output = resistance_raw

            if has_resistance:
                # LLM确认有阻力
                session.resistance.llm_confirmed = True
                session.resistance.level = confirmed_level
                session.resistance.consecutive_count = 1  # 确认后设为1
                session.resistance.last_updated_turn = session.turn_count
            else:
                # LLM不认为有阻力，重置
                session.reset_resistance()

        # 3. 更新维度评分 (排除 SFBT 和 Closing 阶段)
        if session.current_stage not in (ConsultationStage.CLOSING_EMPOWERMENT,
                                        ConsultationStage.REFRAMING_SFBT):
            print(f"[Brain] 开始评估维度，当前阶段: {session.current_stage}, 轮次: {session.turn_count}")
            full_json, assessment_raw = self.llm.assess_dimensions_update(user_msg, session.assessment)
            self.last_brain_assessment_raw_output = assessment_raw
            print(f"[Brain] 评估原始输出: {assessment_raw[:200]}...")

            # 提取 updates 字段
            updates = full_json.get("updates", {})
            print(f"[Brain] 评估结果: {updates}")

            for dim_name, data in updates.items():
                # 大小写兼容：将LLM返回的维度名转为大写匹配
                dim_name_upper = dim_name.upper()
                matched_dim = None
                for dim_enum in AssessmentDimension.__members__.keys():
                    if dim_enum.upper() == dim_name_upper:
                        matched_dim = dim_enum
                        break

                if matched_dim:
                    score = data.get('score')
                    evidence = data.get('evidence')

                    if score is not None:
                        session.assessment.update_score(
                            dimension=AssessmentDimension[matched_dim],
                            score=score,
                            evidence=evidence or "",
                            current_turn=session.turn_count
                        )
                        print(f"[Brain] 维度 {matched_dim} 评分: {score}")

        # 4. 阶段流转判断 (为下一轮做准备）
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

    def _generate_resistance_instruction(self, session: SessionState, level: ResistanceLevel) -> str:
        """
        生成阻力应对指令（三档策略）

        Args:
            session: 会话状态
            level: 阻力等级

        Returns:
            对应等级的应对指令
        """
        level_to_prompt = {
            ResistanceLevel.PASSIVE: "passive_response",
            ResistanceLevel.DEFENSIVE: "defensive_response",
            ResistanceLevel.HOSTILE: "hostile_response"
        }

        prompt_key = level_to_prompt.get(level, "passive_response")
        return config.prompts.get("resistance", prompt_key)

    def _check_time_condition(self, session: SessionState, stage_key: str) -> bool:
        """
        检查时间条件是否满足

        Args:
            session: 会话状态
            stage_key: 阶段键名（如"WARM_UP_SCAN"）

        Returns:
            bool: 时间条件是否满足
        """
        elapsed_seconds = (datetime.now() - session.start_time).total_seconds()
        elapsed_minutes = elapsed_seconds / 60.0

        # 计算该阶段应该持续的时间
        stage_duration = config.MAX_SESSION_TIME_MINUTES * config.STAGE_TIME_ALLOCATION.get(stage_key, 0)

        return elapsed_minutes >= stage_duration

    def consolidate_evidence(self, session: SessionState) -> str:
        """
        整合各维度的 evidence 为上下文文本
        Deep Dive 结束时调用，结果存入 session 供 SFBT 使用
        """
        lines = []
        for dim, state in session.assessment.dimensions.items():
            if state.evidence:
                # 添加维度名称的中文说明
                dim_cn = {
                    "PHYSICAL_FUNCTION": "生理功能",
                    "COGNITIVE_DISTORTION": "认知偏差",
                    "ACADEMIC_PRESSURE": "学业压力",
                    "EMOTIONAL_RESILIENCE": "情绪韧度",
                    "SOCIAL_SUPPORT": "社会支持"
                }.get(dim.name, dim.name)
                lines.append(f"【{dim_cn}】{state.evidence}")

        if not lines:
            return ""

        return "=== 评估维度信息 ===\n" + "\n".join(lines)

    def _manage_stage_transition(self, session: SessionState):
        """
        阶段流转逻辑表（时间 OR 轮数，任一条件满足即触发）
        """
        current = session.current_stage
        turn = session.turn_count
        
        # 1. 破冰 -> 共情 (Warmup -> Deep Dive)
        # 时间条件：已过warmup分配时间（10% = 4分钟）
        # 轮数条件：(turn >= min_turns) AND (turn >= max_turns OR 检测到任一信号)
        # 检测到信号包括：风险信号、阻力信号、任何维度被评估
        if current == ConsultationStage.WARM_UP_SCAN:
            time_ready = self._check_time_condition(session, "WARM_UP_SCAN")

            thresholds = config.STAGE_TURN_THRESHOLDS["warmup_to_deepdive"]

            # 检测到任一信号：风险、阻力、维度评估
            has_signal = (
                session.risk_level != RiskLevel.NORMAL or  # 风险信号
                session.resistance.level != ResistanceLevel.NONE or  # 阻力信号
                any(dim.is_assessed for dim in session.assessment.dimensions.values())  # 维度被评估
            )

            turn_ready = (turn >= thresholds["min_turns"] and
                         (turn >= thresholds["max_turns"] or has_signal))

            # OR逻辑：时间到 或 轮数条件满足
            if time_ready or turn_ready:
                session.current_stage = ConsultationStage.EMPATHY_DEEP_DIVE
        
        # 2. 共情 -> 重构 (Deep Dive -> Reframing)
        # 时间条件：已过warmup+deepdive分配时间（10%+50% = 24分钟）
        # 轮数条件：(turn >= min_turns) AND (turn >= max_turns OR 核心维度已评估)
        elif current == ConsultationStage.EMPATHY_DEEP_DIVE:
            # 计算累计时间（warmup + deep_dive）
            elapsed_minutes = (datetime.now() - session.start_time).total_seconds() / 60.0
            cumulative_time = config.MAX_SESSION_TIME_MINUTES * (
                config.STAGE_TIME_ALLOCATION["WARM_UP_SCAN"] +
                config.STAGE_TIME_ALLOCATION["EMPATHY_DEEP_DIVE"]
            )
            time_ready = elapsed_minutes >= cumulative_time

            # 检查核心维度是否已评估
            phy_assessed = session.assessment.dimensions[AssessmentDimension.PHYSICAL_FUNCTION].is_assessed
            cog_assessed = session.assessment.dimensions[AssessmentDimension.COGNITIVE_DISTORTION].is_assessed
            core_assessed = phy_assessed and cog_assessed

            thresholds = config.STAGE_TURN_THRESHOLDS["deepdive_to_sfbt"]
            # 25轮设为下限（至少25轮），上限只看时间
            turn_ready = (turn >= thresholds["min_turns"] and core_assessed) or time_ready

            # OR逻辑：时间到 或 轮数条件满足，且非危机状态
            if (time_ready or turn_ready) and session.risk_level != RiskLevel.CRISIS:
                # 进入 SFBT 前整合评估信息并加入对话历史
                assessment_context = self.consolidate_evidence(session)
                print(f"[Brain] 评估上下文: {assessment_context[:300] if assessment_context else '无'}")
                if assessment_context:
                    try:
                        session.history.appendleft(Message(
                            role=MessageRole.SYSTEM,
                            content=assessment_context
                        ))
                        print(f"[Brain] 评估上下文已添加到历史")
                    except Exception as e:
                        # deque 不支持 insert/appendleft 时降级为 append
                        print(f"[WARNING] history.appendleft failed: {e}")
                        session.history.append(Message(
                            role=MessageRole.SYSTEM,
                            content=assessment_context
                        ))
                session.current_stage = ConsultationStage.REFRAMING_SFBT
                print(f"[Brain] 阶段切换到 REFRAMING_SFBT")

        # 3. 重构 -> 结束 (Reframing -> Closing)
        # 时间条件：已过warmup+deepdive+sfbt分配时间（10%+50%+35% = 38分钟）
        # 轮数条件：sfbt轮数 >= 15
        # 两个条件都满足才切换（AND逻辑）
        elif current == ConsultationStage.REFRAMING_SFBT:
            # 计算累计时间（warmup + deep_dive + sfbt）
            elapsed_minutes = (datetime.now() - session.start_time).total_seconds() / 60.0
            cumulative_time = config.MAX_SESSION_TIME_MINUTES * (
                config.STAGE_TIME_ALLOCATION["WARM_UP_SCAN"] +
                config.STAGE_TIME_ALLOCATION["EMPATHY_DEEP_DIVE"] +
                config.STAGE_TIME_ALLOCATION["REFRAMING_SFBT"]
            )
            time_ready = elapsed_minutes >= cumulative_time

            thresholds = config.STAGE_TURN_THRESHOLDS["sfbt_to_closing"]
            sfbt_turn = session.sfbt_turn_count
            turn_ready = sfbt_turn >= thresholds["max_turns"]

            # AND逻辑：时间到 且 轮数条件都满足才切换
            if time_ready and turn_ready:
                session.current_stage = ConsultationStage.CLOSING_EMPOWERMENT

    def _generate_instruction(self, session: SessionState, user_msg: str) -> str:
        """
        核心策略生成器 - Normal Mode指令路由

        返回的instruction会在Avatar中组装为：persona + instruction + history
        """
        stage = session.current_stage

        # === Stage 1: Warmup ===
        # 组装形式: persona + warmup + history
        if stage == ConsultationStage.WARM_UP_SCAN:
            return config.prompts.get("stages", "warmup")

        # === Stage 2: Topic Follow (Deep Dive) ===
        # 基础形式: persona + topic_follow + history
        # 增强形式: persona + bridging.XXX + history (替代topic_follow，不拼接)
        elif stage == ConsultationStage.EMPATHY_DEEP_DIVE:
            # 检查是否有需要评估的维度
            idle_dim = session.assessment.get_idle_dimension(
                current_turn=session.turn_count,
                pending_dimensions=session.pending_dimensions,
                dimension_retry_count=session.dimension_retry_count,
                retry_delay=config.DIMENSION_RETRY_DELAY,
                max_retries=config.MAX_DIMENSION_RETRIES
            )

            # 策略分支（替代逻辑，非拼接）：
            if idle_dim:
                # 有待评估维度 -> 使用bridging替代topic_follow
                return self._get_bridging_instruction(idle_dim)
            else:
                # 无待评估维度 -> 使用默认topic_follow
                return config.prompts.get("stages", "topic_follow")

        # === Stage 3: SFBT ===
        # 组装形式: persona + sfbt_XXX + history
        elif stage == ConsultationStage.REFRAMING_SFBT:
            return self._handle_sfbt_logic(session, user_msg)

        # === Stage 4: Closing ===
        # 组装形式: persona + closing + history
        elif stage == ConsultationStage.CLOSING_EMPOWERMENT:
           return config.prompts.get("stages", "closing")

        return "保持共情，自然回应."

    def _handle_sfbt_logic(self, session: SessionState, user_msg: str) -> str:
        """
        SFBT 核心逻辑：
        - 如果上一轮已选中 method，跳过快脑，直接调用策略器
        - 否则并行调用快脑和策略器
        """
        # 递增 SFBT 轮数计数
        session.sfbt_turn_count += 1
        print(f"[SFBT] 当前第 {session.sfbt_turn_count} 轮")

        # 准备历史对话
        history = []
        for msg in list(session.history)[-12:]:  # 最近12条消息
            history.append({"role": msg.role.value, "content": msg.content})

        # 检查是否需要跳过快脑判断
        # 如果上一轮有选中 method，说明已经在干预过程中，跳过快脑，直接调用策略器
        if session.last_method:
            print(f"[跳过快脑] 上一轮已选中 method: {session.last_method}，直接调用策略器")

            # 直接调用策略器（不调用快脑）
            package, reasoning = self.sfbt_generator.generate_strategy(
                user_msg=user_msg,
                full_history=history,
                previous_method=session.last_method,
                previous_score=session.last_score,
                previous_action=session.last_action,
                previous_action_status=session.previous_action_status,
                in_progress_count=session.in_progress_count,
                current_module=session.last_current_module
            )

            # 处理策略器结果，继续执行后面的逻辑...
        else:
            # 首次进入，需要快脑判断
            # 用于存储结果的容器
            result_container = {"strategy_done": False, "strategy_result": None}
            need_sfbt = None  # 快脑判断结果

            def run_splitter():
                """快脑判断线程"""
                nonlocal need_sfbt
                split_result = self.sfbt_splitter.check_need_sfbt(user_msg, history)
                need_sfbt = split_result.need_sfbt
                print(f"[快脑] 判断结果: {'需要' if need_sfbt else '不需要'} SFBT 干预")

            def run_generator():
                """策略器生成线程"""
                nonlocal result_container
                package, reasoning = self.sfbt_generator.generate_strategy(
                    user_msg=user_msg,
                    full_history=history,
                    previous_method=session.last_method or "",
                    previous_score=session.last_score,
                    previous_action=session.last_action,
                    previous_action_status=session.previous_action_status,
                    in_progress_count=session.in_progress_count,
                    current_module=session.last_current_module
                )
                result_container["strategy_done"] = True
                result_container["strategy_result"] = (package, reasoning)

            # 并行启动快脑和策略器
            splitter_thread = threading.Thread(target=run_splitter)
            generator_thread = threading.Thread(target=run_generator)

            splitter_thread.start()
            generator_thread.start()

            # 等待快脑先完成
            splitter_thread.join()

            # 如果快脑判断不需要 SFBT，立即返回跟随对话
            if not need_sfbt:
                # 等待策略器线程结束（可选，避免资源泄漏）
                generator_thread.join(timeout=1)
                # 重置状态（包括 last_method 和 last_plugin）
                session.action = ""
                session.last_method = ""
                session.last_plugin = ""
                session.last_action = ""
                session.previous_action_status = ""
                session.in_progress_count = 0
                return self._get_sfbt_prompt("跟随对话", "", session.last_score)

            # 快脑判断需要 SFBT，等待策略器完成
            generator_thread.join()

            if not result_container["strategy_done"]:
                # 策略器未完成
                session.action = ""
                session.last_method = ""
                session.last_plugin = ""
                session.last_action = ""
                session.previous_action_status = ""
                return self._get_sfbt_prompt("跟随对话", "", session.last_score)

            package, reasoning = result_container["strategy_result"]

        # 保存 SFBT 策略器的原始 JSON 输出
        if package.raw_data:
            import json
            self.last_sfbt_raw_output = json.dumps(package.raw_data, ensure_ascii=False)

        # 检查 method 是否切换，如果切换则清空 action 相关字段
        method_switched = (session.last_method != package.selected_method and session.last_method != "")
        if method_switched:
            print(f"\n🔄 [DEBUG] Method 切换: '{session.last_method}' → '{package.selected_method}'")
            print(f"  - 清空 action 相关字段")
            print(f"    - action: '{session.action}' → ''")
            print(f"    - last_action: '{session.last_action}' → ''")
            print(f"    - previous_action_status: '{session.previous_action_status}' → ''")
            print(f"    - in_progress_count: {session.in_progress_count} → 0")
            session.action = ""
            session.last_action = ""
            session.previous_action_status = ""
            session.in_progress_count = 0

        # 记录上一轮信息（供下一轮使用）
        session.last_method = package.selected_method
        session.last_current_module = package.current_module
        session.last_score = package.score
        session.last_plugin = package.plugin
        # 注意：action 的更新在后面 action_status 处理之后进行

        # 获取 action_status（默认 in_progress）
        action_status = package.action_status or "in_progress"
        session.previous_action_status = action_status

        # 更新 in_progress 计数器
        if action_status == "in_progress":
            session.in_progress_count += 1
            print(f"[计数器] in_progress 次数: {session.in_progress_count}")
            # 超过3次自动放弃
            if session.in_progress_count >= 3:
                print(f"[计数器] 达到3次，自动放弃")
                action_status = "abandoned"
        else:
            session.in_progress_count = 0

        # 处理 action_status
        if action_status == "abandoned":
            # 放弃：清空所有状态，下一轮重新调用快脑选择新方法
            session.action = ""
            # 清空 last_method，下一轮重新调用快脑
            session.last_method = ""
            session.last_action = ""
            session.previous_action_status = ""
            session.last_plugin = ""  # 清空插件
            session.in_progress_count = 0
            # score 保留（累计值）
            # 返回跟随对话，让下一轮重新选择方法
            return self._get_sfbt_prompt("跟随对话", "", session.last_score)

        # 检查 method 是否为空（跟随对话）
        if not package.selected_method:
            session.action = ""
            # 清空 last_method，下一轮重新调用快脑判断
            session.last_method = ""
            session.last_action = ""
            session.previous_action_status = ""
            return self._get_sfbt_prompt("跟随对话", "", session.last_score)

        # 更新 action
        # action = 当前要执行的内部动作
        if action_status == "in_progress":
            # in_progress: 保持上一轮的action不变，继续执行当前action
            # 但如果session.action为空（第一轮），则需要初始化
            if not session.action:
                session.action = package.action if package.action else ""
            action_to_use = session.action  # 使用当前session中的action（上一轮设置的）
            # 【调试】打印action保持
            print(f"\n🐛 [DEBUG] Session状态保持 (in_progress):")
            print(f"  - action_status: 'in_progress'")
            print(f"  - session.action: '{session.action}'")
            print(f"  - package.action: '{package.action}'")
        elif action_status == "completed":
            # completed: 更新为新的action（进入下一个action）
            session.action = package.action if package.action else ""
            action_to_use = session.action
            # 【调试】打印action更新
            print(f"\n🐛 [DEBUG] Session状态更新 (completed):")
            print(f"  - action_status: 'completed'")
            print(f"  - package.action: '{package.action}'")
            print(f"  - session.action更新为: '{session.action}'")
        else:
            # abandoned 或其他情况
            session.action = ""
            action_to_use = ""
            # 【调试】打印清空操作
            print(f"\n🐛 [DEBUG] Session状态清空:")
            print(f"  - action_status: '{action_status}'")
            print(f"  - session.action清空为: ''")

        # 更新 last_action（在 action_status 处理之后）
        if session.action:
            session.last_action = session.action

        # 拼接: persona + 方法子步骤 + plugin + intend + score
        return self._get_sfbt_prompt(
            package.selected_method,
            package.strategic_intend,
            package.score,
            action_to_use,
            package.plugin
        )

    def _get_sfbt_prompt(self, method: str, intend: str, score: int = 0, action: str = "", plugin: str = "") -> str:
        """
        获取 SFBT prompt：persona + 方法子步骤 + plugin + intend + score
        """
        # 加载 sfbt_prompt.yaml
        sfbt_prompt_path = Path(__file__).parent.parent / "sfbt" / "sfbt_prompt.yaml"

        if not sfbt_prompt_path.exists():
            return f"[请填写 sfbt_prompt.yaml] 方法: {method}, intend: {intend}"

        with open(sfbt_prompt_path, 'r', encoding='utf-8') as f:
            import yaml
            sfbt_data = yaml.safe_load(f) or {}

        # 获取 persona
        persona = config.prompts.get("persona") or ""

        # 获取方法数据（嵌套结构）
        method_data = sfbt_data.get("methods", {}).get(method, {})

        # 构建方法 prompt
        method_prompt = self._build_method_prompt(method, method_data, action)

        # 拼接: persona + 方法子步骤 + plugin + intend + score
        result = f"{persona}\n\n{method_prompt}"

        # 如果有插件，叠加插件 prompt
        if plugin and plugin != "无":
            plugin_data = sfbt_data.get("methods", {}).get(plugin, {})
            plugin_prompt = self._build_method_prompt(plugin, plugin_data, "")
            if plugin_prompt:
                result += f"\n\n【plugin】{plugin_prompt}"

        if action:
            result += f"\n\n当前内部动作：{action}"
        if intend:
            result += f"\n\n本轮指导：{intend}"
        if score:
            result += f"\n\n当前累计分数：{score}"

        return result

    def _build_method_prompt(self, method: str, method_data: dict, step_name: str) -> str:
        """
        构建方法 prompt

        支持两种格式：
        1. 嵌套结构：core_task + steps
        2. 直接内容：直接是字符串或顶层字段（用于插件）

        Args:
            method: 方法名
            method_data: 方法数据字典
            step_name: 内部动作名称，如果为空则返回完整方法 prompt
        """
        # 处理字符串类型的方法数据（如关系问题、赞美等直接是字符串）
        if isinstance(method_data, str):
            # 直接返回字符串内容
            return method_data

        # 检查是否是直接内容格式（没有 core_task 和 steps）
        has_core_task = "core_task" in method_data
        has_steps = "steps" in method_data

        if not has_core_task and not has_steps:
            # 直接内容格式（插件），把所有顶层字段拼接起来
            parts = []
            for key, value in method_data.items():
                if value:  # 跳过空值
                    parts.append(f"{value}")
            return "\n".join(parts) if parts else ""

        # 获取核心任务
        core_task = method_data.get("core_task", "")

        # 获取步骤
        steps = method_data.get("steps", {})

        # 如果有指定步骤，格式化该步骤
        if step_name and step_name in steps:
            step_data = steps[step_name]
            purpose = step_data.get("purpose", "")
            example = step_data.get("example", "")
            warning = step_data.get("warning", "")
            evaluation = step_data.get("evaluation", "")

            prompt = f"## 核心任务\n{core_task}\n\n## 当前步骤：{step_name}\n"
            if purpose:
                prompt += f"目的：{purpose}\n"
            if example:
                prompt += f"示例：{example}\n"
            if warning:
                prompt += f"注意：{warning}\n"
            if evaluation:
                prompt += f"评价标准：{evaluation}\n"
            return prompt
        else:
            # 没有指定步骤，返回完整方法（只包含 core_task）
            # 跟随对话特殊处理
            if method == "跟随对话":
                default_step = steps.get("默认", {})
                example = default_step.get("example", "")
                warning = default_step.get("warning", "")
                evaluation = default_step.get("evaluation", "")
                prompt = f"## 核心任务\n{core_task}\n"
                if example:
                    prompt += f"操作指引：{example}\n"
                if warning:
                    prompt += f"禁止事项：{warning}\n"
                if evaluation:
                    prompt += f"评价标准：{evaluation}\n"
                return prompt
            else:
                # 其他方法，如果没有指定步骤，返回提示
                if steps:
                    available_steps = ", ".join(steps.keys())
                    return f"## 核心任务\n{core_task}\n\n可用的内部动作：{available_steps}"
                else:
                    return f"## 核心任务\n{core_task}"

    def _get_bridging_instruction(self, dimension: AssessmentDimension) -> str:
        return config.prompts.get("bridging", dimension.name)

