from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Optional, Dict, Deque, Tuple
from collections import deque
from datetime import datetime
import core.config as config

# =============================================================================
# 枚举定义 (Enums)
# =============================================================================

class AssessmentDimension(Enum):
    """五大评估维度 """
    PHYSICAL_FUNCTION = "PHYSICAL_FUNCTION"     # 生理功能
    COGNITIVE_DISTORTION = "COGNITIVE_DISTORTION" # 认知偏差
    ACADEMIC_PRESSURE = "ACADEMIC_PRESSURE"     # 学业压力
    EMOTIONAL_RESILIENCE = "EMOTIONAL_RESILIENCE" # 情绪韧度
    SOCIAL_SUPPORT = "SOCIAL_SUPPORT"           # 社会支持

class RiskLevel(Enum):
    """CAMS/PFA 风险等级熔断机制 [cite: 12, 147]"""
    NORMAL = 0          # 无风险
    IDEATION = 1        # 黄色预警：模糊的死亡意念 [cite: 13]
    CRISIS = 2          # 红色警报：明确计划/工具 -> 触发生命锚点模式 [cite: 14, 16]

class ResistanceLevel(Enum):
    """用户阻力等级 - 评估对话参与度与防御机制"""
    NONE = 0         # 无阻力：正常对话，积极回应
    PASSIVE = 1     # 回避/被动："不知道"、"还好"、"没啥想说的"、省略号、极短回复
    DEFENSIVE = 2   # 怀疑/理性化："问这个有用吗？"、"你是机器人不懂我"、"这太傻了"
    HOSTILE = 3     # 敌意/拒绝："关你屁事"、"不想说了"、"闭嘴"、攻击性语言

class ConsultationStage(Enum):
    """对话阶段流转 """
    WARM_UP_SCAN = "warmup"        # 破冰与扫描
    EMPATHY_DEEP_DIVE = "deep_dive" # 共情与深挖 (最复杂逻辑)
    REFRAMING_SFBT = "reframing"   # 重构与干预
    CLOSING_EMPOWERMENT = "closing" # 结束与赋能

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    THOUGHT = "thought"  # 内部思考过程记录

# =============================================================================
# 数据类定义 (Dataclasses)
# =============================================================================

@dataclass
class DimensionState:
    """单个维度的评估状态"""
    dimension: AssessmentDimension
    score: int = 1  # 默认为1 (健康/低风险) [cite: 145]
    evidence: str = ""  # 评分依据 (Thought过程) [cite: 33]
    last_updated_turn: int = 0  # 用于计算闲置轮数 [cite: 40]
    is_assessed: bool = False  # 是否已被评估过（独立于score值）
    score_history: List[dict] = field(default_factory=list)  # 评分历史 [(turn, score, evidence), ...]

@dataclass
class ResistanceState:
    """用户阻力状态 - 独立于心理风险评估的对话质量指标"""
    level: ResistanceLevel = ResistanceLevel.NONE
    evidence: str = ""  # 触发的原话
    last_updated_turn: int = 0
    consecutive_count: int = 0  # 连续触发次数
    llm_confirmed: bool = False  # 是否经过LLM确认（首次正则检测后需8B确认）

    # 缓冲窗口机制：防止过于敏感的重置
    positive_responses_after_resistance: int = 0  # 阻力后的积极回应次数
    reset_threshold: int = 2  # 需要连续N次积极回应才完全重置

    def reset(self):
        """重置阻力状态（用户积极响应时）"""
        self.consecutive_count = 0
        self.llm_confirmed = False
        self.evidence = ""
        self.positive_responses_after_resistance = 0

@dataclass
class AssessmentState:
    """整体评估状态 (The Brain 的核心记忆) """
    dimensions: Dict[AssessmentDimension, DimensionState] = field(default_factory=dict)
    
    def __post_init__(self):
        # 初始化所有维度
        for dim in AssessmentDimension:
            self.dimensions[dim] = DimensionState(dimension=dim)

    def update_score(self, dimension: AssessmentDimension, score: int, evidence: str, current_turn: int):
        """更新分数与依据，同时追加到历史"""
        if dimension in self.dimensions:
            # 追加到历史
            self.dimensions[dimension].score_history.append({
                "turn": current_turn,
                "score": score,
                "evidence": evidence
            })
            # 更新当前值
            self.dimensions[dimension].score = score
            self.dimensions[dimension].evidence = evidence
            self.dimensions[dimension].last_updated_turn = current_turn
            self.dimensions[dimension].is_assessed = True  # 标记为已评估

    def get_highest_risk_dimension(self) -> Optional[AssessmentDimension]:
        """获取当前分数最高(风险最大)的维度，用于Deep Dive阶段的追问 [cite: 31]"""
        # 简单逻辑：返回分数最高的维度，若分数相同则按权重排序
        sorted_dims = sorted(
            self.dimensions.values(),
            key=lambda x: (x.score, config.ASSESSMENT_RULES[x.dimension.name]["weight"]),
            reverse=True
        )
        if sorted_dims and sorted_dims[0].score > 1:
            return sorted_dims[0].dimension
        return None

    def get_idle_dimension(self, current_turn: int, pending_dimensions: List[Tuple[AssessmentDimension, int]],
                           dimension_retry_count: Dict[AssessmentDimension, int],
                           retry_delay: int = 5, max_retries: int = 2) -> Optional[AssessmentDimension]:
        """
        获取需要评估的维度（支持待办栈机制）

        优先级：
        1. 从待办栈中取出到期的维度（间隔足够轮数）
        2. 按权重检查从未评估过的维度
        3. 返回None表示所有维度都已评估或重试次数用尽

        Args:
            current_turn: 当前轮数
            pending_dimensions: 待办栈 [(维度, 入栈轮数), ...]
            dimension_retry_count: 重试次数字典
            retry_delay: 重试延迟轮数
            max_retries: 最大重试次数
        """
        # 1. 优先检查待办栈（按权重排序，高权重优先）
        if pending_dimensions:
            # 按权重排序待办栈
            sorted_pending = sorted(
                pending_dimensions,
                key=lambda x: config.ASSESSMENT_RULES[x[0].name]["weight"],
                reverse=True
            )

            for dim, pending_turn in sorted_pending:
                # 检查是否到期可以重试
                if (current_turn - pending_turn) >= retry_delay:
                    # 检查重试次数是否用尽
                    retry_count = dimension_retry_count.get(dim, 0)
                    if retry_count < max_retries:
                        return dim

        # 2. 检查从未评估过的维度（按权重优先级）
        sorted_keys = sorted(
            config.ASSESSMENT_RULES.keys(),
            key=lambda k: config.ASSESSMENT_RULES[k]["weight"],
            reverse=True
        )

        for key in sorted_keys:
            dim_enum = AssessmentDimension[key]
            dim_state = self.dimensions[dim_enum]

            # 只返回完全未被触及的维度
            if not dim_state.is_assessed and dim_state.last_updated_turn == 0:
                return dim_enum

        return None
    
    def to_prompt_block(self) -> str:
        """生成用于 System Prompt 的状态文本块"""
        lines = ["【当前心理评估状态】"]
        for dim_state in self.dimensions.values():
            lines.append(str(dim_state))
        return "\n".join(lines)

@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TurnLog:
    """记录单轮对话的完整元数据"""
    turn_id: int
    user_input: str
    brain_instruction: str  # Brain 给的指令
    avatar_response: str
    risk_level_snapshot: str # 当时的风险等级
    assessment_snapshot: Dict[str, int] # 当时的评分快照
    # 模型原始输出 (用于调试)
    guard_raw_output: str = ""  # 1.7B Guard 模型的原始输出
    brain_risk_raw_output: str = ""  # 8B Brain 风险分析的原始输出
    brain_assessment_raw_output: str = ""  # 8B Brain 维度评估的原始输出
    # 阻力相关字段（新增）
    resistance_level_snapshot: str = ""  # 当时的阻力等级
    resistance_count_snapshot: int = 0  # 连续触发次数
    resistance_raw_output: str = ""  # Brain 阻力判断的原始输出
    # SFBT 策略原始输出（JSON格式）
    sfbt_raw_output: str = ""  # 策略器的完整JSON输出

@dataclass
class SessionState:
    """会话全局状态对象"""
    session_id: str
    user_id: str  # 【新增】用户ID，用于关联长期记忆
    start_time: datetime = field(default_factory=datetime.now)
    turn_count: int = 0  # 对话轮数 [cite: 10]
    sfbt_turn_count: int = 0  # SFBT阶段轮数

    # 状态机核心
    current_stage: ConsultationStage = ConsultationStage.WARM_UP_SCAN
    risk_level: RiskLevel = RiskLevel.NORMAL
    assessment: AssessmentState = field(default_factory=AssessmentState)

    # 用户阻力状态（新增）
    resistance: ResistanceState = field(default_factory=ResistanceState)

    # 历史记录 (仅保留最近10轮)
    history: Deque[Message] = field(default_factory=lambda: deque(maxlen=config.HISTORY_WINDOW_SIZE))

    # 用于记录 Brain 最后一次主动询问的维度，用于软更新逻辑
    last_targeted_dimension: Optional[AssessmentDimension] = None

    # 待办栈机制：用于延迟重试未能评估的维度
    pending_dimensions: List[Tuple[AssessmentDimension, int]] = field(default_factory=list)
    # 记录每个维度的重试次数
    dimension_retry_count: Dict[AssessmentDimension, int] = field(default_factory=dict)

    # CRISIS状态追踪
    crisis_triggered: bool = False  # 是否触发过CRISIS
    crisis_trigger_time: Optional[datetime] = None  # CRISIS首次触发时间
    crisis_stabilization_turns: int = 0  # 危机稳定化观察轮数（连续N轮非危机才退出）

    # SFBT阶段追踪
    last_method: str = ""  # 上一轮使用的SFBT方法
    last_current_module: str = "S1_合作构建"  # 上一轮的模块，默认S1
    last_score: int = 0  # 上一轮的累计分数
    last_plugin: str = ""  # 上一轮使用的插件
    last_action: str = ""  # 上一轮的 action
    previous_action_status: str = ""  # 上一轮的 action_status
    action: str = ""  # 当前要执行的内部动作
    in_progress_count: int = 0  # 连续 in_progress 计数（达到3次自动放弃）

    @property
    def remaining_minutes(self) -> float:
        """计算剩余时间（分钟）"""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60.0
        return max(0, config.MAX_SESSION_TIME_MINUTES - elapsed)

    # 详细日志列表
    logs: List[TurnLog] = field(default_factory=list)

    def add_message(self, role: MessageRole, content: str):
        """添加消息并自增轮数"""
        self.history.append(Message(role=role, content=content))
        if role == MessageRole.USER:
            self.turn_count += 1
            
    def get_history_text(self) -> str:
        """格式化输出历史对话供Prompt使用"""
        return "\n".join([f"{msg.role.value}: {msg.content}" for msg in self.history])
    
    def add_log(self, user_msg: str, instruction: str, response: str,
                guard_raw: str = "", brain_risk_raw: str = "", brain_assessment_raw: str = "",
                resistance_raw: str = "", sfbt_raw: str = ""):
        """记录这一轮的详细数据"""
        # 创建评分快照
        scores = {k.name: v.score for k, v in self.assessment.dimensions.items()}

        log = TurnLog(
            turn_id=self.turn_count,
            user_input=user_msg,
            brain_instruction=instruction,
            avatar_response=response,
            risk_level_snapshot=self.risk_level.name,
            assessment_snapshot=scores,
            guard_raw_output=guard_raw,
            brain_risk_raw_output=brain_risk_raw,
            brain_assessment_raw_output=brain_assessment_raw,
            resistance_level_snapshot=self.resistance.level.name,
            resistance_count_snapshot=self.resistance.consecutive_count,
            resistance_raw_output=resistance_raw,
            sfbt_raw_output=sfbt_raw
        )
        self.logs.append(log)

    def update_last_log_model_outputs(self, brain_risk_raw: str, brain_assessment_raw: str, resistance_raw: str = "", turn_id: int = None):
        """
        更新最后一条日志的慢速回路模型输出

        Args:
            brain_risk_raw: Brain风险分析原始输出
            brain_assessment_raw: Brain评估原始输出
            resistance_raw: 阻力判断原始输出
            turn_id: 指定要更新的轮次ID（防止并发竞态）
        """
        if not self.logs:
            return

        # 修复并发竞态：如果指定了turn_id，查找对应的日志条目
        if turn_id is not None:
            # 从后往前查找对应的turn_id
            for i in range(len(self.logs) - 1, -1, -1):
                if self.logs[i].turn_id == turn_id:
                    self.logs[i].brain_risk_raw_output = brain_risk_raw
                    self.logs[i].brain_assessment_raw_output = brain_assessment_raw
                    if resistance_raw:  # 只有非空时才更新
                        self.logs[i].resistance_raw_output = resistance_raw
                    return
        else:
            # 未指定turn_id，保持原有逻辑（更新最后一条）
            self.logs[-1].brain_risk_raw_output = brain_risk_raw
            self.logs[-1].brain_assessment_raw_output = brain_assessment_raw
            if resistance_raw:
                self.logs[-1].resistance_raw_output = resistance_raw

    def export_logs(self) -> dict:
        """导出为字典格式"""
        return {
            "session_id": self.session_id,
            "total_turns": self.turn_count,
            "final_risk": self.risk_level.name,
            "final_resistance_level": self.resistance.level.name,
            "total_resistance_triggers": self.resistance.consecutive_count,
            "dialogue_logs": [asdict(log) for log in self.logs]
        }

    @property
    def is_crisis_mode(self) -> bool:
        """判断是否处于红色危机模式 [cite: 14]"""
        return self.risk_level == RiskLevel.CRISIS

    def update_resistance(self, level: ResistanceLevel, evidence: str, turn: int):
        """
        更新阻力状态

        Args:
            level: 阻力等级
            evidence: 触发的原话
            turn: 当前轮次
        """
        self.resistance.level = level
        self.resistance.evidence = evidence
        self.resistance.last_updated_turn = turn
        self.resistance.consecutive_count += 1

    def reset_resistance(self):
        """重置阻力状态（用户积极响应时调用）"""
        self.resistance.reset()
