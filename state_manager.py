from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Deque
from collections import deque
from datetime import datetime
import config
from dataclasses import asdict

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
    PASSIVE = 1      # 回避/被动："不知道"、"还好"、"没啥想说的"、省略号、极短回复
    DEFENSIVE = 2    # 怀疑/理性化："问这个有用吗？"、"你是机器人不懂我"、"这太傻了"
    HOSTILE = 3      # 敌意/拒绝："关你屁事"、"不想说了"、"闭嘴"、攻击性语言

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

    def mark_soft_update(self, turn: int):
        """
        软更新：仅更新时间戳并设置默认风险分数，用于模糊回应。
        逻辑：虽然问了但没评估出结果，但为了区分"完全未评估"(score=1)，
        设为2分(轻微风险)，表示"问过了但结果不确定"。
        注意：不将 is_assessed 设为 True，保留"未确认评估"的状态。
        """
        self.last_updated_turn = turn
        self.score = 2  # 设置为轻微风险，区分未评估状态

@dataclass
class ResistanceState:
    """用户阻力状态 - 独立于心理风险评估的对话质量指标"""
    level: ResistanceLevel = ResistanceLevel.PASSIVE
    evidence: str = ""  # 触发的原话
    last_updated_turn: int = 0
    consecutive_count: int = 0  # 连续触发次数
    llm_confirmed: bool = False  # 是否经过LLM确认（首次正则检测后需8B确认）

    def reset(self):
        """重置阻力状态（用户积极响应时）"""
        self.consecutive_count = 0
        self.llm_confirmed = False
        self.evidence = ""

@dataclass
class AssessmentState:
    """整体评估状态 (The Brain 的核心记忆) """
    dimensions: Dict[AssessmentDimension, DimensionState] = field(default_factory=dict)
    
    def __post_init__(self):
        # 初始化所有维度
        for dim in AssessmentDimension:
            self.dimensions[dim] = DimensionState(dimension=dim)

    def update_score(self, dimension: AssessmentDimension, score: int, evidence: str, current_turn: int):
        """更新分数与依据 [cite: 39]"""
        if dimension in self.dimensions:
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

    def get_idle_dimension(self, current_turn: int) -> Optional[AssessmentDimension]:
        """获取超过闲置阈值的维度 """
        # 按权重优先级检查闲置
        sorted_keys = sorted(
            config.ASSESSMENT_RULES.keys(),
            key=lambda k: config.ASSESSMENT_RULES[k]["weight"],
            reverse=True
        )
        
        for key in sorted_keys:
            dim_enum = AssessmentDimension[key]
            dim_state = self.dimensions[dim_enum]
            threshold = config.ASSESSMENT_RULES[key]["idle_threshold"]
            
            if (current_turn - dim_state.last_updated_turn) >= threshold:
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

@dataclass
class SessionState:
    """会话全局状态对象"""
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    turn_count: int = 0  # 对话轮数 [cite: 10]

    # 状态机核心
    current_stage: ConsultationStage = ConsultationStage.WARM_UP_SCAN
    risk_level: RiskLevel = RiskLevel.NORMAL
    assessment: AssessmentState = field(default_factory=AssessmentState)

    # 用户阻力状态（新增）
    resistance: ResistanceState = field(default_factory=ResistanceState)

    # 历史记录 (仅保留最近10轮)
    history: Deque[Message] = field(default_factory=lambda: deque(maxlen=config.HISTORY_WINDOW_SIZE))

    # 用于记录 SFBT 阶段是否已经完成了评量提问
    sfbt_scaling_done: bool = False

    # 用于记录 Brain 最后一次主动询问的维度，用于软更新逻辑
    last_targeted_dimension: Optional[AssessmentDimension] = None

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
                resistance_raw: str = ""):
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
            resistance_raw_output=resistance_raw
        )
        self.logs.append(log)

    def update_last_log_model_outputs(self, brain_risk_raw: str, brain_assessment_raw: str, resistance_raw: str = ""):
        """更新最后一条日志的慢速回路模型输出"""
        if self.logs:
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
