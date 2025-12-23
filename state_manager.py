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
    
    def add_log(self, user_msg: str, instruction: str, response: str):
        """记录这一轮的详细数据"""
        # 创建评分快照
        scores = {k.name: v.score for k, v in self.assessment.dimensions.items()}
        
        log = TurnLog(
            turn_id=self.turn_count,
            user_input=user_msg,
            brain_instruction=instruction,
            avatar_response=response,
            risk_level_snapshot=self.risk_level.name,
            assessment_snapshot=scores
        )
        self.logs.append(log)

    def export_logs(self) -> dict:
        """导出为字典格式"""
        return {
            "session_id": self.session_id,
            "total_turns": self.turn_count,
            "final_risk": self.risk_level.name,
            "dialogue_logs": [asdict(log) for log in self.logs]
        }

    @property
    def is_crisis_mode(self) -> bool:
        """判断是否处于红色危机模式 [cite: 14]"""
        return self.risk_level == RiskLevel.CRISIS
    
