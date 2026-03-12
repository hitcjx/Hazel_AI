import os
from pathlib import Path
from enum import Enum
from typing import Dict, TypedDict, Any
import yaml

# =============================================================================
# API后端配置
# =============================================================================
class ModelBackend(Enum):
    """模型后端类型"""
    LOCAL = "local"      # 本地模型（transformers）
    VLLM = "vllm"        # vLLM 高并发引擎
    OPENAI = "openai"    # OpenAI格式API（包括兼容接口）

class APIConfig:
    """API配置 - 支持从环境变量读取"""

    # Avatar模型配置（14B 温柔对话）
    AVATAR_BACKEND = ModelBackend.OPENAI  # 默认本地
    AVATAR_API_KEY = os.getenv("AVATAR_API_KEY", "sk-uUaPPQMrETWan8TfHbBsdEebNCIQ6AZ3g581N0aI7hCsF52y")
    AVATAR_API_MODEL = os.getenv("AVATAR_API_MODEL", "qwen3-235b-a22b-instruct-2507")
    AVATAR_API_BASE_URL = os.getenv("AVATAR_API_BASE_URL", "https://api.chatanywhere.org/v1")  # 自定义endpoint

    # Brain模型配置（8B 逻辑判断）
    BRAIN_BACKEND = ModelBackend.OPENAI  # 使用API模式
    BRAIN_API_KEY = os.getenv("BRAIN_API_KEY", "sk-uUaPPQMrETWan8TfHbBsdEebNCIQ6AZ3g581N0aI7hCsF52y")
    BRAIN_API_MODEL = os.getenv("BRAIN_API_MODEL", "gpt-5-mini")
    BRAIN_API_BASE_URL = os.getenv("BRAIN_API_BASE_URL", "https://api.chatanywhere.org/v1")

    # Guard模型配置（1.7B 快速检查 - 本地模式）
    GUARD_BACKEND = ModelBackend.LOCAL
    GUARD_API_KEY = os.getenv("GUARD_API_KEY", "")
    GUARD_API_MODEL = os.getenv("GUARD_API_MODEL", "")
    GUARD_API_BASE_URL = os.getenv("GUARD_API_BASE_URL", "")

class MemoryConfig:
    """记忆系统配置 - mem0 Cloud"""

    # mem0 Cloud API 配置
    MEM0_API_KEY = os.getenv("MEM0_API_KEY", "m0-HUd2t4rT5uaZTZO4RnwmfAzeB5WIMXSfEtQMXRLj")
    MEM0_API_BASE_URL = os.getenv("MEM0_API_BASE_URL", "https://api.mem0.ai/v1")

    # 用于 mem0 的 embedding（使用 chatanywhere API）
    MEMORY_QUICK_API_KEY = os.getenv("MEMORY_QUICK_API_KEY", APIConfig.AVATAR_API_KEY)
    MEMORY_QUICK_BASE_URL = os.getenv("MEMORY_QUICK_BASE_URL", APIConfig.AVATAR_API_BASE_URL)

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path(__file__).parent.absolute()
PROMPTS_YAML_PATH = BASE_DIR / "prompts.yaml"

class PromptManager:
    """单例模式管理 Prompt 加载与渲染"""
    def __init__(self, path: Path):
        self.data = self._load_yaml(path)
        
        # 预加载量表内容 (Rubric)
        self.rubric_content = self._load_rubric_fallback()

    def _load_yaml(self, path) -> Dict[str, Any]:
        if not path.exists():
            print(f"[Warning] Prompt file not found: {path}")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_rubric_fallback(self) -> str:
        """优先从 prompts.yaml 读 rubric_content，其次读 assessment_rubric.txt"""
        # 1. 优先从 yaml 中读 rubric_content
        if "rubric_content" in self.data:
            content = self.data.get("rubric_content")
            if content:
                return content
        
        return "Assessment Rubric Missing."

    def get(self, *keys, **kwargs) -> str:
        """
        获取 prompt，支持嵌套 key 和 .format() 参数
        例如: config.prompts.get('stages', 'warmup')
        """
        curr = self.data
        try:
            for k in keys:
                curr = curr[k]
            if isinstance(curr, str):
                return curr.format(**kwargs)
            return str(curr)
        except (KeyError, AttributeError, ValueError) as e:
            return f"[Error: Prompt key {'->'.join(keys)} failed: {str(e)}]"

# 全局单例
prompts = PromptManager(PROMPTS_YAML_PATH)

# =============================================================================
# 模型配置
# =============================================================================
class ModelConfig:
    """双模型配置"""
    # Avatar负责生成温柔的回复 [cite: 1]
    AVATAR_MODEL_PATH = "OpenPipe/Qwen3-14B-Instruct"
    # Brain负责逻辑判断和指令生成 [cite: 7]
    BRAIN_MODEL_PATH = "Qwen/Qwen3-8B"
    # Guard负责初步安全过滤
    GUARD_MODEL_PATH = "Qwen/Qwen3-1.7B-Base"

    # LoRA适配器配置
    USE_LORA_FOR_BRAIN = True  # 是否为Brain模型使用LoRA适配器
    BRAIN_LORA_ADAPTER_PATH = str(BASE_DIR / "checkpoint-210")  # Brain模型的LoRA适配器路径（绝对路径）
    USE_LORA_FOR_GUARD = True  # 是否为Guard模型使用LoRA适配器
    GUARD_LORA_ADAPTER_PATH = str(BASE_DIR / "checkpoint_guard")  # Guard模型的LoRA适配器路径（绝对路径）

    # 生成参数配置
    DEFAULT_TEMP = 0.7
    BRAIN_TEMP = 0.2  # Brain需要逻辑严谨，温度较低

    # vLLM 配置（用于高并发场景）
    # 服务器环境可设置为 True，利用 vLLM 的连续批处理提升并发性能
    USE_VLLM_FOR_BRAIN = False   # Brain 模型是否使用 vLLM
    USE_VLLM_FOR_GUARD = False   # Guard 模型是否使用 vLLM

    # vLLM 引擎参数
    VLLM_MAX_MODEL_LEN = 2048        # 最大序列长度
    VLLM_GPU_MEMORY_UTIL = 0.9       # GPU 显存利用率（服务器可设高一些）
    VLLM_QUANTIZATION = None         # 量化方式: None, "bitsandbytes", "awq", "gptq"

    # Transformers 本地模型配置（备选方案）
    USE_4BIT_QUANTIZATION = True     # 是否使用 4-bit 量化（BitsAndBytes）

CRISIS_KEYWORDS = [
    # 直接行为词 - 使用灵活匹配
    r"自杀", r"自尽", r"轻生", r"上吊", r"跳楼", r"跳河", r"跳海", r"割腕",
    r"服药", r"吃药睡过去", r"安眠药", r" overdose", r"药量加倍",

    # 意图表达 - 修复：使用.{0,2}匹配中间插入的字
    r"想.{0,2}死",  # 匹配：想死、想去死、想要死、想着死
    r"不想活", r"活够了", r"死了算了", r"一了百了",
    r"结束生命", r"自我了断", r"寻死", r"求死",

    # 解脱/价值否定
    r"死是解脱", r"死了就解脱了", r"没人会在乎我死了",
    r"世界没有我更好", r"我是累赘", r"我不想在这个世界上了",

    # 告别信号
    r"遗书", r"告别信", r"最后一条", r"永别了", r"再也不见",
    r"来生再见", r"下辈子见", r"跟大家说再见"
]

# =============================================================================
# 心理评估维度配置
# =============================================================================
class DimensionConfig(TypedDict):
    weight: float
    idle_threshold: int
    name_cn: str

# 依据 定义的维度权重与闲置阈值
ASSESSMENT_RULES: Dict[str, DimensionConfig] = {
    "PHYSICAL_FUNCTION": {
        "weight": 0.30, 
        "idle_threshold": 5, 
        "name_cn": "生理功能"
    },
    "COGNITIVE_DISTORTION": {
        "weight": 0.25, 
        "idle_threshold": 8, 
        "name_cn": "认知偏差"
    },
    "ACADEMIC_PRESSURE": {
        "weight": 0.20, 
        "idle_threshold": 10, 
        "name_cn": "学业压力"
    },
    "EMOTIONAL_RESILIENCE": {
        "weight": 0.15, 
        "idle_threshold": 12, 
        "name_cn": "情绪韧度"
    },
    "SOCIAL_SUPPORT": {
        "weight": 0.10, 
        "idle_threshold": 12, 
        "name_cn": "社会支持"
    }
}

# =============================================================================
# 会话管理配置
# =============================================================================
HISTORY_WINDOW_SIZE = 10  # 历史对话窗口大小

# 待办栈机制配置
DIMENSION_RETRY_DELAY = 5  # 维度重试间隔轮数
MAX_DIMENSION_RETRIES = 2  # 最大重试次数

# =============================================================================
# 阶段流转配置（时间 OR 轮数，任一条件满足即触发）
# =============================================================================
# 总会话时长
MAX_SESSION_TIME_MINUTES = 40  # 总时长40分钟

# 各阶段时间占比（总和应为1.0）
STAGE_TIME_ALLOCATION = {
    "WARM_UP_SCAN": 0.10,           # 10% = 4分钟
    "EMPATHY_DEEP_DIVE": 0.50,      # 50% = 20分钟
    "REFRAMING_SFBT": 0.35,         # 35% = 14分钟
    "CLOSING_EMPOWERMENT": 0.05     # 5% = 2分钟
}

# 各阶段的轮数阈值（作为时间的补充判断，OR逻辑）
STAGE_TURN_THRESHOLDS = {
    "warmup_to_deepdive": {
        "min_turns": 2,     # 至少2轮才能切换
        "max_turns": 5,     # 最多5轮强制切换（防止时间未到但已充分破冰）
    },
    "deepdive_to_sfbt": {
        "min_turns": 25,    # 至少深挖25轮（用户要求）
    },
    "sfbt_to_closing": {
        "max_turns": 15,    # 最多15轮SFBT（防止过度干预）
    }
}

# =============================================================================
# 阻力状态管理配置
# =============================================================================
RESISTANCE_RESET_THRESHOLD = 2  # 连续N次积极回应才完全重置阻力状态

# =============================================================================
# 危机状态管理配置
# =============================================================================
CRISIS_STABILIZATION_REQUIRED = 3  # 连续N轮稳定（risk<CRISIS）才退出危机模式

# =============================================================================
# API 调用稳定性配置
# =============================================================================
RETRY_CONFIG = {
    "max_retries": 3,           # 最大重试次数
    "retry_delay": 0.5,         # 初始重试延迟（秒）
    "retry_multiplier": 1.5,    # 指数退避倍数
    "timeout": 20,              # API 调用超时（秒）- 降低以匹配前端60s超时
} 