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
    OPENAI = "openai"    # OpenAI格式API（包括兼容接口）

class APIConfig:
    """API配置 - 支持从环境变量读取"""

    # Avatar模型配置（14B 温柔对话）
    AVATAR_BACKEND = ModelBackend.LOCAL  # 默认本地
    AVATAR_API_KEY = os.getenv("AVATAR_API_KEY", "")
    AVATAR_API_MODEL = os.getenv("AVATAR_API_MODEL", "gpt-4")
    AVATAR_API_BASE_URL = os.getenv("AVATAR_API_BASE_URL", "")  # 自定义endpoint

    # Brain模型配置（8B 逻辑判断）
    BRAIN_BACKEND = ModelBackend.LOCAL  # 默认本地
    BRAIN_API_KEY = os.getenv("BRAIN_API_KEY", "")
    BRAIN_API_MODEL = os.getenv("BRAIN_API_MODEL", "gpt-3.5-turbo")
    BRAIN_API_BASE_URL = os.getenv("BRAIN_API_BASE_URL", "")

    # Guard模型配置（1.7B 快速检查 - 建议保持本地）
    GUARD_BACKEND = ModelBackend.LOCAL
    GUARD_API_KEY = os.getenv("GUARD_API_KEY", "")
    GUARD_API_MODEL = os.getenv("GUARD_API_MODEL", "")
    GUARD_API_BASE_URL = os.getenv("GUARD_API_BASE_URL", "")

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
        例如: config.prompts.get('sfbt', 'coping', score=3)
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
    
    # 生成参数配置
    DEFAULT_TEMP = 0.7
    BRAIN_TEMP = 0.2  # Brain需要逻辑严谨，温度较低

CRISIS_KEYWORDS = [
    # 直接行为词
    r"自杀", r"自尽", r"轻生", r"上吊", r"跳楼", r"跳河", r"跳海", r"割腕",
    r"服药", r"吃药睡过去", r"安眠药", r" overdose", r"药量加倍",

    # 意图表达
    r"想死", r"不想活", r"活够了", r"死了算了", r"一了百了",
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

# 风险等级阈值
HISTORY_WINDOW_SIZE = 10  # 
MAX_SESSION_TIME_MINUTES = 40 # 