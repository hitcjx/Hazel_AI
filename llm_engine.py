import json
import re
from typing import Dict, List, Optional, Iterator
from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import config
from config import ModelConfig, ModelBackend
from state_manager import AssessmentState, RiskLevel, AssessmentDimension, ResistanceLevel
import threading
from threading import Thread

# =============================================================================
# 用户阻力检测模式（正则）
# =============================================================================
RESISTANCE_PATTERNS = {
    ResistanceLevel.PASSIVE: [
        r"^不知道\.?$",
        r"^还好吧?\.?$",
        r"^没啥\.?$",
        r"^没什么\.?$",
        r"^随便\.?$",
        r"^无所谓\.?$",
        r"^\.{2,}$",  # 省略号（2个及以上点）
        r"^…+$",      # 中文省略号
        r"^.{1,3}$",  # 1-3字极短回复
        r"^嗯{1,3}\.?$",  # 嗯、嗯嗯
        r"^哦{1,3}\.?$",  # 哦、哦哦
    ],
    ResistanceLevel.DEFENSIVE: [
        r"有用吗",
        r"没用",
        r"机器人",
        r"AI",
        r"不懂我",
        r"不理解",
        r"太傻",
        r"没意义",
        r"瞎问",
        r"问这个干嘛",
        r"你懂什么",
    ],
    ResistanceLevel.HOSTILE: [
        r"关你.*事",
        r"不想说",
        r"别问了",
        r"闭嘴",
        r"滚",
        r"烦死",
        r"恶心",
        r"去死",
        r"傻[Xx逼比]",
        r"妈的",
        r"[操草艹]",
    ]
}

# =============================================================================
# 模型适配器抽象层
# =============================================================================
class ModelAdapter(ABC):
    """统一的模型接口 - 支持本地模型和API调用"""

    @abstractmethod
    def generate(self, messages: List[dict], max_tokens: int, temperature: float) -> str:
        """同步生成"""
        pass

    @abstractmethod
    def generate_stream(self, messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        """流式生成"""
        pass

# =============================================================================
# 本地模型适配器（Transformers + BitsAndBytes）
# =============================================================================
class LocalModelAdapter(ModelAdapter):
    """本地模型适配器 - 使用 transformers 库加载本地模型"""

    def __init__(self, model_path: str, gpu_lock: threading.Lock):
        """
        初始化本地模型

        Args:
            model_path: 模型路径（HuggingFace格式）
            gpu_lock: GPU互斥锁（多模型共享显存时使用）
        """
        self.model_path = model_path
        self.gpu_lock = gpu_lock

        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        print(f"Loading Local Model: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
        print(f"✓ Model loaded: {model_path}")

    def generate(self, messages: List[dict], max_tokens: int, temperature: float) -> str:
        """同步生成"""
        # 构建 Prompt
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])

        # 加锁生成
        with self.gpu_lock:
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )

        # 只解码新生成的token（跳过输入prompt）
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def generate_stream(self, messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        """流式生成"""
        # 构建 Prompt
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 初始化流式传输器
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 在独立线程中运行 generate
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        def thread_target(**kwargs):
            with self.gpu_lock:  # 加锁
                self.model.generate(**kwargs)

        thread = threading.Thread(target=thread_target, kwargs=generation_kwargs)
        thread.start()

        # 返回生成器
        return streamer

# =============================================================================
# OpenAI 格式 API 适配器
# =============================================================================
class OpenAIAdapter(ModelAdapter):
    """OpenAI格式API适配器 - 支持OpenAI及兼容接口（vLLM, ollama等）"""

    def __init__(self, api_key: str, model_name: str, base_url: str = ""):
        """
        初始化 OpenAI API 客户端

        Args:
            api_key: API密钥
            model_name: 模型名称（如 gpt-4, gpt-3.5-turbo）
            base_url: 自定义 API endpoint（可选，用于兼容接口）
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI adapter requires 'openai' package. "
                "Install it with: pip install openai"
            )

        self.model_name = model_name

        # 初始化客户端
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        print(f"✓ OpenAI API initialized: {model_name} (endpoint: {base_url or 'default'})")

    def generate(self, messages: List[dict], max_tokens: int, temperature: float) -> str:
        """同步生成"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[OpenAI API Error] {str(e)}")
            return f"[API调用失败: {str(e)}]"

    def generate_stream(self, messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        """流式生成"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"[OpenAI API Stream Error] {str(e)}")
            yield f"[API流式调用失败: {str(e)}]"

class LLMEngine:
    """
    负责底层模型调用的适配层 - 支持本地模型和API调用。
    """
    def __init__(self, use_mock: bool = False):
        """
        初始化LLM引擎

        Args:
            use_mock: 是否使用mock模式（测试用）
        """
        self.use_mock = use_mock

        if self.use_mock:
            print("⚠️ Mock模式启用 - 不加载任何模型")
            self.avatar_adapter = None
            self.brain_adapter = None
            self.guard_adapter = None
            self.gpu_lock = None
            return

        # 初始化GPU锁（如果有本地模型）
        self.gpu_lock = self._init_gpu_lock()

        # 使用工厂方法创建适配器
        print("⏳ 初始化模型适配器...")
        self.avatar_adapter = self._create_adapter(
            backend=config.APIConfig.AVATAR_BACKEND,
            local_path=ModelConfig.AVATAR_MODEL_PATH,
            api_key=config.APIConfig.AVATAR_API_KEY,
            api_model=config.APIConfig.AVATAR_API_MODEL,
            api_base_url=config.APIConfig.AVATAR_API_BASE_URL,
            role="Avatar"
        )

        self.brain_adapter = self._create_adapter(
            backend=config.APIConfig.BRAIN_BACKEND,
            local_path=ModelConfig.BRAIN_MODEL_PATH,
            api_key=config.APIConfig.BRAIN_API_KEY,
            api_model=config.APIConfig.BRAIN_API_MODEL,
            api_base_url=config.APIConfig.BRAIN_API_BASE_URL,
            role="Brain"
        )

        self.guard_adapter = self._create_adapter(
            backend=config.APIConfig.GUARD_BACKEND,
            local_path=ModelConfig.GUARD_MODEL_PATH,
            api_key=config.APIConfig.GUARD_API_KEY,
            api_model=config.APIConfig.GUARD_API_MODEL,
            api_base_url=config.APIConfig.GUARD_API_BASE_URL,
            role="Guard"
        )

        print("✅ 所有模型适配器初始化完成\n")

    def _init_gpu_lock(self) -> Optional[threading.Lock]:
        """只有当有本地模型时才初始化GPU锁"""
        has_local = any([
            config.APIConfig.AVATAR_BACKEND == ModelBackend.LOCAL,
            config.APIConfig.BRAIN_BACKEND == ModelBackend.LOCAL,
            config.APIConfig.GUARD_BACKEND == ModelBackend.LOCAL
        ])
        return threading.Lock() if has_local else None

    def _create_adapter(self, backend: ModelBackend, local_path: str,
                       api_key: str, api_model: str, api_base_url: str,
                       role: str = "") -> ModelAdapter:
        """
        工厂方法：根据配置创建对应的适配器

        Args:
            backend: 后端类型（本地/OpenAI）
            local_path: 本地模型路径
            api_key: API密钥
            api_model: API模型名称
            api_base_url: API基础URL
            role: 模型角色（用于日志）

        Returns:
            ModelAdapter实例
        """
        role_prefix = f"[{role}] " if role else ""

        if backend == ModelBackend.LOCAL:
            print(f"{role_prefix}使用本地模型: {local_path}")
            return LocalModelAdapter(local_path, self.gpu_lock)

        elif backend == ModelBackend.OPENAI:
            print(f"{role_prefix}使用OpenAI API: {api_model}")
            return OpenAIAdapter(api_key, api_model, api_base_url)

        else:
            raise ValueError(f"不支持的后端类型: {backend}")

    def _call_model(self, prompt: str, is_brain: bool = True) -> str:
        """
        底层推理调用 - 使用适配器

        Args:
            prompt: 输入提示
            is_brain: True=Brain模型, False=Avatar模型

        Returns:
            模型生成的文本
        """
        adapter = self.brain_adapter if is_brain else self.avatar_adapter
        temp = ModelConfig.BRAIN_TEMP if is_brain else ModelConfig.DEFAULT_TEMP

        messages = [
            {
                "role": "system",
                "content": "You are a logic engine outputting JSON." if is_brain else "You are a helpful assistant."
            },
            {"role": "user", "content": prompt}
        ]

        return adapter.generate(
            messages=messages,
            max_tokens=512 if is_brain else 1024,
            temperature=temp
        )
    
    def fast_risk_check(self, text: str) -> tuple[bool, str]:
        """
        快速双重熔断机制
        1. 正则匹配 (0ms)
        2. Guard模型确认 (100ms)
        返回: (True (有风险), False (安全)), Guard模型原始输出
        """
        # Step 1: 正则初筛
        has_keyword = any(re.search(p, text) for p in config.CRISIS_KEYWORDS)
        if not has_keyword:
            return False, ""

        # Step 2: 小模型进一步确认 (防止"我笑死了"被误判)
        messages = [
            {"role": "system", "content": "你是一个安全审核员。判断用户输入是否有自杀风险。只回答'是'或'否'。"},
            {"role": "user", "content": text}
        ]

        response = self.guard_adapter.generate(
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )

        # 简单起见，直接找关键词
        has_risk = "是" in response or "Yes" in response
        return has_risk, response

    def quick_resistance_check(self, text: str) -> Optional[ResistanceLevel]:
        """
        快速正则阻力检测 - 三档模式

        Args:
            text: 用户输入文本

        Returns:
            检测到的阻力等级，未检测到返回None
        """
        text_stripped = text.strip()

        # 按严重程度倒序检查（从敌意到被动）
        for level in [ResistanceLevel.HOSTILE, ResistanceLevel.DEFENSIVE, ResistanceLevel.PASSIVE]:
            patterns = RESISTANCE_PATTERNS[level]
            for pattern in patterns:
                if re.search(pattern, text_stripped, re.IGNORECASE):
                    return level

        return None

    def analyze_resistance(self, text: str, history: str) -> tuple[bool, Optional[ResistanceLevel], str]:
        """
        LLM深度阻力判断 - 使用8B模型确认正则检测结果

        Args:
            text: 用户输入文本
            history: 对话历史

        Returns:
            (是否确认有阻力, 阻力等级, 模型原始输出)
        """
        prompt = config.prompts.get(
            "logic", "resistance_analysis",
            text=text,
            history=history[-200:]  # 只取最近的历史
        )

        response = self._call_model(prompt, is_brain=True)

        # 解析JSON输出
        parsed = self._parse_json_from_response(response)

        has_resistance = parsed.get("has_resistance", False)
        if not has_resistance:
            return False, None, response

        # 提取等级
        level_num = parsed.get("level", 1)
        level_map = {
            1: ResistanceLevel.PASSIVE,
            2: ResistanceLevel.DEFENSIVE,
            3: ResistanceLevel.HOSTILE
        }
        level = level_map.get(level_num, ResistanceLevel.PASSIVE)

        return True, level, response

    def analyze_risk_level(self, text: str, history: str) -> tuple[RiskLevel, str]:
        """
        [cite: 12-14, 147] 全局熔断逻辑：判断风险等级

        8B 模型可能输出复杂的推理过程，需要智能提取关键信息
        策略：从最后一行/最后的数字开始提取，优先匹配"2"，再匹配"1"

        返回: (风险等级, 模型原始输出)
        """
        prompt = config.prompts.get(
            "logic", "risk_analysis",
            text=text,
            history=history[-200:]
        )

        response = self._call_model(prompt, is_brain=True)

        # 提取关键分类信息
        risk_score = self._extract_risk_classification(response)

        if risk_score == 2:
            return RiskLevel.CRISIS, response
        elif risk_score == 1:
            return RiskLevel.IDEATION, response
        else:
            return RiskLevel.NORMAL, response

    def _extract_risk_classification(self, response: str) -> int:
        """
        修复：更严格的提取逻辑，防止误判
        """
        if not response or not isinstance(response, str):
            return 0
        
        response = response.strip()
        
        # 尝试解析 JSON (因为 Brain 被要求输出 JSON)
        try:
            # 寻找可能的 JSON 块
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                if "risk_level" in data:
                    return int(data["risk_level"])
        except:
            pass
        
        # 策略1：严格匹配 "Risk Level: X" 或 "risk_level": X
        # 避免匹配到日期或其他数字
        patterns = [
            r'[\"\']?risk_level[\"\']?\s*[:=]\s*(\d)',  # JSON key style
            r'Risk\s*Level\s*[:=]\s*(\d)',              # Text style
            r'等级\s*[:=]\s*(\d)',
            r'Risk\s*[:=]\s*(\d)'
        ]
        
        for p in patterns:
            match = re.search(p, response, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # 策略2：如果上面都失败了，不要盲目搜索数字。
        # 只有当回复非常短（例如只回复了一个数字 "0"）才尝试直接转换
        if len(response) < 5 and response.isdigit():
             return int(response)

        # 默认返回 0 (正常)，防止误报危机
        return 0

    def assess_dimensions_update(self, user_msg: str, current_assessment: AssessmentState) -> tuple[Dict[str, dict], str]:
        """
        [cite: 33-39, 141-150] 证据提取与逻辑比对
        使用 Chain-of-Thought 提取证据并返回 JSON

        返回: (解析后的JSON字典, 模型原始输出)
        """
        # 加载量表标准
        rubric = config.prompts.rubric_content

        prompt = config.prompts.get(
            "logic", "assessment_extraction",
            rubric=rubric,
            state_block=current_assessment.to_prompt_block(),
            user_msg=user_msg
        )

        response = self._call_model(prompt, is_brain=True)
        parsed_json = self._parse_json_from_response(response)
        return parsed_json, response

    def extract_scaling_score(self, text: str) -> Optional[float]:
        """
        [cite: 106] SFBT 打分提取 - 智能打分识别
        
        优先级策略：
        1. 查找"X分"模式（最可靠）
        2. 查找"分数是X"/"打X分"等明确的打分表述
        3. 查找最后出现的数字（用户最终答案）
        4. 处理特殊表达（"满分"→10, "零分"→0）
        """
        # 特殊表达处理
        special_cases = {
            '满分': 10, '十分': 10, '十': 10,
            '零分': 0, '零': 0,
            '五分': 5, '五': 5,
        }
        for text_pattern, score in special_cases.items():
            if text_pattern in text:
                return float(score)
        
        # 策略1: 查找 "X分" 模式（最可靠）
        # 例如："我打7分"、"8.5分"、"给个9分"
        score_with_unit = re.search(r'(\d+(?:\.\d+)?)\s*分', text)
        if score_with_unit:
            try:
                score = float(score_with_unit.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                pass
        
        # 策略2: 查找 "分数"、"评分"等明确表述
        # 例如："我的分数是7"、"评分8"
        explicit_pattern = re.search(r'(?:分数|评分|打|给|是|在)\s*(\d+(?:\.\d+)?)', text)
        if explicit_pattern:
            try:
                score = float(explicit_pattern.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                pass
        
        # 策略3: 找最后出现的数字（用户最终答案）
        all_numbers = list(re.finditer(r'(\d+(?:\.\d+)?)', text))
        if all_numbers:
            for match in reversed(all_numbers):
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 10:
                        return score
                except ValueError:
                    continue
        
        return None

    def generate_avatar_response(self, full_prompt: str):
        """
        Avatar 流式生成 - 使用适配器，返回一个迭代器
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]

        return self.avatar_adapter.generate_stream(
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )

    def _parse_json_from_response(self, text: str) -> dict:
        """
        从 LLM 输出中清洗 JSON - 轻量级容错
        
        只处理最常见的错误，避免过度处理
        """
        if not text or not isinstance(text, str):
            return {}
        
        # 快速路径：如果整个文本是有效 JSON，直接返回
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # 提取 {...} 之间的内容
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            return {}
        
        json_str = text[start:end]
        
        # 尝试解析提取的 JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # 只做一次轻量修复：移除末尾逗号 + 修复引号
        try:
            # 移除末尾逗号：,} 或 ,]
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            # 修复中文引号
            json_str = json_str.replace('"', '"').replace('"', '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # 全部失败，返回空字典
        return {}