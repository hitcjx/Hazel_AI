import json
import re
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import config
from config import ModelConfig
from state_manager import AssessmentState, RiskLevel, AssessmentDimension
import threading
from threading import Thread

class LLMEngine:
    """
    负责底层模型调用的适配层。
    """
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.brain_model = None
        self.brain_tokenizer = None
        self.avatar_model = None
        self.avatar_tokenizer = None
        self.guard_model = None 
        self.guard_tokenizer = None

        # 显卡互斥锁：任何时候只能有一个模型在 generate
        self.gpu_lock = threading.Lock()

        if not self.use_mock:
            self._load_models()

    def _load_models(self):
        # 定义 4-bit 量化配置 (这是核心)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        """加载本地模型 (根据实际显存情况调整)"""
        print(f"Loading Brain Model: {ModelConfig.BRAIN_MODEL_PATH}...")
        self.brain_tokenizer = AutoTokenizer.from_pretrained(ModelConfig.BRAIN_MODEL_PATH, trust_remote_code=True)
        self.brain_model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.BRAIN_MODEL_PATH, 
            quantization_config=bnb_config,
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

        print(f"Loading Avatar Model: {ModelConfig.AVATAR_MODEL_PATH}...")
        self.avatar_tokenizer = AutoTokenizer.from_pretrained(ModelConfig.AVATAR_MODEL_PATH, trust_remote_code=True)
        self.avatar_model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.AVATAR_MODEL_PATH, 
            quantization_config=bnb_config,
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

        print(f"Loading Guard Model: {ModelConfig.GUARD_MODEL_PATH}...")
        self.guard_tokenizer = AutoTokenizer.from_pretrained(ModelConfig.GUARD_MODEL_PATH, trust_remote_code=True)
        self.guard_model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.GUARD_MODEL_PATH, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

    def _call_model(self, prompt: str, is_brain: bool = True) -> str:
        """底层推理调用 - 修复 Prompt 移除逻辑"""
        model = self.brain_model if is_brain else self.avatar_model
        tokenizer = self.brain_tokenizer if is_brain else self.avatar_tokenizer
        temp = ModelConfig.BRAIN_TEMP if is_brain else ModelConfig.DEFAULT_TEMP

        messages = [
            {"role": "system", "content": "You are a helpful assistant." if not is_brain else "You are a logic engine outputting JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # 1. 构建 Prompt
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])

        # 2. 生成
        # 加锁
        with self.gpu_lock:
             generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=512 if is_brain else 1024,
                temperature=temp,
                top_p=0.9
             )
        
        # 3. 关键修复：直接跳过 Input Tokens 进行解码
        # 这样能 100% 确保返回的只有模型新生成的内容，不会包含 Prompt
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def fast_risk_check(self, text: str) -> bool:
        """
        快速双重熔断机制
        1. 正则匹配 (0ms)
        2. 小模型确认 (100ms)
        返回: True (有风险), False (安全)
        """
        # Step 1: 正则初筛
        has_keyword = any(re.search(p, text) for p in config.CRISIS_KEYWORDS)
        if not has_keyword:
            return False
        
        # Step 2: 小模型进一步确认 (防止"我笑死了"被误判)
        # 只有命中关键词才调小模型，极其节省资源
        messages = [
            {"role": "system", "content": "你是一个安全审核员。判断用户输入是否有自杀风险。只回答'是'或'否'。"},
            {"role": "user", "content": text}
        ]
        
        # 使用 apply_chat_template
        input_text = self.guard_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.guard_tokenizer([input_text], return_tensors="pt").to(self.guard_model.device)
        
        generated_ids = self.guard_model.generate(
            inputs.input_ids, 
            max_new_tokens=10, # 给够空间
            temperature=0.1    # 低温，保证确定性
        )
        response = self.guard_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 简单起见，直接找关键词
        return "是" in response or "Yes" in response

    def analyze_risk_level(self, text: str, history: str) -> RiskLevel:
        """
        [cite: 12-14, 147] 全局熔断逻辑：判断风险等级
        
        8B 模型可能输出复杂的推理过程，需要智能提取关键信息
        策略：从最后一行/最后的数字开始提取，优先匹配"2"，再匹配"1"
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
            return RiskLevel.CRISIS
        elif risk_score == 1:
            return RiskLevel.IDEATION
        else:
            return RiskLevel.NORMAL

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

    def assess_dimensions_update(self, user_msg: str, current_assessment: AssessmentState) -> Dict[str, dict]:
        """
        [cite: 33-39, 141-150] 证据提取与逻辑比对
        使用 Chain-of-Thought 提取证据并返回 JSON
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
        return self._parse_json_from_response(response)

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
        [优化] Avatar 流式生成，返回一个迭代器
        """
        model = self.avatar_model
        tokenizer = self.avatar_tokenizer
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 初始化流式传输器
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 必须在独立线程中运行 generate，否则会阻塞主线程无法读取 stream
        generation_kwargs = dict(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=1024, 
            temperature=0.7
        )
        def thread_target(**kwargs):
            with self.gpu_lock: # [关键修改] 只有拿到锁才能开始生成
                self.avatar_model.generate(**kwargs)
                
        thread = threading.Thread(target=thread_target, kwargs=generation_kwargs)
        thread.start()

        # 返回生成器，让主程序去 print
        return streamer

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