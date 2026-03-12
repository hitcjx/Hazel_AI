"""
SFBT策略生成器

负责调用模型进行SFBT策略推理，生成下一轮的干预指令包。

作者：Claude
日期：2025-01-12
版本：v1.2（支持可配置API）
"""

import json
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.config import APIConfig
from sfbt import scoring_rules


@dataclass
class StrategyPackage:
    """SFBT策略包"""
    selected_method: str = ""      # 选中的干预方法
    strategic_intend: str = ""     # 战略意图
    current_module: str = ""       # 当前子模块
    action_status: str = ""        # action状态: in_progress/completed/abandoned
    action: str = ""               # 当前要执行的内部动作
    plugin: str = ""               # 增强插件: 无/赞美/关系问题
    score: int = 0                 # 当前累计分数
    raw_data: dict = None          # 原始完整数据(后台日志用)


@dataclass
class SplitResult:
    """快脑判断结果"""
    need_sfbt: bool          # 是否需要 SFBT 干预
    reason: str              # 判断理由
    raw_data: dict = None   # 原始完整数据(后台日志用)


class SFBTSplitter:
    """快脑 - 快速判断是否需要 SFBT 干预"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """初始化快脑"""
        # 改用 AVATAR 配置（qwen3-235b，更快）
        api_key = api_key or APIConfig.AVATAR_API_KEY
        base_url = base_url or APIConfig.AVATAR_API_BASE_URL
        model = model or APIConfig.AVATAR_API_MODEL

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

        # 加载 Prompt 模板
        self.prompt_template = self._load_split_prompt()

    def _load_split_prompt(self) -> dict:
        """加载 split_prompt.yaml"""
        prompt_path = Path(__file__).parent / "split_prompt.yaml"

        if not prompt_path.exists():
            print(f"[警告] split_prompt.yaml 不存在，使用空模板")
            return {}

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _build_prompt(self, user_msg: str, full_history: List[Dict]) -> str:
        """构建快脑 Prompt"""
        # 格式化历史对话
        history_text = ""
        for msg in full_history[-6:]:  # 最近6轮
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        # 简单的模板拼接（用户会自定义）
        template = self.prompt_template.get("system_role", "") or ""
        template += "\n\n" + (self.prompt_template.get("output_format", "") or "")

        prompt = f"""{template}

当前用户消息：{user_msg}

对话历史：
{history_text}
"""
        return prompt

    def check_need_sfbt(
        self,
        user_msg: str,
        full_history: List[Dict]
    ) -> SplitResult:
        """
        判断是否需要 SFBT 干预（流式获取 need_sfbt）

        Returns:
            SplitResult: 包含 need_sfbt, reason, raw_data
        """
        prompt = self._build_prompt(user_msg, full_history)

        # 流式调用
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个快速判断助手"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            stream=True
        )

        full_content = ""
        need_sfbt = None  # 还没收到判断

        # 第一个 chunk 立即判断
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content

                # 尝试从已收到的内容中解析 need_sfbt
                # 只需要前几个字符就能判断
                if need_sfbt is None:
                    # 尝试解析 JSON
                    try:
                        # 简单检查是否包含 true/false
                        if '"need_sfbt": true' in full_content or '"need_sfbt":true' in full_content:
                            need_sfbt = True
                            print(f"[快脑] 判断结果: 需要 SFBT 干预")
                        elif '"need_sfbt": false' in full_content or '"need_sfbt":false' in full_content:
                            need_sfbt = False
                            print(f"[快脑] 判断结果: 不需要 SFBT 干预")
                    except:
                        pass

        # 完整解析
        reason = ""
        raw_data = {}
        try:
            # 尝试提取 need_sfbt 和 reason
            if '"need_sfbt"' in full_content:
                data = json.loads(full_content)
                need_sfbt = data.get("need_sfbt", True)
                reason = data.get("reason", "")
                raw_data = data
        except:
            # 解析失败，根据内容判断
            need_sfbt = True if need_sfbt is None else need_sfbt
            reason = full_content

        return SplitResult(
            need_sfbt=need_sfbt,
            reason=reason,
            raw_data=raw_data
        )


class SFBTStrategyGenerator:
    """SFBT策略生成器"""

    # 可用方法白名单
    AVAILABLE_METHODS = [
        "例外情境",
        "奇迹问题",
        "应对问题",
        "关系问题",
        "量尺问题",
        "赞美"
    ]

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化策略生成器

        Args:
            api_key: API密钥（默认使用config.BRAIN_API_KEY）
            base_url: API地址（默认使用config.BRAIN_API_BASE_URL）
            model: 模型名称（默认使用config.BRAIN_API_MODEL）
        """
        # 使用config中的默认值，如果参数未提供
        # 改用AVATAR的配置（同款模型）
        api_key = api_key or APIConfig.AVATAR_API_KEY
        base_url = base_url or APIConfig.AVATAR_API_BASE_URL
        model = model or APIConfig.AVATAR_API_MODEL

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model

        # 加载Prompt模板
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> dict:
        """加载decider_prompt.yaml"""
        prompt_path = Path(__file__).parent / "decider_prompt.yaml"

        if not prompt_path.exists():
            print(f"[错误] decider_prompt.yaml不存在: {prompt_path}")
            raise FileNotFoundError(f"找不到decider_prompt.yaml: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 检查是否为空
        if not data or data is None:
            print("[错误] decider_prompt.yaml内容为空，请先填写Prompt内容")
            raise ValueError("decider_prompt.yaml内容为空，无法加载")

        return data

    def generate_strategy(
        self,
        user_msg: str,
        full_history: List[Dict],
        previous_method: str,
        previous_score: int = 0,
        previous_action: str = "",
        previous_action_status: str = "",
        in_progress_count: int = 0,
        current_module: str = ""
    ) -> Tuple[StrategyPackage, str]:
        """
        生成SFBT策略包

        Args:
            user_msg: 用户当前回复
            full_history: 完整对话历史（每轮包含role和content）
            previous_method: 上一轮使用的干预方法
            previous_score: 上一轮的累计分数
            previous_action: 上一轮的 action (当前动作)
            previous_action_status: 上一轮的 action_status
            in_progress_count: 连续 in_progress 次数

        Returns:
            (StrategyPackage, reasoning): 策略包和完整推理过程
        """
        # 构建输入数据（快脑判断已在外部完成）
        t_input_start = time.time()
        input_data = self._format_input(
            user_msg, full_history, previous_method,
            previous_score, previous_action, previous_action_status,
            in_progress_count, current_module
        )
        t_input = time.time() - t_input_start

        # 2. 构建完整Prompt
        t_build_prompt_start = time.time()
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(input_data)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        t_build_prompt = time.time() - t_build_prompt_start

        # 打印日志
        print(f"\n📝 [LOG] Prompt长度分析:")
        print(f"  - system_prompt: {len(system_prompt)} 字符")
        print(f"  - user_prompt: {len(user_prompt)} 字符")
        print(f"  - 总计: {len(full_prompt)} 字符 (~{len(full_prompt)/4:.0f} tokens)")

        t_api_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                stream=True
            )

            # 4. 处理流式响应，收集全部内容
            full_content = ""
            reasoning_content = ""

            t_stream_start = time.time()
            for chunk in response:
                try:
                    # 调试：打印chunk结构
                    # print(f"\n[DEBUG] Chunk: {chunk}")

                    if len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            full_content += delta.content
                            # 实时打印进度
                            print(".", end="", flush=True)
                        elif hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                            # GPT-4o-mini可能返回reasoning_content
                            reasoning_content += delta.reasoning_content
                    else:
                        # chunk.choices为空，可能已经结束
                        pass
                except Exception as e:
                    print(f"\n⚠️ Chunk解析错误: {e}")
                    print(f"Chunk类型: {type(chunk)}")
                    print(f"Chunk内容: {chunk}")
                    continue

            t_stream = time.time() - t_stream_start
            print()  # 换行
            t_api_total = time.time() - t_api_start

            # 5. 分离决策过程(CoT)和JSON
            t_parse_start = time.time()
            reasoning_content, json_str = self._split_cot_and_json(full_content)
            t_parse = time.time() - t_parse_start

            # 打印原始输出
            print(f"\n📄 [LOG] LLM原始输出:")
            print(f"full_content 长度: {len(full_content)} 字符")
            print(f"json_str 长度: {len(json_str)} 字符")
            print(f"json_str 内容:\n{json_str[:1500]}")

            # 6. 解析JSON
            try:
                package_dict = json.loads(json_str)

                # 提取 tags
                tags = package_dict.get("tags", {})

                # 根据 tags 计算 score
                if tags:
                    # 使用传入的 current_module 作为当前模块来计算分数
                    calculated_score = scoring_rules.calculate_score(tags, current_module)
                else:
                    calculated_score = 0
                    calculated_module = "S1_合作构建"

                # 提取 method（可能为空）
                method = package_dict.get("method", "")
                if method == "空" or method == "":
                    final_method = ""  # 空表示跟随对话
                else:
                    final_method = self._normalize_method(method)

                # 提取 action_status（白名单校验）
                raw_action_status = package_dict.get("action_status", "")
                if raw_action_status in ("in_progress", "completed", "abandoned"):
                    action_status = raw_action_status
                else:
                    action_status = "in_progress"  # 非法值默认为in_progress

                # 提取 plugin（白名单校验）
                raw_plugin = package_dict.get("plugin", "")
                if raw_plugin in ("关系问题", "赞美"):
                    plugin = raw_plugin
                else:
                    plugin = ""  # 非法值清空

                # 【约束1】method为空 → 清空action/plugin/action_status
                if not final_method:
                    action_status = ""
                    action = ""
                    plugin = ""
                # 【约束2】abandoned → 清空method/action/plugin
                elif action_status == "abandoned":
                    final_method = ""
                    action = ""
                    plugin = ""
                else:
                    # 正常情况（in_progress/completed）
                    action = package_dict.get("action", "")

                # 使用指数平滑计算累计分数: S_t = 0.6 * s_t + 0.4 * S_{t-1}, S_0 = s_0
                if previous_score == 0:
                    score = calculated_score  # 第一轮: S_0 = s_0
                else:
                    score = 0.6 * calculated_score + 0.4 * previous_score  # S_t = 0.6*s_t + 0.4*S_{t-1}
                # 使用累计分数来计算模块
                if not tags:
                    calculated_module = "S1_合作构建"
                else:
                    calculated_module = scoring_rules.get_current_module(score, current_module)
                final_current_module = calculated_module

                # 【调试】打印action相关字段
                print(f"\n🐛 [DEBUG] Action字段解析:")
                print(f"  - package_dict.get('action'): '{package_dict.get('action', '')}'")
                print(f"  - package_dict.get('action_status'): '{package_dict.get('action_status', '')}'")
                print(f"  - local variable 'action': '{action}'")
                print(f"  - local variable 'action_status': '{action_status}'")
                print(f"  - final_method: '{final_method}'")

                # 构建策略包对象
                package = StrategyPackage(
                    selected_method=final_method,
                    strategic_intend=package_dict.get("intend", ""),
                    current_module=final_current_module,
                    action_status=action_status,
                    action=action,
                    plugin=plugin,
                    score=score,
                    raw_data=package_dict  # 完整原始数据存入后台日志
                )

                # 【调试】打印package的action字段
                print(f"  - package.action: '{package.action}'")
                print(f"  - package.action_status: '{package.action_status}'")

                # 9. 输出时间分解
                print(f"\n⏱️ 时间分解:")
                print(f"  - 输入格式化: {t_input:.3f}秒")
                print(f"  - Prompt构建: {t_build_prompt:.3f}秒")
                print(f"  - API调用总计: {t_api_total:.3f}秒")
                print(f"    - 流式接收: {t_stream:.3f}秒")
                print(f"    - API往返延迟: {t_api_total - t_stream:.3f}秒")
                print(f"  - 解析分离: {t_parse:.3f}秒")
                print(f"  - 总耗时: {t_input + t_build_prompt + t_api_total + t_parse:.3f}秒")

                # 10. 返回策略包和推理过程
                # reasoning_content 是分离出的决策过程(CoT)
                return package, reasoning_content

            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"JSON字符串: {json_str[:500]}...")
                # 返回默认策略包（解包后）
                fallback_package, fallback_reasoning = self._get_fallback_package()
                return fallback_package, fallback_reasoning

        except Exception as e:
            print(f"❌ API调用失败: {e}")
            # 返回默认策略包（解包后）
            fallback_package, fallback_reasoning = self._get_fallback_package()
            return fallback_package, fallback_reasoning

    def _format_input(
        self,
        user_msg: str,
        full_history: List[Dict],
        previous_method: str,
        previous_score: int = 0,
        previous_action: str = "",
        previous_action_status: str = "",
        in_progress_count: int = 0,
        current_module: str = ""
    ) -> str:
        """格式化输入数据"""

        # 格式化历史对话
        history_text = self._format_history(full_history)

        # 计算连续 in_progress 的次数描述
        progress_status = f"连续 in_progress: {in_progress_count}次"
        if in_progress_count >= 3:
            progress_status += "（已达到3次上限，必须 abandoned）"

        # 上一轮 action 信息
        action_info = ""
        if previous_action:
            action_info = f"上一轮 action：{previous_action}\n"
        if previous_action_status:
            action_info += f"上一轮状态：{previous_action_status}"

        # 当前模块信息
        module_info = ""
        if current_module:
            available_methods = scoring_rules.get_available_methods(current_module)
            module_info = f"""
当前模块：{current_module}
可用方法：{available_methods}
"""

        input_data = f"""
## 当前输入数据

用户回复：
{user_msg}

上一轮方法：{previous_method}
上一轮累计分数：{previous_score}
{action_info}
{progress_status}
{module_info}
完整对话历史：
{history_text}
"""

        # 检测是否需要添加奇迹问题过渡提示
        # 条件：上一轮是奇迹问题，且 action_status 为 completed
        if previous_method == "奇迹问题" and previous_action_status == "completed":
            transition_notice = self._get_transition_notice()
            if transition_notice:
                input_data += f"\n\n{transition_notice}"

        return input_data

    def _get_transition_notice(self) -> str:
        """获取奇迹问题过渡提示"""
        # 加载 sfbt_prompt.yaml
        sfbt_prompt_path = Path(__file__).parent / "sfbt_prompt.yaml"

        if not sfbt_prompt_path.exists():
            return ""

        try:
            with open(sfbt_prompt_path, 'r', encoding='utf-8') as f:
                sfbt_data = yaml.safe_load(f) or {}
            return sfbt_data.get("transition_notice", "")
        except:
            return ""

    def _format_history(self, history: List[Dict]) -> str:
        """格式化历史对话"""
        lines = []
        for msg in history[-10:]:  # 确保只取10轮
            role = msg.get("role", "")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """构建系统Prompt"""
        template = self.prompt_template.get("system_role", "")
        return template.strip()

    def _build_user_prompt(self, input_data: str) -> str:
        """构建用户Prompt"""
        tags = self.prompt_template.get("tags", "")
        methods = self.prompt_template.get("methods", "")
        modules = self.prompt_template.get("modules", "")
        output = self.prompt_template.get("output_format", "")

        prompt = f"{tags}\n\n{methods}\n\n{modules}\n\n{output}\n\n{input_data}"
        return prompt

    def _normalize_method(self, raw_method: str) -> str:
        """
        规范化方法名（白名单+模糊匹配）

        Args:
            raw_method: LLM输出的方法名

        Returns:
            规范化后的方法名
        """
        # 1. 完全匹配
        if raw_method in self.AVAILABLE_METHODS:
            return raw_method

        # 2. 模糊匹配（包含关键词）
        method_keywords = {
            "例外情境": ["例外", "寻找例外", "例外情境"],
            "奇迹问题": ["奇迹", "奇迹问题", "奇迹"],
            "应对问题": ["应对", "应对问题", "肯定", "支持"],
            "关系问题": ["关系", "他人", "重要他人"],
            "量尺问题": ["量尺", "分数", "打分", "量化"],
            "赞美": ["赞美", "肯定", "鼓励"]
        }

        for standard_name, keywords in method_keywords.items():
            if any(kw in raw_method for kw in keywords):
                print(f"⚠️ 方法名模糊匹配: '{raw_method}' -> '{standard_name}'")
                return standard_name

        # 3. 无法匹配，返回默认
        print(f"❌ 无法识别方法: '{raw_method}'，使用默认方法'应对问题'")
        return "应对问题"

    def _split_cot_and_json(self, full_content: str) -> tuple[str, str]:
        """
        分离决策过程(CoT)和JSON

        Args:
            full_content: 完整输出（决策过程 + JSON）

        Returns:
            (reasoning_content, json_string)
        """
        # 查找JSON代码块标记
        json_start = full_content.find("```json")

        if json_start == -1:
            # 没有找到```json，尝试查找单独的```标记
            json_start = full_content.find("```")

        if json_start == -1:
            # 都找不到，假设全部是JSON
            print("⚠️ 未检测到JSON代码块标记，尝试解析全部内容")
            return "", full_content.strip()

        # 分离CoT和JSON
        reasoning_content = full_content[:json_start].strip()
        json_part = full_content[json_start:]

        # 提取JSON字符串（移除```json和```标记）
        lines = json_part.split('\n')
        json_lines = []
        in_json = False

        for line in lines:
            if line.strip().startswith("```"):
                in_json = not in_json
                continue
            if in_json or line.strip().startswith('{'):
                json_lines.append(line)

        json_str = '\n'.join(json_lines).strip()

        return reasoning_content, json_str

    def _get_fallback_package(self) -> Tuple[StrategyPackage, str]:
        """获取默认策略包（兜底）"""
        package = StrategyPackage(
            selected_method="应对问题",
            strategic_intend="提供支持和陪伴，当前处于S1_合作构建阶段",
            current_module="S1_合作构建",
            raw_data={}  # 兜底时无原始数据
        )
        reasoning = "使用兜底策略包"
        return package, reasoning

    def save_test_result(self, test_name: str, package: StrategyPackage, reasoning: str):
        """保存测试结果到文件"""
        result = {
            "test_name": test_name,
            "package": {
                "selected_method": package.selected_method,
                "strategic_intend": package.strategic_intend,
                "current_module": package.current_module,
                # raw_data 存入后台日志
            },
            "raw_data": package.raw_data,  # 完整原始数据
            "reasoning": reasoning
        }

        output_path = Path(__file__).parent / "test_results.json"

        # 追加模式写入
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    data = []
                data.append(result)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([result], f, ensure_ascii=False, indent=2)

        print(f"✅ 测试结果已保存到: {output_path}")


# =============================================================================
# 测试代码
# =============================================================================
def test_strategy_generator():
    """测试策略生成器"""

    # 初始化（使用config中的配置）
    generator = SFBTStrategyGenerator()

    # 加载测试数据
    test_data_path = Path(__file__).parent / "test_data.yaml"
    if not test_data_path.exists():
        print(f"❌ 测试数据文件不存在: {test_data_path}")
        return

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_cases = yaml.safe_load(f)

    # 运行测试
    for case in test_cases.get("test_cases", []):
        print(f"\n{'='*60}")
        print(f"测试案例: {case.get('name', 'Unknown')}")
        print(f"{'='*60}")

        # 构建输入
        full_history = case.get("context", "").split('\n')
        history_list = []
        user_msg = ""
        for line in full_history:
            if ': ' in line:
                role, content = line.split(':', 1)
                history_list.append({"role": role.strip(), "content": content.strip()})
                if role.strip() == "U":
                    user_msg = content.strip()

        # 记录开始时间
        start_time = time.time()

        # 生成策略
        package, reasoning = generator.generate_strategy(
            user_msg=user_msg,
            full_history=history_list,
            previous_method=case.get("previous_method", ""),
            previous_score=case.get("previous_score", 0),
            previous_action=case.get("previous_action", ""),
            previous_action_status=case.get("previous_action_status", ""),
            in_progress_count=case.get("in_progress_count", 0)
        )

        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 输出结果
        print(f"\n✅ 生成结果 (总耗时: {elapsed_time:.2f}秒):")
        print(f"  方法: {package.selected_method}")
        print(f"  状态: {package.action_status}")
        print(f"  动作: {package.action}")
        print(f"  意图: {package.strategic_intend}")
        print(f"  模块: {package.current_module}")
        # raw_data 已移至后台日志

        # 保存结果
        generator.save_test_result(case.get('name', 'unknown'), package, reasoning)

        print(f"\n推理过程前500字:")
        print(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)


if __name__ == "__main__":
    test_strategy_generator()
