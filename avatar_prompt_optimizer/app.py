"""
Avatar Prompt 测试工具 - Streamlit 版本
支持实时编辑和测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from openai import OpenAI
import yaml

# 直接配置 Avatar API
AVATAR_API_KEY = "sk-uUaPPQMrETWan8TfHbBsdEebNCIQ6AZ3g581N0aI7hCsF52y"
AVATAR_API_MODEL = "gpt-4o-mini"  # 使用和主项目一样的模型
AVATAR_API_BASE_URL = "https://api.chatanywhere.org/v1"

# 初始化 OpenAI 客户端（不缓存）
def get_client():
    return OpenAI(api_key=AVATAR_API_KEY, base_url=AVATAR_API_BASE_URL)

# 尝试导入 mem0
try:
    from memory_manager import MemoryManager
    memory_manager = MemoryManager()
except ImportError:
    memory_manager = None

# 从本地 avatar_prompts.yaml 加载（而不是主项目的 prompts.yaml）
class AvatarPromptsLoader:
    def __init__(self):
        import os
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.yaml_path = os.path.join(script_dir, "avatar_prompts.yaml")
        print(f"[DEBUG] AvatarPromptsLoader: yaml_path = {self.yaml_path}")
        print(f"[DEBUG] File exists: {os.path.exists(self.yaml_path)}")
        self._load()

    def _load(self):
        import os
        print(f"[DEBUG] Loading YAML from: {self.yaml_path}")
        print(f"[DEBUG] File exists: {os.path.exists(self.yaml_path)}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)

        print(f"[DEBUG] Loaded keys: {list(self.data.keys())}")
        persona = self.data.get('persona')
        print(f"[DEBUG] Persona value type: {type(persona)}")
        print(f"[DEBUG] Persona is None: {persona is None}")
        print(f"[DEBUG] Persona length: {len(persona) if persona else 0}")
        if persona:
            print(f"[DEBUG] Persona first 100 chars: {persona[:100]}")

    def reload(self):
        """重新加载 YAML 文件"""
        self._load()

    def get(self, *keys, default=None):
        """获取嵌套的值"""
        data = self.data
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data

# 不在模块级别初始化，而是在每次运行时初始化
def get_avatar_prompts():
    """获取 avatar prompts（每次都重新加载）"""
    if '_avatar_prompts' not in st.session_state:
        st.session_state._avatar_prompts = AvatarPromptsLoader()
    return st.session_state._avatar_prompts
INSTRUCTIONS = {
    "stages.warmup": "正常-破冰",
    "stages.topic_follow": "正常-话题跟随",
    "stages.closing": "正常-结束",

    "bridging.PHYSICAL_FUNCTION": "评估-生理功能",
    "bridging.COGNITIVE_DISTORTION": "评估-认知偏差",
    "bridging.ACADEMIC_PRESSURE": "评估-学业压力",
    "bridging.EMOTIONAL_RESILIENCE": "评估-情绪韧度",
    "bridging.SOCIAL_SUPPORT": "评估-社会支持",

    "sfbt.量尺问题": "SFBT-量尺问题",
    "sfbt.应对问题": "SFBT-应对问题",
    "sfbt.例外情境": "SFBT-例外情境",
    "sfbt.奇迹问题": "SFBT-奇迹问题",
    "sfbt.关系问题": "SFBT-关系问题",
    "sfbt.赞美": "SFBT-赞美",
    "sfbt.跟随对话": "SFBT-跟随对话",

    "resistance.passive_response": "阻力-被动",
    "resistance.defensive_response": "阻力-防御",
    "resistance.hostile_response": "阻力-敌意",

    "safety.crisis_instruction": "危机-干预",
}


def get_instruction_content(key: str) -> str:
    """获取 instruction 内容（从 session_state 优先）"""
    # 如果用户编辑过，用编辑后的
    if "edited_instructions" in st.session_state and key in st.session_state.edited_instructions:
        return st.session_state.edited_instructions[key]

    # 否则从本地 avatar_prompts.yaml 加载
    parts = key.split(".")
    avatar_prompts = get_avatar_prompts()
    data = avatar_prompts.data
    for part in parts:
        if part in data:
            data = data[part]
        else:
            return f"Error: {key} not found"
    return data if isinstance(data, str) else str(data)


def save_instruction_to_file(key: str, content: str):
    """保存 instruction 到 avatar_prompts.yaml"""
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "avatar_prompts.yaml")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 解析 key（如 "stages.warmup"）
    parts = key.split(".")
    if len(parts) == 2:
        category, item = parts
        if category in data and item in data[category]:
            data[category][item] = content

    # 写回文件
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def save_persona_to_file(content: str):
    """保存 persona 到 avatar_prompts.yaml"""
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "avatar_prompts.yaml")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    data["persona"] = content

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def assemble_prompt(persona: str, instruction: str, history: list, user_msg: str, user_id: str = "test_user") -> str:
    """组装完整 prompt"""
    parts = []

    # 确保 persona 不是 None
    if persona:
        parts.append(persona)

    # mem0 记忆
    if memory_manager:
        try:
            memories = memory_manager.search(query=user_msg, user_id=user_id, limit=5)
            if memories:
                memory_texts = [m.get('memory', '') if isinstance(m, dict) else str(m) for m in list(memories)[:5]]
                if memory_texts:
                    parts.append("【相关记忆】\n" + "\n".join(f"- {m}" for m in memory_texts))
        except Exception as e:
            print(f"[Warning] Memory search failed: {e}")

    # 确保 instruction 不是 None
    if instruction:
        parts.append(instruction)

    # 历史对话（最近10轮）
    if history:
        recent = history[-10:]
        history_lines = []
        for role, content in recent:
            prefix = "U" if role == "user" else "A"
            content_short = content[:100] + ("..." if len(content) > 100 else "")
            history_lines.append(f"{prefix}: {content_short}")
        parts.append("【对话历史】\n" + "\n".join(history_lines))

    # 当前输入
    parts.append(f"\n用户: {user_msg}")
    parts.append("Assistant:")

    # 过滤掉 None 值
    parts = [p for p in parts if p is not None]

    return "\n\n".join(parts)


def call_avatar_api(client, prompt: str):
    """调用 Avatar API（流式版本）"""
    import time
    start_time = time.time()

    response = client.chat.completions.create(
        model=AVATAR_API_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
        stream=True  # 启用流式输出
    )

    def generate():
        full_content = ""
        first_chunk_time = None
        chunk_count = 0

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, 'content') and delta.content:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    time_to_first_chunk = first_chunk_time - start_time
                    print(f"[性能] 首字耗时: {time_to_first_chunk:.2f}秒")

                full_content += delta.content
                chunk_count += 1
                yield delta.content

        elapsed = time.time() - start_time
        print(f"[性能] 总耗时: {elapsed:.2f}秒 | chunks: {chunk_count} | tokens: {len(full_content)}")

    return generate()


# ==================== 页面布局 ====================

st.set_page_config(page_title="Avatar Prompt 测试工具", layout="wide")

st.title("🎭 Avatar Prompt 测试工具")

# 初始化 session state
avatar_prompts = get_avatar_prompts()

# 调试：把加载状态写入 session_state
st.session_state._debug_avatar_loaded = True
st.session_state._debug_persona_length = len(avatar_prompts.get("persona") or "")
st.session_state._debug_keys = list(avatar_prompts.data.keys())

if "persona" not in st.session_state or not st.session_state.persona:
    persona_val = avatar_prompts.get("persona")
    st.session_state.persona = persona_val if persona_val else ""
    st.session_state._debug_init_persona_length = len(persona_val or "")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_instruction" not in st.session_state:
    st.session_state.selected_instruction = "stages.warmup"
if "user_id" not in st.session_state:
    st.session_state.user_id = "test_user"
if "edited_instructions" not in st.session_state:
    st.session_state.edited_instructions = {}

# ==================== 左右分栏 ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Prompt 编辑")

    # 调试信息（展开查看）
    with st.expander("🔍 调试信息（点击查看）"):
        st.write(f"- avatar_prompts loaded: {st.session_state.get('_debug_avatar_loaded', False)}")
        st.write(f"- persona length (from file): {st.session_state.get('_debug_persona_length', 0)}")
        st.write(f"- persona length (after init): {st.session_state.get('_debug_init_persona_length', 0)}")
        st.write(f"- YAML keys: {st.session_state.get('_debug_keys', [])}")
        st.write(f"- session_state.persona length: {len(st.session_state.get('persona', '') or '')}")

    # Persona 编辑
    st.markdown("**1. Persona（可直接编辑）**")

    # 调试信息
    if st.session_state.persona:
        st.caption(f"✓ Persona 已加载（{len(st.session_state.persona)} 字符）")
    else:
        st.caption("⚠️ Persona 为空，请点击重新加载")

    edited_persona = st.text_area(
        "Persona 内容",
        st.session_state.persona or "",
        height=300,
        key="persona_editor",
        help="直接编辑 Persona 内容，修改后点击上方保存按钮"
    )
    st.session_state.persona = edited_persona

    # 保存 persona 按钮
    if st.button("💾 保存 Persona 到文件", key="save_persona"):
        try:
            save_persona_to_file(edited_persona)
            st.success("✅ 已保存 Persona 到 avatar_prompts.yaml")
        except Exception as e:
            st.error(f"❌ 保存失败: {str(e)}")

    # Instruction 选择和编辑
    st.markdown("**2. Instruction（可选择和编辑）**")

    # 当切换时重置编辑状态
    def on_instruction_change():
        if 'last_selected_instruction' in st.session_state:
            if st.session_state.last_selected_instruction != st.session_state.selected_instruction:
                # 切换了，清除之前可能存在的编辑状态
                if st.session_state.last_selected_instruction in st.session_state.edited_instructions:
                    # 如果之前有编辑但没保存，询问用户（这里先简单处理：清除）
                    pass
        st.session_state.last_selected_instruction = st.session_state.selected_instruction

    instruction_key = st.selectbox(
        "选择 Instruction",
        options=list(INSTRUCTIONS.keys()),
        format_func=lambda x: f"{x} - {INSTRUCTIONS[x]}",
        index=list(INSTRUCTIONS.keys()).index(st.session_state.selected_instruction),
        key="instruction_selector",
        on_change=on_instruction_change
    )

    # 如果切换了，更新 session_state
    if instruction_key != st.session_state.selected_instruction:
        st.session_state.selected_instruction = instruction_key
        st.rerun()

    # Instruction 编辑框
    current_instruction = get_instruction_content(instruction_key)
    edited_instruction = st.text_area(
        "Instruction 内容（可直接编辑）",
        current_instruction,
        height=200,
        key=f"instruction_editor_{instruction_key}"  # 动态 key，每个 instruction 独立
    )

    # 保存编辑后的 instruction 到 session_state
    if edited_instruction != current_instruction:
        st.session_state.edited_instructions[instruction_key] = edited_instruction
        current_instruction = edited_instruction

    # 保存到文件按钮
    col_save1, col_save2 = st.columns(2)
    with col_save1:
        if st.button("💾 保存 Instruction 到文件", key="save_instruction"):
            try:
                save_instruction_to_file(instruction_key, edited_instruction)
                st.success(f"✅ 已保存 {instruction_key} 到 avatar_prompts.yaml")
                # 清除编辑记录，因为已经保存到文件了
                if instruction_key in st.session_state.edited_instructions:
                    del st.session_state.edited_instructions[instruction_key]
            except Exception as e:
                st.error(f"❌ 保存失败: {str(e)}")

    with col_save2:
        if st.button("🔄 重置 Instruction", key="reset_instruction"):
            if instruction_key in st.session_state.edited_instructions:
                del st.session_state.edited_instructions[instruction_key]
            st.rerun()

    # User ID
    st.markdown("**3. 其他设置**")
    st.session_state.user_id = st.text_input("User ID", st.session_state.user_id)

    # 重新加载按钮
    if st.button("🔄 重新加载 avatar_prompts.yaml", key="reload_yaml"):
        # 清除缓存并重新初始化
        if '_avatar_prompts' in st.session_state:
            del st.session_state._avatar_prompts
        avatar_prompts = get_avatar_prompts()
        # 强制重新加载所有内容
        persona_val = avatar_prompts.get("persona")
        st.session_state.persona = persona_val if persona_val else ""
        st.session_state.edited_instructions = {}
        st.session_state.last_selected_instruction = None
        st.success("✅ 已重新加载 avatar_prompts.yaml")
        st.rerun()

with col2:
    st.subheader("💬 对话测试")

    # 显示对话历史
    chat_container = st.container()

    with chat_container:
        for role, content in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)

    # 用户输入
    if user_input := st.chat_input("输入消息..."):
        # 添加用户消息
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        # 组装 prompt
        full_prompt = assemble_prompt(
            st.session_state.persona,
            current_instruction,
            st.session_state.chat_history,
            user_input,
            st.session_state.user_id
        )

        # 调用 API（流式输出）
        with st.chat_message("assistant"):
            try:
                client = get_client()
                stream = call_avatar_api(client, full_prompt)

                # 实时显示流式输出
                full_response = ""
                placeholder = st.empty()
                with st.spinner("AI 思考中..."):
                    for chunk in stream:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                # 移除光标并显示最终结果
                placeholder.markdown(full_response)

                # 保存到历史
                st.session_state.chat_history.append(("assistant", full_response))

                # 保存到 mem0
                if memory_manager:
                    try:
                        memory_manager.add_conversation(
                            user_id=st.session_state.user_id,
                            user_msg=user_input,
                            assistant_msg=full_response
                        )
                    except:
                        pass

            except Exception as e:
                st.error(f"❌ 错误: {str(e)}")
                st.session_state.chat_history.pop()

    # 清空历史按钮
    if st.button("🔄 清空对话历史"):
        st.session_state.chat_history = []
        st.rerun()

# ==================== 底部：查看完整 Prompt ====================
st.markdown("---")
st.subheader("📋 当前组装的完整 Prompt")

if st.session_state.chat_history:
    last_user_msg = st.session_state.chat_history[-1][1] if st.session_state.chat_history[-1][0] == "user" else "你好"
else:
    last_user_msg = "你好"

# 获取当前选中的 instruction
current_instruction_preview = get_instruction_content(st.session_state.selected_instruction)

full_prompt_preview = assemble_prompt(
    st.session_state.persona,
    current_instruction_preview,
    st.session_state.chat_history,
    last_user_msg,
    st.session_state.user_id
)

st.text_area(
    "完整 Prompt",
    full_prompt_preview,
    height=300,
    disabled=True,
    label_visibility="collapsed"
)
