# Avatar Prompt 测试工具

用于测试和优化 Avatar 的 persona 和 instruction。

## 🚀 启动

```bash
cd /home/dazzle/Hazel_AI/avatar_prompt_optimizer
chainlit run app.py
```

启动后会自动打开浏览器（通常是 http://localhost:8000）

## 📖 使用说明

### 1. 选择 Instruction
在右侧设置面板选择要测试的 Instruction：
- 正常场景（warmup, topic_follow, closing）
- 评估场景（5个维度）
- SFBT 场景（4种）
- 阻力场景（3个等级）
- 危机场景

### 2. 开始对话测试
在输入框输入消息，AI 会根据当前组装的 prompt 回复：
- Persona（从 avatar_prompts.yaml 读取）
- 选中的 Instruction
- 最近 10 轮对话历史
- mem0 记忆（自动搜索相关记忆）

### 3. 切换 Instruction
随时在设置面板切换 Instruction，测试不同场景的效果。

### 4. 快捷操作
- **📝 查看当前 Prompt**：查看完整组装的 prompt
- **📊 显示对话历史**：查看当前所有对话
- **🔄 清空历史**：清空对话历史，重新开始

## 📝 修改 Prompt

直接编辑 `avatar_prompts.yaml` 文件，刷新页面后生效。

修改满意后，复制回主项目的 `prompts.yaml`。

## 🔧 配置

### User ID
设置面板中的"用户 ID"用于 mem0 记忆系统，不同 user_id 会有独立的记忆。

### mem0 记忆
默认启用，对话会自动保存到 mem0，并在后续对话中检索相关记忆。

如需禁用，注释掉 `app.py` 中的 memory 相关代码。

## 📂 文件说明

```
avatar_prompt_optimizer/
├── app.py                 # Chainlit 应用主程序
├── avatar_prompts.yaml    # 可编辑的 prompt 副本
└── README.md             # 本文件
```

## ⚠️ 注意事项

1. 此工具仅用于测试，不会修改主项目的 prompts.yaml
2. mem0 中的记忆会永久保存，测试时建议使用固定的 test_user
3. 对话历史在刷新页面后会清空（mem0 记忆不会）
