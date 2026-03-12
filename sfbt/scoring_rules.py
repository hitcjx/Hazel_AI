"""
SFBT 评分规则模块
根据 tags 计算分数和决定当前模块
"""

# 评分规则定义
SCORING_RULES = {
    "S1_合作构建": {
        "threshold": (None, 3),
        "scores": {
            "stance": {"投入": 4, "合作": 3, "被动": -2, "阻抗": -3},
            "talk_focus": {"问题": -3, "波动": 1, "解决": 3},
            "energy_status": {"竭尽": -3, "波动": 1, "积极": 3},
            "goal_clarity": {"缺失": -1, "模糊": 2, "具体": 4},
            "exception_detected": {"闪烁": 2, "稳固": 4},
            "trend": {"好转": 2, "恶化": -2}
        },
        "available_methods": "应对问题 / 例外情境（轻度使用，不可过度深入）"
    },
    "S2_目标构建&资源挖掘": {
        "threshold": (3, 8),
        "scores": {
            "goal_clarity": {"模糊": 2, "具体": 5},
            "exception_detected": {"闪烁": 3, "稳固": 6},
            "talk_focus": {"波动": 1, "解决": 3, "问题": -2},
            "stance": {"投入": 3, "合作": 1, "被动": -3, "阻抗": -5},
            "energy_status": {"竭尽": -3, "积极": 2},
            "trend": {"好转": 1, "恶化": -3}
        },
        "available_methods": "奇迹问题 / 例外情境 / 量尺问题（轻度）"
    },
    "S3_赋能行动": {
        "threshold": (8, None),
        "scores": {
            "exception_detected": {"稳固": 6, "闪烁": 2, "无": -3},
            "trend": {"好转": 3, "恶化": -5},
            "talk_focus": {"解决": 1, "问题": -5},
            "stance": {"投入": 1, "被动": -3, "阻抗": -6},
            "goal_clarity": {"具体": 2, "模糊": -4},
            "energy_status": {"积极": 2, "竭尽": -5}
        },
        "available_methods": "量尺问题 / 例外情境（强化）"
    }
}

# 模块名称映射（用于显示）
MODULE_NAMES = {
    "S1_合作构建": "S1_合作构建",
    "S2_目标构建&资源挖掘": "S2_目标构建&资源挖掘",
    "S3_赋能行动": "S3_赋能行动"
}


def calculate_score(tags: dict, current_module: str = None) -> int:
    """
    根据 tags 计算累计分数

    Args:
        tags: LLM 输出的标签 dict
        current_module: 当前模块（如果知道的话，用于使用对应的评分规则）

    Returns:
        累计分数
    """
    if not tags:
        return 0

    score = 0

    # 确定使用哪个模块的评分规则
    # 如果提供了 current_module，使用对应规则
    # 否则尝试所有模块的规则并累加（因为是累计分数）
    if current_module and current_module in SCORING_RULES:
        rules = SCORING_RULES[current_module]["scores"]
        for dimension, value in tags.items():
            if dimension in rules and value in rules[dimension]:
                score += rules[dimension][value]
    else:
        # 没有指定模块，尝试所有模块的规则并累加
        for module_name, module_data in SCORING_RULES.items():
            rules = module_data["scores"]
            for dimension, value in tags.items():
                if dimension in rules and value in rules[dimension]:
                    score += rules[dimension][value]

    return score


def get_current_module(score: int, previous_module: str = None) -> str:
    """
    根据分数返回当前模块（支持回退）

    Args:
        score: 平滑后的累计分数
        previous_module: 上一轮模块名称（可选）

    Returns:
        模块名称
    """
    # 模块回退顺序：S3 -> S2 -> S1
    MODULE_ORDER = ["S3_赋能行动", "S2_目标构建&资源挖掘", "S1_合作构建"]

    # 如果传入了上一轮模块，检查是否需要回退
    if previous_module and previous_module in MODULE_ORDER:
        current_idx = MODULE_ORDER.index(previous_module)
        # 获取当前分数对应的目标模块
        if score < 3:
            target_module = "S1_合作构建"
        elif score < 8:
            target_module = "S2_目标构建&资源挖掘"
        else:
            target_module = "S3_赋能行动"

        target_idx = MODULE_ORDER.index(target_module)

        # 如果目标模块索引更小（更高级），升级
        if target_idx < current_idx:
            return target_module
        # 如果目标模块索引更大（更低级），回退
        elif target_idx > current_idx:
            return target_module
        # 如果相同，保持上一轮模块
        return previous_module

    # 如果没有传入上一轮模块，使用原有逻辑（只升级不降级）
    if score < 3:
        return "S1_合作构建"
    elif score < 8:
        return "S2_目标构建&资源挖掘"
    else:
        return "S3_赋能行动"


def get_available_methods(module: str) -> str:
    """
    获取当前模块的可用方法描述

    Args:
        module: 模块名称

    Returns:
        可用方法描述字符串
    """
    if module in SCORING_RULES:
        return SCORING_RULES[module]["available_methods"]
    return "无限制"


def get_module_info(score: int = None, current_module: str = None) -> dict:
    """
    获取完整的模块信息

    Args:
        score: 分数（可选）
        current_module: 当前模块（可选）

    Returns:
        包含 score, current_module, available_methods 的 dict
    """
    # 计算分数
    if score is None:
        score = 0

    # 确定当前模块
    if current_module:
        final_module = current_module
    elif score > 0:
        final_module = get_current_module(score)
    else:
        final_module = "S1_合作构建"  # 默认初始模块

    # 如果传入了新的分数，重新计算模块
    if score > 0:
        final_module = get_current_module(score)

    return {
        "score": score,
        "current_module": final_module,
        "available_methods": get_available_methods(final_module)
    }
