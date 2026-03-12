"""
记忆管理器 - 简化版，直接使用 mem0 + 简单缓存
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time

try:
    from mem0 import Memory
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False
    print("[Warning] mem0ai not installed. Install with: pip install mem0ai")


class MemoryManager:
    """
    mem0 记忆管理器 - 简化版

    直接使用 mem0 的自动提取功能，不做额外处理
    """

    def __init__(self, cache_ttl: int = 600):
        """
        初始化 mem0 客户端 + 缓存

        Args:
            cache_ttl: 缓存过期时间（秒），默认10分钟
        """
        if not HAS_MEM0:
            raise ImportError("mem0ai is not installed. Run: pip install mem0ai")

        import os
        from config import MemoryConfig

        # 配置 OpenAI 客户端（用于 embedding）
        os.environ["OPENAI_API_KEY"] = MemoryConfig.MEMORY_QUICK_API_KEY
        os.environ["OPENAI_BASE_URL"] = MemoryConfig.MEMORY_QUICK_BASE_URL

        # 使用 mem0 Cloud
        try:
            api_key = MemoryConfig.MEM0_API_KEY
            self.client = Memory.from_config({"api_key": api_key})
            print("✓ mem0 Cloud initialized")
            self.is_cloud = True
        except Exception as e:
            print(f"[Warning] Failed to initialize mem0 Cloud: {e}")
            print("Falling back to local storage...")
            self.memories = {}
            self.is_cloud = False
            print("✓ mem0 Local initialized")

        # 初始化缓存
        self._cache: Dict[str, Tuple[List[dict], float]] = {}
        self._cache_ttl = cache_ttl
        self._cache_stats = {"hits": 0, "misses": 0}
        print(f"✓ 缓存已启用 (TTL={cache_ttl}秒)")

    def add_conversation(self, user_id: str, user_msg: str, assistant_msg: str, metadata: dict = None):
        """
        添加对话到 mem0，让 mem0 自动提取记忆

        Args:
            user_id: 用户ID
            user_msg: 用户消息
            assistant_msg: 助手回复
            metadata: 额外的元数据
        """
        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]

            # 合并元数据
            base_metadata = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            if metadata:
                base_metadata.update(metadata)

            # 添加到 mem0，让它自动提取
            self.client.add(messages, user_id=user_id, metadata=base_metadata)

            print(f"[Memory] 对话已添加到 mem0 (user_id: {user_id})")

        except Exception as e:
            print(f"[Error] 添加对话到 mem0 失败: {e}")

    def search(self, query: str, user_id: str, limit: int = 5) -> List[dict]:
        """
        搜索记忆（带缓存）

        Args:
            query: 搜索查询
            user_id: 用户ID
            limit: 返回结果数量

        Returns:
            匹配的记忆列表
        """
        # 生成缓存key
        cache_key = f"{user_id}:{query}"

        # 检查缓存
        if cache_key in self._cache:
            results, timestamp = self._cache[cache_key]

            # 检查是否过期
            if time.time() - timestamp < self._cache_ttl:
                self._cache_stats["hits"] += 1
                age = int(time.time() - timestamp)
                print(f"  ✅ [Memory Cache] 命中 (查询: {query[:30]}... 缓存年龄: {age}秒)")
                return results
            else:
                # 缓存过期，删除
                del self._cache[cache_key]
                print(f"  ⏰ [Memory Cache] 过期 (查询: {query[:30]}...)")

        # 缓存未命中，调用API
        self._cache_stats["misses"] += 1
        print(f"  🔄 [Memory Cache] 未命中，调用API (查询: {query[:30]}...)")

        try:
            results = self.client.search(query=query, user_id=user_id, limit=limit)

            # 存入缓存
            self._cache[cache_key] = (results, time.time())

            return results

        except Exception as e:
            print(f"[Error] 搜索记忆失败: {e}")
            return []

    def get_all(self, user_id: str) -> List[dict]:
        """
        获取用户的所有记忆

        Args:
            user_id: 用户ID

        Returns:
            所有记忆列表
        """
        try:
            result = self.client.get_all(user_id=user_id)

            # mem0 Cloud 返回 {'results': [...]}
            if isinstance(result, dict) and 'results' in result:
                return result['results']
            else:
                return result if result else []

        except Exception as e:
            print(f"[Error] 获取所有记忆失败: {e}")
            return []

    def get_all_as_text(self, user_id: str, limit: int = 10) -> str:
        """
        获取用户的所有记忆，格式化为文本

        Args:
            user_id: 用户ID
            limit: 最多返回多少条记忆

        Returns:
            格式化的记忆文本
        """
        try:
            memories = self.get_all(user_id)

            if not memories:
                return "暂无历史记忆"

            # 只取前 N 条
            memories = memories[:limit]

            # 格式化为文本
            lines = []
            for i, memory in enumerate(memories, 1):
                if isinstance(memory, dict):
                    content = memory.get('memory', '')
                else:
                    content = str(memory)

                lines.append(f"{i}. {content}")

            return "\n".join(lines)

        except Exception as e:
            print(f"[Error] 获取记忆文本失败: {e}")
            return "获取历史记忆失败"

    def clear_cache(self):
        """清空所有缓存"""
        self._cache.clear()
        print("✓ [Memory Cache] 缓存已清空")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        获取缓存统计信息

        Returns:
            包含缓存统计的字典
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self._cache),
            "ttl": self._cache_ttl
        }

    def cleanup_expired_cache(self):
        """清理所有过期的缓存条目"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._cache_ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            print(f"✓ [Memory Cache] 清理了 {len(expired_keys)} 个过期缓存")

        return len(expired_keys)


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    import time

    print("=== 测试 mem0 记忆管理器（简化版）===\n")

    # 初始化
    manager = MemoryManager()
    user_id = f"test_user_{int(time.time())}"

    # 测试 1：添加对话
    print("【测试 1】添加对话")
    manager.add_conversation(
        user_id,
        "你好，我叫小明，在上海读大学",
        "你好小明！很高兴认识你。"
    )

    time.sleep(1)

    # 测试 2：搜索记忆
    print("\n【测试 2】搜索记忆")
    results = manager.search("小明", user_id=user_id)
    print(f"搜索结果: {results}")

    # 测试 3：获取所有记忆
    print("\n【测试 3】获取所有记忆")
    all_memories = manager.get_all_as_text(user_id)
    print(f"所有记忆:\n{all_memories}")

    print("\n✓ 测试完成")
