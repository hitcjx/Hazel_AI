"""流式气泡分割器 - 在流式输出中实时检测分割点"""
import re


class StreamBubbleSplitter:
    """支持流式处理的分割器

    在流式输出过程中，维护缓冲区累积文本，检测到标点符号（。？！…）时
    自动插入 [SPLIT] 分隔符，实现多气泡对话效果。
    """

    # 分割信号：句号、问号、感叹号、换行符、波浪号
    SPLIT_MARKS = ['。', '？', '！', '\n', '~', '～']

    def __init__(self, min_bubble_length: int = 5, max_bubbles: int = 3):
        """
        初始化分割器

        Args:
            min_bubble_length: 最小气泡长度（字符数），小于此值不分割
            max_bubbles: 最大气泡数，超过此数量后不再分割
        """
        self.buffer = ""
        self.min_bubble_length = min_bubble_length
        self.max_bubbles = max_bubbles
        self.split_count = 0

    def process(self, chunk: str) -> list[str]:
        """
        处理一个chunk，返回需要发送的部分

        Args:
            chunk: LLM 输出的文本片段

        Returns:
            list[str]: 需要依次发送的文本片段（可能包含 [SPLIT]）
        """
        self.buffer += chunk

        # 合并连续的换行符，避免过度分割
        self.buffer = re.sub(r'\n+', '\n', self.buffer)

        parts = []

        # 检查是否达到分割上限
        if self.split_count >= self.max_bubbles - 1:
            # 已达上限，直接返回剩余内容
            if self.buffer:
                parts.append(self.buffer)
                self.buffer = ""
            return parts

        # 检测分割标点（首段和后续使用相同逻辑）
        for mark in self.SPLIT_MARKS:
            while mark in self.buffer:
                idx = self.buffer.index(mark) + len(mark)

                # 检查长度是否满足最小要求
                if idx >= self.min_bubble_length:
                    split_content = self.buffer[:idx]
                    # 去除末尾的换行符
                    if split_content.endswith('\n'):
                        split_content = split_content[:-1]
                    parts.append(split_content)
                    parts.append("[SPLIT]")
                    self.buffer = self.buffer[idx:]
                    self.split_count += 1

                    # 达到上限则停止
                    if self.split_count >= self.max_bubbles - 1:
                        break
                else:
                    # 太短，跳过这个标点，继续查找下一个
                    self.buffer = self.buffer[idx:]

            # 检查是否达到上限
            if self.split_count >= self.max_bubbles - 1:
                break

        return parts

    def finalize(self) -> list[str]:
        """
        返回剩余内容（流结束时调用）

        Returns:
            list[str]: 剩余的文本片段
        """
        if self.buffer:
            result = [self.buffer]
            self.buffer = ""
            self.split_count = 0
            return result
        return []
