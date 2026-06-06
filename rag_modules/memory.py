"""
短期记忆模块 — 维护对话历史会话列表，为多轮对话提供上下文。

功能：
1. 滑动窗口管理对话历史（保留最近 N 轮）
2. 将历史格式化为生成阶段的上下文前缀
3. 支持查询重写时引用历史对话消解指代
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    role: str           # "user" | "assistant"
    content: str
    query_type: str = "general"     # 查询类型标记


@dataclass
class ConversationMemory:
    """
    短期记忆管理器。
    用滑动窗口保留最近 N 轮对话，供查询重写和答案生成时使用。
    """

    max_turns: int = 10          # 最多保留的对话轮数（一问一答算 2 轮）
    _history: List[ConversationTurn] = field(default_factory=list)

    def add_turn(self, role: str, content: str, query_type: str = "general") -> None:
        """添加一轮对话到历史中。"""
        turn = ConversationTurn(role=role, content=content, query_type=query_type)
        self._history.append(turn)

        # 滑动窗口：超出上限时移除最早的轮次
        while len(self._history) > self.max_turns:
            self._history.pop(0)

    def add_user_query(self, query: str, query_type: str = "general") -> None:
        """记录用户问题。"""
        self.add_turn("user", query, query_type)

    def add_assistant_response(self, answer: str) -> None:
        """记录助手回答。"""
        self.add_turn("assistant", answer)

    def get_history(self) -> List[ConversationTurn]:
        """返回完整对话历史（副本）。"""
        return list(self._history)

    def get_recent(self, n: int = 4) -> List[ConversationTurn]:
        """返回最近 n 轮对话。"""
        return self._history[-n:] if len(self._history) >= n else list(self._history)

    def format_for_context(self, max_turns: Optional[int] = None) -> str:
        """
        将对话历史格式化为上下文字符串，供生成阶段使用。

        Args:
            max_turns: 最多包含的轮数，None 表示全部

        Returns:
            格式化的历史对话字符串
        """
        history = self._history if max_turns is None else self.get_recent(max_turns)
        if not history:
            return "（暂无历史对话）"

        lines = ["## 对话历史"]
        for i, turn in enumerate(history, 1):
            label = "👤 用户" if turn.role == "user" else "🤖 助手"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)

    def format_for_rewrite(self) -> str:
        """
        将对话历史格式化为查询重写可用的简洁参考。
        只保留最近的用户问题，用于消解指代。
        """
        recent = self.get_recent(6)
        if not recent:
            return "（无历史）"

        lines = []
        for turn in recent:
            if turn.role == "user":
                lines.append(f"- 用户问: {turn.content}")
            else:
                lines.append(f"  助手答: {turn.content[:100]}{'...' if len(turn.content) > 100 else ''}")
        return "\n".join(lines)

    def get_last_user_query(self) -> Optional[str]:
        """获取最近一次用户问题，用于消解指代。"""
        for turn in reversed(self._history):
            if turn.role == "user":
                return turn.content
        return None

    def get_last_assistant_response(self) -> Optional[str]:
        """获取最近一次助手回答。"""
        for turn in reversed(self._history):
            if turn.role == "assistant":
                return turn.content
        return None

    def clear(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0

    @property
    def turn_count(self) -> int:
        """返回用户提问次数（不含助手回答）。"""
        return sum(1 for t in self._history if t.role == "user")

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"<ConversationMemory turns={len(self._history)} max={self.max_turns}>"
