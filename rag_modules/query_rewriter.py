"""
查询重写模块 — 在检索前用 LLM 改写模糊查询，提升检索命中率。

解决的问题：
1. 指代消解：用户问"它的步骤是什么？" → 结合历史改写为"番茄炒蛋的步骤是什么？"
2. 省略补全：用户问"有没有推荐的？" → 改写为"有没有推荐的科幻书籍？"
3. 多义消歧：用户问"Python 怎么学？" → 保持原样（合理）
4. 口语规范化：用户问"咋整啊这个" → 改写为标准书面语

设计原则：
- 只改写确实模糊的查询，清晰查询原样返回
- 改写后的问题必须语义等价，不引入新信息
- 结合对话历史进行指代消解
"""

from typing import Optional
from langchain_deepseek import ChatDeepSeek


QUERY_REWRITE_SYSTEM_PROMPT = """你是一个查询改写助手。你的任务是判断用户问题是否需要改写，只输出改写后的问题。

规则：
1. 如果问题包含"它"、"这个"、"那个"、"其"等指代词，参考对话历史将指代替换为具体内容。
2. 如果问题省略了关键主语或宾语（依赖上下文才能理解），参考历史补全。
3. 如果问题本身清晰完整，**原样输出，不要做任何修改**。
4. 改写后的问题必须与原始意图完全一致，不能添加用户没问的内容。
5. 只输出改写后的问题文本，不要加任何解释、前缀或后缀。

## 示例

对话历史：
- 用户问: 番茄炒蛋怎么做？
  助手答: 番茄炒蛋的步骤是：1. 准备食材...
- 用户问: 有什么推荐的科幻书？
  助手答: 推荐《三体》《银河帝国》...

示例1：
用户问题: 它的步骤是什么？
输出: 番茄炒蛋的步骤是什么？

示例2：
用户问题: 还有别的推荐吗？
输出: 除了《三体》《银河帝国》之外还有什么科幻书籍推荐？

示例3：
用户问题: Python 装饰器怎么用？
输出: Python 装饰器怎么用？

示例4：
用户问题: 那第二本讲的是什么？
输出: 《银河帝国》第二本讲的是什么？
"""

QUERY_REWRITE_USER_TEMPLATE = """对话历史：
{history}

用户问题: {query}

输出:"""


class QueryRewriter:
    """
    查询改写器。
    使用 LLM 分析用户问题，结合对话历史将模糊查询改写为明确查询。
    """

    def __init__(self, llm: ChatDeepSeek):
        """
        Args:
            llm: 已初始化的大模型实例（用于轻量改写，区别于主生成模型）
        """
        self.llm = llm

    def rewrite(
        self,
        query: str,
        history_text: str = "",
    ) -> str:
        """
        改写用户查询。

        Args:
            query: 原始用户问题
            history_text: 格式化的对话历史文本（由 ConversationMemory.format_for_rewrite() 生成）

        Returns:
            改写后的问题（如果无需改写则原样返回）
        """
        # 快速短路：如果无历史且问题长度足够，大概率不需要改写
        if (not history_text or history_text == "（无历史）") and self._seems_clear(query):
            return query

        # 用 LLM 判断并改写
        try:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", QUERY_REWRITE_SYSTEM_PROMPT),
                ("human", QUERY_REWRITE_USER_TEMPLATE),
            ])
            messages = prompt.format_messages(history=history_text, query=query)
            response = self.llm.invoke(messages)
            rewritten = response.content.strip()

            # 安全兜底：如果 LLM 返回空或过长，用原始查询
            if not rewritten or len(rewritten) > len(query) * 3:
                return query

            return rewritten

        except Exception as e:
            # 改写失败时兜底：直接用原始查询
            print(f"[WARN] 查询改写失败: {e}，使用原始查询")
            return query

    @staticmethod
    def _seems_clear(query: str) -> bool:
        """
        启发式判断查询是否已经足够清晰，无需 LLM 改写。

        清晰标准：
        - 问题长度 >= 8 个字符
        - 不包含明显指代词（它、这个、那个、其、该）
        - 不包含省略信号词（还有、另外的、别的）
        """
        # 太短的问题可能不完整
        if len(query) < 8:
            return False

        # 检查指代词
        ambiguous_words = ["它", "这个", "那个", "其", "该", "他", "她"]
        for word in ambiguous_words:
            if word in query:
                return False

        # 检查省略信号（单用这些词通常是追问，需要上下文）
        omission_signals = ["还有吗", "别的呢", "另外的", "其他的", "除此之外"]
        for signal in omission_signals:
            if signal in query:
                return False

        return True
