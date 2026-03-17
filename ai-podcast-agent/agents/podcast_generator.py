"""
Claude 播客脚本生成 Agent（使用 Agent SDK Loop）

使用 Anthropic SDK 的 Tool Use 和 Agent Loop 机制生成播客脚本：
- 真正的 Loop：在单一对话会话中循环，Claude 主导流程
- 工具调用：Claude 可以切换主持人、查询历史、获取论文等
- 上下文管理：集成历史论文数据到对话中

这是符合 Anthropic Agent SDK 设计理念的实现
"""

import logging
import json
from typing import List, Dict, Optional, Any
from anthropic import Anthropic
from datetime import datetime

logger = logging.getLogger(__name__)


class PodcastGenerator:
    """播客脚本生成器（Agent Loop 实现）"""

    def __init__(self, config, context_manager=None):
        """
        初始化播客生成器

        Args:
            config: 配置对象
            context_manager: 上下文管理器（可选）
        """
        self.config = config
        self.context_manager = context_manager

        # 支持自定义 BASE_URL
        client_kwargs = {"api_key": config.ANTHROPIC_API_KEY}
        if config.ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = config.ANTHROPIC_BASE_URL
        self.anthropic_client = Anthropic(**client_kwargs)

        self.model = config.CLAUDE_MODEL
        self.max_tokens = config.CLAUDE_MAX_TOKENS

        self.generator_config = config.PODCAST_GENERATOR_CONFIG
        self.opening_template = self.generator_config["opening_template"]
        self.closing_template = self.generator_config["closing_template"]

    def generate_podcast_script(self, analyzed_papers: List[Dict], daily_summary: str = None) -> Dict:
        """
        使用 Agent Loop 生成完整的播客脚本

        Args:
            analyzed_papers: 分析后的论文列表
            daily_summary: 每日总览（可选）

        Returns:
            播客脚本字典
        """
        logger.info("开始使用 Agent Loop 生成播客脚本...")

        # 构建上下文
        context = self._build_context(analyzed_papers, daily_summary)

        # 获取历史上下文
        historical_context = self._get_historical_context()

        # 使用 Agent Loop 生成脚本
        script_segments = self._run_agent_loop(context, historical_context, analyzed_papers)

        # 构建完整脚本
        full_script = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'paper_count': len(analyzed_papers),
            'paper_ids': [p['paper_id'] for p in analyzed_papers],
            'segments': script_segments,
            'estimated_duration_minutes': len(script_segments) * 0.5,
            'context': context
        }

        logger.info(f"播客脚本生成完成，共 {len(script_segments)} 个片段")
        return full_script

    def _build_context(self, analyzed_papers: List[Dict], daily_summary: str = None) -> Dict:
        """构建播客上下文信息"""
        context = {
            'date': datetime.now().strftime('%Y年%m月%d日'),
            'paper_count': len(analyzed_papers),
            'daily_summary': daily_summary or "今天我们有几篇有趣的 AI 论文要分享",
            'papers': []
        }

        for paper in analyzed_papers:
            context['papers'].append({
                'title': paper['title'],
                'summary': paper.get('summary', ''),
                'innovation': paper.get('innovation', ''),
                'impact': paper.get('impact', ''),
                'key_points': paper.get('key_points', []),
                'authors': paper.get('authors', [])[:3]
            })

        return context

    def _get_historical_context(self) -> str:
        """
        从 Context Manager 获取历史上下文

        Returns:
            历史上下文描述
        """
        if not self.context_manager:
            return "这是第一期播客。"

        try:
            # 获取历史论文统计
            stats = self.context_manager.get_statistics()

            # 获取热门话题
            topics = self.context_manager.get_top_topics(limit=5)
            topics_str = ", ".join([f"{t[0]}({t[1]}次)" for t in topics]) if topics else "暂无"

            historical_context = f"""
历史统计：
- 已讨论论文：{stats.get('total_papers', 0)} 篇
- 已使用论文：{stats.get('used_papers', 0)} 篇
- 热门话题：{topics_str}
"""
            return historical_context.strip()

        except Exception as e:
            logger.warning(f"获取历史上下文失败: {e}")
            return "历史数据暂不可用。"

    def _run_agent_loop(self, context: Dict, historical_context: str, analyzed_papers: List[Dict]) -> List[Dict]:
        """
        运行 Agent Loop 生成播客对话

        这是核心的 Agent Loop 实现：
        1. 定义工具供 Claude 调用
        2. 在单一对话会话中循环
        3. Claude 主导对话流程

        Args:
            context: 播客上下文
            historical_context: 历史上下文
            analyzed_papers: 论文列表

        Returns:
            对话片段列表
        """
        # 定义 Claude 可调用的工具
        tools = self._define_tools()

        # 构建系统提示
        system_prompt = self._build_agent_system_prompt(context, historical_context)

        # 初始化对话
        messages = [{
            "role": "user",
            "content": f"""请开始生成今天的播客。

今天要讨论的论文：
{json.dumps(context['papers'], ensure_ascii=False, indent=2)}

要求：
1. 以开场白开始
2. 为每篇论文生成对话片段（主持人介绍 → 专家分析 → 互动提问 → 专家回答）
3. 以结尾收尾
4. 使用工具来输出每个对话片段

请开始！"""
        }]

        # 存储生成的片段
        script_segments = []

        # 工具调用的内部状态
        tool_state = {
            'current_paper_index': 0,
            'papers': analyzed_papers,
            'context': context
        }

        # Agent Loop：持续对话直到完成
        max_iterations = 50  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent Loop 迭代 {iteration}/{max_iterations}")

            try:
                # 调用 Claude
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0.8,
                    system=system_prompt,
                    tools=tools,
                    messages=messages
                )

                # 检查停止原因
                stop_reason = response.stop_reason
                logger.info(f"Stop reason: {stop_reason}")

                # 处理响应
                if stop_reason == "tool_use":
                    # Claude 调用了工具
                    tool_results = []

                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            # 执行工具
                            tool_name = content_block.name
                            tool_input = content_block.input
                            tool_id = content_block.id

                            logger.info(f"执行工具: {tool_name}")

                            # 执行工具并获取结果
                            result = self._execute_tool(
                                tool_name,
                                tool_input,
                                tool_state,
                                script_segments
                            )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": json.dumps(result, ensure_ascii=False)
                            })

                    # 将 Claude 的响应和工具结果添加到消息历史
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                elif stop_reason == "end_turn":
                    # Claude 完成了当前轮次
                    # 检查是否有文本输出
                    text_content = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            text_content += content_block.text

                    if text_content:
                        logger.info(f"Claude 输出: {text_content[:100]}...")

                    # 将响应添加到历史
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # 检查是否完成
                    if "播客已完成" in text_content or len(script_segments) >= len(analyzed_papers) * 4 + 2:
                        logger.info("播客生成完成")
                        break

                    # 继续下一轮
                    messages.append({
                        "role": "user",
                        "content": "请继续。"
                    })

                elif stop_reason == "max_tokens":
                    logger.warning("达到最大 token 限制")
                    # 继续对话
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": "请继续。"
                    })

                else:
                    logger.warning(f"未知停止原因: {stop_reason}")
                    break

            except Exception as e:
                logger.error(f"Agent Loop 错误: {e}")
                import traceback
                traceback.print_exc()
                break

        # 如果 Agent Loop 未生成足够片段，使用回退方案
        if len(script_segments) < 2:
            logger.warning("Agent Loop 生成片段不足，使用回退方案")
            script_segments = self._fallback_generation(context, analyzed_papers)

        return script_segments

    def _define_tools(self) -> List[Dict]:
        """
        定义 Claude 可调用的工具（遵循 Gather-Take-Verify 循环）

        Returns:
            工具定义列表
        """
        tools = [
            # ========== GATHER CONTEXT 工具 ==========
            {
                "name": "get_paper_details",
                "description": "【Gather Context】获取指定论文的详细信息，包括标题、摘要、创新点、影响等。在生成内容前使用此工具了解论文。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paper_index": {
                            "type": "integer",
                            "description": "论文索引（从0开始）"
                        }
                    },
                    "required": ["paper_index"]
                }
            },
            {
                "name": "get_historical_context",
                "description": "【Gather Context】获取历史上下文信息，包括已讨论的论文、热门话题等。帮助避免重复内容。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "context_type": {
                            "type": "string",
                            "enum": ["statistics", "topics", "recent_papers"],
                            "description": "上下文类型：statistics=统计信息，topics=热门话题，recent_papers=最近论文"
                        }
                    },
                    "required": ["context_type"]
                }
            },
            {
                "name": "get_current_progress",
                "description": "【Gather Context】获取当前播客生成进度，包括已生成片段数量、类型等。",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },

            # ========== TAKE ACTION 工具 ==========
            {
                "name": "add_podcast_segment",
                "description": "【Take Action】添加一个播客对话片段。在充分收集上下文后，使用此工具生成内容。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "enum": ["host_a", "host_b"],
                            "description": "发言人：host_a=主持人张三（好奇型），host_b=嘉宾李四（专家型）"
                        },
                        "text": {
                            "type": "string",
                            "description": "发言内容（40-150字）"
                        },
                        "segment_type": {
                            "type": "string",
                            "enum": ["opening", "paper_intro", "innovation_analysis", "application_question", "application_answer", "closing", "other"],
                            "description": "片段类型"
                        },
                        "paper_id": {
                            "type": "string",
                            "description": "相关论文ID（如果适用）"
                        }
                    },
                    "required": ["speaker", "text", "segment_type"]
                }
            },

            # ========== VERIFY WORK 工具 ==========
            {
                "name": "verify_segment",
                "description": "【Verify Work】验证最后生成的片段质量。检查长度、连贯性、准确性等。返回验证结果和改进建议。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "segment_index": {
                            "type": "integer",
                            "description": "要验证的片段索引（-1表示最后一个）"
                        },
                        "check_aspects": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["length", "coherence", "accuracy", "tone", "flow"]
                            },
                            "description": "要检查的方面：length=长度，coherence=连贯性，accuracy=准确性，tone=语气，flow=流畅度"
                        }
                    },
                    "required": ["segment_index"]
                }
            },
            {
                "name": "revise_segment",
                "description": "【Verify Work】修订指定的片段。如果验证发现问题，使用此工具改进内容。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "segment_index": {
                            "type": "integer",
                            "description": "要修订的片段索引"
                        },
                        "new_text": {
                            "type": "string",
                            "description": "修订后的文本"
                        },
                        "revision_reason": {
                            "type": "string",
                            "description": "修订原因"
                        }
                    },
                    "required": ["segment_index", "new_text", "revision_reason"]
                }
            },

            # ========== 完成工具 ==========
            {
                "name": "finish_podcast",
                "description": "完成播客生成。在所有片段生成并验证通过后调用此工具。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "生成总结"
                        },
                        "quality_score": {
                            "type": "integer",
                            "description": "自评质量分数（1-10）"
                        }
                    },
                    "required": ["summary"]
                }
            }
        ]

        return tools

    def _build_agent_system_prompt(self, context: Dict, historical_context: str) -> str:
        """
        构建 Agent 系统提示（强调 Gather-Take-Verify 循环）

        Args:
            context: 播客上下文
            historical_context: 历史上下文

        Returns:
            系统提示字符串
        """
        system_prompt = f"""你是一个遵循 Gather-Take-Verify 循环的播客生成 AI Agent。

# 核心工作模式：Gather Context → Take Action → Verify Work

你必须严格遵循这个循环：

## 1️⃣ GATHER CONTEXT（收集上下文）
在生成任何内容前，先使用工具收集信息：
- `get_paper_details` - 了解论文详情
- `get_historical_context` - 查看历史话题，避免重复
- `get_current_progress` - 检查当前进度

**示例流程：**
→ 调用 get_paper_details(0) 获取第1篇论文
→ 调用 get_historical_context("topics") 检查是否讨论过类似话题
→ 基于收集的信息决定如何介绍

## 2️⃣ TAKE ACTION（执行操作）
基于收集的上下文，生成内容：
- `add_podcast_segment` - 添加播客片段

**示例流程：**
→ 调用 add_podcast_segment 生成开场白
→ 调用 add_podcast_segment 生成论文介绍

## 3️⃣ VERIFY WORK（验证工作）
生成后立即验证质量：
- `verify_segment` - 检查片段质量
- `revise_segment` - 如有问题则修订

**示例流程：**
→ 调用 verify_segment(-1) 验证最后生成的片段
→ 如果发现问题，调用 revise_segment 修改
→ 如果验证通过，继续下一个片段

---

# 播客信息
- 日期：{context['date']}
- 论文数量：{context['paper_count']}
- 主题：AI 前沿日报

# 主持人设定
1. **张三（host_a）**：女声，好奇型主持人
   - 负责开场、介绍论文、提问
   - 语气：亲切、好奇、引导性
   - 风格：简洁明快，通俗易懂

2. **李四（host_b）**：男声，AI 专家
   - 负责深度分析、回答问题
   - 语气：专业、友好、善于举例
   - 风格：深入浅出，有理有据

# 历史上下文
{historical_context}

# 播客结构
1. 开场白（张三）：30-80字
2. 对于每篇论文：
   a. 论文介绍（张三）：40-60字
   b. 创新分析（李四）：60-100字
   c. 应用提问（张三）：30-50字
   d. 应用回答（李四）：50-80字
3. 结尾（张三）：40-60字

# 质量标准
- 长度符合要求
- 对话自然流畅
- 专业内容用通俗语言表达
- 适当使用类比和举例
- 保持轻松专业的风格

# 完整工作流程示例

```
步骤1: GATHER - 了解论文
→ get_paper_details(0)

步骤2: GATHER - 检查历史
→ get_historical_context("topics")

步骤3: TAKE ACTION - 生成开场
→ add_podcast_segment(speaker="host_a", text="...", segment_type="opening")

步骤4: VERIFY - 验证开场
→ verify_segment(-1)
→ 如果通过，继续；如果不通过，revise_segment

步骤5: TAKE ACTION - 介绍论文
→ add_podcast_segment(speaker="host_a", text="...", segment_type="paper_intro")

步骤6: VERIFY - 验证介绍
→ verify_segment(-1)

... 重复 GATHER → TAKE → VERIFY 循环 ...

最后: 完成
→ finish_podcast(summary="...", quality_score=9)
```

**重要提醒：**
- 每次 TAKE ACTION 后必须 VERIFY
- 如果 VERIFY 不通过，必须修订
- 不要跳过任何步骤

现在开始生成播客，严格遵循 Gather-Take-Verify 循环！"""

        return system_prompt

    def _execute_tool(self, tool_name: str, tool_input: Dict, tool_state: Dict, script_segments: List[Dict]) -> Dict:
        """
        执行工具调用（支持 Gather-Take-Verify 循环）

        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数
            tool_state: 工具状态
            script_segments: 脚本片段列表（会被修改）

        Returns:
            工具执行结果
        """
        # ========== GATHER CONTEXT 工具 ==========
        if tool_name == "get_paper_details":
            # 获取论文详情
            paper_index = tool_input['paper_index']
            papers = tool_state['papers']

            if 0 <= paper_index < len(papers):
                paper = papers[paper_index]
                return {
                    "success": True,
                    "paper": {
                        'title': paper['title'],
                        'summary': paper.get('summary', ''),
                        'innovation': paper.get('innovation', ''),
                        'impact': paper.get('impact', ''),
                        'key_points': paper.get('key_points', []),
                        'authors': paper.get('authors', [])
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"论文索引 {paper_index} 超出范围（0-{len(papers)-1}）"
                }

        elif tool_name == "get_historical_context":
            # 获取历史上下文
            context_type = tool_input.get('context_type', 'statistics')

            if not self.context_manager:
                return {
                    "success": True,
                    "message": "这是第一期播客，暂无历史数据",
                    "context_type": context_type
                }

            try:
                if context_type == "statistics":
                    stats = self.context_manager.get_statistics()
                    return {
                        "success": True,
                        "context_type": "statistics",
                        "data": stats
                    }
                elif context_type == "topics":
                    topics = self.context_manager.get_top_topics(limit=10)
                    topics_list = [{"topic": t[0], "count": t[1]} for t in topics]
                    return {
                        "success": True,
                        "context_type": "topics",
                        "data": topics_list
                    }
                elif context_type == "recent_papers":
                    recent = self.context_manager.get_recent_papers(days=7, limit=5)
                    return {
                        "success": True,
                        "context_type": "recent_papers",
                        "data": [{"title": p['title'], "date": p['published_date']} for p in recent]
                    }
                else:
                    return {
                        "success": False,
                        "error": f"未知上下文类型: {context_type}"
                    }
            except Exception as e:
                logger.warning(f"获取历史上下文失败: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        elif tool_name == "get_current_progress":
            # 获取当前进度
            segment_types = {}
            for seg in script_segments:
                seg_type = seg.get('segment_type', 'other')
                segment_types[seg_type] = segment_types.get(seg_type, 0) + 1

            return {
                "success": True,
                "total_segments": len(script_segments),
                "segment_types": segment_types,
                "last_segment": script_segments[-1] if script_segments else None
            }

        # ========== TAKE ACTION 工具 ==========
        elif tool_name == "add_podcast_segment":
            # 添加播客片段
            segment = {
                'speaker': tool_input['speaker'],
                'text': tool_input['text'],
                'segment_type': tool_input['segment_type']
            }

            if 'paper_id' in tool_input:
                segment['paper_id'] = tool_input['paper_id']

            script_segments.append(segment)

            logger.info(f"[TAKE ACTION] 添加片段: {tool_input['segment_type']} by {tool_input['speaker']}")

            return {
                "success": True,
                "message": f"片段已添加（共 {len(script_segments)} 个片段）",
                "total_segments": len(script_segments),
                "segment_index": len(script_segments) - 1
            }

        # ========== VERIFY WORK 工具 ==========
        elif tool_name == "verify_segment":
            # 验证片段
            segment_index = tool_input.get('segment_index', -1)
            check_aspects = tool_input.get('check_aspects', ['length', 'coherence'])

            if not script_segments:
                return {
                    "success": False,
                    "error": "没有片段可验证"
                }

            # 获取目标片段
            if segment_index == -1:
                segment_index = len(script_segments) - 1

            if segment_index < 0 or segment_index >= len(script_segments):
                return {
                    "success": False,
                    "error": f"片段索引 {segment_index} 超出范围"
                }

            segment = script_segments[segment_index]
            text = segment['text']
            issues = []
            passed = True

            # 检查长度
            if 'length' in check_aspects:
                text_len = len(text)
                seg_type = segment['segment_type']

                # 根据类型判断长度要求
                if seg_type == "opening":
                    if text_len < 30 or text_len > 80:
                        issues.append(f"开场白长度不符（{text_len}字，应为30-80字）")
                        passed = False
                elif seg_type in ["paper_intro", "application_question"]:
                    if text_len < 30 or text_len > 60:
                        issues.append(f"片段长度不符（{text_len}字，应为30-60字）")
                        passed = False
                elif seg_type in ["innovation_analysis", "application_answer"]:
                    if text_len < 50 or text_len > 100:
                        issues.append(f"片段长度不符（{text_len}字，应为50-100字）")
                        passed = False

            # 检查连贯性（简单检查）
            if 'coherence' in check_aspects:
                if len(text) < 10:
                    issues.append("内容过短，缺乏连贯性")
                    passed = False

            # 检查准确性（检查是否有占位符）
            if 'accuracy' in check_aspects:
                if '...' in text or '待补充' in text or 'TODO' in text:
                    issues.append("包含占位符或待补充内容")
                    passed = False

            logger.info(f"[VERIFY] 片段 {segment_index}: {'通过' if passed else '不通过'}")

            return {
                "success": True,
                "segment_index": segment_index,
                "passed": passed,
                "issues": issues,
                "message": "验证通过" if passed else f"验证失败：{', '.join(issues)}"
            }

        elif tool_name == "revise_segment":
            # 修订片段
            segment_index = tool_input['segment_index']
            new_text = tool_input['new_text']
            reason = tool_input.get('revision_reason', '未说明')

            if segment_index < 0 or segment_index >= len(script_segments):
                return {
                    "success": False,
                    "error": f"片段索引 {segment_index} 超出范围"
                }

            old_text = script_segments[segment_index]['text']
            script_segments[segment_index]['text'] = new_text
            script_segments[segment_index]['revised'] = True
            script_segments[segment_index]['revision_reason'] = reason

            logger.info(f"[VERIFY] 修订片段 {segment_index}: {reason}")

            return {
                "success": True,
                "segment_index": segment_index,
                "message": "片段已修订",
                "old_text": old_text[:50] + "..." if len(old_text) > 50 else old_text,
                "new_text": new_text[:50] + "..." if len(new_text) > 50 else new_text
            }

        # ========== 完成工具 ==========
        elif tool_name == "finish_podcast":
            # 完成播客
            quality_score = tool_input.get('quality_score', 0)
            logger.info(f"播客生成完成，质量评分：{quality_score}")
            return {
                "success": True,
                "message": "播客已完成",
                "total_segments": len(script_segments),
                "summary": tool_input.get('summary', ''),
                "quality_score": quality_score
            }

        else:
            return {
                "success": False,
                "error": f"未知工具: {tool_name}"
            }

    def _fallback_generation(self, context: Dict, analyzed_papers: List[Dict]) -> List[Dict]:
        """
        回退方案：如果 Agent Loop 失败，使用传统方法生成

        Args:
            context: 播客上下文
            analyzed_papers: 论文列表

        Returns:
            脚本片段列表
        """
        logger.info("使用回退方案生成播客...")

        script_segments = []

        # 开场
        opening = f"{self.opening_template} 今天是{context['date']}，我们为大家带来 {context['paper_count']} 篇精彩的 AI 论文解读。"
        script_segments.append({
            'speaker': 'host_a',
            'text': opening,
            'segment_type': 'opening'
        })

        # 为每篇论文生成简单对话
        for i, paper in enumerate(analyzed_papers, 1):
            # 介绍
            intro = f"今天第{i}篇论文是《{paper['title']}》，{paper.get('summary', '这是一篇有趣的研究。')}"
            script_segments.append({
                'speaker': 'host_a',
                'text': intro,
                'segment_type': 'paper_intro',
                'paper_id': paper['paper_id']
            })

            # 分析
            analysis = paper.get('innovation', '这篇论文提出了创新的方法。')
            script_segments.append({
                'speaker': 'host_b',
                'text': analysis,
                'segment_type': 'innovation_analysis',
                'paper_id': paper['paper_id']
            })

        # 结尾
        closing = f"今天的分享就到这里。{self.closing_template}"
        script_segments.append({
            'speaker': 'host_a',
            'text': closing,
            'segment_type': 'closing'
        })

        return script_segments

    def format_script_for_tts(self, script: Dict) -> List[tuple]:
        """
        将脚本格式化为 TTS 输入格式

        Args:
            script: 播客脚本

        Returns:
            [(speaker, text), ...] 列表
        """
        tts_input = []

        for segment in script['segments']:
            speaker = segment['speaker']
            text = segment['text']
            tts_input.append((speaker, text))

        return tts_input


def main():
    """测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 导入配置
    import sys
    sys.path.append('..')
    import config
    from agents.context_manager import ContextManager

    # 测试数据
    test_papers = [
        {
            'paper_id': 'test_001',
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani', 'Shazeer', 'Parmar'],
            'summary': '提出了完全基于注意力机制的 Transformer 架构',
            'innovation': '抛弃了 RNN 和 CNN，完全使用自注意力机制',
            'impact': '成为现代大语言模型的基础架构',
            'key_points': [
                '多头自注意力机制',
                '位置编码',
                '并行化训练'
            ]
        }
    ]

    # 创建上下文管理器
    context_manager = ContextManager(config)

    # 创建生成器
    generator = PodcastGenerator(config, context_manager)

    # 生成脚本
    print("\n正在使用 Agent Loop 生成播客脚本...")
    script = generator.generate_podcast_script(test_papers)

    print("\n播客脚本：")
    print(f"日期：{script['date']}")
    print(f"论文数量：{script['paper_count']}")
    print(f"\n对话片段（共 {len(script['segments'])} 段）：\n")

    for i, segment in enumerate(script['segments'], 1):
        speaker_name = "张三" if segment['speaker'] == 'host_a' else "李四"
        print(f"{i}. 【{speaker_name}】{segment['text']}\n")


if __name__ == "__main__":
    main()
