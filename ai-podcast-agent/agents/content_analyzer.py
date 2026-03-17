"""
Claude 内容分析 Agent

使用 Claude API 分析论文内容，提取关键信息：
1. 核心创新点
2. 技术突破
3. 潜在影响
4. 关键要点

输出结构化的分析结果供播客生成使用
"""

import logging
import json
from typing import List, Dict, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """内容分析 Agent"""

    def __init__(self, config):
        """
        初始化内容分析器

        Args:
            config: 配置对象
        """
        self.config = config
        # 支持自定义 BASE_URL
        client_kwargs = {"api_key": config.ANTHROPIC_API_KEY}
        if config.ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = config.ANTHROPIC_BASE_URL
        self.anthropic_client = Anthropic(**client_kwargs)
        self.model = config.CLAUDE_MODEL
        self.max_tokens = config.CLAUDE_MAX_TOKENS
        self.temperature = config.CLAUDE_TEMPERATURE

        self.analyzer_config = config.CONTENT_ANALYZER_CONFIG
        self.system_prompt = self.analyzer_config["system_prompt"]
        self.max_papers = self.analyzer_config["max_papers"]
        self.output_language = self.analyzer_config["output_language"]

    def analyze_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        分析多篇论文

        Args:
            papers: 论文列表

        Returns:
            分析结果列表
        """
        # 限制分析数量
        papers_to_analyze = papers[:self.max_papers]
        logger.info(f"开始分析 {len(papers_to_analyze)} 篇论文...")

        analyzed_papers = []

        for i, paper in enumerate(papers_to_analyze, 1):
            logger.info(f"正在分析第 {i}/{len(papers_to_analyze)} 篇: {paper['title']}")

            try:
                analysis = self.analyze_single_paper(paper)
                analyzed_papers.append(analysis)
            except Exception as e:
                logger.error(f"分析论文失败 {paper['title']}: {e}")
                # 创建一个基本的分析结果
                analyzed_papers.append({
                    'paper_id': paper['paper_id'],
                    'title': paper['title'],
                    'summary': paper['abstract'][:500] + "...",
                    'innovation': "分析失败",
                    'impact': "未知",
                    'key_points': [],
                    'error': str(e)
                })

        logger.info(f"分析完成，成功分析 {len(analyzed_papers)} 篇论文")
        return analyzed_papers

    def analyze_single_paper(self, paper: Dict) -> Dict:
        """
        分析单篇论文

        Args:
            paper: 论文信息字典

        Returns:
            分析结果字典
        """
        # 构建分析提示
        user_prompt = self._build_analysis_prompt(paper)

        # 调用 Claude API
        response = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # 解析响应
        response_text = response.content[0].text

        # 尝试解析 JSON
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # 如果不是 JSON，则尝试提取信息
            logger.warning(f"响应不是有效的 JSON，使用文本解析")
            analysis = self._parse_text_response(response_text, paper)

        # 添加原始论文信息
        analysis['paper_id'] = paper['paper_id']
        analysis['url'] = paper['url']
        analysis['authors'] = paper['authors']
        analysis['published_date'] = paper['published_date']

        return analysis

    def _build_analysis_prompt(self, paper: Dict) -> str:
        """
        构建分析提示词

        Args:
            paper: 论文信息

        Returns:
            提示词字符串
        """
        authors_str = ", ".join(paper['authors'][:5])
        if len(paper['authors']) > 5:
            authors_str += " 等"

        prompt = f"""请分析以下 AI 论文，并以 JSON 格式输出分析结果。

论文信息：
标题：{paper['title']}
作者：{authors_str}
摘要：{paper['abstract']}

请提供以下分析（用中文）：

1. summary: 一句话总结论文核心内容（50字以内）
2. innovation: 核心创新点是什么？（100字以内）
3. impact: 这篇论文的潜在影响和应用场景（100字以内）
4. key_points: 3-5个关键要点（数组形式，每个要点30字以内）
5. technical_depth: 技术深度评分（1-5，1=入门，5=前沿突破）
6. practical_value: 实用价值评分（1-5，1=纯理论，5=可立即应用）

输出格式示例：
{{
    "title": "论文标题",
    "summary": "一句话总结",
    "innovation": "核心创新点描述",
    "impact": "潜在影响描述",
    "key_points": ["要点1", "要点2", "要点3"],
    "technical_depth": 4,
    "practical_value": 3
}}

请直接输出 JSON，不要有其他解释文字。
"""

        return prompt

    def _parse_text_response(self, text: str, paper: Dict) -> Dict:
        """
        从文本响应中提取信息（备用方案）

        Args:
            text: 响应文本
            paper: 原始论文信息

        Returns:
            分析结果字典
        """
        # 简单的文本解析
        analysis = {
            'title': paper['title'],
            'summary': text[:200] + "..." if len(text) > 200 else text,
            'innovation': "待进一步分析",
            'impact': "待进一步分析",
            'key_points': [text[:100]] if text else [],
            'technical_depth': 3,
            'practical_value': 3
        }

        return analysis

    def rank_papers(self, analyzed_papers: List[Dict]) -> List[Dict]:
        """
        对分析后的论文进行排序

        Args:
            analyzed_papers: 分析结果列表

        Returns:
            排序后的列表
        """
        # 简单的排序策略：技术深度 * 0.6 + 实用价值 * 0.4
        def score(paper):
            tech = paper.get('technical_depth', 3)
            practical = paper.get('practical_value', 3)
            return tech * 0.6 + practical * 0.4

        sorted_papers = sorted(analyzed_papers, key=score, reverse=True)
        return sorted_papers

    def generate_daily_summary(self, analyzed_papers: List[Dict]) -> str:
        """
        生成每日论文总览

        Args:
            analyzed_papers: 分析结果列表

        Returns:
            总览文本
        """
        if not analyzed_papers:
            return "今天没有找到相关论文。"

        # 构建总览提示
        papers_info = []
        for i, paper in enumerate(analyzed_papers, 1):
            info = f"{i}. {paper['title']}\n   创新点：{paper.get('innovation', '未知')}\n"
            papers_info.append(info)

        prompt = f"""请基于以下论文分析，生成一段简短的每日 AI 前沿总览（100-150字）。

今日论文：
{"".join(papers_info)}

总览应该：
1. 总结今天的主要技术趋势
2. 突出最重要的创新点
3. 用通俗易懂的语言
4. 保持轻松专业的风格

请直接输出总览文字，不要有标题或前缀。
"""

        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            summary = response.content[0].text.strip()
            return summary

        except Exception as e:
            logger.error(f"生成总览失败: {e}")
            return f"今天我们找到了 {len(analyzed_papers)} 篇有趣的 AI 论文，涵盖了多个前沿领域。"


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

    # 测试论文
    test_paper = {
        'paper_id': 'test_001',
        'title': 'Attention Is All You Need',
        'authors': ['Vaswani', 'Shazeer', 'Parmar'],
        'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
        'url': 'https://arxiv.org/abs/1706.03762',
        'published_date': '2017-06-12T00:00:00'
    }

    # 创建分析器
    analyzer = ContentAnalyzer(config)

    # 分析论文
    print("\n正在分析论文...")
    analysis = analyzer.analyze_single_paper(test_paper)

    print("\n分析结果：")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
