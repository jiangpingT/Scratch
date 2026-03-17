"""
论文获取 Agent

从多个来源获取最新的 AI 相关论文，包括：
- arXiv API
- Hugging Face Daily Papers
- Semantic Scholar (可选)

主要功能：
1. 查询最新论文
2. 过滤和筛选
3. 提取关键信息
"""

import requests
import feedparser
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
import re

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """论文数据类"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    url: str
    source: str  # arxiv, huggingface, semantic_scholar
    categories: List[str]
    pdf_url: Optional[str] = None

    def to_dict(self):
        """转换为字典"""
        data = asdict(self)
        data['published_date'] = self.published_date.isoformat()
        return data


class PaperFetcher:
    """论文获取器"""

    def __init__(self, config):
        """
        初始化论文获取器

        Args:
            config: 配置对象，包含 PAPER_SOURCES 和 PAPER_FILTER
        """
        self.config = config
        self.sources_config = config.PAPER_SOURCES
        self.filter_config = config.PAPER_FILTER

    def fetch_all(self) -> List[Paper]:
        """
        从所有启用的源获取论文

        Returns:
            论文列表
        """
        all_papers = []

        if self.sources_config["arxiv"]["enabled"]:
            logger.info("正在从 arXiv 获取论文...")
            arxiv_papers = self.fetch_from_arxiv()
            all_papers.extend(arxiv_papers)
            logger.info(f"从 arXiv 获取到 {len(arxiv_papers)} 篇论文")

        if self.sources_config["huggingface"]["enabled"]:
            logger.info("正在从 Hugging Face 获取论文...")
            hf_papers = self.fetch_from_huggingface()
            all_papers.extend(hf_papers)
            logger.info(f"从 Hugging Face 获取到 {len(hf_papers)} 篇论文")

        if self.sources_config["semantic_scholar"]["enabled"]:
            logger.info("正在从 Semantic Scholar 获取论文...")
            ss_papers = self.fetch_from_semantic_scholar()
            all_papers.extend(ss_papers)
            logger.info(f"从 Semantic Scholar 获取到 {len(ss_papers)} 篇论文")

        # 过滤论文
        filtered_papers = self.filter_papers(all_papers)
        logger.info(f"过滤后剩余 {len(filtered_papers)} 篇论文")

        # 去重（基于标题相似度）
        deduped_papers = self.deduplicate_papers(filtered_papers)
        logger.info(f"去重后剩余 {len(deduped_papers)} 篇论文")

        return deduped_papers

    def fetch_from_arxiv(self) -> List[Paper]:
        """
        从 arXiv API 获取论文

        Returns:
            论文列表
        """
        papers = []
        config = self.sources_config["arxiv"]

        # 构建查询
        categories = config["categories"]
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])

        # 构建 URL
        base_url = config["base_url"]
        params = {
            "search_query": category_query,
            "max_results": config["max_results"],
            "sortBy": config["sort_by"],
            "sortOrder": config["sort_order"]
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            # 解析 XML 响应
            root = ET.fromstring(response.content)

            # 命名空间
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            # 提取论文信息
            for entry in root.findall('atom:entry', ns):
                try:
                    # 提取基本信息
                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                    summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                    published = entry.find('atom:published', ns).text
                    link = entry.find('atom:id', ns).text

                    # 提取作者
                    authors = [
                        author.find('atom:name', ns).text
                        for author in entry.findall('atom:author', ns)
                    ]

                    # 提取分类
                    categories = [
                        cat.attrib['term']
                        for cat in entry.findall('atom:category', ns)
                    ]

                    # PDF 链接
                    pdf_link = None
                    for link_elem in entry.findall('atom:link', ns):
                        if link_elem.attrib.get('title') == 'pdf':
                            pdf_link = link_elem.attrib['href']
                            break

                    # 提取 arXiv ID
                    arxiv_id = link.split('/')[-1]

                    # 解析日期
                    published_date = datetime.fromisoformat(published.replace('Z', '+00:00'))

                    paper = Paper(
                        paper_id=f"arxiv_{arxiv_id}",
                        title=title,
                        authors=authors,
                        abstract=summary,
                        published_date=published_date,
                        url=link,
                        source="arxiv",
                        categories=categories,
                        pdf_url=pdf_link
                    )

                    papers.append(paper)

                except Exception as e:
                    logger.warning(f"解析 arXiv 论文条目失败: {e}")
                    continue

        except Exception as e:
            logger.error(f"从 arXiv 获取论文失败: {e}")

        return papers

    def fetch_from_huggingface(self) -> List[Paper]:
        """
        从 Hugging Face Daily Papers 获取论文

        注意：这需要解析 HTML，可能需要调整

        Returns:
            论文列表
        """
        papers = []

        # Hugging Face 提供的 RSS feed
        # 注意：实际实现可能需要使用爬虫或官方 API
        try:
            # 这里使用 Hugging Face Papers 的 RSS feed（如果存在）
            # 或者使用他们的 API
            logger.warning("Hugging Face 获取功能需要实现爬虫或使用 API")
            # TODO: 实现 Hugging Face 论文获取

        except Exception as e:
            logger.error(f"从 Hugging Face 获取论文失败: {e}")

        return papers

    def fetch_from_semantic_scholar(self) -> List[Paper]:
        """
        从 Semantic Scholar API 获取论文

        Returns:
            论文列表
        """
        papers = []
        config = self.sources_config["semantic_scholar"]

        if not config.get("api_key"):
            logger.warning("Semantic Scholar API key 未配置，跳过")
            return papers

        try:
            # TODO: 实现 Semantic Scholar API 调用
            logger.warning("Semantic Scholar 获取功能待实现")

        except Exception as e:
            logger.error(f"从 Semantic Scholar 获取论文失败: {e}")

        return papers

    def filter_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        根据配置过滤论文

        Args:
            papers: 论文列表

        Returns:
            过滤后的论文列表
        """
        filtered = []
        now = datetime.now(papers[0].published_date.tzinfo) if papers else datetime.now()

        min_hours = self.filter_config["min_hours_ago"]
        max_hours = self.filter_config["max_hours_ago"]
        keywords = self.filter_config.get("keywords", [])
        exclude_keywords = self.filter_config.get("exclude_keywords", [])

        for paper in papers:
            # 时间过滤
            hours_ago = (now - paper.published_date).total_seconds() / 3600
            if hours_ago < min_hours or hours_ago > max_hours:
                continue

            # 关键词过滤（可选）
            if keywords:
                text = (paper.title + " " + paper.abstract).lower()
                if not any(kw.lower() in text for kw in keywords):
                    continue

            # 排除关键词
            if exclude_keywords:
                text = (paper.title + " " + paper.abstract).lower()
                if any(kw.lower() in text for kw in exclude_keywords):
                    continue

            filtered.append(paper)

        return filtered

    def deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        基于标题相似度去重

        Args:
            papers: 论文列表

        Returns:
            去重后的论文列表
        """
        if not papers:
            return []

        # 简单的去重：基于标题的标准化
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            # 标准化标题（去除标点、转小写、去除多余空格）
            normalized_title = re.sub(r'[^\w\s]', '', paper.title.lower())
            normalized_title = ' '.join(normalized_title.split())

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)

        return unique_papers


def main():
    """测试函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 导入配置
    import sys
    sys.path.append('..')
    import config

    # 创建获取器
    fetcher = PaperFetcher(config)

    # 获取论文
    papers = fetcher.fetch_all()

    # 打印结果
    print(f"\n总共获取到 {len(papers)} 篇论文：\n")
    for i, paper in enumerate(papers[:5], 1):  # 只显示前5篇
        print(f"{i}. {paper.title}")
        print(f"   作者: {', '.join(paper.authors[:3])}")
        print(f"   来源: {paper.source}")
        print(f"   发布: {paper.published_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   链接: {paper.url}")
        print()


if __name__ == "__main__":
    main()
