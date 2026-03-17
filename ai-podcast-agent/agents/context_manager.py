"""
上下文管理器

管理历史论文数据和上下文信息，包括：
1. 论文历史记录（去重）
2. 热点话题追踪
3. 上下文状态维护

使用 SQLite 数据库存储
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


class ContextManager:
    """上下文管理器"""

    def __init__(self, config):
        """
        初始化上下文管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.db_path = config.STORAGE_CONFIG["database_path"]
        self.max_history = config.CONTEXT_CONFIG["max_history_papers"]
        self.dedup_window_days = config.CONTEXT_CONFIG["dedup_window_days"]
        self.track_topics = config.CONTEXT_CONFIG["track_topics"]
        self.topic_threshold = config.CONTEXT_CONFIG["topic_threshold"]

        # 确保数据库目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建论文表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                published_date TEXT NOT NULL,
                url TEXT,
                source TEXT,
                categories TEXT,
                pdf_url TEXT,
                fetched_date TEXT NOT NULL,
                used_in_podcast INTEGER DEFAULT 0,
                podcast_date TEXT
            )
        ''')

        # 创建话题表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE NOT NULL,
                count INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
        ''')

        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_paper_id ON papers(paper_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_published_date ON papers(published_date)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_used_in_podcast ON papers(used_in_podcast)
        ''')

        conn.commit()
        conn.close()

        logger.info(f"数据库初始化完成: {self.db_path}")

    def add_papers(self, papers: List) -> int:
        """
        添加论文到数据库

        Args:
            papers: 论文列表（Paper 对象）

        Returns:
            新添加的论文数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        added_count = 0

        for paper in papers:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO papers
                    (paper_id, title, authors, abstract, published_date,
                     url, source, categories, pdf_url, fetched_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    paper.paper_id,
                    paper.title,
                    json.dumps(paper.authors),
                    paper.abstract,
                    paper.published_date.isoformat(),
                    paper.url,
                    paper.source,
                    json.dumps(paper.categories),
                    paper.pdf_url,
                    datetime.now().isoformat()
                ))

                if cursor.rowcount > 0:
                    added_count += 1

                    # 提取和更新话题
                    if self.track_topics:
                        self._update_topics(paper)

            except Exception as e:
                logger.error(f"添加论文失败 {paper.paper_id}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"成功添加 {added_count} 篇新论文到数据库")
        return added_count

    def _update_topics(self, paper):
        """
        从论文中提取话题并更新话题统计

        Args:
            paper: 论文对象
        """
        # 简单的话题提取：从分类和标题中提取关键词
        topics = set()

        # 从分类中提取
        for category in paper.categories:
            topics.add(category)

        # 从标题中提取常见技术关键词（简化版）
        keywords = [
            "transformer", "llm", "gpt", "bert", "agent", "multimodal",
            "reasoning", "fine-tuning", "rlhf", "retrieval", "rag",
            "vision", "language model", "neural network", "deep learning"
        ]

        title_lower = paper.title.lower()
        for kw in keywords:
            if kw in title_lower:
                topics.add(kw)

        # 更新数据库中的话题统计
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for topic in topics:
            cursor.execute('''
                INSERT INTO topics (topic, count, first_seen, last_seen)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    count = count + 1,
                    last_seen = ?
            ''', (topic, now, now, now))

        conn.commit()
        conn.close()

    def is_duplicate(self, paper_id: str) -> bool:
        """
        检查论文是否已存在

        Args:
            paper_id: 论文 ID

        Returns:
            是否重复
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM papers WHERE paper_id = ?
        ''', (paper_id,))

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def get_recent_papers(self, days: int = 7, limit: int = 100) -> List[Dict]:
        """
        获取最近的论文

        Args:
            days: 最近多少天
            limit: 最多返回多少篇

        Returns:
            论文列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT * FROM papers
            WHERE published_date >= ?
            ORDER BY published_date DESC
            LIMIT ?
        ''', (cutoff_date, limit))

        papers = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # 解析 JSON 字段
        for paper in papers:
            paper['authors'] = json.loads(paper['authors'])
            paper['categories'] = json.loads(paper['categories'])

        return papers

    def get_unused_papers(self, limit: int = 10) -> List[Dict]:
        """
        获取尚未用于播客的论文

        Args:
            limit: 最多返回多少篇

        Returns:
            论文列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM papers
            WHERE used_in_podcast = 0
            ORDER BY published_date DESC
            LIMIT ?
        ''', (limit,))

        papers = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # 解析 JSON 字段
        for paper in papers:
            paper['authors'] = json.loads(paper['authors'])
            paper['categories'] = json.loads(paper['categories'])

        return papers

    def mark_papers_used(self, paper_ids: List[str], podcast_date: str = None):
        """
        标记论文已用于播客

        Args:
            paper_ids: 论文 ID 列表
            podcast_date: 播客日期
        """
        if not podcast_date:
            podcast_date = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for paper_id in paper_ids:
            cursor.execute('''
                UPDATE papers
                SET used_in_podcast = 1, podcast_date = ?
                WHERE paper_id = ?
            ''', (podcast_date, paper_id))

        conn.commit()
        conn.close()

        logger.info(f"标记 {len(paper_ids)} 篇论文为已使用")

    def get_trending_topics(self, limit: int = 10, min_count: int = None) -> List[Dict]:
        """
        获取热门话题

        Args:
            limit: 返回多少个话题
            min_count: 最少出现次数

        Returns:
            话题列表 [{"topic": str, "count": int, ...}, ...]
        """
        if min_count is None:
            min_count = self.topic_threshold

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM topics
            WHERE count >= ?
            ORDER BY count DESC, last_seen DESC
            LIMIT ?
        ''', (min_count, limit))

        topics = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return topics

    def cleanup_old_papers(self):
        """清理旧论文，保持数据库大小"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 保留最近的论文和已用于播客的论文
        cursor.execute('''
            DELETE FROM papers
            WHERE id NOT IN (
                SELECT id FROM papers
                WHERE used_in_podcast = 1
                UNION
                SELECT id FROM papers
                ORDER BY published_date DESC
                LIMIT ?
            )
        ''', (self.max_history,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"清理了 {deleted_count} 篇旧论文")

    def get_top_topics(self, limit: int = 10) -> List[tuple]:
        """
        获取热门话题（简化版，返回元组列表）

        Args:
            limit: 返回多少个话题

        Returns:
            [(topic, count), ...] 列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT topic, count FROM topics
            WHERE count >= ?
            ORDER BY count DESC, last_seen DESC
            LIMIT ?
        ''', (self.topic_threshold, limit))

        topics = cursor.fetchall()
        conn.close()

        return topics

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 总论文数
        cursor.execute('SELECT COUNT(*) FROM papers')
        stats['total_papers'] = cursor.fetchone()[0]

        # 已使用论文数
        cursor.execute('SELECT COUNT(*) FROM papers WHERE used_in_podcast = 1')
        stats['used_papers'] = cursor.fetchone()[0]

        # 未使用论文数
        stats['unused_papers'] = stats['total_papers'] - stats['used_papers']

        # 话题数
        cursor.execute('SELECT COUNT(*) FROM topics')
        stats['total_topics'] = cursor.fetchone()[0]

        # 最近 7 天的论文数
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM papers WHERE published_date >= ?', (cutoff,))
        stats['recent_papers_7d'] = cursor.fetchone()[0]

        conn.close()

        return stats


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

    # 创建管理器
    manager = ContextManager(config)

    # 获取统计信息
    stats = manager.get_statistics()
    print("\n数据库统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 获取热门话题
    topics = manager.get_trending_topics(limit=5)
    print("\n热门话题:")
    for topic in topics:
        print(f"  {topic['topic']}: {topic['count']} 次")


if __name__ == "__main__":
    main()
