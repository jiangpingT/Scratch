#!/usr/bin/env python3
"""
测试 Claude API 功能
"""

import sys
import json

print("=" * 60)
print("Claude API 测试")
print("=" * 60)

# 导入配置
print("\n【1】加载配置...")
try:
    import config
    print(f"✓ 配置加载成功")
    print(f"  - API Key: {config.ANTHROPIC_API_KEY[:20]}...")
    print(f"  - Base URL: {config.ANTHROPIC_BASE_URL}")
    print(f"  - Model: {config.CLAUDE_MODEL}")
except Exception as e:
    print(f"✗ 配置加载失败: {e}")
    sys.exit(1)

# 测试论文获取
print("\n【2】获取测试论文...")
try:
    from agents.paper_fetcher import PaperFetcher

    fetcher = PaperFetcher(config)
    config.PAPER_FILTER["max_hours_ago"] = 168  # 7天

    papers = fetcher.fetch_from_arxiv()
    print(f"✓ 获取到 {len(papers)} 篇论文")

    if not papers:
        print("✗ 没有获取到论文，无法继续测试")
        sys.exit(1)

    test_paper = papers[0]
    print(f"\n  测试论文:")
    print(f"  标题: {test_paper.title}")
    print(f"  作者: {', '.join(test_paper.authors[:3])}")

except Exception as e:
    print(f"✗ 论文获取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 Claude 内容分析
print("\n【3】测试 Claude 内容分析...")
try:
    from agents.content_analyzer import ContentAnalyzer

    analyzer = ContentAnalyzer(config)
    print("✓ 分析器初始化成功")

    # 将 Paper 对象转换为字典
    paper_dict = {
        'paper_id': test_paper.paper_id,
        'title': test_paper.title,
        'authors': test_paper.authors,
        'abstract': test_paper.abstract,
        'url': test_paper.url,
        'published_date': test_paper.published_date.isoformat()
    }

    print("\n  正在调用 Claude API 分析论文...")
    print("  这可能需要几秒钟...")

    analysis = analyzer.analyze_single_paper(paper_dict)

    print("\n✓ 分析完成！")
    print("\n分析结果:")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

except Exception as e:
    print(f"\n✗ Claude 分析失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试播客脚本生成（简化版）
print("\n【4】测试 Claude 播客脚本生成...")
try:
    from agents.podcast_generator import PodcastGenerator

    generator = PodcastGenerator(config)
    print("✓ 播客生成器初始化成功")

    print("\n  正在生成播客脚本...")
    print("  这可能需要几十秒...")

    # 使用刚才分析的论文
    script = generator.generate_podcast_script([analysis])

    print("\n✓ 播客脚本生成完成！")
    print(f"\n脚本信息:")
    print(f"  - 日期: {script['date']}")
    print(f"  - 论文数: {script['paper_count']}")
    print(f"  - 片段数: {len(script['segments'])}")
    print(f"  - 预估时长: {script['estimated_duration_minutes']:.1f} 分钟")

    print(f"\n前 3 个对话片段:")
    for i, segment in enumerate(script['segments'][:3], 1):
        speaker = "张三" if segment['speaker'] == 'host_a' else "李四"
        print(f"\n  {i}. 【{speaker}】")
        print(f"     {segment['text'][:100]}...")

except Exception as e:
    print(f"\n✗ 播客生成失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Claude API 测试完成！")
print("=" * 60)
