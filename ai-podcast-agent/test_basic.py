#!/usr/bin/env python3
"""
基础功能测试脚本

测试不需要 API 密钥的基础功能
"""

import sys
print("Python 版本:", sys.version)
print("=" * 60)

# 测试 1: 导入配置
print("\n【测试 1】导入配置模块...")
try:
    import config
    print("✓ 配置模块导入成功")
    print(f"  - 设备: {config.DEVICE}")
    print(f"  - 数据库路径: {config.STORAGE_CONFIG['database_path']}")
    print(f"  - 调度时间: {config.SCHEDULER_CONFIG['daily_run_time']}")
except Exception as e:
    print(f"✗ 配置模块导入失败: {e}")

# 测试 2: 论文获取器
print("\n【测试 2】测试论文获取器...")
try:
    from agents.paper_fetcher import PaperFetcher

    # 创建获取器（使用测试配置）
    import os
    os.environ['TEST_MODE'] = 'True'

    fetcher = PaperFetcher(config)
    print("✓ 论文获取器初始化成功")

    # 修改配置以获取更多时间范围的论文
    config.PAPER_FILTER["max_hours_ago"] = 168  # 7天

    print("  正在从 arXiv 获取论文（最近 7 天）...")
    papers = fetcher.fetch_from_arxiv()
    print(f"✓ 获取到 {len(papers)} 篇论文")

    if papers:
        print(f"\n  示例论文:")
        for i, paper in enumerate(papers[:2], 1):
            print(f"    {i}. {paper.title[:60]}...")
            print(f"       作者: {', '.join(paper.authors[:2])}")
            print(f"       日期: {paper.published_date.strftime('%Y-%m-%d')}")

except Exception as e:
    print(f"✗ 论文获取器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 上下文管理器
print("\n【测试 3】测试上下文管理器...")
try:
    from agents.context_manager import ContextManager

    manager = ContextManager(config)
    print("✓ 上下文管理器初始化成功")

    # 获取统计信息
    stats = manager.get_statistics()
    print(f"  数据库统计:")
    print(f"    - 总论文数: {stats['total_papers']}")
    print(f"    - 已使用: {stats['used_papers']}")
    print(f"    - 未使用: {stats['unused_papers']}")

    # 如果有论文，添加到数据库
    if 'papers' in locals() and papers:
        print(f"\n  添加 {len(papers[:3])} 篇论文到数据库（测试）...")
        new_count = manager.add_papers(papers[:3])
        print(f"✓ 新增 {new_count} 篇论文")

        # 再次获取统计
        stats = manager.get_statistics()
        print(f"  更新后的统计:")
        print(f"    - 总论文数: {stats['total_papers']}")

except Exception as e:
    print(f"✗ 上下文管理器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 调度器
print("\n【测试 4】测试调度器...")
try:
    from scheduler import PodcastScheduler

    def dummy_task():
        print("  这是一个测试任务")

    scheduler = PodcastScheduler(config, dummy_task)
    print("✓ 调度器初始化成功")
    print(f"  配置的运行时间: {scheduler.daily_run_time}")
    print(f"  时区: {scheduler.timezone}")

except Exception as e:
    print(f"✗ 调度器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "=" * 60)
print("基础功能测试完成！")
print("\n注意事项:")
print("  • 音频处理功能需要 ffmpeg 和 Python 3.12 或更早版本")
print("  • Claude 内容分析需要 ANTHROPIC_API_KEY")
print("  • TTS 功能需要 OPENAI_API_KEY")
print("\n下一步:")
print("  1. 配置 .env 文件（复制 .env.example）")
print("  2. 填入 API 密钥")
print("  3. 运行: python main.py --mode once --test")
print("=" * 60)
