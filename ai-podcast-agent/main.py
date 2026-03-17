#!/usr/bin/env python3
"""
AI 播客 Agent - 主程序

每天自动生成 AI 前沿播客的完整流程：
1. 获取最新论文
2. 分析论文内容
3. 生成播客脚本
4. 合成语音
5. 后期处理
6. 发布播客

使用 Claude Agent SDK 的 Loop 和 Context Management
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# 导入配置和模块
import config
from agents.paper_fetcher import PaperFetcher
from agents.context_manager import ContextManager
from agents.content_analyzer import ContentAnalyzer
from agents.podcast_generator import PodcastGenerator
from services.tts_service import TTSService
from services.audio_processor import AudioProcessor
from scheduler import PodcastScheduler


def setup_logging():
    """配置日志系统"""
    log_config = config.LOGGING_CONFIG

    # 创建日志目录
    log_config["log_dir"].mkdir(parents=True, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config["log_level"]),
        format=log_config["log_format"],
        handlers=[
            # 控制台输出
            logging.StreamHandler(sys.stdout),
            # 文件输出（带轮转）
            logging.handlers.RotatingFileHandler(
                log_config["log_dir"] / "podcast_agent.log",
                maxBytes=log_config["log_file_max_bytes"],
                backupCount=log_config["log_file_backup_count"],
                encoding='utf-8'
            )
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("AI 播客 Agent 启动")
    logger.info(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return logger


def generate_daily_podcast():
    """
    生成每日播客的主流程

    这是核心的任务函数，被调度器调用
    """
    logger = logging.getLogger(__name__)

    try:
        # ===== 步骤 1: 初始化所有组件 =====
        logger.info("\n【步骤 1/6】初始化组件...")

        paper_fetcher = PaperFetcher(config)
        context_manager = ContextManager(config)
        content_analyzer = ContentAnalyzer(config)
        podcast_generator = PodcastGenerator(config, context_manager)  # 传递 context_manager
        tts_service = TTSService(config)
        audio_processor = AudioProcessor(config)

        logger.info("所有组件初始化完成 ✓")

        # ===== 步骤 2: 获取最新论文 =====
        logger.info("\n【步骤 2/6】获取最新论文...")

        papers = paper_fetcher.fetch_all()

        if not papers:
            logger.warning("未获取到任何论文，跳过今天的播客生成")
            return

        logger.info(f"成功获取 {len(papers)} 篇论文 ✓")

        # ===== 步骤 3: 上下文管理和去重 =====
        logger.info("\n【步骤 3/6】上下文管理和去重...")

        # 添加到数据库
        new_papers_count = context_manager.add_papers(papers)
        logger.info(f"新增 {new_papers_count} 篇论文到数据库")

        # 获取未使用的论文
        unused_papers = context_manager.get_unused_papers(
            limit=config.CONTENT_ANALYZER_CONFIG["max_papers"]
        )

        if not unused_papers:
            logger.warning("没有未使用的论文，跳过今天的播客生成")
            return

        logger.info(f"选择 {len(unused_papers)} 篇论文进行分析 ✓")

        # ===== 步骤 4: 分析论文内容 =====
        logger.info("\n【步骤 4/6】使用 Claude 分析论文...")

        analyzed_papers = content_analyzer.analyze_papers(unused_papers)

        # 排序论文
        analyzed_papers = content_analyzer.rank_papers(analyzed_papers)

        # 生成每日总览
        daily_summary = content_analyzer.generate_daily_summary(analyzed_papers)
        logger.info(f"每日总览：{daily_summary}")

        logger.info(f"论文分析完成，共 {len(analyzed_papers)} 篇 ✓")

        # ===== 步骤 5: 生成播客脚本 =====
        logger.info("\n【步骤 5/6】使用 Claude Loop 生成播客脚本...")

        podcast_script = podcast_generator.generate_podcast_script(
            analyzed_papers,
            daily_summary
        )

        # 保存脚本
        script_file = config.STORAGE_CONFIG["local_output_dir"] / f"script_{datetime.now().strftime('%Y%m%d')}.json"
        with open(script_file, 'w', encoding='utf-8') as f:
            json.dump(podcast_script, f, ensure_ascii=False, indent=2)

        logger.info(f"播客脚本已保存：{script_file} ✓")

        # ===== 步骤 6: 生成音频 =====
        logger.info("\n【步骤 6/6】生成音频...")

        # 6.1 TTS 生成
        logger.info("正在使用 TTS 生成语音...")
        segments_info_file = tts_service.generate_audio(
            podcast_script,
            output_filename=f"podcast_{datetime.now().strftime('%Y%m%d')}.mp3"
        )

        # 6.2 音频后期处理
        logger.info("正在进行音频后期处理...")
        final_audio_file = audio_processor.process_podcast(segments_info_file)

        if not final_audio_file:
            logger.error("音频处理失败")
            return

        # 获取音频信息
        audio_info = audio_processor.get_audio_info(final_audio_file)
        logger.info(f"音频生成成功：{final_audio_file}")
        logger.info(f"时长：{audio_info.get('duration_formatted', '未知')}")
        logger.info(f"文件大小：{audio_info.get('file_size_mb', 0):.2f} MB ✓")

        # ===== 步骤 7: 更新上下文 =====
        logger.info("\n【步骤 7/6】更新上下文...")

        paper_ids = [p['paper_id'] for p in analyzed_papers]
        context_manager.mark_papers_used(paper_ids)

        # 清理旧论文
        context_manager.cleanup_old_papers()

        logger.info("上下文更新完成 ✓")

        # ===== 完成 =====
        logger.info("\n" + "=" * 80)
        logger.info("🎉 播客生成完成！")
        logger.info(f"📁 输出文件：{final_audio_file}")
        logger.info(f"📄 脚本文件：{script_file}")
        logger.info(f"⏱️  时长：{audio_info.get('duration_formatted', '未知')}")
        logger.info(f"📊 论文数：{len(analyzed_papers)} 篇")
        logger.info("=" * 80)

        # TODO: 发送通知
        # TODO: 上传到云存储

    except Exception as e:
        logger.error(f"播客生成失败: {e}", exc_info=True)
        raise


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AI 播客 Agent')
    parser.add_argument(
        '--mode',
        choices=['once', 'schedule'],
        default='schedule',
        help='运行模式：once=运行一次，schedule=定时调度（默认）'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式（使用测试配置）'
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    # 测试模式
    if args.test:
        logger.info("⚠️  测试模式已启用")
        import os
        os.environ['TEST_MODE'] = 'True'

    # 检查 API 密钥
    if not config.ANTHROPIC_API_KEY:
        logger.error("❌ ANTHROPIC_API_KEY 未设置，请在环境变量中配置")
        sys.exit(1)

    if not config.OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY 未设置，请在环境变量中配置")
        sys.exit(1)

    # 运行模式
    if args.mode == 'once':
        logger.info("🚀 单次运行模式")
        generate_daily_podcast()
    else:
        logger.info("⏰ 定时调度模式")
        scheduler = PodcastScheduler(config, generate_daily_podcast)
        scheduler.start()


if __name__ == "__main__":
    main()
