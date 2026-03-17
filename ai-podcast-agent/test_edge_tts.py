#!/usr/bin/env python3
"""
使用免费 Edge TTS 测试完整播客生成

无需 OpenAI API，完全免费！
"""

import asyncio
import sys
import json
from pathlib import Path

print("=" * 60)
print("免费 Edge TTS 播客生成测试")
print("=" * 60)

# 检查 edge-tts 是否安装
try:
    import edge_tts
    print("✓ Edge TTS 已安装")
except ImportError:
    print("✗ Edge TTS 未安装")
    print("\n请运行：pip install edge-tts")
    sys.exit(1)

# 导入项目模块
print("\n【1】加载配置...")
import config
from agents.paper_fetcher import PaperFetcher
from agents.content_analyzer import ContentAnalyzer
from agents.podcast_generator import PodcastGenerator
from agents.context_manager import ContextManager

print("✓ 所有模块加载成功")

# 获取论文
print("\n【2】获取测试论文...")
fetcher = PaperFetcher(config)
config.PAPER_FILTER["max_hours_ago"] = 168
papers = fetcher.fetch_from_arxiv()

if not papers:
    print("✗ 未获取到论文")
    sys.exit(1)

test_paper = papers[0]
print(f"✓ 获取到 {len(papers)} 篇论文")
print(f"  测试论文: {test_paper.title[:60]}...")

# 分析论文
print("\n【3】Claude 分析论文...")
analyzer = ContentAnalyzer(config)
paper_dict = {
    'paper_id': test_paper.paper_id,
    'title': test_paper.title,
    'authors': test_paper.authors,
    'abstract': test_paper.abstract,
    'url': test_paper.url,
    'published_date': test_paper.published_date.isoformat()
}

analysis = analyzer.analyze_single_paper(paper_dict)
print("✓ 论文分析完成")

# 生成播客脚本（使用 Agent Loop 和 Context Management）
print("\n【4】使用 Agent Loop 生成播客脚本...")
context_manager = ContextManager(config)
generator = PodcastGenerator(config, context_manager)
script = generator.generate_podcast_script([analysis])
print(f"✓ 脚本生成完成（{len(script['segments'])} 个片段）")

# 保存脚本
output_dir = Path("output/podcasts")
output_dir.mkdir(parents=True, exist_ok=True)

script_file = output_dir / "podcast_edge_tts_test.json"
with open(script_file, 'w', encoding='utf-8') as f:
    json.dump(script, f, ensure_ascii=False, indent=2)
print(f"  脚本已保存: {script_file}")

# 使用 Edge TTS 生成音频
print("\n【5】使用免费 Edge TTS 生成音频...")
print("  这可能需要几分钟...")


async def generate_podcast_audio():
    """异步生成播客音频"""
    segments_audio = []

    # 定义声音
    voice_a = "zh-CN-XiaoxiaoNeural"  # 女声（主持人张三）
    voice_b = "zh-CN-YunxiNeural"     # 男声（嘉宾李四）

    print(f"\n  使用声音：")
    print(f"    - 主持人张三: {voice_a}")
    print(f"    - 嘉宾李四: {voice_b}")

    # 为每个片段生成音频
    for i, segment in enumerate(script['segments']):
        speaker = segment['speaker']
        text = segment['text']

        # 选择声音
        voice = voice_a if speaker == 'host_a' else voice_b
        speaker_name = "张三" if speaker == 'host_a' else "李四"

        print(f"\n  生成片段 {i+1}/{len(script['segments'])} ({speaker_name})...")

        # 生成音频文件
        audio_file = output_dir / f"segment_{i:03d}_{speaker}.mp3"

        tts = edge_tts.Communicate(text, voice)
        await tts.save(str(audio_file))

        segments_audio.append(audio_file)
        print(f"    ✓ {audio_file.name}")

    return segments_audio


# 运行异步函数
try:
    segments = asyncio.run(generate_podcast_audio())
    print(f"\n✓ 成功生成 {len(segments)} 个音频片段")
except Exception as e:
    print(f"\n✗ 音频生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 合并音频（如果 pydub 可用）
print("\n【6】合并音频片段...")
try:
    from pydub import AudioSegment

    # 合并所有片段
    combined = AudioSegment.empty()
    for segment_file in segments:
        audio = AudioSegment.from_mp3(str(segment_file))
        combined += audio
        # 添加 0.5 秒停顿
        combined += AudioSegment.silent(duration=500)

    # 导出最终音频
    final_audio = output_dir / "podcast_final_edge_tts.mp3"
    combined.export(str(final_audio), format="mp3", bitrate="192k")

    # 获取音频信息
    duration_sec = len(combined) / 1000
    duration_min = int(duration_sec // 60)
    duration_sec = int(duration_sec % 60)
    file_size_mb = final_audio.stat().st_size / 1024 / 1024

    print(f"✓ 音频合并成功！")
    print(f"\n📁 最终播客文件:")
    print(f"  文件: {final_audio}")
    print(f"  时长: {duration_min}:{duration_sec:02d}")
    print(f"  大小: {file_size_mb:.2f} MB")

    # 清理临时文件
    for segment_file in segments:
        segment_file.unlink()
    print(f"\n✓ 临时文件已清理")

except ImportError:
    print("⚠️  pydub 不可用（Python 3.13 问题），音频片段未合并")
    print(f"\n📁 音频片段已生成:")
    for segment_file in segments:
        print(f"  - {segment_file}")
    print(f"\n您可以手动合并这些文件，或使用 Python 3.12")

except Exception as e:
    print(f"✗ 音频合并失败: {e}")
    print(f"\n📁 音频片段:")
    for segment_file in segments:
        print(f"  - {segment_file}")

# 总结
print("\n" + "=" * 60)
print("🎉 免费播客生成完成！")
print("=" * 60)
print(f"\n使用的技术:")
print(f"  - 论文获取: arXiv API")
print(f"  - 内容分析: Claude API (自定义端点)")
print(f"  - 脚本生成: Claude Loop 机制")
print(f"  - 语音合成: Edge TTS (免费)")
print(f"\n成本: $0 (完全免费！)")
print("=" * 60)
