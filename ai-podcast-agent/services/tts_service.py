"""
TTS (Text-to-Speech) 服务

支持多种 TTS 提供商：
- OpenAI TTS API
- ElevenLabs (可选)
- Azure TTS (可选)

将文本转换为音频文件
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)


class TTSService:
    """TTS 服务类"""

    def __init__(self, config):
        """
        初始化 TTS 服务

        Args:
            config: 配置对象
        """
        self.config = config

        # OpenAI TTS
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.tts_model = config.OPENAI_TTS_MODEL
        self.voice_a = config.OPENAI_TTS_VOICE_A
        self.voice_b = config.OPENAI_TTS_VOICE_B

        # TTS 配置
        self.tts_config = config.TTS_CONFIG
        self.speed = self.tts_config["speed"]
        self.output_format = self.tts_config["output_format"]

        # 输出目录
        self.output_dir = config.STORAGE_CONFIG["local_output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 是否使用 ElevenLabs
        self.use_elevenlabs = config.USE_ELEVENLABS

    def generate_audio(self, script: Dict, output_filename: str = None) -> Path:
        """
        从脚本生成完整的播客音频

        Args:
            script: 播客脚本字典
            output_filename: 输出文件名（可选）

        Returns:
            生成的音频文件路径
        """
        if not output_filename:
            date_str = script.get('date', 'unknown').replace('-', '')
            output_filename = f"podcast_{date_str}.mp3"

        logger.info(f"开始生成音频：{output_filename}")

        # 为每个片段生成音频
        segment_files = []

        for i, segment in enumerate(script['segments']):
            logger.info(f"正在生成第 {i+1}/{len(script['segments'])} 个片段...")

            speaker = segment['speaker']
            text = segment['text']

            # 生成单个片段的音频
            segment_file = self._generate_segment_audio(
                text=text,
                speaker=speaker,
                segment_id=i
            )

            if segment_file:
                segment_files.append(segment_file)

                # 如果需要添加停顿
                if self.tts_config["add_pauses"] and i < len(script['segments']) - 1:
                    pause_file = self._generate_pause(
                        duration=self.tts_config["pause_duration"]
                    )
                    if pause_file:
                        segment_files.append(pause_file)

        logger.info(f"成功生成 {len(segment_files)} 个音频片段")

        # 合并所有音频片段
        final_audio_path = self.output_dir / output_filename

        # 这里返回片段列表，实际合并由 AudioProcessor 完成
        # 临时保存片段列表信息
        import json
        segments_info_path = self.output_dir / f"{output_filename}.segments.json"
        with open(segments_info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'output_file': str(final_audio_path),
                'segment_files': [str(f) for f in segment_files]
            }, f, indent=2)

        logger.info(f"音频片段信息已保存到：{segments_info_path}")
        return segments_info_path

    def _generate_segment_audio(self, text: str, speaker: str, segment_id: int) -> Optional[Path]:
        """
        生成单个片段的音频

        Args:
            text: 文本内容
            speaker: 说话人（host_a 或 host_b）
            segment_id: 片段 ID

        Returns:
            音频文件路径
        """
        # 选择声音
        voice = self.voice_a if speaker == 'host_a' else self.voice_b

        # 生成文件名
        segment_filename = f"segment_{segment_id:03d}_{speaker}.mp3"
        segment_path = self.output_dir / "temp" / segment_filename
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 调用 OpenAI TTS API
            response = self.openai_client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                speed=self.speed
            )

            # 保存音频文件
            response.stream_to_file(str(segment_path))

            logger.debug(f"片段音频已生成：{segment_path}")
            return segment_path

        except Exception as e:
            logger.error(f"生成片段音频失败 (segment {segment_id}): {e}")
            return None

    def _generate_pause(self, duration: float) -> Optional[Path]:
        """
        生成静音片段

        Args:
            duration: 静音时长（秒）

        Returns:
            静音文件路径
        """
        try:
            from pydub import AudioSegment

            # 生成静音
            silence = AudioSegment.silent(duration=int(duration * 1000))  # 毫秒

            # 保存到临时文件
            pause_path = self.output_dir / "temp" / f"pause_{duration}s.mp3"
            pause_path.parent.mkdir(parents=True, exist_ok=True)

            silence.export(str(pause_path), format="mp3")

            return pause_path

        except Exception as e:
            logger.error(f"生成静音片段失败: {e}")
            return None

    def test_voices(self):
        """
        测试两个声音

        用于预览声音效果
        """
        test_text_a = "大家好，我是主持人张三，欢迎收听今天的 AI 前沿日报。"
        test_text_b = "你好，我是李四。今天有几篇非常有趣的论文要和大家分享。"

        logger.info("正在生成声音测试音频...")

        # 生成测试音频 A
        test_file_a = self._generate_segment_audio(test_text_a, "host_a", 9999)
        if test_file_a:
            logger.info(f"主持人 A 测试音频：{test_file_a}")

        # 生成测试音频 B
        test_file_b = self._generate_segment_audio(test_text_b, "host_b", 9998)
        if test_file_b:
            logger.info(f"主持人 B 测试音频：{test_file_b}")

        return test_file_a, test_file_b


class ElevenLabsTTSService:
    """
    ElevenLabs TTS 服务（可选）

    提供更高质量的语音合成
    """

    def __init__(self, config):
        """
        初始化 ElevenLabs TTS

        Args:
            config: 配置对象
        """
        self.config = config
        self.api_key = config.ELEVENLABS_API_KEY

        # TODO: 实现 ElevenLabs API 集成
        logger.warning("ElevenLabs TTS 功能待实现")

    def generate_audio(self, text: str, voice_id: str) -> Optional[Path]:
        """
        使用 ElevenLabs 生成音频

        Args:
            text: 文本
            voice_id: 声音 ID

        Returns:
            音频文件路径
        """
        # TODO: 实现
        raise NotImplementedError("ElevenLabs TTS 待实现")


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

    # 创建服务
    tts = TTSService(config)

    # 测试声音
    print("\n测试声音效果...")
    tts.test_voices()

    print("\n音频生成完成！")


if __name__ == "__main__":
    main()
