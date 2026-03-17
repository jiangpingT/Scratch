"""
音频处理器

处理音频的后期制作，包括：
1. 合并音频片段
2. 添加背景音乐
3. 添加片头片尾
4. 音量标准化
5. 格式转换
"""

import logging
import json
from pathlib import Path
from typing import List, Optional
from pydub import AudioSegment
from pydub.effects import normalize

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器"""

    def __init__(self, config):
        """
        初始化音频处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.audio_config = config.AUDIO_CONFIG

        self.add_background_music = self.audio_config["add_background_music"]
        self.music_volume = self.audio_config["music_volume"]
        self.add_intro = self.audio_config["add_intro"]
        self.add_outro = self.audio_config["add_outro"]
        self.normalize_audio = self.audio_config["normalize_audio"]
        self.output_bitrate = self.audio_config["output_bitrate"]

        self.output_dir = config.STORAGE_CONFIG["local_output_dir"]

    def process_podcast(self, segments_info_path: Path) -> Path:
        """
        处理完整的播客音频

        Args:
            segments_info_path: TTS 生成的片段信息文件路径

        Returns:
            最终音频文件路径
        """
        logger.info("开始音频后期处理...")

        # 读取片段信息
        with open(segments_info_path, 'r', encoding='utf-8') as f:
            segments_info = json.load(f)

        output_file = Path(segments_info['output_file'])
        segment_files = [Path(f) for f in segments_info['segment_files']]

        # 1. 合并音频片段
        logger.info("正在合并音频片段...")
        combined_audio = self._combine_segments(segment_files)

        if combined_audio is None:
            logger.error("合并音频失败")
            return None

        # 2. 添加片头
        if self.add_intro and self.audio_config["intro_file"].exists():
            logger.info("添加片头...")
            combined_audio = self._add_intro(combined_audio)

        # 3. 添加背景音乐
        if self.add_background_music and self.audio_config["background_music_file"].exists():
            logger.info("添加背景音乐...")
            combined_audio = self._add_background_music(combined_audio)

        # 4. 添加片尾
        if self.add_outro and self.audio_config["outro_file"].exists():
            logger.info("添加片尾...")
            combined_audio = self._add_outro(combined_audio)

        # 5. 音量标准化
        if self.normalize_audio:
            logger.info("标准化音量...")
            combined_audio = normalize(combined_audio)

        # 6. 导出最终音频
        logger.info(f"导出最终音频到：{output_file}")
        combined_audio.export(
            str(output_file),
            format="mp3",
            bitrate=self.output_bitrate,
            tags={
                'artist': 'AI 前沿日报',
                'album': 'Daily AI Papers',
                'title': output_file.stem
            }
        )

        # 7. 清理临时文件
        self._cleanup_temp_files(segment_files)

        logger.info(f"音频处理完成！文件大小：{output_file.stat().st_size / 1024 / 1024:.2f} MB")
        return output_file

    def _combine_segments(self, segment_files: List[Path]) -> Optional[AudioSegment]:
        """
        合并音频片段

        Args:
            segment_files: 片段文件列表

        Returns:
            合并后的音频
        """
        try:
            combined = AudioSegment.empty()

            for segment_file in segment_files:
                if not segment_file.exists():
                    logger.warning(f"片段文件不存在：{segment_file}")
                    continue

                segment = AudioSegment.from_mp3(str(segment_file))
                combined += segment

            logger.info(f"成功合并 {len(segment_files)} 个片段，总时长：{len(combined) / 1000:.1f} 秒")
            return combined

        except Exception as e:
            logger.error(f"合并音频片段失败: {e}")
            return None

    def _add_intro(self, audio: AudioSegment) -> AudioSegment:
        """
        添加片头

        Args:
            audio: 主音频

        Returns:
            添加片头后的音频
        """
        try:
            intro_file = self.audio_config["intro_file"]
            intro = AudioSegment.from_file(str(intro_file))

            # 淡出效果
            intro = intro.fade_out(500)

            return intro + audio

        except Exception as e:
            logger.error(f"添加片头失败: {e}")
            return audio

    def _add_outro(self, audio: AudioSegment) -> AudioSegment:
        """
        添加片尾

        Args:
            audio: 主音频

        Returns:
            添加片尾后的音频
        """
        try:
            outro_file = self.audio_config["outro_file"]
            outro = AudioSegment.from_file(str(outro_file))

            # 淡入效果
            outro = outro.fade_in(500)

            return audio + outro

        except Exception as e:
            logger.error(f"添加片尾失败: {e}")
            return audio

    def _add_background_music(self, audio: AudioSegment) -> AudioSegment:
        """
        添加背景音乐

        Args:
            audio: 主音频（人声）

        Returns:
            混合后的音频
        """
        try:
            music_file = self.audio_config["background_music_file"]
            music = AudioSegment.from_file(str(music_file))

            # 降低背景音乐音量
            music = music - (20 * (1 - self.music_volume))  # dB 调整

            # 循环背景音乐以匹配主音频长度
            while len(music) < len(audio):
                music += music

            # 截断到主音频长度
            music = music[:len(audio)]

            # 为背景音乐添加淡入淡出
            music = music.fade_in(2000).fade_out(2000)

            # 混合音频
            combined = audio.overlay(music)

            logger.info("背景音乐添加成功")
            return combined

        except Exception as e:
            logger.error(f"添加背景音乐失败: {e}")
            return audio

    def _cleanup_temp_files(self, temp_files: List[Path]):
        """
        清理临时文件

        Args:
            temp_files: 临时文件列表
        """
        try:
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

            # 删除临时目录
            temp_dir = self.output_dir / "temp"
            if temp_dir.exists() and not list(temp_dir.iterdir()):
                temp_dir.rmdir()

            logger.info("临时文件清理完成")

        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")

    def get_audio_info(self, audio_file: Path) -> dict:
        """
        获取音频文件信息

        Args:
            audio_file: 音频文件路径

        Returns:
            音频信息字典
        """
        try:
            audio = AudioSegment.from_file(str(audio_file))

            return {
                'duration_seconds': len(audio) / 1000,
                'duration_formatted': f"{len(audio) // 60000}:{(len(audio) % 60000) // 1000:02d}",
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'file_size_mb': audio_file.stat().st_size / 1024 / 1024
            }

        except Exception as e:
            logger.error(f"获取音频信息失败: {e}")
            return {}

    def create_sample_assets(self):
        """
        创建示例资源文件（片头、片尾、背景音乐）

        用于测试
        """
        logger.info("创建示例音频资源...")

        assets_dir = self.config.BASE_DIR / "assets"
        assets_dir.mkdir(exist_ok=True)

        try:
            from pydub.generators import Sine

            # 创建简单的片头音效（440Hz 正弦波 2秒）
            intro = Sine(440).to_audio_segment(duration=2000).fade_in(100).fade_out(500)
            intro_path = assets_dir / "intro.mp3"
            intro.export(str(intro_path), format="mp3")
            logger.info(f"片头创建完成：{intro_path}")

            # 创建简单的片尾音效
            outro = Sine(330).to_audio_segment(duration=3000).fade_in(500).fade_out(1000)
            outro_path = assets_dir / "outro.mp3"
            outro.export(str(outro_path), format="mp3")
            logger.info(f"片尾创建完成：{outro_path}")

            # 创建简单的背景音乐（和弦）
            bg_music = (
                Sine(262).to_audio_segment(duration=10000) +  # C
                Sine(330).to_audio_segment(duration=10000) +  # E
                Sine(392).to_audio_segment(duration=10000)    # G
            ) - 20  # 降低音量

            bg_music_path = assets_dir / "background_music.mp3"
            bg_music.export(str(bg_music_path), format="mp3")
            logger.info(f"背景音乐创建完成：{bg_music_path}")

        except Exception as e:
            logger.error(f"创建示例资源失败: {e}")


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

    # 创建处理器
    processor = AudioProcessor(config)

    # 创建示例资源
    print("\n创建示例音频资源...")
    processor.create_sample_assets()

    print("\n音频处理器就绪！")


if __name__ == "__main__":
    main()
