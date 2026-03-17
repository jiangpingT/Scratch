from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import List, Tuple

from .. import config


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def tts_openai(text: str, voice: str, out_path: Path) -> None:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai 未安装：pip install openai") from e
    if not config.OPENAI_API_KEY:
        raise RuntimeError("缺少 OPENAI_API_KEY")
    kwargs = {"api_key": config.OPENAI_API_KEY}
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    client = OpenAI(**kwargs)
    try:
        try:
            with client.audio.speech.with_streaming_response.create(
                    model=config.OPENAI_TTS_MODEL,
                    voice=voice,
                    input=text,
                    audio_format="mp3",
            ) as resp:
                resp.stream_to_file(str(out_path))
        except TypeError:
            # 兼容旧版 SDK 参数名
            with client.audio.speech.with_streaming_response.create(
                    model=config.OPENAI_TTS_MODEL,
                    voice=voice,
                    input=text,
                    format="mp3",
            ) as resp:
                resp.stream_to_file(str(out_path))
    except Exception:
        # 回退：改为静音片段（不再生成嘟嘟声）
        from pydub import AudioSegment  # type: ignore
        dur = max(800, min(3000, len(text) * 40))
        AudioSegment.silent(duration=dur).export(out_path, format="mp3")


async def _tts_edge_async(turns: List[Tuple[str, str]], parts: List[Path]) -> None:
    try:
        import edge_tts  # type: ignore
    except Exception as e:
        raise RuntimeError("edge-tts 未安装：pip install edge-tts") from e

    for (spk, text), part in zip(turns, parts):
        voice = config.TTS_VOICE_A if spk == "A" else config.TTS_VOICE_B
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(part))
        except Exception:
            # 失败回退为静音（不再生成嘟嘟声）
            from pydub import AudioSegment  # type: ignore
            AudioSegment.silent(duration=max(600, min(2500, len(text) * 40))).export(part, format="mp3")


def synthesize_dialogue(turns: List[Tuple[str, str]], out_file: str) -> str:
    _ensure_dir(Path(out_file).parent)
    tmp_dir = Path(Path(out_file).parent, ".tmp")
    _ensure_dir(tmp_dir)

    chunks: List[Path] = []
    for i, (spk, text) in enumerate(turns):
        voice = config.TTS_VOICE_A if spk == "A" else config.TTS_VOICE_B
        part = tmp_dir / f"part_{i:03d}_{spk}.mp3"
        chunks.append(part)

    if config.TTS_PROVIDER == "openai":
        for (spk, text), part in zip(turns, chunks):
            tts_openai(text, config.TTS_VOICE_A if spk == "A" else config.TTS_VOICE_B, part)
    elif config.TTS_PROVIDER == "edge":
        # 使用 Edge TTS 一次性并发生成各段
        try:
            asyncio.run(_tts_edge_async(turns, chunks))
        except RuntimeError:
            # 在已存在事件循环的环境（如某些框架）下的兼容处理
            loop = asyncio.get_event_loop()
            coro = _tts_edge_async(turns, chunks)
            loop.run_until_complete(coro)
    elif config.TTS_PROVIDER in {"stub", "none"} or config.DRY_RUN:
        # 占位改为静音片段（无嘟嘟声）
        from pydub import AudioSegment  # type: ignore
        for (spk, text), part in zip(turns, chunks):
            AudioSegment.silent(duration=max(600, min(2500, len(text) * 40))).export(part, format="mp3")
    else:
        raise RuntimeError("未配置可用的TTS_PROVIDER。可设置 TTS_PROVIDER=edge/openai 或 TTS_PROVIDER=stub 进行占位。")

    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as e:
        raise RuntimeError("pydub 未安装：pip install pydub，并确保系统有 ffmpeg") from e

    combined = AudioSegment.silent(duration=300)
    for c in chunks:
        combined += AudioSegment.from_file(c)
        combined += AudioSegment.silent(duration=150)
    combined.export(out_file, format="mp3")

    for c in chunks:
        try:
            os.remove(c)
        except Exception:
            pass
    return out_file
