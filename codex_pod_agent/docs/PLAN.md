#+ Agent 实施计划（当前）

- [x] 初始化 Python 包与目录结构：创建 `src/python/podagent`、`Makefile` 与脚本骨架。
- [x] 实现抓取 arXiv 与 RSS：`fetchers/arxiv.py`、`fetchers/rss.py` 聚合最新论文与资讯。
- [x] 集成 Claude 总结与上下文管理：`llm/claude_agent.py`（Messages），`llm/claude_loop.py`（Loop 占位，可开关）。
- [x] 生成双人对话与 TTS 占位：从要点生成 A/B 对话脚本并合成音频（OpenAI TTS）。
- [x] 加入调度与 RSS 播客输出：`runner.py` 生成当日节目，`feed/generate_feed.py` 输出 `feed.xml`。
- [x] 添加脚本、Makefile 与 .env 示例：`scripts/*.sh`、`.env.example`、`requirements.txt`。
- [x] 提供使用说明与后续建议：`docs/configuration.md` 与命令示例。
 - [x] 引入上下文管理与去重：`sessions/context.jsonl` 存储历史，生成时参考脉络并过滤重复链接。
 - [x] 每日部署：GitHub Pages 自动发布 `dist/`；可选 S3 同步（基于仓库 Secrets）。

## 使用要点
- 一次运行：`python -m podagent.runner --once` 或 `make dev`
- 生成 RSS：`python -m podagent.feed.generate_feed` 或 `make build`
- 定时（示例 cron，本地时区 09:00）：
  - `0 9 * * * cd /path/to/repo && /usr/bin/env -S bash -lc "source .venv/bin/activate; make build"`

## 配置摘要
- 必填：`ANTHROPIC_API_KEY`；若使用 TTS(OpenAI)，需 `OPENAI_API_KEY`
- 可选：`CLAUDE_USE_LOOP=true` 启用 Claude Agents SDK Loop（当前回退到 Messages）
- 输出：`dist/episodes/YYYY-MM-DD-daily-ai-duo.mp3` 与 `dist/feed.xml`
