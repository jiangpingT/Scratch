# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` by language: `src/python/`, `src/js/`.
- Tests mirror sources under `tests/`: `tests/python/`, `tests/js/`.
- Tooling/docs/assets: `scripts/` (helpers), `docs/` (documentation), `assets/` (static).
- Sessions and logs: `sessions/`, `log/`.
- Example mapping: `src/python/package/module.py` ↔ `tests/python/package/test_module.py`.

## Build, Test, and Development Commands
- `make dev`: 本地运行一次管线（尊重 `.env` 中 `DRY_RUN` 设置）。
- `make build`: 生成最新节目与 RSS（`dist/episodes/*.mp3` 与 `dist/feed.xml`）。
- `make test`: 运行测试（如配置了 pytest）。
- `make lint`: 代码格式/静态检查。

## Coding Style & Naming Conventions
- Python: PEP8, 4-space indent; files/functions `snake_case`, classes `PascalCase`.
- JS/TS: Prettier + `eslint:recommended`; `camelCase` for vars/functions, `PascalCase` for classes.
- Shell: format with `shfmt`; script filenames use kebab-case (e.g., `release-version.sh`).
- Enforce whitespace/newlines with `.editorconfig`.

## Testing Guidelines
- Frameworks: Python `pytest`; Node `npm test`.
- Naming: Python `tests/**/test_*.py`; JS/TS `**/*.spec.ts`.
- Coverage: target ≥80% on changed code; mark slow/integration tests (`@pytest.mark.slow`, `it.skip`).
- Run locally with `make test`.

## Architecture Overview（简要）
- 抓取层：`fetchers/*` 收集 arXiv 与 RSS。
- 上下文：`sessions/context.jsonl` 存储历史要点/脚本/链接，支持去重与脉络传递。
- LLM 层：`llm/claude_agent.py`（Messages）与 `llm/claude_loop.py`（Loop 工具：history/seen_links/fetch_url/rank/cluster），支持 `DRY_RUN` 占位。
- 播客渲染：`podcast/*` 对话解析、TTS 合成、音频拼接。
- 分析：`analysis/rank.py`（打分排序）、`analysis/cluster.py`（轻量聚类），`utils/readability.py`（正文提取）。
- 策略：`config/loop_strategy.yaml` 外置 Loop 策略（topk/阈值/工具步数）。
- 输出：`dist/episodes/*.mp3` 与 `dist/feed.xml`；CI 部署到 Pages/S3。
- 校验（Verify）：`verify.py` 对要点与脚本进行二次审校与修正（Claude验证），并与解析清洗共同保障可播质量。
- 平台上下文：`context/manager.py` 合并本地与平台侧上下文，`context/platform.py` 通过自定义 API URL/KEY 读写历史。

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `build:`).
- PRs: clear description, linked issues, before/after notes or screenshots (if output/UI changes), and testing instructions.
- Keep patches focused and consistent with existing style.

## Security & Configuration Tips
- 切勿提交密钥。使用 `.env`（基于 `.env.example`）。详情见 `docs/configuration.md`。
- 忽略构建产物与缓存：`dist/`、`.venv/`、`node_modules/`、`__pycache__/`。
- 无密钥验证：`DRY_RUN=true` 与 `TTS_PROVIDER=stub` 可进行离线跑通。
- 免费语音：使用 `TTS_PROVIDER=edge` 可通过 Edge TTS 合成（需网络）。

## Agent‑Specific Instructions
- Prefer small, targeted patches; explore with `rg`, edit with `apply_patch`.
- Only change task‑relevant code; update docs/scripts when adding commands.
- Validate locally with `make test` and `make lint` before handoff.
