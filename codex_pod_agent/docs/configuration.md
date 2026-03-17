#+ Configuration

- 复制 `.env.example` 为 `.env` 并填写值。
- 必填：`ANTHROPIC_API_KEY`（Claude）；选择 `TTS_PROVIDER` 并提供对应密钥。
- 可选：`OPENAI_API_KEY`（若使用 OpenAI TTS）。
  - 若使用自定义网关/代理，可设置：`OPENAI_BASE_URL`（如 `https://your-gateway/v1`）与 `OPENAI_TTS_MODEL`（默认 `gpt-4o-mini-tts`）。

### 较简便的免费 TTS（Edge TTS）
- 无需密钥：设置 `TTS_PROVIDER=edge`。
- 语音示例：`zh-CN-XiaoxiaoNeural`（女声），`zh-CN-YunxiNeural`（男声）。
- 注意：需要网络访问；如需其他语种/音色，可参考 edge-tts 文档列表。
- 兼容：`ANTHROPIC_AUTH_TOKEN` 等价于 `ANTHROPIC_API_KEY`。
- 自定义：`ANTHROPIC_BASE_URL` 指定企业代理/私有网关；`ANTHROPIC_MODEL` 可选择如 `claude-sonnet-4-5`。
- 思考模式：`ANTHROPIC_THINKING=true`（若模型支持），可设置 `ANTHROPIC_THINK_TOKENS` 预算。
- 设置 `PODCAST_BASE_URL` 指向托管 `dist/` 的域名（如 GitHub Pages、S3）。
- `SCHEDULE_TIME` 控制每日运行时间（本地时区，HH:MM）。
- `DRY_RUN=true`：不开启真实API调用，使用本地占位LLM与TTS快速验证流程。

### 上下文管理（Context Management & Loop 工具）
- 历史上下文保存在 `sessions/context.jsonl`（每日一行，包含要点、脚本、链接）。
- 系统会基于历史链接去重，避免重复播报；并在提炼要点时参考近几天脉络。
- 启用 Claude Loop：`.env` 设 `CLAUDE_USE_LOOP=true`。Loop 使用工具调用：
  - `get_history(limit?)`：读取近N天要点与日期
  - `get_seen_links()`：读取历史已播报链接
  - `fetch_url(url)`：抓取网页正文（需要 `requests` 与 `beautifulsoup4`）
- Loop 会在总结或生成脚本过程中按需调用工具；若工具不可用会自动回退合理输出。

#### 平台侧上下文（Claude Developer Platform）
- 可选：通过环境变量开启平台侧上下文：
  - `CLAUDE_PLATFORM_API_URL`：平台上下文服务地址（例如你的中台接口）
  - `CLAUDE_PLATFORM_API_KEY`：鉴权密钥
- 运行时会：
  - 从平台读取最近N条历史（与本地合并）
  - 将当日要点/脚本/链接回写到平台
  - 接口期望：
    - GET `${CLAUDE_PLATFORM_API_URL}/context/recent?limit=N` 返回 `[{date, bullets, script, links}]`
    - POST `${CLAUDE_PLATFORM_API_URL}/context/episode` body `{date, bullets, script, links}`

### Loop 策略配置
- 文件：`codex_pod_agent/config/loop_strategy.yaml`
- 字段：
  - `topk`：要点最多条数（默认 10）
  - `cluster_threshold`：聚类阈值（Jaccard，默认 0.55）
  - `max_tool_steps_summary`：总结阶段最多工具步数（默认 4）
  - `max_tool_steps_script`：脚本阶段最多工具步数（默认 2）

安装依赖：

```
pip install -r requirements.txt
```

本地运行：

```
make dev
```

构建最新节目与RSS：

```
make build
```

示例：启用真实调用

```
# .env
DRY_RUN=false
TTS_PROVIDER=openai
```

### 发布到 GitHub Pages
- 默认每日工作流会将 `dist/` 部署为 Pages 静态站点（可在仓库 Settings → Pages 启用）。
- 订阅地址：`<Pages域名>/feed.xml`；音频：`<Pages域名>/<mp3文件>`。
- 同步更新 `.env` 中 `PODCAST_BASE_URL` 为 Pages 域名（如 `https://<user>.github.io/<repo>`）。

### 发布到 S3（可选）
- 在仓库 Secrets 配置：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_REGION`、`S3_BUCKET`、（可选）`S3_PREFIX`。
- 工作流会将 `dist/` 同步到指定存储桶并设置 `public-read`。
- 将 `PODCAST_BASE_URL` 设为 S3 网站或 CDN 域名。
