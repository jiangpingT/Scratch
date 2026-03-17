# AI 播客生成 Agent

## 1. 项目目标

创建一个自动化代理（Agent），该代理能够每天定时完成以下任务：
1.  **获取信息**：搜集关于大模型（LLM）、AI Agent、机器人领域的最新论文或深度观点。
2.  **内容生成**：将搜集到的信息制作成一期双人对话形式的播客。
3.  **便捷播放**：让用户能够在手机上方便地收听生成的播客。

**触发机制**：每天早上 09:00 自动执行。

## 2. 核心模块设计

-   **信息搜集员 (Scout)**: 使用 SerpApi 在 Google News 上搜索最新文章。
-   **分析师 (Analyst)**: 使用 Claude-3.5-Sonnet 模型阅读文章，提炼核心观点。
-   **剧作家 (Writer)**: 使用 Claude-3.5-Sonnet 模型将观点改编成双人播客脚本。
-   **播音员 (Voice)**: 使用 gTTS 将脚本转换为中文语音 MP3 文件。
-   **发布者 (Publisher)**: (可选) 启动一个本地 Web 服务器，方便手机访问。
-   **调度器 (Scheduler)**: (可选) 使用 `cron` 实现每日自动执行。

---

## 3. 使用与测试手册

请按照以下步骤来配置和运行您的 AI 播客 Agent。

### **第一步：配置 API 密钥**

本项目需要两个 API 密钥才能工作。

1.  **Anthropic (Claude) 密钥**:
    *   **用途**: 用于内容分析和脚本生成。
    *   **变量名**: `ANTHROPIC_AUTH_TOKEN`
    *   **代理地址**: `ANTHROPIC_BASE_URL` (已根据您的提供配置为 `https://vibe.deepminer.ai`)

2.  **SerpApi 密钥**:
    *   **用途**: 用于从 Google 搜索最新文章。
    *   **获取地址**: [SerpApi 官网](https://serpapi.com/users/sign_up)
    *   **变量名**: `SERPAPI_API_KEY`

**配置方式 (二选一):**

*   **(推荐) 设置环境变量**: 在终端中运行以下命令。这种方式更安全，无需修改代码。
    ```bash
    export ANTHROPIC_AUTH_TOKEN="sk-..." # 填入您的 Claude 密钥
    export SERPAPI_API_KEY="..."         # 填入您的 SerpApi 密钥
    ```

*   **(备用) 直接修改代码**: 打开 `main.py` 文件，在文件顶部的“配置”区域，直接修改以下几行：
    ```python
    ANTHROPIC_API_KEY = "sk-..." # 您的 Claude 密钥
    # ANTHROPIC_BASE_URL = "..." # 您的代理地址 (如果需要修改)
    SERPAPI_API_KEY = "..."      # 您的 SerpApi 密钥
    ```

### **第二步：安装依赖**

进入项目目录，然后运行命令来安装所有必需的 Python 库。

```bash
cd /Users/mlamp/Workspace/Scratch/podcast_agent
pip install -r requirements.txt
```

### **第三步：测试运行**

完成配置和安装后，您就可以进行第一次测试运行了。

```bash
# 确保您在 podcast_agent 目录下
python main.py
```

您将看到终端开始打印各个模块的运行日志，例如“正在搜索最新文章...”、“正在生成摘要和观点...”等。

如果一切顺利，几分钟后，您会看到“任务成功完成！”的提示。此时，在 `podcast_agent` 目录下会生成一个名为 `podcast_YYYY-MM-DD.mp3` 的音频文件。您可以直接在电脑上播放它。

### **第四步 (可选): 手机播放**

如果您想在手机上收听播客，可以启动内置的 Web 服务器。

1.  打开 `main.py` 文件。
2.  找到文件末尾的 `main` 函数。
3.  取消最后一行 `# start_server()` 的注释。
4.  重新运行 `python main.py`。

脚本在生成 MP3 文件后，会启动一个 Web 服务器。在您手机的浏览器中访问 `http://<你的电脑IP地址>:8000/`，点击对应的 `.mp3` 文件即可在线播放。

> **提示**: 您可以在终端中使用 `ifconfig | grep "inet "` (macOS) 或 `ipconfig` (Windows) 命令来查找您电脑的 IP 地址。

### **第五步 (可选): 自动化**

如果您希望 Agent 每天早上 9 点自动运行，可以设置一个 `cron` 定时任务。

```bash
# 打开 crontab 编辑器
crontab -e

# 在文件末尾添加以下一行，并保存
# 注意：请将路径修改为您项目的实际绝对路径
0 9 * * * /usr/bin/python3 /Users/mlamp/Workspace/Scratch/podcast_agent/main.py
```
这会让系统每天定时为您执行脚本，生成最新的播客。