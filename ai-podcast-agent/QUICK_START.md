# 🚀 快速开始 - 生成完整音频播客

根据您的环境，我为您准备了 3 个方案，**推荐方案 2（最简单）**。

---

## 📊 当前环境检查

- ✅ Python 3.13.2（已安装）
- ✅ ffmpeg（已安装）
- ✅ Claude API（已配置）
- ❌ Python 3.12（未安装）
- ❌ Docker（未安装）
- ❌ pyenv（未安装）

---

## 🎯 推荐方案（按难度排序）

### ⭐ 方案 1：安装 Python 3.12（10 分钟）

**最彻底的解决方案，支持所有功能**

```bash
# 1. 使用 Homebrew 安装 Python 3.12
brew install python@3.12

# 2. 回到项目目录
cd /Users/mlamp/Workspace/Scratch/ai-podcast-agent

# 3. 删除旧虚拟环境
rm -rf venv

# 4. 使用 Python 3.12 创建新环境
/opt/homebrew/bin/python3.12 -m venv venv

# 5. 激活环境
source venv/bin/activate

# 6. 安装依赖
pip install -r requirements.txt python-dotenv

# 7. 配置 OpenAI API（如果有的话）
# 编辑 .env 文件，添加：
# OPENAI_API_KEY=sk-your_key_here

# 8. 运行测试
python main.py --mode once --test
```

**需要**：
- OpenAI API 密钥（约 $0.15/期）

---

### ⭐⭐⭐ 方案 2：使用免费 Edge TTS（5 分钟，推荐！）

**完全免费，音质好，不需要换 Python 版本**

我已经为您创建了 `services/edge_tts_service.py`（见下方），使用微软的免费 TTS。

```bash
cd /Users/mlamp/Workspace/Scratch/ai-podcast-agent

# 1. 激活当前环境
source venv/bin/activate

# 2. 安装 Edge TTS
pip install edge-tts

# 3. 测试免费 TTS
python -c "
import asyncio
import edge_tts

async def test():
    tts = edge_tts.Communicate('大家好，欢迎收听 AI 前沿日报', 'zh-CN-XiaoxiaoNeural')
    await tts.save('test_voice.mp3')
    print('✓ 音频生成成功：test_voice.mp3')

asyncio.run(test())
"

# 4. 使用免费版本运行（我已修改代码）
python test_edge_tts.py
```

**优点**：
- ✅ 完全免费
- ✅ 不需要 API 密钥
- ✅ 不需要换 Python 版本
- ✅ 音质不错（微软 Azure 同款）
- ✅ 支持多种中文声音

**缺点**：
- ⚠️ 需要网络连接（调用微软服务器）

---

### ⭐⭐ 方案 3：使用 Docker（15 分钟）

**适合生产环境，一次配置永久使用**

```bash
# 1. 安装 Docker Desktop for Mac
# 访问：https://www.docker.com/products/docker-desktop
# 或使用 Homebrew：
brew install --cask docker

# 2. 启动 Docker Desktop

# 3. 回到项目目录
cd /Users/mlamp/Workspace/Scratch/ai-podcast-agent

# 4. 配置 .env（如果要用 OpenAI TTS）
# 编辑 .env，添加：
# OPENAI_API_KEY=sk-your_key_here

# 5. 构建并运行
docker-compose build
docker-compose run podcast python test_basic.py

# 6. 定时运行（后台）
docker-compose up -d
```

---

## 🎤 声音对比

### OpenAI TTS（需付费）
- 声音：alloy, echo, fable, onyx, nova, shimmer
- 音质：⭐⭐⭐⭐⭐
- 成本：$0.015/1000字符

### Edge TTS（免费）
- 声音：
  - 女声：`zh-CN-XiaoxiaoNeural` (推荐)
  - 男声：`zh-CN-YunxiNeural`
  - 女声2：`zh-CN-XiaoyiNeural`
- 音质：⭐⭐⭐⭐
- 成本：免费

### 本地 TTS（离线，音质一般）
- pyttsx3
- 音质：⭐⭐
- 成本：免费

---

## 💡 我的建议

### 如果您想快速体验（推荐）：
👉 **选择方案 2（免费 Edge TTS）**
- 5 分钟即可完成
- 完全免费
- 音质很好

### 如果您有 OpenAI API 且想要最佳音质：
👉 **选择方案 1（Python 3.12）**
- 音质最好
- 支持所有功能

### 如果您要部署到服务器：
👉 **选择方案 3（Docker）**
- 环境隔离
- 便于管理

---

## 🎬 立即开始

**我推荐您现在就试试方案 2（免费 Edge TTS）：**

```bash
cd /Users/mlamp/Workspace/Scratch/ai-podcast-agent
source venv/bin/activate
pip install edge-tts
python test_edge_tts.py
```

这将在几分钟内生成您的第一期 AI 播客！🎉

---

## ❓ 常见问题

### Q: Edge TTS 需要联网吗？
A: 是的，它调用微软的 Azure TTS 服务（免费）。

### Q: Edge TTS 有使用限制吗？
A: 没有官方限制，但建议合理使用。

### Q: 我可以混用不同 TTS 吗？
A: 可以！在 `config.py` 中切换。

### Q: 哪个方案最省钱？
A: 方案 2（Edge TTS）完全免费。

### Q: 哪个方案音质最好？
A: 方案 1（OpenAI TTS），但需付费。

---

请告诉我您选择哪个方案，我立即帮您配置！🚀
