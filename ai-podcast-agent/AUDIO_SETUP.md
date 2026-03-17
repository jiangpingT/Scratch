# 完整音频播客生成配置指南

## 问题说明

当前环境：
- ✅ Python 3.13.2
- ✅ Claude API 已配置
- ❌ OpenAI API 未配置
- ❌ Python 3.13 与 pydub 不兼容

## 解决方案

### 方案 A：使用 Python 3.12 环境（推荐）

#### 步骤 1：安装 Python 3.12

```bash
# 使用 Homebrew 安装 pyenv（如果还没安装）
brew install pyenv

# 安装 Python 3.12
pyenv install 3.12.7

# 在项目目录设置使用 Python 3.12
cd /Users/mlamp/Workspace/Scratch/ai-podcast-agent
pyenv local 3.12.7
```

#### 步骤 2：重新创建虚拟环境

```bash
# 删除旧环境
rm -rf venv

# 创建新环境（使用 Python 3.12）
python3.12 -m venv venv

# 激活环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install python-dotenv
```

#### 步骤 3：配置 OpenAI API

编辑 `.env` 文件：
```bash
# 添加 OpenAI API 密钥
OPENAI_API_KEY=sk-your_openai_api_key_here
```

#### 步骤 4：安装 ffmpeg

```bash
# Mac 用户
brew install ffmpeg

# 验证安装
ffmpeg -version
```

#### 步骤 5：测试完整流程

```bash
source venv/bin/activate
python main.py --mode once --test
```

---

### 方案 B：使用 Docker（生产环境推荐）

#### 步骤 1：创建 Dockerfile

已为您创建 `Dockerfile`（见下方）

#### 步骤 2：创建 docker-compose.yml

已为您创建 `docker-compose.yml`（见下方）

#### 步骤 3：配置环境变量

在 `.env` 中添加：
```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
```

#### 步骤 4：构建和运行

```bash
# 构建镜像
docker-compose build

# 运行一次
docker-compose run podcast python main.py --mode once

# 定时运行（后台）
docker-compose up -d
```

---

## OpenAI API 密钥获取

1. 访问：https://platform.openai.com/api-keys
2. 登录账号
3. 点击 "Create new secret key"
4. 复制密钥到 `.env` 文件

**预估成本**：
- TTS API: ~$0.015 / 1000 字符
- 每期播客约 1000 字 = ~$0.15
- 每月（30 期）= ~$4.5

---

## 替代方案：使用免费 TTS

如果暂时不想用 OpenAI TTS，可以使用：

### Edge TTS（免费，微软）

修改 `services/tts_service.py`：
```python
# 安装
pip install edge-tts

# 使用示例
import edge_tts
async def generate_audio(text, output_file):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(output_file)
```

### pyttsx3（本地，离线）

```bash
pip install pyttsx3

# 优点：完全免费，离线
# 缺点：音质一般
```

---

## 快速启动命令

### 如果您有 OpenAI API Key：

```bash
# 1. 添加到 .env
echo "OPENAI_API_KEY=your_key_here" >> .env

# 2. 选择一个方案：

# 方案 A（Python 3.12）
pyenv install 3.12.7
cd /Users/mlamp/Workspace/Scratch/ai-podcast-agent
pyenv local 3.12.7
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt python-dotenv
brew install ffmpeg
python main.py --mode once --test

# 方案 B（Docker）
docker-compose up --build
```

---

## 故障排查

### 问题 1：ffmpeg 未找到
```bash
brew install ffmpeg
```

### 问题 2：OpenAI API 额度不足
- 检查账户余额：https://platform.openai.com/usage
- 充值或使用免费 TTS 方案

### 问题 3：音频文件过大
- 调整 `config.py` 中的 `output_bitrate` 为 "128k"
- 减少 `max_papers` 数量

---

## 下一步

请选择：
1. 提供 OpenAI API 密钥 → 我帮您配置
2. 使用免费 TTS → 我修改代码
3. 直接使用 Docker → 我创建配置文件
