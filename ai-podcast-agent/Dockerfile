FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-dotenv

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p database logs output/podcasts assets

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令
CMD ["python", "main.py", "--mode", "schedule"]
