import torch

try:
    # 加载模型文件
    model = torch.load('saved_models/model_best.pth', map_location=torch.device('cpu'))
    
    # 打开文件准备写入信息
    with open('docs/train.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transformer 训练记录</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        }
        .info-block { 
            margin: 20px 0; 
            padding: 15px; 
            border-left: 4px solid #4CAF50; 
            background: #f9f9f9; 
        }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="index.html">首页</a>
            <a href="#model-info">模型信息</a>
            <a href="#preparation">数据准备</a>
            <a href="#training">训练流程</a>
            <a href="#optimization">优化技巧</a>
        </nav>

        <h1>Transformer 训练记录</h1>
        <section id="model-info">
''')
        
        # 写入模型基本信息
        f.write('            <h2>模型基本信息</h2>\n            <div class="info-block">\n')
        for key, value in model.items():
            f.write(f'                <h3>{key}</h3>\n                <pre>{str(value)}</pre>\n')
        f.write('            </div>\n        </section>\n')
        
        # 写入结束标签
        f.write('''
    </div>
</body>
</html>
''')
    
    print('成功将模型信息写入到 docs/train.html')
    
except Exception as e:
    print(f'错误: {e}')