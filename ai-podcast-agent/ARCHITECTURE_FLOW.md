# 🏗️ AI 播客 Agent 完整架构与执行流程

## 📊 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          用户入口                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  python test_edge_tts.py  或  python main.py --mode once                │
│                                                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        主流程协调器 (main.py)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. 初始化所有组件                                                         │
│  2. 协调各个 Agent 执行                                                   │
│  3. 处理异常和日志                                                         │
│                                                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ PaperFetcher │    │ ContextManager   │    │ ContentAnalyzer   │
│              │    │                  │    │                   │
│ 获取论文      │───▶│ 去重&历史管理     │───▶│ Claude 分析论文   │
│ (arXiv API)  │    │ (SQLite)         │    │ (Anthropic API)   │
└──────────────┘    └──────────────────┘    └──────────┬────────┘
                                                        │
                                                        ▼
                            ┌────────────────────────────────────────┐
                            │      PodcastGenerator                  │
                            │                                        │
                            │  🔥 核心：Agent Loop + Tool Use        │
                            │  🔥 Gather-Take-Verify 循环            │
                            │                                        │
                            └──────────────┬─────────────────────────┘
                                          │
                                          ▼
                            ┌────────────────────────┐
                            │     TTSService         │
                            │                        │
                            │  Edge TTS 语音合成      │
                            │  (免费，微软 Azure)     │
                            └──────────┬─────────────┘
                                      │
                                      ▼
                            ┌────────────────────────┐
                            │   AudioProcessor       │
                            │                        │
                            │  ffmpeg 音频合并        │
                            └──────────┬─────────────┘
                                      │
                                      ▼
                            ┌────────────────────────┐
                            │     最终输出            │
                            │                        │
                            │  podcast_complete.mp3  │
                            └────────────────────────┘
```

---

## 🔥 核心：Agent Loop 详细执行流程

### 1. 初始化阶段

```
┌─────────────────────────────────────────────────────────────┐
│ PodcastGenerator.__init__()                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 创建 Anthropic Client                                   │
│     ├─ API Key: config.ANTHROPIC_API_KEY                   │
│     └─ Base URL: config.ANTHROPIC_BASE_URL (自定义端点)    │
│                                                             │
│  2. 初始化 Context Manager (SQLite)                         │
│     ├─ 读取历史论文数据                                      │
│     ├─ 读取热门话题                                          │
│     └─ 提供上下文查询接口                                     │
│                                                             │
│  3. 定义 7 个工具 (Tools)                                   │
│     ├─ GATHER: get_paper_details                           │
│     ├─ GATHER: get_historical_context                      │
│     ├─ GATHER: get_current_progress                        │
│     ├─ TAKE: add_podcast_segment                           │
│     ├─ VERIFY: verify_segment                              │
│     ├─ VERIFY: revise_segment                              │
│     └─ COMPLETE: finish_podcast                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. Agent Loop 启动

```python
# 入口函数
generator.generate_podcast_script(analyzed_papers)
  │
  └─▶ _run_agent_loop(context, historical_context, analyzed_papers)
```

```
┌─────────────────────────────────────────────────────────────┐
│ _run_agent_loop() 开始                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 定义工具列表                                             │
│     tools = _define_tools()  # 7 个工具                     │
│                                                             │
│  2. 构建 System Prompt                                      │
│     system_prompt = _build_agent_system_prompt()           │
│     ├─ 包含 GTV 循环指令                                     │
│     ├─ 包含历史上下文                                        │
│     ├─ 包含主持人设定                                        │
│     └─ 包含质量标准                                          │
│                                                             │
│  3. 初始化对话历史                                           │
│     messages = [{                                           │
│       "role": "user",                                       │
│       "content": "请开始生成今天的播客..."                    │
│     }]                                                      │
│                                                             │
│  4. 初始化工具状态                                           │
│     tool_state = {                                          │
│       'papers': analyzed_papers,                            │
│       'context': context                                    │
│     }                                                       │
│                                                             │
│  5. 初始化片段列表                                           │
│     script_segments = []                                    │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              【进入 Agent Loop】
```

---

### 3. Agent Loop 核心循环（Tool Use 机制）

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Agent Loop (While 循环)                        │
│                     iteration = 1, 2, 3, ... max_iterations           │
└───────────────────────────────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤 1: 调用 Claude API (带 Tool Use)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  response = anthropic_client.messages.create(                        │
│      model="claude-sonnet-4-5",                                      │
│      max_tokens=4096,                                                │
│      temperature=0.8,                                                │
│      system=system_prompt,        ← 包含 GTV 指令                     │
│      tools=tools,                 ← 7 个工具定义                      │
│      messages=messages            ← 完整对话历史                      │
│  )                                                                    │
│                                                                       │
│  【这里发生了什么？】                                                   │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │ Claude 收到请求后：                                          │       │
│  │ 1. 读取 system prompt（理解 GTV 循环要求）                  │       │
│  │ 2. 读取对话历史（了解当前状态）                              │       │
│  │ 3. 查看可用工具列表（7 个工具）                              │       │
│  │ 4. 思考：我应该调用什么工具？                                │       │
│  │ 5. 决定：先 GATHER 信息，调用 get_paper_details             │       │
│  │ 6. 构造工具调用请求                                         │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤 2: 检查 Claude 的响应                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  stop_reason = response.stop_reason                                  │
│                                                                       │
│  可能的值：                                                            │
│  ├─ "tool_use"     ← Claude 调用了工具                               │
│  ├─ "end_turn"     ← Claude 完成了当前轮次                           │
│  ├─ "max_tokens"   ← 达到 token 限制                                │
│  └─ "stop_sequence" ← 遇到停止序列                                   │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
                        【分支判断】
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
  stop_reason ==          stop_reason ==          stop_reason ==
   "tool_use"              "end_turn"              "max_tokens"
        │                       │                       │
        ▼                       ▼                       ▼
  【执行工具】              【检查完成】              【继续对话】
```

---

### 4. Tool Use 详细流程（核心机制）

```
┌───────────────────────────────────────────────────────────────────────┐
│ 当 stop_reason == "tool_use" 时                                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ for content_block in response.content:                                 │
│     if content_block.type == "tool_use":                               │
│                                                                         │
│         【Claude 返回的工具调用信息】                                    │
│         ┌─────────────────────────────────────────────────┐            │
│         │ content_block = {                                │            │
│         │   "type": "tool_use",                            │            │
│         │   "id": "toolu_01ABC123...",                     │            │
│         │   "name": "get_paper_details",    ← 工具名       │            │
│         │   "input": {                      ← 工具参数     │            │
│         │     "paper_index": 0                             │            │
│         │   }                                              │            │
│         │ }                                                │            │
│         └─────────────────────────────────────────────────┘            │
│                                                                         │
│         tool_name = content_block.name      # "get_paper_details"      │
│         tool_input = content_block.input    # {"paper_index": 0}       │
│         tool_id = content_block.id          # "toolu_01ABC123..."      │
│                                                                         │
│         ┌─────────────────────────────────────────────────┐            │
│         │ 执行工具                                          │            │
│         │ result = _execute_tool(                          │            │
│         │     tool_name="get_paper_details",               │            │
│         │     tool_input={"paper_index": 0},               │            │
│         │     tool_state=tool_state,                       │            │
│         │     script_segments=script_segments              │            │
│         │ )                                                │            │
│         └─────────────────────────────────────────────────┘            │
│                                                                         │
│         【工具返回结果】                                                 │
│         ┌─────────────────────────────────────────────────┐            │
│         │ result = {                                       │            │
│         │   "success": true,                               │            │
│         │   "paper": {                                     │            │
│         │     "title": "Reward Models are...",            │            │
│         │     "summary": "揭示奖励模型...",                │            │
│         │     "innovation": "首次系统性...",               │            │
│         │     "impact": "...",                             │            │
│         │     "key_points": [...]                          │            │
│         │   }                                              │            │
│         │ }                                                │            │
│         └─────────────────────────────────────────────────┘            │
│                                                                         │
│         【构造工具结果消息】                                             │
│         tool_result = {                                                │
│             "type": "tool_result",                                     │
│             "tool_use_id": tool_id,          ← 关联到工具调用 ID       │
│             "content": json.dumps(result)    ← 工具执行结果            │
│         }                                                              │
│                                                                         │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 步骤 3: 将工具调用和结果添加到对话历史                                   │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  # Claude 的响应（包含工具调用）                                         │
│  messages.append({                                                     │
│      "role": "assistant",                                              │
│      "content": response.content  # 包含 tool_use block                │
│  })                                                                    │
│                                                                         │
│  # 工具执行结果                                                          │
│  messages.append({                                                     │
│      "role": "user",                                                   │
│      "content": [tool_result]  # tool_result block                    │
│  })                                                                    │
│                                                                         │
│  【对话历史示例】                                                         │
│  messages = [                                                          │
│    {"role": "user", "content": "请开始生成播客..."},                     │
│    {"role": "assistant", "content": [                                  │
│      {"type": "text", "text": "我先了解论文信息"},                      │
│      {"type": "tool_use", "name": "get_paper_details", ...}            │
│    ]},                                                                 │
│    {"role": "user", "content": [                                       │
│      {"type": "tool_result", "content": "{\"paper\": {...}}"}          │
│    ]},                                                                 │
│    ...                                                                 │
│  ]                                                                     │
│                                                                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
                      【回到 Loop 开始，继续下一次迭代】
```

---

### 5. Gather-Take-Verify 循环实例

```
┌───────────────────────────────────────────────────────────────────────┐
│                   完整的 GTV 循环示例                                   │
└───────────────────────────────────────────────────────────────────────┘

迭代 1: GATHER - 获取论文详情
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: get_paper_details(paper_index=0)         │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "paper": {                                            │
│     "title": "Reward Models are Metrics...",           │
│     "summary": "...",                                   │
│     "innovation": "...",                                │
│     "impact": "..."                                     │
│   }                                                     │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

迭代 2: GATHER - 检查历史话题
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: get_historical_context("topics")         │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "data": [                                             │
│     {"topic": "reward model", "count": 3},              │
│     {"topic": "evaluation", "count": 2}                 │
│   ]                                                     │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

迭代 3: TAKE ACTION - 生成开场白
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: add_podcast_segment(                     │
│   speaker="host_a",                                     │
│   text="大家好，欢迎收听AI前沿日报！我是主持人张三...",   │
│   segment_type="opening"                                │
│ )                                                       │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "message": "片段已添加（共 1 个片段）",                │
│   "segment_index": 0                                    │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

迭代 4: VERIFY - 验证开场白
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: verify_segment(                          │
│   segment_index=-1,                                     │
│   check_aspects=["length", "coherence"]                 │
│ )                                                       │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "passed": false,                   ← 验证失败！        │
│   "issues": [                                           │
│     "开场白长度不符（106字，应为30-80字）"               │
│   ],                                                    │
│   "message": "验证失败：开场白长度不符..."               │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

迭代 5: VERIFY - 修订开场白
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: revise_segment(                          │
│   segment_index=0,                                      │
│   new_text="大家好，欢迎来到AI前沿日报！我是张三...",    │
│   revision_reason="开场白长度超标，需缩短至30-80字"      │
│ )                                                       │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "message": "片段已修订",                               │
│   "old_text": "大家好，欢迎收听AI前沿日报！...",         │
│   "new_text": "大家好，欢迎来到AI前沿日报！..."          │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

迭代 6: VERIFY - 再次验证
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: verify_segment(segment_index=0)          │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "passed": true,                    ← 验证通过！        │
│   "issues": [],                                         │
│   "message": "验证通过"                                  │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

... 继续生成下一个片段 ...

迭代 N: COMPLETE - 完成播客
┌─────────────────────────────────────────────────────────┐
│ Claude → Tool: finish_podcast(                          │
│   summary="成功生成6个片段，4个经过修订",                 │
│   quality_score=9                                       │
│ )                                                       │
│ Tool → Claude: {                                        │
│   "success": true,                                      │
│   "message": "播客已完成",                               │
│   "total_segments": 6,                                  │
│   "quality_score": 9                                    │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

Loop 结束，返回 script_segments
```

---

## 🔧 工具执行机制详解

### _execute_tool() 函数流程

```python
def _execute_tool(tool_name, tool_input, tool_state, script_segments):
    """
    根据工具名称分发到对应的处理逻辑
    """

    if tool_name == "get_paper_details":
        # GATHER: 从 tool_state 获取论文数据
        paper = tool_state['papers'][tool_input['paper_index']]
        return {
            "success": True,
            "paper": {
                "title": paper['title'],
                "summary": paper['summary'],
                ...
            }
        }

    elif tool_name == "get_historical_context":
        # GATHER: 从 Context Manager 查询历史数据
        context_type = tool_input['context_type']
        if context_type == "topics":
            topics = self.context_manager.get_top_topics(limit=10)
            return {
                "success": True,
                "data": [{"topic": t[0], "count": t[1]} for t in topics]
            }

    elif tool_name == "add_podcast_segment":
        # TAKE ACTION: 添加片段到列表
        segment = {
            'speaker': tool_input['speaker'],
            'text': tool_input['text'],
            'segment_type': tool_input['segment_type']
        }
        script_segments.append(segment)
        return {
            "success": True,
            "message": f"片段已添加（共 {len(script_segments)} 个片段）"
        }

    elif tool_name == "verify_segment":
        # VERIFY: 验证片段质量
        segment = script_segments[tool_input['segment_index']]
        text_len = len(segment['text'])

        # 检查长度
        if segment['segment_type'] == "opening":
            if text_len < 30 or text_len > 80:
                return {
                    "passed": False,
                    "issues": [f"开场白长度不符（{text_len}字，应为30-80字）"]
                }

        return {"passed": True, "issues": []}

    elif tool_name == "revise_segment":
        # VERIFY: 修订片段
        segment_index = tool_input['segment_index']
        script_segments[segment_index]['text'] = tool_input['new_text']
        script_segments[segment_index]['revised'] = True
        script_segments[segment_index]['revision_reason'] = tool_input['revision_reason']
        return {"success": True, "message": "片段已修订"}
```

---

## 📡 API 调用时序图

```
┌─────────┐      ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  主程序  │      │   Claude    │      │  Tool        │      │  Context    │
│         │      │   API       │      │  Executor    │      │  Manager    │
└────┬────┘      └──────┬──────┘      └──────┬───────┘      └──────┬──────┘
     │                  │                     │                     │
     │  1. 调用 API     │                     │                     │
     │  (带 tools)      │                     │                     │
     ├─────────────────>│                     │                     │
     │                  │                     │                     │
     │                  │  2. Claude 思考     │                     │
     │                  │  决定调用工具        │                     │
     │                  │                     │                     │
     │  3. 返回响应     │                     │                     │
     │  (tool_use)      │                     │                     │
     │<─────────────────┤                     │                     │
     │                  │                     │                     │
     │  4. 解析工具调用  │                     │                     │
     │                  │                     │                     │
     │  5. 执行工具                            │                     │
     ├────────────────────────────────────────>│                     │
     │                  │                     │                     │
     │                  │                     │  6. 查询历史数据      │
     │                  │                     ├────────────────────>│
     │                  │                     │                     │
     │                  │                     │  7. 返回数据         │
     │                  │                     │<────────────────────┤
     │                  │                     │                     │
     │  8. 工具结果      │                     │                     │
     │<────────────────────────────────────────┤                     │
     │                  │                     │                     │
     │  9. 将结果作为    │                     │                     │
     │  user message    │                     │                     │
     │  发送回 Claude    │                     │                     │
     ├─────────────────>│                     │                     │
     │                  │                     │                     │
     │                  │  10. Claude 继续思考 │                     │
     │                  │  基于工具结果        │                     │
     │                  │                     │                     │
     │  11. 返回响应    │                     │                     │
     │  (可能是更多      │                     │                     │
     │   tool_use)      │                     │                     │
     │<─────────────────┤                     │                     │
     │                  │                     │                     │
     │  ... 循环继续 ...│                     │                     │
     │                  │                     │                     │
```

---

## 🔄 消息历史演进示例

```javascript
// 初始状态
messages = [
  {
    role: "user",
    content: "请开始生成今天的播客。今天要讨论的论文：..."
  }
]

// 第 1 次 API 调用后
messages = [
  {
    role: "user",
    content: "请开始生成今天的播客..."
  },
  {
    role: "assistant",
    content: [
      {
        type: "text",
        text: "我先了解一下论文的详细信息"
      },
      {
        type: "tool_use",
        id: "toolu_01ABC",
        name: "get_paper_details",
        input: { paper_index: 0 }
      }
    ]
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "toolu_01ABC",
        content: '{"success": true, "paper": {"title": "...", ...}}'
      }
    ]
  }
]

// 第 2 次 API 调用后
messages = [
  ... 之前的内容 ...,
  {
    role: "assistant",
    content: [
      {
        type: "text",
        text: "了解了论文内容，现在检查历史话题避免重复"
      },
      {
        type: "tool_use",
        id: "toolu_02DEF",
        name: "get_historical_context",
        input: { context_type: "topics" }
      }
    ]
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "toolu_02DEF",
        content: '{"success": true, "data": [...]}'
      }
    ]
  }
]

// 第 3 次 API 调用后 - 生成内容
messages = [
  ... 之前的内容 ...,
  {
    role: "assistant",
    content: [
      {
        type: "text",
        text: "好的，我现在生成开场白"
      },
      {
        type: "tool_use",
        id: "toolu_03GHI",
        name: "add_podcast_segment",
        input: {
          speaker: "host_a",
          text: "大家好，欢迎收听AI前沿日报！...",
          segment_type: "opening"
        }
      }
    ]
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "toolu_03GHI",
        content: '{"success": true, "message": "片段已添加"}'
      }
    ]
  }
]

// 第 4 次 API 调用后 - 验证
messages = [
  ... 之前的内容 ...,
  {
    role: "assistant",
    content: [
      {
        type: "text",
        text: "让我验证一下这个开场白的质量"
      },
      {
        type: "tool_use",
        id: "toolu_04JKL",
        name: "verify_segment",
        input: { segment_index: -1 }
      }
    ]
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "toolu_04JKL",
        content: '{"passed": false, "issues": ["开场白长度不符..."]}'
      }
    ]
  }
]

// 第 5 次 API 调用后 - 修订
messages = [
  ... 之前的内容 ...,
  {
    role: "assistant",
    content: [
      {
        type: "text",
        text: "验证失败，我需要修订开场白，缩短长度"
      },
      {
        type: "tool_use",
        id: "toolu_05MNO",
        name: "revise_segment",
        input: {
          segment_index: 0,
          new_text: "大家好，欢迎来到AI前沿日报！我是张三...",
          revision_reason: "开场白长度超标"
        }
      }
    ]
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "toolu_05MNO",
        content: '{"success": true, "message": "片段已修订"}'
      }
    ]
  }
]

// ... 继续循环直到完成 ...
```

---

## 🎯 关键概念说明

### 1. **Tool Use vs MCP**

#### Tool Use（本项目使用）
```
┌─────────────────────────────────────────────────────────┐
│                    Tool Use 机制                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  特点：                                                   │
│  ✅ 在单个 API 调用中定义工具                             │
│  ✅ Claude 在一次对话中可以多次调用工具                   │
│  ✅ 工具在本地 Python 代码中实现                          │
│  ✅ 适合自包含的应用                                      │
│                                                          │
│  流程：                                                   │
│  1. 定义工具（JSON Schema）                              │
│  2. 发送给 Claude（作为 API 参数）                       │
│  3. Claude 返回工具调用                                  │
│  4. 本地执行工具                                         │
│  5. 将结果返回给 Claude                                  │
│  6. Claude 继续对话                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### MCP（Model Context Protocol）
```
┌─────────────────────────────────────────────────────────┐
│                    MCP 协议                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  特点：                                                   │
│  ✅ 标准化的协议                                          │
│  ✅ 工具可以是远程服务                                    │
│  ✅ 支持多种数据源（文件系统、数据库、API等）              │
│  ✅ 可跨应用共享工具                                      │
│                                                          │
│  使用场景：                                               │
│  - 需要访问外部系统（如 GitHub、Slack）                  │
│  - 需要共享工具定义                                       │
│  - 企业级应用                                            │
│                                                          │
│  本项目未使用 MCP 的原因：                                │
│  - 所有工具都在本地实现                                   │
│  - 不需要外部服务集成                                     │
│  - Tool Use 已满足需求                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2. **Context Manager 的作用**

```
┌─────────────────────────────────────────────────────────┐
│              Context Manager (SQLite)                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  数据表：                                                 │
│  ┌─────────────────────────────────────┐                │
│  │ papers 表                            │                │
│  ├─────────────────────────────────────┤                │
│  │ - paper_id (唯一)                    │                │
│  │ - title                              │                │
│  │ - abstract                           │                │
│  │ - published_date                     │                │
│  │ - used_in_podcast (0/1)              │                │
│  │ - podcast_date                       │                │
│  └─────────────────────────────────────┘                │
│                                                          │
│  ┌─────────────────────────────────────┐                │
│  │ topics 表                            │                │
│  ├─────────────────────────────────────┤                │
│  │ - topic (唯一)                       │                │
│  │ - count                              │                │
│  │ - first_seen                         │                │
│  │ - last_seen                          │                │
│  └─────────────────────────────────────┘                │
│                                                          │
│  功能：                                                   │
│  1. 去重：避免重复讨论同一篇论文                          │
│  2. 历史记录：追踪所有讨论过的论文                        │
│  3. 话题追踪：识别热门话题                                │
│  4. 上下文注入：将历史数据注入到 Agent                    │
│                                                          │
│  Agent 如何使用：                                         │
│  - get_historical_context("topics") → 获取热门话题       │
│  - get_historical_context("statistics") → 获取统计       │
│  - 在 system prompt 中包含历史信息                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 数据流图

```
┌────────────┐
│ arXiv API  │
└─────┬──────┘
      │
      │ 论文 XML
      ▼
┌──────────────┐
│ PaperFetcher │ ──┐
└──────┬───────┘   │
       │           │
       │ Paper对象  │
       ▼           │
┌─────────────────┐│
│ ContextManager  ││ 去重
│   (SQLite)      │<┘
└────────┬────────┘
         │
         │ 未使用的论文
         ▼
┌──────────────────┐
│ ContentAnalyzer  │
│  (Claude API)    │
└────────┬─────────┘
         │
         │ 分析结果 (JSON)
         │ {
         │   title, summary,
         │   innovation, impact,
         │   key_points
         │ }
         ▼
┌────────────────────────────┐
│   PodcastGenerator         │
│   (Agent Loop + Tool Use)  │
│                            │
│   ┌──────────────────┐     │
│   │ Gather Context   │     │
│   │ ↓                │     │
│   │ Take Action      │ ───┼──→ script_segments[]
│   │ ↓                │     │     [
│   │ Verify Work      │     │       {speaker, text, type},
│   │ ↓                │     │       {speaker, text, type},
│   │ Revise (if fail) │     │       ...
│   └──────────────────┘     │     ]
└────────────────┬───────────┘
                 │
                 │ 播客脚本 (JSON)
                 ▼
┌─────────────────────────┐
│    TTSService           │
│    (Edge TTS)           │
│                         │
│  for each segment:      │
│    tts.save(mp3)        │
└────────┬────────────────┘
         │
         │ 音频片段
         │ segment_000.mp3
         │ segment_001.mp3
         │ ...
         ▼
┌─────────────────────────┐
│   AudioProcessor        │
│   (ffmpeg concat)       │
└────────┬────────────────┘
         │
         │ 最终音频
         ▼
┌─────────────────────────┐
│ podcast_complete.mp3    │
└─────────────────────────┘
```

---

## 🔍 关键代码片段

### 定义工具

```python
# agents/podcast_generator.py: _define_tools()

tools = [
    {
        "name": "get_paper_details",
        "description": "【Gather Context】获取论文详细信息...",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_index": {
                    "type": "integer",
                    "description": "论文索引（从0开始）"
                }
            },
            "required": ["paper_index"]
        }
    },
    # ... 其他 6 个工具
]
```

### 调用 Claude API

```python
# agents/podcast_generator.py: _run_agent_loop()

response = self.anthropic_client.messages.create(
    model=self.model,                    # "claude-sonnet-4-5"
    max_tokens=self.max_tokens,          # 4096
    temperature=0.8,
    system=system_prompt,                # 包含 GTV 指令
    tools=tools,                         # 7 个工具
    messages=messages                    # 完整对话历史
)
```

### 处理工具调用

```python
# agents/podcast_generator.py: _run_agent_loop()

if stop_reason == "tool_use":
    tool_results = []

    for content_block in response.content:
        if content_block.type == "tool_use":
            # 执行工具
            result = self._execute_tool(
                content_block.name,
                content_block.input,
                tool_state,
                script_segments
            )

            # 构造工具结果
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": content_block.id,
                "content": json.dumps(result)
            })

    # 添加到对话历史
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
```

---

## 🎓 总结

### 核心机制

1. **Tool Use（Function Calling）**
   - Claude 在对话中主动调用工具
   - 工具调用和结果都是对话的一部分
   - 形成连续的上下文

2. **Agent Loop**
   - While 循环持续调用 API
   - 每次调用都基于完整历史
   - Agent 自主决定何时结束

3. **Gather-Take-Verify**
   - 收集上下文 → 执行操作 → 验证质量
   - 自我修正的闭环
   - 保证输出质量

4. **Context Management**
   - SQLite 存储历史数据
   - 通过工具查询注入到对话
   - 影响 Agent 决策

### 数据流

```
用户请求
  → 初始化组件
  → 获取论文（arXiv）
  → 去重（SQLite）
  → 分析论文（Claude API）
  → 生成播客（Agent Loop + Tool Use）
    ├─ GATHER: 查询论文和历史
    ├─ TAKE: 生成片段
    └─ VERIFY: 验证和修订
  → 语音合成（Edge TTS）
  → 音频合并（ffmpeg）
  → 最终输出
```

---

**完整实现了 Anthropic Agent SDK 的 Tool Use 和 Gather-Take-Verify 循环！** 🎉
