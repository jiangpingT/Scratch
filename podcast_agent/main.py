import os
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import anthropic
import datetime
import time
from serpapi import GoogleSearch

# --- 1. 配置 (Configuration) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN", "sk-Y1QjrOKehRvj8IolqV-hWw")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://vibe.deepminer.ai")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "3df9cec93f45bf621c656ebfafa98d71edf1d8d9a8acf4196cac88b7dcb55610")

# --- 2. 上下文管理 (Context Management) ---
class AgentContext:
    """一个用于在 Agent 之间传递状态的独立上下文对象。"""
    def __init__(self):
        self.status_log = []
        self.article_url = None
        self.article_content = None
        self.summary = None
        self.script = None
        self.audio_filename = None

    def log(self, message):
        print(message)
        self.status_log.append(f"[{datetime.datetime.now()}] {message}")

# --- 3. 子 Agent 定义 (Sub-Agents) ---
class ScoutAgent:
    """子 Agent 1: 负责搜集上下文 (Gather Context)，已升级，具备自动重试能力。"""
    def run(self, context: AgentContext) -> AgentContext:
        context.log("-> ScoutAgent: 开始执行...")
        try:
            params = {
                "engine": "google",
                "q": "latest deep dive on large language models OR AI agents OR robotics",
                "tbm": "nws", "num": "5", "api_key": SERPAPI_API_KEY
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            news_results = results.get("news_results", [])
            
            if not news_results:
                context.log("ScoutAgent: 未能找到任何相关新闻文章。")
                return context

            context.log(f"ScoutAgent: 找到 {len(news_results)} 篇文章，将逐一尝试获取...")

            for result in news_results:
                url = result["link"]
                context.log(f"ScoutAgent: 正在尝试 URL: {url}")
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    response = requests.get(url, headers=headers, timeout=20)
                    response.raise_for_status()  # 对4xx/5xx错误抛出异常

                    soup = BeautifulSoup(response.text, 'html.parser')
                    # 简单检查内容是否过短
                    if len(soup.get_text()) < 500:
                        context.log(f"ScoutAgent: 内容过短，可能不是有效文章。跳过。")
                        continue
                    
                    context.article_url = url
                    context.article_content = soup.get_text()
                    context.log(f"ScoutAgent: 成功提取文章内容。")
                    return context  # 成功，立即返回
                except requests.RequestException as e:
                    context.log(f"ScoutAgent: 无法获取 URL {url}。原因: {e}。正在尝试下一篇...")
                    continue  # 尝试下一个URL
            
            context.log("ScoutAgent: 尝试了所有找到的 URL，均未能成功获取内容。")
            return context

        except Exception as e:
            context.log(f"ScoutAgent: 执行失败 (SerpApi可能出错)。原因: {e}")
        return context

class CreativeAgent:
    """子 Agent 2: 负责核心创作 (Take Action)。"""
    def __init__(self, client):
        self.claude_client = client

    def run(self, context: AgentContext) -> AgentContext:
        context.log("-> CreativeAgent: 开始执行...")
        if not context.article_content:
            context.log("CreativeAgent: 上下文中无文章内容，跳过执行。")
            return context
        try:
            context.log("CreativeAgent: 正在生成摘要...")
            summary_message = self.claude_client.messages.create(
                model="claude-sonnet-4-5", max_tokens=1024,
                messages=[{"role": "user", "content": f"分析以下文章，提炼核心观点、创新和影响，总结在300字内。\n\n文章：\n{context.article_content}"}]
            )
            context.summary = summary_message.content[0].text

            context.log("CreativeAgent: 正在生成播客脚本...")
            script_message = self.claude_client.messages.create(
                model="claude-sonnet-4-5", max_tokens=2048,
                messages=[{"role": "user", "content": f"根据以下核心观点，创作一段双人播客对话（主持人Alex和Ben）。\n\n核心观点：\n{context.summary}"}]
            )
            context.script = script_message.content[0].text
            context.log("CreativeAgent: 成功生成摘要和脚本。")
        except Exception as e:
            context.log(f"CreativeAgent: 执行失败。原因: {e}")
        return context

class OutputAgent:
    """子 Agent 3: 负责生成最终产物并验证 (Take Action & Verify Work)。"""
    def run(self, context: AgentContext) -> AgentContext:
        context.log("-> OutputAgent: 开始执行...")
        if not context.script:
            context.log("OutputAgent: 上下文中无脚本内容，跳过执行。")
            return context
        try:
            # Take Action: Generate Audio
            context.log("OutputAgent: 正在生成音频文件...")
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            audio_filename = f"podcast_{today_str}.mp3"
            tts = gTTS(text=context.script, lang='zh-cn')
            tts.save(audio_filename)
            context.audio_filename = audio_filename
            context.log(f"OutputAgent: 成功生成音频文件: {audio_filename}")

            # Verify Work
            context.log("OutputAgent: 正在验证工作成果...")
            if os.path.exists(audio_filename) and os.path.getsize(audio_filename) > 0:
                context.log(f"OutputAgent: 验证成功，文件 '{audio_filename}' 已正确创建。")
            else:
                context.log(f"OutputAgent: 验证失败，文件 '{audio_filename}' 未能正确生成。")
        except Exception as e:
            context.log(f"OutputAgent: 执行失败。原因: {e}")
        return context

# --- 4. 主 Agent 定义 (Master Agent) ---
class MasterAgent:
    """主 Agent: 负责编排子 Agent 并执行主循环。"""
    def __init__(self):
        self._initialize_clients()
        self.scout_agent = ScoutAgent()
        self.creative_agent = CreativeAgent(self.claude_client)
        self.output_agent = OutputAgent()
        print("Master Agent 及所有子 Agent 初始化完毕。" )

    def _initialize_clients(self):
        if "YOUR_ANTHROPIC" in ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY:
            raise ValueError("错误: 请设置您的 ANTHROPIC_AUTH_TOKEN。" )
        self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)

    def run_single_cycle(self):
        """按顺序调用子 Agent，执行一次完整的任务循环。"""
        context = AgentContext()
        context.log(f"\n[{datetime.datetime.now()}] Master Agent: 开始新一轮任务循环...")
        
        # 1. Gather Context
        context = self.scout_agent.run(context)
        
        # 2. Take Action
        context = self.creative_agent.run(context)
        context = self.output_agent.run(context)
        
        context.log("Master Agent: 本轮任务循环结束。" )

    def start_loop(self, run_at_hour=9, run_at_minute=0):
        """启动 Agent 的主循环，按计划每日执行。"""
        print(f"Agent 主循环已启动。将在每天 {run_at_hour:02d}:{run_at_minute:02d} 执行任务。" )
        last_run_date = None
        while True:
            now = datetime.datetime.now()
            if now.hour == run_at_hour and now.minute == run_at_minute and now.date() != last_run_date:
                self.run_single_cycle()
                last_run_date = now.date()
            time.sleep(60)

if __name__ == "__main__":
    try:
        master_agent = MasterAgent()
        # 立即执行一次进行测试
        master_agent.run_single_cycle()
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        print("\nAgent 已手动停止。" )
