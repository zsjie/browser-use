"""
Simple demonstration of the CDP feature.

To test this locally, follow these steps:
1. Create a shortcut for the executable Chrome file.
2. Add the following argument to the shortcut:
   - On Windows: `--remote-debugging-port=9222`
3. Open a web browser and navigate to `http://localhost:9222/json/version` to verify that the Remote Debugging Protocol (CDP) is running.
4. Launch this example.

@dev You need to set the `QWEN_API_KEY` environment variable before proceeding.
"""

import os
import sys
from typing import cast

from dotenv import load_dotenv
from pydantic import SecretStr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

from chat_qwen import ChatQwen

load_dotenv()

# 获取并验证API密钥
api_key_raw = os.getenv('QWEN_API_KEY')
if not api_key_raw:
	raise ValueError('QWEN_API_KEY is not set')

# 创建SecretStr
api_key = SecretStr(api_key_raw)

browser = Browser(
	config=BrowserConfig(
		headless=False,
		cdp_url='http://localhost:9222',
	)
)
controller = Controller()


async def main():
	task = 'go to google.com'
	task += ' and search dashscope'
	model = ChatQwen(
		model='qwen2.5-32b-instruct',
		api_key=api_key,
	)
	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser=browser,
		max_failures=1
	)

	await agent.run()
	await browser.close()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())
