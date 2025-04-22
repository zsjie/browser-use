import asyncio
import os

from dotenv import load_dotenv
from pydantic import SecretStr

from browser_use import Agent
from chat_dashscope import ChatDashscope

# dotenv
load_dotenv()

api_key = os.getenv('DASHSCOPE_API_KEY', '')
if not api_key:
	raise ValueError('DASHSCOPE_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=('go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result'),
		llm=ChatDashscope(
			model='qwen-plus',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
		max_failures=2,
		max_actions_per_step=1,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())
