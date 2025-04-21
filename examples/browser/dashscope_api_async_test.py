import asyncio
import platform
from dashscope.aigc.generation import AioGeneration
import os

from dotenv import load_dotenv
import json
from pprint import pprint

load_dotenv()

api_key_raw = os.getenv('QWEN_API_KEY')
if not api_key_raw:
	raise ValueError('QWEN_API_KEY is not set')

# 定义异步任务列表
async def task(question):
    print(f"发送问题: {question}")
    
    # 确保API密钥不为None
    if not api_key_raw:
        raise ValueError("API密钥不能为空")
    
    response_gen = await AioGeneration.call(
        api_key=api_key_raw,
        model="qwen-plus",  # 模型名称，可按需更换
        prompt=question,
        stream=True
    )
    
    # 处理流式响应
    full_response = ""
    
    # 检查response_gen是否为可迭代对象
    # 如果是异步生成器，正常处理
    async for response in response_gen:
        if hasattr(response, "output") and hasattr(response.output, "text"):
            text = response.output.text
            full_response += text
            print(f"接收到部分答案: {text}")
    
    print(f"问题 '{question}' 的完整回答: {full_response}")
    return full_response

# 主异步函数
async def main():
    questions = ["你是谁？"]
    tasks = [task(q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    # 打印所有结果的汇总
    print("\n所有回答的汇总:")
    for i, result in enumerate(results):
        print(f"问题 {i+1}: {questions[i]}")
        print(f"回答: {result}")
        print("-" * 50)

if __name__ == '__main__':
    # 设置事件循环策略
    if platform.system() == 'Windows':
        # 为Windows设置正确的事件循环策略
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行主协程
    asyncio.run(main())