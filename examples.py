"""
示例脚本，展示如何使用ChatQwen类。
"""

import os
import asyncio
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, SecretStr

from chat_qwen import ChatQwen

from dotenv import load_dotenv

load_dotenv()

api_key_dashscope = os.getenv('DASHSCOPE_API_KEY', '')
if not api_key_dashscope:
	raise ValueError('DASHSCOPE_API_KEY is not set')

# 基本用法示例
def basic_usage():
    print("\n=== 基本用法示例 ===")
    
    # 初始化模型
    chat = ChatQwen(
        temperature=0.7,
        model="qwen-plus",
        dashscope_api_key=SecretStr(api_key_dashscope)
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="介绍一下杭州。")
    ]
    
    # 获取响应
    response = chat.invoke(messages)
    print(f"响应内容: {response.content}")


# 流式输出示例
def streaming_example():
    print("\n=== 流式输出示例 ===")
    
    # 初始化支持流式输出的模型
    chat = ChatQwen(
        temperature=0.7,
        model="qwen-plus",
        streaming=True,
        dashscope_api_key=SecretStr(api_key_dashscope)
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="详细介绍一下人工智能的发展历史。")
    ]
    
    # 获取流式响应
    print("开始流式输出:")
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n流式输出结束")


# 结构化输出示例 - 使用Pydantic v2的方式定义
class Person(BaseModel):
    """表示一个人的结构化数据。"""
    name: str = Field(description="人的姓名")
    age: int = Field(description="人的年龄")
    hobbies: List[str] = Field(description="人的兴趣爱好列表")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "张三",
                    "age": 30,
                    "hobbies": ["阅读", "旅游", "摄影"]
                }
            ]
        }
    }


def structured_output_example():
    print("\n=== 结构化输出示例 ===")
    
    # 初始化模型
    chat = ChatQwen(
        temperature=0.1,  # 结构化输出需要较低的温度以提高确定性
        model="qwen-plus",
        dashscope_api_key=SecretStr(api_key_dashscope)
    )
    
    # 配置结构化输出
    structured_chat = chat.with_structured_output(Person)
    
    # 创建消息
    messages = [
        SystemMessage(content="你需要从用户输入中提取人的信息，并以结构化格式返回。"),
        HumanMessage(content="我叫张三，今年28岁，喜欢篮球、编程和旅游。")
    ]
    
    # 获取结构化响应
    response = structured_chat.invoke(messages)
    print(f"结构化输出: {response}")
    print(f"- 姓名: {response.name}")
    print(f"- 年龄: {response.age}")
    print(f"- 爱好: {', '.join(response.hobbies)}")


# 异步调用示例
async def async_example():
    print("\n=== 异步调用示例 ===")
    
    # 初始化模型
    chat = ChatQwen(
        temperature=0.7,
        model="qwen-plus",
        dashscope_api_key=SecretStr(api_key_dashscope)
    )
    
    # 创建消息
    messages1 = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="简单介绍一下北京。")
    ]
    
    messages2 = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="简单介绍一下上海。")
    ]
    
    # 并行调用模型
    print("开始并行异步调用...")
    results = await asyncio.gather(
        chat.ainvoke(messages1),
        chat.ainvoke(messages2)
    )
    
    print(f"北京简介: {results[0].content[:100]}...")
    print(f"上海简介: {results[1].content[:100]}...")


# 批量处理示例
def batch_example():
    print("\n=== 批量处理示例 ===")
    
    # 初始化模型
    chat = ChatQwen(
        temperature=0.7,
        model="qwen-plus",
        dashscope_api_key=SecretStr(api_key_dashscope)
    )
    
    # 创建多组消息
    batch_messages = [
        [
            SystemMessage(content="你是一个有用的AI助手。"),
            HumanMessage(content="你能用一句话介绍一下西湖吗？")
        ],
        [
            SystemMessage(content="你是一个有用的AI助手。"),
            HumanMessage(content="你能用一句话介绍一下长城吗？")
        ],
        [
            SystemMessage(content="你是一个有用的AI助手。"),
            HumanMessage(content="你能用一句话介绍一下故宫吗？")
        ]
    ]
    
    # 批量获取响应
    print("开始批量处理...")
    results = chat.batch(batch_messages)
    
    for i, result in enumerate(results):
        print(f"响应 {i+1}: {result.generations[0].message.content}")


# 异步流式输出示例
async def async_streaming_example():
    print("\n=== 异步流式输出示例 ===")
    
    # 初始化支持流式输出的模型
    chat = ChatQwen(
        temperature=0.7,
        model="qwen-plus",
        streaming=True,
        dashscope_api_key=SecretStr(api_key_dashscope)
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="简要介绍一下太阳系的行星。")
    ]
    
    # 获取异步流式响应
    print("开始异步流式输出:")
    full_response = ""
    async for chunk in chat.astream(messages):
        chunk_content = str(chunk.content)
        print(chunk_content, end="", flush=True)
        full_response += chunk_content
    print("\n异步流式输出结束")


# 主函数
async def main():
    # 运行各种示例
    basic_usage()
    # streaming_example()
    # structured_output_example()
    # await async_example()
    # batch_example()
    # await async_streaming_example()


if __name__ == "__main__":
    asyncio.run(main()) 