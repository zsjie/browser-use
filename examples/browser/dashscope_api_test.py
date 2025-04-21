import os
from dashscope import Generation
from dotenv import load_dotenv
import json
from pprint import pprint

load_dotenv()

api_key_raw = os.getenv('QWEN_API_KEY')
if not api_key_raw:
	raise ValueError('QWEN_API_KEY is not set')

messages = [
    {'role':'system','content':'you are a helpful assistant'},
    {'role': 'user','content': '你是谁？请用 JSON 格式介绍你自己'}]
responses = Generation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=api_key_raw,
    model="qwen-plus", # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=messages,
    response_format={"type": "json_object"},
    result_format='message',
    stream=True,
    # 增量式流式输出
    incremental_output=True
    )
full_content = ""
print("流式输出内容为：")
response_count = 0
for response in responses:
    # 打印usage信息
    # if hasattr(response, 'usage'):
    #     print(f"Token使用情况: 输入 {response.usage.input_tokens} tokens, 输出 {response.usage.output_tokens} tokens")
    # full_content += response.output.choices[0].message.content
    print(response)
    
print(f"完整内容为：{full_content}")
