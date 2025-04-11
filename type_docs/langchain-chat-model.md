# Langchain Chat Model

## _stream 方法

### _stream 方法入参

让我分析 `_stream` 方法的参数：

| 参数名 | 类型 | 是否可选 | 描述 |
|--------|------|----------|------|
| messages | List[BaseMessage] | 必选 | 要发送的消息列表 |
| stop | Optional[List[str]] | 可选 | 停止生成的标记列表 |
| run_manager | Optional[CallbackManagerForLLMRun] | 可选 | 用于管理LLM运行的回调管理器 |
| stream_usage | Optional[bool] | 可选 | 是否在流式输出中包含使用情况元数据 |
| **kwargs | - | 可选 | 其他参数，包括： |
| stream | bool | 可选 | 是否启用流式输出，默认为False |
| temperature | float | 可选 | 控制随机性的温度参数 |
| top_p | float | 可选 | 核采样参数 |
| max_tokens | int | 可选 | 最大生成长度 |
| presence_penalty | float | 可选 | 存在惩罚参数 |
| frequency_penalty | float | 可选 | 频率惩罚参数 |
| seed | int | 可选 | 随机种子 |
| logprobs | bool | 可选 | 是否返回对数概率 |
| top_logprobs | int | 可选 | 返回的对数概率数量 |
| logit_bias | Dict[int, int] | 可选 | 对数概率偏置 |
| n | int | 可选 | 生成的数量 |
| response_format | dict | 可选 | 响应格式配置 |
| include_response_headers | bool | 可选 | 是否包含响应头 |
| disabled_params | Dict[str, Any] | 可选 | 禁用的参数 |
| use_responses_api | bool | 可选 | 是否使用响应API |
