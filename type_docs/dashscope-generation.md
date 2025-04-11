# Dashscope Generation API

## call 方法

### call 方法入参

| 参数名 | 类型 | 是否可选 | 描述 |
|--------|------|----------|------|
| model | str | 必选 | 请求的模型名称，如 qwen-turbo |
| prompt | Any | 可选 | 输入提示词 |
| history | list | 可选 | 用户提供的历史对话记录，已废弃 |
| api_key | str | 可选 | API密钥，如果为None则使用默认规则获取 |
| messages | List[Message] | 可选 | 生成消息列表，包含角色和内容 |
| plugins | Union[str, Dict[str, Any]] | 可选 | 插件配置，可以是插件配置字符串或字典 |
| workspace | str | 可选 | DashScope工作空间ID |
| **kwargs | - | 可选 | 其他参数，包括： |
| stream | bool | 可选 | 是否启用服务器发送事件，默认为False |
| temperature | float | 可选 | 控制随机性和多样性的程度，范围(0, 2) |
| top_p | float | 可选 | 核采样策略，只考虑概率质量前top_p的token |
| top_k | int | 可选 | 生成时的候选集大小，默认为0 |
| enable_search | bool | 可选 | 是否启用网络搜索，默认为False |
| customized_model_id | str | 可选 | 企业特定的大模型ID |
| result_format | str | 可选 | 结果格式，可选message或text |
| incremental_output | bool | 可选 | 控制流式输出模式，默认为False |
| stop | list[str]或list[list[int]] | 可选 | 控制生成停止的条件 |
| max_tokens | int | 可选 | 期望输出的最大token数 |
| repetition_penalty | float | 可选 | 控制生成时的重复性，1.0表示无惩罚 |

注意：

1. prompt 和 messages 至少需要提供一个
2. model 是必选参数
3. history 参数已废弃，建议使用 messages 参数
4. 其他参数都是可选的，通过 **kwargs 传入
