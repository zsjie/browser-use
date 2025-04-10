"""基于阿里云Dashscope API实现的ChatQwen模型接口。

这个模块提供了一个与langchain兼容的ChatQwen类，便于在Agent中使用。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

# 需要先安装dashscope: pip install dashscope
import dashscope
from dashscope import Generation
from dashscope.aigc.generation import AioGeneration

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, SecretStr, field_validator, model_validator


logger = logging.getLogger(__name__)


def _convert_message_to_dashscope(message: BaseMessage) -> Dict[str, Any]:
    """将LangChain消息转换为Dashscope API所需的消息格式。"""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        # 处理可能包含工具调用的AI消息
        msg_dict = {"role": "assistant", "content": message.content}
        
        try:
            # 记录AIMessage的全部属性，帮助调试
            logger.info(f"AIMessage类型: {type(message)}")
            logger.info(f"AIMessage属性: {dir(message)}")
            
            # 直接获取tool_calls属性
            if hasattr(message, "tool_calls") and message.tool_calls:
                # 检查tool_calls是否可迭代
                try:
                    # 尝试迭代处理工具调用
                    formatted_tool_calls = []
                    for tc in message.tool_calls:
                        formatted_call = _format_tool_call(tc)
                        if formatted_call:
                            formatted_tool_calls.append(formatted_call)
                    
                    if formatted_tool_calls:
                        msg_dict["tool_calls"] = formatted_tool_calls
                except TypeError:
                    # 如果不可迭代，则作为单个对象处理
                    logger.warning("tool_calls不可迭代，尝试作为单个对象处理")
                    formatted_call = _format_tool_call(message.tool_calls)
                    if formatted_call:
                        msg_dict["tool_calls"] = [formatted_call]
                        
        except Exception as e:
            logger.warning(f"处理AI消息时出错: {e}", exc_info=True)
            
        return msg_dict
    elif isinstance(message, ToolMessage):
        # ToolMessage在DashScope中需要特殊处理
        # 在Dashscope API中，tool消息必须是对前一条包含tool_calls的消息的响应
        # 需要包含tool_call_id字段
        try:
            # 记录ToolMessage的全部属性，帮助调试
            logger.info(f"ToolMessage类型: {type(message)}")
            logger.info(f"ToolMessage属性: {dir(message)}")
            
            if not hasattr(message, "tool_call_id") or not message.tool_call_id:
                # 如果没有tool_call_id，作为普通用户消息处理
                logger.warning("ToolMessage没有tool_call_id，作为普通用户消息处理")
                return {"role": "user", "content": message.content}
            
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id
            }
        except Exception as e:
            logger.warning(f"处理Tool消息时出错: {e}", exc_info=True)
            return {"role": "user", "content": str(message.content)}
    else:
        logger.warning(f"未支持的消息类型: {type(message)}")
        return {"role": "user", "content": str(message.content)}


def _format_tool_call(tc) -> Optional[Dict[str, Any]]:
    """格式化单个工具调用为DashScope API所需的格式。
    
    Args:
        tc: 工具调用对象或字典
        
    Returns:
        格式化后的工具调用字典，如果无法格式化则返回None
    """
    try:
        # 记录接收的工具调用类型和内容
        logger.info(f"工具调用类型: {type(tc)}")
        logger.info(f"工具调用内容: {tc}")
        
        if tc is None:
            return None
            
        # 检查是否是字典
        if isinstance(tc, dict):
            # 确保必要的字段存在
            tool_call = {
                "id": tc.get("id", f"call_{id(tc)}"),
                "type": tc.get("type", "function")
            }
            
            # 确保function字段存在
            if "function" in tc:
                tool_call["function"] = tc["function"]
            else:
                # 如果没有function字段，创建一个默认的
                tool_call["function"] = {
                    "name": tc.get("name", "unknown_function"),
                    "arguments": tc.get("arguments", "{}")
                }
            
            return tool_call
        
        # 如果是ToolCall类型的对象
        # 创建一个新的工具调用字典
        call_id = str(getattr(tc, "id", f"call_{id(tc)}"))
        
        # 创建最终的工具调用格式
        formatted_call: Dict[str, Any] = {
            "id": call_id,
            "type": "function"
        }
        
        # 检查是否有name和args/arguments属性
        name = None
        args = "{}"
        
        # 处理不同结构的工具调用
        # 1. 直接有name和args属性
        if hasattr(tc, "name") and (hasattr(tc, "args") or hasattr(tc, "arguments")):
            name = tc.name
            args = getattr(tc, "args", getattr(tc, "arguments", "{}"))
        # 2. 有function属性，function是一个对象
        elif hasattr(tc, "function"):
            func = tc.function
            if isinstance(func, dict):
                name = func.get("name")
                args = func.get("arguments", "{}")
            else:
                name = getattr(func, "name", None)
                args = getattr(func, "arguments", getattr(func, "args", "{}"))
        
        # 确保args是字符串
        if not isinstance(args, str):
            import json
            try:
                args = json.dumps(args)
            except:
                args = str(args)
                
        # 创建function字段
        function_obj: Dict[str, str] = {
            "name": str(name or "unknown_function"),
            "arguments": str(args)
        }
        
        # 添加function字段到工具调用
        formatted_call["function"] = function_obj
                
        return formatted_call
            
    except Exception as e:
        logger.warning(f"格式化工具调用时出错: {e}", exc_info=True)
        return None


def _create_dashscope_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """将LangChain消息序列转换为Dashscope消息列表。"""
    return [_convert_message_to_dashscope(message) for message in messages]


def _parse_dashscope_chat_response(response) -> AIMessage:
    """解析Dashscope API的响应为AIMessage。
    
    Args:
        response: Dashscope API响应对象
        
    Returns:
        解析后的AIMessage
        
    Raises:
        ValueError: 如果响应包含错误
    """
    try:
        # 记录响应内容帮助调试
        logger.info(f"Dashscope响应: {response}")
        
        if response.status_code == 200:
            # 尝试从响应中提取消息内容
            try:
                message_content = response.output.choices[0].message.content
                return AIMessage(content=message_content)
            except AttributeError as e:
                # 如果无法访问消息内容属性，尝试其他方式获取
                logger.warning(f"无法提取标准消息内容: {e}")
                
                # 尝试获取输出文本
                if hasattr(response.output, "text"):
                    return AIMessage(content=response.output.text)
                # 尝试获取原始输出
                elif hasattr(response, "output") and response.output:
                    # 将输出转换为字符串
                    return AIMessage(content=str(response.output))
                else:
                    # 没有找到可用的输出内容
                    logger.error("无法从响应中提取任何输出内容")
                    return AIMessage(content="[无法获取响应内容]")
        else:
            # 响应包含错误
            error_message = f"Dashscope API错误: HTTP {response.status_code}"
            
            if hasattr(response, "code"):
                error_message += f", 错误码: {response.code}"
            
            if hasattr(response, "message"):
                error_message += f", 错误信息: {response.message}"
                
            raise ValueError(error_message)
    except Exception as e:
        # 捕获所有异常，确保不会因为解析响应而崩溃
        logger.error(f"解析Dashscope响应时出错: {e}", exc_info=True)
        raise ValueError(f"解析响应时出错: {e}")


class ChatQwen(BaseChatModel):
    """与阿里云Dashscope API集成的Qwen聊天模型。

    Attributes:
        model: 要使用的模型名称
        temperature: 生成温度
        max_tokens: 最大生成长度
        top_p: 核采样多样性参数
        top_k: 核采样多样性参数
        api_key: Dashscope API密钥
        stop: 停止词列表
        streaming: 是否启用流式输出
        request_timeout: 请求超时时间
        retry_count: 重试次数
        result_format: 结果格式，一般为message
    """

    model: str = "qwen-plus"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    api_key: SecretStr
    stop: Optional[List[str]] = None
    streaming: bool = False
    request_timeout: Optional[int] = None
    retry_count: int = 3
    result_format: str = "message"
    
    class Config:
        """配置此Pydantic对象。"""
        arbitrary_types_allowed = True

    # 检查温度参数是否在有效范围内
    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("temperature必须在0到1之间")
        return v

    # 验证API密钥
    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """验证API密钥。"""
        api_key = values.get("api_key")
        
        if isinstance(api_key, str):
            values["api_key"] = SecretStr(api_key)
        elif api_key is None:
            raise ValueError("必须提供 api_key")
            
        return values
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型。"""
        return "chat-qwen"

    def _get_api_key(self) -> str:
        """获取API密钥。"""
        return self.api_key.get_secret_value()

    def _convert_params(self) -> Dict[str, Any]:
        """构建API请求参数。"""
        params = {
            "api_key": self._get_api_key(),
            "model": self.model,
            "temperature": self.temperature,
            "result_format": self.result_format,
        }
        
        # 添加可选参数
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.top_p:
            params["top_p"] = self.top_p
        if self.top_k:
            params["top_k"] = self.top_k
        if self.stop:
            params["stop"] = self.stop
        if self.request_timeout:
            params["timeout"] = self.request_timeout
            
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步调用模型生成响应。"""
        try:
            # 转换消息格式并记录
            dashscope_messages = _create_dashscope_messages(messages)
            logger.debug(f"发送到Dashscope的消息: {dashscope_messages}")
            
            # 构建参数
            params = self._convert_params()
            logger.debug(f"基础参数: {params}")
            
            # 合并传入的参数
            params = {**params, **kwargs}
            if stop:
                params["stop"] = stop
            
            params["messages"] = dashscope_messages
            params["stream"] = False
            
            # 记录最终的请求参数
            logger.debug(f"最终请求参数: {params}")
            
            # 发起请求
            logger.info(f"正在调用Dashscope API，模型: {self.model}")
            response = Generation.call(**params)
            logger.info(f"Dashscope响应状态码: {response.status_code}")
            
            # 解析响应
            message = _parse_dashscope_chat_response(response)
            logger.info(f"处理后的消息: {message}")
            
            # 如果有回调管理器，通知回调
            if run_manager:
                # 确保内容是字符串
                content = message.content
                if isinstance(content, str):
                    run_manager.on_llm_new_token(content)
                elif isinstance(content, list):
                    # 如果是复杂结构，转换为字符串
                    run_manager.on_llm_new_token(str(content))
                else:
                    # 其他情况，尝试转换为字符串
                    run_manager.on_llm_new_token(str(content))
            
            return ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            logger.error(f"调用Dashscope模型时出错: {e}", exc_info=True)
            # 重新抛出异常，以便上层代码可以处理
            raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步调用模型生成响应。"""
        try:
            # 转换消息格式并记录
            dashscope_messages = _create_dashscope_messages(messages)
            logger.debug(f"发送到Dashscope的消息: {dashscope_messages}")
            
            # 构建参数
            params = self._convert_params()
            logger.info(f"基础参数: {params}")
            
            # 合并传入的参数
            params = {**params, **kwargs}
            if stop:
                params["stop"] = stop
            
            params["messages"] = dashscope_messages
            params["stream"] = False
            
            # 记录最终的请求参数
            logger.debug(f"最终请求参数: {params}")
            
            # 发起异步请求
            logger.info(f"正在异步调用Dashscope API，模型: {self.model}")
            response = await AioGeneration.call(**params)
            logger.info(f"Dashscope响应状态码: {response.status_code}")
            
            # 解析响应
            message = _parse_dashscope_chat_response(response)
            logger.info(f"处理后的消息: {message}")
            
            # 如果有回调管理器，通知回调
            if run_manager:
                # 确保内容是字符串
                content = message.content
                if isinstance(content, str):
                    await run_manager.on_llm_new_token(content)
                elif isinstance(content, list):
                    # 如果是复杂结构，转换为字符串
                    await run_manager.on_llm_new_token(str(content))
                else:
                    # 其他情况，尝试转换为字符串
                    await run_manager.on_llm_new_token(str(content))
            
            return ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            logger.error(f"异步调用Dashscope模型时出错: {e}", exc_info=True)
            # 重新抛出异常，以便上层代码可以处理
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用模型。"""
        dashscope_messages = _create_dashscope_messages(messages)
        params = self._convert_params()
        
        # 合并传入的参数
        params = {**params, **kwargs}
        if stop:
            params["stop"] = stop
        
        params["messages"] = dashscope_messages
        params["stream"] = True
        params["incremental_output"] = True
        
        # 发起流式请求
        responses = Generation.call(**params)
        
        for response in responses:
            if response.status_code == 200:
                chunk = response.output.choices[0].message.content
                if chunk:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=chunk)
                    )
            else:
                raise ValueError(
                    f"Dashscope API错误: HTTP {response.status_code}, "
                    f"错误码: {response.code}, "
                    f"错误信息: {response.message}"
                )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步流式调用模型。"""
        dashscope_messages = _create_dashscope_messages(messages)
        params = self._convert_params()
        
        # 合并传入的参数
        params = {**params, **kwargs}
        if stop:
            params["stop"] = stop
        
        params["messages"] = dashscope_messages
        params["stream"] = True
        params["incremental_output"] = True
        
        # 发起异步流式请求
        async for response in AioGeneration.call(**params):
            if response.status_code == 200:
                chunk = response.output.choices[0].message.content
                if chunk:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=chunk)
                    )
            else:
                raise ValueError(
                    f"Dashscope API错误: HTTP {response.status_code}, "
                    f"错误码: {response.code}, "
                    f"错误信息: {response.message}"
                )

    def with_structured_output(
        self,
        schema: Union[Dict, Type, object],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """配置模型以返回结构化输出。"""
        # 准备响应格式参数
        params = {"response_format": {"type": "json_object"}}
        
        # 合并传入的其他参数
        params.update(kwargs)
        
        # 返回一个配置了结构化输出的新实例
        new_instance = self.bind(**params)
        
        # 简单方法：使用langchain的标准方法
        from langchain_core.output_parsers import JsonOutputParser
        
        chain = new_instance
        if not include_raw:
            # 如果不需要原始输出，添加JSON解析器
            parser = JsonOutputParser()
            chain = chain | parser
            
            # 如果提供了模式，尝试转换为相应的类型
            if schema is not None:
                from pydantic import BaseModel
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    # 如果schema是一个Pydantic模型类，使用它解析
                    chain = chain | (lambda x: schema.model_validate(x))
        
        return chain

    def batch(
        self,
        messages: List[List[BaseMessage]],
        *,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[ChatResult]:
        """批量处理多组消息。"""
        results = []
        for msg in messages:
            results.append(self._generate(msg, **kwargs))
        return results

    async def abatch(
        self,
        messages: List[List[BaseMessage]],
        *,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[ChatResult]:
        """异步批量处理多组消息。"""
        tasks = [self._agenerate(msg, **kwargs) for msg in messages]
        return await asyncio.gather(*tasks)

    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """直接调用模型，与Browser-use Agent兼容。
        
        重写基类的invoke方法，确保返回格式与Browser-use Agent兼容。
        """
        # 记录调用信息
        logger.info(f"调用invoke方法，输入消息数量: {len(input)}")
        
        # 调用底层的生成方法
        result = self._generate(input, **kwargs)
        
        # 获取AIMessage内容
        if result and result.generations and len(result.generations) > 0:
            message = result.generations[0].message
            
            # 记录完整消息
            logger.info(f"完整的返回消息: {message}")
            content = message.content
            
            # 直接返回内容字符串，让Browser-use Agent自己处理
            return content
        
        # 如果没有生成内容，返回空字符串
        return ""
    
    async def ainvoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """异步调用模型，与Browser-use Agent兼容。
        
        重写基类的ainvoke方法，确保返回格式与Browser-use Agent兼容。
        """
        # 记录调用信息
        logger.info(f"调用ainvoke方法，输入消息数量: {len(input)}")
        
        # 调用底层的异步生成方法
        result = await self._agenerate(input, **kwargs)
        
        # 获取AIMessage内容
        if result and result.generations and len(result.generations) > 0:
            message = result.generations[0].message
            
            # 记录完整消息
            logger.info(f"完整的返回消息: {message}")
            content = message.content
            
            # 直接返回内容字符串，让Browser-use Agent自己处理
            return content
        
        # 如果没有生成内容，返回空字符串
        return "" 