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
        return {"role": "assistant", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content}
    else:
        raise ValueError(f"未支持的消息类型: {type(message)}")


def _create_dashscope_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """将LangChain消息序列转换为Dashscope消息列表。"""
    return [_convert_message_to_dashscope(message) for message in messages]


def _parse_dashscope_chat_response(response) -> AIMessage:
    """解析Dashscope API的响应为AIMessage。"""
    if response.status_code == 200:
        return AIMessage(content=response.output.choices[0].message.content)
    else:
        raise ValueError(
            f"Dashscope API错误: HTTP {response.status_code}, "
            f"错误码: {response.code}, "
            f"错误信息: {response.message}"
        )


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
        dashscope_messages = _create_dashscope_messages(messages)
        params = self._convert_params()
        
        # 合并传入的参数
        params = {**params, **kwargs}
        if stop:
            params["stop"] = stop
        
        params["messages"] = dashscope_messages
        params["stream"] = False
        
        # 发起请求
        response = Generation.call(**params)
        message = _parse_dashscope_chat_response(response)
        
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步调用模型生成响应。"""
        dashscope_messages = _create_dashscope_messages(messages)
        params = self._convert_params()
        
        # 合并传入的参数
        params = {**params, **kwargs}
        if stop:
            params["stop"] = stop
        
        params["messages"] = dashscope_messages
        params["stream"] = False
        
        # 发起异步请求
        response = await AioGeneration.call(**params)
        message = _parse_dashscope_chat_response(response)
        
        return ChatResult(generations=[ChatGeneration(message=message)])

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