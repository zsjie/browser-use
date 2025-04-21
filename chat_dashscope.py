"""Dashscope chat wrapper."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import ssl
import sys
import warnings
from functools import partial
from io import BytesIO
from json import JSONDecodeError
from math import ceil
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import urlparse

import certifi
import openai
import tiktoken
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.base import (
    LangSmithParams,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.tools.base import _stringify
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    TypeBaseModel,
    is_basemodel_subclass,
)
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self

from dashscope.api_entities.dashscope_response import (GenerationResponse,
                                                       Message, Role)

from dashscope import (AioGeneration, Generation)

if TYPE_CHECKING:
    from openai.types.responses import Response

logger = logging.getLogger(__name__)

global_ssl_context = ssl.create_default_context(cafile=certifi.where())

_FUNCTION_CALL_IDS_MAP_KEY = "__dashscope_function_call_ids__"


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role in ("system", "developer"):
        if role == "developer":
            additional_kwargs = {"__openai_role__": role}
        else:
            additional_kwargs = {}
        return SystemMessage(
            content=_dict.get("content", ""),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]



def _handle_openai_bad_request(e: openai.BadRequestError) -> None:
    if (
        "'response_format' of type 'json_schema' is not supported with this model"
    ) in e.message:
        message = (
            "This model does not support OpenAI's structured output feature, which "
            "is the default method for `with_structured_output` as of "
            "langchain-openai==0.3. To use `with_structured_output` with this model, "
            'specify `method="function_calling"`.'
        )
        warnings.warn(message)
        raise e
    elif "Invalid schema for response_format" in e.message:
        message = (
            "Invalid schema for OpenAI's structured output feature, which is the "
            "default method for `with_structured_output` as of langchain-openai==0.3. "
            'Specify `method="function_calling"` instead or update your schema. '
            "See supported schemas: "
            "https://platform.openai.com/docs/guides/structured-outputs#supported-schemas"  # noqa: E501
        )
        warnings.warn(message)
        raise e
    else:
        raise


class _FunctionCall(TypedDict):
    name: str

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]

def _update_token_usage(
    overall_token_usage: Union[int, dict], new_usage: Union[int, dict]
) -> Union[int, dict]:
    # Token usage is either ints or dictionaries
    # `reasoning_tokens` is nested inside `completion_tokens_details`
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return new_usage + overall_token_usage
    elif isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return {
            k: _update_token_usage(overall_token_usage.get(k, 0), v)
            for k, v in new_usage.items()
        }
    else:
        warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
        return new_usage

def _create_usage_metadata(oai_token_usage: dict) -> UsageMetadata:
    input_tokens = oai_token_usage.get("prompt_tokens", 0)
    output_tokens = oai_token_usage.get("completion_tokens", 0)
    total_tokens = oai_token_usage.get("total_tokens", input_tokens + output_tokens)
    input_token_details: dict = {
        "audio": (oai_token_usage.get("prompt_tokens_details") or {}).get(
            "audio_tokens"
        ),
        "cache_read": (oai_token_usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
    }
    output_token_details: dict = {
        "audio": (oai_token_usage.get("completion_tokens_details") or {}).get(
            "audio_tokens"
        ),
        "reasoning": (oai_token_usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
    }
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None}
        ),
        output_token_details=OutputTokenDetails(
            **{k: v for k, v in output_token_details.items() if v is not None}
        ),
    )

def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    elif role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content, id=id_, additional_kwargs=additional_kwargs
        )
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    else:
        return default_class(content=content, id=id_)  # type: ignore

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        if "audio" in message.additional_kwargs:
            # openai doesn't support passing the data back - only the id
            # https://platform.openai.com/docs/guides/audio/multi-turn-conversations
            raw_audio = message.additional_kwargs["audio"]
            audio = (
                {"id": message.additional_kwargs["audio"]["id"]}
                if "id" in raw_audio
                else raw_audio
            )
            message_dict["audio"] = audio
    elif isinstance(message, SystemMessage):
        message_dict["role"] = message.additional_kwargs.get(
            "__openai_role__", "system"
        )
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict

def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        formatted_content = []
        for block in content:
            # Remove unexpected block types
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] in ("tool_use", "thinking")
            ):
                continue
            # Anthropic image blocks
            elif (
                isinstance(block, dict)
                and block.get("type") == "image"
                and (source := block.get("source"))
                and isinstance(source, dict)
            ):
                if source.get("type") == "base64" and (
                    (media_type := source.get("media_type"))
                    and (data := source.get("data"))
                ):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                elif source.get("type") == "url" and (url := source.get("url")):
                    formatted_content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                else:
                    continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content



class BaseChatDashscope(BaseChatModel):
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    root_client: Any = Field(default=None, exclude=True)  #: :meta private:
    root_async_client: Any = Field(default=None, exclude=True)  #: :meta private:


    model_name: str = Field(default="qwen-turbo", alias="model")
    """Model name to use."""
    temperature: Optional[float] = None
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    dashscope_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("DASHSCOPE_API_KEY", default=None)
    )
    dashscope_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    dashscope_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `DASHSCOPE_ORG_ID` if not provided."""
    # to support explicit proxy for Dashscope
    dashscope_proxy: Optional[str] = Field(
        default_factory=from_env("DASHSCOPE_PROXY", default=None)
    )
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Dashscope completion API. Can be float, httpx.Timeout or 
        None."""
    stream_usage: bool = False
    """Whether to include usage metadata in streaming output. If True, an additional
    message chunk will be generated during the stream including usage metadata.

    .. versionadded:: 0.3.9
    """
    max_retries: Optional[int] = None
    """Maximum number of retries to make when generating."""
    presence_penalty: Optional[float] = None
    """Penalizes repeated tokens."""
    frequency_penalty: Optional[float] = None
    """Penalizes repeated tokens according to frequency."""
    seed: Optional[int] = None
    """Seed for generation"""
    logprobs: Optional[bool] = None
    """Whether to return logprobs."""
    top_logprobs: Optional[int] = None
    """Number of most likely tokens to return at each token position, each with
     an associated log probability. `logprobs` must be set to true 
     if this parameter is used."""
    logit_bias: Optional[Dict[int, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: Optional[int] = None
    """Number of chat completions to generate for each prompt."""
    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""
    max_tokens: Optional[int] = Field(default=None)
    """Maximum number of tokens to generate."""
    reasoning_effort: Optional[str] = None
    """Constrains effort on reasoning for reasoning models. 
    
    Reasoning models only, like OpenAI o1 and o3-mini.

    Currently supported values are low, medium, and high. Reducing reasoning effort 
    can result in faster responses and fewer tokens used on reasoning in a response.
    
    .. versionadded:: 0.2.14
    """
    tiktoken_model_name: Optional[str] = None
    """The model name to pass to tiktoken when using this class. 
    Tiktoken is used to count the number of tokens in documents to constrain 
    them to be under a certain limit. By default, when set to None, this will 
    be the same as the embedding model name. However, there are some cases 
    where you may want to use this Embedding class with a model name not 
    supported by tiktoken. This can include when using Azure embeddings or 
    when using one of the many model providers that expose an OpenAI-like 
    API but with different models. In those cases, in order to avoid erroring 
    when tiktoken is called, you can specify a model name to use here."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = Field(default=None, exclude=True)
    """Optional httpx.Client. Only used for sync invocations. Must specify 
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = Field(default=None, exclude=True)
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify 
        http_client as well if you'd like a custom client for sync invocations."""
    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    extra_body: Optional[Mapping[str, Any]] = None
    """Optional additional JSON properties to include in the request parameters when
    making requests to OpenAI compatible APIs, such as vLLM."""
    include_response_headers: bool = False
    """Whether to include response headers in the output message response_metadata."""
    disabled_params: Optional[Dict[str, Any]] = Field(default=None)
    """Parameters of the OpenAI client or chat.completions endpoint that should be 
    disabled for the given model.
    
    Should be specified as ``{"param": None | ['val1', 'val2']}`` where the key is the 
    parameter and the value is either None, meaning that parameter should never be
    used, or it's a list of disabled values for the parameter.
    
    For example, older models may not support the 'parallel_tool_calls' parameter at 
    all, in which case ``disabled_params={"parallel_tool_calls": None}`` can be passed 
    in.
    
    If a parameter is disabled then it will not be used by default in any methods, e.g.
    in :meth:`~langchain_openai.chat_models.base.ChatOpenAI.with_structured_output`.
    However this does not prevent a user from directly passed in the parameter during
    invocation. 
    """

    use_responses_api: Optional[bool] = None
    """Whether to use the Responses API instead of the Chat API.

    If not specified then will be inferred based on invocation params.

    .. versionadded:: 0.3.9
    """

    model_config = ConfigDict(populate_by_name=True)

    plugins: Optional[Union[str, Dict[str, Any]]] = None
    """插件配置，可以是插件配置字符串或字典"""

    workspace: Optional[str] = None
    """DashScope工作空间ID"""

    stream: Optional[bool] = False
    """是否启用服务器发送事件，默认为False"""

    top_k: Optional[int] = 0
    """生成时的候选集大小，默认为0"""

    enable_search: Optional[bool] = False
    """是否启用搜索，默认为False"""

    customized_model_id: Optional[str] = None
    """企业特定的大模型ID"""

    result_format: Optional[str] = None
    """结果格式，可选message或text"""

    incremental_output: Optional[bool] = False
    """控制流式输出模式，默认为False"""

    repetition_penalty: Optional[float] = None
    """控制生成时的重复性，1.0表示无惩罚"""

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values
    
    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        elif self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")
        
        self.dashscope_organization = (
            self.dashscope_organization
            or os.getenv("DASHSCOPE_ORG_ID")
            or os.getenv("DASHSCOPE_ORGANIZATION")
        )
        self.dashscope_api_base = self.dashscope_api_base or os.getenv("DASHSCOPE_API_BASE")
        client_params: dict = {
            "api_key": (
                self.dashscope_api_key.get_secret_value() if self.dashscope_api_key else None
            ),
            "organization": self.dashscope_organization,
            "base_url": self.dashscope_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries
        
        if self.dashscope_proxy and (self.http_client or self.http_async_client):
            dashscope_proxy = self.dashscope_proxy
            http_client = self.http_client
            http_async_client = self.http_async_client
            raise ValueError(
                "Cannot specify 'dashscope_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{dashscope_proxy=}\n{http_client=}\n{http_async_client=}"
            )
        if not self.client:
            if self.dashscope_proxy and not self.http_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_client = httpx.Client(
                    proxy=self.dashscope_proxy, verify=global_ssl_context
                )
            sync_specific = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific) # type: ignore
            self.client = self.root_client.chat.completions
        if not self.async_client:
            if self.dashscope_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_async_client = httpx.AsyncClient(
                    proxy=self.dashscope_proxy, verify=global_ssl_context
                )
                async_specific = {"http_client": self.http_async_client}
                self.root_async_client = openai.AsyncOpenAI(**client_params, **async_specific) # type: ignore
                self.async_client = self.root_async_client.chat.completions
        return self
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Dashscope API."""
        exclude_if_none = {
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "logit_bias": self.logit_bias,
            "stop": self.stop or None,  # also exclude empty list for this
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "n": self.n,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

        params = {
            "model": self.model_name,
            "stream": self.streaming,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
            **self.model_kwargs,
        }

        return params
    
    @property
    def _default_params_dashscope(self) -> Dict[str, Any]:
        """Get the default parameters for calling Dashscope API."""
        exclude_if_none = {
            "plugins": self.plugins,
            "workspace": self.workspace,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_search": self.enable_search,
            "customized_model_id": self.customized_model_id,
            "result_format": self.result_format,
            "incremental_output": self.incremental_output,
            "stop": self.stop or None,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        params = {
            "model": self.model_name,
            "stream": self.stream,
            "api_key": self.dashscope_api_key.get_secret_value() if self.dashscope_api_key else None,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
        }
        return params
    
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output.get("token_usage")
            if token_usage is not None:
                for k, v in token_usage.items():
                    if v is None:
                        continue
                    if k in overall_token_usage:
                        overall_token_usage[k] = _update_token_usage(
                            overall_token_usage[k], v
                        )
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined
    
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        if chunk.get("type") == "content.delta":  # from beta.chat.completions.stream
            return None
        token_usage = chunk.get("usage")
        choices = (
            chunk.get("choices", [])
            # from beta.chat.completions.stream
            or chunk.get("chunk", {}).get("choices", [])
        )

        usage_metadata: Optional[UsageMetadata] = (
            _create_usage_metadata(token_usage) if token_usage else None
        )
        if len(choices) == 0:
            # logprobs is implicitly None
            generation_chunk = ChatGenerationChunk(
                message=default_chunk_class(content="", usage_metadata=usage_metadata)
            )
            return generation_chunk
        
        choice = choices[0]
        if choice["delta"] is None:
            return None
        
        message_chunk = _convert_delta_to_message_chunk(
            choice["delta"], default_chunk_class
        )
        generation_info = {**base_generation_info} if base_generation_info else {}

        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            if model_name := chunk.get("model"):
                generation_info["model_name"] = model_name
            if system_fingerprint := chunk.get("system_fingerprint"):
                generation_info["system_fingerprint"] = system_fingerprint

        logprobs = choice.get("logprobs")
        if logprobs:
            generation_info["logprobs"] = logprobs

        if usage_metadata and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = usage_metadata

        generation_chunk = ChatGenerationChunk(
            message=message_chunk, generation_info=generation_info or None
        )
        return generation_chunk
    
    def _stream_responses(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if self.include_response_headers:
            raw_context_manager = self.root_client.with_raw_response.responses.create(
                **payload
            )
            context_manager = raw_context_manager.parse()
            headers = {"headers": dict(raw_context_manager.headers)}
        else:
            context_manager = self.root_client.responses.create(**payload)
            headers = {}
        original_schema_obj = kwargs.get("response_format")

        with context_manager as response:
            is_first_chunk = True
            for chunk in response:
                metadata = headers if is_first_chunk else {}
                if generation_chunk := _convert_responses_chunk_to_generation_chunk(
                    chunk, schema=original_schema_obj, metadata=metadata
                ):
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk
                        )
                    is_first_chunk = False
                    yield generation_chunk

    async def _astream_responses(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if self.include_response_headers:
            raw_context_manager = (
                await self.root_async_client.with_raw_response.responses.create(
                    **payload
                )
            )
            context_manager = raw_context_manager.parse()
            headers = {"headers": dict(raw_context_manager.headers)}
        else:
            context_manager = await self.root_async_client.responses.create(**payload)
            headers = {}
        original_schema_obj = kwargs.get("response_format")

        async with context_manager as response:
            is_first_chunk = True
            async for chunk in response:
                metadata = headers if is_first_chunk else {}
                if generation_chunk := _convert_responses_chunk_to_generation_chunk(
                    chunk, schema=original_schema_obj, metadata=metadata
                ):
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk
                        )
                    is_first_chunk = False
                    yield generation_chunk

    def _should_stream_usage(
        self, stream_usage: Optional[bool] = None, **kwargs: Any
    ) -> bool:
        """Determine whether to include usage metadata in streaming output.

        For backwards compatibility, we check for `stream_options` passed
        explicitly to kwargs or in the model_kwargs and override self.stream_usage.
        """
        stream_usage_sources = [  # order of precedence
            stream_usage,
            kwargs.get("stream_options", {}).get("include_usage"),
            self.model_kwargs.get("stream_options", {}).get("include_usage"),
            self.stream_usage,
        ]
        for source in stream_usage_sources:
            if isinstance(source, bool):
                return source
        return self.stream_usage

    def _get_dashscope_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, FunctionMessage):
            role = "bot"
        elif isinstance(message, ToolMessage):
            role = "bot"
        elif isinstance(message, ChatMessage):
            role = "user"
        else:
            msg = f"Got unsupported message type: {message}"
            raise ValueError(msg)  # noqa: TRY004
        return role
            

    def _convert_to_dashscope_message(self, messages: List[BaseMessage]) -> List[Message]:
        """Convert a list of BaseMessage to a list of dict."""
        dashscope_messages = []
        for message in messages:
            if isinstance(message.content, str):
                dashscope_messages.append(Message(role=self._get_dashscope_role(message), content=message.content))
            else:
                # TODO: 处理多模态内容
                raise ValueError(f"Unsupported message content type: {type(message.content)}")
        return dashscope_messages
    
    def _get_request_payload_dashscope(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        
        messages = self._convert_to_dashscope_message(self._convert_input(input_).to_messages())
        
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params_dashscope, **kwargs}
        payload["messages"] = messages
        return payload
    
    def _stream_dashscope(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream"] = True

        if stream_usage is not None:
            warnings.warn("stream_usage is not supported in Dashscope. "
                          "Token usage will be included in the response by default."
            )

        payload = self._get_request_payload_dashscope(messages, stop=stop, **kwargs)

        if "response_format" in payload:
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )
        
        # Call Generation API with streaming
        response_stream = Generation.call(**payload)
        
        # Convert each GenerationResponse to ChatGenerationChunk
        for response in response_stream:
            # Extract content from the first choice's message
            content = ""
            finish_reason = None
            if hasattr(response.output, 'choices') and response.output.choices:
                choice = response.output.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = str(choice.message.content)  # Ensure content is a string
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
            
            # Create AIMessageChunk
            message_chunk = AIMessageChunk(
                content=content,
                additional_kwargs={},
                usage_metadata=UsageMetadata(
                    input_tokens=response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else 0,
                    output_tokens=response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0,
                    total_tokens=response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                ) if hasattr(response, 'usage') else None
            )
            
            # Create ChatGenerationChunk
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info={
                    "finish_reason": finish_reason,
                    "model": response.model if hasattr(response, 'model') else self.model_name,
                    "request_id": response.request_id if hasattr(response, 'request_id') else None
                }
            )
            
            # Call callback if provided
            if run_manager:
                run_manager.on_llm_new_token(
                    content,
                    chunk=generation_chunk
                )
            
            yield generation_chunk

    def _generate_dashscope(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream_dashscope(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        
        payload = self._get_request_payload_dashscope(messages, stop=stop, **kwargs)

        if "response_format" in payload:
            payload.pop("stream")
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )
            
        response = Generation.call(**payload)

        return self._create_chat_result_dashscope(cast(GenerationResponse, response))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            response_stream = self.root_client.beta.chat.completions.stream(**payload)
            context_manager = response_stream
        else:
            if self.include_response_headers:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = self.client.create(**payload)
            context_manager = response
        try:
            with context_manager as response:
                is_first_chunk = True
                for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            try:
                response = self.root_client.beta.chat.completions.parse(**payload)
            except openai.BadRequestError as e:
                _handle_openai_bad_request(e)
        elif self._use_responses_api(payload):
            original_schema_obj = kwargs.get("response_format")
            if original_schema_obj and _is_pydantic_class(original_schema_obj):
                response = self.root_client.responses.parse(**payload)
            else:
                if self.include_response_headers:
                    raw_response = self.root_client.with_raw_response.responses.create(
                        **payload
                    )
                    response = raw_response.parse()
                    generation_info = {"headers": dict(raw_response.headers)}
                else:
                    response = self.root_client.responses.create(**payload)
            return _construct_lc_result_from_responses_api(
                response, schema=original_schema_obj, metadata=generation_info
            )
        elif self.include_response_headers:
            raw_response = self.client.with_raw_response.create(**payload)
            response = raw_response.parse()
            generation_info = {"headers": dict(raw_response.headers)}
        else:
            response = self.client.create(**payload)
        return self._create_chat_result(response, generation_info)

    def _use_responses_api(self, payload: dict) -> bool:
        if isinstance(self.use_responses_api, bool):
            return self.use_responses_api
        else:
            return _use_responses_api(payload)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}
        if self._use_responses_api(payload):
            payload = _construct_responses_api_payload(messages, payload)
        else:
            payload["messages"] = [_convert_message_to_dict(m) for m in messages]
        return payload
    
    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        generations = []

        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        # Sometimes the AI Model calling will get error, we should raise it.
        # Otherwise, the next code 'choices.extend(response["choices"])'
        # will throw a "TypeError: 'NoneType' object is not iterable" error
        # to mask the true error. Because 'response["choices"]' is None.
        if response_dict.get("error"):
            raise ValueError(response_dict.get("error"))

        token_usage = response_dict.get("usage")
        for res in response_dict["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            generation_info = generation_info or {}
            generation_info["finish_reason"] = (
                res.get("finish_reason")
                if res.get("finish_reason") is not None
                else generation_info.get("finish_reason")
            )
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response_dict.get("model", self.model_name),
            "system_fingerprint": response_dict.get("system_fingerprint", ""),
        }
        if "id" in response_dict:
            llm_output["id"] = response_dict["id"]

        if isinstance(response, openai.BaseModel) and getattr(
            response, "choices", None
        ):
            message = response.choices[0].message  # type: ignore[attr-defined]
            if hasattr(message, "parsed"):
                generations[0].message.additional_kwargs["parsed"] = message.parsed
            if hasattr(message, "refusal"):
                generations[0].message.additional_kwargs["refusal"] = message.refusal

        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_chat_result_dashscope(
        self,
        response: GenerationResponse,
    ) -> ChatResult:
        generations = []

        if response.status_code == 200 and response.output and response.output.choices:
            choice = response.output.choices[0]
            content = ""
            finish_reason = None

            if choice.message and choice.message.content:
                content = str(choice.message.content)
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            message = AIMessage(content=content)
            generation = ChatGeneration(message=message, generation_info={"finish_reason": finish_reason})

            generations.append(generation)

        token_usage = {}
        if hasattr(response, 'usage'):
            if hasattr(response.usage, 'input_tokens'):
                token_usage["input_tokens"] = response.usage.input_tokens
            if hasattr(response.usage, 'output_tokens'):
                token_usage["output_tokens"] = response.usage.output_tokens
            if hasattr(response.usage, 'total_tokens'):
                token_usage["total_tokens"] = response.usage.total_tokens

        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "request_id": response.request_id,
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream_dashscope(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True

        if stream_usage is not None:
            warnings.warn("stream_usage is not supported in Dashscope. "
                          "Token usage will be included in the response by default."
            )

        payload = self._get_request_payload_dashscope(messages, stop=stop, **kwargs)

        if "response_format" in payload:
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )

        # AioGeneration.call 返回异步生成器
        response_stream = await AioGeneration.call(**payload)
        
        # 处理异步生成器
        async for response in response_stream:
            # 提取内容
            content = ""
            finish_reason = None
            if hasattr(response.output, 'choices') and response.output.choices:
                choice = response.output.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = str(choice.message.content)
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
            
            # 创建 AIMessageChunk
            message_chunk = AIMessageChunk(
                content=content,
                additional_kwargs={},
                usage_metadata=UsageMetadata(
                    input_tokens=response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else 0,
                    output_tokens=response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0,
                    total_tokens=response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                ) if hasattr(response, 'usage') else None
            )
            
            # 创建 ChatGenerationChunk
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info={
                    "finish_reason": finish_reason,
                    "model": response.model if hasattr(response, 'model') else self.model_name,
                    "request_id": response.request_id if hasattr(response, 'request_id') else None
                }
            )
            
            # 调用回调
            if run_manager:
                await run_manager.on_llm_new_token(
                    content,
                    chunk=generation_chunk
                )
            
            yield generation_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            response_stream = self.root_client.beta.chat.completions.stream(**payload)
            context_manager = response_stream
        else:
            if self.include_response_headers:
                raw_response = await self.async_client.with_raw_response.create(
                    **payload
                )
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = await self.async_client.create(**payload)
            context_manager = response
        try:
            async with context_manager as response:
                is_first_chunk = True
                async for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = await response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk
            

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            try:
                response = await self.root_async_client.beta.chat.completions.parse(
                    **payload
                )
            except openai.BadRequestError as e:
                _handle_openai_bad_request(e)
        elif self._use_responses_api(payload):
            original_schema_obj = kwargs.get("response_format") 
            if original_schema_obj and _is_pydantic_class(original_schema_obj):
                response = await self.root_async_client.responses.parse(**payload)
            else:
                if self.include_response_headers:
                    raw_response = (
                        await self.root_async_client.with_raw_response.responses.create(
                            **payload
                        )
                    )
                    response = raw_response.parse()
                    generation_info = {"headers": dict(raw_response.headers)}
                else:
                    response = await self.root_async_client.responses.create(**payload)
            return _construct_lc_result_from_responses_api(
                response, schema=original_schema_obj, metadata=generation_info
            )
        elif self.include_response_headers:
            raw_response = await self.async_client.with_raw_response.create(**payload)
            response = raw_response.parse()
            generation_info = {"headers": dict(raw_response.headers)}
        else:
            response = await self.async_client.create(**payload)
        return await run_in_executor(
            None, self._create_chat_result, response, generation_info
        )
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}
    
    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }
    
    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="openai",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens) or params.get(
            "max_completion_tokens", self.max_tokens
        ):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-chat"
    
    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        return model, encoding
    
    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)
    
    def get_num_tokens_from_messages(
        self,
        messages: List[BaseMessage],
        tools: Optional[
            Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]
        ] = None,
    ) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        **Requirements**: You must have the ``pillow`` installed if you want to count
        image tokens if you are specifying the image as a base64 string, and you must
        have both ``pillow`` and ``httpx`` installed if you are specifying the image
        as a URL. If these aren't installed image inputs will be ignored in token
        counting.

        OpenAI reference: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of dict, BaseModel, function, or BaseTools
                to be converted to tool schemas.
        """
        # TODO: Count bound tools as part of input.
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools."
            )
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}. See "
                "https://platform.openai.com/docs/guides/text-generation/managing-tokens"  # noqa: E501
                " for information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # This is an inferred approximation. OpenAI does not document how to
                # count tool message tokens.
                if key == "tool_call_id":
                    num_tokens += 3
                    continue
                if isinstance(value, list):
                    # content or tool calls
                    for val in value:
                        if isinstance(val, str) or val["type"] == "text":
                            text = val["text"] if isinstance(val, dict) else val
                            num_tokens += len(encoding.encode(text))
                        elif val["type"] == "image_url":
                            if val["image_url"].get("detail") == "low":
                                num_tokens += 85
                            else:
                                image_size = _url_to_size(val["image_url"]["url"])
                                if not image_size:
                                    continue
                                num_tokens += _count_image_tokens(*image_size)
                        # Tool/function call token counting is not documented by OpenAI.
                        # This is an approximation.
                        elif val["type"] == "function":
                            num_tokens += len(
                                encoding.encode(val["function"]["arguments"])
                            )
                            num_tokens += len(encoding.encode(val["function"]["name"]))
                        else:
                            raise ValueError(
                                f"Unrecognized content block type\n\n{val}"
                            )
                elif not value:
                    continue
                else:
                    # Cast str(value) in case the message value is not a string
                    # This occurs with function messages
                    num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:

                - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                - ``"auto"``: automatically selects a tool (including no tool).
                - ``"none"``: does not call a tool.
                - ``"any"`` or ``"required"`` or ``True``: force at least one tool to be called.
                - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                - ``False`` or ``None``: no effect, default OpenAI behavior.
            strict: If True, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. If True, the input schema will be
                validated according to
                https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.
                If False, input schema will not be validated and model output will not
                be validated.
                If None, ``strict`` argument will not be passed to the model.
            parallel_tool_calls: Set to ``False`` to disable parallel tool use.
                Defaults to ``None`` (no specification, which allows parallel tool use).
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_openai.chat_models.base.ChatOpenAI.bind`.

        .. versionchanged:: 0.1.21

            Support for ``strict`` argument added.

        """  # noqa: E501

        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
            else:
                pass
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in (
                    "file_search",
                    "web_search_preview",
                    "computer_use_preview",
                ):
                    tool_choice = {"type": tool_choice}
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                elif tool_choice == "any":
                    tool_choice = "required"
                else:
                    pass
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class (support added in 0.1.20),
                - or a Pydantic class.

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method: The method for steering model generation, one of:

                - "function_calling":
                    Uses OpenAI's tool-calling (formerly called function calling)
                    API: https://platform.openai.com/docs/guides/function-calling
                - "json_schema":
                    Uses OpenAI's Structured Output API: https://platform.openai.com/docs/guides/structured-outputs
                    Supported for "gpt-4o-mini", "gpt-4o-2024-08-06", "o1", and later
                    models.
                - "json_mode":
                    Uses OpenAI's JSON mode. Note that if using JSON mode then you
                    must include instructions for formatting the output into the
                    desired schema into the model call:
                    https://platform.openai.com/docs/guides/structured-outputs/json-mode

                Learn more about the differences between the methods and which models
                support which methods here:

                - https://platform.openai.com/docs/guides/structured-outputs/structured-outputs-vs-json-mode
                - https://platform.openai.com/docs/guides/structured-outputs/function-calling-vs-response-format

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".
            strict:

                - True:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to
                    https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
                - False:
                    Input schema will not be validated and model output will not be
                    validated.
                - None:
                    ``strict`` argument will not be passed to the model.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            | If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            | If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - "raw": BaseMessage
            - "parsed": None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - "parsing_error": Optional[BaseException]

        .. versionchanged:: 0.1.20

            Added support for TypedDict class ``schema``.

        .. versionchanged:: 0.1.21

            Support for ``strict`` argument added.
            Support for ``method`` = "json_schema" added.
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        if strict is not None and method == "json_mode":
            raise ValueError(
                "Argument `strict` is not supported with `method`='json_mode'"
            )
        is_pydantic_schema = _is_pydantic_class(schema)

        if method == "json_schema":
            # Check for Pydantic BaseModel V1
            if (
                is_pydantic_schema and issubclass(schema, BaseModelV1)  # type: ignore[arg-type]
            ):
                warnings.warn(
                    "Received a Pydantic BaseModel V1 schema. This is not supported by "
                    'method="json_schema". Please use method="function_calling" '
                    "or specify schema via JSON Schema or Pydantic V2 BaseModel. "
                    'Overriding to method="function_calling".'
                )
                method = "function_calling"
            # Check for incompatible model
            if self.model_name and (
                self.model_name.startswith("gpt-3")
                or self.model_name.startswith("gpt-4-")
                or self.model_name == "gpt-4"
            ):
                warnings.warn(
                    f"Cannot use method='json_schema' with model {self.model_name} "
                    f"since it doesn't support OpenAI's Structured Output API. You can "
                    f"see supported models here: "
                    f"https://platform.openai.com/docs/guides/structured-outputs#supported-models. "  # noqa: E501
                    "To fix this warning, set `method='function_calling'. "
                    "Overriding to method='function_calling'."
                )
                method = "function_calling"

        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            bind_kwargs = self._filter_disabled_params(
                tool_choice=tool_name,
                parallel_tool_calls=False,
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": method, "strict": strict},
                    "schema": schema,
                },
            )

            llm = self.bind_tools([schema], **bind_kwargs)
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            response_format = _convert_to_openai_response_format(schema, strict=strict)
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format={
                    "kwargs": {"method": method, "strict": strict},
                    "schema": convert_to_openai_tool(schema),
                },
            )
            if is_pydantic_schema:
                output_parser = RunnableLambda(
                    partial(_oai_structured_outputs_parser, schema=cast(type, schema)) # type: ignore
                ).with_types(output_type=cast(type, schema))
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
        
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def _filter_disabled_params(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.disabled_params:
            return kwargs
        filtered = {}
        for k, v in kwargs.items():
            # Skip param
            if k in self.disabled_params and (
                self.disabled_params[k] is None or v in self.disabled_params[k]
            ):
                continue
            # Keep param
            else:
                filtered[k] = v
        return filtered

    def _get_generation_chunk_from_completion(
        self, completion: openai.BaseModel
    ) -> ChatGenerationChunk:
        """Get chunk from completion (e.g., from final completion of a stream)."""
        chat_result = self._create_chat_result(completion)
        chat_message = chat_result.generations[0].message
        if isinstance(chat_message, AIMessage):
            usage_metadata = chat_message.usage_metadata
            # Skip tool_calls, already sent as chunks
            if "tool_calls" in chat_message.additional_kwargs:
                chat_message.additional_kwargs.pop("tool_calls")
        else:
            usage_metadata = None
        message = AIMessageChunk(
            content="",
            additional_kwargs=chat_message.additional_kwargs,
            usage_metadata=usage_metadata,
        )
        return ChatGenerationChunk(
            message=message, generation_info=chat_result.llm_output
        )

class ChatDashscope(BaseChatDashscope):  # type: ignore[override]
    max_tokens: Optional[int] = Field(default=None, alias="max_completion_tokens")

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}
    
    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "dashscope"]
    
    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.dashscope_organization:
            attributes["dashscope_organization"] = self.dashscope_organization  
            
        if self.dashscope_api_base:
            attributes["dashscope_api_base"] = self.dashscope_api_base

        if self.dashscope_proxy:
            attributes["dashscope_proxy"] = self.dashscope_proxy
            
        return attributes
            

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = super()._default_params
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        return params
    
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # max_tokens was deprecated in favor of max_completion_tokens
        # in September 2024 release
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        # Mutate system message role to "developer" for o-series models
        if self.model_name and re.match(r"^o\d", self.model_name):
            for message in payload.get("messages", []):
                if message["role"] == "system":
                    message["role"] = "developer"
        return payload
    
    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        else:
            return super()._stream(*args, **kwargs)
        
    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk
            
    
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        return super().with_structured_output(
            schema, method=method, include_raw=include_raw, strict=strict, **kwargs
        )

def _create_usage_metadata_responses(oai_token_usage: dict) -> UsageMetadata:
    input_tokens = oai_token_usage.get("input_tokens", 0)
    output_tokens = oai_token_usage.get("output_tokens", 0)
    total_tokens = oai_token_usage.get("total_tokens", input_tokens + output_tokens)

    output_token_details: dict = {
        "audio": (oai_token_usage.get("completion_tokens_details") or {}).get(
            "audio_tokens"
        ),
        "reasoning": (oai_token_usage.get("output_token_details") or {}).get(
            "reasoning_tokens"
        ),
    }
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        output_token_details=OutputTokenDetails(
            **{k: v for k, v in output_token_details.items() if v is not None}
        ),
    )



def _is_builtin_tool(tool: dict) -> bool:
    return "type" in tool and tool["type"] != "function"

def _use_responses_api(payload: dict) -> bool:
    uses_builtin_tools = "tools" in payload and any(
        _is_builtin_tool(tool) for tool in payload["tools"]
    )
    responses_only_args = {"previous_response_id", "text", "truncation", "include"}
    return bool(uses_builtin_tools or responses_only_args.intersection(payload))

def _convert_to_openai_response_format(
    schema: Union[Dict[str, Any], Type], *, strict: Optional[bool] = None
) -> Union[Dict, TypeBaseModel]: # type: ignore
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema

    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_function(schema, strict=strict)
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    if strict is not None and strict is not response_format["json_schema"].get(
        "strict"
    ):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to " # type: ignore
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format



def _construct_responses_api_payload(
    messages: Sequence[BaseMessage], payload: dict
) -> dict:
    # Rename legacy parameters
    for legacy_token_param in ["max_tokens", "max_completion_tokens"]:
        if legacy_token_param in payload:
            payload["max_output_tokens"] = payload.pop(legacy_token_param)
    if "reasoning_effort" in payload:
        payload["reasoning"] = {"effort": payload.pop("reasoning_effort")}

    payload["input"] = _construct_responses_api_input(messages)
    if tools := payload.pop("tools", None):
        new_tools: list = []
        for tool in tools:
            # chat api: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}, "strict": ...}}  # noqa: E501
            # responses api: {"type": "function", "name": "...", "description": "...", "parameters": {...}, "strict": ...}  # noqa: E501
            if tool["type"] == "function" and "function" in tool:
                new_tools.append({"type": "function", **tool["function"]})
            else:
                new_tools.append(tool)
        payload["tools"] = new_tools
    if tool_choice := payload.pop("tool_choice", None):
        # chat api: {"type": "function", "function": {"name": "..."}}
        # responses api: {"type": "function", "name": "..."}
        if (
            isinstance(tool_choice, dict)
            and tool_choice["type"] == "function"
            and "function" in tool_choice
        ):
            payload["tool_choice"] = {"type": "function", **tool_choice["function"]}
        else:
            payload["tool_choice"] = tool_choice

    # Structured output
    if schema := payload.pop("response_format", None):
        if payload.get("text"):
            text = payload["text"]
            raise ValueError(
                "Can specify at most one of 'response_format' or 'text', received both:"
                f"\n{schema=}\n{text=}"
            )

        # For pydantic + non-streaming case, we use responses.parse.
        # Otherwise, we use responses.create.
        strict = payload.pop("strict", None)
        if not payload.get("stream") and _is_pydantic_class(schema):
            payload["text_format"] = schema
        else:
            if _is_pydantic_class(schema):
                schema_dict = schema.model_json_schema()
                strict = True
            else:
                schema_dict = schema
            if schema_dict == {"type": "json_object"}:  # JSON mode
                payload["text"] = {"format": {"type": "json_object"}}
            elif (
                (
                    response_format := _convert_to_openai_response_format(
                        schema_dict, strict=strict
                    )
                )
                and (isinstance(response_format, dict))
                and (response_format["type"] == "json_schema")
            ):
                payload["text"] = {
                    "format": {"type": "json_schema", **response_format["json_schema"]}
                }
            else:
                pass
    return payload

def _make_computer_call_output_from_message(message: ToolMessage) -> dict:
    computer_call_output: dict = {
        "call_id": message.tool_call_id,
        "type": "computer_call_output",
    }
    if isinstance(message.content, list):
        # Use first input_image block
        output = next(
            block
            for block in message.content
            if cast(dict, block)["type"] == "input_image"
        )
    else:
        # string, assume image_url
        output = {"type": "input_image", "image_url": message.content}
    computer_call_output["output"] = output
    return computer_call_output



def _construct_responses_api_input(messages: Sequence[BaseMessage]) -> list:
    input_ = []
    for lc_msg in messages:
        msg = _convert_message_to_dict(lc_msg)
        # "name" parameter unsupported
        if "name" in msg:
            msg.pop("name")
        if msg["role"] == "tool":
            tool_output = msg["content"]
            if lc_msg.additional_kwargs.get("type") == "computer_call_output":
                computer_call_output = _make_computer_call_output_from_message(
                    cast(ToolMessage, lc_msg)
                )
                input_.append(computer_call_output)
            else:
                if not isinstance(tool_output, str):
                    tool_output = _stringify(tool_output)
                function_call_output = {
                    "type": "function_call_output",
                    "output": tool_output,
                    "call_id": msg["tool_call_id"],
                }
                input_.append(function_call_output)
        elif msg["role"] == "assistant":
            # Reasoning items
            reasoning_items = []
            if reasoning := lc_msg.additional_kwargs.get("reasoning"):
                reasoning_items.append(reasoning)
            # Function calls
            function_calls = []
            if tool_calls := msg.pop("tool_calls", None):
                # TODO: should you be able to preserve the function call object id on
                #  the langchain tool calls themselves?
                function_call_ids = lc_msg.additional_kwargs.get(
                    _FUNCTION_CALL_IDS_MAP_KEY
                )
                for tool_call in tool_calls:
                    function_call = {
                        "type": "function_call",
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                        "call_id": tool_call["id"],
                    }
                    if function_call_ids is not None and (
                        _id := function_call_ids.get(tool_call["id"])
                    ):
                        function_call["id"] = _id
                    function_calls.append(function_call)
            # Computer calls
            computer_calls = []
            tool_outputs = lc_msg.additional_kwargs.get("tool_outputs", [])
            for tool_output in tool_outputs:
                if tool_output.get("type") == "computer_call":
                    computer_calls.append(tool_output)
            msg["content"] = msg.get("content") or []
            if lc_msg.additional_kwargs.get("refusal"):
                if isinstance(msg["content"], str):
                    msg["content"] = [
                        {
                            "type": "output_text",
                            "text": msg["content"],
                            "annotations": [],
                        }
                    ]
                msg["content"] = msg["content"] + [
                    {"type": "refusal", "refusal": lc_msg.additional_kwargs["refusal"]}
                ]
            if isinstance(msg["content"], list):
                new_blocks = []
                for block in msg["content"]:
                    # chat api: {"type": "text", "text": "..."}
                    # responses api: {"type": "output_text", "text": "...", "annotations": [...]}  # noqa: E501
                    if block["type"] == "text":
                        new_blocks.append(
                            {
                                "type": "output_text",
                                "text": block["text"],
                                "annotations": block.get("annotations") or [],
                            }
                        )
                    elif block["type"] in ("output_text", "refusal"):
                        new_blocks.append(block)
                    else:
                        pass
                msg["content"] = new_blocks
            if msg["content"]:
                input_.append(msg)
            input_.extend(reasoning_items)
            input_.extend(function_calls)
            input_.extend(computer_calls)
        elif msg["role"] == "user":
            if isinstance(msg["content"], list):
                new_blocks = []
                for block in msg["content"]:
                    # chat api: {"type": "text", "text": "..."}
                    # responses api: {"type": "input_text", "text": "..."}
                    if block["type"] == "text":
                        new_blocks.append({"type": "input_text", "text": block["text"]})
                    # chat api: {"type": "image_url", "image_url": {"url": "...", "detail": "..."}}  # noqa: E501
                    # responses api: {"type": "image_url", "image_url": "...", "detail": "...", "file_id": "..."}  # noqa: E501
                    elif block["type"] == "image_url":
                        new_block = {
                            "type": "input_image",
                            "image_url": block["image_url"]["url"],
                        }
                        if block["image_url"].get("detail"):
                            new_block["detail"] = block["image_url"]["detail"]
                        new_blocks.append(new_block)
                    elif block["type"] in ("input_text", "input_image", "input_file"):
                        new_blocks.append(block)
                    else:
                        pass
                msg["content"] = new_blocks
            input_.append(msg)
        else:
            input_.append(msg)

    return input_

def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)

def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }

def _construct_lc_result_from_responses_api(
    response: Response,
    schema: Optional[Type[_BM]] = None,
    metadata: Optional[dict] = None,
) -> ChatResult:
    """Construct ChatResponse from OpenAI Response API response."""
    if response.error:
        raise ValueError(response.error)

    response_metadata = {
        k: v
        for k, v in response.model_dump(exclude_none=True, mode="json").items()
        if k
        in (
            "created_at",
            "id",
            "incomplete_details",
            "metadata",
            "object",
            "status",
            "user",
            "model",
        )
    }
    if metadata:
        response_metadata.update(metadata)
    # for compatibility with chat completion calls.
    response_metadata["model_name"] = response_metadata.get("model")
    if response.usage:
        usage_metadata = _create_usage_metadata_responses(response.usage.model_dump())
    else:
        usage_metadata = None

    content_blocks: list = []
    tool_calls = []
    invalid_tool_calls = []
    additional_kwargs: dict = {}
    msg_id = None
    for output in response.output:
        if output.type == "message":
            for content in output.content:
                if content.type == "output_text":
                    block = {
                        "type": "text",
                        "text": content.text,
                        "annotations": [
                            annotation.model_dump()
                            for annotation in content.annotations
                        ],
                    }
                    content_blocks.append(block)
                    if hasattr(content, "parsed"):
                        additional_kwargs["parsed"] = content.parsed # type: ignore
                if content.type == "refusal":
                    additional_kwargs["refusal"] = content.refusal
            msg_id = output.id
        elif output.type == "function_call":
            try:
                args = json.loads(output.arguments, strict=False)
                error = None
            except JSONDecodeError as e:
                args = output.arguments
                error = str(e)
            if error is None:
                tool_call = {
                    "type": "tool_call",
                    "name": output.name,
                    "args": args,
                    "id": output.call_id,
                }
                tool_calls.append(tool_call)
            else:
                tool_call = {
                    "type": "invalid_tool_call",
                    "name": output.name,
                    "args": args,
                    "id": output.call_id,
                    "error": error,
                }
                invalid_tool_calls.append(tool_call)
            if _FUNCTION_CALL_IDS_MAP_KEY not in additional_kwargs:
                additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {}
            additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][output.call_id] = output.id
        elif output.type == "reasoning":
            additional_kwargs["reasoning"] = output.model_dump(
                exclude_none=True, mode="json"
            )
        else:
            tool_output = output.model_dump(exclude_none=True, mode="json")
            if "tool_outputs" in additional_kwargs:
                additional_kwargs["tool_outputs"].append(tool_output)
            else:
                additional_kwargs["tool_outputs"] = [tool_output]
    # Workaround for parsing structured output in the streaming case.
    #    from openai import OpenAI
    #    from pydantic import BaseModel

    #    class Foo(BaseModel):
    #        response: str

    #    client = OpenAI()

    #    client.responses.parse(
    #        model="gpt-4o-mini",
    #        input=[{"content": "how are ya", "role": "user"}],
    #        text_format=Foo,
    #        stream=True,  # <-- errors
    #    )
    if (
        schema is not None
        and "parsed" not in additional_kwargs
        and response.output_text  # tool calls can generate empty output text
        and response.text
        and (text_config := response.text.model_dump())
        and (format_ := text_config.get("format", {}))
        and (format_.get("type") == "json_schema")
    ):
        try:
            parsed_dict = json.loads(response.output_text)
            if schema and _is_pydantic_class(schema):
                parsed = schema(**parsed_dict)
            else:
                parsed = parsed_dict
            additional_kwargs["parsed"] = parsed
        except json.JSONDecodeError:
            pass
    message = AIMessage(
        content=content_blocks,
        id=msg_id,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )
    return ChatResult(generations=[ChatGeneration(message=message)])



def _convert_responses_chunk_to_generation_chunk(
    chunk: Any, schema: Optional[Type[_BM]] = None, metadata: Optional[dict] = None
) -> Optional[ChatGenerationChunk]:
    content = []
    tool_call_chunks: list = []
    additional_kwargs: dict = {}
    if metadata:
        response_metadata = metadata
    else:
        response_metadata = {}
    usage_metadata = None
    id = None
    if chunk.type == "response.output_text.delta":
        content.append(
            {"type": "text", "text": chunk.delta, "index": chunk.content_index}
        )
    elif chunk.type == "response.output_text.annotation.added":
        content.append(
            {
                "annotations": [
                    chunk.annotation.model_dump(exclude_none=True, mode="json")
                ],
                "index": chunk.content_index,
            }
        )
    elif chunk.type == "response.created":
        response_metadata["id"] = chunk.response.id
    elif chunk.type == "response.completed":
        msg = cast(
            AIMessage,
            (
                _construct_lc_result_from_responses_api(chunk.response, schema=schema)
                .generations[0]
                .message
            ),
        )
        if parsed := msg.additional_kwargs.get("parsed"):
            additional_kwargs["parsed"] = parsed
        if reasoning := msg.additional_kwargs.get("reasoning"):
            additional_kwargs["reasoning"] = reasoning
        usage_metadata = msg.usage_metadata
        response_metadata = {
            k: v for k, v in msg.response_metadata.items() if k != "id"
        }
    elif chunk.type == "response.output_item.added" and chunk.item.type == "message":
        id = chunk.item.id
    elif (
        chunk.type == "response.output_item.added"
        and chunk.item.type == "function_call"
    ):
        tool_call_chunks.append(
            {
                "type": "tool_call_chunk",
                "name": chunk.item.name,
                "args": chunk.item.arguments,
                "id": chunk.item.call_id,
                "index": chunk.output_index,
            }
        )
        additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {
            chunk.item.call_id: chunk.item.id
        }
    elif chunk.type == "response.output_item.done" and chunk.item.type in (
        "web_search_call",
        "file_search_call",
        "computer_call",
    ):
        additional_kwargs["tool_outputs"] = [
            chunk.item.model_dump(exclude_none=True, mode="json")
        ]
    elif chunk.type == "response.function_call_arguments.delta":
        tool_call_chunks.append(
            {
                "type": "tool_call_chunk",
                "args": chunk.delta,
                "index": chunk.output_index,
            }
        )
    elif chunk.type == "response.refusal.done":
        additional_kwargs["refusal"] = chunk.refusal
    else:
        return None

    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=content,  # type: ignore[arg-type]
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
            additional_kwargs=additional_kwargs,
            id=id,
        )
    )

def _url_to_size(image_source: str) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        logger.info(
            "Unable to count image tokens. To count image tokens please install "
            "`pip install -U pillow httpx`."
        )
        return None
    if _is_url(image_source):
        try:
            import httpx
        except ImportError:
            logger.info(
                "Unable to count image tokens. To count image tokens please install "
                "`pip install -U httpx`."
            )
            return None
        response = httpx.get(image_source)
        response.raise_for_status()
        width, height = Image.open(BytesIO(response.content)).size
        return width, height
    elif _is_b64(image_source):
        _, encoded = image_source.split(",", 1)
        data = base64.b64decode(encoded)
        width, height = Image.open(BytesIO(data)).size
        return width, height
    else:
        return None


def _count_image_tokens(width: int, height: int) -> int:
    # Reference: https://platform.openai.com/docs/guides/vision/calculating-costs
    width, height = _resize(width, height)
    h = ceil(height / 512)
    w = ceil(width / 512)
    return (170 * h * w) + 85

def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _resize(width: int, height: int) -> Tuple[int, int]:
    # larger side must be <= 2048
    if width > 2048 or height > 2048:
        if width > height:
            height = (height * 2048) // width
            width = 2048
        else:
            width = (width * 2048) // height
            height = 2048
    # smaller side must be <= 768
    if width > 768 and height > 768:
        if width > height:
            width = (width * 768) // height
            height = 768
        else:
            height = (width * 768) // height
            width = 768
    return width, height

def _oai_structured_outputs_parser(
    ai_msg: AIMessage, schema: Type[_BM]
) -> PydanticBaseModel: # type: ignore
    if parsed := ai_msg.additional_kwargs.get("parsed"):
        if isinstance(parsed, dict):
            return schema(**parsed)
        else:
            return parsed
    elif ai_msg.additional_kwargs.get("refusal"):
        raise OpenAIRefusalError(ai_msg.additional_kwargs["refusal"])
    else:
        raise ValueError(
            "Structured Output response does not have a 'parsed' field nor a 'refusal' "
            f"field. Received message:\n\n{ai_msg}"
        )

class OpenAIRefusalError(Exception):
    """Error raised when OpenAI Structured Outputs API returns a refusal.

    When using OpenAI's Structured Outputs API with user-generated input, the model
    may occasionally refuse to fulfill the request for safety reasons.

    See here for more on refusals:
    https://platform.openai.com/docs/guides/structured-outputs/refusals

    .. versionadded:: 0.1.21
    """
