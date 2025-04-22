"""Dashscope chat wrapper."""
from __future__ import annotations

import logging
import os
import re
import ssl
import warnings
from operator import itemgetter
from typing import (
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
    TypeVar,
    Union,
    cast,
)

import certifi
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
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import (
    UsageMetadata,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    is_basemodel_subclass,
)
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from dashscope.api_entities.dashscope_response import (GenerationResponse,
                                                       Message)

from dashscope import (AioGeneration, Generation)


logger = logging.getLogger(__name__)

global_ssl_context = ssl.create_default_context(cafile=certifi.where())

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]

class BaseChatDashscope(BaseChatModel):
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

        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
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
    
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        
        messages = self._convert_to_dashscope_message(self._convert_input(input_).to_messages())
        
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}
        payload["messages"] = messages
        return payload
    
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

        if stream_usage is not None:
            warnings.warn("stream_usage is not supported in Dashscope. "
                          "Token usage will be included in the response by default."
            )

        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        if "response_format" in payload:
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "_stream: JSON Schema response format is not supported in Dashscope."
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

        if "response_format" in payload:
            payload.pop("stream")
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "_generate: JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )
            
        response = Generation.call(**payload)

        return self._create_chat_result(cast(GenerationResponse, response))

    def _create_chat_result(
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

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        logger.info("astream")
        kwargs["stream"] = True

        if stream_usage is not None:
            warnings.warn("stream_usage is not supported in Dashscope. "
                          "Token usage will be included in the response by default."
            )

        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        if "response_format" in payload:
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "_astream: JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )

        # AioGeneration.call 返回异步生成器
        response_stream = await AioGeneration.call(**payload)

        # 处理异步生成器
        async for response in response_stream: # type: ignore
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

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            logger.info("调用streaming模式 ================================")
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        payload.pop("stream")

        if "response_format" in payload:
            if payload["response_format"]["type"] == "json_schema":
                warnings.warn(
                    "_agenerate: JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )

        logger.info("调用AioGeneration.call ================================")
        response_generator = await AioGeneration.call(**payload)

        # 处理异步生成器，获取最后一个响应作为完整响应
        final_response = None
        async for response in response_generator: # type: ignore
            final_response = response
        
        if final_response is None:
            raise ValueError("没有从生成器中获取到响应")
        
        logger.info("处理完成的响应 ================================")
        return self._create_chat_result(cast(GenerationResponse, final_response))

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
            ls_provider="dashscope",
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
        return "dashscope-chat"

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
                    "code_interpreter",
                    "wanx",
                    "search",
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
            warnings.warn(
                    "_with_structured_output: JSON Schema response format is not supported in Dashscope."
                    "You can clearly describe the key-value structure and data types of the required JSON in the prompt, and provide standard data examples."
                    "This will help the model achieve similar results."
                )
            method = "function_calling"

        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            tool_name = convert_to_dashscope_tool(schema)["function"]["name"]
            print("_with_structured_output: tool_name", tool_name)
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

def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)

def convert_to_dashscope_tool(
    tool: Union[dict[str, Any], type[BaseModel], Callable, BaseTool],
    *,
    strict: Optional[bool] = None,
) -> dict[str, Any]:
    if isinstance(tool, dict):
        if tool.get("type") in ("function", "search", "wanx", "code_interpreter"):
            return tool
    oai_function = convert_to_openai_function(tool, strict=strict)
    return {"type": "function", "function": oai_function}

