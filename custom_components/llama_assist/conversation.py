import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, cast

import openai
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import AbstractConversationAgent, ConversationEntityFeature, SystemContent
from homeassistant.components.conversation import ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import yaml as yaml_util
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
from voluptuous_openapi import convert

from . import DOMAIN, LlamaAssistAPI, LlamaAPIClientsConfigEntry
from .const import CONF_PROMPT, LLAMA_LLM_API, \
    CONF_EMBEDDINGS_TOOLS, LOGGER, MAX_TOOL_ITERATIONS, CONF_EMBEDDINGS_ENTITIES
from .embeddings import EmbeddingsDatabase


async def async_setup_entry(hass: HomeAssistant, config_entry: LlamaAPIClientsConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up conversation entities."""
    agent = LlamaConversationEntity(config_entry)
    async_add_entities([agent])



def _format_tool(
        tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _convert_content_to_param(
        content: conversation.Content,
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""
    if content.role == "tool_result":
        assert type(content) is conversation.ToolResultContent
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )
    if content.role != "assistant" or not content.tool_calls:  # type: ignore[union-attr]
        role = content.role
        if role == "system":
            role = "developer"
        return cast(
            ChatCompletionMessageParam,
            {"role": content.role, "content": content.content},  # type: ignore[union-attr]
        )

    # Handle the Assistant content including tool calls.
    assert type(content) is conversation.AssistantContent
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=content.content,
        tool_calls=[
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=Function(
                    arguments=json.dumps(tool_call.tool_args),
                    name=tool_call.tool_name,
                ),
                type="function",
            )
            for tool_call in content.tool_calls
        ],
    )


async def _transform_stream(
        result: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    current_tool_call: dict | None = None
    async for chunk in result:
        LOGGER.debug("Received chunk: %s", chunk)
        choice = chunk.choices[0]

        if choice.finish_reason:
            if current_tool_call:
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=current_tool_call["id"],
                            tool_name=current_tool_call["tool_name"],
                            tool_args=json.loads(current_tool_call["tool_args"]),
                        )
                    ]
                }

            break

        delta = chunk.choices[0].delta

        # We can yield delta messages not continuing or starting tool calls
        if current_tool_call is None and not delta.tool_calls:
            yield {  # type: ignore[misc]
                key: value
                for key in ("role", "content")
                if (value := getattr(delta, key)) is not None
            }
            continue

        # When doing tool calls, we should always have a tool call
        # object or we have gotten stopped above with a finish_reason set.
        if (
                not delta.tool_calls
                or not (delta_tool_call := delta.tool_calls[0])
                or not delta_tool_call.function
        ):
            raise ValueError("Expected delta with tool call")

        if current_tool_call and delta_tool_call.index == current_tool_call["index"]:
            current_tool_call["tool_args"] += delta_tool_call.function.arguments or ""
            continue

        # We got tool call with new index, so we need to yield the previous
        if current_tool_call:
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call["id"],
                        tool_name=current_tool_call["tool_name"],
                        tool_args=json.loads(current_tool_call["tool_args"]),
                    )
                ]
            }

        current_tool_call = {
            "index": delta_tool_call.index,
            "id": delta_tool_call.id,
            "tool_name": delta_tool_call.function.name,
            "tool_args": delta_tool_call.function.arguments or "",
        }


class LlamaConversationEntity(ConversationEntity, AbstractConversationAgent):
    """Llama Assist conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: LlamaAPIClientsConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_name = entry.title
        self._attr_unique_id = f"{entry.entry_id}_{Platform.CONVERSATION}"

        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="LlamaAssist",
            model="LocalLLM",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        apis = self.entry.runtime_data
        self.completion_client: AsyncOpenAI = apis.completion_client
        self.embeddings_client: AsyncOpenAI | None = apis.embeddings_client
        self.embeddings_db: EmbeddingsDatabase | None = apis.embeddings_db

        # TODO: investigate if we can add features here to access from llm context (see pipeline.py)
        self._attr_supported_features = (
            ConversationEntityFeature.CONTROL
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        settings = {**self.entry.data, **self.entry.options}

        try:
            await chat_log.async_update_llm_data(
                conversing_domain=DOMAIN,
                user_input=user_input,
                # user_llm_hass_api=settings.get(CONF_LLM_HASS_API),
                user_llm_hass_api=LLAMA_LLM_API,
                user_llm_prompt=settings.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools_to_use: list[llm.Tool] = []
        if settings.get(CONF_EMBEDDINGS_ENTITIES) or settings.get(CONF_EMBEDDINGS_TOOLS):
            # If using embeddings, we need to get the user input vector
            user_input_vector = await self.embeddings_client.embeddings.create(input=user_input.text, model="none",
                                                                               encoding_format="float")

            if user_input_vector:
                user_input_vector = user_input_vector[0].model_extra.get('embedding')[0]

                if settings.get(CONF_EMBEDDINGS_TOOLS):
                    await self.embeddings_db.store_tools(chat_log.llm_api.tools)
                    tools_to_use = await self.embeddings_db.matching_tools(user_input=user_input_vector)

                if settings.get(CONF_EMBEDDINGS_ENTITIES):
                    # If using embeddings entities, we need to get the relevant entities and add them to the chat log for the LLM
                    if isinstance(chat_log.llm_api.api, LlamaAssistAPI):
                        _all_exposed_entities = chat_log.llm_api.api.all_exposed_entities

                        await self.embeddings_db.store_entities(_all_exposed_entities)
                        matching_entities = await self.embeddings_db.matching_entities(user_input=user_input_vector)

                        prompt = []

                        if matching_entities:
                            prompt.append(
                                "Static Context Update:"
                            )
                            prompt.append(yaml_util.dump(list(matching_entities.values())))

                        chat_log.content.insert(
                            len(chat_log.content) - 1,
                            SystemContent(content="\n".join(prompt))
                        )
            else:
                LOGGER.warning("No user input vector found, not using embeddings tools or entities.")

        await self._async_handle_chat_log(chat_log, tools_to_use)

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_handle_chat_log(
            self,
            chat_log: conversation.ChatLog,
            tools: list[llm.Tool],
    ) -> None:
        """Generate an answer for the chat log."""

        # Convert tools to FunctionToolParam format
        tools = [
            _format_tool(tool, chat_log.llm_api.custom_serializer)
            for tool in tools
        ]

        # Convert chat log content to OpenAI message format
        messages = [_convert_content_to_param(content) for content in chat_log.content]

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": "none",
                "messages": messages,
                # "max_output_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                # "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                # "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "stream": True,
            }
            if tools:
                model_args["tools"] = tools

            # if model.startswith("o"):
            #     model_args["reasoning"] = {
            #         "effort": options.get(
            #             CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
            #         )
            #     }
            # else:
            #     model_args["store"] = False
            model_args["store"] = False

            try:
                result = await self.completion_client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            messages.extend(
                [
                    _convert_content_to_param(content)
                    async for content in chat_log.async_add_delta_content_stream(
                    self.entity_id, _transform_stream(result)
                )
                ]
            )

            if not chat_log.unresponded_tool_results:
                break

    # def _trim_history(self, message_history: MessageHistory, max_messages: int) -> None:
    #     """Trims excess messages from a single history.
    #
    #     This sets the max history to allow a configurable size history may take
    #     up in the context window.
    #
    #     Note that some messages in the history may not be from ollama only, and
    #     may come from other anents, so the assumptions here may not strictly hold,
    #     but generally should be effective.
    #     """
    #     if max_messages < 1:
    #         # Keep all messages
    #         return
    #
    #     # Ignore the in progress user message
    #     num_previous_rounds = message_history.num_user_messages - 1
    #     if num_previous_rounds >= max_messages:
    #         # Trim history but keep system prompt (first message).
    #         # Every other message should be an assistant message, so keep 2x
    #         # message objects. Also keep the last in progress user message
    #         num_keep = 2 * max_messages + 1
    #         drop_index = len(message_history.messages) - num_keep
    #         message_history.messages = [message_history.messages[0]] + message_history.messages[drop_index:]

    async def _async_entry_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)

# def _parse_reasoning_message(msg: str) -> str:
#     msg = msg.replace("\n\n", "")
#
#     if "<think>" in msg and "</think>" in msg:
#         # Extract the thought content
#         return msg.split("</think>")[1].strip()
#     return msg
