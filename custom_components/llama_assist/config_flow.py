"""Config flow for Llama Assist integration."""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

import openai
import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigEntry, OptionsFlow
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import TextSelector, TextSelectorConfig, TextSelectorType, SelectOptionDict, \
    TemplateSelector, SelectSelector, SelectSelectorConfig, NumberSelector, NumberSelectorConfig, NumberSelectorMode

from . import HEALTHCHECK_TIMEOUT
from .const import DOMAIN, CONF_PROMPT, CONF_CONVERSATION_MAX_HISTORY, DEFAULT_CONVERSATION_MAX_HISTORY, \
    CONF_CONVERSATION_DISABLE_REASONING, EXISTING_TOOLS, CONF_BLACKLIST_TOOLS, CONF_EMBEDDINGS_TOOLS, \
    CONF_EMBEDDINGS_ENTITIES, CONF_SERVER_EMBEDDINGS_URL, CONF_CONVERSATION_SERVER_URL, CONF_EMBEDDINGS_OVERWRITE, \
    CONF_TTS_SERVER_URL, CONF_TTS_MODEL, CONF_TTS_VOICE, CONF_CONVERSATION_API_KEY, \
    DEFAULT_CONVERSATION_API_KEY, CONF_EMBEDDINGS_API_KEY, DEFAULT_EMBEDDINGS_API_KEY, CONF_TTS_API_KEY, \
    DEFAULT_TTS_API_KEY, CONF_ENTITY_PREFIX, CONF_ACTION_DEVICE_TYPE, DEFAULT_CONVERSATION_WANT_TO_USE_EMBEDDINGS, \
    CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS, DEFAULT_TTS_MODEL, DEFAULT_TTS_VOICE, \
    DEFAULT_CONVERSATION_DISABLE_REASONING, DEFAULT_EMBEDDINGS_OVERWRITE, DEFAULT_EMBEDDINGS_TOOLS, \
    DEFAULT_EMBEDDINGS_ENTITIES

_LOGGER = logging.getLogger(__name__)


def get_user_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the user schema for the flows."""
    return vol.Schema({
        vol.Required(CONF_ENTITY_PREFIX,
                     default=user_input.get(CONF_ENTITY_PREFIX) if user_input else None): TextSelector(
            TextSelectorConfig(type=TextSelectorType.TEXT)
        ),
        vol.Required(CONF_ACTION_DEVICE_TYPE,
                     default=user_input.get(CONF_ACTION_DEVICE_TYPE) if user_input else None): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(label="Conversation engine", value="conversation"),
                    SelectOptionDict(label="TTS engine", value="tts"),
                ],
                multiple=False
            )
        )
    })


def get_conversation_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the conversation schema for the flows."""
    return vol.Schema({
        vol.Required(CONF_CONVERSATION_SERVER_URL,
                     default=user_input.get(CONF_CONVERSATION_SERVER_URL) if user_input else None): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_CONVERSATION_API_KEY, default=user_input.get(
            CONF_CONVERSATION_API_KEY) if user_input else DEFAULT_CONVERSATION_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_PROMPT, default=user_input.get(
            CONF_PROMPT) if user_input else llm.DEFAULT_INSTRUCTIONS_PROMPT): TemplateSelector(),
        vol.Optional(CONF_CONVERSATION_DISABLE_REASONING, default=user_input.get(
            CONF_CONVERSATION_DISABLE_REASONING) if user_input else DEFAULT_CONVERSATION_DISABLE_REASONING): bool,
        vol.Optional(CONF_CONVERSATION_MAX_HISTORY, default=user_input.get(
            CONF_CONVERSATION_MAX_HISTORY) if user_input else DEFAULT_CONVERSATION_MAX_HISTORY): NumberSelector(
            NumberSelectorConfig(
                min=0, max=sys.maxsize, step=1, mode=NumberSelectorMode.BOX
            )
        ),
        vol.Optional(CONF_BLACKLIST_TOOLS,
                     default=user_input.get(CONF_BLACKLIST_TOOLS) if user_input else []): SelectSelector(
            SelectSelectorConfig(
                options=[SelectOptionDict(label=tool, value=tool) for tool in EXISTING_TOOLS],
                multiple=True
            )
        ),
        vol.Optional(CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS,
                     default=user_input.get(
                         CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS) if user_input else DEFAULT_CONVERSATION_WANT_TO_USE_EMBEDDINGS): bool,
    })


def get_embeddings_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the embeddings schema for the flows."""
    return vol.Schema({
        vol.Required(CONF_SERVER_EMBEDDINGS_URL,
                     default=user_input.get(CONF_SERVER_EMBEDDINGS_URL) if user_input else None): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_EMBEDDINGS_API_KEY, default=user_input.get(
            CONF_EMBEDDINGS_API_KEY) if user_input else DEFAULT_EMBEDDINGS_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_EMBEDDINGS_OVERWRITE, default=user_input.get(
            CONF_EMBEDDINGS_OVERWRITE) if user_input else DEFAULT_EMBEDDINGS_OVERWRITE): bool,
        vol.Optional(CONF_EMBEDDINGS_TOOLS,
                     default=user_input.get(CONF_EMBEDDINGS_TOOLS) if user_input else DEFAULT_EMBEDDINGS_TOOLS): bool,
        vol.Optional(CONF_EMBEDDINGS_ENTITIES, default=user_input.get(
            CONF_EMBEDDINGS_ENTITIES) if user_input else DEFAULT_EMBEDDINGS_ENTITIES): bool,
    })


def get_tts_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the TTS schema for the flows."""
    return vol.Schema({
        vol.Required(CONF_TTS_SERVER_URL,
                     default=user_input.get(CONF_TTS_SERVER_URL) if user_input else None): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_TTS_API_KEY,
                     default=user_input.get(CONF_TTS_API_KEY) if user_input else DEFAULT_TTS_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_TTS_MODEL,
                     default=user_input.get(CONF_TTS_MODEL) if user_input else DEFAULT_TTS_MODEL): TextSelector(),
        vol.Optional(CONF_TTS_VOICE,
                     default=user_input.get(CONF_TTS_VOICE) if user_input else DEFAULT_TTS_VOICE): TextSelector(),
    })


async def validate_openai_connection(hass: HomeAssistant, api_key: str, base_url: str) -> None:
    """Validate the OpenAI connection with the provided API key and base URL."""
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=get_async_client(hass)
    )
    await client.with_options(timeout=HEALTHCHECK_TIMEOUT).models.list()


class LlamaAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Llama Assist."""

    VERSION = 1
    CURRENT_STEP = "user"

    data: Optional[Dict[str, Any]]

    async def async_step_tts(self, user_input: dict[str, Any] | None = None):
        """Handle the TTS step."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            url = user_input.get(CONF_TTS_SERVER_URL, "")
            api_key = user_input.get(CONF_TTS_API_KEY, DEFAULT_TTS_API_KEY)

            if not url:
                errors["base"] = "missing_url"
            else:
                try:
                    await validate_openai_connection(self.hass, api_key, url)
                except openai.APIConnectionError:
                    errors["base"] = "cannot_connect"
                except openai.AuthenticationError:
                    errors["base"] = "invalid_auth"
                except Exception:
                    _LOGGER.exception("Unexpected exception")
                    errors["base"] = "unknown"

            if not errors:
                # Input for TTS step is valid
                self.data.update(user_input)

                return self.async_create_entry(
                    title=f"{self.data[CONF_ENTITY_PREFIX]}-{self.data[CONF_ACTION_DEVICE_TYPE]}",
                    data=self.data
                )

        return self.async_show_form(
            step_id="tts",
            data_schema=get_tts_schema(),
            errors=errors
        )

    async def async_step_conversation(self, user_input: dict[str, Any] | None = None):
        """Handle the conversation step."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            url = user_input.get(CONF_CONVERSATION_SERVER_URL, "")
            api_key = user_input.get(CONF_CONVERSATION_API_KEY, DEFAULT_CONVERSATION_API_KEY)
            want_to_use_embeds = user_input.get(CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS,
                                                DEFAULT_CONVERSATION_WANT_TO_USE_EMBEDDINGS)

            if not url:
                errors["base"] = "missing_url"
            else:
                try:
                    await validate_openai_connection(self.hass, api_key, url)
                except openai.APIConnectionError:
                    errors["base"] = "cannot_connect"
                except openai.AuthenticationError:
                    errors["base"] = "invalid_auth"
                except Exception:
                    _LOGGER.exception("Unexpected exception")
                    errors["base"] = "unknown"

            if not errors:
                # Input for conversation step is valid
                self.data.update(user_input)

                if want_to_use_embeds:
                    return await self.async_step_embeddings()
                else:
                    return self.async_create_entry(
                        title=f"{self.data[CONF_ENTITY_PREFIX]}-{self.data[CONF_ACTION_DEVICE_TYPE]}",
                        data=self.data
                    )

        return self.async_show_form(
            step_id="conversation",
            data_schema=get_conversation_schema(),
            errors=errors
        )

    async def async_step_embeddings(self, user_input: dict[str, Any] | None = None):
        """Handle the embeddings step."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            url = user_input.get(CONF_SERVER_EMBEDDINGS_URL, "")
            api_key = user_input.get(CONF_EMBEDDINGS_API_KEY, DEFAULT_EMBEDDINGS_API_KEY)
            if not url:
                errors["base"] = "missing_url"
            else:
                try:
                    await validate_openai_connection(self.hass, api_key, url)
                except openai.APIConnectionError:
                    errors["base"] = "cannot_connect"
                except openai.AuthenticationError:
                    errors["base"] = "invalid_auth"
                except Exception:
                    _LOGGER.exception("Unexpected exception")
                    errors["base"] = "unknown"

            if not errors:
                # Input for embeddings step is valid
                self.data.update(user_input)

                return self.async_create_entry(
                    title=f"{self.data[CONF_ENTITY_PREFIX]}-{self.data[CONF_ACTION_DEVICE_TYPE]}",
                    data=self.data
                )

        return self.async_show_form(
            step_id="embeddings",
            data_schema=get_embeddings_schema(),
            errors=errors
        )

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            prefix = user_input.get(CONF_ENTITY_PREFIX, "")
            engine_type = user_input.get(CONF_ACTION_DEVICE_TYPE)

            if not prefix:
                errors["base"] = "missing_prefix"
            else:
                registry = entity_registry.async_get(self.hass)

                exists = registry.async_get_entity_id(DOMAIN, engine_type, f"{prefix}_{engine_type}")
                if exists:
                    errors["base"] = "entity_exists"

            if not errors:
                self.data = user_input

                if engine_type == "conversation":
                    # Proceed to conversation step for conversation engine
                    return await self.async_step_conversation()
                elif engine_type == "tts":
                    # Proceed to TTS step for TTS engine
                    return await self.async_step_tts()

        return self.async_show_form(
            step_id="user",
            data_schema=get_user_schema(),
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return LlamaAssistOptionsFlow(config_entry)


class LlamaAssistOptionsFlow(OptionsFlow):
    """Ollama options flow."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.engine_type = config_entry.data.get(CONF_ACTION_DEVICE_TYPE)
        self.settings = {**config_entry.data, **config_entry.options}

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if self.engine_type == Platform.CONVERSATION:
            return self.async_show_menu(
                step_id="init",

                menu_options=["conversation", "embeddings"],
            )
        elif self.engine_type == Platform.TTS:
            return self.async_show_form(
                step_id="tts",
                data_schema=get_tts_schema(user_input=user_input),
            )
        else:
            raise ValueError(
                f"Unsupported engine type '{self.engine_type}'. Supported types: {', '.join(Platform)}"
            )

    async def async_step_conversation(self, user_input=None):
        """Manage the conversation options."""
        errors: Dict[str, str] = {}
        self.settings.update(user_input or {})

        if user_input is not None:
            # Validate the user input
            prompt = user_input.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)
            max_history = user_input.get(CONF_CONVERSATION_MAX_HISTORY, DEFAULT_CONVERSATION_MAX_HISTORY)
            disable_reasoning = user_input.get(CONF_CONVERSATION_DISABLE_REASONING,
                                               DEFAULT_CONVERSATION_DISABLE_REASONING)
            blacklist_tools = user_input.get(CONF_BLACKLIST_TOOLS, [])
            want_to_use_embeddings = user_input.get(CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS,
                                                    DEFAULT_CONVERSATION_WANT_TO_USE_EMBEDDINGS)

            if not prompt:
                errors["base"] = "missing_prompt"
            elif max_history < 0:
                errors["base"] = "invalid_max_history"

            if not errors:
                # Input for conversation options is valid
                data_new = self.config_entry.data.copy()
                data_new.update({
                    CONF_PROMPT: prompt,
                    CONF_CONVERSATION_MAX_HISTORY: max_history,
                    CONF_CONVERSATION_DISABLE_REASONING: disable_reasoning,
                    CONF_BLACKLIST_TOOLS: blacklist_tools,
                    CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS: want_to_use_embeddings,
                })

                return self.async_create_entry(
                    title=self.config_entry.title,
                    data=data_new
                )

        return self.async_show_form(
            step_id="conversation",
            data_schema=get_conversation_schema(user_input=self.settings),
            errors=errors
        )
#
