"""The Llama Assist integration."""

from __future__ import annotations

from pathlib import Path

import openai
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import llm as ha_llm
from homeassistant.helpers.httpx_client import get_async_client
from posthog.ai.openai import AsyncOpenAI

from .const import DOMAIN, HEALTHCHECK_TIMEOUT, LLAMA_LLM_API, CONF_BLACKLIST_TOOLS, CONF_SERVER_EMBEDDINGS_URL, \
    CONF_CONVERSATION_SERVER_URL, CONF_EMBEDDINGS_TOOLS, DEFAULT_EMBEDDINGS_TOOLS, PLATFORMS, EMBEDDINGS_SQLITE, \
    DEFAULT_EMBEDDINGS_OVERWRITE, CONF_EMBEDDINGS_OVERWRITE, CONF_TTS_SERVER_URL, CONF_ACTION_DEVICE_TYPE, \
    CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS
from .embeddings import EmbeddingsDatabase
from .llm import LlamaAssistAPI

type LlamaAPIClientsConfigEntry = ConfigEntry[LlamaAPIClients]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Llama Assist integration."""
    settings = {**entry.data, **entry.options}

    engine_type = settings.get(CONF_ACTION_DEVICE_TYPE)

    if engine_type not in PLATFORMS:
        raise ConfigEntryNotReady(
            f"Unsupported engine type '{engine_type}'. Supported types: {', '.join(PLATFORMS)}"
        )

    if engine_type == Platform.CONVERSATION:
        if not any([x.id == LLAMA_LLM_API for x in ha_llm.async_get_apis(hass)]):
            ha_llm.async_register_api(hass, LlamaAssistAPI(hass))

        completion_client = openai.AsyncOpenAI(
            api_key="none",
            base_url=settings.get(CONF_CONVERSATION_SERVER_URL, ""),
            http_client=get_async_client(hass)
        )

        use_embeds = settings.get(CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS)
        embeddings_client = None
        embeddings_db = None
        if use_embeds:
            embeddings_client = openai.AsyncOpenAI(
                api_key="none",
                base_url=settings.get(CONF_SERVER_EMBEDDINGS_URL, ""),
                http_client=get_async_client(hass)
            )

            db_path = Path(hass.config.path(DOMAIN + "/" + EMBEDDINGS_SQLITE))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            embeddings_db = EmbeddingsDatabase(
                embeddings_client,
                db_path=str(db_path),
                overwrite=settings.get(CONF_EMBEDDINGS_OVERWRITE, DEFAULT_EMBEDDINGS_OVERWRITE),
                blacklist_tools=settings.get(CONF_BLACKLIST_TOOLS, []),
            )

        apis = LlamaAPIClients(
            completion_client=completion_client,
            embeddings_client=embeddings_client,
            embeddings_db=embeddings_db,
        )
        entry.runtime_data = apis
        await hass.config_entries.async_forward_entry_setups(entry, (Platform.CONVERSATION,))

    elif engine_type == Platform.TTS:
        tts_client = openai.AsyncOpenAI(
            api_key="none",
            base_url=settings.get(CONF_TTS_SERVER_URL, ""),
            http_client=get_async_client(hass)
        )

        apis = LlamaAPIClients(
            tts_client=tts_client,
        )
        entry.runtime_data = apis

        await hass.config_entries.async_forward_entry_setups(entry, (Platform.TTS,))

    # for client in [completion_client, embeddings_client, tts_client]:
    #     if not client:
    #         continue
    #     if not client.base_url:
    #         raise ConfigEntryNotReady(
    #             f"Missing required configuration for {client.__class__.__name__}: base_url"
    #         )
    #
    #     try:
    #         await hass.async_add_executor_job(client.with_options(timeout=HEALTHCHECK_TIMEOUT).models.list)
    #     except openai.OpenAIError as err:
    #         raise ConfigEntryNotReady(
    #             f"Failed to connect to {client.__class__.__name__} at {client.base_url}: {err}"
    #         ) from err

    return True


async def async_unload_entry(hass: HomeAssistant, entry: LlamaAPIClientsConfigEntry) -> bool:
    """Unload Ollama."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    hass.data[DOMAIN].pop(entry.entry_id)  # TODO: fix unload
    return True


class LlamaAPIClients:
    """A class to hold the API clients for Llama Assist."""

    def __init__(
            self,
            completion_client: AsyncOpenAI | None = None,
            embeddings_client: AsyncOpenAI | None = None,
            embeddings_db: EmbeddingsDatabase | None = None,
            tts_client: AsyncOpenAI | None = None,
    ):
        self.completion_client = completion_client
        self.embeddings_client = embeddings_client
        self.tts_client = tts_client
        self.embeddings_db = embeddings_db
