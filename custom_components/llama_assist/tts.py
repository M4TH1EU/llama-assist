"""Support for Wyoming text-to-speech services."""

import io
import logging
import wave
from collections import defaultdict
from typing import Any

from homeassistant.components import tts
from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from openai import AsyncOpenAI

from custom_components.llama_assist import LlamaAPIClientsConfigEntry
from custom_components.llama_assist.const import CONF_TTS_MODEL

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        config_entry: LlamaAPIClientsConfigEntry,
        async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    agent = LlamaAssistTtsProvider(config_entry)
    async_add_entities([agent])


class LlamaAssistTtsProvider(TextToSpeechEntity):
    """Wyoming text-to-speech provider."""

    def __init__(self, entry: LlamaAPIClientsConfigEntry) -> None:
        """Set up provider."""
        self.entry = entry
        self._attr_name = entry.title
        self._attr_unique_id = f"{entry.entry_id}_{Platform.TTS}"

        voice_languages: set[str] = set()
        self._voices: dict[str, list[tts.Voice]] = defaultdict(list)

        voice_languages.add("en")
        self._voices["en"].extend([
            tts.Voice(voice_id="default", name="Default Voice"),
            tts.Voice(voice_id="af_bella+af_sky", name="Bella+Sky Voice"),
        ])
        self._supported_languages: list[str] = list(voice_languages)

        apis = self.entry.runtime_data
        self.tts_client: AsyncOpenAI | None = apis.tts_client

        # self.service = service
        # self._tts_service = next(tts for tts in service.info.tts if tts.installed)

        # for voice in self._tts_service.voices:
        #     if not voice.installed:
        #         continue
        #
        #     voice_languages.update(voice.languages)
        #     for language in voice.languages:
        #         self._voices[language].append(
        #             tts.Voice(
        #                 voice_id=voice.name,
        #                 name=voice.description or voice.name,
        #             )
        #         )

        # Sort voices by name
        # for language in self._voices:
        #     self._voices[language] = sorted(
        #         self._voices[language], key=lambda v: v.name
        #     )

        # self._attr_name = self._tts_service.name

    @property
    def default_language(self):
        """Return default language."""
        if not self._supported_languages:
            return None

        return self._supported_languages[0]

    @property
    def supported_languages(self):
        """Return list of supported languages."""
        return self._supported_languages

    @property
    def supported_options(self):
        """Return list of supported options like voice, emotion."""
        return [
            tts.ATTR_AUDIO_OUTPUT,
            tts.ATTR_VOICE,
            # ATTR_SPEAKER,
        ]

    @property
    def default_options(self):
        """Return a dict include default options."""
        return {}

    @callback
    def async_get_supported_voices(self, language: str) -> list[tts.Voice] | None:
        """Return a list of supported voices for a language."""
        return self._voices.get(language)

    async def async_get_tts_audio(self, message: str, language: str, options: dict[str, Any]) -> TtsAudioType:
        """Stream TTS audio from OpenAI backend and return as WAV."""
        if not self.tts_client:
            _LOGGER.error("No TTS client configured")
            return None, None

        voice = options.get("voice") if options else None
        model = self.entry.data.get(CONF_TTS_MODEL)

        try:
            _LOGGER.debug("Sending TTS request to OpenAI backend (voice: %s)", voice)

            # OpenAI's streaming response gives us raw PCM data
            with self.tts_client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=message,
                    response_format="pcm"  # PCM 16-bit, 24kHz, mono
            ) as response:
                pcm_data = b""
                async for chunk in response.iter_bytes(chunk_size=1024):
                    pcm_data += chunk

            # Convert PCM to WAV in memory (24kHz, 16-bit, mono)
            with io.BytesIO() as wav_io:
                wav_writer = wave.open(wav_io, "wb")
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)  # 16-bit
                wav_writer.setframerate(24000)
                wav_writer.writeframes(pcm_data)
                wav_writer.close()
                data = wav_io.getvalue()

            _LOGGER.debug("TTS audio successfully received and converted to WAV")
            return "wav", data

        except Exception as e:
            _LOGGER.exception("Failed to generate TTS audio via OpenAI: %s", e)
            return None, None
