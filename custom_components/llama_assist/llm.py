from functools import cache, partial

import slugify as unicode_slug
from homeassistant.components.calendar import (
    DOMAIN as CALENDAR_DOMAIN,
)
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.components.todo import DOMAIN as TODO_DOMAIN
from homeassistant.components.weather import INTENT_GET_WEATHER
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import area_registry, floor_registry, device_registry, intent
from homeassistant.helpers.llm import LLMContext, API, APIInstance, _get_exposed_entities, _selector_serializer, \
    NO_ENTITIES_PROMPT, DYNAMIC_CONTEXT_PROMPT, Tool, IntentTool, CalendarGetEventsTool, TodoGetItemsTool, ScriptTool, \
    GetLiveContextTool
from homeassistant.util import yaml as yaml_util

from .const import LLAMA_LLM_API, DEFAULT_EMBEDDINGS_ENTITIES


class LlamaAssistAPI(API):
    """LLama Assist API."""

    IGNORE_INTENTS = {
        intent.INTENT_GET_TEMPERATURE,
        INTENT_GET_WEATHER,
        INTENT_OPEN_COVER,  # deprecated
        INTENT_CLOSE_COVER,  # deprecated
        intent.INTENT_GET_STATE,
        intent.INTENT_NEVERMIND,
        intent.INTENT_TOGGLE,
        intent.INTENT_GET_CURRENT_DATE,
        intent.INTENT_GET_CURRENT_TIME,
        intent.INTENT_RESPOND,
    }

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the API."""
        super().__init__(
            hass=hass,
            id=LLAMA_LLM_API,
            name="Llama Assist",
        )
        self.cached_slugify = cache(
            partial(unicode_slug.slugify, separator="_", lowercase=False)
        )

        self.all_exposed_entities: dict | None = None
        self.use_embedding_for_entities = DEFAULT_EMBEDDINGS_ENTITIES

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Return the instance of the API."""
        exposed_entities: dict | None = None

        # # Hack to check if option "use embeddings for entities" is set without additional function parameters
        # for entry in self.hass.config_entries.async_entries(DOMAIN):
        #     url_from_context = llm_context.context.origin_event.data.get("old_state", {}).name.removeprefix(
        #         "Llama Assist (").removesuffix(")")
        #
        #     if entry.data.get(CONF_COMPLETION_SERVER_URL, "") == url_from_context:
        #         # If the entry's base URL matches the target ID, use the option for embeddings
        #         self.use_embedding_for_entities = entry.options.get(CONF_USE_EMBEDDINGS_ENTITIES, USE_EMBEDDINGS_ENTITIES)
        #         break
        self.use_embedding_for_entities = True  # TODO, find a way to fix this

        if llm_context.assistant:
            _exposed_entities = _get_exposed_entities(self.hass, llm_context.assistant, include_state=False)

            if self.use_embedding_for_entities:
                # Save the exposed entities to add only the useful ones at each prompt
                self.all_exposed_entities = _exposed_entities
            else:
                # Default behavior, expose all entities (performance may be affected)
                exposed_entities = _exposed_entities

        return APIInstance(
            api=self,
            api_prompt=self._async_get_api_prompt(llm_context, exposed_entities),
            llm_context=llm_context,
            tools=self._async_get_tools(llm_context, self.all_exposed_entities or exposed_entities),
            custom_serializer=_selector_serializer,
        )

    @callback
    def _async_get_api_prompt(self, llm_context: LLMContext, exposed_entities: dict | None) -> str:
        if not exposed_entities or not exposed_entities["entities"]:
            return NO_ENTITIES_PROMPT

        parts = [*self._async_get_preable(llm_context)]

        if not self.use_embedding_for_entities:
            parts += self._async_get_exposed_entities_prompt(llm_context, exposed_entities)

        return "\n".join(parts)

    @callback
    def _async_get_preable(self, llm_context: LLMContext) -> list[str]:
        """Return the prompt for the API."""

        prompt = [
            (
                "When controlling Home Assistant always call the intent tools. "
                "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
                "When controlling a device, prefer passing just name and domain. "
                "When controlling an area, prefer passing just area name and domain."
            )
        ]
        area: area_registry.AreaEntry | None = None
        floor: floor_registry.FloorEntry | None = None
        extra = ""
        if llm_context.device_id:
            device_reg = device_registry.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = area_registry.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = floor_registry.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)

            extra = "and all generic commands like 'turn on the lights' should target this area."

        if floor and area:
            prompt.append(f"You are in area {area.name} (floor {floor.name}) {extra}")
        elif area:
            prompt.append(f"You are in area {area.name} {extra}")
        else:
            prompt.append(
                "When a user asks to turn on all devices of a specific type, "
                "ask user to specify an area, unless there is only one device of that type."
            )

        if not llm_context.device_id or not async_device_supports_timers(
                self.hass, llm_context.device_id
        ):
            prompt.append("This device is not able to start timers.")

        prompt.append(DYNAMIC_CONTEXT_PROMPT)

        return prompt

    @callback
    def _async_get_exposed_entities_prompt(
            self, llm_context: LLMContext, exposed_entities: dict | None
    ) -> list[str]:
        """Return the prompt for the API for exposed entities."""
        prompt = []

        if exposed_entities and exposed_entities["entities"]:
            prompt.append(
                "Static Context: An overview of the areas and the devices in this smart home:"
            )
            prompt.append(yaml_util.dump(list(exposed_entities["entities"].values())))

        return prompt

    @callback
    def _async_get_tools(
            self, llm_context: LLMContext, exposed_entities: dict | None
    ) -> list[Tool]:
        """Return a list of LLM tools."""
        ignore_intents = self.IGNORE_INTENTS
        if not llm_context.device_id or not async_device_supports_timers(
                self.hass, llm_context.device_id
        ):
            ignore_intents = ignore_intents | {
                intent.INTENT_START_TIMER,
                intent.INTENT_CANCEL_TIMER,
                intent.INTENT_INCREASE_TIMER,
                intent.INTENT_DECREASE_TIMER,
                intent.INTENT_PAUSE_TIMER,
                intent.INTENT_UNPAUSE_TIMER,
                intent.INTENT_TIMER_STATUS,
            }

        intent_handlers = [
            intent_handler
            for intent_handler in intent.async_get(self.hass)
            if intent_handler.intent_type not in ignore_intents
        ]

        exposed_domains: set[str] | None = None
        if exposed_entities is not None:
            exposed_domains = {
                info["domain"] for info in exposed_entities["entities"].values()
            }

            intent_handlers = [
                intent_handler
                for intent_handler in intent_handlers
                if intent_handler.platforms is None
                   or intent_handler.platforms & exposed_domains
            ]

        tools: list[Tool] = [
            IntentTool(self.cached_slugify(intent_handler.intent_type), intent_handler)
            for intent_handler in intent_handlers
        ]

        if exposed_entities:
            if exposed_entities[CALENDAR_DOMAIN]:
                names = []
                for info in exposed_entities[CALENDAR_DOMAIN].values():
                    names.extend(info["names"].split(", "))
                tools.append(CalendarGetEventsTool(names))

            if exposed_domains is not None and TODO_DOMAIN in exposed_domains:
                names = []
                for info in exposed_entities["entities"].values():
                    if info["domain"] != TODO_DOMAIN:
                        continue
                    names.extend(info["names"].split(", "))
                tools.append(TodoGetItemsTool(names))

            tools.extend(
                ScriptTool(self.hass, script_entity_id)
                for script_entity_id in exposed_entities[SCRIPT_DOMAIN]
            )

        if exposed_domains:
            tools.append(GetLiveContextTool())

        return tools
