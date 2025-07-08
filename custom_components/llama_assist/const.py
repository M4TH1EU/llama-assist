import logging
from enum import IntFlag

from homeassistant.const import Platform

DOMAIN = "llama_assist"
LLAMA_LLM_API = DOMAIN + "_api"
LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,Platform.TTS,)


CONF_ENTITY_PREFIX = "llama_assist_entity_prefix"
CONF_ACTION_DEVICE_TYPE = "llama_assist_add_action"

#
# Conversation API Configuration
#
CONF_CONVERSATION_SERVER_URL = "conversation-server-url"

CONF_PROMPT = "prompt"
DEFAULT_PROMPT = "You are a helpful assistant."

CONF_CONVERSATION_API_KEY = "conversation-api-key"
DEFAULT_CONVERSATION_API_KEY = "none"

CONF_CONVERSATION_MAX_HISTORY = "max_history"
DEFAULT_CONVERSATION_MAX_HISTORY = 20

CONF_CONVERSATION_DISABLE_REASONING = "disable_reasoning"
DEFAULT_CONVERSATION_DISABLE_REASONING = False

CONF_CONVERSATION_WANT_TO_USE_EMBEDDINGS = "want_to_use_embeddings"
DEFAULT_CONVERSATION_WANT_TO_USE_EMBEDDINGS = False

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


#
# Embeddings API Configuration
#
CONF_SERVER_EMBEDDINGS_URL = "embeddings-server-url"

CONF_EMBEDDINGS_API_KEY = "embeddings-api-key"
DEFAULT_EMBEDDINGS_API_KEY = "none"

CONF_EMBEDDINGS_TOOLS = "use_embeddings_tools"
DEFAULT_EMBEDDINGS_TOOLS = False

CONF_EMBEDDINGS_ENTITIES = "use_embeddings_entities"
DEFAULT_EMBEDDINGS_ENTITIES = False

CONF_EMBEDDINGS_OVERWRITE = "overwrite_embeddings"
DEFAULT_EMBEDDINGS_OVERWRITE = False

EMBEDDINGS_MIN_SCORE = 0.65
EMBEDDINGS_SQLITE = "llama_assist_embeddings.db"


#
# TTS API Configuration
#
CONF_TTS_SERVER_URL = "tts-server-url"

CONF_TTS_API_KEY = "tts-api-key"
DEFAULT_TTS_API_KEY = "none"

CONF_TTS_MODEL = "tts-model"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"

CONF_TTS_VOICE = "tts-voice"
DEFAULT_TTS_VOICE = "alloy"


# Timeout settings
HEALTHCHECK_TIMEOUT = 2
CONVERSATION_TIMEOUT = 30
EMBEDDINGS_TIMEOUT = 10


CONF_BLACKLIST_TOOLS = "blacklist_tools"
EXISTING_TOOLS = [
    'HassTurnOn',
    'HassTurnOff',
    'HassCancelAllTimers',
    'HassMediaUnpause',
    'HassMediaPause',
    'HassMediaNext',
    'HassMediaPrevious',
    'HassSetVolume',
    'HassMediaSearchAndPlay',
    'HassListAddItem',
    'HassListCompleteItem',
    'todo_get_items',
    'GetLiveContext',
    'HassSetPosition',
    'HassShoppingListAddItem',
    'HassShoppingListLastItems',
]

