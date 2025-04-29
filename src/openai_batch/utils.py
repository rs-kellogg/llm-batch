import openai
import yaml
import tiktoken
from pathlib import Path
from typing import List, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# -----------------------------------------------------------------------------
def config(config_file: Path) -> Dict:
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        return conf


# -----------------------------------------------------------------------------
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, kwargs) -> openai.types.chat.chat_completion.ChatCompletion:
    response = client.chat.completions.create(**kwargs)
    return response


# -----------------------------------------------------------------------------
def make_message_body(system_prompt, user_prompt) -> List[Dict]:
    message_body = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]
    return message_body


# -----------------------------------------------------------------------------
def count_tokens(text: str, model: str) -> int:
    assert text is not None and len(text) > 0
    assert model is not None

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens


# -----------------------------------------------------------------------------
def validate_response(result: str) -> bool:
    assert result is not None
    return True
