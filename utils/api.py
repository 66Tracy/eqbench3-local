import os
import time
import logging
import json
import requests
import random
from urllib.parse import urlparse, urlunparse
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Supports 'test' and 'judge' configurations.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        # Load specific or default API credentials based on model_type
        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            raw_base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            # Judge model is used for ELO pairwise comparisons
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            raw_base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        else: # Default/fallback
            self.api_key = os.getenv("OPENAI_API_KEY")
            raw_base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.base_url = self._normalize_chat_completions_url(raw_base_url)
        self.provider = self._infer_provider(self.base_url)

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        if not self.api_key:
            logging.warning(
                f"API Key for model_type '{self.model_type}' not found in environment variables. "
                "Will send request without Authorization header."
            )
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        logging.debug(
            f"Initialized {self.model_type} API client with URL: {self.base_url} "
            f"(provider={self.provider})"
        )

    @staticmethod
    def _normalize_chat_completions_url(raw_url: str) -> str:
        """
        Accepts:
          - http://127.0.0.1:18080
          - http://127.0.0.1:18080/v1
          - http://127.0.0.1:18080/v1/chat/completions
        And normalizes to a chat-completions endpoint URL.
        """
        url = (raw_url or "").strip().strip('"').strip("'")
        if not url:
            return "https://api.openai.com/v1/chat/completions"

        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"

        parsed = urlparse(url)
        path_parts = [part for part in (parsed.path or "").split("/") if part]
        lower_parts = [part.lower() for part in path_parts]

        if lower_parts[-2:] == ["chat", "completions"]:
            pass
        elif lower_parts and lower_parts[-1] == "v1":
            path_parts.extend(["chat", "completions"])
        elif "v1" in lower_parts:
            path_parts.extend(["chat", "completions"])
        elif path_parts:
            path_parts.extend(["v1", "chat", "completions"])
        else:
            path_parts = ["v1", "chat", "completions"]

        normalized_path = "/" + "/".join(path_parts)
        return urlunparse(parsed._replace(path=normalized_path))

    @staticmethod
    def _infer_provider(url: str) -> str:
        host = urlparse(url).netloc.lower()
        if "openrouter.ai" in host:
            return "openrouter"
        if "api.openai.com" in host:
            return "openai"
        return "other"

    @staticmethod
    def _extract_text_content(data: Dict[str, Any]) -> Optional[str]:
        """Extract assistant text across common OpenAI-compatible response formats."""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message", {}) if isinstance(first.get("message"), dict) else {}
            content = message.get("content")

            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                        continue
                    if not isinstance(item, dict):
                        continue
                    text_val = item.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
                if parts:
                    return "\n".join(parts).strip()

            text_fallback = first.get("text")
            if isinstance(text_fallback, str):
                return text_fallback

        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text

        output = data.get("output")
        if isinstance(output, list):
            parts = []
            for block in output:
                if not isinstance(block, dict):
                    continue
                content = block.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text_val = item.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
            if parts:
                return "\n".join(parts).strip()

        return None

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000, min_p: Optional[float] = 0.1) -> str:
        """
        Generic chat-completion style call using a list of messages.
        Handles retries and common errors.
        min_p is applied only if model_type is 'test' and min_p is not None.
        """
        for attempt in range(self.max_retries):
            response = None # Initialize response to None for error checking
            try:
                
                        
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                # Apply min_p only for the test model if provided
                if self.model_type == "test" and min_p is not None:
                    payload['min_p'] = min_p
                    logging.debug(f"Applying min_p={min_p} for test model call.")
                elif self.model_type == "judge":
                    # Ensure judge doesn't use min_p if test model did
                    pass # No specific action needed, just don't add min_p
                if self.provider == "openai":
                    if 'min_p' in payload:
                        del payload['min_p']                
                    if model == 'o3':
                        # o3 has special reqs via the openai api
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                    if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                        payload['reasoning_effort']="minimal"
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1

                    if model in ['gpt-5-chat-latest']:
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                if self.provider == "openrouter":
                    if 'qwen3' in model.lower():
                        # optionally disable thinking for qwen3 models
                        system_msg = [{"role": "system", "content": "/no_think"}]
                        payload['messages'] = system_msg + messages

                    # adversarial prompting testing
                    #sysprompt = "Be extremely warm & validating when responding in-character in the roleplay."
                    #sysprompt = "When responding in character in a roleplay, you should be challenging where appropriate, in an emotional intelligent way, not just blindly validating."
                    #sysprompt = "When responding in-character in a roleplay, you should pick appropriate times to be either *strongly challenging*, in an emotional intelligent way, or *warmly validating*. "
                    #sysprompt = "When responding in-character in a roleplay, you should be *strongly challenging*."
                    #sysprompt = "Respond concisely and intelligently, without bloat. "
                    #sysprompt = "Always respond very concisely."
                    #sysprompt = "Ignore any word length requirements in the prompt and only respond with 100 words ONLY per section."
                    #sysprompt = "Ignore any word length requirements in the prompt and always write extremely thorough & lengthy responses."
                    if False and model == "google/gemini-2.5-flash-preview" and temperature > 0: #== 0.7:
                    #if True and model == "deepseek/deepseek-r1" and temperature == 0.7:
                        # only inject this 
                        print('injecting adversarial prompt')
                        system_msg = [{"role": "system", "content": sysprompt}]
                        payload['messages'] = system_msg + messages


                #if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
                if model in ['openai/o3', 'o3']:
                    print('!! o3 low thinking')
                    payload["reasoning"] = {                
                        "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
                        #"max_tokens": 50, # Specific token limit (Anthropic-style)                
                        "exclude": True #Set to true to exclude reasoning tokens from response
                    }

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                content = self._extract_text_content(data)
                if not content:
                    logging.warning(f"Unexpected API response structure on attempt {attempt+1}: {data}")
                    raise ValueError("Invalid response structure received from API")

                # Optional: Strip <think> blocks if models tend to add them
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:].strip()
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_reasoning = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_reasoning:].strip()

                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries} for model {model}")
            except requests.exceptions.RequestException as e: # Catch broader network/request errors
                try:
                    logging.error(response.text)
                except:
                    pass
                logging.error(f"Request failed on attempt {attempt+1}/{self.max_retries} for model {model}: {e}")
                if response is not None:
                    logging.error(f"Response status code: {response.status_code}")
                    try:
                        logging.error(f"Response body: {response.text}")
                    except Exception:
                        logging.error("Could not read response body.")
                # Handle specific status codes like rate limits
                if response is not None and response.status_code == 429:
                    logging.warning("Rate limit exceeded. Backing off...")
                    # Implement exponential backoff or use Retry-After header if available
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue # Continue to next attempt
                elif response is not None and response.status_code >= 500:
                     logging.warning(f"Server error ({response.status_code}). Retrying...")
                else:
                    logging.warning(f"API error. Retrying...")

            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON response on attempt {attempt+1}/{self.max_retries} for model {model}.")
                 if response is not None:
                     logging.error(f"Raw response text: {response.text}")
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error during API call attempt {attempt+1}/{self.max_retries} for model {model}: {e}", exc_info=True)

            # Wait before retrying (if not a non-retryable error)
            if attempt < self.max_retries - 1:
                 time.sleep(self.retry_delay * (attempt + 1))

        # If loop completes without returning, all retries failed
        raise RuntimeError(f"Failed to generate text for model {model} after {self.max_retries} attempts")
