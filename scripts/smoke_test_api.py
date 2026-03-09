import argparse
import json
import logging
import os
import sys

import requests
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.api import APIClient
from utils.logging_setup import setup_logging


def _clean_env_value(value: str) -> str:
    return (value or "").strip().strip('"').strip("'")


def _resolve_model_name(model_type: str, override_model: str) -> str:
    if override_model:
        return override_model

    env_key = "TEST_MODEL_NAME" if model_type == "test" else "JUDGE_MODEL_NAME"
    env_value = _clean_env_value(os.getenv(env_key, ""))
    if env_value:
        return env_value
    raise ValueError(f"Missing model name for '{model_type}'. Set {env_key} in .env or pass --model.")


def _run_client_check(model_type: str, model_name: str, prompt: str) -> dict:
    client = APIClient(model_type=model_type, max_retries=1, retry_delay=1)
    messages = [{"role": "user", "content": prompt}]
    output = client.generate(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=64,
        min_p=None,
    )
    return {
        "endpoint": client.base_url,
        "preview": output[:120],
    }


def _run_raw_request_check(model_type: str, model_name: str, prompt: str) -> dict:
    client = APIClient(model_type=model_type, max_retries=1, retry_delay=1)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    headers = {"Content-Type": "application/json"}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    response = requests.post(
        client.base_url,
        headers=headers,
        json=payload,
        timeout=client.request_timeout,
    )
    response.raise_for_status()
    data = response.json()
    content = APIClient._extract_text_content(data)
    if not content:
        raise RuntimeError(f"Raw request returned unsupported response format: {data}")
    return {
        "endpoint": client.base_url,
        "preview": content[:120],
    }


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Smoke test EQBench API config against local OpenAI-compatible endpoints."
    )
    parser.add_argument("--type", choices=["test", "judge", "both"], default="both")
    parser.add_argument("--model", help="Optional override model name for the selected type(s).")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Prompt used in smoke call (keep short to reduce token cost).",
    )
    parser.add_argument("--skip-raw-request", action="store_true", help="Only run APIClient check.")
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args()

    setup_logging(args.verbosity)

    target_types = ["test", "judge"] if args.type == "both" else [args.type]
    summary = {}
    failed = False

    for model_type in target_types:
        try:
            model_name = _resolve_model_name(model_type, args.model)
            logging.info(f"[SmokeTest] Running client check for {model_type} using model '{model_name}'")
            type_result = {
                "client": _run_client_check(model_type, model_name, args.prompt)
            }
            if not args.skip_raw_request:
                logging.info(f"[SmokeTest] Running raw requests check for {model_type}")
                type_result["raw_request"] = _run_raw_request_check(model_type, model_name, args.prompt)
            summary[model_type] = {"model": model_name, "status": "ok", "checks": type_result}
        except Exception as exc:
            failed = True
            logging.error(f"[SmokeTest] {model_type} check failed: {exc}", exc_info=True)
            summary[model_type] = {"status": "failed", "error": str(exc)}

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
