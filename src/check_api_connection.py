"""Helper script: Verify OpenAI API key and model connectivity across all five configured roles."""

import os
import sys
import re
from pathlib import Path
import dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, NotFoundError, BadRequestError

ROLES = [
    ("ORCHESTRATOR", "OPENAI_MODEL_ORCHESTRATOR", "gpt-5.5"),
    ("RETRIEVAL", "OPENAI_MODEL_RETRIEVAL", "gpt-5-mini"),
    ("EVIDENCE", "OPENAI_MODEL_EVIDENCE", "gpt-5-mini"),
    ("REASONING", "OPENAI_MODEL_REASONING", "gpt-5.5"),
    ("VALIDATION", "OPENAI_MODEL_VALIDATION", "gpt-5-mini"),
]


def redact_keys(text: str) -> str:
    """Safely redact any sk-... API key patterns from text."""
    return re.sub(r"sk-[a-zA-Z0-9_\-]{15,}", "[REDACTED]", str(text))


def check_role_connection(client: OpenAI, role_name: str, model_name: str) -> bool:
    """Test a single role with a minimal call using at least 200 max_completion_tokens."""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Respond only with 'Pong'."}],
        "max_completion_tokens": 200,
    }

    # Only pass temperature for non-gpt-5 non-reasoning models
    if not (model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3")):
        kwargs["temperature"] = 0

    try:
        response = client.chat.completions.create(**kwargs)
        first_content = response.choices[0].message.content if response.choices else ""
        answer_text = (first_content or "").strip()
        print(f"[{role_name}] Model: {model_name} -> PASS (Response: {answer_text!r})")
        return True

    except (NotFoundError, BadRequestError) as e:
        safe_msg = redact_keys(str(e))
        print(f"[{role_name}] Model: {model_name} -> FAIL")
        print(f"Error: {safe_msg}")
        return False

    except RateLimitError as e:
        safe_msg = redact_keys(str(e))
        print(f"[{role_name}] Model: {model_name} -> FAIL (RateLimit / Insufficient Quota)")
        print(f"Error: {safe_msg}")
        return False

    except AuthenticationError as e:
        safe_msg = redact_keys(str(e))
        print(f"[{role_name}] Model: {model_name} -> FAIL (Authentication Error)")
        print(f"Error: {safe_msg}")
        return False

    except Exception as e:
        safe_msg = redact_keys(str(e))
        print(f"[{role_name}] Model: {model_name} -> FAIL")
        print(f"Error: {safe_msg}")
        return False


def check_all_roles() -> bool:
    env_path = Path(".env")
    if not env_path.exists():
        print("[FAIL] .env does not exist.")
        return False

    dotenv.load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[FAIL] OPENAI_API_KEY is not set in .env.")
        return False

    client = OpenAI(api_key=api_key)
    all_passed = True

    print("\n--- Verifying All 5 OpenAI Role Configurations ---")
    for role_name, env_var, default_model in ROLES:
        model_name = os.getenv(env_var, default_model)
        passed = check_role_connection(client, role_name, model_name)
        if not passed:
            all_passed = False
            # As instructed: If a model ID is rejected, show error and STOP
            print(f"\n[STOPPING] Role '{role_name}' model '{model_name}' was rejected.")
            return False

    print("\nAll 5 role models verified successfully!")
    return True


if __name__ == "__main__":
    success = check_all_roles()
    if not success:
        sys.exit(1)
