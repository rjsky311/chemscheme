"""AI intent parsing: natural language -> scheme.json via Claude API.

The Claude API drafts the whole scheme, SMILES included. The draft passes
schema validation here; run the chemistry and reaction validators before use.
"""

import json
import os
import re

import requests

_CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-sonnet-4-20250514"


class AiServiceError(Exception):
    """Raised when the AI service is unavailable or returns invalid data."""


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from AI response text.

    Handles ```json, ```JSON, and bare ``` wrappers, even with surrounding text (MR3-4).
    """
    stripped = text.strip()
    # Prefer ```json-labeled blocks over bare ``` blocks (MR4-13)
    match = re.search(r"```(?:json|JSON)\s*\n?(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback to bare code block
    match = re.search(r"```\s*\n?(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _call_claude_api(prompt: str) -> dict:
    """Call Claude API to parse NL description into scheme.json.

    Returns parsed scheme dict. Raises on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise AiServiceError("ANTHROPIC_API_KEY not set")

    system_prompt = (
        "You are a chemistry assistant. Convert the user's reaction description "
        "into a scheme.json format. Respond with ONLY valid JSON, no markdown. "
        "The JSON must have 'schema_version' (string, use \"1\") at top level "
        "and a 'steps' array where each step has: "
        "'id' (string), 'reactants' (list of SMILES), 'products' (list of SMILES), "
        "'arrow_above' (string with reagents/conditions). "
        "Use valid SMILES strings for all molecules."
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": _MODEL,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(_CLAUDE_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise AiServiceError(f"Claude API returned status {resp.status_code}")

    data = resp.json()
    # Safe access: handle empty content array (MR5R2-G5)
    content_list = data.get("content", [])
    content = content_list[0].get("text", "") if content_list else ""

    # Strip markdown fences if present (G2)
    content = _strip_markdown_fences(content)

    try:
        result: dict = json.loads(content)
    except json.JSONDecodeError as e:
        raise AiServiceError(f"AI returned invalid JSON: {e}") from e

    return result


def _validate_scheme(scheme: dict) -> None:
    """Validate that scheme has required structure.

    Uses the full schema validator and converts errors to AiServiceError.
    Injects schema_version if missing (C1).
    """
    from chemscheme.schema.scheme_schema import validate_scheme

    # Guard: must be a dict (MR5R2-G1/C3)
    if not isinstance(scheme, dict):
        raise AiServiceError(
            f"AI returned invalid scheme structure: expected dict, got {type(scheme).__name__}"
        )

    # Ensure schema_version is present (inject if AI forgot)
    if "schema_version" not in scheme:
        scheme["schema_version"] = "1"

    if not isinstance(scheme.get("steps"), list):
        raise AiServiceError("AI returned invalid scheme: 'steps' must be a list")

    # Check each step is a dict before deeper validation
    for i, step in enumerate(scheme["steps"]):
        if not isinstance(step, dict):
            raise AiServiceError(
                f"AI returned invalid scheme: step {i} must be a dict, got {type(step).__name__}"
            )

    # Run the full schema validator
    errors = validate_scheme(scheme)
    if errors:
        raise AiServiceError(f"AI returned invalid scheme: {'; '.join(errors)}")


def parse_intent(description: str) -> dict:
    """Parse natural language description into a scheme.json dict.

    Raises AiServiceError on failure (timeout, auth, invalid response).
    """
    try:
        scheme = _call_claude_api(description)
    except AiServiceError:
        raise
    except requests.exceptions.Timeout as e:
        raise AiServiceError("AI service unavailable, please try again") from e
    except requests.exceptions.ConnectionError as e:
        raise AiServiceError(f"AI service unreachable: {e}") from e
    except Exception as e:
        raise AiServiceError(f"AI service unavailable: {e}") from e

    _validate_scheme(scheme)
    return scheme
