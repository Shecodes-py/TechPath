"""
Drop-in LLM callers for TechPath.

Use call_claude() or call_gemini_latest() instead of looping through
hardcoded Gemini version names. Both return raw text; parse with parse_llm_json().
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("TechPath.LLMClients")

import asyncio

GEMINI_MODEL = "gemini-1.5-flash"
CLAUDE_MODEL = "claude-sonnet-4-5"

async def call_claude(prompt: str, api_key: str, system: str | None = None) -> str:
    """Single Anthropic Messages call. Returns raw text."""
    payload: dict[str, Any] = {
        "model": CLAUDE_MODEL,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


async def call_gemini_latest(prompt: str, api_key: str) -> str:
    """Call pinned gemini-1.5-flash with 3-attempt retry loop for 503 transient errors."""
    models = ["gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    last_exc = None

    async with httpx.AsyncClient(timeout=90) as client:
        for model in models:
            for attempt in range(1, 4):
                try:
                    logger.info(f"Invoking Gemini API model {model} (attempt {attempt}/3)...")
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={"x-goog-api-key": api_key, "content-type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "responseMimeType": "application/json",
                            },
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as exc:
                    logger.warning(f"Gemini model {model} (attempt {attempt}/3) failed: {exc}")
                    last_exc = exc
                    if attempt < 3:
                        await asyncio.sleep(2)

    raise last_exc or RuntimeError("All Gemini API attempts failed.")


def parse_llm_json(raw_text: str) -> dict:
    """Parse model text into JSON, including fenced ```json blocks."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise
