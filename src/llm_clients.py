"""
Drop-in LLM callers for TechPath supporting Groq / Grok, Google Gemini, Anthropic Claude, and OpenAI.
All functions return raw response text; parse with parse_llm_json().
Includes 3-attempt retry loops with exponential backoff for transient 503/429 errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("TechPath.LLMClients")

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]
GEMINI_MODELS = ["gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-1.5-pro"]
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
OPENAI_MODEL = "gpt-4o-mini"


async def call_groq(prompt: str, api_key: str, system: Optional[str] = None) -> str:
    """Call Groq / Grok API endpoint with multi-model fallback."""
    logger.info(f"[LLM] Request started. Provider: Groq/Grok | Available Models: {GROQ_MODELS}")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=90.0) as client:
        for model in GROQ_MODELS:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
            for attempt in range(1, 4):
                try:
                    logger.info(f"[LLM] Invoking Groq model '{model}' (attempt {attempt}/3)...")
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                    if response.status_code != 200:
                        logger.warning(f"[LLM] Groq API model '{model}' HTTP {response.status_code}: {response.text[:200]}")
                    response.raise_for_status()
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    logger.info(f"[LLM] Groq model '{model}' response received successfully ({len(text)} chars).")
                    return text
                except Exception as exc:
                    logger.warning(f"[LLM] Groq model '{model}' (attempt {attempt}/3) failed: {exc}")
                    if attempt < 3:
                        await asyncio.sleep(1.5 * attempt)
    raise RuntimeError("All Groq API attempts failed across available models.")


async def call_claude(prompt: str, api_key: str, system: Optional[str] = None) -> str:
    """Single Anthropic Messages API call with retries. Returns raw text."""
    logger.info(f"[LLM] Request started. Provider: Claude | Model: {CLAUDE_MODEL}")
    payload: dict[str, Any] = {
        "model": CLAUDE_MODEL,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(1, 4):
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code != 200:
                    logger.warning(f"[LLM] Claude API HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                data = response.json()
                text = data["content"][0]["text"]
                logger.info(f"[LLM] Claude response received successfully ({len(text)} chars).")
                return text
            except Exception as exc:
                logger.warning(f"[LLM] Claude attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    await asyncio.sleep(2.0 * attempt)
        raise RuntimeError("All Anthropic Claude API attempts failed.")


async def call_gemini_latest(prompt: str, api_key: str) -> str:
    """Call Google Gemini API with 3-attempt retry loop for transient errors."""
    logger.info(f"[LLM] Request started. Provider: Gemini | Available Models: {GEMINI_MODELS}")
    last_exc = None

    async with httpx.AsyncClient(timeout=90.0) as client:
        for model in GEMINI_MODELS:
            for attempt in range(1, 4):
                try:
                    logger.info(f"[LLM] Invoking Gemini API model '{model}' (attempt {attempt}/3)...")
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
                    if response.status_code != 200:
                        logger.warning(f"[LLM] Gemini API model '{model}' HTTP {response.status_code}: {response.text[:200]}")
                    response.raise_for_status()
                    data = response.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"[LLM] Gemini '{model}' response received successfully ({len(text)} chars).")
                    return text
                except Exception as exc:
                    logger.warning(f"[LLM] Gemini model '{model}' (attempt {attempt}/3) failed: {exc}")
                    last_exc = exc
                    if attempt < 3:
                        await asyncio.sleep(2.0 * attempt)

    raise last_exc or RuntimeError("All Gemini API attempts failed.")


async def call_openai(prompt: str, api_key: str, system: Optional[str] = None) -> str:
    """Call OpenAI Chat Completions API. Returns raw text response."""
    logger.info(f"[LLM] Request started. Provider: OpenAI | Model: {OPENAI_MODEL}")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(1, 4):
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                if response.status_code != 200:
                    logger.warning(f"[LLM] OpenAI API HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                logger.info(f"[LLM] OpenAI response received successfully ({len(text)} chars).")
                return text
            except Exception as exc:
                logger.warning(f"[LLM] OpenAI attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    await asyncio.sleep(2.0 * attempt)
        raise RuntimeError("All OpenAI API attempts failed.")


def parse_llm_json(raw_text: str) -> dict:
    """Parse raw LLM response text into JSON, cleaning markdown code fences."""
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
            raise ValueError(f"Failed to parse LLM response into JSON. Raw text: {raw_text[:200]}...")
