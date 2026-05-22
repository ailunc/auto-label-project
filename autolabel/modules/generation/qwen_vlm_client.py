from __future__ import annotations

import json
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autolabel.utils import image_to_data_url

from .grid import grid_ids, validate_grid_id
from .prompts import build_qwen_fine_grid_prompt, build_qwen_grid_prompt, build_qwen_review_prompt


class VLMResponseError(RuntimeError):
    pass


class VLMNetworkError(VLMResponseError):
    pass


class VLMJsonError(VLMResponseError):
    pass


@dataclass(frozen=True)
class GridCandidate:
    grid: str
    score: float
    reason: str


@dataclass(frozen=True)
class GridSelection:
    selected_grid: str
    confidence: float
    candidate_grids: list[GridCandidate]
    edit_region_hint: str
    risk_note: str
    rejected_region_reason: str
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ReviewResult:
    is_valid: bool
    anomaly_type_match: bool
    location_reasonable: bool
    visual_artifact: bool
    score: float
    reason: str
    raw_response: dict[str, Any]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _as_score(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


class QwenVLMClient:
    def __init__(
        self,
        model_name: str = "qwen3.6-plus",
        dry_run: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        request_image_max_side: int = 768,
        request_timeout_seconds: int = 60,
        json_retry_count: int = 1,
        enable_grid_fallback: bool = True,
        jpeg_quality: int = 82,
    ) -> None:
        self.model_name = model_name
        self.dry_run = dry_run
        self.api_key = api_key or os.getenv("QWEN_VLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("QWEN_VLM_BASE_URL")
            or os.getenv("DASHSCOPE_VLM_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.base_url = "".join(str(self.base_url).split())
        self.request_image_max_side = request_image_max_side
        self.request_timeout_seconds = request_timeout_seconds
        self.json_retry_count = max(0, json_retry_count)
        self.enable_grid_fallback = enable_grid_fallback
        self.jpeg_quality = jpeg_quality

    def select_grid(self, grid_preview_image: str | Path, anomaly_type: str, grid_layout: str = "4x4") -> GridSelection:
        prompt = build_qwen_grid_prompt(anomaly_type)
        if self.dry_run:
            return self._dummy_grid_selection(grid_layout, edit_region_hint=f"dry-run {anomaly_type} candidate")
        try:
            payload = self._call_json_model(grid_preview_image, prompt, retry_prompt=prompt + "\nReturn JSON only. No prose.")
            return self._parse_grid_selection(payload, grid_layout)
        except (VLMResponseError, ValueError, json.JSONDecodeError) as exc:
            if self.enable_grid_fallback:
                return self._dummy_grid_selection(
                    grid_layout,
                    edit_region_hint=f"fallback {anomaly_type} candidate after Qwen failure",
                    fallback_reason=str(exc),
                )
            if isinstance(exc, VLMResponseError):
                raise
            raise VLMResponseError(f"Qwen VLM grid selection failed: {exc}") from exc

    def select_fine_grid(self, fine_grid_preview_image: str | Path, anomaly_type: str, coarse_grid: str) -> GridSelection:
        prompt = build_qwen_fine_grid_prompt(anomaly_type, coarse_grid)
        if self.dry_run:
            return self._dummy_grid_selection("3x3", edit_region_hint=f"dry-run fine selection within {coarse_grid}")
        try:
            payload = self._call_json_model(fine_grid_preview_image, prompt, retry_prompt=prompt + "\nReturn JSON only. No prose.")
            return self._parse_grid_selection(payload, "3x3")
        except (VLMResponseError, ValueError, json.JSONDecodeError) as exc:
            if self.enable_grid_fallback:
                return self._dummy_grid_selection(
                    "3x3",
                    edit_region_hint=f"fallback fine selection within {coarse_grid} after Qwen failure",
                    fallback_reason=str(exc),
                )
            if isinstance(exc, VLMResponseError):
                raise
            raise VLMResponseError(f"Qwen VLM fine grid selection failed: {exc}") from exc

    def review_generation(self, image_path: str | Path, anomaly_type: str) -> ReviewResult:
        prompt = build_qwen_review_prompt(anomaly_type)
        if self.dry_run:
            raw = {
                "is_valid": True,
                "anomaly_type_match": True,
                "location_reasonable": True,
                "visual_artifact": False,
                "score": 0.88,
                "reason": "dry-run review accepts deterministic synthetic anomaly",
            }
            return self._parse_review(raw)
        payload = self._call_json_model(image_path, prompt, retry_prompt=prompt + "\nReturn JSON only. No prose.")
        return self._parse_review(payload)

    def _call_json_model(self, image_path: str | Path, prompt: str, retry_prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise VLMResponseError("Qwen VLM API key is not configured. Use --dry-run or set DASHSCOPE_API_KEY/QWEN_VLM_API_KEY.")
        prompts = [prompt] + [retry_prompt] * self.json_retry_count
        last_error: Exception | None = None
        for attempt_index, current_prompt in enumerate(prompts, 1):
            try:
                raw_text = self._openai_compatible_vision_call(image_path, current_prompt)
                return _extract_json_object(raw_text)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                raise VLMNetworkError(f"Qwen VLM HTTP {exc.code}: {body}") from exc
            except (TimeoutError, socket.timeout, ssl.SSLError, urllib.error.URLError) as exc:
                raise VLMNetworkError(f"Qwen VLM network error: {exc}") from exc
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt_index < len(prompts):
                    continue
            except Exception as exc:  # noqa: BLE001 - record retry cause for API/debug flow.
                last_error = exc
                if attempt_index < len(prompts):
                    continue
        raise VLMJsonError(f"Qwen VLM response was not valid JSON after {len(prompts)} attempt(s): {last_error}")

    def _openai_compatible_vision_call(self, image_path: str | Path, prompt: str) -> str:
        data_url = image_to_data_url(image_path, max_side=self.request_image_max_side, jpeg_quality=self.jpeg_quality)
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        request_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a strict JSON API. Return only valid JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:  # noqa: S310 - endpoint is user-configured.
            payload = json.loads(response.read().decode("utf-8"))
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise VLMResponseError(f"Qwen VLM returned an empty response: {payload}")
        return content

    def _parse_grid_selection(self, payload: dict[str, Any], grid_layout: str) -> GridSelection:
        selected_grid = str(payload.get("selected_grid", "")).strip().upper()
        validate_grid_id(selected_grid, grid_layout)
        raw_candidates = payload.get("candidate_grids")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise VLMResponseError("candidate_grids must be a non-empty list")

        candidates: list[GridCandidate] = []
        seen: set[str] = set()
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            grid = str(item.get("grid", "")).strip().upper()
            validate_grid_id(grid, grid_layout)
            if grid in seen:
                continue
            seen.add(grid)
            candidates.append(
                GridCandidate(
                    grid=grid,
                    score=_as_score(item.get("score")),
                    reason=str(item.get("reason") or ""),
                )
            )
        if selected_grid not in seen:
            candidates.insert(0, GridCandidate(selected_grid, _as_score(payload.get("confidence")), "selected_grid fallback"))
        if len(candidates) < 3:
            for grid in grid_ids(grid_layout):
                if grid not in {candidate.grid for candidate in candidates}:
                    candidates.append(GridCandidate(grid, 0.01, "automatic fallback to keep top-3 candidate flow"))
                if len(candidates) >= 3:
                    break
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:3]
        return GridSelection(
            selected_grid=selected_grid,
            confidence=_as_score(payload.get("confidence")),
            candidate_grids=candidates,
            edit_region_hint=str(payload.get("edit_region_hint") or ""),
            risk_note=str(payload.get("risk_note") or ""),
            rejected_region_reason=str(payload.get("rejected_region_reason") or ""),
            raw_response=payload,
        )

    def _parse_review(self, payload: dict[str, Any]) -> ReviewResult:
        return ReviewResult(
            is_valid=bool(payload.get("is_valid")),
            anomaly_type_match=bool(payload.get("anomaly_type_match")),
            location_reasonable=bool(payload.get("location_reasonable")),
            visual_artifact=bool(payload.get("visual_artifact")),
            score=_as_score(payload.get("score")),
            reason=str(payload.get("reason") or ""),
            raw_response=payload,
        )

    def _dummy_grid_selection(self, grid_layout: str, edit_region_hint: str, fallback_reason: str | None = None) -> GridSelection:
        ids = grid_ids(grid_layout)
        rows_cols = grid_layout.lower().split("x")
        rows, cols = int(rows_cols[0]), int(rows_cols[1])
        center = f"{chr(ord('A') + min(rows - 1, max(0, rows // 2)))}{min(cols, max(1, (cols + 1) // 2))}"
        candidates = [center]
        for candidate in ("C2", "C3", "B2", "B1", "A1"):
            if candidate in ids and candidate not in candidates:
                candidates.append(candidate)
        for candidate in ids:
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= 3:
                break
        scores = [0.86, 0.74, 0.62]
        raw_candidates = [
            {"grid": grid, "score": scores[idx], "reason": "deterministic fallback candidate" if fallback_reason else "deterministic dry-run candidate"}
            for idx, grid in enumerate(candidates[:3])
        ]
        raw = {
            "selected_grid": candidates[0],
            "confidence": scores[0],
            "candidate_grids": raw_candidates,
            "edit_region_hint": edit_region_hint,
            "risk_note": "Qwen grid selection fallback was used" if fallback_reason else "dry-run does not use an external VLM",
            "rejected_region_reason": "fallback skips visual semantics and validates pipeline continuity" if fallback_reason else "dry-run skips visual semantics and validates pipeline structure",
            "fallback": bool(fallback_reason),
            "fallback_reason": fallback_reason,
        }
        return self._parse_grid_selection(raw, grid_layout)
