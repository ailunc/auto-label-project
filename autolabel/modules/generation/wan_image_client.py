from __future__ import annotations

import base64
import json
import os
import random
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from autolabel.utils import image_to_data_url

from .grid import BBox, clip_bbox


@dataclass(frozen=True)
class WanGenerationResult:
    output_path: Path
    raw_response: dict[str, Any]


class WanImageClient:
    def __init__(
        self,
        model_name: str = "wan2.7-image-pro",
        dry_run: bool = False,
        api_key: str | None = None,
        request_timeout_seconds: int = 300,
    ) -> None:
        self.model_name = model_name
        self.dry_run = dry_run
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("WAN_IMAGE_API_KEY")
        self.endpoint = os.getenv("DASHSCOPE_WAN_ENDPOINT") or os.getenv("WAN_IMAGE_EDIT_ENDPOINT")
        if self.endpoint:
            self.endpoint = "".join(str(self.endpoint).split())
        self.request_timeout_seconds = request_timeout_seconds

    def is_configured(self) -> tuple[bool, str | None]:
        if self.dry_run:
            return True, None
        if not self.api_key:
            return False, "Wan API key is not configured. Set DASHSCOPE_API_KEY or WAN_IMAGE_API_KEY."
        if not self.endpoint:
            return False, "Wan endpoint is not configured. Set DASHSCOPE_WAN_ENDPOINT or WAN_IMAGE_EDIT_ENDPOINT."
        return True, None

    def generate(
        self,
        original_image_path: str | Path,
        bbox_list: list[BBox],
        prompt: str,
        negative_prompt: str,
        output_path: str | Path,
        response_path: str | Path,
        anomaly_type: str,
    ) -> WanGenerationResult:
        output = Path(output_path)
        response_output = Path(response_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        response_output.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            raw = self._generate_dry_run(original_image_path, bbox_list, output, anomaly_type)
        else:
            raw = self._generate_live(original_image_path, bbox_list, prompt, negative_prompt, output)
        response_output.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return WanGenerationResult(output_path=output, raw_response=raw)

    def _generate_live(
        self,
        original_image_path: str | Path,
        bbox_list: list[BBox],
        prompt: str,
        negative_prompt: str,
        output_path: Path,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Wan API key is not configured. Use --dry-run or set DASHSCOPE_API_KEY/WAN_IMAGE_API_KEY.")
        if not self.endpoint:
            raise RuntimeError(
                "WAN_IMAGE_EDIT_ENDPOINT/DASHSCOPE_WAN_ENDPOINT is not configured. "
                "This wrapper expects an endpoint returning image_base64, image_url, or local_path; use --dry-run to validate locally."
            )
        if "dashscope" in self.endpoint and "/multimodal-generation/generation" in self.endpoint:
            return self._generate_dashscope_official(original_image_path, bbox_list, prompt, output_path)

        with open(original_image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        request_payload = {
            "model": self.model_name,
            "image_base64": image_b64,
            "bbox_list": [list(map(int, bbox)) for bbox in bbox_list],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:  # noqa: S310 - endpoint is user-configured.
            payload = json.loads(response.read().decode("utf-8"))

        image_b64_response = payload.get("image_base64") or payload.get("generated_image_base64")
        if image_b64_response:
            output_path.write_bytes(base64.b64decode(image_b64_response))
        elif payload.get("local_path"):
            shutil.copyfile(payload["local_path"], output_path)
        elif payload.get("image_url"):
            with urllib.request.urlopen(payload["image_url"], timeout=self.request_timeout_seconds) as image_response:  # noqa: S310
                output_path.write_bytes(image_response.read())
        else:
            raise RuntimeError("Wan endpoint response did not include image_base64, image_url, or local_path.")
        payload["saved_output_path"] = str(output_path)
        payload["input_was_clean_original"] = True
        return payload

    def _generate_dashscope_official(
        self,
        original_image_path: str | Path,
        bbox_list: list[BBox],
        prompt: str,
        output_path: Path,
    ) -> dict[str, Any]:
        with Image.open(original_image_path) as source:
            width, height = source.size
        image_data_url = image_to_data_url(original_image_path, max_side=None, jpeg_quality=95)
        request_payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_data_url},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "bbox_list": [[list(map(int, bbox)) for bbox in bbox_list]],
                "size": f"{width}*{height}",
                "n": 1,
                "watermark": False,
                "thinking_mode": True,
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:  # noqa: S310 - official endpoint is user-configured.
            payload = json.loads(response.read().decode("utf-8"))

        image_url = self._extract_dashscope_image_url(payload)
        if not image_url:
            raise RuntimeError(f"DashScope Wan response did not include an output image URL: {payload}")
        with urllib.request.urlopen(image_url, timeout=self.request_timeout_seconds) as image_response:  # noqa: S310 - URL is returned by DashScope.
            output_path.write_bytes(image_response.read())
        payload["saved_output_path"] = str(output_path)
        payload["input_was_clean_original"] = True
        payload["request_bbox_list"] = request_payload["parameters"]["bbox_list"]
        payload["request_size"] = request_payload["parameters"]["size"]
        return payload

    @staticmethod
    def _extract_dashscope_image_url(payload: dict[str, Any]) -> str | None:
        output = payload.get("output") or {}
        for choice in output.get("choices") or []:
            message = choice.get("message") or {}
            for item in message.get("content") or []:
                if isinstance(item, dict) and item.get("image"):
                    return str(item["image"])
        for result in output.get("results") or []:
            if isinstance(result, dict) and result.get("url"):
                return str(result["url"])
        return None

    def _generate_dry_run(self, original_image_path: str | Path, bbox_list: list[BBox], output_path: Path, anomaly_type: str) -> dict[str, Any]:
        with Image.open(original_image_path) as source:
            base = source.convert("RGB")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        width, height = base.size
        rng = random.Random(str(output_path))
        for bbox in bbox_list:
            x1, y1, x2, y2 = clip_bbox(bbox, width, height)
            bw, bh = x2 - x1, y2 - y1
            color = self._leak_color(anomaly_type)
            center_x = x1 + int(bw * (0.45 + rng.random() * 0.15))
            center_y = y1 + int(bh * (0.55 + rng.random() * 0.25))
            puddle_w = max(6, int(bw * (0.18 + rng.random() * 0.12)))
            puddle_h = max(4, int(bh * (0.06 + rng.random() * 0.08)))
            for idx in range(5):
                dx = int((rng.random() - 0.5) * puddle_w * 0.9)
                dy = int((rng.random() - 0.5) * puddle_h * 0.9)
                scale = 1.0 - idx * 0.12
                box = (
                    center_x + dx - int(puddle_w * scale / 2),
                    center_y + dy - int(puddle_h * scale / 2),
                    center_x + dx + int(puddle_w * scale / 2),
                    center_y + dy + int(puddle_h * scale / 2),
                )
                draw.ellipse(box, fill=color)
            trail_points = []
            for idx in range(5):
                px = center_x - int(puddle_w * 0.45) + int(idx * puddle_w / 4)
                py = center_y - int(puddle_h * 1.6) + int(rng.random() * puddle_h)
                trail_points.append((px, py))
            draw.line(trail_points, fill=color, width=max(2, int(min(bw, bh) * 0.015)))
        generated = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        generated.save(output_path, quality=95)
        return {
            "dry_run": True,
            "model": self.model_name,
            "bbox_list": [list(map(int, bbox)) for bbox in bbox_list],
            "saved_output_path": str(output_path),
            "input_was_clean_original": True,
            "note": "Synthetic local leak drawn for offline validation; no grid/text/box annotations added.",
        }

    @staticmethod
    def _leak_color(anomaly_type: str) -> tuple[int, int, int, int]:
        if anomaly_type == "oil_leak":
            return (25, 18, 12, 150)
        if anomaly_type == "coolant_leak":
            return (115, 210, 170, 115)
        if anomaly_type == "water_leak":
            return (245, 250, 255, 145)
        return (228, 184, 75, 105)
