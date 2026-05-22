from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from autolabel.modules.generation.grid import expand_bbox, grid_id_to_bbox
from autolabel.modules.generation.main import main as vlm_wan_main
from autolabel.modules.generation.prompts import build_negative_prompt, build_qwen_grid_prompt, build_wan_edit_prompt
from autolabel.modules.generation.qwen_vlm_client import QwenVLMClient
from autolabel.utils import read_json
from autolabel.validators import validate_sample_contract


class VLMWanAutoLabelTests(unittest.TestCase):
    def test_grid_bbox_and_expansion(self) -> None:
        self.assertEqual(grid_id_to_bbox("A1", 400, 200, "4x4"), (0, 0, 100, 50))
        self.assertEqual(grid_id_to_bbox("D4", 400, 200, "4x4"), (300, 150, 400, 200))
        self.assertEqual(expand_bbox((100, 50, 200, 100), 400, 200, 0.2), (80, 40, 220, 110))

    def test_water_leak_prompts_are_distinct_from_oil_diesel_coolant(self) -> None:
        qwen_prompt = build_qwen_grid_prompt("water_leak")
        wan_prompt = build_wan_edit_prompt("water_leak", "near a drain pipe and equipment base", "early")
        negative_prompt = build_negative_prompt()
        self.assertIn("漏水 / 清水泄漏", qwen_prompt)
        self.assertIn("纯白透明", qwen_prompt)
        self.assertIn("纯白色、透明偏白", wan_prompt)
        self.assertIn("不是黄色柴油", wan_prompt)
        self.assertIn("银绿色防冻液", negative_prompt)
        self.assertIn("黑褐色油污", negative_prompt)
        self.assertIn("高压喷水", negative_prompt)

    def test_qwen_grid_selection_falls_back_on_timeout(self) -> None:
        class TimeoutQwen(QwenVLMClient):
            def _openai_compatible_vision_call(self, image_path: str | Path, prompt: str) -> str:
                raise TimeoutError("simulated timeout")

        client = TimeoutQwen(dry_run=False, api_key="test-key", request_timeout_seconds=1)
        selection = client.select_grid("missing-grid.jpg", "water_leak", "4x4")
        self.assertTrue(selection.raw_response["fallback"])
        self.assertIn("simulated timeout", selection.raw_response["fallback_reason"])
        self.assertEqual(len(selection.candidate_grids), 3)

    def test_dry_run_writes_valid_metadata_with_diff_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            output_root = root / "outputs"
            image_root.mkdir()
            image_path = image_root / "engine_room.jpg"
            image = Image.new("RGB", (360, 240), (115, 118, 120))
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 50, 320, 170), fill=(80, 84, 86), outline=(160, 160, 160), width=3)
            draw.line((130, 105, 300, 105), fill=(185, 150, 42), width=10)
            draw.rectangle((0, 170, 360, 240), fill=(92, 94, 96))
            image.save(image_path, quality=95)
            tasks = root / "tasks.csv"
            tasks.write_text(
                "task_id,image_path,anomaly_type,source_type\n"
                "task_0001,engine_room.jpg,water_leak,generated\n",
                encoding="utf-8",
            )

            code = vlm_wan_main(
                [
                    "--tasks",
                    str(tasks),
                    "--image-root",
                    str(image_root),
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    "--export-labelstudio",
                ]
            )
            self.assertEqual(code, 0)
            metadata_path = output_root / "metadata" / "sample_task_0001.json"
            sample = read_json(metadata_path)
            validate_sample_contract(sample)
            obj = sample["objects"][0]
            params = obj["geometry_detail"]["generation_params"]
            self.assertTrue(Path(obj["geometry_detail"]["mask_uri"]).exists())
            self.assertEqual(obj["geometry_detail"]["mask_format"], "png")
            self.assertNotEqual(
                [obj["box"]["x1"], obj["box"]["y1"], obj["box"]["x2"], obj["box"]["y2"]],
                params["expanded_edit_bbox"],
            )
            self.assertIn("background_preservation_score", params)
            self.assertIn("anomaly_visibility_score", params)
            self.assertEqual(obj["classification"]["multi_labels"][0]["label_value"], "water_leak")
            self.assertTrue((output_root / "metadata" / "import.json").exists())
            top_level_dirs = {path.name for path in output_root.iterdir() if path.is_dir()}
            self.assertEqual(top_level_dirs, {"crops", "masks", "metadata"})


if __name__ == "__main__":
    unittest.main()
