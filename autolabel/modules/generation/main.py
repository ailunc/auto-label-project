from __future__ import annotations

import argparse
import concurrent.futures
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from autolabel.utils import now_iso_shanghai, read_csv, write_json

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from autolabel.modules.generation.config import normalize_task_row, positive_float, positive_int, prepare_output_paths, resolve_task_image_path
    from autolabel.modules.generation.cropper import crop_with_expand
    from autolabel.modules.generation.diff_localizer import LocalizationThresholds, localize_difference
    from autolabel.modules.generation.exporter_labelstudio import export_metadata_dir
    from autolabel.modules.generation.grid import bbox_to_dict, expand_bbox, fine_grid_id_to_bbox, grid_id_to_bbox, make_fine_grid_preview, make_grid_preview
    from autolabel.modules.generation.metadata_builder import build_autolabel_sample
    from autolabel.modules.generation.prompts import build_negative_prompt, build_wan_edit_prompt
    from autolabel.modules.generation.quality import anomaly_visibility_score, background_preservation_score, passes_quality
    from autolabel.modules.generation.qwen_vlm_client import GridCandidate, GridSelection, QwenVLMClient
    from autolabel.modules.generation.utils import JsonlLogger
    from autolabel.modules.generation.validators import validate_autolabel_sample
    from autolabel.modules.generation.wan_image_client import WanImageClient
else:
    from .config import normalize_task_row, positive_float, positive_int, prepare_output_paths, resolve_task_image_path
    from .cropper import crop_with_expand
    from .diff_localizer import LocalizationThresholds, localize_difference
    from .exporter_labelstudio import export_metadata_dir
    from .grid import bbox_to_dict, expand_bbox, fine_grid_id_to_bbox, grid_id_to_bbox, make_fine_grid_preview, make_grid_preview
    from .metadata_builder import build_autolabel_sample
    from .prompts import build_negative_prompt, build_wan_edit_prompt
    from .quality import anomaly_visibility_score, background_preservation_score, passes_quality
    from .qwen_vlm_client import GridCandidate, GridSelection, QwenVLMClient
    from .utils import JsonlLogger
    from .validators import validate_autolabel_sample
    from .wan_image_client import WanImageClient


@dataclass
class EvaluatedGeneration:
    score: float
    sample: dict[str, Any]
    metadata_path: Path
    background_score: float
    visibility_score: float
    selected_grid: str
    generated_image_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM grid selection + Wan local image editing + diff AutoLabel pipeline.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--vlm-model", default="qwen3.6-plus")
    parser.add_argument("--image-model", default="wan2.7-image-pro")
    parser.add_argument("--vlm-request-image-max-side", type=lambda value: positive_int(value, "vlm-request-image-max-side"), default=768)
    parser.add_argument("--vlm-timeout-seconds", type=lambda value: positive_int(value, "vlm-timeout-seconds"), default=60)
    parser.add_argument("--vlm-json-retry-count", type=int, default=1)
    parser.add_argument("--disable-vlm-fallback", action="store_true")
    parser.add_argument("--wan-timeout-seconds", type=lambda value: positive_int(value, "wan-timeout-seconds"), default=300)
    parser.add_argument("--grid-layout", default="4x4")
    parser.add_argument("--edit-bbox-expand-ratio", type=lambda value: positive_float(value, "edit-bbox-expand-ratio"), default=0.20)
    parser.add_argument("--crop-expand-ratio", type=lambda value: positive_float(value, "crop-expand-ratio"), default=0.10)
    parser.add_argument("--num-generations-per-candidate", type=lambda value: positive_int(value, "num-generations-per-candidate"), default=1)
    parser.add_argument("--enable-fine-grid", action="store_true")
    parser.add_argument("--enable-vlm-review", action="store_true")
    parser.add_argument("--export-labelstudio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-candidate-grids", type=lambda value: positive_int(value, "max-candidate-grids"), default=3)
    parser.add_argument("--severity-level", default="early", choices=["early", "moderate", "obvious_but_controlled"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of task-level workers. Keep low to avoid API rate limits.")
    parser.add_argument(
        "--baseline-mode",
        default="ours_qwen_grid_wan_diff",
        choices=["baseline_random_grid", "baseline_no_diff_use_grid_bbox", "ours_qwen_grid_wan_diff"],
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    outputs = prepare_output_paths(args.output_root)
    logger = JsonlLogger(outputs.logs / f"run_{now_iso_shanghai().replace(':', '').replace('-', '').replace('+', '_')}.jsonl")
    qwen = QwenVLMClient(
        model_name=args.vlm_model,
        dry_run=args.dry_run,
        request_image_max_side=args.vlm_request_image_max_side,
        request_timeout_seconds=args.vlm_timeout_seconds,
        json_retry_count=args.vlm_json_retry_count,
        enable_grid_fallback=not args.disable_vlm_fallback,
    )
    wan = WanImageClient(model_name=args.image_model, dry_run=args.dry_run, request_timeout_seconds=args.wan_timeout_seconds)
    if not args.dry_run:
        wan_ok, wan_reason = wan.is_configured()
        if not wan_ok:
            logger.log("preflight", "wan_config", "failed", reason=wan_reason)
            logger.close()
            print(f"FAILED preflight: {wan_reason}", file=sys.stderr)
            return 1
        if args.disable_vlm_fallback and not qwen.api_key:
            reason = "Qwen API key is not configured and --disable-vlm-fallback was set."
            logger.log("preflight", "qwen_config", "failed", reason=reason)
            logger.close()
            print(f"FAILED preflight: {reason}", file=sys.stderr)
            return 1
        if not qwen.api_key:
            logger.log("preflight", "qwen_config", "warning", reason="Qwen API key is not configured; grid fallback will be used.")
    rows = read_csv(args.tasks)
    if args.limit is not None:
        rows = rows[: args.limit]
    success_count = 0
    try:
        worker_count = max(1, int(args.workers))
        if worker_count == 1:
            for index, raw_row in enumerate(rows, 1):
                task_id, success, error = process_raw_row(index, raw_row, args, outputs, qwen, wan, logger)
                success_count += int(success)
                if error:
                    print(f"FAILED {task_id}: {error}", file=sys.stderr)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_row = {
                    executor.submit(process_raw_row, index, raw_row, args, outputs, qwen, wan, logger): (index, raw_row)
                    for index, raw_row in enumerate(rows, 1)
                }
                for future in concurrent.futures.as_completed(future_to_row):
                    index, raw_row = future_to_row[future]
                    try:
                        task_id, success, error = future.result()
                    except Exception as exc:  # noqa: BLE001 - defensive guard; process_raw_row should catch row failures.
                        task_id = str(raw_row.get("task_id") or f"row_{index}")
                        success = False
                        error = str(exc)
                        logger.log(task_id, "task", "failed", reason=error, row=raw_row)
                    success_count += int(success)
                    if error:
                        print(f"FAILED {task_id}: {error}", file=sys.stderr)
        if args.export_labelstudio:
            export_path = outputs.labelstudio / "import.json"
            tasks = export_metadata_dir(outputs.metadata, export_path)
            print(f"Wrote {len(tasks)} Label Studio tasks to {export_path}")
    finally:
        logger.close()
    print(f"Completed {success_count}/{len(rows)} tasks. Output root: {outputs.root}")
    return 0 if success_count == len(rows) else 1


def process_raw_row(
    index: int,
    raw_row: dict[str, Any],
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
) -> tuple[str, bool, str | None]:
    try:
        task = normalize_task_row(raw_row, index)
        task["severity_level"] = raw_row.get("severity_level") or args.severity_level
        task["severity_level"] = task["severity_level"] or "early"
        if task["severity_level"] not in {"early", "moderate", "obvious_but_controlled"}:
            raise ValueError(f"Unsupported severity_level: {task['severity_level']}")
        success = process_task(task, args, outputs, qwen, wan, logger)
        return str(task["task_id"]), success, None
    except Exception as exc:  # noqa: BLE001 - row-level failure should be logged and continue.
        task_id = str(raw_row.get("task_id") or f"row_{index}")
        logger.log(task_id, "task", "failed", reason=str(exc), row=raw_row)
        return task_id, False, str(exc)


def process_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
) -> bool:
    task_id = task["task_id"]
    existing_metadata_path = outputs.metadata / f"{task.get('sample_id') or f'sample_{task_id}'}.json"
    if args.skip_existing and existing_metadata_path.exists():
        logger.log(task_id, "metadata", "skipped", metadata_path=str(existing_metadata_path))
        return True

    image_path = resolve_task_image_path(task, args.image_root)
    if not image_path.exists():
        logger.log(task_id, "input", "failed", reason="image not found", image_path=str(image_path))
        return False

    with Image.open(image_path) as source:
        image_width, image_height = source.size
    grid_preview = outputs.grid_previews / f"{task_id}_grid.jpg"
    make_grid_preview(image_path, grid_preview, args.grid_layout)
    logger.log(task_id, "grid_preview", "success", image_path=str(image_path), grid_preview=str(grid_preview))

    selection = qwen.select_grid(grid_preview, task["anomaly_type"], args.grid_layout)
    selection_status = "fallback" if selection.raw_response.get("fallback") else "success"
    logger.log(
        task_id,
        "vlm_grid_selection",
        selection_status,
        selected_grid=selection.selected_grid,
        candidate_grids=[candidate.__dict__ for candidate in selection.candidate_grids],
        fallback_reason=selection.raw_response.get("fallback_reason"),
    )

    best_overall: EvaluatedGeneration | None = None
    for candidate in selection.candidate_grids[: args.max_candidate_grids]:
        candidate_best = evaluate_candidate(
            task=task,
            candidate=candidate,
            selection=selection,
            original_image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            args=args,
            outputs=outputs,
            qwen=qwen,
            wan=wan,
            logger=logger,
        )
        if candidate_best is not None:
            best_overall = candidate_best
            break

    if best_overall is None:
        logger.log(task_id, "task", "failed", reason="all VLM candidate grids failed")
        return False

    if args.skip_existing and best_overall.metadata_path.exists():
        logger.log(task_id, "metadata", "skipped", metadata_path=str(best_overall.metadata_path))
        return True
    write_json(best_overall.metadata_path, best_overall.sample)
    logger.log(
        task_id,
        "task",
        "success",
        metadata_path=str(best_overall.metadata_path),
        generated_image_path=str(best_overall.generated_image_path),
        selected_grid=best_overall.selected_grid,
        background_preservation_score=best_overall.background_score,
        anomaly_visibility_score=best_overall.visibility_score,
    )
    return True


def evaluate_candidate(
    task: dict[str, Any],
    candidate: GridCandidate,
    selection: GridSelection,
    original_image_path: Path,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
) -> EvaluatedGeneration | None:
    task_id = task["task_id"]
    coarse_bbox = grid_id_to_bbox(candidate.grid, image_width, image_height, args.grid_layout)
    grid_bbox = coarse_bbox
    selected_grid_label = candidate.grid
    fine_selection_raw: dict[str, Any] | None = None
    if args.enable_fine_grid:
        fine_preview = outputs.grid_previews / f"{task_id}_{candidate.grid}_fine_grid.jpg"
        make_fine_grid_preview(original_image_path, coarse_bbox, fine_preview, "3x3")
        fine_selection = qwen.select_fine_grid(fine_preview, task["anomaly_type"], candidate.grid)
        fine_grid = fine_selection.candidate_grids[0].grid
        grid_bbox = fine_grid_id_to_bbox(fine_grid, coarse_bbox, "3x3")
        selected_grid_label = f"{candidate.grid}/{fine_grid}"
        fine_selection_raw = fine_selection.raw_response
        fine_status = "fallback" if fine_selection.raw_response.get("fallback") else "success"
        logger.log(task_id, "vlm_fine_grid_selection", fine_status, coarse_grid=candidate.grid, fine_grid=fine_grid)

    expanded_edit_bbox = expand_bbox(grid_bbox, image_width, image_height, args.edit_bbox_expand_ratio)
    prompt = build_wan_edit_prompt(task["anomaly_type"], selection.edit_region_hint or candidate.reason, task["severity_level"])
    negative_prompt = build_negative_prompt()
    thresholds = LocalizationThresholds.for_anomaly(task["anomaly_type"])
    best: EvaluatedGeneration | None = None

    for generation_index in range(1, args.num_generations_per_candidate + 1):
        suffix = f"{task_id}_{selected_grid_label.replace('/', '_')}_g{generation_index}"
        generated_path = outputs.generated_images / f"{suffix}.jpg"
        response_path = outputs.api_responses / f"{suffix}_wan_response.json"
        mask_path = outputs.masks / f"{suffix}_mask.png"
        crop_path = outputs.crops / f"{suffix}_crop.jpg"
        try:
            wan_result = wan.generate(
                original_image_path=original_image_path,
                bbox_list=[expanded_edit_bbox],
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=generated_path,
                response_path=response_path,
                anomaly_type=task["anomaly_type"],
            )
            with Image.open(generated_path) as generated_source:
                if generated_source.size != (image_width, image_height):
                    raise RuntimeError(f"generated image size mismatch: {generated_source.size} != {(image_width, image_height)}")
            localization = localize_difference(original_image_path, generated_path, expanded_edit_bbox, mask_path, thresholds)
            if not localization.success or localization.final_bbox is None or localization.mask_path is None:
                logger.log(task_id, "diff_localization", "failed", selected_grid=selected_grid_label, reason=localization.reason, metrics=localization.metrics)
                continue
            background_score = background_preservation_score(original_image_path, generated_path, expanded_edit_bbox)
            visibility_score = anomaly_visibility_score(localization.metrics)
            min_background_score = min_background_quality_score(task["anomaly_type"])
            quality_ok, quality_reason = passes_quality(background_score, visibility_score, min_background_score=min_background_score)
            if not quality_ok:
                logger.log(
                    task_id,
                    "quality",
                    "failed",
                    selected_grid=selected_grid_label,
                    reason=quality_reason,
                    background_preservation_score=background_score,
                    anomaly_visibility_score=visibility_score,
                    min_background_score=min_background_score,
                )
                continue
            crop_info = crop_with_expand(generated_path, localization.final_bbox, crop_path, args.crop_expand_ratio)
            review_payload = None
            review_score = None
            if args.enable_vlm_review:
                review = qwen.review_generation(crop_path, task["anomaly_type"])
                review_payload = review.raw_response
                review_score = review.score
                if not (review.is_valid and review.anomaly_type_match and review.location_reasonable and not review.visual_artifact):
                    logger.log(task_id, "vlm_review", "failed", selected_grid=selected_grid_label, review=review.raw_response)
                    continue
                visibility_score = anomaly_visibility_score(localization.metrics, review_score=review.score)
            combined_score = 0.50 * visibility_score + 0.35 * background_score + 0.15 * (review_score if review_score is not None else 1.0)
            generation_params = {
                "localization_pipeline": "vlm_grid_selection_wan_edit_diff_localization",
                "grid_layout": args.grid_layout,
                "enable_fine_grid": bool(args.enable_fine_grid),
                "selected_grid": selected_grid_label,
                "candidate_grids": [item.grid for item in selection.candidate_grids],
                "grid_bbox": list(map(int, grid_bbox)),
                "expanded_edit_bbox": list(map(int, expanded_edit_bbox)),
                "final_bbox_source": "image_difference_connected_components",
                "vlm_model": args.vlm_model,
                "image_generation_model": args.image_model,
                "severity_level": task["severity_level"],
                "prompt_version": "v1.0",
                "negative_prompt_version": "v1.0",
                "background_preservation_score": background_score,
                "anomaly_visibility_score": visibility_score,
                "min_background_score": min_background_score,
                "vlm_review_score": review_score,
                "wan_prompt": prompt,
                "negative_prompt": negative_prompt,
                "wan_input_image_uri": str(original_image_path),
                "wan_input_is_clean_original": True,
                "wan_response_path": str(response_path),
                "wan_raw_response": wan_result.raw_response,
                "qwen_grid_selection": selection.raw_response,
                "qwen_grid_selection_fallback": bool(selection.raw_response.get("fallback")),
                "qwen_fine_grid_selection": fine_selection_raw,
                "qwen_fine_grid_selection_fallback": bool(fine_selection_raw and fine_selection_raw.get("fallback")),
                "qwen_review": review_payload,
                "localization_metrics": localization.metrics,
                "baseline_mode": args.baseline_mode,
            }
            sample = build_autolabel_sample(
                task=task,
                generated_image_path=generated_path,
                original_image_path=original_image_path,
                image_width=image_width,
                image_height=image_height,
                final_bbox=localization.final_bbox,
                crop_info=crop_info,
                mask_path=localization.mask_path,
                generation_params=generation_params,
            )
            ok, errors = validate_autolabel_sample(sample)
            if not ok:
                logger.log(task_id, "required_fields_v1", "failed", selected_grid=selected_grid_label, errors=errors)
                continue
            logger.log(
                task_id,
                "candidate_generation",
                "success",
                selected_grid=selected_grid_label,
                generation_index=generation_index,
                final_bbox=bbox_to_dict(localization.final_bbox),
                mask_uri=str(localization.mask_path),
                background_preservation_score=background_score,
                anomaly_visibility_score=visibility_score,
            )
            sample_id = sample["sample_id"]
            metadata_path = outputs.metadata / f"{sample_id}.json"
            evaluated = EvaluatedGeneration(
                score=combined_score,
                sample=sample,
                metadata_path=metadata_path,
                background_score=background_score,
                visibility_score=visibility_score,
                selected_grid=selected_grid_label,
                generated_image_path=generated_path,
            )
            if best is None or evaluated.score > best.score:
                best = evaluated
        except Exception as exc:  # noqa: BLE001 - candidate fallback continues after logging.
            logger.log(task_id, "candidate_generation", "failed", selected_grid=selected_grid_label, generation_index=generation_index, reason=str(exc))
    return best


def min_background_quality_score(anomaly_type: str) -> float:
    if anomaly_type == "water_leak":
        return 0.74
    return 0.78


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
