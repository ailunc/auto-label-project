# AutoLabel Generation Module

This folder contains the industrial anomaly image-generation and auto-label pipeline for `autolabel`.
Run it from the repository root with the package entrypoint; the old `vlm_wan_autolabel/src/main.py` entry remains as a compatibility wrapper.

## Install

```bash
python -m pip install -r autolabel/modules/generation/requirements.txt
```

## Environment

Do not hardcode API keys. The CLI automatically loads `.env` from the repository root and `autolabel/modules/generation/.env` before reading environment variables. Current shell exports still take precedence over `.env` values.

Create a local `.env` file, which is ignored by git:

```bash
DASHSCOPE_API_KEY=your_api_key_here
QWEN_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_WAN_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

If live Wan editing is unavailable, use `--dry-run` to validate grid generation, dummy Qwen selection, local synthetic edit, diff localization, mask/crop writing, metadata construction, and RequiredFieldsV1 validation.

## Supported Anomaly Types

```text
diesel_leak
oil_leak
coolant_leak
water_leak
```

`water_leak` is clear-water leakage: pure white / transparent whitish / bright reflective clean water. It must not look like yellow diesel, silver-green coolant, black-brown oil, fluorescent liquid, muddy water, foam, high-pressure spray, or flooding.

## Run

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks_water_leak_test.csv \
  --image-root data/raw/images \
  --output-root data/processed \
  --vlm-model qwen3.6-plus \
  --image-model wan2.7-image-pro \
  --grid-layout 4x4 \
  --edit-bbox-expand-ratio 0.20 \
  --crop-expand-ratio 0.10 \
  --num-generations-per-candidate 1
```

Dry-run:

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks_water_leak_test.csv \
  --image-root data/raw/images \
  --output-root data/processed \
  --vlm-model qwen3.6-plus \
  --image-model wan2.7-image-pro \
  --grid-layout 4x4 \
  --edit-bbox-expand-ratio 0.20 \
  --crop-expand-ratio 0.10 \
  --num-generations-per-candidate 1 \
  --dry-run \
  --export-labelstudio
```
