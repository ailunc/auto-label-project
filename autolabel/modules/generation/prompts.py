from __future__ import annotations

ANOMALY_SELECTION_HINTS = {
    "diesel_leak": "fuel pipe, yellow metal supply pipe, fuel flange, valve joint, generator lower side, equipment-floor boundary",
    "oil_leak": "engine block, bolt flange, sealing cover edge, oil cooler connection, middle or upper-middle metal connection structures",
    "coolant_leak": "black rubber hose, silver clamp, radiator connection pipe, cooling circuit hose joint, engine-radiator connection, middle-lower hose area",
    "water_leak": (
        "water pipe, condensate pipe, drain pipe, pipe joint, valve, metal connector, equipment lower side, "
        "base edge, drainage path, equipment-floor boundary, low floor area where clear water can drip and spread"
    ),
}

ANOMALY_WAN_APPEARANCE = {
    "diesel_leak": "clear transparent light yellow or amber liquid, low viscosity, water-like flow, thin reflective wet film on gray epoxy floor",
    "oil_leak": "deep black or black-brown, opaque, high viscosity, poor flowability, local sticky oil stain attached to engine block, flange, or seal area",
    "coolant_leak": "silver-green or light silver-green transparent liquid, clear and bright but not sci-fi fluorescent, low viscosity, reflective green liquid film",
    "water_leak": (
        "pure white, transparent whitish, clean clear water with bright highlights, high-reflection thin water film, "
        "no yellow diesel tint, no silver-green coolant tint, no black oil stain, no fluorescent color"
    ),
}

SEVERITY_HINTS = {
    "early": "very small early-stage trace, subtle but visible, controlled scale",
    "moderate": "moderate local leakage, visible but still controlled and localized",
    "obvious_but_controlled": "obvious localized leakage, still controlled, no disaster scene, no broad spill",
}


def build_qwen_grid_prompt(anomaly_type: str) -> str:
    if anomaly_type == "water_leak":
        return """
你是一名工业巡检图像分析专家。现在给你一张已经叠加 4x4 网格编号的工业机房监控图像，网格编号从左到右、从上到下依次为：

A1 A2 A3 A4
B1 B2 B3 B4
C1 C2 C3 C4
D1 D2 D3 D4

你的任务是判断哪一个网格区域最适合进行“漏水 / 清水泄漏”异常图像编辑。

漏水异常的合理发生位置通常包括：
1. 设备下部靠近水管、冷凝水管、排水管、接口、阀门或接头的位置；
2. 管路与设备外壳、底座、墙面或地面交界的位置；
3. 机组底座边缘、地面低洼处或液体可能滴落并扩散的位置；
4. 靠近管道支架、软管连接、金属接头、排水口或设备冷凝水排放路径的区域；
5. 能够表现清水沿设备表面、管路下缘或底座边缘向下流动，并在灰色环氧地坪形成湿痕的位置。

漏水异常不应发生在：
1. 天花板、远处墙面、纸箱、蓝色托盘等无关背景区域；
2. 完全远离设备、管路、接头和底座的普通地面中央；
3. 发电机顶部排气管、空气滤清器等不适合产生清水泄漏的位置；
4. 没有任何设备结构、管路结构或液体合理来源的区域。

请优先选择能够同时包含“可能漏水起点”和“水可能向下流动或在地面形成湿痕”的网格。如果一个网格只包含普通地面但没有设备连接部位，不应优先选择。

请严格输出 JSON，不要输出额外解释文字。必须输出 top-3 candidate_grids。

输出格式：
{
  "selected_grid": "C2",
  "confidence": 0.86,
  "candidate_grids": [
    {
      "grid": "C2",
      "score": 0.86,
      "reason": "该区域靠近设备下部、管路接头或底座边缘，并且下方有地面可形成清水湿痕，适合生成漏水异常。"
    },
    {
      "grid": "C3",
      "score": 0.73,
      "reason": "该区域可能包含底座边缘和地面扩散区域，可作为清水流动或湿痕扩散的备选区域。"
    },
    {
      "grid": "B2",
      "score": 0.62,
      "reason": "该区域包含部分设备结构，但与地面湿痕形成路径不如首选区域明确。"
    }
  ],
  "edit_region_hint": "在所选网格内靠近管路接头、设备底部、底座边缘或排水路径的位置生成少量清水泄漏，并让水沿设备或地面自然形成纯白透明的湿痕。",
  "risk_note": "漏水应表现为低压、微量、清澈、纯白透明偏高亮的水迹，不应表现为黄色油液、绿色防冻液、黑色机油、喷射、爆裂或大面积洪流。",
  "rejected_region_reason": "避开天花板、远处墙面、纸箱、托盘、普通空地、无管路或无设备连接结构的区域。"
}
""".strip()
    hint = ANOMALY_SELECTION_HINTS[anomaly_type]
    return f"""
You are an industrial inspection image analysis expert. You receive a clean industrial image preview with a visible 4x4 grid and grid IDs A1-D4.
Select the top-3 grid cells most suitable for realistic {anomaly_type} image editing.
Prefer realistic industrial anomaly locations: {hint}.
Do not select wall, ceiling, paper boxes, pallets, ordinary empty floor, or unrelated background unless no better region exists.
Return strict JSON only, with no markdown and no extra text. The JSON schema is:
{{
  "selected_grid": "C2",
  "confidence": 0.86,
  "candidate_grids": [
    {{"grid": "C2", "score": 0.86, "reason": "..."}},
    {{"grid": "C3", "score": 0.74, "reason": "..."}},
    {{"grid": "B2", "score": 0.62, "reason": "..."}}
  ],
  "edit_region_hint": "...",
  "risk_note": "...",
  "rejected_region_reason": "..."
}}
Always output exactly three candidate_grids when possible. Scores must be between 0 and 1.
""".strip()


def build_qwen_fine_grid_prompt(anomaly_type: str, coarse_grid: str) -> str:
    if anomaly_type == "water_leak":
        return f"""
你正在对粗网格 {coarse_grid} 内的“漏水 / 清水泄漏”编辑区域做 3x3 局部细选。局部网格编号为 A1-C3。
请选择最适合生成微量清水泄漏的位置，优先靠近水管、冷凝水管、排水管、阀门、接头、设备底部、底座边缘或排水路径。
细选区域最好同时包含可能漏水起点，以及清水向下流动或在灰色环氧地坪形成纯白透明高亮湿痕的空间。
不要选择墙面、天花板、普通空地中央、纸箱、托盘、无设备结构或无液体来源的位置。
请严格输出 JSON，不要输出额外解释文字，格式为：
{{
  "selected_grid": "B2",
  "confidence": 0.82,
  "candidate_grids": [
    {{"grid": "B2", "score": 0.82, "reason": "靠近设备底部或管路接头，且下方有清水湿痕扩散空间。"}},
    {{"grid": "B1", "score": 0.68, "reason": "包含部分可能漏水起点，但地面扩散路径较弱。"}},
    {{"grid": "C2", "score": 0.61, "reason": "适合形成地面清水湿痕，但漏水起点不如首选明确。"}}
  ],
  "edit_region_hint": "在局部网格内靠近接头、阀门、底座边缘或排水路径处生成纯白透明偏高亮的清水水迹。",
  "risk_note": "不要生成黄色柴油、绿色防冻液、黑褐色机油、荧光液体、高压喷水或大面积洪水。",
  "rejected_region_reason": "避开无管路、无接头、无设备底部结构、无法形成清水滴落路径的位置。"
}}
""".strip()
    hint = ANOMALY_SELECTION_HINTS[anomaly_type]
    return f"""
You are refining a coarse grid selection for {anomaly_type}. The image is a crop from coarse grid {coarse_grid} with a local 3x3 grid A1-C3.
Select the best local fine grid for a small realistic anomaly near: {hint}.
Avoid plain wall, ceiling, unrelated empty floor, boxes, labels, and visual clutter.
Return strict JSON only with this schema:
{{
  "selected_grid": "B2",
  "confidence": 0.82,
  "candidate_grids": [
    {{"grid": "B2", "score": 0.82, "reason": "..."}},
    {{"grid": "B1", "score": 0.68, "reason": "..."}},
    {{"grid": "C2", "score": 0.61, "reason": "..."}}
  ],
  "edit_region_hint": "...",
  "risk_note": "...",
  "rejected_region_reason": "..."
}}
""".strip()


def build_wan_edit_prompt(anomaly_type: str, edit_region_hint: str, severity_level: str = "early") -> str:
    if anomaly_type == "water_leak":
        severity = SEVERITY_HINTS.get(severity_level, SEVERITY_HINTS["early"])
        return f"""
请基于输入的原始工业监控图像进行局部图像编辑，只在指定的 bbox_list 区域附近生成“漏水 / 清水泄漏”异常，其余区域必须严格保持与原图一致。

整体画面必须保持原图的固定 CCTV 监控视角、工业机房布局、设备位置、背景物体、照明条件、监控画面质感和空间关系。不得改变主体设备结构，不得改变机房内原有设备和背景元素，不得新增人物、车辆、工具、文字、箭头、红框、标注框、网格线或网格编号。不要生成高压喷涌、大规模爆裂漏水、洪水、烟雾、火焰、爆炸、蒸汽或任何灾难化效果。

异常类型为漏水 / 清水泄漏。当前严重程度为：{severity}。

漏水应发生在指定 bbox_list 区域内最合理的管路、接头、阀门、设备底部、底座边缘或排水路径附近。区域提示：{edit_region_hint or '靠近水管、冷凝水管、排水管、接口、阀门、设备底部或地面低洼处'}。该异常属于微量、持续、低压的早期清水泄漏，不应表现为高压喷射，也不应表现为大面积管道爆裂或严重积水事故。

泄漏出的液体为清水。水必须表现为纯白色、透明偏白、清澈、干净、具有高亮反光的液体。整体颜色应接近冷白色灯光下的清水湿痕，而不是黄色柴油、银绿色防冻液、黑褐色机油或任何荧光液体。水的粘度极低，流动特性与普通清水一致。

清水应先从管路接头、阀门缝隙、设备下缘、底座边缘或排水路径附近细微渗出。随后水沿着设备表面、管道下缘或底座边缘形成一条细小、清澈、纯白透明的水迹。水迹可以在最低点形成少量液滴，并自然滴落到灰色环氧地坪上，但整体应保持微量、持续、受控的早期漏水状态。

如果水滴落到灰色环氧地坪，应形成半透明、纯白偏亮、带明显镜面反光的湿润区域。由于清水流动性强，地面水迹应像薄薄的水膜一样铺展，边缘自然、不规则、厚度很薄。水膜本身应接近无色透明，但在冷白色顶部灯光反射下呈现纯白色高亮反光，能够破坏原本地面反光的一致性。

漏水区域不能呈现油腻感，不能带黄色、绿色、黑色或彩虹油膜效果。水中不应包含明显杂质、泡沫、泥污或浑浊感。最终结果应突出“纯白透明清水、低压微量泄漏、局部水迹、地面高亮反光湿痕”的工业异常特征。

请确保异常只出现在指定 bbox_list 区域内部或其非常邻近位置。除漏水水迹、水滴和地面湿痕之外，所有设备、背景、警戒线、地面纹理、时间戳和机房环境必须尽量保持与原图一致。
""".strip()
    appearance = ANOMALY_WAN_APPEARANCE[anomaly_type]
    severity = SEVERITY_HINTS.get(severity_level, SEVERITY_HINTS["early"])
    return f"""
Edit only near bbox_list. Keep the original CCTV view, device layout, background, lighting, timestamp, and camera perspective unchanged.
Generate a realistic {severity} {anomaly_type}. Physical appearance: {appearance}.
Preferred local context: {edit_region_hint or 'industrial pipe, flange, hose, valve, seal, or equipment-floor boundary'}.
Do not change non-anomaly regions. Do not add people, vehicles, arrows, labels, red/yellow boxes, grid lines, grid IDs, masks, or text.
Do not generate fire, smoke, explosion, vapor, spark, high-pressure spray, large splash, or disaster scene.
The anomaly must be localized around bbox_list and must not affect unrelated background.
""".strip()


def build_negative_prompt() -> str:
    return (
        "people, vehicles, red box, yellow box, bounding box, arrow, label, text, grid line, grid id, mask overlay, "
        "segmentation color, fire, smoke, explosion, vapor, spark, high-pressure spray, large splash, disaster scene, "
        "changed camera angle, changed machine layout, rewritten background, unrealistic liquid, sci-fi glow, "
        "yellow oil liquid, light yellow diesel, amber oil, silver-green coolant, green liquid, black engine oil, "
        "black-brown oil stain, fluorescent liquid, sci-fi glowing liquid, rainbow oil film, large-scale flood, "
        "high-pressure water spray, water jet, pipe burst, foam, muddy water, turbid sewage, detection box, text label, "
        "黄色油液，淡黄色柴油，琥珀色油液，银绿色防冻液，绿色液体，黑色机油，黑褐色油污，荧光液体，"
        "科幻发光液体，彩虹油膜，大面积洪水，高压喷水，水柱喷射，管道爆裂，泡沫，泥水，浑浊污水，"
        "红框，检测框，箭头，网格线，网格编号，文字标签"
    )


def build_retry_visibility_prompt(anomaly_type: str) -> str:
    return (
        f"Make the {anomaly_type} slightly more visible while keeping it early-stage, controlled, local, and realistic. "
        "Do not expand outside bbox_list and do not alter the background."
    )


def build_qwen_review_prompt(anomaly_type: str) -> str:
    return f"""
You are a strict industrial anomaly quality reviewer. Check whether the image contains a realistic, early-stage, controlled {anomaly_type}.
Verify: anomaly type match, reasonable industrial location, no grid lines, no red/yellow boxes, no text, no arrows, no masks, and no severe disaster artifacts.
Return strict JSON only:
{{
  "is_valid": true,
  "anomaly_type_match": true,
  "location_reasonable": true,
  "visual_artifact": false,
  "score": 0.87,
  "reason": "..."
}}
""".strip()
