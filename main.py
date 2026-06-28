import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from statistics import median
from PIL import Image


# 食品包装标签字段提取规则
FIELD_PATTERNS = {
    "生产许可证": re.compile(r"SC\s*[\dA-Z]{14,}", re.IGNORECASE),
    "产地": re.compile(r"(?:产\s*地|origin)[：:]\s*(.+)"),
    "生产日期": re.compile(r"(?:生产日期|生产时间|制造日期)[：:\s]*(\d{4}[-./年]\d{1,2}[-./月]\d{1,2})"),
    "保质期": re.compile(r"(?:保质期|shelf\s*life)[：:]\s*(.+)"),
    "净含量": re.compile(r"(?:净含量|净重)[：:]\s*([\d.]+\s*(?:g|kg|ml|L|克|千克|毫升|升))"),
    "配料": re.compile(r"(?:配料|成分|配料表|原料|ingredients)[：:]\s*(.{10,}?)(?=\n|$|执行|生产|贮存|储存)"),
    "执行标准": re.compile(r"(?:执行标准|产品标准)[：:]\s*([A-Z/\d.]+)"),
    "贮存条件": re.compile(r"(?:贮存条件|储存方法|贮存方法)[：:]\s*(.+)"),
}

# 区域检测调参
GAP_MIN_WIDTH_RATIO = 0.05
LINE_MERGE_THRESHOLD_RATIO = 1.5
BARCODE_MAX_CHARS = 10
LABEL_KEYWORD_MIN_HITS = 2

LABEL_KEYWORDS = [
    "配料", "成分", "原料", "保质期", "shelf life", "产地", "origin",
    "SC", "生产许可", "执行标准", "产品标准", "贮存条件", "储存方法",
    "净含量", "净重", "生产日期", "生产时间",
]


def extract_fields(full_text: str) -> dict:
    results = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = pattern.search(full_text)
        if match:
            # 有捕获组则取第1组，否则取整个匹配
            results[field] = match.group(1).strip() if match.lastindex else match.group(0).strip()
    return results


def poly_to_bbox(poly) -> tuple:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def find_column_splits(bboxes: list, img_width: int, gap_min_width: int) -> list:
    histogram = [0] * (img_width + 1)
    for x0, _, x1, _ in bboxes:
        for x in range(max(0, x0), min(img_width + 1, x1 + 1)):
            histogram[x] += 1

    splits = []
    gap_start = None
    for x, count in enumerate(histogram):
        if count == 0:
            if gap_start is None:
                gap_start = x
        else:
            if gap_start is not None:
                gap_width = x - gap_start
                if gap_width >= gap_min_width:
                    splits.append((gap_start + x) // 2)
                gap_start = None
    return splits


def group_blocks_into_columns(blocks: list, split_xs: list) -> list:
    if not split_xs:
        return [blocks]

    boundaries = [0] + split_xs + [float("inf")]
    columns = [[] for _ in range(len(boundaries) - 1)]
    for block in blocks:
        cx = (block["bbox"][0] + block["bbox"][2]) // 2
        for i, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
            if lo <= cx < hi:
                columns[i].append(block)
                break
    return [col for col in columns if col]


def merge_column_into_sections(column_blocks: list, threshold_ratio: float) -> list:
    sorted_blocks = sorted(column_blocks, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)

    heights = [(b["bbox"][3] - b["bbox"][1]) for b in sorted_blocks]
    med_height = median(heights) if heights else 20
    threshold = threshold_ratio * med_height

    sections = []
    current = [sorted_blocks[0]]
    for block in sorted_blocks[1:]:
        prev_bottom = current[-1]["bbox"][3]
        curr_top = block["bbox"][1]
        if curr_top - prev_bottom <= threshold:
            current.append(block)
        else:
            sections.append(current)
            current = [block]
    sections.append(current)

    result = []
    for group in sections:
        x0 = min(b["bbox"][0] for b in group)
        y0 = min(b["bbox"][1] for b in group)
        x1 = max(b["bbox"][2] for b in group)
        y1 = max(b["bbox"][3] for b in group)
        result.append({"bbox": (x0, y0, x1, y1), "blocks": group})
    return result


def classify_section(section: dict, is_leftmost: bool) -> str:
    joined = " ".join(b["text"] for b in section["blocks"])

    if "营养成分表" in joined:
        return "nutrition"

    hits = sum(1 for kw in LABEL_KEYWORDS if kw in joined)
    if hits >= LABEL_KEYWORD_MIN_HITS:
        return "label"

    char_count = len(joined.replace(" ", ""))
    x0, y0, x1, y1 = section["bbox"]
    aspect = (y1 - y0) / max(x1 - x0, 1)
    if char_count <= BARCODE_MAX_CHARS and aspect > 2.5:
        return "barcode"

    if is_leftmost:
        return "front"
    return "other"


def detect_sections(polys: list, texts: list, scores: list) -> list:
    bboxes = [poly_to_bbox(p) for p in polys]
    blocks = [
        {"text": t, "score": s, "bbox": bb}
        for t, s, bb in zip(texts, scores, bboxes)
    ]

    if not blocks:
        return []

    img_width = max(bb[2] for bb in bboxes)
    gap_min_width = max(1, int(img_width * GAP_MIN_WIDTH_RATIO))

    split_xs = find_column_splits(bboxes, img_width, gap_min_width)
    columns = group_blocks_into_columns(blocks, split_xs)

    # determine leftmost column x0
    col_x0s = [min(b["bbox"][0] for b in col) for col in columns]
    leftmost_x0 = min(col_x0s) if col_x0s else 0

    sections = []
    for col in columns:
        is_leftmost = min(b["bbox"][0] for b in col) == leftmost_x0
        raw_sections = merge_column_into_sections(col, LINE_MERGE_THRESHOLD_RATIO)
        for sec in raw_sections:
            sec_type = classify_section(sec, is_leftmost)
            lines = [b["text"] for b in sec["blocks"]]
            section_text = "\n".join(lines)
            fields = extract_fields(section_text) if sec_type in ("label", "nutrition") else {}
            sections.append({
                "type": sec_type,
                "bbox": sec["bbox"],
                "lines": lines,
                "fields": fields,
            })

    sections.sort(key=lambda s: s["bbox"][0])
    return sections


MAX_SIDE = 3000


def preprocess_image(img_path: str) -> np.ndarray:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def run_ocr_and_extract(img_path: str, output_dir: str = "./output",
                        section_filter: str = "all", output_json: bool = False):
    from paddleocr import PaddleOCR

    os.makedirs(output_dir, exist_ok=True)

    img = preprocess_image(img_path)
    ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
    result = ocr.predict(img)

    if not result:
        print("未识别到文字")
        return

    res = result[0]
    texts = res.get("rec_texts", [])
    polys = res.get("rec_polys", [])
    scores = res.get("rec_scores", [])

    if not texts:
        print("未识别到文字")
        return

    # 按 bbox 顶部 y 坐标排序
    lines = sorted(zip(polys, texts, scores), key=lambda x: x[0][0][1])
    text_lines = [text for _, text, _ in lines]
    full_text = "\n".join(text_lines)

    print("=== OCR 识别文本 ===")
    for line in text_lines:
        print(f"  {line}")

    # 区域分段
    sections = detect_sections(polys, texts, scores)

    print("\n=== 区域分段结果 ===")
    for sec in sections:
        if section_filter != "all" and sec["type"] != section_filter:
            continue
        x0, y0, x1, y1 = sec["bbox"]
        print(f"  [{sec['type']}]  bbox=({x0},{y0},{x1},{y1})  行数={len(sec['lines'])}")
        if sec["fields"]:
            for field, value in sec["fields"].items():
                print(f"    {field}: {value}")
        else:
            for line in sec["lines"][:3]:
                print(f"    {line}")
            if len(sec["lines"]) > 3:
                print(f"    ... 共 {len(sec['lines'])} 行")

    print("\n=== 字段提取结果 ===")
    fields = extract_fields(full_text)
    if fields:
        for field, value in fields.items():
            print(f"  {field}: {value}")
    else:
        print("  未匹配到已知字段（可根据实际文本扩展 FIELD_PATTERNS）")

    # 保存原始 OCR 文本
    stem = Path(img_path).stem
    out_file = Path(output_dir) / f"{stem}_ocr.txt"
    out_file.write_text(full_text, encoding="utf-8")
    print(f"\n原始文本已保存: {out_file}")

    if output_json:
        json_data = {
            "sections": [
                {
                    "type": s["type"],
                    "bbox": list(s["bbox"]),
                    "lines": s["lines"],
                    "fields": s["fields"],
                }
                for s in sections
                if section_filter == "all" or s["type"] == section_filter
            ]
        }
        json_file = Path(output_dir) / f"{stem}_sections.json"
        json_file.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"分段结果已保存: {json_file}")

    return {"text": full_text, "fields": fields, "sections": sections}


def main():
    parser = argparse.ArgumentParser(description="食品包装标签识别 (路线A: OCR + 规则提取)")
    parser.add_argument("img", nargs="?", help="图片路径")
    parser.add_argument("--output", "-o", default="./output", help="输出目录 (默认: ./output)")
    parser.add_argument(
        "--section", "-s",
        choices=["label", "nutrition", "barcode", "front", "other", "all"],
        default="all",
        help="只显示/保存指定区域 (默认: all)",
    )
    parser.add_argument("--json", action="store_true", help="输出分段 JSON 文件")
    args = parser.parse_args()

    if not args.img:
        print("用法: uv run main.py <图片路径>")
        print("示例: uv run main.py label.jpg")
        return

    run_ocr_and_extract(args.img, args.output, section_filter=args.section, output_json=args.json)


if __name__ == "__main__":
    main()
