import os
import re
import argparse
from pathlib import Path


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


def extract_fields(full_text: str) -> dict:
    results = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = pattern.search(full_text)
        if match:
            # 有捕获组则取第1组，否则取整个匹配
            results[field] = match.group(1).strip() if match.lastindex else match.group(0).strip()
    return results


def run_ocr_and_extract(img_path: str, output_dir: str = "./output"):
    from paddleocr import PaddleOCR

    os.makedirs(output_dir, exist_ok=True)

    # use_angle_cls=True 处理旋转文字（包装上常见）
    ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
    result = ocr.predict(img_path)

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

    print("\n=== 字段提取结果 ===")
    fields = extract_fields(full_text)
    if fields:
        for field, value in fields.items():
            print(f"  {field}: {value}")
    else:
        print("  未匹配到已知字段（可根据实际文本扩展 FIELD_PATTERNS）")

    # 保存原始 OCR 文本
    out_file = Path(output_dir) / f"{Path(img_path).stem}_ocr.txt"
    out_file.write_text(full_text, encoding="utf-8")
    print(f"\n原始文本已保存: {out_file}")

    return {"text": full_text, "fields": fields}


def main():
    parser = argparse.ArgumentParser(description="食品包装标签识别 (路线A: OCR + 规则提取)")
    parser.add_argument("img", nargs="?", help="图片路径")
    parser.add_argument("--output", "-o", default="./output", help="输出目录 (默认: ./output)")
    args = parser.parse_args()

    if not args.img:
        print("用法: uv run main.py <图片路径>")
        print("示例: uv run main.py label.jpg")
        return

    run_ocr_and_extract(args.img, args.output)


if __name__ == "__main__":
    main()
