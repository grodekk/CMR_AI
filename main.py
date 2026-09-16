import argparse
import json
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from datetime import datetime, timezone
from paddleocr import PaddleOCR


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare_image(source, output):
    original = output / f"original{source.suffix.lower()}"
    shutil.copy2(source, original)

    with Image.open(original) as opened:
        if opened.format not in {"JPEG", "PNG"}:
            raise ValueError("The file must contain a JPEG or PNG image.")

        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.save(output / "input.png")

    return image, original


def extract_fragments(result):
    fragments = []

    ocr_items = zip(
        result["rec_texts"],
        result["rec_scores"],
        result["rec_polys"],
        strict=True,
    )

    for index, (text, score, polygon) in enumerate(ocr_items, start=1):
        fragments.append({
            "id": index,
            "text": str(text),
            "score": float(score),
            "polygon": [[float(x), float(y)] for x, y in polygon],
        })

    return fragments


def draw_preview(image, fragments, output, font_path):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    font_size = max(12, min(28, image.width // 70))
    font = ImageFont.truetype(str(font_path), size=font_size)

    for fragment in fragments:
        points = [tuple(point) for point in fragment["polygon"]]

        draw.polygon(points, outline="red", width=2)

        label = (
            f'{fragment["id"]}: '
            f'{fragment["text"]} '
            f'({fragment["score"]:.2f})'
        )

        x = int(min(point[0] for point in points))
        y = int(min(point[1] for point in points))

        draw.text(
            (x, max(0, y - font_size)),
            label,
            font=font,
            fill="blue",
            stroke_width=2,
            stroke_fill="white",
        )

    preview.save(output / "preview.png")


def run(args):
    source = args.image.expanduser().resolve(strict=True)

    if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Provide a JPG, JPEG, or PNG image.")

    if not args.font.is_file():
        raise ValueError(f"Font not found: {args.font}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() / timestamp
    output.mkdir(parents=True)

    print(f"Output directory: {output}")

    image, original = prepare_image(source, output)

    ocr = PaddleOCR(
        lang=args.lang,
        ocr_version="PP-OCRv5",
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=0.0,
    )

    image_array = np.asarray(image)
    results = list(ocr.predict(image_array))

    if len(results) != 1:
        raise RuntimeError(
            f"Expected one result, got: {len(results)}"
        )

    result = results[0]
    result.save_to_json(str(output / "raw_0.json"))

    fragments = extract_fragments(result)

    result_data = {
        "source_name": source.name,
        "original": original.name,
        "image": "input.png",
        "preview": "preview.png",
        "width": image.width,
        "height": image.height,
        "fragments": fragments,
    }

    write_json(output / "result.json", result_data)
    draw_preview(image, fragments, output, args.font)

    print(
        f"Recognized {len(fragments)} fragments. "
        f"Preview: {output / 'preview.png'}"
    )


def run(args):
    source = args.image.expanduser().resolve(strict=True)

    if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Provide a JPG, JPEG, or PNG image.")

    if not args.font.is_file():
        raise ValueError(f"Font not found: {args.font}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() / timestamp
    output.mkdir(parents=True)

    print(f"Output directory: {output}")

    image, original = prepare_image(source, output)

    ocr = PaddleOCR(
        lang=args.lang,
        ocr_version="PP-OCRv5",
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=0.0,
    )

    image_array = np.asarray(image)
    results = list(ocr.predict(image_array))

    if len(results) != 1:
        raise RuntimeError(
            f"Expected one result, got: {len(results)}"
        )

    result = results[0]
    result.save_to_json(str(output / "raw_0.json"))

    fragments = extract_fragments(result)

    result_data = {
        "source_name": source.name,
        "original": original.name,
        "image": "input.png",
        "preview": "preview.png",
        "width": image.width,
        "height": image.height,
        "fragments": fragments,
    }

    write_json(output / "result.json", result_data)
    draw_preview(image, fragments, output, args.font)

    print(
        f"Recognized {len(fragments)} fragments. "
        f"Preview: {output / 'preview.png'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Local OCR for a single CMR image."
    )

    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--device",
        choices=("gpu:0", "cpu"),
        default="gpu:0",
    )
    parser.add_argument("--lang", default="pl")
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("C:/Windows/Fonts/arial.ttf"),
    )

    args = parser.parse_args()

    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        parser.error(
            "Python 3.10–3.13 is required; 3.12 is recommended."
        )

    run(args)


if __name__ == "__main__":
    main()