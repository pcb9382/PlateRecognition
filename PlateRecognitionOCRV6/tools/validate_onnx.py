#!/usr/bin/env python3
"""Run PP-OCRv6 ONNX with the same embedded sparse mappings as C++."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort


TIERS = ("tiny", "small", "medium")
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[2]
    / "PlateRecognition"
    / "test"
    / "data"
    / "苏A0GY63_006235.jpg"
)
INPUT_HEIGHT = 48
MIN_WIDTH = 160
OPT_WIDTH = 320
MAX_WIDTH = 3200

CPP_MAPPING_SYMBOLS = {
    "kTinyClassCount": "kTinyClassIndexByPlateCharacter",
    "kFullClassCount": "kFullClassIndexByPlateCharacter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PP-OCRv6 ONNX and print only PlateRecognition text."
    )
    parser.add_argument("--tier", choices=TIERS, default="medium")
    parser.add_argument("--all-tiers", action="store_true")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"image is empty or unreadable: {path}")
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV could not decode image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"unsupported image dimensions: {image.shape}")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported channel count: {image.shape[2]}")


def load_plate_characters() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    header = (repo_root / "common" / "PlateRecognitionCharset.h").read_text(
        encoding="utf-8"
    )
    marker = "static const std::vector<std::string> characters"
    start = header.find(marker)
    end = header.find("};", start)
    if start < 0 or end < 0:
        raise ValueError("could not parse common/PlateRecognitionCharset.h")
    characters = re.findall(r'"([^"\\]*)"', header[start:end])
    if len(characters) != 77 or len(set(characters)) != 77:
        raise ValueError("PlateRecognition visible alphabet must have 77 entries")
    return characters


def load_cpp_mappings() -> dict[int, tuple[int, ...]]:
    module_root = Path(__file__).resolve().parents[1]
    source = (module_root / "src" / "OCRV6Text.cpp").read_text(encoding="utf-8")
    header = (module_root / "src" / "OCRV6Text.h").read_text(encoding="utf-8")
    mappings: dict[int, tuple[int, ...]] = {}
    for count_symbol, mapping_symbol in CPP_MAPPING_SYMBOLS.items():
        count_match = re.search(
            rf"\b{re.escape(count_symbol)}\s*=\s*(\d+)U?\s*;", header
        )
        mapping_match = re.search(
            rf"\b{re.escape(mapping_symbol)}\s*\[\s*\]\s*=\s*"
            rf"\{{(?P<body>.*?)\}};",
            source,
            flags=re.DOTALL,
        )
        if count_match is None or mapping_match is None:
            raise ValueError(f"could not parse C++ contract for {count_symbol}")
        class_count = int(count_match.group(1))
        indices = tuple(
            int(value)
            for value in re.findall(r"\d+", mapping_match.group("body"))
        )
        if (
            len(indices) != 77
            or len(set(indices)) != 77
            or any(index <= 0 or index >= class_count for index in indices)
        ):
            raise ValueError(f"invalid C++ sparse mapping: {mapping_symbol}")
        mappings[class_count] = indices
    return mappings


def preprocess_bgr(image: np.ndarray) -> np.ndarray:
    ratio_width = int(np.ceil(INPUT_HEIGHT * image.shape[1] / image.shape[0]))
    content_width = max(1, min(MAX_WIDTH, ratio_width))
    canvas_width = max(MIN_WIDTH, min(MAX_WIDTH, max(OPT_WIDTH, content_width)))
    resized = cv2.resize(
        image, (content_width, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    normalized = resized.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    tensor = np.zeros((1, 3, INPUT_HEIGHT, canvas_width), dtype=np.float32)
    tensor[0, :, :, :content_width] = normalized.transpose(2, 0, 1)
    return tensor


def decode_ctc(
    logits: np.ndarray,
    plate_characters: list[str],
    class_indices_by_count: dict[int, tuple[int, ...]],
) -> str:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"output must have shape [1,time,classes], got {logits.shape}")
    class_count = int(logits.shape[2])
    class_indices = class_indices_by_count.get(class_count)
    if class_indices is None:
        raise ValueError(f"unsupported PP-OCRv6 class count: {class_count}")
    if len(class_indices) != len(plate_characters):
        raise ValueError("embedded mapping and PlateRecognition alphabet differ")
    mapped = dict(zip(class_indices, plate_characters))

    result: list[str] = []
    previous = -1
    for best_value in np.argmax(logits[0], axis=1):
        best = int(best_value)
        if best == 0:
            previous = 0
            continue
        if best == previous:
            continue
        previous = best
        character = mapped.get(best)
        if character is not None:
            result.append(character)
    return "".join(result)


def resolve_model_path(model_root: Path, tier: str) -> Path:
    copied = model_root / tier / "inference.onnx"
    source = model_root / tier / "rec" / "inference.onnx"
    if copied.is_file():
        return copied
    if source.is_file():
        return source
    raise FileNotFoundError(f"missing {tier} inference.onnx under {model_root}")


def validate_tier(
    tier: str,
    model_root: Path,
    image: np.ndarray,
    plate_characters: list[str],
    class_indices_by_count: dict[int, tuple[int, ...]],
) -> str:
    model_path = resolve_model_path(model_root, tier)
    tensor = preprocess_bgr(image)
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError(f"{tier} model must have one input and one output")
    output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: tensor},
    )[0]
    return decode_ctc(output, plate_characters, class_indices_by_count)


def main() -> int:
    args = parse_args()
    tiers: Iterable[str] = TIERS if args.all_tiers else (args.tier,)
    image = read_image(args.image)
    plate_characters = load_plate_characters()
    class_indices_by_count = load_cpp_mappings()
    for tier in tiers:
        print(
            validate_tier(
                tier,
                args.model_root,
                image,
                plate_characters,
                class_indices_by_count,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
