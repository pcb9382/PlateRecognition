#!/usr/bin/env python3
"""Run all PP-OCRv6 ONNX tiers and assert public plate text results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import onnxruntime as ort


CPP_MAPPING_SYMBOLS = {
    "tiny": "kTinyClassIndexByPlateCharacter",
    "small": "kFullClassIndexByPlateCharacter",
    "medium": "kFullClassIndexByPlateCharacter",
}

CPP_CLASS_COUNT_SYMBOLS = {
    "tiny": "kTinyClassCount",
    "small": "kFullClassCount",
    "medium": "kFullClassCount",
}

EXPECTED = {
    "tiny": {
        "苏A0GY63_006235.jpg": "5AGY63",
        "藏ABT766_000915.jpg": "ABT766",
    },
    "small": {
        "苏A0GY63_006235.jpg": "A0GY63",
        "藏ABT766_000915.jpg": "藏ABT766",
    },
    "medium": {
        "苏A0GY63_006235.jpg": "苏A0GY63",
        "藏ABT766_000915.jpg": "藏ABT766",
    },
}


def parse_args() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]
    repository_root = module_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=module_root / "models")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=repository_root / "PlateRecognition" / "test" / "data",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=tuple(EXPECTED),
        default=list(EXPECTED),
    )
    parser.add_argument(
        "--source-model-root",
        type=Path,
        help=(
            "optional original PPOCRV6 models root; audits each "
            "<tier>/rec/dict.txt against the embedded sparse indices"
        ),
    )
    return parser.parse_args()


def load_plate_characters(repository_root: Path) -> list[str]:
    header = (repository_root / "common" / "PlateRecognitionCharset.h").read_text(
        encoding="utf-8"
    )
    marker = header.index("characters =")
    begin = header.index("{", marker)
    end = header.index("};", begin)
    tokens = re.findall(r'"((?:\\.|[^"\\])*)"', header[begin:end])
    characters = [json.loads(f'"{token}"') for token in tokens]
    if len(characters) != 77 or len(set(characters)) != 77:
        raise RuntimeError("PlateRecognition alphabet must contain 77 unique entries")
    return characters


def load_cpp_contract(module_root: Path, tier: str) -> tuple[int, tuple[int, ...]]:
    source = (module_root / "src" / "OCRV6Text.cpp").read_text(encoding="utf-8")
    header = (module_root / "src" / "OCRV6Text.h").read_text(encoding="utf-8")
    mapping_symbol = CPP_MAPPING_SYMBOLS[tier]
    mapping_match = re.search(
        rf"\b{re.escape(mapping_symbol)}\s*\[\s*\]\s*=\s*\{{(?P<body>.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    if mapping_match is None:
        raise RuntimeError(f"could not find C++ mapping array {mapping_symbol}")
    indices = tuple(
        int(value) for value in re.findall(r"\d+", mapping_match.group("body"))
    )
    if len(indices) != 77 or len(set(indices)) != 77:
        raise RuntimeError(f"{mapping_symbol} must contain 77 unique class indices")

    class_count_symbol = CPP_CLASS_COUNT_SYMBOLS[tier]
    count_match = re.search(
        rf"\b{re.escape(class_count_symbol)}\s*=\s*(\d+)U?\s*;", header
    )
    if count_match is None:
        raise RuntimeError(f"could not find C++ class count {class_count_symbol}")
    return int(count_match.group(1)), indices


def verify_source_dictionary(
    source_model_root: Path,
    tier: str,
    mapping: dict[int, str],
    plate_characters: list[str],
) -> None:
    path = source_model_root / tier / "rec" / "dict.txt"
    entries = path.read_text(encoding="utf-8-sig").splitlines()
    allowed = set(plate_characters)
    actual = {
        index + 1: value for index, value in enumerate(entries) if value in allowed
    }
    if actual != mapping:
        raise RuntimeError(f"{tier}: embedded sparse mapping differs from {path}")


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    content_width = max(1, min(3200, int(np.ceil(48 * image.shape[1] / image.shape[0]))))
    canvas_width = max(160, min(3200, max(320, content_width)))
    resized = cv2.resize(
        image, (content_width, 48), interpolation=cv2.INTER_LINEAR
    )
    normalized = resized.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    tensor = np.zeros((1, 3, 48, canvas_width), dtype=np.float32)
    tensor[0, :, :, :content_width] = normalized.transpose(2, 0, 1)
    return tensor


def decode_sparse(
    logits: np.ndarray, class_count: int, mapping: dict[int, str]
) -> str:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != class_count:
        raise RuntimeError(f"unexpected V6 output shape: {list(logits.shape)}")
    result: list[str] = []
    previous = class_count
    for value in np.argmax(logits[0], axis=1).tolist():
        if value != 0 and value != previous:
            character = mapping.get(value)
            if character is not None:
                result.append(character)
        previous = value
    return "".join(result)


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    module_root = Path(__file__).resolve().parents[1]
    plate_characters = load_plate_characters(repository_root)
    results: list[dict[str, object]] = []

    for tier in args.tiers:
        class_count, class_indices = load_cpp_contract(module_root, tier)
        if len(class_indices) != len(plate_characters):
            raise RuntimeError(f"{tier}: sparse mapping must contain 77 entries")
        mapping = dict(zip(class_indices, plate_characters))
        if len(mapping) != 77:
            raise RuntimeError(f"{tier}: sparse mapping indices must be unique")
        if args.source_model_root is not None:
            verify_source_dictionary(
                args.source_model_root, tier, mapping, plate_characters
            )

        model_path = args.model_root / tier / "inference.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise RuntimeError(f"{tier}: expected exactly one input and output")

        for image_name, expected in EXPECTED[tier].items():
            tensor = preprocess(read_image(args.image_root / image_name))
            started = time.perf_counter()
            output = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: tensor},
            )[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            actual = decode_sparse(output, class_count, mapping)
            if actual != expected:
                raise AssertionError(
                    f"{tier}/{image_name}: expected {expected!r}, got {actual!r}"
                )
            results.append(
                {
                    "tier": tier,
                    "image": image_name,
                    "input_shape": list(tensor.shape),
                    "output_shape": list(output.shape),
                    "plate_text": actual,
                    "expected": expected,
                    "inference_ms_ort_cpu": round(elapsed_ms, 3),
                }
            )

    report = {
        "backend": "ONNX Runtime CPUExecutionProvider",
        "decoder": "sparse mapping parsed from C++ OCRV6Text.cpp",
        "source_dictionaries_checked": args.source_model_root is not None,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
