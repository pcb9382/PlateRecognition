#!/usr/bin/env python3
"""Run reproducible PP-OCRv5 ONNX Runtime recognition regressions.

This test deliberately decodes with the same sparse class-index mapping that
is embedded in the C++ runtime.  A complete PP-OCR dictionary is therefore not
required to run the test.  When ``--source-dictionary`` is supplied, the test
also proves that every embedded index still matches the original dictionary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import onnxruntime as ort


CLASS_COUNT = 18385

MODEL_FILES = {
    "mobile": Path("mobile") / "inference.onnx",
    "server": Path("server") / "inference.onnx",
}

CPP_MAPPING_SYMBOLS = {
    "mobile": "kMobileClassIndices",
    "server": "kServerClassIndices",
}

EXPECTED = {
    "mobile": {
        "苏A0GY63_006235.jpg": "A0GY63",
        "藏ABT766_000915.jpg": "藏ABT766",
    },
    "server": {
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
        "--models",
        nargs="+",
        choices=tuple(MODEL_FILES),
        default=list(MODEL_FILES),
    )
    parser.add_argument(
        "--source-dictionary",
        type=Path,
        help="optional original ppocrv5_key.json used to audit all sparse indices",
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


def load_cpp_mapping(module_root: Path, model_name: str) -> tuple[int, ...]:
    source = (module_root / "src" / "OCRV5Text.cpp").read_text(encoding="utf-8")
    symbol = CPP_MAPPING_SYMBOLS[model_name]
    match = re.search(
        rf"\b{re.escape(symbol)}\s*\[\s*\]\s*=\s*\{{(?P<body>.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"could not find C++ mapping array {symbol}")
    indices = tuple(int(value) for value in re.findall(r"\d+", match.group("body")))
    if len(indices) != 77 or len(set(indices)) != 77:
        raise RuntimeError(f"{symbol} must contain 77 unique class indices")
    return indices


def verify_source_dictionary(
    path: Path,
    model_name: str,
    mapping: dict[int, str],
    plate_characters: list[str],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = [payload[str(index)] for index in range(len(payload))]
    if len(entries) < CLASS_COUNT - 1:
        raise RuntimeError("source dictionary is shorter than the model class count")
    allowed = set(plate_characters)
    actual = {
        index + 1: value
        for index, value in enumerate(entries[: CLASS_COUNT - 1])
        if value in allowed
    }
    if actual != mapping:
        raise RuntimeError(
            f"{model_name}: C++ sparse mapping differs from source dictionary"
        )


def preprocess(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise RuntimeError("input must be an 8-bit BGR image")
    scaled = image.shape[1] * 48.0 / image.shape[0]
    rounded = max(1, int(np.floor(scaled + 0.5)))
    width = max(20, min(1280, ((rounded + 19) // 20) * 20))
    resized = cv2.resize(image, (width, 48), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])


def decode_sparse(logits: np.ndarray, mapping: dict[int, str]) -> str:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != CLASS_COUNT:
        raise RuntimeError(f"unexpected V5 output shape: {list(logits.shape)}")
    result: list[str] = []
    previous = 0
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
    for model_name in args.models:
        class_indices = load_cpp_mapping(module_root, model_name)
        mapping = dict(zip(class_indices, plate_characters))
        if args.source_dictionary is not None:
            verify_source_dictionary(
                args.source_dictionary, model_name, mapping, plate_characters
            )
        model_path = args.model_root / MODEL_FILES[model_name]
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise RuntimeError(f"{model_name}: expected exactly one input and output")

        for image_name, expected in EXPECTED[model_name].items():
            image_path = args.image_root / image_name
            encoded = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read image: {image_path}")
            tensor = preprocess(image)
            started = time.perf_counter()
            output = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: tensor},
            )[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            actual = decode_sparse(output, mapping)
            if actual != expected:
                raise AssertionError(
                    f"{model_name}/{image_name}: expected {expected!r}, got {actual!r}"
                )
            results.append(
                {
                    "model": model_name,
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
        "decoder": "sparse mapping parsed from C++ OCRV5Text.cpp",
        "source_dictionary_checked": args.source_dictionary is not None,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
