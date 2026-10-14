"""Static QDQ INT8 quantization of an FP32 ONNX model with ONNX Runtime.

Usage:
    python quantize.py                                    # defaults below
    python quantize.py --input results/student_fp32.onnx --output results/student_int8.onnx \
        --method minmax --calib-size 500 [--keep-first-last-fp32]

Calibration images are a class-stratified, seeded sample of the TRAIN indices in
val_split.json, so no validation image is ever seen during calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

SEED = 42

CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}

# Use the same preprocessing as the competition PDF
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def stratified_calibration_indices(dataset, train_indices, size, seed=SEED):
    """Equal number of images per class, drawn from the train split only."""
    targets = np.array(dataset.targets)
    train_indices = np.array(train_indices)
    classes = np.unique(targets)
    per_class = size // len(classes)
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in classes:
        class_train = train_indices[targets[train_indices] == class_id]
        chosen.extend(rng.choice(class_train, per_class, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


class EuroSATCalibrationReader(CalibrationDataReader):
    def __init__(self, loader, input_name):
        self.loader = loader
        self.input_name = input_name
        self.iterator = iter(loader)

    def get_next(self):
        try:
            images, _ = next(self.iterator)
            return {self.input_name: images.numpy().astype(np.float32)}
        except StopIteration:
            return None

    def rewind(self):
        self.iterator = iter(self.loader)


def first_conv_and_last_gemm(model_path):
    nodes = onnx.load(model_path).graph.node
    first_conv = next(n.name for n in nodes if n.op_type == "Conv")
    last_gemm = [n.name for n in nodes if n.op_type in ("Gemm", "MatMul")][-1]
    return [first_conv, last_gemm]


def quantize(input_path, output_path, method="minmax", calib_size=500,
             keep_first_last_fp32=False, split_path="val_split.json"):
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_path = output_path.with_name(output_path.stem + "_preprocessed.onnx")

    # ORT quantizers expect a shape-inferred graph (PDF Section 6.2)
    quant_pre_process(str(input_path), str(preprocessed_path))

    dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)
    train_indices = json.loads(Path(split_path).read_text())["train"]
    calibration_indices = stratified_calibration_indices(dataset, train_indices, calib_size)
    # Batch of 1: the exported graph has a fixed (1, 3, 224, 224) input
    calibration_loader = DataLoader(Subset(dataset, calibration_indices), batch_size=1, shuffle=False)

    input_name = onnx.load(str(preprocessed_path)).graph.input[0].name
    nodes_to_exclude = first_conv_and_last_gemm(str(preprocessed_path)) if keep_first_last_fp32 else []

    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=EuroSATCalibrationReader(calibration_loader, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CALIBRATION_METHODS[method],
        nodes_to_exclude=nodes_to_exclude,
    )
    preprocessed_path.unlink()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/student_fp32.onnx")
    parser.add_argument("--output", default="results/student_int8.onnx")
    parser.add_argument("--method", choices=CALIBRATION_METHODS, default="minmax")
    parser.add_argument("--calib-size", type=int, default=500)
    parser.add_argument("--keep-first-last-fp32", action="store_true")
    args = parser.parse_args()

    print(f"Quantizing {args.input} (method={args.method}, calib_size={args.calib_size}, "
          f"keep_first_last_fp32={args.keep_first_last_fp32})")
    output = quantize(args.input, args.output, args.method, args.calib_size, args.keep_first_last_fp32)
    print("Saved:", output)
