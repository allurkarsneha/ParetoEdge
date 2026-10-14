import torch
from torch import nn
from torchvision import models

print("Starting ParetoEdge...")

# Initialize MobileNetV2
student = models.mobilenet_v2(weights=None)

# Modify classifier for 10 EuroSAT classes
student.classifier[1] = nn.Linear(
    student.last_channel, 10
)

# Load pretrained student model
student.load_state_dict(
    torch.load(
        "models/student_mobilenetv2_distilled.pt",
        map_location="cpu",
        weights_only=True
    )
)

student.eval()

print("MobileNetV2 loaded successfully!")

# Test model with dummy input
dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = student(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)

print("ParetoEdge setup successful!")

import os
import onnx
import onnxruntime as ort
import numpy as np

# Create output directory
os.makedirs("results", exist_ok=True)

# Export the original MobileNetV2 to ONNX
onnx_path = "results/student_fp32.onnx"

torch.onnx.export(
    student,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamo=False
)

print("ONNX export completed!")

# Validate ONNX model structure
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("ONNX model validation successful!")

# Test inference using ONNX Runtime
session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

onnx_output = session.run(
    None,
    {input_name: dummy_input.numpy()}
)[0]

print("ONNX output shape:", onnx_output.shape)

# Compare PyTorch and ONNX predictions
pytorch_output = output.detach().numpy()

max_difference = np.max(
    np.abs(pytorch_output - onnx_output)
)

print("Maximum output difference:", max_difference)

print("ParetoEdge ONNX export successful!")