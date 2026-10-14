import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

print("Starting ParetoEdge accuracy evaluation...")

# Same preprocessing used during quantization
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load EuroSAT
dataset = datasets.EuroSAT(
    root="./data",
    download=False,
    transform=transform
)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

# Reproducible provisional evaluation subset.
# Exclude images 0–199 used for INT8 calibration.
rng = np.random.default_rng(42)

eligible_indices = np.arange(200, len(dataset))
evaluation_indices = rng.choice(
    eligible_indices,
    size=1000,
    replace=False
)

evaluation_dataset = Subset(
    dataset,
    evaluation_indices.tolist()
)

evaluation_loader = DataLoader(
    evaluation_dataset,
    batch_size=32,
    shuffle=False
)

models = {
    "FP32": "results/student_fp32.onnx",
    "INT8": "results/student_int8.onnx"
}

results = {}

for model_name, model_path in models.items():

    print(f"\nEvaluating {model_name}...")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # The exported ONNX model may require batch size 1.
    input_shape = session.get_inputs()[0].shape
    batch_size = input_shape[0]

    correct = 0
    total = 0

    class_correct = np.zeros(len(dataset.classes), dtype=np.int64)
    class_total = np.zeros(len(dataset.classes), dtype=np.int64)

    for images, labels in evaluation_loader:

        images_np = images.numpy()

        # Run individual images to support fixed batch-size-1 models
        for image, label in zip(images_np, labels.numpy()):

            image = np.expand_dims(image, axis=0)

            output = session.run(
                None,
                {input_name: image}
            )[0]

            prediction = int(np.argmax(output, axis=1)[0])
            actual = int(label)

            total += 1
            class_total[actual] += 1

            if prediction == actual:
                correct += 1
                class_correct[actual] += 1

    accuracy = 100 * correct / total

    per_class_recall = {}

    for i, class_name in enumerate(dataset.classes):
        if class_total[i] > 0:
            recall = 100 * class_correct[i] / class_total[i]
            per_class_recall[class_name] = round(float(recall), 2)

    worst_class_recall = min(per_class_recall.values())

    results[model_name] = {
        "accuracy_percent": round(accuracy, 2),
        "worst_class_recall_percent": worst_class_recall,
        "per_class_recall_percent": per_class_recall,
        "evaluated_images": total
    }

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Worst-class recall: {worst_class_recall:.2f}%")

    print("\nPer-class recall:")

    for class_name, recall in per_class_recall.items():
        print(f"  {class_name}: {recall:.2f}%")

# Compare accuracy
accuracy_drop = (
    results["FP32"]["accuracy_percent"]
    - results["INT8"]["accuracy_percent"]
)

results["comparison"] = {
    "accuracy_drop_percentage_points": round(accuracy_drop, 2)
}

results["evaluation_details"] = {
    "dataset": "EuroSAT",
    "split": "Provisional random subset, NOT official validation split",
    "seed": 42,
    "sample_size": 1000,
    "excluded_calibration_indices": [0, 199]
}

Path("results").mkdir(exist_ok=True)

with open("results/accuracy_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== PARETOEDGE ACCURACY COMPARISON =====")
print(
    "FP32 accuracy:",
    results["FP32"]["accuracy_percent"],
    "%"
)
print(
    "INT8 accuracy:",
    results["INT8"]["accuracy_percent"],
    "%"
)
print(
    "Accuracy drop:",
    round(accuracy_drop, 2),
    "percentage points"
)

print("\nSaved: results/accuracy_comparison.json")