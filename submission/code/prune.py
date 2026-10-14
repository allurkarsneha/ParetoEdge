"""Structured channel pruning of the MobileNetV2 student, optional fine-tuning, ONNX export.

What gets pruned:
  - the hidden (expansion) channels inside each InvertedResidual block. They are internal
    to the block, so removing them never breaks a residual connection. Pruning a hidden
    channel removes one output of the 1x1 expand conv, one depthwise filter, and one input
    of the 1x1 project conv.
  - optionally the 1280 output channels of the final 1x1 conv (and matching classifier inputs).
Channel selection (default --criterion stats), measured on 256 train images:
  importance = std of the channel's activation x L2 norm of the weights that carry it onward,
  i.e. how much the channel's output actually varies the next layer. A near-constant channel
  scores ~0 even if its weights are large.
  Bias compensation: each removed channel's mean activation is folded into the next BatchNorm
  running mean (or the classifier bias), so removing it does not shift downstream activations.
--criterion weight is the earlier weight-only score (project-conv L1 x |depthwise BN gamma|),
kept for the ablation in the report: it was no better than random on some blocks.
Kept counts are rounded to a multiple of 8, which vectorized CPU kernels and NPUs prefer.

Usage:
    python prune.py sensitivity
        Prune one stage at a time at several ratios, no fine-tuning, report val accuracy.
        Shows where the accuracy cliff is and whether it is the same for every stage.
    python prune.py run --ratio 0.3 --last-ratio 0.3 --epochs 1 --kd --name p30_kd
        Prune, fine-tune (hard labels, or --kd for teacher distillation), export ONNX,
        then run measure.py on the export.

Devices: fine-tuning uses the Apple GPU (MPS) when available, else CUDA, else CPU (--device to
force one). On our M4 the PyTorch CPU build trains MobileNetV2 at ~5 img/s vs ~67 img/s on MPS,
so a CPU-only run reproduces the same pipeline but much more slowly. All accuracy numbers come
from the exported ONNX model run on CPU by ONNX Runtime, which is what judges benchmark.
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

SEED = 42
STUDENT_PATH = "models/student_mobilenetv2_distilled.pt"
TEACHER_PATH = "models/teacher_resnet50_eurosat.pt"
TEACHER_LOGITS_PATH = Path("results/teacher_train_logits.pt")
OUT_DIR = Path("results/pruned")

# MobileNetV2 stages (blocks sharing a resolution/width) -> indices into model.features.
# Block 1 has expand_ratio 1 (no hidden layer), so it is never pruned.
STAGES = {
    "s2_56px": [2, 3],
    "s3_28px": [4, 5, 6],
    "s4_14px": [7, 8, 9, 10],
    "s5_14px": [11, 12, 13],
    "s6_7px": [14, 15, 16],
    "s7_7px": [17],
}
PRUNABLE_BLOCKS = [i for blocks in STAGES.values() for i in blocks]

# Same preprocessing as the competition PDF
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------- models

def load_student():
    student = models.mobilenet_v2(weights=None)
    student.classifier[1] = nn.Linear(student.last_channel, 10)
    student.load_state_dict(torch.load(STUDENT_PATH, map_location="cpu", weights_only=True))
    return student.eval()


def load_teacher():
    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location="cpu", weights_only=True))
    return teacher.eval()


def slice_bn(bn, keep):
    new = nn.BatchNorm2d(len(keep), eps=bn.eps, momentum=bn.momentum)
    new.weight.data = bn.weight.data[keep].clone()
    new.bias.data = bn.bias.data[keep].clone()
    new.running_mean = bn.running_mean[keep].clone()
    new.running_var = bn.running_var[keep].clone()
    return new


def hidden_importance(block):
    project = block.conv[2].weight.data  # [out, hidden, 1, 1]
    dw_gamma = block.conv[1][1].weight.data.abs()
    return project.abs().sum(dim=(0, 2, 3)) * dw_gamma


def removed_indices(total, keep):
    mask = torch.ones(total, dtype=torch.bool)
    mask[keep] = False
    return mask.nonzero().flatten()


def prune_block(block, keep, channel_means=None):
    """Keep only the given hidden channels of an InvertedResidual block.
    channel_means (per hidden channel, post-ReLU6) enables bias compensation."""
    expand, expand_bn = block.conv[0][0], block.conv[0][1]
    depthwise, depthwise_bn = block.conv[1][0], block.conv[1][1]
    project = block.conv[2]
    k = len(keep)

    if channel_means is not None:
        # Removed channels' average contribution to the project conv output, absorbed by its BN
        removed = removed_indices(project.in_channels, keep)
        shift = project.weight.data[:, removed, 0, 0] @ channel_means[removed]
        block.conv[3].running_mean -= shift

    new_expand = nn.Conv2d(expand.in_channels, k, 1, bias=False)
    new_expand.weight.data = expand.weight.data[keep].clone()
    new_depthwise = nn.Conv2d(k, k, 3, stride=depthwise.stride, padding=depthwise.padding, groups=k, bias=False)
    new_depthwise.weight.data = depthwise.weight.data[keep].clone()
    new_project = nn.Conv2d(k, project.out_channels, 1, bias=False)
    new_project.weight.data = project.weight.data[:, keep].clone()

    block.conv[0][0], block.conv[0][1] = new_expand, slice_bn(expand_bn, keep)
    block.conv[1][0], block.conv[1][1] = new_depthwise, slice_bn(depthwise_bn, keep)
    block.conv[2] = new_project


def last_conv_importance(model):
    classifier = model.classifier[1].weight.data  # [10, 1280]
    gamma = model.features[18][1].weight.data.abs()
    return classifier.abs().sum(dim=0) * gamma


def prune_last_conv(model, keep, channel_means=None):
    conv, bn = model.features[18][0], model.features[18][1]
    if channel_means is not None:
        # Removed channels' average pooled activation, folded into the classifier bias
        removed = removed_indices(conv.out_channels, keep)
        model.classifier[1].bias.data += model.classifier[1].weight.data[:, removed] @ channel_means[removed]
    new_conv = nn.Conv2d(conv.in_channels, len(keep), 1, bias=False)
    new_conv.weight.data = conv.weight.data[keep].clone()
    model.features[18][0], model.features[18][1] = new_conv, slice_bn(bn, keep)
    old_fc = model.classifier[1]
    new_fc = nn.Linear(len(keep), old_fc.out_features)
    new_fc.weight.data = old_fc.weight.data[:, keep].clone()
    new_fc.bias.data = old_fc.bias.data.clone()
    model.classifier[1] = new_fc
    model.last_channel = len(keep)


def activation_stats(model, loader, max_images=256):
    """Per-channel mean and std of each hidden layer (after the depthwise ReLU6) and of the
    pooled final-conv features, over a few train images."""
    sums, squares, counts, hooks = {}, {}, {}, []

    def record(key, activation, dims):
        sums[key] = sums.get(key, 0) + activation.sum(dim=dims).double()
        squares[key] = squares.get(key, 0) + (activation.double() ** 2).sum(dim=dims)
        counts[key] = counts.get(key, 0) + activation.numel() // activation.shape[1]

    for index in PRUNABLE_BLOCKS:
        hooks.append(model.features[index].conv[1].register_forward_hook(
            lambda _m, _i, out, key=index: record(key, out, (0, 2, 3))))
    hooks.append(model.features[18].register_forward_hook(
        lambda _m, _i, out: record("last", out.mean(dim=(2, 3)), (0,))))

    model.eval()
    seen = 0
    with torch.no_grad():
        for images, *_ in loader:
            model(images)
            seen += len(images)
            if seen >= max_images:
                break
    for hook in hooks:
        hook.remove()
    stats = {}
    for key in sums:
        mean = sums[key] / counts[key]
        std = (squares[key] / counts[key] - mean ** 2).clamp(min=0).sqrt()
        stats[key] = {"mean": mean.float(), "std": std.float()}
    return stats


def calibration_loader(dataset, train_indices, size=256):
    rng = np.random.default_rng(SEED)
    indices = rng.choice(train_indices, size, replace=False).tolist()
    return DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)


def kept_count(total, ratio):
    return max(8, int(round(total * (1 - ratio) / 8)) * 8)


def apply_pruning(model, block_ratios, last_ratio=0.0, stats=None, compensate=True):
    """Prune in place. With stats (from activation_stats on the unpruned model) channels are
    ranked by activation std x outgoing weight norm; without, by the weight-only score.
    Returns the architecture config needed to rebuild the model."""
    config = {"hidden": {}, "last_channel": model.features[18][0].out_channels}
    for index in PRUNABLE_BLOCKS:
        block = model.features[index]
        hidden = block.conv[0][0].out_channels
        ratio = block_ratios.get(index, 0.0)
        if ratio > 0:
            k = kept_count(hidden, ratio)
            if stats is None:
                score, means = hidden_importance(block), None
            else:
                outgoing = block.conv[2].weight.data[:, :, 0, 0].norm(dim=0)
                score = stats[index]["std"] * outgoing
                means = stats[index]["mean"] if compensate else None
            keep = torch.argsort(score, descending=True)[:k].sort().values
            prune_block(block, keep, means)
            hidden = k
        config["hidden"][index] = hidden
    if last_ratio > 0:
        k = kept_count(model.features[18][0].out_channels, last_ratio)
        if stats is None:
            score, means = last_conv_importance(model), None
        else:
            score = stats["last"]["std"] * model.classifier[1].weight.data.norm(dim=0)
            means = stats["last"]["mean"] if compensate else None
        keep = torch.argsort(score, descending=True)[:k].sort().values
        prune_last_conv(model, keep, means)
        config["last_channel"] = k
    model.train(model.training)  # new BN layers default to train mode; match the parent model
    return config


def build_pruned_model(config):
    """Rebuild an empty model with the pruned shapes, ready for load_state_dict."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    for index, hidden in config["hidden"].items():
        prune_block(model.features[int(index)], torch.arange(hidden))
    if config["last_channel"] != model.features[18][0].out_channels:
        prune_last_conv(model, torch.arange(config["last_channel"]))
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- data

class IndexedSubset(Dataset):
    """Subset that also returns the position, used to look up cached teacher logits."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        return image, label, position


def load_data(split_path="val_split.json"):
    dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)
    split = json.loads(Path(split_path).read_text())
    return dataset, split["train"], split["val"]


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, num_classes=10):
    """Accuracy of the exported ONNX graph under ONNX Runtime (CPU), one image at a time
    because the exported graph has a fixed (1, 3, 224, 224) input."""
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export_onnx(model, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, labels in loader:
        for image, label in zip(images.numpy(), labels.tolist()):
            prediction = int(session.run(None, {input_name: image[None]})[0].argmax())
            confusion[label, prediction] += 1
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)
    return {
        "accuracy": (confusion.diag().sum() / confusion.sum()).item(),
        "worst_class_accuracy": recall.min().item(),
        "per_class_accuracy": recall.tolist(),
    }


def teacher_logits(dataset, train_indices, device):
    """Teacher outputs for every train image, computed once and cached.
    Valid because fine-tuning uses no random augmentation, so each image is always identical."""
    if TEACHER_LOGITS_PATH.exists():
        cached = torch.load(TEACHER_LOGITS_PATH)
        if cached["train_indices"] == list(train_indices):
            return cached["logits"]
    print("Computing teacher logits for the train split (one-time)...")
    teacher = load_teacher().to(device)
    loader = DataLoader(Subset(dataset, train_indices), batch_size=64, shuffle=False)
    outputs = []
    start = time.perf_counter()
    with torch.no_grad():
        for images, _ in loader:
            outputs.append(teacher(images.to(device)).cpu())
    logits = torch.cat(outputs)
    print(f"  done in {time.perf_counter() - start:.0f}s")
    TEACHER_LOGITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_indices": list(train_indices), "logits": logits}, TEACHER_LOGITS_PATH)
    return logits


def fine_tune(model, dataset, train_indices, epochs, kd, device, lr=0.01, temperature=4.0, alpha=0.7):
    """Short SGD fine-tune. With kd, loss = alpha * T^2 * KL(student/T || teacher/T) + (1 - alpha) * CE."""
    soft_targets = teacher_logits(dataset, train_indices, device).to(device) if kd else None
    model.to(device)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(IndexedSubset(dataset, train_indices), batch_size=64, shuffle=True, generator=generator)
    total_steps = max(1, int(math.ceil(epochs * len(loader))))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    model.train()
    step = 0
    start = time.perf_counter()
    while step < total_steps:
        for images, labels, positions in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if kd:
                teacher_soft = F.softmax(soft_targets[positions.to(device)] / temperature, dim=1)
                kd_loss = F.kl_div(F.log_softmax(logits / temperature, dim=1), teacher_soft,
                                   reduction="batchmean") * temperature ** 2
                loss = alpha * kd_loss + (1 - alpha) * loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0 or step == total_steps:
                print(f"  step {step}/{total_steps} loss {loss.item():.4f} ({time.perf_counter() - start:.0f}s)")
            if step >= total_steps:
                break
    model.cpu().eval()
    return time.perf_counter() - start


def export_onnx(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), str(path),
        input_names=["input"], output_names=["output"], opset_version=13, dynamo=False,
    )


# ---------------------------------------------------------------- commands

def pruning_stats(args, dataset, train_indices):
    if args.criterion == "weight":
        return None
    return activation_stats(load_student(), calibration_loader(dataset, train_indices))


def sensitivity(args):
    dataset, train_indices, val_indices = load_data(args.split)
    stats = pruning_stats(args, dataset, train_indices)
    compensate = not args.no_compensation
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=128, shuffle=False)
    base = evaluate(load_student(), val_loader)
    print(f"unpruned: acc {base['accuracy'] * 100:.2f}% worst {base['worst_class_accuracy'] * 100:.2f}%")

    rows = [{"stage": "none", "ratio": 0.0, "params": count_params(load_student()), **base}]
    targets = list(STAGES.items()) + [("last_conv", None)]
    for stage, blocks in targets:
        for ratio in args.ratios:
            model = load_student()
            if blocks is None:
                apply_pruning(model, {}, ratio, stats, compensate)
            else:
                apply_pruning(model, {b: ratio for b in blocks}, 0.0, stats, compensate)
            result = evaluate(model, val_loader)
            rows.append({"stage": stage, "ratio": ratio, "params": count_params(model), **result})
            print(f"{stage:10s} ratio {ratio:.2f}: params {count_params(model) / 1e6:.3f}M "
                  f"acc {result['accuracy'] * 100:6.2f}% worst {result['worst_class_accuracy'] * 100:6.2f}%")
    suffix = args.criterion + ("_nocomp" if args.no_compensation and stats is not None else "")
    out = Path(f"results/prune_sensitivity_{suffix}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"Saved: {out}")


def run(args):
    torch.manual_seed(SEED)
    dataset, train_indices, val_indices = load_data(args.split)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=128, shuffle=False)

    stages = args.stages.split(",") if args.stages else list(STAGES)
    block_ratios = {b: args.ratio for stage in stages for b in STAGES[stage]}
    stage_ratios = dict(item.split("=") for item in args.stage_ratios.split(",")) if args.stage_ratios else {}
    for stage, ratio in stage_ratios.items():
        block_ratios.update({b: float(ratio) for b in STAGES[stage]})
    model = load_student()
    params_before = count_params(model)
    stats = pruning_stats(args, dataset, train_indices)
    config = apply_pruning(model, block_ratios, args.last_ratio, stats, not args.no_compensation)
    record = {
        "name": args.name, "ratio": args.ratio, "last_ratio": args.last_ratio, "stages": stages,
        "stage_ratios": stage_ratios,
        "criterion": args.criterion, "compensation": stats is not None and not args.no_compensation,
        "epochs": args.epochs, "kd": args.kd, "lr": args.lr,
        "params_before": params_before, "params_after": count_params(model), "config": config,
    }
    device = pick_device(args.device)
    record["device"] = str(device)
    print(f"Params {params_before / 1e6:.3f}M -> {record['params_after'] / 1e6:.3f}M | fine-tune device {device}")

    record["before_finetune"] = evaluate(model, val_loader)
    print(f"Before fine-tune: acc {record['before_finetune']['accuracy'] * 100:.2f}% "
          f"worst {record['before_finetune']['worst_class_accuracy'] * 100:.2f}%")

    if args.epochs > 0:
        record["finetune_seconds"] = fine_tune(model, dataset, train_indices, args.epochs, args.kd, device, args.lr)
        record["after_finetune"] = evaluate(model, val_loader)
        print(f"After fine-tune ({record['finetune_seconds']:.0f}s): "
              f"acc {record['after_finetune']['accuracy'] * 100:.2f}% "
              f"worst {record['after_finetune']['worst_class_accuracy'] * 100:.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, OUT_DIR / f"{args.name}.pt")
    onnx_path = OUT_DIR / f"{args.name}_fp32.onnx"
    export_onnx(model, onnx_path)
    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(record, indent=2))
    print(f"Saved: {onnx_path}")

    if not args.no_measure:
        subprocess.run([sys.executable, "measure.py", str(onnx_path), "--name", f"{args.name}_fp32",
                        "--skip-energy", "--split", args.split], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_split.json")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--criterion", choices=["stats", "weight"], default="stats",
                        help="channel ranking: activation statistics (default) or weights only")
    parser.add_argument("--no-compensation", action="store_true", help="disable bias compensation")
    commands = parser.add_subparsers(dest="command", required=True)

    sens = commands.add_parser("sensitivity")
    sens.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])

    r = commands.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--ratio", type=float, default=0.0, help="fraction of hidden channels removed per block")
    r.add_argument("--stages", default="", help=f"comma list of stages to prune (default all): {','.join(STAGES)}")
    r.add_argument("--stage-ratios", default="",
                   help="per-stage overrides, e.g. s7_7px=0.75,s6_7px=0.25")
    r.add_argument("--last-ratio", type=float, default=0.0, help="fraction of the 1280 final channels removed")
    r.add_argument("--epochs", type=float, default=0.0, help="fine-tune epochs (fractions allowed)")
    r.add_argument("--kd", action="store_true", help="distill from the ResNet-50 teacher while fine-tuning")
    r.add_argument("--lr", type=float, default=0.01)
    r.add_argument("--no-measure", action="store_true")
    r.add_argument("--device", default="auto", help="auto (mps > cuda > cpu), mps, cuda or cpu")

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    {"sensitivity": sensitivity, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
