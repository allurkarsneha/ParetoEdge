"""Compile and profile an ONNX model on Qualcomm AI Hub (PDF Section 7).

Requires: pip install qai-hub && qai-hub configure --api_token YOUR_TOKEN

Usage:
    python profile_aihub.py submission/model_int8.onnx
    python profile_aihub.py submission/model_int8.onnx --device "Samsung Galaxy S24 (Family)"
"""
import argparse
import json
from pathlib import Path

import onnx
import qai_hub as hub

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("--device", default="Samsung Galaxy S24 (Family)", help="AI Hub device name")
args = parser.parse_args()

# Strip value_info entries that collide with graph outputs (known export issue, PDF Section 7)
model = onnx.load(str(args.model))
output_names = {o.name for o in model.graph.output}
kept = [v for v in model.graph.value_info if v.name not in output_names]
del model.graph.value_info[:]
model.graph.value_info.extend(kept)
clean_path = args.model.with_name(args.model.stem + "_aihub.onnx")
onnx.save(model, str(clean_path))

device = hub.Device(args.device)
compile_job = hub.submit_compile_job(
    model=str(clean_path),
    device=device,
    input_specs={"input": (1, 3, 224, 224)},
)
print("Compile job:", compile_job.url)
compile_job.wait()

profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device)
print("Profile job:", profile_job.url)
profile_job.wait()
profile = profile_job.download_profile()

summary = profile.get("execution_summary", {})
print(json.dumps(summary, indent=2))
out = Path("results/aihub_profile.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(profile, indent=2, default=str))
print(f"Saved: {out}")
print(f"Paste into metrics.json ai_hub_profile_summary: device={args.device}, "
      f"inference_time_us={summary.get('estimated_inference_time')}, "
      f"peak_memory_bytes={summary.get('estimated_inference_peak_memory')}")
