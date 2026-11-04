"""Run AffectGPT's scripts using the complete native Omni Thinker.

Example (Linux GPU runtime):
  python run_affectgpt_omni.py train --source /path/to/AffectGPT/AffectGPT \
      --runtime /workspace/omni-thinker-sft --dataset-root /workspace/dataset

Use `prepare` to create/check the isolated execution copy without downloading
weights or running anything. Scoring scripts run without model substitutions.
"""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

SOURCE_COMMIT = "95b9257758df29fa6481c138252cdd317cce1735"
REFERENCE_CONFIG = "train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml"
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "Official Repository Code/AffectGPT/AffectGPT"
SCRIPTS = {"train": "train.py", "infer": "inference_hybird.py",
           "evaluate": "evaluation.py", "score": "evaluation-scoreonly.py"}
WORKFLOW = "native-thinker-synchronized-av-v3"


def prepare_execution(source, runtime, dataset):
    source, runtime, dataset = source.resolve(), runtime.resolve(), dataset.resolve()
    if source == runtime or source in runtime.parents or runtime in source.parents:
        raise ValueError("Execution directory must be separate from the official source")
    if not dataset.is_dir():
        raise ValueError("Original dataset directory does not exist")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != SOURCE_COMMIT or dirty:
        raise ValueError("Official source must be clean at the pinned commit")
    execution = runtime / "AffectGPT"
    dataset_link = runtime / "dataset"
    receipt = runtime / "preparation.json"
    if runtime.exists() and any(runtime.iterdir()):
        if not receipt.is_file() or json.loads(receipt.read_text()).get("workflow") != WORKFLOW:
            raise ValueError("Use a fresh runtime for native Thinker; existing artifacts are not imported")
    runtime.mkdir(parents=True, exist_ok=True)
    if dataset_link.exists() or dataset_link.is_symlink():
        if dataset_link.resolve() != dataset:
            raise ValueError("Runtime dataset link points to different inputs")
    else:
        dataset_link.symlink_to(dataset, target_is_directory=True)
    # The execution copy contains identical source/resources, but none of the
    # authors' supplied output files or any source-tree model/cache directories.
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if relative.parts[0] in {"output", "models", ".git"} or "__pycache__" in relative.parts:
            continue
        if not item.is_file():
            continue
        target = execution / relative
        data = item.read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"Execution code differs from source: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    (execution / "output").mkdir(exist_ok=True)
    record = {"workflow": WORKFLOW, "source_commit": head,
              "dataset_root": str(dataset), "record_type": "source_preparation"}
    # This is a preparation receipt, never evidence that training ran.
    (runtime / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", *SCRIPTS])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args, original_args = parser.parse_known_args()
    if original_args[:1] == ["--"]:
        original_args = original_args[1:]
    if args.stage in {"train", "prepare"} and original_args:
        parser.error("No training overrides are allowed; the original YAML is used verbatim")
    if args.stage in {"evaluate", "score"} and original_args:
        parser.error("The pinned scoring scripts do not accept command-line options")
    execution = prepare_execution(args.source, args.runtime, args.dataset_root)
    if args.stage == "prepare":
        print(f"Prepared unchanged source at {execution}; no training started")
        return
    runtime = args.runtime.resolve()
    # Source/output data remain isolated from author files and retired runtimes.
    os.environ["HF_HOME"] = str(runtime / "cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(runtime / "cache" / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(runtime / "cache" / "transformers")
    os.environ["TORCH_HOME"] = str(runtime / "cache" / "torch")
    os.chdir(execution)
    sys.path.insert(0, str(execution))
    if args.stage in {"train", "infer"}:
        import torch
        if not torch.__version__.startswith(("2.4.0", "2.1.0")):
            raise RuntimeError("The unchanged AffectGPT CUDA AMP helper requires torch 2.4.0 or 2.1.0")
        from huggingface_hub import snapshot_download
        from omni_components import MODEL_ID, MODEL_REVISION, install
        snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                     cache_dir=str(runtime / "cache" / "omni"))
        install(snapshot)
    script = SCRIPTS[args.stage]
    sys.argv = [script]
    if args.stage in {"train", "infer"}:
        sys.argv += ["--cfg-path", REFERENCE_CONFIG]
    if args.stage == "infer":
        # Match AffectGPT README's OV-label inference invocation. The script's
        # bare defaults (merbench, zeroshot=False) are not a runnable protocol.
        # Later explicit original CLI arguments retain their usual precedence.
        sys.argv += ["--dataset", "inferenceData", "--zeroshot"]
    sys.argv += original_args
    # No edits to train.py, inference_hybird.py, scoring, dataset classes or YAML.
    runpy.run_path(str(execution / script), run_name="__main__")


if __name__ == "__main__":
    main()
