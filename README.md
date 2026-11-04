# SFT-on-Qwen2.5Omni7B
Fine-tune Qwen2.5-Omni using AffectGPT’s data and training recipe. Use the complete native Omni Thinker—including its encoders, projection layers, LLM, and processor—and remove AffectGPT’s additional projection and fusion modules.

## Repository checkout

This repository contains the SFT implementation synchronized from AgentEmotion
commit `21aaa0bf11dfb30869b3988ec7f6bc865caf120d`. The four implementation/test
files are byte-for-byte copies of that implementation. Existing reproduction
notes are retained verbatim; the current `AGENTS.md` and `Experiment/adjustment.md`
supersede their historical three-component replacement instruction.

AffectGPT is a Git submodule pinned to the reference commit, not modified source.
Fetch it before running the launcher or tests:

```sh
git clone --recurse-submodules https://github.com/haoxu2525-tech/SFT-on-Qwen2.5Omni7B.git
cd SFT-on-Qwen2.5Omni7B
# For an existing checkout:
git submodule update --init --recursive
```

Only source, tests and documentation are synchronized. Datasets, weights,
runtime caches and generated outputs are not uploaded. Upstream author outputs
inside the pinned submodule are not our training results.


Use the complete official `Qwen/Qwen2.5-Omni-7B` Thinker and its native processor.
Freeze its encoders, native projections and language base weights; train only
language-layer LoRA. AffectGPT's added projections, aggregation and fusion are
not instantiated. This replaces the earlier component-transplant plan.

Code: `Experiment/SFT Base Model Training`.
Branch: `main`.
AffectGPT source: `Official Repository Code/AffectGPT/AffectGPT`, unchanged at
`95b9257758df29fa6481c138252cdd317cce1735`.
Omni revision: `ae9e1690543ffd5c0221dc27f79834d0294cba00`.
Reference: `train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml`.

The original dataset readers, sampled frames/clips, prompt prose, target answers,
runner, optimizer, scheduler and scoring remain the owners. The processor turns
model-specific placeholders into native media tokens. Original sampled frames
are supplied as independent images and the eight audio clips as independent
audio inputs; they are not concatenated into a fictitious synchronized video.
`<MultiHere>` supplies no external fusion feature; only this placeholder is
removed, leaving the original surrounding prose and all other media present.

The user authorized a capacity increase: training 2048 tokens before the
original BOS/EOS additions; inference total budget 4096, with the reference
1200 new-token budget. This avoids the measured 1038-token short-prompt example
exceeding the old 1024 training / 800 inference-input allowances. Exceeding the
new limits raises an error instead of dropping media, labels or samples.
No claim of unchanged evaluation scores follows from these interface changes.

## Running

Create a fresh Linux runtime with the official environment dependencies and the
`requirements-omni.txt` overlay. Keep original datasets on remote storage.

```sh
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" prepare --runtime /workspace/omni-thinker-sft --dataset-root /workspace/dataset
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" train --runtime /workspace/omni-thinker-sft --dataset-root /workspace/dataset
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" infer --runtime /workspace/omni-thinker-sft --dataset-root /workspace/dataset
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" evaluate --runtime /workspace/omni-thinker-sft --dataset-root /workspace/dataset
```

Inference accepts its original script arguments after `--` (including checkpoint
locations). The launcher selects the original `inferenceData` dataset group by
default; the source's literal `merbench` default has no matching dataset class.
The scoring scripts take no CLI options. Before scoring, provision the original
official Qwen2.5-7B-Instruct extraction model under the execution directory's
`models/Qwen2.5-7B-Instruct`, along with the official scoring dependencies. This
evaluation model is separate from the Thinker being fine-tuned. Their dataset
list, selection rules, decoding and scoring code are unchanged.
Preparation excludes author outputs and pretrained
weight folders, and records source hashes. Native Thinker only is loaded, without
Talker or token-to-wave generation.

## Validation and reconstruction

`test_omni_components.py` uses the actual official processor/tokenizer and a tiny
complete native Thinker. It checks media counts/order, answer masks, training
scope, generation and trained-weight reconstruction; it is not a 7B result.
Set `OMNI_PROCESSOR_SNAPSHOT` to a local snapshot containing the pinned official
processor/tokenizer files and run pytest on that file.

Preserve the original runner checkpoints (LoRA plus optimizer/scaler state),
Omni revision, processor/tokenizer, runtime dependency versions and
`omni-thinker.json` source/interface manifest together. Frozen weights must be
reconstructible from that exact official revision. See `Experiment/adjustment.md`
for validation status and outstanding runtime checks.
