# SFT on Qwen2.5-Omni-7B

Use the complete pretrained Omni Thinker with AffectGPT's annotations,
supervision, training runner and scoring. Source is maintained directly on
`main` at https://github.com/haoxu2525-tech/SFT-on-Qwen2.5Omni7B.

## Published implementations

- AffectGPT: https://github.com/zeroQiaoba/AffectGPT at
  `95b9257758df29fa6481c138252cdd317cce1735` (unchanged submodule).
- Omni: `Qwen/Qwen2.5-Omni-7B` revision
  `ae9e1690543ffd5c0221dc27f79834d0294cba00`.
- Native model/processor: `transformers==4.52.4`.
- Published media loader: `qwen-omni-utils==0.0.8`, `process_mm_info`.
  https://github.com/QwenLM/Qwen2.5-Omni/tree/main/qwen-omni-utils

The previous independent-images/independent-audio-clips interface is retired.
There is no local media decoder, timestamp calculator, feature projector,
aggregation or fusion implementation. The actual media call chain is:

```python
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos, video_kwargs = process_mm_info(
    messages, use_audio_in_video=True, return_video_kwargs=True)
inputs = processor(text=text, images=images, videos=videos, audio=audios,
                   use_audio_in_video=True, fps=video_kwargs["fps"][0],
                   do_resize=False, return_tensors="pt", padding=False,
                   truncation=False, add_special_tokens=False)
inputs["use_audio_in_video"] = True
```

Each row uses ONE original video file with its embedded audio. The official
helper receives `nframes=8`, `resized_height=224`, `resized_width=224` from the
reference input budget. It performs the decoding, frame selection and resizing;
the processor does not resize a second time. Its returned FPS is forwarded
unchanged (one-element list to scalar for Transformers 4.52.4's interface).
Native video grids, interleaved audio/video tokens and TMRoPE are used in both
training and generation. A video without an audio track fails in the official
loader; no separate WAV, face array or historical cache is silently substituted.

This uses the original full-frame video and continuous audio, NOT AffectGPT's
face crops and eight sampled audio clips. Sample identities and annotations
stay the same; the media representation is an explicit correction requested by
the user. Native frame indices are chosen by Qwen's published sampler.

## Training and evaluation boundary

The original `BaseTask.train_step`, runner, optimizer, scheduler, checkpoint
selection, benchmark membership and scoring files are unchanged. A small model
signature wrapper accepts the original `model(samples)["loss"]` call and calls
the complete Thinker. Data interfaces provide the video path and native batch.
Inference's model adapter calls native `generate` with the source script's
explicit decoding parameters and original stopping/parser behavior.

Qwen's published system prompt and chat template supply role/media tokens.
The text body comes verbatim from AffectGPT's existing `textonly` prompt branch
with the SAME subtitle and question. This branch is used only to obtain the
text body: actual input remains paired video/audio. Legacy Human/Assistant
wrappers are replaced by native chat roles; no new question is authored.
Original hybrid description/ovlabel selection, answer and `###` suffix remain.
Only the answer plus EOS is supervised after native media expansion.

Freeze all pretrained Thinker weights, including both encoders/projections;
train language-layer LoRA with r=16, alpha=32, dropout=.05 and the original
attention/MLP targets. Preserve the reference FP16/AMP runner policy; GPU
numerical behavior is not yet verified. No custom precision policy is added.
Expanded training sequences, INCLUDING chat/EOS, must fit 2048. Inference
requires input+1200 output budget <=4096. Overflow fails; no media/answer is
cut by the interface. The original Dataset's item function is called through a
text-serialization view: it keeps annotation lookup, hybrid QA choice, answer
suffix and masks, but tokenizes complete text and defers the legacy 1024 check
to the final native 2048 check. Length overflow escapes from the collater instead
of entering the Dataset's random-replacement handler. Original retries for
other annotation/data errors remain. Source Dataset code and YAML stay intact.

## Checkout and run

```sh
git clone --recurse-submodules https://github.com/haoxu2525-tech/SFT-on-Qwen2.5Omni7B.git
cd SFT-on-Qwen2.5Omni7B
# Existing checkout: git submodule update --init --recursive
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" prepare --runtime /workspace/omni-native-av-sft --dataset-root /workspace/dataset
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" train --runtime /workspace/omni-native-av-sft --dataset-root /workspace/dataset
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" infer --runtime /workspace/omni-native-av-sft --dataset-root /workspace/dataset
python "Experiment/SFT Base Model Training/run_affectgpt_omni.py" evaluate --runtime /workspace/omni-native-av-sft --dataset-root /workspace/dataset
```

Use a fresh original AffectGPT environment plus `requirements-omni.txt` and a
working FFmpeg executable for its official audio loader. Supply original video
files WITH audio, not only face/audio preprocessing products. The launcher
rejects retired v1/v2 runtimes. It excludes author outputs from the execution
copy. The preparation receipt records the source commit, workflow and dataset
path; the exact model revision is pinned in the interface code.

Inference supplies `--dataset inferenceData --zeroshot`, as in the original
README's OV-label invocation. The source's bare `merbench`/non-zeroshot defaults
are not runnable. Original CLI arguments can follow `--`, including checkpoint
options or a specific original dataset; the source YAML checkpoint-selection
defaults are otherwise retained. No question or decoding setting is invented.
Scoring has no CLI overrides and requires the original separate
`models/Qwen2.5-7B-Instruct` extraction model and scoring dependencies.

## Verification and reconstruction

Tests use a real synthetic video container WITH continuous audio, the published
Qwen loader, real pinned processor and a tiny native Thinker. They compare
processor tensors with the direct official call chain, check native token
interleaving/TMRoPE time correspondence, original training-step compatibility,
LoRA updates and checkpoint reconstruction. This is not 7B/GPU training or an
evaluation result. See `Experiment/adjustment.md` for test status/limitations.

Preserve runner checkpoints, exact Omni revision and processor, interface code,
environment versions and preparation receipt together. Reconstruct the same frozen
base and LoRA configuration before loading trained state. No retired v1/v2 checkpoint
is accepted as a substitute for the corrected input interface. Historical
reproduction notes retain their technical content with personal details removed;
current AGENTS.md and adjustment.md override
their obsolete three-component-replacement instructions.
