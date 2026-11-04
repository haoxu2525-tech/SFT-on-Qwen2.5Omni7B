# Native Omni Thinker SFT decisions

Updated 2026-11-04 (Australia/Sydney).

## Current user decision

Use the complete official Qwen2.5-Omni-7B Thinker and native processor with
AffectGPT's original datasets, prompt prose, target labels, hybrid supervision,
training recipe, test membership and scoring scripts. Remove the AffectGPT
projection/aggregation/fusion model entirely. Freeze all pretrained Thinker
parameters (including visual merger and audio output projection); train only
language-layer LoRA using the reference rank/alpha/dropout/targets.

This explicitly supersedes the earlier prohibition on full Thinker replacement.
The user chose `Experiment/SFT Base Model Training` as the code location. The
old `Experiment/Experiment Code` folder remains; its four untracked draft files
were rewritten/moved, not imported as a trained model or historical Qwen output.

## Interface behavior

- Load `Qwen2_5OmniThinkerForConditionalGeneration` directly and retain both
  encoders and native projections inside it. No custom feature pooling or
  AffectGPT model constructor runs.
- Keep original dataset readers and sampling. The current reference supplies
  eight 224-pixel face images and eight independently sampled two-second audio
  clips. Serialize these as independent images/audios through the processor.
  No new FPS, AV alignment, whole-video resampling or concatenated waveform is
  invented. Native media positions are computed by Thinker.
- Preserve original natural-language prompt characters and target token IDs.
  Replace `<FaceHere>`/`<AudioHere>` etc. with native media markers. Remove only
  `<MultiHere>`, which no longer has an external fused feature. Its surrounding
  original prose and `<Multi></Multi>` wrapper remain. Do not add a new system
  prompt or apply a different chat template to the original prompt prose.
- Rebuild answer-only labels after processor expansion. Keep the original
  BOS/EOS convention, target `###` suffix and padding-label behavior.
- Bind original Task.train_step and Chat calls to native forward/generate.
  Keep the original runner, optimizer groups, scheduler, loop, inference
  datasets, checkpoint selection, stopping criteria, parser and NPZ schema.
- Keep the official repository and YAML unchanged. Runtime manifests record
  the native model revision, interface hashes and effective capacity settings.
- Inference explicitly selects the source's `inferenceData` branch by default
  (its literal default `merbench` has no corresponding dataset class), retaining
  all nine original benchmarks. Scoring runs the untouched main programs with
  their original result discovery. It still requires the separate official
  Qwen2.5-7B-Instruct extraction model at the original configured local path;
  it does not use the fine-tuned Thinker as a replacement judge/extractor.

## Authorized sequence capacity change

A real pinned processor measurement with the original prompt function, subtitle
`I am fine.`, eight 224x224 images and eight 32000-sample clips produced 1038
input tokens before any answer. There are 512 visual and 400 audio feature
positions, plus markers and prompt text. This is a synthetic interface example,
not a dataset-wide length measurement or a training result.

The user instructed: adjust capacity without reducing training content.
Native expanded training capacity is therefore 2048 tokens before the original
BOS/EOS additions; inference total capacity is 4096 with the original 1200
new-token budget. Sampling, image size, prompt prose, target answers, batch
sizes and training duration are unchanged. Original dataset text-token checks
remain as in the reference; the increased limit is applied after native media
expansion. An overflow beyond these capacities raises an explicit error instead
of cutting media/answers or silently selecting a different training example.
Increasing capacity is not a guarantee of identical final evaluation scores.

## Runtime compatibility

Use a fresh environment, caches and outputs. Transformers 4.52.4 supplies the
native Thinker missing from reference 4.49.0. Other reference runtime pins are
retained. On torch 2.4.0, disable the config's unused tensor-parallel placement
plan: Transformers 4.52.4 otherwise checks it against an unavailable registry.
This changes no layers/weights or math; the original runner uses DDP, not TP.

The isolated Windows CPU tests pin one CPU thread because repeated tiny-model
audio-attention evaluation varied with two threads on this torch 2.4 build.
This is a test setting, not a change to the CUDA training implementation.

## Settings inspected in the pinned code


The source is authoritative. This table is an inspection aid, not a replacement
configuration or authorization to override unlisted settings.

| Item | Pinned reference |
| --- | --- |
| Dataset | MERCaptionPlus, multiface_audio_face_text, label_type: hybird |
| LR schedule | linear_warmup_cosine_lr |
| Initial/minimum/warmup LR | 1e-5 / 1e-5 / 1e-6 |
| Weight decay | 0.05 |
| Epochs / iterations per epoch / warmup steps | 60 / 5000 / 5000 |
| Training/evaluation batch size | 3 / 3 per GPU |
| Seed / workers / AMP | 42 / 4 / True |
| Resume | null |
| Gradient accumulation | Runner default 1 when absent from YAML |
| Distributed configuration | YAML world_size: 1, distributed: True; verify effective launcher topology |
| Input length / visual sampling | max_length: 1024, n_frms: 8, image_size: 224 |
| Language-model LoRA | Default rank 16, alpha 32, dropout 0.05; original attention and MLP projection targets |
| Hybrid inference | num_beams=1, temperature=1, do_sample=True, top_p=0.9, max_new_tokens=1200, max_length=2000 |

Preserve optimizer parameter grouping/defaults, loss masking, checkpoint
save/selection behavior, benchmark membership, response parsing, emotion-wheel
mapping and metrics. Use official evaluation scripts and their dependencies.
Historical AVERE exclusions and old Qwen scores do not define evaluation scope.

## Data inspection

The two existing archives in C:/MerUniBench contain 38,416 file entries.
Their names, uncompressed sizes and stored CRC values match all 38,416 entries
in the existing retention inventory, with no missing/extra/mismatched entries.
All 19 CSV/NPZ annotation files were read successfully with ZIP CRC verification.
Media payloads were not fully reread/CRC-checked during this inspection.
This is a local consistency check, not independent proof of download provenance.

OVMERD+ contains 532 open-set labels, 532 subtitles and 332 descriptions.
These are existing source files; differing row counts are not authorization to
filter, fill or rewrite annotations. Preserve the original dataset behavior.

The official MERCaptionPlus reader uses track2_train_mercaptionplus.csv,
track3_train_mercaptionplus.csv and subtitle_chieng.csv from mer2025-dataset,
with its original audio and face inputs. Live Runpod training-data completeness
was not checked in this local cleanup. No dataset or remote resource was changed.

## Validation status

Eight CPU integration tests passed with the pinned real processor/tokenizer and
a tiny complete native Thinker (torch 2.4.0+cpu, Transformers 4.52.4, PEFT 0.13.2).
They verify media ordering/counts, answer-only masks, original prompt/label
functions, explicit overflow, frozen weights, a language-LoRA optimizer step,
bitwise-equal logits after checkpoint reconstruction, native generation, script
call bindings and source-copy exclusions. No new 7B training, GPU smoke run,
benchmark inference or scoring result is claimed. Original datasets and
author outputs remain preserved. Full fresh-runtime dependency integration,
actual training-data coverage and CUDA memory use still require verification.

For reconstruction, retain the exact official Omni revision, processor/tokenizer,
interface files, runtime dependency record and original runner checkpoints
containing language LoRA plus optimizer/scaler state. Never substitute retired
Qwen artifacts or mix author outputs with this model's results.
