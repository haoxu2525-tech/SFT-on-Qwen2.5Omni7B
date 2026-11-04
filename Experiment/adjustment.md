# Native Omni Thinker SFT decisions

Updated 2026-11-04 (Australia/Sydney).

## Required correction: published Omni audio/video path

The user rejected the independent-image/independent-audio representation and
explicitly required the complete native Thinker and published upstream code.
The earlier v1 implementation is retired. Its eight passing interface tests
were insufficient: they did not verify synchronized video/audio behavior.

Use `qwen-omni-utils==0.0.8` directly, without editing/copying/reimplementing its
loader. `process_mm_info(..., use_audio_in_video=True, return_video_kwargs=True)`
extracts video and continuous embedded audio from the SAME original file.
Native `processor.apply_chat_template`, processor token interleaving, Thinker
forward/generate and native positional computation are retained. The boolean
must be passed to both processor and model. The reference eight-frame/224x224
budget is passed through supported official utility options. Its sampled FPS
is forwarded unchanged; scalar unwrapping is required by Transformers 4.52.4.
The resized video is not resized again by the processor.

No handwritten PyAV decoder, audio clipping/resampling rule, timestamp matching
policy, feature pooling, projector, encoder replacement or fusion layer remains.
Any media behavior/limitations are those of the pinned published utility. In
particular, this does not prove arbitrary variable-frame-rate or stream-offset
support. Missing embedded audio errors instead of falling back to unrelated
WAVs or claiming unverified synchronization.

This explicitly changes media representation from face crops + eight sampled
clips to original video frames + full continuous embedded audio. Annotation
membership, subtitles, questions, answers and scoring references are retained.
Do not claim exact preservation of original sampled pixels/waveform segments.
No source media, annotation, split, exclusion or original checkpoint is changed.

## Prompt, supervision and original code boundary

Call the original AffectGPT `get_prompt_for_multimodal(..., 'textonly', ...)`
branch for its existing subtitle/question prose, then replace ONLY its old
Human/Assistant role envelope with the native Omni chat template. The actual
sample is NOT text-only: a native video message supplies the paired AV input.
The Qwen official example's system prompt is used verbatim. Original hybrid
supervision and answer suffix `###` are kept; answer labels are relocated after
native media expansion. Native role tokens already contain BOS; do not add a
second legacy BOS. Padding masks are based on actual row lengths, retaining EOS
attention and avoiding an accidental EOS-as-padding interpretation.

`BaseTask.train_step` is no longer monkey-patched. A model signature wrapper
accepts its exact `model(samples)['loss']` contract. BaseDataset's media input,
prompt serialization and collater are interface bindings only. The original
training loop, optimizer/scheduler, backward, checkpoint writer, inference
iteration, stopping criteria, response parser, membership and metric scripts
remain unchanged on disk. Inference uses the source's `min_length` argument,
not a silently substituted `min_new_tokens`. Scoring still uses the original
separate Qwen2.5-7B-Instruct extractor.

Pretrained Thinker and both native projections remain frozen. Language LoRA
uses reference r=16, alpha=32, dropout=.05 and explicit language-layer targets.
Reference FP16/AMP is retained; it is not the pretrained config's BF16 default.
No attention implementation/quantization/optimizer change is introduced.
The torch 2.4 unused TP-metadata compatibility fix is retained.

## Capacity and isolation

Training checks the complete native chat + target + EOS against 2048; inference
checks native prompt + original 1200 output budget against 4096. There is no
new truncation/fallback. Full audio may exceed these budgets on long clips:
fail and report rather than silently shorten it. The original Dataset's
upstream 1024 text-token truncation/retry remains, and is not claimed removed.

Fresh runtime workflow is `native-thinker-synchronized-av-v2`; v1 receipts and
artifacts are rejected. Preserve full checkpoint reconstruction information.
Sync only scoped files directly to the new repository main; create no branches.

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

Ten corrected CPU tests passed (49.68 seconds) against real synthetic MOV
containers with embedded audio, the published loader, pinned processor and a
tiny native Thinker. They verify exact tensors against the direct published
loader/processor call chain, audio/video interleaving and temporal-position
correspondence, duration-derived FPS, missing-audio failure, multi-sample native
forward, original target labels, unchanged original BaseTask.train_step, LoRA
updates, bitwise-equal logits after checkpoint reconstruction, generation and
original inference-call binding. Source git status is clean at the pinned
AffectGPT commit. No 7B training, CUDA memory/numerical result, complete real
training-data audit or benchmark score is claimed.

The Windows CPU test environment uses Python 3.12, torch 2.4.0+cpu, PyAV 13.1.0,
NumPy 1.26.4 and librosa 0.10.2.post1, with FFmpeg available on PATH. This differs
from the unchanged original Linux environment (Python 3.10, NumPy 1.23.5,
librosa 0.8.1); production dependency integration and numerical equivalence are
not yet verified. The requirements overlay adds only pinned Transformers and
qwen-omni-utils. CPU tests use one thread as previously documented.
