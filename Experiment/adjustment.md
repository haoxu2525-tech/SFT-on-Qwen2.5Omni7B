# Native Omni Thinker SFT decisions

Updated 2026-11-04.

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
The latest user instruction authorizes BF16 instead of reference FP16/AMP.
Thinker loading uses torch.bfloat16. The model signature adapter uses BF16 CUDA
autocast for forward/generate. The launcher passes run.amp=False through the
original --options API, disabling FP16 autocast/scaling in the unchanged runner.
No attention implementation/quantization/optimizer change is introduced.
The torch 2.4 unused TP-metadata compatibility fix is retained.

## Capacity and isolation

Training checks the complete native chat + target + EOS against 4096; inference
checks native prompt + original 1200 output budget against 4096. There is no
new truncation/fallback. Full audio may exceed these budgets on long clips:
fail and report rather than silently shorten it. The interface now calls the
ORIGINAL Dataset.__getitem__ through NativeTextSerialization. This view delegates
annotation lookup, original QA methods/hybrid sampling and ordinary data-error
retries to the original dataset. It changes only text serialization: no text
truncation and no preliminary text-only capacity cutoff. The view's unbounded
intermediate max_length is never passed to the tokenizer or model; the actual
4096 limit is enforced after native media expansion, outside the source retry
handler. The dataset object's max_length and the source YAML remain untouched.
The user offered either target truncation or increased training capacity.
We implement increased capacity (2048 to 4096), preserving complete answers
and EOS. Inputs still exceeding 4096 fail explicitly; no unlimited fallback.
Epochs, generation budget and labels remain unchanged.

Fresh runtime workflow is `native-thinker-synchronized-av-v4-bf16`; v1/v2/v3 receipts and
artifacts are rejected. Preserve full checkpoint reconstruction information.
Sync only scoped files directly to the new repository main; create no branches.

The latest user instruction also authorizes adjustable batch size after OOM.
The launcher accepts positive --batch-size-train and --batch-size-eval values
for train, using original run.batch_size_train/eval config overrides. Defaults
remain 3/3. Effective batch sizes and precision/capacity are recorded in
training-options.json. Lower batches change the effective batch; no automatic
gradient-accumulation/LR/duration adjustment or OOM retry is introduced.

Execution models is now a symlink to source/models, as requested. This permits
the unchanged evaluation scripts to load their separate Qwen2.5-7B-Instruct
extractor without installing the Omni adapter. No weights are copied/downloaded
by preparation, and a missing source models directory leaves a dangling link
until provisioned. Wrong links or a conflicting real directory are rejected.
Author output files remain excluded from the execution copy.

## Original inference entrypoint correction

The launcher previously supplied only --cfg-path. This exposed the upstream
script's non-runnable bare defaults: merbench has no dataset class, and
get_user_message has no assigned question when zeroshot=False without an outside
question. The launcher now supplies --dataset inferenceData --zeroshot from the
official README's OV-label command. Original script/YAML, dataset list, question
function, stopping/decoding and checkpoint selection are unchanged. Explicit
original CLI options are still forwarded, with normal argparse precedence.
The README's earlier claim about an automatically written omni-thinker.json /
interface-hash manifest was inaccurate and has been removed: preparation.json
records source commit, workflow and dataset path; model revision is pinned in code.

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

The two existing dataset archives contain 38,416 file entries.
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

Current v4 validation on 2026-11-04: 27 CPU tests passed, 3 skipped in 23.82s.
New coverage includes BF16 tiny-Thinker forward/backward, LoRA updates and
bitwise-equal checkpoint reconstruction, BF16 loader/autocast selection,
complete >2048-token answers/subtitles within 4096, and batch override dispatch,
recording and invalid-value rejection. Three real symlink integration cases
were skipped because this Windows account lacks directory-symlink privileges
(WinError 1314). Linux symlink integration and full 7B/CUDA execution remain
unverified; the source models directory/weights must be provisioned there.

Historical v3 validation: 21 CPU tests passed in 14.99 seconds on 2026-11-04.
Real synthetic MOV containers with embedded audio, the published loader, a fresh
copy of the pinned official processor (including chat_template.json), and a tiny
native Thinker verify the native AV path, forward/generation, LoRA updates and
bitwise-equal logits after checkpoint reconstruction. Additional regressions
verify original description/ovlabel/hybrid selection and RNG consumption,
complete >1024-token subtitles/answers that fit the previous 2048 budget, explicit
final overflow without random replacement, and the 4096/1200 inference boundary.
The installed Dataset wrapper is exercised as well as the original train_step.
Launcher tests intercept model download/script dispatch, execute the original
inference argparse statements and question-selector function, and check train,
infer, evaluate and score arguments. They do not run the full GPU scripts.
Official source git status remains clean at the pinned commit. No 7B training,
CUDA memory/numerical result, complete real training-data audit or benchmark
score is claimed.

The fresh isolated Windows CPU test environment uses Python 3.12.4,
torch 2.4.0+cpu, torchvision 0.19.0+cpu, Transformers 4.52.4, PEFT 0.13.2,
accelerate 1.0.0, tokenizers 0.21.0, safetensors 0.4.5, PyAV 13.1.0,
NumPy 1.26.4 and librosa 0.10.2.post1, with FFmpeg available on PATH. This differs
from the unchanged original Linux environment (Python 3.10, NumPy 1.23.5,
librosa 0.8.1); production dependency integration and numerical equivalence are
not yet verified. The requirements overlay adds only pinned Transformers and
qwen-omni-utils. CPU tests use one thread as previously documented.

To run the tests, set OMNI_PROCESSOR_SNAPSHOT to the complete pinned processor
directory, put FFmpeg on PATH, and run pytest on
Experiment/SFT Base Model Training/test_omni_components.py in the isolated test
environment. The initial local processor cache lacked chat_template.json;
fetching the original file at the pinned revision fixed the test setup without
adding a custom template or altering model code.
