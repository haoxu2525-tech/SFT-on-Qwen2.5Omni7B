# Active project instructions

Read README.md, Experiment/adjustment.md and
Reproduction/Reproduction Code/readme.txt before implementation.

The current user decision supersedes the earlier three-component replacement:
use the complete official Qwen2.5-Omni-7B Thinker and its processor. Do not split
encoders out or retain AffectGPT projection, aggregation or fusion modules.

- SFT base-model code belongs in Experiment/SFT Base Model Training, the folder
  explicitly requested by the user. Keep the other existing folders.
- Preserve AffectGPT datasets, annotations, question/subtitle prose, target answers,
  hybrid supervision, runner/optimizer/scheduler,
  duration, decoding, checkpoint selection and scoring. Batch sizes default to
  reference 3/3; the user now authorizes explicit batch-size reduction after OOM.
- Freeze the complete pretrained Thinker, including encoders and native
  projections; train language-layer LoRA with the reference LoRA settings.
- Call the published qwen-omni-utils process_mm_info directly for ONE original
  video and its continuous embedded audio. Use native chat template, processor,
  forward/generate with use_audio_in_video=True and the utility's returned FPS.
  Pass reference 8-frame/224x224 budget as official utility parameters.
  Do not write custom decoders/samplers/timestamp logic or revert to separate
  images/eight audio clips. No independent WAV fallback or invented alignment.
- Use the original AffectGPT textonly prompt branch only to obtain its existing
  question/subtitle prose; native video messages supply the actual AV content.
  Do not author new task prompts. Preserve original supervision and scoring.
- Keep BaseTask.train_step and runner unchanged; only model signature/data
  serialization/label masks/inference model calls belong in the interface.
- The user authorized increased length capacity without reducing input or
  supervision: training 4096; inference total 4096; output budget remains 1200.
  Fail explicitly if exceeded; never silently truncate media or target answers.
- Latest user decision: load Thinker in BF16, use BF16 CUDA autocast in the
  model adapter and disable original FP16 AMP/scaling through run.amp=False.
  Keep official runner/task files unchanged. Link execution/models to the
  original source/models so original scoring can access its extraction model.
- Keep official source/configuration/author outputs unchanged at commit
  95b9257758df29fa6481c138252cdd317cce1735. Author results are not our results.
- Preserve original datasets/annotations. Do not regenerate labels/splits,
  inherit retired Qwen exclusions, delete media, or import retired artifacts.
- Use fresh environment/cache/output paths. Record concrete incompatibilities;
  further experimental changes require the user's decision.
- Preserve trained weights and verify reconstruction. Distinguish CPU interface
  checks from full 7B/GPU training, evaluation and verified results.
- Sync scoped changes directly to main in haoxu2525-tech/SFT-on-Qwen2.5Omni7B.
  Do not create branches unless explicitly requested. Exclude unrelated changes.
