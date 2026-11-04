# Active project instructions

Read README.md, Experiment/adjustment.md and
Reproduction/Reproduction Code/readme.txt before implementation.

The current user decision supersedes the earlier three-component replacement:
use the complete official Qwen2.5-Omni-7B Thinker and its processor. Do not split
encoders out or retain AffectGPT projection, aggregation or fusion modules.

- SFT base-model code belongs in Experiment/SFT Base Model Training, the folder
  explicitly requested by the user. Keep the other existing folders.
- Preserve AffectGPT datasets, annotations, sampled media, natural-language
  prompts, target answers, hybrid supervision, runner/optimizer/scheduler,
  batch sizes, duration, decoding, checkpoint selection and scoring.
- Freeze the complete pretrained Thinker, including encoders and native
  projections; train language-layer LoRA with the reference LoRA settings.
- Use the native processor and Thinker forward/generate. Limit integration code
  to data serialization, label masks and original training/inference calls.
- The user authorized increased length capacity without reducing input or
  supervision: training 2048; inference total 4096; output budget remains 1200.
  Fail explicitly if exceeded; never silently truncate media or target answers.
- Keep official source/configuration/author outputs unchanged at commit
  95b9257758df29fa6481c138252cdd317cce1735. Author results are not our results.
- Preserve original datasets/annotations. Do not regenerate labels/splits,
  inherit retired Qwen exclusions, delete media, or import retired artifacts.
- Use fresh environment/cache/output paths. Record concrete incompatibilities;
  further experimental changes require the user's decision.
- Preserve trained weights and verify reconstruction. Distinguish CPU interface
  checks from full 7B/GPU training, evaluation and verified results.
- Sync scoped changes directly to main in haoxu2525-tech/SFT-on-Qwen2.5Omni7B.
  Do not create branches unless the user explicitly requests one.
  Exclude unrelated parent-repository changes.
