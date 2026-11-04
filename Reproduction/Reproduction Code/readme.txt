This is folder for our reproduction code, we should replicate the original paper code in our own device like local and rented pod.

Remember:
1. The structure of official code is not changeable in our reproduction code. Though there will be difficulties to actually run our reproduction on the rented device and downloading before running replicated code from official repositories, but we should never touch the content of official code. We could only add code to download datasets.

2. Compare the 'Reproduction' folder in 'Reproduction Code' folder with 'Official Paper Repository' folder, any difference except GPU adapation is taken as violation of reproduction. We only accept adapations for GPU devices difference and the adapations could adjust speed only and it must make no influence on training output.

3. We should never create new folders in 'Reproduction Code' folder. We only write code we use to download datasets in 'Download datasets' folder. We only write code to fix blockers and errors we encounter in 'Fix Blocker and Error' folder.

4. The code we write in 'Fix blocker and errors' should only never touch variables and algorithms in Reproduction. We could only solve non-experimental issues when downloading and running on a rented device there.

5. We should always synchronise our code in our github repositories for version control every time we finish editing files.

6.do not create new files for new versions of existing files after we try to update for a new goal, just overwrite the existing file

7.we download the enormous datasets on runpod storage only, in our own device we only maintain the code and results

8.all coding files in this folder should be really utilized in generation of our final version results

9.The pipeline should run all required programs on Runpod through result generation without frequent manual intervention.

10. There must exist discrepancies in availability to baseline papers as some files may require permission or be unopen to public route. List all the discrepancies carefully and wait for my decision about how we adjust our reproduction or experiment. These adjustments we make should be recorded in adjustment.md file.

Current project override (2026-11-04):
Use the complete Qwen2.5-Omni-7B Thinker and native processor, including its
encoders, projections and LLM. Do not use AffectGPT projection, aggregation or
fusion modules. Freeze pretrained Thinker weights and train language-layer
LoRA with AffectGPT's reference settings, data, supervision and evaluation.
Interface code belongs in Experiment/SFT Base Model Training. Keep official
files unchanged. The latest user instruction authorizes BF16, training capacity
4096, explicit batch-size reduction after OOM, and a source models symlink.
See Experiment/adjustment.md for the effective settings and validation limits.
Any additional experimental change requires the project maintainer's decision. Preserve original
datasets/annotations and do not reuse retired Qwen artifacts. Other folder,
storage and version-control requirements above remain in force.
