This is folder for our reproduction code, we should replicate the original paper code in our own device like local and rented pod.

Remember:
1. The structure of official code is not changeable in our reproduction code. Though there will be difficulties to actually run our reproduction on the rented device and downloading before running replicated code from official repositories, but we should never touch the content of official code. We could only add code to download datasets.

2. Compare the 'Reproduction' folder in 'Reproduction Code' folder with 'Official Paper Repository' folder, any difference except GPU adapation is taken as violation of reproduction. We only accept adapations for GPU devices difference and the adapations could adjust speed only and it must make no influence on training output.

3. We should never create new folders in 'Reproduction Code' folder. We only write code we use to download datasets in 'Download datasets' folder. We only write code to fix blockers and errors we encounter in 'Fix Blocker and Error' folder.

4. The code we write in 'Fix blocker and errors' should only never touch variables and algorithms in Reproduction. We could only solve non-experimental issues when downloading and running on a rented device there.

5. We should always synchronise our code in our github repositories for version control every time we finish editing files. I've connected HuggingFace and Github with OpenAi account as plugin so do not ask questions for login again.

6.do not create new files for new versions of existing files after we try to update for a new goal, just overwrite the existing file

7.we download the enormous datasets on runpod storage only, in our own device we only maintain the code and results

8.all coding files in this folder should be really utilized in generation of our final version results

9.We need a thorough pipeline program to run all needed programs on one go as Xu Hao is too busy to frequently check with Codex what have we done, but just arrange the compelete task on runpod and let the program run by itself till all results come out.

10. There must exist discrepancies in availability to baseline papers as some files may require permission or be unopen to public route. List all the discrepancies carefully and wait for my decision about how we adjust our reproduction or experiment. These adjustments we make should be recorded in adjustment.md file.

Current project override (2026-11-04, Australia/Sydney):
Replace only AffectGPT's language backbone and audio/visual encoders with
Qwen2.5-Omni-7B's corresponding components. Keep AffectGPT's projection,
aggregation and fusion modules, trainability policy, data, supervision,
training settings and evaluation unchanged. This is not a language-LoRA-only
or end-to-end Omni replacement. Native component interfaces may be adapted in
the existing Experiment/Experiment Code directory without editing official files.
Any additional experimental change requires Xu Hao's decision. Preserve original
datasets/annotations and do not reuse retired Qwen artifacts. Other folder,
storage and version-control requirements above remain in force.
