"""Complete native Thinker; AffectGPT data/runner/scoring remain the owners.

No encoder extraction, feature pooling, custom projector or replacement trainer.
"""
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MODEL_REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
IGNORE_INDEX = -100
TRAIN_MAX_LENGTH = 2048
INFERENCE_MAX_LENGTH = 4096
MEDIA_MARKERS = ("<ImageHere>", "<FrameHere>", "<FaceHere>", "<AudioHere>", "<MultiHere>")


class SequenceLimitError(ValueError):
    """Stop instead of silently changing samples, media or supervision."""


@lru_cache(maxsize=2)
def load_processor(snapshot):
    return Qwen2_5OmniProcessor.from_pretrained(snapshot, local_files_only=True)


def load_thinker_config(snapshot):
    cfg = Qwen2_5OmniConfig.from_pretrained(snapshot, local_files_only=True).thinker_config
    if torch.__version__.startswith("2.4.0"):
        # Transformers 4.52 validates TP metadata against a None registry on
        # torch<2.5. This run uses AffectGPT's DDP, never tensor parallelism.
        # Removing unused placement metadata changes no layer or computation.
        cfg.text_config.base_model_tp_plan = {}
    return cfg


def make_lora_model(thinker, rank=16):
    thinker.requires_grad_(False)
    targets = [f"model.layers.{i}.{suffix}"
               for i in range(len(thinker.model.layers))
               for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                              "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj")]
    model = get_peft_model(thinker, LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=rank, lora_alpha=32, lora_dropout=0.05, target_modules=targets,
    ))
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_" not in name or ".model.layers." not in name):
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
    thinker.visual.eval()
    thinker.audio_tower.eval()
    return model


def load_trainable_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    required = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint lacks trained weights: {sorted(missing)[:5]}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {result.unexpected_keys[:5]}")


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def native_prompt(processor, prompt, media):
    """Replace placeholders only, preserving all natural-language characters.

    Frames are independent images; clips are independent audios. No fabricated
    timestamps or concatenated audio. MultiHere has no external feature anymore.
    """
    images, audios = [], []

    def replace(match):
        marker = match.group()
        if marker == "<MultiHere>":
            return ""
        kind = {"<ImageHere>": "image", "<FrameHere>": "frame",
                "<FaceHere>": "face", "<AudioHere>": "audio"}[marker]
        value = media.get("raw_" + kind)
        if value is None:
            raise ValueError(f"Prompt requires missing raw {kind}")
        value = _numpy(value)
        if kind == "audio":
            if value.ndim != 3 or value.shape[1] != 1:
                raise ValueError("Expected original audio clips [A,1,S]")
            audios.extend(np.asarray(x[0], dtype=np.float32) for x in value)
            return (processor.audio_bos_token + processor.audio_token
                    + processor.audio_eos_token) * len(value)
        if value.ndim != 4 or value.shape[0] != 3:
            raise ValueError("Expected original visual frames [3,T,H,W]")
        frames = value.transpose(1, 2, 3, 0).astype(np.uint8)
        images.extend(frames)
        return (processor.vision_bos_token + processor.image_token
                + processor.vision_eos_token) * len(frames)

    text = re.sub("|".join(map(re.escape, MEDIA_MARKERS)), replace, prompt)
    if "<MultiHere>" in prompt and not images and not audios:
        raise ValueError("Fusion-only prompt has no native media positions")
    return text, images, audios


def encode_prompt(processor, prompt, media):
    text, images, audios = native_prompt(processor, prompt, media)
    kwargs = dict(text=text, return_tensors="pt", add_special_tokens=False,
                  padding=False, truncation=False, use_audio_in_video=False)
    if images:
        kwargs.update(images=images, images_kwargs={"do_resize": False})
    if audios:
        kwargs.update(audio=audios, sampling_rate=16000)
    return dict(processor(**kwargs))


def collate_native(processor, tokenizer, instances, max_length):
    """Retain original sampled media/target IDs; align masks after expansion."""
    rows, labels, media_rows = [], [], []
    for sample in instances:
        original_ids, original_labels = sample["text_input"], sample["label"]
        prefix_len = int((original_labels == IGNORE_INDEX).sum())
        if not torch.all(original_labels[:prefix_len] == IGNORE_INDEX):
            raise ValueError("Expected a masked prompt followed by an answer")
        target = original_ids[prefix_len:].long()
        if not torch.equal(target, original_labels[prefix_len:].long()):
            raise ValueError("Original answer labels and token IDs disagree")
        prompt_ids = original_ids[:prefix_len].tolist()
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        if tokenizer(prompt, add_special_tokens=False)["input_ids"] != prompt_ids:
            raise ValueError("Original prompt cannot be losslessly decoded")
        inputs = encode_prompt(processor, prompt, sample)
        prefix = inputs.pop("input_ids")[0]
        inputs.pop("attention_mask", None)
        if len(prefix) + len(target) > max_length:
            raise SequenceLimitError(
                f"Expanded prompt {len(prefix)} + answer {len(target)} exceeds "
                f"expanded-sequence capacity={max_length}. No truncation or sample replacement was performed."
            )
        # Original collater adds BOS/EOS after the max_length check.
        rows.append(torch.cat((torch.tensor([tokenizer.bos_token_id]), prefix,
                               target, torch.tensor([tokenizer.eos_token_id]))))
        labels.append(torch.cat((torch.full((len(prefix) + 1,), IGNORE_INDEX),
                                 target, torch.tensor([tokenizer.eos_token_id]))))
        media_rows.append(inputs)
    ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True,
                                          padding_value=tokenizer.pad_token_id)
    batch = dict(input_ids=ids, attention_mask=ids.ne(tokenizer.pad_token_id).long(),
                 labels=torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                        padding_value=IGNORE_INDEX))
    for key in set().union(*(row.keys() for row in media_rows)):
        batch[key] = torch.cat([row[key] for row in media_rows if key in row], dim=0)
    return {"omni": batch}


def thinker_inputs(model, inputs):
    dtype = next(model.parameters()).dtype
    return {k: v.to(device=model.device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()}


def install(snapshot):
    """Bind native calls in memory; never modify official files."""
    import config
    from my_affectgpt.common.registry import registry
    from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
    from my_affectgpt.tasks.base_task import BaseTask
    from my_affectgpt.conversation import conversation_video

    snapshot = str(Path(snapshot).resolve())
    config.PATH_TO_LLM["Qwen25"] = snapshot


    class NativeThinkerFactory:
        @classmethod
        def from_config(cls, cfg):
            native = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                snapshot, config=load_thinker_config(snapshot),
                local_files_only=True, torch_dtype=torch.float16,
            )
            model = make_lora_model(native, cfg.get("lora_r", 16))
            for key in ("ckpt", "ckpt_2", "ckpt_3"):
                if cfg.get(key):
                    load_trainable_checkpoint(model, cfg[key])
            return model

    def collater(dataset, instances):
        return collate_native(load_processor(snapshot), dataset.tokenizer,
                              instances, TRAIN_MAX_LENGTH)

    def train_step(task, model, samples):
        core = model.module if hasattr(model, "module") else model
        core.get_base_model().visual.eval()
        core.get_base_model().audio_tower.eval()
        return model(**thinker_inputs(core, samples["omni"])).loss

    original_chat = conversation_video.Chat

    class NativeChat(original_chat):
        def __init__(self, model, model_cfg, device="cuda:0"):
            from my_affectgpt.models.tokenizer import load_tokenizer_from_LLM
            model.llama_tokenizer = load_tokenizer_from_LLM(model_cfg.llama_model)
            super().__init__(model, model_cfg, device)
            self.processor = load_processor(snapshot)

        def postprocess_audio(self, sample):
            return None, sample.get("raw_audio")

        def postprocess_face(self, sample):
            return None, sample.get("raw_face")

        def postprocess_frame(self, sample):
            return None, sample.get("raw_frame")

        def postprocess_image(self, sample):
            return None, sample.get("raw_image")

        def postprocess_multi(self, *args):
            return None, None

        @torch.no_grad()
        def answer_sample(self, prompt, img_list, num_beams=1, temperature=1.0,
                          do_sample=True, top_p=0.9, max_new_tokens=1000,
                          min_length=1, max_length=2000, repetition_penalty=1.0,
                          length_penalty=1.0):
            media = {"raw_" + k: v for k, v in img_list.items() if k != "multi"}
            inputs = encode_prompt(self.processor, prompt, media)
            prefix_length = inputs["input_ids"].shape[1]
            max_length = INFERENCE_MAX_LENGTH
            if prefix_length + max_new_tokens > max_length:
                raise SequenceLimitError(
                    f"Expanded inference prompt {prefix_length} + generation budget "
                    f"{max_new_tokens} exceeds configured capacity {max_length}; no media was cut."
                )
            output = self.model.generate(
                **thinker_inputs(self.model, inputs),
                num_beams=num_beams, temperature=temperature, do_sample=do_sample,
                top_p=top_p, max_new_tokens=max_new_tokens, min_new_tokens=min_length,
                repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                stopping_criteria=self.stopping_criteria,
            )
            response = self.tokenizer.decode(output[0, prefix_length:], skip_special_tokens=False)
            if self.tokenizer.bos_token in response:
                response = response.split(self.tokenizer.bos_token)[1]
            if self.tokenizer.eos_token in response:
                response = response.split(self.tokenizer.eos_token)[0]
            return response.rsplit("###", 1)[0].split("Assistant:")[-1].strip()

    registry.mapping["model_name_mapping"]["affectgpt"] = NativeThinkerFactory
    BaseDataset.collater = collater
    BaseTask.train_step = train_step
    conversation_video.Chat = NativeChat
