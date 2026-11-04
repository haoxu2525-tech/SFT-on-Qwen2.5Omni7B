"""Complete native Thinker; AffectGPT data/runner/scoring remain the owners.

No encoder extraction, feature pooling, custom projector or replacement trainer.
"""
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MODEL_REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
IGNORE_INDEX = -100
TRAIN_MAX_LENGTH = 2048
INFERENCE_MAX_LENGTH = 4096
MEDIA_MARKERS = ("<ImageHere>", "<FrameHere>", "<FaceHere>", "<AudioHere>", "<MultiHere>")
VIDEO_FRAMES = 8


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


@dataclass(frozen=True)
class VideoSource:
    """One original container owns BOTH streams and their shared clock."""
    path: str


def native_messages(prompt, source):
    """AffectGPT's original textonly prompt inside Qwen's official chat format."""
    if not isinstance(source, VideoSource):
        raise ValueError("Expected original video file, not separate images/audio clips")
    prefix, suffix = "###Human: ", " ###Assistant: "
    if not prompt.startswith(prefix) or not prompt.endswith(suffix):
        raise ValueError("Expected the original AffectGPT textonly prompt envelope")
    text = prompt[len(prefix):-len(suffix)]
    if any(marker in text for marker in MEDIA_MARKERS):
        raise ValueError("Legacy media placeholders must not enter native Omni chat")
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "video", "video": source.path, "nframes": VIDEO_FRAMES,
             "resized_height": 224, "resized_width": 224},
            {"type": "text", "text": text},
        ]},
    ]


def encode_prompt(processor, prompt, source):
    # Qwen official example: apply_chat_template -> process_mm_info -> processor.
    # Decoder, sampling, continuous audio extraction and sampled FPS all belong
    # to the published utility. No local AV decoder or position calculation.
    from qwen_omni_utils import process_mm_info
    messages = native_messages(prompt, source)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos, video_kwargs = process_mm_info(
        messages, use_audio_in_video=True, return_video_kwargs=True,
    )
    if len(videos) != 1 or len(audios) != 1 or len(video_kwargs["fps"]) != 1:
        raise ValueError("Expected one original video and its embedded continuous audio track")
    # qwen-omni-utils returns one FPS per video; pinned Transformers 4.52.4
    # accepts a scalar FPS. Each dataset row here contains exactly one video.
    inputs = dict(processor(
        text=text, images=images, videos=videos, audio=audios,
        return_tensors="pt", add_special_tokens=False, padding=False,
        truncation=False, use_audio_in_video=True, fps=video_kwargs["fps"][0],
        do_resize=False,  # official utility already applied the reference 224x224 size
    ))
    # Validate actual audio/video tokens, not just a boolean in our Python call.
    audio_lengths = (inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
    audio_lengths = (audio_lengths - 2) // 2 + 1
    video_lengths = inputs["video_grid_thw"].prod(-1) // processor.video_processor.merge_size**2
    for token, expected in ((processor.audio_token, audio_lengths.sum()),
                            (processor.video_token, video_lengths.sum())):
        count = (inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids(token)).sum()
        if count != expected:
            raise ValueError(f"Native synchronized token/feature mismatch for {token}: {count} vs {expected}")
    # Processor does not return this flag; Thinker needs it in BOTH forward and generate.
    inputs["use_audio_in_video"] = True
    return inputs


def collate_native(processor, tokenizer, instances, max_length):
    """Retain original target IDs; align masks after native video expansion."""
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
        inputs = encode_prompt(processor, prompt, sample["raw_frame"])
        prefix = inputs.pop("input_ids")[0]
        inputs.pop("attention_mask", None)
        inputs.pop("use_audio_in_video")
        if len(prefix) + len(target) + 1 > max_length:
            raise SequenceLimitError(
                f"Expanded prompt {len(prefix)} + answer {len(target)} + EOS exceeds "
                f"expanded-sequence capacity={max_length}. No truncation or sample replacement was performed."
            )
        # Native chat template already supplies role/BOS delimiters.
        rows.append(torch.cat((prefix, target, torch.tensor([tokenizer.eos_token_id]))))
        labels.append(torch.cat((torch.full((len(prefix),), IGNORE_INDEX),
                                 target, torch.tensor([tokenizer.eos_token_id]))))
        media_rows.append(inputs)
    ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True,
                                          padding_value=processor.tokenizer.pad_token_id)
    lengths = torch.tensor([len(row) for row in rows])
    batch = dict(input_ids=ids, attention_mask=(torch.arange(ids.shape[1])[None] < lengths[:, None]).long(),
                 labels=torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                        padding_value=IGNORE_INDEX))
    for key in set().union(*(row.keys() for row in media_rows)):
        values = [row[key] for row in media_rows if key in row]
        batch[key] = torch.cat(values, dim=0)
    batch["use_audio_in_video"] = True
    return {"omni": batch}


def thinker_inputs(model, inputs):
    dtype = next(model.parameters()).dtype
    media_keys = {"pixel_values", "pixel_values_videos", "input_features"}
    return {k: v.to(device=model.device, dtype=dtype if k in media_keys else v.dtype) if torch.is_tensor(v) else v
            for k, v in inputs.items()}


class AffectGPTThinker(torch.nn.Module):
    """Model signature adapter: original BaseTask.train_step stays unchanged."""
    def __init__(self, thinker):
        super().__init__()
        self.thinker = thinker

    @property
    def device(self):
        return self.thinker.device

    def train(self, mode=True):
        super().train(mode)
        self.thinker.get_base_model().visual.eval()
        self.thinker.get_base_model().audio_tower.eval()
        return self

    def forward(self, samples):
        return {"loss": self.thinker(**thinker_inputs(self.thinker, samples["omni"])).loss}

    def generate(self, **kwargs):
        return self.thinker.generate(**kwargs)


def install(snapshot):
    """Bind native calls in memory; never modify official files."""
    import config
    from my_affectgpt.common.registry import registry
    from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
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
            model = AffectGPTThinker(make_lora_model(native, cfg.get("lora_r", 16)))
            for key in ("ckpt", "ckpt_2", "ckpt_3"):
                if cfg.get(key):
                    load_trainable_checkpoint(model, cfg[key])
            return model

    def collater(dataset, instances):
        return collate_native(load_processor(snapshot), dataset.tokenizer,
                              instances, TRAIN_MAX_LENGTH)

    def read_media(dataset, video_path=None, face_npy=None, audio_path=None, image_path=None):
        # Paths come from the unchanged original dataset/index. Decode in collater,
        # outside the source __getitem__ retry block: AV failures are never resampled.
        if video_path is None:
            raise ValueError("Original synchronized video path is required")
        result = {k: None for k in ("face", "frame", "image", "audio", "raw_face", "raw_image", "raw_audio")}
        result["raw_frame"] = VideoSource(str(video_path))
        return result

    original_chat = conversation_video.Chat
    original_prompt = BaseDataset.get_prompt_for_multimodal

    def text_prompt(dataset, face_or_frame, subtitle, user_message):
        # Reuse published prompt prose verbatim; native Omni supplies media/roles.
        return original_prompt(dataset, "textonly", subtitle, user_message)

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
            inputs = encode_prompt(self.processor, prompt, img_list["frame"])
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
                top_p=top_p, max_new_tokens=max_new_tokens, min_length=min_length,
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
    BaseDataset.read_frame_face_audio_text = read_media
    BaseDataset.get_prompt_for_multimodal = text_prompt
    conversation_video.Chat = NativeChat
