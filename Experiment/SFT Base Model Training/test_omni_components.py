"""Native processor + tiny complete Thinker integration, without 7B weights.

Set OMNI_PROCESSOR_SNAPSHOT to a local official processor/tokenizer snapshot.
Official methods are compiled unchanged from their AST to avoid importing the
unrelated video/GPU/scoring dependency stack during these CPU tests.
"""
import ast
import copy
import json
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2_5OmniConfig, Qwen2_5OmniThinkerForConditionalGeneration

from omni_components import (MEDIA_MARKERS, SequenceLimitError, collate_native,
    encode_prompt, load_processor, load_thinker_config, load_trainable_checkpoint, make_lora_model,
    native_prompt, thinker_inputs)
from run_affectgpt_omni import prepare_execution, SOURCE_COMMIT

SOURCE = Path(__file__).resolve().parents[2] / "Official Repository Code/AffectGPT/AffectGPT"


def original_class(file, name, namespace):
    tree = ast.parse((SOURCE / file).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE / file), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def processor():
    snapshot = os.environ.get("OMNI_PROCESSOR_SNAPSHOT")
    if not snapshot:
        pytest.fail("OMNI_PROCESSOR_SNAPSHOT must name an official local processor snapshot")
    return load_processor(snapshot)


@pytest.fixture(scope="module")
def tokenizer(processor):
    tok = copy.deepcopy(processor.tokenizer)
    tok.bos_token = "<|im_start|>"
    tok.pad_token = tok.eos_token
    tok.add_tokens(list(MEDIA_MARKERS), special_tokens=True)
    return tok


def sample(tokenizer, frames=1, clips=1, size=28, samples=3200, answer="happy###", prompt=None):
    prompt = prompt or "###Human: <Multi><MultiHere></Multi> <AudioHere> <FaceHere> Emotion? ###Assistant: "
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return dict(text_input=torch.tensor(p + a), label=torch.tensor([-100]*len(p) + a),
                raw_face=torch.arange(3*frames*size*size).reshape(3, frames, size, size).remainder(256).float(),
                raw_audio=torch.linspace(-.1, .1, clips*samples).reshape(clips, 1, samples))


def tiny_native(processor):
    torch.manual_seed(42)
    # Pin CPU test threading: this Windows torch 2.4 build produced varying
    # audio-attention outputs with two threads, even on repeated eval forwards.
    # This setting is test-only; the GPU runtime is not modified.
    torch.set_num_threads(1)
    cfg = load_thinker_config(os.environ["OMNI_PROCESSOR_SNAPSHOT"])
    text = cfg.text_config
    text.hidden_size, text.intermediate_size = 32, 64
    text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 2, 4, 2
    text.rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]}
    vision = cfg.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size = 2, 32, 64
    vision.num_heads, vision.out_hidden_size, vision.fullatt_block_indexes = 4, 32, [1]
    audio = cfg.audio_config
    audio.encoder_layers, audio.encoder_attention_heads = 2, 4
    audio.encoder_ffn_dim, audio.d_model, audio.output_dim = 64, 32, 32
    cfg._attn_implementation = "eager"
    return Qwen2_5OmniThinkerForConditionalGeneration(cfg)


def tiny_model(processor):
    return make_lora_model(tiny_native(processor))


def test_prompt_and_media_preserved(processor, tokenizer):
    item = sample(tokenizer, frames=8, clips=8, size=224, samples=32000)
    text = tokenizer.decode(item["text_input"][item["label"] == -100], clean_up_tokenization_spaces=False)
    native, images, audio = native_prompt(processor, text, item)
    assert len(images) == len(audio) == 8
    assert np.array_equal(images[3], item["raw_face"][:, 3].permute(1, 2, 0).numpy().astype(np.uint8))
    assert np.array_equal(audio[4], item["raw_audio"][4, 0].numpy())
    image_marker = processor.vision_bos_token + processor.image_token + processor.vision_eos_token
    audio_marker = processor.audio_bos_token + processor.audio_token + processor.audio_eos_token
    restored = native.replace(image_marker*8, "<FaceHere>").replace(audio_marker*8, "<AudioHere>")
    assert restored == text.replace("<MultiHere>", "")
    encoded = encode_prompt(processor, text, item)
    ids = encoded["input_ids"]
    assert (ids == processor.tokenizer.convert_tokens_to_ids(processor.image_token)).sum() == 512
    assert (ids == processor.tokenizer.convert_tokens_to_ids(processor.audio_token)).sum() == 400
    assert encoded["image_grid_thw"].shape == (8, 3)


def test_labels_padding_and_media_order(processor, tokenizer):
    items = [sample(tokenizer), sample(tokenizer, frames=2, clips=2, answer="sad and worried###")]
    batch = collate_native(processor, tokenizer, items, 1024)["omni"]
    for i, item in enumerate(items):
        expected = torch.cat((item["label"][item["label"] != -100], torch.tensor([tokenizer.eos_token_id])))
        assert torch.equal(batch["labels"][i][batch["labels"][i] != -100], expected)
    assert batch["attention_mask"].dtype == torch.long
    assert batch["image_grid_thw"].shape[0] == 3
    assert batch["input_features"].shape[0] == 3
    assert torch.all(batch["labels"][0, -3:] == -100)


def test_original_item_and_prompt(processor, tokenizer):
    cfg = SimpleNamespace(IGNORE_INDEX=-100)
    for kind in ("IMAGE", "AUDIO", "FRAME", "FACE", "MULTI"):
        setattr(cfg, f"DEFAULT_{kind}_PATCH_TOKEN", f"<{kind.title()}Here>")
    cls = original_class("my_affectgpt/datasets/datasets/base_dataset.py", "BaseDataset",
                         dict(torch=torch, random=random, config=cfg))
    dataset = cls.__new__(cls)
    dataset.annotation = [dict(name="synthetic", subtitle="I am fine.", description="The person is happy.", ovlabel="happy")]
    dataset.label_type_candidates, dataset.label_type = ["description"], "description"
    dataset.dataset, dataset.face_or_frame = "MERCaptionPlus", "multiface_audio_face_text"
    dataset.max_length, dataset.tokenizer = 1024, tokenizer
    for kind in ("image", "video", "audio", "multi"):
        setattr(dataset, f"num_{kind}_query_token", 1)
    raw = sample(tokenizer)
    media = {key: raw.get(key) for key in ("raw_face", "raw_audio", "raw_frame", "raw_image")}
    media.update(face=None, frame=None, audio=None, image=None)
    dataset.read_frame_face_audio_text = lambda *a: media
    item = dataset[0]
    batch = collate_native(processor, tokenizer, [item], dataset.max_length)["omni"]
    assert tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100]) == "The person is happy.###" + tokenizer.eos_token
    # Original 8x224 images + 8x2s clips already exceed the 800-token inference
    # allowance (2000 total minus 1200 generation), without any answer text.
    full = sample(tokenizer, frames=8, clips=8, size=224, samples=32000)
    prompt = dataset.get_prompt_for_multimodal(dataset.face_or_frame, "I am fine.", "Please recognize all possible emotional states of the character.")
    length = encode_prompt(processor, prompt, full)["input_ids"].shape[1]
    print(f"Reference prompt with eight frames/eight clips: {length} input tokens")
    assert length > 800
    item.update(raw_face=full["raw_face"], raw_audio=full["raw_audio"])
    expanded = collate_native(processor, tokenizer, [item], 2048)["omni"]
    assert 1024 < expanded["input_ids"].shape[1] <= 2050
    assert torch.equal(expanded["labels"][0][expanded["labels"][0] != -100],
                       torch.cat((item["label"][item["label"] != -100], torch.tensor([tokenizer.eos_token_id]))))
    with pytest.raises(SequenceLimitError):
        collate_native(processor, tokenizer, [item], 1024)


def test_overflow_is_not_silent(processor, tokenizer):
    with pytest.raises(SequenceLimitError, match="No truncation"):
        collate_native(processor, tokenizer, [sample(tokenizer)], 10)


def test_native_forward_lora_step_and_reconstruction(processor, tokenizer, tmp_path):
    model = tiny_model(processor)
    batch = collate_native(processor, tokenizer, [sample(tokenizer)], 1024)["omni"]
    native = model.get_base_model()
    assert native.visual is not None and native.audio_tower is not None
    assert not hasattr(native, "multi_llama_proj")
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=.05)
    loss = model(**thinker_inputs(model, batch)).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert all(torch.equal(p, frozen_before[n]) for n, p in model.named_parameters() if not p.requires_grad)
    # The exact trainable-only filtering of the original checkpoint writer.
    params = dict(model.named_parameters())
    state = {n: v for n, v in model.state_dict().items() if n not in params or params[n].requires_grad}
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": state}, path)
    restored = tiny_model(processor)
    load_trainable_checkpoint(restored, path)
    different = [n for n, v in model.state_dict().items() if not torch.equal(v, restored.state_dict()[n])]
    assert not different, different[:10]
    model.eval()
    restored.eval()
    with torch.no_grad():
        before = model(**thinker_inputs(model, batch)).logits
        after = restored(**thinker_inputs(restored, batch)).logits
    assert torch.equal(before, after)
    del state[next(n for n in state if "lora_" in n)]
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="lacks trained"):
        load_trainable_checkpoint(restored, path)


def test_native_generate_prefix_is_excluded(processor, tokenizer):
    model = tiny_model(processor).eval()
    item = sample(tokenizer)
    prompt = tokenizer.decode(item["text_input"][item["label"] == -100], clean_up_tokenization_spaces=False)
    inputs = encode_prompt(processor, prompt, item)
    prefix_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        result = model.generate(**thinker_inputs(model, inputs), max_new_tokens=2, do_sample=False)
    assert torch.equal(result[:, :prefix_len], inputs["input_ids"])
    assert 1 <= result[:, prefix_len:].shape[1] <= 2


def test_execution_copy_includes_model_source(tmp_path, monkeypatch):
    import run_affectgpt_omni as runner
    monkeypatch.setattr(runner.subprocess, "check_output", lambda args, **kwargs: SOURCE_COMMIT if "rev-parse" in args else "")
    source = tmp_path / "source"
    for rel in ("my_affectgpt/models/affectgpt.py", "models/weights.bin", "output/author.npz", "train.py"):
        p = source / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    # A real directory at the intended link target avoids Windows symlink privileges.
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "preparation.json").write_text(json.dumps({"workflow": "native-thinker-v1"}))
    (runtime / "dataset").mkdir()
    execution = prepare_execution(source, runtime, runtime / "dataset")
    assert (execution / "my_affectgpt/models/affectgpt.py").read_text() == "my_affectgpt/models/affectgpt.py"
    assert not (execution / "models").exists()
    assert not (execution / "output/author.npz").exists()
    assert not (dataset / "train.py").exists()


def test_original_script_bindings(processor, tokenizer, monkeypatch):
    import sys
    from types import ModuleType
    from transformers import StoppingCriteria, StoppingCriteriaList
    import omni_components as bridge

    def module(name, **attrs):
        obj = ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            if parent not in sys.modules:
                module(parent)
            setattr(sys.modules[parent], leaf, obj)
        return obj

    cfg = SimpleNamespace(PATH_TO_LLM={"Qwen25": "unused"})
    namespace = dict(torch=torch, config=cfg, StoppingCriteria=StoppingCriteria,
                     StoppingCriteriaList=StoppingCriteriaList)
    original_class("my_affectgpt/conversation/conversation_video.py", "StoppingCriteriaSub", namespace)
    chat = original_class("my_affectgpt/conversation/conversation_video.py", "Chat", namespace)
    dataset_cls = type("BaseDataset", (), {})
    task_cls = type("BaseTask", (), {})
    registry = SimpleNamespace(mapping={"model_name_mapping": {}})
    module("config", PATH_TO_LLM=cfg.PATH_TO_LLM)
    module("my_affectgpt.common.registry", registry=registry)
    module("my_affectgpt.datasets.datasets.base_dataset", BaseDataset=dataset_cls)
    module("my_affectgpt.tasks.base_task", BaseTask=task_cls)
    conv = module("my_affectgpt.conversation.conversation_video", Chat=chat)
    module("my_affectgpt.models.tokenizer", load_tokenizer_from_LLM=lambda _: tokenizer)
    monkeypatch.setattr(bridge.Qwen2_5OmniThinkerForConditionalGeneration, "from_pretrained",
                        lambda *args, **kwargs: tiny_native(processor))
    bridge.install(os.environ["OMNI_PROCESSOR_SNAPSHOT"])
    model = registry.mapping["model_name_mapping"]["affectgpt"].from_config({})
    dataset = dataset_cls()
    dataset.tokenizer, dataset.max_length = tokenizer, 1024
    batch = dataset.collater([sample(tokenizer)])
    model.train()
    loss = task_cls().train_step(model, batch)
    assert torch.isfinite(loss)
    assert not model.get_base_model().visual.training
    assert not model.get_base_model().audio_tower.training
    config = SimpleNamespace(llama_model="Qwen25", num_video_query_token=1,
                             num_audio_query_token=1, num_multi_query_token=1, num_image_query_token=1)
    chat = conv.Chat(model, config, device="cpu")
    media = sample(tokenizer)
    img_list = {"audio": chat.postprocess_audio(media)[1], "face": chat.postprocess_face(media)[1],
                "frame": None, "image": None, "multi": chat.postprocess_multi(None, None)[1]}
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        answer = tokenizer("happy###", add_special_tokens=False, return_tensors="pt")["input_ids"]
        return torch.cat((kwargs["input_ids"], answer), dim=1)

    monkeypatch.setattr(model, "generate", generate)
    response = chat.answer_sample("###Human: <AudioHere><FaceHere> ###Assistant: ", img_list,
                                  max_new_tokens=1200, num_beams=1, do_sample=True, top_p=.9)
    assert response == "happy"
    assert calls[0]["max_new_tokens"] == 1200 and calls[0]["top_p"] == .9
    assert calls[0]["pixel_values"].shape[0] == 4
