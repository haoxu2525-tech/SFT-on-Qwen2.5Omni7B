"""Real synchronized-container + native processor/Thinker interface tests."""
import ast
import copy
import json
import os
import random
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

from omni_components import (MEDIA_MARKERS, TRAIN_MAX_LENGTH, SequenceLimitError, AffectGPTThinker,
    NativeTextSerialization,
    native_messages, collate_native, encode_prompt, load_processor,
    load_thinker_config, load_trainable_checkpoint, make_lora_model, thinker_inputs)
from run_affectgpt_omni import prepare_execution, SOURCE_COMMIT, WORKFLOW

SOURCE = Path(__file__).resolve().parents[2] / "Official Repository Code/AffectGPT/AffectGPT"
PROMPT = (
    "###Human: The subtitle of this video is: <Subtitle>I am fine.</Subtitle>. "
    "Now, please answer my question based on all the provided information. What is the emotion? ###Assistant: "
)


def original_class(file, name, namespace):
    tree = ast.parse((SOURCE / file).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE / file), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def processor():
    torch.set_num_threads(1)
    return load_processor(os.environ["OMNI_PROCESSOR_SNAPSHOT"])


@pytest.fixture(scope="module")
def tokenizer(processor):
    tok = copy.deepcopy(processor.tokenizer)
    tok.bos_token = "<|im_start|>"
    tok.pad_token = tok.eos_token
    tok.add_tokens(list(MEDIA_MARKERS), special_tokens=True)
    return tok


def write_av(path, seconds=4, fps=8, audio=True, audio_offset=0):
    """One real MOV container, frame-index pixels and continuous PCM audio."""
    with av.open(str(path), "w", format="mov") as out:
        video = out.add_stream("mpeg4", rate=fps)
        video.width = video.height = 56
        video.pix_fmt = "yuv420p"
        sound = out.add_stream("pcm_s16le", rate=16000) if audio else None
        if sound is not None:
            sound.layout = "mono"
        for i in range(seconds * fps):
            pixels = np.full((56, 56, 3), i * 5, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if sound is not None:
            t = np.arange(seconds * 16000) / 16000
            wave = (.1 * np.sin(2 * np.pi * np.where(t < seconds/2, 220, 440) * t) * 32767).astype(np.int16)
            for start in range(0, len(wave), 1024):
                frame = av.AudioFrame.from_ndarray(wave[None, start:start+1024], format="s16", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = 16000, start + audio_offset, Fraction(1,16000)
                for packet in sound.encode(frame):
                    out.mux(packet)
            for packet in sound.encode():
                out.mux(packet)
    return str(path)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return write_av(tmp_path_factory.mktemp("synchronized") / "sample.mov")


def sample(tokenizer, video, answer="happy###", prompt=PROMPT):
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return dict(text_input=torch.tensor(prefix+target), label=torch.tensor([-100]*len(prefix)+target), raw_frame=video)


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


def test_matches_published_qwen_pipeline(processor, video):
    from qwen_omni_utils import process_mm_info
    messages = native_messages(PROMPT, video)
    audios, images, videos, video_kwargs = process_mm_info(messages, use_audio_in_video=True, return_video_kwargs=True)
    assert len(audios) == len(videos) == 1 and images is None
    assert len(audios[0]) == 64000  # complete continuous four seconds, not eight chunks
    assert videos[0].shape == (8,3,224,224)
    assert video_kwargs["fps"] == [2.0]
    reference = processor(
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=images, videos=videos, audio=audios, return_tensors="pt",
        add_special_tokens=False, padding=False, truncation=False,
        use_audio_in_video=True, fps=video_kwargs["fps"][0], do_resize=False,
    )
    actual = encode_prompt(processor,PROMPT,video)
    for key, value in reference.items():
        assert torch.equal(value,actual[key]),key
    assert actual["use_audio_in_video"] is True


def test_missing_audio_and_legacy_media_fail(processor, tmp_path):
    silent = write_av(tmp_path / "silent.mov", audio=False)
    with pytest.raises(AssertionError, match="audio track"):
        encode_prompt(processor,PROMPT,silent)
    with pytest.raises(ValueError, match="separate images/audio clips"):
        native_messages(PROMPT,torch.zeros(3,8,224,224))


def test_native_video_interleaving_and_time_positions(processor, video):
    inputs = encode_prompt(processor, PROMPT, video)
    assert inputs["use_audio_in_video"] is True
    assert "pixel_values" not in inputs and "image_grid_thw" not in inputs
    assert inputs["video_grid_thw"].tolist() == [[4,16,16]]
    assert inputs["video_second_per_grid"].tolist() == [1.0]
    ids = inputs["input_ids"][0]
    vid = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    aud = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    assert (ids == vid).sum() == 256
    assert (ids == aud).sum() == 100
    # Native two-second chunks: eight video tokens, fifty audio tokens, repeated.
    media_ids = ids[(ids == vid) | (ids == aud)].tolist()
    assert media_ids == [vid]*128 + [aud]*50 + [vid]*128 + [aud]*50
    native = tiny_native(processor)
    positions, _ = native.get_rope_index(
        inputs["input_ids"], video_grid_thw=inputs["video_grid_thw"],
        attention_mask=inputs["attention_mask"], use_audio_in_video=True,
        audio_seqlens=inputs["feature_attention_mask"].sum(-1),
        second_per_grids=inputs["video_second_per_grid"],
    )
    video_t = positions[0,0,ids == vid].reshape(4,64)[:,0]
    audio_t = positions[0,0,ids == aud]
    assert torch.equal(video_t-video_t[0], torch.tensor([0,25,50,75]))
    assert video_t[0] == audio_t[0] and video_t[2] == audio_t[50]
    text = processor.apply_chat_template(native_messages(PROMPT, video), tokenize=False, add_generation_prompt=True)
    assert text[0].endswith("<|im_start|>assistant\n")
    assert "merged info" not in text[0] and "###Human" not in text[0] and "<FaceHere>" not in text[0]
    assert "<Subtitle>I am fine.</Subtitle>" in text[0] and "What is the emotion?" in text[0]


def test_frame_interval_comes_from_source_duration(processor, tmp_path):
    slower = write_av(tmp_path / "longer.mov", seconds=8)
    inputs = encode_prompt(processor, PROMPT, slower)
    assert inputs["video_second_per_grid"].tolist() == [2.0]
    ids = inputs["input_ids"][0]
    assert (ids == processor.tokenizer.convert_tokens_to_ids(processor.audio_token)).sum() == 200


def test_labels_padding_and_native_batch_forward(processor, tokenizer, video):
    items = [sample(tokenizer, video), sample(tokenizer, video, answer="sad and worried###")]
    batch = collate_native(processor, tokenizer, items, 2048)["omni"]
    assert batch["use_audio_in_video"] is True
    assert batch["video_grid_thw"].shape == (2,3)
    assert batch["input_features"].shape[0] == 2
    for i, item in enumerate(items):
        expected = torch.cat((item["label"][item["label"] != -100], torch.tensor([tokenizer.eos_token_id])))
        assert torch.equal(batch["labels"][i][batch["labels"][i] != -100], expected)
        valid = int(batch["attention_mask"][i].sum())
        assert batch["labels"][i, valid-1] == tokenizer.eos_token_id
        assert torch.all(batch["labels"][i, valid:] == -100)
    model = tiny_model(processor)
    assert torch.isfinite(model(**thinker_inputs(model, batch)).loss)
    with pytest.raises(SequenceLimitError, match="No truncation"):
        collate_native(processor, tokenizer, items, 10)


def reference_dataset(tokenizer, video):
    cfg = SimpleNamespace(IGNORE_INDEX=-100)
    for kind in ("IMAGE", "AUDIO", "FRAME", "FACE", "MULTI"):
        setattr(cfg, f"DEFAULT_{kind}_PATCH_TOKEN", f"<{kind.title()}Here>")
    cls = original_class("my_affectgpt/datasets/datasets/base_dataset.py", "BaseDataset", dict(torch=torch, random=random, config=cfg))
    dataset = cls.__new__(cls)
    dataset.annotation = [dict(name="synthetic", subtitle="I am fine.", description="The person is happy.", ovlabel="happy")]
    dataset.label_type_candidates, dataset.label_type = ["description", "ovlabel"], "description"
    dataset.dataset, dataset.face_or_frame = "MERCaptionPlus", "multiface_audio_face_text"
    dataset.max_length, dataset.tokenizer = 1024, tokenizer
    for kind in ("image", "video", "audio", "multi"):
        setattr(dataset, f"num_{kind}_query_token", 1)
    media = {key:None for key in ("raw_face", "raw_audio", "raw_frame", "raw_image", "face", "frame", "audio", "image")}
    media["raw_frame"] = video
    dataset.read_frame_face_audio_text = lambda *args: media
    dataset.get_prompt_for_multimodal = lambda mode, subtitle, question: cls.get_prompt_for_multimodal(dataset, "textonly", subtitle, question)
    return dataset


@pytest.mark.parametrize("label_type", ["description", "ovlabel", "hybird"])
def test_original_item_preserves_supervision(processor, tokenizer, video, label_type):
    dataset = reference_dataset(tokenizer, video)
    dataset.label_type = label_type
    rng = random.getstate()
    item = dataset[0]
    after = random.getstate()
    random.setstate(rng)
    adapted = type(dataset).__getitem__(NativeTextSerialization(dataset), 0)
    assert random.getstate() == after
    assert torch.equal(item["text_input"], adapted["text_input"])
    assert torch.equal(item["label"], adapted["label"])
    batch = collate_native(processor, tokenizer, [adapted], 2048)["omni"]
    expected = torch.cat((item["label"][item["label"] != -100], torch.tensor([tokenizer.eos_token_id])))
    assert torch.equal(batch["labels"][0][batch["labels"][0] != -100], expected)


@pytest.mark.parametrize("long_field,words,overflows", [
    ("description", 1100, False), ("subtitle", 1100, False),
    ("description", 2200, False), ("subtitle", 2200, False),
    ("description", 4300, True),
])
def test_original_item_defers_capacity_without_truncation_or_replacement(
        processor, tokenizer, video, long_field, words, overflows):
    dataset = reference_dataset(tokenizer, video)
    dataset.annotation[0][long_field] = "happy " * words
    reads = []

    class Annotations(list):
        def __getitem__(self, index):
            reads.append(index)
            return super().__getitem__(index)

    dataset.annotation = Annotations(dataset.annotation)
    item = type(dataset).__getitem__(NativeTextSerialization(dataset), 0)
    assert reads == [0]  # no exception/retry discarded the requested long row
    assert dataset.max_length == 1024  # no mutation of original dataset/YAML
    assert len(item["text_input"]) > 1024
    decoded = tokenizer.decode(item["text_input"], clean_up_tokenization_spaces=False)
    assert "happy " * words in decoded
    expected_answer = ("happy " * words if long_field == "description" else "The person is happy.") + "###"
    assert tokenizer.decode(item["label"][item["label"] != -100]) == expected_answer
    if overflows:
        with pytest.raises(SequenceLimitError, match="capacity=4096"):
            collate_native(processor, tokenizer, [item], TRAIN_MAX_LENGTH)
    else:
        batch = collate_native(processor, tokenizer, [item], TRAIN_MAX_LENGTH)["omni"]
        assert tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100]) == expected_answer + tokenizer.eos_token
    assert reads == [0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_train_step_lora_and_checkpoint(processor, tokenizer, video, tmp_path, dtype):
    model = AffectGPTThinker(make_lora_model(tiny_native(processor).to(dtype=dtype))).train()
    task_cls = original_class("my_affectgpt/tasks/base_task.py", "BaseTask",
                              dict(torch=torch, main_process=lambda f:f))
    batch = collate_native(processor, tokenizer, [sample(tokenizer,video)], 2048)
    before = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=.05)
    loss = task_cls().train_step(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if not p.requires_grad)
    assert not model.thinker.get_base_model().visual.training
    assert not model.thinker.get_base_model().audio_tower.training
    params = dict(model.named_parameters())
    state = {n:v for n,v in model.state_dict().items() if n not in params or params[n].requires_grad}
    path = tmp_path / "checkpoint.pth"
    torch.save({"model":state},path)
    restored = AffectGPTThinker(make_lora_model(tiny_native(processor).to(dtype=dtype)))
    load_trainable_checkpoint(restored,path)
    model.eval(); restored.eval()
    with torch.no_grad():
        left = model.thinker(**thinker_inputs(model.thinker,batch["omni"])).logits
        right = restored.thinker(**thinker_inputs(restored.thinker,batch["omni"])).logits
    assert torch.equal(left,right)
    del state[next(n for n in state if "lora_" in n)]
    torch.save({"model":state},path)
    with pytest.raises(ValueError,match="lacks trained"):
        load_trainable_checkpoint(restored,path)


def test_adapter_selects_bf16_autocast(monkeypatch):
    from contextlib import nullcontext
    calls = []
    def autocast(device_type, **kwargs):
        calls.append((device_type, kwargs))
        return nullcontext()
    monkeypatch.setattr(torch, "autocast", autocast)
    adapter = AffectGPTThinker(SimpleNamespace(device=torch.device("cuda")))
    with adapter.precision_context():
        pass
    assert calls == [("cuda", {"dtype": torch.bfloat16})]


@pytest.mark.parametrize("flags", [
    ["train", "--batch-size-train", "0"],
    ["train", "--batch-size-eval", "-1"],
    ["infer", "--batch-size-train", "1"],
])
def test_batch_options_reject_invalid_requests(tmp_path, monkeypatch, flags):
    import sys
    import run_affectgpt_omni as launcher
    def unexpected_prepare(*args):
        pytest.fail("Invalid options must fail before preparing a runtime")
    monkeypatch.setattr(launcher, "prepare_execution", unexpected_prepare)
    monkeypatch.setattr(sys, "argv", ["launcher", *flags, "--runtime", str(tmp_path),
                                    "--dataset-root", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2


def test_native_generate_passes_synchronization(processor, video):
    model = tiny_model(processor).eval()
    inputs = encode_prompt(processor,PROMPT,video)
    with torch.no_grad():
        output = model.generate(**thinker_inputs(model,inputs),max_new_tokens=2,do_sample=False)
    n = inputs["input_ids"].shape[1]
    assert torch.equal(output[:,:n],inputs["input_ids"])
    assert 1 <= output.shape[1]-n <= 2


@pytest.mark.parametrize("retired_workflow", ["native-thinker-v1", "native-thinker-synchronized-av-v2",
                                            "native-thinker-synchronized-av-v3"])
def test_execution_isolation(tmp_path, monkeypatch, retired_workflow):
    import run_affectgpt_omni as runner
    monkeypatch.setattr(runner.subprocess,"check_output",lambda args,**kwargs:SOURCE_COMMIT if "rev-parse" in args else "")
    source=tmp_path/"source"
    for rel in ("my_affectgpt/models/affectgpt.py","models/weights.bin","output/author.npz","train.py"):
        path=source/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(rel)
    runtime=tmp_path/"runtime"; runtime.mkdir(); (runtime/"dataset").mkdir()
    receipt=runtime/"preparation.json"
    receipt.write_text(json.dumps({"workflow":retired_workflow}))
    with pytest.raises(ValueError,match="fresh runtime"):
        prepare_execution(source,runtime,runtime/"dataset")
    receipt.write_text(json.dumps({"workflow":WORKFLOW}))
    # Real filesystem integration: do not pretend a mocked link proves access.
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(source / "models", target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows requires Developer Mode/admin for real directory symlinks")
        raise
    probe.unlink()
    execution=prepare_execution(source,runtime,runtime/"dataset")
    assert (execution/"my_affectgpt/models/affectgpt.py").is_file()
    assert (execution/"models").is_symlink()
    assert (execution/"models").resolve() == (source/"models").resolve()
    assert (execution/"models/weights.bin").read_text() == "models/weights.bin"
    assert not (execution/"output/author.npz").exists()
    assert prepare_execution(source,runtime,runtime/"dataset") == execution
    # A pre-existing link to different model assets must not be accepted.
    (execution/"models").unlink()
    (execution/"models").symlink_to(runtime/"dataset", target_is_directory=True)
    with pytest.raises(ValueError, match="Runtime models"):
        prepare_execution(source,runtime,runtime/"dataset")


@pytest.mark.parametrize("stage,extra", [
    ("train", []), ("train", ["--batch-size-train", "1", "--batch-size-eval", "2"]),
    ("infer", []),
    ("infer", ["--dataset", "MER2023", "--options", "inference.test_epoch=30"]),
    ("evaluate", []), ("score", []),
])
def test_launcher_dispatches_original_scripts(tmp_path, monkeypatch, stage, extra):
    import argparse
    import sys
    import huggingface_hub
    import omni_components as bridge
    import run_affectgpt_omni as launcher
    calls, installs = [], []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        monkeypatch.setenv(key, "test-before-launch")
    monkeypatch.setattr(launcher, "prepare_execution", lambda *args: SOURCE)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: "test-snapshot")
    monkeypatch.setattr(bridge, "install", lambda snapshot: installs.append(snapshot))
    monkeypatch.setattr(launcher.runpy, "run_path", lambda path, **kwargs: calls.append((path, list(sys.argv))))
    monkeypatch.setattr(sys, "argv", ["launcher", stage, "--runtime", str(tmp_path),
                                    "--dataset-root", str(tmp_path),
                                    *([] if stage == "train" else ["--"]), *extra])
    launcher.main()
    path, argv = calls[0]
    assert Path(path) == SOURCE / launcher.SCRIPTS[stage]
    if stage in {"evaluate", "score"}:
        assert argv == [launcher.SCRIPTS[stage]] and installs == []
        return
    assert installs == ["test-snapshot"]
    assert argv[1:3] == ["--cfg-path", launcher.REFERENCE_CONFIG]
    if stage == "train":
        expected = ["run.amp=False"]
        if extra:
            expected += ["run.batch_size_train=1", "run.batch_size_eval=2"]
        assert argv[3:] == ["--options", *expected]
        record = json.loads((tmp_path/"training-options.json").read_text())
        assert record["precision"] == "bfloat16" and record["gradient_scaling"] is False
        assert record["batch_size_train"] == (1 if extra else 3)
        assert record["batch_size_eval"] == (2 if extra else 3)
        assert record["train_max_length"] == TRAIN_MAX_LENGTH == 4096
        return
    # Execute the ACTUAL original argparse statements against dispatched argv.
    tree = ast.parse((SOURCE / "inference_hybird.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test))
    parser_nodes = []
    for node in main.body:
        parser_nodes.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            break
    namespace = dict(argparse=argparse)
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), "original-inference-cli", "exec"), namespace)
    args = namespace["args"]
    assert args.dataset == ("MER2023" if extra else "inferenceData")
    assert args.zeroshot is True and args.outside_user_message is None
    assert args.options == (["inference.test_epoch=30"] if extra else None)
    # Check original question selection reaches the existing dataset function.
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_user_message")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original-question", "exec"), namespace)
    selected = []
    dataset = SimpleNamespace(func_get_qa_ovlabel=lambda **kwargs: selected.append(kwargs) or "original question")
    assert namespace["get_user_message"](dataset, args.zeroshot, args.outside_user_message) == "original question"
    assert selected == [dict(sample=None, question_only=True)]


def test_install_preserves_task_and_inference_contract(processor, tokenizer, video, monkeypatch):
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
            monkeypatch.setattr(sys.modules[parent], leaf, obj, raising=False)
        return obj

    cfg = SimpleNamespace(PATH_TO_LLM={"Qwen25":"unused"})
    for kind in ("IMAGE", "AUDIO", "FRAME", "FACE", "MULTI"):
        setattr(cfg, f"DEFAULT_{kind}_PATCH_TOKEN", f"<{kind.title()}Here>")
    dataset_cls = original_class("my_affectgpt/datasets/datasets/base_dataset.py", "BaseDataset",
                                 dict(torch=torch, random=random, config=cfg))
    original_qa = dataset_cls.get_qa_pairs
    task_cls = original_class("my_affectgpt/tasks/base_task.py", "BaseTask", dict(torch=torch))
    original_step = task_cls.train_step
    namespace = dict(torch=torch, config=cfg, StoppingCriteria=StoppingCriteria,
                     StoppingCriteriaList=StoppingCriteriaList)
    original_class("my_affectgpt/conversation/conversation_video.py", "StoppingCriteriaSub", namespace)
    chat_cls = original_class("my_affectgpt/conversation/conversation_video.py", "Chat", namespace)
    registry = SimpleNamespace(mapping={"model_name_mapping":{}})
    module("config", PATH_TO_LLM=cfg.PATH_TO_LLM)
    module("my_affectgpt.common.registry",registry=registry)
    module("my_affectgpt.datasets.datasets.base_dataset",BaseDataset=dataset_cls)
    module("my_affectgpt.tasks.base_task",BaseTask=task_cls)
    conv = module("my_affectgpt.conversation.conversation_video",Chat=chat_cls)
    module("my_affectgpt.models.tokenizer",load_tokenizer_from_LLM=lambda _:tokenizer)
    load_calls = []
    def load_native(*args, **kwargs):
        load_calls.append(kwargs)
        return tiny_native(processor)
    monkeypatch.setattr(bridge.Qwen2_5OmniThinkerForConditionalGeneration,"from_pretrained",load_native)
    bridge.install(os.environ["OMNI_PROCESSOR_SNAPSHOT"])
    assert task_cls.train_step is original_step
    assert dataset_cls.get_qa_pairs is original_qa
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.__dict__.update(reference_dataset(tokenizer, video).__dict__)
    # Exercise installed methods, not the direct-test media/prompt stubs.
    del dataset.read_frame_face_audio_text
    del dataset.get_prompt_for_multimodal
    dataset._get_video_path = lambda sample: video
    media = dataset.read_frame_face_audio_text(video,"unused.npy","unused.wav",None)
    assert media["raw_frame"] == video and media["raw_audio"] is None and media["raw_face"] is None
    prompt = dataset.get_prompt_for_multimodal("multiface_audio_face_text","I am fine.","What is the emotion?")
    assert prompt == PROMPT
    dataset.annotation[0]["description"] = "happy " * 1100
    item = dataset[0]
    assert len(item["text_input"]) > 1024
    full_batch = dataset.collater([item])["omni"]
    assert tokenizer.decode(full_batch["labels"][0][full_batch["labels"][0] != -100]) == "happy " * 1100 + "###" + tokenizer.eos_token
    model = registry.mapping["model_name_mapping"]["affectgpt"].from_config({})
    assert isinstance(model,AffectGPTThinker)
    assert load_calls[0]["torch_dtype"] == torch.bfloat16
    batch = dataset.collater([sample(tokenizer,video)])
    assert torch.isfinite(task_cls().train_step(model,batch))
    config = SimpleNamespace(llama_model="Qwen25",num_video_query_token=1,
                             num_audio_query_token=1,num_multi_query_token=1,num_image_query_token=1)
    chat = conv.Chat(model,config,device="cpu")
    for method in ("postprocess_audio", "postprocess_face", "postprocess_image", "postprocess_multi"):
        assert getattr(conv.Chat, method) is getattr(chat_cls, method)
    assert chat.postprocess_multi(None, None) == (None, None)
    img_list = {kind:getattr(chat,"postprocess_"+kind)(media)[1] for kind in ("audio","face","frame","image")}
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        answer=tokenizer("happy###",add_special_tokens=False,return_tensors="pt")["input_ids"]
        return torch.cat((kwargs["input_ids"],answer),dim=1)
    monkeypatch.setattr(model,"generate",generate)
    assert chat.answer_sample(prompt,img_list,max_new_tokens=1200) == "happy"
    assert calls[0]["use_audio_in_video"] is True and calls[0]["max_new_tokens"] == 1200
    assert calls[0]["min_length"] == 1 and "min_new_tokens" not in calls[0]
    assert "pixel_values_videos" in calls[0] and "pixel_values" not in calls[0]
    # Total inference budget includes every prompt token plus the original 1200.
    monkeypatch.setattr(bridge, "encode_prompt", lambda *args: {"input_ids":torch.zeros((1,2896),dtype=torch.long)})
    assert chat.answer_sample(prompt,img_list,max_new_tokens=1200) == "happy"
    monkeypatch.setattr(bridge, "encode_prompt", lambda *args: {"input_ids":torch.zeros((1,2897),dtype=torch.long)})
    with pytest.raises(SequenceLimitError, match="capacity 4096"):
        chat.answer_sample(prompt,img_list,max_new_tokens=1200)
    assert len(calls) == 2  # overflow never reaches generation
