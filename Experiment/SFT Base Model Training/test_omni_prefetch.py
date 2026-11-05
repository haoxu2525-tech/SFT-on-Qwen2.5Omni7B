"""CUDA transport equivalence and lifecycle checks, independent of live training."""
import ast
import copy
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import random
import threading
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from omni_prefetch import OrderedCudaPrefetchLoader

SOURCE = Path(__file__).resolve().parents[2] / "Official Repository Code/AffectGPT/AffectGPT"
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA transport required")


def reference_prefetch():
    """Execute the actual pinned class and transfer functions, without heavy imports."""
    namespace = dict(torch=torch)
    for file, names in [
        ("my_affectgpt/datasets/data_utils.py", {"apply_to_sample", "move_to_cuda"}),
        ("my_affectgpt/datasets/datasets/dataloader_utils.py", {"PrefetchLoader", "record_cuda_stream"}),
    ]:
        tree = ast.parse((SOURCE / file).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / file), "exec"), namespace)
    return namespace["PrefetchLoader"]


class RandomSamples(Dataset):
    def __init__(self, count=24, fail=None, delay=0):
        self.count, self.fail, self.delay = count, fail, delay

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.delay:
            time.sleep(self.delay)
        if index == self.fail:
            raise ValueError("intentional sample error")
        # Exercise all RNGs seeded by the unchanged DataLoader workers.
        return {"x": torch.rand(32), "target": torch.tensor(random.randrange(4)),
                "index": torch.tensor(index), "np": torch.tensor(np.random.rand()),
                "enabled": True}


class PreparedSamples(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return 4
    def __getitem__(self, index):
        return self.rows[index % len(self.rows)]


def loader(dataset, shuffle=True):
    # Match production worker count and default multiprocessing context.
    return DataLoader(dataset, batch_size=3, num_workers=4, pin_memory=True,
                      shuffle=shuffle, drop_last=True)


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1)


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    return value


def equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            equal(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@cuda_required
def test_order_worker_rng_dropout_gradients_and_optimizer():
    def run(factory):
        seed()
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Dropout(.05),
                                    torch.nn.Linear(32, 4)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.05)
        batches, losses, grads = [], [], []
        raw = loader(RandomSamples())
        wrapped = factory(raw)
        for epoch in range(2):
            for batch in wrapped:
                batches.append(snapshot(batch))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(model(batch["x"]), batch["target"])
                loss.backward()
                grads.append([snapshot(p.grad) for p in model.parameters()])
                optimizer.step()
                optimizer.zero_grad()
                losses.append(snapshot(loss))
        return (batches, losses, grads, snapshot(model.state_dict()),
                snapshot(optimizer.state_dict()), torch.get_rng_state(), torch.cuda.get_rng_state())

    baseline = run(reference_prefetch())
    candidate = run(OrderedCudaPrefetchLoader)
    equal(baseline, candidate)
    assert not any(t.name == "omni-cuda-prefetch" for t in threading.enumerate())


@cuda_required
def test_error_propagation_and_early_close():
    seed()
    iterator = iter(OrderedCudaPrefetchLoader(loader(RandomSamples(fail=3), shuffle=False)))
    # Upstream detects the failing following batch before yielding the current.
    with pytest.raises(ValueError, match="intentional sample error"):
        next(iterator)
    iterator.close()
    gc.collect()
    iterator = iter(OrderedCudaPrefetchLoader(loader(RandomSamples(90), shuffle=False)))
    next(iterator)
    iterator.close()
    gc.collect()
    assert not any(t.name == "omni-cuda-prefetch" for t in threading.enumerate())


@cuda_required
def test_real_native_batch_and_lora_update():
    # Real published video/audio preprocessing, tiny native Thinker, and the
    # unchanged BaseTask.train_step. This is NOT full 7B training validation.
    from test_omni_components import (tiny_native, load_processor, write_av, sample,
                                     collate_native, original_class)
    from omni_components import AffectGPTThinker, make_lora_model, MEDIA_MARKERS
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        processor = load_processor(os.environ["OMNI_PROCESSOR_SNAPSHOT"])
        tokenizer = copy.deepcopy(processor.tokenizer)
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_tokens(list(MEDIA_MARKERS), special_tokens=True)
        video = write_av(Path(directory) / "sample.mov")
        rows = [collate_native(processor, tokenizer, [sample(tokenizer, video, answer=a)], 4096)
                for a in ("happy###", "The speaker is happy.###")]
        task = original_class("my_affectgpt/tasks/base_task.py", "BaseTask",
                              dict(torch=torch, main_process=lambda f: f))()

        def run(factory):
            seed()
            model = AffectGPTThinker(make_lora_model(tiny_native(processor).to(torch.bfloat16))).cuda().train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=1e-5, weight_decay=.05)
            raw = DataLoader(PreparedSamples(rows), batch_size=None, num_workers=4, pin_memory=True)
            losses, grads, batches = [], [], []
            for batch in factory(raw):
                batches.append(snapshot(batch))
                loss = task.train_step(model, batch)
                loss.backward()
                grads.append({n: snapshot(p.grad) for n, p in model.named_parameters() if p.requires_grad})
                optimizer.step()
                optimizer.zero_grad()
                losses.append(snapshot(loss))
            return (batches, losses, grads,
                    {n: snapshot(p) for n, p in model.named_parameters() if p.requires_grad},
                    snapshot(optimizer.state_dict()), torch.cuda.get_rng_state())

        equal(run(reference_prefetch()), run(OrderedCudaPrefetchLoader))


@cuda_required
def test_controlled_wait_benchmark(tmp_path):
    """Scheduling microbenchmark only; shared-GPU timing is not 7B throughput."""
    if os.environ.get("OMNI_PREFETCH_BENCHMARK") != "1":
        pytest.skip("Explicitly opt into the scheduling benchmark")
    measurements = {"reference": [], "async": []}
    factories = {"reference": reference_prefetch(), "async": OrderedCudaPrefetchLoader}
    # CPU delay represents original synchronous metric work; same in both arms.
    for repeat in range(3):
        for name in (["reference", "async"] if repeat % 2 == 0 else ["async", "reference"]):
            seed()
            wrapped = factories[name](loader(RandomSamples(60, delay=.08), shuffle=False))
            start = time.perf_counter()
            indexes = []
            for batch in wrapped:
                indexes.extend(batch["index"].tolist())
                time.sleep(.04)
            torch.cuda.synchronize()
            measurements[name].append(time.perf_counter() - start)
            assert indexes == list(range(60))
    print("PREFETCH_SCHEDULING_BENCHMARK=" + json.dumps(measurements))
    (tmp_path / "prefetch-benchmark.json").write_text(json.dumps(measurements, indent=2))


def test_background_lookahead_and_original_failure_boundary(monkeypatch):
    """CPU scheduling test only; fake events cannot verify CUDA correctness."""
    import omni_prefetch as adapter
    fetching_next, allow_next = threading.Event(), threading.Event()

    class Stream:
        def synchronize(self):
            pass
        def wait_event(self, event):
            assert event.recorded

    class Event:
        def record(self, stream):
            self.recorded = True

    class ControlledLoader(DataLoader):
        def __iter__(self):
            yield {"index": 0}
            yield {"index": 1}
            fetching_next.set()
            assert allow_next.wait(5), "test did not release the delayed batch"
            yield {"index": 2}
            raise ValueError("ordered source failure")

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr(adapter, "_to_cuda", lambda batch, device: batch)
    wrapped = OrderedCudaPrefetchLoader(ControlledLoader(RandomSamples(), num_workers=1, pin_memory=True))
    iterator = iter(wrapped)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            first = pool.submit(next, iterator).result(timeout=3)
            assert first == {"index": 0}
            assert fetching_next.wait(1)
        finally:
            allow_next.set()
        assert next(iterator) == {"index": 1}
        with pytest.raises(ValueError, match="ordered source failure"):
            next(iterator)
    assert not any(t.name == "omni-cuda-prefetch" for t in threading.enumerate())


def test_install_keeps_runner_methods_unchanged(monkeypatch):
    import sys
    from types import ModuleType
    from omni_prefetch import install_async_prefetch
    original = object()
    runner = ModuleType("my_affectgpt.runners.runner_base")
    runner.PrefetchLoader = original
    runner.RunnerBase = type("RunnerBase", (), {"train": lambda self: None, "create_loaders": lambda self: None})
    before = dict(runner.RunnerBase.__dict__)
    for name in ("my_affectgpt", "my_affectgpt.runners"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    sys.modules["my_affectgpt"].runners = sys.modules["my_affectgpt.runners"]
    sys.modules["my_affectgpt.runners"].runner_base = runner
    install_async_prefetch()
    assert runner.PrefetchLoader is OrderedCudaPrefetchLoader
    assert dict(runner.RunnerBase.__dict__) == before


def test_producer_setup_failure_reaches_consumer(monkeypatch):
    import queue
    from omni_prefetch import _produce
    def fail(device):
        raise RuntimeError("CUDA context setup failed")
    monkeypatch.setattr(torch.cuda, "device", fail)
    ready = queue.Queue()
    _produce(iter([]), ready, threading.Semaphore(1), threading.Event(), "cuda:0", None)
    kind, error, event = ready.get_nowait()
    assert kind == "error" and "CUDA context setup failed" in str(error)
