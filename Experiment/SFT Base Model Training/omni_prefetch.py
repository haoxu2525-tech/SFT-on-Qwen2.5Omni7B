"""Optional FIFO transport adapter; official runner/task/source stay unchanged.

Only the existing multiprocessing DataLoader is consumed here. Dataset work,
sampler order, worker seeds, collation and model computation remain upstream.
"""
import queue
import threading

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def _to_cuda(value, device):
    if torch.is_tensor(value):
        if value.device.type != "cpu" or not value.is_pinned():
            raise ValueError("Async prefetch requires pinned CPU tensors")
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_cuda(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cuda(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cuda(v, device) for v in value)
    return value


def _record(value, stream):
    if torch.is_tensor(value):
        value.record_stream(stream)
    elif isinstance(value, dict):
        for child in value.values():
            _record(child, stream)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _record(child, stream)


def _produce_batches(iterator, ready, slot, stop, device, stream):
    # Reserved slots cover fetching AND transfer. Never cross a DataLoader
    # epoch or reorder completed batches.
    with torch.cuda.device(device), torch.cuda.stream(stream):
        while not stop.is_set():
            if not slot.acquire(timeout=0.1):
                continue
            if stop.is_set():
                break
            try:
                host = next(iterator)
            except StopIteration:
                ready.put(("end", None, None))
                break
            except BaseException as error:
                ready.put(("error", error, None))
                break
            try:
                batch = _to_cuda(host, device)
                event = torch.cuda.Event()
                event.record(stream)
                ready.put(("batch", batch, event))
                # Pinned allocator tracks asynchronous copy lifetime. Nothing
                # mutates the host tensors after submission.
                del host, batch, event
            except BaseException as error:
                ready.put(("error", error, None))
                break


def _produce(iterator, ready, slot, stop, device, stream):
    try:
        _produce_batches(iterator, ready, slot, stop, device, stream)
    except BaseException as error:
        # CUDA context/stream setup failures must reach the consumer too.
        ready.put(("error", error, None))


class OrderedCudaPrefetchLoader:
    """Overlap ordered lookahead with compute, preserving upstream error order."""

    def __init__(self, loader):
        if not isinstance(loader, DataLoader) or loader.num_workers < 1:
            raise ValueError("Async prefetch requires the original multiprocessing DataLoader")
        if not loader.pin_memory or loader.persistent_workers:
            raise ValueError("Expected pin_memory=True and original non-persistent workers")
        if type(loader.sampler) not in {RandomSampler, SequentialSampler, DistributedSampler}:
            raise ValueError("Custom sampler RNG behavior has not been verified")
        if loader.batch_sampler is not None and type(loader.batch_sampler) is not BatchSampler:
            raise ValueError("Custom batch sampler RNG behavior has not been verified")
        self.loader = loader
        self.device = torch.device("cuda", torch.cuda.current_device())

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        # Create the iterator in the training thread, just as upstream does.
        # Worker seeds and RandomSampler's private generator are initialized
        # here by the original DataLoader's initial prefetch, before our thread.
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        ready = queue.Queue()
        slot = threading.Semaphore(2)
        stop = threading.Event()
        producer = threading.Thread(
            target=_produce, args=(iterator, ready, slot, stop, self.device, stream),
            name="omni-cuda-prefetch", daemon=True,
        )
        producer.start()
        try:
            item = ready.get()
            while item[0] != "end":
                kind, value, event = item
                if kind == "error":
                    raise value
                # Upstream preloads the following batch BEFORE yielding this
                # one. Retain that failure boundary: a bad following batch must
                # not allow an extra optimizer step on the current batch.
                following = ready.get()
                if following[0] == "error":
                    raise following[1]
                current = torch.cuda.current_stream(self.device)
                # Wait for THIS batch only, not a subsequently submitted copy.
                current.wait_event(event)
                _record(value, current)
                slot.release()
                yield value
                item = following
        finally:
            stop.set()
            slot.release()
            # A current CPU fetch finishes through the unchanged DataLoader.
            # Join before destroying its iterator or starting another epoch.
            producer.join()
            stream.synchronize()
            while not ready.empty():
                ready.get_nowait()


def install_async_prefetch():
    """Replace only the loader factory binding; no runner/task method changes."""
    import my_affectgpt.runners.runner_base as runner

    runner.PrefetchLoader = OrderedCudaPrefetchLoader
