"""Shared fixtures. GPU tests skip (with a reason) when CUDA or CuPy is absent,
so the CPU-only CI job runs the allocator and CLI tests and reports the rest
as skipped rather than passed."""

from __future__ import annotations

import pytest
import torch


def _gpu_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False, "CuPy not installed"
    return True, ""


GPU_OK, GPU_REASON = _gpu_available()


def pytest_collection_modifyitems(config, items):
    if GPU_OK:
        return
    skip = pytest.mark.skip(reason=f"GPU test: {GPU_REASON}")
    for item in items:
        if "gpu" in item.keywords or "model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def gpt2():
    """GPT-2 small in fp16 on the GPU, plus its tokenizer."""
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()
    return model, tok
