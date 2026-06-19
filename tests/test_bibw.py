import pytest
import torch

from pytorch_hccl_tests.osu.p2p.osu_bibw import (
    _pattern_tensor,
    _verify_received,
)


def test_pattern_tensor_is_deterministic():
    device = torch.device("cpu")
    a = _pattern_tensor(128, "float16", sender_rank=0, slot=3, device=device)
    b = _pattern_tensor(128, "float16", sender_rank=0, slot=3, device=device)
    assert torch.equal(a, b)


def test_pattern_tensor_differs_by_sender_and_slot():
    device = torch.device("cpu")
    a = _pattern_tensor(64, "int", sender_rank=0, slot=1, device=device)
    b = _pattern_tensor(64, "int", sender_rank=1, slot=1, device=device)
    c = _pattern_tensor(64, "int", sender_rank=0, slot=2, device=device)
    assert not torch.equal(a, b)
    assert not torch.equal(a, c)


def test_verify_received_accepts_matching_payload():
    device = torch.device("cpu")
    recv = _pattern_tensor(32, "float32", sender_rank=1, slot=5, device=device)
    _verify_received(recv, "float32", sender_rank=1, slot=5, device=device)


def test_verify_received_rejects_mismatch():
    device = torch.device("cpu")
    recv = _pattern_tensor(32, "float32", sender_rank=1, slot=5, device=device)
    recv[0] = recv[0] + 1
    with pytest.raises(RuntimeError, match="bibw data mismatch"):
        _verify_received(recv, "float32", sender_rank=1, slot=5, device=device)
