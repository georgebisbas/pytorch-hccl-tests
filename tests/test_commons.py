import types

import pytest
import torch

from pytorch_hccl_tests.commons import (
    dist_init,
    get_device,
    get_dtype,
    get_nbytes_from_dtype,
    get_npu_runtime_details,
    is_integral,
)


def test_get_device():
    rank = 0
    backend = "cpu"
    device = get_device(backend, rank)
    assert device == torch.device("cpu")


def test_get_dtype_int8():
    dtype = get_dtype("int8")
    assert dtype == torch.int8


def test_get_dtype_float():
    dtype = get_dtype("float")
    assert dtype == torch.float


def test_get_dtype_float32():
    dtype = get_dtype("float32")
    assert dtype == torch.float


def test_get_dtype_double():
    dtype = get_dtype("double")
    assert dtype == torch.float64


def test_get_dtype_float64():
    dtype = get_dtype("float64")
    assert dtype == torch.float64


def test_is_integral_int():
    assert is_integral("int")


def test_is_integral_long():
    assert is_integral("long")


def test_is_integral_float():
    assert not is_integral("float")


def test_is_integral_double():
    assert not is_integral("double")


def test_get_nbytes_from_dtype_float16():
    assert get_nbytes_from_dtype("float16") == 2


def test_get_nbytes_from_dtype_float32():
    assert get_nbytes_from_dtype("float") == 4


def test_get_nbytes_from_dtype_int():
    assert get_nbytes_from_dtype("int") == 4


def test_get_nbytes_from_dtype_long():
    assert get_nbytes_from_dtype("long") == 8


def test_get_npu_runtime_details_reports_import_error():
    def importer():
        raise ImportError("missing torch_npu")

    details = get_npu_runtime_details(importer=importer, hccl_checker=lambda: True)

    assert details["torch_npu_version"] is None
    assert details["hccl_available"] is False
    assert "missing torch_npu" in details["import_error"]


def test_get_npu_runtime_details_reports_hccl_status():
    torch_npu = types.SimpleNamespace(__version__="2.9.0")

    details = get_npu_runtime_details(
        importer=lambda: torch_npu,
        hccl_checker=lambda: True,
    )

    assert details["torch_npu_version"] == "2.9.0"
    assert details["hccl_available"] is True
    assert details["import_error"] is None


def test_dist_init_npu_requires_hccl(monkeypatch):
    monkeypatch.setattr(
        "pytorch_hccl_tests.commons.get_npu_runtime_details",
        lambda: {
            "python_version": "3.10.0",
            "torch_version": torch.__version__,
            "ascend_home_path": "unset",
            "ascend_opp_path": "unset",
            "torch_npu_version": "2.9.0",
            "hccl_available": False,
            "import_error": None,
            "hccl_error": None,
        },
    )

    with pytest.raises(RuntimeError, match="HCCL is unavailable"):
        dist_init("npu", 0)
