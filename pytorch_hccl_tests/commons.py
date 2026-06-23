import importlib
import logging
import os
import platform
import sys
from time import perf_counter_ns as now
from typing import Any, Callable, List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


_TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "float16": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def _load_torch_npu() -> Any:
    return importlib.import_module("torch_npu")


def _is_hccl_available() -> bool:
    checker = getattr(torch.distributed, "is_hccl_available", None)
    return bool(checker()) if callable(checker) else False


def get_npu_runtime_details(
    importer: Callable[[], Any] | None = None,
    hccl_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    details = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ascend_home_path": os.getenv("ASCEND_HOME_PATH") or "unset",
        "ascend_opp_path": os.getenv("ASCEND_OPP_PATH") or "unset",
        "torch_npu_version": None,
        "hccl_available": False,
        "import_error": None,
        "hccl_error": None,
    }
    importer = importer or _load_torch_npu
    hccl_checker = hccl_checker or _is_hccl_available

    try:
        torch_npu = importer()
    except Exception as exc:
        details["import_error"] = str(exc)
        return details

    details["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")

    try:
        details["hccl_available"] = bool(hccl_checker())
    except Exception as exc:
        details["hccl_error"] = str(exc)

    return details


def get_dtype(dtype: str) -> torch.dtype:
    dtype = _TORCH_DTYPES.get(dtype, "float16")
    return dtype


def get_nbytes_from_dtype(dtype: str) -> int:
    if dtype in {"uint8", "int8"}:
        return 1
    elif dtype in {"int16", "short", "float16", "bfloat16"}:
        return 2
    elif dtype in {"int", "float", "float32"}:
        return 4
    elif dtype in {"double", "float64", "long", "int64"}:
        return 8
    else:
        raise NotImplementedError(f"dtype '{dtype}' is not supported")


def is_integral(dtype: str) -> bool:
    tensor = torch.zeros(1, dtype=get_dtype(dtype))
    return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)


def safe_rand(size: int, dtype: str) -> torch.Tensor:
    if is_integral(dtype):
        return torch.randint(low=0, high=100, size=(size,), dtype=get_dtype(dtype))
    return torch.rand(size, dtype=get_dtype(dtype))


def get_device(backend: str, local_rank: int):
    "Returns device"
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        n_devices = torch.cuda.device_count()
        if local_rank >= n_devices:
            raise RuntimeError(
                f"Local rank *{local_rank}* greater than number of CUDA devices {n_devices}"
            )
        return torch.device(f"cuda:{local_rank}")
    elif backend == "hccl":
        return torch.device(f"npu:{local_rank}")
    else:
        return torch.device("cpu")


def get_device_event(backend: str):
    "Returns device Event"
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    elif backend == "hccl":
        event = torch.npu.Event(enable_timing=True)
        event.record()
        return event
    else:
        return now()


def sync_device(backend: str):
    "Synchronize device"
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        torch.cuda.synchronize()
    elif backend == "hccl":
        torch.npu.synchronize()


def elaspsed_time_ms(backend: str, start, end):
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        # See https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
        return start.elapsed_time(end)
    elif backend == "hccl":
        return start.elapsed_time(end)
    else:
        return (end - start) / 1000


def dist_init(device: str, local_rank: int):
    logger.info(f"Init distributed env device: {device} / local_rank {local_rank}")
    backend = None
    if device == "cpu":
        backend = "gloo"

    elif device == "npu":
        details = get_npu_runtime_details()
        if details["import_error"] is not None:
            raise ImportError(
                "NPU benchmark requested but torch-npu could not be imported. "
                f"Python={details['python_version']}, torch={details['torch_version']}, "
                f"ASCEND_HOME_PATH={details['ascend_home_path']}. "
                "Install a torch-npu build that matches the local CANN runtime. "
                "The Makefile install-npu-* targets are the supported setup path for this fork."
            )
        if not details["hccl_available"]:
            extra = ""
            if details["hccl_error"]:
                extra = f" HCCL probe error: {details['hccl_error']}."
            raise RuntimeError(
                "torch-npu imported successfully but HCCL is unavailable. "
                f"torch_npu={details['torch_npu_version']}, "
                f"ASCEND_HOME_PATH={details['ascend_home_path']}, "
                f"ASCEND_OPP_PATH={details['ascend_opp_path']}.{extra} "
                "Ensure the torch-npu build matches the local CANN installation."
            )
        torch.npu.set_device(local_rank)
        backend = "hccl"
    elif device == "cuda":
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        raise ValueError("unknown device")

    dist.init_process_group(backend=backend)
    return backend


def wait_all(async_reqs: List[Any]) -> None:
    """Wait for all request handles
    Parameters
    ----------
    async_reqs : List[Any]
        List of torch.distributed communication handles
    """
    n = sum(int(req is not None) for req in async_reqs)
    in_flight_msgs = n
    while in_flight_msgs > 0:
        for req in async_reqs:
            if req is not None:
                req.wait()
                in_flight_msgs = in_flight_msgs - 1


def print_root(vec_size: int, latency: float, bw: float):
    rank = dist.get_rank()
    if rank == 0:
        print(f"(Rank {rank}) {vec_size * 4}   {latency:.3f}  {bw:.6f}")


def setup_loggers(filename: str) -> List[Any]:
    file_handler = logging.FileHandler(filename=f"{filename}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    return [file_handler, stdout_handler]


def log_env_info(device, backend):
    world_size = dist.get_world_size()
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"PyTorch MPI enabled?: {torch.distributed.is_mpi_available()}")
    logger.info(f"PyTorch CUDA enabled?: {torch.cuda.is_available()}")
    logger.info(f"PyTorch NCCL enabled?: {torch.distributed.is_nccl_available()}")
    logger.info(f"PyTorch Gloo enabled?: {torch.distributed.is_gloo_available()}")
    details = get_npu_runtime_details()
    logger.info(f"ASCEND_HOME_PATH: {details['ascend_home_path']}")
    logger.info(f"ASCEND_OPP_PATH: {details['ascend_opp_path']}")
    if details["import_error"] is None:
        logger.info(f"PyTorch HCCL enabled?: {details['hccl_available']}")
        logger.info(
            f"PyTorch Ascend Adapter (NPU) version: {details['torch_npu_version']}"
        )
        if details["hccl_error"]:
            logger.warning(f"PyTorch HCCL probe error: {details['hccl_error']}")
    else:
        logger.warning("*" * 80)
        logger.warning("* PyTorch Ascend (NPU) is NOT installed.")
        logger.warning(
            "* Install a torch-npu build that matches the local CANN runtime. *"
        )
        logger.warning(f"* Import error: {details['import_error']} *")
        logger.warning("*" * 80)

    logger.info(f"Using device *{device}* with *{backend}* backend")
    logger.info(f"World size: {world_size}")
