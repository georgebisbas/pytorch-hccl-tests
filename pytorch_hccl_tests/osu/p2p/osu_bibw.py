import logging

import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv

from pytorch_hccl_tests.commons import (
    BW_RESULT_COLUMNS,
    elaspsed_time_ms,
    get_device,
    get_device_event,
    get_dtype,
    get_nbytes_from_dtype,
    is_integral,
    log_timed_result,
    sync_device,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)

# Fixed tags for each direction (must match on sender and receiver).
TAG_0_TO_1_BASE = 100
TAG_1_TO_0_BASE = 200
PATTERN_SEED = 12345


def _pattern_tensor(size: int, dtype: str, sender_rank: int, slot: int, device):
    torch_dtype = get_dtype(dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PATTERN_SEED + sender_rank * 1000 + slot)
    if is_integral(dtype):
        tensor = torch.randint(
            low=0, high=100, size=(size,), dtype=torch_dtype, generator=generator
        )
    else:
        tensor = torch.rand(size, dtype=torch_dtype, generator=generator)
    return tensor.to(device)


def _verify_received(recv, dtype: str, sender_rank: int, slot: int, device):
    expected = _pattern_tensor(recv.numel(), dtype, sender_rank, slot, device)
    if not torch.equal(recv, expected):
        raise RuntimeError(
            f"bibw data mismatch: expected payload from rank {sender_rank} "
            f"slot {slot}, got max diff "
            f"{(recv - expected).abs().max().item() if recv.is_floating_point() else 'n/a'}"
        )


def _verify_bibw_buffers(
    rank, window_size, s_msgs, r_msgs, dtype, device, peer, pg, backend, sequential
):
    """Run one transfer and verify payloads in both directions."""
    _run_bibw_iteration(
        window_size, s_msgs, r_msgs, peer, pg, rank, sequential=sequential
    )
    sync_device(backend)
    if rank == 0:
        for j in range(window_size):
            _verify_received(r_msgs[j], dtype, 1, j, device)
    else:
        for j in range(window_size):
            _verify_received(r_msgs[j], dtype, 0, j, device)
    dist.barrier()


def _run_bibw_iteration(
    window_size, s_msgs, r_msgs, peer, pg, rank, sequential=False
):
    if sequential:
        for j in range(window_size):
            tag = TAG_0_TO_1_BASE + j
            if rank == 0:
                dist.send(s_msgs[j], peer, pg, tag)
            else:
                dist.recv(r_msgs[j], peer, pg, tag)
        for j in range(window_size):
            tag = TAG_1_TO_0_BASE + j
            if rank == 1:
                dist.send(s_msgs[j], peer, pg, tag)
            else:
                dist.recv(r_msgs[j], peer, pg, tag)
    else:
        ops = []
        for j in range(window_size):
            recv_tag = TAG_1_TO_0_BASE + j if rank == 0 else TAG_0_TO_1_BASE + j
            send_tag = TAG_0_TO_1_BASE + j if rank == 0 else TAG_1_TO_0_BASE + j
            ops.append(P2POp(dist.irecv, r_msgs[j], peer, tag=recv_tag))
            ops.append(P2POp(dist.isend, s_msgs[j], peer, tag=send_tag))
        for req in batch_isend_irecv(ops):
            req.wait()


def bibw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, args.local_rank)
    pg = None
    sequential = args.sequential
    verify = not args.no_verify
    window_size = args.window

    options = Options("Bi-Directional Bandwidth", args)
    Utils.check_numprocs(world_size, rank, limit=2)
    Utils.print_header(options.benchmark, rank)
    if rank == 0:
        mode = "sequential" if sequential else "duplex"
        logger.info("bibw mode=%s window=%d verify=%s", mode, window_size, verify)

    rows = []
    peer = 1 if rank == 0 else 0

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        s_msgs = [
            _pattern_tensor(size, dtype, rank, j, device)
            for j in range(window_size)
        ]
        r_msgs = [
            torch.empty(size, dtype=get_dtype(dtype), device=device)
            for _ in range(window_size)
        ]

        dist.barrier()
        if verify:
            _verify_bibw_buffers(
                rank,
                window_size,
                s_msgs,
                r_msgs,
                dtype,
                device,
                peer,
                pg,
                backend,
                sequential,
            )

        for i in range(options.iterations + options.skip):
            if i == options.skip:
                start_event = get_device_event(backend)
            _run_bibw_iteration(
                window_size,
                s_msgs,
                r_msgs,
                peer,
                pg,
                rank,
                sequential=sequential,
            )
        end_event = get_device_event(backend)
        sync_device(backend)

        if verify:
            _verify_bibw_buffers(
                rank,
                window_size,
                s_msgs,
                r_msgs,
                dtype,
                device,
                peer,
                pg,
                backend,
                sequential,
            )
            if rank == 0:
                logger.info(
                    "Data check passed for size %d bytes",
                    size * get_nbytes_from_dtype(dtype),
                )

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            total_iterations = options.iterations * window_size
            total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
            avg_latency_ms = total_time_ms / total_iterations
            new_row = log_timed_result(
                logger, size_in_bytes, avg_latency_ms, 2 * size_in_bytes
            )
            rows.append(new_row)

    if rank == 0:
        pd.DataFrame(rows, columns=BW_RESULT_COLUMNS).to_csv(
            f"osu_bibw-{device.type}-{dtype}-{world_size}.csv", index=False
        )
