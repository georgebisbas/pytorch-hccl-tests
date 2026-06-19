import logging

import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    BW_RESULT_COLUMNS,
    elaspsed_time_ms,
    get_device,
    get_device_event,
    get_nbytes_from_dtype,
    log_timed_result,
    safe_rand,
    sync_device,
    wait_all,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def bw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Bandwidth", args)
    Utils.check_numprocs(world_size, rank, limit=2)
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=BW_RESULT_COLUMNS)

    window_size = 64
    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        window_sizes = range(window_size)
        requests = [None] * window_size

        dist.barrier()
        if rank == 0:
            s_msg = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)
            ]
            r_msg = safe_rand(4, dtype=dtype).to(device)
            for i in range(options.iterations + options.skip):
                if i == options.skip:
                    start_event = get_device_event(backend)
                for j in window_sizes:
                    requests[j] = dist.isend(s_msg[j], 1, pg, 100)
                wait_all(requests)
                dist.recv(r_msg, 1, pg, 101)
            end_event = get_device_event(backend)
            sync_device(backend)
        elif rank == 1:
            s_msg = safe_rand(4, dtype=dtype).to(device)
            r_msg = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)
            ]
            for i in range(options.iterations + options.skip):
                for j in window_sizes:
                    requests[j] = dist.irecv(r_msg[j], 0, pg, 100)
                wait_all(requests)
                dist.send(s_msg, 0, pg, 101)

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            total_iterations = options.iterations * window_size
            total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
            avg_latency_ms = total_time_ms / total_iterations
            new_row = log_timed_result(
                logger, size_in_bytes, avg_latency_ms, size_in_bytes
            )
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if rank == 0:
        df.to_csv(f"osu_bandwidth-{device.type}-{dtype}-{world_size}.csv", index=False)
