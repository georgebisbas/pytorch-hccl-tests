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

# Tags for the two directions: rank 0 -> 1 uses SEND_TAG, rank 1 -> 0 uses RECV_TAG.
SEND_TAG = 100
RECV_TAG = 101


def bibw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, args.local_rank)
    pg = None

    options = Options("Bi-Directional Bandwidth", args)
    Utils.check_numprocs(world_size, rank, limit=2)
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=BW_RESULT_COLUMNS)

    peer = 1 if rank == 0 else 0
    send_tag = SEND_TAG if rank == 0 else RECV_TAG
    recv_tag = RECV_TAG if rank == 0 else SEND_TAG

    window_size = 64
    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        s_msgs = [safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)]
        r_msgs = [safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)]
        send_requests = [None] * window_size
        recv_requests = [None] * window_size

        dist.barrier()
        for i in range(options.iterations + options.skip):
            if i == options.skip:
                start_event = get_device_event(backend)
            for j in range(window_size):
                recv_requests[j] = dist.irecv(r_msgs[j], peer, pg, recv_tag)
                send_requests[j] = dist.isend(s_msgs[j], peer, pg, send_tag)
            wait_all(recv_requests)
            wait_all(send_requests)
        end_event = get_device_event(backend)
        sync_device(backend)

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            total_iterations = options.iterations * window_size
            total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
            avg_latency_ms = total_time_ms / total_iterations
            new_row = log_timed_result(
                logger, size_in_bytes, avg_latency_ms, 2 * size_in_bytes
            )
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if rank == 0:
        df.to_csv(f"osu_bibw-{device.type}-{dtype}-{world_size}.csv", index=False)
