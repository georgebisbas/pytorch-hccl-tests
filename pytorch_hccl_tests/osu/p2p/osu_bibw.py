import logging

import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    elaspsed_time_ms,
    get_device,
    get_device_event,
    get_nbytes_from_dtype,
    safe_rand,
    sync_device,
    wait_all,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def bibw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Bi-Directional Bandwidth", args)
    Utils.check_numprocs(world_size, rank, limit=2)

    if rank == 0:
        logger.info("# OMB-Py MPI %s Test" % (options.benchmark))
        logger.info("# %-8s%18s" % ("Size (B)", "Bandwidth (GB/s)"))

    rows = []

    window_size = 64
    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        window_sizes = list(range(window_size))

        # Separate tensor buffer per window slot to avoid concurrent HCCL ops
        # targeting the same memory (matches the unidirectional osu_bw.py pattern).
        s_msg = [safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)]
        r_msg = [safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)]

        send_requests = [None] * window_size
        recv_requests = [None] * window_size

        # Tags are swapped between ranks (canonical OSU C: osu_bibw.c).
        # Both ranks use the same wait ordering (sends before recvs).
        # See https://mvapich.cse.ohio-state.edu/benchmarks/
        if rank == 0:
            partner = 1
            recv_tag = 10
            send_tag = 100
        else:
            partner = 0
            recv_tag = 100
            send_tag = 10

        dist.barrier()
        for i in range(options.iterations + options.skip):
            if i == options.skip:
                start_event = get_device_event(backend)
            for j in window_sizes:
                recv_requests[j] = dist.irecv(r_msg[j], partner, pg, recv_tag)
            for j in window_sizes:
                send_requests[j] = dist.isend(s_msg[j], partner, pg, send_tag)
            wait_all(send_requests)
            wait_all(recv_requests)
        end_event = get_device_event(backend)
        sync_device(backend)

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)

            # Canonical OSU bandwidth formula: aggregate bandwidth across window.
            total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
            t_sec = total_time_ms / 1000.0
            bw_gbps = (size_in_bytes * options.iterations * window_size) / (1e9 * t_sec)

            logger.info("%-10d%18.2f" % (size_in_bytes, bw_gbps))
            new_row = {
                "size_in_bytes": int(size_in_bytes),
                "bw_gbps": bw_gbps,
            }
            rows.append(new_row)

    if rank == 0:
        pd.DataFrame(rows).to_csv(
            f"osu_bibw_gbps-{device.type}-{dtype}-{world_size}.csv", index=False
        )
