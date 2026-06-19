"""
Copyright (C) 2002-2022 the Network-Based Computing Laboratory
(NBCL), The Ohio State University.

Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)

For detailed copyright and licensing information, please refer to the
copyright file COPYRIGHT in the top level OMB directory.
"""


class Options:
    def __init__(self, benchmark_name, args):
        self.args = args
        self.iterations = 10000
        self.iterations_large = 100
        self.skip = 1000
        self.skip_large = 10
        self.min_message_size = 1 << 10
        self.max_message_size = 1 << 29
        self.large_message_size = 8192
        self.buffer = None
        self.benchmark = benchmark_name
        self.update_options()

    def update_options(self):
        bench = self.args.benchmark.lower()
        pt2pt = {"latency", "bw", "bibw", "multi_lat", "bandwidth", "multi-latency"}
        bw_benchs = {"bw", "bibw", "bandwidth"}
        coll_reduce = {"reduce", "allreduce", "reduce_scatter"}

        if self.args.buffer:
            self.buffer = self.args.buffer
        if self.args.iterations:
            self.iterations = self.args.iterations
            self.iterations_large = int(self.args.iterations / 100) + 1
        elif bench in bw_benchs:
            self.iterations = 100
            self.iterations_large = 30
        if self.args.skip:
            self.skip = self.args.skip
            self.skip_large = int(self.args.skip / 100) + 1
        elif bench in bw_benchs:
            self.skip = 10
            self.skip_large = 3
        if self.args.max:
            self.max_message_size = self.args.max
        elif bench in bw_benchs:
            # bw/bibw keep 64 in-flight buffers per rank; cap message size to
            # avoid NPU OOM or driver crashes at the largest sizes (~256 MiB/msg).
            self.max_message_size = 1 << 26
        elif bench in pt2pt:
            self.max_message_size = 1 << 27
        if self.args.min:
            self.min_message_size = self.args.min
        elif bench in {"latency", "multi_lat", "multi-latency"}:
            self.min_message_size = 0
        elif bench in coll_reduce:
            self.min_message_size = 4
