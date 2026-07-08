"""Persistent multi-GPU self-play workers, one process per device.

Each worker owns a SelfPlayPool driver and a NetEvaluator pinned to
its GPU; the trainer broadcasts fresh weights every iteration and
gathers samples. Communication is over multiprocessing queues with
CPU state dicts, so no distributed backend is needed for a single
node.
"""

import numpy as np
import torch
import torch.multiprocessing as mp

from train.batchplay import play_games


def _worker_loop(device_id: int, net_kwargs: dict, in_queue, out_queue):
    import signal

    from train.net import PolicyValueNet
    from train.selfplay import NetEvaluator

    # Graceful-stop signals (e.g. scancel --signal=TERM --full hits
    # every process in the job) are the trainer's business; a worker
    # dying mid-request leaves the trainer waiting forever.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Inference runs on the GPU; a fat default OMP pool per worker
    # just thrashes the job's CPU allocation (observed: 8 H200s at 2%).
    torch.set_num_threads(2)
    # Workers beyond the GPU count stack onto devices round-robin: the
    # C++ search is one thread per worker, and on big boards a single
    # worker cannot feed a fast GPU by itself.
    torch.cuda.set_device(device_id % torch.cuda.device_count())
    net = PolicyValueNet(**net_kwargs)
    device = torch.device(
        f"cuda:{device_id % torch.cuda.device_count()}")
    evaluator = NetEvaluator(net, device, compile=True)
    while True:
        message = in_queue.get()
        if message is None:
            return
        state_dict, play_kwargs, seed = message
        net.load_state_dict(state_dict)
        net.eval()
        try:
            result = play_games(evaluator.evaluate_planes,
                                rng=np.random.default_rng(seed),
                                **play_kwargs)
            out_queue.put(("ok", result))
        except Exception as error:  # surface, don't hang the trainer
            out_queue.put(("error", repr(error)))


class SelfPlayWorkers:
    def __init__(self, n_workers: int, net_kwargs: dict):
        self.n_workers = n_workers
        context = mp.get_context("spawn")
        self.in_queues = [context.Queue() for _ in range(n_workers)]
        self.out_queue = context.Queue()
        self.processes = [
            context.Process(target=_worker_loop,
                            args=(i, net_kwargs, self.in_queues[i],
                                  self.out_queue),
                            daemon=True)
            for i in range(n_workers)
        ]
        for process in self.processes:
            process.start()

    def play(self, net, n_games: int, play_kwargs: dict,
             rng: np.random.Generator):
        """Split n_games across the workers; returns merged
        (game_samples, margins, stats)."""
        state_dict = {key: value.cpu()
                      for key, value in net.state_dict().items()}
        share = [n_games // self.n_workers] * self.n_workers
        for i in range(n_games % self.n_workers):
            share[i] += 1
        active = 0
        for i, games in enumerate(share):
            if games == 0:
                continue
            kwargs = dict(play_kwargs, n_games=games)
            if i != 0:  # only one worker writes the spectate file
                kwargs["spectate_path"] = None
            self.in_queues[i].put(
                (state_dict, kwargs, int(rng.integers(0, 2**63 - 1))))
            active += 1

        all_samples = []
        margins = []
        stats: dict = {}
        for _ in range(active):
            status, payload = self.out_queue.get()
            if status == "error":
                raise RuntimeError(f"self-play worker failed: {payload}")
            samples, worker_margins, worker_stats = payload
            all_samples.extend(samples)
            margins.extend(worker_margins)
            for key, value in worker_stats.items():
                stats[key] = stats.get(key, 0) + value
        return all_samples, margins, stats

    def close(self) -> None:
        for queue in self.in_queues:
            queue.put(None)
        for process in self.processes:
            process.join(timeout=30)
