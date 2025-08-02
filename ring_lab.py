import queue
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mapping_job import MappingJob, JobManager
from payload import PigeonTapePayload


class RingNode(nn.Module):
    def __init__(self, param_dim: int = 1):
        super().__init__()
        self.learned_params = nn.Parameter(torch.randn(param_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.stats = None

    def effective_weights(self):
        if self.stats is None:
            return self.learned_params
        return self.alpha * self.stats + self.beta * self.learned_params


class PigeonRingLab:
    def __init__(
        self,
        bit_depth: int = 8,
        param_dim: int = 1,
        feedback_delay_frames: int = 0,
        nn_mode: str = "meta",
        visualizer_factory: Optional[Callable] = None,
        kernel: Optional[Callable] = None,
    ):
        self.bit_depth = bit_depth
        self.param_dim = param_dim
        self.feedback_delay_frames = feedback_delay_frames
        self.nn_mode = nn_mode
        self.visualizer_factory = visualizer_factory
        self.kernel = kernel
        self.ring_jobs: List[MappingJob] = self._build_ring_jobs()
        self.num_layers = len(self.ring_jobs)
        self.meta_graph = (
            self._build_meta_graph() if nn_mode in ("meta", "both") else None
        )
        self.geometric_graph = (
            self._build_geometric_graph() if nn_mode in ("geometric", "both") else None
        )
        self.results: Dict[int, Tuple[Dict[int, List[PigeonTapePayload]], any]] = {}
        self.job_manager = JobManager(self.ring_jobs, self.results)
        self.job_manager.daemon = True

    def _build_ring_bins(self) -> List[int]:
        up = [2 ** i for i in range(0, self.bit_depth + 1)]
        down = [2 ** i for i in range(self.bit_depth - 1, -1, -1)]
        return up + down

    def _build_ring_jobs(self) -> List[MappingJob]:
        binsizes = self._build_ring_bins()
        jobs: List[MappingJob] = []
        for idx, in_bins in enumerate(binsizes):
            out_bins = binsizes[(idx + 1) % len(binsizes)]
            job = MappingJob(
                source=queue.Queue(),
                input_bins=in_bins,
                output_bins=out_bins,
                samples_per_in=[1] * in_bins,
                asap=True,
                exclusion=0.125,
                layer_id=idx,
                tape_mode="append",
                pigeon_kernel=self.kernel,
                grid_info={},
            )
            jobs.append(job)
        for i in range(jobs[0].input_bins):
            jobs[0].source.put((i, PigeonTapePayload(value=i, origin_bin=i)))
        return jobs

    def _build_meta_graph(self):
        return nn.ModuleList(
            [RingNode(self.param_dim) for job in self.ring_jobs for _ in range(job.output_bins)]
        )

    def _build_geometric_graph(self):
        return nn.ModuleList([RingNode(self.param_dim) for _ in range(len(self.ring_jobs))])

    def nn_inject(self, idx: int, tensor: torch.Tensor):
        if self.meta_graph:
            self.meta_graph[idx].stats = tensor.detach().clone()

    def nn_collect(self, idx: int) -> torch.Tensor:
        if self.meta_graph:
            return self.meta_graph[idx].effective_weights().detach().clone()
        raise ValueError("Meta graph not initialized")

    def opengl_visualize(self):
        if self.visualizer_factory:
            vis = self.visualizer_factory(self)
            vis.run()

    def ring_on_demand(self, bit_depth: int = None, param_dim: int = None):
        return PigeonRingLab(
            bit_depth=bit_depth or self.bit_depth,
            param_dim=param_dim or self.param_dim,
            feedback_delay_frames=self.feedback_delay_frames,
            nn_mode=self.nn_mode,
            visualizer_factory=self.visualizer_factory,
            kernel=self.kernel,
        )

