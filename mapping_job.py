import random
import queue
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Any, Iterator, Dict
import pandas as pd
from pigeon_core.payload import PigeonTapePayload
from pigeon_core.color import ColorMap
from pigeon_core.logging import get_logger, TRACE
from pigeon_crusher_helper import PigeonCrushKernelHelper

logger = get_logger("mapping_job")

class PipelineStageBase:
    def __init__(self):
        self.sources = []
        self.sinks = []
        self.in_buffer = []
        self.out_buffer = []
    def set_source(self, src):
        self.sources.append(src)
    def set_sink(self, sink):
        self.sinks.append(sink)
    def sink(self):
        """Default sink method, can be overridden by subclasses. assumes 
        items in sinks are all queues that can directly be put() into."""
        for s in self.sinks:
            for i in range(len(self.out_buffer)):
                s.put(self.out_buffer[i])  # Pop from buffer and put into each sink
    def source(self):
        """Default source method, can be overridden by subclasses. assumes
        items in sources are all queues that can directly be get() from."""
        for s in self.sources:
            patience = 10  # wait for up to 10 items
            while not s.empty() and patience > 0:
                old_len = len(self.in_buffer)
                self.in_buffer.append(s.get_nowait())
                if len(self.in_buffer) == old_len:
                    patience -= 1



class JobJoiner:
    @staticmethod
    def build_graph(nodes: List[PipelineStageBase], edges: List[Tuple[Tuple[int, int]]]) -> JobGraph:
        graph = JobGraph()
        for i, job in enumerate(nodes):
            graph.add_node(JobGraphNode(job, id(job)))
        for (src, dst) in edges:
            graph.add_edge(src, dst)
        return graph
    @staticmethod
    def flash_edges(graph: JobGraph):
        edges = graph.edges
        for src, dst in edges:
            src_node = graph.nodes[src]
            dst_node = graph.nodes[dst]
            logger.trace(f"Connecting {src_node.id} -> {dst_node.id}")
            src_node.job.set_sink(dst_node.job.source)
            dst_node.job.set_source(src_node.job.sink)


class PipePayload:
    """A standard wrapper for items flowing through the pipeline.
    It can hold either a valid data value or an exception.
    """
    def __init__(self, value: Any = None, error: Optional[Exception] = None, source_id: Optional[str] = None):
        self.value = value
        self.error = error
        self.source_id = source_id

    @property
    def is_ok(self) -> bool:
        """Returns True if this payload contains valid data."""
        return self.error is None

    @property
    def is_error(self) -> bool:
        """Returns True if this payload represents an error."""
        return self.error is not None

    def __repr__(self) -> str:
        if self.is_ok:
            return f"<PipePayload OK value={self.value}>"
        else:
            return f"<PipePayload ERROR source='{self.source_id}' error='{type(self.error).__name__}'>"

class PipelineStageBase:
    def __init__(self):
        self.sources = []
        self.sinks = []
        self.in_buffer = []
        self.out_buffer = []

    def set_source(self, src):
        self.sources.append(src)

    def set_sink(self, sink):
        self.sinks.append(sink)

    def source(self):
        for s in self.sources:
            while not s.empty():
                self.in_buffer.append(s.get_nowait())

    def sink(self):
        for s in self.sinks:
            for item in self.out_buffer:
                s.put(item)
class MappingJob(PipelineStageBase):
    def __init__(
        self,
        source: Optional[queue.Queue],  # changed from Iterator to queue.Queue
        input_bins: int,
        output_bins: int,
        samples_per_in: List[int],
        asap: bool = True,
        exclusion: float = 0.125,
        layer_id: int = 0,
        tape_mode: str = 'append',
        pigeon_kernel: Optional[Callable] = None,
        grid_info: Optional[Dict] = None
    ):
        super().__init__()
        self.asap = asap
        self.source = source  # now expects a queue.Queue
        self.input_bins = input_bins
        self.output_bins = output_bins
        self.samples_per_in = samples_per_in
        self.exclusion = exclusion
        self.layer_id = layer_id
        self.tape_mode = tape_mode
        self.pigeon_kernel = pigeon_kernel or PigeonCrushKernelHelper.classic_kernel
        self.grid_info = grid_info or {}
        self.rng = random.Random()
        self.mapping_df = pd.DataFrame()
        self.buffer = []
        self.done = False
        self.in_hues = PigeonCrusher.get_hues(self.input_bins, self.exclusion)
        self.out_hues = PigeonCrusher.get_hues(self.output_bins, self.exclusion)
        if self.source is not None:
            if isinstance(self.source, queue.Queue):
                items = []
                while True:
                    try:
                        items.append(self.source.get_nowait())
                    except queue.Empty:
                        break
                self._fill_input_buffer(items)
            else:
                self._fill_input_buffer(self.source)

    def get_next(self, blocking: bool = False) -> Tuple[int, PigeonTapePayload]:
        if not blocking:
            if isinstance(self.source, queue.Queue):
                return self.source.get_nowait()
        else:
            while True:
                try:
                    return self.source.get(timeout=0.01)
                except queue.Empty:
                    continue
        raise StopIteration("No more items in source queue")
    def step(self):
        """
        Pulls all items from self.source (assumed to be a queue.Queue),
        processes the batch in one vectorized call (no per-item for-loops),
        and returns (result_bins, df) as the output of this step.
        """
        # === DRAIN QUEUE ENTIRELY ===
        batch = []
        while True:
            try:
                item = self.source.get_nowait()
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            self.done = True
            return defaultdict(list), pd.DataFrame()

        # === VECTORIZE INPUTS ===
        src_bins   = []
        payloads   = []
        positions  = []
        angles     = []

        for idx, payload in batch:
            src_bins.append(idx)
            payloads.append(payload)
            positions.append(self.rng.random())
            angles.append(self.rng.random())

        S = self.input_bins
        D = self.output_bins

        # === BATCH KERNEL CALL ===
        # The kernel must support arrays: (src_bins, positions, angles, S, D, rng, grid_info)
        dst_idxs, stats = self.pigeon_kernel(
            src_bins, positions, angles, S, D, self.rng, self.grid_info
        )

        # === BUILD OUTPUTS, NO LOOPS OVER INDIVIDUAL BINS, ONLY BULK ARRAY OPS ===
        result_bins = defaultdict(list)
        rows = []
        for i in range(len(payloads)):
            dst_idx = dst_idxs[i]
            stat = stats[i]
            payload = payloads[i]
            logger.trace(f"Processing payload {i}: src_bin={src_bins[i]}, dst_idx={dst_idx}, stat={stat}, out_hues len={len(self.out_hues)}")  
            dst_hue = self.out_hues[dst_idx]
            payload.add_stage(dst_idx, dst_hue, tape_mode=self.tape_mode)
            result_bins[dst_idx].append(payload)
            rows.append({
                'layer': self.layer_id,
                'src_bin': src_bins[i],
                'src_val': payload.value,
                'src_hue': payload.tape[0],
                'src_idx': i,
                'subbin': stat.get('subbin', -1),
                'dst_bin': dst_idx,
                'dst_hue': dst_hue,
                'bin_path': payload.bin_path.copy(),
                'tape': payload.tape.copy()
            })

        df = pd.DataFrame(rows)
        self.mapping_df = pd.concat([self.mapping_df, df], ignore_index=True)
        self.done = True
        return result_bins, df

    def _fill_input_buffer(self, items: List[Tuple[int, PigeonTapePayload]]):
        """
        Directly injects a list of items into the job's processing buffer,
        bypassing the source queue.
        """
        self.buffer.extend(items)


    # NEW: Add a simple mapping helper to translate an input index to an output bin index
    def map_input_idx(self, idx: int) -> int:
        # Simple mapping logic: evenly partition input_bins to output_bins.
        # This is a placeholder; advanced implementations may call crush logic.
        return (idx * self.output_bins) // self.input_bins

class TeeStage(PipelineStageBase):
    def __init__(self, n: int):
        super().__init__()
        self.queues = [deque() for _ in range(n)]
    def set_sinks(self, sinks: List[Callable]):
        # each sink sits on its own queue
        self._sinks = sinks

class DiscreteScatterStage(PipelineStageBase):
    def __init__(self, n: int, key_fn: Callable = None, hues: Optional[List[float]] = None):
        super().__init__()
        self.n = n
        self.key_fn = key_fn or (lambda item, i=[0]: (i.append(i.pop(0)+1) or i[0]-1) % n)
        self.hues = hues

    def scatter(self) -> List[Iterator[PipePayload]]:
        queues = [deque() for _ in range(self.n)]
        source_exhausted = threading.Event()

        def _scatter_items():
            try:
                for item in self.source:
                    try:
                        idx = self.key_fn(item)
                        if not 0 <= idx < self.n:
                            raise IndexError(f"key_fn returned out-of-bounds index {idx}.")
                        item_to_queue = item
                        if self.hues:
                            item_to_queue = append_hue_to_tape(item, self.hues[idx])
                        queues[idx].append(PipePayload(value=item_to_queue))
                    except Exception as e:
                        err_payload = PipePayload(value=item, error=e, source_id="scatter:key_fn")
                        for q in queues:
                            q.append(err_payload)
            except Exception as e:
                err_payload = PipePayload(error=e, source_id="scatter:source")
                for q in queues:
                    q.append(err_payload)
            finally:
                source_exhausted.set()

        threading.Thread(target=_scatter_items, daemon=True).start()

        def _gen(my_queue: deque) -> Iterator[PipePayload]:
            while not (source_exhausted.is_set() and not my_queue):
                try:
                    yield my_queue.popleft()
                except IndexError:
                    time.sleep(0.001)

        return [_gen(q) for q in queues]
class HistogramGatherStage(PipelineStageBase):
    def __init__(self, bins: int, sample_counts: List[int]):
        super().__init__()
        self.bins = bins
        self.sample_counts = sample_counts
    def _validate_histogram(self):
        buffers = defaultdict(list)
        for payload in self.source:
            if payload.is_error:
                self.sink(payload)
            else:
                bidx, item = payload.value
                buffers[bidx].append(item)
                if all(len(buffers[i]) >= self.sample_counts[i] for i in range(self.bins)):
                    pack = {i: buffers[i][:self.sample_counts[i]] for i in range(self.bins)}
                    for i in range(self.bins):
                        buffers[i] = buffers[i][self.sample_counts[i]:]
                    self.sink(PipePayload(value=pack))


class JobManager(threading.Thread):
    def __init__(self, jobs: List[MappingJob], results: Dict[int, Tuple[List[List[PigeonTapePayload]], pd.DataFrame]]):
        super().__init__()
        self.jobs = jobs
        self.results = results
        self.asap = jobs.asap if hasattr(jobs, 'asap') else True



    def run(self):
        """
        Continuously process jobs in a loop. This is suitable for a pipeline
        with a feedback loop where jobs will be ready to process data repeatedly.
        """
        while True:
            processed_in_cycle = False
            for i, job in enumerate(self.jobs):
                try:
                    res, df = job.step()
                    self.results[i] = (res, df)
                    processed_in_cycle = True
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Layer {i}: completed cycle with {df.shape[0]} mappings.")
                    if logger.isEnabledFor(TRACE):
                        logger.trace(f"LAYER {i} DF:\n{df}")
                except StopIteration:
                    pass
            # Removed time.sleep to prevent any blocking
            if not processed_in_cycle:
                time.sleep(0.01)