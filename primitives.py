
class PigeonCrushedBase:
    """
    Mixin/base for crushed primitives.
    Provides:
      - by_output_index(idx): raw access
      - get_mapped_output(idx): dry-run, just mapping, no storage access
      - get_full_mapping(): full mapping as dict or list
    """
    @classmethod
    def _harvest_base_methods(cls, base):
        # Get the unbound base methods for get/set/del
        cls._backing_get = base.__getitem__
        cls._backing_set = base.__setitem__
        cls._backing_del = base.__delitem__
        if hasattr(base, '__contains__'):
            cls._backing_contains = base.__contains__
        if hasattr(base, 'get'):
            cls._backing_get_key = base.get
        if hasattr(base, 'keys'):
            cls._backing_keys = base.keys
        if hasattr(base, 'values'):
            cls._backing_values = base.values
        if hasattr(base, 'items'):
            cls._backing_items = base.items
        if hasattr(base, 'remove'):
            cls._backing_remove = base.remove

    def by_output_index(self, idx):
        """Should be overridden to access raw backing (list/dict/set)."""
        raise NotImplementedError

    def get_mapped_output(self, in_idx, n_samples=1):
        """
        Returns a list of output indices that this input index would be mapped to
        using the current pigeon mapping and cache, for n_samples.
        Does NOT mutate or access actual backing storage.
        """
        results = []
        for _ in range(n_samples):
            # Pull mapping from the IndexCache, but don't touch storage
            in_idx, out_idx, tape, payload = self.cache.get(in_idx)
            results.append(out_idx)
            # Optionally, you could return (out_idx, tape, payload) for full provenance
        return results

    def get_full_mapping(self, n_samples=1):
        """
        Returns a mapping for all input bins (domain: 0..input_bins-1), with n_samples each.
        Result: dict of {in_idx: [out_idx, ...]}
        """
        mapping = {}
        for in_idx in range(self.input_bins):
            mapping[in_idx] = self.get_mapped_output(in_idx, n_samples)
        return mapping




class IndexCache:
    """
    Probabilistic, repeat-permitting batch index mapping cache.
    For each input index, keeps a queue of (input_idx, output_idx, hue_tape, payload).
    """
    def __init__(self, job, fill_batch=256):
        self.job = job
        self.fill_batch = fill_batch
        self.caches = defaultdict(deque)  # input_idx -> queue of results

    def clear(self):
        self.caches.clear()

    def bulk_fill(self, in_indices: List[int]):
        """
        Fills the cache by running a batch mapping job. It calculates the required
        samples per input bin and uses the job's blocking 'process' method.
        """
        if not in_indices:
            return

        # 1. Count how many samples are requested for each input bin
        counts = defaultdict(int)
        for i in in_indices:
            counts[i] += 1
        
        required_samples = [counts[i] for i in range(self.job.input_bins)]

        # 2. Temporarily set the job's sample configuration for this run
        original_samples = self.job.samples_per_in
        self.job.samples_per_in = required_samples

        # 3. Create the source queue with proper payloads
        in_queue = queue.Queue()
        for i in in_indices:
            in_queue.put((i, PigeonTapePayload(value=i, origin_bin=i)))
        # 4. Call the new blocking process method
        bins, df = self.job.process(in_queue)

        # 5. Restore the job's original configuration
        self.job.samples_per_in = original_samples

        # 6. Populate the cache with the results
        # The mapping dataframe is the most reliable source of truth
        for _, row in df.iterrows():
            in_idx = row['src_bin']
            out_idx = row['dst_bin']
            # Recreate a payload object for the cache from the record
            payload = PigeonTapePayload(
                value=row['src_val'],
                origin_bin=in_idx,
                tape=row['tape'],
                bin_path=row['bin_path']
            )
            self.caches[in_idx].append((in_idx, out_idx, row['tape'], payload))

    def get(self, in_idx):
        """Return (and remove) the next cached mapping for this input index, filling as needed."""
        while not self.caches[in_idx]:
            self.bulk_fill([in_idx]*self.fill_batch)
        return self.caches[in_idx].popleft()

    def __contains__(self, in_idx):
        return bool(self.caches[in_idx])

class CrushList(PigeonCrushedBase, list):
    """
    List with all index translation (get/set) performed by a batch-filling, probabilistic IndexCache.
    """
    def __init__(self, *args, input_bins=None, pigeon_kernel=None, grid_info=None, **kwargs):
        PigeonCrushedBase._harvest_base_methods(list)
        super().__init__(*args, **kwargs)
        self.input_bins = input_bins if input_bins is not None else len(self)
        self.pigeon_kernel = pigeon_kernel
        self.grid_info = grid_info
        self._make_job_and_cache()
        
    def _make_job_and_cache(self):
        self._job = MappingJob(
            source=queue.Queue(),  # changed from None/iterator to queue.Queue
            input_bins=self.input_bins,
            output_bins=len(self),
            samples_per_in=[1] * self.input_bins,
            layer_id=0,
            tape_mode='append',
            pigeon_kernel=self.pigeon_kernel,
            grid_info=self.grid_info
        )
        self.cache = IndexCache(self._job, fill_batch=max(32, len(self)*8))

    def __getitem__(self, idx):
        # All access goes through the cache: input index mapped to output index (may repeat probabilistically)
        in_idx, out_idx, tape, payload = self.cache.get(idx)
        return super().__getitem__(out_idx)

    def __setitem__(self, idx, value):
        in_idx, out_idx, tape, payload = self.cache.get(idx)
        super().__setitem__(out_idx, value)

    def append(self, value):
        super().append(value)
        self._make_job_and_cache()  # reset mapping/cache on structure change

    def insert(self, idx, value):
        super().insert(idx, value)
        self._make_job_and_cache()

    def __delitem__(self, idx):
        in_idx, out_idx, tape, payload = self.cache.get(idx)
        super().__delitem__(out_idx)
        self._make_job_and_cache()

    def pop(self, idx=-1):
        if idx == -1:
            # Pop last output bin
            val = super().pop()
        else:
            in_idx, out_idx, tape, payload = self.cache.get(idx)
            val = super().pop(out_idx)
        self._make_job_and_cache()
        return val

    def set_input_bins(self, input_bins):
        self.input_bins = input_bins
        self._make_job_and_cache()

    def __repr__(self):
        return f"CrushList({list(self)})"
from collections import deque, defaultdict
import numpy as np
from pigeon_crushing import MappingJob  # Use your real import

class CrushDict(PigeonCrushedBase, dict):
    """
    Dictionary with integer-indexed (sorted key) rebinning via IndexCache.
    All get/set/del by integer index go through the probabilistic mapping cache.
    """
    def __init__(self, *args, input_bins=None, pigeon_kernel=None, grid_info=None, **kwargs):
        PigeonCrushedBase._harvest_base_methods(dict)
        super().__init__(*args, **kwargs)
        self.input_bins = input_bins if input_bins is not None else len(self)
        self.pigeon_kernel = pigeon_kernel
        self.grid_info = grid_info
        self._refresh_keys()
        self._make_job_and_cache()

    def _refresh_keys(self):
        # Keep a sorted list of keys for index mapping
        self._ordered_keys = sorted(self.keys())

    def _make_job_and_cache(self):
        self._refresh_keys()
        self._job = MappingJob(
            source=queue.Queue(),  # changed from None/iterator to queue.Queue
            input_bins=self.input_bins,
            output_bins=len(self._ordered_keys),
            samples_per_in=[1] * self.input_bins,
            layer_id=0,
            tape_mode='append',
            pigeon_kernel=self.pigeon_kernel,
            grid_info=self.grid_info
        )
        self.cache = IndexCache(self._job, fill_batch=max(32, len(self)*8))

    # --- dict-style access ---
    def __getitem__(self, key):
        # If key is int, rebin; otherwise, behave as dict
        if isinstance(key, int):
            in_idx, out_idx, tape, payload = self.cache.get(key)
            k = self._ordered_keys[out_idx]
            return super().__getitem__(k)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            in_idx, out_idx, tape, payload = self.cache.get(key)
            k = self._ordered_keys[out_idx]
            super().__setitem__(k, value)
        else:
            super().__setitem__(key, value)
        self._make_job_and_cache()

    def __delitem__(self, key):
        if isinstance(key, int):
            in_idx, out_idx, tape, payload = self.cache.get(key)
            k = self._ordered_keys[out_idx]
            super().__delitem__(k)
        else:
            super().__delitem__(key)
        self._make_job_and_cache()

    def set_input_bins(self, input_bins):
        self.input_bins = input_bins
        self._make_job_and_cache()

    # All normal dict methods work as expected (by key)
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._make_job_and_cache()

    def pop(self, key=None):
        if key is None:
            k = self._ordered_keys[-1]
            val = super().pop(k)
        elif isinstance(key, int):
            in_idx, out_idx, tape, payload = self.cache.get(key)
            k = self._ordered_keys[out_idx]
            val = super().pop(k)
        else:
            val = super().pop(key)
        self._make_job_and_cache()
        return val

    def clear(self):
        super().clear()
        self._make_job_and_cache()

    def __repr__(self):
        return f"CrushDict({dict(self)})"
class CrushSet(PigeonCrushedBase, set):
    """
    Set with integer-indexed (sorted element) rebinning via IndexCache.
    All get/set/del by integer index go through the probabilistic mapping cache.
    """
    def __init__(self, *args, input_bins=None, pigeon_kernel=None, grid_info=None):
        PigeonCrushedBase._harvest_base_methods(set)
        super().__init__(*args)
        self.input_bins = input_bins if input_bins is not None else len(self)
        self.pigeon_kernel = pigeon_kernel
        self.grid_info = grid_info
        self._refresh_elements()
        self._make_job_and_cache()

    def _refresh_elements(self):
        self._ordered_elements = sorted(self)

    def _make_job_and_cache(self):
        self._refresh_elements()
        self._job = MappingJob(
            source=queue.Queue(),  # changed from None/iterator to queue.Queue
            input_bins=self.input_bins,
            output_bins=len(self._ordered_elements),
            samples_per_in=[1] * self.input_bins,
            layer_id=0,
            tape_mode='append',
            pigeon_kernel=self.pigeon_kernel,
            grid_info=self.grid_info
        )
        self.cache = IndexCache(self._job, fill_batch=max(32, len(self)*8))

    def __getitem__(self, idx):
        # Always integer index rebinned to sorted element via cache
        in_idx, out_idx, tape, payload = self.cache.get(idx)
        return self._ordered_elements[out_idx]

    def __delitem__(self, idx):
        in_idx, out_idx, tape, payload = self.cache.get(idx)
        elem = self._ordered_elements[out_idx]
        super().__delitem__(elem)
        self._make_job_and_cache()

    def add(self, elem):
        super().add(elem)
        self._make_job_and_cache()

    def remove(self, elem):
        super().remove(elem)
        self._make_job_and_cache()

    def discard(self, elem):
        super().discard(elem)
        self._make_job_and_cache()

    def pop(self):
        elem = self._ordered_elements[-1]
        super().__remove__(elem)
        self._make_job_and_cache()
        return elem

    def clear(self):
        super().clear()
        self._make_job_and_cache()

    def set_input_bins(self, input_bins):
        self.input_bins = input_bins
        self._make_job_and_cache()

    def __repr__(self):
        return f"CrushSet({set(self)})"
