import math
import random
import threading
from collections import defaultdict   # Correct the defaultdict import:
import queue
from typing import Callable, List, Optional, Tuple, Any, Iterator, Dict
from pidgeon_core.payload import PidgeonTapePayload, TapeFrame, TapeHeaderEntry, TapeHeaderFrame
from pidgeon_core.color import normalize_weights, ColorMap, make_palette
from pidgeon_core.mapping_job import MappingJob
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
from collections import deque
import ctypes
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexSubImage2D,
    glBegin, glEnd, glTexCoord2f, glVertex2f, glClear, glClearColor, glViewport,
    glLoadIdentity, glOrtho, glMatrixMode, glLoadIdentity, glEnable,
    GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_QUADS, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_PROJECTION,
    GL_MODELVIEW, GL_NEAREST
)
from pidgeon_crusher_helper import PigeonCrushKernelHelper  
import logging  # keep for other logging uses
from pidgeon_core.logging import get_logger, TRACE
logger = get_logger("pidgeoncrusher")

def normalize_weights(weights: List[float], total: int) -> List[int]:
    raw = [w * total for w in weights]
    floored = [int(x) for x in raw]
    residuals = [x - int(x) for x in raw]
    shortfall = total - sum(floored)
    for i in sorted(range(len(residuals)), key=lambda i: -residuals[i])[:shortfall]:
        floored[i] += 1
    return floored

class PidgeonTapePayload:
    def __init__(self, value: Any, origin_bin: int, tape: Optional[List[float]] = None, bin_path: Optional[List[int]] = None):
        self.value = value
        self.origin_bin = origin_bin
        self.tape = tape or []
        self.bin_path = bin_path or []

    def add_stage(self, bin_idx: int, hue: float, tape_mode: str = 'append'):
        self.bin_path.append(bin_idx)
        if tape_mode == 'append':
            self.tape.append(hue)
            logger.trace(f"TAPE APPEND: Appended hue {hue:.3f}. New tape: {self.tape}")
        elif tape_mode == 'replace':
            original_hue = self.tape[-1] if self.tape else None
            if self.tape:
                self.tape[-1] = hue
            else:
                self.tape.append(hue)
            logger.trace(f"TAPE REPLACE: Replaced hue {original_hue} with {hue:.3f}. New tape: {self.tape}")

    def tape_color(self, depth: int = 0, mode: str = 'first', complement: bool = False):
        if not self.tape:
            return 0.0
        
        idx = 0
        if mode == 'first':
            idx = 0
        elif mode == 'last':
            idx = -1
        else: # indexed mode
            idx = depth

        # Bounds check for safety
        if not (-len(self.tape) <= idx < len(self.tape)):
            # Fallback to last element if index is out of bounds
            idx = -1

        hue = self.tape[idx]
        if complement:
            hue = (hue + 0.5) % 1.0
        return hue

    def __repr__(self):
        return f"<PidgeonTapePayload val={self.value} tape={self.tape} bins={self.bin_path}>"




class TapeFrame(ctypes.Structure):
    _fields_ = [
        ('hue', ctypes.c_float),        # hue in [0, 1)
        ('painter', ctypes.c_uint32),   # id of the system/branch
    ]

    def __repr__(self):
        return f"TapeFrame(hue={self.hue}, painter={self.painter})"


MAX_PAINT_UNITS = 64  # or any limit you want

class TapeHeaderEntry(ctypes.Structure):
    _fields_ = [
        ('painter_id', ctypes.c_uint32),
        ('desc_offset', ctypes.c_uint32),  # offset into an external char buffer for descriptions
    ]

class TapeHeaderFrame(ctypes.Structure):
    _fields_ = [
        ('count', ctypes.c_uint32),
        ('entries', TapeHeaderEntry * MAX_PAINT_UNITS),
    ]


def append_hue_to_tape_in_place(item, hue: float):
    """Given (idx, PidgeonTapePayload), add hue to tape in place."""
    idx, payload = item
    payload.tape.append(hue)
    return (idx, payload)

def append_hue_to_tape(item, hue: float):
    idx, payload = item
    new_tape = payload.tape + [hue]
    logger.trace(f"TAPE CLONE/APPEND: Appended hue {hue:.3f} to tape from {payload.tape}. New tape: {new_tape}")
    # Deep copy bin_path to avoid accidental mutation
    new_payload = PidgeonTapePayload(payload.value, payload.origin_bin, new_tape, payload.bin_path[:])
    return (idx, new_payload)


def append_bin_to_job(item: 'MappingJob', insertion_idx: int, insertion_location: int, shift_hues= True):
    """
    Given a MappingJob item, append the insertion_idx to the input or output bins.
    If insertion_location == 0, append to input_bins; otherwise, append to output_bins.
    """
    if insertion_location == 0:
        item.input_bins.push(insertion_idx)
        if shift_hues:
            item.in_hues = PidgeonCrusher.get_hues(len(item.input_bins), item.exclusion)
    elif insertion_location == 1:
        item.output_bins.push(insertion_idx)
        if shift_hues:
            item.out_hues = PidgeonCrusher.get_hues(len(item.output_bins), item.exclusion)
    
class PidgeonCrusher:
    @staticmethod
    def get_hues(count: int, exclusion: float = 0.125, margin=0.07, gamma=0.8, mode="rainbow") -> List[float]:
        try:
            if exclusion is not None and margin == 0.07:
                margin = exclusion
            if count <= 0:
                raise ValueError("Count must be a positive integer")
            logger.trace(f"Generating {count} hues with mode='{mode}', margin={margin:.3f}, gamma={gamma:.2f}")
            cmap = ColorMap(count, margin=0.07, gamma=0.8, mode="rainbow")
            hues = cmap.hues()
            return hues
        except ValueError as e:
            logger.error(f"Invalid parameters for ColorMap: {e}")
                
            ε = exclusion
            if count == 1:
                return [0.5]
            return [ε + (1 - 2 * ε) * i / (count - 1) for i in range(count)]

    @staticmethod
    def dummy_payloads(
        bins: int, 
        count: int, 
        value: Any = None, 
        exclusion: float = 0.125, 
        t_fn: Optional[Callable[[], float]] = None
    ) -> Iterator[Tuple[int, PidgeonTapePayload]]:
        logger.trace(f"Generating {count} dummy payloads for {bins} bins.")
        hues = PidgeonCrusher.get_hues(bins, exclusion)
        per_bin = count // bins
        idx = 0
        while idx < bins * per_bin:
            bin_idx = idx % bins
            initial_tape = [hues[bin_idx]]
            logger.trace(f"TAPE INIT: Dummy payload for bin {bin_idx} created with tape: {initial_tape}")
            yield (bin_idx, PidgeonTapePayload(value, origin_bin=bin_idx, tape=initial_tape, bin_path=[bin_idx]))
            idx += 1
            if t_fn:
                import time
                time.sleep(t_fn())
        for bin_idx in range(count % bins):
            initial_tape = [hues[bin_idx]]
            logger.trace(f"TAPE INIT: Dummy payload for bin {bin_idx} created with tape: {initial_tape}")
            yield (bin_idx, PidgeonTapePayload(value, origin_bin=bin_idx, tape=[hues[bin_idx]], bin_path=[bin_idx]))
            if t_fn:
                import time
                time.sleep(t_fn())
import logging


import numpy as np

from typing import Any, Optional

import queue
import threading
import time
from collections import deque, defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple






def plot_bins(bins: List[List[PidgeonTapePayload]], title: str, tape_depth: int = 0, tape_mode: str = 'first', complement: bool = False, figsize=(16, 2)):
    avg_hues = []
    for b in bins:
        hues = [p.tape_color(depth=tape_depth, mode=tape_mode, complement=complement) for p in b]
        if hues:
            avg = sum(hues) / len(hues)
        else:
            avg = None
        avg_hues.append(avg)
    rgb = [
        colorsys.hsv_to_rgb(h, 1, 1) if h is not None else (0.0, 0.0, 0.0)
        for h in avg_hues
    ]
    fig, ax = plt.subplots(figsize=figsize)
    for i, color in enumerate(rgb):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(rgb))
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title)
    plt.show()


from collections import deque, defaultdict
import numpy as np
from pidgeon_crushing import MappingJob  # adjust import to your module structure


import threading
import numpy as np
import colorsys
import pygame
import time
from math import ceil, sqrt


### --- DEMO CONFIGURATION ---

# List of (input_bins, output_bins, color_map_mode) per job/row
JOB_TEMPLATES = [
    (64, 8,   "rainbow", PigeonCrushKernelHelper.classic_kernel, {}),
    (32, 16,  "hsv_linear", PigeonCrushKernelHelper.float_ray_kernel, {'grid_height': 1.5}),
    (128, 32, "rainbow", PigeonCrushKernelHelper.diffusion_kernel, {}),
    (24, 24,  "rainbow", PigeonCrushKernelHelper.classic_kernel, {}),
    (12, 48,  "hsv_linear", PigeonCrushKernelHelper.float_ray_kernel, {'grid_height': 0.5}),
    (100, 10, "rainbow", PigeonCrushKernelHelper.diffusion_kernel, {}),
]

TILE_SIZE = 16  # pixels per tile (adjust for window size/performance)
FPS = 60
STEPS_PER_FRAME = 400  # How many mapping steps each job runs between redraws

# --- END CONFIG ---

def make_palette(output_bins, mode="rainbow"):
    cmap = ColorMap(output_bins, mode=mode)
    return cmap.hues()
# pidgeon_demo.py
import pygame
import numpy as np
import colorsys
from OpenGL.GL import *
import time
from typing import List, Tuple, Dict, Callable
from pidgeon_crushing import MappingJob, JobManager, PidgeonCrusher, PidgeonTapePayload
from pidgeon_crusher_helper import PigeonCrushKernelHelper
import pygame
import numpy as np
import colorsys
from OpenGL.GL import *
from typing import List, Tuple, Dict, Callable
from pidgeon_crushing import MappingJob, JobManager, PidgeonCrusher, PidgeonTapePayload
from pidgeon_crusher_helper import PigeonCrushKernelHelper

# ring_lab.py
import torch
import torch.nn as nn
from collections import deque
from typing import List, Callable, Optional, Dict, Any, Sequence
from pidgeon_crushing import (
    MappingJob, PidgeonTapePayload, PidgeonCrusher, JobJoiner
)




# visualizer.py
import pygame
import numpy as np
import colorsys
from OpenGL.GL import *
from typing import List, Tuple, Dict
from collections import deque, defaultdict
from pidgeon_crushing import MappingJob, PidgeonTapePayload, PidgeonCrusher, ColorMap, normalize_weights, JobManager

if __name__ == "__main__":
    logger.setLevel(TRACE)
    lab = PidgeonRingLab(
        bit_depth=16,
        kernel=PigeonCrushKernelHelper.classic_kernel,
        visualizer_factory=None,
    )
    vis = PidgeonVisualizer(
        lab=lab,
        win_width=960,
        win_height=540,
        max_fps=144,
        feedback_delay_frames=0,
        input_hue_tape_depth=0,
        bit_depth=16,
        row_height_gamma=2.0,
    )
    vis.run()


