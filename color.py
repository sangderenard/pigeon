from typing import List, Callable, Optional
import numpy as np
import colorsys

def normalize_weights(weights: List[float], total: int) -> List[int]:
    raw = [w * total for w in weights]
    floored = [int(x) for x in raw]
    residuals = [x - int(x) for x in raw]
    shortfall = total - sum(floored)
    for i in sorted(range(len(residuals)), key=lambda i: -residuals[i])[:shortfall]:
        floored[i] += 1
    return floored


class ColorMap:
    def __init__(
        self, 
        n: int, 
        margin: float = 0.125, 
        gamma: float = 1.0, 
        mode: str = "hsv_linear", 
        custom_fn: Optional[Callable[[int, int], float]] = None
    ):
        self.n = n
        self.margin = margin
        self.gamma = gamma
        self.mode = mode
        self.custom_fn = custom_fn  # Takes (i, n) and returns a hue

    @staticmethod
    def interpolate_hues(source_hues: List[float], target_width: int) -> np.ndarray:
        """
        Downsamples a large list of hues to a smaller width using true linear
        interpolation for a smooth, anti-aliased result.
        """
        source_width = len(source_hues)
        if source_width == 0:
            return np.zeros((1, target_width, 4), dtype=np.uint8)

        # 1. Define the x-coordinates for our source and target data.
        # We map the source bins to a range of [0, 1] for interpolation.
        source_x = np.linspace(0, 1, source_width)
        target_x = np.linspace(0, 1, target_width)

        # Handle cases where source might contain None
        valid_hues = [h for h in source_hues if h is not None]
        if not valid_hues:
            # If no valid data, return a black texture
            return np.zeros((1, target_width, 4), dtype=np.uint8)
        
        # Use the first valid hue for all None values to ensure np.interp works
        fallback_hue = valid_hues[0]
        source_y = np.array([h if h is not None else fallback_hue for h in source_hues])

        # 2. Perform the linear interpolation. NumPy does all the heavy lifting.
        interpolated_hues = np.interp(target_x, source_x, source_y)

        # 3. Convert the resulting interpolated hues to an RGBA texture.
        output_rgba = np.empty((1, target_width, 4), dtype=np.uint8)
        for i, hue in enumerate(interpolated_hues):
            r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
            output_rgba[0, i, :] = (int(r * 255), int(g * 255), int(b * 255), 255)
            
        return output_rgba

    def bin_to_hue(self, i: int) -> float:
        """Map bin index i (0 to n-1) to hue in [0,1) according to color map spec."""
        if self.custom_fn:
            hue = self.custom_fn(i, self.n)
        elif self.mode == "hsv_linear":
            hue = self.margin + (1 - 2*self.margin) * (i / (self.n-1 if self.n > 1 else 1))
        elif self.mode == "hsv_reverse":
            hue = self.margin + (1 - 2*self.margin) * ((self.n-1-i) / (self.n-1 if self.n > 1 else 1))
        elif self.mode == "rainbow":
            hue = (0.7 * i / max(1, self.n-1)) ** self.gamma + self.margin
        else:
            hue = i / self.n
        # Apply gamma curve to the normalized position, not hue directly
        norm = (i / (self.n-1)) if self.n > 1 else 0.5
        hue = self.margin + (1 - 2*self.margin) * (norm ** self.gamma)
        return hue % 1.0

    def hues(self) -> List[float]:
        return [self.bin_to_hue(i) for i in range(self.n)]

    def hue_to_bin(self, hue: float) -> int:
        """Given a hue in [0,1), infer the bin index."""
        # Invert the mapping formula (works for gamma != 1)
        margin = self.margin
        norm = (hue - margin) / max(1e-10, (1 - 2*margin))
        if norm < 0: norm = 0
        if norm > 1: norm = 1
        i_float = (norm ** (1/self.gamma)) * (self.n-1) if self.n > 1 else 0
        return int(round(i_float))
    @staticmethod
    def fallback_hue(idx: int, kind: str = "output") -> float:
        # Return 0.0 for black (HSV), or pick any custom hue logic you want
        # If you want to visually debug, try 0.0 for red, or None for black (see below)
        return None

def make_palette(output_bins: int, mode: str = "rainbow") -> List[float]:
    cmap = ColorMap(output_bins, mode=mode)
    return cmap.hues()
