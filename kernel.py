import math
import random

from scipy import stats

class PigeonCrushKernelHelper:
    """
    Helper for bin-mapping (pigeon crushing) with pluggable kernel logic.

    For each mapping event:
        - Provides a normalized position (float in 0..1) within the input bin
        - Provides a normalized angle (float in 0..1)
        - Calls user-supplied kernel: kernel(input_bin, pos, angle, S, D, rng, grid_info)
            Should return (output_bin, stat_frame)
        - Falls back to LCM integer mapping if no kernel is supplied.

    Example kernels included: classic LCM, float-grid (ray), and a stub for diffusion/physics.
    """

    @staticmethod
    def classic_kernel(input_bin, pos, angle, S, D, rng, grid_info):
        import numpy as np
        """Default integer mapping via LCM, ignores pos/angle."""
        # Always convert to 1D arrays, even if already lists or scalars
        input_bin = np.atleast_1d(input_bin)
        pos = np.atleast_1d(pos)
        angle = np.atleast_1d(angle)

        L = math.lcm(S, D)
        sub_per_src = L // S
        sub_per_dst = L // D
        subbin = input_bin * sub_per_src + (pos * sub_per_src).astype(int)
        output_bin = subbin // sub_per_dst
        # Stat frame for provenance
        output_bin = output_bin.tolist()
        input_bin = input_bin.tolist()
        pos = pos.tolist()
        angle = angle.tolist()
        subbin = subbin.tolist()

        statistics = []
        for i in range(len(output_bin)):
            statistics.append({
                'input_bin': input_bin[i],
                'pos': pos[i],
                'angle': angle[i],
                'S': S,
                'D': D,
            'subbin': subbin[i],
            'output_bin': output_bin[i],
            'method': 'classic_lcm'
        })
        return output_bin, statistics

    @staticmethod
    def float_ray_kernel(input_bin, pos, angle, S, D, rng, grid_info):
        """
        Physics-inspired: treat pos/angle as launching a ray from the input bin.
        - pos: normalized entry point in input bin (0..1)
        - angle: normalized (0..1) mapped to physical angle (-pi/2 to pi/2)
        - grid_info may include 'grid_height'
        """
        grid_height = grid_info.get('grid_height', 1.0)
        # Entry x in [input_bin, input_bin+1)
        entry_x = input_bin + pos
        # Angle in radians: center = 0, -pi/2 = left, +pi/2 = right (customizable)
        theta = (angle - 0.5) * math.pi  # or change range as needed
        # Compute exit x at output plane
        exit_x = entry_x + math.tan(theta) * grid_height
        output_bin = int(D * (exit_x / S))
        output_bin = max(0, min(D-1, output_bin))
        stat = {
            'input_bin': input_bin,
            'pos': pos,
            'angle': angle,
            'theta': theta,
            'entry_x': entry_x,
            'exit_x': exit_x,
            'output_bin': output_bin,
            'grid_height': grid_height,
            'method': 'float_ray'
        }
        return output_bin, stat

    @staticmethod
    def diffusion_kernel(input_bin, pos, angle, S, D, rng, grid_info):
        """
        Placeholder for custom diffusion kernels, e.g., Gaussian, stochastic, etc.
        """
        # Example: use angle as Gaussian noise control, pos as drift
        mean = pos * D
        stddev = (angle + 0.01) * (D / 8)
        sample = rng.gauss(mean, stddev)
        output_bin = int(round(sample))
        output_bin = max(0, min(D-1, output_bin))
        stat = {
            'input_bin': input_bin,
            'pos': pos,
            'angle': angle,
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'output_bin': output_bin,
            'method': 'diffusion'
        }
        return output_bin, stat

    @staticmethod
    def map_bins(
        S, D, N,
        kernel=None,
        rng=None,
        grid_info=None
    ):
        """
        Map N events from S input bins to D output bins using provided kernel.
        Each event is: choose input_bin, pos, angle (all random), call kernel.
        Returns list of (input_bin, pos, angle, output_bin, stat)
        """
        rng = rng or random
        kernel = kernel or PigeonCrushKernelHelper.classic_kernel
        grid_info = grid_info or {}

        results = []
        for _ in range(N):
            input_bin = rng.randrange(S)
            pos = rng.random()  # normalized 0..1
            angle = rng.random()  # normalized 0..1
            output_bin, stat = kernel(input_bin, pos, angle, S, D, rng, grid_info)
            results.append({
                'input_bin': input_bin,
                'pos': pos,
                'angle': angle,
                'output_bin': output_bin,
                'stat': stat
            })
        return results

    @staticmethod
    def batch_bins(
        S, D, input_bins=None, pos=None, angle=None, kernel=None, grid_info=None, rng=None
    ):
        """
        Batch interface for numpy/torch. (supply arrays for input_bins, pos, angle; returns arrays)
        """
        import numpy as np
        kernel = kernel or PigeonCrushKernelHelper.classic_kernel
        rng = rng or random
        grid_info = grid_info or {}

        if input_bins is None or pos is None or angle is None:
            raise ValueError("input_bins, pos, and angle arrays required")
        N = len(input_bins)
        out = []
        for i in range(N):
            ib = input_bins[i]
            p = pos[i]
            a = angle[i]
            ob, stat = kernel(ib, p, a, S, D, rng, grid_info)
            out.append((ob, stat))
        return out

# Example usage:
if __name__ == "__main__":
    # Map 100 events, 8 input bins, 16 output bins, using float-ray kernel
    results = PigeonCrushKernelHelper.map_bins(
        S=8,
        D=16,
        N=100,
        kernel=PigeonCrushKernelHelper.float_ray_kernel,
        grid_info={'grid_height': 2.0}
    )
    for r in results[:5]:
        print(r)
