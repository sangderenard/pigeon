from kernel import PigeonCrushKernelHelper
from ring_lab import PigeonRingLab
from visualizer import PigeonVisualizer
from pigeon_logging import get_logger, TRACE

logger = get_logger("pigeon", level=TRACE)


def main() -> None:
    """Run a simple ring lab visualization using the classic kernel."""
    lab = PigeonRingLab(
        bit_depth=16,
        kernel=PigeonCrushKernelHelper.classic_kernel,
        visualizer_factory=None,
    )
    vis = PigeonVisualizer(
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


if __name__ == "__main__":
    main()
