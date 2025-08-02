from typing import Any, Optional, List

class PigeonTapePayload:
    def __init__(self, value: Any, origin_bin: int, tape: Optional[List[float]] = None, bin_path: Optional[List[int]] = None):
        self.value = value
        self.origin_bin = origin_bin
        self.tape = tape or []
        self.bin_path = bin_path or []

    def add_stage(self, bin_idx: int, hue: float, tape_mode: str = 'append'):
        self.bin_path.append(bin_idx)
        if tape_mode == 'append':
            self.tape.append(hue)
        elif tape_mode == 'replace':
            if self.tape:
                self.tape[-1] = hue
            else:
                self.tape.append(hue)

    def tape_color(self, depth: int = 0, mode: str = 'first', complement: bool = False) -> float:
        if not self.tape:
            return 0.0
        if mode == 'first':
            idx = 0
        elif mode == 'last':
            idx = -1
        else:
            idx = depth
        if not (-len(self.tape) <= idx < len(self.tape)):
            idx = -1
        hue = self.tape[idx]
        if complement:
            hue = (hue + 0.5) % 1.0
        return hue

    def __repr__(self):
        return f"<PigeonTapePayload val={self.value} tape={self.tape} bins={self.bin_path}>"
