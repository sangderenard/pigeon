"""Low-level tape structures for provenance headers and frames."""

import ctypes


class TapeFrame(ctypes.Structure):
    _fields_ = [
        ('hue', ctypes.c_float),
        ('painter', ctypes.c_uint32),
    ]

    def __repr__(self):
        return f"TapeFrame(hue={self.hue}, painter={self.painter})"

MAX_PAINT_UNITS = 64

class TapeHeaderEntry(ctypes.Structure):
    _fields_ = [
        ('painter_id', ctypes.c_uint32),
        ('desc_offset', ctypes.c_uint32),
    ]

class TapeHeaderFrame(ctypes.Structure):
    _fields_ = [
        ('count', ctypes.c_uint32),
        ('entries', TapeHeaderEntry * MAX_PAINT_UNITS),
    ]
