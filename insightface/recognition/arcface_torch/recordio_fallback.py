"""Small MXNet RecordIO reader fallback.

Used when mxnet cannot be imported on newer Kaggle/Python images. It supports
the indexed RecordIO files used by InsightFace datasets.
"""

import struct
import os
from collections import namedtuple
from pathlib import Path

import numpy as np


IRHeader = namedtuple("HEADER", ["flag", "label", "id", "id2"])

_MAGIC = 0xCED7230A
_LENGTH_MASK = (1 << 29) - 1
_HEADER = struct.Struct("<IfQQ")
_UINT32 = struct.Struct("<I")


def unpack_image_record(record):
    flag, label, rec_id, rec_id2 = _HEADER.unpack(record[: _HEADER.size])
    body = record[_HEADER.size :]
    if flag > 0:
        labels = np.frombuffer(body[: flag * 4], dtype=np.float32, count=flag)
        body = body[flag * 4 :]
        label = labels
    return IRHeader(flag, label, rec_id, rec_id2), body


class MXIndexedRecordIOFallback:
    def __init__(self, idx_path, rec_path):
        self.idx_path = Path(idx_path)
        self.rec_path = Path(rec_path)
        self.idx = {}
        self.keys = []
        with open(self.idx_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                key = int(parts[0])
                self.idx[key] = int(parts[1])
                self.keys.append(key)
        self._pid = None
        self._record = None
        self._open_record()

    def _open_record(self):
        if self._record is not None:
            self._record.close()
        self._record = open(self.rec_path, "rb")
        self._pid = os.getpid()

    def _ensure_process_file(self):
        if self._record is None or self._pid != os.getpid():
            self._open_record()

    def read_idx(self, idx):
        self._ensure_process_file()
        self._record.seek(self.idx[int(idx)])
        return self._read_record()

    def close(self):
        if self._record is not None:
            self._record.close()
            self._record = None

    def _read_segment(self):
        magic_raw = self._record.read(_UINT32.size)
        if len(magic_raw) != _UINT32.size:
            return None, None
        magic = _UINT32.unpack(magic_raw)[0]
        if magic != _MAGIC:
            raise ValueError(f"Invalid RecordIO magic: {magic:#x}")

        lrec = _UINT32.unpack(self._record.read(_UINT32.size))[0]
        cflag = (lrec >> 29) & 7
        length = lrec & _LENGTH_MASK
        data = self._record.read(length)
        pad = (4 - (length % 4)) % 4
        if pad:
            self._record.read(pad)
        return cflag, data

    def _read_record(self):
        parts = []
        while True:
            cflag, data = self._read_segment()
            if cflag is None:
                return None
            if cflag == 0:
                if parts:
                    parts.append(data)
                    return b"".join(parts)
                return data
            if cflag == 1:
                parts = [data]
            elif cflag == 2:
                parts.append(data)
            elif cflag == 3:
                parts.append(data)
                return b"".join(parts)
            else:
                raise ValueError(f"Unsupported RecordIO continuation flag: {cflag}")
