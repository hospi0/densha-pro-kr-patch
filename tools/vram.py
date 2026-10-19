# -*- coding: utf-8 -*-
r"""DuckStation 스테이트(.sav) → VRAM (512×1024 u16). VRAM = 기계 상태 프레임 0x201DDB(RAM 2MB 뒤 0x379 B) (2026-10-18 실측)
  python tools/vram.py <sav> <out.png>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import savestate as SS  # noqa: E402

RAM_OFF = 0x1A62
VRAM_OFF = RAM_OFF + 0x200379                # RAM 뒤 0x379 B(CLUT·TXS 좌표 대조로 확정 2026-10-18; 0x201A62/63 은 틀림)


def state(path):
    return max(SS.frames(path), key=len)


def vram(path):
    st = state(path)
    return np.frombuffer(st[VRAM_OFF:VRAM_OFF + 0x100000], '<u2').reshape(512, 1024).copy()


def ram(path):
    return state(path)[RAM_OFF:RAM_OFF + 0x200000]


if __name__ == '__main__':
    SS.png(vram(sys.argv[1]), sys.argv[2])
