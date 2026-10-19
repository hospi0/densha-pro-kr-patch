# -*- coding: utf-8 -*-
r"""스테이트 RAM 에서 SPRT(0x64~0x67) 패킷을 느슨하게 뽑기 (화면 좌표·크기·uv·clut) (2026-10-18)
  python tools/sprites.py <sav>
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vram as V  # noqa: E402


def sprites(ram):
    out = set()
    for i in range(0, len(ram) - 16, 4):
        w = struct.unpack_from('<4I', ram, i)
        if 0x64 <= w[0] >> 24 <= 0x67:
            x = w[1] & 0xFFFF; y = w[1] >> 16
            x = x - 0x10000 if x & 0x8000 else x; y = y - 0x10000 if y & 0x8000 else y
            ww, hh = w[3] & 0xFFFF, w[3] >> 16
            if 0 < ww <= 256 and 0 < hh <= 256 and -64 <= x <= 640 and -64 <= y <= 512:
                out.add((y, x, ww, hh, w[2] & 0xFF, (w[2] >> 8) & 0xFF, w[2] >> 16))
    return sorted(out)


if __name__ == '__main__':
    for s in sprites(V.ram(sys.argv[1])):
        y, x, w, h, u, v, c = s
        print('scr(%d,%d) %dx%d uv(%d,%d) clut %04X(%d,%d)' % (x, y, w, h, u, v, c, (c & 0x3F) * 16, c >> 6))
