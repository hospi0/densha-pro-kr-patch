# -*- coding: utf-8 -*-
r"""스테이트 RAM 에서 GPU 그리기 패킷(텍스처 사각형) 찾기 (2026-10-18)
  POLY_GT4(0x3C~0x3F, 12워드)·POLY_FT4(0x2C~0x2F): 태그 + [색|cmd] [xy0] [uv0|clut] [xy1] [uv1|tpage] [xy2] [uv2] [xy3] [uv3]
  SPRT(0x64~0x67): 태그 + [색|cmd] [xy] [uv|clut] [wh]
  python tools/gpu_prims.py <sav> → 화면 좌표·UV·CLUT·tpage 목록
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vram as V  # noqa: E402


def xy(w):
    x = w & 0xFFFF; y = w >> 16
    return (x - 0x10000 if x & 0x8000 else x), (y - 0x10000 if y & 0x8000 else y)


def prims(ram):
    out = []
    for i in range(0, len(ram) - 40, 4):
        tag = struct.unpack_from('<I', ram, i)[0]
        n = tag >> 24; nxt = tag & 0xFFFFFF
        if not (0 < n <= 12) or not (nxt == 0xFFFFFF or nxt < 0x200000):
            continue
        cmd = ram[i + 7]
        if n == 9 and 0x2C <= cmd <= 0x2F:
            w = struct.unpack_from('<9I', ram, i + 4)
            x0, y0 = xy(w[1]); x3, y3 = xy(w[7])
            if not (-64 <= x0 <= 640 and -64 <= y0 <= 512):
                continue
            u0, v0, clut = w[2] & 0xFF, (w[2] >> 8) & 0xFF, w[2] >> 16
            u3, v3 = w[8] & 0xFF, (w[8] >> 8) & 0xFF
            tp = w[4] >> 16
            out.append(('FT4', i, x0, y0, x3 - x0, y3 - y0, u0, v0, u3 - u0, v3 - v0, clut, tp))
        elif n == 12 and 0x3C <= cmd <= 0x3F:                       # POLY_GT4
            w = struct.unpack_from('<12I', ram, i + 4)
            x0, y0 = xy(w[1]); x3, y3 = xy(w[10])
            if not (-64 <= x0 <= 640 and -64 <= y0 <= 512):
                continue
            u0, v0, clut = w[2] & 0xFF, (w[2] >> 8) & 0xFF, w[2] >> 16
            u3, v3 = w[11] & 0xFF, (w[11] >> 8) & 0xFF
            out.append(('GT4', i, x0, y0, x3 - x0, y3 - y0, u0, v0, u3 - u0, v3 - v0, clut, w[5] >> 16))
        elif n == 4 and 0x64 <= cmd <= 0x67:
            w = struct.unpack_from('<4I', ram, i + 4)
            x0, y0 = xy(w[1]); ww, hh = w[3] & 0xFFFF, w[3] >> 16
            if not (0 < ww <= 512 and 0 < hh <= 256 and -64 <= x0 <= 640 and -64 <= y0 <= 512):
                continue
            out.append(('SPR', i, x0, y0, ww, hh, w[2] & 0xFF, (w[2] >> 8) & 0xFF, ww, hh, w[2] >> 16, None))
    return out


if __name__ == '__main__':
    r = V.ram(sys.argv[1])
    for p in prims(r):
        k, i, x, y, w, h, u, v, uw, vh, clut, tp = p
        cx, cy = (clut & 0x3F) * 16, clut >> 6
        tps = '' if tp is None else 'tp x%d y%d %dbpp' % ((tp & 15) * 64, ((tp >> 4) & 1) * 256, 4 << ((tp >> 7) & 3))
        print('%s %06X scr(%d,%d %dx%d) uv(%d,%d %dx%d) clut(%d,%d) %s' % (k, i, x, y, w, h, u, v, uw, vh, cx, cy, tps))
