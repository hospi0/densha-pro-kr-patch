# -*- coding: utf-8 -*-
r"""스테이트 RAM 에서 SPRT(0x64~0x67) 패킷 찾기 — 앞 E1(tpage) 포함 꼴 [태그 len7][E1][0][0][SPRT 색|cmd][xy][uv|clut][wh] (2026-10-19, 노란 띠·계기판)
  python tools/sprt_scan.py <sav> [x0 y0 x1 y1] → 화면 좌표·UV·CLUT·크기·tpage
"""
import struct
import sys

import vram as V


def xy(w):
    x = w & 0xFFFF; y = w >> 16
    return (x - 0x10000 if x & 0x8000 else x), (y - 0x10000 if y & 0x8000 else y)


def scan(ram):
    out = []
    for o in range(16, len(ram) - 16, 4):
        w = struct.unpack_from('<I', ram, o)[0]
        if w >> 24 not in (0x64, 0x65, 0x66, 0x67):
            continue
        tp = None
        for k in range(1, 5):
            q = struct.unpack_from('<I', ram, o - 4 * k)[0]
            if q >> 24 == 0xE1:
                tp = q & 0xFFFF; break
        if tp is None:
            continue
        x, y = xy(struct.unpack_from('<I', ram, o + 4)[0]); uvc, wh = struct.unpack_from('<II', ram, o + 8)
        out.append(dict(o=o, x=x, y=y, u=uvc & 0xFF, v=(uvc >> 8) & 0xFF, clut=uvc >> 16, w=wh & 0xFFFF, h=wh >> 16, tp=tp))
    return out


if __name__ == '__main__':
    ram = V.ram(sys.argv[1]); box = list(map(int, sys.argv[2:6])) if len(sys.argv) > 5 else None
    seen = set()
    for s in scan(ram):
        if box and not (box[0] <= s['x'] <= box[2] and box[1] <= s['y'] <= box[3]):
            continue
        k = tuple(v for kk, v in s.items() if kk != 'o')
        if k in seen:
            continue
        seen.add(k)
        print('%06X' % s['o'], 'scr', s['x'], s['y'], 'uv', s['u'], s['v'], 'wh', s['w'], s['h'], 'clut %04X' % s['clut'], 'tp %04X' % s['tp'])
