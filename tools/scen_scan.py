# -*- coding: utf-8 -*-
r"""풍경 이름표 레코드 표 찾기 (2026-10-19, 큐슈 스테이트 RAM 0x725B4 에서 발견)
  레코드 12 B = [u][v][w][h=16][0 ×4][tpage u16 (0x80xx)][CLUT u16] — 노선 모듈 안에 연속으로 있음(큐슈 PAC 0x375B0 근처).
  python tools/scen_scan.py → 표마다 레코드 목록(PAC 위치·VRAM x,y·w)
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')


def rec(pac, o):
    u, v, w, h = pac[o:o + 4]; z = struct.unpack_from('<I', pac, o + 4)[0]
    tp, cl = struct.unpack_from('<HH', pac, o + 8)
    return u, v, w, h, z, tp, cl


def ok(r):
    u, v, w, h, z, tp, cl = r
    return h == 16 and z == 0 and (tp & 0xFF00) == 0x8000 and w and w % 16 == 0 and u % 16 == 0 and v % 16 == 0 and cl


def tables(pac):
    out = []; o = 0
    while o < len(pac) - 12:
        if ok(rec(pac, o)):
            s = o; rs = []
            while o < len(pac) - 12 and ok(rec(pac, o)):
                rs.append((o, rec(pac, o))); o += 12
            if len(rs) >= 2 and len({r[1][6] for r in rs}) == 1:
                out.append((s, rs))
        else:
            o += 4
    return out


if __name__ == '__main__':
    pac = open(PAC, 'rb').read()
    for s, rs in tables(pac):
        print('%X  %d개  CLUT %04X' % (s, len(rs), rs[0][1][6]))
        for o, (u, v, w, h, z, tp, cl) in rs:
            x = (tp & 15) * 64; y = ((tp >> 4) & 1) * 256
            print('   %X tp %04X vram(%d,%d) u%d v%d w%d' % (o, tp, x, y, u, v, w))
