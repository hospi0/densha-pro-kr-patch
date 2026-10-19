# -*- coding: utf-8 -*-
r"""실행 파일 안 디스크 «섹터 지도» (2026-10-18)
  0x80047B70(n): LBA = 23 + Σ_{i<n}(크기[i] + (종류[i] ∈ {0x10,0x0D,0x0E} ? 0 : 1))
  종류 u8 표 0x80048468(1032칸) · 크기 u16 표 0x80048870
  python tools/secmap.py → 항목별 LBA·섹터 수·PAC 위치
"""
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'work', 'disc', 'SLPM_863.78')
B = 0x8000F800
TY, SZ, N = 0x80048468, 0x80048870, 0x408
PAC_LBA = 175


def load(exe=None):
    d = exe if exe is not None else open(EXE, 'rb').read()
    ty = d[TY - B:TY - B + N]
    sz = struct.unpack_from('<%dH' % N, d, SZ - B)
    return ty, sz


def nsec(ty, sz, i):
    return sz[i] + (0 if ty[i] in (0x10, 0x0D, 0x0E) else 1)


def table(exe=None):
    ty, sz = load(exe); out = []; a = 23
    for i in range(N):
        out.append((i, a, nsec(ty, sz, i), ty[i])); a += nsec(ty, sz, i)
    return out


def entry_of(pac_off):
    for i, lba, n, t in table():
        if lba <= PAC_LBA + pac_off // 2048 < lba + n:
            return i, (lba - PAC_LBA) * 2048, n * 2048
    return None


if __name__ == '__main__':
    for i, lba, n, t in table()[:60]:
        print('%4d LBA %6d %5d섹터 종류 %02X PAC %X' % (i, lba, n, t, (lba - PAC_LBA) * 2048))
