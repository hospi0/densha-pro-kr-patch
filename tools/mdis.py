# -*- coding: utf-8 -*-
r"""MIPS 역어셈(capstone, LE)
  python tools/mdis.py <메모리주소 hex> [줄수]                      → SLPM_863.78 (메모리 = 파일 + 0x8000F800)
  python tools/mdis.py <메모리주소 hex> <줄수> <파일> <파일0의 메모리주소 hex>   → 오버레이·RAM 덤프
"""
import os
import sys

import capstone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'work', 'disc', 'SLPM_863.78')
BASE = 0x8000F800

if __name__ == '__main__':
    a = int(sys.argv[1], 16); n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    path, base = (sys.argv[3], int(sys.argv[4], 16)) if len(sys.argv) > 4 else (EXE, BASE)
    d = open(path, 'rb').read()
    md = capstone.Cs(capstone.CS_ARCH_MIPS, capstone.CS_MODE_MIPS32 + capstone.CS_MODE_LITTLE_ENDIAN)
    off = a - base
    for ins in md.disasm(d[off:off + n * 4], a):
        print('%08X  %-8s %s' % (ins.address, ins.mnemonic, ins.op_str))
