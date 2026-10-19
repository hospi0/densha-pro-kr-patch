# -*- coding: utf-8 -*-
r"""파일 안 SJIS 전각 연속(기본 4자 이상) 찾기 (2026-10-18)
  python tools/sjis_scan.py <파일> [최소글자] [최대 출력]
"""
import sys


def runs(b, minc=4):
    out = []; i = 0; n = len(b)
    while i < n - 1:
        j = i; c = 0
        while j < n - 1 and (0x81 <= b[j] <= 0x9f or 0xe0 <= b[j] <= 0xef) and 0x40 <= b[j + 1] <= 0xfc and b[j + 1] != 0x7f:
            j += 2; c += 1
        if c >= minc:
            out.append((i, j)); i = j
        else:
            i += 1
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    d = open(sys.argv[1], 'rb').read()
    mc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    r = runs(d, mc)
    print('runs', len(r), 'bytes', sum(b - a for a, b in r))
    for a, b in r[:mx]:
        print('%08X' % a, d[a:b].decode('cp932', 'replace'))
