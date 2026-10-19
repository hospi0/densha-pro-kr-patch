# -*- coding: utf-8 -*-
r"""MDEC 정지 영상 한글판 굽기 (2026-10-19) — 제목 24장(title_kr) + 총 주행 거리 11장(distance_kr) + 사진 설명 397장(photo_kr)
  python tools/mdec_kr.py make  → work/bs_out/<PAC 위치>.bs (원래 자리 크기, BS v3) + 되풀어 검산(PSNR)
  build.py 는 apply(pac) 로 work/bs_out 의 파일을 원래 자리에 넣음(파일 없으면 건너뜀 — 먼저 make).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bs  # noqa: E402

OUT = os.path.join(ROOT, 'work', 'bs_out')
RANGE = (0x95E3800, 0xABF4800)                     # 제목·거리(~0x975D000) + 사진 397장(0x9B96800~0xABF4800, photo_kr)


def make():
    import distance_kr as D
    import title_kr as T
    import photo_kr as PK
    pac = open(bs.PAC, 'rb').read()
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, '*.bs')):
        os.remove(f)
    res = T.build_all(pac) + D.build_all(pac) + PK.build_all(pac)
    for o, n, old, new in res:
        b = bs.encode(new, 640, 480, n, 'v3')
        assert len(b) <= n and b[2:4] == b'\x00\x38', (hex(o), len(b), n)
        d = bs.decode(b, 640, 480)
        psnr = 10 * np.log10(255 ** 2 / max(1e-6, ((d.astype(float) - new) ** 2).mean()))
        open(os.path.join(OUT, '%X.bs' % o), 'wb').write(b + bytes(n - len(b)))
        print('%X %d/%d PSNR %.1f' % (o, len(b), n, psnr))


def apply(pac):
    n = 0
    for f in sorted(glob.glob(os.path.join(OUT, '*.bs'))):
        o = int(os.path.basename(f)[:-3], 16); b = open(f, 'rb').read()
        assert RANGE[0] <= o and o + len(b) <= RANGE[1]
        pac[o:o + len(b)] = b; n += 1
    return n


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1] == 'make':
        make()
