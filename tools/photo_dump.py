# -*- coding: utf-8 -*-
r"""사진 플로어 BS(제목·거리 35장·체험판 3장 뺀 397장) 전부 풀기 + 설명 글 판독 시트 (2026-10-19)
  python tools/photo_dump.py → work/photo/<위치>.png · work/photo/sheet_NN.png(설명 글 띠 v380~480, 20장씩, 번호 붙임)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bs  # noqa: E402

OUT = os.path.join(ROOT, 'work', 'photo')
SKIP_LO, SKIP_HI = 0x95E3800, 0x9772800 + 1          # 제목 24·거리 11·체험판 3(975D000·9767800·9772800)


def photos():
    L = [l.split('\t') for l in open(os.path.join(ROOT, 'work', 'bs_list.txt'), encoding='utf-8').read().split('\n') if l]
    return [(int(a[0], 16), int(a[4].split()[-1])) for a in L if not SKIP_LO <= int(a[0], 16) < SKIP_HI]


def main():
    pac = open(bs.PAC, 'rb').read(); os.makedirs(OUT, exist_ok=True)
    P = photos(); print(len(P))
    caps = []
    for k, (o, n) in enumerate(P):
        p = os.path.join(OUT, '%X.png' % o)
        if not os.path.exists(p):
            Image.fromarray(bs.decode(pac[o:o + n], 640, 480)).save(p)
        caps.append((k, o, np.array(Image.open(p))[360:480]))
    for s in range(0, len(caps), 16):
        cv = Image.new('RGB', (1280, 8 * 134), (0, 0, 0)); d = ImageDraw.Draw(cv)
        for j, (k, o, c) in enumerate(caps[s:s + 16]):
            x, y = (j % 2) * 640, (j // 2) * 134
            cv.paste(Image.fromarray(c), (x, y + 14)); d.text((x + 2, y), '%d %X' % (k, o), fill=(255, 255, 0))
        cv.save(os.path.join(OUT, 'sheet_%02d.png' % (s // 16)))


if __name__ == '__main__':
    main()
