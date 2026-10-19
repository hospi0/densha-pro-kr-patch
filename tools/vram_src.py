# -*- coding: utf-8 -*-
r"""스테이트 VRAM 텍스처 영역(x<448 등 화면 밖) 줄 조각을 TRAIN.PAC 에서 찾아 «VRAM 칸 → PAC 위치» 대응 (2026-10-18)
  32 halfword(=64 B) 조각, 0·단조 조각 제외. 같은 PAC 블록이면 (PAC − VRAM 선형 위치) 차가 일정 → 블록으로 묶음.
  python tools/vram_src.py → work/vram_src.json [{state, x, y, pac, stride}]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import vram as V  # noqa: E402


def main():
    pac = open(os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC'), 'rb').read()
    paths = [('%d' % n, os.path.join(ROOT, 'work', 'shots', '%d.sav' % n)) for n in range(1, 31)]
    paths.append(('rc1', os.path.join(ROOT, 'work', 'shots', 'route_complex_1.sav')))
    cache = {}; out = []
    for name, p in paths:
        v = V.vram(p)
        for y in range(0, 512, 4):
            for x in range(0, 1024, 32):
                row = v[y, x:x + 32].tobytes()
                if row.count(0) > 32 or len(set(row)) < 10:
                    continue
                if row not in cache:
                    cache[row] = pac.find(row)
                o = cache[row]
                if o >= 0:
                    out.append((name, x, y, o))
        print(name, len(out))
    json.dump(out, open(os.path.join(ROOT, 'work', 'vram_src.json'), 'w'))


if __name__ == '__main__':
    main()
