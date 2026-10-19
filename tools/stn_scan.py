# -*- coding: utf-8 -*-
r"""역 이름 칸 레코드 × 노선 그림 9장 판독표 (2026-10-19) — 묶음(노선) 경계 정하기용
  python tools/stn_scan.py → work/stn_scan.json · 화면에 칸마다 노선별 최고 이름·점수
"""
import json, os, sys, warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXS = [0x9269800, 0x926E800, 0x9273800, 0x9278800, 0x927D800, 0x9282800, 0x9287800, 0x928C800, 0x9291800]
TABLES = ((0x7E5AC, 131), (0x7EBD0, 61))


def img(pac, off):
    a = off + 0x800; raw = np.frombuffer(pac[a:a + 64 * 128 * 2], np.uint8).reshape(128, 128)
    return np.stack([raw & 15, raw >> 4], -1).reshape(128, 256)


def recs(pac):
    out = []
    for base, n in TABLES:
        for k in range(n):
            o = base + 12 * k; out.append((o,) + tuple(pac[o:o + 4]))
    return out


def names():
    d = {}
    for l in open(os.path.join(ROOT, 'work', 'text', 'stn_cols.tsv'), encoding='utf-8').read().split('\n')[1:]:
        if l:
            t, u, n = l.split('\t'); d.setdefault(int(t), set()).update(n.split(' '))
    return d


F = ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc', 12)


def rend(name, h, w):
    im = Image.new('L', (w, 12 * len(name))); d = ImageDraw.Draw(im)
    for i, c in enumerate(name):
        d.text((w - 13, 12 * i), c, font=F, fill=255)
    return (np.array(im.resize((w, h))) > 80).astype(float)


def score(a, u, v, w, h, nm):
    if v + h > 128 or u + w > 256:
        return -1
    c = a[v:v + h, u:u + w]
    if (c == 0).mean() > 0.2:
        return -1
    x = ((c > 0) & (c < 6)).astype(float)
    if x.sum() < 10:
        return -1
    r = rend(nm, h, w); best = -1
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            best = max(best, np.corrcoef(x.ravel(), np.roll(np.roll(r, dx, 1), dy, 0).ravel())[0, 1])
    return float(best)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pac = open(os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC'), 'rb').read()
    I = [img(pac, o) for o in TXS]; N = names(); out = []
    for i, (o, u, v, w, h) in enumerate(recs(pac)):
        row = []
        for t, a in enumerate(I):
            best = (-1, '')
            if w != 8:
                for nm in N[t]:
                    if abs(len(nm) * 12 - h) > len(nm) * 3 + 2:
                        continue
                    s = score(a, u, v, w, h, nm)
                    if s > best[0]:
                        best = (round(s, 2), nm)
            row.append(best)
        out.append(dict(i=i, o=o, u=u, v=v, w=w, h=h, per=row))
        print(i, '%X' % o, u, v, w, h, ' '.join('%d:%s%.2f' % (t, b[1], b[0]) if b[0] >= 0.3 else '%d:-' % t for t, b in enumerate(row)))
    json.dump(out, open(os.path.join(ROOT, 'work', 'stn_scan.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
