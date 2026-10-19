# -*- coding: utf-8 -*-
r"""시각표(운전 승무) 역 이름 조각 조사 (2026-10-18)
  역 이름 = 34번 파일(PAC 0xBA800, RAM 0x8006F000) 16 B 레코드(menu_kr 꼴) 중 tpage 0xD, CLUT 38A8(위 2비트)·38E8(아래 2비트).
  역 하나 = x0 레코드부터 다음 x0 전까지(둘째·셋째 칸 x24·x48, 넉 자 이름은 64폭 한 조각).
  그림 = 운전 승무 아틀라스 TXS 7E54800 (VRAM 640,0 256×256) 의 tpage D 칸(아틀라스 열 768~1023), 칸 16×24.
  칸 판독 = 역 사전(work/text/stations.tsv) 글자 + 가나를 MS 고딕으로 그려 상관 비교.
  python tools/tt_stations.py → 역 목록(레코드·일본어·사전 번역)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import menu_kr as MK  # noqa: E402

FILE34 = 0xBA800
TXS = 0x7E54800
RUNS = ((0xAA7E, 94), (0xB06E, 469))
LAYER = {0x38A8: 2, 0x38E8: 0}


def records(pac):
    F = pac[FILE34:FILE34 + 0xE000]; out = []
    for base, n in RUNS:
        for k in range(n):
            o = base + 16 * k; w = struct.unpack_from('<8H', F, o)
            out.append(dict(o=o, u=w[0] & 0xFF, v=w[0] >> 8, x=MK.s16(w[1]), y=MK.s16(w[2]), w=w[3], h=w[4], tp=w[5], clut=w[6], fl=w[7]))
    return out


def stations(recs):
    """역 = tpage D 레코드 연속에서 x0 로 시작하는 묶음"""
    out = []; cur = None; prev = None
    for r in recs:
        isst = r['tp'] == 0xD and r['clut'] in LAYER
        if not isst:
            cur = None; prev = r; continue
        if cur is None or r['x'] == 0 or (prev and r['o'] - prev['o'] != 16):
            cur = []; out.append(cur)
        cur.append(r); prev = r
    return out


def block(pac):
    p = TXS + 0x800
    raw = np.frombuffer(pac[p:p + 256 * 256 * 2], np.uint8).reshape(256, 512)
    return np.stack([raw & 15, raw >> 4], -1).reshape(256, 1024)[:, 768:].copy()


def cell(blk, r, k=0):
    sh = LAYER[r['clut']]
    a = (blk[r['v']:r['v'] + 24, r['u'] + 16 * k:r['u'] + 16 * k + 16] >> sh) & 3
    return a


def templates(chars):
    f = ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc', 18)
    out = {}
    for c in chars:
        im = Image.new('L', (16, 24)); ImageDraw.Draw(im).text((-1, 2), c, font=f, fill=255)
        out[c] = np.array(im, float)
    return out


def ocr(a, T):
    x = (a > 0).astype(float)
    if x.sum() < 6:
        return '　', 0
    best = None
    for c, t in T.items():
        # 위치 흔들림 ±2
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-1, 0, 1):
                tt = np.roll(np.roll(t, dy, 0), dx, 1)
                s = np.corrcoef(x.ravel(), tt.ravel())[0, 1]
                if best is None or s > best[1]:
                    best = (c, s)
    return best


def cellmap():
    """work/text/tt_cells.txt(받아 적은 칸) → {(CLUT, u, v): 글자}"""
    rows = [l.rstrip(chr(10)) for l in open(os.path.join(ROOT, 'work', 'text', 'tt_cells.txt'), encoding='utf-8') if not l.startswith('#')]
    assert len(rows) == 20 and all(len(r) == 16 for r in rows), [len(r) for r in rows]
    out = {}
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            out[(0x38A8 if i < 10 else 0x38E8, 16 * j, 24 * (i % 10))] = '　' if c == '＿' else c
    return out


def name_of(st, cm):
    s = ''
    for r in st:
        for k in range(max(1, r['w'] // 16)):
            s += cm.get((r['clut'], r['u'] + 16 * k, r['v']), '？')
    return s


def match(name, dic):
    """사전에서 같은 길이·차이 최소(？ 는 아무 글자) 항목 → (항목, 차이)"""
    best = (None, 99)
    for k in dic:
        if len(k) != len(name):
            continue
        d = sum(1 for a, b in zip(name, k) if a != '？' and a != b)
        if d < best[1]:
            best = (k, d)
    return best


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pac = open(MK.PAC, 'rb').read()
    dic = dict(l.rstrip(chr(10)).split(chr(9))[:2] for l in open(os.path.join(ROOT, 'work', 'text', 'stations.tsv'), encoding='utf-8') if chr(9) in l)
    cm = cellmap(); recs = records(pac)
    for st in stations(recs):
        nm = name_of(st, cm).replace('　', '')
        k, d = match(nm, dic)
        print('%X|%d|%s|%s|%s|%d'.replace('|', chr(9)) % (st[0]['o'], len(st), nm, k or '', dic.get(k, '') if k else '', d))


if __name__ == '__main__':
    main()
